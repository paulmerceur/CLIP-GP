import torch
import torch.nn as nn
import torch.nn.functional as F
import gpytorch


class GaussianProcessTemplateWeighter(gpytorch.models.ApproximateGP):
    """
    Simplified GP-based template weighter that leans *entirely* on
    GPyTorchʼs built-ins.

    ➤ *No* hand-rolled KL, priors, or extra regularisation.
    ➤ One batched variational GP (one task == one class).
    ➤ Templates are the *inducing* points - nothing to learn there.
    """

    def __init__(self, text_embeddings: torch.Tensor, cfg: dict, **kwargs) -> None:
        # Keep original dtype/device for later but run GP in fp32 for numerical
        # stability (GPyTorch is primarily tested in fp32).
        self.orig_dtype = text_embeddings.dtype
        self.orig_device = text_embeddings.device

        text_embeddings_fp32 = text_embeddings.to(dtype=torch.float32)

        self.num_classes, self.num_templates, self.dim = text_embeddings_fp32.shape
        self.num_mc_samples = cfg.TRAINER.ADAPTER.GP_NUM_MC_SAMPLES

        # ------------------------------------------------------------------
        #  One independent GP per class (batched). We do NOT wrap the
        #  strategy in an extra IndependentMultitaskVariationalStrategy
        #  because the batch dimension **already** indexes the classes. The
        #  previous double-wrapping duplicated the class dimension, leading
        #  to a weight matrix of shape [K, K, M] instead of the intended
        #  [K, M]. This, in turn, produced almost uniform (and thus
        #  ineffective) template weights during optimisation.
        # ------------------------------------------------------------------

        batch_shape = torch.Size([self.num_classes])  # one GP per class

        # Use the (frozen) template encodings as inducing inputs.
        inducing_inputs = text_embeddings_fp32.detach().clone().view(
            self.num_classes, self.num_templates, self.dim
        )

        variational_dist = gpytorch.variational.CholeskyVariationalDistribution(
            num_inducing_points=self.num_templates,
            batch_shape=batch_shape,
        )

        variational_strategy = gpytorch.variational.VariationalStrategy(
            self,
            inducing_inputs,
            variational_dist,
            learn_inducing_locations=False,
        )

        super().__init__(variational_strategy)

        # Mean and covariance modules -------------------------------------------------
        self.mean_module = gpytorch.means.ConstantMean(batch_shape=batch_shape)

        if cfg.TRAINER.ADAPTER.GP_KERNEL_TYPE.lower() == "rbf":
            base_kernel = gpytorch.kernels.RBFKernel(
                batch_shape=batch_shape, ard_num_dims=self.dim, length_scale=cfg.TRAINER.ADAPTER.GP_LENGTH_SCALE
            )
        elif cfg.TRAINER.ADAPTER.GP_KERNEL_TYPE.lower() == "cosine":
            base_kernel = gpytorch.kernels.CosineKernel(batch_shape=batch_shape)
        elif cfg.TRAINER.ADAPTER.GP_KERNEL_TYPE.lower() == "linear":
            base_kernel = gpytorch.kernels.LinearKernel(batch_shape=batch_shape)
        else:
            raise ValueError(f"Unsupported kernel: {cfg.TRAINER.ADAPTER.GP_KERNEL_TYPE}")

        self.covar_module = gpytorch.kernels.ScaleKernel(
            base_kernel, batch_shape=batch_shape
        )

        # ------------------------------------------------------------------
        #  Break symmetry: randomise the variational mean so that early
        #  softmaxes are *not* exactly uniform. The attribute path has changed
        #  now that we removed the extra multitask wrapper.
        # ------------------------------------------------------------------
        with torch.no_grad():
            vm = self.variational_strategy._variational_distribution.variational_mean
            vm.fill_(1.0 / self.num_templates)

        # Register the (fixed) template embeddings (original dtype) for downstream
        # use (e.g., for normalising prototypes/pruning). This buffer is not
        # used by the GP computations directly, hence it stays in original
        # precision to align with CLIP.
        self.register_buffer("_templates", text_embeddings.detach())

    # ------------------------------------------------------------------
    #  Standard GP forward - required by gpytorch
    # ------------------------------------------------------------------
    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

    # ------------------------------------------------------------------
    #  Main entry point (keeps the old signature for drop-in replacement)
    # ------------------------------------------------------------------
    def forward_and_kl(self, *, use_mean: bool = False):
        """
        Return class prototypes and KL term.

        Args
        ----
        use_mean:  Use posterior mean instead of MC sampling (deterministic).
        """
        # Ensure we are not using memoised distributions from a different device
        if hasattr(self.variational_strategy, "_clear_cache"):
            self.variational_strategy._clear_cache()

        q = self.variational_strategy(self._templates.to(torch.float32))  # MultivariateNormal (batch)

        # ------------------------------------------------------------------
        #  Use Monte-Carlo estimate during training; deterministic mean at eval.
        # ------------------------------------------------------------------

        if self.training and not use_mean:
            alpha = q.rsample(torch.Size([self.num_mc_samples]))  # Expected shape [S, K, M] but some configs
            # Handle potential (S, M, K) swap originating from input shape
            if alpha.dim() == 3 and alpha.size(1) == self.num_templates and alpha.size(2) == self.num_classes:
                # Transpose to [S, K, M]
                alpha = alpha.permute(0, 2, 1)
            w = F.softmax(alpha, dim=-1).mean(0) # [K, M]
        else:
            mu = q.mean
            # Detect and fix swapped dims [M, K] -> [K, M]
            if mu.size(0) == self.num_templates and mu.size(1) == self.num_classes:
                mu = mu.t()
            w = F.softmax(mu, dim=-1) # [K, M]

        if w.dtype != self._templates.dtype:
            w = w.to(dtype=self._templates.dtype)
        prototypes = torch.einsum("km,kmd->kd", w, self._templates)

        prototypes = prototypes.to(dtype=self._templates.dtype, device=self._templates.device)
        kl = self.variational_strategy.kl_divergence().sum()
        return prototypes, kl
    
    def get_weight_distribution(self):
        return self.variational_strategy(self._templates.to(torch.float32))

    # ------------------------------------------------------------------
    #  Helper for Monte-Carlo ELBO
    # ------------------------------------------------------------------
    def sample_prototypes(self, num_samples: int, *, use_mean: bool = False) -> torch.Tensor:
        """Draw *num_samples* sets of template-weighted class prototypes.

        Returns
        -------
        Tensor
            A tensor of shape ``[S, K, D]`` where *S* is ``num_samples``,
            *K* the number of classes and *D* the embedding dimension.
        """
        # Always operate in fp32 for GPyk computations; cast to original
        # dtype just before returning so that downstream code can combine
        # the prototypes with CLIP features without extra conversions.
        q = self.get_weight_distribution()

        if use_mean:
            w = torch.softmax(q.mean, dim=-1).unsqueeze(0)  # [1,K,M]
        else:
            alpha = q.rsample(torch.Size([num_samples]))  # [S,K,M]
            w = torch.softmax(alpha, dim=-1)              # [S,K,M]

        # Linear map  weights → prototypes
        prototypes = torch.einsum("skm,kmd->skd", w.to(dtype=self._templates.dtype), self._templates)

        # Keep original precision/device for compatibility with CLIP.
        return prototypes.to(dtype=self._templates.dtype, device=self._templates.device)