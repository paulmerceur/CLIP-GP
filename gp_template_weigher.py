import torch
import torch.nn as nn
import torch.nn.functional as F
import gpytorch
import numpy as np
from gpytorch.utils.memoize import clear_cache_hook


class GaussianProcessTemplateWeighter(gpytorch.models.ApproximateGP):
    """
    Simplified GP-based template weighter that leans entirely on GPyTorch's built-ins.

    * No hand-rolled KL, priors, or extra regularisation.
    * One batched variational GP (one task == one class).
    * Templates are the *inducing* points - nothing to learn there.
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
        self.mean_module = PerTemplateMean(self.num_classes, self.num_templates)

        if cfg.TRAINER.ADAPTER.GP_KERNEL_TYPE.lower() == "rbf":
            base_kernel = gpytorch.kernels.RBFKernel(batch_shape=batch_shape, ard_num_dims=self.dim)
            base_kernel.initialize(lengthscale=cfg.TRAINER.ADAPTER.GP_LENGTH_SCALE)
            # Debug: verify that the initial length-scale is set as expected
            print(f"[DEBUG] RBF kernel initial length-scale: {base_kernel.lengthscale.detach().cpu().view(-1)}")
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
        #  Break symmetry: initialise variational mean with small noise
        # ------------------------------------------------------------------
        with torch.no_grad():
            vm = self.variational_strategy._variational_distribution.variational_mean
            # Small random perturbation instead of a perfectly uniform start
            vm.normal_(mean=0.0, std=1e-2)

        # ------------------------------------------------------------------
        #  Learnable softmax temperature (log_tau) to control concentration
        # ------------------------------------------------------------------
        self.log_tau = nn.Parameter(torch.tensor(0.0))  # softplus(0) ≈ 0.693

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

        q = self.variational_strategy(self._templates.to(torch.float32))  # MultivariateNormal (batch)

        # ------------------------------------------------------------------
        #  Use Monte-Carlo estimate during training; deterministic mean at eval.
        # ------------------------------------------------------------------

        if self.training and not use_mean:
            # ------------------------------------------------------------------
            #  Draw *one* stochastic sample for each class during training so that
            #  every forward pass receives a fresh prototype realisation. This
            #  preserves the Monte-Carlo nature assumed by the ELBO without
            #  collapsing it into the mean of several samples.
            # ------------------------------------------------------------------

            alpha = q.rsample()  # Shape: [K, M]

            # Handle potential (M, K) swap originating from input shape
            if alpha.dim() == 2 and alpha.size(0) == self.num_templates and alpha.size(1) == self.num_classes:
                # Transpose to [K, M]
                alpha = alpha.t()

            tau = F.softplus(self.log_tau) + 1e-4
            w = F.softmax(alpha / tau, dim=-1)  # [K, M]  (temperature-scaled)
        else:
            mu = q.mean
            # Detect and fix swapped dims [M, K] -> [K, M]
            if mu.size(0) == self.num_templates and mu.size(1) == self.num_classes:
                mu = mu.t()
            tau = F.softplus(self.log_tau) + 1e-4
            w = F.softmax(mu / tau, dim=-1)  # [K, M]  (temperature-scaled)

        # Keep all GP maths in fp32 for numerical stability; cast back *only* at the end
        # Compute prototypes using fp32 representations of templates
        prototypes = torch.einsum("km,kmd->kd", w, self._templates.float())

        # Return in the original dtype/device so downstream CLIP code stays unchanged
        prototypes = prototypes.to(dtype=self.orig_dtype, device=self.orig_device)
        kl = self.variational_strategy.kl_divergence().sum() / self.num_classes
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
            tau = F.softplus(self.log_tau) + 1e-4
            w = torch.softmax(q.mean / tau, dim=-1).unsqueeze(0)  # [1,K,M] – fp32
        else:
            alpha = q.rsample(torch.Size([num_samples]))  # [S,K,M]
            tau = F.softplus(self.log_tau) + 1e-4
            w = torch.softmax(alpha / tau, dim=-1)              # [S,K,M] – fp32

        # Compute prototypes in fp32 then cast back to CLIP precision
        prototypes = torch.einsum("skm,kmd->skd", w, self._templates.float())
        return prototypes.to(dtype=self.orig_dtype, device=self._templates.device)


class PerTemplateMean(gpytorch.means.Mean):
    """Learnable mean of shape [K, M] (one bias per class & template)."""

    def __init__(self, num_classes: int, num_templates: int):
        super().__init__()
        # One scalar per (class, template) pair – initialised at 0.
        self.mean_param = torch.nn.Parameter(torch.zeros(num_classes, num_templates))

    def forward(self, x):  # noqa: D401 – simple forward override
        """Return mean of shape [K, N] where N = x.size(-2).

        If *N* equals the number of stored template biases (*M*), we use them
        directly.  Otherwise (e.g. when *x* is the concatenation of inducing
        and test points, hence N = 2 M) we broadcast the **average** bias of
        each class across all points.  This keeps shapes compatible while still
        learning a per-class offset.  Feel free to refine this later.
        """

        N = x.size(-2)
        K, M = self.mean_param.shape

        if N == M:
            return self.mean_param

        # Otherwise use per-class scalar (mean over templates) and broadcast
        per_class_bias = self.mean_param.mean(dim=-1, keepdim=True)  # [K,1]
        return per_class_bias.expand(K, N)