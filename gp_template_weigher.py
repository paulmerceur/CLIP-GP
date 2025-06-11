import torch
import torch.nn as nn
import torch.nn.functional as F
import gpytorch


class GaussianProcessTemplateWeighter(gpytorch.models.ApproximateGP):
    """Simplified GP‑based template weighter that leans *entirely* on
    GPyTorchʼs built‑ins.

    ➤ *No* hand‑rolled KL, priors, or extra regularisation.
    ➤ One batched variational GP (one task == one class).
    ➤ Templates are the *inducing* points – nothing to learn there.
    """

    def __init__(
        self,
        text_embeddings: torch.Tensor,  # [K, M, D]  (frozen)
        *,
        kernel_type: str = "rbf",
        num_mc_samples: int = 4,
        use_diag: bool = True,
        lengthscale: float = 1.0,
        outputscale: float = 1.0,
        noise_var: float = 1e-4,
    ) -> None:
        # Keep original dtype/device for later but run GP in fp32 for numerical
        # stability (GPyTorch is primarily tested in fp32).
        self.orig_dtype = text_embeddings.dtype
        self.orig_device = text_embeddings.device

        text_embeddings_fp32 = text_embeddings.to(dtype=torch.float32)

        self.num_classes, self.num_templates, self.dim = text_embeddings_fp32.shape
        self.num_mc_samples = num_mc_samples

        # ------------------------------------------------------------------
        #  Build a batched variational GP
        # ------------------------------------------------------------------
        batch_shape = torch.Size([self.num_classes])  # one GP per class

        # Use the (frozen) template encodings as inducing inputs.
        inducing_inputs = text_embeddings_fp32.detach().clone().view(-1, self.dim)

        # Full‑covariance variational posterior (Cholesky factor learnt).
        variational_dist = gpytorch.variational.CholeskyVariationalDistribution(
            num_inducing_points=self.num_templates,
            batch_shape=batch_shape,
        )

        base_vs = gpytorch.variational.VariationalStrategy(
            self,
            inducing_inputs.view(self.num_classes, self.num_templates, self.dim),
            variational_dist,
            learn_inducing_locations=False,
        )

        # One independent GP *per class* (tasks are independent).
        variational_strategy = gpytorch.variational.IndependentMultitaskVariationalStrategy(
            base_vs, num_tasks=self.num_classes
        )

        super().__init__(variational_strategy)

        # Mean and covariance modules -------------------------------------------------
        self.mean_module = gpytorch.means.ConstantMean(batch_shape=batch_shape)

        if kernel_type.lower() == "rbf":
            base_kernel = gpytorch.kernels.RBFKernel(
                batch_shape=batch_shape, ard_num_dims=self.dim
            )
        elif kernel_type.lower() == "cosine":
            base_kernel = gpytorch.kernels.CosineKernel(batch_shape=batch_shape)
        elif kernel_type.lower() == "linear":
            base_kernel = gpytorch.kernels.LinearKernel(batch_shape=batch_shape)
        else:
            raise ValueError(f"Unsupported kernel: {kernel_type}")

        self.covar_module = gpytorch.kernels.ScaleKernel(
            base_kernel, batch_shape=batch_shape
        )

        # ------------------------------------------------------------------
        #  Break symmetry: initialise variational mean with a **larger** noise
        #  so that the subsequent softmax is able to deviate from the uniform
        #  distribution early in training.  Using too small a std (e.g. 0.05)
        #  made the initial logits almost zero, resulting in near-uniform
        #  weights and vanishing gradients. A std≈1 provides logit differences
        #  of order ±1, i.e. exp(±1)≈{0.37,2.7}, which yields noticeable but
        #  still moderate departure from the uniform prior.
        # ------------------------------------------------------------------
        with torch.no_grad():
            vm = self.variational_strategy.base_variational_strategy._variational_distribution.variational_mean
            vm.normal_(mean=0.0, std=1.0)

        # Learnable temperature (scales α before the softmax); initial value 1.
        # Larger values sharpen the distribution, smaller values flatten it.
        self.register_parameter("weight_temperature", nn.Parameter(torch.tensor(1.0)))

        # Register the (fixed) template embeddings (original dtype) for downstream
        # use (e.g., for normalising prototypes/pruning). This buffer is not
        # used by the GP computations directly, hence it stays in original
        # precision to align with CLIP.
        self.register_buffer("_templates", text_embeddings.detach())

    # ------------------------------------------------------------------
    #  Standard GP forward – required by gpytorch
    # ------------------------------------------------------------------
    def forward(self, x):  # x: [..., D]
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

    # ------------------------------------------------------------------
    #  Public API compatible with previous code
    # ------------------------------------------------------------------
    def _weight_distribution(self):
        """Return the **posterior** distribution over template weights α.

        Note
        ----
        The previous implementation mistakenly called ``self.forward``
        directly, which only returns the *prior* (as it bypasses the
        variational strategy).  As a consequence, the distribution – and
        therefore the expected weights – were completely *independent* of
        the learnable variational parameters, so the optimisation had no
        effect on the template weights.

        The correct way to obtain the posterior in a GPyTorch
        ``ApproximateGP`` is to invoke the model *via* its variational
        strategy.  The simplest (and officially supported) path is to call
        the strategy directly, letting it take care of wrapping the prior
        produced by ``forward``.
        """

        # Evaluate the GP at the (fixed) template embeddings while
        # transparently passing through the variational strategy so that the
        # returned distribution depends on the variational parameters.

        x = self._templates.to(torch.float32)

        # ``IndependentMultitaskVariationalStrategy`` expects an input of
        # shape ``[K, M, D]`` (already satisfied) and will return a batched
        # ``MultivariateNormal`` with batch dimension ``K`` and event shape
        # ``M`` – exactly what we need.

        return self.variational_strategy(x)

    def _kl(self) -> torch.Tensor:
        """Analytic KL(q‖p) with robust device/dtype alignment.

        ``gpytorch`` sometimes memoizes the prior distribution on the device
        on which the model was *first* called (CPU, in our case). If we later
        move the module to GPU, the cached prior stays on CPU, causing the
        ``kl_divergence`` call to raise a device-mismatch error.  We catch
        that situation and rebuild the prior on the correct device.
        """

        try:
            return self.variational_strategy.kl_divergence().sum()
        except RuntimeError as err:
            if "device" not in str(err):
                raise  # Different error – re-raise

            # ---- Device mismatch detected: compute KL manually on a common device ----
            q = self.variational_strategy.variational_distribution
            p = self.variational_strategy.prior_distribution

            target_device = q.mean.device
            target_dtype = q.mean.dtype

            # Convert both distributions to dense-tensor versions on the same device
            q_mean = q.mean.to(device=target_device, dtype=target_dtype)
            p_mean = p.mean.to(device=target_device, dtype=target_dtype)

            q_cov = q.lazy_covariance_matrix.to_dense().to(device=target_device, dtype=target_dtype)
            p_cov = p.lazy_covariance_matrix.to_dense().to(device=target_device, dtype=target_dtype)

            torch_mvn_q = torch.distributions.MultivariateNormal(q_mean, q_cov)
            torch_mvn_p = torch.distributions.MultivariateNormal(p_mean, p_cov)

            return torch.distributions.kl.kl_divergence(torch_mvn_q, torch_mvn_p).sum()

    def _prototype_from_weights(self, weights: torch.Tensor) -> torch.Tensor:
        # weights: [K, M] (fp32)   templates: [K, M, D] (could be fp16)
        # Cast weights to the template dtype to avoid mixed-precision errors
        if weights.dtype != self._templates.dtype:
            weights = weights.to(dtype=self._templates.dtype)
        else:
            print("GaussianProcessTemplateWeighter._prototype_from_weights: weights.dtype == self._templates.dtype")

        return torch.einsum("km,kmd->kd", weights, self._templates)

    # ------------------------------------------------------------------
    #  Main entry point (keeps the old signature for drop‑in replacement)
    # ------------------------------------------------------------------
    def forward_and_kl(self, *, use_mean: bool = False):
        """Return class prototypes **and** KL term.

        Args
        ----
        use_mean:  Use posterior mean instead of MC sampling (deterministic).
        """
        # Ensure we are not using memoised distributions from a different device
        if hasattr(self.variational_strategy, "_clear_cache"):
            # Clear caches that could hold CPU tensors when the module is moved to GPU
            self.variational_strategy._clear_cache()

        q = self._weight_distribution()  # MultivariateNormal (batch)

        # ------------------------------------------------------------------
        #  Use Monte-Carlo estimate during training; deterministic mean at eval.
        # ------------------------------------------------------------------

        if self.training and not use_mean:
            alpha = q.rsample(torch.Size([self.num_mc_samples]))  # Expected shape [S, K, M] but some configs
            # Handle potential (S, M, K) swap originating from input shape
            if alpha.dim() == 3 and alpha.size(1) == self.num_templates and alpha.size(2) == self.num_classes:
                # Transpose to [S, K, M]
                alpha = alpha.permute(0, 2, 1)
            w = F.softmax(alpha * self.weight_temperature, dim=-1).mean(0)                 # [K, M]
        else:
            mu = q.mean
            # Detect and fix swapped dims [M, K] -> [K, M]
            if mu.size(0) == self.num_templates and mu.size(1) == self.num_classes:
                mu = mu.t()
            w = F.softmax(mu * self.weight_temperature, dim=-1)                             # [K, M]

        prototypes = self._prototype_from_weights(w)              # [K, D]

        prototypes = prototypes.to(dtype=self._templates.dtype, device=self._templates.device)
        return prototypes, self._kl()

    # Keep compatibility with the old call signature used in adapters.py
    def __call__(self, *_unused, use_mean: bool = False):  # type: ignore[override]
        return self.forward_and_kl(use_mean=use_mean)
