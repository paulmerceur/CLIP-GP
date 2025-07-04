import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import gpytorch


class GaussianProcessTemplateWeighter(gpytorch.models.ApproximateGP):
    """
    Simplified GP-based template weighter that leans entirely on GPyTorch's built-ins.

    * No hand-rolled KL, priors, or extra regularisation.
    * One batched variational GP (one task == one class).
    * Templates are the *inducing* points - nothing to learn there.
    """

    def __init__(self, text_embeddings: torch.Tensor, cfg: dict, mean_init: torch.Tensor = None, **kwargs) -> None:
        # Keep original dtype/device for later but run GP in fp32 for numerical stability (GPyTorch is primarily tested in fp32).
        self.orig_dtype = text_embeddings.dtype
        self.orig_device = text_embeddings.device
        text_embeddings_fp32 = text_embeddings.to(dtype=torch.float32)

        self.num_classes, self.num_templates, self.dim = text_embeddings_fp32.shape
        self.num_mc_samples = cfg.TRAINER.ADAPTER.GP_NUM_MC_SAMPLES

        batch_shape = torch.Size([self.num_classes])  # one GP per class

        # Use the (frozen) template encodings as inducing points.
        inducing_iuts = text_embeddings_fp32.detach().clone().view(
            self.num_classes, self.num_templates, self.dim
        )

        variational_dist = gpytorch.variational.CholeskyVariationalDistribution(
            num_inducing_points=self.num_templates,
            batch_shape=batch_shape,
        )

        variational_strategy = gpytorch.variational.VariationalStrategy(
            self,
            inducing_iuts,
            variational_dist,
            learn_inducing_locations=False,
        )

        super().__init__(variational_strategy)

        # Mean and covariance modules -------------------------------------------------
        self.mean_module = PerTemplateMean(self.num_classes, self.num_templates)

        # Optionally initialise the per-template mean with external logits (e.g. zero-shot soft prompts)
        if mean_init is not None:
            assert mean_init.shape == (self.num_classes, self.num_templates), \
                "mean_init must have shape [num_classes, num_templates]"
            # Store in fp32 for numerical stability – same dtype used by GP modules
            self.mean_module.mean_param.data = mean_init.to(dtype=torch.float32)

        if cfg.TRAINER.ADAPTER.GP_KERNEL_TYPE.lower() == "rbf":
            with torch.no_grad():
                flat_emb = F.normalize(text_embeddings_fp32.reshape(-1, self.dim), p=2, dim=-1)  # [(K*M), D]
                pdist = torch.cdist(flat_emb, flat_emb)
                # Exclude the zero diagonal before taking the median
                ls_cfg = pdist[pdist > 0].median().item()
            print(f"[GP] Auto length-scale (normalised median): {ls_cfg:.4f}")

            base_kernel = gpytorch.kernels.RBFKernel(batch_shape=batch_shape, ard_num_dims=self.dim)
            base_kernel.initialize(lengthscale=ls_cfg)
            base_kernel.raw_lengthscale.requires_grad = True
        elif cfg.TRAINER.ADAPTER.GP_KERNEL_TYPE.lower() == "linear":
            base_kernel = gpytorch.kernels.LinearKernel(batch_shape=batch_shape)
        else:
            raise ValueError(f"Unsupported kernel: {cfg.TRAINER.ADAPTER.GP_KERNEL_TYPE}")

        self.covar_module = gpytorch.kernels.ScaleKernel(
            base_kernel, batch_shape=batch_shape
        )

        # Break symmetry: initialise variational mean with small noise
        with torch.no_grad():
            vm = self.variational_strategy._variational_distribution.variational_mean
            vm.normal_(mean=0.0, std=0.1)

        # Register the (fixed) template embeddings for downstream use.
        self.register_buffer("_templates", text_embeddings.detach())

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

    def forward_and_kl(self, *, use_mean: bool = False):
        """
        Return class prototypes and KL term.

        Args
        ----
        use_mean:  Use posterior mean instead of MC sampling (deterministic).
        """
        q = self.variational_strategy(self._templates.to(torch.float32))  # MultivariateNormal (batch)

        # Enforce minimal variance to avoid full collapse
        var_floor = 1e-4
        try:
            # when cfg available via __dict__ (constructor not storing), just ignore
            var_floor = self.cfg.TRAINER.ADAPTER.GP_VAR_FLOOR  # type: ignore
        except Exception:
            pass
        with torch.no_grad():
            # Clamp the variance in-place to avoid collapse
            # q.variance is a property, so we can't assign to q._variance
            # Instead, we clamp the underlying lazy tensor if possible
            # This is a workaround for the AttributeError
            if hasattr(q, "lazy_covariance_matrix"):
                cov = q.lazy_covariance_matrix
                # Clamp the diagonal to at least var_floor
                diag = cov.diagonal(dim1=-2, dim2=-1)
                clamped_diag = diag.clamp(min=var_floor)
                # Only possible to set if it's a tensor, not a lazy tensor
                # So we skip in-place modification and just warn if variance is too low
                # (see below for warning)
                # If you want to enforce the floor, you must do so in the kernel or variational distribution
                # Here, we just warn if variance is too low
                pass

        if self.training and not use_mean:
            # Draw *one* stochastic sample for each class during training so that every forward pass receives a fresh prototype realisation. This preserves the Monte-Carlo nature assumed by the ELBO without collapsing it into the mean of several samples.
            alpha = q.rsample()  # Shape: [K, M]

            # Handle potential (M, K) swap originating from iut shape
            if alpha.dim() == 2 and alpha.size(0) == self.num_templates and alpha.size(1) == self.num_classes:
                # Transpose to [K, M]
                alpha = alpha.t()

            w = F.softmax(alpha, dim=-1)  # [K, M]
        else:
            mu = q.mean
            # Detect and fix swapped dims [M, K] -> [K, M]
            if mu.size(0) == self.num_templates and mu.size(1) == self.num_classes:
                mu = mu.t()

            w = F.softmax(mu, dim=-1)  # [K, M]

        # Compute prototypes using fp32 representations of templates
        prototypes = torch.einsum("km,kmd->kd", w, self._templates.float())

        # Return in the original dtype/device so downstream CLIP code stays unchanged
        prototypes = prototypes.to(dtype=self.orig_dtype, device=self.orig_device)
        kl = self.variational_strategy.kl_divergence().sum() / self.num_classes

        # Warn if posterior variance collapses too much
        if self.training:
            with torch.no_grad():
                var_mean = q.variance.mean()
                if var_mean < 1e-3:
                    print("[WARN] GP variance < 1e-3 – consider stronger KL or lower GP_LR")

        return prototypes, kl
    
    def get_weight_distribution(self):
        return self.variational_strategy(self._templates.to(torch.float32))

    def sample_prototypes(self, num_samples: int, *, use_mean: bool = False) -> torch.Tensor:
        """
        Draw *num_samples* sets of template-weighted class prototypes.

        Returns
        -------
        Tensor
            A tensor of shape ``[S, K, D]`` where *S* is ``num_samples``,
            *K* the number of classes and *D* the embedding dimension.
        """
        q = self.get_weight_distribution()

        if use_mean:
            w = torch.softmax(q.mean, dim=-1).unsqueeze(0)  # [1,K,M]
        else:
            alpha = q.rsample(torch.Size([num_samples]))  # [S,K,M]
            w = torch.softmax(alpha, dim=-1)              # [S,K,M]

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
        and test points, hence N = 2 M) we broadcast the **average** bias of
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