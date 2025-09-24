import torch
import torch.nn.functional as F
import gpytorch
from typing import Any, Optional, cast


class GaussianProcessTemplateWeighter(gpytorch.models.ApproximateGP):
    """
    GP-based template weighter that leans entirely on GPyTorch's built-ins.
    """

    def __init__(self, text_embeddings: torch.Tensor, cfg: Any, **kwargs) -> None:
        self.orig_device = text_embeddings.device
        text_embeddings_fp32 = text_embeddings
        
        self.num_classes, self.num_templates, self.dim = text_embeddings_fp32.shape
        
        # Handle both old cfg format and new config format
        def get_config_value(key, default):
            # New config format - convert key to lowercase format
            if key == 'GP_NUM_MC_SAMPLES':
                return getattr(cfg.adapter, 'gp_num_mc_samples', default)
            elif key == 'GP_KERNEL_TYPE':
                return getattr(cfg.adapter, 'gp_kernel_type', default)
            else:
                return default
        
        self.num_mc_samples = get_config_value('GP_NUM_MC_SAMPLES', 5)

        batch_shape = torch.Size([self.num_classes])  # one GP per class

        # Use the template encodings as inducing points.
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

        # Initialise the per-template mean
        self.mean_module = PerTemplateMean(self.num_classes, self.num_templates)
        with torch.no_grad():
            class_mean = text_embeddings_fp32.mean(dim=1, keepdim=True)  # [K,1,D]
            text_norm = F.normalize(text_embeddings_fp32, p=2, dim=-1)   # [K,M,D]
            class_mean_norm = F.normalize(class_mean, p=2, dim=-1)       # [K,1,D]
            mean_init = (text_norm * class_mean_norm).sum(-1)            # [K,M]
        self.mean_module.mean_param.data = mean_init

        kernel_type = get_config_value('GP_KERNEL_TYPE', 'rbf').lower()
        if kernel_type == "rbf":
            with torch.no_grad():
                flat_emb = F.normalize(text_embeddings_fp32.reshape(-1, self.dim), p=2, dim=-1)  # [(K*M), D]
                pdist = torch.cdist(flat_emb, flat_emb)
                # Exclude the zero diagonal before taking the median
                ls_cfg = pdist[pdist > 0].median().item()
            print(f"[GP] Auto length-scale (normalised median): {ls_cfg:.4f}")

            base_kernel = gpytorch.kernels.RBFKernel(batch_shape=batch_shape, ard_num_dims=self.dim)
            base_kernel.initialize(lengthscale=ls_cfg)
            # Fix linter issue: set requires_grad through parameter
            base_kernel.raw_lengthscale.requires_grad_(True)
        elif kernel_type == "linear":
            base_kernel = gpytorch.kernels.LinearKernel(batch_shape=batch_shape)
        else:
            raise ValueError(f"Unsupported kernel: {kernel_type}")

        self.covar_module = gpytorch.kernels.ScaleKernel(
            base_kernel, batch_shape=batch_shape
        )
        # Gaussian likelihood for supervised regression on per-template targets
        self.likelihood = gpytorch.likelihoods.GaussianLikelihood(batch_shape=batch_shape)

        # Break symmetry: initialise variational mean with small noise
        with torch.no_grad():
            vm = self.variational_strategy._variational_distribution.variational_mean
            vm.normal_(mean=0.0, std=0.1)

        # Register the (fixed) template embeddings for downstream use.
        self._templates: torch.Tensor
        self.register_buffer("_templates", text_embeddings.detach())

    @torch.no_grad()
    def initialize_from_weights(self, weights_km: torch.Tensor, temperature: float = 1.0) -> None:
        """Initialize GP logits from per-class template weights.

        Parameters
        ----------
        weights_km: torch.Tensor
            Tensor of shape [K, M] with nonnegative weights summing to 1 per class.
        temperature: float
            Temperature to scale the initialization logits; >1 makes them softer.
        """
        w = weights_km.to(device=self._templates.device)
        w = torch.clamp(w, min=1e-12)
        f_init = torch.log(w) / max(float(temperature), 1e-6)  # [K, M]
        try:
            # Initialize mean module
            if hasattr(self, "mean_module") and hasattr(self.mean_module, "mean_param"):
                self.mean_module.mean_param.data.copy_(f_init)
        except Exception:
            pass
        try:
            # Initialize variational mean
            vm = self.variational_strategy._variational_distribution.variational_mean
            vm.data.copy_(f_init)
        except Exception:
            pass

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        if not isinstance(mean_x, torch.Tensor):
            raise TypeError("Mean module must return a tensor")

        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

    def forward_and_kl(self):
        """
        Return class prototypes and KL term for a single stochastic sample.
        """
        proto_s = self.sample_prototypes(num_samples=1).squeeze(0)  # [K,D]
        kl = self.variational_strategy.kl_divergence().sum()
        return proto_s.to(device=self.orig_device), kl
    
    def sample_prototypes(self, num_samples: int) -> torch.Tensor:
        """
        Draw *num_samples* sets of template-weighted class prototypes.
        
        Always uses stochastic sampling for maximum stochasticity.

        Returns
        -------
        Tensor
            A tensor of shape ``[S, K, D]`` where *S* is ``num_samples``,
            *K* the number of classes and *D* the embedding dimension.
        """
        # Distribution over latent function values at template inputs
        qf = self(self._templates)
        # Stochastic sampling of latent function values
        f_samples = qf.rsample(torch.Size([num_samples]))  # [S,K,M]
        # Map function values to convex weights per class/template
        w = torch.softmax(f_samples, dim=-1)  # [S,K,M]

        # Compute prototypes in fp32 then cast back to CLIP precision
        prototypes = torch.einsum("skm,kmd->skd", w, self._templates)
        return prototypes

    @torch.no_grad()
    def prototypes_from_posterior_mean(self) -> torch.Tensor:
        """Deterministic prototypes using posterior mean weights.

        Returns a tensor of shape [K, D].
        """
        self.eval()
        try:
            self.likelihood.eval()
        except Exception:
            pass
        f_dist = self(self._templates)
        f_mean = f_dist.mean  # [K,M]
        w = torch.softmax(f_mean, dim=-1)  # [K,M]
        prototypes = torch.einsum("km,kmd->kd", w, self._templates)  # [K,D]
        return prototypes


class PerTemplateMean(gpytorch.means.Mean):
    """Learnable mean of shape [K, M] (one bias per class & template)."""

    def __init__(self, num_classes: int, num_templates: int):
        super().__init__()
        # One scalar per (class, template) pair; values will be set from mean_init
        self.mean_param = torch.nn.Parameter(torch.empty(num_classes, num_templates))

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