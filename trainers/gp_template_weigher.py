import torch
import torch.nn as nn
import torch.nn.functional as F
import gpytorch
from typing import Any, Optional


class GaussianProcessTemplateWeighter(gpytorch.models.ApproximateGP):
    """
    Simplified GP-based template weighter that leans entirely on GPyTorch's built-ins.

    * No hand-rolled KL, priors, or extra regularisation.
    * One batched variational GP (one task == one class).
    * Templates are the *inducing* points - nothing to learn there.
    """

    def __init__(self, text_embeddings: torch.Tensor, cfg: Any, mean_init: Optional[torch.Tensor] = None, **kwargs) -> None:
        # Keep original dtype/device for later but run GP in fp32 for numerical stability (GPyTorch is primarily tested in fp32).
        self.orig_dtype = text_embeddings.dtype
        self.orig_device = text_embeddings.device
        text_embeddings_fp32 = text_embeddings.to(dtype=torch.float32)

        self.num_classes, self.num_templates, self.dim = text_embeddings_fp32.shape
        
        # Handle both old cfg format and new config format
        def get_config_value(key, default):
            if hasattr(cfg, 'TRAINER'):
                # Old Dassl config format
                return getattr(cfg.TRAINER.ADAPTER, key, default)
            else:
                # New config format - convert key to lowercase format
                if key == 'GP_NUM_MC_SAMPLES':
                    return getattr(cfg.adapter, 'gp_num_mc_samples', default)
                elif key == 'GP_KERNEL_TYPE':
                    return getattr(cfg.adapter, 'gp_kernel_type', default)
                else:
                    return default
        
        self.num_mc_samples = get_config_value('GP_NUM_MC_SAMPLES', 5)

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

        # Break symmetry: initialise variational mean with small noise
        with torch.no_grad():
            vm = self.variational_strategy._variational_distribution.variational_mean
            vm.normal_(mean=0.0, std=0.1)

        # Register the (fixed) template embeddings for downstream use.
        self.register_buffer("_templates", text_embeddings.detach())

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        # Ensure mean_x is a tensor to satisfy type checker
        if not isinstance(mean_x, torch.Tensor):
            raise TypeError("Mean module must return a tensor")
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

    def forward_and_kl(self):
        """
        Return class prototypes and KL term.
        
        Always uses stochastic sampling for maximum stochasticity.
        """
        q = self.variational_strategy(self._templates.to(torch.float32))  # MultivariateNormal (batch)

        # Note: Variance floor enforcement would require kernel-level modifications in GPyTorch
        # For now, we rely on the initialization and regularization to prevent collapse

        # Always draw stochastic samples for maximum stochasticity  
        alpha = q.rsample()  # Shape should be [K, M]

        # Ensure correct shape [K, M] (classes × templates)
        if alpha.size(0) == self.num_templates and alpha.size(1) == self.num_classes:
            alpha = alpha.t()  # Transpose to [K, M]
            
        # Use raw alpha values as template weights
        w = alpha  # [K, M]

        # Compute prototypes using fp32 representations of templates
        prototypes = torch.einsum("km,kmd->kd", w, self._templates.float())

        # Return in the original dtype/device so downstream CLIP code stays unchanged
        prototypes = prototypes.to(dtype=self.orig_dtype, device=self.orig_device)
        # Return *sum* of KL across classes; scaling will be handled in the
        # training loop so that it can be normalised per-image.
        kl = self.variational_strategy.kl_divergence().sum()

        # Warn if posterior variance collapses too much
        if self.training:
            with torch.no_grad():
                var_mean = q.variance.mean()
                if var_mean < 1e-3:
                    print("[WARN] GP variance < 1e-3 – consider stronger KL or lower GP_LR")

        return prototypes, kl
    
    def get_weight_distribution(self):
        return self.variational_strategy(self._templates.to(torch.float32))

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
        q = self.get_weight_distribution()

        # Always use stochastic sampling
        alpha = q.rsample(torch.Size([num_samples]))  # [S,K,M]
        w = alpha  # [S,K,M] - use raw alpha values as weights

        # Compute prototypes in fp32 then cast back to CLIP precision
        prototypes = torch.einsum("skm,kmd->skd", w, self._templates.float())
        # Ensure device is properly typed
        templates_device = self._templates.device
        if isinstance(templates_device, torch.device):
            target_device = templates_device
        else:
            target_device = self.orig_device
        return prototypes.to(dtype=self.orig_dtype, device=target_device)

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