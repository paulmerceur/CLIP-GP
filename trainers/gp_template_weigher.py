import torch
import torch.nn.functional as F
import gpytorch
from typing import Any, Optional, cast


class GaussianProcessTemplateWeighter(gpytorch.models.ApproximateGP):
    """
    GP-based template weighter that leans entirely on GPyTorch's built-ins.
    """

    def __init__(self, text_embeddings: torch.Tensor, cfg: Any, **kwargs) -> None:
        # Keep original dtype/device for later but run GP in fp32 for numerical stability (GPyTorch is primarily tested in fp32).
        self.orig_dtype = text_embeddings.dtype
        self.orig_device = text_embeddings.device
        text_embeddings_fp32 = text_embeddings.to(dtype=torch.float32)
        
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
        w = weights_km.to(device=self._templates.device, dtype=torch.float32)
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
        return proto_s.to(dtype=self.orig_dtype, device=self.orig_device), kl
    
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
        qf = self(self._templates.to(torch.float32))
        # Stochastic sampling of latent function values
        f_samples = qf.rsample(torch.Size([num_samples]))  # [S,K,M]
        # Map function values to convex weights per class/template
        w = torch.softmax(f_samples, dim=-1)  # [S,K,M]

        # Compute prototypes in fp32 then cast back to CLIP precision
        prototypes = torch.einsum("skm,kmd->skd", w, self._templates.float())
        templates_tensor = cast(torch.Tensor, self._templates)
        return prototypes.to(templates_tensor)

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
        f_dist = self(self._templates.to(torch.float32))
        f_mean = f_dist.mean  # [K,M]
        w = torch.softmax(f_mean, dim=-1)  # [K,M]
        prototypes = torch.einsum("km,kmd->kd", w, self._templates.float())  # [K,D]
        templates_tensor = cast(torch.Tensor, self._templates)
        return prototypes.to(templates_tensor)

    def fit_targets(self, targets: torch.Tensor, epochs: int = 300, lr: float = 1e-2, weight_decay: float = 0.0, verbose: bool = True) -> None:
        """Fit the batched GP to per-template targets of shape [K, M].

        Optimizes the variational ELBO with a Gaussian likelihood.
        """
        # Validate shapes
        if targets.dim() != 2:
            raise ValueError("targets must have shape [num_classes, num_templates]")
        if int(targets.shape[0]) != int(self.num_classes) or int(targets.shape[1]) != int(self.num_templates):
            raise ValueError("targets shape must match [K, M] of stored templates")

        device = self._templates.device
        self.to(device=device)
        self.likelihood.to(device=device)

        self.train()
        self.likelihood.train()

        # Separate model vs likelihood parameters to avoid overlaps
        lik_params = list(self.likelihood.parameters())
        lik_param_ids = {id(p) for p in lik_params}
        model_params = [p for p in self.parameters() if id(p) not in lik_param_ids]
        optimizer = torch.optim.Adam([
            {"params": model_params, "lr": lr, "weight_decay": weight_decay},
            {"params": lik_params, "lr": lr, "weight_decay": 0.0},
        ])
        mll = gpytorch.mlls.VariationalELBO(self.likelihood, self, num_data=int(self.num_templates))

        y = targets.to(device=device, dtype=torch.float32)
        x = self._templates.to(device=device, dtype=torch.float32)

        if verbose:
            try:
                print(f"[GP] Prefit: x={tuple(x.shape)} y={tuple(y.shape)} K={self.num_classes} M={self.num_templates} device={x.device}")
            except Exception:
                pass
        for ep in range(int(epochs)):
            optimizer.zero_grad(set_to_none=True)
            output = self(x)
            elbo = mll(output, y)  # shape [K]
            loss = (-elbo).sum()
            loss.backward()
            optimizer.step()
            if verbose and ((ep + 1) % max(1, int(epochs) // 5) == 0 or ep == 0):
                try:
                    print(f"[GP] Prefit ep {ep+1}/{epochs} loss={float(loss.detach().item()):.4f} elbo_mean={float(elbo.mean().detach().item()):.4f}")
                except Exception:
                    pass



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