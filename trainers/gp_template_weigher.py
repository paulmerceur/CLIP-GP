import torch
import torch.nn.functional as F
import gpytorch
from entmax import sparsemax
from typing import Any


class GaussianProcessTemplateWeighter(gpytorch.models.ApproximateGP):
    """
    GP-based template weighter that leans entirely on GPyTorch's built-ins.
    """

    def __init__(self, text_embeddings: torch.Tensor, cfg: Any, **kwargs) -> None:
        self.orig_device = text_embeddings.device

        # self.num_classes, self.num_templates, self.dim = text_embeddings_fp32.shape
        self.num_classes, self.num_templates, self.dim = text_embeddings.shape

        # Remove these if don't work
        K, M, D = self.num_classes, self.num_templates, self.dim

        # PCA
        self.input_dim = D
        self.red_dim = int(getattr(cfg.adapter, 'gp_pca_dim', 128))

        with torch.no_grad():
            X = text_embeddings.reshape(-1, self.input_dim)  # (K*M, D)
            mu = X.mean(dim=0, keepdim=True)  # (1, D)
            X_centered = X - mu
            U, S, Vt = torch.linalg.svd(X_centered, full_matrices=False)

            max_rank = Vt.shape[0]
            self.red_dim = min(self.red_dim, max_rank)
            
            W = Vt[:self.red_dim].T  # (D, d)
        self._pca_mean = mu.squeeze(0)  # (D,)
        self._pca_W = W  # (D, d)


        def _project(x): # x: [*, D] -> [*, d]
            return (x - self._pca_mean) @ self._pca_W
        def _lift(z): # z: [*, d] -> [*, D]
            return z @ self._pca_W.T + self._pca_mean
        
        self._project = _project
        self._lift = _lift

        with torch.no_grad():
            cls_mean = text_embeddings.mean(dim=1, keepdim=True)  # [K,1,D]

        templates_red = self._project(text_embeddings.view(-1, self.input_dim)).view(K, M, self.red_dim)  # [K, M, d]
        cls_mean_red = self._project(cls_mean.view(-1, self.input_dim)).view(K, 1, self.red_dim)  # [K, 1, d]

        batch_shape = torch.Size([self.num_classes])  # one GP per class
        self.scores = None

        # Use the template encodings as inducing points.
        # inducing_inputs = text_embeddings.detach().clone().view(self.num_classes, self.num_templates, self.dim)
        # inducing_inputs = torch.cat([inducing_inputs, cls_mean], dim=1)  # [K, M+1, D]
        N_learnable_tokens = 1
        inducing_inputs = torch.cat([templates_red, cls_mean_red], dim=1)  # [K, M+N_learnable_tokens, d]
        variational_dist = gpytorch.variational.CholeskyVariationalDistribution(num_inducing_points=self.num_templates+N_learnable_tokens,batch_shape=batch_shape,)
        variational_strategy = gpytorch.variational.VariationalStrategy(self, inducing_inputs, variational_dist, learn_inducing_locations=True)

        super().__init__(variational_strategy)

        # Learnable kernel specific linear map
        self.A = torch.nn.Linear(self.red_dim, self.red_dim, bias=False)
        with torch.no_grad():
            self.A.weight.copy_(torch.eye(self.red_dim, device=text_embeddings.device, dtype=text_embeddings.dtype))

        # Zero grads on first M inducing
        mask = torch.zeros(K, M+N_learnable_tokens, self.red_dim, device=inducing_inputs.device)
        mask[:, M:, :] = 1.0
        self.register_buffer("_ind_mask", mask)
        def _freeze_templates_hook(grad):
            return grad * self._ind_mask
        
        self.variational_strategy.inducing_points.register_hook(_freeze_templates_hook)

        # Initialise the per-template mean
        #self.mean_module = PerTemplateMean(self.num_classes, self.num_templates)
        with torch.no_grad():
            class_mean = text_embeddings.mean(dim=1, keepdim=True)  # [K,1,D]
            text_norm = F.normalize(text_embeddings, p=2, dim=-1)   # [K,M,D]
            class_mean_norm = F.normalize(class_mean, p=2, dim=-1)       # [K,1,D]
            mean_init = (text_norm * class_mean_norm).sum(-1)            # [K,M]
        #self.mean_module.mean_param.data = mean_init

        tau = float(getattr(cfg.adapter, 'gp_prior_temp', 1.0) or 1.)  # optional temp
        with torch.no_grad():
            # mean_init is cosine(template, class_center)  [K, M]
            prior_logits = mean_init / max(tau, 1e-6)                 # [K, M]
            w0 = torch.softmax(prior_logits, dim=-1).clamp_min(1e-12) # zero-shot template prior
            f0 = torch.log(w0)                                        # logits space for GP

            # self.mean_module.mean_param.copy_(f0.to(torch.float32))
        self.mean_module = ResidualMeanWithBias(f0_logits=f0.to(torch.float32))
        # self.mean_module = ContextAwareMean(f0_logits=f0.to(torch.float32), D=self.dim)

        kernel_type = getattr(cfg.adapter, 'gp_kernel_type', 'rbf')
        if kernel_type == "rbf":
            with torch.no_grad():
                flat_emb = F.normalize(templates_red.reshape(-1, self.red_dim), p=2, dim=-1)  # [(K*M), d]
                pdist = torch.cdist(flat_emb, flat_emb)
                # Exclude the zero diagonal before taking the median
                ls_cfg = pdist[pdist > 0].median().item()
            print(f"[GP] Auto length-scale (normalised median): {ls_cfg:.4f}")

            base_kernel = gpytorch.kernels.RBFKernel(batch_shape=batch_shape, ard_num_dims=self.red_dim)
            base_kernel.initialize(lengthscale=ls_cfg)
            # Fix linter issue: set requires_grad through parameter
            base_kernel.raw_lengthscale.requires_grad_(True)
        elif kernel_type == "matern":
            base_kernel = gpytorch.kernels.MaternKernel(nu=0.5, batch_shape=batch_shape, ard_num_dims=self.red_dim)
        elif kernel_type == "linear":
            base_kernel = gpytorch.kernels.LinearKernel(batch_shape=batch_shape)
        else:
            raise ValueError(f"Unsupported kernel: {kernel_type}")

        self.covar_module = gpytorch.kernels.ScaleKernel(base_kernel, batch_shape=batch_shape)
        # Gaussian likelihood for supervised regression on per-template targets
        self.likelihood = gpytorch.likelihoods.GaussianLikelihood(batch_shape=batch_shape)

        # Register the (fixed) template embeddings for downstream use.
        self._templates: torch.Tensor
        self.register_buffer("_templates", text_embeddings.detach())
        self.register_buffer("_templates_red", templates_red.detach())
        self.register_buffer("_cls_mean_init", cls_mean)

    def _to_kernel_space(self, x_red):
        z = self.A(x_red)
        z = torch.nn.functional.normalize(z, p=2, dim=-1)
        return z
    
    @torch.no_grad()
    def initialize_from_weights(self, weights_km: torch.Tensor, temperature: float = 1.0) -> None:
        """Initialize GP logits from per-class template weights.

        Parameters
        ----------
        weights_km: torch.Tensor
            Tensor of shape [K, M] or [1, M] with nonnegative weights summing to 1 per class.
        temperature: float
            Temperature to scale the initialization logits; >1 makes them softer.
        """
        w = weights_km.to(device=self._templates.device)
        K = self.num_classes
        # If sharing weights across classes, broadcast from (1, M) to (K, M)
        if w.shape[0] == 1 and K > 1:
            w = w.expand(K, -1)
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
        # z = self._to_kernel_space(x)
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
        kl = self.variational_strategy.kl_divergence().sum() + self.reg
        return proto_s.to(device=self.orig_device), kl
    
    def sample_prototypes(self, num_samples: int, visual_embeddings: torch.Tensor = None) -> torch.Tensor:
        """
        Draw *num_samples* sets of template-weighted class prototypes.

        Returns
        -------
        Tensor
            A tensor of shape ``[S, K, D]`` where *S* is ``num_samples``,
            *K* the number of classes and *D* the embedding dimension.
        """
        # Use visual embeddings as context -> [B, D] where for training B=K classes and testing B=batch size != K
        N_templates = self._templates_red.shape[1]
        # visual_token = self.variational_strategy.inducing_points[:, -1, :]  # [K, D]
        # visual_token = F.normalize(visual_token, p=2, dim=-1).unsqueeze(1)  # [K, 1, D]

        if (visual_embeddings is not None) and (visual_embeddings.shape[0] == self.num_classes):
            # visual_embeddings = visual_embeddings.unsqueeze(1)
            visual_embeddings = self._project(visual_embeddings).unsqueeze(1)  # [K, 1, d]
            
            # gp_input = torch.cat([self._templates, visual_token, visual_embeddings], dim=1)  # [K, M+1, d]
            gp_input = torch.cat([self._templates_red, visual_embeddings], dim=1)  # [K, M+1, d]
        else:
            # Test time: no extra context
            # visual_mean = visual_embeddings.mean(dim=0, keepdim=True)  # [1, 1, D]
            # visual_mean = visual_mean.expand(self.num_classes, -1, -1)  # [K, 1, D]
            # gp_input = torch.cat([self._templates_reduced, visual_mean], dim=1)  # [K, M+1, D]
            # print(f"Shape of gp_input at test time: {gp_input.shape}")
            gp_input = self._templates_red  # [K, M, D]

        # Distribution over latent function values at template inputs
        qf = self(gp_input)
        # Stochastic sampling of latent function values
        f_samples = qf.rsample(torch.Size([num_samples]))[:, :, :N_templates]  # [S,K,M]
        # Map function values to convex weights per class/template
        w = sparsemax(f_samples, dim=-1)  # [S,K,M]

        self.scores = w
        
        prototypes = torch.einsum("skm,kmd->skd", w, self._templates_red)
        prototypes = self._lift(prototypes)
        return prototypes

class ResidualMeanWithBias(gpytorch.means.Mean):
    def __init__(self, f0_logits: torch.Tensor):  # [K,M] frozen prior (log w0)
        super().__init__()
        K, M = f0_logits.shape
        self.register_buffer('f0', f0_logits.clone())       # frozen prior
        self.cls_bias = torch.nn.Parameter(torch.zeros(K, 1))
        self.tmp_bias = torch.nn.Parameter(torch.zeros(1, M))

    def forward(self, x):
        K, M = self.f0.shape
        N = x.size(-2)
        # Ensure device/dtype alignment with inputs
        f0 = self.f0.to(device=x.device, dtype=self.cls_bias.dtype)
        base = f0 + self.cls_bias + self.tmp_bias       # [K,M]
        if N == M:
            return base
        extra = N - M
        tail = (self.cls_bias + self.tmp_bias.mean(dim=1, keepdim=True)).expand(K, extra)
        return torch.cat([base, tail], dim=1)                # [K,N]


class ContextAwareMean(gpytorch.means.Mean):
    def __init__(self, f0_logits: torch.Tensor, D: int):
        super().__init__()
        K, M = f0_logits.shape
        self.register_buffer('f0', f0_logits.clone())
        self.cls_bias = torch.nn.Parameter(torch.zeros(K,1))
        self.W = torch.nn.Linear(D, 1, bias=False)  # projects context feature to a scalar

    def forward(self, x):
        # x shape: [K, N, D]; last column may be your class context feature
        K, M = self.f0.shape
        N = x.size(-2)
        base = self.f0 + self.cls_bias
        if N == M:
            return base
        ctx = x[:, -1, :]                 # [K, D] the extra context point
        delta = self.W(ctx)               # [K,1] per-class shift from context
        tail = (self.cls_bias + delta)    # for the extra point(s)
        return torch.cat([base, tail.expand(K, N - M)], dim=1)


class LinearFeatMean(gpytorch.means.Mean):
    def __init__(self, in_dim):
        super().__init__()
        self.beta = torch.nn.Parameter(torch.zeros(in_dim))
    def forward(self, feats):  # feats = φ(x)
        return feats @ self.beta


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