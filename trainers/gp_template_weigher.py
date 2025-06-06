from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import MultivariateNormal
from torch.distributions.kl import kl_divergence

import gpytorch
from gpytorch.kernels import RBFKernel, LinearKernel, ScaleKernel


class GaussianProcessTemplateWeighter(nn.Module):
    r"""Variational GP weighter for CLIP templates.

    Each class‑specific weight vector :math:`\alpha_k \in \mathbb{R}^M`
    is the evaluation of a latent GP defined over template embeddings.
    A Gaussian variational posterior :math:`q(\alpha_k)` is fitted by
    maximising the ELBO:

    .. math::
        \mathcal{L}
        = \mathbb{E}_{q}[\log p(y\mid\alpha)]
          - \beta \;\mathrm{KL}\bigl[q(\alpha)\,\|\,p(\alpha)\bigr].

    The likelihood term is handled upstream (cross‑entropy).  This module
    returns both the softmax‑weighted prototypes **and** the analytic KL
    term so it can be added to the loss.
    """

    def __init__(
        self,
        num_classes: int,
        num_templates: int,
        embedding_dim: int = 512,
        *,
        kernel_type: str = "rbf",
        lengthscale: float = 1.0,
        outputscale: float = 1.0,
        noise_var: float = 1e-4,
        num_mc_samples: int = 5,
        use_diagonal_cov: bool = True,
    ) -> None:
        super().__init__()
        self.K = num_classes
        self.M = num_templates
        self.D = embedding_dim
        self.num_mc = num_mc_samples
        self.kernel_type = kernel_type.lower()
        self.use_diag = use_diagonal_cov

        # ---- Fixed GP hyper‑parameters (buffers so they move with .to()) ---- #
        self.register_buffer("lengthscale", torch.tensor(lengthscale))
        self.register_buffer("outputscale", torch.tensor(outputscale))
        self.register_buffer("noise_var", torch.tensor(noise_var))

        # ---- Variational parameters (learnable) ---------------------------- #
        self.variational_mean = nn.Parameter(torch.zeros(num_classes, num_templates))

        if self.use_diag:
            # Diagonal covariance Σ = diag(σ²); store log σ² for stability.
            self.log_var = nn.Parameter(torch.full((num_classes, num_templates), -4.0))
        else:
            # Full covariance via scale‑tril (Cholesky factor).
            init = 0.05 * torch.eye(num_templates).unsqueeze(0)
            self.chol = nn.Parameter(init.repeat(num_classes, 1, 1))

    # --------------------------------------------------------------------- #
    #  Helpers                                                               #
    # --------------------------------------------------------------------- #
    def _kernel(self) -> gpytorch.kernels.Kernel:
        if self.kernel_type == "rbf":
            base = RBFKernel(ard_num_dims=self.D)
            base.lengthscale = self.lengthscale
        elif self.kernel_type == "linear":
            base = LinearKernel()
        else:
            raise ValueError(f"Unsupported kernel: {self.kernel_type}")
        return ScaleKernel(base, outputscale=self.outputscale)

    def _prior_cov(self, f: torch.Tensor) -> torch.Tensor:
        """Compute batched prior covariance *K_k* (adds noise variance)."""
        kern = self._kernel().to(f)
        covs = []
        eye = torch.eye(self.M, device=f.device, dtype=f.dtype)
        for k in range(self.K):
            Kk = kern(f[k], f[k]).evaluate() + self.noise_var * eye
            covs.append(Kk)
        return torch.stack(covs)  # [K, M, M]

    def _q_dist(self) -> MultivariateNormal:
        if self.use_diag:
            var = torch.exp(self.log_var)                       # [K,M]
            cov = torch.diag_embed(var)                         # [K,M,M]
            return MultivariateNormal(self.variational_mean, covariance_matrix=cov)
        L = torch.tril(self.chol)
        return MultivariateNormal(self.variational_mean, scale_tril=L)

    # --------------------------------------------------------------------- #
    #  Forward                                                               #
    # --------------------------------------------------------------------- #
    def forward(
        self,
        text_embeddings: torch.Tensor,  # [K, M, D]
        return_kl: bool = True,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Combine templates into class prototypes.

        Args
        ----
        text_embeddings:
            Tensor ``[K, M, D]`` of template CLIP embeddings.
        return_kl:
            Whether to also return KL[q‖p] for the ELBO.

        Returns
        -------
        prototypes:
            Tensor ``[K, D]`` weighted class prototypes.
        kl_divergence:
            Scalar (or *None*) analytic KL term.
        """
        f = text_embeddings.float()  # safer numerics
        q = self._q_dist()

        # ---- KL divergence ------------------------------------------------ #
        if return_kl:
            prior_cov = self._prior_cov(f)
            p = MultivariateNormal(loc=f.new_zeros(self.K, self.M), covariance_matrix=prior_cov)
            kl = kl_divergence(q, p).sum()  # scalar
        else:
            kl = None

        # ---- Monte‑Carlo combine ------------------------------------------ #
        alpha = q.rsample((self.num_mc,))          # [S,K,M]
        w = F.softmax(alpha, dim=-1)               # weights on Δ^{M-1}
        # einsum:   samples S, class K, templates M, dim D
        proto = torch.einsum("skm,kmd->skd", w, f).mean(0)  # [K,D]

        return proto.to(text_embeddings.dtype), kl