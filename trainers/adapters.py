import os
import os.path as osp
import random
from re import template
import time
import os.path as osp
import datetime
import copy
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.cuda.amp import GradScaler, autocast
import numpy as np
import math

from dassl.engine import TRAINER_REGISTRY, SimpleTrainer
from dassl.metrics import compute_accuracy
from dassl.utils import load_pretrained_weights, load_checkpoint, AverageMeter, MetricMeter
from dassl.optim import build_optimizer, build_lr_scheduler

from clip import clip
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer
from datasets.imagenet_templates import IMAGENET_TEMPLATES, IMAGENET_TEMPLATES_SELECT

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.allow_tf32 = True

_tokenizer = _Tokenizer()


def reset_gpu_state():
    """Reset GPU state to avoid CUDA errors between runs."""
    try:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            # Reset CUDA context if needed
            torch.cuda.reset_peak_memory_stats()
            print("GPU state reset successfully")
    except Exception as e:
        print(f"Warning: Could not reset GPU state: {e}")


CUSTOM_TEMPLATES = {
    "OxfordPets": "a photo of a {}, a type of pet.",
    "OxfordFlowers": "a photo of a {}, a type of flower.",
    "FGVCAircraft": "a photo of a {}, a type of aircraft.",
    "DescribableTextures": "{} texture.",
    "EuroSAT": "a centered satellite photo of {}.",
    "StanfordCars": "a photo of a {}.",
    "Food101": "a photo of {}, a type of food.",
    "SUN397": "a photo of a {}.",
    "Caltech101": "a photo of a {}.",
    "UCF101": "a photo of a person doing {}.",
    "ImageNet": "a photo of a {}.",
    "ImageNetSketch": "a photo of a {}.",
    "ImageNetV2": "a photo of a {}.",
    "ImageNetA": "a photo of a {}.",
    "ImageNetR": "a photo of a {}.",
}


class GaussianProcessTemplateWeighter(nn.Module):
    """Gaussian Process Template Weighter with Variational Inference.
    
    Implements the GP framework from the specifications:
    - GP prior on weight vectors α_k ∈ ℝ^M for each class k
    - Variational approximation q(α_k) = N(μ_k, Σ_k)
    - ELBO objective with Monte Carlo sampling
    - Kernel-based covariance matrices
    """

    def __init__(
        self,
        num_classes: int,
        num_templates: int,
        embedding_dim: int = 512,
        kernel_type: str = "rbf",
        lengthscale: float = 1.0,
        outputscale: float = 1.0,
        noise_var: float = 1e-4,
        num_mc_samples: int = 5,
        use_diagonal_cov: bool = True,
    ):
        super().__init__()
        self.K = num_classes  # K classes
        self.M = num_templates  # M templates  
        self.D = embedding_dim
        self.kernel_type = kernel_type
        self.num_mc_samples = num_mc_samples
        self.use_diagonal_cov = use_diagonal_cov
        
        # GP prior parameters (fixed)
        self.register_buffer("lengthscale", torch.tensor(lengthscale))
        self.register_buffer("outputscale", torch.tensor(outputscale))
        self.register_buffer("noise_var", torch.tensor(noise_var))
        
        # Prior mean (zero mean function)
        self.register_buffer("prior_mean", torch.zeros(num_templates))
        
        # Variational parameters for q(α_k) = N(μ_k, Σ_k)
        # μ_k ∈ ℝ^M for each class k
        self.variational_mean = nn.Parameter(
            torch.randn(num_classes, num_templates) * 0.01
        )
        
        if use_diagonal_cov:
            # Diagonal covariance: log(σ²_k) for numerical stability
            self.variational_logvar = nn.Parameter(
                torch.full((num_classes, num_templates), -2.0)  # σ² ≈ 0.135
            )
        else:
            # Full covariance (lower triangular L such that Σ = LL^T)
            self.variational_chol = nn.Parameter(
                torch.eye(num_templates).unsqueeze(0).repeat(num_classes, 1, 1) * 0.1
            )
        
    def _compute_prior_covariance(self, text_embeddings):
        """Compute GP prior covariance K_k based on text embeddings.
        
        Args:
            text_embeddings: [K, M, D] - text features for computing kernel
            
        Returns:
            prior_cov: [K, M, M] - prior covariance matrices for each class
        """
        K, M, D = text_embeddings.shape
        
        if self.kernel_type == "rbf":
            # RBF kernel: k(f_i, f_j) = σ² * exp(-||f_i - f_j||² / (2 * l²))
            prior_cov = []
            for k in range(K):
                # Get embeddings for class k: [M, D]
                f_k = text_embeddings[k]  # [M, D]
                
                # Compute pairwise distances
                f_k_norm = f_k.norm(dim=1, keepdim=True)  # [M, 1]
                distances_sq = f_k_norm.pow(2) + f_k_norm.pow(2).t() - 2 * f_k @ f_k.t()
                distances_sq = torch.clamp(distances_sq, min=0.0)  # Numerical stability
                
                # RBF kernel
                K_k = self.outputscale * torch.exp(-distances_sq / (2 * self.lengthscale.pow(2)))
                
                # Add noise for numerical stability
                K_k = K_k + self.noise_var * torch.eye(M, device=K_k.device, dtype=K_k.dtype)
                
                prior_cov.append(K_k)
                
            return torch.stack(prior_cov)  # [K, M, M]
            
        elif self.kernel_type == "linear":
            # Linear kernel: k(f_i, f_j) = σ² * f_i^T f_j
            prior_cov = []
            for k in range(K):
                f_k = text_embeddings[k]  # [M, D]
                K_k = self.outputscale * (f_k @ f_k.t())
                K_k = K_k + self.noise_var * torch.eye(M, device=K_k.device, dtype=K_k.dtype)
                prior_cov.append(K_k)
            return torch.stack(prior_cov)  # [K, M, M]
            
        else:
            raise ValueError(f"Unsupported kernel type: {self.kernel_type}")
    
    def _sample_alpha(self, batch_size=None):
        """Sample α_k from variational distribution q(α_k) = N(μ_k, Σ_k).
        
        Args:
            batch_size: If None, return [K, M]. If int, return [batch_size, K, M]
            
        Returns:
            alpha_samples: Sampled weight vectors
        """
        if batch_size is None:
            batch_size = 1
            squeeze = True
        else:
            squeeze = False
            
        if self.use_diagonal_cov:
            # Diagonal covariance case
            std = torch.exp(0.5 * self.variational_logvar)  # [K, M]
            eps = torch.randn(batch_size, self.K, self.M, device=std.device, dtype=std.dtype)
            alpha_samples = self.variational_mean.unsqueeze(0) + eps * std.unsqueeze(0)
        else:
            # Full covariance case
            eps = torch.randn(batch_size, self.K, self.M, device=self.variational_mean.device, dtype=self.variational_mean.dtype)
            # Use lower triangular matrix to ensure positive definite covariance
            L = torch.tril(self.variational_chol)  # [K, M, M]
            alpha_samples = self.variational_mean.unsqueeze(0) + torch.bmm(
                L.unsqueeze(0).expand(batch_size, -1, -1, -1).contiguous().view(-1, self.M, self.M),
                eps.view(-1, self.M, 1)
            ).view(batch_size, self.K, self.M)
            
        if squeeze:
            alpha_samples = alpha_samples.squeeze(0)  # [K, M]
            
        return alpha_samples
    
    def _compute_kl_divergence(self, text_embeddings):
        """Compute KL divergence KL(q(α_k) || p(α_k)) for all classes.
        
        Args:
            text_embeddings: [K, M, D] - used to compute prior covariance
            
        Returns:
            kl_div: Scalar KL divergence summed over all classes
        """
        # Compute prior covariance matrices
        prior_cov = self._compute_prior_covariance(text_embeddings)  # [K, M, M]
        
        total_kl = 0.0
        
        for k in range(self.K):
            # Prior: N(0, K_k)
            prior_mean_k = self.prior_mean  # [M]
            prior_cov_k = prior_cov[k]  # [M, M]
            
            # Variational: N(μ_k, Σ_k)
            var_mean_k = self.variational_mean[k]  # [M]
            
            if self.use_diagonal_cov:
                var_cov_k = torch.diag(torch.exp(self.variational_logvar[k]))  # [M, M]
            else:
                L_k = torch.tril(self.variational_chol[k])  # [M, M]
                var_cov_k = L_k @ L_k.t()  # [M, M]
            
            # KL divergence between two multivariate Gaussians
            # KL(q||p) = 0.5 * [tr(Σ_p^-1 Σ_q) + (μ_p - μ_q)^T Σ_p^-1 (μ_p - μ_q) - k + log(det(Σ_p)/det(Σ_q))]
            
            try:
                # Cholesky decomposition for numerical stability
                L_prior = torch.linalg.cholesky(prior_cov_k)  # Lower triangular
                
                # Solve triangular systems
                mean_diff = var_mean_k - prior_mean_k  # [M]
                
                # Compute terms efficiently using Cholesky
                # tr(Σ_p^-1 Σ_q) = ||L_p^-1 L_q||_F^2 where Σ_q = L_q L_q^T
                if self.use_diagonal_cov:
                    var_std = torch.exp(0.5 * self.variational_logvar[k])  # [M]
                    # For diagonal case, solve directly without matrix operations
                    solved_std = torch.linalg.solve_triangular(L_prior, var_std.unsqueeze(-1), upper=False).squeeze(-1)
                    trace_term = (solved_std ** 2).sum()
                    logdet_var = 2 * torch.sum(torch.log(var_std))
                else:
                    L_var = torch.tril(self.variational_chol[k])
                    solved_var = torch.linalg.solve_triangular(L_prior, L_var, upper=False)
                    trace_term = (solved_var ** 2).sum()
                    logdet_var = 2 * torch.sum(torch.log(torch.diag(L_var)))
                
                # Mean term: (μ_p - μ_q)^T Σ_p^-1 (μ_p - μ_q)
                solved_mean = torch.linalg.solve_triangular(L_prior, mean_diff.unsqueeze(-1), upper=False).squeeze(-1)
                mean_term = (solved_mean ** 2).sum()
                
                # Log determinant terms
                logdet_prior = 2 * torch.sum(torch.log(torch.diag(L_prior)))
                
                # KL divergence
                kl_k = 0.5 * (trace_term + mean_term - self.M + logdet_prior - logdet_var)
                total_kl = total_kl + kl_k
                
            except Exception as e:
                # Fallback: add small regularization if Cholesky fails
                print(f"Warning: Cholesky failed for class {k}, using regularized computation: {e}")
                reg_prior_cov = prior_cov_k + 1e-3 * torch.eye(self.M, device=prior_cov_k.device, dtype=prior_cov_k.dtype)
                
                try:
                    L_prior = torch.linalg.cholesky(reg_prior_cov)
                    
                    mean_diff = var_mean_k - prior_mean_k
                    
                    if self.use_diagonal_cov:
                        var_std = torch.exp(0.5 * self.variational_logvar[k])
                        solved_std = torch.linalg.solve_triangular(L_prior, var_std.unsqueeze(-1), upper=False).squeeze(-1)
                        trace_term = (solved_std ** 2).sum()
                        logdet_var = 2 * torch.sum(torch.log(var_std))
                    else:
                        L_var = torch.tril(self.variational_chol[k])
                        solved_var = torch.linalg.solve_triangular(L_prior, L_var, upper=False)
                        trace_term = (solved_var ** 2).sum()
                        logdet_var = 2 * torch.sum(torch.log(torch.diag(L_var)))
                    
                    solved_mean = torch.linalg.solve_triangular(L_prior, mean_diff.unsqueeze(-1), upper=False).squeeze(-1)
                    mean_term = (solved_mean ** 2).sum()
                    
                    logdet_prior = 2 * torch.sum(torch.log(torch.diag(L_prior)))
                    
                    kl_k = 0.5 * (trace_term + mean_term - self.M + logdet_prior - logdet_var)
                    total_kl = total_kl + kl_k
                    
                except Exception as e2:
                    print(f"Warning: Even regularized Cholesky failed for class {k}: {e2}")
                    # Use simple L2 regularization as fallback
                    kl_k = 0.01 * (var_mean_k ** 2).sum()
                    if not self.use_diagonal_cov:
                        kl_k = kl_k + 0.01 * (self.variational_chol[k] ** 2).sum()
                    else:
                        kl_k = kl_k + 0.01 * (self.variational_logvar[k] ** 2).sum()
                    total_kl = total_kl + kl_k
        
        return total_kl
    
    def forward(self, text_embeddings, return_kl=True):
        """Apply GP-weighted template combination.
        
        Args:
            text_embeddings: [K, M, D] - text embeddings for all classes and templates
            return_kl: Whether to return KL divergence for ELBO loss
            
        Returns:
            weighted_embeddings: [K, D] - weighted text prototypes
            kl_divergence: Scalar KL divergence (if return_kl=True)
        """
        orig_dtype = text_embeddings.dtype
        text_embeddings = text_embeddings.float()  # Use fp32 for numerical stability
        
        K, M, D = text_embeddings.shape
        assert K == self.K and M == self.M
        
        # Sample α_k from variational distribution (Monte Carlo sampling)
        alpha_samples = self._sample_alpha(batch_size=self.num_mc_samples)  # [num_mc_samples, K, M]
        
        # Apply softmax to get normalized weights for each sample
        weights = F.softmax(alpha_samples, dim=-1)  # [num_mc_samples, K, M]
        
        # Compute weighted embeddings for each MC sample
        # weights: [num_mc_samples, K, M, 1], text_embeddings: [1, K, M, D]
        weighted_samples = []
        for s in range(self.num_mc_samples):
            weighted_s = (text_embeddings * weights[s].unsqueeze(-1)).sum(dim=1)  # [K, D]
            weighted_samples.append(weighted_s)
        
        # Average over MC samples
        weighted_embeddings = torch.stack(weighted_samples).mean(dim=0)  # [K, D]
        
        # Compute KL divergence for ELBO loss
        if return_kl:
            kl_divergence = self._compute_kl_divergence(text_embeddings)
            return weighted_embeddings.to(orig_dtype), kl_divergence
        else:
            return weighted_embeddings.to(orig_dtype)

def load_clip_to_cpu(cfg):
    backbone_name = cfg.MODEL.BACKBONE.NAME
    url = clip._MODELS[backbone_name]
    model_path = clip._download(url)

    try:
        # loading JIT archive
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None

    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu")

    model = clip.build_model(state_dict or model.state_dict())

    return model


class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    def forward(self, prompts, tokenized_prompts):
        x = prompts + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection

        return x


def _get_base_text_features(cfg, classnames, clip_model, text_encoder, pretrained_projection=None):
    device = next(text_encoder.parameters()).device
    
    # Add robust CUDA handling
    if clip_model.dtype == torch.float16:
        try:
            # Clear GPU cache first to avoid memory issues
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            
            # Check if we can safely move to CUDA
            if device.type == 'cuda' or torch.cuda.is_available():
                text_encoder = text_encoder.cuda()
            else:
                print("Warning: CUDA not available, using CPU for text encoder")
                
        except RuntimeError as e:
            print(f"CUDA error when moving text encoder: {e}")
            print("Falling back to CPU computation")
            # Force CPU computation if CUDA fails
            text_encoder = text_encoder.cpu()
            
        if pretrained_projection is not None:
            # Load pretrained projection from TaskRes Work
            pretrained_text_projection = torch.load(pretrained_projection)

            # Move weight to current CLIP model
            state_dict = text_encoder.state_dict()
            state_dict['text_projection'] = pretrained_text_projection['state_dict']['weight'].t()
            text_encoder.load_state_dict(state_dict)
            print(">> Pretrained text encoder loaded!")
            params = pretrained_text_projection['state_dict']['weight'].size(0) * \
                     pretrained_text_projection['state_dict']['weight'].size(1)
            print(">> Text projection parameters: ", params)
            print(pretrained_text_projection['state_dict'].keys())

    dataset = cfg.DATASET.NAME

    if dataset == "ImageNet":
        TEMPLATES = IMAGENET_TEMPLATES_SELECT
    elif cfg.TRAINER.ADAPTER.NUM_TEMPLATES > 1:
        # Use deterministic template selection with fixed seed for consistency
        rng_state = random.getstate()
        random.seed(42)
        TEMPLATES = random.sample(IMAGENET_TEMPLATES, cfg.TRAINER.ADAPTER.NUM_TEMPLATES - 1)
        random.setstate(rng_state)
    else:
        TEMPLATES = []
    TEMPLATES += [CUSTOM_TEMPLATES[dataset]]

    # FIX: Conditionally use torch.no_grad() based on GP usage
    # When GP is enabled, we need gradients to flow through text embeddings
    use_grad_context = cfg.TRAINER.ADAPTER.USE_GP and cfg.TRAINER.ADAPTER.NUM_TEMPLATES > 1
    
    if use_grad_context:
        # Don't use no_grad context - allow gradients to flow
        text_embeddings = []
        for text in classnames:
            tokens = clip.tokenize([template.format(text) for template in TEMPLATES])  # tokenized prompts are indices
            embeddings = clip_model.token_embedding(tokens).type(clip_model.dtype)
            prototype = text_encoder(embeddings.cuda(), tokens.cuda())
            text_embeddings.append(prototype)
    else:
        # Use no_grad context for efficiency when GP is not used
        with torch.no_grad():
            text_embeddings = []
            for text in classnames:
                tokens = clip.tokenize([template.format(text) for template in TEMPLATES])  # tokenized prompts are indices
                embeddings = clip_model.token_embedding(tokens).type(clip_model.dtype)
                prototype = text_encoder(embeddings.cuda(), tokens.cuda())
                text_embeddings.append(prototype)

    text_embeddings = torch.stack(text_embeddings)
    
    # Ensure we use the correct device (where text_embeddings are located)
    actual_device = text_embeddings.device
    
    # Choose between GP weighting and simple averaging
    if cfg.TRAINER.ADAPTER.USE_GP and text_embeddings.shape[1] > 1:
        # Initialize GP weighting module on the same device as text embeddings
        gp_weighter = GaussianProcessTemplateWeighter(
            num_classes=text_embeddings.shape[0],
            num_templates=text_embeddings.shape[1],
            embedding_dim=text_embeddings.shape[-1],
            kernel_type=cfg.TRAINER.ADAPTER.GP_KERNEL_TYPE,
            lengthscale=cfg.TRAINER.ADAPTER.GP_LENGTHSCALE,
            outputscale=cfg.TRAINER.ADAPTER.GP_OUTPUTSCALE,
            noise_var=cfg.TRAINER.ADAPTER.GP_NOISE_VAR,
            num_mc_samples=cfg.TRAINER.ADAPTER.GP_NUM_MC_SAMPLES,
            use_diagonal_cov=cfg.TRAINER.ADAPTER.GP_USE_DIAGONAL_COV,
        ).to(actual_device)
        
        # Apply GP weighting
        text_embeddings_avg, kl_divergence = gp_weighter(text_embeddings)
        
        return text_embeddings_avg, text_embeddings, gp_weighter, kl_divergence
    else:
        text_embeddings_avg = text_embeddings.mean(1)
        return text_embeddings_avg, text_embeddings, None, None


class AdapterMethod(nn.Module):
    def __init__(self, cfg, clip_model, base_text_features):
        super().__init__()
        self.device = clip_model.dtype
        self.logit_scale = clip_model.logit_scale
        self.initialization = cfg.TRAINER.ADAPTER.INIT
        self.apply_constraint = cfg.TRAINER.ADAPTER.CONSTRAINT
        self.distance = "l2"
        self.register_buffer("base_text_features", base_text_features)
        self.alpha_constraint = torch.zeros((base_text_features.shape[0])).to(self.device)
        self.base_text_features = base_text_features
        self.augmentations = True  # True
        self.epochs_aumentation = 20  # 20
        self.use_gp = cfg.TRAINER.ADAPTER.USE_GP  # Store GP usage flag

        if self.initialization == "RANDOM":  # Randomly initialized Linear Probing
            print("Using RANDOM initialization in Linear Probing")
            self.prototypes = nn.Parameter(torch.nn.init.kaiming_normal_(torch.empty(base_text_features.shape)))
        elif "ZS" in self.initialization:  # Linear Probe initialized with zero-shot weights
            print("Using Zero-Shot initialization in Linear Probing")
            self.prototypes = nn.Parameter(base_text_features.clone())
        elif "TR" in self.initialization:  # Task Residual Adapter form Yu et al. (2023)
            print("Using Task_residual approach for Linear Probing")
            self.init_TR(alpha=0.5)
        elif "ClipA" in self.initialization:  # CLIP-Adapter form Gao et al. (2023)
            self.init_clipA()
        elif "TipA" in self.initialization:  # TIP-Adapter form Zhang et al. (2022)
            self.init_tipA()
        elif "CrossModal" in self.initialization:  # Cross-Modal Linear Probing form Lin et al. (2023)
            print("Using CrossModal for Linear Probing")
            self.init_MultiModal()
        else:
            print("Initialization for Linear Probing not implemented")
            assert False

        if self.apply_constraint != "none":
            print("Applying constraint to the logistic regression weights: " + str(self.distance))

    def init_MultiModal(self):
        print("Using Zero-Shot initialization in Linear Probing")
        self.prototypes = nn.Parameter(self.base_text_features.clone())
    def init_TR(self, alpha=0.5):
        self.alpha = alpha
        self.grid_search_param = {"lr": [1e-1, 1e-2, 1e-3],
                                  "alpha": list(np.arange(0.2, 1.2, 0.2))}
        print("Using Task_residual approach for Linear Probing")
        self.prototypes = nn.Parameter(torch.zeros_like(self.base_text_features.clone()))

    def init_clipA(self, ratio=0.2):
        print("Using CLIP-Adapter")
        self.grid_search_param = {"lr": [1e-1, 1e-2, 1e-3],
                                  "ratio": list(np.arange(0.2, 1, 0.2))}
        self.ratio = ratio
        self.prototypes = nn.Parameter(self.base_text_features.clone())
        self.prototypes.requires_grad = False
        self.mlp = nn.Sequential(
            nn.Linear(self.base_text_features.shape[-1], self.base_text_features.shape[-1] // 4, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(self.base_text_features.shape[-1] // 4, self.base_text_features.shape[-1], bias=False),
            nn.ReLU(inplace=True)
        ).to(self.device)

    def init_tipA(self, beta=1, alpha=1):

        if "-f-" in self.initialization:
            self.grid_search_param = {"lr": [1e-1, 1e-2],
                                      "alpha": list(np.arange(1, 50, 50/10)),
                                      "beta": list(np.arange(1, 28, 28/10))}
        else:
            self.grid_search_param = {"alpha": list(np.arange(1, 50, 50/20)),
                                      "beta": list(np.arange(1, 28, 28/20))}

        print("Using Tip-Adapter")
        self.beta = beta
        self.alpha = alpha

        self.prototypes = nn.Parameter(self.base_text_features.clone())
        self.prototypes.requires_grad = False

        self.cache_keys = None  # Features
        self.cache_values = None  # labels

    def init_tipadapter(self, features_train, labels_train):
        self.cache_keys = nn.Parameter(features_train.clone().to(self.device))
        self.cache_keys.requires_grad = True
        self.cache_values = nn.Parameter(torch.nn.functional.one_hot(labels_train).clone().to(torch.float32).to(self.device))
        self.cache_values.requires_grad = False

    def zero_shot_constraint(self):

        # Compute constraint
        if "l2" in self.apply_constraint:
            disimilitude = (self.prototypes - self.base_text_features.clone()).pow(2).sum(-1)
        elif "cosine" in self.apply_constraint:
            disimilitude = (1 - torch.nn.functional.cosine_similarity(self.prototypes, self.base_text_features.clone()))
        else:
            print("Dissimilitude metric for constraint not implemented")
            assert False

        return torch.mean(self.alpha_constraint * disimilitude)

    def init_lagrangian_multipliers(self, labels_ds, logits_ds):
        if "balanced" in self.apply_constraint:
            performance = torch.ones(logits_ds.shape[-1]).to(torch.float)
        else:
            with torch.no_grad():

                # Get one-hot encoding ground-truth
                labels_one_hot = torch.nn.functional.one_hot(labels_ds).cpu()

                # Get zero_shot performance
                performance = torch.diag(torch.softmax(logits_ds, -1).t() @ labels_one_hot.to(torch.float32)) /\
                                      labels_one_hot.sum(0)

                if "corrected" in self.apply_constraint:
                    performance *= (logits_ds.shape[-1] / torch.sum(performance).item())
                if "constant" in self.apply_constraint:
                    performance = torch.ones(logits_ds.shape[-1]).to(torch.float) * torch.mean(performance).item()

        # set new alphas
        self.alpha_constraint = torch.clone(performance).to(self.device)
        self.penalty_parameter = torch.zeros_like(self.alpha_constraint).to(self.device)

    def outer_step(self):
        def phr(h, lambd, rho):
            x = lambd + rho * h
            y_sup = 1 / (2 * rho) * (x ** 2 - lambd ** 2)
            y_inf = - 1 / (2 * rho) * (lambd ** 2)

            grad_y_sup = x
            grad_y_inf = torch.zeros_like(h)

            sup = x >= 0
            return (
                torch.where(sup, y_sup, y_inf),
                torch.where(sup, grad_y_sup, grad_y_inf)
            )

        print("Outer step on Augmented Lagrangian Multiplier")

        # Cmpute current constraints
        disimilitude = (self.prototypes - self.base_text_features.clone()).pow(2).sum(-1)

        # Compute phr
        phr_value, phr_grad = phr(disimilitude, self.alpha_constraint, self.penalty_parameter)

        # Update lagrangian multipliers
        self.alpha_constraint = phr_grad.detach().clone()

        # Update penalty parameters rho
        self.penalty_parameter = disimilitude.detach().clone()

        print("New lagrangian multipliers:")
        print(self.alpha_constraint[0:5].detach().cpu().numpy())

    def forward(self):
        return self.prototypes

    def update_base_features(self, new_base_features):
        """Update base text features for GP mode."""
        # Update the base features reference (used for constraints, etc.)
        self.base_text_features = new_base_features
        
        # CRITICAL FIX: Do NOT update prototypes during training when GP is active!
        # The prototypes should be allowed to learn independently from the base features.
        # Only update prototypes at initialization or when not training.
        if hasattr(self, 'prototypes') and "ZS" in self.initialization:
            # Only update prototypes if we're not in training mode or GP is not active
            if not self.use_gp or not getattr(self, '_is_training', True):
                if not self.use_gp:
                    with torch.no_grad():
                        self.prototypes.data = new_base_features.clone()
                else:
                    # Allow gradients to flow through when GP is active but not training
                    self.prototypes.data = new_base_features.clone().detach()
            # If GP is active and we're training, let prototypes learn independently!


class CustomCLIP(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        self.image_encoder = clip_model.visual
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype  # float16
        
        # Store necessary attributes for Fix 2
        self.cfg = cfg
        self.classnames = classnames
        # FIX: Instead of storing clip_model reference, store token_embedding as a proper submodule
        # so it gets moved to GPU with the rest of the model
        self.token_embedding = clip_model.token_embedding
        
        text_encoder = TextEncoder(clip_model)
        self.text_encoder = text_encoder

        # CRITICAL FIX: Store the templates used during initialization to reuse consistently
        # Generate templates exactly as in _get_base_text_features
        dataset = cfg.DATASET.NAME
        if dataset == "ImageNet":
            self.templates = IMAGENET_TEMPLATES_SELECT
        elif cfg.TRAINER.ADAPTER.NUM_TEMPLATES > 1:
            # Use deterministic template selection with fixed seed for consistency
            rng_state = random.getstate()
            random.seed(42)
            self.templates = random.sample(IMAGENET_TEMPLATES, cfg.TRAINER.ADAPTER.NUM_TEMPLATES - 1)
            random.setstate(rng_state)
        else:
            self.templates = []
        self.templates += [CUSTOM_TEMPLATES[dataset]]

        # For TaskRes (Yu et al.) enhanced base - or regular CLIP base
        if cfg.TRAINER.ADAPTER.ENHANCED_BASE == "none":
            print(">> Use regular base!")
            base_text_features, text_embeddings_all, gp_weighter, kl_divergence = _get_base_text_features(cfg, classnames, clip_model, text_encoder)
        else:
            print(">> Use enhanced base!")
            base_text_features, text_embeddings_all, gp_weighter, kl_divergence = _get_base_text_features(
                cfg, classnames, clip_model, text_encoder, cfg.TRAINER.TaskRes.ENHANCED_BASE)

        self.text_embeddings_all = text_embeddings_all
        self.gp_weighter = gp_weighter  # Store GP weighter for training
        self.adapter = AdapterMethod(cfg, clip_model, base_text_features)
        
        # Store latest KL divergence for ELBO loss computation
        self.latest_kl_divergence = None

    def forward(self, image, return_features=False):
        try:
            image_features = self.image_encoder(image.type(self.dtype))
        except:
            image_features = self.image_encoder(image.float())

        # Update base features if using GP and in training mode
        # Only update during actual training epochs, not during feature extraction
        # Note: Don't check self.training since model is set to eval mode during training epochs
        if (self.gp_weighter is not None and 
            hasattr(self, '_in_training_epoch') and 
            self._in_training_epoch):
            
            # Set training flag for adapter
            self.adapter._is_training = True
            
            # CRITICAL FIX: Use the stored templates instead of regenerating random ones
            text_encoder = self.text_encoder
            classnames = self.classnames
            TEMPLATES = self.templates  # Use the consistent templates stored during initialization
            
            # Handle text_encoder device and dtype properly
            if self.dtype == torch.float16:
                try:
                    # Clear GPU cache and check CUDA availability
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                        text_encoder = text_encoder.cuda()
                    else:
                        print("Warning: CUDA not available during forward pass")
                except RuntimeError as e:
                    print(f"CUDA error in forward pass: {e}")
                    print("Continuing with current device")
            
            # FIX: Update text_encoder.dtype to match actual parameter dtype after model conversion
            text_encoder.dtype = next(text_encoder.parameters()).dtype
            
            # Create text embeddings WITH gradients using the SAME templates as baseline
            text_embeddings = []
            for text in classnames:
                tokens = clip.tokenize([template.format(text) for template in TEMPLATES])  # tokenized prompts are indices
                
                # Move tokens to the same device as token_embedding safely
                try:
                    tokens = tokens.to(next(self.token_embedding.parameters()).device)
                except RuntimeError as e:
                    print(f"Warning: Could not move tokens to device: {e}")
                    tokens = tokens.cuda() if torch.cuda.is_available() else tokens
                
                # Use the actual dtype of the token_embedding instead of stored self.dtype for consistency
                embeddings = self.token_embedding(tokens).type(next(self.token_embedding.parameters()).dtype)
                
                # Convert embeddings to the actual dtype of text_encoder to avoid dtype mismatch
                embeddings = embeddings.type(next(text_encoder.parameters()).dtype)
                
                try:
                    prototype = text_encoder(embeddings.cuda(), tokens.cuda())  # move to GPU for text encoder
                except RuntimeError as e:
                    print(f"Warning: CUDA error in text encoding: {e}")
                    prototype = text_encoder(embeddings, tokens)  # fallback to current device
                text_embeddings.append(prototype)
            
            text_embeddings = torch.stack(text_embeddings)
            
            # Apply GP weighting to fresh embeddings with consistent templates
            updated_base_features, kl_divergence = self.gp_weighter(text_embeddings)
            self.adapter.update_base_features(updated_base_features)
            
            # Store KL divergence for ELBO loss computation
            self.latest_kl_divergence = kl_divergence
        else:
            # Not in training mode with GP
            if hasattr(self.adapter, '_is_training'):
                self.adapter._is_training = False
            # Reset KL divergence when not using GP
            self.latest_kl_divergence = None

        if "TR" in self.adapter.initialization:
            logits = self.forward_task_residual(image_features)
        elif "ClipA" in self.adapter.initialization:
            logits = self.forward_clipadapter(image_features)
        elif "TipA" in self.adapter.initialization:
            logits = self.forward_tipadapter(image_features)
        else:
            logits = self.forward_lp(image_features)

        if return_features:
            return logits, image_features
        else:
            return logits

    def forward_features(self, features):
        
        # Apply GP weighting during training if enabled
        # This ensures GP learning happens even when using pre-extracted image features
        # Note: Don't check self.training since model is set to eval mode during training epochs
        if (self.gp_weighter is not None and 
            hasattr(self, '_in_training_epoch') and 
            self._in_training_epoch):
            
            # Set training flag for adapter
            self.adapter._is_training = True
            
            # CRITICAL FIX: Use the stored templates instead of regenerating random ones
            text_encoder = self.text_encoder
            classnames = self.classnames
            TEMPLATES = self.templates  # Use the consistent templates stored during initialization
            
            # Handle text_encoder device and dtype properly
            if self.dtype == torch.float16:
                try:
                    # Clear GPU cache and check CUDA availability
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                        text_encoder = text_encoder.cuda()
                    else:
                        print("Warning: CUDA not available during forward pass")
                except RuntimeError as e:
                    print(f"CUDA error in forward pass: {e}")
                    print("Continuing with current device")
            
            # FIX: Update text_encoder.dtype to match actual parameter dtype after model conversion
            text_encoder.dtype = next(text_encoder.parameters()).dtype
            
            # Create text embeddings WITH gradients using the SAME templates as baseline
            text_embeddings = []
            for text in classnames:
                tokens = clip.tokenize([template.format(text) for template in TEMPLATES])
                try:
                    tokens = tokens.to(next(self.token_embedding.parameters()).device)
                except RuntimeError as e:
                    print(f"Warning: Could not move tokens to device: {e}")
                    tokens = tokens.cuda() if torch.cuda.is_available() else tokens
                    
                embeddings = self.token_embedding(tokens).type(next(self.token_embedding.parameters()).dtype)
                embeddings = embeddings.type(next(text_encoder.parameters()).dtype)
                
                try:
                    prototype = text_encoder(embeddings.cuda(), tokens.cuda())
                except RuntimeError as e:
                    print(f"Warning: CUDA error in text encoding: {e}")
                    prototype = text_encoder(embeddings, tokens)
                text_embeddings.append(prototype)
            
            text_embeddings = torch.stack(text_embeddings)
            
            # Apply GP weighting to fresh embeddings with consistent templates
            updated_base_features, kl_divergence = self.gp_weighter(text_embeddings)
            self.adapter.update_base_features(updated_base_features)
            
            # Store KL divergence for ELBO loss computation
            self.latest_kl_divergence = kl_divergence
        else:
            # Not in training mode with GP
            if hasattr(self.adapter, '_is_training'):
                self.adapter._is_training = False
            # Reset KL divergence when not using GP
            self.latest_kl_divergence = None

        if "TR" in self.adapter.initialization:
            logits = self.forward_task_residual(features)
        elif "ClipA" in self.adapter.initialization:
            logits = self.forward_clipadapter(features)
        elif "TipA" in self.adapter.initialization:
            logits = self.forward_tipadapter(features)
        else:
            logits = self.forward_lp(features)

        return logits

    def forward_lp(self, features):

        # Get trained prototype
        prototypes = self.adapter()

        image_features_norm = features / features.norm(dim=-1, keepdim=True)
        prototypes_norm = prototypes / prototypes.norm(dim=-1, keepdim=True)

        logit_scale = self.logit_scale.exp()
        logits = image_features_norm @ prototypes_norm.t() * logit_scale

        return logits

    def forward_task_residual(self, features):

        # Get trained prototype
        prototypes = self.adapter()

        # Sum residual features to base zero-shot prototypes
        prototypes = self.adapter.base_text_features + self.adapter.alpha * prototypes

        image_features_norm = features / features.norm(dim=-1, keepdim=True)
        prototypes_norm = prototypes / prototypes.norm(dim=-1, keepdim=True)

        logit_scale = self.logit_scale.exp()
        logits = image_features_norm @ prototypes_norm.t() * logit_scale

        return logits

    def forward_clipadapter(self, features):

        # Get zero-shot weights
        prototypes = self.adapter()

        # Produce residual features on vision features
        x = self.adapter.mlp(features)
        features = self.adapter.ratio * x + (1 - self.adapter.ratio) * features

        # Normalize features
        image_features_norm = features / features.norm(dim=-1, keepdim=True)
        prototypes_norm = prototypes / prototypes.norm(dim=-1, keepdim=True)

        # Obtain logits
        logit_scale = self.logit_scale.exp()
        logits = image_features_norm @ prototypes_norm.t() * logit_scale

        return logits

    def forward_tipadapter(self, features):

        # Get zero-shot weights
        prototypes = self.adapter()

        # Normalize features
        image_features_norm = features / features.norm(dim=-1, keepdim=True)
        prototypes_norm = prototypes / prototypes.norm(dim=-1, keepdim=True)

        # Obtain  zero-shot logits
        logit_scale = self.logit_scale.exp()
        logits = image_features_norm @ prototypes_norm.t() * logit_scale

        if self.adapter.cache_keys is not None:
            # normalize cache keys
            cache_keys = self.adapter.cache_keys / self.adapter.cache_keys.norm(dim=-1, keepdim=True)

            # Get affinity betwen train features and test
            affinity = features @ cache_keys.t().cuda().to(torch.float)

            cache_logits = torch.exp(((-1) * (self.adapter.beta - self.adapter.beta * affinity))) @ self.adapter.cache_values.cuda().to(torch.float)

            logits += self.adapter.alpha * cache_logits

        return logits


class TrainerXCostume(SimpleTrainer):
    """A base trainer using labeled data only."""

    def run_epoch(self):
        # Mark that we're in a training epoch for Fix 2
        self.model._in_training_epoch = True
        
        # Eval mode - not updating batchnorm statistics
        self.set_model_mode("eval")

        # Init kpis tracker
        losses = MetricMeter()
        batch_time = AverageMeter()
        data_time = AverageMeter()

        # Set number of batches to sample
        self.num_batches = len(self.train_loader_x)
        self.batch_size = self.train_loader_x.batch_size

        # Set features
        features = self.features_train.clone().cpu().numpy()
        labels = self.labels_train.clone()

        # Sample half dataset - to tackle previous oversample with text prompts
        if "CrossModal" in self.model.adapter.initialization:
            idx = np.random.choice(list(np.arange(0, features.shape[0])), features.shape[0] // 2)
            features = features[idx, :]
            labels = labels[idx]

        # Randomly shuffle
        idx = np.random.rand(features.shape[0]).argsort(axis=0)
        features = features[idx, :]
        labels = labels[idx]

        end = time.time()
        for self.batch_idx in range(self.num_batches):
            batch_init = self.batch_idx * self.batch_size
            batch_end = (self.batch_idx + 1) * self.batch_size

            data_time.update(time.time() - end)
            loss_summary = self.forward_backward(features[batch_init:batch_end],
                                                 labels[batch_init:batch_end])
            batch_time.update(time.time() - end)
            losses.update(loss_summary)

            meet_freq = (self.batch_idx + 1) % self.cfg.TRAIN.PRINT_FREQ == 0
            only_few_batches = self.num_batches < self.cfg.TRAIN.PRINT_FREQ
            if meet_freq or only_few_batches:
                nb_remain = 0
                nb_remain += self.num_batches - self.batch_idx - 1
                nb_remain += (
                    self.max_epoch - self.epoch - 1
                ) * self.num_batches
                eta_seconds = batch_time.avg * nb_remain
                eta = str(datetime.timedelta(seconds=int(eta_seconds)))

                if self.epoch % 10 == 0 or self.epoch == self.max_epoch - 1:
                    info = []
                    info += [f"epoch [{self.epoch + 1}/{self.max_epoch}]"]
                    info += [f"batch [{self.batch_idx + 1}/{self.num_batches}]"]
                    info += [f"{losses}"]
                    info += [f"eta {eta}"]
                    print(" ".join(info))

            n_iter = self.epoch * self.num_batches + self.batch_idx
            for name, meter in losses.meters.items():
                self.write_scalar("train/" + name, meter.avg, n_iter)
            self.write_scalar("train/lr", self.get_current_lr(), n_iter)

            end = time.time()
        
        # Unmark training epoch flag  
        self.model._in_training_epoch = False
        
        return loss_summary


@TRAINER_REGISTRY.register()
class ADAPTER(TrainerXCostume):
    """General Adapter
    """

    def check_cfg(self, cfg):
        assert cfg.TRAINER.ADAPTER.PREC in ["fp16", "fp32", "amp"]

    def build_model(self):
        cfg = self.cfg
        classnames = self.dm.dataset.classnames

        # Reset GPU state to avoid CUDA errors
        reset_gpu_state()

        print(f"Loading CLIP (backbone: {cfg.MODEL.BACKBONE.NAME})")
        clip_model = load_clip_to_cpu(cfg)

        if cfg.TRAINER.ADAPTER.PREC == "fp32" or cfg.TRAINER.ADAPTER.PREC == "amp":
            # CLIP's default precision is fp16
            clip_model.float()

        print("Building custom CLIP")
        self.model = CustomCLIP(cfg, classnames, clip_model)

        print("Turning off gradients in both the image and the text encoder")
        for name, param in self.model.named_parameters():
            if "adapter" not in name and "gp_weighter" not in name:
                param.requires_grad_(False)
            else:
                # Ensure adapter and GP parameters require gradients
                param.requires_grad_(True)
                print(f"  Parameter {name} requires_grad: {param.requires_grad}")

        if cfg.MODEL.INIT_WEIGHTS:
            load_pretrained_weights(self.model.adapter, cfg.MODEL.INIT_WEIGHTS)

        self.model.to(self.device)
        self.model = self.model.float()
        
        # NOTE: only give adapter (and optionally GP) parameters to the optimizer
        if cfg.TRAINER.ADAPTER.USE_GP and self.model.gp_weighter is not None:
            # Create parameter groups with different learning rates
            adapter_params = list(self.model.adapter.parameters())
            gp_params = list(self.model.gp_weighter.parameters())
            
            # Get weight decay if available
            weight_decay = getattr(cfg.OPTIM, 'WEIGHT_DECAY', 0.0)
            gp_lr = getattr(cfg.TRAINER.ADAPTER, 'GP_LR', cfg.OPTIM.LR)
            
            # Create parameter groups
            param_groups = [
                {'params': adapter_params, 'lr': cfg.OPTIM.LR, 'weight_decay': weight_decay},
                {'params': gp_params, 'lr': gp_lr, 'weight_decay': 0.0}  # No weight decay for GP params
            ]
            
            # Create optimizer with parameter groups
            from torch.optim import SGD, Adam
            if cfg.OPTIM.NAME.lower() == "sgd":
                self.optim = SGD(param_groups, momentum=getattr(cfg.OPTIM, 'MOMENTUM', 0.9))
            elif cfg.OPTIM.NAME.lower() == "adam":
                self.optim = Adam(param_groups)
            else:
                # Fallback to SGD
                self.optim = SGD(param_groups, momentum=getattr(cfg.OPTIM, 'MOMENTUM', 0.9))
        else:
            self.optim = build_optimizer(self.model.adapter, cfg.OPTIM)
            
        self.sched = build_lr_scheduler(self.optim, cfg.OPTIM)
        self.register_model("adapter", self.model.adapter, self.optim, self.sched)

        self.scaler = GradScaler() if cfg.TRAINER.ADAPTER.PREC == "amp" else None

    def train(self):
        self.set_model_mode("eval")

        # Feature extraction on test set
        self.labels_test, output_test, self.features_test = self.extract_features(partition="test")
        print("Zero-Shot accuracy on test: " +
              str(round(compute_accuracy(output_test.cuda(), self.labels_test.cuda())[0].item(), 2)))

        # Feature extraction on training set
        self.labels_train, self.logits_zs, self.features_train = self.extract_features(
            partition="train", reps=self.model.adapter.epochs_aumentation, transforms=self.model.adapter.augmentations)

        if "CrossModal" in self.model.adapter.initialization:
            print("Preparing cross-modal dataset... resampling text prompts")
            # Cross-Modal: Add zero-shot prototypes as samples
            zs_prototypes = self.model.text_embeddings_all.cpu().numpy()
            zs_labels = np.repeat(np.expand_dims(np.arange(0, zs_prototypes.shape[0]), (0)), zs_prototypes.shape[1], 0)

            zs_prototypes = np.reshape(np.transpose(zs_prototypes, (2, 1, 0)),
                                       (zs_prototypes.shape[-1], zs_prototypes.shape[0]*zs_prototypes.shape[1])).transpose()
            zs_labels = np.transpose(zs_labels, (1, 0)).flatten()

            # Resample for a balanced dataset between modalities
            idx = np.random.choice(list(np.arange(0, len(zs_labels))), self.features_train.shape[0])
            zs_labels = zs_labels[idx]
            zs_prototypes = zs_prototypes[idx, :]

            self.features_train = torch.cat([self.features_train, torch.tensor(zs_prototypes)], dim=0)
            self.labels_train = torch.cat([self.labels_train, torch.tensor(zs_labels).cuda()])

        # Init alphas in constraint formulation
        if self.model.adapter.apply_constraint != "none":
            print("Getting initial lagrangian multipliers for constraint formulation")
            self.model.adapter.device = self.device
            self.model.adapter.init_lagrangian_multipliers(self.labels_train, self.logits_zs)
            print("Lagrangian multipliers: ")
            print(list(torch.round(self.model.adapter.alpha_constraint.detach(), decimals=3).cpu().numpy()))

        # In the case of tip-adapter, register cache features
        if "TipA" in self.model.adapter.initialization:
            # Given the new key features, register again the weights to optimizer
            self.model.adapter.init_tipadapter(self.features_train, self.labels_train)
            self.optim = build_optimizer(self.model.adapter, self.cfg.OPTIM)  # Update optimizer with new params
            self.sched = build_lr_scheduler(self.optim, self.cfg.OPTIM)
            self.register_model("adapter_tipa-f-", self.model.adapter, self.optim, self.sched)

            if "-f-" not in self.model.adapter.initialization:
                self.max_epoch = 1  # Not training, just one forward round for predicting test performance

        summary_grid = []
        if "grid_search" in self.model.adapter.initialization:
            from sklearn.model_selection import ParameterGrid
            import pandas as pd

            best_acc = 0.0
            best_setting = []
            grid = ParameterGrid(self.model.adapter.grid_search_param)
            for params in grid:
                print("Iteration grid hyperparameters search: ")
                print(params)
                self.reset_hyperparams(params)

                # Training of adapter
                self.before_train()
                for self.epoch in range(self.start_epoch, self.max_epoch):

                    # Train and update weights per epoch
                    self.before_epoch()
                    loss_summary = self.run_epoch()

                    if loss_summary["acc_test"] > best_acc:
                        best_acc = loss_summary["acc_test"]
                        best_setting = params

                    self.epoch = -1  # To avoid saving weights
                    self.after_epoch()

                params["acc_test"] = loss_summary["acc_test"]
                summary_grid.append(params)

                # Print current configuration performance
                print("Current configuration: ")
                print(params)
                print("A on test:")
                print(loss_summary["acc_test"])

            # Print best configuration performance:
            print("Best configuration: ")
            print(best_setting)
            print("Best accuracy on test:")
            print(best_acc)
            df = pd.DataFrame(summary_grid)
            df.to_csv(self.cfg.OUTPUT_DIR + "/grid_search.csv")
        else:
            # Training of adapter
            self.before_train()
            for self.epoch in range(self.start_epoch, self.max_epoch):

                # Train and update weights per epoch
                self.before_epoch()
                self.run_epoch()

                # Update lagrangian parameter and multiplier
                if "adaptative" in self.model.adapter.apply_constraint:
                    self.model.adapter.outer_step()

                self.after_epoch()

        self.after_train()

    def reset_hyperparams(self, params):
        import random

        if "ClipA" in self.model.adapter.initialization:
            self.model.adapter.init_clipA(ratio=params["ratio"])
        if "TipA" in self.model.adapter.initialization:
            self.model.adapter.init_tipA(alpha=params["alpha"], beta=params["beta"])
        if "TR" in self.model.adapter.initialization:
            self.model.adapter.init_TR(alpha=params["alpha"])

        # In the case of tip-adapter, register cache features
        if "TipA" in self.model.adapter.initialization:
            # Given the new key features, register again the weights to optimizer
            self.model.adapter.init_tipadapter(self.features_train, self.labels_train)
            if "-f-" in self.model.adapter.initialization:
                # Put epochs as in the original paper
                self.max_epoch = 20

        self.model.to(self.device)
        self.model = self.model.float()

        if "lr" in list(params.keys()):
            self.cfg.OPTIM["LR"] = params["lr"]
            
        # Recreate optimizer considering GP parameters
        if self.cfg.TRAINER.ADAPTER.USE_GP and self.model.gp_weighter is not None:
            # Create parameter groups with different learning rates
            adapter_params = list(self.model.adapter.parameters())
            gp_params = list(self.model.gp_weighter.parameters())
            
            # Get weight decay if available
            weight_decay = getattr(self.cfg.OPTIM, 'WEIGHT_DECAY', 0.0)
            gp_lr = getattr(self.cfg.TRAINER.ADAPTER, 'GP_LR', self.cfg.OPTIM.LR)
            
            # Create parameter groups
            param_groups = [
                {'params': adapter_params, 'lr': self.cfg.OPTIM.LR, 'weight_decay': weight_decay},
                {'params': gp_params, 'lr': gp_lr, 'weight_decay': 0.0}  # No weight decay for GP params
            ]
            
            # Create optimizer with parameter groups
            from torch.optim import SGD, Adam
            if self.cfg.OPTIM.NAME.lower() == "sgd":
                self.optim = SGD(param_groups, momentum=getattr(self.cfg.OPTIM, 'MOMENTUM', 0.9))
            elif self.cfg.OPTIM.NAME.lower() == "adam":
                self.optim = Adam(param_groups)
            else:
                # Fallback to SGD
                self.optim = SGD(param_groups, momentum=getattr(self.cfg.OPTIM, 'MOMENTUM', 0.9))
        else:
            self.optim = build_optimizer(self.model.adapter, self.cfg.OPTIM)
            
        self.sched = build_lr_scheduler(self.optim, self.cfg.OPTIM)
        self._models.popitem(), self._optims.popitem(),self._scheds.popitem()
        self.register_model("adapter" + str(random.random()), self.model.adapter, self.optim, self.sched)

        return 1

    def after_train(self):
        print("Finish training")

        # Show elapsed time
        elapsed = round(time.time() - self.time_start)
        elapsed = str(datetime.timedelta(seconds=elapsed))
        print(f"Elapsed: {elapsed}")

        # Close writer
        self.close_writer()

    def forward_backward(self, features, labels):
        
        prec = self.cfg.TRAINER.ADAPTER.PREC
        if prec == "amp":
            with autocast():
                # Cross-entropy loss (likelihood term in ELBO)
                output = self.model.forward_features(torch.tensor(features).to(self.device))
                
                # Softmax cross-entropy
                loss_ce = F.cross_entropy(output, labels)
                
                # Constraint to zero-shot (CLAP)
                if self.model.adapter.apply_constraint != "none":
                    loss_constraint = self.model.adapter.zero_shot_constraint()
                    loss = loss_ce + loss_constraint
                else:
                    loss = loss_ce
                
                # ELBO loss: -E[log p(y|α)] + β * KL(q(α)||p(α))
                if self.cfg.TRAINER.ADAPTER.USE_GP and self.model.gp_weighter is not None:
                    if self.model.latest_kl_divergence is not None:
                        kl_divergence = self.model.latest_kl_divergence
                        loss = loss + self.cfg.TRAINER.ADAPTER.GP_BETA * kl_divergence
                    
            self.optim.zero_grad()
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optim)
            self.scaler.update()
        else:
            # Cross-entropy loss (likelihood term in ELBO)
            output = self.model.forward_features(torch.tensor(features).to(self.device))
            
            # Softmax cross-entropy
            loss_ce = F.cross_entropy(output, labels)
            
            # Constraint to zero-shot (CLAP)
            if self.model.adapter.apply_constraint != "none":
                loss_constraint = self.model.adapter.zero_shot_constraint()
                loss = loss_ce + loss_constraint
            else:
                loss = loss_ce
            
            # ELBO loss: -E[log p(y|α)] + β * KL(q(α)||p(α))
            if self.cfg.TRAINER.ADAPTER.USE_GP and self.model.gp_weighter is not None:
                if self.model.latest_kl_divergence is not None:
                    kl_divergence = self.model.latest_kl_divergence
                    loss = loss + self.cfg.TRAINER.ADAPTER.GP_BETA * kl_divergence

            self.model_backward_and_update(loss)

        with torch.no_grad():
            output_test = self.model.forward_features(self.features_test.clone().detach().to(self.device))

        loss_summary = {
            "loss": loss.item(),
            "acc_train": compute_accuracy(output, labels)[0].item(),
            "acc_test": compute_accuracy(output_test, self.labels_test)[0].item(),
        }
        
        # Add KL divergence to loss summary for monitoring
        if self.cfg.TRAINER.ADAPTER.USE_GP and self.model.latest_kl_divergence is not None:
            loss_summary["kl_divergence"] = self.model.latest_kl_divergence.item()

        if (self.batch_idx + 1) == self.num_batches:
            self.update_lr()

        torch.cuda.empty_cache()
        return loss_summary

    def load_model(self, directory, cfg, epoch=None):
        if not directory:
            print("Note that load_model() is skipped as no pretrained model is given")
            return
        else:
            print("Pretrained model given")

        if self.model.adapter.initialization == "TipA":
            epoch = 1

        names = self.get_model_names()

        # By default, the best model is loaded
        model_file = "model-best.pth.tar"

        if epoch is not None:
            model_file = "model.pth.tar-" + str(epoch)

        for name in names:
            model_path = osp.join(directory, name, model_file)

            if not osp.exists(model_path):
                raise FileNotFoundError('Model not found at "{}"'.format(model_path))
            else:
                print('Model found at "{}"'.format(model_path))

            checkpoint = load_checkpoint(model_path)
            state_dict = checkpoint["state_dict"]

            if "TipA" in self.model.adapter.initialization:
                self.model.adapter.cache_keys = nn.Parameter(state_dict['cache_keys'].clone())
                self.model.adapter.cache_values = nn.Parameter(state_dict['cache_values'].clone())

            if self.cfg.DATASET.NAME == 'ImageNetA' or self.cfg.DATASET.NAME == 'ImageNetR':
                if self.cfg.DATASET.NAME == 'ImageNetA':
                    from datasets.imagenet_a_r_indexes_v2 import find_imagenet_a_indexes as find_indexes
                else:
                    from datasets.imagenet_a_r_indexes_v2 import find_imagenet_r_indexes as find_indexes
                imageneta_indexes = find_indexes()
                print("Parameters found: ")
                print(state_dict.keys())
                state_dict['base_text_features'] = state_dict['base_text_features'][imageneta_indexes]
                state_dict['prototypes'] = state_dict['prototypes'][imageneta_indexes]

                if "TipA" in self.model.adapter.initialization:
                    state_dict['cache_values'] = state_dict['cache_values'][:, imageneta_indexes]
                    self.model.adapter.cache_keys = nn.Parameter(state_dict['cache_keys'].clone())
                    self.model.adapter.cache_values = nn.Parameter(state_dict['cache_values'].clone())
            epoch = checkpoint["epoch"]

            # Ignore fixed token vectors
            if "token_prefix" in state_dict:
                del state_dict["token_prefix"]

            if "token_suffix" in state_dict:
                del state_dict["token_suffix"]

            print("Loading weights to {} " 'from "{}" (epoch = {})'.format(name, model_path, epoch))
            # set strict=False
            self._models[name].load_state_dict(state_dict, strict=False)
            self.model.float()

    def extract_features(self, partition, reps=1, transforms=False):
        print("Extracting features from: " + partition)
        self.set_model_mode("eval")

        if partition == "train":

            # Copy safe version of training dataloader
            data_loader = copy.deepcopy(self.train_loader_x)

            # Set data augmentation transforms
            if not transforms:
                data_loader.dataset.transform = self.val_loader.dataset.transform

            # Set data loader with drop last to false for not losing samples
            data_loader = torch.utils.data.DataLoader(
                copy.deepcopy(self.train_loader_x.dataset), batch_size=self.train_loader_x.batch_size,
                sampler=self.train_loader_x.sampler, num_workers=self.train_loader_x.num_workers,
                drop_last=False, pin_memory=self.train_loader_x.pin_memory)

        elif partition == "val":
            data_loader = copy.deepcopy(self.val_loader)
        elif partition == "test":
            data_loader = copy.deepcopy(self.test_loader)
        else:
            assert False

        if "TipA" not in self.model.adapter.initialization:

            labels_ds, logits_ds, features_ds = [], [], []
            for rep in range(reps):
                for batch_idx, batch in enumerate(data_loader):
                    with torch.no_grad():
                        input, label = self.parse_batch_test(batch)
                        logits, features = self.model(input,  return_features=True)
                        labels_ds.append(label), logits_ds.append(logits.cpu()),  features_ds.append(features.cpu())

            # Concatenate outputs
            labels_ds = torch.cat(labels_ds, dim=0)
            logits_ds = torch.cat(logits_ds, dim=0)
            features_ds = torch.cat(features_ds, dim=0)

        else:

            labels_ds, logits_ds, features_ds = [], [], []
            for rep in range(reps):
                labels_ds_irep, logits_dsirep, features_ds_irep = [], [], []
                for batch_idx, batch in enumerate(data_loader):
                    with torch.no_grad():
                        input, label = self.parse_batch_test(batch)
                        logits, features = self.model(input, return_features=True)
                        labels_ds_irep.append(label), logits_dsirep.append(logits.cpu()), features_ds_irep.append(features.cpu())
                # Concatenate outputs for dataset
                labels_ds_irep = torch.cat(labels_ds_irep, dim=0)
                logits_dsirep = torch.cat(logits_dsirep, dim=0)
                features_ds_irep = torch.cat(features_ds_irep, dim=0)
                # Concatenate outputs for repetitions
                labels_ds.append(labels_ds_irep.unsqueeze(0))
                logits_ds.append(logits_dsirep.unsqueeze(0))
                features_ds.append(features_ds_irep.unsqueeze(0))

            # Concatenate outputs
            labels_ds = torch.cat(labels_ds, dim=0)[0, :]
            logits_ds = torch.cat(logits_ds, dim=0).mean(0)
            features_ds = torch.cat(features_ds, dim=0).mean(0)

        return labels_ds, logits_ds, features_ds
