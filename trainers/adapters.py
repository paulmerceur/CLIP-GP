import os
import os.path as osp
from typing import List, Optional
import random
from re import template
import time
import os.path as osp
import datetime
import copy
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.amp import GradScaler, autocast
import numpy as np

from dassl.engine import TRAINER_REGISTRY, SimpleTrainer
from dassl.metrics import compute_accuracy
from dassl.utils import load_pretrained_weights, load_checkpoint, AverageMeter, MetricMeter
from dassl.optim import build_optimizer, build_lr_scheduler

from clip import clip
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer
from datasets.imagenet_templates import IMAGENET_TEMPLATES, IMAGENET_TEMPLATES_SELECT
from gp_template_weigher import GaussianProcessTemplateWeighter

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.allow_tf32 = True

_tokenizer = _Tokenizer()


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

def _build_templates(cfg) -> List[str]:
    """Fixed diverse set of proven templates based on CoOp paper."""
    dataset = cfg.DATASET.NAME
    
    # Use fixed diverse templates proven effective in CoOp
    # These are semantically diverse and high-performing
    FIXED_DIVERSE_TEMPLATES = [
        "a photo of a {}.",
        "a photograph of a {}.",
        "an image of a {}.",
        "a cropped photo of a {}.",
        "a good photo of a {}.",
        "a bad photo of a {}.",
        "a photo of the {}."
    ]
    
    if cfg.TRAINER.ADAPTER.NUM_TEMPLATES == 1:
        # Single template case - use dataset specific
        base = [CUSTOM_TEMPLATES.get(dataset, "a photo of a {}.")]
    else:
        # Multiple templates - use fixed diverse set
        num_needed = min(cfg.TRAINER.ADAPTER.NUM_TEMPLATES, len(FIXED_DIVERSE_TEMPLATES))
        base = FIXED_DIVERSE_TEMPLATES[:num_needed]
    
    print(f"Selected {len(base)} fixed diverse templates:")
    for i, template in enumerate(base):
        print(f"  {i}: {template}")
    
    return base

def _get_base_text_features(
    cfg,
    classnames: List[str],
    clip_model,
    text_encoder: TextEncoder,
    pretrained_projection: Optional[str] = None,
):
    """Computes and caches embeddings inside closure."""

    device = next(text_encoder.parameters()).device
    templates = _build_templates(cfg)

    # Encode all prompts once - returned tensor is reused by caller.
    emb_list = []
    with torch.no_grad():
        for name in classnames:
            tok = clip.tokenize([t.format(name) for t in templates]).to(device)
            e = clip_model.token_embedding(tok).type(clip_model.dtype)
            emb = text_encoder(e, tok)
            emb_list.append(emb)
    text_embeds = torch.stack(emb_list)  # [K,M,D]

    # Instantiate GP
    if cfg.TRAINER.ADAPTER.USE_GP and text_embeds.size(1) > 1:
        gp = GaussianProcessTemplateWeighter(text_embeddings=text_embeds, cfg=cfg).to(device)
        proto, kl = gp.forward_and_kl()
        return proto, text_embeds, gp, kl

    # GP disabled -> simple average
    return text_embeds.mean(1), text_embeds, None, None

class AdapterMethod(nn.Module):
    def __init__(self, cfg, clip_model, base_text_features):
        super().__init__()
        self.dtype = clip_model.dtype
        self.logit_scale = clip_model.logit_scale
        self.initialization = cfg.TRAINER.ADAPTER.INIT
        self.apply_constraint = cfg.TRAINER.ADAPTER.CONSTRAINT
        self.distance = "l2"
        self.register_buffer("base_text_features", base_text_features)
        self.alpha_constraint = torch.zeros((base_text_features.shape[0])).to(self.dtype)
        self.base_text_features = base_text_features
        self.augmentations = True
        self.epochs_aumentation = 20
        self.use_gp = cfg.TRAINER.ADAPTER.USE_GP

        if self.initialization == "RANDOM":  # Randomly initialized Linear Probing
            print("Using RANDOM initialization in Linear Probing")
            self.prototypes = nn.Parameter(torch.nn.init.kaiming_normal_(torch.empty(base_text_features.shape)))
        elif "ZS" in self.initialization:  # Linear Probe initialized with zero-shot weights
            print("Using Zero-Shot initialization in Linear Probing")
            if self.use_gp:
                print("GP is active: prototypes are not independent learnable parameters.")
                # When GP is active, prototypes will be updated by GP - initialize as non-parameter
                self.prototypes = base_text_features.clone().detach()
            else:
                # Default: make the zero-shot prototypes *trainable* so that
                # the linear probe can actually learn from the labelled
                # data. They start from the zero-shot weights but will be
                # updated during training.
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
            
        # Ensure all components are on the same device after initialization
        target_device = base_text_features.device
        target_dtype = base_text_features.dtype
        self.to(device=target_device, dtype=target_dtype)

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
        ).to(self.dtype)

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
        self.cache_keys = nn.Parameter(features_train.clone().to(self.dtype))
        self.cache_keys.requires_grad = True
        self.cache_values = nn.Parameter(torch.nn.functional.one_hot(labels_train).clone().to(torch.float32).to(self.dtype))
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
        self.alpha_constraint = torch.clone(performance).to(self.dtype)
        self.penalty_parameter = torch.zeros_like(self.alpha_constraint).to(self.dtype)

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


class LowRankLinear(nn.Module):
    """
    Low-rank factorisation of a square projection matrix.
    W = V ∘ U with U: d → r, V: r → d.
    A small residual (initially zero) is thus added on top of the identity
    when r < d.  A `.weight` property exposes V.weight so existing dtype
    checks that reference `visual_proj.weight` continue to work unchanged.
    """
    def __init__(self, dim: int, rank: int = 64, bias: bool = False):
        super().__init__()
        self.U = nn.Linear(dim, rank, bias=bias)
        self.V = nn.Linear(rank, dim, bias=bias)

        # Start close to identity: U initializes to zeros, V to identity rows
        nn.init.zeros_(self.U.weight)
        if self.V.weight.shape[0] == self.V.weight.shape[1]:
            nn.init.eye_(self.V.weight)
        else:
            # When rank < dim, use small values to approximate identity effect
            nn.init.normal_(self.V.weight, mean=0.0, std=1e-4)

    @property
    def weight(self):
        # Expose the out-projection weight for dtype checks
        return self.V.weight

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply low-rank residual on top of identity.

        Returns:  x + V(U(x))  so that the layer starts exactly as the identity
        mapping (because U is zero-initialised).  This prevents zero vectors
        which caused division-by-zero issues during \|x\| normalisation.
        """
        return x + self.V(self.U(x))


class CustomCLIP(nn.Module):
    """CLIP + optional GP-weighted template prototypes (cached)."""

    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        self.cfg = cfg
        self.classnames = classnames

        self.image_encoder = clip_model.visual
        self.token_embedding = clip_model.token_embedding
        self.text_encoder = TextEncoder(clip_model)
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype

        # ---------------------------------------------------------------
        #  Learnable visual projection (only active when GP is enabled)
        # ---------------------------------------------------------------
        self.use_visual_proj = cfg.TRAINER.ADAPTER.USE_GP  # flag for later use
        self.vision_dim = clip_model.text_projection.shape[1]
        # Use low-rank projection only when GP template weighting is active
        if cfg.TRAINER.ADAPTER.USE_GP:
            # Rank fixed to 64 for an initial experiment (≈13% params of full)
            self.visual_proj = LowRankLinear(self.vision_dim, rank=64, bias=False)
        else:
            # Baseline/no-GP setup keeps an identity mapping (no parameters)
            self.visual_proj = nn.Identity()

        # Use helper to build all text-related tensors 
        base_proto, self.text_embeddings_all, self.gp_weighter, _ = _get_base_text_features(cfg, classnames, clip_model, self.text_encoder)

        # Cache embeddings for fast GP updates
        self.register_buffer("text_embeddings_static", self.text_embeddings_all.float())

        # Adapter
        self.adapter = AdapterMethod(cfg, clip_model, base_proto)

        self._in_training_epoch = False  # toggled by trainer each epoch
        self._batch_count = 0  # Track batches for diagnostics
        self._epoch_count = 0  # Track epochs for diagnostics

    def get_gp_kl_divergence(self):
        """Compute KL divergence for GP loss term."""
        
        return getattr(self, "_last_gp_kl", None)

    # ----------------------------------------------------------------- #
    #  Forward passes                                                   #
    # ----------------------------------------------------------------- #
    def _forward_impl(self, x, *, is_feature: bool) -> torch.Tensor:
        # Update batch count for diagnostics
        if self._in_training_epoch:
            self._batch_count += 1
            
        # Obtain image features (pre-computed or freshly encoded)
        feats = x.type(self.dtype) if is_feature else self.image_encoder(x.type(self.dtype))

        # Optionally apply visual projection (only when GP is used)
        if self.use_visual_proj:
            # Cast input to projection dtype to avoid matmul dtype mismatch
            if feats.dtype != self.visual_proj.weight.dtype:
                feats = feats.to(dtype=self.visual_proj.weight.dtype)

            feats = self.visual_proj(feats)
            feats = feats / feats.norm(dim=-1, keepdim=True)

        # Get current prototypes - simple and direct
        if self.gp_weighter is not None:
            # Use GP prototypes with appropriate sampling
            use_mean = not self.training
            current_prototypes, kl = self.gp_weighter.forward_and_kl(use_mean=use_mean)
            self._last_gp_kl = kl # cache KL so the trainer can reuse it later
        else:
            # Use standard adapter prototypes
            current_prototypes = self.adapter.prototypes
            self._last_gp_kl = None

        if "TR" in self.adapter.initialization:
            logits = self.forward_task_residual(feats, current_prototypes)
        elif "ClipA" in self.adapter.initialization:
            logits = self.forward_clipadapter(feats, current_prototypes)
        elif "TipA" in self.adapter.initialization:
            logits = self.forward_tipadapter(feats, current_prototypes)
        else:
            logits = self.forward_lp(feats, current_prototypes)
        return logits

    def forward(self, image, *, return_features: bool = False):
        logits = self._forward_impl(image, is_feature=False)
        if return_features:
            feats = self.image_encoder(image.type(self.dtype))
            return logits, feats
        return logits

    def forward_features(self, features):
        return self._forward_impl(features, is_feature=True)

    # ----------------------------------------------------------------- #
    #  Monte-Carlo forward (for ELBO)                                   #
    # ----------------------------------------------------------------- #
    def forward_features_mc(self, features, num_samples: int):
        """Compute logits for *num_samples* GP draws.

        Returns
        -------
        logits : Tensor
            Shape ``[S, B, K]`` where *S* is ``num_samples``.
        kl : Tensor | None
            KL divergence term from the GP (or *None* if GP disabled).
        """
        # Obtain image features once (shared across samples) ------------------
        feats = features.type(self.dtype)

        if self.use_visual_proj:
            # Cast input to projection dtype to avoid matmul dtype mismatch
            if feats.dtype != self.visual_proj.weight.dtype:
                feats = feats.to(dtype=self.visual_proj.weight.dtype)

            feats = self.visual_proj(feats)
            feats = feats / feats.norm(dim=-1, keepdim=True)

        # If no GP, fall back to deterministic adapter -----------------------
        if self.gp_weighter is None:
            logits = self.forward_lp(feats)  # [B,K]
            return logits.unsqueeze(0), None

        # Sample prototypes ---------------------------------------------------
        prototypes_s = self.gp_weighter.sample_prototypes(num_samples, use_mean=False)  # [S,K,D]
        kl = self.gp_weighter.variational_strategy.kl_divergence().sum()

        # Normalise once for stability
        feats_norm = feats / feats.norm(dim=-1, keepdim=True)          # [B,D]
        prot_norm = prototypes_s / prototypes_s.norm(dim=-1, keepdim=True)  # [S,K,D]

        # Ensure dtype alignment before matmul
        if prot_norm.dtype != feats.dtype:
            prot_norm = prot_norm.to(dtype=feats.dtype)

        # Compute logits:  (B,D)  ·  (S,K,D)ᵀ  →  (S,B,K)
        logits = torch.einsum("bd,skd->sbk", feats_norm, prot_norm)
        # Cast logit_scale to logits dtype for safe multiplication
        logits = logits * self.logit_scale.exp().to(dtype=logits.dtype)

        return logits, kl

    # ----------------------------------------------------------------- #
    #  Adapter‑specific logits                                          #
    # ----------------------------------------------------------------- #
    def forward_lp(self, features, prototypes=None):
        if prototypes is None:
            prototypes = self.adapter()
        
        # Ensure prototypes are on the same device and dtype as features
        if prototypes.device != features.device or prototypes.dtype != features.dtype:
            prototypes = prototypes.to(device=features.device, dtype=features.dtype)
            
        features_norm = features / features.norm(dim=-1, keepdim=True)
        prototypes_norm = prototypes / prototypes.norm(dim=-1, keepdim=True)

        return (features_norm @ prototypes_norm.t()) * self.logit_scale.exp()

    def forward_task_residual(self, features, prototypes=None):
        if prototypes is None:
            prot = self.adapter()
        else:
            prot = prototypes
        prot = self.adapter.base_text_features + self.adapter.alpha * prot
        
        # Ensure prototypes are on the same device and dtype as features
        if prot.device != features.device or prot.dtype != features.dtype:
            prot = prot.to(device=features.device, dtype=features.dtype)
            
        features_norm = features / features.norm(dim=-1, keepdim=True)
        prototypes_norm = prot / prot.norm(dim=-1, keepdim=True)

        return (features_norm @ prototypes_norm.t()) * self.logit_scale.exp()

    def forward_clipadapter(self, features, prototypes=None):
        if prototypes is None:
            prot = self.adapter()
        else:
            prot = prototypes
        
        # Ensure prototypes are on the same device and dtype as features
        if prot.device != features.device or prot.dtype != features.dtype:
            prot = prot.to(device=features.device, dtype=features.dtype)
            
        x = self.adapter.mlp(features)
        feats = self.adapter.ratio * x + (1 - self.adapter.ratio) * features

        features_norm = feats / feats.norm(dim=-1, keepdim=True)
        prototypes_norm = prot / prot.norm(dim=-1, keepdim=True)

        return (features_norm @ prototypes_norm.t()) * self.logit_scale.exp()

    def forward_tipadapter(self, features, prototypes=None):
        if prototypes is None:
            prot = self.adapter()
        else:
            prot = prototypes
        
        # Ensure prototypes are on the same device and dtype as features
        if prot.device != features.device or prot.dtype != features.dtype:
            prot = prot.to(device=features.device, dtype=features.dtype)
            
        features_norm = features / features.norm(dim=-1, keepdim=True)
        prototypes_norm = prot / prot.norm(dim=-1, keepdim=True)
        
        logits = (features_norm @ prototypes_norm.t()) * self.logit_scale.exp()
        
        if self.adapter.cache_keys is not None:
            ck = self.adapter.cache_keys / self.adapter.cache_keys.norm(dim=-1, keepdim=True)
            affinity = features_norm @ ck.t().float()  # Use projected features for consistency
            cache_logits = torch.exp((-1) * (self.adapter.beta - self.adapter.beta * affinity)) @ self.adapter.cache_values.float()
            logits += self.adapter.alpha * cache_logits
        
        return logits

class TrainerXCostume(SimpleTrainer):
    """A base trainer using labeled data only."""

    def run_epoch(self):
        # Start-of-epoch bookkeeping -------------------------------------------------
        self.model._in_training_epoch = True
        self.model._batch_count = 0
        self.model._epoch_count += 1
        
        # ------------------------------------------------------------------
        # Train mode for adapter + GP so gradients flow, but keep frozen CLIP
        # encoders in eval to avoid BatchNorm/Dropout updates.
        # ------------------------------------------------------------------
        self.set_model_mode("train")
        # Explicitly freeze encoders at eval behaviour
        if hasattr(self.model, "image_encoder"):
            self.model.image_encoder.eval()
        if hasattr(self.model, "text_encoder"):
            self.model.text_encoder.eval()
        # Sanity-check: log mode of key sub-modules
        if self.epoch % 10 == 0:
            print("[DEBUG] modes -> adapter.train():", self.model.adapter.training,
                  "gp.train():", self.model.gp_weighter.training if self.model.gp_weighter else "N/A",
                  "vision.eval():", not self.model.image_encoder.training,
                  "text.eval():", not self.model.text_encoder.training)

            # ---- Extra DEBUG: GP template weights for class 0 ----
            if self.model.gp_weighter is not None:
                with torch.no_grad():
                    dist = self.model.gp_weighter.get_weight_distribution()
                    weights_mean = torch.softmax(dist.mean, dim=-1)  # [K, M]
                    print("[DEBUG] GP weights (class 0, first 10 templates):",
                          weights_mean[0, :10].detach().cpu().numpy())

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

                if (self.epoch+1) % 10 == 0 or self.epoch == 0:
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

        print(f"Loading CLIP (backbone: {cfg.MODEL.BACKBONE.NAME})")
        clip_model = load_clip_to_cpu(cfg)

        if cfg.TRAINER.ADAPTER.PREC == "fp32" or cfg.TRAINER.ADAPTER.PREC == "amp":
            # CLIP's default precision is fp16
            clip_model.float()

        print("Building custom CLIP")
        self.model = CustomCLIP(cfg, classnames, clip_model)

        print("Turning off gradients in frozen modules and honour pre-set flags")
        for name, param in self.model.named_parameters():
            if name == "logit_scale":
                param.requires_grad = True
            elif "visual_proj" in name:
                param.requires_grad = cfg.TRAINER.ADAPTER.USE_GP  # train only with GP
            else:
                if ("adapter" not in name) and ("gp_weighter" not in name) and ("visual_proj" not in name):
                    param.requires_grad = False

        if cfg.MODEL.INIT_WEIGHTS:
            load_pretrained_weights(self.model.adapter, cfg.MODEL.INIT_WEIGHTS)

        self.model.to(self.device)
        self.model = self.model.float()
        
        # NOTE: only give adapter (and optionally GP) parameters to the optimizer
        if cfg.TRAINER.ADAPTER.USE_GP and self.model.gp_weighter is not None:
            # Exclude frozen adapter params (e.g., visual_projection) ---------
            adapter_params = list({id(p): p for p in self.model.adapter.parameters() if p.requires_grad}.values())
            adapter_params += [p for p in self.model.visual_proj.parameters() if p.requires_grad]
            
            print(f"Number of adapter parameters: {len(adapter_params)}")
            
            # List actual parameter names and shapes
            print("Adapter parameters:")
            for name, param in self.model.adapter.named_parameters():
                print(f"  {name}: {param.shape}")
            print("Visual projection parameters:")
            for name, param in self.model.visual_proj.named_parameters():
                print(f"  {name}: {param.shape}, trainable={param.requires_grad}")
            
            print("GP parameters:")
            for name, param in self.model.gp_weighter.named_parameters():
                print(f"  {name}: {param.shape}")
                
            # Combine all parameters into a single list for simplicity
            all_params = adapter_params + [p for p in self.model.gp_weighter.parameters() if p.requires_grad]
            self.optim = build_optimizer(all_params, cfg.OPTIM)
        else:
            # Optimise adapter (+ visual projection if trainable) and logit_scale
            baseline_params = list(self.model.adapter.parameters())
            baseline_params += [p for p in self.model.visual_proj.parameters() if p.requires_grad]
            if self.model.logit_scale.requires_grad:
                baseline_params.append(self.model.logit_scale)
            self.optim = build_optimizer(baseline_params, cfg.OPTIM)
            
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
            # Re-build optimiser including visual projection parameters
            tipa_params = list(self.model.adapter.parameters())
            tipa_params += [p for p in self.model.visual_proj.parameters() if p.requires_grad]
            self.optim = build_optimizer(tipa_params, self.cfg.OPTIM)  # Update optimizer with new params
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
            # Simple training loop
            self.before_train()
            for self.epoch in range(self.start_epoch, self.max_epoch):
                self.before_epoch()
                self.run_epoch()
                
                # Update lagrangian parameter and multiplier if needed
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
            # Exclude frozen adapter params
            adapter_params = list({id(p): p for p in self.model.adapter.parameters() if p.requires_grad}.values())
            adapter_params += [p for p in self.model.visual_proj.parameters() if p.requires_grad]
            
            # Get weight decay if available
            weight_decay = getattr(self.cfg.OPTIM, 'WEIGHT_DECAY', 0.0)
            gp_lr = getattr(self.cfg.TRAINER.ADAPTER, 'GP_LR', self.cfg.OPTIM.LR)
            
            # Create parameter groups
            param_groups = [
                {'params': adapter_params, 'lr': self.cfg.OPTIM.LR, 'weight_decay': weight_decay},
                {'params': self.model.visual_proj.parameters() if self.model.visual_proj.requires_grad else [], 'lr': self.cfg.OPTIM.LR, 'weight_decay': weight_decay},
                {'params': [p for p in self.model.gp_weighter.parameters() if p.requires_grad], 'lr': gp_lr, 'weight_decay': 0.0},
            ]
            
            # Create optimizer with parameter groups
            from torch.optim import SGD, Adam
            if self.cfg.OPTIM.NAME.lower() == "sgd":
                self.optim = SGD(param_groups, momentum=getattr(self.cfg.OPTIM, "MOMENTUM", 0.9))
            elif self.cfg.OPTIM.NAME.lower() == "adam":
                self.optim = Adam(param_groups)
            else:
                # Fallback to SGD
                self.optim = SGD(param_groups, momentum=getattr(self.cfg.OPTIM, "MOMENTUM", 0.9))
        else:
            # Re-build optimiser including visual projection parameters
            baseline_params = list(self.model.adapter.parameters())
            baseline_params += [p for p in self.model.visual_proj.parameters() if p.requires_grad]
            self.optim = build_optimizer(baseline_params, self.cfg.OPTIM)
            
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
        """
        Perform a forward pass, compute the unified ELBO loss (CE + β·KL), and
        run the backward/update step.

        Args:
            features (Tensor): Pre-computed visual features (CPU or GPU).
            labels   (Tensor): Corresponding ground-truth labels (already on
                              `self.device`).

        Returns:
            dict: Dictionary with training/test accuracy, loss, and (optionally)
                  KL-divergence statistics.
        """

        use_amp = self.cfg.TRAINER.ADAPTER.PREC == "amp"

        # Ensure tensors are on correct device
        features = torch.as_tensor(features, device=self.device)

        kl_divergence = None  # For logging purposes

        # ------------------------------------------------------------------
        #  Forward pass with Monte-Carlo samples                              
        # ------------------------------------------------------------------
        S = self.cfg.TRAINER.ADAPTER.GP_NUM_MC_SAMPLES if (self.cfg.TRAINER.ADAPTER.USE_GP and self.model.gp_weighter is not None) else 1

        with autocast(enabled=use_amp, device_type=self.device.type):
            logits_mc, kl_divergence = self.model.forward_features_mc(features, S)

            # Flatten samples for CE:   [S,B,K] → [S*B,K]
            logits_flat = logits_mc.reshape(-1, logits_mc.shape[-1])
            labels_flat = labels.unsqueeze(0).repeat(S, 1).reshape(-1)

            loss_ce = F.cross_entropy(logits_flat, labels_flat)

            # Optional zero-shot constraint (CLAP)
            if self.model.adapter.apply_constraint != "none":
                loss_constraint = self.model.adapter.zero_shot_constraint()
                loss = loss_ce + loss_constraint
            else:
                loss = loss_ce

            # Add β·KL term when GP is active
            if kl_divergence is not None:
                loss = loss + self.cfg.TRAINER.ADAPTER.GP_BETA * kl_divergence

        # ------------------------------------------------------------------
        # Backward + optimiser step
        # ------------------------------------------------------------------
        if use_amp:
            self.optim.zero_grad(set_to_none=True)
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optim)
            self.scaler.update()
        else:
            self.model_backward_and_update(loss)

        # ------------------------------------------------------------------
        # Diagnostics (test accuracy etc.)
        # ------------------------------------------------------------------
        with torch.no_grad():
            # Use deterministic prototypes (posterior mean) for evaluation
            output_test = self.model.forward_features(
                self.features_test.clone().detach().to(self.device)
            )

        # Training accuracy from mean logits across samples
        output_mean = logits_mc.mean(0).contiguous()  # [B,K]

        loss_summary = {
            "loss": loss.item(),
            "acc_train": compute_accuracy(output_mean, labels)[0].item(),
            "acc_test": compute_accuracy(output_test, self.labels_test)[0].item(),
        }

        if kl_divergence is not None:
            loss_summary["kl_divergence"] = kl_divergence.item()

        # Scheduler step once per epoch (when last batch is processed)
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

            # Handle cross-dataset evaluation: remove class-specific tensors if size mismatch
            current_num_classes = self.model.adapter.base_text_features.shape[0]
            checkpoint_num_classes = state_dict.get('base_text_features', torch.tensor([])).shape[0] if 'base_text_features' in state_dict else current_num_classes
            
            if checkpoint_num_classes != current_num_classes:
                print(f"Cross-dataset evaluation detected: checkpoint has {checkpoint_num_classes} classes, target has {current_num_classes} classes")
                print("Removing class-specific tensors from checkpoint (prototypes, base_text_features)")
                
                # Remove class-specific tensors that would cause size mismatches
                if 'prototypes' in state_dict:
                    del state_dict['prototypes']
                if 'base_text_features' in state_dict:
                    del state_dict['base_text_features']

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