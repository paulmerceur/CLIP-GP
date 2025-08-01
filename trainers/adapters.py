import os.path as osp
from typing import List, TYPE_CHECKING
import time
import datetime
import copy
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.amp.grad_scaler import GradScaler    
from torch.amp.autocast_mode import autocast
import numpy as np
import math

from dassl.engine import TRAINER_REGISTRY, SimpleTrainer
from dassl.metrics import compute_accuracy
from dassl.utils import load_pretrained_weights, load_checkpoint, AverageMeter, MetricMeter
from dassl.optim import build_optimizer, build_lr_scheduler

from clip import clip
from datasets.imagenet_templates import IMAGENET_TEMPLATES_SELECT, IMAGENET_TEMPLATES
from .gp_template_weigher import GaussianProcessTemplateWeighter

if TYPE_CHECKING:
    from typing import Any

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.allow_tf32 = True


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
        model = None

    if model is not None:
        model = clip.build_model(state_dict or model.state_dict())
    elif state_dict is not None:
        model = clip.build_model(state_dict)
    else:
        raise RuntimeError("Unable to load CLIP model")

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


def _get_base_text_features(cfg, classnames: List[str], clip_model, text_encoder: TextEncoder, pretrained_projection: str | None = None):
    """Computes and caches embeddings inside closure."""

    device = next(text_encoder.parameters()).device

    templates = ["a photo of a {}."]

    if cfg.TRAINER.ADAPTER.NUM_TEMPLATES > 1:
        num_needed = min(
            cfg.TRAINER.ADAPTER.NUM_TEMPLATES,
            len(IMAGENET_TEMPLATES_SELECT) - 1,
        )
        templates += IMAGENET_TEMPLATES_SELECT[:num_needed]
    if cfg.TRAINER.ADAPTER.NUM_TEMPLATES > 1 + len(IMAGENET_TEMPLATES_SELECT):
        # Add templates from IMAGENET_TEMPLATES
        templates += IMAGENET_TEMPLATES[:cfg.TRAINER.ADAPTER.NUM_TEMPLATES - 1 - len(IMAGENET_TEMPLATES_SELECT)]
    print(f"[DEBUG] templates: {templates}")

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
    if cfg.TRAINER.ADAPTER.USE_GP and len(templates) > 1:
        # -------------------------------------------------------------
        # Better prior: use per-template zero-shot logits.  For each
        # class k and template m we compute the logit obtained by
        # measuring the similarity of that template to the **average**
        # prototype of the class.  This centres the GP posterior on a
        # reasonably calibrated location instead of a flat 0.
        # -------------------------------------------------------------
        with torch.no_grad():
            class_mean = text_embeds.mean(dim=1, keepdim=True)          # [K,1,D]
            zs_logits = (text_embeds * class_mean).sum(-1)              # [K,M]
        mean_init = zs_logits.to(dtype=torch.float32, device=device)
        gp = GaussianProcessTemplateWeighter(text_embeddings=text_embeds, cfg=cfg, mean_init=mean_init).to(device)
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
                # Make the zero-shot prototypes trainable
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

        self.vision_dim = clip_model.text_projection.shape[1]

        # Visual projection logic: use it for GP method OR when explicitly enabled for baselines
        if cfg.TRAINER.ADAPTER.USE_GP or cfg.TRAINER.ADAPTER.USE_VISUAL_PROJ:
            # Use a full learnable d×d projection initialised as identity
            print("[VP] Using FULL Linear projection (d×d matrix)")
            self.visual_proj = nn.Linear(self.vision_dim, self.vision_dim, bias=False)
            with torch.no_grad():
                self.visual_proj.weight.copy_(torch.eye(self.vision_dim))
        else:
            self.visual_proj = nn.Identity()

        # Build all text-related tensors 
        base_proto, self.text_embeddings_all, self.gp_weighter, _ = _get_base_text_features(cfg, classnames, clip_model, self.text_encoder)
        gp = self.gp_weighter is not None

        # Cache embeddings for fast GP updates (no need to keep in original dtype)
        self.register_buffer("text_embeddings_static", self.text_embeddings_all.float())

        # Adapter (prototypes)
        self.adapter = AdapterMethod(cfg, clip_model, base_proto)

        self._in_training_epoch = False  # toggled by trainer each epoch
        self._batch_count = 0  # Track batches for diagnostics
        self._epoch_count = 0  # Track epochs for diagnostics (for debugging)

    def get_gp_kl_divergence(self):
        """Compute KL divergence for GP loss term."""
        
        return getattr(self, "_last_gp_kl", None)

    def _forward_impl(self, x, *, is_feature: bool) -> torch.Tensor:
        # Update batch count for diagnostics
        if self._in_training_epoch:
            self._batch_count += 1
            
        # Obtain image features (pre-computed or freshly encoded)
        feats = x.type(self.dtype) if is_feature else self.image_encoder(x.type(self.dtype))

        # Apply visual projection if trainable
        if isinstance(self.visual_proj, nn.Linear):
            if feats.dtype != self.visual_proj.weight.dtype:
                feats = feats.to(dtype=self.visual_proj.weight.dtype)
            feats = self.visual_proj(feats)
            feats = feats / feats.norm(dim=-1, keepdim=True)

        # Determine current prototypes
        if self.gp_weighter is not None:
            current_prototypes, kl = self.gp_weighter.forward_and_kl()
            self._last_gp_kl = kl
        else:
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
        """Forward pass.

        During **evaluation** (i.e. ``self.training == False``) and when the GP
        template weighter is active, we average the logits obtained from multiple
        Monte-Carlo draws of the GP posterior. This retains predictive
        uncertainty at test time and noticeably improves calibration.
        """

        if (not self.training) and (self.gp_weighter is not None):
            # Encode images once to visual features
            feats = self.image_encoder(image.type(self.dtype))

            # Draw several sets of prototypes and average the resulting probabilities rather than logits.
            num_mc = max(10, self.cfg.TRAINER.ADAPTER.x * 3)
            logits_mc, _ = self.forward_features_mc(feats, num_samples=num_mc)  # [S,B,K]

            # Convert to probabilities, average, and map back to log-space
            probs_mean = torch.softmax(logits_mc, dim=-1).mean(0)  # [B,K]
            # Clamp for numerical stability before log
            logits = torch.log(probs_mean.clamp(min=1e-8))

            if return_features:
                return logits, feats
            return logits

        # Default behaviour (training or GP disabled)
        logits = self._forward_impl(image, is_feature=False)
        if return_features:
            feats = self.image_encoder(image.type(self.dtype))
            return logits, feats
        return logits

    def forward_features(self, features):
        """Forward pass for **pre-computed visual features**.

        During evaluation and when the GP weighter is present we average
        probabilities over multiple Monte-Carlo samples, identical to the
        behaviour implemented in :py:meth:`forward`.
        """

        if (not self.training) and (self.gp_weighter is not None):
            num_mc = max(10, self.cfg.TRAINER.ADAPTER.x * 3)
            logits_mc, _ = self.forward_features_mc(features, num_samples=num_mc)

            probs_mean = torch.softmax(logits_mc, dim=-1).mean(0)
            logits = torch.log(probs_mean.clamp(min=1e-8))
            return logits.to(self.dtype)

        return self._forward_impl(features, is_feature=True)

    def forward_features_mc(self, features, num_samples: int):
        """Compute logits for *num_samples* GP draws.

        Returns
        -------
        logits : Tensor
            Shape ``[S, B, K]`` where *S* is ``num_samples``.
        kl : Tensor | None
            KL divergence term from the GP (or *None* if GP disabled).
        """
        # Obtain image features once (shared across samples)
        feats = features.type(self.dtype)

        # Apply visual projection if trainable
        if isinstance(self.visual_proj, nn.Linear):
            if feats.dtype != self.visual_proj.weight.dtype:
                feats = feats.to(dtype=self.visual_proj.weight.dtype)
            feats = self.visual_proj(feats)
            feats = feats / feats.norm(dim=-1, keepdim=True)

        # If no GP, fall back to deterministic adapter
        if self.gp_weighter is None:
            logits = self.forward_lp(feats)  # [B,K]
            return logits.unsqueeze(0), None

        # Always use stochastic sampling
        prototypes_s = self.gp_weighter.sample_prototypes(num_samples)  # [S,K,D]

        kl = self.gp_weighter.variational_strategy.kl_divergence().sum()

        # Normalise once for stability
        feats_norm = feats / feats.norm(dim=-1, keepdim=True)          # [B,D]
        prot_norm = prototypes_s / prototypes_s.norm(dim=-1, keepdim=True)  # [S,K,D]

        # Perform computation in float32 for numerical stability
        feats32 = feats_norm.float()
        prot32  = prot_norm.float()

        # Compute logits in fp32:  (B,D)  ·  (S,K,D)ᵀ  →  (S,B,K)
        logits_fp32 = torch.einsum("bd,skd->sbk", feats32, prot32)
        logits_fp32 = logits_fp32 * self.logit_scale.exp().float()

        # Cast back to the original CLIP dtype (usually fp16) before returning
        return logits_fp32.to(self.dtype), kl

    def forward_lp(self, features, prototypes=None):
        if prototypes is None:
            prototypes = self.adapter()
        
        # Ensure prototypes are on the same device and dtype as features
        if prototypes.device != features.device or prototypes.dtype != features.dtype:
            prototypes = prototypes.to(device=features.device, dtype=features.dtype)
            
        # Compute logits in fp32 to prevent numerical underflow
        feats32 = (features / features.norm(dim=-1, keepdim=True)).float()
        prot32 = (prototypes / prototypes.norm(dim=-1, keepdim=True)).float()

        logits_fp32 = (feats32 @ prot32.t()) * self.logit_scale.exp().float()
        return logits_fp32.to(self.dtype)

    def forward_task_residual(self, features, prototypes=None):
        if prototypes is None:
            prot = self.adapter()
        else:
            prot = prototypes
        prot = self.adapter.base_text_features + self.adapter.alpha * prot
        
        # Ensure prototypes are on the same device and dtype as features
        if prot.device != features.device or prot.dtype != features.dtype:
            prot = prot.to(device=features.device, dtype=features.dtype)
            
        feats32 = (features / features.norm(dim=-1, keepdim=True)).float()
        prot32  = (prot / prot.norm(dim=-1, keepdim=True)).float()

        logits_fp32 = (feats32 @ prot32.t()) * self.logit_scale.exp().float()
        return logits_fp32.to(self.dtype)

    def forward_clipadapter(self, features, prototypes=None):
        if prototypes is None:
            prot = self.adapter()
        else:
            prot = prototypes
        
        # Ensure prototypes are on the same device and dtype as features
        if prot.device != features.device or prot.dtype != features.dtype:
            prot = prot.to(device=features.device, dtype=features.dtype)
            
        # CLIP-Adapter feature blend
        x = self.adapter.mlp(features)
        feats = self.adapter.ratio * x + (1 - self.adapter.ratio) * features

        feats32 = (feats / feats.norm(dim=-1, keepdim=True)).float()
        prot32 = (prot / prot.norm(dim=-1, keepdim=True)).float()
        logits_fp32 = (feats32 @ prot32.t()) * self.logit_scale.exp().float()
        return logits_fp32.to(self.dtype)

    def forward_tipadapter(self, features, prototypes=None):
        if prototypes is None:
            prot = self.adapter()
        else:
            prot = prototypes
        
        # Ensure prototypes are on the same device and dtype as features
        if prot.device != features.device or prot.dtype != features.dtype:
            prot = prot.to(device=features.device, dtype=features.dtype)
            
        feats32 = (features / features.norm(dim=-1, keepdim=True)).float()
        prot32  = (prot / prot.norm(dim=-1, keepdim=True)).float()

        logits_fp32 = (feats32 @ prot32.t()) * self.logit_scale.exp().float()
        logits = logits_fp32

        if self.adapter.cache_keys is not None and self.adapter.cache_values is not None:
            ck = self.adapter.cache_keys / self.adapter.cache_keys.norm(dim=-1, keepdim=True)
            affinity = feats32 @ ck.t().float()  # Use projected features for consistency
            cache_logits = torch.exp((-1) * (self.adapter.beta - self.adapter.beta * affinity)) @ self.adapter.cache_values.float()
            logits += self.adapter.alpha * cache_logits
        
        return logits.to(self.dtype)

class TrainerXCostume(SimpleTrainer):
    """A base trainer using labeled data only."""
    
    model: "CustomCLIP"  # Type annotation to help linter
    features_train: torch.Tensor
    labels_train: torch.Tensor
    features_test: torch.Tensor
    labels_test: torch.Tensor
    batch_size: int

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

            # ---- DEBUG: GP template weights for class 0 ----
            if self.model.gp_weighter is not None:
                with torch.no_grad():
                    dist = self.model.gp_weighter.get_weight_distribution()
                    w_mean = torch.softmax(dist.mean, dim=-1)      # K × M
                    w_std  = dist.variance.sqrt()
                    print("\n")
                    print(f"[DEBUG] w_mean[0]: {w_mean[0,:7].cpu().numpy()}")
                    print(f"[DEBUG] w_std [0]: {w_std [0,:7].cpu().numpy()}")

        # Init kpis tracker
        losses = MetricMeter()
        batch_time = AverageMeter()
        data_time = AverageMeter()

        # Set number of batches to sample
        self.num_batches = len(self.train_loader_x)
        self.batch_size = self.train_loader_x.batch_size or 1  # Default to 1 if None

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

        # Initialize loss_summary in case loop doesn't execute
        loss_summary = {"loss": 0.0, "acc_train": 0.0, "acc_test": 0.0}

        end = time.time()
        for self.batch_idx in range(self.num_batches):
            batch_init = self.batch_idx * self.batch_size
            batch_end = (self.batch_idx + 1) * self.batch_size

            data_time.update(time.time() - end)
            batch_data = (features[batch_init:batch_end], labels[batch_init:batch_end])
            loss_summary = self.forward_backward(batch_data)
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


@TRAINER_REGISTRY.register()  # type: ignore[misc]
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

        for name, param in self.model.named_parameters():
            if name == "logit_scale":
                param.requires_grad = True  # allow adaptation with regularisation
            elif "visual_proj" in name and (cfg.TRAINER.ADAPTER.USE_GP or cfg.TRAINER.ADAPTER.USE_VISUAL_PROJ):
                param.requires_grad = True
            else:
                if ("adapter" not in name) and ("gp_weighter" not in name) and ("visual_proj" not in name):
                    param.requires_grad = False

        if cfg.MODEL.INIT_WEIGHTS:
            load_pretrained_weights(self.model.adapter, cfg.MODEL.INIT_WEIGHTS)

        # Store reference value of logit_scale for later regularisation
        self.logit_scale_ref = self.model.logit_scale.clone().detach()

        self.model.to(self.device)
        self.model = self.model.float()
        
        # NOTE: only give adapter (and optionally GP) parameters to the optimizer
        if cfg.TRAINER.ADAPTER.USE_GP and self.model.gp_weighter is not None:
            # param-group 1: adapter, visual proj, logit_scale
            base_params = [
                p for p in self.model.adapter.parameters() if p.requires_grad
            ] + [p for p in self.model.visual_proj.parameters() if p.requires_grad]

            # make sure logit_scale is updated
            if self.model.logit_scale.requires_grad:
                base_params.append(self.model.logit_scale)

            # param-group 2: GP parameters (usually need a smaller LR)
            gp_params = [p for p in self.model.gp_weighter.parameters() if p.requires_grad]

            # visual_proj parameters are already in the list above – avoid double insertion
            seen = set()
            base_params_unique = []
            for p in base_params:
                if id(p) not in seen:
                    base_params_unique.append(p)
                    seen.add(id(p))

            param_groups = [
                {
                    'params': base_params_unique,
                    'lr': cfg.OPTIM.LR,
                    'weight_decay': getattr(cfg.OPTIM, 'WEIGHT_DECAY', 0.0)
                },
                {
                    'params': [p for p in gp_params if p.requires_grad],
                    'lr': float(getattr(cfg.TRAINER.ADAPTER, 'GP_LR', cfg.OPTIM.LR)),
                    'weight_decay': 0.0
                },
            ]

            optim_map = {"sgd": torch.optim.SGD, "adam": torch.optim.Adam}
            BaseOptim = optim_map.get(cfg.OPTIM.NAME.lower(), torch.optim.SGD)

            # Use the global optimiser type for all parameter groups
            self.optim = BaseOptim(param_groups)
        else:
            # Optimise adapter (+ visual projection if trainable) and logit_scale
            baseline_params = list(self.model.adapter.parameters())
            baseline_params += [p for p in self.model.visual_proj.parameters() if p.requires_grad]
            self.optim = build_optimizer(baseline_params, cfg.OPTIM)
            for i, pg in enumerate(self.optim.param_groups):
                n_params = sum(p.numel() for p in pg['params'])
                print(f"[DEBUG] Optim group {i}: {n_params} params · lr={pg['lr']} · wd={pg.get('weight_decay',0)}")
            
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
            self.model.adapter.device = self.device  # type: ignore[assignment]
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

                # Initialize loss_summary in case loops don't execute
                loss_summary = {"loss": 0.0, "acc_train": 0.0, "acc_test": 0.0}

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
                {'params': [p for p in self.model.gp_weighter.parameters() if p.requires_grad], 'lr': gp_lr, 'weight_decay': 0.0},
            ]
            
            # Create optimizer with parameter groups
            from torch.optim import SGD, Adam
            if self.cfg.OPTIM.NAME.lower() == "sgd":
                self.optim = SGD([param_groups[0]], momentum=getattr(self.cfg.OPTIM, "MOMENTUM", 0.9))
            elif self.cfg.OPTIM.NAME.lower() == "adam":
                self.optim = Adam(param_groups)
            else:
                # Fallback to SGD
                self.optim = SGD([param_groups[0]], momentum=getattr(self.cfg.OPTIM, "MOMENTUM", 0.9))
        else:
            # Re-build optimiser including visual projection parameters
            baseline_params = list(self.model.adapter.parameters())
            baseline_params += [p for p in self.model.visual_proj.parameters() if p.requires_grad]
            self.optim = build_optimizer(baseline_params, self.cfg.OPTIM)
            
        self.sched = build_lr_scheduler(self.optim, self.cfg.OPTIM)
        self._models.popitem()
        self._optims.popitem()
        self._scheds.popitem()
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

    def forward_backward(self, batch):
        """
        Perform a forward pass, compute the unified ELBO loss (CE + β·KL), and
        run the backward/update step.

        Args:
            batch (tuple): Tuple of (features, labels) where:
                features (Tensor): Pre-computed visual features (CPU or GPU).
                labels   (Tensor): Corresponding ground-truth labels (already on
                                  `self.device`).

        Returns:
            dict: Dictionary with training/test accuracy, loss, and (optionally)
                  KL-divergence statistics.
        """
        features, labels = batch

        use_amp = self.cfg.TRAINER.ADAPTER.PREC == "amp"

        # Ensure tensors are on correct device
        features = torch.as_tensor(features, device=self.device)

        # Forward pass with Monte-Carlo samples
        kl_divergence = None  # For logging purposes
        S = self.cfg.TRAINER.ADAPTER.GP_NUM_MC_SAMPLES if (self.cfg.TRAINER.ADAPTER.USE_GP and self.model.gp_weighter is not None) else 1

        with autocast(enabled=use_amp, device_type=self.device.type):
            logits_mc, kl_divergence = self.model.forward_features_mc(features, S)

            # logits_mc: [S,B,K]
            # 1) Compute per-sample log-probabilities
            logprob_mc = F.log_softmax(logits_mc, dim=-1)  # [S,B,K]

            # 2) Monte-Carlo estimate of expected log-probability  E_q[log p(y|⋅)]
            logprob = logprob_mc.mean(dim=0)  # [B,K]

            # 3) Negative log-likelihood (cross-entropy)
            loss_ce = F.nll_loss(logprob, labels)

            # No extra regularisation terms—keep only CE and optional GP KL
            loss = loss_ce

            # -------------------------------------------------------------
            # β·KL (GP regulariser) from GP posterior
            # -------------------------------------------------------------
            beta = getattr(self.cfg.TRAINER.ADAPTER, "GP_BETA", 0)
            if kl_divergence is not None:
                # Normalise KL by the number of images in the batch so it
                # scales the same way as the cross-entropy.
                kl_per_image = kl_divergence / labels.size(0)
                loss = loss + beta * kl_per_image

            # -------------------------------------------------------------
            # L2 regularization on visual projection
            # -------------------------------------------------------------
            w_reg_lambda = getattr(self.cfg.TRAINER.ADAPTER, "GP_W_REG_COEF", 0)
            shots = max(1, int(self.cfg.DATASET.NUM_SHOTS))
            if self.model.visual_proj is not None and isinstance(self.model.visual_proj, nn.Linear):
                d = self.model.visual_proj.weight.size(0)
                eye = torch.eye(d, device=self.model.visual_proj.weight.device)
                diff = (self.model.visual_proj.weight - eye).pow(2).sum()
                loss += w_reg_lambda * diff / (labels.size(0) * shots)

        # -------------------
        # Backward + optimiser step with optional gradient clipping on GP
        # -------------------
        clip_val = 1.0  # hard-coded max-norm for GP parameters

        if use_amp and self.scaler is not None:
            self.optim.zero_grad(set_to_none=True)
            self.scaler.scale(loss).backward()

            # Unscale before clipping so the norm is computed in fp32
            self.scaler.unscale_(self.optim)
            if self.model.gp_weighter is not None:
                torch.nn.utils.clip_grad_norm_(self.model.gp_weighter.parameters(), clip_val)

            # _JointOptimizer.step() will forward to both underlying optimisers
            self.scaler.step(self.optim)
            self.scaler.update()
        else:
            self.optim.zero_grad(set_to_none=True)
            loss.backward()

            if self.model.gp_weighter is not None:
                torch.nn.utils.clip_grad_norm_(self.model.gp_weighter.parameters(), clip_val)

            # _JointOptimizer forwards .step() to each underlying optimiser
            self.optim.step()

        # Diagnostics (test accuracy etc.)
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

    def load_model(self, directory, epoch=None):
        cfg = self.cfg  # Get cfg from instance instead of parameter
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
            if not transforms and self.val_loader is not None:
                try:
                    data_loader.dataset.transform = self.val_loader.dataset.transform  # type: ignore[attr-defined]
                except AttributeError:
                    # Skip if dataset or transform attributes don't exist
                    pass

            # Set data loader with drop last to false for not losing samples
            data_loader = torch.utils.data.DataLoader(
                copy.deepcopy(self.train_loader_x.dataset), batch_size=self.train_loader_x.batch_size,
                sampler=self.train_loader_x.sampler, num_workers=self.train_loader_x.num_workers,
                drop_last=False, pin_memory=False)

        elif partition == "val":
            data_loader = copy.deepcopy(self.val_loader)
        elif partition == "test":
            data_loader = copy.deepcopy(self.test_loader)
        else:
            assert False

        if "TipA" not in self.model.adapter.initialization:

            labels_ds, logits_ds, features_ds = [], [], []
            for rep in range(reps):
                if data_loader is not None:
                    for batch_idx, batch in enumerate(data_loader):
                        with torch.no_grad():
                            input, label = self.parse_batch_test(batch)
                            logits, features = self.model(input,  return_features=True)
                            labels_ds.append(label)
                            logits_ds.append(logits.cpu())
                            features_ds.append(features.cpu())

            # Concatenate outputs
            labels_ds = torch.cat(labels_ds, dim=0)
            logits_ds = torch.cat(logits_ds, dim=0)
            features_ds = torch.cat(features_ds, dim=0)

        else:

            labels_ds, logits_ds, features_ds = [], [], []
            for rep in range(reps):
                labels_ds_irep, logits_dsirep, features_ds_irep = [], [], []
                if data_loader is not None:
                    for batch_idx, batch in enumerate(data_loader):
                        with torch.no_grad():
                            input, label = self.parse_batch_test(batch)
                            logits, features = self.model(input, return_features=True)
                            labels_ds_irep.append(label)
                            logits_dsirep.append(logits.cpu())
                            features_ds_irep.append(features.cpu())
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

# -----------------------------------------------------------------------------
# Helper class: wrap multiple optimisers so the training loop can treat them as
# a single object supporting .zero_grad(), .step(), .param_groups, ...
# -----------------------------------------------------------------------------


class _JointOptimizer(torch.optim.Optimizer):
    """Thin wrapper that forwards calls to several underlying optimisers.

    Only the subset of the *torch.optim.Optimizer* interface used in this
    training code is implemented (``zero_grad``, ``step``, ``state_dict``,
    ``load_state_dict`` and the *param_groups* property).
    """

    def __init__(self, optim_list):
        if not isinstance(optim_list, (list, tuple)) or len(optim_list) == 0:
            raise ValueError("optim_list must be a non-empty list/tuple")
        self._optimisers = list(optim_list)

    # -------------------------------------------------------------
    # Minimal interface required by the rest of the codebase
    # -------------------------------------------------------------

    def zero_grad(self, set_to_none: bool | None = None):  # noqa: D401
        for opt in self._optimisers:
            if set_to_none is None:
                opt.zero_grad()
            else:
                opt.zero_grad(set_to_none=set_to_none)

    def step(self, closure=None):  # noqa: D401  # type: ignore[override]
        for opt in self._optimisers:
            opt.step(closure)

    # ----------------------
    # Checkpoint utilities
    # ----------------------
    def state_dict(self):  # noqa: D401  # type: ignore[override]
        return [opt.state_dict() for opt in self._optimisers]

    def load_state_dict(self, state_dict):  # noqa: D401
        if not isinstance(state_dict, (list, tuple)) or len(state_dict) != len(self._optimisers):
            raise ValueError("state_dict must match the optimisers list length")
        for opt, sd in zip(self._optimisers, state_dict):
            opt.load_state_dict(sd)

    # --------------
    # Param groups
    # --------------
    @property
    def param_groups(self):  # noqa: D401  # type: ignore[override]
        groups = []
        for opt in self._optimisers:
            groups.extend(opt.param_groups)
        return groups