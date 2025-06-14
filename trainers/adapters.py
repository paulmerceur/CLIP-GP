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
from torch.cuda.amp import GradScaler, autocast
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
    """Deterministic template list for the given dataset with improved diversity."""
    dataset = cfg.DATASET.NAME
    
    if dataset == "ImageNet":
        base = IMAGENET_TEMPLATES_SELECT.copy()
    elif cfg.TRAINER.ADAPTER.NUM_TEMPLATES > 1:
        rng = random.Random(42)  # Fixed seed for reproducibility
        base = [CUSTOM_TEMPLATES[dataset]] # Always include the custom template for this dataset
        
        # Then add diverse general templates
        remaining_count = cfg.TRAINER.ADAPTER.NUM_TEMPLATES - 1
        if remaining_count > 0:
            # Select a diverse set of templates rather than purely random
            # Sort templates by length to get varied styles
            sorted_templates = sorted(IMAGENET_TEMPLATES, key=len)
            
            # Select templates with good spacing
            if remaining_count >= len(sorted_templates):
                # If we need more templates than available, use all
                selected = sorted_templates
            else:
                # Select evenly spaced templates for diversity
                step = len(sorted_templates) // remaining_count
                selected = []
                for i in range(remaining_count):
                    idx = min(i * step, len(sorted_templates) - 1)
                    selected.append(sorted_templates[idx])
            
            base.extend(selected)
    else:
        # Single template case
        base = [CUSTOM_TEMPLATES[dataset]]
    
    print(f"Selected {len(base)} templates for GP weighting:")
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
    """Original helper - now *also* caches embeddings inside closure."""

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
        # NOTE: Initial weights are uniform after softmax, so the initial proto is a simple average.
        proto, kl = gp(text_embeds)
        return proto, text_embeds, gp, kl

    # GP disabled -> simple average
    return text_embeds.mean(1), text_embeds, None, None

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
        self.augmentations = True
        self.epochs_aumentation = 20
        self.use_gp = cfg.TRAINER.ADAPTER.USE_GP  # Store GP usage flag

        # Visual projection matrix W (as per mathematical formulation)
        # W ∈ R^{d×d} for projecting visual features: v_tilde = Wv
        feat_dim = base_text_features.shape[-1]
        
        # Make visual projection optional via config
        self.use_visual_projection = getattr(cfg.TRAINER.ADAPTER, 'USE_VISUAL_PROJECTION', True)
        
        if self.use_visual_projection:
            self.visual_projection = nn.Linear(feat_dim, feat_dim, bias=False)
            # Initialize as identity for stability
            nn.init.eye_(self.visual_projection.weight)
            # Ensure correct device and dtype
            self.visual_projection = self.visual_projection.to(device=base_text_features.device, dtype=base_text_features.dtype)
            # ------------------------------------------------------------------
            #  If GP is enabled, **freeze** the visual projection so that the
            #  optimisation pressure is forced onto the GP template weights
            #  instead of letting the very-large matrix W soak up all the
            #  gradients (which we observed in the previous runs).
            # ------------------------------------------------------------------
            # Optionally freeze visual projection when GP is active (helps isolate GP effects)
            freeze_vp = getattr(cfg.TRAINER.ADAPTER, 'GP_FREEZE_VP', True)
            if cfg.TRAINER.ADAPTER.USE_GP and freeze_vp:
                for p in self.visual_projection.parameters():
                    p.requires_grad = False
                print("[INFO] Visual projection frozen (GP active) - prototypes must adapt instead.")
            else:
                print("[INFO] Visual projection TRAINABLE while GP active (experimentation mode)")
        else:
            # Identity projection (no-op)
            self.visual_projection = nn.Identity()

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
                self.prototypes = nn.Parameter(base_text_features.clone())
                self.prototypes.requires_grad = False
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

        # Use helper to build all text-related tensors 
        base_proto, self.text_embeddings_all, self.gp_weighter, _ = _get_base_text_features(cfg, classnames, clip_model, self.text_encoder)

        # Cache embeddings for fast GP updates
        self.register_buffer("text_embeddings_static", self.text_embeddings_all.float())

        # Adapter
        self.adapter = AdapterMethod(cfg, clip_model, base_proto)

        self._in_training_epoch = False  # toggled by trainer each epoch
        self._batch_count = 0  # Track batches for GP update frequency
        self._epoch_count = 0  # Track epochs for GP update frequency
        self.gp_update_freq = cfg.TRAINER.ADAPTER.GP_UPDATE_FREQ if hasattr(cfg.TRAINER.ADAPTER, 'GP_UPDATE_FREQ') else 1
        
        # Cache current prototypes to avoid recomputation every forward pass
        self._cached_prototypes = None
        self._last_update_batch = -1

        # ------------------------------------------------------------------
        #  Two-phase fine-tuning helpers
        #   • During the GP warm-up (epoch < warmup_epochs) we optionally keep
        #     the visual projection frozen *and* skip prototype ℓ2-normalisation
        #     to amplify the effect of weight shifts.
        #   • After the warm-up we re-enable normalisation and (optionally)
        #     unfreeze the projection matrix.  Both behaviours can be toggled
        #     at runtime by the trainer via the public attributes below.
        # ------------------------------------------------------------------

        # Start with normalisation ON – can be disabled via cfg if desired.
        self.normalize_prototypes: bool = True
        # Expose a quick helper so the trainer can (un)freeze visual projection
        self.visual_projection_frozen: bool = cfg.TRAINER.ADAPTER.GP_FREEZE_VP

    def _update_prototypes_if_needed(self):
        """Update GP prototypes only when needed (not every forward pass)."""
        if self.gp_weighter is None:
            return  # No GP, nothing to update
            
        # Only update periodically to avoid constant recomputation
        should_update = (
            self._cached_prototypes is None or
            (self._in_training_epoch and 
             self._batch_count % self.gp_update_freq == 0 and
             self._batch_count != self._last_update_batch)
        )
        
        if should_update:
            with torch.no_grad():
                # Use mean for consistent behavior during both training and testing
                proto, _ = self.gp_weighter(self.text_embeddings_static, use_mean=True)
                
                # Ensure prototypes have correct device and dtype
                target_device = next(self.adapter.parameters()).device
                target_dtype = self.dtype
                
                self._cached_prototypes = proto.to(device=target_device, dtype=target_dtype)
                self.adapter.prototypes = self._cached_prototypes
                self._last_update_batch = self._batch_count

    def _gp_prototypes(self):
        """Get current GP prototypes for gradient computation during training."""
        if self.gp_weighter is None:
            return self.adapter.prototypes          # plain LP
        
        # Use MC sampling during training, mean during eval
        if self._in_training_epoch and self._batch_count == 1:
            mode = 'MC' if self.training else 'mean'

        proto, _ = self.gp_weighter(
            self.text_embeddings_static,
            use_mean=not self.training,
        )

        # --- Lightweight diagnostics (print sparingly) ------------------------
        if self._in_training_epoch and self._batch_count == 1:
            with torch.no_grad():
                q = self.gp_weighter._weight_distribution()
                probs = F.softmax(q.mean[0], dim=-1)
                entropy = -(probs * probs.log()).sum().item()
                mu_std = q.mean.std().item()
                # Additional sanity checks
                temp = self.gp_weighter.weight_temperature.item()
                if self._in_training_epoch and self._batch_count == 1 and self._epoch_count % 10 == 0:
                    print(
                        f"[INFO][GP] epoch {self._epoch_count} entropy={entropy:.3f} | mu_std={mu_std:.4f} | temp={temp:.2f} | "
                        f"probs {[round(v.item(),4) for v in probs]}"
                    )

        return proto

    def get_gp_kl_divergence(self):
        """Compute KL divergence for GP loss term."""
        if self.gp_weighter is None:
            return None
        # Always use MC sampling to get proper gradients for KL
        _, kl = self.gp_weighter(self.text_embeddings_static, use_mean=False)
        
        return kl

    def update_gp_prototypes(self):
        """Force update GP prototypes (called externally if needed)."""
        if self.gp_weighter is None:
            return
            
        with torch.no_grad():
            proto, _ = self.gp_weighter(self.text_embeddings_static, use_mean=True)
            
            # Ensure prototypes have correct device and dtype
            target_device = next(self.adapter.parameters()).device
            target_dtype = self.dtype
            
            self.adapter.prototypes = proto.to(device=target_device, dtype=target_dtype)
            self._cached_prototypes = self.adapter.prototypes

    # ----------------------------------------------------------------- #
    #  Forward passes                                                   #
    # ----------------------------------------------------------------- #
    def _forward_impl(self, x, *, is_feature: bool) -> torch.Tensor:
        # Update batch count and prototypes only when needed
        if self._in_training_epoch:
            self._batch_count += 1
            
        # Update cached prototypes periodically (not every forward pass!)
        self._update_prototypes_if_needed()

        feats = x if is_feature else self.image_encoder(x.type(self.dtype))

        # ----- GP prototypes with gradients (ONLY during training loss computation) -----
        if self.cfg.TRAINER.ADAPTER.USE_GP and self.training:
            # Get GP prototypes with gradients for this forward pass
            # This is crucial for GP parameter learning
            current_prototypes = self._gp_prototypes()
        else:
            # Use cached prototypes for inference or when GP is disabled
            current_prototypes = self.adapter.prototypes

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
    #  Adapter‑specific logits                                          #
    # ----------------------------------------------------------------- #
    def forward_lp(self, features, prototypes=None):
        if prototypes is None:
            prot = self.adapter()
        else:
            prot = prototypes
        
        # Ensure prototypes are on the same device and dtype as features
        if prot.device != features.device or prot.dtype != features.dtype:
            prot = prot.to(device=features.device, dtype=features.dtype)
            
        # Apply visual projection W: v_tilde = Wv
        feats_proj = self.adapter.visual_projection(features)
        feats_n = feats_proj / feats_proj.norm(dim=-1, keepdim=True)

        if getattr(self, "normalize_prototypes", True):
            prot_n = prot / prot.norm(dim=-1, keepdim=True)
        else:
            prot_n = prot

        return (feats_n @ prot_n.t()) * self.logit_scale.exp()

    def forward_task_residual(self, features, prototypes=None):
        if prototypes is None:
            prot = self.adapter()
        else:
            prot = prototypes
        prot = self.adapter.base_text_features + self.adapter.alpha * prot
        
        # Ensure prototypes are on the same device and dtype as features
        if prot.device != features.device or prot.dtype != features.dtype:
            prot = prot.to(device=features.device, dtype=features.dtype)
            
        # Apply visual projection W: v_tilde = Wv
        feats_proj = self.adapter.visual_projection(features)
        feats_n = feats_proj / feats_proj.norm(dim=-1, keepdim=True)
        prot_n = prot / prot.norm(dim=-1, keepdim=True)
        return (feats_n @ prot_n.t()) * self.logit_scale.exp()

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
        # Apply visual projection W: v_tilde = Wv  
        feats_proj = self.adapter.visual_projection(feats)
        feats_n = feats_proj / feats_proj.norm(dim=-1, keepdim=True)
        prot_n = prot / prot.norm(dim=-1, keepdim=True)
        return (feats_n @ prot_n.t()) * self.logit_scale.exp()

    def forward_tipadapter(self, features, prototypes=None):
        if prototypes is None:
            prot = self.adapter()
        else:
            prot = prototypes
        
        # Ensure prototypes are on the same device and dtype as features
        if prot.device != features.device or prot.dtype != features.dtype:
            prot = prot.to(device=features.device, dtype=features.dtype)
            
        # Apply visual projection W: v_tilde = Wv
        feats_proj = self.adapter.visual_projection(features)
        feats_n = feats_proj / feats_proj.norm(dim=-1, keepdim=True)
        prot_n = prot / prot.norm(dim=-1, keepdim=True)
        logits = (feats_n @ prot_n.t()) * self.logit_scale.exp()
        if self.adapter.cache_keys is not None:
            ck = self.adapter.cache_keys / self.adapter.cache_keys.norm(dim=-1, keepdim=True)
            affinity = feats_proj @ ck.t().float()  # Use projected features for consistency
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
        # Sanity-check: log mode of key sub-modules (once every 20 epochs)
        if self.epoch % 20 == 0:
            print("[DEBUG] modes -> adapter.train():", self.model.adapter.training,
                  "gp.train():", self.model.gp_weighter.training if self.model.gp_weighter else "N/A",
                  "vision.eval():", not self.model.image_encoder.training,
                  "text.eval():", not self.model.text_encoder.training)

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
            # Keep a trainable temperature (logit_scale) even though it's at top-level.
            if name == "logit_scale":
                param.requires_grad = True
            else:
                # Freeze CLIP encoders ➜ no 'adapter' nor 'gp_weighter' in the name
                if ("adapter" not in name) and ("gp_weighter" not in name):
                    param.requires_grad = False

        if cfg.MODEL.INIT_WEIGHTS:
            load_pretrained_weights(self.model.adapter, cfg.MODEL.INIT_WEIGHTS)

        self.model.to(self.device)
        self.model = self.model.float()
        
        # Force GP update to initialize prototypes properly on correct device
        if cfg.TRAINER.ADAPTER.USE_GP and self.model.gp_weighter is not None:
            self.model.update_gp_prototypes()
        
        # NOTE: only give adapter (and optionally GP) parameters to the optimizer
        if cfg.TRAINER.ADAPTER.USE_GP and self.model.gp_weighter is not None:
            # Exclude frozen adapter params (e.g., visual_projection) ---------
            adapter_params = list({id(p): p for p in self.model.adapter.parameters() if p.requires_grad}.values())
            # Add CLIP temperature if trainable and not already present
            if self.model.logit_scale.requires_grad and id(self.model.logit_scale) not in {id(p) for p in adapter_params}:
                adapter_params.append(self.model.logit_scale)
            
            print(f"Number of adapter parameters: {len(adapter_params)}")
            
            # List actual parameter names and shapes
            print("Adapter parameters:")
            for name, param in self.model.adapter.named_parameters():
                print(f"  {name}: {param.shape}")
            
            print("GP parameters:")
            for name, param in self.model.gp_weighter.named_parameters():
                print(f"  {name}: {param.shape}")
                
            # Get weight decay if available
            weight_decay = getattr(cfg.OPTIM, 'WEIGHT_DECAY', 0.0)
            gp_lr = getattr(cfg.TRAINER.ADAPTER, 'GP_LR', cfg.OPTIM.LR)
            
            # Create parameter groups
            param_groups = [
                {'params': adapter_params, 'lr': cfg.OPTIM.LR, 'weight_decay': weight_decay},
                {'params': [p for p in self.model.gp_weighter.parameters() if p.requires_grad], 'lr': gp_lr, 'weight_decay': 0.0}
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
            # -------------------- TWO-PHASE TRAIN LOOP --------------------
            #  Phase-0 setup: expose a mutable β and warm-up configuration
            self.gp_beta = self.cfg.TRAINER.ADAPTER.GP_BETA
            warmup_epochs = getattr(self.cfg.TRAINER.ADAPTER, "GP_WARMUP_EPOCHS", 20)
            phase2_lr_factor = getattr(self.cfg.TRAINER.ADAPTER, "GP_PHASE2_LR_FACTOR", 0.1)
            phase2_beta = getattr(self.cfg.TRAINER.ADAPTER, "GP_PHASE2_BETA", self.gp_beta)
            unfreeze_vp = getattr(self.cfg.TRAINER.ADAPTER, "GP_PHASE2_UNFREEZE_VP", True)

            self.before_train()
            for self.epoch in range(self.start_epoch, self.max_epoch):

                # Train and update weights per epoch
                self.before_epoch()
                self.run_epoch()

                # ---------------- Phase switch -------------------
                if (self.epoch + 1) == warmup_epochs:
                    print("[INFO] === Switching to phase-2 (joint fine-tuning) ===")

                    # 1)  Update KL weight β
                    self.gp_beta = phase2_beta
                    print(f"[INFO] New GP β = {self.gp_beta}")

                    # 2)  Un-freeze visual projection if requested
                    if unfreeze_vp and self.model.adapter.visual_projection is not None:
                        for p in self.model.adapter.visual_projection.parameters():
                            p.requires_grad = True
                        # Re-enable prototype normalisation for logits
                        if hasattr(self.model, "normalize_prototypes"):
                            self.model.normalize_prototypes = True

                    # 3)  Rebuild optimizer with (optionally) reduced GP LR and new VP params
                    adapter_params = list({id(p): p for p in self.model.adapter.parameters() if p.requires_grad}.values())
                    if self.model.logit_scale.requires_grad and id(self.model.logit_scale) not in {id(p) for p in adapter_params}:
                        adapter_params.append(self.model.logit_scale)

                    gp_params = [p for p in self.model.gp_weighter.parameters() if p.requires_grad]

                    # Update GP LR
                    gp_lr = getattr(self.cfg.TRAINER.ADAPTER, "GP_LR", self.cfg.OPTIM.LR) * phase2_lr_factor

                    from torch.optim import SGD, Adam
                    weight_decay_val = getattr(self.cfg.OPTIM, "WEIGHT_DECAY", 0.0)
                    param_groups = [
                        {"params": adapter_params, "lr": self.cfg.OPTIM.LR, "weight_decay": weight_decay_val},
                        {"params": gp_params, "lr": gp_lr, "weight_decay": 0.0},
                    ]

                    # Construct new optimizer of same type
                    if self.cfg.OPTIM.NAME.lower() == "sgd":
                        new_optim = SGD(param_groups, momentum=getattr(self.cfg.OPTIM, "MOMENTUM", 0.9))
                    else:
                        new_optim = Adam(param_groups)

                    # Replace in trainer bookkeeping
                    self.optim = new_optim
                    self.sched = build_lr_scheduler(self.optim, self.cfg.OPTIM)
                    # Update registry entry so that checkpoints contain new opt
                    self.register_model("adapter_phase2", self.model.adapter, self.optim, self.sched)

                # --------------------------------------------------

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
            # Exclude frozen adapter params (e.g., visual_projection) ---------
            adapter_params = list({id(p): p for p in self.model.adapter.parameters() if p.requires_grad}.values())
            # Add CLIP temperature if trainable and not already present
            if self.model.logit_scale.requires_grad and id(self.model.logit_scale) not in {id(p) for p in adapter_params}:
                adapter_params.append(self.model.logit_scale)
            
            # Get weight decay if available
            weight_decay = getattr(self.cfg.OPTIM, 'WEIGHT_DECAY', 0.0)
            gp_lr = getattr(self.cfg.TRAINER.ADAPTER, 'GP_LR', self.cfg.OPTIM.LR)
            
            # Create parameter groups
            param_groups = [
                {'params': adapter_params, 'lr': self.cfg.OPTIM.LR, 'weight_decay': weight_decay},
                {'params': [p for p in self.model.gp_weighter.parameters() if p.requires_grad], 'lr': gp_lr, 'weight_decay': 0.0}
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
                    kl_divergence = self.model.get_gp_kl_divergence()
                    if kl_divergence is not None:
                        beta = getattr(self, "gp_beta", self.cfg.TRAINER.ADAPTER.GP_BETA)
                        loss = loss + beta * kl_divergence
                    
                
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
            kl_divergence = None
            if self.cfg.TRAINER.ADAPTER.USE_GP and self.model.gp_weighter is not None:
                kl_divergence = self.model.get_gp_kl_divergence()
                if kl_divergence is not None:
                    beta = getattr(self, "gp_beta", self.cfg.TRAINER.ADAPTER.GP_BETA)
                    loss = loss + beta * kl_divergence
            
            self.model_backward_and_update(loss)
        
        with torch.no_grad():
            output_test = self.model.forward_features(self.features_test.clone().detach().to(self.device))

        loss_summary = {
            "loss": loss.item(),
            "acc_train": compute_accuracy(output, labels)[0].item(),
            "acc_test": compute_accuracy(output_test, self.labels_test)[0].item(),
        }
        
        # Add KL divergence to loss summary for monitoring
        if self.cfg.TRAINER.ADAPTER.USE_GP and kl_divergence is not None:
            loss_summary["kl_divergence"] = kl_divergence.item()

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