from typing import cast
import time
import datetime
import torch
import torch.nn as nn
from torch.nn import functional as F
import gpytorch  # used for ELBO in joint GP training
from torch.amp.grad_scaler import GradScaler    
import numpy as np
import math

from utils.trainer import BaseTrainer
from utils.metrics import compute_accuracy, AverageMeter
 
from utils.optimization import build_optimizer, build_lr_scheduler, build_optimizer_from_param_groups
from utils.trainer_registry import TRAINER_REGISTRY

from clip import clip
from datasets.imagenet_templates import IMAGENET_TEMPLATES_SELECT, IMAGENET_TEMPLATES
from .gp_template_weigher import GaussianProcessTemplateWeighter

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.allow_tf32 = True


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
        x = x.permute(1, 0, 2) # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2) # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection

        return x


def load_clip_to_cpu(config):
    backbone_name = config.model.backbone_name
    url = clip._MODELS[backbone_name]
    model_path = clip._download(url)

    try:
        # loading JIT archive
        jit_model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = jit_model.state_dict()
    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu")

    model = clip.build_model(state_dict)
    return model


def _get_base_text_features(config, classnames, clip_model, text_encoder=None):
    """Extract text features for all templates and classes."""
    device = next(clip_model.parameters()).device
    
    # Get template strings 
    templates = ["a photo of a {}."]
    
    # Add templates from IMAGENET_TEMPLATES_SELECT
    if config.adapter.num_templates > 1:
        num_needed = min(
            config.adapter.num_templates - 1,
            len(IMAGENET_TEMPLATES_SELECT),
        )
        templates += IMAGENET_TEMPLATES_SELECT[:num_needed]
    # If we need more templates, add the rest from IMAGENET_TEMPLATES
    if config.adapter.num_templates > 1 + len(IMAGENET_TEMPLATES_SELECT):
        templates += IMAGENET_TEMPLATES[:config.adapter.num_templates - 1 - len(IMAGENET_TEMPLATES_SELECT)]
    
    # Encode all prompts once - returned tensor is reused by caller.
    emb_list = []
    with torch.no_grad():
        for name in classnames:
            tok = clip.tokenize([t.format(name) for t in templates]).to(device)
            if text_encoder is not None:
                e = clip_model.token_embedding(tok).type(clip_model.dtype)
                emb = text_encoder(e, tok)
            else:
                emb = clip_model.encode_text(tok)
            emb_list.append(emb)
    text_embeds = torch.stack(emb_list) # [K,M,D]

    # GP disabled -> simple average
    return text_embeds.mean(1), text_embeds


class LinearAdapter(nn.Module):
    """Legacy linear probe (not used in training)."""

    def __init__(self, base_text_features: torch.Tensor):
        super().__init__()
        self.register_buffer("base_text_features", base_text_features)
        self.base_text_features = base_text_features
        # Per-class l2 regularization multipliers
        num_classes = base_text_features.size(0)
        self.register_buffer("alpha_constraint", torch.ones(num_classes, dtype=base_text_features.dtype, device=base_text_features.device))

        # ZS Initialization
        self.prototypes = nn.Parameter(base_text_features.clone())

    def constraint(self):
        # Compare raw embeddings; forward already normalizes for logits
        dissimilitude = (self.prototypes - self.base_text_features.clone()).pow(2).sum(dim=-1)
        alpha = cast(torch.Tensor, self.alpha_constraint)
        return torch.mean(alpha * dissimilitude)

    @torch.no_grad()
    def init_alpha_from_zero_shot(self, labels: torch.Tensor, logits: torch.Tensor):
        """
        Initialize per-class alpha multipliers from zero-shot predictions.
        Uses per-class error rate, normalized to have mean 1 for scale stability.
        """
        alpha_buf = cast(torch.Tensor, self.alpha_constraint)
        dev = alpha_buf.device
        labels = labels.to(dev)
        logits = logits.to(dev)
        preds = logits.argmax(dim=1)
        num_classes = int(alpha_buf.shape[0])
        # Compute per-class counts and corrects via bincount
        labels_i64 = labels.to(torch.int64)
        counts = torch.bincount(labels_i64, minlength=num_classes).clamp_min(1)
        correct_labels = labels_i64[preds.eq(labels)]
        corrects = torch.bincount(correct_labels, minlength=num_classes)
        acc = corrects.float() / counts.float()
        error = 1.0 - acc
        # Normalize to mean 1
        mean_error = error.mean().clamp_min(1e-6)
        alpha = error / mean_error
        alpha_buf.copy_(alpha.to(dtype=alpha_buf.dtype))

    def forward(self) -> torch.Tensor:
        return self.prototypes


class CustomCLIP(nn.Module):
    """Custom CLIP model with adapter and optional GP weighting."""
    
    def __init__(self, config, classnames, clip_model):
        super().__init__()
        self.config = config
        # Store CLIP components
        self.image_encoder = clip_model.visual
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype
        
        # Create TextEncoder for proper text encoding
        self.text_encoder = TextEncoder(clip_model)
        
        # Get text features and setup GP if needed
        base_text_features, self.text_embeddings_all = _get_base_text_features(config, classnames, clip_model, self.text_encoder)
        
        # Register base text features (frozen zero-shot prototypes)
        self.register_buffer("base_text_features", base_text_features)

        # Learnable visual projection W (full-rank, bias-free), initialized to identity
        dim = int(base_text_features.shape[-1])
        self.visual_proj = nn.Linear(dim, dim, bias=False)
        with torch.no_grad():
            eye = torch.eye(dim)
            self.visual_proj.weight.copy_(eye)
        # Align projection layer dtype/device with CLIP modules
        try:
            target_device = next(self.image_encoder.parameters()).device
        except StopIteration:
            target_device = base_text_features.device
        self.visual_proj.to(device=target_device, dtype=self.dtype)

        # Create GP weighter if needed
        self.gp_weighter = None
        self.gp_num_mc_samples = int(getattr(config.adapter, 'gp_num_mc_samples', 1) or 1)
        if getattr(config.adapter, 'use_gp', False):
            self.gp_weighter = GaussianProcessTemplateWeighter(
                text_embeddings=self.text_embeddings_all,
                cfg=config,
            )
    
    def forward_prototypes(self, num_samples: int = 1):
        """Get class prototypes; if GP is enabled, average over num_samples.

        When GP is disabled, use the zero-shot base text features (no trainable
        linear probe over class prototypes).
        """
        target_device = next(self.parameters()).device
        if self.gp_weighter is not None:
            # Prefer deterministic posterior-mean prototypes when not sampling
            if num_samples <= 1:
                current_prototypes = self.gp_weighter.prototypes_from_posterior_mean()
            else:
                proto_s = self.gp_weighter.sample_prototypes(num_samples)  # [S,K,D]
                current_prototypes = proto_s.mean(dim=0)

            if current_prototypes.device != target_device:
                current_prototypes = current_prototypes.to(target_device)
            return current_prototypes
        # Baseline when GP is disabled: use zero-shot base text features (frozen)
        base = cast(torch.Tensor, self.base_text_features)
        if base.device != target_device:
            base = base.to(target_device)
        return base
    
    def forward_features(self, features: torch.Tensor) -> torch.Tensor:
        """Compute logits from precomputed visual features (CLAP-style forward_lp).

        This normalizes features, obtains (possibly GP-sampled) prototypes,
        normalizes them, and returns the scaled cosine similarities.
        """
        num_samples = self.gp_num_mc_samples
        prototypes = self.forward_prototypes(num_samples=num_samples)
        
        # Ensure same dtype/device
        if features.dtype != prototypes.dtype:
            prototypes = prototypes.to(dtype=features.dtype)
        if features.device != prototypes.device:
            prototypes = prototypes.to(device=features.device)

        # Ensure features match projection dtype/device before matmul
        proj_weight = self.visual_proj.weight
        if features.dtype != proj_weight.dtype:
            features = features.to(dtype=proj_weight.dtype)
        if features.device != proj_weight.device:
            features = features.to(device=proj_weight.device)
        
        # Apply learnable visual projection W before normalization
        projected = self.visual_proj(features)
        features_norm = F.normalize(projected, p=2, dim=-1)
        prototypes_norm = F.normalize(prototypes, p=2, dim=-1)
        features_norm = features_norm.to(prototypes_norm.dtype)
        scale = self.logit_scale.exp().to(features_norm.dtype)
        return scale * (features_norm @ prototypes_norm.t())

    def forward(self, image: torch.Tensor, return_features: bool = False):
        """CLAP-like forward.

        - Extract image features
        - Compute logits via forward_features (normalization + prototypes)
        - Optionally return both logits and features
        """
        features = self.image_encoder(image.type(self.dtype))
        logits = self.forward_features(features)
        if return_features:
            return logits, features
        return logits


@TRAINER_REGISTRY.register("Trainer")
class Trainer(BaseTrainer):
    """Unified adapter trainer supporting both baseline and GP methods."""
    
    def __init__(self, config, dataset_manager):
        super().__init__(config, dataset_manager)

    def check_cfg(self, config) -> None:
        assert config.adapter.prec in ["fp16", "fp32", "amp"]

    def build_model(self):
        config = self.config
        classnames = self.dm.dataset.classnames

        print(f"Loading CLIP (backbone: {config.model.backbone_name})")
        clip_model = load_clip_to_cpu(config)

        # Move CLIP to target device BEFORE building CustomCLIP so that
        # prompt/text encoding in _get_base_text_features runs on GPU.
        clip_model = clip_model.to(self.device)

        # Precision handling:
        # - CLIP's build_model already applies selective fp16 via convert_weights.
        # - For fp32 or amp we upcast to float, but we avoid generic .half() to
        #   preserve LN/buffer dtypes expected by CLIP.
        if config.adapter.prec == "fp32" or config.adapter.prec == "amp":
            clip_model.float()

        print("Building custom CLIP")
        self.model = CustomCLIP(config, classnames, clip_model)
        self.model.to(self.device)

        # Setup parameter groups: train only visual_proj and GP (if enabled)
        for name, param in self.model.named_parameters():
            if "visual_proj" in name or "gp_weighter" in name:
                param.requires_grad = True
            else:
                param.requires_grad = False

        # Setup optimizer with different learning rates for GP
        if config.adapter.use_gp and self.model.gp_weighter is not None:
            # Two parameter groups: base params and GP params
            base_params = []
            base_params.extend([p for p in self.model.visual_proj.parameters() if p.requires_grad])
            gp_params = [p for p in self.model.gp_weighter.parameters() if p.requires_grad]

            param_groups = [
                {
                    'params': base_params,
                    'lr': float(config.optim.lr),
                    'weight_decay': float(config.optim.weight_decay)
                },
                {
                    'params': gp_params,
                    'lr': float(config.adapter.gp_lr),
                    'weight_decay': 0.0
                },
            ]

            # Use generic builder (supports sgd/adam/adamw with param groups)
            self.optim = build_optimizer_from_param_groups(param_groups, config.optim)
            self.sched = build_lr_scheduler(self.optim, config.optim)
        else:
            # Single parameter group for baseline
            baseline_params = []
            baseline_params.extend(list(self.model.visual_proj.parameters()))
            
            # Use utils optimization functions
            self.optim = build_optimizer(baseline_params, config.optim)
            self.sched = build_lr_scheduler(self.optim, config.optim)
            
        self.scaler = GradScaler() if config.adapter.prec == "amp" else None

    def forward_backward(self, batch):
        """Forward pass and backward pass with loss computation."""
        model = cast(CustomCLIP, self.model)
        features, labels = batch
        # Convert to tensors and move to device
        features = torch.tensor(features, dtype=torch.float32).to(self.device)
        if not torch.is_tensor(labels):
            labels = torch.tensor(labels).to(self.device)
        else:
            labels = labels.detach().clone().to(self.device)
        projected_features = features
        
        # Get prototypes (for diagnostics only)
        prototypes = model.forward_prototypes(num_samples=int(getattr(self.config.adapter, 'gp_num_mc_samples', 1) or 1))
        # Track prototype norm stats (useful to spot collapse/explosions)
        try:
            with torch.no_grad():
                proto_norms = prototypes.norm(dim=-1)
                self._dbg_proto_stats = {
                    "mean": float(proto_norms.mean().item()),
                    "std": float(proto_norms.std(unbiased=False).item()),
                    "min": float(proto_norms.min().item()),
                    "max": float(proto_norms.max().item()),
                }
        except Exception:
            self._dbg_proto_stats = None
        
        if projected_features.dtype != prototypes.dtype:
            prototypes = prototypes.to(dtype=projected_features.dtype)
        if projected_features.device != prototypes.device:
            prototypes = prototypes.to(device=projected_features.device)
        
        # Compute loss with Monte Carlo expectation over GP prototypes (if enabled)
        num_samples = int(getattr(self.config.adapter, 'gp_num_mc_samples', 1) or 1)
        loss = self.compute_loss(projected_features, labels, num_samples=num_samples)

        # Compute logits via the model's centralized path for metrics
        logits = model.forward_features(projected_features)

        # Backward pass
        self._backward_and_update(loss)
        
        # Compute accuracies for logging
        with torch.no_grad():
            # Training accuracy
            acc_train = compute_accuracy(logits, labels)[0]
            # Test accuracy (using stored test features)
            test_features = self.features_test.to(self.device)
            # Test features are CLIP visual features
            test_projected = test_features
            # Quick test metric: use the same unified path
            num_samples = int(getattr(self.config.adapter, 'gp_num_mc_samples', 1) or 1)
            test_prototypes = model.forward_prototypes(num_samples=num_samples)
            if test_projected.dtype != test_prototypes.dtype:
                test_prototypes = test_prototypes.to(dtype=test_projected.dtype)
            test_logits = model.forward_features(test_projected)
            acc_test = compute_accuracy(test_logits, self.labels_test.to(self.device))[0]
        return {
            "loss": loss.item(),
            "acc_train": acc_train,
            "acc_test": acc_test
        }

    def compute_loss(self, features: torch.Tensor, labels: torch.Tensor, num_samples: int = 1):
        """Compute loss using MC expectation over GP prototypes when enabled.

        If GP is enabled and num_samples > 1, draws num_samples prototype sets,
        computes per-sample cross-entropy, and averages them. Otherwise falls
        back to a single forward.
        """
        from typing import cast
        model = cast(CustomCLIP, self.model)

        use_gp = bool(getattr(self.config.adapter, 'use_gp', False) and getattr(model, 'gp_weighter', None) is not None)
        gp_weighter = getattr(model, 'gp_weighter', None)
        num_samples = int(num_samples or 1)

        if use_gp and num_samples > 1 and gp_weighter is not None:
            # Sample S prototype sets [S,K,D] in fp32 for stability
            protos = gp_weighter.sample_prototypes(num_samples)  # [S,K,D]
            # Ensure device/dtype match
            proj_weight = model.visual_proj.weight
            target_device = proj_weight.device
            target_dtype = proj_weight.dtype
            if protos.device != target_device:
                protos = protos.to(device=target_device)
            if protos.dtype != target_dtype:
                protos = protos.to(dtype=target_dtype)

            # Normalize features once
            f = features
            if f.device != target_device:
                f = f.to(device=target_device)
            if f.dtype != target_dtype:
                f = f.to(dtype=target_dtype)
            projected = model.visual_proj(f)
            features_norm = F.normalize(projected, p=2, dim=-1)
            scale = model.logit_scale.exp().to(dtype=features_norm.dtype)

            ce_vals = []
            for s in range(num_samples):
                prototypes_s = protos[s]  # [K,D]
                prototypes_norm = F.normalize(prototypes_s, p=2, dim=-1)
                logits_s = scale * (features_norm @ prototypes_norm.t())
                ce_vals.append(F.cross_entropy(logits_s, labels))
            ce_loss = torch.stack(ce_vals, dim=0).mean()
        else:
            # Single-sample path (or GP disabled): reuse model path
            f = features
            proj_weight = model.visual_proj.weight
            if f.device != proj_weight.device:
                f = f.to(device=proj_weight.device)
            if f.dtype != proj_weight.dtype:
                f = f.to(dtype=proj_weight.dtype)
            logits = model.forward_features(f)
            ce_loss = F.cross_entropy(logits, labels)

        total_loss = ce_loss

        # If joint GP training is enabled, add ELBO (data term) and KL
        if use_gp and gp_weighter is not None and bool(getattr(self.config.adapter, 'gp_joint_training', False)):
            # Ensure targets exist (compute lazily if needed)
            if not hasattr(self, '_gp_targets'):
                try:
                    self._gp_targets = self._compute_gp_template_targets_prob()
                except Exception:
                    self._gp_targets = None
            if getattr(self, '_gp_targets', None) is not None:
                try:
                    # GP operates at template inputs; build ELBO in-place
                    gp_weighter.train()
                    if hasattr(gp_weighter, 'likelihood'):
                        gp_weighter.likelihood.train()
                    x = gp_weighter._templates.to(dtype=torch.float32, device=gp_weighter._templates.device)
                    y = cast(torch.Tensor, self._gp_targets)
                    if y.device != x.device:
                        y = y.to(x.device)
                    if y.dtype != torch.float32:
                        y = y.to(torch.float32)

                    # VariationalELBO expects num_data as number of points per batch GP
                    mll = gpytorch.mlls.VariationalELBO(
                        gp_weighter.likelihood,
                        gp_weighter,
                        num_data=int(gp_weighter.num_templates),
                        beta=float(self.config.adapter.gp_beta),
                    )

                    out = gp_weighter(x)
                    elbo = mll(out, y)  # may return per-class tensor for batched GPs
                    # Reduce to scalar to be compatible with CE scalar loss
                    if elbo.dim() > 0:
                        elbo_scalar = elbo.mean()
                    else:
                        elbo_scalar = elbo
                    total_loss = total_loss + (-elbo_scalar)
                    try:
                        if hasattr(self, 'batch_idx') and (int(self.batch_idx) % max(1, int(getattr(self.config.train, 'print_freq', 50)))) == 0:
                            print(f"  [DBG][ELBO] value={float(elbo_scalar.detach().item()):.6f} beta={float(self.config.adapter.gp_beta):.4f}")
                    except Exception:
                        pass
                except Exception:
                    pass

        # Identity regularizer on visual projection W
        try:
            W = model.visual_proj.weight
            eye = torch.eye(W.shape[0], device=W.device, dtype=W.dtype)
            l2_reg = (W - eye).pow(2).sum()
            l2_lambda = self.config.adapter.l2_lambda
            # Scale by number of shots
            shots = self.config.dataset.num_shots
            l2_reg = l2_reg * l2_lambda / shots
            total_loss += l2_reg
        except Exception:
            print("Error in l2 regularization")
            l2_reg = None

        # Debug components
        try:
            self._dbg_loss_components = {
                "ce": float(ce_loss.detach().item()),
                "elbo": float(elbo_scalar.detach().item()),
                "l2_reg": float(l2_reg.detach().item()) if l2_reg is not None else 0.0,
                "total": float(total_loss.detach().item()),
            }
        except Exception:
            pass

        return total_loss

    def _backward_and_update(self, loss):
        """Backward pass and optimizer step."""
        self.optim.zero_grad()
        if self.scaler:
            self.scaler.scale(loss).backward()
            # Unscale to get true gradients for logging
            try:
                self.scaler.unscale_(self.optim)
            except Exception:
                pass
            # Capture gradient norms (debug)
            try:
                self._capture_grad_norms()
            except Exception:
                pass
            self.scaler.step(self.optim)
            self.scaler.update()
        else:
            loss.backward()
            # Capture gradient norms (debug)
            try:
                self._capture_grad_norms()
            except Exception:
                pass
            self.optim.step()

    def _capture_grad_norms(self):
        """Compute and store gradient norms for base vs GP params for diagnostics."""
        base_norm_sq = 0.0
        gp_norm_sq = 0.0
        try:
            if hasattr(self, "optim") and hasattr(self.optim, "param_groups"):
                # Group 0: base params
                base_group = self.optim.param_groups[0]
                for p in base_group.get('params', []):
                    if p is not None and getattr(p, 'grad', None) is not None:
                        base_norm_sq += float(p.grad.detach().pow(2).sum().item())
                # Group 1: gp params (if present)
                if len(self.optim.param_groups) > 1:
                    gp_group = self.optim.param_groups[1]
                    for p in gp_group.get('params', []):
                        if p is not None and getattr(p, 'grad', None) is not None:
                            gp_norm_sq += float(p.grad.detach().pow(2).sum().item())
        except Exception:
            pass
        self._dbg_grad_norms = {
            "base": math.sqrt(base_norm_sq) if base_norm_sq > 0 else 0.0,
            "gp": math.sqrt(gp_norm_sq) if gp_norm_sq > 0 else 0.0,
        }

    def parse_batch_train(self, batch):
        input_data = batch["img"]
        labels = batch["label"]
        input_data = input_data.to(self.device)
        labels = labels.to(self.device)
        return input_data, labels

    def train(self):
        """Training loop with feature extraction and evaluation."""
        # Build model first (this is normally done in BaseTrainer.train())
        self.build_model()
        
        self.set_model_mode("eval")

        # Feature extraction on test set
        self.labels_test, output_test, self.features_test = self.extract_features(partition="test")
        print("Zero-Shot accuracy on test: " + 
              str(round(compute_accuracy(output_test.cuda(), self.labels_test.cuda())[0], 2)))

        # Feature extraction on training set
        self.labels_train, logits_zs, self.features_train = self.extract_features(partition="train")

        # Optional GP prefit on per-template targets
        try:
            if getattr(self.config.adapter, 'use_gp', False) and getattr(self.model, 'gp_weighter', None) is not None:
                # Compute and cache targets once for both prefit and joint ELBO
                self._gp_targets = self._compute_gp_template_targets_prob()
                if getattr(self.config.adapter, 'gp_reg_prefit', True):
                    gp_epochs = int(getattr(self.config.adapter, 'gp_reg_epochs', 400))
                    gp_lr = float(getattr(self.config.adapter, 'gp_reg_lr', 1e-2))
                    self.model.gp_weighter.fit_targets(self._gp_targets, epochs=gp_epochs, lr=gp_lr)
                    # Freeze GP after prefit if joint training is disabled
                    if not bool(getattr(self.config.adapter, 'gp_joint_training', False)):
                        try:
                            for p in self.model.gp_weighter.parameters():
                                p.requires_grad = False
                            if hasattr(self, 'optim') and hasattr(self.optim, 'param_groups'):
                                gp_param_ids = {id(p) for p in self.model.gp_weighter.parameters()}
                                for group in self.optim.param_groups:
                                    if any(id(p) in gp_param_ids for p in group.get('params', [])):
                                        group['lr'] = 0.0
                        except Exception:
                            pass
        except Exception as e:
            print(f"[WARN] GP prefit skipped due to error: {e}")

        # Run the actual training using the base class training loop
        self.before_train()
        for self.epoch in range(self.start_epoch, self.max_epoch):
            self.before_epoch()
            self.run_epoch()
            self.after_epoch()
        self.after_train()

    def run_epoch(self):
        """Run one training epoch."""
        losses = AverageMeter()
        batch_time = AverageMeter()
        data_time = AverageMeter()

        # Set model to train mode but keep encoders frozen
        self.set_model_mode("train")
        try:
            if hasattr(self.model, "image_encoder"):
                self.model.image_encoder.eval() # type: ignore
        except AttributeError:
            pass
        try:
            if hasattr(self.model, "text_encoder"):
                self.model.text_encoder.eval() # type: ignore
        except AttributeError:
            pass

        # Set number of batches to sample
        self.num_batches = len(self.train_loader_x)
        self.batch_size = self.train_loader_x.batch_size or 1

        # Set features (ensure attributes exist)
        if not hasattr(self, 'features_train') or not hasattr(self, 'labels_train'):
            raise RuntimeError("features_train and labels_train must be extracted before training")
        
        features = self.features_train.clone().cpu().numpy()
        labels = self.labels_train.clone()

        # Randomly shuffle
        idx = np.random.rand(features.shape[0]).argsort(axis=0)
        features = features[idx, :]
        labels = labels[idx]

        loss_summary = {"loss": 0.0, "acc_train": 0.0, "acc_test": 0.0}

        end = time.time()
        for self.batch_idx in range(self.num_batches):
            batch_init = self.batch_idx * self.batch_size
            batch_end = (self.batch_idx + 1) * self.batch_size

            data_time.update(time.time() - end)
            batch_data = (features[batch_init:batch_end], labels[batch_init:batch_end])
            loss_summary = self.forward_backward(batch_data)
            batch_time.update(time.time() - end)
            losses.update(loss_summary['loss'])

            meet_freq = (self.batch_idx + 1) % self.config.train.print_freq == 0
            only_few_batches = self.num_batches < self.config.train.print_freq
            if meet_freq or only_few_batches:
                nb_remain = 0
                nb_remain += self.num_batches - self.batch_idx - 1
                nb_remain += (self.max_epoch - self.epoch - 1) * self.num_batches
                eta_seconds = batch_time.avg * nb_remain
                eta = str(datetime.timedelta(seconds=int(eta_seconds)))

                if (self.epoch + 1) % 10 == 0 or self.epoch == 0:
                    info = []
                    info += [f"epoch [{self.epoch + 1}/{self.max_epoch}]"]
                    info += [f"batch [{self.batch_idx + 1}/{self.num_batches}]"]
                    info += [f"loss {loss_summary['loss']:.4f}"]
                    info += [f"acc_train {loss_summary['acc_train']:.4f}"]
                    info += [f"acc_test {loss_summary['acc_test']:.4f}"]
                    info += [f"eta {eta}"]
                    print(" ".join(info))

                    # ---- GP/adapter diagnostics (printed at same cadence) ----
                    try:
                        model = cast(CustomCLIP, self.model)
                        # Loss breakdown
                        if hasattr(self, "_dbg_loss_components"):
                            comp = getattr(self, "_dbg_loss_components")
                            # Avoid overly long lines; show key parts
                            print(
                                f"  [DBG] loss: CE={comp.get('ce', 0):.4f} KL(raw)={comp.get('kl_raw', 0):.4f} "
                                f"beta={comp.get('kl_beta', 0):.3f} KL*beta={comp.get('kl', 0):.4f} "
                                f"l2_reg={comp.get('l2_reg', comp.get('l2', 0)):.4f} Total={comp.get('total', 0):.4f}"
                            )
                        # Prototype stats
                        if hasattr(self, "_dbg_proto_stats") and self._dbg_proto_stats is not None:
                            ps = self._dbg_proto_stats
                            print(
                                f"  [DBG] proto_norms: mean={ps['mean']:.4f} std={ps['std']:.4f} "
                                f"min={ps['min']:.4f} max={ps['max']:.4f}"
                            )
                        # Optimizer LRs (base vs GP when applicable) and grad norms
                        try:
                            if hasattr(self, "optim") and hasattr(self.optim, "param_groups"):
                                if len(self.optim.param_groups) == 2:
                                    lr_base = float(self.optim.param_groups[0]['lr'])
                                    lr_gp = float(self.optim.param_groups[1]['lr'])
                                    print(f"  [DBG] lr_base={lr_base:.6f} lr_gp={lr_gp:.6f}")
                                elif len(self.optim.param_groups) == 1:
                                    lr0 = float(self.optim.param_groups[0]['lr'])
                                    print(f"  [DBG] lr={lr0:.6f}")
                            if hasattr(self, "_dbg_grad_norms"):
                                gn = self._dbg_grad_norms
                                print(f"  [DBG] grad_norms: base={gn.get('base', 0.0):.6f} gp={gn.get('gp', 0.0):.6f}")
                        except Exception:
                            pass
                        # GP-specific stats
                        if getattr(self.config.adapter, "use_gp", False):
                            gp = getattr(model, "gp_weighter", None)
                            if gp is not None:
                                # Kernel hyperparameters (robust across kernel types)
                                try:
                                    covar_module = gp.covar_module if hasattr(gp, "covar_module") else None
                                    if covar_module is not None:
                                        outscale = covar_module.outputscale if hasattr(covar_module, "outputscale") else None
                                        outscale_val = float(outscale.detach().mean().item()) if outscale is not None else float('nan')
                                        base_k = covar_module.base_kernel if hasattr(covar_module, "base_kernel") else None
                                        if base_k is not None and hasattr(base_k, "lengthscale"):
                                            ls_val = float(base_k.lengthscale.detach().mean().item())
                                        else:
                                            ls_val = float('nan')
                                    else:
                                        outscale_val = float('nan')
                                        ls_val = float('nan')
                                except Exception:
                                    outscale_val = float('nan')
                                    ls_val = float('nan')
                                # Posterior variance over inducing/template points
                                # Posterior variance is a gpytorch attribute; skip if not present
                                var_mean = float('nan')
                                # Mean param magnitude
                                try:
                                    mean_module = gp.mean_module if hasattr(gp, "mean_module") else None
                                    if mean_module is not None and hasattr(mean_module, "mean_param"):
                                        mean_param = mean_module.mean_param.detach()
                                        mean_norm = float(mean_param.norm().item())
                                        mean_abs = float(mean_param.abs().mean().item())
                                    else:
                                        mean_norm = float('nan')
                                        mean_abs = float('nan')
                                except Exception:
                                    mean_norm = float('nan')
                                    mean_abs = float('nan')
                                print(
                                    f"  [DBG][GP] var_mean={var_mean:.6f} lengthscale={ls_val:.6f} "
                                    f"outputscale={outscale_val:.6f} mean_param_norm={mean_norm:.4f} mean_abs={mean_abs:.4f}"
                                )
                            else:
                                print("  [DBG][GP] GP disabled at runtime (no weighter present).")
                    except Exception:
                        # Diagnostics should never crash training
                        pass

            n_iter = self.epoch * self.num_batches + self.batch_idx
            self.write_scalar("train/loss", loss_summary['loss'], n_iter)
            self.write_scalar("train/lr", self.get_current_lr(), n_iter)

            end = time.time()

        return loss_summary

    def extract_features(self, partition="train", reps=1, transforms=None):
        """Extract features from specified data partition."""
        import copy
        print("Extracting features from: " + partition)
        self.set_model_mode("eval")

        if partition == "train":
            # Copy safe version of training dataloader
            data_loader = copy.deepcopy(self.train_loader_x)
            # Set data loader with drop last to false for not losing samples
            data_loader = torch.utils.data.DataLoader(
                copy.deepcopy(self.train_loader_x.dataset), 
                batch_size=self.train_loader_x.batch_size,
                sampler=self.train_loader_x.sampler, 
                num_workers=self.train_loader_x.num_workers,
                drop_last=False, 
                pin_memory=False
            )
        elif partition == "val":
            data_loader = copy.deepcopy(self.val_loader)
        elif partition == "test":
            data_loader = copy.deepcopy(self.test_loader)
        else:
            raise ValueError(f"Unknown partition: {partition}")

        labels_ds, logits_ds, features_ds = [], [], []
        for rep in range(reps):
            if data_loader is not None:
                for batch_idx, batch in enumerate(data_loader):
                    input_data, labels = self.parse_batch_train(batch)
                    with torch.no_grad():
                        logits, features = self.model(input_data, return_features=True)  # type: ignore
                    labels_ds.append(labels.cpu())
                    logits_ds.append(logits.cpu())
                    features_ds.append(features.cpu())

        # Concatenate outputs
        labels_ds = torch.cat(labels_ds, dim=0)
        logits_ds = torch.cat(logits_ds, dim=0)
        features_ds = torch.cat(features_ds, dim=0)

        return labels_ds, logits_ds, features_ds

    @torch.no_grad()
    def _compute_gp_template_targets_prob(self) -> torch.Tensor:
        """Compute per-template targets y[k, m] as mean correct-class probability.

        Computes targets from train features and text embeddings in fp32 on CPU.
        Returns a tensor of shape [K, M].
        """
        features = self.features_train.detach().cpu().to(torch.float32)  # [N, D]
        labels = self.labels_train.detach().cpu().to(torch.int64)        # [N]
        text_emb = cast(torch.Tensor, self.model.text_embeddings_all).detach().cpu().to(torch.float32)  # [K, M, D]

        K, M, D = int(text_emb.shape[0]), int(text_emb.shape[1]), int(text_emb.shape[2])
        N = int(features.shape[0])

        # Apply current projection W if present
        try:
            W = cast(torch.nn.Linear, self.model.visual_proj).weight.detach().cpu().to(torch.float32)  # [D, D]
            feats_proj = features @ W.t()
        except Exception:
            feats_proj = features

        feats_norm = F.normalize(feats_proj, p=2, dim=-1)  # [N, D]
        scale = float(cast(torch.Tensor, self.model.logit_scale).exp().detach().cpu().item())

        # Prepare per-class aggregation
        labels_one_hot = torch.zeros(N, K, dtype=torch.float32)
        labels_one_hot[torch.arange(N), labels] = 1.0
        class_counts = labels_one_hot.sum(dim=0).clamp_min(1.0)  # [K]

        targets = torch.zeros(K, M, dtype=torch.float32)
        for m in range(M):
            prot_m = text_emb[:, m, :]                 # [K, D]
            prot_m = F.normalize(prot_m, p=2, dim=-1)  # [K, D]
            logits = scale * (feats_norm @ prot_m.t()) # [N, K]
            probs = torch.softmax(logits, dim=-1)      # [N, K]
            sum_probs = (labels_one_hot * probs).sum(dim=0)  # [K]
            targets[:, m] = sum_probs / class_counts

        return targets
