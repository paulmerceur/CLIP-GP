from typing import cast
import time
import datetime
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.amp.grad_scaler import GradScaler    
import numpy as np
import math

from utils.trainer import BaseTrainer
from utils.metrics import compute_accuracy, AverageMeter
from utils.checkpoint import load_pretrained_weights
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
    """Simple linear adapter with learnable scaling."""

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
        
        self.gp_num_mc_samples = int(getattr(config.adapter, 'gp_num_mc_samples', 1) or 1)
        
        # Create TextEncoder for proper text encoding
        self.text_encoder = TextEncoder(clip_model)
        
        # Get text features and setup GP if needed
        base_text_features, self.text_embeddings_all = _get_base_text_features(config, classnames, clip_model, self.text_encoder)
        
        # Create adapter
        self.adapter = LinearAdapter(base_text_features=base_text_features)

        # Create GP weighter if needed
        self.gp_weighter = None
        if getattr(config.adapter, 'use_gp', False):
            self.gp_weighter = GaussianProcessTemplateWeighter(
                text_embeddings=self.text_embeddings_all,
                cfg=config,
            )
    
    def encode_image(self, image: torch.Tensor) -> torch.Tensor:
        """Extract visual features from images without grad."""
        with torch.no_grad():
            features = self.image_encoder(image.type(self.dtype))
        return features
    
    def forward_prototypes(self, num_samples: int = 1):
        """Get class prototypes; if GP is enabled, average over num_samples."""
        target_device = next(self.parameters()).device
        if self.gp_weighter is not None:
            if num_samples <= 1: num_samples = 1
            proto_s = self.gp_weighter.sample_prototypes(num_samples)  # [S,K,D]
            current_prototypes = proto_s.squeeze(0) if num_samples == 1 else proto_s.mean(dim=0)

            if current_prototypes.device != target_device:
                current_prototypes = current_prototypes.to(target_device)
                
            # Mirror CLAP behavior: keep adapter prototypes in sync with the prototypes used for logits
            # so that the zero-shot constraint applies to the same vectors.
            try:
                if self.training and hasattr(self, 'adapter') and hasattr(self.adapter, 'prototypes'):
                    # Match norms to base_text_features so L2 constraint compares like-for-like
                    with torch.no_grad():
                        base = cast(torch.Tensor, self.adapter.base_text_features)
                        target_norm = base.norm(dim=-1, keepdim=True).clamp_min(1e-6)
                        cur_norm = current_prototypes.norm(dim=-1, keepdim=True).clamp_min(1e-6)
                        current_prototypes_rescaled = current_prototypes * (target_norm / cur_norm)
                    self.adapter.prototypes.data = current_prototypes_rescaled.detach().to(
                        dtype=self.adapter.prototypes.dtype, device=self.adapter.prototypes.device
                    )
            except Exception:
                pass
            return current_prototypes
        # Baseline when GP is disabled: use learnable prototypes (adapter)
        prototypes = self.adapter.forward()
        if prototypes.device != target_device:
            prototypes = prototypes.to(target_device)
        return prototypes
    
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
        features_norm = F.normalize(features, p=2, dim=-1)
        prototypes_norm = F.normalize(prototypes, p=2, dim=-1)
        features_norm = features_norm.to(prototypes_norm.dtype)
        return self.logit_scale.exp() * features_norm @ prototypes_norm.t()

    def forward(self, image: torch.Tensor, return_features: bool = False):
        """CLAP-like forward.

        - Extract image features (encode_image)
        - Compute logits via forward_features (normalization + prototypes)
        - Optionally return both logits and features
        """
        features = self.encode_image(image)
        logits = self.forward_features(features)
        if return_features:
            return logits, features
        return logits
    
    def sample_forward(self, image, num_samples=50):
        """Deprecated: forward() already supports averaging via gp_num_mc_samples."""
        return self.forward(image)



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

        if config.adapter.prec == "fp32" or config.adapter.prec == "amp":
            clip_model.float()

        print("Building custom CLIP")
        self.model = CustomCLIP(config, classnames, clip_model)

        # Setup parameter groups
        for name, param in self.model.named_parameters():
            if "adapter" in name or "gp_weighter" in name:
                param.requires_grad = True
            else:
                param.requires_grad = False

        if config.model.init_weights:
            load_pretrained_weights(self.model.adapter, config.model.init_weights)

        self.model.to(self.device)
        self.model.float()
        
            
        # Setup optimizer with different learning rates for GP
        if config.adapter.use_gp and self.model.gp_weighter is not None:
            # Two parameter groups: base params and GP params
            base_params = []
            base_params.extend([p for p in self.model.adapter.parameters() if p.requires_grad])
            
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
            baseline_params.extend(list(self.model.adapter.parameters()))
            
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
        
        # Get prototypes
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
            # Keep training resilient if any debug metric fails
            self._dbg_proto_stats = None
        # Ensure same dtype and device
        if projected_features.dtype != prototypes.dtype:
            prototypes = prototypes.to(dtype=projected_features.dtype)
        if projected_features.device != prototypes.device:
            prototypes = prototypes.to(device=projected_features.device)
        # Compute logits via the model's centralized path (keeps logic in model)
        logits = model.forward_features(projected_features)
        # Compute loss
        loss = self.compute_loss(logits, labels)
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

    def compute_loss(self, logits, labels):
        from typing import cast
        model = cast(CustomCLIP, self.model)
        """Compute loss including GP KL term and visual projection regularization."""
        # Cross-entropy loss
        ce_loss = F.cross_entropy(logits, labels)
        total_loss = ce_loss

        # Add GP KL divergence if using GP
        if self.config.adapter.use_gp and model.gp_weighter is not None:
            kl_loss = model.gp_weighter.kl_term
            beta = self.config.adapter.gp_beta
            kl_contribution = beta * kl_loss
            total_loss += kl_contribution
        else:
            kl_loss = None
            kl_contribution = None
            beta = getattr(self.config.adapter, "gp_beta", 0.0)
        
        # Adapter regularization: keep learnable prototypes close to zero-shot base
        l2_contribution = model.adapter.constraint()
        total_loss += l2_contribution
        
        # Store debug components for periodic printing
        try:
            batch_size = float(labels.size(0)) if hasattr(labels, 'size') else 1.0
            self._dbg_loss_components = {
                "ce": float(ce_loss.detach().item()),
                "kl_raw": float(kl_loss.detach().item()) if kl_loss is not None else 0.0,
                "kl_beta": float(beta),
                "kl": float(kl_contribution.detach().item()) if kl_contribution is not None else 0.0,
                "kl_per_img": float((kl_contribution / batch_size).detach().item()) if kl_contribution is not None and batch_size > 0 else 0.0,
                "l2": float(l2_contribution.detach().item()) if l2_contribution is not None else 0.0,
                "total": float(total_loss.detach().item()),
            }
        except Exception:
            pass

        return total_loss

    def model_inference(self, input_data):
        """Model inference with GP sampling during evaluation."""
        from typing import cast
        model = cast(CustomCLIP, self.model)
        if not model.training and model.gp_weighter is not None:
            # Use GP sampling for evaluation
            num_samples = self.config.adapter.gp_num_mc_samples
            return model.sample_forward(input_data, num_samples)
        else:
            return model(input_data)

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
        # Initialize per-class alpha_constraint once from zero-shot train logits
        model_ref = cast(CustomCLIP, self.model)
        if hasattr(model_ref, 'adapter') and hasattr(model_ref.adapter, 'init_alpha_from_zero_shot'):
            model_ref.adapter.init_alpha_from_zero_shot(self.labels_train, logits_zs)

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
                                f"L2={comp.get('l2', 0):.4f} Total={comp.get('total', 0):.4f}"
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
