from typing import Any, List, Tuple, Optional, cast
import math
import time
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.trainer import BaseTrainer
from utils.metrics import compute_accuracy, AverageMeter
from utils.optimization import build_optimizer_from_param_groups, build_lr_scheduler
from utils.trainer_registry import TRAINER_REGISTRY

from clip import clip

# Reuse helper utilities from adapters trainer
from .adapters import TextEncoder, load_clip_to_cpu, _get_base_text_features
from .gp_template_weigher import GaussianProcessTemplateWeighter


class FinetuneCLIP(nn.Module):
    """CLIP fine-tuning wrapper supporting linear or prototype heads."""

    def __init__(self, config, classnames: List[str], clip_model) -> None:
        super().__init__()
        self.config = config
        self.classnames = classnames

        # Core CLIP modules
        self.image_encoder = clip_model.visual
        self.text_encoder = TextEncoder(clip_model)
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype

        # Dimension and classes
        # Both ModifiedResNet and VisionTransformer expose output_dim
        dim = int(self.image_encoder.output_dim)  # type: ignore[attr-defined]
        self.num_classes = len(classnames)

        # Prepare text features for zero-shot init and prototype head
        base_text_features, text_embeddings_all = _get_base_text_features(
            config, classnames, clip_model, self.text_encoder
        )
        self.register_buffer("base_text_features", base_text_features)
        self.register_buffer("text_embeddings_all", text_embeddings_all)

        # Head selection
        head_type = str(getattr(config.finetune, 'ft_head_type', 'linear')).lower()
        if head_type not in ["linear", "prototypes"]:
            raise ValueError(f"Unsupported ft_head_type: {head_type}")
        self.head_type = head_type

        if self.head_type == "linear":
            # Linear classifier initialized from zero-shot prototypes
            self.classifier = nn.Linear(dim, self.num_classes, bias=False)
            with torch.no_grad():
                # Normalize base prototypes before copying
                init_w = F.normalize(base_text_features, p=2, dim=-1)
                self.classifier.weight.copy_(init_w)
            # Keep a frozen copy for L2-to-ZS regularization if enabled
            self.register_buffer("_zs_init_weight", init_w.clone())
        else:
            self.classifier = None  # type: ignore[assignment]

        # Optional GP for template weighting in prototype head (B1 style)
        self.gp_weighter: Optional[GaussianProcessTemplateWeighter] = None
        if self.head_type == "prototypes" and bool(getattr(config.finetune, 'use_gp', False)):
            # Instantiate GP in fp32; works over the stored template embeddings
            self.gp_weighter = GaussianProcessTemplateWeighter(
                text_embeddings=self.text_embeddings_all,
                cfg=config,
            )

    def _encode_images(self, images: torch.Tensor) -> torch.Tensor:
        feats = self.image_encoder(images.type(self.dtype))
        # Use a safe eps to avoid fp16 underflow in normalization
        feats = F.normalize(feats, p=2, dim=-1, eps=1e-6)
        return feats

    def _current_class_weights(self) -> torch.Tensor:
        """Return class weight vectors of shape [K, D], normalized."""
        if self.head_type == "linear":
            W = cast(nn.Linear, self.classifier).weight  # [K, D]
            return F.normalize(W, p=2, dim=-1)

        # Prototype head: compute from text templates
        # If GP is enabled (B1), use GP weights over current templates with detached weights
        use_gp = self.gp_weighter is not None and bool(getattr(self.config.finetune, 'use_gp', False))
        text_trainable = bool(getattr(self.config.finetune, 'ft_train_text', False))
        if use_gp:
            # Prepare current template embeddings (recompute if text trainable)
            if text_trainable:
                device = next(self.parameters()).device
                templates = ["a photo of a {}."]
                from datasets.imagenet_templates import IMAGENET_TEMPLATES_SELECT, IMAGENET_TEMPLATES
                if getattr(self.config.finetune, 'num_templates', 1) > 1:
                    num_needed = min(
                        int(self.config.finetune.num_templates) - 1,
                        len(IMAGENET_TEMPLATES_SELECT),
                    )
                    templates += IMAGENET_TEMPLATES_SELECT[:num_needed]
                if getattr(self.config.finetune, 'num_templates', 1) > 1 + len(IMAGENET_TEMPLATES_SELECT):
                    extra = int(self.config.finetune.num_templates) - 1 - len(IMAGENET_TEMPLATES_SELECT)
                    templates += IMAGENET_TEMPLATES[:extra]

                emb_list = []
                with torch.no_grad():
                    for name in self.classnames:
                        tok = clip.tokenize([t.format(name) for t in templates]).to(device)
                        e = clip.token_embedding(tok).type(self.dtype)
                        emb = self.text_encoder(e, tok)
                        emb_list.append(emb)
                text_embeds = torch.stack(emb_list)  # [K,M,D]
            else:
                text_embeds = cast(torch.Tensor, self.text_embeddings_all)

            # Compute GP mean logits over templates and detach weights for prototype mix
            gp = cast(GaussianProcessTemplateWeighter, self.gp_weighter)
            gp.eval()
            try:
                if hasattr(gp, 'likelihood'):
                    gp.likelihood.eval()
            except Exception:
                pass
            with torch.no_grad():
                f_mean = gp(text_embeds.to(torch.float32)).mean  # [K,M]
                w = torch.softmax(f_mean, dim=-1).detach()  # [K,M]
            # Weighted prototypes (grad flows to text_embeds if trainable, weights detached)
            # Compute in fp32 then cast back to CLIP dtype
            protos = torch.einsum("km,kmd->kd", w, text_embeds.float())
            return F.normalize(protos.to(self.dtype), p=2, dim=-1, eps=1e-6)

        # Fallback: mean over templates (or cached base when text is frozen)
        if text_trainable:
            device = next(self.parameters()).device
            templates = ["a photo of a {}."]
            from datasets.imagenet_templates import IMAGENET_TEMPLATES_SELECT, IMAGENET_TEMPLATES
            if getattr(self.config.finetune, 'num_templates', 1) > 1:
                num_needed = min(
                    int(self.config.finetune.num_templates) - 1,
                    len(IMAGENET_TEMPLATES_SELECT),
                )
                templates += IMAGENET_TEMPLATES_SELECT[:num_needed]
            if getattr(self.config.finetune, 'num_templates', 1) > 1 + len(IMAGENET_TEMPLATES_SELECT):
                extra = int(self.config.finetune.num_templates) - 1 - len(IMAGENET_TEMPLATES_SELECT)
                templates += IMAGENET_TEMPLATES[:extra]

            emb_list = []
            with torch.no_grad():
                for name in self.classnames:
                    tok = clip.tokenize([t.format(name) for t in templates]).to(device)
                    e = clip.token_embedding(tok).type(self.dtype)
                    emb = self.text_encoder(e, tok)
                    emb_list.append(emb)
            text_embeds = torch.stack(emb_list)  # [K,M,D]
            protos = text_embeds.mean(1)
            return F.normalize(protos, p=2, dim=-1, eps=1e-6)
        else:
            protos = cast(torch.Tensor, self.base_text_features)
            return F.normalize(protos, p=2, dim=-1, eps=1e-6)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        feats = self._encode_images(images)
        class_w = self._current_class_weights()  # [K,D]
        # Align dtypes/devices
        if feats.dtype != class_w.dtype:
            class_w = class_w.to(dtype=feats.dtype)
        if feats.device != class_w.device:
            class_w = class_w.to(device=feats.device)
        scale = self.logit_scale.exp().to(dtype=feats.dtype)
        logits = scale * (feats @ class_w.t())
        return logits


@TRAINER_REGISTRY.register("FineTune")
class FineTune(BaseTrainer):
    """Full fine-tuning trainer for CLIP with linear/prototype heads."""

    def __init__(self, config, dataset_manager):
        super().__init__(config, dataset_manager)

    def check_cfg(self, config) -> None:
        assert config.adapter.prec in ["fp16", "fp32", "amp"]

    def build_model(self):
        config = self.config
        classnames = self.dm.dataset.classnames

        print(f"Loading CLIP (backbone: {config.model.backbone_name})")
        clip_model = load_clip_to_cpu(config)
        clip_model = clip_model.to(self.device)

        # Precision control
        if config.adapter.prec in ["fp32", "amp"]:
            clip_model.float()

        print("Building FinetuneCLIP")
        self.model = FinetuneCLIP(config, classnames, clip_model)
        self.model.to(self.device)

        # Set trainable parameters per config
        ft_visual = bool(getattr(config.finetune, 'ft_train_visual', True))
        ft_text = bool(getattr(config.finetune, 'ft_train_text', False))
        ft_logit = bool(getattr(config.finetune, 'ft_train_logit_scale', True))

        # Default: freeze all
        for p in self.model.parameters():
            p.requires_grad = False

        # Visual encoder
        for p in self.model.image_encoder.parameters():
            p.requires_grad = ft_visual

        # Text encoder
        for p in self.model.text_encoder.parameters():
            p.requires_grad = ft_text

        # Linear head (if applicable)
        if getattr(self.model, 'classifier', None) is not None:
            for p in self.model.classifier.parameters():
                p.requires_grad = True

        # Logit scale
        try:
            self.model.logit_scale.requires_grad_(ft_logit)
        except Exception:
            pass

        # BatchNorm freezing (for ResNet visual)
        if bool(getattr(config.finetune, 'freeze_bn', False)):
            try:
                for m in self.model.modules():
                    if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):  # type: ignore[attr-defined]
                        m.eval()
                        for p in m.parameters():
                            p.requires_grad = False
            except Exception:
                pass

        # Build optimizer with param groups
        base_lr = float(getattr(config.optim, 'backbone_lr', None) or config.optim.lr)
        head_lr = float(getattr(config.optim, 'head_lr', None) or config.optim.lr)
        logit_lr = 0.1 * min(base_lr, head_lr)

        backbone_params = []
        head_params = []
        logit_params = []
        gp_params = []

        # Collect parameters by role
        for name, p in self.model.named_parameters():
            if not p.requires_grad:
                continue
            if name.startswith('image_encoder') or name.startswith('text_encoder'):
                backbone_params.append(p)
            elif name.startswith('classifier'):
                head_params.append(p)
            elif 'logit_scale' in name:
                logit_params.append(p)
            elif name.startswith('gp_weighter'):
                gp_params.append(p)
            else:
                # Default to backbone group
                backbone_params.append(p)

        param_groups = []
        if len(backbone_params) > 0:
            param_groups.append({
                'params': backbone_params,
                'lr': base_lr,
                'weight_decay': float(config.optim.weight_decay)
            })
        if len(head_params) > 0:
            param_groups.append({
                'params': head_params,
                'lr': head_lr,
                'weight_decay': float(config.optim.weight_decay)
            })
        if len(logit_params) > 0:
            param_groups.append({
                'params': logit_params,
                'lr': logit_lr,
                'weight_decay': 0.0
            })
        if len(gp_params) > 0:
            param_groups.append({
                'params': gp_params,
                'lr': float(getattr(config.finetune, 'gp_lr', 0.01)),
                'weight_decay': 0.0
            })

        self.optim = build_optimizer_from_param_groups(param_groups, config.optim)
        self.sched = build_lr_scheduler(self.optim, config.optim)

        # AMP scaler
        from torch.amp.grad_scaler import GradScaler
        self.scaler = GradScaler() if config.adapter.prec == "amp" else None

        # Store zero-shot init for regularization if needed
        self._ft_l2_to_init = float(getattr(config.finetune, 'ft_l2_to_init', 0.0))
        # ELBO control for GP (B1)
        self._use_gp = bool(getattr(config.finetune, 'use_gp', False)) and getattr(self.model, 'gp_weighter', None) is not None
        self._gp_use_elbo = bool(getattr(config.finetune, 'gp_use_elbo', True))
        self._gp_beta = float(getattr(config.finetune, 'gp_beta', 0.001))
        self._gp_targets: Optional[torch.Tensor] = None
        self._gp_targets_epoch = -1

    def parse_batch_train(self, batch):
        input_data = batch["img"]
        labels = batch["label"]
        input_data = input_data.to(self.device)
        labels = labels.to(self.device)
        return input_data, labels

    def forward_backward(self, batch):
        images, labels = batch
        model = cast(FinetuneCLIP, self.model)

        # Forward
        logits = model(images)

        # CE with optional label smoothing
        ls = float(getattr(self.config.finetune, 'label_smoothing', 0.0) or 0.0)
        try:
            loss = F.cross_entropy(logits, labels, label_smoothing=ls)
        except TypeError:
            # Older PyTorch without label_smoothing
            loss = F.cross_entropy(logits, labels)

        # Optional L2-to-ZS regularizer for linear head
        reg = None
        if self._ft_l2_to_init > 0.0 and getattr(model, 'classifier', None) is not None:
            W = cast(nn.Linear, model.classifier).weight
            init_w = cast(torch.Tensor, model._zs_init_weight)
            if W.dtype != init_w.dtype:
                init_w = init_w.to(dtype=W.dtype)
            if W.device != init_w.device:
                init_w = init_w.to(device=W.device)
            reg = (W - init_w).pow(2).sum() * self._ft_l2_to_init
            loss = loss + reg

        # Optional GP ELBO (detached targets) — B1 style
        if self._use_gp and self._gp_use_elbo:
            try:
                gp = cast(GaussianProcessTemplateWeighter, model.gp_weighter)
                gp.train()
                if hasattr(gp, 'likelihood'):
                    gp.likelihood.train()
                # Refresh targets per policy
                epoch_now = int(self.epoch)
                refresh_every = int(getattr(self.config.finetune, 'gp_targets_refresh_every', 0) or 0)
                need_refresh = (self._gp_targets is None) or (refresh_every > 0 and epoch_now % max(1, refresh_every) == 0 and epoch_now != self._gp_targets_epoch)
                if need_refresh:
                    # Compute per-template targets from current train set using posterior-mean baseline
                    with torch.no_grad():
                        # Mimic adapters._compute_gp_template_targets_prob but inside FT trainer context is heavy; using existing method requires features cache
                        # Here we compute fast targets from the current batch as a proxy (research trade-off)
                        # NOTE: For more faithful targets, implement a cached feature pass like adapters trainer if needed.
                        device = images.device
                        # Build current template embeddings (fp32 CPU for GP stability)
                        text_emb = cast(torch.Tensor, model.text_embeddings_all).detach().cpu().to(torch.float32)
                        K, M, D = int(text_emb.shape[0]), int(text_emb.shape[1]), int(text_emb.shape[2])
                        # Quick batch-based targets: probs for true class per template
                        # Compute logits against each template set, average per-class
                        # First, get normalized image features in model dtype
                        feats = model._encode_images(images).detach().cpu().to(torch.float32)  # [B,D]
                        scale = float(model.logit_scale.exp().detach().cpu().item())
                        y = labels.detach().cpu().to(torch.int64)
                        # One-hot per-class counts in batch
                        counts = torch.bincount(y, minlength=K).clamp_min(1)
                        targets = torch.zeros(K, M, dtype=torch.float32)
                        for m in range(M):
                            prot_m = F.normalize(text_emb[:, m, :], p=2, dim=-1)  # [K,D]
                            logits_m = scale * (feats @ prot_m.t())  # [B,K]
                            probs_m = torch.softmax(logits_m, dim=-1)  # [B,K]
                            # Sum of probabilities of true labels per class
                            sums = torch.zeros(K, dtype=torch.float32)
                            sums.index_add_(0, y, probs_m[torch.arange(probs_m.size(0)), y])
                            targets[:, m] = sums / counts.to(torch.float32)
                        self._gp_targets = targets.to(device=gp._templates.device, dtype=torch.float32)
                        self._gp_targets_epoch = epoch_now

                if self._gp_targets is not None:
                    import gpytorch
                    mll = gpytorch.mlls.VariationalELBO(
                        gp.likelihood,
                        gp,
                        num_data=int(gp.num_templates),
                        beta=self._gp_beta,
                    )
                    x = gp._templates.to(dtype=torch.float32)
                    out = gp(x)
                    elbo = mll(out, self._gp_targets)
                    # Subtract ELBO (maximize) -> add -elbo to loss
                    loss = loss + (-elbo.mean())
            except Exception:
                pass

        # Backward
        self.optim.zero_grad(set_to_none=True)
        if getattr(self, 'scaler', None) is not None:
            self.scaler.scale(loss).backward()
            # Optional grad clip
            try:
                gc = float(getattr(self.config.finetune, 'grad_clip', 0.0) or 0.0)
                if gc > 0:
                    self.scaler.unscale_(self.optim)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), gc)
            except Exception:
                pass
            self.scaler.step(self.optim)
            self.scaler.update()
        else:
            loss.backward()
            # Optional grad clip
            try:
                gc = float(getattr(self.config.finetune, 'grad_clip', 0.0) or 0.0)
                if gc > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), gc)
            except Exception:
                pass
            self.optim.step()

        with torch.no_grad():
            acc = compute_accuracy(logits, labels)[0]

        return {
            "loss": float(loss.detach().item()),
            "acc": float(acc)
        }

    def run_epoch(self):
        losses = AverageMeter()
        batch_time = AverageMeter()
        data_time = AverageMeter()

        self.set_model_mode("train")
        try:
            # Keep encoders in train as configured; BN freezing handled in build_model
            pass
        except Exception:
            pass

        self.num_batches = len(self.train_loader_x)
        self.batch_size = self.train_loader_x.batch_size or 1

        end = time.time()
        for self.batch_idx, batch in enumerate(self.train_loader_x):
            data_time.update(time.time() - end)
            images, labels = self.parse_batch_train(batch)

            out = self.forward_backward((images, labels))
            batch_time.update(time.time() - end)
            losses.update(out['loss'])

            if (self.batch_idx + 1) % self.config.train.print_freq == 0 or self.num_batches < self.config.train.print_freq:
                print(
                    f"epoch [{self.epoch + 1}/{self.max_epoch}] batch [{self.batch_idx + 1}/{self.num_batches}] "
                    f"loss {out['loss']:.4f} acc {out['acc']:.4f} lr {self.get_current_lr():.6f}"
                )

            n_iter = self.epoch * self.num_batches + self.batch_idx
            self.write_scalar("train/loss", out['loss'], n_iter)
            self.write_scalar("train/lr", self.get_current_lr(), n_iter)

            end = time.time()

        return {"loss": losses.avg}


