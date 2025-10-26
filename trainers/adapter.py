from typing import cast
import time
import datetime
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
import gpytorch
import numpy as np
import math
import copy

from clip import clip
from utils.trainer import BaseTrainer, TextEncoder, load_clip, _get_templates
from utils.metrics import compute_accuracy, AverageMeter
from utils.optimization import build_optimizer, build_lr_scheduler, build_optimizer_from_param_groups
from utils.trainer_registry import TRAINER_REGISTRY
from utils.dataset_base import build_dataset, TorchDatasetWrapper
from utils.transforms import build_transform

from .gp_template_weigher import GaussianProcessTemplateWeighter

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.conv.fp32_precision = 'tf32'


@torch.no_grad()
def _get_text_embeddings(templates, classnames, clip_model, text_encoder=None):
    """Extract text features for all templates and classes."""
    device = next(clip_model.parameters()).device
    emb_list = []
    with torch.no_grad():
        for name in classnames:
            tok = clip.tokenize([t.format(name) for t in templates]).to(device)
            if text_encoder is not None:
                e = clip_model.token_embedding(tok)
                emb = text_encoder(e, tok)
            else:
                emb = clip_model.encode_text(tok)
            emb_list.append(emb)
    text_embeds = torch.stack(emb_list) # [K,M,D]

    return text_embeds


@torch.no_grad()
def _get_template_weights(config, text_embeddings: torch.Tensor, features: torch.Tensor | None, labels: torch.Tensor | None, logit_scale: torch.Tensor | float) -> torch.Tensor:
    """Compute per-class template weights of shape [K, M] (rows sum to 1).

    Methods (config.adapter.template_init_method):
      - "uniform": uniform 1/M weights per class
      - "val_weighted": per-class per-template accuracy used as logits
      - "top3": keep global top-3 templates by average accuracy, mask others
      - "minmax": per-class min–max rescale of accuracies in [0,1]

    The pipeline:
      1) Compute zero-shot logits per template on provided features
      2) Build per-class accuracy scores S[k,m]
      3) Optionally transform S per method
      4) Map to weights via row-wise softmax over log(S+eps)/temperature
    """
    method = str(getattr(config.adapter, 'template_init_method', 'uniform')).lower()
    temperature = 1.0
    E = text_embeddings
    K = int(E.shape[0])
    M = int(E.shape[1])
    if M == 0:
        return torch.empty(K, 0, device=E.device, dtype=E.dtype)
    if bool(getattr(config.adapter, 'prefit_on_full_set', False)):
        try:
            cfg_full = copy.deepcopy(config)
            cfg_full.dataset.num_shots = 0
            ds_full = build_dataset(cfg_full)
            tfm = build_transform(cfg_full, is_train=True)
            dl = DataLoader(
                TorchDatasetWrapper(ds_full.train_x, transform=tfm, is_train=False),
                batch_size=int(getattr(cfg_full.dataloader, 'batch_size_train', 128)),
                shuffle=False,
                num_workers=int(getattr(cfg_full.dataloader, 'num_workers', 8)),
                drop_last=False,
                pin_memory=False,
            )
            clip_model_tmp = load_clip(cfg_full, E.device)
            clip_model_tmp.eval()
            feats_list = []
            labels_list = []
            with torch.no_grad():
                for batch in dl:
                    imgs = batch["img"].to(E.device)
                    lbs = batch["label"]
                    f = clip_model_tmp.visual(imgs)
                    feats_list.append(f.cpu())
                    labels_list.append(lbs.cpu())
            features = torch.cat(feats_list, dim=0).to(device=E.device)
            labels = torch.cat(labels_list, dim=0).to(device=E.device)
            print(f"[INFO] Prefit on full set: {len(features)} samples used.")
        except Exception as e:
            print(f"[WARN] prefit_on_full_set failed ({e}); falling back to provided few-shot features.")
    if method == 'uniform' or features is None or labels is None:
        return torch.full((K, M), 1.0 / float(M), device=E.device, dtype=E.dtype)
    feats = features.to(device=E.device)
    feats = F.normalize(feats, p=2, dim=-1)
    labels_i64 = labels.to(device=feats.device, dtype=torch.int64)
    E_float = E.to(device=feats.device)
    scale = logit_scale if isinstance(logit_scale, torch.Tensor) else torch.tensor(float(logit_scale), device=feats.device)
    scale = scale.to(dtype=feats.dtype)
    counts_k = torch.bincount(labels_i64, minlength=K).to(feats.dtype).clamp_min(1)
    scores = torch.zeros(K, M, dtype=feats.dtype, device=feats.device)
    for m in range(M):
        prot_m = F.normalize(E_float[:, m, :], p=2, dim=-1)
        logits = scale * (feats @ prot_m.t())
        preds = logits.argmax(dim=1)
        corr = (preds == labels_i64).to(feats.dtype)
        sums_k = torch.zeros(K, dtype=feats.dtype, device=feats.device)
        sums_k.index_add_(0, labels_i64, corr)
        scores[:, m] = sums_k / counts_k
    if method == 'top3':
        avg_over_classes = scores.mean(dim=0)
        top_k = min(3, M)
        _, top_idx = torch.topk(avg_over_classes, k=top_k, largest=True)
        keep = torch.zeros(M, dtype=scores.dtype, device=scores.device)
        keep[top_idx] = 1.0
        scores = scores * keep.view(1, -1)
        row_sum = scores.sum(dim=1, keepdim=True)
        zero_rows = row_sum.squeeze(1) <= 1e-12
        if bool(zero_rows.any()):
            num_sel = float(top_k)
            uniform_sel = (keep / num_sel).view(1, -1).expand(int(zero_rows.sum().item()), -1)
            scores[zero_rows] = uniform_sel
    elif method == 'minmax':
        s = scores.clone()
        s_min = s.min(dim=1, keepdim=True).values
        s_max = s.max(dim=1, keepdim=True).values
        rng = (s_max - s_min)
        flat = rng.le(1e-12)
        s = torch.where(flat, torch.full_like(s, 1.0 / float(M)), (s - s_min) / rng.clamp_min(1e-12))
        scores = s
    logits_w = torch.log(scores.clamp_min(1e-12)) / max(temperature, 1e-6)
    weights = torch.softmax(logits_w, dim=1)
    return weights.to(device=E.device)


class CustomCLIP(nn.Module):
    """Custom CLIP model with adapter and optional GP weighting."""
    
    def __init__(self, config, classnames, clip_model):
        super().__init__()
        self.config = config
        self.image_encoder = clip_model.visual
        self.logit_scale = clip_model.logit_scale
        self.text_encoder = TextEncoder(clip_model)
        self.templates = _get_templates(config)
        self.text_embeddings = _get_text_embeddings(self.templates, classnames, clip_model, self.text_encoder)

        # Optionally make template weights trainable for baseline
        use_gp = bool(getattr(config.adapter, 'use_gp', False))
        train_tw = bool(getattr(config.adapter, 'train_template_weights', False))
        use_linear_tw = bool(getattr(config.adapter, 'use_linear_template_weighting', False))

        if (not use_gp) and train_tw and not use_linear_tw:
            # Original approach: direct template weights matrix
            K = int(self.text_embeddings.shape[0])
            M = int(self.text_embeddings.shape[1])
            if M > 0:
                init = torch.full((K, M), 1.0 / float(M), dtype=self.text_embeddings.dtype, device=self.text_embeddings.device)
                self.template_weights = nn.Parameter(init, requires_grad=True)
        elif (not use_gp) and use_linear_tw:
            # New approach: linear layer that transforms embeddings to weights
            D = int(self.text_embeddings.shape[-1])
            self.template_weight_linear = nn.Linear(D, 1, bias=False)
            # Initialize to produce roughly uniform weights
            with torch.no_grad():
                # We want the linear layer to initially produce similar values for all templates
                # So initialize weights to small random values
                self.template_weight_linear.weight.data.normal_(0, 0.01)

        # Learnable visual projection W (full-rank, bias-free), initialized to identity
        dim = int(self.text_embeddings.shape[-1])
        self.visual_proj = nn.Linear(dim, dim, bias=False)
        with torch.no_grad():
            eye = torch.eye(dim)
            self.visual_proj.weight.copy_(eye)
        # Align projection layer dtype/device with CLIP modules
        try:
            target_device = next(self.image_encoder.parameters()).device
        except StopIteration:
            target_device = self.text_embeddings.device
        self.visual_proj.to(device=target_device)
        self.gp_weighter = None
        self.gp_num_mc_samples_train = int(getattr(config.adapter, 'gp_num_mc_samples_train', 1) or 1)
        self.gp_num_mc_samples_eval = int(getattr(config.adapter, 'gp_num_mc_samples_eval', 1) or 1)
        if use_gp:
            self.gp_weighter = GaussianProcessTemplateWeighter(
                text_embeddings=self.text_embeddings,
                cfg=config,
            )

    def get_prototypes(self, num_samples: int = 1, visual_embeddings: torch.Tensor = None):
        """Get class prototypes; if GP is enabled, average over num_samples.

        When GP is disabled, use the zero-shot base text features.
        """
        target_device = next(self.parameters()).device
        if self.gp_weighter is not None:
            if num_samples <= 1: num_samples = 1
            prototypes = self.gp_weighter.sample_prototypes(num_samples, visual_embeddings)  # [S,K,D]
        else:
            if hasattr(self, 'template_weight_linear'):
                # Linear weighting: transform embeddings to scalars, then softmax
                logits = self.template_weight_linear(self.text_embeddings).squeeze(-1)  # [K, M, D] -> [K, M, 1] -> [K, M]
                template_weights = torch.softmax(logits, dim=-1)  # [K, M] normalized weights
                prototypes = torch.einsum("km,kmd->kd", template_weights, self.text_embeddings)
            elif isinstance(getattr(self, 'template_weights', None), torch.Tensor):
                # Original approach: direct template weights matrix
                template_weights = cast(torch.Tensor, self.template_weights)
                K = self.text_embeddings.shape[0]
                # If sharing weights across classes, broadcast from (1, M) to (K, M)
                if template_weights.shape[0] == 1 and K > 1:
                    template_weights = template_weights.expand(K, -1)
                prototypes = torch.einsum("km,kmd->kd", template_weights, self.text_embeddings)
            else:
                # Fallback to uniform mean if weights not initialized yet
                prototypes = self.text_embeddings.mean(dim=1)
        if prototypes.device != target_device:
            prototypes = prototypes.to(target_device)
        return prototypes
    
    def forward_features(self, features: torch.Tensor, num_samples: int | None = None) -> torch.Tensor:
        """Compute logits from precomputed visual features.

        This normalizes features, obtains (possibly GP-sampled) prototypes, normalizes them, and returns the scaled cosine similarities.
        """
        # Ensure features match projection dtype/device before any op
        proj_weight = self.visual_proj.weight
        if features.device != proj_weight.device:
            features = features.to(device=proj_weight.device)
        projected = self.visual_proj(features)
        features_norm = F.normalize(projected, p=2, dim=-1)
        scale = self.logit_scale.exp()
        num_samples = self.gp_num_mc_samples_train if self.training else self.gp_num_mc_samples_eval
        prototypes = self.get_prototypes(num_samples=num_samples, visual_embeddings=projected)
        if prototypes.device != features_norm.device:
            prototypes = prototypes.to(device=features_norm.device)
        prototypes_norm = F.normalize(prototypes, p=2, dim=-1)
        if prototypes.dim() == 3:
            logits_s = scale * torch.einsum("bd,skd->bsk", features_norm, prototypes_norm)
            base_logits = logits_s.mean(dim=1)
        else:
            base_logits = scale * (features_norm @ prototypes_norm.t())
        return base_logits

    def forward(self, image: torch.Tensor, return_features: bool = False):
        features = self.image_encoder(image)
        logits = self.forward_features(features)
        if return_features:
            return logits, features
        return logits


@TRAINER_REGISTRY.register("Adapter")
class Trainer(BaseTrainer):
    """Unified adapter trainer supporting both baseline and GP methods."""
    
    def __init__(self, config, dataset_manager):
        super().__init__(config, dataset_manager)

    def build_model(self):
        config = self.config
        classnames = self.dm.dataset.classnames
        print(f"Loading CLIP (backbone: {config.model.backbone_name})")
        clip_model = load_clip(config, self.device)
        print("Building custom CLIP")
        self.model = CustomCLIP(config, classnames, clip_model)
        self.model.to(self.device)
        for name, param in self.model.named_parameters():
            allow_template = (not config.adapter.use_gp) and bool(getattr(config.adapter, 'train_template_weights', False))
            allow_linear_template = (not config.adapter.use_gp) and bool(getattr(config.adapter, 'use_linear_template_weighting', False))
            if bool(getattr(config.adapter, 'freeze_visual_proj', False)) and ("visual_proj" in name):
                param.requires_grad = False
                continue
            if ("visual_proj" in name) or ("gp_weighter" in name) or (allow_template and ("template_weights" in name)) or (allow_linear_template and ("template_weight_linear" in name)):
                param.requires_grad = True
            else:
                param.requires_grad = False
        if config.adapter.use_gp and self.model.gp_weighter is not None:
            base_params = []
            base_params.extend([p for p in self.model.visual_proj.parameters() if p.requires_grad])
            gp_params = [p for p in self.model.gp_weighter.parameters() if p.requires_grad]
            if (len(base_params) + len(gp_params)) == 0:
                self.optim = None
                self.sched = None
            else:
                param_groups = [
                    {
                        'params': base_params,
                        'lr': float(config.optim.lr),
                        'weight_decay': float(config.optim.weight_decay)
                    },
                    {
                        'params': gp_params,
                        'lr': float(config.adapter.gp_lr),
                        'weight_decay': float(config.optim.weight_decay)
                    },
                ]
                self.optim = build_optimizer_from_param_groups(param_groups, config.optim)
                self.sched = build_lr_scheduler(self.optim, config.optim)
        else:
            baseline_params = []
            baseline_params.extend([p for p in self.model.visual_proj.parameters() if p.requires_grad])
            if hasattr(self.model, 'template_weights') and isinstance(self.model.template_weights, torch.nn.Parameter):
                if self.model.template_weights.requires_grad:
                    baseline_params.append(self.model.template_weights)
            if hasattr(self.model, 'template_weight_linear'):
                baseline_params.extend([p for p in self.model.template_weight_linear.parameters() if p.requires_grad])
            if len(baseline_params) == 0:
                self.optim = None
                self.sched = None
            else:
                self.optim = build_optimizer(baseline_params, config.optim)
                self.sched = build_lr_scheduler(self.optim, config.optim)
        self.scaler = None

    def forward_backward(self, batch):
        """Forward pass and backward pass with loss computation."""
        model = cast(CustomCLIP, self.model)
        features, labels = batch
        features = torch.tensor(features).to(self.device)
        if not torch.is_tensor(labels):
            labels = torch.tensor(labels).to(self.device)
        else:
            labels = labels.detach().clone().to(self.device)
        projected_features = features
        proj_features = self.model.visual_proj(projected_features.to(self.model.visual_proj.weight.dtype))
        num_samples = int(getattr(self.config.adapter, 'gp_num_mc_samples_train', 1) or 1)
        prototypes = model.get_prototypes(num_samples=num_samples, visual_embeddings=proj_features)
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
        
        # Compute loss with Monte Carlo expectation over GP prototypes (if enabled)
        loss = self.compute_loss(projected_features, labels, num_samples=num_samples)

        # Compute logits via the model's centralized path for metrics
        logits = model.forward_features(projected_features, num_samples=num_samples)

        # Backward pass
        self._backward_and_update(loss)
        
        # Compute accuracies for logging
        with torch.no_grad():
            # Training accuracy
            acc_train = compute_accuracy(logits, labels)[0]
            was_training = self.model.training
            self.model.eval()
            num_samples = int(getattr(self.config.adapter, 'gp_num_mc_samples_eval', 1) or 1)
            # Test accuracy (using stored test features)
            test_features = self.features_test.to(self.device)
            # Test features are CLIP visual features
            test_projected = test_features
            # Quick test metric: use the same unified path
            test_proj = self.model.visual_proj(test_features.to(self.model.visual_proj.weight.dtype))
            test_prototypes = model.get_prototypes(num_samples=num_samples, visual_embeddings=test_proj)
            if test_projected.dtype != test_prototypes.dtype:
                test_prototypes = test_prototypes.to(dtype=test_projected.dtype)
            test_logits = model.forward_features(test_projected)
            acc_test = compute_accuracy(test_logits, self.labels_test.to(self.device))[0]
            self.model.train(was_training)
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

        # Cross entropy
        if use_gp and num_samples > 1 and gp_weighter is not None:
            # Sample S prototype sets [S,K,D] in fp32 for stability
            proj_features = self.model.visual_proj(features.to(self.model.visual_proj.weight.dtype))
            protos = gp_weighter.sample_prototypes(num_samples, visual_embeddings=proj_features)
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
            scale = model.logit_scale.exp()
            ce_vals = []
            for s in range(num_samples):
                prototypes_s = protos[s] # [K,D]
                prototypes_norm = F.normalize(prototypes_s, p=2, dim=-1)
                logits_s = scale * (features_norm @ prototypes_norm.t())
                ce_vals.append(F.cross_entropy(logits_s, labels))
            ce_loss = torch.stack(ce_vals, dim=0).mean()
        else:
            f = features
            proj_weight = model.visual_proj.weight
            if f.device != proj_weight.device:
                f = f.to(device=proj_weight.device)
            if f.dtype != proj_weight.dtype:
                f = f.to(dtype=proj_weight.dtype)
            logits = model.forward_features(f)
            ce_loss = F.cross_entropy(logits, labels)
        total_loss = ce_loss

        # # GP ELBO
        # if use_gp and gp_weighter is not None and bool(getattr(self.config.adapter, 'gp_use_elbo', True)):
        #     try:
        #         gp_weighter.train()
        #         if hasattr(gp_weighter, 'likelihood'):
        #             gp_weighter.likelihood.train()
        #         if not hasattr(self, '_gp_targets') or getattr(self, '_gp_targets', None) is None:
        #             try:
        #                 self._gp_targets = self._compute_gp_template_targets_prob()
        #             except Exception:
        #                 self._gp_targets = None
        #         if getattr(self, '_gp_targets', None) is not None:
        #             x = gp_weighter._templates
        #             y = cast(torch.Tensor, self._gp_targets)
        #             if y.device != x.device:
        #                 y = y.to(x.device)
        #             mll = gpytorch.mlls.VariationalELBO(
        #                 gp_weighter.likelihood,
        #                 gp_weighter,
        #                 num_data=int(gp_weighter.num_templates),
        #                 beta=float(self.config.adapter.gp_beta),
        #             )
        #             out = gp_weighter(x)
        #             elbo_val = mll(out, y)
        #             elbo_mean = elbo_val.mean() if elbo_val.dim() > 0 else elbo_val
        #             total_loss = total_loss + (-elbo_mean)
        #             try:
        #                 self._dbg_loss_components = getattr(self, '_dbg_loss_components', {})
        #                 self._dbg_loss_components['elbo'] = float((-elbo_mean).detach().item())
        #             except Exception:
        #                 pass
        #     except Exception:
        #         pass

        # L2 regularization
        l2_reg = None
        try:
            # Visual projection regularization (to identity)
            W = model.visual_proj.weight
            if W.requires_grad:
                eye = torch.eye(W.shape[0], device=W.device, dtype=W.dtype)
                l2_reg = (W - eye).pow(2).sum()
                l2_lambda = self.config.adapter.l2_lambda
                shots = self.config.dataset.num_shots
                l2_reg = l2_reg * l2_lambda / shots
                total_loss += l2_reg

            # Linear template weighting regularization (to small weights)
            if hasattr(model, 'template_weight_linear') and model.template_weight_linear.weight.requires_grad:
                linear_reg = model.template_weight_linear.weight.pow(2).sum()
                linear_reg = linear_reg * l2_lambda / shots
                total_loss += linear_reg
                if l2_reg is not None:
                    l2_reg += linear_reg
                else:
                    l2_reg = linear_reg

        except Exception:
            print("Error in L2 regularization")
            l2_reg = None

        # Learnable token regularizer
        if use_gp and gp_weighter is not None:
            # proj visuals
            proj = self.model.visual_proj(features.to(self.model.visual_proj.weight.dtype)).detach()
            labels_i64 = labels.to(torch.int64)
            
            K = int(self.model.text_embeddings.shape[0])
            D = proj.shape[-1]
            device = proj.device
            dtype = proj.dtype

            cls_sum = torch.zeros(K, D, dtype=proj.dtype, device=proj.device) # [K,D]
            cls_count = torch.zeros(K, 1, dtype=proj.dtype, device=proj.device) # [K,1]
            cls_sum.index_add_(0, labels_i64, proj)  # [K,D]
            one = torch.ones(labels_i64.size(0), 1, device=proj.device, dtype=proj.dtype)
            cls_count.index_add_(0, labels_i64, one)

            # classes present in the batch
            present = (cls_count > 0).squeeze(1) # [K]
            if present.any():
                cls_mean = torch.zeros_like(cls_sum)
                cls_mean[present] = cls_sum[present] / cls_count[present].clamp_min(1.0) # [K,D]

                Z = gp_weighter.variational_strategy.inducing_points # [K, M+1, d]
                token_red = Z[:, -1, :] # [K,d]
                token = gp_weighter._lift(token_red) # [K,D]
                
                lam = float(getattr(self.config.adapter, "learn_token_lambda", 1e-3))
                reg = lam * (token[present] - cls_mean[present]).pow(2).mean()
                total_loss = total_loss + reg
                try:
                    self._dbg_loss_components["learn_token_reg"] = float(reg.detach().item())
                except Exception:
                    pass
            w = gp_weighter.scores
            
            orth = None
            # A_kernel = gp_weighter.A.weight
            # orth = 1.e-2 * ((A_kernel @ A_kernel.t()) - torch.eye(A_kernel.size(0), device=A_kernel.device, dtype=A_kernel.dtype)).pow(2).sum()
            # total_loss = total_loss + orth
            # eps = 1e-8
            # lam_entropy = 1.e-2
            # H = -(w.clamp_min(eps) * w.clamp_min(eps).log()).sum(dim=-1)
            # score_entropy_reg = -lam_entropy * H.mean()
            # total_loss = total_loss + score_entropy_reg

        # Debug components
        try:
            # Calculate individual regularization components
            vis_reg = None
            linear_reg = None
            try:
                if model.visual_proj.weight.requires_grad:
                    eye = torch.eye(model.visual_proj.weight.shape[0], device=model.visual_proj.weight.device, dtype=model.visual_proj.weight.dtype)
                    vis_reg = (model.visual_proj.weight - eye).pow(2).sum()
                    vis_reg = vis_reg * self.config.adapter.l2_lambda / self.config.dataset.num_shots
            except:
                pass
            try:
                if hasattr(model, 'template_weight_linear') and model.template_weight_linear.weight.requires_grad:
                    linear_reg = model.template_weight_linear.weight.pow(2).sum()
                    linear_reg = linear_reg * self.config.adapter.l2_lambda / self.config.dataset.num_shots
            except:
                pass

            self._dbg_loss_components = {
                "ce": float(ce_loss.detach().item()),
                "l2_reg": float(l2_reg.detach().item()) if l2_reg is not None else 0.0,
                "vis_reg": float(vis_reg.detach().item()) if vis_reg is not None else 0.0,
                "linear_reg": float(linear_reg.detach().item()) if linear_reg is not None else 0.0,
                "learn_token_reg": float(self._dbg_loss_components.get("learn_token_reg", 0.0)) if hasattr(self, '_dbg_loss_components') else 0.0,
                "orth": float(orth.detach().item()) if orth is not None else 0.0,
                "scores": w.mean(0)[0].detach().cpu() if (use_gp and gp_weighter is not None) else None,
                "total": float(total_loss.detach().item()),
            }
        except Exception:
            pass

        return total_loss

    def _backward_and_update(self, loss):
        """Backward pass and optimizer step."""
        if getattr(self, 'optim', None) is None:
            return
        self.optim.zero_grad()
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
        start_time = time.time()
        self.build_model()
        self.set_model_mode("eval")

        # Feature extraction on test set
        self.labels_test, output_test, self.features_test = self.extract_features(partition="test")
        print("Zero-Shot accuracy on test: " + str(round(compute_accuracy(output_test.cuda(), self.labels_test.cuda())[0], 2)))
        self.labels_train, logits_zs, self.features_train = self.extract_features(partition="train")
        model = cast(CustomCLIP, self.model)
        feats = self.features_train.to(self.device)
        template_weights = _get_template_weights(
            self.config,
            text_embeddings=model.text_embeddings,
            features=feats,
            labels=self.labels_train.to(self.device),
            logit_scale=model.logit_scale.exp(),
        )

        # If sharing template weights across classes, average over the class dimension
        if getattr(self.config.adapter, 'shared_template_weights', False):
            template_weights = template_weights.mean(dim=0, keepdim=True)  # [1, M]

        if hasattr(model, 'template_weight_linear'):
            # For linear weighting, no need to set template_weights since weights are computed dynamically
            pass
        elif hasattr(model, 'template_weights') and isinstance(getattr(model, 'template_weights'), torch.nn.Parameter):
            with torch.no_grad():
                model.template_weights.data.copy_(template_weights.to(dtype=model.text_embeddings.dtype, device=model.text_embeddings.device))
        else:
            model.template_weights = template_weights.to(dtype=model.text_embeddings.dtype, device=model.text_embeddings.device)

        if getattr(self.config.adapter, 'use_gp', False) and getattr(model, 'gp_weighter', None) is not None:
            model.gp_weighter.initialize_from_weights(template_weights)
            print("[GP] One-step initialization applied to GP weights.")
        self.before_train()
        
        for self.epoch in range(self.start_epoch, self.max_epoch):
            self.before_epoch()
            self.run_epoch()
            self.after_epoch()
        
        # Optional sanity-check: fine-tune ONLY the template weights on the test set
        try:
            if getattr(self.config.adapter, 'finetune_on_test', False):
                self._finetune_on_test()
        except Exception as e:
            print(f"[WARN] Template weights test-set fine-tune failed: {e}")
            
        self.after_train()
        # Print template weights stats if available
        try:
            if getattr(self.config.adapter, 'use_gp', False):
                print(self.model.gp_weighter.scores.shape)
                mean_vals = self.model.gp_weighter.scores.mean(dim=0)[0].tolist()
                std_vals = self.model.gp_weighter.scores.std(dim=0)[0].tolist()
                print("Weights: mean = [{}]".format(", ".join(f"{v:.4f}" for v in mean_vals)))
                print("          std = [{}]".format(", ".join(f"{v:.4f}" for v in std_vals)))
            elif hasattr(self.model, 'template_weight_linear'):
                # Linear weighting: compute weights and show stats
                with torch.no_grad():
                    logits = self.model.template_weight_linear(self.model.text_embeddings).squeeze(-1)
                    weights = torch.softmax(logits, dim=-1)
                print(weights.shape)
                if weights.shape[0] == 1:
                    # Shared weights across classes
                    weight_vals = weights.squeeze(0).tolist()
                    print("Linear weights: [{}]".format(", ".join(f"{v:.4f}" for v in weight_vals)))
                else:
                    # Per-class weights
                    mean_vals = weights.mean(dim=0).tolist()
                    std_vals = weights.std(dim=0).tolist()
                    print("Linear weights: mean = [{}]".format(", ".join(f"{v:.4f}" for v in mean_vals)))
                    print("                 std = [{}]".format(", ".join(f"{v:.4f}" for v in std_vals)))
            else:
                print(self.model.template_weights.shape)
                if self.model.template_weights.shape[0] == 1:
                    # Shared weights across classes
                    weights = self.model.template_weights.squeeze(0).tolist()
                    print("Shared weights: [{}]".format(", ".join(f"{v:.4f}" for v in weights)))
                else:
                    # Per-class weights
                    mean_vals = self.model.template_weights.mean(dim=0).tolist()
                    std_vals = self.model.template_weights.std(dim=0).tolist()
                    print("Weights: mean = [{}]".format(", ".join(f"{v:.4f}" for v in mean_vals)))
                    print("          std = [{}]".format(", ".join(f"{v:.4f}" for v in std_vals)))
        except Exception:
            pass
        print(f"Training completed in {time.time() - start_time:.2f} seconds")

        # After training completes, compute final test metrics and write metrics.json
        try:
            metrics = self._compute_final_metrics()
            self._write_run_summary_json(metrics, start_time=start_time)
        except Exception as e:
            print(f"[WARN] Failed to write metrics.json: {e}") 

    def run_epoch(self):
        """Run one training epoch."""
        losses = AverageMeter()
        batch_time = AverageMeter()
        data_time = AverageMeter()

        # Set model to train mode but keep encoders frozen
        self.set_model_mode("train")
        try:
            if hasattr(self.model, "image_encoder"):
                self.model.image_encoder.eval()
        except AttributeError:
            pass
        try:
            if hasattr(self.model, "text_encoder"):
                self.model.text_encoder.eval()
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
                                f"learn_token_reg={comp.get('learn_token_reg', 0):.4f} "
                                f"orth={comp.get('orth', 0):.4f} "
                                f"l2_reg={comp.get('l2_reg', comp.get('l2', 0)):.4f} "
                                f"vis_reg={comp.get('vis_reg', 0):.4f} linear_reg={comp.get('linear_reg', 0):.4f} "
                                f"scores={comp.get('scores', None)} "
                                f"Total={comp.get('total', 0):.4f}  "
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
                                    # GP parameters
                                    q = gp.variational_strategy._variational_distribution
                                    q_m = q.variational_mean.detach()
                                    L = q.chol_variational_covar
                                    q_var = (L @ L.transpose(-1, -2)).diagonal(dim1=-2, dim2=-1)
                                    covar_module = gp.covar_module if hasattr(gp, "covar_module") else None
                                    
                                    q_m_min = float(q_m.min().item())
                                    q_m_max = float(q_m.max().item())
                                    q_var_min = float(q_var.min().item())
                                    q_var_max = float(q_var.max().item())

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
                                    f"  [DBG][GP] lengthscale={ls_val:.6f} outputscale={outscale_val:.6f} mean_param_norm={mean_norm:.4f} mean_abs={mean_abs:.4f}"
                                    f"\n  [DBG][GP] q_m[min={q_m_min:.4f} max={q_m_max:.4f}] q_var[min={q_var_min:.4f} max={q_var_max:.4f}]"
                                )
                                # Template weights for class 0 (posterior-mean softmax over templates)
                                try:
                                    with torch.no_grad():
                                        # x_t = gp._templates_red.to(dtype=torch.float32, device=gp._templates_red.device)
                                        # f_mean = gp(x_t).mean  # [K, M]
                                        # w = torch.softmax(f_mean, dim=-1)  # [K, M]
                                        w = gp.scores
                                        w0 = w[0].detach().cpu().tolist()
                                        w0_str = ", ".join(f"{v:.3f}" for v in w0)
                                        print(f"  [DBG][GP] template_weights[class=0]: [{w0_str}]")
                                except Exception:
                                    pass
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
        features = self.features_train.detach().cpu()  # [N, D]
        labels = self.labels_train.detach().cpu().to(torch.int64)        # [N]
        text_emb = cast(torch.Tensor, self.model.text_embeddings).detach().cpu()  # [K, M, D]

        K, M, D = int(text_emb.shape[0]), int(text_emb.shape[1]), int(text_emb.shape[2])
        N = int(features.shape[0])

        # Apply current projection W if present
        try:
            W = cast(torch.nn.Linear, self.model.visual_proj).weight.detach().cpu()  # [D, D]
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

    def _finetune_on_test(self) -> None:
        """Sanity check: cheating on the test set to validate the template weights.

        This freezes all non-template weights parameters (including the visual projection) and
        optimizes only the template weights on the held-out test features to validate
        the ceiling performance of the template weights.
        """
        use_gp = bool(getattr(self.config.adapter, 'use_gp', False)) and getattr(self.model, 'gp_weighter', None) is not None

            # Ensure template weights are registered as trainable parameters when GP is disabled
        if not use_gp:
            tw = getattr(self.model, 'template_weights', None)
            if isinstance(tw, torch.Tensor) and not isinstance(tw, torch.nn.Parameter):
                self.model.template_weights = torch.nn.Parameter(tw.detach().clone(), requires_grad=True)

        # Freeze everything except GP parameters
        for name, param in self.model.named_parameters():
            if use_gp and "gp_weighter" in name:
                param.requires_grad = True
            elif not use_gp and ("template_weights" in name or "template_weight_linear" in name):
                param.requires_grad = True
            else:
                param.requires_grad = False

        # Optimizer
        lr = float(self.config.optim.lr)
        weight_decay = float(self.config.optim.weight_decay)
        param_groups = [{'params': [p for n, p in self.model.named_parameters() if p.requires_grad], 'lr': lr, 'weight_decay': weight_decay}]
        optim = build_optimizer_from_param_groups(param_groups, self.config.optim)
        sched = build_lr_scheduler(optim, self.config.optim)

        # Prepare loop settings
        self.set_model_mode("train")
        try:
            if hasattr(self.model, "image_encoder"):
                self.model.image_encoder.eval()
        except AttributeError:
            pass
        try:
            if hasattr(self.model, "text_encoder"):
                self.model.text_encoder.eval()
        except AttributeError:
            pass

        # Use pre-extracted test features/labels
        if not hasattr(self, 'features_test') or not hasattr(self, 'labels_test'):
            raise RuntimeError("features_test and labels_test must be available before template weights fine-tune")

        features = self.features_test.clone().cpu().numpy()
        labels = self.labels_test.clone()

        # Shuffle once per run
        idx = np.random.rand(features.shape[0]).argsort(axis=0)
        features = features[idx, :]
        labels = labels[idx]

        # Batch/epoch config
        default_bs = int(getattr(self.config.dataloader, 'batch_size_test', getattr(self.config.dataloader, 'batch_size_train', 128)))
        batch_size = max(1, int(getattr(self.config.adapter, 'gp_test_batch_size', default_bs) or default_bs))
        num_batches = int(math.ceil(features.shape[0] / float(batch_size)))
        num_epochs = 100
        num_mc = int(getattr(self.config.adapter, 'gp_num_mc_samples_train', 1) or 1) # for gp

        print(f"[SANITY] Template weights fine-tuning on TEST set: epochs={num_epochs} bs={batch_size} lr={lr}")
        for ep in range(num_epochs):
            running_loss = 0.0
            seen = 0
            for b in range(num_batches):
                b0 = b * batch_size
                b1 = min((b + 1) * batch_size, features.shape[0])
                x_b = torch.tensor(features[b0:b1]).to(self.device)
                y_b = labels[b0:b1]
                if not torch.is_tensor(y_b):
                    y_b = torch.tensor(y_b).to(self.device)
                else:
                    y_b = y_b.detach().clone().to(self.device)

                loss = self.compute_loss(x_b, y_b, num_samples=num_mc)
                optim.zero_grad()
                loss.backward()
                optim.step()

                batch_sz = (b1 - b0)
                running_loss += float(loss.detach().item()) * batch_sz
                seen += batch_sz

            if sched is not None:
                try:
                    sched.step()
                except Exception:
                    pass

            # Evaluate test accuracy after each epoch
            with torch.no_grad():
                logits = self.model.forward_features(self.features_test.to(self.device))
                acc = compute_accuracy(logits, self.labels_test.to(self.device))[0]
            avg_loss = running_loss / max(1, seen)
            print(f"[SANITY] Template weights test fine-tune epoch {ep+1}/{num_epochs}: loss={avg_loss:.4f} acc_test={float(acc):.4f}")


