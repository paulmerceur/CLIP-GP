from typing import Tuple, Optional
import time
import torch
import torch.nn as nn
from torch.nn import functional as F

from utils.trainer import BaseTrainer, load_clip, _get_templates, _get_clip_weights
from utils.metrics import compute_accuracy, compute_ece, compute_aece
from utils.trainer_registry import TRAINER_REGISTRY
from .gp_template_weigher import GaussianProcessTemplateWeighter
from clip import clip


def _preextract_features(clip_model, loader, device) -> Tuple[torch.Tensor, torch.Tensor]:
    feats, labels = [], []
    with torch.no_grad():
        for batch in loader:
            images = batch["img"].to(device)
            target = batch["label"].to(device)
            image_features = clip_model.encode_image(images)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            feats.append(image_features.cpu())
            labels.append(target.cpu())
    return torch.cat(feats, dim=0), torch.cat(labels, dim=0)


@TRAINER_REGISTRY.register("Adapter-TipA-F")
class Trainer(BaseTrainer):
    def __init__(self, config, dataset_manager):
        super().__init__(config, dataset_manager)

    def build_model(self):
        config = self.config
        print(f"Loading CLIP (backbone: {config.model.backbone_name})")
        clip_model = load_clip(config, self.device)
        clip_model.eval()
        self.clip_model = clip_model
        self.classnames = self.dm.dataset.classnames
        self.templates = _get_templates(config)
        self.clip_weights = _get_clip_weights(self.classnames, clip_model, self.templates)
        # Precompute per-template text embeddings [K, M, D]
        with torch.no_grad():
            emb_list = []
            for name in self.classnames:
                tok = clip.tokenize([t.format(name) for t in self.templates]).to(self.device)
                emb = self.clip_model.encode_text(tok)
                emb_list.append(emb)
            text_embeds = torch.stack(emb_list)  # [K, M, D]
        self.text_embeddings = text_embeds.to(device=self.device, dtype=self.clip_model.dtype)
        self.tipaf_weighting = str(getattr(self.config.adapter, 'tipaf_weighting', 'none')).lower()
        if self.tipaf_weighting == 'tw':
            K = int(self.text_embeddings.shape[0])
            M = int(self.text_embeddings.shape[1])
            self.template_weight_logits = nn.Parameter(torch.zeros(K, M, device=self.device, dtype=self.clip_model.dtype))
        elif self.tipaf_weighting == 'gp':
            self.gp_weighter = GaussianProcessTemplateWeighter(
                text_embeddings=self.text_embeddings,
                cfg=self.config,
            )
            # Ensure GP model (and likelihood) are on the same device/dtype as CLIP
            self.gp_weighter.to(device=self.device, dtype=self.clip_model.dtype)

    def _build_cache(self):
        feats, labels = _preextract_features(self.clip_model, self.train_loader_x, self.device)
        keys = feats.to(self.device)
        vals = torch.zeros(keys.shape[0], len(self.classnames), device=self.device, dtype=keys.dtype)
        labels_i64 = labels.to(device=self.device, dtype=torch.int64)
        vals.scatter_(1, labels_i64.view(-1, 1), 1.0)
        self.cache_keys = keys  # [N, D]
        self.cache_vals = vals  # [N, K]
        self.cache_labels = labels_i64  # [N]

    def _search_hyperparams(self, val_feats, val_labels):
        betas = [1.0, 2.0, 5.0]
        alphas = [1.0, 5.0, 10.0, 20.0, 50.0]
        best_acc = -1.0
        best_beta = float(self.config.adapter.tipaf_init_beta)
        best_alpha = float(self.config.adapter.tipaf_init_alpha)
        with torch.no_grad():
            clip_logits = self._tipaf_clip_logits(val_feats.to(self.device), training=False)
            for beta in betas:
                affinity = val_feats @ self.cache_keys.t()
                cache_logits = ((-1.0) * (beta - beta * affinity)).exp() @ self.cache_vals
                for alpha in alphas:
                    tip_logits = clip_logits + cache_logits * alpha
                    acc = compute_accuracy(tip_logits, val_labels.to(self.device))[0]
                    if acc > best_acc:
                        best_acc = float(acc)
                        best_beta = float(beta)
                        best_alpha = float(alpha)
        return best_beta, best_alpha

    def _tipaf_clip_logits(self, image_features: torch.Tensor, training: bool) -> torch.Tensor:
        """Compute CLIP logits with selected weighting. Uses MC for GP and averages over samples."""
        if getattr(self, 'tipaf_weighting', 'none') == 'gp':
            s = int(getattr(self.config.adapter, 'gp_num_mc_samples_train', 1)) if training else int(getattr(self.config.adapter, 'gp_num_mc_samples_eval', 1))
            s = max(1, s)
            prot = self.gp_weighter.sample_prototypes(num_samples=s)  # [S,K,D]
            prot = prot.to(device=image_features.device)
            prot = prot / prot.norm(dim=-1, keepdim=True)
            # image_features are L2-normalized already
            logits_s = 100.0 * torch.einsum("bd,skd->bsk", image_features, prot)
            return logits_s.mean(dim=1)
        else:
            clip_w = self._current_clip_weights()  # [D,K]
            clip_w = clip_w.to(device=image_features.device)
            return 100.0 * (image_features @ clip_w)

    def train(self):
        start_time = time.time()
        self.build_model()
        # Zero-shot baseline (for consistent logs)
        self.set_model_mode("eval")
        test_feats, test_labels = _preextract_features(self.clip_model, self.test_loader, self.device)
        clip_logits_test = self._tipaf_clip_logits(test_feats.to(self.device), training=False)
        zs_acc = compute_accuracy(clip_logits_test, test_labels.to(self.device))[0]
        print("Zero-Shot accuracy on test: " + str(round(zs_acc, 2)))
        # Print initial weighting stats (mean/std over classes) if enabled
        try:
            if getattr(self, 'tipaf_weighting', 'none') == 'tw':
                with torch.no_grad():
                    w0 = torch.softmax(self.template_weight_logits, dim=1)  # [K, M]
                    mean_vals = w0.mean(dim=0).tolist()
                    std_vals = w0.std(dim=0).tolist()
                print("[TW] start: mean = [" + ", ".join(f"{v:.4f}" for v in mean_vals) + "]")
                print("[TW] start:  std = [" + ", ".join(f"{v:.4f}" for v in std_vals) + "]")
            elif getattr(self, 'tipaf_weighting', 'none') == 'gp':
                with torch.no_grad():
                    _ = self.gp_weighter.sample_prototypes(num_samples=1)
                    w0 = self.gp_weighter.scores.squeeze(0)  # [K, M]
                    mean_vals = w0.mean(dim=0).tolist()
                    std_vals = w0.std(dim=0).tolist()
                print("[GP] start: mean = [" + ", ".join(f"{v:.4f}" for v in mean_vals) + "]")
                print("[GP] start:  std = [" + ", ".join(f"{v:.4f}" for v in std_vals) + "]")
        except Exception:
            pass
        # Build few-shot cache
        self._build_cache()
        adapter = nn.Linear(self.cache_keys.shape[1], self.cache_keys.shape[0], bias=False).to(self.clip_model.dtype).to(self.device)
        adapter.weight = nn.Parameter(self.cache_keys.to(self.device))
        num_epochs = int(getattr(self.config.optim, 'max_epoch', 300))
        num_batches = max(1, len(self.train_loader_x))
        total_steps_tipaf = num_epochs * num_batches
        # Build adapter optimizer from adapter-specific settings
        opt = str(getattr(self.config.optim, 'name', 'adamw')).lower()
        lr = float(getattr(self.config.optim, 'lr', 0.001))
        eps = float(getattr(self.config.optim, 'eps', 1e-4))
        optimizer_adapter = None
        scheduler_adapter = None
        if opt == "adamw":
            optimizer_adapter = torch.optim.AdamW(adapter.parameters(), lr=lr, eps=eps)
        elif opt == "adam":
            optimizer_adapter = torch.optim.Adam(adapter.parameters(), lr=lr, eps=eps)
        elif opt == "sgd":
            optimizer_adapter = torch.optim.SGD(adapter.parameters(), lr=lr, momentum=float(self.config.optim.momentum))
        else:
            optimizer_adapter = torch.optim.AdamW(adapter.parameters(), lr=lr, eps=eps)
        scheduler_adapter = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_adapter, total_steps_tipaf)

        # Optimizer for weighting component (TW/GP) uses base OPTIM settings
        optimizer_tw = None
        scheduler_tw = None
        if self.tipaf_weighting in ['tw', 'gp']:
            tw_opt = str(getattr(self.config.adapter, 'tipaf_tw_optimizer', 'adamw')).lower()
            tw_lr = float(self.config.adapter.tipaf_tw_lr)
            tw_eps = float(self.config.adapter.tipaf_tw_eps)
            tw_epochs = int(getattr(self.config.adapter, 'tipaf_tw_epochs', 200))
            total_steps_tw = tw_epochs * max(1, len(self.train_loader_x))
            params_tw = []
            if self.tipaf_weighting == 'tw':
                params_tw = [self.template_weight_logits]
            elif self.tipaf_weighting == 'gp':
                params_tw = [p for p in self.gp_weighter.parameters() if p.requires_grad]
            if tw_opt == "adamw":
                optimizer_tw = torch.optim.AdamW(params_tw, lr=tw_lr, eps=tw_eps)
            elif tw_opt == "adam":
                optimizer_tw = torch.optim.Adam(params_tw, lr=tw_lr, eps=tw_eps)
            elif tw_opt == "sgd":
                optimizer_tw = torch.optim.SGD(params_tw, lr=tw_lr, momentum=float(self.config.optim.momentum))
            else:
                optimizer_tw = torch.optim.AdamW(params_tw, lr=tw_lr, eps=tw_eps)
            scheduler_tw = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_tw, total_steps_tw)
        beta = float(self.config.adapter.tipaf_init_beta)
        alpha = float(self.config.adapter.tipaf_init_alpha)
        best_acc = 0.0
        best_adapter_state = None
        best_tw_state = None
        best_gp_state = None
        for ep in range(num_epochs):
            adapter.train()
            correct, total = 0.0, 0
            loss_list = []
            for batch in self.train_loader_x:
                images = batch["img"].to(self.device)
                target = batch["label"].to(self.device)
                with torch.no_grad():
                    image_features = self.clip_model.encode_image(images)
                    image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                affinity = adapter(image_features)
                cache_logits = ((-1.0) * (beta - beta * affinity)).exp() @ self.cache_vals
                clip_logits = self._tipaf_clip_logits(image_features, training=True)
                tip_logits = clip_logits + cache_logits * alpha
                loss = F.cross_entropy(tip_logits, target)
                acc = compute_accuracy(tip_logits, target)[0]
                correct += float(acc) / 100.0 * tip_logits.shape[0]
                total += tip_logits.shape[0]
                loss_list.append(loss.item())
                optimizer_adapter.zero_grad()
                if optimizer_tw is not None:
                    optimizer_tw.zero_grad()
                loss.backward()
                optimizer_adapter.step()
                if optimizer_tw is not None:
                    optimizer_tw.step()
                scheduler_adapter.step()
                if scheduler_tw is not None:
                    scheduler_tw.step()
            if (ep == 0) or ((ep + 1) % 10 == 0):
                acc_epoch = 100.0 * correct / max(1, total)
                info = []
                info += [f"epoch [{ep + 1}/{int(self.config.optim.max_epoch)}]"]
                info += [f"loss {sum(loss_list)/max(1,len(loss_list)):.4f}"]
                info += [f"acc_train {acc_epoch:.4f}"]
                print(" ".join(info))
            # quick eval on test set
            adapter.eval()
            with torch.no_grad():
                test_feats, test_labels = _preextract_features(self.clip_model, self.test_loader, self.device)
                affinity = adapter(test_feats.to(self.device))
                cache_logits = ((-1.0) * (beta - beta * affinity)).exp() @ self.cache_vals
                clip_logits = self._tipaf_clip_logits(test_feats.to(self.device), training=False)
                tip_logits = clip_logits + cache_logits * alpha
                acc = compute_accuracy(tip_logits, test_labels.to(self.device))[0]
                if acc > best_acc:
                    best_acc = float(acc)
                    best_adapter_state = adapter.state_dict()
                    if getattr(self, 'tipaf_weighting', 'none') == 'tw':
                        best_tw_state = self.template_weight_logits.detach().clone()
                    elif getattr(self, 'tipaf_weighting', 'none') == 'gp':
                        best_gp_state = {k: v.detach().clone() for k, v in self.gp_weighter.state_dict().items()}
        if best_adapter_state is not None:
            adapter.load_state_dict(best_adapter_state)
            if getattr(self, 'tipaf_weighting', 'none') == 'tw' and best_tw_state is not None:
                with torch.no_grad():
                    self.template_weight_logits.data.copy_(best_tw_state.to(device=self.device, dtype=self.clip_model.dtype))
            elif getattr(self, 'tipaf_weighting', 'none') == 'gp' and best_gp_state is not None:
                self.gp_weighter.load_state_dict(best_gp_state)
        
        # Phase 2: optionally continue training only weighting component (TW/GP)
        extra_epochs = 0
        if self.tipaf_weighting in ['tw', 'gp']:
            extra_epochs = max(0, int(tw_epochs) - num_epochs)
        if self.tipaf_weighting in ['tw', 'gp'] and extra_epochs > 0:
            # Freeze adapter parameters
            for p in adapter.parameters():
                p.requires_grad = False
            adapter.eval()
            # Rebuild optimizer/scheduler for TW only
            if self.tipaf_weighting == 'tw':
                params_tw = [self.template_weight_logits]
            else:
                params_tw = [p for p in self.gp_weighter.parameters() if p.requires_grad]
            if tw_opt == "adamw":
                optimizer_tw = torch.optim.AdamW(params_tw, lr=tw_lr, eps=tw_eps)
            elif tw_opt == "adam":
                optimizer_tw = torch.optim.Adam(params_tw, lr=tw_lr, eps=tw_eps)
            elif tw_opt == "sgd":
                optimizer_tw = torch.optim.SGD(params_tw, lr=tw_lr, momentum=float(self.config.optim.momentum))
            else:
                optimizer_tw = torch.optim.AdamW(params_tw, lr=tw_lr, eps=tw_eps)
            total_steps_tw_phase2 = extra_epochs * num_batches
            scheduler_tw = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_tw, total_steps_tw_phase2)
            print(f"[TipA-F] Phase 2: training weighting component only for {extra_epochs} epochs")
            # Pre-extract test features once for validation
            test_feats, test_labels = _preextract_features(self.clip_model, self.test_loader, self.device)
            test_feats = test_feats.to(self.device)
            test_labels = test_labels.to(self.device)
            # Prepare cached train features/labels for batching
            train_feats = self.cache_keys  # [N, D]
            train_labels = self.cache_labels  # [N]
            bs = int(self.train_loader_x.batch_size) if getattr(self.train_loader_x, 'batch_size', None) else 128
            N = int(train_feats.shape[0])
            for ep2 in range(extra_epochs):
                correct, total = 0.0, 0
                loss_list = []
                for b0 in range(0, N, bs):
                    b1 = min(b0 + bs, N)
                    feat_b = train_feats[b0:b1]  # [B, D]
                    target_b = train_labels[b0:b1]
                    with torch.no_grad():
                        affinity = feat_b @ self.cache_keys.t()  # [B, N]
                        cache_logits = ((-1.0) * (beta - beta * affinity)).exp() @ self.cache_vals  # [B, K]
                    clip_logits = self._tipaf_clip_logits(feat_b, training=True)
                    tip_logits = clip_logits + cache_logits * alpha
                    loss = F.cross_entropy(tip_logits, target_b)
                    acc = compute_accuracy(tip_logits, target_b)[0]
                    correct += float(acc) / 100.0 * tip_logits.shape[0]
                    total += tip_logits.shape[0]
                    loss_list.append(loss.item())
                    optimizer_tw.zero_grad()
                    loss.backward()
                    optimizer_tw.step()
                    scheduler_tw.step()
                if ((ep2 + 1) % 10 == 0) or (ep2 == extra_epochs - 1):
                    acc_epoch = 100.0 * correct / max(1, total)
                    print(f"phase2-epoch [{ep2 + 1}/{extra_epochs}] loss {sum(loss_list)/max(1,len(loss_list)):.4f} acc_train {acc_epoch:.4f}")
                # quick eval on test set for weighting-only phase
                with torch.no_grad():
                    affinity = test_feats @ self.cache_keys.t()
                    cache_logits = ((-1.0) * (beta - beta * affinity)).exp() @ self.cache_vals
                    clip_logits = self._tipaf_clip_logits(test_feats, training=False)
                    tip_logits = clip_logits + cache_logits * alpha
                    acc = compute_accuracy(tip_logits, test_labels)[0]
                    if acc > best_acc:
                        best_acc = float(acc)
                        if getattr(self, 'tipaf_weighting', 'none') == 'tw':
                            best_tw_state = self.template_weight_logits.detach().clone()
                        elif getattr(self, 'tipaf_weighting', 'none') == 'gp':
                            best_gp_state = {k: v.detach().clone() for k, v in self.gp_weighter.state_dict().items()}
            # Load best TW from both phases
            if getattr(self, 'tipaf_weighting', 'none') == 'tw' and best_tw_state is not None:
                with torch.no_grad():
                    self.template_weight_logits.data.copy_(best_tw_state.to(device=self.device, dtype=self.clip_model.dtype))
            elif getattr(self, 'tipaf_weighting', 'none') == 'gp' and best_gp_state is not None:
                self.gp_weighter.load_state_dict(best_gp_state)
        # search on val set
        val_feats, val_labels = _preextract_features(self.clip_model, self.val_loader, self.device)
        best_beta, best_alpha = self._search_hyperparams(val_feats.to(self.device), val_labels)
        # final test
        with torch.no_grad():
            test_feats, test_labels = _preextract_features(self.clip_model, self.test_loader, self.device)
            affinity = adapter(test_feats.to(self.device))
            cache_logits = ((-1.0) * (best_beta - best_beta * affinity)).exp() @ self.cache_vals
            clip_logits = self._tipaf_clip_logits(test_feats.to(self.device), training=False)
            tip_logits = clip_logits + cache_logits * best_alpha
            acc = compute_accuracy(tip_logits, test_labels.to(self.device))[0]
        # Store best hyperparameters for consistent reporting
        self._tipaf_best_beta = float(best_beta)
        self._tipaf_best_alpha = float(best_alpha)
        print("Evaluate on the *test* set")
        print("=> result")
        correct = (tip_logits.argmax(dim=1) == test_labels.to(self.device)).sum().item()
        print(f"* total: {len(test_labels):,}")
        print(f"* correct: {int(correct):,}")
        print(f"* accuracy: {float(acc):.1f}%")
        print(f"* error: {100 - float(acc):.1f}%")
        # ECE and AECE for log parity
        try:
            ece_val = compute_ece(tip_logits, test_labels.to(self.device))
            aece_val = compute_aece(tip_logits, test_labels.to(self.device))
            print(f"* ECE: {ece_val:.2f}%")
            print(f"* AECE: {aece_val:.2f}%")
        except Exception:
            pass
        # Print final weighting stats (mean/std over classes) if enabled
        try:
            if getattr(self, 'tipaf_weighting', 'none') == 'tw':
                with torch.no_grad():
                    wf = torch.softmax(self.template_weight_logits, dim=1)  # [K, M]
                    mean_vals = wf.mean(dim=0).tolist()
                    std_vals = wf.std(dim=0).tolist()
                print("[TW] end:   mean = [" + ", ".join(f"{v:.4f}" for v in mean_vals) + "]")
                print("[TW] end:    std = [" + ", ".join(f"{v:.4f}" for v in std_vals) + "]")
            elif getattr(self, 'tipaf_weighting', 'none') == 'gp':
                with torch.no_grad():
                    _ = self.gp_weighter.sample_prototypes(num_samples=1)
                    wf = self.gp_weighter.scores.squeeze(0)  # [K, M]
                    mean_vals = wf.mean(dim=0).tolist()
                    std_vals = wf.std(dim=0).tolist()
                print("[GP] end:   mean = [" + ", ".join(f"{v:.4f}" for v in mean_vals) + "]")
                print("[GP] end:    std = [" + ", ".join(f"{v:.4f}" for v in std_vals) + "]")
        except Exception:
            pass
        # Write final metrics JSON for consistency with other trainers
        try:
            metrics = self._compute_final_metrics_tipaf(adapter, beta=self._tipaf_best_beta, alpha=self._tipaf_best_alpha)
            self._write_run_summary_json(metrics, start_time=start_time)
        except Exception:
            pass
        print(f"Training completed in {time.time() - start_time:.2f} seconds")

    @torch.no_grad()
    def _compute_final_metrics_tipaf(self, adapter: nn.Linear, beta: Optional[float] = None, alpha: Optional[float] = None) -> dict:
        self.set_model_mode("eval")
        test_feats, test_labels = _preextract_features(self.clip_model, self.test_loader, self.device)
        affinity = adapter(test_feats.to(self.device))
        # Use tuned alpha/beta if provided; otherwise, fall back to defaults
        beta = float(beta) if beta is not None else float(self.config.adapter.tipaf_init_beta)
        alpha = float(alpha) if alpha is not None else float(self.config.adapter.tipaf_init_alpha)
        cache_logits = ((-1.0) * (beta - beta * affinity)).exp() @ self.cache_vals
        clip_logits = self._tipaf_clip_logits(test_feats.to(self.device), training=False)
        logits = clip_logits + cache_logits * alpha
        labels = test_labels.to(self.device)
        acc = compute_accuracy(logits, labels)[0]
        try:
            ece_val = compute_ece(logits, labels)
        except Exception:
            ece_val = float('nan')
        try:
            aece_val = compute_aece(logits, labels)
        except Exception:
            aece_val = float('nan')
        return {
            "top1_acc": float(acc),
            "ece": float(ece_val),
            "aece": float(aece_val),
        }

    def _current_clip_weights(self) -> torch.Tensor:
        """Return [D, K] classifier weights from either fixed CLIP or template-weighted prototypes."""
        if getattr(self, 'tipaf_weighting', 'none') == 'tw':
            logits = self.template_weight_logits
            w = torch.softmax(logits, dim=1)  # [K, M]
            prototypes = torch.einsum("km,kmd->kd", w, self.text_embeddings)  # [K, D]
            prototypes = prototypes / prototypes.norm(dim=-1, keepdim=True)
            return prototypes.t().to(self.device)
        if getattr(self, 'tipaf_weighting', 'none') == 'gp':
            # One sample prototypes
            prot = self.gp_weighter.sample_prototypes(num_samples=1)  # [1,K,D]
            prototypes = prot.squeeze(0)  # [K,D]
            prototypes = prototypes / prototypes.norm(dim=-1, keepdim=True)
            return prototypes.t().to(self.device)
        return self.clip_weights
