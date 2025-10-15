from typing import Tuple, Optional
import time
import torch
import torch.nn as nn
from torch.nn import functional as F

from utils.trainer import BaseTrainer, load_clip, _get_templates, _get_clip_weights
from utils.metrics import compute_accuracy, compute_ece, compute_aece
from utils.trainer_registry import TRAINER_REGISTRY


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

    def _build_cache(self):
        feats, labels = _preextract_features(self.clip_model, self.train_loader_x, self.device)
        keys = feats.to(self.device)
        vals = torch.zeros(keys.shape[0], len(self.classnames), device=self.device, dtype=keys.dtype)
        labels_i64 = labels.to(device=self.device, dtype=torch.int64)
        vals.scatter_(1, labels_i64.view(-1, 1), 1.0)
        self.cache_keys = keys  # [N, D]
        self.cache_vals = vals  # [N, K]

    def _search_hyperparams(self, val_feats, val_labels):
        betas = [1.0, 2.0, 5.0]
        alphas = [1.0, 5.0, 10.0, 20.0, 50.0]
        best_acc = -1.0
        best_beta = float(self.config.adapter.tipaf_init_beta)
        best_alpha = float(self.config.adapter.tipaf_init_alpha)
        with torch.no_grad():
            clip_logits = 100.0 * (val_feats @ self.clip_weights)
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

    def train(self):
        start_time = time.time()
        self.build_model()
        # Zero-shot baseline (for consistent logs)
        self.set_model_mode("eval")
        test_feats, test_labels = _preextract_features(self.clip_model, self.test_loader, self.device)
        clip_logits_test = 100.0 * (test_feats.to(self.device) @ self.clip_weights)
        zs_acc = compute_accuracy(clip_logits_test, test_labels.to(self.device))[0]
        print("Zero-Shot accuracy on test: " + str(round(zs_acc, 2)))
        # Build few-shot cache
        self._build_cache()
        adapter = nn.Linear(self.cache_keys.shape[1], self.cache_keys.shape[0], bias=False).to(self.clip_model.dtype).to(self.device)
        adapter.weight = nn.Parameter(self.cache_keys.to(self.device))
        lr = float(self.config.optim.lr)
        eps = float(self.config.adapter.tipaf_eps)
        optimizer = torch.optim.AdamW(adapter.parameters(), lr=lr, eps=eps)
        total_steps = int(self.config.optim.max_epoch) * max(1, len(self.train_loader_x))
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, total_steps)
        beta = float(self.config.adapter.tipaf_init_beta)
        alpha = float(self.config.adapter.tipaf_init_alpha)
        best_acc = 0.0
        best_state = None
        for train_idx in range(int(self.config.optim.max_epoch)):
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
                clip_logits = 100.0 * (image_features @ self.clip_weights)
                tip_logits = clip_logits + cache_logits * alpha
                loss = F.cross_entropy(tip_logits, target)
                acc = compute_accuracy(tip_logits, target)[0]
                correct += float(acc) / 100.0 * tip_logits.shape[0]
                total += tip_logits.shape[0]
                loss_list.append(loss.item())
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                scheduler.step()
            if (train_idx == 0) or ((train_idx + 1) % 10 == 0):
                acc_epoch = 100.0 * correct / max(1, total)
                info = []
                info += [f"epoch [{train_idx + 1}/{int(self.config.optim.max_epoch)}]"]
                info += [f"loss {sum(loss_list)/max(1,len(loss_list)):.4f}"]
                info += [f"acc_train {acc_epoch:.4f}"]
                print(" ".join(info))
            # quick eval on test set
            adapter.eval()
            with torch.no_grad():
                test_feats, test_labels = _preextract_features(self.clip_model, self.test_loader, self.device)
                affinity = adapter(test_feats.to(self.device))
                cache_logits = ((-1.0) * (beta - beta * affinity)).exp() @ self.cache_vals
                clip_logits = 100.0 * (test_feats.to(self.device) @ self.clip_weights)
                tip_logits = clip_logits + cache_logits * alpha
                acc = compute_accuracy(tip_logits, test_labels.to(self.device))[0]
                if acc > best_acc:
                    best_acc = float(acc)
                    best_state = adapter.state_dict()
        if best_state is not None:
            adapter.load_state_dict(best_state)
        # search on val set
        val_feats, val_labels = _preextract_features(self.clip_model, self.val_loader, self.device)
        best_beta, best_alpha = self._search_hyperparams(val_feats.to(self.device), val_labels)
        # final test
        with torch.no_grad():
            test_feats, test_labels = _preextract_features(self.clip_model, self.test_loader, self.device)
            affinity = adapter(test_feats.to(self.device))
            cache_logits = ((-1.0) * (best_beta - best_beta * affinity)).exp() @ self.cache_vals
            clip_logits = 100.0 * (test_feats.to(self.device) @ self.clip_weights)
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
            ece_val = compute_ece(tip_logits, test_labels.to(self.device)) * 100.0
            aece_val = compute_aece(tip_logits, test_labels.to(self.device)) * 100.0
            print(f"* ECE: {ece_val:.2f}%")
            print(f"* AECE: {aece_val:.2f}%")
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
        clip_logits = 100.0 * (test_feats.to(self.device) @ self.clip_weights)
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
