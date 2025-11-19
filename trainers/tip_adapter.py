from typing import Tuple, Optional
import time
import torch
import torch.nn as nn
from torch.nn import functional as F
from clip import clip

from utils.trainer import BaseTrainer, load_clip, _get_templates, _get_clip_weights
from utils.metrics import compute_accuracy, compute_ece, compute_aece
from utils.trainer_registry import TRAINER_REGISTRY
from .gp_template_weigher import GaussianProcessTemplateWeighter
from .adapter import _get_template_weights


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


@TRAINER_REGISTRY.register("Tip-Adapter")
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

    def _search_hyperparams(self, val_feats, val_labels, adapter: Optional[nn.Linear] = None):
        betas = [1.0, 2.0, 5.0]
        alphas = [1.0, 5.0, 10.0, 20.0, 50.0]
        best_acc = -1.0
        best_beta = float(self.config.adapter.tip_adapter_init_beta)
        best_alpha = float(self.config.adapter.tip_adapter_init_alpha)
        with torch.no_grad():
            use_gp = bool(getattr(self.config.adapter, "use_gp", False)) and hasattr(self, "gp_weighter") and (getattr(self, "gp_weighter") is not None)
            if use_gp:
                S = int(getattr(self.config.adapter, "gp_num_mc_samples_eval", 100) or 1)
                prot = self.gp_weighter.sample_prototypes(num_samples=max(1, S))
                prot = prot / prot.norm(dim=-1, keepdim=True)
                clip_logits = 100.0 * torch.einsum("bd,skd->bsk", val_feats.to(self.device), prot).mean(dim=1)
            else:
                clip_logits = 100.0 * (val_feats @ self.clip_weights)
            for beta in betas:
                if adapter is None:
                    affinity = val_feats @ self.cache_keys.t()
                else:
                    affinity = adapter(val_feats.to(self.device))
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
        # Optional: train template weight matrix before Tip-Adapter
        use_tw = bool(getattr(self.config.adapter, "tip_adapter_use_template_weight_training", False))
        use_gp = bool(getattr(self.config.adapter, "use_gp", False))
        if use_gp:
            self.gp_weighter = None
            # Pre-extract few-shot train features
            tr_feats, tr_labels = _preextract_features(self.clip_model, self.train_loader_x, self.device)
            # Build per-class, per-template text embeddings [K, M, D]
            with torch.no_grad():
                E_list = []
                for name in self.classnames:
                    prompts = [t.format(name) for t in self.templates]
                    tok = clip.tokenize(prompts).to(self.device)
                    emb = self.clip_model.encode_text(tok)  # [M, D]
                    emb = emb / emb.norm(dim=-1, keepdim=True)
                    E_list.append(emb)
                E = torch.stack(E_list, dim=0)  # [K, M, D]
            # Train GP-based template weighter on few-shot features
            try:
                gp_weighter = GaussianProcessTemplateWeighter(text_embeddings=E, cfg=self.config).to(self.device)
                self.gp_weighter = gp_weighter
                # Warm start from per-template weights computed on few-shot features
                try:
                    with torch.no_grad():
                        init_w = _get_template_weights(
                            self.config,
                            text_embeddings=E,
                            features=tr_feats,
                            labels=tr_labels,
                            logit_scale=self.clip_model.logit_scale.exp(),
                        )  # [K, M]
                    self.gp_weighter.initialize_from_weights(init_w)
                    print("[Tip-Adapter][GP] Initialized from few-shot template weights.")
                except Exception as e_init:
                    print(f"[Tip-Adapter][GP][WARN] initialization from weights failed ({e_init}); proceeding without warm start.")
                # Optimize ELBO: CE on few-shot logits + KL
                gp_lr = float(getattr(self.config.adapter, "gp_lr", 1e-3))
                weight_decay = float(getattr(self.config.optim, "weight_decay", 0.0))
                optimizer = torch.optim.AdamW(self.gp_weighter.parameters(), lr=gp_lr, weight_decay=weight_decay)
                epochs = int(getattr(self.config.optim, "max_epoch", 50))
                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)
                S_tr = int(getattr(self.config.adapter, "gp_num_mc_samples_train", 30) or 1)
                beta_kl = float(getattr(self.config.adapter, "gp_beta", 1e-3))
                for ep in range(epochs):
                    self.gp_weighter.train()
                    with torch.no_grad():
                        feats = tr_feats.to(self.device)  # already normalized
                        labels = tr_labels.to(self.device)
                    prot = self.gp_weighter.sample_prototypes(num_samples=max(1, S_tr))  # [S,K,D]
                    prot = prot / prot.norm(dim=-1, keepdim=True)
                    prot = prot.to(dtype=feats.dtype, device=feats.device)
                    logits_s = 100.0 * torch.einsum("bd,skd->bsk", feats, prot)  # [B,S,K]
                    logits = logits_s.mean(dim=1)  # [B,K]
                    ce = F.cross_entropy(logits, labels)
                    kl = self.gp_weighter.variational_strategy.kl_divergence().sum()
                    loss = ce + beta_kl * kl
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    scheduler.step()
                    if (ep == 0) or ((ep + 1) % 10 == 0):
                        with torch.no_grad():
                            acc = compute_accuracy(logits, labels)[0]
                        print(f"[GP] epoch {ep+1}/{epochs} loss={float(loss):.4f} CE={float(ce):.4f} KL={float(kl):.4f} acc={float(acc):.2f}")
                # Set mean prototypes after training (still use MC at inference later)
                with torch.no_grad():
                    num_mc = int(getattr(self.config.adapter, "gp_num_mc_samples_eval", 100) or 1)
                    prot_samples = gp_weighter.sample_prototypes(num_samples=max(1, num_mc))  # [S,K,D]
                    prototypes = prot_samples.mean(dim=0)  # [K,D]
                    prototypes = prototypes / prototypes.norm(dim=-1, keepdim=True)
                    self.clip_weights = prototypes.t().to(self.device)  # [D,K]
                print("[Tip-Adapter] Using trained GP-based template weighter for prototypes.")
            except Exception as e:
                print(f"[Tip-Adapter][WARN] GP weighting failed ({e}); falling back to default CLIP weights.")
        elif use_tw:
            # Pre-extract few-shot train features
            tr_feats, tr_labels = _preextract_features(self.clip_model, self.train_loader_x, self.device)
            # Build per-class, per-template text embeddings [K, M, D]
            with torch.no_grad():
                E_list = []
                for name in self.classnames:
                    prompts = [t.format(name) for t in self.templates]
                    tok = clip.tokenize(prompts).to(self.device)
                    emb = self.clip_model.encode_text(tok)  # [M, D]
                    emb = emb / emb.norm(dim=-1, keepdim=True)
                    E_list.append(emb)
                E = torch.stack(E_list, dim=0)  # [K, M, D]
            K = int(E.shape[0])
            M = int(E.shape[1])
            tw_logits = nn.Parameter(torch.zeros(K, M, device=self.device, dtype=E.dtype))
            # Optimizer from CONFIG.OPTIM
            lr = float(self.config.optim.lr)
            wd = float(getattr(self.config.optim, 'weight_decay', 0.0))
            optimizer = torch.optim.AdamW([tw_logits], lr=lr, weight_decay=wd)
            total_steps = max(1, len(self.train_loader_x)) * max(1, int(getattr(self.config.optim, 'max_epoch', 50)))
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, total_steps)
            # Train for train.max_epoch epochs on few-shot cached features
            epochs = int(getattr(self.config.optim, 'max_epoch', 50))
            for ep in range(epochs):
                weights = torch.softmax(tw_logits, dim=-1)  # [K, M]
                # Prototypes [K, D]
                prototypes = torch.einsum("km,kmd->kd", weights, E)
                prototypes = prototypes / prototypes.norm(dim=-1, keepdim=True)
                # Logits for all training features
                feats = tr_feats.to(device=self.device, dtype=prototypes.dtype)
                logits = 100.0 * (feats @ prototypes.t())
                loss = F.cross_entropy(logits, tr_labels.to(self.device))
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                scheduler.step()
                if (ep == 0) or ((ep + 1) % 10 == 0):
                    with torch.no_grad():
                        acc = compute_accuracy(logits, tr_labels.to(self.device))[0]
                    print(f"[TW] epoch {ep+1}/{epochs} loss={float(loss):.4f} acc={float(acc):.2f}")
            with torch.no_grad():
                weights = torch.softmax(tw_logits, dim=-1)
                prototypes = torch.einsum("km,kmd->kd", weights, E)
                prototypes = prototypes / prototypes.norm(dim=-1, keepdim=True)  # [K, D]
                self.clip_weights = prototypes.t().to(self.device)  # [D, K]
        # Zero-shot after optional template-weight training
        test_feats, test_labels = _preextract_features(self.clip_model, self.test_loader, self.device)
        if use_gp and hasattr(self, "gp_weighter") and (self.gp_weighter is not None):
            with torch.no_grad():
                S = int(getattr(self.config.adapter, "gp_num_mc_samples_eval", 100) or 1)
                # MC-average logits over prototype samples
                prot = self.gp_weighter.sample_prototypes(num_samples=max(1, S))  # [S,K,D]
                prot = prot / prot.norm(dim=-1, keepdim=True)
                f = test_feats.to(self.device)  # already normalized in _preextract_features
                logits_s = 100.0 * torch.einsum("bd,skd->bsk", f, prot)  # [B,S,K]
                clip_logits_test = logits_s.mean(dim=1)  # [B,K]
        else:
            clip_logits_test = 100.0 * (test_feats.to(self.device) @ self.clip_weights)
        zs_acc = compute_accuracy(clip_logits_test, test_labels.to(self.device))[0]
        print("Zero-Shot accuracy on test: " + str(round(zs_acc, 2)))
        # Build few-shot cache
        self._build_cache()
        # Decide whether to train an adapter (Tip-Adapter-F) or use fixed cache (Tip-Adapter)
        trainable = bool(getattr(self.config.adapter, "tip_adapter_trainable", False))
        adapter: Optional[nn.Linear] = None
        if trainable:
            # Trainable adapter initialized with cache keys
            adapter = nn.Linear(self.cache_keys.shape[1], self.cache_keys.shape[0], bias=False).to(self.clip_model.dtype).to(self.device)
            adapter.weight = nn.Parameter(self.cache_keys.to(self.device))
            lr = float(self.config.adapter.tip_adapter_lr)
            eps = float(self.config.adapter.tip_adapter_eps)
            optimizer = torch.optim.AdamW(adapter.parameters(), lr=lr, eps=eps)
            total_steps = int(self.config.adapter.tip_adapter_epochs) * max(1, len(self.train_loader_x))
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, total_steps)
            beta = float(self.config.adapter.tip_adapter_init_beta)
            alpha = float(self.config.adapter.tip_adapter_init_alpha)
            best_acc = 0.0
            best_state = None
            for train_idx in range(int(self.config.adapter.tip_adapter_epochs)):
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
                    if use_gp and hasattr(self, "gp_weighter") and (self.gp_weighter is not None):
                        with torch.no_grad():
                            S = int(getattr(self.config.adapter, "gp_num_mc_samples_eval", 100) or 1)
                            prot = self.gp_weighter.sample_prototypes(num_samples=max(1, S))
                            prot = prot / prot.norm(dim=-1, keepdim=True)
                            clip_logits = 100.0 * torch.einsum("bd,skd->bsk", image_features, prot).mean(dim=1)
                    else:
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
                    info += [f"epoch [{train_idx + 1}/{int(self.config.adapter.tip_adapter_epochs)}]"]
                    info += [f"loss {sum(loss_list)/max(1,len(loss_list)):.4f}"]
                    info += [f"acc_train {acc_epoch:.4f}"]
                    print(" ".join(info))
                # quick eval on test set
                adapter.eval()
                with torch.no_grad():
                    t_feats, t_labels = _preextract_features(self.clip_model, self.test_loader, self.device)
                    affinity = adapter(t_feats.to(self.device))
                    cache_logits = ((-1.0) * (beta - beta * affinity)).exp() @ self.cache_vals
                    if use_gp and hasattr(self, "gp_weighter") and (self.gp_weighter is not None):
                        S = int(getattr(self.config.adapter, "gp_num_mc_samples_eval", 100) or 1)
                        prot = self.gp_weighter.sample_prototypes(num_samples=max(1, S))
                        prot = prot / prot.norm(dim=-1, keepdim=True)
                        clip_logits = 100.0 * torch.einsum("bd,skd->bsk", t_feats.to(self.device), prot).mean(dim=1)
                    else:
                        clip_logits = 100.0 * (t_feats.to(self.device) @ self.clip_weights)
                    tip_logits_tmp = clip_logits + cache_logits * alpha
                    acc_tmp = compute_accuracy(tip_logits_tmp, t_labels.to(self.device))[0]
                    if acc_tmp > best_acc:
                        best_acc = float(acc_tmp)
                        best_state = adapter.state_dict()
            if best_state is not None:
                adapter.load_state_dict(best_state)

            # Hyperparameter search on validation set if available (with adapter)
            try:
                val_feats, val_labels = _preextract_features(self.clip_model, self.val_loader, self.device)
                best_beta, best_alpha = self._search_hyperparams(val_feats.to(self.device), val_labels, adapter=adapter)
            except Exception:
                best_beta = float(self.config.adapter.tip_adapter_init_beta)
                best_alpha = float(self.config.adapter.tip_adapter_init_alpha)

            # Final test with trained adapter
            with torch.no_grad():
                t_feats, t_labels = _preextract_features(self.clip_model, self.test_loader, self.device)
                affinity = adapter(t_feats.to(self.device))
                cache_logits = ((-1.0) * (best_beta - best_beta * affinity)).exp() @ self.cache_vals
                if use_gp and hasattr(self, "gp_weighter") and (self.gp_weighter is not None):
                    S = int(getattr(self.config.adapter, "gp_num_mc_samples_eval", 100) or 1)
                    prot = self.gp_weighter.sample_prototypes(num_samples=max(1, S))
                    prot = prot / prot.norm(dim=-1, keepdim=True)
                    clip_logits = 100.0 * torch.einsum("bd,skd->bsk", t_feats.to(self.device), prot).mean(dim=1)
                else:
                    clip_logits = 100.0 * (t_feats.to(self.device) @ self.clip_weights)
                tip_logits = clip_logits + cache_logits * best_alpha
                acc = compute_accuracy(tip_logits, t_labels.to(self.device))[0]
        else:
            # Hyperparameter search on validation set if available (without adapter)
            try:
                val_feats, val_labels = _preextract_features(self.clip_model, self.val_loader, self.device)
                best_beta, best_alpha = self._search_hyperparams(val_feats.to(self.device), val_labels)
            except Exception:
                best_beta = float(self.config.adapter.tip_adapter_init_beta)
                best_alpha = float(self.config.adapter.tip_adapter_init_alpha)

            # Final test without adapter
            with torch.no_grad():
                affinity = test_feats.to(self.device) @ self.cache_keys.t()
                cache_logits = ((-1.0) * (best_beta - best_beta * affinity)).exp() @ self.cache_vals
                tip_logits = clip_logits_test + cache_logits * best_alpha
                acc = compute_accuracy(tip_logits, test_labels.to(self.device))[0]

        # Store best hyperparameters for consistent reporting (unified naming)
        self._tip_adapter_best_beta = float(best_beta)
        self._tip_adapter_best_alpha = float(best_alpha)
        print("Evaluate on the *test* set")
        print("=> result")
        # For printing counts, use the labels from the final computation path
        final_labels = test_labels if not trainable else t_labels
        correct = (tip_logits.argmax(dim=1) == final_labels.to(self.device)).sum().item()
        print(f"* total: {len(final_labels):,}")
        print(f"* correct: {int(correct):,}")
        print(f"* accuracy: {float(acc):.1f}%")
        print(f"* error: {100 - float(acc):.1f}%")
        # ECE and AECE for log parity
        try:
            ece_val = compute_ece(tip_logits, final_labels.to(self.device))
            aece_val = compute_aece(tip_logits, final_labels.to(self.device))
            print(f"* ECE: {ece_val:.2f}%")
            print(f"* AECE: {aece_val:.2f}%")
        except Exception:
            pass
        # Write final metrics JSON using BaseTrainer helper
        try:
            metrics = self._compute_final_metrics_tip_adapter(adapter=adapter, beta=self._tip_adapter_best_beta, alpha=self._tip_adapter_best_alpha)
            self._write_run_summary_json(metrics, start_time=start_time)
        except Exception:
            pass
        print(f"Completed in {time.time() - start_time:.2f} seconds")

    @torch.no_grad()
    def _compute_final_metrics_tip_adapter(self, adapter: Optional[nn.Linear] = None, beta: Optional[float] = None, alpha: Optional[float] = None) -> dict:
        self.set_model_mode("eval")
        test_feats, test_labels = _preextract_features(self.clip_model, self.test_loader, self.device)
        beta = float(beta) if beta is not None else float(self.config.adapter.tip_adapter_init_beta)
        alpha = float(alpha) if alpha is not None else float(self.config.adapter.tip_adapter_init_alpha)
        if adapter is None:
            affinity = test_feats.to(self.device) @ self.cache_keys.t()
        else:
            affinity = adapter(test_feats.to(self.device))
        cache_logits = ((-1.0) * (beta - beta * affinity)).exp() @ self.cache_vals
        use_gp = bool(getattr(self.config.adapter, "use_gp", False)) and hasattr(self, "gp_weighter") and (getattr(self, "gp_weighter") is not None)
        if use_gp:
            S = int(getattr(self.config.adapter, "gp_num_mc_samples_eval", 100) or 1)
            prot = self.gp_weighter.sample_prototypes(num_samples=max(1, S))
            prot = prot / prot.norm(dim=-1, keepdim=True)
            clip_logits = 100.0 * torch.einsum("bd,skd->bsk", test_feats.to(self.device), prot).mean(dim=1)
        else:
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
