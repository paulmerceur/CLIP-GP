import time
import datetime
import torch
import torch.nn as nn
from torch.nn import functional as F

from utils.trainer import BaseTrainer, load_clip, _get_templates, _get_clip_weights
from utils.metrics import compute_accuracy, AverageMeter
from utils.optimization import build_optimizer, build_lr_scheduler, build_optimizer_from_param_groups
from utils.trainer_registry import TRAINER_REGISTRY
from .gp_template_weigher import GaussianProcessTemplateWeighter
from .adapter import _get_template_weights
from clip import clip


class AdapterMLP(nn.Module):
    def __init__(self, in_dim: int, reduction: int = 4, dtype: torch.dtype | None = None):
        super().__init__()
        hidden = max(1, in_dim // max(1, int(reduction)))
        self.fc1 = nn.Linear(in_dim, hidden, bias=False)
        self.act1 = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(hidden, in_dim, bias=False)
        self.act2 = nn.ReLU(inplace=True)
        if dtype is not None:
            self.to(dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act1(x)
        x = self.fc2(x)
        x = self.act2(x)
        return x


class CustomCLIP(nn.Module):
    def __init__(self, config, classnames, clip_model):
        super().__init__()
        self.config = config
        self.image_encoder = clip_model.visual
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype

        # Build adapter: 2-layer MLP as in CLIP-Adapter
        try:
            in_dim = int(getattr(clip_model.visual, 'output_dim', clip_model.ln_final.weight.shape[0]))
        except Exception:
            in_dim = int(getattr(clip_model, 'embed_dim', 1024))

        reduction = int(getattr(config.adapter, 'clip_adapter_reduction', 4))
        self.adapter = AdapterMLP(in_dim=in_dim, reduction=reduction, dtype=self.dtype)

        # Blending ratio between adapted and original features
        self.register_buffer('_blend_ratio', torch.tensor(float(getattr(config.adapter, 'clip_adapter_ratio', 0.2))))

        # Text side: prompts/templates and per-template embeddings
        self.templates = _get_templates(config)
        self.classnames = classnames
        device_ref = next(self.image_encoder.parameters()).device
        with torch.no_grad():
            emb_list = []
            for name in self.classnames:
                tokens = clip.tokenize([t.format(name) for t in self.templates]).to(device_ref)
                emb = clip_model.encode_text(tokens)
                emb_list.append(emb)
            text_embeds = torch.stack(emb_list)  # [K, M, D]
            text_embeds = text_embeds.to(dtype=self.dtype)
            clip_w = _get_clip_weights(self.classnames, clip_model, self.templates)  # [D, K]
            clip_w = clip_w.to(device=device_ref, dtype=self.dtype)
        self.register_buffer('text_embeddings', text_embeds.detach())
        self.register_buffer('clip_weights', clip_w.detach())

        # Optional GP weighter (set and trained by Trainer when enabled)
        self.gp_weighter = None
        self.gp_num_mc_samples_train = int(getattr(config.adapter, 'gp_num_mc_samples_train', 1) or 1)
        self.gp_num_mc_samples_eval = int(getattr(config.adapter, 'gp_num_mc_samples_eval', 1) or 1)

    def _apply_adapter(self, feats: torch.Tensor) -> torch.Tensor:
        ratio = float(self._blend_ratio.item())
        adapted = self.adapter(feats)
        return ratio * adapted + (1.0 - ratio) * feats

    def _classifier_from_weights(self) -> torch.Tensor:
        return self.clip_weights

    def _compute_logits_from_embeddings(self, feats: torch.Tensor, training: bool) -> torch.Tensor:
        feats_norm = F.normalize(feats, p=2, dim=-1)
        scale = self.logit_scale.exp()
        # If GP is enabled and present, average logits over MC prototype samples
        if (self.gp_weighter is not None) and bool(getattr(self.config.adapter, 'use_gp', False)):
            num_samples = self.gp_num_mc_samples_train if training else self.gp_num_mc_samples_eval
            num_samples = max(1, int(num_samples))
            prototypes = self.gp_weighter.sample_prototypes(num_samples=num_samples)  # [S,K,D]
            prototypes = prototypes.to(device=feats_norm.device, dtype=feats_norm.dtype)
            prototypes = prototypes / prototypes.norm(dim=-1, keepdim=True)
            logits = scale * torch.einsum('bd,skd->bsk', feats_norm, prototypes)
            return logits.mean(dim=1)
        classifier = self._classifier_from_weights()
        classifier = classifier.to(device=feats_norm.device, dtype=feats_norm.dtype)
        classifier = F.normalize(classifier, p=2, dim=0)
        return scale * (feats_norm @ classifier)

    def logits_from_features(self, features: torch.Tensor, training: bool = False) -> torch.Tensor:
        device = self.adapter.fc1.weight.device
        feats = features.to(device=device, dtype=self.dtype)
        feats = self._apply_adapter(feats)
        return self._compute_logits_from_embeddings(feats, training=training)

    def forward(self, image: torch.Tensor):
        feats = self.image_encoder(image.type(self.dtype))
        feats = self._apply_adapter(feats)
        return self._compute_logits_from_embeddings(feats, training=self.training)


@TRAINER_REGISTRY.register("CLIP-Adapter")
class Trainer(BaseTrainer):
    def __init__(self, config, dataset_manager):
        super().__init__(config, dataset_manager)

    def build_model(self):
        config = self.config
        classnames = self.dm.dataset.classnames
        print(f"Loading CLIP (backbone: {config.model.backbone_name})")
        clip_model = load_clip(config, self.device)
        clip_model.eval()
        print("Building CLIP-Adapter")
        self.model = CustomCLIP(config, classnames, clip_model)
        self.model.to(self.device)

        for name, param in self.model.named_parameters():
            if 'adapter' in name:
                param.requires_grad = True
            else:
                param.requires_grad = False

        base_params = []
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            if 'adapter' in name:
                base_params.append(param)

        param_groups = []
        weight_decay = float(getattr(config.optim, 'weight_decay', 0.0))
        adapter_lr = float(getattr(config.adapter, 'clip_adapter_lr', getattr(config.optim, 'lr', 1e-3)))
        if base_params:
            param_groups.append({'params': base_params, 'lr': adapter_lr, 'weight_decay': weight_decay})

        if len(param_groups) == 0:
            self.optim = None
            self.sched = None
        else:
            # Build optimizer using adapter-specific settings; reuse global scheduler shape
            class _TmpOptim:
                pass
            tmp = _TmpOptim()
            tmp.name = getattr(config.adapter, 'clip_adapter_optimizer', getattr(config.optim, 'name', 'adam'))
            tmp.lr = adapter_lr
            tmp.max_epoch = int(getattr(config.adapter, 'clip_adapter_epochs', getattr(config.optim, 'max_epoch', 100)))
            tmp.lr_scheduler = getattr(config.optim, 'lr_scheduler', 'cosine')
            tmp.warmup_epoch = getattr(config.optim, 'warmup_epoch', 1)
            tmp.warmup_type = getattr(config.optim, 'warmup_type', 'constant')
            tmp.warmup_cons_lr = getattr(config.optim, 'warmup_cons_lr', 1e-5)
            tmp.weight_decay = weight_decay
            tmp.momentum = getattr(config.optim, 'momentum', 0.9)
            tmp.betas = getattr(config.optim, 'betas', (0.9, 0.999))
            self.optim = build_optimizer_from_param_groups(param_groups, tmp)  # type: ignore
            self.sched = build_lr_scheduler(self.optim, tmp)  # type: ignore

        # Pre-extract test features with frozen visual encoder for faster eval like other trainers
        self._preextract_test_features()

    @torch.no_grad()
    def _preextract_test_features(self):
        feats_list, labels_list = [], []
        self.model.eval()
        for batch in self.test_loader:
            imgs = batch["img"].to(self.device)
            lbs = batch["label"].to(self.device)
            f = self.model.image_encoder(imgs.type(self.model.dtype)).detach()
            feats_list.append(f)
            labels_list.append(lbs.detach())
        self.features_test = torch.cat(feats_list, dim=0)
        self.labels_test = torch.cat(labels_list, dim=0)
        # Zero-shot baseline for consistent logs
        logits = self.model.logits_from_features(self.features_test, training=False)
        zs_acc = compute_accuracy(logits, self.labels_test)[0]
        print("Zero-Shot accuracy on test: " + str(round(zs_acc, 2)))

    @torch.no_grad()
    def _preextract_train_features(self):
        feats_list, labels_list = [], []
        self.model.eval()
        for batch in self.train_loader_x:
            imgs = batch["img"].to(self.device)
            lbs = batch["label"].to(self.device)
            f = self.model.image_encoder(imgs.type(self.model.dtype)).detach()
            f = f / f.norm(dim=-1, keepdim=True)
            feats_list.append(f)
            labels_list.append(lbs.detach())
        feats = torch.cat(feats_list, dim=0)
        labels = torch.cat(labels_list, dim=0)
        return feats, labels

    def _backward_and_update(self, loss):
        if getattr(self, 'optim', None) is None:
            return
        self.optim.zero_grad()
        loss.backward()
        self.optim.step()

    def parse_batch_train(self, batch):
        input_data = batch["img"]
        labels = batch["label"]
        input_data = input_data.to(self.device)
        labels = labels.to(self.device)
        return input_data, labels

    def forward_backward(self, batch):
        image, label = self.parse_batch_train(batch)
        output = self.model(image)
        loss = F.cross_entropy(output, label)
        self._backward_and_update(loss)
        acc_train = compute_accuracy(output, label)[0]
        # Quick eval on pre-extracted test features with current adapter
        with torch.no_grad():
            logits_test = self.model.logits_from_features(self.features_test, training=False)
            acc_test = compute_accuracy(logits_test, self.labels_test)[0]
        return {"loss": loss.item(), "acc_train": acc_train, "acc_test": acc_test}

    def train(self):
        start_time = time.time()
        self.build_model()
        # Optional: train GP template weighter on few-shot features (before adapter training)
        use_gp = bool(getattr(self.config.adapter, "use_gp", False))
        if use_gp:
            try:
                tr_feats, tr_labels = self._preextract_train_features()
                with torch.no_grad():
                    E = self.model.text_embeddings.detach()  # [K, M, D]
                gp_weighter = GaussianProcessTemplateWeighter(text_embeddings=E, cfg=self.config).to(self.device)
                self.model.gp_weighter = gp_weighter
                # Warm start from few-shot template weights
                try:
                    with torch.no_grad():
                        init_w = _get_template_weights(
                            self.config,
                            text_embeddings=E,
                            features=tr_feats,
                            labels=tr_labels,
                            logit_scale=self.model.logit_scale.exp(),
                        )  # [K, M]
                    self.model.gp_weighter.initialize_from_weights(init_w)
                    print("[CLIP-Adapter][GP] Initialized from few-shot template weights.")
                except Exception as e_init:
                    print(f"[CLIP-Adapter][GP][WARN] initialization from weights failed ({e_init}); proceeding without warm start.")
                # Train with CE + KL
                gp_lr = float(getattr(self.config.adapter, "gp_lr", 1e-3))
                weight_decay = float(getattr(self.config.optim, "weight_decay", 0.0))
                optimizer = torch.optim.AdamW(self.model.gp_weighter.parameters(), lr=gp_lr, weight_decay=weight_decay)
                epochs = int(getattr(self.config.optim, "max_epoch", 50))
                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)
                S_tr = int(getattr(self.config.adapter, "gp_num_mc_samples_train", 30) or 1)
                beta_kl = float(getattr(self.config.adapter, "gp_beta", 1e-3))
                for ep in range(epochs):
                    self.model.gp_weighter.train()
                    feats = tr_feats.to(self.device)
                    labels = tr_labels.to(self.device)
                    prot = self.model.gp_weighter.sample_prototypes(num_samples=max(1, S_tr))  # [S,K,D]
                    prot = prot / prot.norm(dim=-1, keepdim=True)
                    prot = prot.to(dtype=feats.dtype, device=feats.device)
                    logits_s = 100.0 * torch.einsum("bd,skd->bsk", feats, prot)  # [B,S,K]
                    logits = logits_s.mean(dim=1)  # [B,K]
                    ce = F.cross_entropy(logits, labels)
                    kl = self.model.gp_weighter.variational_strategy.kl_divergence().sum()
                    loss = ce + beta_kl * kl
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    scheduler.step()
                    if (ep == 0) or ((ep + 1) % 10 == 0):
                        with torch.no_grad():
                            acc = compute_accuracy(logits, labels)[0]
                        print(f"[CLIP-Adapter GP] epoch {ep+1}/{epochs} loss={float(loss):.4f} CE={float(ce):.4f} KL={float(kl):.4f} acc={float(acc):.2f}")
                with torch.no_grad():
                    num_mc = int(getattr(self.config.adapter, "gp_num_mc_samples_eval", 100) or 1)
                    prot_samples = self.model.gp_weighter.sample_prototypes(num_samples=max(1, num_mc))  # [S,K,D]
                    prototypes = prot_samples.mean(dim=0)  # [K,D]
                    prototypes = prototypes / prototypes.norm(dim=-1, keepdim=True)
                    new_w = prototypes.t().to(device=self.model.clip_weights.device, dtype=self.model.clip_weights.dtype)
                    self.model.clip_weights.copy_(new_w)
                print("[CLIP-Adapter] Using trained GP-based template weighter for prototypes.")
            except Exception as e:
                print(f"[CLIP-Adapter][WARN] GP weighting failed ({e}); continuing without GP.")
                self.model.gp_weighter = None
        # Optional: train template weight matrix before CLIP-Adapter (uses global OPTIM.*)
        use_tw = bool(getattr(self.config.adapter, "clip_adapter_use_template_weight_training", False))
        if use_tw:
            tr_feats, tr_labels = self._preextract_train_features()
            with torch.no_grad():
                E = self.model.text_embeddings.detach()  # [K, M, D]
            K = int(E.shape[0])
            M = int(E.shape[1])
            if M > 0:
                tw_logits = nn.Parameter(torch.zeros(K, M, device=self.device, dtype=E.dtype))
                lr = float(self.config.optim.lr)
                wd = float(getattr(self.config.optim, 'weight_decay', 0.0))
                optimizer = torch.optim.AdamW([tw_logits], lr=lr, weight_decay=wd)
                total_steps = max(1, len(self.train_loader_x)) * max(1, int(getattr(self.config.optim, 'max_epoch', 50)))
                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, total_steps)
                epochs = int(getattr(self.config.optim, 'max_epoch', 50))
                for ep in range(epochs):
                    weights = torch.softmax(tw_logits, dim=-1)  # [K, M]
                    # Prototypes [K, D]
                    prototypes = torch.einsum("km,kmd->kd", weights, E)
                    prototypes = prototypes / prototypes.norm(dim=-1, keepdim=True)
                    # Logits and loss on few-shot train features
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
                        print(f"[CLIP-Adapter TW] epoch {ep+1}/{epochs} loss={float(loss):.4f} acc={float(acc):.2f}")
                with torch.no_grad():
                    weights = torch.softmax(tw_logits, dim=-1)
                    prototypes = torch.einsum("km,kmd->kd", weights, E)
                    prototypes = prototypes / prototypes.norm(dim=-1, keepdim=True)  # [K, D]
                    new_w = prototypes.t().to(device=self.model.clip_weights.device, dtype=self.model.clip_weights.dtype)
                    self.model.clip_weights.copy_(new_w)
        self.before_train()
        for self.epoch in range(self.start_epoch, self.max_epoch):
            self.before_epoch()
            self.run_epoch()
            self.after_epoch()
        self.after_train()
        # Silence end-of-training template weight summaries
        # Final metrics json for consistency
        try:
            metrics = self._compute_final_metrics()
            self._write_run_summary_json(metrics, start_time=start_time)
        except Exception:
            pass
        print(f"Training completed in {time.time() - start_time:.2f} seconds")

    def run_epoch(self):
        losses = AverageMeter()
        batch_time = AverageMeter()
        data_time = AverageMeter()
        self.set_model_mode("train")
        try:
            if hasattr(self.model, "image_encoder"):
                self.model.image_encoder.eval()
        except AttributeError:
            pass
        self.num_batches = len(self.train_loader_x)
        end = time.time()
        for self.batch_idx, batch in enumerate(self.train_loader_x):
            data_time.update(time.time() - end)
            loss_summary = self.forward_backward(batch)
            batch_time.update(time.time() - end)
            losses.update(loss_summary['loss'])
            if (self.epoch == 0) or ((self.epoch + 1) % 10 == 0):
                nb_remain = 0
                nb_remain += self.num_batches - self.batch_idx - 1
                nb_remain += (self.max_epoch - self.epoch - 1) * self.num_batches
                eta_seconds = batch_time.avg * nb_remain
                eta = str(datetime.timedelta(seconds=int(eta_seconds)))
                info = []
                info += [f"epoch [{self.epoch + 1}/{self.max_epoch}]"]
                info += [f"batch [{self.batch_idx + 1}/{self.num_batches}]"]
                info += [f"loss {loss_summary['loss']:.4f}"]
                info += [f"acc_train {loss_summary['acc_train']:.4f}"]
                info += [f"acc_test {loss_summary['acc_test']:.4f}"]
                info += [f"eta {eta}"]
                print(" ".join(info))
            n_iter = self.epoch * self.num_batches + self.batch_idx
            self.write_scalar("train/loss", loss_summary['loss'], n_iter)
            self.write_scalar("train/lr", self.get_current_lr(), n_iter)
            end = time.time()
        return losses.avg if hasattr(losses, 'avg') else 0.0


