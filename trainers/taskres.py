import time
import datetime
import torch
import torch.nn as nn
from torch.nn import functional as F

from clip import clip
from utils.trainer import BaseTrainer, TextEncoder, load_clip, _get_templates, _get_clip_weights
from utils.metrics import compute_accuracy, AverageMeter
from utils.optimization import build_optimizer, build_lr_scheduler
from utils.trainer_registry import TRAINER_REGISTRY
from .gp_template_weigher import GaussianProcessTemplateWeighter
from .adapter import _get_template_weights

# Custom templates for different datasets (from TaskRes implementation)
CUSTOM_TEMPLATES = {
    "Caltech101": "a photo of a {}.",
    "OxfordPets": "a photo of a {}, a type of pet.",
    "OxfordFlowers": "a photo of a {}, a type of flower.",
    "FGVCAircraft": "a photo of a {}, a type of aircraft.",
    "DescribableTextures": "{} texture.",
    "EuroSAT": "a centered satellite photo of {}.",
    "StanfordCars": "a photo of a {}.",
    "Food101": "a photo of {}, a type of food.",
    "SUN397": "a photo of a {}.",
    "UCF101": "a photo of a person doing {}.",
    "ImageNet": "a photo of a {}.",
    "ImageNetSketch": "a photo of a {}.",
    "ImageNetV2": "a photo of a {}.",
    "ImageNetA": "a photo of a {}.",
    "ImageNetR": "a photo of a {}.",
}


class TaskResLearner(nn.Module):
    """Task Residual learner that adds learnable residual to frozen base text features."""

    def __init__(self, config, classnames, clip_model, base_text_features):
        super().__init__()
        self.alpha = getattr(config.adapter, 'taskres_residual_scale', 0.5)
        print(f">> TaskRes scale factor: {self.alpha}")
        self.register_buffer("base_text_features", base_text_features)
        self.text_feature_residuals = nn.Parameter(torch.zeros_like(base_text_features))

    def forward(self):
        # t' = t + α * x (where t is base_text_features, x is learnable residual)
        return self.base_text_features + self.alpha * self.text_feature_residuals


class CustomCLIP(nn.Module):
    def __init__(self, config, classnames, clip_model):
        super().__init__()
        self.config = config
        self.image_encoder = clip_model.visual
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype

        # Create text encoder for computing base text features
        text_encoder = TextEncoder(clip_model)

        # Get base text features (either regular or enhanced)
        base_text_features = self._get_base_text_features(config, classnames, clip_model, text_encoder)

        # Create TaskRes learner
        self.taskres_learner = TaskResLearner(config, classnames, clip_model, base_text_features)

        # Optional GP weighter (set and trained by Trainer when enabled)
        self.gp_weighter = None
        self.gp_num_mc_samples_train = int(getattr(config.adapter, 'gp_num_mc_samples_train', 1) or 1)
        self.gp_num_mc_samples_eval = int(getattr(config.adapter, 'gp_num_mc_samples_eval', 1) or 1)

    def _get_base_text_features(self, config, classnames, clip_model, text_encoder):
        """Compute base text features using templates."""
        device = next(text_encoder.parameters()).device

        templates = _get_templates(config)

        with torch.no_grad():
            text_embeddings = []
            for classname in classnames:
                texts = [template.format(classname) for template in templates]
                tokens = clip.tokenize(texts).to(device)
                embeddings = clip_model.token_embedding(tokens).type(clip_model.dtype)
                text_features = text_encoder(embeddings, tokens)
                text_embeddings.append(text_features)

            # Average over templates
            text_embeddings = torch.stack(text_embeddings, dim=0)  # [num_classes, num_templates, embed_dim]
            if len(templates) > 1:
                text_embeddings = text_embeddings.mean(dim=1)  # [num_classes, embed_dim]
            else:
                text_embeddings = text_embeddings.squeeze(1)  # [num_classes, embed_dim]

        return text_embeddings

    def forward(self, image: torch.Tensor):
        # Encode image
        image_features = self.image_encoder(image.type(self.dtype))
        image_features = F.normalize(image_features, p=2, dim=-1)

        # If GP is available and enabled, MC-average logits over prototype samples plus residuals
        use_gp = (self.gp_weighter is not None) and bool(getattr(self.config.adapter, 'use_gp', False))
        scale = self.logit_scale.exp()
        if use_gp:
            num_samples = self.gp_num_mc_samples_train if self.training else self.gp_num_mc_samples_eval
            num_samples = max(1, int(num_samples))
            prototypes = self.gp_weighter.sample_prototypes(num_samples=num_samples)  # [S,K,D]
            prototypes = prototypes.to(device=image_features.device, dtype=image_features.dtype)
            prototypes = prototypes / prototypes.norm(dim=-1, keepdim=True)
            # Add TaskRes residuals
            residuals = self.taskres_learner.alpha * self.taskres_learner.text_feature_residuals  # [K,D]
            text_s = prototypes + residuals.unsqueeze(0)  # [S,K,D]
            text_s = text_s / text_s.norm(dim=-1, keepdim=True)
            # logits: [B,S,K] -> mean over S
            logits = scale * torch.einsum('bd,skd->bsk', image_features, text_s)
            logits = logits.mean(dim=1)
        else:
            # Get text features from TaskRes learner
            text_features = self.taskres_learner()
            text_features = F.normalize(text_features, p=2, dim=-1)
            logits = scale * (image_features @ text_features.t())

        return logits


@TRAINER_REGISTRY.register("TaskRes")
class Trainer(BaseTrainer):
    def __init__(self, config, dataset_manager):
        super().__init__(config, dataset_manager)

    def build_model(self):
        config = self.config
        classnames = self.dm.dataset.classnames

        print(f"Loading CLIP (backbone: {config.model.backbone_name})")
        clip_model = load_clip(config, self.device)

        print("Building TaskRes adapter")
        self.model = CustomCLIP(config, classnames, clip_model)
        self.model.to(self.device)

        # Also keep a separate CLIP for zero-shot baseline computation
        self._clip_zs = load_clip(config, self.device)

        # Freeze everything except TaskRes learner parameters
        for name, param in self.model.named_parameters():
            if "taskres_learner" in name and "text_feature_residuals" in name:
                param.requires_grad = True
            else:
                param.requires_grad = False

        # Get parameters to optimize (only the residuals)
        taskres_params = [p for n, p in self.model.named_parameters() if p.requires_grad]
        if len(taskres_params) == 0:
            self.optim = None
            self.sched = None
        else:
            # Build optimizer using TaskRes-specific adapter settings; reuse global scheduler shape
            class _TmpOptim:
                pass
            tmp = _TmpOptim()
            tmp.name = getattr(config.adapter, 'taskres_optimizer', getattr(config.optim, 'name', 'adam'))
            tmp.lr = float(getattr(config.adapter, 'taskres_lr', getattr(config.optim, 'lr', 1e-3)))
            tmp.max_epoch = int(getattr(config.adapter, 'taskres_epochs', getattr(config.optim, 'max_epoch', 100)))
            tmp.lr_scheduler = getattr(config.optim, 'lr_scheduler', 'cosine')
            tmp.warmup_epoch = getattr(config.optim, 'warmup_epoch', 1)
            tmp.warmup_type = getattr(config.optim, 'warmup_type', 'constant')
            tmp.warmup_cons_lr = getattr(config.optim, 'warmup_cons_lr', 1e-5)
            tmp.weight_decay = float(getattr(config.optim, 'weight_decay', 0.0))
            tmp.momentum = getattr(config.optim, 'momentum', 0.9)
            tmp.betas = getattr(config.optim, 'betas', (0.9, 0.999))
            self.optim = build_optimizer(taskres_params, tmp)
            self.sched = build_lr_scheduler(self.optim, tmp)

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
        return {"loss": loss.item(), "acc_train": acc_train}

    def train(self):
        start_time = time.time()
        self.build_model()

        # Zero-shot CLIP accuracy (baseline)
        self.set_model_mode("eval")
        templates = _get_templates(self.config)
        zs_weights = _get_clip_weights(self.dm.dataset.classnames, self._clip_zs, templates)
        zs_acc = self._compute_zeroshot_accuracy(zs_weights)
        print("Zero-Shot accuracy on test: " + str(round(zs_acc, 2)))

        # Optional: GP-based template weighting before TaskRes
        use_gp = bool(getattr(self.config.adapter, "use_gp", False))
        if use_gp:
            try:
                # Pre-extract few-shot train features
                with torch.no_grad():
                    feats_list, labels_list = [], []
                    for batch in self.train_loader_x:
                        imgs = batch["img"].to(self.device)
                        lbs = batch["label"].to(self.device)
                        f = self._clip_zs.encode_image(imgs)
                        f = f / f.norm(dim=-1, keepdim=True)
                        feats_list.append(f.detach())
                        labels_list.append(lbs.detach())
                    tr_feats = torch.cat(feats_list, dim=0)
                    tr_labels = torch.cat(labels_list, dim=0)
                # Build per-class, per-template text embeddings [K, M, D] using same path as TaskRes base
                with torch.no_grad():
                    classnames = self.dm.dataset.classnames
                    text_encoder = TextEncoder(self._clip_zs)
                    E_list = []
                    for name in classnames:
                        texts = [t.format(name) for t in templates]
                        tokens = clip.tokenize(texts).to(self.device)
                        embeddings = self._clip_zs.token_embedding(tokens).type(self._clip_zs.dtype)
                        text_features = text_encoder(embeddings, tokens)  # [M, D]
                        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
                        E_list.append(text_features)
                    E = torch.stack(E_list, dim=0)  # [K, M, D]
                gp_weighter = GaussianProcessTemplateWeighter(text_embeddings=E, cfg=self.config).to(self.device)
                self.model.gp_weighter = gp_weighter
                # Warm start from per-template weights computed on few-shot features
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
                    print("[TaskRes][GP] Initialized from few-shot template weights.")
                except Exception as e_init:
                    print(f"[TaskRes][GP][WARN] initialization from weights failed ({e_init}); proceeding without warm start.")
                # Optimize ELBO: CE on few-shot logits + KL
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
                        print(f"[TaskRes GP] epoch {ep+1}/{epochs} loss={float(loss):.4f} CE={float(ce):.4f} KL={float(kl):.4f} acc={float(acc):.2f}")
                with torch.no_grad():
                    num_mc = int(getattr(self.config.adapter, "gp_num_mc_samples_eval", 100) or 1)
                    prot_samples = self.model.gp_weighter.sample_prototypes(num_samples=max(1, num_mc))  # [S,K,D]
                    prototypes = prot_samples.mean(dim=0)  # [K,D]
                    prototypes = prototypes / prototypes.norm(dim=-1, keepdim=True)
                    # Initialize TaskRes base text features from GP-mean prototypes
                    dst = self.model.taskres_learner.base_text_features
                    new_base = prototypes.to(device=dst.device, dtype=dst.dtype)
                    dst.copy_(new_base)
                print("[TaskRes] Using trained GP-based template weighter for prototypes.")
            except Exception as e:
                print(f"[TaskRes][WARN] GP weighting failed ({e}); continuing without GP.")
                self.model.gp_weighter = None
        # Else: optional non-GP template weighting before TaskRes (uses CONFIG.OPTIM)
        elif bool(getattr(self.config.adapter, "taskres_use_template_weight_training", False)):
            # Pre-extract few-shot train features
            with torch.no_grad():
                feats_list, labels_list = [], []
                for batch in self.train_loader_x:
                    imgs = batch["img"].to(self.device)
                    lbs = batch["label"].to(self.device)
                    f = self._clip_zs.encode_image(imgs)
                    f = f / f.norm(dim=-1, keepdim=True)
                    feats_list.append(f.detach())
                    labels_list.append(lbs.detach())
                tr_feats = torch.cat(feats_list, dim=0)
                tr_labels = torch.cat(labels_list, dim=0)
            # Build per-class, per-template text embeddings [K, M, D] using same path as TaskRes base
            with torch.no_grad():
                classnames = self.dm.dataset.classnames
                text_encoder = TextEncoder(self._clip_zs)
                E_list = []
                for name in classnames:
                    texts = [t.format(name) for t in templates]
                    tokens = clip.tokenize(texts).to(self.device)
                    embeddings = self._clip_zs.token_embedding(tokens).type(self._clip_zs.dtype)
                    text_features = text_encoder(embeddings, tokens)  # [M, D]
                    E_list.append(text_features)
                E = torch.stack(E_list, dim=0)  # [K, M, D]
            K = int(E.shape[0])
            M = int(E.shape[1]) if E.ndim >= 3 else 0
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
                    ce = F.cross_entropy(logits, tr_labels.to(self.device))
                    # L2 regularizer towards uniform weights
                    try:
                        lambda_tw = float(getattr(self.config.adapter, 'template_tw_l2_lambda', 0.0))
                    except Exception:
                        lambda_tw = 0.0
                    if (M > 0) and (lambda_tw > 0.0):
                        uniform = torch.full_like(weights, 1.0 / float(M))
                        reg = F.mse_loss(weights, uniform, reduction='mean')
                        loss = ce + lambda_tw * reg
                    else:
                        loss = ce
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    scheduler.step()
                    if (ep == 0) or ((ep + 1) % 10 == 0):
                        with torch.no_grad():
                            acc = compute_accuracy(logits, tr_labels.to(self.device))[0]
                        if (M > 0) and (lambda_tw > 0.0):
                            print(f"[TaskRes TW] epoch {ep+1}/{epochs} loss={float(loss):.4f} CE={float(ce):.4f} reg={float(lambda_tw*reg):.4f} acc={float(acc):.2f}")
                        else:
                            print(f"[TaskRes TW] epoch {ep+1}/{epochs} loss={float(loss):.4f} acc={float(acc):.2f}")
                with torch.no_grad():
                    weights = torch.softmax(tw_logits, dim=-1)
                    prototypes = torch.einsum("km,kmd->kd", weights, E)
                    prototypes = prototypes / prototypes.norm(dim=-1, keepdim=True)  # [K, D]
                    # Initialize TaskRes base text features from learned prototypes
                    dst = self.model.taskres_learner.base_text_features
                    new_base = prototypes.to(device=dst.device, dtype=dst.dtype)
                    dst.copy_(new_base)

        self.before_train()
        for self.epoch in range(self.start_epoch, self.max_epoch):
            self.before_epoch()
            self.run_epoch()
            self.after_epoch()

        self.after_train()
        print(f"Training completed in {time.time() - start_time:.2f} seconds")

        # Final metrics
        try:
            metrics = self._compute_final_metrics()
            self._write_run_summary_json(metrics, start_time=start_time)
        except Exception:
            pass

    def _compute_zeroshot_accuracy(self, zs_weights):
        """Compute zero-shot accuracy."""
        self._clip_zs = load_clip(self.config, self.device)
        feats_list, labels_list = [], []
        with torch.no_grad():
            for batch in self.test_loader:
                imgs = batch["img"].to(self.device)
                lbs = batch["label"].to(self.device)
                f = self._clip_zs.encode_image(imgs)
                f = f / f.norm(dim=-1, keepdim=True)
                feats_list.append(f)
                labels_list.append(lbs)
        features_test = torch.cat(feats_list, dim=0)
        labels_test = torch.cat(labels_list, dim=0)
        clip_logits = 100.0 * (features_test @ zs_weights)
        zs_acc = compute_accuracy(clip_logits, labels_test)[0]
        return zs_acc

    def run_epoch(self):
        losses = AverageMeter()
        batch_time = AverageMeter()
        data_time = AverageMeter()

        self.set_model_mode("train")
        # Keep CLIP encoders in eval mode
        self.model.image_encoder.eval()

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
                info += [f"eta {eta}"]
                print(" ".join(info))

            n_iter = self.epoch * self.num_batches + self.batch_idx
            self.write_scalar("train/loss", loss_summary['loss'], n_iter)
            self.write_scalar("train/lr", self.get_current_lr(), n_iter)
            end = time.time()
