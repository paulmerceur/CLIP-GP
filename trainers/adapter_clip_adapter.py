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

        # Template weighting options: "none", "tw", "gp"
        weighting_cfg = getattr(config.adapter, 'clip_adapter_weighting', None)
        if weighting_cfg is None or str(weighting_cfg).strip() == "":
            weighting_cfg = getattr(config.adapter, 'tipaf_weighting', 'none')
        self.weighting_type = str(weighting_cfg).lower()
        self.gp_num_mc_samples_train = int(getattr(config.adapter, 'gp_num_mc_samples_train', 1) or 1)
        self.gp_num_mc_samples_eval = int(getattr(config.adapter, 'gp_num_mc_samples_eval', 1) or 1)

        self.gp_weighter = None
        if self.weighting_type == 'tw':
            K = int(self.text_embeddings.shape[0])
            M = int(self.text_embeddings.shape[1])
            if M > 0:
                init_logits = torch.zeros((K, M), dtype=self.text_embeddings.dtype, device=self.text_embeddings.device)
                self.template_weight_logits = nn.Parameter(init_logits)
            else:
                self.template_weight_logits = nn.Parameter(torch.empty(0, 0, dtype=self.text_embeddings.dtype, device=self.text_embeddings.device))
        elif self.weighting_type == 'gp':
            self.gp_weighter = GaussianProcessTemplateWeighter(
                text_embeddings=self.text_embeddings.detach(),
                cfg=config,
            )
            # Keep GP in float32 for numerical stability; only move device
            self.gp_weighter.to(device=self.text_embeddings.device)
            with torch.no_grad():
                K = int(self.text_embeddings.shape[0])
                M = int(self.text_embeddings.shape[1])
                if M > 0:
                    uniform_w = torch.full((K, M), 1.0 / float(M), device=self.text_embeddings.device, dtype=self.text_embeddings.dtype)
                    self.gp_weighter.initialize_from_weights(uniform_w)

    def _apply_adapter(self, feats: torch.Tensor) -> torch.Tensor:
        ratio = float(self._blend_ratio.item())
        adapted = self.adapter(feats)
        return ratio * adapted + (1.0 - ratio) * feats

    def _classifier_from_weights(self) -> torch.Tensor:
        if self.weighting_type == 'tw' and hasattr(self, 'template_weight_logits'):
            logits = self.template_weight_logits
            if logits.numel() == 0:
                return self.clip_weights
            weights = torch.softmax(logits, dim=-1)  # [K, M]
            prototypes = torch.einsum('km,kmd->kd', weights, self.text_embeddings)  # [K, D]
            return prototypes.t()  # [D, K]
        return self.clip_weights

    def _compute_logits_from_embeddings(self, feats: torch.Tensor, training: bool) -> torch.Tensor:
        feats_norm = F.normalize(feats, p=2, dim=-1)
        scale = self.logit_scale.exp()
        if self.weighting_type == 'gp' and self.gp_weighter is not None:
            num_samples = self.gp_num_mc_samples_train if training else self.gp_num_mc_samples_eval
            num_samples = max(1, int(num_samples))
            prototypes = self.gp_weighter.sample_prototypes(num_samples=num_samples)
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


@TRAINER_REGISTRY.register("Adapter-CLIP-Adapter")
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

        weighting = getattr(self.model, 'weighting_type', 'none')
        allow_tw = weighting == 'tw'
        allow_gp = weighting == 'gp'

        for name, param in self.model.named_parameters():
            if 'adapter' in name:
                param.requires_grad = True
            elif allow_tw and 'template_weight_logits' in name:
                param.requires_grad = True
            elif allow_gp and 'gp_weighter' in name:
                param.requires_grad = True
            else:
                param.requires_grad = False

        base_params, tw_params, gp_params = [], [], []
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            if 'adapter' in name:
                base_params.append(param)
            elif 'template_weight_logits' in name:
                tw_params.append(param)
            elif 'gp_weighter' in name:
                gp_params.append(param)
            else:
                base_params.append(param)

        param_groups = []
        weight_decay = float(getattr(config.optim, 'weight_decay', 0.0))
        if base_params:
            param_groups.append({'params': base_params, 'lr': float(config.optim.lr), 'weight_decay': weight_decay})
        if tw_params:
            tw_lr = float(getattr(config.adapter, 'clip_adapter_tw_lr', config.optim.lr))
            param_groups.append({'params': tw_params, 'lr': tw_lr, 'weight_decay': weight_decay})
        if gp_params:
            gp_lr = float(getattr(config.adapter, 'gp_lr', config.optim.lr))
            param_groups.append({'params': gp_params, 'lr': gp_lr, 'weight_decay': weight_decay})

        if len(param_groups) == 0:
            self.optim = None
            self.sched = None
        else:
            if len(param_groups) == 1 and abs(param_groups[0]['lr'] - float(config.optim.lr)) < 1e-12:
                self.optim = build_optimizer(param_groups[0]['params'], config.optim)
            else:
                self.optim = build_optimizer_from_param_groups(param_groups, config.optim)
            self.sched = build_lr_scheduler(self.optim, config.optim)

        # Silence startup template weight summaries to reduce log noise

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


