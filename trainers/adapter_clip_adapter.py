import time
import datetime
import torch
import torch.nn as nn
from torch.nn import functional as F

from utils.trainer import BaseTrainer, load_clip, _get_templates, _get_clip_weights
from utils.metrics import compute_accuracy, AverageMeter
from utils.optimization import build_optimizer, build_lr_scheduler
from utils.trainer_registry import TRAINER_REGISTRY


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
        # The visual width equals projection dim used by CLIP features
        try:
            in_dim = int(getattr(clip_model.visual, 'output_dim', clip_model.ln_final.weight.shape[0]))
        except Exception:
            in_dim = int(getattr(clip_model, 'embed_dim', 1024))

        reduction = int(getattr(config.adapter, 'clip_adapter_reduction', 4))
        self.adapter = AdapterMLP(in_dim=in_dim, reduction=reduction, dtype=self.dtype)

        # Blending ratio between adapted and original features
        self.register_buffer('_blend_ratio', torch.tensor(float(getattr(config.adapter, 'clip_adapter_ratio', 0.2))))

        # Text side: fixed prompts generated from generic templates
        self.templates = _get_templates(config)
        self.classnames = classnames
        # Precompute fixed text weights (zero-shot) once and reuse
        with torch.no_grad():
            tw = _get_clip_weights(self.classnames, clip_model, self.templates)  # [D, K]
        self.register_buffer('text_weights', tw)

    def forward(self, image: torch.Tensor):
        # Visual encode in CLIP dtype
        feats = self.image_encoder(image.type(self.dtype))
        # Adapter forward
        adapted = self.adapter(feats)
        ratio = float(self._blend_ratio.item())
        feats = ratio * adapted + (1.0 - ratio) * feats
        # Use cached text weights
        text_feats = self.text_weights
        feats = F.normalize(feats, p=2, dim=-1)
        text_feats = F.normalize(text_feats, p=2, dim=-1)
        scale = self.logit_scale.exp()
        logits = scale * (feats @ text_feats)
        return logits


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
        # Freeze all except adapter
        for name, param in self.model.named_parameters():
            if 'adapter' in name:
                param.requires_grad = True
            else:
                param.requires_grad = False
        trainable = [p for p in self.model.parameters() if p.requires_grad]
        if len(trainable) == 0:
            self.optim = None
            self.sched = None
        else:
            self.optim = build_optimizer(trainable, config.optim)
            self.sched = build_lr_scheduler(self.optim, config.optim)

        # Pre-extract test features with frozen visual encoder for faster eval like other trainers
        self._preextract_test_features()

    @torch.no_grad()
    def _preextract_test_features(self):
        feats_list, labels_list = [], []
        self.model.eval()
        for batch in self.test_loader:
            imgs = batch["img"].to(self.device)
            lbs = batch["label"].to(self.device)
            f = self.model.image_encoder(imgs.type(self.model.dtype))
            f = F.normalize(f, p=2, dim=-1)
            feats_list.append(f)
            labels_list.append(lbs)
        self.features_test = torch.cat(feats_list, dim=0)
        self.labels_test = torch.cat(labels_list, dim=0)
        # Zero-shot baseline for consistent logs
        zs_weights = self.model.text_weights
        clip_logits = 100.0 * (self.features_test @ zs_weights)
        zs_acc = compute_accuracy(clip_logits, self.labels_test)[0]
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
            feats = self.features_test
            # Reuse cached text weights
            text_feats = self.model.text_weights
            feats = feats.to(self.device)
            scale = self.model.logit_scale.exp()
            logits_test = scale * (feats @ F.normalize(text_feats, p=2, dim=-1))
            acc_test = compute_accuracy(logits_test, self.labels_test.to(self.device))[0]
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


