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
        self.image_encoder = clip_model.visual
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype

        # Create text encoder for computing base text features
        text_encoder = TextEncoder(clip_model)

        # Get base text features (either regular or enhanced)
        base_text_features = self._get_base_text_features(config, classnames, clip_model, text_encoder)

        # Create TaskRes learner
        self.taskres_learner = TaskResLearner(config, classnames, clip_model, base_text_features)

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

        # Get text features from TaskRes learner
        text_features = self.taskres_learner()
        text_features = F.normalize(text_features, p=2, dim=-1)

        # Compute logits
        scale = self.logit_scale.exp()
        logits = scale * (image_features @ text_features.t())

        return logits


@TRAINER_REGISTRY.register("Adapter-TaskRes")
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
            self.optim = build_optimizer(taskres_params, config.optim)
            self.sched = build_lr_scheduler(self.optim, config.optim)

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
