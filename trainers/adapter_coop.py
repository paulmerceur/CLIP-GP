from typing import cast
import time
import datetime
import torch
import torch.nn as nn
from torch.nn import functional as F

from utils.trainer import BaseTrainer
from utils.metrics import compute_accuracy, AverageMeter
from utils.optimization import build_optimizer, build_lr_scheduler
from utils.trainer_registry import TRAINER_REGISTRY
from utils.config import _config_to_dict
from utils.metrics import compute_ece, compute_aece
import json
import datetime
from pathlib import Path

from clip import clip


def _get_templates(config):
    templates = ["a photo of a {}."]
    from datasets.imagenet_templates import IMAGENET_TEMPLATES_SELECT, IMAGENET_TEMPLATES
    if config.adapter.num_templates > 1:
        templates += IMAGENET_TEMPLATES_SELECT[:config.adapter.num_templates - 1]
    if config.adapter.num_templates > 1 + len(IMAGENET_TEMPLATES_SELECT):
        templates += IMAGENET_TEMPLATES[:config.adapter.num_templates - 1 - len(IMAGENET_TEMPLATES_SELECT)]
    return templates


@torch.no_grad()
def _get_clip_weights(classnames, clip_model, templates):
    device = next(clip_model.parameters()).device
    zeroshot_weights = []
    for classname in classnames:
        texts = [t.format(classname) for t in templates]
        texts = clip.tokenize(texts).to(device)
        class_embeddings = clip_model.encode_text(texts)
        class_embeddings = class_embeddings / class_embeddings.norm(dim=-1, keepdim=True)
        class_embedding = class_embeddings.mean(dim=0)
        class_embedding = class_embedding / class_embedding.norm()
        zeroshot_weights.append(class_embedding)
    zeroshot_weights = torch.stack(zeroshot_weights, dim=1)
    return zeroshot_weights


class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    def forward(self, prompts, tokenized_prompts):
        x = prompts + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)
        x = self.transformer(x)
        x = x.permute(1, 0, 2)
        x = self.ln_final(x).type(self.dtype)
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection
        return x


def load_clip(config, device):
    backbone_name = config.model.backbone_name
    url = clip._MODELS[backbone_name]
    model_path = clip._download(url)
    try:
        jit_model = torch.jit.load(model_path).eval()
        state_dict = jit_model.state_dict()
    except RuntimeError:
        state_dict = torch.load(model_path)
    model = clip.build_model(state_dict)
    return model.to(device, dtype=torch.float32)


class PromptLearnerCoOp(nn.Module):
    def __init__(self, config, classnames, clip_model):
        super().__init__()
        device = next(clip_model.parameters()).device
        dtype = next(clip_model.parameters()).dtype
        n_ctx = int(getattr(config.adapter, 'n_ctx', 16))
        ctx_init = str(getattr(config.adapter, 'ctx_init', '') or '')
        ctx_dim = int(clip_model.ln_final.weight.shape[0])
        if len(ctx_init) > 0:
            ctx_init_clean = ctx_init.replace('_', ' ').strip()
            # Follow reference: set n_ctx by number of words in init phrase
            n_ctx = len(ctx_init_clean.split(' '))
            tok_init = clip.tokenize(ctx_init_clean).to(device)
            with torch.no_grad():
                embedding = clip_model.token_embedding(tok_init).type(dtype).squeeze(0)
            ctx_vectors = embedding[1:1 + n_ctx, :].detach().clone()
            prompt_prefix = ctx_init_clean
        else:
            ctx_vectors = torch.empty(n_ctx, ctx_dim, device=device, dtype=dtype)
            nn.init.normal_(ctx_vectors, std=0.02)
            prompt_prefix = ' '.join(['X'] * n_ctx)
        classnames = [name for name in classnames]
        prompts = [f"{prompt_prefix} {name}." for name in classnames]
        tokenized_prompts = clip.tokenize(prompts).to(device)
        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts).type(dtype)
        self.ctx = nn.Parameter(ctx_vectors)
        self.register_buffer('token_prefix', embedding[:, :1, :])
        self.register_buffer('token_suffix', embedding[:, 1 + n_ctx:, :])
        self.register_buffer('tokenized_prompts', tokenized_prompts)
        self.n_ctx = n_ctx
        self.num_classes = len(classnames)
        self._token_embedding = clip_model.token_embedding

    def build_prompts(self) -> tuple[torch.Tensor, torch.Tensor]:
        K = int(self.num_classes)
        ctx = self.ctx.unsqueeze(0).expand(K, -1, -1)
        prompts = torch.cat([self.token_prefix, ctx, self.token_suffix], dim=1)
        return prompts, self.tokenized_prompts


class CustomCLIP(nn.Module):
    def __init__(self, config, classnames, clip_model):
        super().__init__()
        self.image_encoder = clip_model.visual
        self.logit_scale = clip_model.logit_scale
        self.text_encoder = TextEncoder(clip_model)
        self.prompt_learner = PromptLearnerCoOp(config, classnames, clip_model)
        self.dtype = clip_model.dtype

    def forward(self, image: torch.Tensor):
        image_features = self.image_encoder(image.type(self.dtype))
        prompts, tokenized = self.prompt_learner.build_prompts()
        text_feats = self.text_encoder(prompts, tokenized)
        image_features = F.normalize(image_features, p=2, dim=-1)
        text_feats = F.normalize(text_feats, p=2, dim=-1)
        scale = self.logit_scale.exp()
        logits = scale * (image_features @ text_feats.t())
        return logits


@TRAINER_REGISTRY.register("AdapterCoOp")
class Trainer(BaseTrainer):
    def __init__(self, config, dataset_manager):
        super().__init__(config, dataset_manager)

    def build_model(self):
        config = self.config
        classnames = self.dm.dataset.classnames
        print(f"Loading CLIP (backbone: {config.model.backbone_name})")
        clip_model = load_clip(config, self.device)
        print("Building CoOp adapter")
        self.model = CustomCLIP(config, classnames, clip_model)
        self.model.to(self.device)
        # Also keep a separate CLIP for zero-shot baseline computation
        self._clip_zs = load_clip(config, self.device)
        for name, param in self.model.named_parameters():
            if "prompt_learner" in name:
                param.requires_grad = True
            else:
                param.requires_grad = False
        prompt_params = [p for n, p in self.model.named_parameters() if p.requires_grad]
        if len(prompt_params) == 0:
            self.optim = None
            self.sched = None
        else:
            self.optim = build_optimizer(prompt_params, config.optim)
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
        # Compute acc on pre-extracted test features with current text features
        with torch.no_grad():
            prompts, tokenized = self.model.prompt_learner.build_prompts()
            text_feats = self.model.text_encoder(prompts, tokenized)
            text_feats = F.normalize(text_feats, p=2, dim=-1)
            feats = self.features_test  # normalized already
            scale = self.model.logit_scale.exp()
            logits_test = scale * (feats @ text_feats.t())
            acc_test = compute_accuracy(logits_test, self.labels_test)[0]
        return {"loss": loss.item(), "acc_train": acc_train, "acc_test": acc_test}

    def train(self):
        start_time = time.time()
        self.build_model()
        # Zero-shot CLIP accuracy (baseline, for consistent logs)
        self.set_model_mode("eval")
        templates = _get_templates(self.config)
        zs_weights = _get_clip_weights(self.dm.dataset.classnames, self._clip_zs, templates)
        # Pre-extract test features once using CLIP visual
        feats_list, labels_list = [], []
        with torch.no_grad():
            for batch in self.test_loader:
                imgs = batch["img"].to(self.device)
                lbs = batch["label"].to(self.device)
                f = self._clip_zs.encode_image(imgs)
                f = f / f.norm(dim=-1, keepdim=True)
                feats_list.append(f)
                labels_list.append(lbs)
        self.features_test = torch.cat(feats_list, dim=0)
        self.labels_test = torch.cat(labels_list, dim=0)
        clip_logits = 100.0 * (self.features_test @ zs_weights)
        zs_acc = compute_accuracy(clip_logits, self.labels_test)[0]
        print("Zero-Shot accuracy on test: " + str(round(zs_acc, 2)))
        self.before_train()
        for self.epoch in range(self.start_epoch, self.max_epoch):
            self.before_epoch()
            self.run_epoch()
            self.after_epoch()
        self.after_train()
        print(f"Training completed in {time.time() - start_time:.2f} seconds")
        # Final unified metrics + JSON (use BaseTrainer helpers)
        try:
            metrics = self._compute_final_metrics()
            self._write_run_summary_json(metrics, start_time=start_time)
        except Exception:
            pass

    # Use BaseTrainer's _compute_final_metrics and _write_run_summary_json

    def run_epoch(self):
        losses = AverageMeter()
        batch_time = AverageMeter()
        data_time = AverageMeter()
        self.set_model_mode("train")
        # Keep frozen CLIP encoders in eval mode to avoid BN/stat updates
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


