import time
import datetime
import torch
import torch.nn as nn
from torch.nn import functional as F

from clip import clip
from utils.trainer import BaseTrainer, TextEncoder, load_clip
from utils.metrics import compute_accuracy, AverageMeter
from utils.optimization import build_optimizer, build_lr_scheduler
from utils.trainer_registry import TRAINER_REGISTRY


class PromptLearnerCoCoOp(nn.Module):
    def __init__(self, config, classnames, clip_model):
        super().__init__()
        device = next(clip_model.parameters()).device
        dtype = next(clip_model.parameters()).dtype
        n_ctx = int(getattr(config.adapter, 'n_ctx', 16))
        ctx_init = str(getattr(config.adapter, 'ctx_init', '') or '')
        ctx_dim = int(clip_model.ln_final.weight.shape[0])
        vis_dim = int(clip_model.visual.output_dim)
        prompt_prefix = ctx_init.replace('_', ' ') if len(ctx_init) > 0 else ' '.join(['X'] * n_ctx)
        prompts = [f"{prompt_prefix} {name}." for name in classnames]
        tokenized_prompts = clip.tokenize(prompts).to(device)
        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts)
        if len(ctx_init) > 0:
            n_ctx = int(ctx_init.count(' ') + 1)
            ctx_vectors = embedding[0, 1:1 + n_ctx, :].detach().clone()
        else:
            ctx_vectors = torch.empty(n_ctx, ctx_dim, device=device, dtype=dtype)
            nn.init.normal_(ctx_vectors, std=0.02)
        self.ctx = nn.Parameter(ctx_vectors)
        self.meta_net = nn.Sequential(
            nn.Linear(vis_dim, max(1, vis_dim // 16)),
            nn.ReLU(inplace=True),
            nn.Linear(max(1, vis_dim // 16), ctx_dim),
        )
        self.register_buffer('token_prefix', embedding[:, :1, :])
        self.register_buffer('token_suffix', embedding[:, 1 + n_ctx:, :])
        self.register_buffer('tokenized_prompts', tokenized_prompts)
        self.n_ctx = n_ctx
        self.num_classes = len(classnames)
        self._token_embedding = clip_model.token_embedding

    def build_prompts(self, image_features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        N = int(image_features.shape[0])
        K = int(self.num_classes)
        bias = self.meta_net(image_features).unsqueeze(1)
        ctx = self.ctx.unsqueeze(0)
        ctx_shifted = ctx + bias
        prompts_list = []
        for i in range(N):
            ctx_i = ctx_shifted[i].unsqueeze(0).expand(K, -1, -1)
            pts_i = torch.cat([self.token_prefix, ctx_i, self.token_suffix], dim=1)
            prompts_list.append(pts_i)
        prompts = torch.stack(prompts_list, dim=0)
        return prompts, self.tokenized_prompts


class CustomCLIP(nn.Module):
    def __init__(self, config, classnames, clip_model):
        super().__init__()
        self.image_encoder = clip_model.visual
        self.logit_scale = clip_model.logit_scale
        self.text_encoder = TextEncoder(clip_model)
        self.prompt_learner = PromptLearnerCoCoOp(config, classnames, clip_model)

    def forward(self, image: torch.Tensor):
        projected = self.image_encoder(image)
        img_for_meta = F.normalize(projected, p=2, dim=-1)
        prompts_b, tokenized = self.prompt_learner.build_prompts(img_for_meta)
        logits_list = []
        scale = self.logit_scale.exp()
        for i in range(projected.shape[0]):
            pts_i = prompts_b[i]
            text_i = self.text_encoder(pts_i, tokenized)
            text_i = F.normalize(text_i, p=2, dim=-1)
            logit_i = scale * (img_for_meta[i] @ text_i.t())
            logits_list.append(logit_i)
        return torch.stack(logits_list, dim=0)


@TRAINER_REGISTRY.register("Adapter-CoCoOp")
class Trainer(BaseTrainer):
    def __init__(self, config, dataset_manager):
        super().__init__(config, dataset_manager)

    def build_model(self):
        config = self.config
        classnames = self.dm.dataset.classnames
        print(f"Loading CLIP (backbone: {config.model.backbone_name})")
        clip_model = load_clip(config, self.device)
        print("Building CoCoOp adapter")
        self.model = CustomCLIP(config, classnames, clip_model)
        self.model.to(self.device)
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
        return {"loss": loss.item(), "acc_train": acc_train}

    def train(self):
        start_time = time.time()
        self.build_model()
        self.before_train()
        for self.epoch in range(self.start_epoch, self.max_epoch):
            self.before_epoch()
            self.run_epoch()
            self.after_epoch()
        self.after_train()
        print(f"Training completed in {time.time() - start_time:.2f} seconds")
        try:
            metrics = self._compute_final_metrics()
            self._write_run_summary_json(metrics, start_time=start_time)
        except Exception:
            pass

    def run_epoch(self):
        losses = AverageMeter()
        batch_time = AverageMeter()
        data_time = AverageMeter()
        self.set_model_mode("train")
        self.num_batches = len(self.train_loader_x)
        end = time.time()
        for self.batch_idx, batch in enumerate(self.train_loader_x):
            data_time.update(time.time() - end)
            loss_summary = self.forward_backward(batch)
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


