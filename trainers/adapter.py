from typing import cast
import time
import datetime
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
import gpytorch
import numpy as np
import math
import copy

from utils.trainer import BaseTrainer
from utils.metrics import compute_accuracy, AverageMeter
from utils.optimization import build_optimizer, build_lr_scheduler, build_optimizer_from_param_groups
from utils.trainer_registry import TRAINER_REGISTRY
from utils.dataset_base import build_dataset, TorchDatasetWrapper
from utils.transforms import build_transform

from clip import clip
from datasets.imagenet_templates import IMAGENET_TEMPLATES_SELECT, IMAGENET_TEMPLATES
from .gp_template_weigher import GaussianProcessTemplateWeighter

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.allow_tf32 = True


class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection

    def forward(self, prompts, tokenized_prompts):
        x = prompts + self.positional_embedding
        x = x.permute(1, 0, 2)
        x = self.transformer(x)
        x = x.permute(1, 0, 2)
        x = self.ln_final(x)
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection
        return x


def load_clip(config, device):
    backbone_name = config.model.backbone_name
    url = clip._MODELS[backbone_name]
    model_ = clip._download(url)
    try:
        jit_model = torch.jit.load(model_).eval()
        state_dict = jit_model.state_dict()
    except RuntimeError:
        state_dict = torch.load(model_)
    model = clip.build_model(state_dict)
    return model.to(device, dtype=torch.float32)


def _get_templates(config):
    templates = ["a photo of a {}."]
    if config.adapter.num_templates > 1:
        templates += IMAGENET_TEMPLATES_SELECT[:config.adapter.num_templates - 1]
    if config.adapter.num_templates > 1 + len(IMAGENET_TEMPLATES_SELECT):
        templates += IMAGENET_TEMPLATES[:config.adapter.num_templates - 1 - len(IMAGENET_TEMPLATES_SELECT)]
    return templates


@torch.no_grad()
def _get_text_embeddings(templates, classnames, clip_model, text_encoder=None):
    device = next(clip_model.parameters()).device
    emb_list = []
    with torch.no_grad():
        for name in classnames:
            tok = clip.tokenize([t.format(name) for t in templates]).to(device)
            if text_encoder is not None:
                e = clip_model.token_embedding(tok)
                emb = text_encoder(e, tok)
            else:
                emb = clip_model.encode_text(tok)
            emb_list.append(emb)
    text_embeds = torch.stack(emb_list)
    return text_embeds


@torch.no_grad()
def _get_template_weights(config, text_embeddings: torch.Tensor, features: torch.Tensor | None, labels: torch.Tensor | None, logit_scale: torch.Tensor | float) -> torch.Tensor:
    method = str(getattr(config.adapter, 'template_init_method', 'uniform')).lower()
    temperature = 1.0
    E = text_embeddings
    K = int(E.shape[0])
    M = int(E.shape[1])
    if M == 0:
        return torch.empty(K, 0, device=E.device, dtype=E.dtype)
    if bool(getattr(config.adapter, 'prefit_on_full_set', False)):
        try:
            cfg_full = copy.deepcopy(config)
            cfg_full.dataset.num_shots = 0
            ds_full = build_dataset(cfg_full)
            tfm = build_transform(cfg_full, is_train=True)
            dl = DataLoader(
                TorchDatasetWrapper(ds_full.train_x, transform=tfm, is_train=False),
                batch_size=int(getattr(cfg_full.dataloader, 'batch_size_train', 128)),
                shuffle=False,
                num_workers=int(getattr(cfg_full.dataloader, 'num_workers', 8)),
                drop_last=False,
                pin_memory=False,
            )
            clip_model_tmp = load_clip(cfg_full, E.device)
            clip_model_tmp.eval()
            feats_list = []
            labels_list = []
            with torch.no_grad():
                for batch in dl:
                    imgs = batch["img"].to(E.device)
                    lbs = batch["label"]
                    f = clip_model_tmp.visual(imgs)
                    feats_list.append(f.cpu())
                    labels_list.append(lbs.cpu())
            features = torch.cat(feats_list, dim=0).to(device=E.device)
            labels = torch.cat(labels_list, dim=0).to(device=E.device)
            print(f"[INFO] Prefit on full set: {len(features)} samples used.")
        except Exception as e:
            print(f"[WARN] prefit_on_full_set failed ({e}); falling back to provided few-shot features.")
    if method == 'uniform' or features is None or labels is None:
        return torch.full((K, M), 1.0 / float(M), device=E.device, dtype=E.dtype)
    feats = features.to(device=E.device)
    feats = F.normalize(feats, p=2, dim=-1)
    labels_i64 = labels.to(device=feats.device, dtype=torch.int64)
    E_float = E.to(device=feats.device)
    scale = logit_scale if isinstance(logit_scale, torch.Tensor) else torch.tensor(float(logit_scale), device=feats.device)
    scale = scale.to(dtype=feats.dtype)
    counts_k = torch.bincount(labels_i64, minlength=K).to(feats.dtype).clamp_min(1)
    scores = torch.zeros(K, M, dtype=feats.dtype, device=feats.device)
    for m in range(M):
        prot_m = F.normalize(E_float[:, m, :], p=2, dim=-1)
        logits = scale * (feats @ prot_m.t())
        preds = logits.argmax(dim=1)
        corr = (preds == labels_i64).to(feats.dtype)
        sums_k = torch.zeros(K, dtype=feats.dtype, device=feats.device)
        sums_k.index_add_(0, labels_i64, corr)
        scores[:, m] = sums_k / counts_k
    if method == 'top3':
        avg_over_classes = scores.mean(dim=0)
        top_k = min(3, M)
        _, top_idx = torch.topk(avg_over_classes, k=top_k, largest=True)
        keep = torch.zeros(M, dtype=scores.dtype, device=scores.device)
        keep[top_idx] = 1.0
        scores = scores * keep.view(1, -1)
        row_sum = scores.sum(dim=1, keepdim=True)
        zero_rows = row_sum.squeeze(1) <= 1e-12
        if bool(zero_rows.any()):
            num_sel = float(top_k)
            uniform_sel = (keep / num_sel).view(1, -1).expand(int(zero_rows.sum().item()), -1)
            scores[zero_rows] = uniform_sel
    elif method == 'minmax':
        s = scores.clone()
        s_min = s.min(dim=1, keepdim=True).values
        s_max = s.max(dim=1, keepdim=True).values
        rng = (s_max - s_min)
        flat = rng.le(1e-12)
        s = torch.where(flat, torch.full_like(s, 1.0 / float(M)), (s - s_min) / rng.clamp_min(1e-12))
        scores = s
    logits_w = torch.log(scores.clamp_min(1e-12)) / max(temperature, 1e-6)
    weights = torch.softmax(logits_w, dim=1)
    return weights.to(device=E.device)


class CustomCLIP(nn.Module):
    def __init__(self, config, classnames, clip_model):
        super().__init__()
        self.config = config
        self.image_encoder = clip_model.visual
        self.logit_scale = clip_model.logit_scale
        self.text_encoder = TextEncoder(clip_model)
        self.templates = _get_templates(config)
        self.text_embeddings = _get_text_embeddings(self.templates, classnames, clip_model, self.text_encoder)
        use_gp = bool(getattr(config.adapter, 'use_gp', False))
        train_tw = bool(getattr(config.adapter, 'train_template_weights', False))
        if (not use_gp) and train_tw:
            K = int(self.text_embeddings.shape[0])
            M = int(self.text_embeddings.shape[1])
            if M > 0:
                init = torch.full((K, M), 1.0 / float(M), dtype=self.text_embeddings.dtype, device=self.text_embeddings.device)
                self.template_weights = nn.Parameter(init, requires_grad=True)
        dim = int(self.text_embeddings.shape[-1])
        self.visual_proj = nn.Linear(dim, dim, bias=False)
        with torch.no_grad():
            eye = torch.eye(dim)
            self.visual_proj.weight.copy_(eye)
        try:
            target_device = next(self.image_encoder.parameters()).device
        except StopIteration:
            target_device = self.text_embeddings.device
        self.visual_proj.to(device=target_device)
        self.gp_weighter = None
        self.gp_num_mc_samples_train = int(getattr(config.adapter, 'gp_num_mc_samples_train', 1) or 1)
        self.gp_num_mc_samples_test = int(getattr(config.adapter, 'gp_num_mc_samples_test', 1) or 1)
        if use_gp:
            self.gp_weighter = GaussianProcessTemplateWeighter(
                text_embeddings=self.text_embeddings,
                cfg=config,
            )

    def get_prototypes(self, num_samples: int = 1, visual_embeddings: torch.Tensor = None):
        target_device = next(self.parameters()).device
        if self.gp_weighter is not None:
            if num_samples <= 1:
                prototypes = self.gp_weighter.prototypes_from_posterior_mean()
            else:
                proto_s = self.gp_weighter.sample_prototypes(num_samples, visual_embeddings)
                prototypes = proto_s
        else:
            if isinstance(getattr(self, 'template_weights', None), torch.Tensor):
                prototypes = torch.einsum("km,kmd->kd", cast(torch.Tensor, self.template_weights), self.text_embeddings)
            else:
                prototypes = self.text_embeddings.mean(dim=1)
        if prototypes.device != target_device:
            prototypes = prototypes.to(target_device)
        return prototypes

    def forward_features(self, features: torch.Tensor) -> torch.Tensor:
        proj_weight = self.visual_proj.weight
        if features.device != proj_weight.device:
            features = features.to(device=proj_weight.device)
        projected = self.visual_proj(features)
        features_norm = F.normalize(projected, p=2, dim=-1)
        scale = self.logit_scale.exp()
        num_samples = self.gp_num_mc_samples_train if self.training else self.gp_num_mc_samples_test
        prototypes = self.get_prototypes(num_samples=num_samples, visual_embeddings=projected)
        if prototypes.device != features_norm.device:
            prototypes = prototypes.to(device=features_norm.device)
        prototypes_norm = F.normalize(prototypes, p=2, dim=-1)
        if prototypes.dim() == 3:
            logits_s = scale * torch.einsum("bd,skd->bsk", features_norm, prototypes_norm)
            base_logits = logits_s.mean(dim=1)
        else:
            base_logits = scale * (features_norm @ prototypes_norm.t())
        return base_logits

    def forward(self, image: torch.Tensor, return_features: bool = False):
        features = self.image_encoder(image)
        logits = self.forward_features(features)
        if return_features:
            return logits, features
        return logits


@TRAINER_REGISTRY.register("Adapter")
class Trainer(BaseTrainer):
    def __init__(self, config, dataset_manager):
        super().__init__(config, dataset_manager)

    def build_model(self):
        config = self.config
        classnames = self.dm.dataset.classnames
        print(f"Loading CLIP (backbone: {config.model.backbone_name})")
        clip_model = load_clip(config, self.device)
        print("Building custom CLIP")
        self.model = CustomCLIP(config, classnames, clip_model)
        self.model.to(self.device)
        for name, param in self.model.named_parameters():
            allow_template = (not config.adapter.use_gp) and bool(getattr(config.adapter, 'train_template_weights', False))
            if bool(getattr(config.adapter, 'freeze_visual_proj', False)) and ("visual_proj" in name):
                param.requires_grad = False
                continue
            if ("visual_proj" in name) or ("gp_weighter" in name) or (allow_template and ("template_weights" in name)):
                param.requires_grad = True
            else:
                param.requires_grad = False
        if config.adapter.use_gp and self.model.gp_weighter is not None:
            base_params = []
            base_params.extend([p for p in self.model.visual_proj.parameters() if p.requires_grad])
            gp_params = [p for p in self.model.gp_weighter.parameters() if p.requires_grad]
            if (len(base_params) + len(gp_params)) == 0:
                self.optim = None
                self.sched = None
            else:
                param_groups = [
                    {
                        'params': base_params,
                        'lr': float(config.optim.lr),
                        'weight_decay': float(config.optim.weight_decay)
                    },
                    {
                        'params': gp_params,
                        'lr': float(config.adapter.gp_lr),
                        'weight_decay': float(config.optim.weight_decay)
                    },
                ]
                self.optim = build_optimizer_from_param_groups(param_groups, config.optim)
                self.sched = build_lr_scheduler(self.optim, config.optim)
        else:
            baseline_params = []
            baseline_params.extend([p for p in self.model.visual_proj.parameters() if p.requires_grad])
            if hasattr(self.model, 'template_weights') and isinstance(self.model.template_weights, torch.nn.Parameter):
                if self.model.template_weights.requires_grad:
                    baseline_params.append(self.model.template_weights)
            if len(baseline_params) == 0:
                self.optim = None
                self.sched = None
            else:
                self.optim = build_optimizer(baseline_params, config.optim)
                self.sched = build_lr_scheduler(self.optim, config.optim)
        self.scaler = None

    def forward_backward(self, batch):
        model = cast(CustomCLIP, self.model)
        features, labels = batch
        features = torch.tensor(features).to(self.device)
        if not torch.is_tensor(labels):
            labels = torch.tensor(labels).to(self.device)
        else:
            labels = labels.detach().clone().to(self.device)
        projected_features = features
        proj_features = self.model.visual_proj(projected_features.to(self.model.visual_proj.weight.dtype))
        prototypes = model.get_prototypes(num_samples=int(getattr(self.config.adapter, 'gp_num_mc_samples_train', 1) or 1), visual_embeddings=proj_features)
        try:
            with torch.no_grad():
                proto_norms = prototypes.norm(dim=-1)
                self._dbg_proto_stats = {
                    "mean": float(proto_norms.mean().item()),
                    "std": float(proto_norms.std(unbiased=False).item()),
                    "min": float(proto_norms.min().item()),
                    "max": float(proto_norms.max().item()),
                }
        except Exception:
            self._dbg_proto_stats = None
        num_samples = int(getattr(self.config.adapter, 'gp_num_mc_samples_train', 1) or 1)
        loss = self.compute_loss(projected_features, labels, num_samples=num_samples)
        logits = model.forward_features(projected_features)
        self._backward_and_update(loss)
        with torch.no_grad():
            acc_train = compute_accuracy(logits, labels)[0]
            test_features = self.features_test.to(self.device)
            test_logits = model.forward_features(test_features)
            acc_test = compute_accuracy(test_logits, self.labels_test.to(self.device))[0]
        return {
            "loss": loss.item(),
            "acc_train": acc_train,
            "acc_test": acc_test
        }

    def compute_loss(self, features: torch.Tensor, labels: torch.Tensor, num_samples: int = 1):
        model = cast(CustomCLIP, self.model)
        use_gp = bool(getattr(self.config.adapter, 'use_gp', False) and getattr(model, 'gp_weighter', None) is not None)
        gp_weighter = getattr(model, 'gp_weighter', None)
        num_samples = int(num_samples or 1)
        if use_gp and num_samples > 1 and gp_weighter is not None:
            proj_features = self.model.visual_proj(features.to(self.model.visual_proj.weight.dtype))
            protos = gp_weighter.sample_prototypes(num_samples, visual_embeddings=proj_features)
            proj_weight = model.visual_proj.weight
            target_device = proj_weight.device
            target_dtype = proj_weight.dtype
            if protos.device != target_device:
                protos = protos.to(device=target_device)
            if protos.dtype != target_dtype:
                protos = protos.to(dtype=target_dtype)
            f = features
            if f.device != target_device:
                f = f.to(device=target_device)
            if f.dtype != target_dtype:
                f = f.to(dtype=target_dtype)
            projected = model.visual_proj(f)
            features_norm = F.normalize(projected, p=2, dim=-1)
            scale = model.logit_scale.exp()
            ce_vals = []
            for s in range(num_samples):
                prototypes_s = protos[s]
                prototypes_norm = F.normalize(prototypes_s, p=2, dim=-1)
                logits_s = scale * (features_norm @ prototypes_norm.t())
                ce_vals.append(F.cross_entropy(logits_s, labels))
            ce_loss = torch.stack(ce_vals, dim=0).mean()
        else:
            f = features
            proj_weight = model.visual_proj.weight
            if f.device != proj_weight.device:
                f = f.to(device=proj_weight.device)
            if f.dtype != proj_weight.dtype:
                f = f.to(dtype=proj_weight.dtype)
            logits = model.forward_features(f)
            ce_loss = F.cross_entropy(logits, labels)
        total_loss = ce_loss
        if use_gp and gp_weighter is not None and bool(getattr(self.config.adapter, 'gp_use_elbo', True)):
            try:
                gp_weighter.train()
                if hasattr(gp_weighter, 'likelihood'):
                    gp_weighter.likelihood.train()
                if not hasattr(self, '_gp_targets') or getattr(self, '_gp_targets', None) is None:
                    try:
                        self._gp_targets = self._compute_gp_template_targets_prob()
                    except Exception:
                        self._gp_targets = None
                if getattr(self, '_gp_targets', None) is not None:
                    x = gp_weighter._templates
                    y = cast(torch.Tensor, self._gp_targets)
                    if y.device != x.device:
                        y = y.to(x.device)
                    mll = gpytorch.mlls.VariationalELBO(
                        gp_weighter.likelihood,
                        gp_weighter,
                        num_data=int(gp_weighter.num_templates),
                        beta=float(self.config.adapter.gp_beta),
                    )
                    out = gp_weighter(x)
                    elbo_val = mll(out, y)
                    elbo_mean = elbo_val.mean() if elbo_val.dim() > 0 else elbo_val
                    total_loss = total_loss + (-elbo_mean)
                    try:
                        self._dbg_loss_components = getattr(self, '_dbg_loss_components', {})
                        self._dbg_loss_components['elbo'] = float((-elbo_mean).detach().item())
                    except Exception:
                        pass
            except Exception:
                pass
        try:
            W = model.visual_proj.weight
            if W.requires_grad:
                eye = torch.eye(W.shape[0], device=W.device, dtype=W.dtype)
                l2_reg = (W - eye).pow(2).sum()
                l2_lambda = self.config.adapter.l2_lambda
                shots = self.config.dataset.num_shots
                l2_reg = l2_reg * l2_lambda / shots
                total_loss += l2_reg
            else:
                l2_reg = None
        except Exception:
            l2_reg = None
        try:
            self._dbg_loss_components = {
                "ce": float(ce_loss.detach().item()),
                "l2_reg": float(l2_reg.detach().item()) if l2_reg is not None else 0.0,
                "total": float(total_loss.detach().item()),
            }
        except Exception:
            pass
        return total_loss

    def _backward_and_update(self, loss):
        if getattr(self, 'optim', None) is None:
            return
        self.optim.zero_grad()
        loss.backward()
        try:
            self._capture_grad_norms()
        except Exception:
            pass
        self.optim.step()

    def _capture_grad_norms(self):
        base_norm_sq = 0.0
        gp_norm_sq = 0.0
        try:
            if hasattr(self, "optim") and hasattr(self.optim, "param_groups"):
                base_group = self.optim.param_groups[0]
                for p in base_group.get('params', []):
                    if p is not None and getattr(p, 'grad', None) is not None:
                        base_norm_sq += float(p.grad.detach().pow(2).sum().item())
                if len(self.optim.param_groups) > 1:
                    gp_group = self.optim.param_groups[1]
                    for p in gp_group.get('params', []):
                        if p is not None and getattr(p, 'grad', None) is not None:
                            gp_norm_sq += float(p.grad.detach().pow(2).sum().item())
        except Exception:
            pass
        self._dbg_grad_norms = {
            "base": math.sqrt(base_norm_sq) if base_norm_sq > 0 else 0.0,
            "gp": math.sqrt(gp_norm_sq) if gp_norm_sq > 0 else 0.0,
        }

    def parse_batch_train(self, batch):
        input_data = batch["img"]
        labels = batch["label"]
        input_data = input_data.to(self.device)
        labels = labels.to(self.device)
        return input_data, labels

    def train(self):
        start_time = time.time()
        self.build_model()
        self.set_model_mode("eval")
        self.labels_test, output_test, self.features_test = self.extract_features(partition="test")
        print("Zero-Shot accuracy on test: " + str(round(compute_accuracy(output_test.cuda(), self.labels_test.cuda())[0], 2)))
        self.labels_train, logits_zs, self.features_train = self.extract_features(partition="train")
        if str(getattr(self.config.adapter, 'benchmark_method', 'none')).lower() == 'none':
            try:
                model = cast(CustomCLIP, self.model)
                feats = self.features_train.to(self.device)
                template_weights = _get_template_weights(
                    self.config,
                    text_embeddings=model.text_embeddings,
                    features=feats,
                    labels=self.labels_train.to(self.device),
                    logit_scale=model.logit_scale.exp(),
                )
                if hasattr(model, 'template_weights') and isinstance(getattr(model, 'template_weights'), torch.nn.Parameter):
                    with torch.no_grad():
                        model.template_weights.data.copy_(template_weights.to(dtype=model.text_embeddings.dtype, device=model.text_embeddings.device))
                else:
                    model.template_weights = template_weights.to(dtype=model.text_embeddings.dtype, device=model.text_embeddings.device)
                if getattr(self.config.adapter, 'use_gp', False) and getattr(model, 'gp_weighter', None) is not None:
                    model.gp_weighter.initialize_from_weights(template_weights)
                    print("[GP] One-step initialization applied to GP weights.")
            except Exception as e:
                print(f"[WARN] Template weight init skipped: {e}")
        self.before_train()
        for self.epoch in range(self.start_epoch, self.max_epoch):
            self.before_epoch()
            self.run_epoch()
            self.after_epoch()
        self.after_train()
        # Final unified metrics + JSON
        try:
            metrics = self._compute_final_metrics()
            self._write_run_summary_json(metrics, start_time=start_time)
        except Exception as e:
            print(f"[WARN] Failed to write metrics.json: {e}")
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
        try:
            if hasattr(self.model, "text_encoder"):
                self.model.text_encoder.eval()
        except AttributeError:
            pass
        self.num_batches = len(self.train_loader_x)
        self.batch_size = self.train_loader_x.batch_size or 1
        if not hasattr(self, 'features_train') or not hasattr(self, 'labels_train'):
            raise RuntimeError("features_train and labels_train must be extracted before training")
        features = self.features_train.clone().cpu().numpy()
        labels = self.labels_train.clone()
        idx = np.random.rand(features.shape[0]).argsort(axis=0)
        features = features[idx, :]
        labels = labels[idx]
        loss_summary = {"loss": 0.0, "acc_train": 0.0, "acc_test": 0.0}
        end = time.time()
        for self.batch_idx in range(self.num_batches):
            batch_init = self.batch_idx * self.batch_size
            batch_end = (self.batch_idx + 1) * self.batch_size
            data_time.update(time.time() - end)
            batch_data = (features[batch_init:batch_end], labels[batch_init:batch_end])
            loss_summary = self.forward_backward(batch_data)
            batch_time.update(time.time() - end)
            losses.update(loss_summary['loss'])
            meet_freq = (self.batch_idx + 1) % self.config.train.print_freq == 0
            only_few_batches = self.num_batches < self.config.train.print_freq
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
                try:
                    # Loss breakdown
                    if hasattr(self, "_dbg_loss_components"):
                        comp = getattr(self, "_dbg_loss_components")
                        print(
                            f"  [DBG] loss: CE={comp.get('ce', 0):.4f} l2_reg={comp.get('l2_reg', 0):.4f} Total={comp.get('total', 0):.4f}"
                        )
                    # Prototype stats
                    if hasattr(self, "_dbg_proto_stats") and self._dbg_proto_stats is not None:
                        ps = self._dbg_proto_stats
                        print(
                            f"  [DBG] proto_norms: mean={ps['mean']:.4f} std={ps['std']:.4f} min={ps['min']:.4f} max={ps['max']:.4f}"
                        )
                    # Optimizer LRs and grad norms
                    if hasattr(self, "optim") and hasattr(self.optim, "param_groups"):
                        if len(self.optim.param_groups) == 2:
                            lr_base = float(self.optim.param_groups[0]['lr'])
                            lr_gp = float(self.optim.param_groups[1]['lr'])
                            print(f"  [DBG] lr_base={lr_base:.6f} lr_gp={lr_gp:.6f}")
                        elif len(self.optim.param_groups) == 1:
                            lr0 = float(self.optim.param_groups[0]['lr'])
                            print(f"  [DBG] lr={lr0:.6f}")
                    if hasattr(self, "_dbg_grad_norms"):
                        gn = self._dbg_grad_norms
                        print(f"  [DBG] grad_norms: base={gn.get('base', 0.0):.6f} gp={gn.get('gp', 0.0):.6f}")
                    # GP-specific stats
                    if getattr(self.config.adapter, "use_gp", False):
                        gp = getattr(self.model, "gp_weighter", None)
                        if gp is not None:
                            try:
                                covar_module = gp.covar_module if hasattr(gp, "covar_module") else None
                                if covar_module is not None:
                                    outscale = covar_module.outputscale if hasattr(covar_module, "outputscale") else None
                                    outscale_val = float(outscale.detach().mean().item()) if outscale is not None else float('nan')
                                    base_k = covar_module.base_kernel if hasattr(covar_module, "base_kernel") else None
                                    if base_k is not None and hasattr(base_k, "lengthscale"):
                                        ls_val = float(base_k.lengthscale.detach().mean().item())
                                    else:
                                        ls_val = float('nan')
                                else:
                                    outscale_val = float('nan')
                                    ls_val = float('nan')
                            except Exception:
                                outscale_val = float('nan')
                                ls_val = float('nan')
                            print(
                                f"  [DBG][GP] lengthscale={ls_val:.6f} outputscale={outscale_val:.6f}"
                            )
                            # Template weights for class 0 (posterior-mean softmax over templates)
                            try:
                                with torch.no_grad():
                                    x_t = gp._templates.to(dtype=torch.float32, device=gp._templates.device)
                                    f_mean = gp(x_t).mean
                                    w = torch.softmax(f_mean, dim=-1)
                                    w0 = w[0].detach().cpu().tolist()
                                    w0_str = ", ".join(f"{v:.3f}" for v in w0)
                                    print(f"  [DBG][GP] template_weights[class=0]: [{w0_str}]")
                            except Exception:
                                pass
                        else:
                            print("  [DBG][GP] GP disabled at runtime (no weighter present).")
                except Exception:
                    pass
            n_iter = self.epoch * self.num_batches + self.batch_idx
            self.write_scalar("train/loss", loss_summary['loss'], n_iter)
            self.write_scalar("train/lr", self.get_current_lr(), n_iter)
            end = time.time()
        return loss_summary

    def extract_features(self, partition="train", reps=1, transforms=None):
        print("Extracting features from: " + partition)
        self.set_model_mode("eval")
        if partition == "train":
            data_loader = torch.utils.data.DataLoader(
                copy.deepcopy(self.train_loader_x.dataset),
                batch_size=self.train_loader_x.batch_size,
                sampler=self.train_loader_x.sampler,
                num_workers=self.train_loader_x.num_workers,
                drop_last=False,
                pin_memory=False
            )
        elif partition == "val":
            data_loader = copy.deepcopy(self.val_loader)
        elif partition == "test":
            data_loader = copy.deepcopy(self.test_loader)
        else:
            raise ValueError(f"Unknown partition: {partition}")
        labels_ds, logits_ds, features_ds = [], [], []
        for rep in range(reps):
            if data_loader is not None:
                for batch_idx, batch in enumerate(data_loader):
                    input_data, labels = self.parse_batch_train(batch)
                    with torch.no_grad():
                        logits, features = self.model(input_data, return_features=True)  # type: ignore
                    labels_ds.append(labels.cpu())
                    logits_ds.append(logits.cpu())
                    features_ds.append(features.cpu())
        labels_ds = torch.cat(labels_ds, dim=0)
        logits_ds = torch.cat(logits_ds, dim=0)
        features_ds = torch.cat(features_ds, dim=0)
        return labels_ds, logits_ds, features_ds

    @torch.no_grad()
    def _compute_gp_template_targets_prob(self) -> torch.Tensor:
        features = self.features_train.detach().cpu()
        labels = self.labels_train.detach().cpu().to(torch.int64)
        text_emb = cast(torch.Tensor, self.model.text_embeddings).detach().cpu()
        K, M, D = int(text_emb.shape[0]), int(text_emb.shape[1]), int(text_emb.shape[2])
        N = int(features.shape[0])
        try:
            W = cast(torch.nn.Linear, self.model.visual_proj).weight.detach().cpu()
            feats_proj = features @ W.t()
        except Exception:
            feats_proj = features
        feats_norm = F.normalize(feats_proj, p=2, dim=-1)
        scale = float(cast(torch.Tensor, self.model.logit_scale).exp().detach().cpu().item())
        labels_one_hot = torch.zeros(N, K, dtype=torch.float32)
        labels_one_hot[torch.arange(N), labels] = 1.0
        class_counts = labels_one_hot.sum(dim=0).clamp_min(1.0)
        targets = torch.zeros(K, M, dtype=torch.float32)
        for m in range(M):
            prot_m = text_emb[:, m, :]
            prot_m = F.normalize(prot_m, p=2, dim=-1)
            logits = scale * (feats_norm @ prot_m.t())
            probs = torch.softmax(logits, dim=-1)
            sum_probs = (labels_one_hot * probs).sum(dim=0)
            targets[:, m] = sum_probs / class_counts
        return targets


