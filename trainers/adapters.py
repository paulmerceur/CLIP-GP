import os.path as osp
from typing import List, TYPE_CHECKING, Union, Any
import time
import datetime
import copy
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.amp.grad_scaler import GradScaler    
from torch.amp.autocast_mode import autocast
import numpy as np
import math

from utils.trainer import BaseTrainer
from utils.metrics import compute_accuracy, AverageMeter, MetricMeter
from utils.checkpoint import load_pretrained_weights
from utils.optimization import build_optimizer, build_lr_scheduler
from utils.trainer_registry import TRAINER_REGISTRY

from clip import clip
from datasets.imagenet_templates import IMAGENET_TEMPLATES_SELECT, IMAGENET_TEMPLATES
from .gp_template_weigher import GaussianProcessTemplateWeighter

if TYPE_CHECKING:
    from typing import Any

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.allow_tf32 = True


CUSTOM_TEMPLATES = {
    "OxfordPets": "a photo of a {}, a type of pet.",
    "OxfordFlowers": "a photo of a {}, a type of flower.",
    "FGVCAircraft": "a photo of a {}, a type of aircraft.",
    "DescribableTextures": "{} texture.",
    "EuroSAT": "a centered satellite photo of {}.",
    "StanfordCars": "a photo of a {}.",
    "Food101": "a photo of {}, a type of food.",
    "SUN397": "a photo of a {}.",
    "Caltech101": "a photo of a {}.",
    "UCF101": "a photo of a person doing {}.",
    "ImageNet": "a photo of a {}.",
    "ImageNetSketch": "a photo of a {}.",
    "ImageNetV2": "a photo of a {}.",
    "ImageNetA": "a photo of a {}.",
    "ImageNetR": "a photo of a {}.",
}


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
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection

        return x


def load_clip_to_cpu(config):
    backbone_name = config.model.backbone_name
    url = clip._MODELS[backbone_name]
    model_path = clip._download(url)

    try:
        # loading JIT archive
        jit_model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = jit_model.state_dict()
    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu")

    model = clip.build_model(state_dict)
    return model


def _get_base_text_features(config, classnames, clip_model, text_encoder=None):
    """Extract text features for all templates and classes."""
    device = next(clip_model.parameters()).device
    
    # Get template strings (using the sophisticated logic from the old version)
    templates = ["a photo of a {}."]
    
    if config.adapter.num_templates > 1:
        num_needed = min(
            config.adapter.num_templates - 1,  # -1 because we already have "a photo of a {}"
            len(IMAGENET_TEMPLATES_SELECT),
        )
        templates += IMAGENET_TEMPLATES_SELECT[:num_needed]
    if config.adapter.num_templates > 1 + len(IMAGENET_TEMPLATES_SELECT):
        # Add templates from IMAGENET_TEMPLATES
        templates += IMAGENET_TEMPLATES[:config.adapter.num_templates - 1 - len(IMAGENET_TEMPLATES_SELECT)]
    
    print(f"[DEBUG] templates: {templates}")

    # Encode all prompts once - returned tensor is reused by caller.
    emb_list = []
    with torch.no_grad():
        for name in classnames:
            tok = clip.tokenize([t.format(name) for t in templates]).to(device)
            if text_encoder is not None:
                # Use custom TextEncoder (the correct way!)
                e = clip_model.token_embedding(tok).type(clip_model.dtype)
                emb = text_encoder(e, tok)
            else:
                # Fallback to CLIP's encode_text
                emb = clip_model.encode_text(tok)
            emb_list.append(emb)
    text_embeds = torch.stack(emb_list)  # [K,M,D]

    # Instantiate GP
    if config.adapter.use_gp and len(templates) > 1:
        # Better prior: use per-template zero-shot logits. For each
        # class k and template m we compute the logit obtained by
        # measuring the similarity of that template to the **average**
        # prototype of the class. This centres the GP posterior on a
        # reasonably calibrated location instead of a flat 0.
        with torch.no_grad():
            class_mean = text_embeds.mean(dim=1, keepdim=True)          # [K,1,D]
            zs_logits = (text_embeds * class_mean).sum(-1)              # [K,M]
        mean_init = zs_logits.to(dtype=torch.float32, device=device)
        gp = GaussianProcessTemplateWeighter(text_embeddings=text_embeds, cfg=config, mean_init=mean_init).to(device)
        proto, kl = gp.forward_and_kl()
        return proto, text_embeds, gp, kl

    # GP disabled -> simple average
    return text_embeds.mean(1), text_embeds, None, None


class LinearAdapter(nn.Module):
    """Simple linear adapter with learnable scaling."""
    
    def __init__(self, num_classes: int, feature_dim: int, init_type: str = "ZS"):
        super().__init__()
        self.num_classes = num_classes
        self.feature_dim = feature_dim
        self.init_type = init_type
        
        # Learnable adapter weights
        self.adapter = nn.Linear(feature_dim, num_classes, bias=False)
        
    def forward(self, features):
        """Forward pass through adapter."""
        return self.adapter(features)


class CustomCLIP(nn.Module):
    """Custom CLIP model with adapter and optional GP weighting."""
    
    def __init__(self, config, classnames, clip_model):
        super().__init__()
        self.config = config
        self.num_classes = len(classnames)
        self.dtype = clip_model.dtype
        
        # Store CLIP components
        self.image_encoder = clip_model.visual
        self.logit_scale = clip_model.logit_scale
        # Use the correct vision dimension calculation (like the old version)
        self.vision_dim = clip_model.text_projection.shape[1]
        
        # Create TextEncoder for proper text encoding
        self.text_encoder = TextEncoder(clip_model)
        
        # Always use visual projection (for fair comparison)
        self.visual_proj = nn.Linear(self.vision_dim, self.vision_dim, bias=False)
        with torch.no_grad():
            self.visual_proj.weight.copy_(torch.eye(self.vision_dim))
        
        # Get text features and setup GP if needed
        base_proto, self.text_embeddings_all, self.gp_weighter, _ = _get_base_text_features(
            config, classnames, clip_model, self.text_encoder
        )
        
        # Create adapter
        self.adapter = LinearAdapter(
            num_classes=self.num_classes,
            feature_dim=self.vision_dim,
            init_type=config.adapter.init
        )
        
        # Initialize adapter with base prototypes
        with torch.no_grad():
            self.adapter.adapter.weight.copy_(base_proto)
    
    def get_gp_kl_divergence(self):
        """Get KL divergence from GP (if using GP)."""
        if not hasattr(self, '_last_gp_kl'):
            return None
        return getattr(self, "_last_gp_kl", None)
    
    def forward_features(self, image):
        """Extract and project visual features."""
        with torch.no_grad():
            features = self.image_encoder(image.type(self.dtype))
        
        # Apply visual projection
        if features.dtype != self.visual_proj.weight.dtype:
            features = features.to(dtype=self.visual_proj.weight.dtype)
        features = self.visual_proj(features)
        
        return features
    
    def forward_prototypes(self):
        """Get current class prototypes (baseline or GP-weighted)."""
        if self.gp_weighter is not None:
            current_prototypes, kl = self.gp_weighter.forward_and_kl()
            self._last_gp_kl = kl
            # Ensure prototypes are on the same device as the model
            target_device = next(self.parameters()).device
            if current_prototypes.device != target_device:
                current_prototypes = current_prototypes.to(target_device)
            return current_prototypes
        else:
            # Use base prototypes (mean across templates)
            self._last_gp_kl = None
            prototypes = self.text_embeddings_all.mean(dim=1)  # [K, D]
            # Ensure prototypes are on the same device as the model
            target_device = next(self.parameters()).device
            if prototypes.device != target_device:
                prototypes = prototypes.to(target_device)
            return prototypes
    
    def forward(self, image, label=None):
        """Forward pass for training."""
        # Get visual features
        features = self.forward_features(image)
        
        # Get prototypes
        prototypes = self.forward_prototypes()  # [K, D]
        
        # Ensure same dtype and device
        if features.dtype != prototypes.dtype:
            prototypes = prototypes.to(dtype=features.dtype)
        if features.device != prototypes.device:
            prototypes = prototypes.to(device=features.device)
        
        # Compute logits via similarity
        features_norm = features / features.norm(dim=-1, keepdim=True)
        prototypes_norm = prototypes / prototypes.norm(dim=-1, keepdim=True)
        
        # Ensure same dtype for matrix multiplication
        features_norm = features_norm.to(prototypes_norm.dtype)
        logits = self.logit_scale.exp() * features_norm @ prototypes_norm.t()
        
        return logits
    
    def sample_forward(self, image, num_samples=50):
        """Forward pass with GP sampling (for evaluation)."""
        if self.gp_weighter is None:
            # No GP, just use regular forward
            return self.forward(image)
        
        # Sample multiple prototypes from GP
        prototypes_s = self.gp_weighter.sample_prototypes(num_samples)  # [S, K, D]
        
        # Get visual features
        features = self.forward_features(image)  # [B, D]
        features_norm = features / features.norm(dim=-1, keepdim=True)
        
        # Compute logits for each sample
        logits_samples = []
        for s in range(num_samples):
            proto_s = prototypes_s[s]  # [K, D]
            proto_s_norm = proto_s / proto_s.norm(dim=-1, keepdim=True)
            # Ensure same dtype for matrix multiplication
            features_norm = features_norm.to(proto_s_norm.dtype)
            logits_s = self.logit_scale.exp() * features_norm @ proto_s_norm.t()
            logits_samples.append(logits_s)
        
        # Average across samples
        logits = torch.stack(logits_samples, dim=0).mean(dim=0)
        return logits



@TRAINER_REGISTRY.register("ADAPTER")
class ADAPTER(BaseTrainer):
    """Unified adapter trainer supporting both baseline and GP methods."""
    
    def __init__(self, config, dataset_manager):
        super().__init__(config, dataset_manager)

    def check_cfg(self, config) -> None:
        assert config.adapter.prec in ["fp16", "fp32", "amp"]

    def build_model(self):
        config = self.config
        classnames = self.dm.dataset.classnames

        print(f"Loading CLIP (backbone: {config.model.backbone_name})")
        clip_model = load_clip_to_cpu(config)

        if config.adapter.prec == "fp32" or config.adapter.prec == "amp":
            clip_model.float()

        print("Building custom CLIP")
        self.model = CustomCLIP(config, classnames, clip_model)

        # Setup parameter groups
        for name, param in self.model.named_parameters():
            if name == "logit_scale":
                param.requires_grad = True
            elif "visual_proj" in name:
                param.requires_grad = True
            elif "adapter" in name:
                param.requires_grad = True
            elif "gp_weighter" in name:
                param.requires_grad = True
            else:
                param.requires_grad = False

        if config.model.init_weights:
            load_pretrained_weights(self.model.adapter, config.model.init_weights)

        self.model.to(self.device)
        self.model.float()
        
            
        # Setup optimizer with different learning rates for GP
        if config.adapter.use_gp and self.model.gp_weighter is not None:
            # Two parameter groups: base params and GP params
            base_params = []
            base_params.extend([p for p in self.model.adapter.parameters() if p.requires_grad])
            base_params.extend([p for p in self.model.visual_proj.parameters() if p.requires_grad])
            if self.model.logit_scale.requires_grad:
                base_params.append(self.model.logit_scale)
            
            gp_params = [p for p in self.model.gp_weighter.parameters() if p.requires_grad]

            param_groups = [
                {
                    'params': base_params,
                    'lr': float(config.optim.lr),
                    'weight_decay': float(config.optim.weight_decay)
                },
                {
                    'params': gp_params,
                    'lr': float(config.adapter.gp_lr),
                    'weight_decay': 0.0
                },
            ]

            optim_map = {"sgd": torch.optim.SGD, "adam": torch.optim.Adam}
            BaseOptim = optim_map.get(config.optim.name.lower(), torch.optim.SGD)
            
            # Add SGD-specific parameters for explicit control
            if config.optim.name.lower() == "sgd":
                self.optim = BaseOptim(param_groups, momentum=float(config.optim.momentum))
            else:
                self.optim = BaseOptim(param_groups)
            
            # Create scheduler manually for GP case
            if config.optim.lr_scheduler.lower() == "cosine":
                from torch.optim.lr_scheduler import CosineAnnealingLR
                self.sched = CosineAnnealingLR(self.optim, T_max=config.optim.max_epoch)
            else:
                # Default: no scheduler
                self.sched = None
        else:
            # Single parameter group for baseline
            baseline_params = []
            baseline_params.extend(list(self.model.adapter.parameters()))
            baseline_params.extend([p for p in self.model.visual_proj.parameters() if p.requires_grad])
            if self.model.logit_scale.requires_grad:
                baseline_params.append(self.model.logit_scale)
            
            # Use utils optimization functions
            self.optim = build_optimizer(baseline_params, config.optim)
            self.sched = build_lr_scheduler(self.optim, config.optim)
            
        self.scaler = GradScaler() if config.adapter.prec == "amp" else None

    def forward_backward(self, batch):
        """Forward pass and backward pass with loss computation."""
        from typing import cast
        model = cast(CustomCLIP, self.model)
        features, labels = batch
        # Convert to tensors and move to device
        features = torch.tensor(features, dtype=torch.float32).to(self.device)
        if not torch.is_tensor(labels):
            labels = torch.tensor(labels).to(self.device)
        else:
            labels = labels.detach().clone().to(self.device)
        # Apply visual projection (features are already extracted)
        if features.dtype != model.visual_proj.weight.dtype:
            features = features.to(dtype=model.visual_proj.weight.dtype)
        projected_features = model.visual_proj(features)
        # Get prototypes
        prototypes = model.forward_prototypes()
        # Ensure same dtype and device
        if projected_features.dtype != prototypes.dtype:
            prototypes = prototypes.to(dtype=projected_features.dtype)
        if projected_features.device != prototypes.device:
            prototypes = prototypes.to(device=projected_features.device)
        # Compute similarity logits
        features_norm = projected_features / projected_features.norm(dim=-1, keepdim=True)
        prototypes_norm = prototypes / prototypes.norm(dim=-1, keepdim=True)
        logits = model.logit_scale.exp() * features_norm @ prototypes_norm.t()
        # Compute loss
        loss = self.compute_loss(logits, labels)
        # Backward pass
        self._backward_and_update(loss)
        # Compute accuracies for logging
        with torch.no_grad():
            # Training accuracy
            acc_train = compute_accuracy(logits, labels)[0]
            # Test accuracy (using stored test features)
            test_features = self.features_test.to(self.device)
            if test_features.dtype != model.visual_proj.weight.dtype:
                test_features = test_features.to(dtype=model.visual_proj.weight.dtype)
            test_projected = model.visual_proj(test_features)
            test_prototypes = model.forward_prototypes()
            if test_projected.dtype != test_prototypes.dtype:
                test_prototypes = test_prototypes.to(dtype=test_projected.dtype)
            test_features_norm = test_projected / test_projected.norm(dim=-1, keepdim=True)
            test_prototypes_norm = test_prototypes / test_prototypes.norm(dim=-1, keepdim=True)
            test_logits = model.logit_scale.exp() * test_features_norm @ test_prototypes_norm.t()
            acc_test = compute_accuracy(test_logits, self.labels_test.to(self.device))[0]
        return {
            "loss": loss.item(),
            "acc_train": acc_train,
            "acc_test": acc_test
        }

    def compute_loss(self, logits, labels):
        from typing import cast
        model = cast(CustomCLIP, self.model)
        """Compute loss including GP KL term and visual projection regularization."""
        # Cross-entropy loss
        ce_loss = F.cross_entropy(logits, labels)
        total_loss = ce_loss

        # Add GP KL divergence if using GP
        if self.config.adapter.use_gp and model.gp_weighter is not None:
            kl_loss = model.get_gp_kl_divergence()
            if kl_loss is not None:
                beta = self.config.adapter.gp_beta
                total_loss += beta * kl_loss
        # Add visual projection regularization
        l2_lambda = self.config.adapter.l2_lambda
        if l2_lambda > 0:
            shots = self.config.dataset.num_shots
            identity = torch.eye(
                model.visual_proj.weight.size(0),
                device=model.visual_proj.weight.device,
                dtype=model.visual_proj.weight.dtype
            )
            diff = torch.norm(model.visual_proj.weight - identity, p='fro') ** 2
            total_loss += l2_lambda * diff / (labels.size(0) * shots)
        return total_loss

    def model_inference(self, input_data):
        """Model inference with GP sampling during evaluation."""
        from typing import cast
        model = cast(CustomCLIP, self.model)
        if not model.training and model.gp_weighter is not None:
            # Use GP sampling for evaluation
            num_samples = self.config.adapter.gp_num_mc_samples
            return model.sample_forward(input_data, num_samples)
        else:
            return model(input_data)

    def _backward_and_update(self, loss):
        """Backward pass and optimizer step."""
        self.optim.zero_grad()
        if self.scaler:
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optim)
            self.scaler.update()
        else:
            loss.backward()
            self.optim.step()

    def parse_batch_train(self, batch):
        input_data = batch["img"]
        labels = batch["label"]
        input_data = input_data.to(self.device)
        labels = labels.to(self.device)
        return input_data, labels

    def train(self):
        """Training loop with feature extraction and evaluation."""
        # Build model first (this is normally done in BaseTrainer.train())
        self.build_model()
        
        self.set_model_mode("eval")

        # Feature extraction on test set
        self.labels_test, output_test, self.features_test = self.extract_features(partition="test")
        print("Zero-Shot accuracy on test: " + 
              str(round(compute_accuracy(output_test.cuda(), self.labels_test.cuda())[0], 2)))

        # Feature extraction on training set
        self.labels_train, logits_zs, self.features_train = self.extract_features(partition="train")

        # Run the actual training using the base class training loop
        self.before_train()
        for self.epoch in range(self.start_epoch, self.max_epoch):
            self.before_epoch()
            self.run_epoch()
            self.after_epoch()
        self.after_train()

    def run_epoch(self):
        """Run one training epoch."""
        losses = AverageMeter()
        batch_time = AverageMeter()
        data_time = AverageMeter()

        # Set model to train mode but keep encoders frozen
        self.set_model_mode("train")
        try:
            if hasattr(self.model, "image_encoder"):
                self.model.image_encoder.eval()  # type: ignore
        except AttributeError:
            pass
        try:
            if hasattr(self.model, "text_encoder"):
                self.model.text_encoder.eval()  # type: ignore
        except AttributeError:
            pass

        # Set number of batches to sample
        self.num_batches = len(self.train_loader_x)
        self.batch_size = self.train_loader_x.batch_size or 1

        # Set features (ensure attributes exist)
        if not hasattr(self, 'features_train') or not hasattr(self, 'labels_train'):
            raise RuntimeError("features_train and labels_train must be extracted before training")
        
        features = self.features_train.clone().cpu().numpy()
        labels = self.labels_train.clone()

        # Randomly shuffle
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
            if meet_freq or only_few_batches:
                nb_remain = 0
                nb_remain += self.num_batches - self.batch_idx - 1
                nb_remain += (self.max_epoch - self.epoch - 1) * self.num_batches
                eta_seconds = batch_time.avg * nb_remain
                eta = str(datetime.timedelta(seconds=int(eta_seconds)))

                if (self.epoch + 1) % 10 == 0 or self.epoch == 0:
                    info = []
                    info += [f"epoch [{self.epoch + 1}/{self.max_epoch}]"]
                    info += [f"batch [{self.batch_idx + 1}/{self.num_batches}]"]
                    info += [f"loss {loss_summary['loss']:.4f}"]
                    info += [f"eta {eta}"]
                    print(" ".join(info))

            n_iter = self.epoch * self.num_batches + self.batch_idx
            self.write_scalar("train/loss", loss_summary['loss'], n_iter)
            self.write_scalar("train/lr", self.get_current_lr(), n_iter)

            end = time.time()

        return loss_summary

    def extract_features(self, partition="train", reps=1, transforms=None):
        """Extract features from specified data partition."""
        import copy
        print("Extracting features from: " + partition)
        self.set_model_mode("eval")

        if partition == "train":
            # Copy safe version of training dataloader
            data_loader = copy.deepcopy(self.train_loader_x)
            # Set data loader with drop last to false for not losing samples
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
                        if hasattr(self.model, 'forward_features'):
                            features = self.model.forward_features(input_data)  # type: ignore
                            logits = self.model(input_data)  # type: ignore
                        else:
                            # Fallback if model doesn't have forward_features
                            logits = self.model(input_data)  # type: ignore
                            features = logits  # Use logits as features
                    
                    labels_ds.append(labels.cpu())
                    logits_ds.append(logits.cpu())
                    features_ds.append(features.cpu())

        # Concatenate outputs
        labels_ds = torch.cat(labels_ds, dim=0)
        logits_ds = torch.cat(logits_ds, dim=0)
        features_ds = torch.cat(features_ds, dim=0)

        return labels_ds, logits_ds, features_ds
