"""
Base trainer implementation for CLIP-GP.
"""

import os
import time
import datetime
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path
from typing import Dict, Optional
import numpy as np
from tqdm import tqdm
import json

from clip import clip
from utils.metrics import compute_accuracy, compute_ece, compute_aece, compute_ece_with_bins, compute_aece_with_bins
from datasets.imagenet_templates import IMAGENET_TEMPLATES_SELECT, IMAGENET_TEMPLATES

CUSTOM_TEMPLATES = {
    "Caltech101": [
        "a photo of a {}.",
        "a drawing of a {}.",
        "a sculpture of a {}.",
        "a low-resolution photo of the {}.",
        "a cropped photo of the {}.",
        "a bad photo of a {}.",
        "a blurry photo of a {}.",
        "a low resolution photo of the {}.",
        "a cropped photo of the {}.",
        "a dark photo of a {}.",
        "a detailed photo of a {}.",
        "a close-up photo of a {} object.",
        "a clear photograph of a {} on display.",
        "a professional photo showing a {} in focus.",
        "an image that contains a {} as the main subject.",
    ],
    "OxfordPets": [
        "a photo of a {}, a type of pet.",
        "a close-up photo of a {}, a type of pet.",
        "a photo of my {}.",
        "a photo of a {} sleeping.",
        "a blurry photo of the {}.",
        "a bad photo of a {}, a type of pet.",
        "a blurry photo of a {}, a type of pet.",
        "a low resolution photo of the {}, a type of pet.",
        "a photo of a hard to see {}, a type of pet.",
        "a dark photo of the {}, a type of pet.",
        "a close-up photo of a {}, a domestic animal.",
        "a photo of a {}, a cute pet.",
        "a picture showing a {}, often kept as a pet.",
        "a portrait photo of a {}, a common household animal.",
        "an image of a {}, a type of pet.",
    ],
    "OxfordFlowers": [
        "a photo of a {}, a type of flower.",
        "a close-up photo of a {}, a type of flower.",
        "a painting of the {}.",
        "a photo of a {} in a garden.",
        "a bright photo of the {}.",
        "a bad photo of a {}, a type of flower.",
        "a blurry photo of a {}, a type of flower.",
        "a low resolution photo of the {}, a type of flower.",
        "a cropped photo of the {}, a type of flower.",
        "a dark photo of the {}, a type of flower.",
        "a close-up photo of a {}, a kind of flower.",
        "a macro shot of {}, showing its petals.",
        "a bright and colorful photo of {}, a species of flower.",
        "a picture of {}, a beautiful blooming flower.",
        "an artistic photo of {}, a type of flower.",
    ],
    "FGVCAircraft": [
        "a photo of a {}, a type of aircraft.",
        "a photo of a {} in flight.",
        "a photo of the {} on a runway.",
        "a drawing of the {}.",
        "a photo of a {} taking off.",
        "a bad photo of a {}, a type of aircraft.",
        "a blurry photo of a {}, a type of aircraft.",
        "a low resolution photo of the {}, a type of aircraft.",
        "a photo of a hard to see {}, a type of aircraft.",
        "a dark photo of the {}, a type of aircraft.",
        "a photo of a {}, a model of aircraft.",
        "an image of a {}, a type of airplane.",
        "a picture showing a {} flying in the sky.",
        "a side-view photo of a {}, a kind of aircraft.",
        "a photograph of a {}, seen on the runway.",
    ],
    "DescribableTextures": [
        "{} texture.",
        "a photo of a {} texture.",
        "a close-up of a {} pattern.",
        "a surface that is {}.",
        "an image of something {}.",
        "a blurry {} texture.",
        "a low resolution {} texture.",
        "a pixelated {} texture.",
        "a corrupted {} texture.",
        "a dark {} texture.",
        "a {} texture pattern.",
        "a close-up of a surface with {} texture.",
        "a detailed macro photo showing {} texture.",
        "a repeating pattern with {} characteristics.",
        "an image that captures {} texture.",
    ],
    "EuroSAT": [
        "a centered satellite photo of {}.",
        "a satellite image of {}.",
        "an aerial view of {}.",
        "a low-resolution satellite photo of {}.",
        "a photo from space showing {}.",
        "a low resolution centered satellite photo of {}.",
        "a blurry centered satellite photo of {}.",
        "a noisy centered satellite photo of {}.",
        "a partially obscured centered satellite photo of {}.",
        "a dark centered satellite photo of {}.",
        "a centered satellite photo of {} landscape.",
        "a satellite view showing {} terrain.",
        "an overhead image depicting {} from space.",
        "a top-down photo showing {} region.",
        "a satellite image of an area with {} features.",
    ],
    "StanfordCars": [
        "a photo of a {}.",
        "a photo of a {} driving.",
        "a photo of a {} parked.",
        "a close-up photo of the {}.",
        "a bad photo of the {}.",
        "a bad photo of a {}.",
        "a blurry photo of a {}.",
        "a low resolution photo of the {}.",
        "a cropped photo of the {}.",
        "a dark photo of a {}.",
        "a photo of a {}, a model of car.",
        "a picture of a {}, parked or on the road.",
        "a side-view photo of {}, an automobile.",
        "a detailed image showing {}, a car model.",
        "a photograph of a {}, viewed from the front.",
    ],
    "Food101": [
        "a photo of {}, a type of food.",
        "a close-up photo of {}, a type of food.",
        "a photo of a plate of {}.",
        "a delicious-looking {}.",
        "a bad photo of {}.",
        "a bad photo of {}, a type of food.",
        "a blurry photo of {}, a type of food.",
        "a low resolution photo of the {}, a type of food.",
        "a cropped photo of the {}, a type of food.",
        "a dark photo of the {}, a type of food.",
        "a photo of {}, a delicious dish.",
        "a close-up photo of {}, a kind of food.",
        "a picture showing a serving of {}, a meal.",
        "a professional photo of {}, a popular cuisine.",
        "an image of {}, ready to eat.",
    ],
    "UCF101": [
        "a photo of a person doing {}.",
        "a still frame from a video of someone {}.",
        "a photo of someone {}.",
        "a blurry photo of a person {}.",
        "a person in the middle of {}.",
        "a bad photo of a person doing {}.",
        "a blurry photo of a person doing {}.",
        "a low resolution photo of a person doing {}.",
        "a cropped photo of a person doing {}.",
        "a dark photo of a person doing {}.",
        "a photo of a person performing {}.",
        "a snapshot showing someone doing {} activity.",
        "an image of a human in the act of {}.",
        "a photo capturing a person engaged in {}.",
        "a picture of an athlete doing {}.",
    ],
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


def _get_templates(config):
    if config.adapter.use_custom_templates:
        return CUSTOM_TEMPLATES[config.dataset.name]
        
    templates = ["a photo of a {}."]
    if config.adapter.num_templates > 1:
        templates += IMAGENET_TEMPLATES_SELECT[: config.adapter.num_templates - 1]
    if config.adapter.num_templates > 1 + len(IMAGENET_TEMPLATES_SELECT):
        templates += IMAGENET_TEMPLATES[: config.adapter.num_templates - 1 - len(IMAGENET_TEMPLATES_SELECT)]
    print(f"Templates: {templates}")
    return templates


@torch.no_grad()
def _get_clip_weights(classnames, clip_model, templates):
    device = next(clip_model.parameters()).device
    dtype = next(clip_model.parameters()).dtype
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
    return zeroshot_weights.to(dtype=dtype)


class BaseTrainer:
    """Custom base trainer"""
    
    def __init__(self, config, dataset_manager):
        """Initialize trainer with config and dataset manager"""
        self.config = config
        self.dm = dataset_manager
        
        # Device setup
        if torch.cuda.is_available() and config.use_cuda:
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")
        
        # Training state
        self.start_epoch = self.epoch = 0
        self.max_epoch = config.adapter.clip_adapter_epochs
        self.output_dir = config.output_dir
        
        # Data loaders from dataset manager
        self.train_loader_x = dataset_manager.train_loader_x
        self.val_loader = dataset_manager.val_loader
        self.test_loader = dataset_manager.test_loader
        self.num_classes = dataset_manager.num_classes
        self.lab2cname = dataset_manager.lab2cname
        
        # Training state tracking
        self.best_result = -np.inf
        self.time_start = None
        
        # TensorBoard writer
        self._writer = None
        
        # Model components (to be set by subclasses)
        self.model: Optional[nn.Module] = None
        self.optim = None
        self.sched = None
        
        # Metrics for batch tracking
        self.batch_idx = 0
        self.num_batches = 0
        self.batch_size = 0
    
    def build_model(self):
        """Build model - to be implemented by subclasses"""
        raise NotImplementedError("Subclasses must implement build_model")
    
    def forward_backward(self, batch):
        """Forward and backward pass - to be implemented by subclasses"""
        raise NotImplementedError("Subclasses must implement forward_backward")
    
    def model_inference(self, input_data):
        """Model inference for evaluation"""
        if self.model is None:
            raise RuntimeError("Model not initialized")
        return self.model(input_data)
    
    def parse_batch_train(self, batch):
        """Parse training batch"""
        input_data = batch["img"]
        labels = batch["label"]
        input_data = input_data.to(self.device)
        labels = labels.to(self.device)
        return input_data, labels
    
    def parse_batch_test(self, batch):
        """Parse test batch"""
        input_data = batch["img"]
        labels = batch["label"]
        input_data = input_data.to(self.device)
        labels = labels.to(self.device)
        return input_data, labels
    
    def set_model_mode(self, mode="train"):
        """Set model mode"""
        if self.model is None:
            return
            
        if mode == "train":
            self.model.train()
        elif mode in ["test", "eval"]:
            self.model.eval()
        else:
            raise KeyError(f"Unknown mode: {mode}")
    
    def get_current_lr(self):
        """Get current learning rate"""
        if self.optim is None:
            return 0.0
        return self.optim.param_groups[0]["lr"]
    
    def init_writer(self, log_dir):
        """Initialize TensorBoard writer"""
        if self._writer is None:
            print(f"Initialize tensorboard (log_dir={log_dir})")
            self._writer = SummaryWriter(log_dir=log_dir)
    
    def close_writer(self):
        """Close TensorBoard writer"""
        if self._writer is not None:
            self._writer.close()
    
    def write_scalar(self, tag, scalar_value, global_step=None):
        """Write scalar to TensorBoard"""
        if self._writer is not None:
            self._writer.add_scalar(tag, scalar_value, global_step)
    
    def save_model(self, epoch, output_dir, val_result=None, model_name="model.pth.tar"):
        """Save model checkpoint"""
        if self.model is None or self.optim is None:
            return
            
        save_dir = Path(output_dir) / "adapter"
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # Add epoch to filename if it's the default name
        if model_name == "model.pth.tar":
            model_name = f"model.pth.tar-{epoch + 1}"
        
        save_path = save_dir / model_name
        
        state_dict = {
            "state_dict": self.model.state_dict(),
            "epoch": epoch + 1,
            "optimizer": self.optim.state_dict(),
        }
        
        if self.sched is not None:
            state_dict["scheduler"] = self.sched.state_dict()
        
        if val_result is not None:
            state_dict["val_result"] = val_result
        
        torch.save(state_dict, save_path)
        print(f"Checkpoint saved to {save_path}")
    
    def load_model(self, model_dir, epoch=None):
        """Load model checkpoint"""
        if epoch is not None:
            model_path = Path(model_dir) / "adapter" / f"model.pth.tar-{epoch}"
        else:
            # Find the latest checkpoint
            adapter_dir = Path(model_dir) / "adapter"
            if not adapter_dir.exists():
                return 0
            
            # Look for model-best.pth.tar first, then latest
            best_path = adapter_dir / "model-best.pth.tar"
            if best_path.exists():
                model_path = best_path
            else:
                # Find latest checkpoint
                checkpoints = list(adapter_dir.glob("model.pth.tar-*"))
                if not checkpoints:
                    return 0
                model_path = max(checkpoints, key=lambda p: int(p.name.split("-")[-1]))
        
        if not model_path.exists():
            print(f"No checkpoint found at {model_path}")
            return 0
        
        print(f"Loading checkpoint from {model_path}")
        checkpoint = torch.load(model_path, map_location=self.device)
        
        if self.model is None:
            raise RuntimeError("Model not initialized")
        self.model.load_state_dict(checkpoint["state_dict"])
        
        if "optimizer" in checkpoint and self.optim is not None:
            self.optim.load_state_dict(checkpoint["optimizer"])
        
        if "scheduler" in checkpoint and self.sched is not None:
            self.sched.load_state_dict(checkpoint["scheduler"])
        
        return checkpoint.get("epoch", 0)
    
    def before_train(self):
        """Setup before training"""
        # Resume from checkpoint if specified
        if self.config.resume:
            self.start_epoch = self.load_model(self.config.resume)
        
        # Initialize TensorBoard
        if getattr(self.config.train, 'enable_tensorboard', True):
            writer_dir = Path(self.output_dir) / "tensorboard"
            writer_dir.mkdir(parents=True, exist_ok=True)
            self.init_writer(str(writer_dir))
        
        # Record start time
        self.time_start = time.time()
    
    def after_train(self):
        """Cleanup after training"""
        print("Finish training")
        
        # Test with best or final model
        if not getattr(self.config, 'no_test', False):
            if getattr(self.config, 'final_model', 'last') == "best_val":
                print("Deploy the model with the best val performance")
                self.load_model(self.output_dir)
            else:
                print("Deploy the last-epoch model")
            self.test()
        
        # Show elapsed time
        if self.time_start is not None:
            elapsed = round(time.time() - self.time_start)
            elapsed = str(datetime.timedelta(seconds=elapsed))
            print(f"Elapsed: {elapsed}")
        
        # Close writer
        self.close_writer()
    
    def before_epoch(self):
        """Setup before each epoch"""
        pass
    
    def after_epoch(self):
        """Cleanup after each epoch"""
        last_epoch = (self.epoch + 1) == self.max_epoch
        meet_checkpoint_freq = (
            (self.epoch + 1) % getattr(self.config.train, 'checkpoint_freq', 0) == 0
            if getattr(self.config.train, 'checkpoint_freq', 0) > 0 else False
        )
        
        # Save checkpoint if needed
        if (meet_checkpoint_freq or last_epoch) and getattr(self.config.train, 'enable_adapter_checkpoints', True):
            self.save_model(self.epoch, self.output_dir)
        
        # Update learning rate
        if self.sched is not None:
            self.sched.step()
    
    @torch.no_grad()
    def test(self, split=None):
        """Test the model"""
        self.set_model_mode("eval")
        
        if split is None:
            split = "test"
        
        if split == "val" and self.val_loader is not None:
            data_loader = self.val_loader
        else:
            split = "test"
            data_loader = self.test_loader
        
        print(f"Evaluate on the *{split}* set")
        
        all_outputs = []
        all_labels = []
        
        for batch_idx, batch in enumerate(tqdm(data_loader)):
            input_data, labels = self.parse_batch_test(batch)
            outputs = self.model_inference(input_data)
            
            all_outputs.append(outputs.cpu())
            all_labels.append(labels.cpu())
        
        # Concatenate all outputs and labels
        all_outputs = torch.cat(all_outputs, dim=0)
        all_labels = torch.cat(all_labels, dim=0)
        
        # Compute metrics
        accuracy_list = compute_accuracy(all_outputs, all_labels)
        accuracy = accuracy_list[0]  # Get top-1 accuracy
        
        # Compute macro F1 if available
        try:
            from sklearn.metrics import f1_score
            pred = all_outputs.argmax(dim=1).numpy()
            labels = all_labels.numpy()
            macro_f1 = f1_score(labels, pred, average='macro') * 100
        except ImportError:
            macro_f1 = 0.0
        
        # Compute calibration metrics and bins
        ece = compute_ece(all_outputs, all_labels)
        aece = compute_aece(all_outputs, all_labels)
        try:
            ece_bins_val, ece_bins = compute_ece_with_bins(all_outputs, all_labels, n_bins=10)
        except Exception:
            ece_bins_val, ece_bins = float('nan'), {"bin_acc": [], "bin_conf": [], "bin_count": []}
        try:
            aece_bins_val, aece_bins = compute_aece_with_bins(all_outputs, all_labels, n_bins=10)
        except Exception:
            aece_bins_val, aece_bins = float('nan'), {"bin_acc": [], "bin_conf": [], "bin_count": []}
        
        results = {
            "accuracy": accuracy,
            "macro_f1": macro_f1,
            "ece": ece,
            "aece": aece,
            # Include per-bin calibration for reliability diagrams
            "calibration": ece_bins,
            "adaptive_calibration": aece_bins,
        }
        
        # Print results
        print("=> result")
        print(f"* total: {len(all_labels):,}")
        print(f"* correct: {(all_outputs.argmax(dim=1) == all_labels).sum().item():,}")
        print(f"* accuracy: {accuracy:.1f}%")
        print(f"* error: {100 - accuracy:.1f}%")
        print(f"* macro_f1: {macro_f1:.1f}%")
        print(f"* ECE: {ece:.2f}%")
        print(f"* AECE: {aece:.2f}%")

        # Save summary to json file
        self._write_run_summary_json(results, start_time=self.time_start)

        
        # Write to TensorBoard
        for k, v in results.items():
            tag = f"{split}/{k}"
            self.write_scalar(tag, v, self.epoch)
        
        return accuracy

    @torch.no_grad()
    def _compute_final_metrics(self) -> Dict[str, float]:
        """Compute final test metrics in a unified way for metrics.json."""
        self.set_model_mode("eval")
        data_loader = self.test_loader
        all_outputs = []
        all_labels = []
        for batch_idx, batch in enumerate(data_loader):
            input_data, labels = self.parse_batch_test(batch)
            outputs = self.model_inference(input_data)
            all_outputs.append(outputs.cpu())
            all_labels.append(labels.cpu())
        all_outputs = torch.cat(all_outputs, dim=0)
        all_labels = torch.cat(all_labels, dim=0)
        acc = compute_accuracy(all_outputs, all_labels)[0]
        try:
            ece_val = compute_ece(all_outputs, all_labels)
        except Exception:
            ece_val = float('nan')
        try:
            aece_val = compute_aece(all_outputs, all_labels)
        except Exception:
            aece_val = float('nan')
        # Also compute per-bin arrays to store in JSON (for scripts)
        try:
            _, ece_bins = compute_ece_with_bins(all_outputs, all_labels, n_bins=10)
        except Exception:
            ece_bins = {"bin_acc": [], "bin_conf": [], "bin_count": []}
        try:
            _, aece_bins = compute_aece_with_bins(all_outputs, all_labels, n_bins=10)
        except Exception:
            aece_bins = {"bin_acc": [], "bin_conf": [], "bin_count": []}
        return {
            "top1_acc": float(acc),
            "ece": float(ece_val),
            "aece": float(aece_val),
            "calibration": ece_bins,
            "adaptive_calibration": aece_bins,
        }

    def _write_run_summary_json(self, metrics: Dict[str, float], start_time: float) -> None:
        """Write a JSON summary for this run under output_dir/metrics.json."""
        out_dir = Path(self.output_dir)

        # Infer method for clarity
        try:
            tname = getattr(self.config, 'trainer_name', '')
            if tname == 'Adapter-TipA-F':
                method = 'tipaf'
            elif tname == 'Adapter-TipA':
                method = 'tipa'
            elif tname == 'Adapter-CoOp':
                method = 'coop'
            elif tname == 'Adapter-CoCoOp':
                method = 'cocoop'
            elif tname == 'Adapter-CLIP-Adapter':
                method = 'clip-adapter'
            else:
                method = 'gp' if bool(getattr(self.config.adapter, 'use_gp', False)) else 'baseline'
        except Exception:
            method = 'baseline'

        # Get zero-shot metrics if computed earlier
        zs = getattr(self, "zero_shot_metrics", None)

        # Prepare payload
        payload = {
            "timestamp": datetime.datetime.now().isoformat(),
            "dataset": self.config.dataset.name,
            "shots": int(self.config.dataset.num_shots),
            "seed": int(self.config.seed),
            "method": method,
            "backbone": self.config.model.backbone_name,
            "zero_shot": zs,
            "metrics": metrics,
            "config": self._config_to_dict_for_json(),
            "output_dir": str(out_dir),
            "train_time_s": float(max(0.0, time.time() - start_time)),
        }
        with (out_dir / "metrics.json").open("w") as f:
            json.dump(payload, f, indent=2)

    def _config_to_dict_for_json(self) -> dict:
        try:
            from utils.config import _config_to_dict
            return _config_to_dict(self.config)
        except Exception:
            return {}
    
    def train(self):
        """Main training loop"""
        # Build model first
        self.build_model()
        
        # Training loop
        self.before_train()
        for self.epoch in range(self.start_epoch, self.max_epoch):
            self.before_epoch()
            self.run_epoch()
            self.after_epoch()
        self.after_train()
    
    def run_epoch(self):
        """Run one training epoch - to be implemented by subclasses"""
        raise NotImplementedError("Subclasses must implement run_epoch")
