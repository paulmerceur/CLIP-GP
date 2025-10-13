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
from typing import Dict, Any, Optional, Union
import numpy as np
from tqdm import tqdm
import json

from utils.metrics import compute_accuracy, compute_ece, compute_aece


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
        self.max_epoch = config.optim.max_epoch
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
        
        # Compute calibration metrics (0-1), report as percentage for consistency
        ece = compute_ece(all_outputs, all_labels)
        ece_pct = ece * 100.0
        aece = compute_aece(all_outputs, all_labels)
        aece_pct = aece * 100.0
        
        results = {
            "accuracy": accuracy,
            "macro_f1": macro_f1,
            "ece": ece_pct,
            "aece": aece_pct
        }
        
        # Print results
        print("=> result")
        print(f"* total: {len(all_labels):,}")
        print(f"* correct: {(all_outputs.argmax(dim=1) == all_labels).sum().item():,}")
        print(f"* accuracy: {accuracy:.1f}%")
        print(f"* error: {100 - accuracy:.1f}%")
        print(f"* macro_f1: {macro_f1:.1f}%")
        print(f"* ECE: {ece_pct:.2f}%")
        print(f"* AECE: {aece_pct:.2f}%")
        
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
        return {
            "top1_acc": float(acc),
            "ece": float(ece_val),
            "aece": float(aece_val),
        }

    def _write_run_summary_json(self, metrics: Dict[str, float], start_time: float) -> None:
        """Write a JSON summary for this run under output_dir/metrics.json."""
        out_dir = Path(self.output_dir)
        try:
            out_dir.mkdir(parents=True, exist_ok=True)
        except Exception:
            pass
        # Infer method for clarity
        try:
            tname = getattr(self.config, 'trainer_name', '')
            if tname == 'AdapterTipAF':
                method = 'tipaf'
            elif tname == 'AdapterCoOp':
                method = 'coop'
            elif tname == 'AdapterCoCoOp':
                method = 'cocoop'
            else:
                method = 'gp' if bool(getattr(self.config.adapter, 'use_gp', False)) else 'baseline'
        except Exception:
            method = 'baseline'
        # Prepare payload
        payload = {
            "timestamp": datetime.datetime.now().isoformat(),
            "dataset": self.config.dataset.name,
            "shots": int(self.config.dataset.num_shots),
            "seed": int(self.config.seed),
            "method": method,
            "backbone": self.config.model.backbone_name,
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
