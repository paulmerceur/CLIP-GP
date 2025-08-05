"""
Checkpoint utilities for CLIP-GP.
"""

import torch
import torch.nn as nn
from pathlib import Path
from typing import Optional, Dict, Any
import collections


def load_pretrained_weights(model: nn.Module, weight_path: str, strict: bool = True) -> None:
    """
    Load pretrained weights into model.
    
    Args:
        model: PyTorch model
        weight_path: Path to weights file
        strict: Whether to strictly enforce key matching
    """
    if not weight_path or not Path(weight_path).exists():
        print(f"Warning: Weight path '{weight_path}' does not exist")
        return
    
    print(f"Loading pretrained weights from {weight_path}")
    
    checkpoint = torch.load(weight_path, map_location='cpu')
    
    # Handle different checkpoint formats
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    elif 'model' in checkpoint:
        state_dict = checkpoint['model']
    else:
        state_dict = checkpoint
    
    # Remove module prefix if present (from DataParallel)
    new_state_dict = collections.OrderedDict()
    for k, v in state_dict.items():
        name = k[7:] if k.startswith('module.') else k
        new_state_dict[name] = v
    
    # Load weights
    missing_keys, unexpected_keys = model.load_state_dict(new_state_dict, strict=strict)
    
    if missing_keys:
        print(f"Missing keys: {missing_keys}")
    if unexpected_keys:
        print(f"Unexpected keys: {unexpected_keys}")
    
    print("Pretrained weights loaded successfully")


def save_checkpoint(
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer],
    scheduler: Optional[Any],
    epoch: int,
    save_path: str,
    **kwargs
) -> None:
    """
    Save model checkpoint.
    
    Args:
        model: PyTorch model
        optimizer: Optimizer state
        scheduler: LR scheduler state
        epoch: Current epoch
        save_path: Path to save checkpoint
        **kwargs: Additional items to save
    """
    checkpoint = {
        'epoch': epoch,
        'state_dict': model.state_dict(),
        **kwargs
    }
    
    if optimizer is not None:
        checkpoint['optimizer'] = optimizer.state_dict()
    
    if scheduler is not None:
        checkpoint['scheduler'] = scheduler.state_dict()
    
    # Create directory if needed
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    
    torch.save(checkpoint, save_path)
    print(f"Checkpoint saved to {save_path}")


def load_checkpoint(
    checkpoint_path: str,
    model: Optional[nn.Module] = None,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[Any] = None,
    map_location: str = 'cpu'
) -> Dict[str, Any]:
    """
    Load checkpoint and optionally restore model/optimizer/scheduler states.
    
    Args:
        checkpoint_path: Path to checkpoint file
        model: Model to load state into
        optimizer: Optimizer to load state into
        scheduler: Scheduler to load state into
        map_location: Device to map checkpoint to
        
    Returns:
        Checkpoint dictionary
    """
    if not Path(checkpoint_path).exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    print(f"Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=map_location)
    
    if model is not None and 'state_dict' in checkpoint:
        model.load_state_dict(checkpoint['state_dict'])
        print("Model state loaded")
    
    if optimizer is not None and 'optimizer' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer'])
        print("Optimizer state loaded")
    
    if scheduler is not None and 'scheduler' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler'])
        print("Scheduler state loaded")
    
    return checkpoint


def resume_from_checkpoint(
    checkpoint_path: str,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[Any] = None
) -> int:
    """
    Resume training from checkpoint.
    
    Args:
        checkpoint_path: Path to checkpoint file
        model: Model to restore
        optimizer: Optimizer to restore
        scheduler: Optional scheduler to restore
        
    Returns:
        Epoch to resume from
    """
    checkpoint = load_checkpoint(checkpoint_path, model, optimizer, scheduler)
    start_epoch = checkpoint.get('epoch', 0) + 1
    print(f"Resuming training from epoch {start_epoch}")
    return start_epoch
