"""
Optimization utilities for CLIP-GP.
"""

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.optim import SGD, Adam, AdamW
from torch.optim.lr_scheduler import (
    CosineAnnealingLR, StepLR, MultiStepLR, 
    ExponentialLR, ConstantLR, LinearLR
)
from typing import Any, List, Union, Optional
import math

"""
Optional Muon optimizer support
"""
try:  # Lazy optional dependency; only required if OPTIM.NAME == "muon"
    # Muon is available at: https://github.com/KellerJordan/Muon
    # Install with: pip install git+https://github.com/KellerJordan/Muon
    from muon import MuonWithAuxAdam  # type: ignore
    _HAS_MUON = True
except Exception:
    MuonWithAuxAdam = None  # type: ignore
    _HAS_MUON = False


def _ensure_single_process_distributed_initialized() -> None:
    """Initialize a 1-process default process group if none exists.

    Muon calls torch.distributed.get_world_size() during step(). In single-process
    runs without DDP this raises unless a default group exists. This makes a
    best-effort single-process init using gloo (or nccl if CUDA is available).
    """
    try:
        if not dist.is_available() or dist.is_initialized():
            return
        backend = 'nccl' if torch.cuda.is_available() else 'gloo'
        import os, tempfile
        init_file = os.path.join(tempfile.gettempdir(), f"muon_pg_{os.getpid()}")
        init_method = f"file://{init_file}"
        dist.init_process_group(backend=backend, init_method=init_method, rank=0, world_size=1)
    except Exception:
        # Fall back to gloo if nccl or file init fails
        try:
            if not dist.is_initialized():
                import os, tempfile
                init_file = os.path.join(tempfile.gettempdir(), f"muon_pg_{os.getpid()}_gloo")
                init_method = f"file://{init_file}"
                dist.init_process_group(backend='gloo', init_method=init_method, rank=0, world_size=1)
        except Exception:
            # Ignore; Muon will raise a clear error if it still needs dist
            pass


def build_optimizer(parameters, config) -> torch.optim.Optimizer:
    """
    Build optimizer from configuration.
    
    Args:
        parameters: Model parameters to optimize
        config: Optimization configuration
        
    Returns:
        Configured optimizer
    """
    optimizer_name = config.name.lower()
    lr = config.lr
    weight_decay = getattr(config, 'weight_decay', 0.0)
    
    if optimizer_name == "sgd":
        momentum = getattr(config, 'momentum', 0.9)
        nesterov = getattr(config, 'nesterov', False)
        return SGD(
            parameters,
            lr=lr,
            weight_decay=weight_decay,
            momentum=momentum,
            nesterov=nesterov
        )
    
    elif optimizer_name == "adam":
        betas = getattr(config, 'betas', (0.9, 0.999))
        eps = getattr(config, 'eps', 1e-8)
        return Adam(
            parameters,
            lr=lr,
            weight_decay=weight_decay,
            betas=betas,
            eps=eps
        )
    
    elif optimizer_name == "adamw":
        betas = getattr(config, 'betas', (0.9, 0.999))
        eps = getattr(config, 'eps', 1e-8)
        return AdamW(
            parameters,
            lr=lr,
            weight_decay=weight_decay,
            betas=betas,
            eps=eps
        )
    
    elif optimizer_name == "muon":
        if not _HAS_MUON:
            raise ImportError(
                "Muon optimizer requested but the 'muon' package is not installed. "
                "Install it with: pip install git+https://github.com/KellerJordan/Muon"
            )
        _ensure_single_process_distributed_initialized()
        # Split parameters by dimensionality: >=2D (weights) -> Muon, <2D -> auxiliary AdamW
        params_list = list(parameters)
        muon_params = [p for p in params_list if getattr(p, 'ndim', 0) >= 2 and p.requires_grad]
        aux_params = [p for p in params_list if getattr(p, 'ndim', 0) < 2 and p.requires_grad]

        # Allow overriding aux optimizer hyper-params from the same config fields
        betas = getattr(config, 'betas', (0.9, 0.999))
        eps = getattr(config, 'eps', 1e-8)
        aux_lr = getattr(config, 'aux_lr', lr)
        aux_weight_decay = getattr(config, 'aux_weight_decay', weight_decay)

        param_groups = []
        if len(muon_params) > 0:
            param_groups.append({
                'params': muon_params,
                'lr': lr,
                'weight_decay': weight_decay,
                'use_muon': True
            })
        if len(aux_params) > 0:
            param_groups.append({
                'params': aux_params,
                'lr': aux_lr,
                'weight_decay': aux_weight_decay,
                'betas': betas,
                'eps': eps,
                'use_muon': False
            })

        return MuonWithAuxAdam(param_groups)  # type: ignore
    
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer_name}")


def build_optimizer_from_param_groups(param_groups: List[dict], config) -> torch.optim.Optimizer:
    """
    Build an optimizer from pre-constructed parameter groups.

    This is useful when different groups need different hyper-parameters
    (e.g., base params vs GP params). Honors config for optimizer selection
    and common hyper-parameters.

    Args:
        param_groups: List of parameter group dicts
        config: Optimization configuration (expects .name, .betas, .eps, .momentum)

    Returns:
        Instantiated torch optimizer
    """
    name = getattr(config, 'name', 'sgd').lower()
    if name == 'sgd':
        momentum = getattr(config, 'momentum', 0.9)
        nesterov = getattr(config, 'nesterov', False)
        return SGD(param_groups, momentum=momentum, nesterov=nesterov)
    elif name == 'adam':
        betas = getattr(config, 'betas', (0.9, 0.999))
        eps = getattr(config, 'eps', 1e-8)
        return Adam(param_groups, betas=betas, eps=eps)
    elif name == 'adamw':
        betas = getattr(config, 'betas', (0.9, 0.999))
        eps = getattr(config, 'eps', 1e-8)
        return AdamW(param_groups, betas=betas, eps=eps)
    elif name == 'muon':
        if not _HAS_MUON:
            raise ImportError(
                "Muon optimizer requested but the 'muon' package is not installed. "
                "Install it with: pip install git+https://github.com/KellerJordan/Muon"
            )
        _ensure_single_process_distributed_initialized()
        # Transform incoming param_groups by splitting each into Muon vs auxiliary
        transformed: List[dict] = []
        betas = getattr(config, 'betas', (0.9, 0.999))
        eps = getattr(config, 'eps', 1e-8)
        for group in param_groups:
            group_params = [p for p in group.get('params', []) if p is not None and p.requires_grad]
            if len(group_params) == 0:
                continue
            group_lr = group.get('lr', getattr(config, 'lr', 1e-3))
            group_wd = group.get('weight_decay', getattr(config, 'weight_decay', 0.0))

            muon_params = [p for p in group_params if getattr(p, 'ndim', 0) >= 2]
            aux_params = [p for p in group_params if getattr(p, 'ndim', 0) < 2]

            if len(muon_params) > 0:
                transformed.append({
                    'params': muon_params,
                    'lr': group_lr,
                    'weight_decay': group_wd,
                    'use_muon': True
                })
            if len(aux_params) > 0:
                transformed.append({
                    'params': aux_params,
                    'lr': group_lr,
                    'weight_decay': group_wd,
                    'betas': betas,
                    'eps': eps,
                    'use_muon': False
                })

        return MuonWithAuxAdam(transformed)  # type: ignore
    else:
        raise ValueError(f"Unsupported optimizer: {name}")


def build_lr_scheduler(optimizer: torch.optim.Optimizer, config) -> Any:
    """
    Build learning rate scheduler from configuration.
    
    Args:
        optimizer: Optimizer to schedule
        config: Optimization configuration
        
    Returns:
        Configured scheduler
    """
    scheduler_name = getattr(config, 'lr_scheduler', 'constant').lower()
    max_epoch = config.max_epoch
    
    if scheduler_name == "cosine":
        eta_min = getattr(config, 'eta_min', 0.0)
        return CosineAnnealingLR(
            optimizer,
            T_max=max_epoch,
            eta_min=eta_min
        )
    
    elif scheduler_name == "step":
        step_size = getattr(config, 'step_size', max_epoch // 3)
        gamma = getattr(config, 'gamma', 0.1)
        return StepLR(
            optimizer,
            step_size=step_size,
            gamma=gamma
        )
    
    elif scheduler_name == "multistep":
        milestones = getattr(config, 'milestones', [max_epoch // 2, max_epoch * 3 // 4])
        gamma = getattr(config, 'gamma', 0.1)
        return MultiStepLR(
            optimizer,
            milestones=milestones,
            gamma=gamma
        )
    
    elif scheduler_name == "exponential":
        gamma = getattr(config, 'gamma', 0.95)
        return ExponentialLR(
            optimizer,
            gamma=gamma
        )
    
    elif scheduler_name == "constant":
        return ConstantLR(optimizer, factor=1.0)
    
    elif scheduler_name == "linear":
        start_factor = getattr(config, 'start_factor', 1.0)
        end_factor = getattr(config, 'end_factor', 0.0)
        total_iters = getattr(config, 'total_iters', max_epoch)
        return LinearLR(
            optimizer,
            start_factor=start_factor,
            end_factor=end_factor,
            total_iters=total_iters
        )
    
    else:
        raise ValueError(f"Unsupported scheduler: {scheduler_name}")


class WarmupWrapper:
    """
    Wrapper to add warmup to any scheduler.
    """
    
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        scheduler: Any,
        warmup_epochs: int = 0,
        warmup_type: str = "constant",
        warmup_factor: float = 0.1
    ):
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.warmup_epochs = warmup_epochs
        self.warmup_type = warmup_type.lower()
        self.warmup_factor = warmup_factor
        self.base_lrs = [group['lr'] for group in optimizer.param_groups]
        self.epoch = 0
    
    def step(self, epoch: Optional[int] = None):
        """Step the scheduler"""
        if epoch is not None:
            self.epoch = epoch
        else:
            self.epoch += 1
        
        if self.epoch < self.warmup_epochs:
            self._warmup_step()
        else:
            self.scheduler.step()
    
    def _warmup_step(self):
        """Apply warmup learning rate"""
        if self.warmup_type == "constant":
            factor = self.warmup_factor
        elif self.warmup_type == "linear":
            factor = self.warmup_factor + (1.0 - self.warmup_factor) * self.epoch / self.warmup_epochs
        else:
            raise ValueError(f"Unsupported warmup type: {self.warmup_type}")
        
        for i, param_group in enumerate(self.optimizer.param_groups):
            param_group['lr'] = self.base_lrs[i] * factor
    
    def state_dict(self):
        """Get scheduler state"""
        return {
            'scheduler': self.scheduler.state_dict(),
            'epoch': self.epoch,
            'base_lrs': self.base_lrs
        }
    
    def load_state_dict(self, state_dict):
        """Load scheduler state"""
        self.scheduler.load_state_dict(state_dict['scheduler'])
        self.epoch = state_dict['epoch']
        self.base_lrs = state_dict['base_lrs']


def build_lr_scheduler_with_warmup(optimizer: torch.optim.Optimizer, config) -> Any:
    """
    Build learning rate scheduler with optional warmup.
    
    Args:
        optimizer: Optimizer to schedule
        config: Optimization configuration
        
    Returns:
        Configured scheduler (possibly wrapped with warmup)
    """
    scheduler = build_lr_scheduler(optimizer, config)
    
    warmup_epochs = getattr(config, 'warmup_epoch', 0)
    if warmup_epochs > 0:
        warmup_type = getattr(config, 'warmup_type', 'constant')
        warmup_factor = getattr(config, 'warmup_cons_lr', 1e-5) / config.lr
        
        scheduler = WarmupWrapper(
            optimizer=optimizer,
            scheduler=scheduler,
            warmup_epochs=warmup_epochs,
            warmup_type=warmup_type,
            warmup_factor=warmup_factor
        )
    
    return scheduler
