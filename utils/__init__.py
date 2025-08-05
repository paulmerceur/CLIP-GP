"""
Utility functions for CLIP-GP project.
Replaces commonly used functions from Dassl.
"""

from .logging import setup_logger
from .metrics import compute_accuracy, AverageMeter, MetricMeter
from .reproducibility import set_random_seed
from .checkpoint import load_pretrained_weights
from .optimization import build_optimizer, build_lr_scheduler
from .config import parse_args_to_config, print_config, Config

__all__ = [
    'setup_logger',
    'compute_accuracy',
    'AverageMeter', 
    'MetricMeter',
    'set_random_seed',
    'load_pretrained_weights',
    'build_optimizer',
    'build_lr_scheduler',
    'parse_args_to_config',
    'print_config',
    'Config'
]
