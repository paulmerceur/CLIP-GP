"""
Trainer registry for CLIP-GP.
"""

from typing import Dict, Type, Any
from utils.trainer import BaseTrainer


class TrainerRegistry:
    """Simple trainer registry"""
    
    def __init__(self):
        self._trainers: Dict[str, Type[BaseTrainer]] = {}
    
    def register(self, name: str):
        """Decorator to register a trainer"""
        def wrapper(trainer_cls: Type[BaseTrainer]):
            self._trainers[name] = trainer_cls
            return trainer_cls
        return wrapper
    
    def get(self, name: str) -> Type[BaseTrainer]:
        """Get a trainer class by name"""
        if name not in self._trainers:
            raise ValueError(f"Unknown trainer: {name}. Available: {list(self._trainers.keys())}")
        return self._trainers[name]
    
    def list_trainers(self):
        """List all registered trainers"""
        return list(self._trainers.keys())


# Global trainer registry
TRAINER_REGISTRY = TrainerRegistry()


def build_trainer(config, dataset_manager):
    """Build trainer from config"""
    trainer_name = config.trainer_name
    trainer_cls = TRAINER_REGISTRY.get(trainer_name)
    trainer = trainer_cls(config, dataset_manager)
    return trainer
