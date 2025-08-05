"""
Simple data manager for CLIP-GP project.
Replaces Dassl's DataManager with a lightweight implementation.
Phase 3: Complete Dassl removal.
"""

import torch
from torch.utils.data import DataLoader
from typing import Optional
from pathlib import Path

# Import our custom transforms and dataset utilities
from .transforms import build_transform
from .dataset_base import build_dataset, TorchDatasetWrapper


class SimpleDataManager:
    """Lightweight data manager replacing Dassl's DataManager"""
    
    def __init__(self, config):
        """Initialize data manager with config"""
        self.config = config
        
        # Build dataset using our custom infrastructure  
        self.dataset = build_dataset(config)
        
        # Build transforms
        self.tfm_train = build_transform(config, is_train=True)
        self.tfm_test = build_transform(config, is_train=False)
        
        # Build data loaders
        self.train_loader_x = self._build_data_loader(
            data_source=self.dataset.train_x,
            batch_size=config.dataloader.batch_size_train,
            is_train=True,
            transform=self.tfm_train
        )
        
        self.test_loader = self._build_data_loader(
            data_source=self.dataset.test,
            batch_size=config.dataloader.batch_size_test,
            is_train=False,
            transform=self.tfm_test
        )
        
        # Optional validation loader
        if hasattr(self.dataset, 'val') and self.dataset.val:
            self.val_loader = self._build_data_loader(
                data_source=self.dataset.val,
                batch_size=config.dataloader.batch_size_test,
                is_train=False,
                transform=self.tfm_test
            )
        else:
            self.val_loader = None
        
        # Set dataset attributes
        self.num_classes = self.dataset.num_classes
        self.lab2cname = self.dataset.lab2cname
        
        # Print dataset information
        self._print_dataset_info()
    
    def _build_data_loader(self, data_source, batch_size, is_train, transform):
        """Build a data loader"""
        if not data_source:
            return None
        
        # Use our custom TorchDatasetWrapper
        dataset = TorchDatasetWrapper(
            data_source=data_source,
            transform=transform,
            is_train=is_train
        )
        
        data_loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=is_train,
            num_workers=self.config.dataloader.num_workers,
            drop_last=is_train and len(data_source) >= batch_size,
            pin_memory=(torch.cuda.is_available() and self.config.use_cuda)
        )
        
        return data_loader
    
    def _print_dataset_info(self):
        """Print dataset information"""
        dataset_name = getattr(self.dataset, 'dataset_name', self.dataset.__class__.__name__)
        print(f"---------  ----------")
        print(f"Dataset    {dataset_name}")
        print(f"# classes  {self.num_classes}")
        print(f"# train_x  {len(self.dataset.train_x)}")
        if hasattr(self.dataset, 'val') and self.dataset.val:
            print(f"# val      {len(self.dataset.val)}")
        print(f"# test     {len(self.dataset.test)}")
        print(f"---------  ----------")


def build_data_manager(config):
    """Build data manager from config"""
    return SimpleDataManager(config)
