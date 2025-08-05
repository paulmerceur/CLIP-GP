"""
Simple data manager for CLIP-GP project.
Replaces Dassl's DataManager with a lightweight implementation.
"""

import torch
from torch.utils.data import DataLoader
from typing import Optional
from pathlib import Path

# Import transforms 
from dassl.data.transforms import build_transform

# Import existing dataset building functionality 
from dassl.data.datasets import build_dataset


class SimpleDataManager:
    """Lightweight data manager replacing Dassl's DataManager"""
    
    def __init__(self, config):
        """Initialize data manager with config"""
        self.config = config
        
        # Convert config to Dassl format for dataset building
        dassl_cfg = self._config_to_dassl_format(config)
        
        # Build dataset using existing Dassl infrastructure
        self.dataset = build_dataset(dassl_cfg)
        
        # Build transforms
        self.tfm_train = build_transform(dassl_cfg, is_train=True)
        self.tfm_test = build_transform(dassl_cfg, is_train=False)
        
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
    
    def _config_to_dassl_format(self, config):
        """Convert our config to Dassl format for compatibility"""
        from dassl.config import get_cfg_default
        from yacs.config import CfgNode as CN
        
        cfg = get_cfg_default()
        
        # Dataset configuration
        cfg.DATASET.NAME = config.dataset.name
        cfg.DATASET.ROOT = config.dataset.root
        cfg.DATASET.NUM_SHOTS = config.dataset.num_shots
        cfg.DATASET.SUBSAMPLE_CLASSES = config.dataset.subsample_classes
        
        if config.dataset.source_domains:
            cfg.DATASET.SOURCE_DOMAINS = config.dataset.source_domains
        if config.dataset.target_domains:
            cfg.DATASET.TARGET_DOMAINS = config.dataset.target_domains
        
        # DataLoader configuration
        cfg.DATALOADER.TRAIN_X.BATCH_SIZE = config.dataloader.batch_size_train
        cfg.DATALOADER.TEST.BATCH_SIZE = config.dataloader.batch_size_test
        cfg.DATALOADER.NUM_WORKERS = config.dataloader.num_workers
        
        # Input configuration
        cfg.INPUT.SIZE = config.input.size
        cfg.INPUT.INTERPOLATION = config.input.interpolation
        cfg.INPUT.PIXEL_MEAN = list(config.input.pixel_mean)
        cfg.INPUT.PIXEL_STD = list(config.input.pixel_std)
        cfg.INPUT.TRANSFORMS = config.input.transforms
        
        # Other required settings
        cfg.USE_CUDA = config.use_cuda
        cfg.SEED = config.seed
        
        cfg.freeze()
        return cfg
    
    def _build_data_loader(self, data_source, batch_size, is_train, transform):
        """Build a data loader"""
        if not data_source:
            return None
        
        # Use Dassl's DatasetWrapper for compatibility
        from dassl.data.data_manager import DatasetWrapper
        
        # Convert config to Dassl format for DatasetWrapper
        dassl_cfg = self._config_to_dassl_format(self.config)
        
        dataset = DatasetWrapper(
            cfg=dassl_cfg,
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
