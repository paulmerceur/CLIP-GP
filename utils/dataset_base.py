"""
Base dataset classes to replace Dassl dataset infrastructure.
Phase 3: Complete Dassl removal.
"""

import os
import json
import pickle
import random
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass

import torch
from torch.utils.data import Dataset
from PIL import Image


@dataclass
class Datum:
    """Data sample class - replaces dassl.data.datasets.Datum"""
    impath: str
    label: int
    domain: str = ""
    classname: str = ""
    
    def __post_init__(self):
        """Ensure impath is a string"""
        if isinstance(self.impath, Path):
            self.impath = str(self.impath)


class DatasetBase:
    """
    Base dataset class - replaces dassl.data.datasets.DatasetBase
    """
    
    dataset_dir = ""
    
    def __init__(self, train_x: Optional[List[Datum]] = None, train_u: Optional[List[Datum]] = None, 
                 val: Optional[List[Datum]] = None, test: Optional[List[Datum]] = None):
        self._train_x = train_x if train_x is not None else []
        self._train_u = train_u if train_u is not None else []
        self._val = val if val is not None else []
        self._test = test if test is not None else []
        
        self._num_classes = self._get_num_classes()
        self._lab2cname, self._classnames = self._get_label_mapping()
        
    @property
    def train_x(self) -> List[Datum]:
        return self._train_x
    
    @property 
    def train_u(self) -> List[Datum]:
        return self._train_u
        
    @property
    def val(self) -> List[Datum]:
        return self._val
        
    @property  
    def test(self) -> List[Datum]:
        return self._test
        
    @property
    def lab2cname(self) -> Dict[int, str]:
        return self._lab2cname
        
    @property
    def classnames(self) -> List[str]:
        return self._classnames
        
    @property
    def num_classes(self) -> int:
        return self._num_classes
        
    def _get_num_classes(self) -> int:
        """Get number of classes from all data splits"""
        labels = set()
        for data_source in [self._train_x, self._train_u, self._val, self._test]:
            for item in data_source:
                labels.add(item.label)
        return len(labels)
        
    def _get_label_mapping(self) -> Tuple[Dict[int, str], List[str]]:
        """Get label to class name mapping"""
        lab2cname = {}
        for data_source in [self._train_x, self._train_u, self._val, self._test]:
            for item in data_source:
                if item.label not in lab2cname and item.classname:
                    lab2cname[item.label] = item.classname
                    
        # Create sorted list of class names
        if lab2cname:
            classnames = [lab2cname[i] for i in sorted(lab2cname.keys())]
        else:
            classnames = []
            
        return lab2cname, classnames
    
    def generate_fewshot_dataset(self, data_source: List[Datum], num_shots: int = 1, 
                               repeat: bool = False) -> List[Datum]:
        """Generate few-shot dataset by sampling from data_source"""
        if num_shots < 1:
            return []
            
        # Group by class
        groups = {}
        for item in data_source:
            if item.label not in groups:
                groups[item.label] = []
            groups[item.label].append(item)
            
        # Sample few-shot data
        result = []
        for label, items in groups.items():
            if len(items) >= num_shots:
                sampled = random.sample(items, num_shots)
            else:
                # If not enough samples, take all and repeat if requested
                sampled = items[:]
                if repeat and len(sampled) < num_shots:
                    while len(sampled) < num_shots:
                        sampled.extend(random.choices(items, k=min(len(items), num_shots - len(sampled))))
            
            result.extend(sampled)
            
        return result
    
    @staticmethod
    def read_json(file_path: str) -> Any:
        """Read JSON file"""
        with open(file_path, 'r') as f:
            return json.load(f)
    
    @staticmethod
    def write_json(obj: Any, file_path: str) -> None:
        """Write object to JSON file"""
        mkdir_if_missing(os.path.dirname(file_path))
        with open(file_path, 'w') as f:
            json.dump(obj, f, indent=4, separators=(',', ': '))
    
    @staticmethod
    def read_split(split_path: str, path_prefix: str = "") -> Tuple[List[Datum], List[Datum], List[Datum]]:
        """Read data split from JSON file"""
        def _convert(items: List[List]) -> List[Datum]:
            out = []
            for impath, label, classname in items:
                impath = os.path.join(path_prefix, impath) if path_prefix else impath
                item = Datum(impath=impath, label=int(label), classname=str(classname))
                out.append(item)
            return out
        
        print(f"Reading split from {split_path}")
        split = DatasetBase.read_json(split_path)
        train = _convert(split["train"])
        val = _convert(split["val"])  
        test = _convert(split["test"])
        
        return train, val, test
        
    @staticmethod
    def save_split(train: List[Datum], val: List[Datum], test: List[Datum], 
                  split_path: str, path_prefix: str = "") -> None:
        """Save data split to JSON file"""
        def _extract(data_source: List[Datum]) -> List[List]:
            out = []
            for item in data_source:
                impath = item.impath
                if path_prefix and impath.startswith(path_prefix):
                    impath = os.path.relpath(impath, path_prefix)
                out.append([impath, item.label, item.classname])
            return out
        
        split = {
            "train": _extract(train),
            "val": _extract(val),
            "test": _extract(test)
        }
        
        DatasetBase.write_json(split, split_path)
        print(f"Saved split to {split_path}")


class TorchDatasetWrapper(Dataset):
    """
    PyTorch Dataset wrapper - replaces dassl.data.data_manager.DatasetWrapper
    """
    
    def __init__(self, data_source: List[Datum], transform=None, is_train: bool = False):
        self.data_source = data_source
        self.transform = transform
        self.is_train = is_train
        
    def __len__(self) -> int:
        return len(self.data_source)
        
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item = self.data_source[idx]
        
        # Load image
        image = Image.open(item.impath).convert('RGB')
        
        # Apply transforms
        if self.transform is not None:
            image = self.transform(image)
            
        # Return dictionary format for compatibility with existing code
        return {
            "img": image,
            "label": item.label,
            "impath": item.impath,
            "classname": item.classname
        }


def listdir_nohidden(path: str, sort: bool = True) -> List[str]:
    """List directory contents excluding hidden files - replaces dassl.utils.listdir_nohidden"""
    items = [f for f in os.listdir(path) if not f.startswith('.')]
    if sort:
        items.sort()
    return items


def mkdir_if_missing(dirname: str) -> None:
    """Create directory if it doesn't exist - replaces dassl.utils.mkdir_if_missing"""
    if not os.path.exists(dirname):
        os.makedirs(dirname)


# Dataset registry to replace DATASET_REGISTRY
_DATASET_REGISTRY = {}

def register_dataset(name: str):
    """Decorator to register dataset classes"""
    def decorator(cls):
        _DATASET_REGISTRY[name] = cls
        return cls
    return decorator


def build_dataset(config) -> DatasetBase:
    """Build dataset from config - replaces dassl.data.datasets.build_dataset"""
    dataset_name = config.dataset.name
    
    if dataset_name not in _DATASET_REGISTRY:
        raise ValueError(f"Unknown dataset: {dataset_name}. Available: {list(_DATASET_REGISTRY.keys())}")
    
    dataset_cls = _DATASET_REGISTRY[dataset_name]
    return dataset_cls(config)


class DATASET_REGISTRY:
    """Dataset registry class for backward compatibility"""
    
    @staticmethod
    def register():
        """Register decorator for backward compatibility"""
        def decorator(cls):
            # Extract dataset name from class name
            dataset_name = cls.__name__
            _DATASET_REGISTRY[dataset_name] = cls
            return cls
        return decorator
