"""
Image transforms module for CLIP-GP.
"""

import torchvision.transforms as T
from typing import List, Tuple


def build_transform(config, is_train: bool = False):
    """Build image transforms from config"""
    
    if is_train:
        transforms = _build_train_transform(config)
    else:
        transforms = _build_test_transform(config)
        
    return T.Compose(transforms)


def _build_train_transform(config) -> List:
    """Build training transforms"""
    transforms = []
    
    # Get transform names from config
    transform_names = config.input.transforms
    
    for name in transform_names:
        if name == "random_resized_crop":
            transforms.append(
                T.RandomResizedCrop(
                    config.input.size,
                    scale=(0.08, 1.0),
                    interpolation=_get_interpolation(config.input.interpolation)
                )
            )
        elif name == "random_crop":
            transforms.append(T.RandomCrop(config.input.size))
        elif name == "random_flip":
            transforms.append(T.RandomHorizontalFlip())
        elif name == "random_rotation":
            transforms.append(T.RandomRotation(15))
        elif name == "color_jitter":
            transforms.append(
                T.ColorJitter(
                    brightness=0.4,
                    contrast=0.4,
                    saturation=0.4,
                    hue=0.1
                )
            )
        elif name == "normalize":
            transforms.extend([
                T.ToTensor(),
                T.Normalize(
                    mean=config.input.pixel_mean,
                    std=config.input.pixel_std
                )
            ])
    
    # If normalize wasn't in the transforms, add ToTensor at least
    if "normalize" not in transform_names:
        transforms.append(T.ToTensor())
        
    return transforms


def _build_test_transform(config) -> List:
    """Build test/validation transforms"""
    transforms = []
    
    # Resize
    size = config.input.size
    if isinstance(size, (tuple, list)):
        resize_size = size[0]
    else:
        resize_size = size
        
    transforms.append(
        T.Resize(
            resize_size,
            interpolation=_get_interpolation(config.input.interpolation)
        )
    )
    
    # Center crop
    transforms.append(T.CenterCrop(config.input.size))
    
    # Normalize
    transforms.extend([
        T.ToTensor(),
        T.Normalize(
            mean=config.input.pixel_mean,
            std=config.input.pixel_std
        )
    ])
    
    return transforms


def _get_interpolation(mode: str):
    """Get interpolation mode"""
    if mode == "bilinear":
        return T.InterpolationMode.BILINEAR
    elif mode == "bicubic":
        return T.InterpolationMode.BICUBIC
    elif mode == "nearest":
        return T.InterpolationMode.NEAREST
    else:
        return T.InterpolationMode.BILINEAR  # Default
