"""
Configuration system for CLIP-GP.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Union, Tuple
import argparse
import json
import yaml
from pathlib import Path


@dataclass
class AdapterConfig:
    """Adapter-specific configuration"""
    # Basic adapter settings
    prec: str = "fp16"  # Precision: "fp16", "fp32", "amp"
    num_templates: int = 1  # Number of templates to use
    l2_lambda: float = 0.1  # L2 regularization weight
    template_init_method: str = "uniform"  # "uniform", "val_weighted", "top3", "minmax"
    train_template_weights: bool = False  # Train template weights alongside visual projection (non-GP only)
    
    # GP-specific settings
    use_gp: bool = False  # Whether to use GP weighting for templates
    gp_lr: float = 0.1  # Learning rate for GP parameters
    gp_beta: float = 0.001  # KL weight for ELBO loss
    gp_num_mc_samples: int = 10  # Number of Monte Carlo samples
    gp_kernel_type: str = "rbf"  # Kernel type: "rbf" or "linear"
    gp_use_elbo: bool = True  # If True, add GP ELBO (with KL) during main training
    

@dataclass
class ModelConfig:
    """Model configuration"""
    backbone_name: str = "RN50"  # CLIP backbone: RN50, ViT-B/32, etc.
    head_name: str = ""  # Head name (usually empty for CLIP)
    init_weights: str = ""  # Path to pretrained weights


@dataclass
class DatasetConfig:
    """Dataset configuration"""
    name: str = "Caltech101"  # Dataset name
    root: str = "/mnt/features/VDATA"
    num_shots: int = 1  # Number of shots for few-shot learning
    subsample_classes: str = "all"  # "all", "base", or "new"
    source_domains: Optional[List[str]] = None  # Source domains for DA/DG
    target_domains: Optional[List[str]] = None  # Target domains for DA/DG


@dataclass
class DataLoaderConfig:
    """Data loader configuration"""
    batch_size_train: int = 128  # Training batch size
    batch_size_test: int = 128  # Test batch size
    num_workers: int = 8  # Number of data loading workers
    drop_last: bool = False  # Drop last incomplete batch


@dataclass
class InputConfig:
    """Input preprocessing configuration"""
    size: Tuple[int, int] = (224, 224)  # Input image size
    interpolation: str = "bicubic"  # Interpolation method
    pixel_mean: Tuple[float, float, float] = (0.48145466, 0.4578275, 0.40821073)  # CLIP normalization
    pixel_std: Tuple[float, float, float] = (0.26862954, 0.26130258, 0.27577711)  # CLIP normalization
    transforms: List[str] = field(default_factory=lambda: ["random_resized_crop", "random_flip", "normalize"])


@dataclass
class OptimConfig:
    """Optimization configuration"""
    name: str = "sgd"  # Optimizer name: "sgd", "adam", "adamw"
    lr: float = 0.01  # Base learning rate
    max_epoch: int = 300  # Maximum number of epochs
    lr_scheduler: str = "cosine"  # LR scheduler: "cosine", "step", "constant"
    warmup_epoch: int = 1  # Warmup epochs
    warmup_type: str = "constant"  # Warmup type: "constant", "linear"
    warmup_cons_lr: float = 1e-5  # Constant warmup learning rate
    weight_decay: float = 0.0  # Weight decay
    momentum: float = 0.9  # SGD momentum
    betas: Tuple[float, float] = (0.9, 0.999)  # Adam betas


@dataclass
class TrainConfig:
    """Training configuration"""
    print_freq: int = 5  # Print frequency
    eval_freq: int = 1  # Evaluation frequency
    checkpoint_freq: int = 0  # Checkpoint saving frequency (0 = disabled)


@dataclass
class Config:
    """Complete configuration for CLIP-GP training"""
    # Core components
    trainer_name: str = "Trainer"
    adapter: AdapterConfig = field(default_factory=AdapterConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    dataloader: DataLoaderConfig = field(default_factory=DataLoaderConfig)
    input: InputConfig = field(default_factory=InputConfig)
    optim: OptimConfig = field(default_factory=OptimConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    
    # Environment settings
    output_dir: str = "output/default_experiment"
    resume: str = ""  # Checkpoint path to resume from
    seed: int = 1  # Random seed (-1 for random)
    use_cuda: bool = True  # Use CUDA if available
    verbose: bool = True  # Verbose output
    
    # Evaluation settings
    eval_only: bool = False  # Evaluation only mode
    model_dir: str = ""  # Model directory for evaluation
    load_epoch: Optional[int] = None  # Specific epoch to load
    no_train: bool = False  # Skip training


def load_config_from_yaml(yaml_path: str) -> dict:
    """Load configuration from YAML file"""
    with open(yaml_path, 'r') as f:
        return yaml.safe_load(f)


def merge_config_from_file(config: Config, config_file: str) -> None:
    """Merge configuration from YAML file into existing config.

    Supports optional inheritance via BASE_CONFIG: path/to/base.yaml within the YAML.
    The base is merged first, then the current file to allow overrides.
    Paths in BASE_CONFIG are resolved relative to the referencing YAML file.
    """
    if not config_file:
        return
    path = Path(config_file)
    if not path.exists():
        return

    file_config = load_config_from_yaml(str(path)) or {}

    # Handle BASE_CONFIG if present
    base_cfg_path = file_config.pop("BASE_CONFIG", None)
    if base_cfg_path:
        base_path = Path(base_cfg_path)
        resolved_base: Path
        if base_path.is_absolute():
            resolved_base = base_path
        else:
            candidate1 = (path.parent / base_path)
            candidate2 = (Path.cwd() / base_path)
            if candidate1.exists():
                resolved_base = candidate1
            elif candidate2.exists():
                resolved_base = candidate2
            else:
                resolved_base = candidate1  # fall back
        # Merge base first
        merge_config_from_file(config, str(resolved_base))

    # Merge the current file after the base to allow overrides
    merge_config_dict(config, file_config)


def merge_config_dict(config: Config, config_dict: dict) -> None:
    """Recursively merge dictionary into config object"""
    for key, value in config_dict.items():
        if key.lower() == "dataset" and isinstance(value, str): config.dataset.name = value
        # Handle special cases first before general attribute checking
        if key == "TRAINER" and "ADAPTER" in value:
            adapter_config = value["ADAPTER"]
            for adapter_key, adapter_value in adapter_config.items():
                if hasattr(config.adapter, adapter_key.lower()):
                    setattr(config.adapter, adapter_key.lower(), adapter_value)
        elif key == "DATALOADER":
            if "TRAIN_X" in value and "BATCH_SIZE" in value["TRAIN_X"]:
                config.dataloader.batch_size_train = value["TRAIN_X"]["BATCH_SIZE"]
            if "TEST" in value and "BATCH_SIZE" in value["TEST"]:
                config.dataloader.batch_size_test = value["TEST"]["BATCH_SIZE"]
            if "NUM_WORKERS" in value:
                config.dataloader.num_workers = value["NUM_WORKERS"]
        elif key == "INPUT":
            for input_key, input_value in value.items():
                if hasattr(config.input, input_key.lower()):
                    if input_key.lower() == "size":
                        # Handle both list format [224, 224] and tuple string format "(224, 224)"
                        if isinstance(input_value, list):
                            setattr(config.input, input_key.lower(), tuple(input_value))
                        elif isinstance(input_value, str) and input_value.startswith('(') and input_value.endswith(')'):
                            # Parse tuple string like "(224, 224)"
                            size_str = input_value.strip('()')
                            size_values = [int(x.strip()) for x in size_str.split(',')]
                            setattr(config.input, input_key.lower(), tuple(size_values))
                        else:
                            setattr(config.input, input_key.lower(), input_value)
                    elif input_key.lower() in ["pixel_mean", "pixel_std"] and isinstance(input_value, list):
                        setattr(config.input, input_key.lower(), tuple(input_value))
                    else:
                        setattr(config.input, input_key.lower(), input_value)
        elif key == "OPTIM":
            for optim_key, optim_value in value.items():
                if hasattr(config.optim, optim_key.lower()):
                    setattr(config.optim, optim_key.lower(), optim_value)
        elif key == "TRAIN":
            for train_key, train_value in value.items():
                if hasattr(config.train, train_key.lower()):
                    setattr(config.train, train_key.lower(), train_value)
        elif key == "DATASET":
            for dataset_key, dataset_value in value.items():
                if hasattr(config.dataset, dataset_key.lower()):
                    setattr(config.dataset, dataset_key.lower(), dataset_value)
        elif key == "MODEL":
            if "BACKBONE" in value and "NAME" in value["BACKBONE"]:
                config.model.backbone_name = value["BACKBONE"]["NAME"]
            if "HEAD" in value and "NAME" in value["HEAD"]:
                config.model.head_name = value["HEAD"]["NAME"]
            if "INIT_WEIGHTS" in value:
                config.model.init_weights = value["INIT_WEIGHTS"]
        elif hasattr(config, key.lower()):
            attr = getattr(config, key.lower())
            if isinstance(value, dict) and hasattr(attr, '__dict__'):
                # Recursively merge nested configs
                for sub_key, sub_value in value.items():
                    if hasattr(attr, sub_key.lower()):
                        setattr(attr, sub_key.lower(), sub_value)
            else:
                setattr(config, key.lower(), value)


def parse_args_to_config() -> Config:
    """Parse command line arguments and create configuration"""
    parser = argparse.ArgumentParser(description="CLIP-GP Training")
    
    # Dataset arguments
    parser.add_argument("--root", type=str, default=None, 
                       help="Path to dataset root")
    parser.add_argument("--dataset", type=str, default=None,
                       choices=["Caltech101", "OxfordPets", "OxfordFlowers", "FGVCAircraft", 
                               "DescribableTextures", "EuroSAT", "StanfordCars", "Food101", 
                               "SUN397", "UCF101", "ImageNet", "ImageNetSketch", "ImageNetV2", 
                               "ImageNetA", "ImageNetR"],
                       help="Dataset name")
    parser.add_argument("--shots", type=int, default=None, help="Number of shots")
    
    # Model arguments
    parser.add_argument("--backbone", type=str, default=None, 
                       help="CLIP backbone name")
    parser.add_argument("--trainer", type=str, default=None,
                       help="Trainer name")
    parser.add_argument("--head", type=str, default=None, help="Head name")
    
    # Training arguments
    parser.add_argument("--lr", type=float, default=None, help="Learning rate")
    parser.add_argument("--epochs", type=int, default=None, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=None, help="Batch size")
    parser.add_argument("--optimizer", type=str, default=None, 
                       choices=["sgd", "adam", "adamw"], help="Optimizer")
    
    # GP arguments
    parser.add_argument("--use-gp", action="store_true", help="Use GP weighting")
    parser.add_argument("--gp-lr", type=float, default=None, help="GP learning rate")
    parser.add_argument("--gp-beta", type=float, default=None, help="GP KL weight")
    parser.add_argument("--num-templates", type=int, default=None, help="Number of templates")
    parser.add_argument("--train-template-weights", action="store_true", help="Train template weights (non-GP)")
    
    # Environment arguments
    parser.add_argument("--output-dir", type=str, default=None,
                       help="Output directory")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    parser.add_argument("--resume", type=str, default=None, help="Resume from checkpoint")
    
    # Configuration files
    parser.add_argument("--config-file", type=str, default="",
                       help="Path to trainer config file")
    parser.add_argument("--dataset-config-file", type=str, default="",
                       help="Path to dataset config file")
    
    # Evaluation arguments
    parser.add_argument("--eval-only", action="store_true", help="Evaluation only")
    parser.add_argument("--model-dir", type=str, default="", 
                       help="Model directory for evaluation")
    parser.add_argument("--load-epoch", type=int, help="Epoch to load for evaluation")
    parser.add_argument("--no-train", action="store_true", help="Skip training")
    
    # Additional options
    parser.add_argument("--source-domains", type=str, nargs="+", 
                       help="Source domains for DA/DG")
    parser.add_argument("--target-domains", type=str, nargs="+",
                       help="Target domains for DA/DG")
    parser.add_argument("--transforms", type=str, nargs="+",
                       help="Data augmentation methods")
    
    # Additional config options
    parser.add_argument("opts", default=None, nargs=argparse.REMAINDER,
                       help="Modify config options using command-line")
    
    args = parser.parse_args()
    
    # Create base config
    config = Config()

    # Load configuration files first (BASE_CONFIG handled in merge_config_from_file)
    if args.dataset_config_file:
        merge_config_from_file(config, args.dataset_config_file)
    if args.config_file:
        merge_config_from_file(config, args.config_file)
    
    # Apply command line arguments (only if explicitly provided)
    if args.root is not None:
        config.dataset.root = args.root
    if args.dataset is not None:
        config.dataset.name = args.dataset
    if args.shots is not None:
        config.dataset.num_shots = args.shots
    if args.backbone is not None:
        config.model.backbone_name = args.backbone
    if args.trainer is not None:
        config.trainer_name = args.trainer
    if args.head is not None:
        config.model.head_name = args.head
    if args.lr is not None:
        config.optim.lr = args.lr
    if args.epochs is not None:
        config.optim.max_epoch = args.epochs
    if args.batch_size is not None:
        config.dataloader.batch_size_train = args.batch_size
        config.dataloader.batch_size_test = args.batch_size
    if args.optimizer is not None:
        config.optim.name = args.optimizer
    if args.use_gp:
        config.adapter.use_gp = True
    if args.gp_lr is not None:
        config.adapter.gp_lr = args.gp_lr
    if args.gp_beta is not None:
        config.adapter.gp_beta = args.gp_beta
    if args.num_templates is not None:
        config.adapter.num_templates = args.num_templates
    if args.train_template_weights:
        config.adapter.train_template_weights = True
    if args.output_dir is not None:
        config.output_dir = args.output_dir
    if args.seed is not None:
        config.seed = args.seed
    if args.resume is not None:
        config.resume = args.resume
    if args.eval_only:
        config.eval_only = args.eval_only
    if args.model_dir:
        config.model_dir = args.model_dir
    if args.load_epoch is not None:
        config.load_epoch = args.load_epoch
    if args.no_train:
        config.no_train = args.no_train
    if args.source_domains is not None:
        config.dataset.source_domains = args.source_domains
    if args.target_domains is not None:
        config.dataset.target_domains = args.target_domains
    if args.transforms is not None:
        config.input.transforms = args.transforms

    # Process additional opts
    if args.opts:
        _merge_from_list(config, args.opts)
    
    return config


def _merge_from_list(config: Config, opts: List[str]) -> None:
    """Merge options from command line list (e.g., ['TRAINER.ADAPTER.USE_GP', 'True'])"""
    if len(opts) % 2 != 0:
        raise ValueError("opts must have even length")
    
    for i in range(0, len(opts), 2):
        key = opts[i]
        value = opts[i + 1]
        
        # Parse value
        if value.lower() in ["true", "false"]:
            value = value.lower() == "true"
        elif value.isdigit():
            value = int(value)
        elif value.replace(".", "").replace("-", "").isdigit():
            value = float(value)
        
        # Apply to config
        _set_nested_attr(config, key, value)


def _set_nested_attr(config: Config, key: str, value) -> None:
    """Set nested attribute using dot notation (e.g., 'TRAINER.ADAPTER.USE_GP')"""
    parts = key.split(".")
    obj = config
    
    for part in parts[:-1]:
        part_lower = part.lower()
        if part_lower == "trainer" and parts[-1].upper() in ["ADAPTER"]:
            continue  # Skip trainer level for adapter config
        elif part_lower == "adapter" or (part.upper() == "ADAPTER"):
            obj = config.adapter
        elif part_lower == "model":
            obj = config.model
        elif part_lower == "dataset":
            obj = config.dataset
        elif part_lower == "dataloader":
            obj = config.dataloader
        elif part_lower == "input":
            obj = config.input
        elif part_lower == "optim":
            obj = config.optim
        elif part_lower == "train":
            obj = config.train
        else:
            if hasattr(obj, part_lower):
                obj = getattr(obj, part_lower)
    
    # Set the final attribute
    final_key = parts[-1].lower()
    if hasattr(obj, final_key):
        setattr(obj, final_key, value)


def save_config_to_file(config: Config, filepath: str) -> None:
    """Save configuration to JSON file"""
    config_dict = _config_to_dict(config)
    with open(filepath, 'w') as f:
        json.dump(config_dict, f, indent=2)


def _config_to_dict(config) -> dict:
    """Convert config object to dictionary"""
    if hasattr(config, '__dict__'):
        result = {}
        for key, value in config.__dict__.items():
            if hasattr(value, '__dict__'):
                result[key] = _config_to_dict(value)
            else:
                result[key] = value
        return result
    else:
        return config


def print_config(config: Config) -> None:
    """Print configuration in a readable format"""
    print("************")
    print("** Config **")
    print("************")
    _print_config_recursive(config, prefix="")


def _print_config_recursive(config, prefix: str) -> None:
    """Recursively print configuration"""
    if hasattr(config, '__dict__'):
        for key, value in config.__dict__.items():
            if hasattr(value, '__dict__'):
                print(f"{prefix}{key.upper()}:")
                _print_config_recursive(value, prefix + "  ")
            else:
                print(f"{prefix}{key.upper()}: {value}")
    else:
        print(f"{prefix}{config}")


# Convenience function for backward compatibility
def get_cfg_default() -> Config:
    """Get default configuration"""
    return Config()


if __name__ == "__main__":
    # Test the configuration system
    config = parse_args_to_config()
    print_config(config)
