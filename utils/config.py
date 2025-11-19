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
    use_custom_templates: bool = False  # Use custom templates for the dataset, this will negate the use of num_templates
    num_templates: int = 1  # Number of templates to use
    l2_lambda: float = 0.1  # L2 regularization weight
    template_tw_l2_lambda: float = 0.0  # L2 regularization for learnable template weight matrix
    template_init_method: str = "uniform"  # "uniform", "val_weighted", "top3", "minmax"
    train_template_weights: bool = False  # Train template weights alongside visual projection (non-GP only)
    use_linear_template_weighting: bool = False  # Use linear layer to compute template weights from embeddings (experimental)
    freeze_visual_proj: bool = False  # If True, keep visual projection fixed at identity
    finetune_on_test: bool = False  # Do not use for regular training
    shared_template_weights: bool = False  # If True, use shared template weights (1, num_templates) instead of per-class (num_classes, num_templates)
    
    # GP-specific settings
    use_gp: bool = False  # Whether to use GP weighting for templates
    gp_kernel_type: str = "rbf"  # Kernel type: "rbf", "linear", "matern"
    gp_use_elbo: bool = False  # If True, add GP ELBO (with KL) during main training
    gp_lr: float = 0.001  # Learning rate for GP parameters
    gp_beta: float = 0.001  # KL weight for ELBO loss
    gp_num_mc_samples_train: int = 30  # Number of Monte Carlo samples for training
    gp_num_mc_samples_eval: int = 100  # Number of Monte Carlo samples for testing
    learn_token_lambda: float = 1e-2  # Weight for l2 regularization on visual learnable token inside the gp
    gp_pca_dim: int = 256  # Dimensionality for PCA reduction before GP (0 = no reduction)

    # CLIP-Adapter specific
    clip_adapter_reduction: int = 4   # Bottleneck reduction ratio for adapter MLP
    clip_adapter_ratio: float = 0.2   # Blend ratio between adapted and original features
    clip_adapter_use_template_weight_training: bool = False  # Train template weights before CLIP-Adapter
    clip_adapter_optimizer: str = "adam"
    clip_adapter_lr: float = 0.001
    clip_adapter_epochs: int = 100

    # Prompt-learning (CoOp / CoCoOp)
    n_ctx: int = 16 # number of learnable context tokens
    ctx_init: str = "" # optional initialization phrase (overrides n_ctx)
    csc: bool = False  # Class-Specific Context when no ctx_init is provided

    # Tip-Adapter defaults
    tip_adapter_trainable: bool = False
    tip_adapter_use_template_weight_training: bool = False
    tip_adapter_optimizer: str = "sgd"
    tip_adapter_lr: float = 0.001
    tip_adapter_epochs: int = 20
    tip_adapter_init_alpha: float = 0.0
    tip_adapter_init_beta: float = 0.0
    tip_adapter_eps: float = 0.0

    # TaskRes specific
    taskres_residual_scale: float = 0.5  # Scaling factor α for task residual (0.5 for most datasets, 1.0 for Flowers102)


@dataclass
class ModelConfig:
    """Model configuration"""
    backbone_name: str = "RN50"  # CLIP backbone: RN50, ViT-B/32, etc.
    init_weights: str = ""  # Path to pretrained weights


@dataclass
class DatasetConfig:
    """Dataset configuration"""
    name: str = "Caltech101"  # Dataset name
    root: str = "/mnt/features/VDATA" # Root dataset directory
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
    enable_tensorboard: bool = False  # Create tensorboard subfolder and write scalars
    enable_adapter_checkpoints: bool = False  # Save adapter checkpoints to subfolder


@dataclass
class Config:
    """Complete configuration for CLIP-GP training"""
    # Core components
    trainer_name: str = "Adapter" # "Adapter", "Adapter-CoOp", "Tip-Adapter", "CLIP-Adapter", "Adapter-TaskRes"
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
    parser.add_argument("--root", type=str, default=None, help="Path to dataset root")
    parser.add_argument("--dataset", type=str, default=None,
                       choices=["Caltech101", "OxfordPets", "OxfordFlowers", "FGVCAircraft", 
                               "DescribableTextures", "EuroSAT", "StanfordCars", "Food101", 
                               "SUN397", "UCF101", "ImageNet", "ImageNetSketch", "ImageNetV2", 
                               "ImageNetA", "ImageNetR"],
                       help="Dataset name")
    parser.add_argument("--shots", type=int, default=None, help="Number of shots")
    
    # Model arguments
    parser.add_argument("--backbone", type=str, default=None, choices=["RN50", "RN101", "RN50x4", "RN50x16", "ViT-B/32", "ViT-B/16"], help="CLIP backbone name")
    parser.add_argument("--trainer", type=str, default=None, choices=["Adapter", "Adapter-CoOp", "Adapter-TipA", "Adapter-TipA-F", "Adapter-CLIP-Adapter"], help="Trainer name")
    
    # Training arguments
    parser.add_argument("--lr", type=float, default=None, help="Learning rate")
    parser.add_argument("--epochs", type=int, default=None, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=None, help="Batch size")
    parser.add_argument("--optimizer", type=str, default=None, choices=["sgd", "adam", "adamw"], help="Optimizer")
    
    # Adapter arguments
    parser.add_argument("--num-templates", type=int, default=None, help="Number of templates")
    parser.add_argument("--l2-lambda", type=float, default=None, help="L2 regularization weight")
    parser.add_argument("--template-tw-l2-lambda", type=float, default=None, help="L2 regularization weight for template weight matrix")
    parser.add_argument("--template-init-method", type=str, default=None, choices=["uniform", "val_weighted", "top3", "minmax"], help="Template initialization method")
    parser.add_argument("--train-template-weights", action="store_true", help="Train template weights (non-GP)")
    parser.add_argument("--use-linear-template-weighting", action="store_true", help="Use linear layer to compute template weights from embeddings (experimental)")
    parser.add_argument("--freeze-visual-proj", action="store_true", help="Freeze visual projection (keep identity; no training)")
    parser.add_argument("--finetune-on-test", action="store_true", help="Finetune template weights on test set")
    parser.add_argument("--shared-template-weights", action="store_true", help="Use shared template weights across all classes")

    # GP arguments
    parser.add_argument("--use-gp", action="store_true", help="Use GP weighting")
    parser.add_argument("--gp-kernel-type", type=str, default=None, choices=["rbf", "linear"], help="GP kernel type")
    parser.add_argument("--gp-use-elbo", action="store_true", help="Use GP ELBO")
    parser.add_argument("--gp-lr", type=float, default=None, help="GP learning rate")
    parser.add_argument("--gp-beta", type=float, default=None, help="GP KL weight")
    parser.add_argument("--gp-num-mc-samples-train", type=int, default=None, help="Number of Monte Carlo samples for training")
    parser.add_argument("--gp-num-mc-samples-eval", type=int, default=None, help="Number of Monte Carlo samples for testing")
    parser.add_argument("--learn-token-lambda", type=float, default=None, help="Weight for l2 regularization on visual learnable token inside the gp")
    parser.add_argument("--gp-pca-dim", type=int, default=None, help="Dimensionality for PCA reduction before GP")

    # CoOp / CoCoOp
    parser.add_argument("--n-ctx", type=int, default=None, help="Number of context tokens for prompt learning")
    parser.add_argument("--ctx-init", type=str, default=None, help="Initialization phrase for context tokens")
    parser.add_argument("--csc", action="store_true", help="Use class-specific context for prompt learning")

    # CLIP-Adapter arguments
    parser.add_argument("--clip-adapter-reduction", type=int, default=None, help="Bottleneck reduction ratio for adapter MLP")
    parser.add_argument("--clip-adapter-ratio", type=float, default=None, help="Blend ratio between adapted and original features")
    
    # Environment arguments
    parser.add_argument("--output-dir", type=str, default=None, help="Output directory")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    parser.add_argument("--resume", type=str, default=None, help="Resume from checkpoint")
    
    # Configuration files
    parser.add_argument("--config-file", type=str, default="", help="Path to trainer config file")
    parser.add_argument("--dataset-config-file", type=str, default="", help="Path to dataset config file")
    
    # Evaluation arguments
    parser.add_argument("--eval-only", action="store_true", help="Evaluation only")
    parser.add_argument("--model-dir", type=str, default="", help="Model directory for evaluation")
    parser.add_argument("--load-epoch", type=int, help="Epoch to load for evaluation")
    parser.add_argument("--no-train", action="store_true", help="Skip training")
    
    # Additional options
    parser.add_argument("--source-domains", type=str, nargs="+", help="Source domains for DA/DG")
    parser.add_argument("--target-domains", type=str, nargs="+", help="Target domains for DA/DG")
    parser.add_argument("--transforms", type=str, nargs="+", help="Data augmentation methods")
    parser.add_argument("opts", default=None, nargs=argparse.REMAINDER, help="Modify config options using command-line")
    
    args = parser.parse_args()
    
    # Create base config
    config = Config()

    # Load configuration files first (BASE_CONFIG handled in merge_config_from_file)
    if args.dataset_config_file:
        merge_config_from_file(config, args.dataset_config_file)
    if args.config_file:
        merge_config_from_file(config, args.config_file)
    
    # Dataset arguments
    if args.root is not None:
        config.dataset.root = args.root
    if args.dataset is not None:
        config.dataset.name = args.dataset
    if args.shots is not None:
        config.dataset.num_shots = args.shots

    # Model arguments
    if args.backbone is not None:
        config.model.backbone_name = args.backbone
    if args.trainer is not None:
        config.trainer_name = args.trainer

    # Optimizer arguments
    if args.lr is not None:
        config.optim.lr = args.lr
    if args.epochs is not None:
        config.optim.max_epoch = args.epochs
    if args.batch_size is not None:
        config.dataloader.batch_size_train = args.batch_size
        config.dataloader.batch_size_test = args.batch_size
    if args.optimizer is not None:
        config.optim.name = args.optimizer

    # Adapter arguments
    if args.num_templates is not None:
        config.adapter.num_templates = args.num_templates
    if args.l2_lambda is not None:
        config.adapter.l2_lambda = args.l2_lambda
    if args.template_tw_l2_lambda is not None:
        config.adapter.template_tw_l2_lambda = args.template_tw_l2_lambda
    if args.template_init_method is not None:
        config.adapter.template_init_method = args.template_init_method
    if args.train_template_weights:
        config.adapter.train_template_weights = True
    if args.use_linear_template_weighting:
        config.adapter.use_linear_template_weighting = True
    if args.freeze_visual_proj:
        config.adapter.freeze_visual_proj = True
    if args.finetune_on_test:
        config.adapter.finetune_on_test = True
    if args.shared_template_weights:
        config.adapter.shared_template_weights = True
    
    # GP arguments
    if args.use_gp:
        config.adapter.use_gp = True
    if args.gp_kernel_type is not None:
        config.adapter.gp_kernel_type = args.gp_kernel_type
    if args.gp_use_elbo:
        config.adapter.gp_use_elbo = True
    if args.gp_lr is not None:
        config.adapter.gp_lr = args.gp_lr
    if args.gp_beta is not None:
        config.adapter.gp_beta = args.gp_beta
    if args.gp_num_mc_samples_train is not None:
        config.adapter.gp_num_mc_samples_train = args.gp_num_mc_samples_train
    if args.gp_num_mc_samples_eval is not None:
        config.adapter.gp_num_mc_samples_eval = args.gp_num_mc_samples_eval
    if args.learn_token_lambda is not None:
        config.adapter.learn_token_lambda = args.learn_token_lambda
    if args.gp_pca_dim is not None:
        config.adapter.gp_pca_dim = args.gp_pca_dim

    # CoOp / CoCoOp arguments
    if args.n_ctx is not None:
        config.adapter.n_ctx = args.n_ctx
    if args.ctx_init is not None:
        config.adapter.ctx_init = args.ctx_init
    if args.csc:
        config.adapter.csc = True

    # CLIP-Adapter arguments
    if args.clip_adapter_reduction is not None:
        config.adapter.clip_adapter_reduction = args.clip_adapter_reduction
    if args.clip_adapter_ratio is not None:
        config.adapter.clip_adapter_ratio = args.clip_adapter_ratio

    # Environment arguments
    if args.output_dir is not None:
        config.output_dir = args.output_dir
    if args.seed is not None:
        config.seed = args.seed
    if args.resume is not None:
        config.resume = args.resume

    # Configuration files
    if args.config_file is not None:
        config.config_file = args.config_file
    if args.dataset_config_file is not None:
        config.dataset_config_file = args.dataset_config_file

    # Evaluation arguments
    if args.eval_only:
        config.eval_only = args.eval_only
    if args.model_dir:
        config.model_dir = args.model_dir
    if args.load_epoch is not None:
        config.load_epoch = args.load_epoch
    if args.no_train:
        config.no_train = args.no_train

    # Additional options
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
