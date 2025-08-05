"""
Main training script for CLIP-GP project.
Phase 1 implementation with new configuration system.
"""

import torch
from pathlib import Path

# Import our new configuration and utilities
from utils.config import parse_args_to_config, print_config
from utils import setup_logger, set_random_seed

# Import existing components (will be gradually replaced)
from dassl.engine import build_trainer

# Custom datasets (keep existing imports for now)
import datasets.oxford_pets
import datasets.oxford_flowers
import datasets.fgvc_aircraft
import datasets.dtd
import datasets.eurosat
import datasets.stanford_cars
import datasets.food101
import datasets.sun397
import datasets.caltech101
import datasets.ucf101
import datasets.imagenet
import datasets.imagenet_sketch
import datasets.imagenetv2
import datasets.imagenet_a
import datasets.imagenet_r

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

import trainers.adapters


def convert_config_to_dassl_format(config):
    """
    Temporary function to convert our new config format to Dassl format.
    This will be removed in later phases.
    """
    from dassl.config import get_cfg_default
    from yacs.config import CfgNode as CN
    
    cfg = get_cfg_default()
    
    # Add adapter configuration
    cfg.TRAINER.NAME = config.trainer_name
    cfg.TRAINER.ADAPTER = CN()
    cfg.TRAINER.ADAPTER.INIT = config.adapter.init
    cfg.TRAINER.ADAPTER.CONSTRAINT = config.adapter.constraint
    cfg.TRAINER.ADAPTER.ENHANCED_BASE = config.adapter.enhanced_base
    cfg.TRAINER.ADAPTER.PREC = config.adapter.prec
    cfg.TRAINER.ADAPTER.NUM_TEMPLATES = config.adapter.num_templates
    cfg.TRAINER.ADAPTER.USE_GP = config.adapter.use_gp
    cfg.TRAINER.ADAPTER.GP_LR = config.adapter.gp_lr
    cfg.TRAINER.ADAPTER.GP_BETA = config.adapter.gp_beta
    cfg.TRAINER.ADAPTER.GP_NUM_MC_SAMPLES = config.adapter.gp_num_mc_samples
    cfg.TRAINER.ADAPTER.GP_KERNEL_TYPE = config.adapter.gp_kernel_type
    cfg.TRAINER.ADAPTER.L2_LAMBDA = config.adapter.l2_lambda
    cfg.TRAINER.ADAPTER.RES_L2_COEF = config.adapter.res_l2_coef
    
    # Model configuration
    cfg.MODEL.BACKBONE.NAME = config.model.backbone_name
    cfg.MODEL.HEAD.NAME = config.model.head_name
    cfg.MODEL.INIT_WEIGHTS = config.model.init_weights
    
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
    
    # Optimization configuration
    cfg.OPTIM.NAME = config.optim.name
    cfg.OPTIM.LR = config.optim.lr
    cfg.OPTIM.MAX_EPOCH = config.optim.max_epoch
    cfg.OPTIM.LR_SCHEDULER = config.optim.lr_scheduler
    cfg.OPTIM.WARMUP_EPOCH = config.optim.warmup_epoch
    cfg.OPTIM.WARMUP_TYPE = config.optim.warmup_type
    cfg.OPTIM.WARMUP_CONS_LR = config.optim.warmup_cons_lr
    cfg.OPTIM.WEIGHT_DECAY = config.optim.weight_decay
    cfg.OPTIM.MOMENTUM = config.optim.momentum
    
    # Training configuration
    cfg.TRAIN.PRINT_FREQ = config.train.print_freq
    
    # Environment configuration
    cfg.OUTPUT_DIR = config.output_dir
    cfg.RESUME = config.resume
    cfg.SEED = config.seed
    cfg.USE_CUDA = config.use_cuda
    cfg.VERBOSE = config.verbose
    
    cfg.freeze()
    return cfg


def print_args(config):
    """Print configuration in a readable format"""
    print("***************")
    print("** Arguments **")
    print("***************")
    
    # Print key arguments
    print(f"dataset: {config.dataset.name}")
    print(f"shots: {config.dataset.num_shots}")
    print(f"backbone: {config.model.backbone_name}")
    print(f"use_gp: {config.adapter.use_gp}")
    print(f"num_templates: {config.adapter.num_templates}")
    print(f"lr: {config.optim.lr}")
    print(f"epochs: {config.optim.max_epoch}")
    print(f"output_dir: {config.output_dir}")
    print(f"seed: {config.seed}")


def main():
    """Main training function"""
    # Parse arguments using new configuration system
    config = parse_args_to_config()
    
    # Set up logging
    logger = setup_logger(config.output_dir)
    logger.info("Starting CLIP-GP training with new configuration system")
    
    # Set random seed
    if config.seed >= 0:
        set_random_seed(config.seed)
    
    # Set up CUDA
    if torch.cuda.is_available() and config.use_cuda:
        torch.backends.cudnn.benchmark = True
        logger.info(f"Using CUDA with {torch.cuda.device_count()} GPUs")
    
    # Print configuration
    print_args(config)
    print_config(config)
    
    # Save configuration to output directory
    config_save_path = Path(config.output_dir) / "config.json"
    config_save_path.parent.mkdir(parents=True, exist_ok=True)
    
    # TEMPORARY: Convert to Dassl format for compatibility
    # This will be removed in Phase 2 when we replace the trainer
    logger.info("Converting configuration to Dassl format (temporary)")
    dassl_cfg = convert_config_to_dassl_format(config)
    
    # Build trainer using existing Dassl infrastructure
    logger.info("Building trainer")
    trainer = build_trainer(dassl_cfg)
    
    # Handle different execution modes
    if config.eval_only:
        logger.info("Running evaluation only")
        trainer.load_model(config.model_dir, dassl_cfg, epoch=config.load_epoch)
        trainer.test()
        return
    
    if not config.no_train:
        logger.info("Starting training")
        trainer.train()
        logger.info("Training completed")
    
    logger.info("CLIP-GP training finished successfully")


if __name__ == "__main__":
    main()
