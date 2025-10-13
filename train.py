"""
Main training script for CLIP-GP.
"""

import torch
from pathlib import Path

# Import custom utilities
from utils import (
    parse_args_to_config, print_config, 
    setup_logger, set_random_seed,
    build_data_manager, build_trainer
)

# Import datasets and trainers
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

# Import trainers package to register all adapter trainers
import trainers


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
    logger.info("Starting CLIP-GP training")
    
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
    
    # Build data manager
    logger.info("Building dataset")
    data_manager = build_data_manager(config)
    
    # Build trainer
    logger.info(f"Loading trainer: {config.trainer_name}")
    trainer = build_trainer(config, data_manager)
    
    # Handle different execution modes
    if config.eval_only:
        logger.info("Running evaluation only")
        trainer.load_model(config.model_dir, epoch=config.load_epoch)
        trainer.test()
        return
    
    if not config.no_train:
        logger.info("Starting training")
        trainer.train()
        logger.info("Training completed")
    
    logger.info("CLIP-GP training finished successfully")


if __name__ == "__main__":
    main()
