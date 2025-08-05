"""
Logging utilities for CLIP-GP project.
Replaces Dassl's logging functionality.
"""

import logging
import sys
from pathlib import Path
from typing import Optional


def setup_logger(output_dir: Optional[str] = None, name: str = "CLIP-GP") -> logging.Logger:
    """
    Setup logging to file and console.
    
    Args:
        output_dir: Directory to save log file. If None, only console logging.
        name: Logger name
        
    Returns:
        Configured logger
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    
    # Clear existing handlers
    logger.handlers.clear()
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler (if output directory provided)
    if output_dir:
        log_dir = Path(output_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(log_dir / "log.txt")
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    # Prevent duplicate logs
    logger.propagate = False
    
    return logger


def get_logger(name: str = "CLIP-GP") -> logging.Logger:
    """Get existing logger by name"""
    return logging.getLogger(name)
