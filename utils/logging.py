"""
Logging utilities for CLIP-GP.
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
        log_path = log_dir / "log.txt"
        file_handler = logging.FileHandler(log_path)
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

        # Redirect stdout and stderr to the log file as well as the console
        class Tee(object):
            def __init__(self, *files):
                self.files = files
            def write(self, obj):
                for f in self.files:
                    f.write(obj)
                    f.flush()
            def flush(self):
                for f in self.files:
                    f.flush()

        log_file = open(log_path, "a", buffering=1)
        sys.stdout = Tee(sys.__stdout__, log_file)
        sys.stderr = Tee(sys.__stderr__, log_file)
        # Note: This will capture all print statements and progress bars in the log file

    # Prevent duplicate logs
    logger.propagate = False
    
    return logger


def get_logger(name: str = "CLIP-GP") -> logging.Logger:
    """Get existing logger by name"""
    return logging.getLogger(name)
