"""
Multi-Logger: Local Text
"""

import logging
import os


class MultiLogger:
    def __init__(
        self,
        log_dir,
        name,
        config=None,
        wandb_project=None,
        wandb_entity=None,
        wandb_tags=None,
    ):
        """
        Initialize multiple logging backends automatically
        """
        self.name = name

        # 1. Standard Python File/Console Logger
        os.makedirs(log_dir, exist_ok=True)
        log_file = os.path.join(log_dir, f"{name}.log")

        self.file_logger = logging.getLogger(name)
        self.file_logger.setLevel(logging.INFO)
        self.file_logger.propagate = False  # Prevent double logging

        # Clear existing handlers
        if self.file_logger.hasHandlers():
            self.file_logger.handlers.clear()

        # File handler
        fh = logging.FileHandler(log_file)
        fh.setLevel(logging.INFO)

        # Console handler
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)

        # Formatters
        detailed_formatter = logging.Formatter(
            "%(asctime)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
        )
        clean_formatter = logging.Formatter("%(message)s")

        fh.setFormatter(detailed_formatter)
        ch.setFormatter(clean_formatter)

        self.file_logger.addHandler(fh)
        self.file_logger.addHandler(ch)

        self.has_tb = False
        self.has_wandb = False

    # Standard string logging methods
    def info(self, msg):
        self.file_logger.info(msg)

    def warning(self, msg):
        self.file_logger.warning(msg)

    def error(self, msg):
        self.file_logger.error(msg)

    # Graphical metrics logging method
    def log_metrics(self, metrics, step):
        """Log numeric metrics to TensorBoard and/or WandB"""
        pass

    def close(self):
        """Close all loggers safely"""
        pass
