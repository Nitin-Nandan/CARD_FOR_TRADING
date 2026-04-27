"""
Multi-Logger: Local Text + TensorBoard + WandB
"""

import logging
import os
from .tensorboard_logger import TensorBoardLogger
from .wandb_logger import WandBLogger


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

        # 2. TensorBoard (Local graphical metrics)
        tb_dir = os.path.join(log_dir, "tensorboard")
        self.tb = TensorBoardLogger(log_dir=tb_dir, name=name)
        self.has_tb = self.tb.enabled

        # 3. Weights & Biases (Cloud graphical metrics)
        if wandb_project:
            try:
                self.wandb = WandBLogger(
                    project=wandb_project,
                    name=name,
                    config=config,
                    entity=wandb_entity,
                    tags=wandb_tags,
                    offline=True,  # ← ALWAYS USE OFFLINE MODE FOR UNRELIABLE INTERNET
                )
                self.has_wandb = self.wandb.enabled
            except Exception as e:
                self.file_logger.warning(f"Failed to initialize WandB: {e}")
                self.wandb = None
                self.has_wandb = False
        else:
            self.wandb = None
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
        if self.has_tb:
            self.tb.log_metrics(metrics, step)

        if self.has_wandb:
            self.wandb.log_metrics(metrics, step)

    def close(self):
        """Close all loggers safely"""
        if self.has_tb:
            self.tb.close()

        if self.has_wandb:
            self.wandb.close()
