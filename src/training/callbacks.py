"""
Module: src/training/callbacks.py
Purpose: Handles checkpoint saving and other training callbacks.
Inputs: Model state, optimizer, scheduler, metrics.
Outputs: Saved checkpoint files.
"""

import os
import torch


def save_checkpoint(model, optimizer, scheduler, epoch, val_loss, path, logger):
    """Saves full training state"""
    os.makedirs(os.path.dirname(path), exist_ok=True)

    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
        "val_loss": val_loss,
    }

    torch.save(checkpoint, path)

    # Log model to WandB if enabled
    if getattr(logger, "has_wandb", False) and logger.wandb is not None:
        logger.wandb.log_model(path, name=f"model_epoch_{epoch}")
