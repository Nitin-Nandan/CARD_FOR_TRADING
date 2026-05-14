"""
Module: src/training/evaluator.py
Purpose: Validation loop and metric computation for CARD model.

Cleaned on 2026-05-03:
- Removed spurious squeeze(-1).
- Updated to use RunLogger (metrics to file).
"""

import torch
from tqdm import tqdm

@torch.no_grad()
def validate_epoch(
    model, dataloader, criterion, config, logger, epoch, show_pbar=True
):
    """Validate model without gradients."""
    model.eval()
    total_loss = 0
    total_dir_acc = 0
    n_batches = 0

    pbar = tqdm(
        dataloader,
        desc=f"Epoch {epoch + 1}/{config.MAX_EPOCHS} [Val]  ",
        disable=not show_pbar,
        leave=False
    )

    for batch in pbar:
        X = batch["X"].to(config.DEVICE)
        y = batch["y"].to(config.DEVICE)

        with torch.cuda.amp.autocast(enabled=config.USE_AMP):
            predictions = model(X)
            loss, loss_dict = criterion(predictions, y)

        current_loss = loss_dict["total"]
        dir_acc = loss_dict["direction_accuracy"] * 100.0

        total_loss += current_loss
        total_dir_acc += dir_acc
        n_batches += 1

        pbar.set_postfix({
            "loss": f"{current_loss:.4f}",
            "dir": f"{dir_acc:.1f}%"
        })

    avg_loss = total_loss / n_batches
    avg_dir_acc = total_dir_acc / n_batches

    # Log epoch summary to file
    logger.metric((epoch + 1) * len(dataloader), val_loss=avg_loss, val_dir_acc=avg_dir_acc)

    return avg_loss, avg_dir_acc
