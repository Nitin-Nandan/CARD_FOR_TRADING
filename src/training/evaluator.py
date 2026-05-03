"""
Module: src/training/evaluator.py
Purpose: Handles the validation loop and metric computation.
Inputs: Model, validation dataloader, loss criterion.
Outputs: Validation loss and directional accuracy.
"""

import torch
from tqdm import tqdm


def validate_epoch(
    model, dataloader, criterion, config, logger, epoch, verbose=True, show_pbar=True
):
    """Validate model without gradients"""

    model.eval()
    total_loss = 0
    total_dir_acc = 0

    pbar = tqdm(
        dataloader,
        desc=f"Epoch {epoch + 1}/{config.MAX_EPOCHS} [Val]  ",
        disable=not show_pbar,
    )

    with torch.no_grad():
        for batch in pbar:
            X = batch["X"].to(config.DEVICE)
            y = batch["y"].to(config.DEVICE)

            # Mixed Precision inference
            with torch.amp.autocast("cuda", enabled=config.USE_AMP):
                predictions = model(X).squeeze(-1)
                loss, loss_dict = criterion(predictions, y)

            current_loss = loss_dict["total"]
            dir_acc = loss_dict["direction_accuracy"] * 100.0

            total_loss += current_loss
            total_dir_acc += dir_acc

            pbar.set_postfix(
                {"loss": f"{current_loss:.6f}", "dir_acc": f"{dir_acc:.1f}%"}
            )

    avg_loss = total_loss / len(dataloader)
    avg_dir_acc = total_dir_acc / len(dataloader)

    # Log global epoch validation metrics
    logger.log_metrics(
        {"val/loss": avg_loss, "val/dir_acc": avg_dir_acc},
        step=(epoch + 1) * len(dataloader),
    )  # Rough global alignment

    return avg_loss, avg_dir_acc
