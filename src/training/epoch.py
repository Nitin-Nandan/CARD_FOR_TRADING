"""
Module: src/training/epoch.py
Purpose: Core per-epoch training loop for CARD model.

Cleaned on 2026-05-03:
- Removed spurious squeeze(-1) (CARD daily outputs (B, pred_len)).
- Removed dead diagnostic mode prints.
- Updated to use RunLogger (detail to file, minimal to terminal).
"""

import torch
from tqdm import tqdm

def train_epoch(
    model,
    dataloader,
    criterion,
    optimizer,
    scheduler,
    scaler,
    config,
    logger,
    epoch,
    show_pbar=True,
):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    total_dir_acc = 0
    n_batches = 0

    pbar = tqdm(
        dataloader,
        desc=f"Epoch {epoch + 1}/{config.MAX_EPOCHS} [Train]",
        disable=not show_pbar,
        leave=False
    )

    for batch_idx, batch in enumerate(pbar):
        X = batch["X"].to(config.DEVICE)
        y = batch["y"].to(config.DEVICE)

        # Forward pass with AMP
        with torch.cuda.amp.autocast(enabled=config.USE_AMP):
            predictions = model(X) # (B, pred_len)
            loss, loss_dict = criterion(predictions, y)
            loss = loss / config.ACCUMULATION_STEPS

        # Backward pass
        scaler.scale(loss).backward()

        # Optimization step
        if (batch_idx + 1) % config.ACCUMULATION_STEPS == 0 or (batch_idx + 1) == len(dataloader):
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.GRAD_CLIP_NORM)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        # Metrics
        current_loss = loss_dict["total"]
        dir_acc = loss_dict["direction_accuracy"] * 100.0
        total_loss += current_loss
        total_dir_acc += dir_acc
        n_batches += 1

        pbar.set_postfix({
            "loss": f"{current_loss:.4f}",
            "dir": f"{dir_acc:.1f}%",
            "lr": f"{optimizer.param_groups[0]['lr']:.2e}"
        })

        # Log details to file only
        global_step = epoch * len(dataloader) + batch_idx
        if global_step % config.LOG_INTERVAL == 0:
            logger.metric(global_step, **loss_dict, lr=optimizer.param_groups[0]["lr"])

    avg_loss = total_loss / n_batches
    avg_dir_acc = total_dir_acc / n_batches

    return avg_loss, avg_dir_acc
