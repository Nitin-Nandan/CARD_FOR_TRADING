"""
Module: src/training/epoch.py
Purpose: Contains the core per-epoch training loop with AMP and gradient accumulation.
Inputs: Model, dataloader, criterion, optimizer, scaler, and config.
Outputs: Average training loss and directional accuracy for the epoch.
"""

import os
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
    verbose=True,
    show_pbar=True,
):
    """Train for one epoch using Phase 4 optimizations"""

    model.train()
    total_loss = 0
    total_dir_acc = 0

    # Initialize batch statistics
    batch_losses = []

    # Progress bar
    pbar = tqdm(
        dataloader,
        desc=f"Epoch {epoch + 1}/{config.MAX_EPOCHS} [Train]",
        disable=not show_pbar,
    )

    for batch_idx, batch in enumerate(pbar):
        # 1. Move to device
        X = batch["X"].to(config.DEVICE)
        y = batch["y"].to(config.DEVICE)

        # 2. Forward pass with AMP (Automatic Mixed Precision)
        # Casts operations to float16 where safe (2x speedup on Tensor Cores)
        with torch.amp.autocast("cuda", enabled=config.USE_AMP):
            # Pass absolute prices to model (None for now as Phase 3 trained on normalized features)
            predictions = model(X)  # (B, PRED_LEN, 1)

            # Squeeze output to match y shape
            predictions = predictions.squeeze(-1)  # (B, 15)

            # Calculate loss WITH internal scaling
            loss, loss_dict = criterion(predictions, y)

            # Diagnostic prints
            if os.getenv("DIAGNOSTIC_MODE") == "1":
                print("\n=== LOSS BREAKDOWN ===")
                for k, v in loss_dict.items():
                    print(f"{k}: {v:.8f}")

            # Normalize loss for gradient accumulation
            loss = loss / config.ACCUMULATION_STEPS

        # 3. Backward pass (scaled for AMP)
        scaler.scale(loss).backward()

        # 4. Optimization step (only every ACCUMULATION_STEPS)
        if (batch_idx + 1) % config.ACCUMULATION_STEPS == 0 or (batch_idx + 1) == len(
            dataloader
        ):
            # Unscale gradients before clipping
            scaler.unscale_(optimizer)

            # Gradient Clipping (prevents exploding gradients)
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.GRAD_CLIP_NORM)

            # Step optimizer and update scale
            scaler.step(optimizer)
            scaler.update()

            # Zero gradients for next accumulation cycle
            optimizer.zero_grad(set_to_none=True)

        # Logging & Metrics
        current_loss = loss_dict["total"]
        dir_acc = loss_dict["direction_accuracy"] * 100.0

        total_loss += current_loss
        total_dir_acc += dir_acc
        batch_losses.append(current_loss)

        # Update progress bar
        avg_loss = sum(batch_losses[-100:]) / len(batch_losses[-100:])
        pbar.set_postfix(
            {
                "loss": f"{avg_loss:.6f}",
                "dir_acc": f"{dir_acc:.1f}%",
                "lr": f"{optimizer.param_groups[0]['lr']:.2e}",
            }
        )

        # Log to TensorBoard/WandB every N intervals
        global_step = epoch * len(dataloader) + batch_idx
        if global_step % config.LOG_INTERVAL == 0:
            metrics_to_log = {
                "train/loss": current_loss,
                "train/dir_acc": dir_acc,
                "train/learning_rate": optimizer.param_groups[0]["lr"],
            }
            # Also log component losses
            metrics_to_log.update(
                {f"train/{k}": v for k, v in loss_dict.items() if k != "total"}
            )
            logger.log_metrics(metrics_to_log, step=global_step)

    # Calculate epoch averages
    avg_loss = total_loss / len(dataloader)
    avg_dir_acc = total_dir_acc / len(dataloader)

    # Step scheduler AT END OF EPOCH
    if scheduler is not None:
        scheduler.step()

    return avg_loss, avg_dir_acc
