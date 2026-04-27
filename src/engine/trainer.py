"""
Training Engine for CARD Model
Features: AMP, Gradient Accumulation, Warmup+Cosine LR, and Multi-Logger Support.
"""

import os
import time
import gc
import torch
from torch.utils.data import DataLoader
from pathlib import Path
from tqdm import tqdm

from ..models.card import CARD, CombinedReturnLoss
from ..config import Config, QuickTestConfig
from ..data.dataset import StockWindowsDataset
from .lr_scheduler import WarmupCosineLR
from ..utils.logger import MultiLogger


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
            # Actually, CARD model signature in card_true.py:
            # forward(self, x_enc, x_mark_enc=None)
            # We are currently using feature returns, no explicit mark encoder yet
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
    if logger.has_wandb:
        logger.wandb.log_model(path, name=f"model_epoch_{epoch}")


def train_model(
    config_class=Config,
    run_name="card_phase4",
    use_wandb=True,
    wandb_project="card-stock-prediction",
    verbose=True,
    show_pbar=True,
):
    """
    Main training execution function.
    Can be called by hyperparameter sweep or final training script.
    """
    config = config_class()

    # 1. Setup Logging
    # Ensure directories exist
    os.makedirs(config.LOG_DIR, exist_ok=True)
    os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)

    logger = MultiLogger(
        log_dir=config.LOG_DIR,
        name=run_name,
        config=vars(config),
        wandb_project=wandb_project if use_wandb else None,
        wandb_tags=["production"],
    )

    logger.info(f"Starting CARD Model Training: {run_name}")
    logger.info(f"Using Device: {config.DEVICE}")
    logger.info(f"AMP Enabled: {config.USE_AMP}")
    logger.info(
        f"Accumulation Steps: {config.ACCUMULATION_STEPS} (Effective Batch: {config.BATCH_SIZE * config.ACCUMULATION_STEPS})"
    )

    # 2. Extract active stocks
    # Discovery which stocks have valid .npz windows
    data_dir = Path("data/windows")
    all_stocks = [
        f.stem.replace("_windows", "") for f in data_dir.glob("*_windows.npz")
    ]

    # Priority 1: Direct list in config
    if isinstance(config.STOCKS, list):
        stocks_to_use = [s for s in config.STOCKS if s in all_stocks]
        if not stocks_to_use:
            logger.warning(
                f"None of the stocks in config.STOCKS {config.STOCKS} found in data/windows. Falling back."
            )
            stocks_to_use = all_stocks[:2]
    # Priority 2: QuickTestConfig shortcut
    elif isinstance(config_class, type) and issubclass(config_class, QuickTestConfig):
        stocks_to_use = ["TCS"] if "TCS" in all_stocks else all_stocks[:1]
    # Priority 3: All stocks
    else:
        stocks_to_use = all_stocks

    logger.info(f"Using {len(stocks_to_use)} stocks for this run")

    # 3. Initialize Datasets & Dataloaders
    try:
        train_dataset = StockWindowsDataset(
            stocks=stocks_to_use,
            split="train",
            balance_stocks=config.BALANCE_STOCKS,
            max_windows_per_stock=config.MAX_WINDOWS_PER_STOCK,
            seq_len=config.SEQ_LEN,
            pred_len=config.PRED_LEN,
            verbose=verbose,
        )

        # Explicitly clear memory before second dataset init
        gc.collect()
        torch.cuda.empty_cache()

        val_dataset = StockWindowsDataset(
            stocks=stocks_to_use,
            split="val",
            balance_stocks=config.BALANCE_STOCKS,
            max_windows_per_stock=config.MAX_WINDOWS_PER_STOCK // 5,  # Smaller val set
            seq_len=config.SEQ_LEN,
            pred_len=config.PRED_LEN,
            verbose=verbose,
        )
    except Exception as e:
        logger.error(f"Dataset initialization failed: {e}")
        return None

    # Create DataLoaders
    # CRITICAL: num_workers MUST BE 0 ON WINDOWS!
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        pin_memory=config.PIN_MEMORY,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=config.PIN_MEMORY,
    )

    # 4. Initialize Model
    # Bridge uppercase Config to lowercase CARD internal requirements
    from types import SimpleNamespace

    model_config = SimpleNamespace(
        enc_in=config.ENC_IN,
        seq_len=config.SEQ_LEN,
        pred_len=config.PRED_LEN,
        patch_len=config.PATCH_LEN,
        stride=config.STRIDE,
        d_model=config.D_MODEL,
        n_heads=config.N_HEADS,
        e_layers=config.E_LAYERS,
        d_ff=config.D_FF,
        merge_size=config.MERGE_SIZE,
        dp_rank=config.DP_RANK,
        alpha=config.ALPHA,
        dropout=config.DROPOUT,
        momentum=0.1,  # Default momentum for BatchNorm
    )

    model = CARD(model_config).to(config.DEVICE)

    # 5. Optimization Setup
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.BASE_LR, weight_decay=config.WEIGHT_DECAY
    )

    # Setup Phase 4 learning rate scheduler
    scheduler = WarmupCosineLR(
        optimizer,
        warmup_epochs=config.WARMUP_EPOCHS,
        max_epochs=config.MAX_EPOCHS,
        base_lr=config.BASE_LR,
        min_lr=config.MIN_LR,
    )

    # Setup Combined Rescue Loss
    criterion = CombinedReturnLoss(
        directional_weight=config.DIRECTIONAL_WEIGHT,
        magnitude_scale=config.MAGNITUDE_SCALE,
    )

    # Setup AMP Scaler for mixed precision
    scaler = torch.amp.GradScaler("cuda", enabled=config.USE_AMP)

    # 6. Training Loop
    best_val_loss = float("inf")
    best_val_dir_acc = 0.0

    # Track training metrics associated with the best validation performance
    train_loss_at_best = 0.0
    train_acc_at_best = 0.0

    epochs_without_improvement = 0

    start_time = time.time()
    logger.info("Starting Training Loop...")

    for epoch in range(config.MAX_EPOCHS):
        logger.info(f"\n{'=' * 50}\nEPOCH {epoch + 1}/{config.MAX_EPOCHS}\n{'=' * 50}")

        # Train
        train_loss, train_dir_acc = train_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            scheduler,
            scaler,
            config,
            logger,
            epoch,
            verbose=verbose,
            show_pbar=show_pbar,
        )
        if verbose:
            logger.info(
                f"Train - Loss: {train_loss:.6f} | Dir Acc: {train_dir_acc:.2f}%"
            )

        # Validate
        if (epoch + 1) % config.VAL_INTERVAL == 0:
            val_loss, val_dir_acc = validate_epoch(
                model,
                val_loader,
                criterion,
                config,
                logger,
                epoch,
                verbose=verbose,
                show_pbar=show_pbar,
            )
            if verbose:
                logger.info(
                    f"Val   - Loss: {val_loss:.6f} | Dir Acc: {val_dir_acc:.2f}%"
                )

            # Checkpoint: Save Best Loss
            if config.SAVE_BEST_LOSS and val_loss < best_val_loss:
                best_val_loss = val_loss
                train_loss_at_best = train_loss
                train_acc_at_best = train_dir_acc
                epochs_without_improvement = 0
                path = os.path.join(config.CHECKPOINT_DIR, f"{run_name}_best_loss.pt")
                save_checkpoint(
                    model, optimizer, scheduler, epoch, val_loss, path, logger
                )
                logger.info("  [SAVE] New best validation loss! Model saved.")
            else:
                epochs_without_improvement += 1

            # Checkpoint: Save Best Directional Accuracy
            if config.SAVE_BEST_DIR_ACC and val_dir_acc > best_val_dir_acc:
                best_val_dir_acc = val_dir_acc
                path = os.path.join(config.CHECKPOINT_DIR, f"{run_name}_best_dir.pt")
                save_checkpoint(
                    model, optimizer, scheduler, epoch, val_loss, path, logger
                )
                logger.info("  [SAVE] New best directional accuracy! Model saved.")

            # Early Stopping Check
            if epochs_without_improvement >= config.EARLY_STOP_PATIENCE:
                logger.warning(
                    f"Early stopping triggered after {epochs_without_improvement} epochs without improvement."
                )
                break

        # Checkpoint: Periodic Save
        if config.SAVE_PERIODIC and (epoch + 1) % config.PERIODIC_SAVE_INTERVAL == 0:
            path = os.path.join(
                config.CHECKPOINT_DIR, f"{run_name}_epoch_{epoch + 1}.pt"
            )
            save_checkpoint(model, optimizer, scheduler, epoch, val_loss, path, logger)

        # Resample dataset if balancing enabled
        if hasattr(train_dataset, "on_epoch_end"):
            train_dataset.on_epoch_end()

    # Training Complete
    total_time = (time.time() - start_time) / 3600  # hours
    logger.info(f"\nTraining completed in {total_time:.2f} hours.")
    logger.info(
        f"Best Validation Loss: {best_val_loss:.6f} (Train: {train_loss_at_best:.6f})"
    )
    logger.info(
        f"Best Directional Accuracy: {best_val_dir_acc:.2f}% (Train: {train_acc_at_best:.2f}%)"
    )

    # Return results dictionary (reproducible search)
    return {
        "val_loss": best_val_loss,
        "val_dir_acc": best_val_dir_acc,
        "train_loss": train_loss_at_best,
        "train_dir_acc": train_acc_at_best,
    }


if __name__ == "__main__":
    # Standard entry point uses default config
    train_model(Config, run_name="card_standard_run", use_wandb=False)
