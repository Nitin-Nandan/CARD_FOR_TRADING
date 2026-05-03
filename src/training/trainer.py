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

from ..models.card import CARD
from .loss import CombinedReturnLoss
from .evaluator import validate_epoch
from .epoch import train_epoch
from .callbacks import save_checkpoint
from ..config import Config, QuickTestConfig
from ..data.dataset import StockWindowsDataset
from .lr_scheduler import WarmupCosineLR
from ..utils.logger import MultiLogger




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
