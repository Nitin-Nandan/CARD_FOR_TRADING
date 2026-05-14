"""
Pipeline: Step 11 - Daily CARD Training
Cleaned orchestrator for NSE daily stock prediction.

Key Changes:
- Integrated RunLogger (Terminal is clean, Detail in logs/daily_v1/).
- Uses cleaned src/training/ modules.
- Proper result reporting.
"""

import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, ConcatDataset
import pandas as pd
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.append(str(PROJECT_ROOT))

from src.models.card import CARD
from src.training.loss import CombinedReturnLoss
from src.training.lr_scheduler import WarmupCosineLR
from src.training.callbacks import save_checkpoint
from src.training.epoch import train_epoch
from src.training.evaluator import validate_epoch
from src.data.transforms import LogReturnTransform
from src.features.daily_features import compute_daily_features
from src.data.daily_dataset import build_splits
from src.utils.daily_config import DailyConfig
from src.utils.logger import RunLogger

PROCESSED_DIR = PROJECT_ROOT / "data" / "raw" / "daily"

def _load_stock(csv_path: Path, config: type) -> pd.DataFrame | None:
    try:
        df = pd.read_csv(csv_path, index_col="date", parse_dates=True)
        df.columns = [c.lower() for c in df.columns]
        df = df[["open", "high", "low", "close", "volume"]].dropna()

        if len(df) < config.seq_len + config.pred_len + 60:
            return None

        t = LogReturnTransform()
        df = t.transform(df).dropna()
        feat = compute_daily_features(df).dropna()
        return feat
    except Exception:
        return None

def _build_loaders(config: type):
    csv_files = sorted(PROCESSED_DIR.glob("*.csv"))
    csv_files = [f for f in csv_files if "NIFTY50" not in f.stem]

    train_dsets, val_dsets, test_dsets = [], [], []
    
    for csv_path in tqdm(csv_files, desc="Preparing Stocks", unit="stock", leave=False):
        feat_df = _load_stock(csv_path, config)
        if feat_df is None: continue

        train_ds, val_ds, test_ds, _ = build_splits(
            feat_df, config.seq_len, config.pred_len, config.TRAIN_RATIO, config.VAL_RATIO
        )
        train_dsets.append(train_ds)
        val_dsets.append(val_ds)
        test_dsets.append(test_ds)

    def _loader(dsets, shuffle: bool):
        return DataLoader(
            ConcatDataset(dsets),
            batch_size=config.BATCH_SIZE,
            shuffle=shuffle,
            num_workers=config.NUM_WORKERS,
            pin_memory=config.PIN_MEMORY and torch.cuda.is_available(),
        )

    return _loader(train_dsets, True), _loader(val_dsets, False), _loader(test_dsets, False)

def train(config=DailyConfig, run_name: str = "daily_v1"):
    torch.manual_seed(config.SEED)
    
    # Initialize clean logger
    logger = RunLogger(config.LOG_DIR, run_name)
    
    print(f"\n🚀 Starting CARD Training [{run_name}]")
    print(f"   Mode: Daily NSE (25 Stocks) | Horizon: {config.pred_len} days")
    print(f"   Log: {logger.log_path}\n")

    # 1. DataLoaders
    train_loader, val_loader, test_loader = _build_loaders(config)

    # 2. Model
    model_ns = SimpleNamespace(**{k: v for k, v in vars(config).items() if not k.isupper()})
    model = CARD(model_ns).to(config.DEVICE)
    
    # 3. Optimization
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.BASE_LR, weight_decay=config.WEIGHT_DECAY)
    scheduler = WarmupCosineLR(optimizer, config.WARMUP_EPOCHS, config.MAX_EPOCHS, config.BASE_LR, config.MIN_LR)
    criterion = CombinedReturnLoss(config.DIRECTIONAL_WEIGHT, config.MAGNITUDE_SCALE)
    amp_scaler = torch.cuda.amp.GradScaler(enabled=config.USE_AMP and torch.cuda.is_available())

    # 4. Training loop
    best_val_loss = float("inf")
    best_val_dir_acc = 0.0
    epochs_no_improve = 0

    start_time = time.time()
    for epoch in range(config.MAX_EPOCHS):
        # Train
        train_loss, train_dir = train_epoch(
            model, train_loader, criterion, optimizer, scheduler, amp_scaler, config, logger, epoch
        )
        
        # Validate
        val_loss, val_dir = validate_epoch(model, val_loader, criterion, config, logger, epoch)
        
        # Clean terminal output (No detail, just status)
        print(f"Epoch {epoch+1:02d}/{config.MAX_EPOCHS} | "
              f"T-Loss: {train_loss:.4f} | V-Loss: {val_loss:.4f} | "
              f"Dir: {val_dir:.1f}%")

        # Checkpoints
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_dir_acc = val_dir
            epochs_no_improve = 0
            ckpt = os.path.join(config.CHECKPOINT_DIR, f"{run_name}_best_loss.pt")
            save_checkpoint(model, optimizer, scheduler, epoch, val_loss, ckpt, logger)
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= config.EARLY_STOP_PATIENCE:
            print(f"\n⚠️ Early stopping at epoch {epoch+1}")
            break

    total_time = (time.time() - start_time) / 60
    print(f"\n✅ Done! Best Val Loss: {best_val_loss:.4f} | Dir Acc: {best_val_dir_acc:.1f}%")
    print(f"   Time taken: {total_time:.1f} minutes\n")

if __name__ == "__main__":
    train()
