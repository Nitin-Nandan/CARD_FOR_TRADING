"""
CARD Stock Prediction — Training Script
Phase 0 Clean Rewrite

Changes from old version:
- Predicts close prices (not returns)
- Single-task loss: SignalDecayMAE on close prices
- Removed MultiTaskLoss, scale_factor hack, volatility head
- Dataset loads y_close instead of y_returns/y_volatility
- Clean NaN detection
- LR=1e-4 (no artificial reduction)

Usage:
    python train/04_train_card.py
    python train/04_train_card.py --epochs 5 --batch_size 64
"""

import sys
import os
import argparse
import json
import time
from pathlib import Path
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
import numpy as np

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.append(str(PROJECT_ROOT))
sys.path.append(str(PROJECT_ROOT / "train"))

from train.config import TrainingConfig, CARDModelConfig
from train.stock_dataset import StockWindowsDataset, MultiStockDataLoader
from train.metrics import MetricsCalculator, format_metrics, print_metrics_table

sys.path.append(str(PROJECT_ROOT / "models"))
from models.card_true import CARD, SignalDecayMAE


# ============================================================================
# TRAINER
# ============================================================================

class Trainer:
    def __init__(self, config: TrainingConfig):
        self.config = config

        config.CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)
        config.LOGS_DIR.mkdir(parents=True, exist_ok=True)

        self.device = torch.device(config.DEVICE)
        print(f"\nDevice: {self.device}")
        if torch.cuda.is_available():
            print(f"GPU   : {torch.cuda.get_device_name(0)}")

        # Model
        print("\nBuilding CARD model...")
        model_config = CARDModelConfig(config)
        self.model = CARD(model_config).to(self.device)
        total_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"Parameters: {total_params:,}")

        # Loss — clean, no scale hack
        self.criterion = SignalDecayMAE(horizon=config.PRED_LEN).to(self.device)

        # Optimizer
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=config.LEARNING_RATE,
            weight_decay=config.WEIGHT_DECAY
        )

        # Scheduler
        self.scheduler = self._build_scheduler()

        # AMP
        self.scaler = GradScaler() if config.USE_AMP else None

        # Metrics
        self.train_metrics = MetricsCalculator()
        self.val_metrics   = MetricsCalculator()

        # State
        self.current_epoch          = 0
        self.best_val_loss          = float('inf')
        self.epochs_without_improve = 0
        self.history = {'train_loss': [], 'val_loss': [], 'train_metrics': [], 'val_metrics': []}

    def _build_scheduler(self):
        if self.config.LR_SCHEDULER == 'cosine':
            return optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.config.NUM_EPOCHS,
                eta_min=self.config.MIN_LR
            )
        elif self.config.LR_SCHEDULER == 'step':
            return optim.lr_scheduler.StepLR(self.optimizer, step_size=10, gamma=0.5)
        return None

    # -----------------------------------------------------------------------
    # TRAINING EPOCH
    # -----------------------------------------------------------------------
    def train_epoch(self, train_loader):
        self.model.train()
        self.train_metrics.reset()

        epoch_loss   = 0.0
        num_batches  = len(train_loader)
        skipped      = 0
        pbar         = tqdm(train_loader, desc=f"Train E{self.current_epoch}")

        for batch_idx, (X, y_close, _) in enumerate(pbar):
            # X: (B, 60, 18) raw prices  →  permute  →  (B, 18, 60)
            X       = X.to(self.device).permute(0, 2, 1)
            y_close = y_close.to(self.device)   # (B, 15) — raw close prices in ₹

            self.optimizer.zero_grad()

            # Forward: model RevIN-normalizes X internally, predicts, RevIN-denorms to raw ₹
            pred_close, (revin_mean, revin_std) = self.model(X)   # (B, 15) raw ₹

            # Compute MSE loss in RevIN-normalized space
            y_close_norm    = (y_close - revin_mean) / revin_std  # (B, 15)
            pred_close_norm = (pred_close - revin_mean) / revin_std  # (B, 15)
            mse_loss = self.criterion(pred_close_norm, y_close_norm)

            # --- DIRECTIONAL PENALTY (Trend-following) ---
            # Penalize model if it predicts the wrong direction of the trend relative to the last known price.
            # X shape: (B, 18, 60), close channel = config.CLOSE_CHANNEL_IDX
            x_last = X[:, self.config.CLOSE_CHANNEL_IDX, -1].unsqueeze(-1)  # (B, 1)
            x_last_norm = (x_last - revin_mean) / revin_std                 # (B, 1)

            # Step-over-step differences (first step relative to last known price)
            y_diff = torch.cat([y_close_norm[:, 0:1] - x_last_norm, torch.diff(y_close_norm, dim=1)], dim=1)
            pred_diff = torch.cat([pred_close_norm[:, 0:1] - x_last_norm, torch.diff(pred_close_norm, dim=1)], dim=1)

            # ReLU(-sign(true_diff) * pred_diff) applies a positive penalty only when direction is WRONG.
            dir_penalty = torch.mean(torch.relu(-torch.sign(y_diff) * pred_diff))
            
            # Total Loss
            loss = mse_loss + self.config.DIR_LOSS_WEIGHT * dir_penalty

            # NaN/Inf guard — check BEFORE backward to avoid corrupting gradients
            if not torch.isfinite(loss):
                skipped += 1
                if skipped <= 5:
                    print(f"\n  ⚠️  Bad batch {batch_idx}: loss={loss.item():.4f}, "
                          f"pred_range=[{pred_close.min():.1f}, {pred_close.max():.1f}] — skipping")
                self.optimizer.zero_grad()
                continue

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.MAX_GRAD_NORM)
            self.optimizer.step()

            epoch_loss += loss.item()
            # Metrics on RAW ₹ prices (meaningful for humans)
            self.train_metrics.update(pred_close.detach(), y_close.detach())

            if batch_idx % self.config.PRINT_FREQ == 0:
                postfix = {'loss': f"{loss.item():.4f}",
                           'lr':   f"{self.optimizer.param_groups[0]['lr']:.2e}"}
                if torch.cuda.is_available():
                    postfix['gpu'] = f"{torch.cuda.memory_allocated()/1e9:.1f}GB"
                if skipped:
                    postfix['skip'] = skipped
                pbar.set_postfix(postfix)

        effective = num_batches - skipped
        avg_loss  = epoch_loss / max(effective, 1)
        if skipped:
            print(f"\n  ⚠️  Skipped {skipped}/{num_batches} bad batches this epoch")
        metrics = self.train_metrics.compute_and_reset()
        return avg_loss, metrics

    # -----------------------------------------------------------------------
    # VALIDATION
    # -----------------------------------------------------------------------
    def validate(self, val_loader):
        self.model.eval()
        self.val_metrics.reset()

        epoch_loss  = 0.0
        num_batches = len(val_loader)

        with torch.no_grad():
            for X, y_close, _ in tqdm(val_loader, desc="Val"):
                X       = X.to(self.device).permute(0, 2, 1)
                y_close = y_close.to(self.device)

                pred_close, (revin_mean, revin_std) = self.model(X)
                y_close_norm    = (y_close - revin_mean) / revin_std
                pred_close_norm = (pred_close - revin_mean) / revin_std
                mse_loss = self.criterion(pred_close_norm, y_close_norm)

                # --- DIRECTIONAL PENALTY (Val) ---
                x_last = X[:, self.config.CLOSE_CHANNEL_IDX, -1].unsqueeze(-1)
                x_last_norm = (x_last - revin_mean) / revin_std
                
                y_diff = torch.cat([y_close_norm[:, 0:1] - x_last_norm, torch.diff(y_close_norm, dim=1)], dim=1)
                pred_diff = torch.cat([pred_close_norm[:, 0:1] - x_last_norm, torch.diff(pred_close_norm, dim=1)], dim=1)
                
                dir_penalty = torch.mean(torch.relu(-torch.sign(y_diff) * pred_diff))
                loss = mse_loss + self.config.DIR_LOSS_WEIGHT * dir_penalty

                if torch.isfinite(loss):
                    epoch_loss += loss.item()
                # Metrics on raw ₹
                self.val_metrics.update(pred_close.detach(), y_close.detach())

        avg_loss = epoch_loss / num_batches
        metrics  = self.val_metrics.compute_and_reset()
        return avg_loss, metrics

    # -----------------------------------------------------------------------
    # CHECKPOINT
    # -----------------------------------------------------------------------
    def save_checkpoint(self, val_loss, is_best=False):
        ckpt = {
            'epoch':             self.current_epoch,
            'model_state_dict':  self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'val_loss':          val_loss,
            'best_val_loss':     self.best_val_loss,
            'history':           self.history,
            'config':            self.config.to_dict(),
        }
        path = self.config.CHECKPOINTS_DIR / f"checkpoint_epoch_{self.current_epoch}.pt"
        torch.save(ckpt, path)
        print(f"  Saved: {path}")

        if is_best:
            best_path = self.config.CHECKPOINTS_DIR / "best_model.pt"
            torch.save(ckpt, best_path)
            print(f"  ⭐ Best model: {best_path}")

        # Prune old checkpoints
        all_ckpts = sorted(self.config.CHECKPOINTS_DIR.glob("checkpoint_epoch_*.pt"))
        for old in all_ckpts[:-self.config.KEEP_LAST_N]:
            old.unlink()

    # -----------------------------------------------------------------------
    # MAIN LOOP  (window sampling — all stocks every epoch)
    # -----------------------------------------------------------------------
    def train(self):
        """
        Train with window-sampling: ALL stocks appear every epoch.
        A new random sample of MAX_WINDOWS_PER_STOCK is drawn each epoch,
        guaranteeing diversity without catastrophic forgetting.
        """
        print("\n" + "="*70)
        print("STARTING TRAINING  — all stocks every epoch (window sampling)")
        print(f"Stocks: {len(self.config.STOCKS)}  |  "
              f"{self.config.MAX_WINDOWS_PER_STOCK} windows/stock/epoch  |  "
              f"{len(self.config.STOCKS)*self.config.MAX_WINDOWS_PER_STOCK:,} total/epoch")
        print("="*70)
        t0 = time.time()

        # Build val dataset once (fixed sample, same every epoch)
        print("\nBuilding validation dataset (fixed sample)...")
        val_ds = StockWindowsDataset(
            split       = 'val',
            windows_dir = self.config.WINDOWS_DIR,
            stocks      = self.config.STOCKS,
            train_ratio = self.config.TRAIN_RATIO,
            val_ratio   = self.config.VAL_RATIO,
            test_ratio  = self.config.TEST_RATIO,
            seed        = self.config.SEED,
            max_windows_per_stock = self.config.MAX_VAL_WINDOWS_PER_STOCK,
        )
        val_loader = MultiStockDataLoader(
            val_ds,
            batch_size  = self.config.BATCH_SIZE,
            shuffle     = False,
            num_workers = self.config.NUM_WORKERS,
            pin_memory  = self.config.PIN_MEMORY,
        )
        print(f"Val batches: {len(val_loader):,}")

        for epoch in range(1, self.config.NUM_EPOCHS + 1):
            self.current_epoch = epoch

            # Re-sample training windows each epoch (different seed → different windows)
            train_ds = StockWindowsDataset(
                split       = 'train',
                windows_dir = self.config.WINDOWS_DIR,
                stocks      = self.config.STOCKS,
                train_ratio = self.config.TRAIN_RATIO,
                val_ratio   = self.config.VAL_RATIO,
                test_ratio  = self.config.TEST_RATIO,
                seed        = self.config.SEED + epoch,   # new sample every epoch
                max_windows_per_stock = self.config.MAX_WINDOWS_PER_STOCK,
            )
            train_loader = MultiStockDataLoader(
                train_ds,
                batch_size  = self.config.BATCH_SIZE,
                shuffle     = True,
                num_workers = self.config.NUM_WORKERS,
                pin_memory  = self.config.PIN_MEMORY,
            )

            print(f"\nEpoch {epoch}/{self.config.NUM_EPOCHS}  "
                  f"lr={self.optimizer.param_groups[0]['lr']:.2e}  "
                  f"batches={len(train_loader):,}")
            print("-"*70)

            train_loss, train_mets = self.train_epoch(train_loader)
            val_loss,   val_mets   = self.validate(val_loader)

            # Free train RAM before next epoch
            del train_ds, train_loader

            if self.scheduler:
                self.scheduler.step()

            print(f"\nTrain loss: {train_loss:.5f}   Val loss: {val_loss:.5f}")
            print_metrics_table(train_mets, val_mets, epoch)

            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['train_metrics'].append(train_mets)
            self.history['val_metrics'].append(val_mets)

            is_best = val_loss < self.best_val_loss - self.config.EARLY_STOP_DELTA
            if is_best:
                self.best_val_loss          = val_loss
                self.epochs_without_improve = 0
            else:
                self.epochs_without_improve += 1

            if epoch % self.config.SAVE_FREQ == 0 or is_best:
                self.save_checkpoint(val_loss, is_best)

            if self.epochs_without_improve >= self.config.EARLY_STOP_PATIENCE:
                print(f"\n⚠️  Early stopping at epoch {epoch}")
                break

        elapsed = time.time() - t0
        print("\n" + "="*70)
        print(f"DONE  |  {elapsed/3600:.2f} h  |  best val loss: {self.best_val_loss:.5f}")
        print("="*70)

        hist_path = self.config.LOGS_DIR / f"history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(hist_path, 'w') as f:
            json.dump(
                {k: [float(x) if not isinstance(x, dict) else {kk: float(vv) for kk, vv in x.items()}
                     for x in v]
                 for k, v in self.history.items()},
                f, indent=2
            )
        print(f"History saved: {hist_path}")


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Train CARD (close price prediction)")
    parser.add_argument('--epochs',              type=int,   default=None)
    parser.add_argument('--batch_size',          type=int,   default=None)
    parser.add_argument('--lr',                  type=float, default=None)
    parser.add_argument('--device',              type=str,   default=None)
    parser.add_argument('--windows_per_stock',   type=int,   default=None,
                        help="Override MAX_WINDOWS_PER_STOCK from config")
    args = parser.parse_args()

    config = TrainingConfig
    if args.epochs:            config.NUM_EPOCHS              = args.epochs
    if args.batch_size:        config.BATCH_SIZE              = args.batch_size
    if args.lr:                config.LEARNING_RATE           = args.lr
    if args.device:            config.DEVICE                  = args.device
    if args.windows_per_stock: config.MAX_WINDOWS_PER_STOCK   = args.windows_per_stock

    config.print_config()

    torch.manual_seed(config.SEED)
    np.random.seed(config.SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(config.SEED)

    trainer = Trainer(config)
    trainer.train()
    print("\n✅ Training complete!")


if __name__ == "__main__":
    main()
