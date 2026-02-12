"""
CARD Stock Prediction - Training Script

Complete training pipeline with:
- TRUE CARD model
- Multi-task loss (returns + volatility)
- Mixed precision training
- Checkpointing
- Comprehensive metrics
- Early stopping

Usage:
    python pipeline/04_train_card.py
    
    # Or with custom config
    python pipeline/04_train_card.py --epochs 100 --batch_size 32
"""

import sys
import os
import argparse
import json
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
import numpy as np

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.append(str(PROJECT_ROOT))

# Import our modules
from config import TrainingConfig, CARDModelConfig
from stock_dataset import StockWindowsDataset, MultiStockDataLoader
from metrics import MetricsCalculator, format_metrics, print_metrics_table

# Import TRUE CARD
sys.path.append(str(PROJECT_ROOT / "models"))
from card_true import CARD, MultiTaskLoss


class Trainer:
    """
    Complete training orchestration for CARD
    
    Handles:
    - Training loop
    - Validation
    - Checkpointing
    - Metrics tracking
    - Early stopping
    """
    
    def __init__(self, config: TrainingConfig):
        self.config = config
        
        # Create directories
        config.CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)
        config.LOGS_DIR.mkdir(parents=True, exist_ok=True)
        
        # Setup device
        self.device = torch.device(config.DEVICE)
        print(f"\nUsing device: {self.device}")
        
        if torch.cuda.is_available():
            print(f"GPU: {torch.cuda.get_device_name(0)}")
            print(f"CUDA Version: {torch.version.cuda}")
        
        # Build model
        print("\nBuilding TRUE CARD model...")
        model_config = CARDModelConfig(config)
        self.model = CARD(model_config).to(self.device)
        
        total_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"Model parameters: {total_params:,}")
        
        # Loss function
        self.criterion = MultiTaskLoss(
            horizon=config.PRED_LEN,
            alpha=config.LOSS_ALPHA,
            beta=config.LOSS_BETA
        ).to(self.device)  # Move to GPU to match model device
        
        # Optimizer
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=config.LEARNING_RATE,
            weight_decay=config.WEIGHT_DECAY
        )
        
        # Learning rate scheduler
        self.scheduler = self._build_scheduler()
        
        # Mixed precision training
        self.scaler = GradScaler() if config.USE_AMP else None
        
        # Metrics
        self.train_metrics = MetricsCalculator()
        self.val_metrics = MetricsCalculator()
        
        # Tracking
        self.current_epoch = 0
        self.best_val_loss = float('inf')
        self.epochs_without_improvement = 0
        self.training_history = {
            'train_loss': [],
            'val_loss': [],
            'train_metrics': [],
            'val_metrics': []
        }
    
    def _build_scheduler(self):
        """Build learning rate scheduler"""
        if self.config.LR_SCHEDULER == 'cosine':
            return optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.config.NUM_EPOCHS,
                eta_min=self.config.MIN_LR
            )
        elif self.config.LR_SCHEDULER == 'step':
            return optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=10,
                gamma=0.5
            )
        else:
            return None
    
    def train_epoch(self, train_loader):
        """Train for one epoch"""
        self.model.train()
        self.train_metrics.reset()
        
        epoch_loss = 0
        num_batches = len(train_loader)
        
        pbar = tqdm(train_loader, desc=f"Epoch {self.current_epoch}")
        
        for batch_idx, (X, y_returns, y_volatility, _) in enumerate(pbar):
            # Move to device
            X = X.to(self.device)  # (batch, 60, 18)
            y_returns = y_returns.to(self.device)  # (batch, 15)
            y_volatility = y_volatility.to(self.device)  # (batch, 15)
            
            # CARD expects (batch, channels, seq_len)
            # But we only have returns for single "channel" output
            # Need to reshape: (batch, 60, 18) -> (batch, 18, 60)
            X = X.permute(0, 2, 1)
            
            # Forward pass
            self.optimizer.zero_grad()
            
            if self.config.USE_AMP:
                with autocast():
                    pred_returns, pred_volatility = self.model(X)
                    
                    # pred_returns: (batch, 18, 15) - but we only care about close price
                    # Take channel 0 (close price predictions)
                    pred_returns = pred_returns[:, 0, :]  # (batch, 15)
                    pred_volatility = pred_volatility[:, 0, :]  # (batch, 15)
                    
                    loss, loss_dict = self.criterion(
                        pred_returns, pred_volatility,
                        y_returns, y_volatility
                    )
                
                # Backward pass
                self.scaler.scale(loss).backward()
                
                # Gradient clipping
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.MAX_GRAD_NORM
                )
                
                # Optimizer step
                self.scaler.step(self.optimizer)
                self.scaler.update()
            
            else:
                pred_returns, pred_volatility = self.model(X)
                pred_returns = pred_returns[:, 0, :]
                pred_volatility = pred_volatility[:, 0, :]
                
                loss, loss_dict = self.criterion(
                    pred_returns, pred_volatility,
                    y_returns, y_volatility
                )
                
                loss.backward()
                
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.MAX_GRAD_NORM
                )
                
                self.optimizer.step()
            
            # Update metrics
            epoch_loss += loss.item()
            self.train_metrics.update(pred_returns, y_returns, pred_volatility, y_volatility)
            
            # Update progress bar with GPU memory monitoring
            if batch_idx % self.config.PRINT_FREQ == 0:
                postfix_dict = {
                    'loss': f"{loss.item():.6f}",
                    'lr': f"{self.optimizer.param_groups[0]['lr']:.6f}"
                }
                
                # Add GPU memory if available
                if torch.cuda.is_available():
                    gpu_mem_gb = torch.cuda.memory_allocated() / 1024**3
                    postfix_dict['gpu_mem'] = f"{gpu_mem_gb:.2f}GB"
                
                pbar.set_postfix(postfix_dict)
        
        avg_loss = epoch_loss / num_batches
        metrics = self.train_metrics.compute_and_reset()
        
        return avg_loss, metrics
    
    def validate(self, val_loader):
        """Validate model"""
        self.model.eval()
        self.val_metrics.reset()
        
        epoch_loss = 0
        num_batches = len(val_loader)
        
        with torch.no_grad():
            for X, y_returns, y_volatility, _ in tqdm(val_loader, desc="Validating"):
                X = X.to(self.device).permute(0, 2, 1)
                y_returns = y_returns.to(self.device)
                y_volatility = y_volatility.to(self.device)
                
                # Forward pass
                pred_returns, pred_volatility = self.model(X)
                pred_returns = pred_returns[:, 0, :]
                pred_volatility = pred_volatility[:, 0, :]
                
                loss, _ = self.criterion(
                    pred_returns, pred_volatility,
                    y_returns, y_volatility
                )
                
                epoch_loss += loss.item()
                self.val_metrics.update(pred_returns, y_returns, pred_volatility, y_volatility)
        
        avg_loss = epoch_loss / num_batches
        metrics = self.val_metrics.compute_and_reset()
        
        return avg_loss, metrics
    
    def save_checkpoint(self, val_loss, is_best=False):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'val_loss': val_loss,
            'best_val_loss': self.best_val_loss,
            'training_history': self.training_history,
            'config': self.config.to_dict()
        }
        
        # Save regular checkpoint
        checkpoint_path = self.config.CHECKPOINTS_DIR / f"checkpoint_epoch_{self.current_epoch}.pt"
        torch.save(checkpoint, checkpoint_path)
        print(f"  Saved checkpoint: {checkpoint_path}")
        
        # Save best model
        if is_best:
            best_path = self.config.CHECKPOINTS_DIR / "best_model.pt"
            torch.save(checkpoint, best_path)
            print(f"  ⭐ New best model saved: {best_path}")
        
        # Clean up old checkpoints (keep last N)
        checkpoints = sorted(self.config.CHECKPOINTS_DIR.glob("checkpoint_epoch_*.pt"))
        if len(checkpoints) > self.config.KEEP_LAST_N:
            for old_checkpoint in checkpoints[:-self.config.KEEP_LAST_N]:
                old_checkpoint.unlink()
    
    def train(self, train_loader, val_loader):
        """Complete training loop"""
        print("\n" + "="*70)
        print("STARTING TRAINING")
        print("="*70)
        
        start_time = time.time()
        
        for epoch in range(1, self.config.NUM_EPOCHS + 1):
            self.current_epoch = epoch
            
            print(f"\nEpoch {epoch}/{self.config.NUM_EPOCHS}")
            print("-"*70)
            
            # Train
            train_loss, train_metrics = self.train_epoch(train_loader)
            
            # Validate
            if epoch % self.config.VAL_FREQ == 0:
                val_loss, val_metrics = self.validate(val_loader)
            else:
                val_loss, val_metrics = None, {}
            
            # Update scheduler
            if self.scheduler:
                self.scheduler.step()
            
            # Print metrics
            print("\n" + "-"*70)
            print(f"Train Loss: {train_loss:.6f}")
            if val_loss is not None:
                print(f"Val Loss:   {val_loss:.6f}")
            print("-"*70)
            
            if val_metrics:
                print_metrics_table(train_metrics, val_metrics, epoch)
            
            # Save history
            self.training_history['train_loss'].append(train_loss)
            self.training_history['train_metrics'].append(train_metrics)
            if val_loss is not None:
                self.training_history['val_loss'].append(val_loss)
                self.training_history['val_metrics'].append(val_metrics)
            
            # Check for improvement
            if val_loss is not None:
                is_best = val_loss < self.best_val_loss - self.config.EARLY_STOP_DELTA
                
                if is_best:
                    self.best_val_loss = val_loss
                    self.epochs_without_improvement = 0
                else:
                    self.epochs_without_improvement += 1
                
                # Save checkpoint
                if epoch % self.config.SAVE_FREQ == 0 or is_best:
                    self.save_checkpoint(val_loss, is_best)
                
                # Early stopping
                if self.epochs_without_improvement >= self.config.EARLY_STOP_PATIENCE:
                    print(f"\n⚠️  Early stopping triggered after {epoch} epochs")
                    print(f"   No improvement for {self.config.EARLY_STOP_PATIENCE} epochs")
                    break
        
        # Training complete
        elapsed = time.time() - start_time
        print("\n" + "="*70)
        print("TRAINING COMPLETE")
        print("="*70)
        print(f"Total time: {elapsed/3600:.2f} hours")
        print(f"Best val loss: {self.best_val_loss:.6f}")
        print(f"Final learning rate: {self.optimizer.param_groups[0]['lr']:.6f}")
        
        # Save final model
        self.save_checkpoint(val_loss, is_best=False)
        
        # Save training history
        history_path = self.config.LOGS_DIR / f"training_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(history_path, 'w') as f:
            # Convert numpy types to native Python types
            history_serializable = {
                'train_loss': [float(x) for x in self.training_history['train_loss']],
                'val_loss': [float(x) for x in self.training_history['val_loss']],
                'train_metrics': [
                    {k: float(v) for k, v in m.items()}
                    for m in self.training_history['train_metrics']
                ],
                'val_metrics': [
                    {k: float(v) for k, v in m.items()}
                    for m in self.training_history['val_metrics']
                ]
            }
            json.dump(history_serializable, f, indent=2)
        
        print(f"Training history saved: {history_path}")


def main():
    """Main training function"""
    # Parse arguments
    parser = argparse.ArgumentParser(description="Train CARD Stock Prediction")
    parser.add_argument('--epochs', type=int, default=None)
    parser.add_argument('--batch_size', type=int, default=None)
    parser.add_argument('--lr', type=float, default=None)
    parser.add_argument('--device', type=str, default=None)
    args = parser.parse_args()
    
    # Load config
    config = TrainingConfig
    
    # Override with CLI args
    if args.epochs:
        config.NUM_EPOCHS = args.epochs
    if args.batch_size:
        config.BATCH_SIZE = args.batch_size
    if args.lr:
        config.LEARNING_RATE = args.lr
    if args.device:
        config.DEVICE = args.device
    
    # Print configuration
    config.print_config()
    
    # Set random seeds
    torch.manual_seed(config.SEED)
    np.random.seed(config.SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(config.SEED)
    
    # Load datasets
    print("\nLoading datasets...")
    train_dataset = StockWindowsDataset(
        windows_dir=config.WINDOWS_DIR,
        stocks=config.STOCKS,
        split='train',
        train_ratio=config.TRAIN_RATIO,
        val_ratio=config.VAL_RATIO,
        test_ratio=config.TEST_RATIO
    )
    
    val_dataset = StockWindowsDataset(
        windows_dir=config.WINDOWS_DIR,
        stocks=config.STOCKS,
        split='val',
        train_ratio=config.TRAIN_RATIO,
        val_ratio=config.VAL_RATIO,
        test_ratio=config.TEST_RATIO
    )
    
    # Create data loaders
    train_loader = MultiStockDataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        pin_memory=config.PIN_MEMORY
    )
    
    val_loader = MultiStockDataLoader(
        val_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=config.PIN_MEMORY
    )
    
    print(f"\nTrain batches: {len(train_loader):,}")
    print(f"Val batches: {len(val_loader):,}")
    
    # Create trainer and train
    trainer = Trainer(config)
    trainer.train(train_loader, val_loader)
    
    print("\n✅ Training complete!")


if __name__ == "__main__":
    main()
