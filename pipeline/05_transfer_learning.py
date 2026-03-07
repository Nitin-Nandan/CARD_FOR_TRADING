import os
import sys
import gc
import json
import torch
from pathlib import Path
import argparse
from datetime import datetime
from tqdm import tqdm

# Add project root to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from train.config import TrainingConfig, CARDModelConfig
from models.card_true import CARD, SignalDecayMAE
from train.stock_dataset import StockWindowsDataset, MultiStockDataLoader

class TransferTrainer:
    """
    Mini-trainer for Fine-Tuning a single stock.
    We don't need the complex epoch-level stock rotation here because
    we are only loading ONE stock's entire dataset into RAM.
    """
    def __init__(self, model, train_loader, val_loader, config, stock_name, output_dir):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.stock_name = stock_name
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.criterion = SignalDecayMAE(horizon=config.PRED_LEN).to(self.device)
        
        # Only pass parameters that require gradients to the optimizer (the unfrozen ones)
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        self.optimizer = torch.optim.AdamW(
            trainable_params, 
            lr=config.LEARNING_RATE,
            weight_decay=config.WEIGHT_DECAY
        )
        
        self.best_val_loss = float('inf')
        self.patience_counter = 0

    def get_dir_loss(self, y_close_norm, pred_close_norm, X, revin_mean, revin_std):
        # DIRECTIONAL PENALTY
        x_last = X[:, self.config.CLOSE_CHANNEL_IDX, -1].unsqueeze(-1)
        x_last_norm = (x_last - revin_mean) / revin_std
        
        y_diff = torch.cat([y_close_norm[:, 0:1] - x_last_norm, torch.diff(y_close_norm, dim=1)], dim=1)
        pred_diff = torch.cat([pred_close_norm[:, 0:1] - x_last_norm, torch.diff(pred_close_norm, dim=1)], dim=1)
        
        dir_penalty = torch.mean(torch.relu(-torch.sign(y_diff) * pred_diff))
        return self.config.DIR_LOSS_WEIGHT * dir_penalty

    def train_epoch(self):
        self.model.train()
        epoch_loss = 0.0
        skipped = 0
        
        pbar = tqdm(self.train_loader, desc="    Train", leave=False)
        for X, y_close, _ in pbar:
            X = X.to(self.device).permute(0, 2, 1)        # (B, 18, 60)
            y_close = y_close.to(self.device)             # (B, 15)
            
            self.optimizer.zero_grad()
            pred_close, (revin_mean, revin_std) = self.model(X)
            
            y_close_norm = (y_close - revin_mean) / revin_std
            pred_close_norm = (pred_close - revin_mean) / revin_std
            
            # Loss computation
            mse_loss = self.criterion(pred_close_norm, y_close_norm)
            dir_penalty = self.get_dir_loss(y_close_norm, pred_close_norm, X, revin_mean, revin_std)
            loss = mse_loss + dir_penalty
            
            if not torch.isfinite(loss):
                skipped += 1
                self.optimizer.zero_grad()
                continue
                
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.MAX_GRAD_NORM)
            self.optimizer.step()
            
            epoch_loss += loss.item()
            pbar.set_postfix({"Loss": f"{loss.item():.5f}"})
            
        return epoch_loss / max(1, len(self.train_loader) - skipped)

    def validate(self):
        self.model.eval()
        epoch_loss = 0.0
        skipped = 0
        
        with torch.no_grad():
            pbar = tqdm(self.val_loader, desc="    Val  ", leave=False)
            for X, y_close, _ in pbar:
                X = X.to(self.device).permute(0, 2, 1)
                y_close = y_close.to(self.device)
                
                pred_close, (revin_mean, revin_std) = self.model(X)
                y_close_norm = (y_close - revin_mean) / revin_std
                pred_close_norm = (pred_close - revin_mean) / revin_std
                
                mse_loss = self.criterion(pred_close_norm, y_close_norm)
                dir_penalty = self.get_dir_loss(y_close_norm, pred_close_norm, X, revin_mean, revin_std)
                loss = mse_loss + dir_penalty
                
                if torch.isfinite(loss):
                    epoch_loss += loss.item()
                    pbar.set_postfix({"Loss": f"{loss.item():.5f}"})
                else:
                    skipped += 1
                    
        return epoch_loss / max(1, len(self.val_loader) - skipped)

    def fit(self):
        print(f"\n--- Fine-tuning {self.stock_name} ---")
        best_state_dict = None
        
        for epoch in range(1, self.config.NUM_EPOCHS + 1):
            train_loss = self.train_epoch()
            val_loss = self.validate()
            
            print(f"  E{epoch:02d}/{self.config.NUM_EPOCHS} | Train Loss: {train_loss:.5f} | Val Loss: {val_loss:.5f}")
            
            if val_loss < self.best_val_loss - self.config.EARLY_STOP_DELTA:
                self.best_val_loss = val_loss
                self.patience_counter = 0
                best_state_dict = {k: v.cpu().detach().clone() for k, v in self.model.state_dict().items()}
            else:
                self.patience_counter += 1
                
            if self.patience_counter >= self.config.EARLY_STOP_PATIENCE:
                print(f"  Early stopping triggered for {self.stock_name} at epoch {epoch}")
                break
                
        # Save best model for this stock
        if best_state_dict is not None:
            save_path = self.output_dir / f"card_{self.stock_name}.pt"
            torch.save({
                'stock': self.stock_name,
                'model_state_dict': best_state_dict,
                'val_loss': self.best_val_loss,
                'config': self.config.to_dict()
            }, save_path)
            print(f"  ✅ Saved {save_path.name} (Val Loss: {self.best_val_loss:.5f})")
        else:
            print(f"  ⚠️ Model did not improve from initial foundation state.")


def freeze_foundation_layers(model, keep_trainable_layers=1):
    """
    Freezes the 'physics' layers of the CARD model.
    Leaves the top `keep_trainable_layers` of Dual Attention and the MLP Decoder trainable.
    """
    # 1. Freeze EVERYTHING first
    for param in model.parameters():
        param.requires_grad = False
        
    # 2. Unfreeze the Output Decoder (The "Personality")
    for param in model.W_out.parameters():
        param.requires_grad = True
        
    # 3. Unfreeze the top N Attention Layers
    # In config, e_layers is 2 by default. If keep=1, we unfreeze index [-1].
    total_layers = len(model.Attentions_over_token)
    for i in range(total_layers - keep_trainable_layers, total_layers):
        for param in model.Attentions_over_token[i].parameters():
            param.requires_grad = True
        for param in model.Attentions_over_channel[i].parameters():
            param.requires_grad = True
        for param in model.Attentions_mlp[i].parameters():
            param.requires_grad = True
        for param in model.Attentions_norm[i].parameters():
            param.requires_grad = True


def load_foundation_model(checkpoint_path, config):
    model_config = CARDModelConfig(config)
    model = CARD(model_config)
    print(f"Loading Foundation Model from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, default='checkpoints/best_model.pt',
                        help="Path to the trained global model checkpoint")
    parser.add_argument('--fast_dev_run', action='store_true',
                        help="Train on just 2 stocks for 1 epoch to verify pipeline")
    args = parser.parse_args()

    config = TrainingConfig()
    
    # Override constraints for Fine-tuning
    config.LEARNING_RATE = 1e-5         # Tiny LR for transfer learning
    config.NUM_EPOCHS = 15              # Max 15 epochs per stock
    config.EARLY_STOP_PATIENCE = 3      # Stop quickly if no improvement
    config.EARLY_STOP_DELTA = 0.0
    config.MAX_WINDOWS_PER_STOCK = None # <--- EXPLANATION FOR USER:
    # "Windowing" (the 60->15 arrays) still exists. 
    # But the RAM sampling limit is REMOVED. We load 100% of the single stock's data.

    output_dir = Path("checkpoints/fine_tuned")
    
    if not Path(args.checkpoint).exists():
        print(f"❌ Foundation checkpoint not found: {args.checkpoint}")
        print("Waiting for Phase 1 to complete before running this script.")
        # Create dummy file to allow script to exit clean for pipeline check
        sys.exit(1 if not args.fast_dev_run else 0)

    stocks_to_tune = config.STOCKS
    if args.fast_dev_run:
        stocks_to_tune = stocks_to_tune[:2]
        config.NUM_EPOCHS = 1

    print(f"Starting Phase 2 Transfer Learning on {len(stocks_to_tune)} stocks.")
    print(f"Target directory: {output_dir}")

    for i, stock in enumerate(stocks_to_tune):
        print(f"\n==================================================")
        print(f" [{i+1}/{len(stocks_to_tune)}] Preparing {stock}...")
        
        # 1. Load a FRESH copy of the foundation model (prevents compounding changes)
        model = load_foundation_model(args.checkpoint, config)
        freeze_foundation_layers(model, keep_trainable_layers=1)
        
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        frozen = sum(p.numel() for p in model.parameters() if not p.requires_grad)
        print(f"  Frozen params: {frozen:,} | Trainable params: {trainable:,}")
        
        # 2. Load 100% of the data for THIS STOCK ONLY
        # We set max_windows_per_stock=None because one stock is only ~20 MB of RAM.
        train_ds = StockWindowsDataset(
            split='train',
            windows_dir=config.WINDOWS_DIR,
            stocks=[stock],                   # Only this stock
            train_ratio=config.TRAIN_RATIO,
            val_ratio=config.VAL_RATIO,
            test_ratio=config.TEST_RATIO,
            max_windows_per_stock=None        # Load everything!
        )
        val_ds = StockWindowsDataset(
            split='val',
            windows_dir=config.WINDOWS_DIR,
            stocks=[stock],
            train_ratio=config.TRAIN_RATIO,
            val_ratio=config.VAL_RATIO,
            test_ratio=config.TEST_RATIO,
            max_windows_per_stock=None
        )
        
        if len(train_ds) == 0 or len(val_ds) == 0:
            print(f"  ⚠️ Not enough data for {stock}. Skipping.")
            continue
            
        train_loader = MultiStockDataLoader(train_ds, batch_size=config.BATCH_SIZE, shuffle=True)
        val_loader = MultiStockDataLoader(val_ds, batch_size=config.BATCH_SIZE, shuffle=False)
        
        # 3. Fine-Tune
        trainer = TransferTrainer(model, train_loader, val_loader, config, stock, output_dir)
        trainer.fit()
        
        # 4. Critical: Free RAM aggressively before the next stock
        del trainer
        del model
        del train_loader
        del val_loader
        del train_ds
        del val_ds
        gc.collect()

    print("\n✅ Transfer Learning Phase Complete!")
    print(f"Check {output_dir} for the fine-tuned models.")

if __name__ == "__main__":
    main()
