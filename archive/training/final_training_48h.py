"""
48-HOUR SPRINT TRAINING
Trains CARD model on top 150 stocks for 30 epochs

Expected Time: 40-48 hours
Expected Result: 55-56% direction accuracy

Optimizations:
- Top 150 stocks by liquidity
- 40k windows per stock (capped)
- Aggressive learning rate (3e-3)
- Large batch size (128)
- Early stopping (patience=8)
"""

import sys
from pathlib import Path
import os

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

import torch
import numpy as np
from datetime import datetime, timedelta
import json
import importlib

from train.config import Config

# Dynamic import for numeric filename to avoid SyntaxError
train_card_mod = importlib.import_module("train.04_train_card")
train_phase4 = train_card_mod.train_phase4

def print_header(text, char="="):
    """Print formatted header"""
    print("\n" + char*80)
    print(text)
    print(char*80)

def format_time(hours):
    """Format hours into days and hours"""
    days = int(hours // 24)
    hrs = hours % 24
    if days > 0:
        return f"{days}d {hrs:.1f}h"
    return f"{hrs:.1f}h"

# ============================================
# HEADER
# ============================================

print_header("48-HOUR SPRINT TRAINING - CARD STOCK PREDICTION")

print("\n⚡ SPEED OPTIMIZATIONS ENABLED:")
print("  • Top 150 stocks (high liquidity)")
print("  • 30 epochs max (vs 200 standard)")
print("  • 40k windows/stock (vs 100k standard)")
print("  • Batch size 128 (vs 64 standard)")
print("  • Learning rate 3e-3 (vs 1e-3 standard)")
print("  • Early stop patience 8 (vs 40 standard)")

# ============================================
# CONFIGURATION
# ============================================

config = Config()

print_header("CONFIGURATION", "-")

print(f"\nHyperparameters:")
print(f"  Learning Rate: {config.BASE_LR:.2e}")
print(f"  Directional Weight: {config.DIRECTIONAL_WEIGHT}")
print(f"  Magnitude Scale: {config.MAGNITUDE_SCALE}")
print(f"  Gradient Clip: {config.GRAD_CLIP_NORM}")
print(f"  Dropout: {config.DROPOUT}")
print(f"  Weight Decay: {config.WEIGHT_DECAY:.2e}")

print(f"\nTraining Settings:")
print(f"  Max Epochs: {config.MAX_EPOCHS}")
print(f"  Batch Size: {config.BATCH_SIZE}")
print(f"  Effective Batch: {config.BATCH_SIZE * config.ACCUMULATION_STEPS}")
print(f"  Accumulation Steps: {config.ACCUMULATION_STEPS}")
print(f"  Warmup Epochs: {config.WARMUP_EPOCHS}")
print(f"  LR Schedule: {config.LR_SCHEDULE}")
print(f"  Early Stop Patience: {config.EARLY_STOP_PATIENCE}")

print(f"\nOptimizations:")
print(f"  Mixed Precision (AMP): {config.USE_AMP}")
print(f"  Stock Balancing: {config.BALANCE_STOCKS}")
print(f"  Max Windows/Stock: {config.MAX_WINDOWS_PER_STOCK:,}")

print(f"\nPaths:")
print(f"  Checkpoints: {config.CHECKPOINT_DIR}")
print(f"  Logs: {config.LOG_DIR}")

# Create directories
Path(config.CHECKPOINT_DIR).mkdir(parents=True, exist_ok=True)
Path(config.LOG_DIR).mkdir(parents=True, exist_ok=True)

# ============================================
# SELECT TOP 150 STOCKS BY LIQUIDITY
# ============================================

print_header("SELECTING TOP 150 STOCKS BY LIQUIDITY")

TOP_N = 150

window_dir = Path('data/windows')
all_npz_files = list(window_dir.glob('*_windows.npz'))

print(f"\nFound {len(all_npz_files)} stock files")
print("Analyzing liquidity (window counts)...")

# Get window counts for all stocks
stock_liquidity = []

for npz_file in all_npz_files:
    stock = npz_file.stem.replace('_windows', '')
    
    try:
        data = np.load(str(npz_file), mmap_mode='r')
        
        if 'valid_indices' not in data.files:
            continue
        
        window_count = len(data['valid_indices'])
        stock_liquidity.append((stock, window_count))
        
    except Exception as e:
        print(f"  Warning: Skipping {stock} - {e}")
        continue

# Sort by window count (descending)
stock_liquidity.sort(key=lambda x: x[1], reverse=True)

# Take top N
top_stocks = [stock for stock, count in stock_liquidity[:TOP_N]]
top_counts = [count for stock, count in stock_liquidity[:TOP_N]]

print(f"\n✓ Selected top {TOP_N} stocks by liquidity:")
print(f"\n  Top 5 stocks:")
for i in range(min(5, len(stock_liquidity))):
    stock, count = stock_liquidity[i]
    print(f"    {i+1}. {stock}: {count:,} windows")

print(f"\n  Stock #{TOP_N//2} (median):")
stock, count = stock_liquidity[TOP_N//2]
print(f"    {stock}: {count:,} windows")

print(f"\n  Stock #{TOP_N} (lowest in top {TOP_N}):")
stock, count = stock_liquidity[TOP_N-1]
print(f"    {stock}: {count:,} windows")

total_windows = sum(top_counts)
avg_windows = total_windows / TOP_N
median_windows = int(np.median(top_counts))

print(f"\n  Total windows: {total_windows:,}")
print(f"  Average per stock: {avg_windows:,.0f}")
print(f"  Median per stock: {median_windows:,}")

# ============================================
# TIME ESTIMATE
# ============================================

print_header("TIME ESTIMATE")

# Based on rescue test results
SECONDS_PER_BATCH = 0.25  # Conservative estimate from your test

# Calculate with capping
capped_median = min(median_windows, config.MAX_WINDOWS_PER_STOCK)
samples_per_epoch = TOP_N * capped_median
effective_batch = config.BATCH_SIZE * config.ACCUMULATION_STEPS
batches_per_epoch = samples_per_epoch // effective_batch

train_time_per_epoch = (batches_per_epoch * SECONDS_PER_BATCH) / 3600
val_batches = int(batches_per_epoch * 0.15)  # Assume val is 15% of train
val_time_per_epoch = (val_batches * SECONDS_PER_BATCH * 0.4) / 3600  # Val faster
total_time_per_epoch = train_time_per_epoch + val_time_per_epoch

total_time_30_epochs = total_time_per_epoch * 30
total_time_with_early_stop = total_time_per_epoch * 22  # Assume stops at epoch 22

print(f"\nCalculations:")
print(f"  Stocks: {TOP_N}")
print(f"  Median windows (raw): {median_windows:,}")
print(f"  Median windows (capped): {capped_median:,}")
print(f"  Samples per epoch: {samples_per_epoch:,}")
print(f"  Effective batch size: {effective_batch}")
print(f"  Batches per epoch: {batches_per_epoch:,}")

print(f"\nTime Estimates:")
print(f"  Per epoch (train): {train_time_per_epoch:.2f}h")
print(f"  Per epoch (val): {val_time_per_epoch:.2f}h")
print(f"  Per epoch (total): {total_time_per_epoch:.2f}h")
print(f"  Full 30 epochs: {format_time(total_time_30_epochs)}")
print(f"  With early stop (~22 epochs): {format_time(total_time_with_early_stop)}")

if total_time_with_early_stop > 50:
    print("\n⚠️  WARNING: Estimated time exceeds 48 hours!")
    print("\n  Recommendations to speed up:")
    print(f"  1. Reduce MAX_EPOCHS to {int(48 / total_time_per_epoch)}")
    print(f"  2. Reduce TOP_N to {int(TOP_N * 48 / total_time_with_early_stop)}")
    print("  3. Reduce MAX_WINDOWS_PER_STOCK to 30000")
    print("  4. Increase BATCH_SIZE to 160 (if GPU allows)")
    
    print("\n  Current estimate is conservative - training may finish faster.")
    print("  AMP and optimizations often give 1.5-2x speedup in practice.")
    
elif total_time_with_early_stop < 36:
    print(f"\n✓ Estimated time is UNDER 48 hours ({format_time(total_time_with_early_stop)})")
    print("  You may want to increase MAX_EPOCHS or TOP_N for better results.")

# ============================================
# GPU MEMORY CHECK
# ============================================

print_header("GPU MEMORY CHECK")

if not torch.cuda.is_available():
    print("❌ No GPU available!")
    print("   Training on CPU will be EXTREMELY slow (100x slower)")
    print("   Estimated time: 200+ days")
    
    response = input("\n   Continue anyway? (y/n): ")
    if response.lower() != 'y':
        print("Aborted.")
        sys.exit(0)
else:
    print(f"✓ GPU available: {torch.cuda.get_device_name(0)}")
    
    # Test batch size
    print(f"\nTesting batch size {config.BATCH_SIZE}...")
    
    try:
        from models.card_true import CARD
        
        test_model = CARD(config).cuda()
        test_batch = torch.randn(config.BATCH_SIZE, 81, 60).cuda()
        
        with torch.no_grad():
            _ = test_model(test_batch)
        
        mem_allocated = torch.cuda.memory_allocated() / 1e9
        mem_total = torch.cuda.get_device_properties(0).total_memory / 1e9
        mem_pct = (mem_allocated / mem_total) * 100
        
        print(f"✓ Batch size {config.BATCH_SIZE} fits in GPU")
        print(f"  Memory used: {mem_allocated:.2f} GB / {mem_total:.2f} GB ({mem_pct:.1f}%)")
        
        if mem_pct > 90:
            print(f"\n⚠️  WARNING: Using {mem_pct:.1f}% of GPU memory")
            print("   Risk of OOM errors during training")
            print("   Recommendation: Reduce BATCH_SIZE to 96 or 64")
            
            response = input("\n   Continue anyway? (y/n): ")
            if response.lower() != 'y':
                print("Aborted. Please update BATCH_SIZE in train/config.py")
                sys.exit(0)
        elif mem_pct < 70:
            print(f"\n  You have headroom - could increase BATCH_SIZE to {int(config.BATCH_SIZE * 1.2)}")
        
        # Cleanup
        del test_model, test_batch
        torch.cuda.empty_cache()
        
    except RuntimeError as e:
        print(f"\n❌ Batch size {config.BATCH_SIZE} is TOO LARGE!")
        print(f"   Error: {e}")
        
        print("\n   Automatically reducing to BATCH_SIZE=64...")
        config.BATCH_SIZE = 64
        
        # Recalculate time with smaller batch
        effective_batch = config.BATCH_SIZE * config.ACCUMULATION_STEPS
        batches_per_epoch = samples_per_epoch // effective_batch
        total_time_per_epoch = (batches_per_epoch * SECONDS_PER_BATCH) / 3600 * 1.4
        total_time_with_early_stop = total_time_per_epoch * 22
        
        print(f"   New time estimate: {format_time(total_time_with_early_stop)}")

# ============================================
# FINAL CONFIRMATION
# ============================================

print_header("READY TO START")

print(f"\nConfiguration Summary:")
print(f"  Stocks: {TOP_N} (top by liquidity)")
print(f"  Max Epochs: {config.MAX_EPOCHS}")
print(f"  Batch Size: {config.BATCH_SIZE}")
print(f"  Windows/Stock: {config.MAX_WINDOWS_PER_STOCK:,}")
print(f"  Learning Rate: {config.BASE_LR:.2e}")

print(f"\nEstimated Time: {format_time(total_time_with_early_stop)}")

print(f"\nExpected Results:")
print(f"  Epoch 5:  ~51% direction accuracy")
print(f"  Epoch 10: ~52-53% direction accuracy")
print(f"  Epoch 15: ~53-54% direction accuracy")
print(f"  Epoch 20: ~54-55% direction accuracy")
print(f"  Epoch 25: ~55-56% direction accuracy")
print(f"  Epoch 30: ~55-57% direction accuracy")

print(f"\nCheckpoints will be saved to:")
print(f"  {config.CHECKPOINT_DIR}/")

print("\n" + "="*80)
print("Starting in 10 seconds... (Press Ctrl+C to cancel)")
print("="*80)

import time
for i in range(10, 0, -1):
    print(f"{i}...", end=" ", flush=True)
    time.sleep(1)

print("\n")

# ============================================
# TRAINING
# ============================================

print_header("TRAINING STARTED", "=")
print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"Expected completion: {(datetime.now() + timedelta(hours=total_time_with_early_stop)).strftime('%Y-%m-%d %H:%M:%S')}")

try:
    start_time = datetime.now()
    
    # CRITICAL: Assign the selected top stocks back to config
    config.STOCKS = top_stocks
    
    # Run training
    results = train_phase4(
        config_class=lambda: config, # Use the factory to pass our configured instance
        run_name="card_48h_sprint",
        use_wandb=False 
    )
    
    if results is None:
        print("\n" + "="*80)
        print("❌ TRAINING FAILED TO START OR CRASHED")
        print("="*80)
        print("\nPossible reasons:")
        print("1. RAM Limit: Even with 90% reduction, OS memory might be too low.")
        print("2. GPU OOM: Batch size too large for RTX 4050.")
        print("3. Data Corruption: Check logs/final_training_48h/ for specific data errors.")
        sys.exit(1)
        
    end_time = datetime.now()
    duration_seconds = (end_time - start_time).total_seconds()
    duration_hours = duration_seconds / 3600
    
    # ========================================
    # TRAINING COMPLETE
    # ========================================
    
    print_header("✅ 48-HOUR TRAINING COMPLETE!", "=")
    
    print(f"\nCompleted at: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Duration: {format_time(duration_hours)} ({duration_hours:.2f} hours)")
    
    # Safely get metrics
    best_val_loss = results.get('val_loss', 0.0)
    best_val_dir = results.get('val_dir_acc', 0.0) / 100.0
    
    print(f"\nBest Validation Loss: {best_val_loss:.4f}")
    print(f"Best Direction Accuracy: {best_val_dir:.2%}")
    
    print(f"\nCheckpoints saved to {config.CHECKPOINT_DIR}/")
    
    # ========================================
    # SAVE SUMMARY
    # ========================================
    
    summary = {
        'training_type': '48_hour_sprint',
        'started_at': start_time.isoformat(),
        'completed_at': end_time.isoformat(),
        'duration_hours': duration_hours,
        'stocks_trained': TOP_N,
        'results': results,
        'config': {
            'MAX_EPOCHS': config.MAX_EPOCHS,
            'BATCH_SIZE': config.BATCH_SIZE,
            'BASE_LR': config.BASE_LR,
            'DIRECTIONAL_WEIGHT': config.DIRECTIONAL_WEIGHT,
            'MAGNITUDE_SCALE': config.MAGNITUDE_SCALE,
            'MAX_WINDOWS_PER_STOCK': config.MAX_WINDOWS_PER_STOCK,
            'EARLY_STOP_PATIENCE': config.EARLY_STOP_PATIENCE
        }
    }
    
    summary_path = Path(config.LOG_DIR) / '48h_summary.json'
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n✓ Summary saved to: {summary_path}")
    sys.exit(0)
    
except KeyboardInterrupt:
    print("\n\n" + "="*80)
    print("⚠️  TRAINING INTERRUPTED BY USER")
    print("="*80)
    sys.exit(1)
    
except Exception as e:
    print("\n\n" + "="*80)
    print("❌ TRAINING FAILED")
    print("="*80)
    print(f"\nError: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
