"""
Stock Windows Dataset

PHASE 0 REWRITE:
- Loads y_close (raw close prices) instead of y_returns / y_volatility
- __getitem__ returns (X, y_close, stock_idx)
- Temporal split (not random) to prevent leakage
"""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from typing import List, Dict, Tuple
import random
import os
import sys


class StockWindowsDataset(Dataset):
    """
    PyTorch Dataset for stock windows from .npz files
    
    Pre-loads all data into RAM at initialization for maximum speed.
    Trade-off: ~3.5GB RAM usage for 100-400× faster training.
    
    Args:
        windows_dir: Directory containing {STOCK}_windows.npz files
        stocks: List of stock names to include
        split: 'train', 'val', or 'test'
        train_ratio: Fraction for training (default: 0.7)
        val_ratio: Fraction for validation (default: 0.15)
        test_ratio: Fraction for testing (default: 0.15)
        seed: Random seed for reproducibility (default: 42)
    """
    
    def __init__(
        self,
        windows_dir: str,
        stocks: List[str],
        split: str = 'train',
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
        seed: int = 42,
        max_windows_per_stock: int = None,
    ):
        """
        Args:
            max_windows_per_stock: If set, randomly sample this many windows
                per stock instead of loading all of them.  Allows ALL stocks
                to be loaded every epoch within a fixed RAM budget.
                E.g. max_windows_per_stock=2500, 50 stocks → 125k windows ≈ 540 MB.
        """
        assert split in ['train', 'val', 'test'], "split must be 'train', 'val', or 'test'"
        assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "Ratios must sum to 1"

        self.windows_dir = Path(windows_dir).resolve()
        self.stocks = stocks
        self.split = split
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.seed = seed
        self.max_windows_per_stock = max_windows_per_stock

        self.data = {}
        self.index = []
        self.stock_metadata = {}

        n_desc = f" (max {max_windows_per_stock}/stock)" if max_windows_per_stock else ""
        print(f"Loading {split} split from {len(stocks)} stocks{n_desc}...")
        print(f"Windows directory: {self.windows_dir}")

        if not self.windows_dir.exists():
            print(f"  ⚠️  WARNING: Directory does not exist: {self.windows_dir}")
            return

        rng = np.random.default_rng(seed)

        for stock_idx, stock in enumerate(stocks):
            stock_file = self.windows_dir / f"{stock}_windows.npz"

            if not stock_file.exists():
                print(f"  Warning: {stock_file} not found, skipping {stock}")
                continue

            with np.load(stock_file, mmap_mode='r') as d:
                total_windows = len(d['X'])

                train_end = int(total_windows * train_ratio)
                val_end   = train_end + int(total_windows * val_ratio)

                if split == 'train':
                    start_idx, end_idx = 0, train_end
                elif split == 'val':
                    start_idx, end_idx = train_end, val_end
                else:
                    start_idx, end_idx = val_end, total_windows

                slice_len = end_idx - start_idx

                if max_windows_per_stock and slice_len > max_windows_per_stock:
                    # Sample randomly within the split (temporal order preserved per stock)
                    chosen = np.sort(rng.choice(slice_len, size=max_windows_per_stock, replace=False))
                    X_data  = d['X'][start_idx:end_idx][chosen].copy()
                    y_data  = d['y_close'][start_idx:end_idx][chosen].copy()
                    n_loaded = max_windows_per_stock
                else:
                    X_data  = d['X'][start_idx:end_idx].copy()
                    y_data  = d['y_close'][start_idx:end_idx].copy()
                    n_loaded = slice_len

            self.data[stock_idx] = {'X': X_data, 'y_close': y_data}
            self.stock_metadata[stock_idx] = {'name': stock, 'num_windows': n_loaded}

            for local_idx in range(n_loaded):
                self.index.append((stock_idx, local_idx))

            print(f"  {stock}: {n_loaded:,} windows loaded")

        total = len(self.index)
        ram_mb = sum(v['X'].nbytes + v['y_close'].nbytes for v in self.data.values()) / 1e6
        print(f"\n✅ {split}: {total:,} samples  |  RAM ≈ {ram_mb:.0f} MB")

    def __len__(self):
        return len(self.index)
    
    def __getitem__(self, idx):
        """
        Returns:
            X:        (60, 18) input window  — normalized features
            y_close:  (15,)   raw close prices for next 15 minutes
            stock_idx: int    stock index (for tracking)
        """
        stock_idx, local_window_idx = self.index[idx]

        X       = self.data[stock_idx]['X'][local_window_idx]
        y_close = self.data[stock_idx]['y_close'][local_window_idx]

        X       = torch.from_numpy(X.astype(np.float32))
        y_close = torch.from_numpy(y_close.astype(np.float32))

        return X, y_close, stock_idx
    
    def get_stock_name(self, stock_idx):
        """Get stock name from index"""
        return self.stock_metadata[stock_idx]['name']


class MultiStockDataLoader:
    """
    Wrapper for DataLoader with stock-aware batching
    
    Ensures batches have good stock diversity for better generalization.
    
    WINDOWS FIX: Set num_workers=0 for Windows compatibility
    """
    
    def __init__(
        self,
        dataset: StockWindowsDataset,
        batch_size: int = 64,
        shuffle: bool = True,
        num_workers: int = 0,  # CHANGED: Set to 0 for Windows
        pin_memory: bool = True
    ):
        self.dataset = dataset
        self.batch_size = batch_size
        
        self.loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,  # Use 0 for Windows to avoid multiprocessing issues
            pin_memory=pin_memory if torch.cuda.is_available() else False,
            drop_last=True  # Drop incomplete batches for consistent shapes
        )
    
    def __iter__(self):
        return iter(self.loader)
    
    def __len__(self):
        return len(self.loader)


# ============================================================================
# TESTING
# ============================================================================

if __name__ == "__main__":
    print("="*60)
    print("Testing StockWindowsDataset")
    print("="*60)
    
    # FIXED: Use proper path resolution
    # Get the project root directory
    current_file = Path(__file__).resolve()
    
    # If running from train/ directory
    if current_file.parent.name == 'train':
        project_root = current_file.parent.parent
    else:
        project_root = current_file.parent
    
    windows_dir = project_root / 'data' / 'windows'
    
    print(f"Current file: {current_file}")
    print(f"Project root: {project_root}")
    print(f"Windows directory: {windows_dir}")
    print(f"Windows directory exists: {windows_dir.exists()}")
    
    # Test with a few stocks
    test_stocks = ['RELIANCE', 'TCS', 'INFY']
    
    # List available window files
    if windows_dir.exists():
        print(f"\nAvailable window files:")
        window_files = sorted(windows_dir.glob("*_windows.npz"))
        if window_files:
            for f in window_files[:5]:  # Show first 5
                print(f"  ✓ {f.name}")
            if len(window_files) > 5:
                print(f"  ... and {len(window_files) - 5} more")
        else:
            print("  ⚠️  No window files found!")
            print("  Please run: python pipeline/03_create_windows_60min.py")
            sys.exit(1)
    else:
        print(f"\n⚠️  ERROR: Windows directory not found!")
        print(f"Expected: {windows_dir}")
        print("Please run: python pipeline/03_create_windows_60min.py")
        sys.exit(1)
    
    print("\n" + "="*60)
    print("Creating datasets...")
    print("="*60)
    
    # Create datasets
    train_dataset = StockWindowsDataset(
        windows_dir=str(windows_dir),
        stocks=test_stocks,
        split='train'
    )
    
    val_dataset = StockWindowsDataset(
        windows_dir=str(windows_dir),
        stocks=test_stocks,
        split='val'
    )
    
    print(f"\nTrain samples: {len(train_dataset):,}")
    print(f"Val samples: {len(val_dataset):,}")
    
    if len(train_dataset) == 0:
        print("\n⚠️  ERROR: No training samples found!")
        sys.exit(1)

    X, y_close, stock_idx = train_dataset[0]
    print(f"\nSample shapes:")
    print(f"  X      : {X.shape}")
    print(f"  y_close: {y_close.shape}  (raw close prices)")
    print(f"  Stock  : {train_dataset.get_stock_name(stock_idx)}")

    train_loader = MultiStockDataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=0)
    print(f"\nDataLoader batches: {len(train_loader)}")

    for X_b, y_close_b, stock_ids in train_loader:
        print(f"\nBatch:  X={X_b.shape}  y_close={y_close_b.shape}")
        break

    print("\n" + "="*60)
    print("✅ Dataset test complete!")
    print("="*60)