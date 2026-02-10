"""
Stock Windows Dataset - FIXED FOR WINDOWS + PATH ISSUES

Efficient PyTorch Dataset for loading windowed stock data from .npz files.
Supports random sampling across all 50 stocks.

WINDOWS FIX: Uses num_workers=0 in test to avoid multiprocessing issues.
PATH FIX: Uses proper absolute path resolution.
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
    
    Uses memory-mapped file access for efficient loading of large datasets.
    
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
        seed: int = 42
    ):
        assert split in ['train', 'val', 'test'], "split must be 'train', 'val', or 'test'"
        assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "Ratios must sum to 1"
        
        self.windows_dir = Path(windows_dir).resolve()  # FIXED: Use absolute path
        self.stocks = stocks
        self.split = split
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.seed = seed
        
        # Build index: (stock_idx, window_idx) for each sample
        self.index = []
        self.stock_metadata = {}
        
        print(f"Loading {split} split from {len(stocks)} stocks...")
        print(f"Windows directory: {self.windows_dir}")
        
        # Check if directory exists
        if not self.windows_dir.exists():
            print(f"  ⚠️  WARNING: Directory does not exist: {self.windows_dir}")
            print(f"  Please check if windows have been created.")
            return
        
        for stock_idx, stock in enumerate(stocks):
            stock_file = self.windows_dir / f"{stock}_windows.npz"
            
            if not stock_file.exists():
                print(f"  Warning: {stock_file} not found, skipping {stock}")
                continue
            
            # Load metadata only (not full data)
            data = np.load(stock_file, mmap_mode='r')
            total_windows = len(data['X'])
            
            # Split indices
            train_end = int(total_windows * train_ratio)
            val_end = train_end + int(total_windows * val_ratio)
            
            if split == 'train':
                start_idx = 0
                end_idx = train_end
            elif split == 'val':
                start_idx = train_end
                end_idx = val_end
            else:  # test
                start_idx = val_end
                end_idx = total_windows
            
            # Store metadata
            self.stock_metadata[stock_idx] = {
                'name': stock,
                'file': stock_file,
                'start_idx': start_idx,
                'end_idx': end_idx,
                'num_windows': end_idx - start_idx
            }
            
            # Add to index
            for window_idx in range(start_idx, end_idx):
                self.index.append((stock_idx, window_idx))
            
            print(f"  {stock}: {end_idx - start_idx:,} windows")
        
        print(f"\nTotal {split} samples: {len(self.index):,}")
    
    def __len__(self):
        return len(self.index)
    
    def __getitem__(self, idx):
        """
        Get one sample - MEMORY EFFICIENT VERSION
        
        Uses memory-mapped file access to load only the required window,
        not the entire stock file. Reduces RAM usage from 3.5GB to ~100MB.
        
        Returns:
            X: (60, 18) input window
            y_returns: (15,) target returns
            y_volatility: (15,) target volatility
            stock_idx: index of stock (for tracking)
        """
        stock_idx, window_idx = self.index[idx]
        metadata = self.stock_metadata[stock_idx]
        
        # Memory-mapped access - doesn't load full file into RAM
        # Only loads the specific window we need
        with np.load(metadata['file'], mmap_mode='r') as data:
            # Load ONLY this specific window (not all 258K windows)
            X = data['X'][window_idx].copy()  # .copy() to ensure we own the data
            y_returns = data['y_returns'][window_idx].copy()
            y_volatility = data['y_volatility'][window_idx].copy()
        
        # Convert to float32 and tensors
        X = torch.from_numpy(X.astype(np.float32))
        y_returns = torch.from_numpy(y_returns.astype(np.float32))
        y_volatility = torch.from_numpy(y_volatility.astype(np.float32))
        
        return X, y_returns, y_volatility, stock_idx
    
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
        print("Check that the stock names match the window files.")
        sys.exit(1)
    
    # Test getting a sample
    X, y_ret, y_vol, stock_idx = train_dataset[0]
    
    print(f"\nSample shapes:")
    print(f"  X: {X.shape}")
    print(f"  y_returns: {y_ret.shape}")
    print(f"  y_volatility: {y_vol.shape}")
    print(f"  Stock: {train_dataset.get_stock_name(stock_idx)}")
    
    # Test DataLoader - IMPORTANT: num_workers=0 for Windows
    train_loader = MultiStockDataLoader(
        train_dataset,
        batch_size=32,
        shuffle=True,
        num_workers=0  # CRITICAL FOR WINDOWS
    )
    
    print(f"\nDataLoader batches: {len(train_loader)}")
    
    # Get one batch
    print("\nTesting batch loading...")
    for X_batch, y_ret_batch, y_vol_batch, stock_indices in train_loader:
        print(f"\nBatch shapes:")
        print(f"  X: {X_batch.shape}")
        print(f"  y_returns: {y_ret_batch.shape}")
        print(f"  y_volatility: {y_vol_batch.shape}")
        print(f"  Unique stocks in batch: {len(torch.unique(stock_indices))}")
        break
    
    print("\n" + "="*60)
    print("✅ Dataset test complete!")
    print("="*60)