"""
Stock Dataset for CARD Model
Implements memory-mapping (mmap) for SSD-safe TB-scale dataset loading.
"""

import numpy as np
import torch
from torch.utils.data import Dataset
import os
from pathlib import Path
from .samplers import create_sample_indices
from .normalization import normalize_features


class StockWindowsDataset(Dataset):
    """
    Dataset for CARD training with per-stock balancing.

    CRITICAL WINDOWS FIX:
    - Uses memory mapping (mmap_mode='r') but safely keeps file handles open in memory
    - Avoids Windows BrokenPipeError by mandating NUM_WORKERS=0
    - Prevents 94GB RAM consumption by mapping SSD to memory pointers instead of full loading

    Features:
    - Temporal train/val/test split (70/15/15)
    - Per-stock balancing (equal representation)
    - On-the-fly window slicing from SSD via references
    """

    def __init__(
        self,
        stocks,
        split="train",
        balance_stocks=True,
        max_windows_per_stock=100000,
        seq_len=60,
        pred_len=15,
        verbose=True,
    ):
        """
        Args:
            stocks: List of stock symbols
            split: 'train', 'val', or 'test'
            balance_stocks: If True, equal probability per stock
            max_windows_per_stock: Cap large stocks to prevent dominance
            seq_len: Input sequence length (default 60)
            pred_len: Prediction horizon (default 15)
        """

        self.stocks = stocks
        self.split = split
        self.balance_stocks = balance_stocks
        self.max_windows_per_stock = max_windows_per_stock
        self.seq_len = seq_len
        self.pred_len = pred_len

        self.verbose = verbose

        if self.verbose:
            print(
                f"  [Dataset] {split:5} | Stocks: {len(stocks)} | Balancing: {balance_stocks}"
            )

        # Load window data for each stock INTO RAM
        self.stock_data = {}
        self.stock_window_counts = {}

        # We need to keep a reference to the opened NpzFile objects so they don't close
        self.open_files = []

        for stock in stocks:
            npz_path = Path(f"data/windows/{stock}_windows.npz")

            if not npz_path.exists():
                if self.verbose:
                    print(f"  ⚠️  Skipping {stock} (no window file)")
                continue

            # CRITICAL: Keep file open with mmap on Windows
            # Safe because NUM_WORKERS = 0 guarantees no thread contention
            try:
                data = np.load(str(npz_path), mmap_mode="r")
                self.open_files.append(
                    data
                )  # Prevent garbage collection of the file handle
            except Exception as e:
                if self.verbose:
                    print(f"  ⚠️  Failed to mmap {stock}: {e}")
                continue

            if "valid_indices" not in data.files:
                print(f"  ⚠️  Skipping {stock} (invalid format)")
                continue

            # CRITICAL WINDOWS/8GB RAM FIX:
            # Use uncompressed .npy for true memory mapping if available
            npy_path = npz_path.parent / f"{stock}_data.npy"

            if npy_path.exists():
                # True mmap on uncompressed .npy (Zero RAM)
                full_data_ref = np.load(str(npy_path), mmap_mode="r")
            else:
                # Fallback to .npz (Warning: This will load the whole array into RAM!)
                if self.verbose and not getattr(self, "_mmap_warning_printed", False):
                    print(
                        f"\n  ⚠️  WARNING: {stock}_data.npy not found. Using .npz fallback."
                    )
                    print("      This will likely cause OOM on 8GB RAM.")
                    print("      Run 'python scripts/fix_memory.py' first!")
                    self._mmap_warning_printed = True
                full_data_ref = data["data"]

            # Metadata references (Small, stays in SSD mmap handle)
            valid_indices_ref = data["valid_indices"]
            y_returns_ref = data["y_returns"]
            last_prices_ref = data["last_prices"]

            # Feature Normalization
            # some features (OBV) are in billions. normalize using sample.
            sample_size = min(10000, len(full_data_ref))
            sample = np.array(full_data_ref[:sample_size])
            sample = np.nan_to_num(sample, nan=0.0, posinf=1e6, neginf=-1e6)

            means = np.mean(sample, axis=0)
            stds = np.std(sample, axis=0) + 1e-4
            stds[stds < 1e-6] = 1.0

            # Total windows for this stock
            total_windows = len(valid_indices_ref)

            # Temporal split (70% train, 15% val, 15% test)
            if split == "train":
                start_idx = 0
                end_idx = int(total_windows * 0.70)
            elif split == "val":
                start_idx = int(total_windows * 0.70)
                end_idx = int(total_windows * 0.85)
            else:  # test
                start_idx = int(total_windows * 0.85)
                end_idx = total_windows
            # Get indices for this split
            actual_count = end_idx - start_idx

            if actual_count == 0:
                if self.verbose:
                    print(f"  ⚠️  Skipping {stock} (no data in {split} split)")
                continue

            # Apply cap if balancing
            use_cap = balance_stocks and actual_count > max_windows_per_stock
            final_count = max_windows_per_stock if use_cap else actual_count

            # FINAL METADATA ACCESS: Slice ONLY the needed indices into RAM
            # If use_cap=False, we store a range object or 0..N indices to save RAM
            if use_cap:
                # Use np.arange to avoid Python list-of-ints
                full_split_indices = np.arange(start_idx, end_idx, dtype=np.uint32)
                sampled_indices = np.random.choice(
                    full_split_indices, size=final_count, replace=False
                )
                valid_indices = valid_indices_ref[sampled_indices]
                y_returns = y_returns_ref[sampled_indices]
                last_prices = last_prices_ref[sampled_indices]
                # local indices for the sampled metadata are just 0..final_count-1
                indices_to_sample = np.arange(final_count, dtype=np.uint32)
            else:
                # Range slice (0 RAM)
                valid_indices = valid_indices_ref[start_idx:end_idx]
                y_returns = y_returns_ref[start_idx:end_idx]
                last_prices = last_prices_ref[start_idx:end_idx]
                indices_to_sample = np.arange(final_count, dtype=np.uint32)

            # Store references
            self.stock_data[stock] = {
                "data": full_data_ref,
                "valid_indices": valid_indices,
                "y_returns": y_returns,
                "last_prices": last_prices,
                "means": means,
                "stds": stds,
                "indices": indices_to_sample,
                "count": final_count,
            }

            self.stock_window_counts[stock] = final_count

            if self.verbose:
                if use_cap:
                    # print(f"  {stock}: {actual_count:,} -> {final_count:,} (capped)")
                    pass
                else:
                    # print(f"  {stock}: {final_count:,}")
                    pass

        if len(self.stock_data) == 0:
            raise ValueError(f"No valid stocks found for {split} split!")

        # Build stock list for ID mapping
        self.stock_list = sorted(self.stock_data.keys())
        self.stock_to_id = {s: i for i, s in enumerate(self.stock_list)}

        # Build sample indices using efficient numpy arrays
        self.indices_stock_ids, self.indices_window_idx = create_sample_indices(
            self.balance_stocks, self.stock_list, self.stock_to_id, self.stock_data, self.stock_window_counts
        )

        if self.verbose:
            print(
                f"  [Dataset] {split:5} | Total Samples: {len(self.indices_stock_ids):,}"
            )



    def on_epoch_end(self):
        """Resample balanced indices at end of each epoch"""
        if self.balance_stocks:
            self.indices_stock_ids, self.indices_window_idx = create_sample_indices(
                self.balance_stocks, self.stock_list, self.stock_to_id, self.stock_data, self.stock_window_counts
            )

    def validate_no_leakage(self):
        """
        Audit method to ensure no data leakage across time splits.
        Returns True if passed, raises AssertionError if leakage detected.
        """
        # TODO: Implement strict validation of split boundaries, warm-up gaps,
        # and normalization isolation as requested by Data Agent rules.
        pass

    def __len__(self):
        return len(self.indices_stock_ids)



    def __getitem__(self, idx):
        """
        Get a training sample
        """
        stock_id = self.indices_stock_ids[idx]
        local_idx = self.indices_window_idx[idx]
        stock = self.stock_list[stock_id]

        # Get from RAM cache metadata
        stock_cache = self.stock_data[stock]

        # Get window start index
        start_idx = stock_cache["valid_indices"][local_idx]

        # Extract features (60 timesteps × 81 features)
        # This streams the tiny slice directly from the SSD via the pointer!
        X = np.array(
            stock_cache["data"][start_idx : start_idx + self.seq_len]
        )  # (60, 81)

        # Check 4: Is Z-score normalization too aggressive? (Diagnostics)
        if os.getenv("DIAGNOSTIC_MODE") == "1" and not hasattr(self, "_diag_printed"):
            print("\n=== NORMALIZATION CHECK ===")
            print(f"Features before Z-score - mean: {X.mean():.2f}, std: {X.std():.2f}")
            X_norm_temp = (X - stock_cache["means"]) / stock_cache["stds"]
            print(
                f"Features after Z-score  - mean: {X_norm_temp.mean():.6f}, std: {X_norm_temp.std():.2f}"
            )
            print(
                f"Features after Z-score  - min: {X_norm_temp.min():.2f}, max: {X_norm_temp.max():.2f}"
            )
            saturated_high = np.sum(X_norm_temp > 10.0)
            saturated_low = np.sum(X_norm_temp < -10.0)
            print(
                f"Saturated (+10): {saturated_high} / {X_norm_temp.size} ({100 * saturated_high / X_norm_temp.size:.1f}%)"
            )
            print(
                f"Saturated (-10): {saturated_low} / {X_norm_temp.size} ({100 * saturated_low / X_norm_temp.size:.1f}%)"
            )
            self._diag_printed = True

        # Apply robust normalization (Wide clipping + NaN handling)
        X = normalize_features(X)

        # Get returns (already computed in Phase 2)
        y_returns = stock_cache["y_returns"][local_idx]  # (15,)
        last_price = stock_cache["last_prices"][local_idx]  # scalar

        # Reconstruct prices (for metrics)
        y_close = last_price * (1 + y_returns)  # (15,)

        return {
            "X": torch.tensor(X.T, dtype=torch.float32),  # (81, 60) - transposed
            "y": torch.tensor(y_returns, dtype=torch.float32),  # (15,) - PRIMARY TARGET
            "y_close": torch.tensor(
                y_close, dtype=torch.float32
            ),  # (15,) - for metrics
            "last_price": torch.tensor([last_price], dtype=torch.float32),  # (1,)
            "stock": stock,  # stock symbol (string)
        }
