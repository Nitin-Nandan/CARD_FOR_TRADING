"""
Dataset Builder: Index-Based Windowing
Implements strict gap detection for financial time series.
"""

import numpy as np
from pathlib import Path
from numpy.lib.stride_tricks import sliding_window_view


class WindowBuilder:
    """
    Modular builder to create sliding windows for CARD training.
    Strictly handles overnight/weekend/holiday gap detection.
    """

    def __init__(self, input_len=60, pred_len=15, stride=1, gap_threshold=2):
        self.input_len = input_len
        self.pred_len = pred_len
        self.stride = stride
        self.gap_threshold = gap_threshold
        self.total_len = input_len + pred_len

    def create_windows(self, df, stock_name):
        """Extract valid window starting indices from a processed stock DataFrame."""
        feature_cols = [col for col in df.columns if col != "timestamp"]
        if "close" not in feature_cols:
            raise KeyError(f"'close' column not found for {stock_name}")

        close_idx = feature_cols.index("close")
        data = df[feature_cols].values.astype(np.float32)
        timestamps = df["timestamp"].values

        if len(data) < self.total_len:
            return None

        # 1. Gap detection
        time_diffs = np.diff(timestamps).astype("timedelta64[m]").astype(int)
        invalid_gaps = time_diffs > self.gap_threshold
        window_gaps = sliding_window_view(invalid_gaps, self.total_len - 1)
        valid_starts = np.where(window_gaps.sum(axis=1) == 0)[0]

        # 2. NaN detection
        has_nan = np.isnan(data).any(axis=1)
        window_nans = sliding_window_view(has_nan, self.total_len)
        valid_nan_starts = np.where(window_nans.sum(axis=1) == 0)[0]

        base_indices = np.intersect1d(valid_starts, valid_nan_starts)

        # 3. Compute returns and filter extremes
        y_returns = []
        last_prices = []
        final_indices = []

        for idx in base_indices:
            last = data[idx + self.input_len - 1, close_idx]
            future = data[idx + self.input_len : idx + self.total_len, close_idx]

            if last == 0:
                continue

            rets = (future / last) - 1
            if np.abs(rets).max() > 0.10:
                continue  # Sanity filter

            y_returns.append(rets)
            last_prices.append(last)
            final_indices.append(idx)

        return {
            "data": data,
            "timestamps": timestamps,
            "valid_indices": np.array(final_indices, dtype=np.int32),
            "y_returns": np.array(y_returns, dtype=np.float32),
            "last_prices": np.array(last_prices, dtype=np.float32),
            "feature_names": feature_cols,
            "close_idx": close_idx,
            "stock_name": stock_name,
        }

    def save_npz(self, windows_data, output_dir):
        """Save to compressed .npz format"""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        stock_name = windows_data["stock_name"]
        path = output_dir / f"{stock_name}_windows.npz"

        np.savez_compressed(
            path,
            data=windows_data["data"],
            timestamps=windows_data["timestamps"],
            valid_indices=windows_data["valid_indices"],
            y_returns=windows_data["y_returns"],
            last_prices=windows_data["last_prices"],
            feature_names=np.array(windows_data["feature_names"]),
            close_idx=np.array(windows_data["close_idx"]),
            stock_name=np.array(stock_name),
        )
        return path
