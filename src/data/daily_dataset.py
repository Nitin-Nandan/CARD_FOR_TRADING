"""
PyTorch Dataset for daily NSE CARD training.

Sliding-window dataset over daily feature DataFrames.

    X : (enc_in, seq_len)    e.g. (15, 120)  — 15 features × 6-month lookback
    y : (pred_len,)          e.g. (5,)        — 5-day log return of close

Temporal integrity rules:
  1. Splits are made by row-index position, never by random shuffle.
  2. The RollingZScoreScaler is fit only on training rows — then applied to
     val/test without re-fitting. This is enforced inside build_splits().

Usage:
    from src.data.daily_dataset import build_splits
    train_ds, val_ds, test_ds = build_splits(df, DailyConfig)
"""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import logging
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Scaler (fit-on-train only)
# ---------------------------------------------------------------------------

class RollingZScoreScaler:
    """
    Standardizes each feature column using mean/std computed on training rows.
    Applied identically to val/test rows (no re-fitting).

    Cyclical features (dow_sin, dow_cos, month_sin, month_cos) are
    already bounded [-1, 1] — they are skipped.
    """

    SKIP_COLS = {"dow_sin", "dow_cos", "month_sin", "month_cos"}

    def __init__(self):
        self.mean_: pd.Series | None = None
        self.std_: pd.Series | None = None

    def fit(self, df: pd.DataFrame) -> "RollingZScoreScaler":
        """Fit on training slice only."""
        cols = [c for c in df.columns if c not in self.SKIP_COLS]
        self.mean_ = df[cols].mean()
        self.std_ = df[cols].std().clip(lower=1e-8)
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Standardize all non-cyclical columns. Returns a copy."""
        if self.mean_ is None:
            raise RuntimeError("Scaler not fitted. Call fit() on the training slice first.")
        out = df.copy()
        for col in self.mean_.index:
            if col in out.columns:
                out[col] = (out[col] - self.mean_[col]) / self.std_[col]
        # Clip to prevent extreme outliers from saturating the model
        out = out.clip(lower=-10.0, upper=10.0)
        return out


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class DailyReturnDataset(Dataset):
    """
    Sliding-window dataset over daily features.

    Args:
        features : numpy array of shape (T, enc_in) — already scaled
        targets  : numpy array of shape (T,) — log_return_close (unscaled)
        seq_len  : lookback window in trading days
        pred_len : forecast horizon in trading days
    """

    def __init__(
        self,
        features: np.ndarray,
        targets: np.ndarray,
        seq_len: int,
        pred_len: int,
    ):
        self.features = features.astype(np.float32)
        self.targets = targets.astype(np.float32)
        self.seq_len = seq_len
        self.pred_len = pred_len
        self._len = max(0, len(features) - seq_len - pred_len + 1)

    def __len__(self) -> int:
        return self._len

    def __getitem__(self, idx: int) -> dict:
        x_start = idx
        x_end = idx + self.seq_len
        y_start = x_end
        y_end = y_start + self.pred_len

        # X: (enc_in, seq_len) — transpose from (seq_len, enc_in)
        X = torch.from_numpy(self.features[x_start:x_end].T)
        # y: (pred_len,) — raw log returns (not scaled)
        y = torch.from_numpy(self.targets[y_start:y_end])

        return {"X": X, "y": y}


# ---------------------------------------------------------------------------
# Split builder (the only public entry point for training code)
# ---------------------------------------------------------------------------

def build_splits(
    df: pd.DataFrame,
    seq_len: int,
    pred_len: int,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
) -> tuple["DailyReturnDataset", "DailyReturnDataset", "DailyReturnDataset", "RollingZScoreScaler"]:
    """
    Split a processed feature DataFrame into train/val/test datasets.

    LEAKAGE GUARANTEE: The RollingZScoreScaler is fit exclusively on
    training rows (rows 0..train_end). Val and test rows are transformed
    with the same fitted scaler — never re-fitted.

    Args:
        df         : DataFrame with FEATURE_NAMES columns + DatetimeIndex.
                     Must already contain log_return_close as first column.
        seq_len    : lookback window (bars)
        pred_len   : forecast horizon (bars)
        train_ratio: fraction of rows used for training
        val_ratio  : fraction of rows used for validation

    Returns:
        (train_ds, val_ds, test_ds, scaler)
    """
    n = len(df)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))

    # --- Scaler: fit on train ONLY ---
    scaler = RollingZScoreScaler()
    scaler.fit(df.iloc[:train_end])

    # --- Scale all splits ---
    df_scaled = scaler.transform(df)

    features = df_scaled.values          # (T, enc_in)
    targets = df["log_return_close"].values  # (T,) — unscaled close returns

    # Log split sizes
    logger.info(
        f"Dataset split: train={train_end} | val={val_end - train_end} "
        f"| test={n - val_end} rows"
    )

    train_ds = DailyReturnDataset(features[:train_end], targets[:train_end], seq_len, pred_len)
    val_ds = DailyReturnDataset(features[train_end:val_end], targets[train_end:val_end], seq_len, pred_len)
    test_ds = DailyReturnDataset(features[val_end:], targets[val_end:], seq_len, pred_len)

    logger.info(
        f"Windows: train={len(train_ds)} | val={len(val_ds)} | test={len(test_ds)}"
    )
    return train_ds, val_ds, test_ds, scaler
