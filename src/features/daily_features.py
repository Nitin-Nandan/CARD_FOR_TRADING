"""
Daily Feature Engineering for NSE stocks.

Computes 15 stationary daily features from OHLCV data (after log-return
transformation). All features are either in log-return space or bounded ratios
— no raw price levels are used. All rolling windows end at bar t (no lookahead).

Feature index map (enc_in = 15):
    0  log_return_close
    1  vol_log_return
    2  atr_pct_14
    3  rsi_14
    4  macd_signal
    5  bb_width_20
    6  close_ema20_ratio
    7  close_ema50_ratio
    8  high_low_range
    9  overnight_gap
    10 volume_zscore_20
    11 dow_sin
    12 dow_cos
    13 month_sin
    14 month_cos
"""

import numpy as np
import pandas as pd

# Channel index for the prediction target (used by CARD's close_channel_idx)
CLOSE_CHANNEL_IDX = 0
FEATURE_NAMES = [
    "log_return_close",
    "vol_log_return",
    "atr_pct_14",
    "rsi_14",
    "macd_signal",
    "bb_width_20",
    "close_ema20_ratio",
    "close_ema50_ratio",
    "high_low_range",
    "overnight_gap",
    "volume_zscore_20",
    "dow_sin",
    "dow_cos",
    "month_sin",
    "month_cos",
]
N_FEATURES = len(FEATURE_NAMES)  # 15


def _rsi(series: pd.Series, window: int = 14) -> pd.Series:
    """RSI computed on log returns (bounded [0, 100])."""
    delta = series.diff()
    gain = delta.clip(lower=0).rolling(window).mean()
    loss = (-delta.clip(upper=0)).rolling(window).mean()
    rs = gain / (loss + 1e-8)
    return 100.0 - 100.0 / (1.0 + rs)


def _ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


def compute_daily_features(
    df: pd.DataFrame,
    nifty_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    Compute 15 daily features from a OHLCV DataFrame that already contains
    log_return_close and vol_log_return columns (output of LogReturnTransform).

    Args:
        df        : DataFrame with columns [open, high, low, close, volume,
                    log_return_close, vol_log_return, ...]. Index must be DatetimeIndex.
        nifty_df  : Optional Nifty50 DataFrame with same DatetimeIndex. Currently
                    reserved for future cross-asset features; not used in Phase 1.

    Returns:
        DataFrame with exactly 15 feature columns in FEATURE_NAMES order.
        First ~50 rows will contain NaN from rolling windows — caller must drop.
    """
    c = df["close"].astype(float)
    h = df["high"].astype(float)
    lo = df["low"].astype(float)
    o = df["open"].astype(float)
    vol = df["volume"].astype(float).replace(0, np.nan)
    ret = df["log_return_close"].astype(float)

    feat = pd.DataFrame(index=df.index)

    # 0. Log return of close
    feat["log_return_close"] = ret

    # 1. Volume log return
    feat["vol_log_return"] = df["vol_log_return"].astype(float)

    # 2. ATR(14) / close — normalized daily volatility
    tr = pd.concat(
        [h - lo, (h - c.shift(1)).abs(), (lo - c.shift(1)).abs()], axis=1
    ).max(axis=1)
    atr14 = tr.rolling(14).mean()
    feat["atr_pct_14"] = atr14 / (c + 1e-8)

    # 3. RSI(14) on log returns — bounded [0, 100] → rescaled to [-1, 1]
    feat["rsi_14"] = (_rsi(ret, 14) / 50.0) - 1.0

    # 4. MACD signal line histogram (12/26/9 EMAs on log returns)
    ema12 = _ema(ret, 12)
    ema26 = _ema(ret, 26)
    macd_line = ema12 - ema26
    feat["macd_signal"] = macd_line - _ema(macd_line, 9)  # histogram

    # 5. Bollinger Band width / close — volatility proxy
    sma20 = c.rolling(20).mean()
    std20 = c.rolling(20).std()
    feat["bb_width_20"] = (4 * std20) / (sma20 + 1e-8)

    # 6. close / EMA(20) - 1 — mean reversion signal
    feat["close_ema20_ratio"] = (c / (_ema(c, 20) + 1e-8)) - 1.0

    # 7. close / EMA(50) - 1 — trend signal
    feat["close_ema50_ratio"] = (c / (_ema(c, 50) + 1e-8)) - 1.0

    # 8. (high - low) / close — intraday range
    feat["high_low_range"] = (h - lo) / (c + 1e-8)

    # 9. overnight gap = open / prev_close - 1
    feat["overnight_gap"] = (o / (c.shift(1) + 1e-8)) - 1.0

    # 10. volume z-score over 20 days
    vol_mean = vol.rolling(20).mean()
    vol_std = vol.rolling(20).std()
    feat["volume_zscore_20"] = (vol - vol_mean) / (vol_std + 1e-8)

    # 11-12. Day-of-week cyclical encoding
    dow = df.index.dayofweek.astype(float)  # 0=Mon, 4=Fri
    feat["dow_sin"] = np.sin(2 * np.pi * dow / 5.0)
    feat["dow_cos"] = np.cos(2 * np.pi * dow / 5.0)

    # 13-14. Month cyclical encoding
    month = df.index.month.astype(float)
    feat["month_sin"] = np.sin(2 * np.pi * month / 12.0)
    feat["month_cos"] = np.cos(2 * np.pi * month / 12.0)

    return feat[FEATURE_NAMES]
