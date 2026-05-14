"""
Stationary data transforms for daily NSE OHLCV.

Primary: LogReturnTransform
  - Converts price columns to log returns: log(P_t / P_{t-1})
  - Stateless by design (no fit needed for log differencing)
  - Follows sklearn convention for future swap-in (e.g. fractional diff)

Secondary: check_stationarity
  - ADF test on a pandas Series
  - Logs a warning if p-value > 0.05
"""

import numpy as np
import pandas as pd
import logging
logger = logging.getLogger(__name__)

PRICE_COLS = ["open", "high", "low", "close"]
VOLUME_COL = "volume"


class LogReturnTransform:
    """
    Converts OHLCV price columns to log returns.

    Usage:
        t = LogReturnTransform()
        df_out = t.transform(df_raw)          # adds log_return_* columns
        prices = t.inverse_transform(rets, last_close)
    """

    def fit(self, df: pd.DataFrame) -> "LogReturnTransform":
        """No-op — log returns require no fitting. Follows sklearn convention."""
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Return a new DataFrame with log-return columns appended.

        Added columns:
            log_return_open/high/low/close  : log(P_t / P_{t-1})
            vol_log_return                  : log(volume_t / volume_{t-1})

        First row will contain NaN — caller must drop it via .dropna().
        Original OHLCV columns are preserved.
        """
        out = df.copy()
        for col in PRICE_COLS:
            if col in df.columns:
                out[f"log_return_{col}"] = np.log(
                    df[col].astype(float) / df[col].astype(float).shift(1)
                )
        if VOLUME_COL in df.columns:
            vol = df[VOLUME_COL].astype(float).replace(0, np.nan)
            out["vol_log_return"] = np.log(vol / vol.shift(1).replace(0, np.nan))
        return out

    def inverse_transform(
        self, log_returns: np.ndarray, last_price: float
    ) -> np.ndarray:
        """
        Reconstruct price path from log returns.

        Args:
            log_returns : 1-D array of shape (pred_len,)
            last_price  : closing price at t=0 (the day before the first return)

        Returns:
            np.ndarray of shape (pred_len,) — reconstructed prices
        """
        cumulative = np.cumsum(log_returns)
        return float(last_price) * np.exp(cumulative)


def check_stationarity(series: pd.Series, significance: float = 0.05) -> dict:
    """
    Augmented Dickey-Fuller test for stationarity.

    Args:
        series       : time series to test (NaNs are dropped)
        significance : p-value threshold (default 0.05)

    Returns:
        dict with keys: p_value, stationary, adf_stat
    """
    try:
        from statsmodels.tsa.stattools import adfuller
    except ImportError:
        logger.warning("statsmodels not installed — skipping ADF test.")
        return {"p_value": None, "stationary": None, "note": "statsmodels missing"}

    result = adfuller(series.dropna(), autolag="AIC")
    p_val = float(result[1])
    is_stationary = p_val < significance

    if is_stationary:
        logger.info(f"ADF: stationary (p={p_val:.4f})")
    else:
        logger.warning(
            f"ADF: possibly NON-stationary (p={p_val:.4f}). "
            "Consider fractional differencing or wavelet denoising."
        )

    return {"p_value": p_val, "stationary": is_stationary, "adf_stat": float(result[0])}
