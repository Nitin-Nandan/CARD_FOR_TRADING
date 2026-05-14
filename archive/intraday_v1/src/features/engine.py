"""
Feature Engine: 81-Channel Feature Engineering
"""

import numpy as np
import pandas as pd
from datetime import time
from .momentum import add_momentum_features
from .volatility import add_volatility_features
from .volume import add_volume_features


class FeatureEngine:
    """
    Modular engine to compute 81 technical and market features.

    Channels:
    - OHLCV (5)
    - Momentum Indicators (15)
    - Volatility Indicators (10)
    - Volume Indicators (12)
    - Price-based Features (8)
    - Time-based Features (7)
    - Cross-Asset (Nifty) Features (5)
    - Pattern Recognition (3)
    - Phase 3 Regimes (3)
    - Additional EMA/Returns (13)
    """

    def __init__(self, market_open=time(9, 15), market_close=time(15, 30)):
        self.market_open = market_open
        self.market_close = market_close

        self.expected_features = [
            "timestamp",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "ema_5",
            "ema_10",
            "roi_1m",
            "bb_upper",
            "bb_middle",
            "bb_lower",
            "nifty_close",
            "nifty_return",
            "hour_sin",
            "hour_cos",
            "realized_vol_30m",
            "volume_ratio",
            "volume_surge",
            "rsi_7",
            "rsi_14",
            "rsi_21",
            "macd",
            "macd_signal",
            "macd_hist",
            "stoch_k",
            "stoch_d",
            "roc_5",
            "roc_10",
            "roc_20",
            "williams_r",
            "ema_12",
            "ema_26",
            "ema_20",
            "atr_7",
            "atr_14",
            "hv_10",
            "hv_20",
            "hv_30",
            "vol_ratio_short",
            "vol_expansion",
            "parkinson_vol",
            "gk_vol",
            "true_range",
            "volume_sma_10",
            "volume_sma_20",
            "volume_roc",
            "obv",
            "obv_ema",
            "mfi",
            "vpt",
            "force_index",
            "force_index_ema",
            "ad_line",
            "money_flow",
            "typical_price",
            "ema_50",
            "price_position",
            "hl_ratio",
            "oc_ratio",
            "dist_from_vwap",
            "gap_pct",
            "intraday_return",
            "vwap",
            "minute_sin",
            "minute_cos",
            "is_monday",
            "is_friday",
            "week_of_month",
            "days_to_month_end",
            "is_expiry_week",
            "relative_strength",
            "relative_strength_10",
            "beta_60",
            "beta_20",
            "corr_nifty_30",
            "near_20d_high",
            "near_20d_low",
            "breakout_up",
            "regime_volatility",
            "regime_trend",
            "regime_time",
        ]

    def process(self, df, nifty_df=None):
        """Orchestrate full feature computation"""
        df = df.copy()

        # 1. Clean and filter
        df = self.filter_trading_hours(df)
        df = self.filter_weekends(df)
        df = self.handle_missing_data(df)

        # 2. Base Features
        df = add_momentum_features(df)
        df = add_volatility_features(df)
        df = add_volume_features(df)
        df = self.add_price_features(df)
        df = self.add_time_features(df)
        df = self.add_pattern_features(df)

        # 3. Market Context
        if nifty_df is not None:
            df = self.add_cross_asset_features(df, nifty_df)
        else:
            # placeholders for compilation and schema safety
            df["nifty_close"] = 0.0
            df["nifty_return"] = 0.0
            df["relative_strength"] = 0.0
            df["relative_strength_10"] = 0.0
            df["beta_60"] = 1.0
            df["beta_20"] = 1.0
            df["corr_nifty_30"] = 1.0

        # 4. Phase 3 Regimes
        df = self.add_regime_features(df)

        # 5. Finalize
        df = df.ffill().bfill()
        df = df.dropna().reset_index(drop=True)

        # Safety check: ensure 81 features + timestamp
        for col in self.expected_features:
            if col not in df.columns:
                df[col] = 0.0

        return df[self.expected_features]

    # --- Feature Computation Methods (Internal) ---

    def filter_trading_hours(self, df):
        df["time"] = df["timestamp"].dt.time
        mask = (df["time"] >= self.market_open) & (df["time"] <= self.market_close)
        return df[mask].drop(columns=["time"]).copy()

    def filter_weekends(self, df):
        return df[df["timestamp"].dt.dayofweek < 5].copy()

    def handle_missing_data(self, df):
        return df.ffill(limit=5).dropna().copy()



    def add_price_features(self, df):
        df["ema_20"] = df["close"].ewm(span=20).mean()
        df["ema_50"] = df["close"].ewm(span=50).mean()
        df["ema_5"] = df["close"].ewm(span=5).mean()
        df["ema_10"] = df["close"].ewm(span=10).mean()
        df["roi_1m"] = df["close"].pct_change()

        mid = df["close"].rolling(20).mean()
        std = df["close"].rolling(20).std()
        df["bb_middle"], df["bb_upper"], df["bb_lower"] = (
            mid,
            mid + 2 * std,
            mid - 2 * std,
        )

        low_20_min = df["low"].rolling(20).min()
        df["price_position"] = (df["close"] - low_20_min) / (
            df["high"].rolling(20).max() - low_20_min + 1e-8
        )
        df["hl_ratio"] = (df["high"] - df["low"]) / (df["close"] + 1e-8)
        df["oc_ratio"] = (df["close"] - df["open"]) / (df["close"] + 1e-8)
        df["dist_from_vwap"] = (df["close"] - df["vwap"]) / (df["vwap"] + 1e-8)
        df["gap_pct"] = (df["open"] / df["close"].shift(1) - 1) * 100
        df["intraday_return"] = (df["close"] / (df["open"] + 1e-8) - 1) * 100
        df["realized_vol_30m"] = df["roi_1m"].rolling(30).std()
        return df

    def add_time_features(self, df):
        h, m = df["timestamp"].dt.hour, df["timestamp"].dt.minute
        dh = h + m / 60.0
        df["hour_sin"], df["hour_cos"] = (
            np.sin(2 * np.pi * dh / 24),
            np.cos(2 * np.pi * dh / 24),
        )

        mod = (h - 9) * 60 + m - 15
        df["minute_sin"], df["minute_cos"] = (
            np.sin(2 * np.pi * mod / 375),
            np.cos(2 * np.pi * mod / 375),
        )

        dow = df["timestamp"].dt.dayofweek
        df["is_monday"], df["is_friday"] = (
            (dow == 0).astype(int),
            (dow == 4).astype(int),
        )
        df["week_of_month"] = ((df["timestamp"].dt.day - 1) // 7).astype(int)
        df["days_to_month_end"] = (
            df["timestamp"].dt.days_in_month - df["timestamp"].dt.day
        )
        df["is_expiry_week"] = (df["days_to_month_end"] <= 5).astype(int)
        return df

    def add_cross_asset_features(self, df, nifty_df):
        nd = nifty_df[["timestamp", "close"]].rename(columns={"close": "nifty_close"})
        df = df.merge(nd, on="timestamp", how="left")
        df["nifty_close"] = df["nifty_close"].ffill()
        df["nifty_return"] = df["nifty_close"].pct_change().fillna(0)
        df["relative_strength"] = df["roi_1m"] - df["nifty_return"]
        df["relative_strength_10"] = df["close"].pct_change(10) - df[
            "nifty_close"
        ].pct_change(10)

        def beta(s_rets, m_rets, w):
            return s_rets.rolling(w).cov(m_rets) / (m_rets.rolling(w).var() + 1e-8)

        df["beta_60"], df["beta_20"] = (
            beta(df["roi_1m"], df["nifty_return"], 60),
            beta(df["roi_1m"], df["nifty_return"], 20),
        )
        df["corr_nifty_30"] = df["roi_1m"].rolling(30).corr(df["nifty_return"])
        return df

    def add_pattern_features(self, df):
        h20, l20 = df["high"].rolling(7500).max(), df["low"].rolling(7500).min()
        df["near_20d_high"] = (df["close"] / (h20 + 1e-8) > 0.98).astype(int)
        df["near_20d_low"] = (df["close"] / (l20 + 1e-8) < 1.02).astype(int)
        df["breakout_up"] = (df["close"] > h20.shift(1)).astype(int)
        return df

    def add_regime_features(self, df):
        """Volatility, Trend, and Time regimes."""
        mvol = df["realized_vol_30m"].rolling(375 * 5).median()
        df["regime_volatility"] = df["realized_vol_30m"] / (mvol + 1e-8)
        s50, s200 = df["close"].rolling(50).mean(), df["close"].rolling(200).mean()
        df["regime_trend"] = (s50 / (s200 + 1e-8)) - 1
        mod = (df["timestamp"].dt.hour - 9) * 60 + df["timestamp"].dt.minute - 15
        df["regime_time"] = ((mod <= 60) | (mod >= 315)).astype(float)
        return df
