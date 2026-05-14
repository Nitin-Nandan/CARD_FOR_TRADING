import pandas as pd
import numpy as np

def add_volume_features(df):
    df["volume_sma_10"] = df["volume"].rolling(10).mean()
    df["volume_sma_20"] = df["volume"].rolling(20).mean()
    df["volume_ratio"] = df["volume"] / (df["volume_sma_20"] + 1e-8)
    df["volume_roc"] = df["volume"].pct_change(10) * 100
    df["obv"] = (np.sign(df["close"].diff()) * df["volume"]).fillna(0).cumsum()
    df["obv_ema"] = df["obv"].ewm(span=20).mean()

    df["typical_price"] = (df["high"] + df["low"] + df["close"]) / 3
    df["money_flow"] = df["typical_price"] * df["volume"]

    df["vwap"] = (df["typical_price"] * df["volume"]).groupby(
        df["timestamp"].dt.date
    ).cumsum() / (df["volume"].groupby(df["timestamp"].dt.date).cumsum() + 1e-8)

    def mfi(df, p=14):
        mf = df["typical_price"] * df["volume"]
        pos = (
            mf.where(df["typical_price"] > df["typical_price"].shift(1), 0)
            .rolling(p)
            .sum()
        )
        neg = (
            mf.where(df["typical_price"] < df["typical_price"].shift(1), 0)
            .rolling(p)
            .sum()
        )
        return 100 - (100 / (1 + (pos / (neg + 1e-8))))

    df["mfi"] = mfi(df)
    df["vpt"] = (df["volume"] * df["close"].pct_change()).fillna(0).cumsum()
    df["force_index"] = df["close"].diff() * df["volume"]
    df["force_index_ema"] = df["force_index"].ewm(span=13).mean()
    clv = ((df["close"] - df["low"]) - (df["high"] - df["close"])) / (
        df["high"] - df["low"] + 1e-8
    )
    df["ad_line"] = (clv.fillna(0) * df["volume"]).cumsum()
    df["volume_surge"] = (df["volume_ratio"] > 2.0).astype(int)
    return df
