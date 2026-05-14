import pandas as pd
import numpy as np

def add_momentum_features(df):
    def rsi(series, period=14):
        delta = series.diff()
        gain = (delta.where(delta > 0, 0)).rolling(period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
        return 100 - (100 / (1 + (gain / (loss + 1e-8))))

    df["rsi_7"] = rsi(df["close"], 7)
    df["rsi_14"] = rsi(df["close"], 14)
    df["rsi_21"] = rsi(df["close"], 21)

    df["ema_12"] = df["close"].ewm(span=12).mean()
    df["ema_26"] = df["close"].ewm(span=26).mean()
    df["macd"] = df["ema_12"] - df["ema_26"]
    df["macd_signal"] = df["macd"].ewm(span=9).mean()
    df["macd_hist"] = df["macd"] - df["macd_signal"]

    low_14, high_14 = df["low"].rolling(14).min(), df["high"].rolling(14).max()
    df["stoch_k"] = 100 * (df["close"] - low_14) / (high_14 - low_14 + 1e-8)
    df["stoch_d"] = df["stoch_k"].rolling(3).mean()

    df["roc_5"] = df["close"].pct_change(5) * 100
    df["roc_10"] = df["close"].pct_change(10) * 100
    df["roc_20"] = df["close"].pct_change(20) * 100
    df["williams_r"] = -100 * (high_14 - df["close"]) / (high_14 - low_14 + 1e-8)
    return df
