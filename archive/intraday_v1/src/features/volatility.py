import pandas as pd
import numpy as np

def add_volatility_features(df):
    tr = pd.concat(
        [
            df["high"] - df["low"],
            abs(df["high"] - df["close"].shift(1)),
            abs(df["low"] - df["close"].shift(1)),
        ],
        axis=1,
    ).max(axis=1)
    df["true_range"] = tr
    df["atr_7"] = tr.rolling(7).mean()
    df["atr_14"] = tr.rolling(14).mean()

    rets = df["close"].pct_change()
    df["hv_10"] = rets.rolling(10 * 375).std() * np.sqrt(252 * 375)
    df["hv_20"] = rets.rolling(20 * 375).std() * np.sqrt(252 * 375)
    df["hv_30"] = rets.rolling(30 * 375).std() * np.sqrt(252 * 375)

    df["vol_ratio_short"] = df["hv_10"] / (df["hv_30"] + 1e-8)
    df["vol_expansion"] = df["atr_7"] / (df["atr_14"] + 1e-8)

    df["parkinson_vol"] = np.sqrt(
        (np.log(df["high"] / df["low"].replace(0, 1e-8)) ** 2).rolling(20).mean()
        / (4 * np.log(2))
    )
    df["gk_vol"] = np.sqrt(
        (
            0.5 * np.log(df["high"] / df["low"].replace(0, 1e-8)) ** 2
            - (2 * np.log(2) - 1)
            * np.log(df["close"] / df["open"].replace(0, 1e-8)) ** 2
        )
        .rolling(20)
        .mean()
    )
    return df
