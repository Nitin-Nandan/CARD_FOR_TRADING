"""
CARD Model Demo Server — v2 (Predicted vs Actual Chart Edition)
================================================================
- Fetches the ENTIRE day's 1-min candles from Fyers with retry/rate-limit handling
- Caches data in-memory per stock (no repeated API hits)
- POST /api/predict accepts {stock, window_idx} where window_idx selects
  which 60-min input window to use.  The next 15 actual candles are returned
  alongside the model predictions so the UI can draw a predicted-vs-actual chart.
"""

import sys, os, time, json, gc
from datetime import datetime, timedelta, date
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from dotenv import load_dotenv
from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS
from fyers_apiv3 import fyersModel

# ── Paths ─────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).parent
sys.path.append(str(PROJECT_ROOT))
from train.config import TrainingConfig, CARDModelConfig
from models.card_true import CARD

load_dotenv()
app = Flask(__name__, static_folder="demo_ui", static_url_path="")
CORS(app)

# ── Config ────────────────────────────────────────────────────────────────────
CLIENT_ID    = os.getenv("CLIENT_ID")
ACCESS_TOKEN = os.getenv("ACCESS_TOKEN")
CHECKPOINTS  = PROJECT_ROOT / "checkpoints"
FINE_TUNED   = CHECKPOINTS / "fine_tuned"
GLOBAL_MODEL = CHECKPOINTS / "best_model.pt"

SEQ_LEN  = 60    # model input window
PRED_LEN = 15    # model output horizon

# Fyers symbol map
SYMBOL_MAP = {
    "TCS":        "NSE:TCS-EQ",
    "INFY":       "NSE:INFY-EQ",
    "HCLTECH":    "NSE:HCLTECH-EQ",
    "WIPRO":      "NSE:WIPRO-EQ",
    "TECHM":      "NSE:TECHM-EQ",
    "HDFCBANK":   "NSE:HDFCBANK-EQ",
    "ICICIBANK":  "NSE:ICICIBANK-EQ",
    "SBIN":       "NSE:SBIN-EQ",
    "RELIANCE":   "NSE:RELIANCE-EQ",
    "KOTAKBANK":  "NSE:KOTAKBANK-EQ",
    "AXISBANK":   "NSE:AXISBANK-EQ",
    "BAJFINANCE": "NSE:BAJFINANCE-EQ",
    "MARUTI":     "NSE:MARUTI-EQ",
    "ITC":        "NSE:ITC-EQ",
    "TITAN":      "NSE:TITAN-EQ",
    "SUNPHARMA":  "NSE:SUNPHARMA-EQ",
    "NTPC":       "NSE:NTPC-EQ",
    "ONGC":       "NSE:ONGC-EQ",
}
NIFTY_SYMBOL = "NSE:NIFTY50-INDEX"

FEATURE_COLS = [
    "open", "high", "low", "close", "volume",
    "ema_5", "ema_10", "roi_1m", "bb_upper", "bb_middle", "bb_lower",
    "nifty_close", "nifty_return", "hour_sin", "hour_cos",
    "realized_vol_30m", "volume_ratio", "volume_surge",
]

FINE_TUNED_STOCKS = ([f.stem.replace("card_", "") for f in FINE_TUNED.glob("card_*.pt")]
                     if FINE_TUNED.exists() else [])

# ── In-memory cache: {stock: (featured_df, fetched_date)} ─────────────────────
_data_cache: dict = {}
_fyers = None


def get_fyers():
    global _fyers
    if _fyers is None:
        _fyers = fyersModel.FyersModel(
            client_id=CLIENT_ID,
            token=ACCESS_TOKEN,
            log_path=str(PROJECT_ROOT / "logs"),
            is_async=False,
        )
    return _fyers


# ── Fyers data fetch with retry ────────────────────────────────────────────────
def _fetch_raw(symbol: str, date_str: str, retries: int = 4, backoff: float = 2.0) -> pd.DataFrame:
    """Fetch 1-min candles for one day/symbol with exponential-backoff retries."""
    fyers = get_fyers()
    for attempt in range(retries):
        try:
            resp = fyers.history({
                "symbol":      symbol,
                "resolution":  "1",
                "date_format": "1",
                "range_from":  date_str,
                "range_to":    date_str,
                "cont_flag":   "1",
            })
            status = resp.get("s", "")
            if status == "ok":
                candles = resp.get("candles", [])
                if not candles:
                    raise RuntimeError(f"No candles returned for {symbol} on {date_str}")
                df = pd.DataFrame(candles, columns=["timestamp","open","high","low","close","volume"])
                df["timestamp"] = (
                    pd.to_datetime(df["timestamp"], unit="s", utc=True)
                      .dt.tz_convert("Asia/Kolkata")
                      .dt.tz_localize(None)
                )
                return df.sort_values("timestamp").reset_index(drop=True)

            err_msg = resp.get("message", str(resp))
            if "rate" in err_msg.lower() or "limit" in err_msg.lower():
                wait = backoff ** attempt
                print(f"  Rate-limited. Waiting {wait:.1f}s …")
                time.sleep(wait)
                continue
            raise RuntimeError(f"Fyers error: {err_msg}")

        except RuntimeError:
            raise
        except Exception as e:
            if attempt < retries - 1:
                wait = backoff ** attempt
                print(f"  Error ({e}). Retry in {wait:.1f}s …")
                time.sleep(wait)
            else:
                raise
    raise RuntimeError(f"Exhausted {retries} retries for {symbol}")


def _last_trading_day() -> str:
    """Return today if it is a weekday, else last Friday. Format YYYY-MM-DD."""
    d = date.today()
    while d.weekday() >= 5:          # Sat=5, Sun=6
        d -= timedelta(days=1)
    return d.strftime("%Y-%m-%d")


def fetch_and_cache(stock: str) -> pd.DataFrame:
    """
    Return fully-featured DataFrame for `stock` on the last trading day.
    Caches result in _data_cache; re-fetches only if the cached date is stale.
    """
    trading_day = _last_trading_day()
    if stock in _data_cache:
        df_cached, cached_date = _data_cache[stock]
        if cached_date == trading_day:
            return df_cached
        del _data_cache[stock]
        gc.collect()

    print(f"Fetching {stock} + Nifty for {trading_day} …")

    # Fyers rate-limit: pause between calls
    raw_df   = _fetch_raw(SYMBOL_MAP[stock], trading_day)
    time.sleep(1.5)
    nifty_df = _fetch_raw(NIFTY_SYMBOL,      trading_day)

    # Filter trading hours 9:15–15:30
    def filter_hours(df):
        t = df["timestamp"].dt.time
        return df[(t >= pd.Timestamp("09:15").time()) & (t <= pd.Timestamp("15:30").time())].copy()

    raw_df   = filter_hours(raw_df)
    nifty_df = filter_hours(nifty_df)

    featured = _add_features(raw_df, nifty_df)
    _data_cache[stock] = (featured, trading_day)
    print(f"  Cached {len(featured)} rows for {stock}")
    return featured


# ── Feature engineering (identical to pipeline/02_process_stocks.py) ──────────
def _add_features(df: pd.DataFrame, nifty_df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["ema_5"]   = df["close"].ewm(span=5,  adjust=False).mean()
    df["ema_10"]  = df["close"].ewm(span=10, adjust=False).mean()
    df["roi_1m"]  = df["close"].pct_change()
    rm = df["close"].rolling(20).mean()
    rs = df["close"].rolling(20).std()
    df["bb_middle"] = rm
    df["bb_upper"]  = rm + 2 * rs
    df["bb_lower"]  = rm - 2 * rs

    nifty_slim = nifty_df[["timestamp","close"]].rename(columns={"close":"nifty_close"})
    df = df.merge(nifty_slim, on="timestamp", how="left")
    df["nifty_close"]  = df["nifty_close"].ffill()
    df["nifty_return"] = df["nifty_close"].pct_change().fillna(0)

    dh = df["timestamp"].dt.hour + df["timestamp"].dt.minute / 60.0
    df["hour_sin"] = np.sin(2 * np.pi * dh / 24)
    df["hour_cos"] = np.cos(2 * np.pi * dh / 24)

    df["realized_vol_30m"] = df["roi_1m"].rolling(30).std()
    vol_ma = df["volume"].rolling(20).mean()
    df["volume_ratio"] = df["volume"] / (vol_ma + 1e-8)
    df["volume_surge"] = (df["volume_ratio"] > 2.0).astype(int)

    return df.dropna().reset_index(drop=True)


# ── Model loading (cached) ────────────────────────────────────────────────────
_model_cache: dict = {}

def load_model(stock: str):
    fine_tuned_path = FINE_TUNED / f"card_{stock}.pt"
    model_path = fine_tuned_path if fine_tuned_path.exists() else GLOBAL_MODEL
    key = str(model_path)
    if key not in _model_cache:
        cfg = CARDModelConfig(TrainingConfig())
        model = CARD(cfg)
        ckpt = torch.load(model_path, map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()
        _model_cache[key] = model
        print(f"  Loaded model: {model_path.name}")
    return _model_cache[key], model_path.name


# ═══════════════════════════════════════════════════════════════════════════════
# ROUTES
# ═══════════════════════════════════════════════════════════════════════════════

@app.route("/")
def index():
    return send_from_directory("demo_ui", "index.html")


@app.route("/api/stocks")
def get_stocks():
    ordered = FINE_TUNED_STOCKS + [s for s in SYMBOL_MAP if s not in FINE_TUNED_STOCKS]
    return jsonify({"stocks": ordered, "fine_tuned": FINE_TUNED_STOCKS})


@app.route("/api/day_data", methods=["POST"])
def day_data():
    """
    Fetch (or return cached) the full day's close prices for a stock.
    Used to draw the full-day actual price line on the chart.
    """
    body  = request.get_json() or {}
    stock = body.get("stock", "TCS").upper()
    if stock not in SYMBOL_MAP:
        return jsonify({"status": "error", "message": f"Unknown stock: {stock}"}), 400
    try:
        df = fetch_and_cache(stock)
        return jsonify({
            "status":     "ok",
            "stock":      stock,
            "date":       df["timestamp"].iloc[-1].strftime("%d %b %Y"),
            "timestamps": df["timestamp"].dt.strftime("%H:%M").tolist(),
            "closes":     df["close"].round(2).tolist(),
            "total_rows": len(df),
            # First valid window index: need 60 input + 15 output = 75 rows from index
            "min_window_idx": 0,
            "max_window_idx": max(0, len(df) - SEQ_LEN - PRED_LEN),
        })
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/api/predict", methods=["POST"])
def predict():
    """
    Run model inference on a specific 60-min window.
    Body: {"stock": "TCS", "window_idx": 42}
    Returns: predictions (15 floats) + actual next-15 prices for comparison.
    """
    body       = request.get_json() or {}
    stock      = body.get("stock", "TCS").upper()
    window_idx = int(body.get("window_idx", 0))

    if stock not in SYMBOL_MAP:
        return jsonify({"status": "error", "message": f"Unknown stock: {stock}"}), 400
    try:
        df = fetch_and_cache(stock)

        max_idx = len(df) - SEQ_LEN - PRED_LEN
        window_idx = max(0, min(window_idx, max_idx))

        # ── Input window ──────────────────────────────────────────────────────
        win = df.iloc[window_idx : window_idx + SEQ_LEN]

        # Check all feature columns present and no NaN
        missing = [c for c in FEATURE_COLS if c not in win.columns]
        if missing:
            return jsonify({"status": "error", "message": f"Missing features: {missing}"}), 500

        X = win[FEATURE_COLS].values.astype(np.float32)   # (60, 18)
        x_tensor = torch.tensor(X).unsqueeze(0).permute(0, 2, 1)  # (1, 18, 60)

        last_input_ts    = str(win["timestamp"].iloc[-1])
        last_input_price = float(win["close"].iloc[-1])

        # ── Actual next 15 candles ────────────────────────────────────────────
        actual_window = df.iloc[window_idx + SEQ_LEN : window_idx + SEQ_LEN + PRED_LEN]
        actual_prices    = actual_window["close"].round(2).tolist()
        actual_timestamps = actual_window["timestamp"].dt.strftime("%H:%M").tolist()

        # ── Model inference ───────────────────────────────────────────────────
        model, model_name = load_model(stock)
        with torch.no_grad():
            pred_close, _ = model(x_tensor)    # (1, 15)
        predictions = [round(p, 2) for p in pred_close.squeeze(0).tolist()]

        # ── Input window context for chart ────────────────────────────────────
        input_timestamps = win["timestamp"].dt.strftime("%H:%M").tolist()
        input_closes     = win["close"].round(2).tolist()

        return jsonify({
            "status":           "ok",
            "stock":            stock,
            "model_used":       model_name,
            "is_fine_tuned":    (FINE_TUNED / f"card_{stock}.pt").exists(),
            "window_idx":       window_idx,
            "last_input_ts":    last_input_ts,
            "last_input_price": last_input_price,
            "input_timestamps": input_timestamps,
            "input_closes":     input_closes,
            "pred_timestamps":  actual_timestamps,
            "predictions":      predictions,
            "actual_prices":    actual_prices,
        })
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"status": "error", "message": str(e)}), 500


if __name__ == "__main__":
    print("=" * 60)
    print("CARD Model Demo Server  (Predicted vs Actual Edition)")
    print(f"Fine-tuned models: {FINE_TUNED_STOCKS or 'none'}")
    print(f"Global model     : {GLOBAL_MODEL}")
    print("Open             : http://localhost:5000")
    print("=" * 60)
    app.run(host="0.0.0.0", port=5000, debug=False)
