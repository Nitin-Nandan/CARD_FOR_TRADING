"""
Script 03: Create 60-Minute Windows — TARGET: Future Close Prices

PHASE 0 REWRITE:
- Target (y) = next 15 CLOSE PRICES (not returns, not volatility)
- This matches CARD's same-domain design: input has close prices, output is close prices
- RevIN in the model will handle normalization/denormalization correctly

Window structure:
    Input  X:  minutes [t-60 .. t-1]  → shape (60, 18)  — 60 min of 18 features
    Target y:  minutes [t .. t+14]    → shape (15,)      — next 15 close prices (raw ₹)

Dense sampling (stride=1) for maximum training data.

Usage:
    python pipeline/03_create_windows_60min.py

Output:
    data/windows/{STOCK}_windows.npz
    Contains: X (60,18), y_close (15,), timestamps
"""

import sys
import warnings
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
from loguru import logger
from tqdm import tqdm

warnings.filterwarnings('ignore')

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.append(str(PROJECT_ROOT))

from scripts.config_nifty50 import get_stock_symbols, symbol_to_name

# =====================
# CONFIGURATION
# =====================

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
WINDOWS_DIR   = PROJECT_ROOT / "data" / "windows"
LOG_DIR       = PROJECT_ROOT / "logs"

WINDOWS_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

log_file = LOG_DIR / f"windowing_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
logger.add(log_file, rotation="100 MB", level="INFO")
logger.add(sys.stdout, level="INFO")

INPUT_LEN    = 60   # minutes of input context
PRED_LEN     = 15   # minutes of future to predict
STRIDE       = 1    # dense sampling
NUM_FEATURES = 18   # features in X (all columns except timestamp)


# =====================
# WINDOWING
# =====================

def find_close_index(feature_cols):
    """Find the index of 'close' in the feature column list."""
    if 'close' not in feature_cols:
        raise ValueError(f"'close' column not found! Available: {feature_cols}")
    return feature_cols.index('close')


def create_windows(df, stock_name):
    """
    Create sliding windows from a processed stock DataFrame.

    Returns dict with:
        X          : (num_windows, INPUT_LEN, NUM_FEATURES)  float32
        y_close    : (num_windows, PRED_LEN)                 float32  ← RAW CLOSE PRICES
        timestamps : (num_windows,)  datetime64 — start of prediction window
        feature_names, stock_name
    """
    logger.info(f"\nCreating windows for {stock_name}...")

    feature_cols = [col for col in df.columns if col != 'timestamp']
    if len(feature_cols) != NUM_FEATURES:
        logger.warning(f"Expected {NUM_FEATURES} features, got {len(feature_cols)}: {feature_cols}")

    close_idx = find_close_index(feature_cols)
    logger.info(f"  close_idx = {close_idx} (column: '{feature_cols[close_idx]}')")

    data       = df[feature_cols].values.astype(np.float32)  # (rows, 18)
    timestamps = df['timestamp'].values

    total_rows   = len(data)
    num_windows  = total_rows - INPUT_LEN - PRED_LEN + 1

    if num_windows <= 0:
        logger.error(f"Not enough data! Need {INPUT_LEN + PRED_LEN} rows, got {total_rows}")
        return None

    logger.info(f"  Data rows: {total_rows:,}")
    logger.info(f"  Windows  : {num_windows:,} (stride={STRIDE})")

    # Pre-allocate
    X          = np.zeros((num_windows, INPUT_LEN, NUM_FEATURES), dtype=np.float32)
    y_close    = np.zeros((num_windows, PRED_LEN),                dtype=np.float32)
    win_ts     = np.zeros(num_windows, dtype='datetime64[ns]')

    for i in tqdm(range(num_windows), desc=f"  {stock_name}", leave=False):
        start     = i
        input_end = start + INPUT_LEN
        pred_end  = input_end + PRED_LEN

        # Input: 60 minutes × 18 features
        X[i] = data[start:input_end]

        # Target: next 15 raw close prices  ← KEY CHANGE from old script
        y_close[i] = data[input_end:pred_end, close_idx]

        # Timestamp = first bar of prediction window
        win_ts[i] = timestamps[input_end]

    logger.info(f"  X shape      : {X.shape}")
    logger.info(f"  y_close shape: {y_close.shape}")
    logger.info(f"  y_close range: [{y_close.min():.2f}, {y_close.max():.2f}]")
    logger.info(f"  y_close mean : {y_close.mean():.2f}")

    return {
        'X':             X,
        'y_close':       y_close,
        'timestamps':    win_ts,
        'feature_names': feature_cols,
        'close_idx':     close_idx,
        'stock_name':    stock_name,
    }


def save_windows(windows_data, stock_name):
    """Save windows to compressed .npz"""
    output_path = WINDOWS_DIR / f"{stock_name}_windows.npz"

    np.savez_compressed(
        output_path,
        X             = windows_data['X'],
        y_close       = windows_data['y_close'],
        timestamps    = windows_data['timestamps'],
        feature_names = np.array(windows_data['feature_names']),
        close_idx     = np.array(windows_data['close_idx']),
        stock_name    = np.array(windows_data['stock_name']),
    )

    size_mb = output_path.stat().st_size / (1024 * 1024)
    logger.info(f"  ✅ Saved: {output_path}  ({size_mb:.1f} MB)")
    return output_path


def process_stock(stock_name):
    """Full windowing pipeline for one stock."""
    logger.info(f"\n{'='*60}")
    logger.info(f"Processing: {stock_name}")
    logger.info(f"{'='*60}")

    processed_path = PROCESSED_DIR / f"{stock_name}_processed.csv"
    if not processed_path.exists():
        logger.error(f"  ❌ Processed file not found: {processed_path}")
        return False

    try:
        df = pd.read_csv(processed_path, parse_dates=['timestamp'])
        logger.info(f"  Loaded: {len(df):,} rows")

        windows_data = create_windows(df, stock_name)
        if windows_data is None:
            return False

        save_windows(windows_data, stock_name)
        return True

    except Exception as e:
        logger.exception(f"  ❌ Error: {e}")
        return False


# =====================
# MAIN
# =====================

def main():
    logger.info("\n" + "="*60)
    logger.info("WINDOWING (60-MIN INPUT, 15-MIN CLOSE PRICE TARGET)")
    logger.info("="*60)

    stocks      = get_stock_symbols()
    stock_names = [symbol_to_name(s) for s in stocks]

    logger.info(f"\nTotal stocks : {len(stock_names)}")
    logger.info(f"Input length : {INPUT_LEN} min")
    logger.info(f"Pred length  : {PRED_LEN} min")
    logger.info(f"Target       : y_close (raw close prices, NOT returns)")
    logger.info(f"Stride       : {STRIDE}")

    successful, failed, total_wins = 0, 0, 0
    failed_stocks = []

    for stock_name in tqdm(stock_names, desc="Windowing", unit="stock"):
        ok = process_stock(stock_name)
        if ok:
            successful += 1
            wf = WINDOWS_DIR / f"{stock_name}_windows.npz"
            with np.load(wf) as d:
                total_wins += len(d['X'])
        else:
            failed += 1
            failed_stocks.append(stock_name)

    logger.info("\n" + "="*60)
    logger.info("SUMMARY")
    logger.info("="*60)
    logger.info(f"Successful   : {successful} ✅")
    logger.info(f"Failed       : {failed} ❌")
    logger.info(f"Total windows: {total_wins:,}")
    if successful:
        logger.info(f"Avg per stock: {total_wins // successful:,}")
    if failed_stocks:
        logger.warning(f"Failed: {failed_stocks}")

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        logger.warning("Interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.exception(f"Unexpected error: {e}")
        sys.exit(1)
