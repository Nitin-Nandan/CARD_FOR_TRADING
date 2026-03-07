"""
Script 04: Pass Windows Through (No Pre-normalization)

CRITICAL DESIGN NOTE:
======================
CARD uses RevIN (Reversible Instance Normalization) as its ONLY normalizer.
RevIN normalizes each window at runtime using that window's own statistics.

Pre-normalizing X with StandardScaler BREAKS RevIN because:
  - RevIN stores (mean, std) of its input to undo later
  - If input is already normalized, RevIN's stored stats are in normalized space (~0, ~0.1)
  - Denormed output is still in normalized space, NOT raw ₹
  - y_close is in raw ₹ → domain mismatch → loss = ₹1000+ instead of ₹10

CORRECT design (this script):
  - X is stored as RAW values (open, high, low, close in ₹; indicators in natural units)
  - RevIN receives raw X → normalizes at runtime → model runs → RevIN denorms back to raw ₹
  - y_close is raw ₹ → loss is computed in matching raw ₹ space

Usage:
    python pipeline/04_normalize_windows.py
"""

import sys
from pathlib import Path
import numpy as np
from tqdm import tqdm
from loguru import logger

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.append(str(PROJECT_ROOT))

from scripts.config_nifty50 import get_stock_symbols, symbol_to_name

WINDOWS_DIR = PROJECT_ROOT / "data" / "windows"


def passthrough_stock_windows(stock_name):
    """
    Pass-through: reload the npz and resave with raw X (no StandardScaler).
    Also saves per-stock close channel stats for reference.
    """
    window_file = WINDOWS_DIR / f"{stock_name}_windows.npz"
    if not window_file.exists():
        logger.warning(f"  File not found: {window_file}")
        return False

    try:
        data = np.load(window_file, allow_pickle=True)

        X             = data['X']             # (N, 60, 18) — may be pre-normalized (old)
        y_close       = data['y_close']       # (N, 15)  — raw prices
        timestamps    = data['timestamps']
        feature_names = data['feature_names']
        close_idx     = data['close_idx']
        stock_name_   = data['stock_name']

        # Check if already raw (close price >> 10 means raw, not normalized)
        close_channel = int(close_idx)
        close_max = X[:, :, close_channel].max()
        already_raw = close_max > 50.0   # raw Nifty prices are ₹50+

        if already_raw:
            logger.info(f"  X already raw (close max={close_max:.1f} ₹) — saving as-is")
        else:
            logger.warning(f"  X appears pre-normalized (close max={close_max:.2f}) — "
                           f"cannot un-normalize without original scaler. "
                           f"Re-run pipeline/03_create_windows_60min.py to get raw windows.")
            # Still save — the caller must re-run step 3 for correct raw data
            return False

        logger.info(f"  X range: [{X.min():.2f}, {X.max():.2f}]")
        logger.info(f"  y_close range: [{y_close.min():.2f}, {y_close.max():.2f}]")

        # Save with a flag so we know this stock has raw X
        np.savez_compressed(
            window_file,
            X             = X.astype(np.float32),
            y_close       = y_close.astype(np.float32),
            timestamps    = timestamps,
            feature_names = feature_names,
            close_idx     = close_idx,
            stock_name    = stock_name_,
            x_normalized  = np.array(False),   # flag: X is RAW, RevIN normalizes at runtime
        )
        logger.info(f"  ✅ Saved: {window_file}")
        return True

    except Exception as e:
        logger.exception(f"  ❌ Error: {e}")
        return False


def main():
    logger.info("\n" + "="*60)
    logger.info("VERIFYING WINDOWS ARE RAW (RevIN is the only normalizer)")
    logger.info("="*60)
    logger.info(
        "CARD design: X must be RAW prices so RevIN can normalize/denormalize correctly.\n"
        "If X is pre-normalized, re-run pipeline/03_create_windows_60min.py first."
    )

    stocks      = get_stock_symbols()
    stock_names = [symbol_to_name(s) for s in stocks]
    logger.info(f"Total stocks: {len(stock_names)}")

    ok, fail = 0, 0
    for stock_name in tqdm(stock_names, desc="Verifying", unit="stock"):
        logger.info(f"\n{stock_name}")
        if passthrough_stock_windows(stock_name):
            ok += 1
        else:
            fail += 1

    logger.info(f"\nDone — OK: {ok} ✅  Failed: {fail} ❌")
    if fail > 0:
        logger.error(
            f"\n{'='*60}\n"
            f"  {fail} stocks failed because X is pre-normalized.\n"
            f"  Action: re-run pipeline/03_create_windows_60min.py\n"
            f"  Then re-run this script to verify.\n"
            f"{'='*60}"
        )
    return 0 if fail == 0 else 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        logger.warning("Interrupted")
        sys.exit(1)
    except Exception as e:
        logger.exception(f"Unexpected error: {e}")
        sys.exit(1)
