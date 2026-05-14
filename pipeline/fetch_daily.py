"""
Pipeline: Step 10 - Daily Data Acquisition
Downloads daily OHLCV for 25 NSE stocks + Nifty50 index via yfinance.
Saves raw CSVs to data/raw/daily/.
"""

import sys
from pathlib import Path

import yfinance as yf
import pandas as pd
from tqdm import tqdm
import logging
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.append(str(PROJECT_ROOT))

from data.configs.daily_universe import (  # noqa: E402
    get_daily_symbols,
    ticker_to_name,
    NIFTY50_SYMBOL,
)

START_DATE = "2018-01-01"
END_DATE = "2026-12-31"
RAW_DAILY_DIR = PROJECT_ROOT / "data" / "raw" / "daily"


def _download_and_save(ticker: str, start: str, end: str, out_path: Path) -> bool:
    """Download one ticker and write to CSV. Returns True on success."""
    try:
        df = yf.download(
            ticker,
            start=start,
            end=end,
            progress=False,
            auto_adjust=True,
        )
        if df.empty:
            logger.warning(f"  [SKIP] {ticker}: empty DataFrame from yfinance")
            return False

        # Flatten MultiIndex columns if present (yfinance >=0.2 quirk)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        df.index.name = "date"
        df.columns = [c.lower() for c in df.columns]
        df = df[["open", "high", "low", "close", "volume"]].copy()
        df = df.dropna()
        df.to_csv(out_path)
        return True
    except Exception as exc:
        logger.error(f"  [ERROR] {ticker}: {exc}")
        return False


def main() -> int:
    RAW_DAILY_DIR.mkdir(parents=True, exist_ok=True)

    symbols = get_daily_symbols()
    all_tickers = [(NIFTY50_SYMBOL, "NIFTY50")] + [
        (sym, ticker_to_name(sym)) for sym in symbols
    ]

    ok, fail = 0, 0

    for ticker, name in tqdm(all_tickers, desc="Downloading daily OHLCV", unit="stock"):
        out_path = RAW_DAILY_DIR / f"{name}.csv"
        if out_path.exists():
            logger.info(f"  [SKIP] {name} already exists")
            ok += 1
            continue

        logger.info(f"  Downloading {name} ({ticker})")
        success = _download_and_save(ticker, START_DATE, END_DATE, out_path)
        if success:
            ok += 1
        else:
            fail += 1

    logger.info(f"\nDone — {ok} downloaded, {fail} failed.")
    return 0 if fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
