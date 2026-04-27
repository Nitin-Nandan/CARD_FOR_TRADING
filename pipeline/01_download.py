"""
Pipeline: Step 01 - Data Acquisition
Handles chunked historical data downloading for the Nifty 500 universe.
"""

import os
import sys
from pathlib import Path
from loguru import logger
from dotenv import load_dotenv

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.append(str(PROJECT_ROOT))

from src.data.downloader import FyersDownloader  # noqa: E402
from data.configs.stock_universe import (  # noqa: E402
    get_stock_symbols,
    symbol_to_name,
    get_index_symbol,
)


def main():
    load_dotenv()
    client_id = os.getenv("CLIENT_ID")
    access_token = os.getenv("ACCESS_TOKEN")

    if not access_token:
        logger.error("ACCESS_TOKEN missing. Run pipeline/00_auth.py first.")
        return 1

    downloader = FyersDownloader(client_id, access_token)

    # Download range
    periods = [("2022-01-01", "2026-02-28")]

    # 1. Download Index
    index_symbol = get_index_symbol()
    logger.info(f"Downloading index: {index_symbol}")
    for start, end in periods:
        ranges = downloader.generate_date_ranges(start, end)
        df_index = downloader.download_symbol(index_symbol, ranges)
        if df_index is not None:
            out = PROJECT_ROOT / "data" / "market_indices" / "NIFTY50.csv"
            out.parent.mkdir(parents=True, exist_ok=True)
            df_index.to_csv(out, index=False)

    # 2. Download Stocks
    stocks = get_stock_symbols()
    for symbol in stocks:
        stock_name = symbol_to_name(symbol)
        out_dir = PROJECT_ROOT / "data" / "raw" / stock_name
        if (out_dir / "full.csv").exists():
            logger.info(f"Skipping {stock_name} (exists)")
            continue

        logger.info(f"Downloading {stock_name}")
        for start, end in periods:
            ranges = downloader.generate_date_ranges(start, end)
            df = downloader.download_symbol(symbol, ranges)
            if df is not None:
                out_dir.mkdir(parents=True, exist_ok=True)
                df.to_csv(out_dir / "full.csv", index=False)

    return 0


if __name__ == "__main__":
    sys.exit(main())
