"""
Pipeline: Step 03 - Windowing & Gap Detection
Orchestrates index-based windowing to ensure temporal integrity and gap-free training samples.
"""

import sys
import pandas as pd
from pathlib import Path
from loguru import logger
from tqdm import tqdm

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.append(str(PROJECT_ROOT))

from src.data.builder import WindowBuilder  # noqa: E402
from data.configs.stock_universe import get_stock_symbols, symbol_to_name  # noqa: E402


def main():
    builder = WindowBuilder()

    stocks = get_stock_symbols()
    processed_dir = PROJECT_ROOT / "data" / "processed"
    windows_dir = PROJECT_ROOT / "data" / "windows"

    for symbol in tqdm(stocks, desc="Building windows"):
        stock_name = symbol_to_name(symbol)
        proc_path = processed_dir / f"{stock_name}_processed.parquet"

        if not proc_path.exists():
            continue

        df = pd.read_parquet(proc_path)
        windows = builder.create_windows(df, stock_name)

        if windows is not None:
            builder.save_npz(windows, windows_dir)

    logger.info("Window building complete.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
