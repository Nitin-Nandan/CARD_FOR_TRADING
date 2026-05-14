"""
Pipeline: Step 02 - Feature Engineering
Triggers the 81-channel feature engine to transform raw OHLCV into predictive tensors.
"""

import sys
import pandas as pd
from pathlib import Path
from loguru import logger
from tqdm import tqdm

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.append(str(PROJECT_ROOT))

from src.features.engine import FeatureEngine  # noqa: E402
from data.configs.stock_universe import get_stock_symbols, symbol_to_name  # noqa: E402


def main():
    engine = FeatureEngine()

    # Load Nifty for cross-asset features
    nifty_path = PROJECT_ROOT / "data" / "market_indices" / "NIFTY50.csv"
    nifty_df = (
        pd.read_csv(nifty_path, parse_dates=["timestamp"])
        if nifty_path.exists()
        else None
    )

    stocks = get_stock_symbols()
    processed_dir = PROJECT_ROOT / "data" / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)

    for symbol in tqdm(stocks, desc="Processing features"):
        stock_name = symbol_to_name(symbol)
        raw_path = PROJECT_ROOT / "data" / "raw" / stock_name / "full.csv"

        if not raw_path.exists():
            continue

        df = pd.read_csv(raw_path, parse_dates=["timestamp"])
        processed_df = engine.process(df, nifty_df=nifty_df)

        out_path = processed_dir / f"{stock_name}_processed.parquet"
        processed_df.to_parquet(out_path, engine="pyarrow", index=False)

    logger.info("Feature engineering complete.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
