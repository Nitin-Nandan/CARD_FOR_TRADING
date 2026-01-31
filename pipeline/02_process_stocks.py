"""
Script 02: Process Stocks - Clean & Add 18 Features

Processes raw OHLCV data:
1. Filter trading hours only (9:15 AM - 3:30 PM IST)
2. Remove weekends and holidays
3. Handle missing data (interpolation)
4. Add 18 features:
   - Technical indicators (6)
   - Market context (4)
   - Volatility & volume (3)
   - Original OHLCV (5)

Usage:
    python scripts/02_process_stocks.py

Output:
    - data/processed/{STOCK_NAME}_processed.csv (18 features per stock)
    - data/processed/NIFTY50_processed.csv (index with features)
    - logs/processing_log.txt
"""

import sys
import warnings
from pathlib import Path
from datetime import time, datetime

import numpy as np
import pandas as pd
from loguru import logger
from tqdm import tqdm

warnings.filterwarnings('ignore')

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.append(str(PROJECT_ROOT))

from scripts.config_nifty50 import get_stock_symbols, symbol_to_name

# =====================
# CONFIGURATION
# =====================

# Paths
RAW_DIR = PROJECT_ROOT / "data" / "raw"
INDEX_DIR = PROJECT_ROOT / "data" / "market_indices"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
LOG_DIR = PROJECT_ROOT / "logs"

# Create directories
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

# Setup logging
log_file = LOG_DIR / f"processing_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
logger.add(log_file, rotation="100 MB", level="INFO")
logger.add(sys.stdout, level="INFO")

# Trading hours (IST)
MARKET_OPEN = time(9, 15)   # 9:15 AM
MARKET_CLOSE = time(15, 30)  # 3:30 PM

# Feature calculation windows
EMA_SHORT = 5
EMA_LONG = 10
BB_WINDOW = 20
VOLATILITY_WINDOW = 30
VOLUME_WINDOW = 20

# =====================
# PROCESSING FUNCTIONS
# =====================

def filter_trading_hours(df):
    """
    Keep only data during trading hours (9:15 AM - 3:30 PM IST)
    Remove pre-market and post-market data
    
    Args:
        df: DataFrame with 'timestamp' column
        
    Returns:
        Filtered DataFrame
    """
    df = df.copy()
    df['time'] = df['timestamp'].dt.time
    
    # Filter: 9:15 AM <= time <= 3:30 PM
    mask = (df['time'] >= MARKET_OPEN) & (df['time'] <= MARKET_CLOSE)
    df_filtered = df[mask].copy()
    
    # Drop temporary column
    df_filtered = df_filtered.drop(columns=['time'])
    
    logger.info(f"  Trading hours filter: {len(df):,} → {len(df_filtered):,} rows "
                f"({len(df) - len(df_filtered):,} removed)")
    
    return df_filtered


def filter_weekends(df):
    """
    Remove weekend data (Saturday=5, Sunday=6)
    
    Args:
        df: DataFrame with 'timestamp' column
        
    Returns:
        Filtered DataFrame
    """
    df = df.copy()
    df['day_of_week'] = df['timestamp'].dt.dayofweek
    
    # Keep Monday(0) to Friday(4)
    df_filtered = df[df['day_of_week'] < 5].copy()
    
    # Drop temporary column
    df_filtered = df_filtered.drop(columns=['day_of_week'])
    
    logger.info(f"  Weekend filter: {len(df):,} → {len(df_filtered):,} rows "
                f"({len(df) - len(df_filtered):,} removed)")
    
    return df_filtered


def handle_missing_data(df):
    """
    Handle missing values:
    1. Forward fill for small gaps (<5 rows)
    2. Drop rows with remaining NaN
    
    Args:
        df: DataFrame with OHLCV data
        
    Returns:
        DataFrame with missing data handled
    """
    df = df.copy()
    
    # Count initial NaN
    nan_before = df.isnull().sum().sum()
    
    if nan_before > 0:
        # Forward fill (limit to 5 rows = 5 minutes)
        df = df.ffill(limit=5)
        
        # Drop any remaining NaN
        df = df.dropna()
        
        nan_after = df.isnull().sum().sum()
        
        logger.info(f"  Missing data: {nan_before} NaN values → "
                    f"{nan_after} after handling "
                    f"({nan_before - nan_after} filled)")
    
    return df


def add_technical_indicators(df):
    """
    Add 6 technical indicator features:
    1. EMA_5 (5-period exponential moving average)
    2. EMA_10 (10-period exponential moving average)
    3. ROI_1m (1-minute return / rate of change)
    4. BB_upper (Bollinger Band upper)
    5. BB_middle (Bollinger Band middle/SMA)
    6. BB_lower (Bollinger Band lower)
    
    Args:
        df: DataFrame with 'close' column
        
    Returns:
        DataFrame with technical indicators added
    """
    df = df.copy()
    
    # EMAs
    df['ema_5'] = df['close'].ewm(span=EMA_SHORT, adjust=False).mean()
    df['ema_10'] = df['close'].ewm(span=EMA_LONG, adjust=False).mean()
    
    # ROI (1-minute return)
    df['roi_1m'] = df['close'].pct_change()
    
    # Bollinger Bands (20-period)
    rolling_mean = df['close'].rolling(window=BB_WINDOW).mean()
    rolling_std = df['close'].rolling(window=BB_WINDOW).std()
    
    df['bb_middle'] = rolling_mean
    df['bb_upper'] = rolling_mean + 2 * rolling_std
    df['bb_lower'] = rolling_mean - 2 * rolling_std
    
    logger.info(f"  Added 6 technical indicators")
    
    return df


def add_market_context(df, nifty_df):
    """
    Add 4 market context features:
    1. nifty_close (Nifty 50 index price)
    2. nifty_return (Nifty 50 return)
    3. hour_sin (hour of day, sine encoding)
    4. hour_cos (hour of day, cosine encoding)
    
    Args:
        df: Stock DataFrame with 'timestamp'
        nifty_df: Nifty index DataFrame
        
    Returns:
        DataFrame with market context features
    """
    df = df.copy()
    
    # Merge with Nifty data (on timestamp)
    nifty_data = nifty_df[['timestamp', 'close']].copy()
    nifty_data = nifty_data.rename(columns={'close': 'nifty_close'})
    
    df = df.merge(nifty_data, on='timestamp', how='left')
    
    # Nifty return
    df['nifty_return'] = df['nifty_close'].pct_change()
    
    # Time of day encoding (cyclic)
    df['hour'] = df['timestamp'].dt.hour
    df['minute'] = df['timestamp'].dt.minute
    df['decimal_hour'] = df['hour'] + df['minute'] / 60.0
    
    # Sine/Cosine encoding (24-hour cycle)
    df['hour_sin'] = np.sin(2 * np.pi * df['decimal_hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['decimal_hour'] / 24)
    
    # Drop temporary columns
    df = df.drop(columns=['hour', 'minute', 'decimal_hour'])
    
    # Check for missing Nifty data
    nifty_missing = df['nifty_close'].isnull().sum()
    if nifty_missing > 0:
        logger.warning(f"  {nifty_missing} rows missing Nifty data (will forward fill)")
        df['nifty_close'] = df['nifty_close'].ffill()
        df['nifty_return'] = df['nifty_return'].fillna(0)
    
    logger.info(f"  Added 4 market context features")
    
    return df


def add_volatility_volume_features(df):
    """
    Add 3 volatility and volume features:
    1. realized_vol_30m (30-minute realized volatility)
    2. volume_ratio (volume vs 20-min average)
    3. volume_surge (boolean: volume > 2× average)
    
    Args:
        df: DataFrame with 'close' and 'volume'
        
    Returns:
        DataFrame with volatility/volume features
    """
    df = df.copy()
    
    # Realized volatility (30-minute rolling std of returns)
    df['realized_vol_30m'] = df['roi_1m'].rolling(window=VOLATILITY_WINDOW).std()
    
    # Volume ratio (current / 20-min average)
    volume_ma = df['volume'].rolling(window=VOLUME_WINDOW).mean()
    df['volume_ratio'] = df['volume'] / (volume_ma + 1e-8)  # avoid division by zero
    
    # Volume surge indicator (volume > 2× average)
    df['volume_surge'] = (df['volume_ratio'] > 2.0).astype(int)
    
    logger.info(f"  Added 3 volatility/volume features")
    
    return df


def process_stock(stock_name, nifty_df):
    """
    Complete processing pipeline for one stock
    
    Args:
        stock_name: Stock name (e.g., "RELIANCE")
        nifty_df: Processed Nifty index DataFrame
        
    Returns:
        Tuple of (success: bool, processed_rows: int)
    """
    logger.info(f"\n{'='*60}")
    logger.info(f"Processing: {stock_name}")
    logger.info(f"{'='*60}")
    
    # Load raw data
    raw_path = RAW_DIR / stock_name / "full.csv"
    
    if not raw_path.exists():
        logger.error(f"  ❌ File not found: {raw_path}")
        return False, 0
    
    try:
        df = pd.read_csv(raw_path, parse_dates=['timestamp'])
        logger.info(f"  Loaded: {len(df):,} rows")
        
        # Step 1: Filter trading hours
        df = filter_trading_hours(df)
        
        # Step 2: Filter weekends
        df = filter_weekends(df)
        
        # Step 3: Handle missing data
        df = handle_missing_data(df)
        
        # Step 4: Add technical indicators
        df = add_technical_indicators(df)
        
        # Step 5: Add market context
        df = add_market_context(df, nifty_df)
        
        # Step 6: Add volatility/volume features
        df = add_volatility_volume_features(df)
        
        # Step 7: Drop initial NaN rows (from rolling windows)
        rows_before = len(df)
        df = df.dropna().reset_index(drop=True)
        rows_dropped = rows_before - len(df)
        
        if rows_dropped > 0:
            logger.info(f"  Dropped {rows_dropped} rows with NaN (from rolling windows)")
        
        # Verify final feature count
        expected_features = [
            'timestamp',
            # OHLCV (5)
            'open', 'high', 'low', 'close', 'volume',
            # Technical indicators (6)
            'ema_5', 'ema_10', 'roi_1m', 'bb_upper', 'bb_middle', 'bb_lower',
            # Market context (4)
            'nifty_close', 'nifty_return', 'hour_sin', 'hour_cos',
            # Volatility/volume (3)
            'realized_vol_30m', 'volume_ratio', 'volume_surge'
        ]
        
        # Reorder columns to match expected order
        df = df[expected_features]
        
        logger.info(f"  Final features: {len(df.columns)} (expected: {len(expected_features)})")
        
        if len(df.columns) != len(expected_features):
            logger.error(f"  ❌ Feature mismatch! Expected {len(expected_features)}, got {len(df.columns)}")
            return False, 0
        
        # Save processed data
        output_path = PROCESSED_DIR / f"{stock_name}_processed.csv"
        df.to_csv(output_path, index=False)
        
        logger.info(f"  ✅ Saved: {output_path}")
        logger.info(f"     Final rows: {len(df):,}")
        logger.info(f"     Date range: {df['timestamp'].min()} → {df['timestamp'].max()}")
        
        return True, len(df)
        
    except Exception as e:
        logger.exception(f"  ❌ Error processing {stock_name}: {e}")
        return False, 0


def process_nifty_index():
    """
    Process Nifty 50 index data (same pipeline as stocks, without market context)
    
    Returns:
        Processed Nifty DataFrame
    """
    logger.info(f"\n{'='*60}")
    logger.info(f"Processing: NIFTY 50 INDEX")
    logger.info(f"{'='*60}")
    
    # Load raw data
    raw_path = INDEX_DIR / "NIFTY50.csv"
    
    if not raw_path.exists():
        logger.error(f"  ❌ File not found: {raw_path}")
        return None
    
    df = pd.read_csv(raw_path, parse_dates=['timestamp'])
    logger.info(f"  Loaded: {len(df):,} rows")
    
    # Step 1: Filter trading hours
    df = filter_trading_hours(df)
    
    # Step 2: Filter weekends
    df = filter_weekends(df)
    
    # Step 3: Handle missing data
    df = handle_missing_data(df)
    
    # Step 4: Add basic features (for later use)
    # Note: We don't add market context to Nifty itself
    df['return'] = df['close'].pct_change()
    df['volatility'] = df['return'].rolling(window=30).std()
    
    # Drop NaN
    df = df.dropna().reset_index(drop=True)
    
    # Save processed Nifty
    output_path = PROCESSED_DIR / "NIFTY50_processed.csv"
    df.to_csv(output_path, index=False)
    
    logger.info(f"  ✅ Saved: {output_path}")
    logger.info(f"     Final rows: {len(df):,}")
    
    return df


# =====================
# MAIN PROCESSING
# =====================

def main():
    """
    Main processing orchestration
    """
    logger.info("\n" + "="*60)
    logger.info("NIFTY 50 DATA PROCESSING")
    logger.info("="*60)
    
    # Process Nifty index first (needed for market context)
    logger.info("\nSTEP 1: PROCESS NIFTY 50 INDEX")
    logger.info("="*60)
    
    nifty_df = process_nifty_index()
    
    if nifty_df is None:
        logger.error("❌ Failed to process Nifty index. Cannot continue.")
        return 1
    
    # Get stock list
    stocks = get_stock_symbols()
    stock_names = [symbol_to_name(s) for s in stocks]
    
    logger.info("\n" + "="*60)
    logger.info(f"STEP 2: PROCESS {len(stock_names)} STOCKS")
    logger.info("="*60)
    
    # Process all stocks
    summary = {
        'total': len(stock_names),
        'successful': 0,
        'failed': 0,
        'total_rows': 0,
        'failed_stocks': []
    }
    
    # Progress bar
    stock_pbar = tqdm(stock_names, desc="Processing stocks", unit="stock")
    
    for stock_name in stock_pbar:
        stock_pbar.set_postfix_str(stock_name)
        
        success, rows = process_stock(stock_name, nifty_df)
        
        if success:
            summary['successful'] += 1
            summary['total_rows'] += rows
        else:
            summary['failed'] += 1
            summary['failed_stocks'].append(stock_name)
    
    # Print summary
    logger.info("\n" + "="*60)
    logger.info("PROCESSING SUMMARY")
    logger.info("="*60)
    logger.info(f"Total stocks: {summary['total']}")
    logger.info(f"Successful: {summary['successful']} ✅")
    logger.info(f"Failed: {summary['failed']} ❌")
    logger.info(f"Total processed rows: {summary['total_rows']:,}")
    
    if summary['failed_stocks']:
        logger.warning("\nFailed stocks:")
        for stock in summary['failed_stocks']:
            logger.warning(f"  - {stock}")
    
    logger.info(f"\nProcessed data saved in: {PROCESSED_DIR}")
    logger.info(f"Log file: {log_file}")
    
    logger.info("\n" + "="*60)
    logger.info("✅ PROCESSING COMPLETE")
    logger.info("="*60)
    
    # Calculate statistics
    if summary['successful'] > 0:
        avg_rows = summary['total_rows'] / summary['successful']
        logger.info(f"\nAverage rows per stock: {avg_rows:,.0f}")
        logger.info(f"Expected features per stock: 19 columns (timestamp + 18 features)")
    
    # Exit code
    if summary['failed'] == 0:
        logger.info("\n🎉 All stocks processed successfully!")
        return 0
    else:
        logger.warning(f"\n⚠️  {summary['failed']} stocks failed")
        return 1


if __name__ == "__main__":
    try:
        exit_code = main()
        sys.exit(exit_code)
    except KeyboardInterrupt:
        logger.warning("\n⚠️  Processing interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.exception(f"❌ Unexpected error: {e}")
        sys.exit(1)
