"""
Script 01: Download Nifty 50 Stocks + Index Data

Downloads minute-level OHLCV data for all 50 Nifty stocks plus Nifty 50 index
from January 2022 to December 2025 using Fyers API.

Features:
- Progress bars for each stock
- Chunked downloads to avoid API limits
- Error handling and retry logic
- Automatic gap detection
- Summary statistics

Usage:
    python scripts/01_download_nifty50.py

Output:
    - data/raw/{STOCK_NAME}/full.csv
    - data/market_indices/NIFTY50.csv
    - logs/download_log.txt
"""

import os
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from fyers_apiv3 import fyersModel
from loguru import logger
from tqdm import tqdm

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.append(str(PROJECT_ROOT))

from config_nifty50 import get_stock_symbols, get_index_symbol, symbol_to_name

# Load environment variables
load_dotenv()

# =====================
# CONFIGURATION
# =====================

CLIENT_ID = os.getenv("CLIENT_ID")
ACCESS_TOKEN = os.getenv("ACCESS_TOKEN")
REDIRECT_URI = "https://127.0.0.1/"

# Date range
START_DATE = "2022-01-01"
END_DATE = "2025-12-31"

# Chunking (to avoid API limits)
CHUNK_DAYS = 30  # Download 30 days at a time

# Rate limiting
SLEEP_BETWEEN_CHUNKS = 1.0  # seconds
SLEEP_BETWEEN_STOCKS = 2.0  # seconds

# Retry logic
MAX_RETRIES = 3
RETRY_DELAY = 5  # seconds

# Paths
DATA_DIR = PROJECT_ROOT / "data" / "raw"
INDEX_DIR = PROJECT_ROOT / "data" / "market_indices"
LOG_DIR = PROJECT_ROOT / "logs"

# Create directories
DATA_DIR.mkdir(parents=True, exist_ok=True)
INDEX_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

# Setup logging
log_file = LOG_DIR / f"download_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
logger.add(log_file, rotation="100 MB", level="INFO")
logger.add(sys.stdout, level="INFO")

# =====================
# INITIALIZE FYERS
# =====================

logger.info("Initializing Fyers API connection...")

try:
    fyers = fyersModel.FyersModel(
        client_id=CLIENT_ID,
        token=ACCESS_TOKEN,
        log_path=str(LOG_DIR)
    )
    logger.info("✅ Fyers API initialized successfully")
except Exception as e:
    logger.error(f"❌ Failed to initialize Fyers API: {e}")
    sys.exit(1)

# =====================
# HELPER FUNCTIONS
# =====================

def generate_date_ranges(start_date, end_date, chunk_days=30):
    """
    Split date range into chunks to avoid API limits.
    
    Args:
        start_date: Start date string (YYYY-MM-DD)
        end_date: End date string (YYYY-MM-DD)
        chunk_days: Number of days per chunk
    
    Returns:
        List of date range dictionaries
    """
    start = datetime.strptime(start_date, "%Y-%m-%d")
    end = datetime.strptime(end_date, "%Y-%m-%d")
    
    ranges = []
    current = start
    
    while current < end:
        chunk_end = min(current + timedelta(days=chunk_days), end)
        ranges.append({
            'start': current.strftime("%Y-%m-%d"),
            'end': chunk_end.strftime("%Y-%m-%d")
        })
        current = chunk_end + timedelta(days=1)
    
    return ranges


def download_chunk(symbol, range_from, range_to, retry_count=0):
    """
    Download one chunk of data with retry logic.
    
    Args:
        symbol: Fyers symbol (e.g., "NSE:RELIANCE-EQ")
        range_from: Start date (YYYY-MM-DD)
        range_to: End date (YYYY-MM-DD)
        retry_count: Current retry attempt
    
    Returns:
        DataFrame with OHLCV data or None if failed
    """
    data = {
        "symbol": symbol,
        "resolution": "1",  # 1-minute candles
        "date_format": "1",  # Unix timestamp
        "range_from": range_from,
        "range_to": range_to,
        "cont_flag": "1"  # Continuous data
    }
    
    try:
        response = fyers.history(data)
        
        # Check response status
        if response.get("s") != "ok":
            error_msg = response.get("message", "Unknown error")
            logger.warning(f"  ⚠️  API error: {error_msg}")
            
            # Retry logic
            if retry_count < MAX_RETRIES:
                logger.info(f"  🔄 Retrying ({retry_count + 1}/{MAX_RETRIES})...")
                time.sleep(RETRY_DELAY)
                return download_chunk(symbol, range_from, range_to, retry_count + 1)
            else:
                logger.error(f"  ❌ Max retries reached for {range_from} to {range_to}")
                return None
        
        # Extract candles
        candles = response.get("candles", [])
        
        if not candles:
            logger.warning(f"  ⚠️  No data returned for {range_from} to {range_to}")
            return None
        
        # Convert to DataFrame
        df = pd.DataFrame(
            candles,
            columns=["timestamp", "open", "high", "low", "close", "volume"]
        )
        
        # Convert timestamp to IST
        df["timestamp"] = (
            pd.to_datetime(df["timestamp"], unit="s", utc=True)
              .dt.tz_convert("Asia/Kolkata")
              .dt.tz_localize(None)
        )
        
        # Sort by timestamp
        df = df.sort_values("timestamp").reset_index(drop=True)
        
        return df
        
    except Exception as e:
        logger.error(f"  ❌ Exception during download: {e}")
        
        # Retry logic
        if retry_count < MAX_RETRIES:
            logger.info(f"  🔄 Retrying ({retry_count + 1}/{MAX_RETRIES})...")
            time.sleep(RETRY_DELAY)
            return download_chunk(symbol, range_from, range_to, retry_count + 1)
        else:
            return None


def download_stock(symbol, stock_name, date_ranges):
    """
    Download complete data for one stock.
    
    Args:
        symbol: Fyers symbol
        stock_name: Stock name (e.g., "RELIANCE")
        date_ranges: List of date range dictionaries
    
    Returns:
        Tuple of (success: bool, total_rows: int)
    """
    logger.info(f"\n{'='*60}")
    logger.info(f"Downloading: {stock_name} ({symbol})")
    logger.info(f"{'='*60}")
    
    all_chunks = []
    failed_chunks = []
    
    # Progress bar for chunks
    chunk_pbar = tqdm(
        date_ranges,
        desc=f"{stock_name:15s}",
        unit="chunk",
        leave=True,
        ncols=100
    )
    
    for date_range in chunk_pbar:
        range_start = date_range['start']
        range_end = date_range['end']
        
        # Update progress bar description
        chunk_pbar.set_postfix_str(f"{range_start} → {range_end}")
        
        # Download chunk
        df_chunk = download_chunk(symbol, range_start, range_end)
        
        if df_chunk is not None and len(df_chunk) > 0:
            all_chunks.append(df_chunk)
            logger.info(f"  ✅ Downloaded {len(df_chunk):,} rows")
        else:
            failed_chunks.append((range_start, range_end))
            logger.warning(f"  ⚠️  Failed to download chunk")
        
        # Rate limiting
        time.sleep(SLEEP_BETWEEN_CHUNKS)
    
    # Check if any data was downloaded
    if not all_chunks:
        logger.error(f"❌ No data downloaded for {stock_name}")
        return False, 0
    
    # Combine all chunks
    logger.info(f"\nCombining {len(all_chunks)} chunks...")
    df_full = pd.concat(all_chunks, ignore_index=True)
    df_full = df_full.sort_values("timestamp").reset_index(drop=True)
    
    # Remove duplicates (in case of overlapping chunks)
    before_dedup = len(df_full)
    df_full = df_full.drop_duplicates(subset=["timestamp"], keep="first")
    after_dedup = len(df_full)
    
    if before_dedup != after_dedup:
        logger.info(f"Removed {before_dedup - after_dedup:,} duplicate timestamps")
    
    # Save to CSV
    stock_dir = DATA_DIR / stock_name
    stock_dir.mkdir(exist_ok=True)
    
    output_path = stock_dir / "full.csv"
    df_full.to_csv(output_path, index=False)
    
    logger.info(f"✅ Saved: {output_path}")
    logger.info(f"   Total rows: {len(df_full):,}")
    logger.info(f"   Date range: {df_full['timestamp'].min()} → {df_full['timestamp'].max()}")
    
    if failed_chunks:
        logger.warning(f"   Failed chunks: {len(failed_chunks)}")
        for start, end in failed_chunks:
            logger.warning(f"     - {start} to {end}")
    
    return True, len(df_full)


def download_nifty_index(date_ranges):
    """
    Download Nifty 50 index data.
    
    Args:
        date_ranges: List of date range dictionaries
    
    Returns:
        Tuple of (success: bool, total_rows: int)
    """
    symbol = get_index_symbol()
    
    logger.info(f"\n{'='*60}")
    logger.info(f"Downloading: NIFTY 50 INDEX ({symbol})")
    logger.info(f"{'='*60}")
    
    all_chunks = []
    
    # Progress bar for chunks
    chunk_pbar = tqdm(
        date_ranges,
        desc="NIFTY50 INDEX ",
        unit="chunk",
        leave=True,
        ncols=100
    )
    
    for date_range in chunk_pbar:
        range_start = date_range['start']
        range_end = date_range['end']
        
        chunk_pbar.set_postfix_str(f"{range_start} → {range_end}")
        
        df_chunk = download_chunk(symbol, range_start, range_end)
        
        if df_chunk is not None and len(df_chunk) > 0:
            all_chunks.append(df_chunk)
        
        time.sleep(SLEEP_BETWEEN_CHUNKS)
    
    if not all_chunks:
        logger.error("❌ No index data downloaded")
        return False, 0
    
    # Combine and save
    df_full = pd.concat(all_chunks, ignore_index=True)
    df_full = df_full.sort_values("timestamp").reset_index(drop=True)
    df_full = df_full.drop_duplicates(subset=["timestamp"], keep="first")
    
    output_path = INDEX_DIR / "NIFTY50.csv"
    df_full.to_csv(output_path, index=False)
    
    logger.info(f"✅ Saved: {output_path}")
    logger.info(f"   Total rows: {len(df_full):,}")
    logger.info(f"   Date range: {df_full['timestamp'].min()} → {df_full['timestamp'].max()}")
    
    return True, len(df_full)


# =====================
# MAIN DOWNLOAD PROCESS
# =====================

def main():
    """
    Main download orchestration function.
    """
    logger.info("\n" + "="*60)
    logger.info("NIFTY 50 DATA DOWNLOAD")
    logger.info("="*60)
    logger.info(f"Date range: {START_DATE} to {END_DATE}")
    logger.info(f"Chunk size: {CHUNK_DAYS} days")
    logger.info("="*60)
    
    # Generate date ranges
    date_ranges = generate_date_ranges(START_DATE, END_DATE, CHUNK_DAYS)
    logger.info(f"Total chunks per stock: {len(date_ranges)}")
    
    # Get stock list
    stocks = get_stock_symbols()
    logger.info(f"Total stocks to download: {len(stocks)}")
    
    # Download summary
    summary = {
        'total_stocks': len(stocks),
        'successful': 0,
        'failed': 0,
        'total_rows': 0,
        'failed_stocks': []
    }
    
    # Download Nifty 50 index first
    logger.info("\n" + "="*60)
    logger.info("STEP 1: DOWNLOAD NIFTY 50 INDEX")
    logger.info("="*60)
    
    index_success, index_rows = download_nifty_index(date_ranges)
    
    if index_success:
        logger.info("✅ Nifty 50 index downloaded successfully")
        summary['nifty_index_rows'] = index_rows
    else:
        logger.warning("⚠️  Nifty 50 index download failed (will continue with stocks)")
    
    time.sleep(SLEEP_BETWEEN_STOCKS)
    
    # Download all stocks
    logger.info("\n" + "="*60)
    logger.info("STEP 2: DOWNLOAD NIFTY 50 STOCKS")
    logger.info("="*60)
    
    for i, symbol in enumerate(stocks, 1):
        stock_name = symbol_to_name(symbol)

        # Check if we already have the full file
        if (DATA_DIR / stock_name / "full.csv").exists():
            logger.info(f"[{i}/{len(stocks)}] Skipping {stock_name} (Already downloaded) ✅")
            continue
        
        logger.info(f"\n[{i}/{len(stocks)}] Processing {stock_name}...")
        
        success, rows = download_stock(symbol, stock_name, date_ranges)
        
        if success:
            summary['successful'] += 1
            summary['total_rows'] += rows
        else:
            summary['failed'] += 1
            summary['failed_stocks'].append(stock_name)
        
        # Rate limiting between stocks
        if i < len(stocks):
            time.sleep(SLEEP_BETWEEN_STOCKS)
    
    # Print final summary
    logger.info("\n" + "="*60)
    logger.info("DOWNLOAD SUMMARY")
    logger.info("="*60)
    logger.info(f"Total stocks: {summary['total_stocks']}")
    logger.info(f"Successful: {summary['successful']} ✅")
    logger.info(f"Failed: {summary['failed']} ❌")
    logger.info(f"Total data rows: {summary['total_rows']:,}")
    
    if summary['failed_stocks']:
        logger.warning("\nFailed stocks:")
        for stock in summary['failed_stocks']:
            logger.warning(f"  - {stock}")
    
    if index_success:
        logger.info(f"\nNifty 50 index rows: {summary['nifty_index_rows']:,}")
    
    logger.info("\n" + "="*60)
    logger.info("✅ DOWNLOAD COMPLETE")
    logger.info("="*60)
    logger.info(f"Data saved in: {DATA_DIR}")
    logger.info(f"Index saved in: {INDEX_DIR}")
    logger.info(f"Log file: {log_file}")
    
    # Exit code
    if summary['failed'] == 0:
        logger.info("\n🎉 All downloads successful!")
        return 0
    else:
        logger.warning(f"\n⚠️  {summary['failed']} stocks failed to download")
        return 1


if __name__ == "__main__":
    try:
        exit_code = main()
        sys.exit(exit_code)
    except KeyboardInterrupt:
        logger.warning("\n⚠️  Download interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.exception(f"❌ Unexpected error: {e}")
        sys.exit(1)
