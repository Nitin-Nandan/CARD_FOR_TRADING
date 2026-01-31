"""
Script 03: Create 60-Minute Windows (Stride=1 Dense Sampling)

Creates overlapping 60-minute windows from processed data for training.

Window structure:
- Input: 60 minutes of data (18 features)
- Output: Next 15 minutes of returns and volatility

Dense sampling (stride=1):
- Window 1: [0:59] → predict [60:74]
- Window 2: [1:60] → predict [61:75]
- Window 3: [2:61] → predict [62:76]
- ...

This creates maximum training data (~370k windows per stock).

Usage:
    python pipeline/03_create_windows_60min.py

Output:
    - data/windows/{STOCK}_windows.npz
    - Contains: X_train, y_returns, y_volatility, timestamps
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

# Add project root
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.append(str(PROJECT_ROOT))

from scripts.config_nifty50 import get_stock_symbols, symbol_to_name

# =====================
# CONFIGURATION
# =====================

# Paths
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
WINDOWS_DIR = PROJECT_ROOT / "data" / "windows"
LOG_DIR = PROJECT_ROOT / "logs"

# Create directories
WINDOWS_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

# Setup logging
log_file = LOG_DIR / f"windowing_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
logger.add(log_file, rotation="100 MB", level="INFO")
logger.add(sys.stdout, level="INFO")

# Window parameters
INPUT_LEN = 60      # 60 minutes of input
PRED_LEN = 15       # 15 minutes of prediction
STRIDE = 1          # Dense sampling (every minute)
NUM_FEATURES = 18   # 18 features (timestamp excluded)


# =====================
# WINDOWING FUNCTIONS
# =====================

def create_windows(df, stock_name):
    """
    Create 60-minute windows with dense sampling (stride=1).
    
    Args:
        df: DataFrame with processed data
        stock_name: Name of stock (for logging)
    
    Returns:
        Dictionary with windows data
    """
    logger.info(f"\nCreating windows for {stock_name}...")
    
    # Get feature columns (exclude timestamp)
    feature_cols = [col for col in df.columns if col != 'timestamp']
    
    if len(feature_cols) != NUM_FEATURES:
        logger.warning(f"Expected {NUM_FEATURES} features, got {len(feature_cols)}")
    
    # Convert to numpy for faster indexing
    data = df[feature_cols].values  # (num_rows, 18)
    timestamps = df['timestamp'].values
    
    # Calculate number of windows
    total_rows = len(data)
    max_start_idx = total_rows - INPUT_LEN - PRED_LEN
    
    if max_start_idx < 0:
        logger.error(f"Not enough data! Need {INPUT_LEN + PRED_LEN} rows, got {total_rows}")
        return None
    
    # Number of windows with stride=1
    num_windows = max_start_idx + 1
    
    logger.info(f"  Data shape: {data.shape}")
    logger.info(f"  Creating {num_windows:,} windows (stride={STRIDE})")
    
    # Pre-allocate arrays
    X = np.zeros((num_windows, INPUT_LEN, NUM_FEATURES), dtype=np.float32)
    y_returns = np.zeros((num_windows, PRED_LEN), dtype=np.float32)
    y_volatility = np.zeros((num_windows, PRED_LEN), dtype=np.float32)
    window_timestamps = np.zeros(num_windows, dtype='datetime64[ns]')
    
    # Close price index (for returns and volatility)
    close_idx = feature_cols.index('close')
    
    # Create windows
    for i in tqdm(range(num_windows), desc=f"  {stock_name}", leave=False):
        start_idx = i * STRIDE
        input_end = start_idx + INPUT_LEN
        pred_end = input_end + PRED_LEN
        
        # Input window (60 minutes, 18 features)
        X[i] = data[start_idx:input_end]
        
        # Target returns (next 15 minutes)
        close_prices = data[input_end:pred_end, close_idx]
        last_input_close = data[input_end - 1, close_idx]
        
        # Calculate returns: (price[t] - price[t-1]) / price[t-1]
        returns = (close_prices - last_input_close) / (last_input_close + 1e-8)
        y_returns[i] = returns
        
        # Target volatility (rolling std of returns in next 15 minutes)
        # For each future timestep, calculate volatility as std of returns up to that point
        for t in range(PRED_LEN):
            if t == 0:
                # First timestep: use just that return
                y_volatility[i, t] = abs(returns[t])
            else:
                # Subsequent: rolling std
                y_volatility[i, t] = np.std(returns[:t+1])
        
        # Store timestamp (start of prediction window)
        window_timestamps[i] = timestamps[input_end]
    
    logger.info(f"  Created {num_windows:,} windows successfully")
    logger.info(f"  X shape: {X.shape}")
    logger.info(f"  y_returns shape: {y_returns.shape}")
    logger.info(f"  y_volatility shape: {y_volatility.shape}")
    
    # Return data dict
    return {
        'X': X,
        'y_returns': y_returns,
        'y_volatility': y_volatility,
        'timestamps': window_timestamps,
        'feature_names': feature_cols,
        'stock_name': stock_name
    }


def save_windows(windows_data, stock_name):
    """Save windows to disk as compressed .npz file"""
    
    output_path = WINDOWS_DIR / f"{stock_name}_windows.npz"
    
    np.savez_compressed(
        output_path,
        X=windows_data['X'],
        y_returns=windows_data['y_returns'],
        y_volatility=windows_data['y_volatility'],
        timestamps=windows_data['timestamps'],
        feature_names=windows_data['feature_names'],
        stock_name=windows_data['stock_name']
    )
    
    # Calculate file size
    file_size_mb = output_path.stat().st_size / (1024 * 1024)
    
    logger.info(f"  ✅ Saved: {output_path}")
    logger.info(f"     Size: {file_size_mb:.1f} MB")
    
    return output_path


def process_stock_windows(stock_name):
    """Complete windowing pipeline for one stock"""
    
    logger.info(f"\n{'='*60}")
    logger.info(f"Processing: {stock_name}")
    logger.info(f"{'='*60}")
    
    # Load processed data
    processed_path = PROCESSED_DIR / f"{stock_name}_processed.csv"
    
    if not processed_path.exists():
        logger.error(f"  ❌ Processed file not found: {processed_path}")
        return False
    
    try:
        df = pd.read_csv(processed_path, parse_dates=['timestamp'])
        logger.info(f"  Loaded: {len(df):,} rows")
        
        # Create windows
        windows_data = create_windows(df, stock_name)
        
        if windows_data is None:
            return False
        
        # Save windows
        save_windows(windows_data, stock_name)
        
        return True
        
    except Exception as e:
        logger.exception(f"  ❌ Error: {e}")
        return False


# =====================
# MAIN PROCESSING
# =====================

def main():
    """Main windowing orchestration"""
    
    logger.info("\n" + "="*60)
    logger.info("NIFTY 50 WINDOWING (60-MIN DENSE SAMPLING)")
    logger.info("="*60)
    
    # Get stock list
    stocks = get_stock_symbols()
    stock_names = [symbol_to_name(s) for s in stocks]
    
    logger.info(f"\nTotal stocks: {len(stock_names)}")
    logger.info(f"Window config:")
    logger.info(f"  Input length: {INPUT_LEN} minutes")
    logger.info(f"  Prediction length: {PRED_LEN} minutes")
    logger.info(f"  Stride: {STRIDE} (dense sampling)")
    logger.info(f"  Features: {NUM_FEATURES}")
    
    # Process all stocks
    summary = {
        'total': len(stock_names),
        'successful': 0,
        'failed': 0,
        'total_windows': 0,
        'failed_stocks': []
    }
    
    logger.info("\n" + "="*60)
    logger.info("PROCESSING STOCKS")
    logger.info("="*60)
    
    for stock_name in tqdm(stock_names, desc="Overall progress", unit="stock"):
        success = process_stock_windows(stock_name)
        
        if success:
            summary['successful'] += 1
            
            # Load to count windows
            window_file = WINDOWS_DIR / f"{stock_name}_windows.npz"
            data = np.load(window_file)
            num_windows = len(data['X'])
            summary['total_windows'] += num_windows
            
        else:
            summary['failed'] += 1
            summary['failed_stocks'].append(stock_name)
    
    # Print summary
    logger.info("\n" + "="*60)
    logger.info("WINDOWING SUMMARY")
    logger.info("="*60)
    logger.info(f"Total stocks: {summary['total']}")
    logger.info(f"Successful: {summary['successful']} ✅")
    logger.info(f"Failed: {summary['failed']} ❌")
    logger.info(f"Total windows: {summary['total_windows']:,}")
    
    if summary['successful'] > 0:
        avg_windows = summary['total_windows'] / summary['successful']
        logger.info(f"Average windows per stock: {avg_windows:,.0f}")
    
    if summary['failed_stocks']:
        logger.warning("\nFailed stocks:")
        for stock in summary['failed_stocks']:
            logger.warning(f"  - {stock}")
    
    logger.info(f"\nWindows saved in: {WINDOWS_DIR}")
    logger.info(f"Log file: {log_file}")
    
    logger.info("\n" + "="*60)
    logger.info("✅ WINDOWING COMPLETE")
    logger.info("="*60)
    
    # Exit code
    if summary['failed'] == 0:
        logger.info("\n🎉 All stocks windowed successfully!")
        return 0
    else:
        logger.warning(f"\n⚠️  {summary['failed']} stocks failed")
        return 1


if __name__ == "__main__":
    try:
        exit_code = main()
        sys.exit(exit_code)
    except KeyboardInterrupt:
        logger.warning("\n⚠️  Windowing interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.exception(f"❌ Unexpected error: {e}")
        sys.exit(1)
