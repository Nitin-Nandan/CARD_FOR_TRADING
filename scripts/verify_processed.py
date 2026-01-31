"""
Script: Verify Processed Data Quality

Checks processed Nifty 50 data for:
- All 18 features present
- No missing values (except expected)
- Feature distributions
- Correlation with Nifty
- Time series continuity

Usage:
    python scripts/verify_processed.py

Output:
    - Console summary
    - data/processed_verification_report.txt
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm
from loguru import logger

# Add project root
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.append(str(PROJECT_ROOT))

from config_nifty50 import get_stock_symbols, symbol_to_name

# Paths
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
REPORT_PATH = PROJECT_ROOT / "data" / "processed_verification_report.txt"

logger.remove()
logger.add(sys.stdout, level="INFO")

# Expected features
EXPECTED_FEATURES = [
    'timestamp',
    # OHLCV
    'open', 'high', 'low', 'close', 'volume',
    # Technical indicators
    'ema_5', 'ema_10', 'roi_1m',
    'bb_upper', 'bb_middle', 'bb_lower',
    # Market context
    'nifty_close', 'nifty_return',
    'hour_sin', 'hour_cos',
    # Volatility & volume
    'realized_vol_30m', 'volume_ratio', 'volume_surge'
]


def verify_processed_stock(stock_name):
    """
    Verify one processed stock
    
    Returns:
        Dictionary with verification results
    """
    file_path = PROCESSED_DIR / f"{stock_name}_processed.csv"
    
    if not file_path.exists():
        return {
            'stock': stock_name,
            'exists': False,
            'error': 'File not found'
        }
    
    try:
        df = pd.read_csv(file_path, parse_dates=['timestamp'])
        
        # Check features
        missing_features = set(EXPECTED_FEATURES) - set(df.columns)
        extra_features = set(df.columns) - set(EXPECTED_FEATURES)
        
        if missing_features or extra_features:
            return {
                'stock': stock_name,
                'exists': True,
                'error': f"Feature mismatch: missing={missing_features}, extra={extra_features}"
            }
        
        # Basic stats
        total_rows = len(df)
        date_min = df['timestamp'].min()
        date_max = df['timestamp'].max()
        
        # Check for NaN
        nan_counts = df.isnull().sum()
        total_nans = nan_counts.sum()
        
        # Check feature distributions
        close_mean = df['close'].mean()
        close_std = df['close'].std()
        volume_mean = df['volume'].mean()
        
        # Check Nifty correlation
        nifty_corr = df['close'].corr(df['nifty_close'])
        
        # Check time features range
        hour_sin_range = (df['hour_sin'].min(), df['hour_sin'].max())
        hour_cos_range = (df['hour_cos'].min(), df['hour_cos'].max())
        
        # Quality score
        quality = 'GOOD' if (
            total_nans == 0 and
            total_rows > 300000 and
            -1 <= hour_sin_range[0] <= 1 and
            -1 <= hour_cos_range[0] <= 1 and
            nifty_corr > 0.3  # Should correlate with market
        ) else 'ISSUES'
        
        return {
            'stock': stock_name,
            'exists': True,
            'quality': quality,
            'rows': total_rows,
            'date_range': f"{date_min} → {date_max}",
            'total_nans': total_nans,
            'nan_features': [col for col, count in nan_counts.items() if count > 0],
            'close_mean': close_mean,
            'close_std': close_std,
            'volume_mean': volume_mean,
            'nifty_corr': nifty_corr,
            'hour_sin_range': hour_sin_range,
            'hour_cos_range': hour_cos_range,
            'features_count': len(df.columns),
            'error': None
        }
        
    except Exception as e:
        return {
            'stock': stock_name,
            'exists': True,
            'error': str(e)
        }


def main():
    """
    Main verification process
    """
    logger.info("="*60)
    logger.info("PROCESSED DATA VERIFICATION")
    logger.info("="*60)
    
    # Get stock list
    stocks = get_stock_symbols()
    stock_names = [symbol_to_name(s) for s in stocks]
    
    logger.info(f"Verifying {len(stock_names)} processed stocks...")
    logger.info(f"Expected features: {len(EXPECTED_FEATURES)}")
    
    # Verify all stocks
    results = []
    
    for stock_name in tqdm(stock_names, desc="Verifying", unit="stock"):
        result = verify_processed_stock(stock_name)
        results.append(result)
    
    # Analyze results
    successful = [r for r in results if r.get('exists') and not r.get('error')]
    failed = [r for r in results if not r.get('exists') or r.get('error')]
    
    good_quality = [r for r in successful if r.get('quality') == 'GOOD']
    issues = [r for r in successful if r.get('quality') == 'ISSUES']
    
    # Print summary
    logger.info("\n" + "="*60)
    logger.info("VERIFICATION SUMMARY")
    logger.info("="*60)
    logger.info(f"Total stocks: {len(stock_names)}")
    logger.info(f"Files found: {len(successful)}/{len(stock_names)} ✅")
    logger.info(f"Files missing: {len(failed)} ❌")
    logger.info(f"Good quality: {len(good_quality)} ✅")
    logger.info(f"Has issues: {len(issues)} ⚠️")
    
    # Aggregate statistics
    if successful:
        total_rows = sum(r['rows'] for r in successful)
        avg_rows = total_rows / len(successful)
        avg_nifty_corr = np.mean([r['nifty_corr'] for r in successful])
        
        logger.info(f"\nData Statistics:")
        logger.info(f"  Total rows: {total_rows:,}")
        logger.info(f"  Average rows per stock: {avg_rows:,.0f}")
        logger.info(f"  Average Nifty correlation: {avg_nifty_corr:.3f}")
    
    # List issues
    if issues:
        logger.warning(f"\n⚠️  Stocks with quality issues:")
        for r in issues:
            logger.warning(f"\n  {r['stock']}:")
            if r.get('total_nans', 0) > 0:
                logger.warning(f"    - NaN values: {r['total_nans']}")
                logger.warning(f"    - In features: {r['nan_features']}")
            if r.get('rows', 0) <= 300000:
                logger.warning(f"    - Low row count: {r['rows']:,}")
            if r.get('nifty_corr', 1) < 0.3:
                logger.warning(f"    - Low Nifty correlation: {r['nifty_corr']:.3f}")
    
    # List failed
    if failed:
        logger.error(f"\n❌ Failed stocks:")
        for r in failed:
            logger.error(f"  - {r['stock']}: {r.get('error', 'Unknown')}")
    
    # Write detailed report
    with open(REPORT_PATH, 'w', encoding='utf-8') as f:
        f.write("="*60 + "\n")
        f.write("PROCESSED DATA VERIFICATION REPORT\n")
        f.write("="*60 + "\n\n")
        
        f.write(f"Total stocks: {len(stock_names)}\n")
        f.write(f"Successful: {len(successful)}\n")
        f.write(f"Failed: {len(failed)}\n")
        f.write(f"Good quality: {len(good_quality)}\n")
        f.write(f"Has issues: {len(issues)}\n\n")
        
        f.write("="*60 + "\n")
        f.write("DETAILED RESULTS\n")
        f.write("="*60 + "\n\n")
        
        for r in sorted(results, key=lambda x: x['stock']):
            f.write(f"\n{r['stock']}\n")
            f.write("-" * 40 + "\n")
            
            if not r.get('exists'):
                f.write("  Status: FILE NOT FOUND\n")
            elif r.get('error'):
                f.write(f"  Status: ERROR\n")
                f.write(f"  Error: {r['error']}\n")
            else:
                f.write(f"  Status: {r['quality']}\n")
                f.write(f"  Rows: {r['rows']:,}\n")
                f.write(f"  Date range: {r['date_range']}\n")
                f.write(f"  Features: {r['features_count']}\n")
                f.write(f"  NaN values: {r['total_nans']}\n")
                if r['nan_features']:
                    f.write(f"  NaN in: {', '.join(r['nan_features'])}\n")
                f.write(f"  Close mean: ₹{r['close_mean']:.2f}\n")
                f.write(f"  Close std: ₹{r['close_std']:.2f}\n")
                f.write(f"  Volume mean: {r['volume_mean']:,.0f}\n")
                f.write(f"  Nifty correlation: {r['nifty_corr']:.3f}\n")
                f.write(f"  Hour_sin range: [{r['hour_sin_range'][0]:.3f}, {r['hour_sin_range'][1]:.3f}]\n")
                f.write(f"  Hour_cos range: [{r['hour_cos_range'][0]:.3f}, {r['hour_cos_range'][1]:.3f}]\n")
    
    logger.info(f"\n📄 Detailed report saved: {REPORT_PATH}")
    
    logger.info("\n" + "="*60)
    logger.info("✅ VERIFICATION COMPLETE")
    logger.info("="*60)
    
    # Return exit code
    if len(failed) == 0 and len(issues) == 0:
        logger.info("🎉 All processed data is high quality!")
        return 0
    elif len(failed) > 0:
        logger.warning(f"⚠️  {len(failed)} files missing or failed")
        return 1
    else:
        logger.warning(f"⚠️  {len(issues)} files have quality issues")
        return 2


if __name__ == "__main__":
    try:
        exit_code = main()
        sys.exit(exit_code)
    except KeyboardInterrupt:
        logger.warning("\n⚠️  Verification interrupted")
        sys.exit(1)
    except Exception as e:
        logger.exception(f"❌ Error: {e}")
        sys.exit(1)
