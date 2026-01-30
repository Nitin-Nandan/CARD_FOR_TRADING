"""
Script: Verify Downloaded Data Quality

Checks downloaded Nifty 50 data for:
- File existence
- Data completeness
- Gaps in timestamps
- Trading hours correctness
- Missing values
- Price anomalies

Usage:
    python scripts/verify_download.py

Output:
    - Console summary
    - data/verification_report.txt
"""

import sys
from pathlib import Path
from datetime import timedelta

import pandas as pd
import numpy as np
from tqdm import tqdm
from loguru import logger

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.append(str(PROJECT_ROOT))

from config_nifty50 import get_stock_symbols, symbol_to_name

# Paths
DATA_DIR = PROJECT_ROOT / "data" / "raw"
INDEX_DIR = PROJECT_ROOT / "data" / "market_indices"
REPORT_PATH = PROJECT_ROOT / "data" / "verification_report.txt"

logger.remove()
logger.add(sys.stdout, level="INFO")


def verify_file_exists(stock_name):
    """Check if CSV file exists"""
    file_path = DATA_DIR / stock_name / "full.csv"
    return file_path.exists(), file_path


def verify_stock_data(stock_name):
    """
    Comprehensive verification of stock data
    
    Returns:
        Dictionary with verification results
    """
    exists, file_path = verify_file_exists(stock_name)
    
    if not exists:
        return {
            'stock': stock_name,
            'exists': False,
            'error': 'File not found'
        }
    
    try:
        # Load data
        df = pd.read_csv(file_path, parse_dates=["timestamp"])
        
        # Basic stats
        total_rows = len(df)
        date_min = df['timestamp'].min()
        date_max = df['timestamp'].max()
        days_span = (date_max - date_min).days
        
        # Check for missing values
        missing_values = df.isnull().sum().sum()
        
        # Check for zero/negative prices
        invalid_prices = (
            (df['open'] <= 0).sum() +
            (df['high'] <= 0).sum() +
            (df['low'] <= 0).sum() +
            (df['close'] <= 0).sum()
        )
        
        # Check for extreme price changes (>50% in 1 minute)
        df['price_change_pct'] = df['close'].pct_change().abs()
        extreme_changes = (df['price_change_pct'] > 0.5).sum()
        
        # Check trading hours
        df['hour'] = df['timestamp'].dt.hour
        df['minute'] = df['timestamp'].dt.minute
        
        # NSE trading: 9:15 AM - 3:30 PM
        outside_hours = len(df[
            ~(
                ((df['hour'] == 9) & (df['minute'] >= 15)) |
                ((df['hour'] >= 10) & (df['hour'] <= 14)) |
                ((df['hour'] == 15) & (df['minute'] <= 30))
            )
        ])
        
        # Check for gaps (missing minutes within trading day)
        df['time_diff'] = df['timestamp'].diff()
        gaps = df[df['time_diff'] > pd.Timedelta(minutes=2)]
        num_gaps = len(gaps)
        
        # Average daily rows (should be ~375 for full trading day)
        avg_daily_rows = total_rows / days_span if days_span > 0 else 0
        
        return {
            'stock': stock_name,
            'exists': True,
            'total_rows': total_rows,
            'date_range': f"{date_min} → {date_max}",
            'days_span': days_span,
            'missing_values': missing_values,
            'invalid_prices': invalid_prices,
            'extreme_changes': extreme_changes,
            'outside_trading_hours': outside_hours,
            'gaps': num_gaps,
            'avg_daily_rows': avg_daily_rows,
            'quality_score': 'GOOD' if (
                missing_values == 0 and
                invalid_prices == 0 and
                extreme_changes == 0 and
                outside_hours == 0 and
                num_gaps < 50 and
                avg_daily_rows >= 300
            ) else 'ISSUES',
            'error': None
        }
        
    except Exception as e:
        return {
            'stock': stock_name,
            'exists': True,
            'error': str(e)
        }


def verify_index_data():
    """Verify Nifty 50 index data"""
    index_path = INDEX_DIR / "NIFTY50.csv"
    
    if not index_path.exists():
        return {
            'exists': False,
            'error': 'Index file not found'
        }
    
    try:
        df = pd.read_csv(index_path, parse_dates=["timestamp"])
        
        return {
            'exists': True,
            'total_rows': len(df),
            'date_range': f"{df['timestamp'].min()} → {df['timestamp'].max()}",
            'missing_values': df.isnull().sum().sum(),
            'quality_score': 'GOOD' if df.isnull().sum().sum() == 0 else 'ISSUES'
        }
    except Exception as e:
        return {
            'exists': True,
            'error': str(e)
        }


def main():
    """Main verification process"""
    logger.info("="*60)
    logger.info("DATA VERIFICATION")
    logger.info("="*60)
    
    # Get stock list
    stocks = get_stock_symbols()
    stock_names = [symbol_to_name(s) for s in stocks]
    
    logger.info(f"Verifying {len(stock_names)} stocks...")
    
    # Verify all stocks
    results = []
    
    for stock_name in tqdm(stock_names, desc="Verifying", unit="stock"):
        result = verify_stock_data(stock_name)
        results.append(result)
    
    # Verify index
    logger.info("\nVerifying Nifty 50 index...")
    index_result = verify_index_data()
    
    # Analyze results
    successful = [r for r in results if r.get('exists') and not r.get('error')]
    failed = [r for r in results if not r.get('exists') or r.get('error')]
    
    good_quality = [r for r in successful if r.get('quality_score') == 'GOOD']
    issues = [r for r in successful if r.get('quality_score') == 'ISSUES']
    
    # Print summary
    logger.info("\n" + "="*60)
    logger.info("VERIFICATION SUMMARY")
    logger.info("="*60)
    logger.info(f"Total stocks: {len(stock_names)}")
    logger.info(f"Files found: {len(successful)}/{len(stock_names)} ✅")
    logger.info(f"Files missing: {len(failed)} ❌")
    logger.info(f"Good quality: {len(good_quality)} ✅")
    logger.info(f"Has issues: {len(issues)} ⚠️")
    
    if index_result['exists']:
        logger.info(f"\nNifty 50 Index: {index_result['quality_score']}")
        logger.info(f"  Rows: {index_result['total_rows']:,}")
        logger.info(f"  Range: {index_result['date_range']}")
    else:
        logger.warning("\nNifty 50 Index: NOT FOUND ❌")
    
    # Detailed statistics
    if successful:
        total_rows = sum(r['total_rows'] for r in successful)
        avg_rows = total_rows / len(successful)
        
        logger.info(f"\nData Statistics:")
        logger.info(f"  Total rows across all stocks: {total_rows:,}")
        logger.info(f"  Average rows per stock: {avg_rows:,.0f}")
    
    # List issues
    if issues:
        logger.warning(f"\n⚠️  Stocks with data quality issues:")
        for r in issues:
            logger.warning(f"\n  {r['stock']}:")
            if r.get('missing_values', 0) > 0:
                logger.warning(f"    - Missing values: {r['missing_values']}")
            if r.get('invalid_prices', 0) > 0:
                logger.warning(f"    - Invalid prices: {r['invalid_prices']}")
            if r.get('extreme_changes', 0) > 0:
                logger.warning(f"    - Extreme changes: {r['extreme_changes']}")
            if r.get('outside_trading_hours', 0) > 0:
                logger.warning(f"    - Outside trading hours: {r['outside_trading_hours']}")
            if r.get('gaps', 0) > 0:
                logger.warning(f"    - Time gaps: {r['gaps']}")
            if r.get('avg_daily_rows', 0) < 300:
                logger.warning(f"    - Low avg daily rows: {r['avg_daily_rows']:.0f}")
    
    # List failed
    if failed:
        logger.error(f"\n❌ Failed stocks:")
        for r in failed:
            logger.error(f"  - {r['stock']}: {r.get('error', 'File not found')}")
    
    # Write detailed report
    with open(REPORT_PATH, 'w', encoding='utf-8') as f:
        f.write("="*60 + "\n")
        f.write("NIFTY 50 DATA VERIFICATION REPORT\n")
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
                f.write(f"  Status: {r['quality_score']}\n")
                f.write(f"  Rows: {r['total_rows']:,}\n")
                f.write(f"  Date range: {r['date_range']}\n")
                f.write(f"  Days span: {r['days_span']}\n")
                f.write(f"  Avg daily rows: {r['avg_daily_rows']:.0f}\n")
                f.write(f"  Missing values: {r['missing_values']}\n")
                f.write(f"  Invalid prices: {r['invalid_prices']}\n")
                f.write(f"  Extreme changes: {r['extreme_changes']}\n")
                f.write(f"  Outside hours: {r['outside_trading_hours']}\n")
                f.write(f"  Time gaps: {r['gaps']}\n")
    
    logger.info(f"\n📄 Detailed report saved: {REPORT_PATH}")
    
    logger.info("\n" + "="*60)
    logger.info("✅ VERIFICATION COMPLETE")
    logger.info("="*60)
    
    # Return exit code
    if len(failed) == 0 and len(issues) == 0:
        logger.info("🎉 All data files are present and high quality!")
        return 0
    elif len(failed) > 0:
        logger.warning(f"⚠️  {len(failed)} files missing")
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
