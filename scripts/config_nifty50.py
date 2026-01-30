"""
Nifty 50 Stock List (as of January 2026)

This list contains the 50 stocks in the NSE Nifty 50 index.
Used for data download and processing.
"""

NIFTY_50_STOCKS = [
    # --- IT Sector ---
    "NSE:TCS-EQ",
    "NSE:INFY-EQ",
    "NSE:HCLTECH-EQ",
    "NSE:WIPRO-EQ",
    "NSE:TECHM-EQ",

    # --- Banking & Finance ---
    "NSE:HDFCBANK-EQ",
    "NSE:ICICIBANK-EQ",
    "NSE:SBIN-EQ",
    "NSE:KOTAKBANK-EQ",
    "NSE:AXISBANK-EQ",
    "NSE:BAJFINANCE-EQ",
    "NSE:BAJAJFINSV-EQ",
    "NSE:SHRIRAMFIN-EQ",      
    # "NSE:JIOFIN-EQ",        <-- REMOVED (Listed Aug 2023, no data for 2022)
    "NSE:HDFCLIFE-EQ",
    "NSE:SBILIFE-EQ",

    # --- Energy, Oil & Gas ---
    "NSE:RELIANCE-EQ",
    "NSE:NTPC-EQ",
    "NSE:POWERGRID-EQ",
    "NSE:ONGC-EQ",
    "NSE:COALINDIA-EQ",
    "NSE:ADANIENT-EQ",
    "NSE:ADANIPORTS-EQ",
    "NSE:BPCL-EQ",            # <-- ADDED (Classic stock with full history)

    # --- Automobiles ---
    "NSE:MARUTI-EQ",
    "NSE:M&M-EQ",
    "NSE:BAJAJ-AUTO-EQ",
    # "NSE:TATAMOTORS-EQ",      <-- REMOVED BECAUSE OF DEMERGER
    "NSE:HEROMOTOCO-EQ",
    "NSE:EICHERMOT-EQ",

    # --- FMCG & Consumption ---
    "NSE:HINDUNILVR-EQ",
    "NSE:ITC-EQ",
    "NSE:NESTLEIND-EQ",
    "NSE:TATACONSUM-EQ",      
    "NSE:TITAN-EQ",
    "NSE:ASIANPAINT-EQ",
    "NSE:TRENT-EQ",           
    # "NSE:ZOMATO-EQ",          <-- REMOVED BEACUSE FYER'S API DIDN'T RESPOND
    "NSE:BRITANNIA-EQ",

    # --- Pharma & Healthcare ---
    "NSE:SUNPHARMA-EQ",
    "NSE:DRREDDY-EQ",
    "NSE:CIPLA-EQ",
    "NSE:APOLLOHOSP-EQ",
    "NSE:MAXHEALTH-EQ",       # (Listed Aug 2020, Safe for 2022)

    # --- Metals & Mining ---
    "NSE:TATASTEEL-EQ",
    "NSE:HINDALCO-EQ",
    "NSE:JSWSTEEL-EQ",

    # --- Telecom ---
    "NSE:BHARTIARTL-EQ",

    # --- Infrastructure & Construction ---
    "NSE:LT-EQ",              
    "NSE:ULTRACEMCO-EQ",
    "NSE:GRASIM-EQ",

    # --- Defence & Aviation ---
    "NSE:BEL-EQ",             
    "NSE:INDIGO-EQ",          
]

# Nifty 50 Index symbol
NIFTY_INDEX_SYMBOL = "NSE:NIFTY50-INDEX"

# Alternative if above doesn't work
NIFTY_INDEX_ALTERNATIVES = [
    "NSE:NIFTY-INDEX",
    "NSE:NIFTY 50",
]

def get_stock_symbols():
    """
    Returns list of Nifty 50 stock symbols in Fyers format
    """
    return NIFTY_50_STOCKS

def get_index_symbol():
    """
    Returns Nifty 50 index symbol
    """
    return NIFTY_INDEX_SYMBOL

def symbol_to_name(symbol):
    """
    Convert Fyers symbol to stock name
    
    Example: NSE:RELIANCE-EQ -> RELIANCE
    """
    return symbol.replace("NSE:", "").replace("-EQ", "")

def validate_symbol(symbol):
    """
    Validate if symbol is in Nifty 50
    """
    return symbol in NIFTY_50_STOCKS

if __name__ == "__main__":
    print("=" * 60)
    print("NIFTY 50 STOCKS")
    print("=" * 60)
    print(f"Total stocks: {len(NIFTY_50_STOCKS)}")
    print(f"\nFirst 10:")
    for i, stock in enumerate(NIFTY_50_STOCKS[:10], 1):
        print(f"  {i:2d}. {symbol_to_name(stock):20s} ({stock})")
    print(f"\nIndex symbol: {NIFTY_INDEX_SYMBOL}")
