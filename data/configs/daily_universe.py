"""
Stock Universe: 25 liquid NSE stocks for daily prediction.
Tickers are in yfinance format (symbol + .NS suffix).
"""

# fmt: off
# 25 liquid Nifty 50 blue-chips, selected for data availability and liquidity
_DAILY_SYMBOLS = [
    "RELIANCE.NS",
    "HDFCBANK.NS",
    "ICICIBANK.NS",
    "INFY.NS",
    "TCS.NS",
    "HINDUNILVR.NS",
    "ITC.NS",
    "SBIN.NS",
    "BAJFINANCE.NS",
    "MARUTI.NS",
    "AXISBANK.NS",
    "LT.NS",
    "KOTAKBANK.NS",
    "ASIANPAINT.NS",
    "WIPRO.NS",
    "HCLTECH.NS",
    "TITAN.NS",
    "ULTRACEMCO.NS",
    "POWERGRID.NS",
    "NTPC.NS",
    "TATASTEEL.NS",
    "TATAMOTORS.NS",
    "SUNPHARMA.NS",
    "BHARTIARTL.NS",
    "NESTLEIND.NS",
]
# fmt: on

# Nifty 50 index (benchmark for cross-asset feature)
NIFTY50_SYMBOL = "^NSEI"

# Human-readable names keyed by ticker
_SYMBOL_NAMES = {
    "RELIANCE.NS": "RELIANCE",
    "HDFCBANK.NS": "HDFCBANK",
    "ICICIBANK.NS": "ICICIBANK",
    "INFY.NS": "INFY",
    "TCS.NS": "TCS",
    "HINDUNILVR.NS": "HINDUNILVR",
    "ITC.NS": "ITC",
    "SBIN.NS": "SBIN",
    "BAJFINANCE.NS": "BAJFINANCE",
    "MARUTI.NS": "MARUTI",
    "AXISBANK.NS": "AXISBANK",
    "LT.NS": "LT",
    "KOTAKBANK.NS": "KOTAKBANK",
    "ASIANPAINT.NS": "ASIANPAINT",
    "WIPRO.NS": "WIPRO",
    "HCLTECH.NS": "HCLTECH",
    "TITAN.NS": "TITAN",
    "ULTRACEMCO.NS": "ULTRACEMCO",
    "POWERGRID.NS": "POWERGRID",
    "NTPC.NS": "NTPC",
    "TATASTEEL.NS": "TATASTEEL",
    "TATAMOTORS.NS": "TATAMOTORS",
    "SUNPHARMA.NS": "SUNPHARMA",
    "BHARTIARTL.NS": "BHARTIARTL",
    "NESTLEIND.NS": "NESTLEIND",
}


def get_daily_symbols() -> list[str]:
    """Return the list of 25 NSE yfinance tickers."""
    return list(_DAILY_SYMBOLS)


def ticker_to_name(ticker: str) -> str:
    """Convert yfinance ticker (e.g. 'RELIANCE.NS') to short name ('RELIANCE')."""
    return _SYMBOL_NAMES.get(ticker, ticker.replace(".NS", ""))
