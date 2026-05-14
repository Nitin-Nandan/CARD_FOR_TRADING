"""
Fyers Data Downloader
"""

import time
import pandas as pd
from datetime import datetime, timedelta
from fyers_apiv3 import fyersModel
from loguru import logger
from tqdm import tqdm


class FyersDownloader:
    """
    Handles chunked downloading of historical 1-minute data from Fyers.
    """

    def __init__(self, client_id, access_token, log_path="logs"):
        self.fyers = fyersModel.FyersModel(
            client_id=client_id, token=access_token, log_path=str(log_path)
        )
        self.max_retries = 3
        self.retry_delay = 5

    def generate_date_ranges(self, start_date, end_date, chunk_days=30):
        start = datetime.strptime(start_date, "%Y-%m-%d")
        end = datetime.strptime(end_date, "%Y-%m-%d")
        ranges = []
        curr = start
        while curr < end:
            c_end = min(curr + timedelta(days=chunk_days), end)
            ranges.append((curr.strftime("%Y-%m-%d"), c_end.strftime("%Y-%m-%d")))
            curr = c_end + timedelta(days=1)
        return ranges

    def download_chunk(self, symbol, start, end, retry=0):
        query = {
            "symbol": symbol,
            "resolution": "1",
            "date_format": "1",
            "range_from": start,
            "range_to": end,
            "cont_flag": "1",
        }
        try:
            response = self.fyers.history(query)
            if response.get("s") == "no_data":
                return None
            if response.get("s") != "ok":
                if retry < self.max_retries:
                    time.sleep(self.retry_delay)
                    return self.download_chunk(symbol, start, end, retry + 1)
                return None

            candles = response.get("candles", [])
            if not candles:
                return None

            df = pd.DataFrame(
                candles, columns=["timestamp", "open", "high", "low", "close", "volume"]
            )
            df["timestamp"] = (
                pd.to_datetime(df["timestamp"], unit="s", utc=True)
                .dt.tz_convert("Asia/Kolkata")
                .dt.tz_localize(None)
            )
            return df.sort_values("timestamp")
        except Exception as e:
            logger.error(f"Download error: {e}")
            return None

    def download_symbol(self, symbol, ranges, sleep_secs=1.0):
        all_dfs = []
        pbar = tqdm(ranges, desc=f"Downloading {symbol}", leave=False)
        for start, end in pbar:
            df = self.download_chunk(symbol, start, end)
            if df is not None:
                all_dfs.append(df)
            time.sleep(sleep_secs)

        if not all_dfs:
            return None
        return pd.concat(all_dfs).drop_duplicates("timestamp").sort_values("timestamp")
