"""
Market data fetching via yfinance.

For prop firm backtesting we use hourly bars:
  - yfinance provides up to ~730 days of 1h data for US equities
  - SPY is used as a liquid proxy for ES (S&P 500 futures)
  - NQ proxy: QQQ; YM proxy: DIA; RTY proxy: IWM

We work in US/Eastern time so that session boundaries (9:30 AM open) are clean.
"""

import pandas as pd
import pytz
from datetime import datetime, timedelta
from typing import Optional

try:
    import yfinance as yf
except ImportError:
    raise ImportError("yfinance is required: pip install yfinance")


# Mapping from common futures names to liquid ETF proxies
FUTURES_PROXIES = {
    "ES": "SPY",   # S&P 500 E-mini → SPY
    "NQ": "QQQ",   # NASDAQ-100 E-mini → QQQ
    "YM": "DIA",   # Dow E-mini → DIA
    "RTY": "IWM",  # Russell 2000 E-mini → IWM
    "GC": "GLD",   # Gold → GLD
    "CL": "USO",   # Crude Oil → USO
}

EASTERN = pytz.timezone("US/Eastern")
MARKET_OPEN = "09:30"
MARKET_CLOSE = "16:00"


def fetch_hourly(
    ticker: str,
    period: str = "2y",
    start: Optional[str] = None,
    end: Optional[str] = None,
    use_proxy: bool = True,
) -> pd.DataFrame:
    """
    Fetch hourly OHLCV data for a ticker.

    Args:
        ticker:     Ticker symbol (e.g., "SPY", "ES", "NQ")
        period:     yfinance period string (e.g., "2y", "1y", "6mo")
                    Ignored if start/end are provided.
        start:      Start date string "YYYY-MM-DD"
        end:        End date string "YYYY-MM-DD"
        use_proxy:  If True, auto-convert futures names (ES→SPY, etc.)

    Returns:
        DataFrame with DatetimeIndex (US/Eastern), columns: Open High Low Close Volume
        Only includes regular trading hours (9:30–16:00 ET).
    """
    symbol = FUTURES_PROXIES.get(ticker.upper(), ticker) if use_proxy else ticker

    if ticker.upper() != symbol:
        print(f"[fetcher] Using {symbol} as proxy for {ticker.upper()}")

    dl_kwargs = dict(interval="1h", auto_adjust=True, progress=False)
    if start:
        dl_kwargs["start"] = start
        dl_kwargs["end"] = end or datetime.now().strftime("%Y-%m-%d")
    else:
        dl_kwargs["period"] = period

    raw = yf.download(symbol, **dl_kwargs)

    if raw.empty:
        raise ValueError(f"No data returned for {symbol}. Check ticker and period.")

    # yfinance returns tz-aware index in UTC; convert to Eastern
    if raw.index.tz is None:
        raw.index = raw.index.tz_localize("UTC")
    raw.index = raw.index.tz_convert(EASTERN)

    # Flatten MultiIndex columns if present (yfinance ≥0.2.x sometimes returns them)
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)

    # Keep only regular trading hours
    raw = raw.between_time(MARKET_OPEN, MARKET_CLOSE, inclusive="left")

    # Drop rows with zero volume (non-trading days that slipped through)
    raw = raw[raw["Volume"] > 0]

    raw.index.name = "Datetime"
    return raw[["Open", "High", "Low", "Close", "Volume"]].copy()


def session_dates(df: pd.DataFrame) -> list:
    """Return sorted list of unique trading dates in the DataFrame."""
    return sorted(df.index.normalize().unique())


def get_session(df: pd.DataFrame, date: pd.Timestamp) -> pd.DataFrame:
    """Extract all hourly bars for a single trading day."""
    day = date.date() if hasattr(date, "date") else date
    mask = df.index.date == day
    return df[mask]


def fetch_and_cache(
    ticker: str = "SPY",
    cache_path: str = "data_cache.parquet",
    period: str = "2y",
    force_refresh: bool = False,
) -> pd.DataFrame:
    """
    Fetch data and cache it to a parquet file to avoid repeated API calls.
    Subsequent calls load from cache unless force_refresh=True.
    """
    import os

    if os.path.exists(cache_path) and not force_refresh:
        print(f"[fetcher] Loading cached data from {cache_path}")
        df = pd.read_parquet(cache_path)
        # Re-localize if timezone info was lost
        if df.index.tz is None:
            df.index = df.index.tz_localize(EASTERN)
        return df

    print(f"[fetcher] Fetching {period} of hourly {ticker} data from yfinance...")
    df = fetch_hourly(ticker, period=period)
    df.to_parquet(cache_path)
    print(f"[fetcher] Cached {len(df)} rows to {cache_path}")
    return df
