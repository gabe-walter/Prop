"""
Databento data fetcher for CME Globex futures (MNQ, NQ, ES, MES).

Fetches 1-minute OHLCV bars from GLBX.MDP3 and resamples to 15-minute bars.
Requires the DATABENTO_API_KEY environment variable.

Supported symbols (pass the short name or the full Databento continuous notation):
    MNQ  →  MNQ.c.0   (Micro E-mini NASDAQ-100, front month)
    NQ   →  NQ.c.0    (E-mini NASDAQ-100, front month)
    ES   →  ES.c.0    (E-mini S&P 500, front month)
    MES  →  MES.c.0   (Micro E-mini S&P 500, front month)
"""

import os
import pandas as pd
import pytz
from datetime import datetime, timedelta

try:
    import databento as db
except ImportError:
    raise ImportError("databento is required: pip install 'databento>=0.34'")

EASTERN = pytz.timezone("US/Eastern")

SYMBOL_MAP = {
    "MNQ": "MNQ.c.0",
    "NQ":  "NQ.c.0",
    "ES":  "ES.c.0",
    "MES": "MES.c.0",
}

# CME Globex hours we keep: wide enough to warm up EMAs before the 08:30 open.
# 06:00–15:00 ET gives ~10 pre-market 15-min bars every day for EMA seeding.
_SESSION_START = "06:00"
_SESSION_END   = "15:00"   # inclusive upper bound for between_time


def fetch_15m_bars(
    start: str,
    end: str,
    symbol: str = "MNQ.c.0",
    dataset: str = "GLBX.MDP3",
) -> pd.DataFrame:
    """
    Fetch 15-minute OHLCV bars via Databento by resampling 1-minute source data.

    Args:
        start:   Start date "YYYY-MM-DD" (inclusive)
        end:     End date "YYYY-MM-DD" (exclusive for the API, so pass day+1 if needed)
        symbol:  Databento continuous-contract symbol or short alias (e.g. "MNQ")
        dataset: Databento dataset (default: GLBX.MDP3 for CME Globex)

    Returns:
        DataFrame with tz-aware DatetimeIndex (US/Eastern),
        columns: Open, High, Low, Close, Volume.
        Rows are 15-min bars between 06:00 and 15:00 ET.
    """
    api_key = os.environ.get("DATABENTO_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "DATABENTO_API_KEY environment variable is not set.\n"
            "Set it with: export DATABENTO_API_KEY=your_key"
        )

    resolved = SYMBOL_MAP.get(symbol.upper(), symbol)
    print(f"[databento] Fetching 1-min bars for {resolved}  {start} → {end} ...")

    client = db.Historical(key=api_key)
    store  = client.timeseries.get_range(
        dataset=dataset,
        symbols=[resolved],
        schema="ohlcv-1m",
        start=start,
        end=end,
    )

    df = store.to_df()
    if df.empty:
        raise ValueError(f"No data returned for {resolved} ({start} → {end}). "
                         "Check your symbol, date range, and API key permissions.")

    print(f"[databento] Received {len(df):,} 1-min rows")

    # ── Normalise price scale ─────────────────────────────────────────────────
    # Databento stores CME prices as int64 fixed-point with scale 1e-9.
    # The Python SDK may or may not auto-convert; we detect and fix either way.
    price_cols = [c for c in ("open", "high", "low", "close") if c in df.columns]
    if not price_cols:
        raise ValueError(f"Expected price columns not found. Got: {df.columns.tolist()}")

    sample_median = float(df[price_cols[0]].median())
    if sample_median > 1_000_000:          # raw int64 fixed-point (e.g. 20_000 * 1e9)
        for col in price_cols:
            df[col] = df[col] / 1e9

    # ── Convert index to Eastern ──────────────────────────────────────────────
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    df.index = df.index.tz_convert(EASTERN)
    df.index.name = "Datetime"

    # ── Standardise column names ──────────────────────────────────────────────
    rename_map = {
        "open":   "Open",
        "high":   "High",
        "low":    "Low",
        "close":  "Close",
        "volume": "Volume",
    }
    df = df.rename(columns=rename_map)
    for col in ("Open", "High", "Low", "Close", "Volume"):
        if col not in df.columns:
            raise ValueError(f"Column '{col}' missing after rename. "
                             f"Available: {df.columns.tolist()}")
    df = df[["Open", "High", "Low", "Close", "Volume"]].copy()

    # ── Resample 1m → 15m (left-labelled, left-closed) ───────────────────────
    df_15m = (
        df.resample("15min", label="left", closed="left")
        .agg(Open=("Open", "first"),
             High=("High",  "max"),
             Low= ("Low",   "min"),
             Close=("Close","last"),
             Volume=("Volume","sum"))
        .dropna(subset=["Open"])
    )

    # Drop zero-volume bars (extended-hours gaps, roll gaps, holidays)
    df_15m = df_15m[df_15m["Volume"] > 0]

    # Keep only the session window we care about (EMA warmup + trade hours)
    df_15m = df_15m.between_time(_SESSION_START, _SESSION_END)

    print(f"[databento] {len(df_15m):,} 15-min bars after resample + session filter")
    return df_15m


def fetch_and_cache(
    symbol: str = "MNQ.c.0",
    start: str = None,
    end: str = None,
    cache_path: str = None,
    force_refresh: bool = False,
) -> pd.DataFrame:
    """
    Fetch 15-min bars with local parquet caching to avoid repeated API calls.

    Default window: the 365 calendar days ending today.
    Subsequent calls with the same arguments load from cache unless
    force_refresh=True is passed.
    """
    today     = datetime.now()
    if end is None:
        end   = today.strftime("%Y-%m-%d")
    if start is None:
        start = (today - timedelta(days=365)).strftime("%Y-%m-%d")

    resolved = SYMBOL_MAP.get(symbol.upper(), symbol)

    if cache_path is None:
        safe_sym  = resolved.replace(".", "_").lower()
        cache_path = f"{safe_sym}_15m_{start}_{end}.parquet"

    if os.path.exists(cache_path) and not force_refresh:
        print(f"[databento] Loading cached data from {cache_path}")
        df = pd.read_parquet(cache_path)
        if df.index.tz is None:
            df.index = df.index.tz_localize(EASTERN)
        return df

    df = fetch_15m_bars(start=start, end=end, symbol=resolved)
    df.to_parquet(cache_path)
    print(f"[databento] Cached {len(df):,} rows → {cache_path}")
    return df
