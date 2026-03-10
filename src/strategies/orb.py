"""
Opening Range Breakout (ORB) Strategy.

The video identifies ORB as a natural fit for the prop firm payoff structure because:
  - Taking profit at a fraction of the opening range (small TP relative to SL)
    produces a naturally high win rate + low RR geometry
  - Even a very simple ORB has slight positive expected value on trending instruments
  - Deployed at the right RR for the challenge phase, it passes ~50% of the time

Strategy Logic (hourly bars):
  1. Opening Range: the first hourly bar of the trading session (9:30–10:30 ET)
  2. Signal: if any subsequent bar's high > OR high → go LONG
             if any subsequent bar's low  < OR low  → go SHORT
             (first signal wins per day; no re-entry)
  3. Entry:  at the breakout price (OR high or OR low)
  4. Stop:   at the opposite side of the OR (OR low for long, OR high for short)
  5. Target: OR high/low ± (tp_multiplier × OR size)
  6. EOD:    if neither TP nor SL hit by close, exit at last close price
"""

import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from typing import Optional

from src.data.fetcher import session_dates, get_session


@dataclass
class ORBConfig:
    """
    Configuration for the Opening Range Breakout strategy.

    Risk geometry is controlled by tp_multiplier relative to sl_size (1 OR range):
        RR = tp_multiplier / 1.0

    Examples from the video:
        tp_multiplier=0.5  →  RR 0.5:1  (67% zero-EV win rate)
        tp_multiplier=1.0  →  RR 1.0:1  (50% zero-EV win rate)
        tp_multiplier=2.0  →  RR 2.0:1  (33% zero-EV win rate)

    The SL is always the full opening range width (entry to opposite side).
    """
    tp_multiplier: float = 0.5      # TP = tp_multiplier × OR size (from entry)
    risk_dollars: float = 150.0     # Fixed dollar risk per trade (SL hit)
    min_range_pct: float = 0.0010   # Skip days where OR is < 0.10% of price (low vol)
    max_range_pct: float = 0.0300   # Skip days where OR is > 3% of price (high vol)
    direction: str = "both"         # "long", "short", or "both"


@dataclass
class Trade:
    """Record of a single completed trade."""
    date: pd.Timestamp
    direction: str           # "long" or "short"
    or_high: float
    or_low: float
    or_size: float
    entry_price: float
    stop_price: float
    target_price: float
    exit_price: float
    exit_reason: str         # "target", "stop", "eod"
    pnl_points: float        # Raw point P&L
    pnl_dollars: float       # Dollar P&L (sized by risk_dollars / or_size in price)
    win: bool


def _compute_position_size(risk_dollars: float, sl_points: float) -> float:
    """Dollar value per point needed to risk exactly risk_dollars on the SL."""
    if sl_points <= 0:
        return 0.0
    return risk_dollars / sl_points


def generate_trades(
    df: pd.DataFrame,
    config: ORBConfig,
) -> list[Trade]:
    """
    Apply ORB strategy to hourly OHLCV data and return a list of Trade records.

    Args:
        df:      Hourly OHLCV DataFrame with US/Eastern DatetimeIndex
        config:  ORBConfig settings

    Returns:
        List of Trade objects, one per day a trade was taken.
    """
    trades = []
    dates = session_dates(df)

    for date in dates:
        session = get_session(df, date)

        if len(session) < 2:
            # Need at least the opening bar + one more bar to trade
            continue

        # Bar 0 = opening range (9:30–10:30)
        or_bar = session.iloc[0]
        or_high = float(or_bar["High"])
        or_low = float(or_bar["Low"])
        or_size = or_high - or_low
        or_mid = (or_high + or_low) / 2.0

        # Skip degenerate days
        range_pct = or_size / or_mid
        if range_pct < config.min_range_pct or range_pct > config.max_range_pct:
            continue

        # Bars 1+ = potential signal and execution bars
        intraday = session.iloc[1:]

        trade_taken = False
        for _, bar in intraday.iterrows():
            if trade_taken:
                break

            # --- LONG signal: bar breaks above OR high ---
            if config.direction in ("long", "both") and float(bar["High"]) > or_high:
                entry = or_high
                stop = or_low
                target = or_high + config.tp_multiplier * or_size
                sl_points = entry - stop  # = or_size

                dollar_per_point = _compute_position_size(config.risk_dollars, sl_points)
                trade = _simulate_trade(
                    "long", entry, stop, target, dollar_per_point,
                    intraday, bar.name, date, or_high, or_low, or_size,
                )
                trades.append(trade)
                trade_taken = True

            # --- SHORT signal: bar breaks below OR low ---
            elif config.direction in ("short", "both") and float(bar["Low"]) < or_low:
                entry = or_low
                stop = or_high
                target = or_low - config.tp_multiplier * or_size
                sl_points = stop - entry  # = or_size

                dollar_per_point = _compute_position_size(config.risk_dollars, sl_points)
                trade = _simulate_trade(
                    "short", entry, stop, target, dollar_per_point,
                    intraday, bar.name, date, or_high, or_low, or_size,
                )
                trades.append(trade)
                trade_taken = True

    return trades


def _simulate_trade(
    direction: str,
    entry: float,
    stop: float,
    target: float,
    dollar_per_point: float,
    remaining_bars: pd.DataFrame,
    entry_bar_time: pd.Timestamp,
    date: pd.Timestamp,
    or_high: float,
    or_low: float,
    or_size: float,
) -> Trade:
    """
    Simulate a trade from the entry bar onward, checking TP and SL on each bar.
    Conservative fill assumption: TP fills at target, SL fills at stop (no slippage).
    """
    # Start checking from the bar where signal fired and all subsequent bars
    active_bars = remaining_bars.loc[remaining_bars.index >= entry_bar_time]

    exit_price = None
    exit_reason = "eod"

    for _, bar in active_bars.iterrows():
        high = float(bar["High"])
        low = float(bar["Low"])

        if direction == "long":
            # Check SL first (conservative)
            if low <= stop:
                exit_price = stop
                exit_reason = "stop"
                break
            if high >= target:
                exit_price = target
                exit_reason = "target"
                break
        else:  # short
            if high >= stop:
                exit_price = stop
                exit_reason = "stop"
                break
            if low <= target:
                exit_price = target
                exit_reason = "target"
                break

    # EOD exit at last bar's close
    if exit_price is None:
        exit_price = float(active_bars.iloc[-1]["Close"])
        exit_reason = "eod"

    # P&L
    if direction == "long":
        pnl_points = exit_price - entry
    else:
        pnl_points = entry - exit_price

    pnl_dollars = pnl_points * dollar_per_point

    return Trade(
        date=date,
        direction=direction,
        or_high=or_high,
        or_low=or_low,
        or_size=or_size,
        entry_price=entry,
        stop_price=stop,
        target_price=target,
        exit_price=exit_price,
        exit_reason=exit_reason,
        pnl_points=pnl_points,
        pnl_dollars=pnl_dollars,
        win=pnl_dollars > 0,
    )


def trades_to_dataframe(trades: list[Trade]) -> pd.DataFrame:
    """Convert list of Trade objects to a tidy DataFrame."""
    if not trades:
        return pd.DataFrame()
    return pd.DataFrame([t.__dict__ for t in trades])


def strategy_summary(trades: list[Trade]) -> dict:
    """Return a summary dict of basic strategy statistics."""
    if not trades:
        return {}

    df = trades_to_dataframe(trades)
    wins = df[df["win"]]
    losses = df[~df["win"]]

    total_pnl = df["pnl_dollars"].sum()
    win_rate = len(wins) / len(df)

    avg_win = wins["pnl_dollars"].mean() if len(wins) > 0 else 0.0
    avg_loss = losses["pnl_dollars"].mean() if len(losses) > 0 else 0.0
    rr_actual = abs(avg_win / avg_loss) if avg_loss != 0 else float("inf")

    ev_per_trade = df["pnl_dollars"].mean()

    targets = df[df["exit_reason"] == "target"]
    stops = df[df["exit_reason"] == "stop"]
    eods = df[df["exit_reason"] == "eod"]

    return {
        "n_trades": len(df),
        "win_rate": win_rate,
        "actual_rr": rr_actual,
        "ev_per_trade_dollars": ev_per_trade,
        "total_pnl_dollars": total_pnl,
        "avg_win_dollars": avg_win,
        "avg_loss_dollars": avg_loss,
        "target_exits_pct": len(targets) / len(df),
        "stop_exits_pct": len(stops) / len(df),
        "eod_exits_pct": len(eods) / len(df),
        "sharpe_approx": df["pnl_dollars"].mean() / df["pnl_dollars"].std() if df["pnl_dollars"].std() > 0 else 0,
    }
