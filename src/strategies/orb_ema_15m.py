"""
15-Minute Opening Range Breakout + EMA Filter Strategy.

Faithful Python port of the Pine Script v5 strategy shared by the user.
All parameters, level calculations, and exit logic match the original exactly.

Key design decisions
────────────────────
OR window   : 08:30–08:44 ET  →  single 15-min bar on a 15-min chart
Trade window: 08:45–14:54 ET  →  entry allowed; 14:55 ET forces EOD close
EMA filter  : fast EMA > slow EMA → only longs; < → only shorts
Entry       : stop order at OR high (long) / OR low (short)
Stop        : OR midpoint  (= entry ± halfOR)
Target      : entry ± reward_risk × halfOR
Break-even  : stop moves to entry after price moves 1 × halfOR in our favour
              long  BE trigger : orH + halfOR
              short BE trigger : orL − halfOR
Sizing       : % risk of running equity  (compounding)
              risk_per_contract = halfOR × point_value
              qty = floor(equity × risk_pct% / risk_per_contract)

Within-bar resolution order (conservative, matching TV default):
  1. SL / BE-stop checked against bar Low (long) or High (short)
  2. TP checked only if SL was NOT hit
"""

import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import Optional


# ── Configuration ─────────────────────────────────────────────────────────────

@dataclass
class ORBEMAConfig:
    # Session times (ET strings, 24-h "HH:MM")
    or_start:  str   = "08:30"   # first bar of opening range
    or_end:    str   = "08:45"   # exclusive: trade window begins here
    eod_time:  str   = "14:55"   # force-close at or after this time

    # EMA (applied to Close, adjust=False to match Pine Script)
    fast_ema:  int   = 9
    slow_ema:  int   = 20

    # Entry
    one_per_day: bool = True     # only one trade per calendar day

    # Exit
    reward_risk: float = 1.0     # TP = entry ± reward_risk × halfOR
    move_to_be:  bool  = True    # move stop to BE after 1R in-the-money

    # Sizing
    size_mode:      str   = "% Risk"   # "% Risk" or "Fixed"
    fixed_qty:      int   = 1
    risk_pct:       float = 1.0        # % of equity risked per trade
    point_value:    float = 2.0        # $ per point  (MNQ=2, NQ=20, ES=50, MES=5)
    max_contracts:  int   = 50
    initial_capital: float = 50_000.0


# ── Trade record ──────────────────────────────────────────────────────────────

@dataclass
class Trade:
    date:          pd.Timestamp
    direction:     str           # "long" | "short"
    or_high:       float
    or_low:        float
    or_mid:        float
    half_or:       float
    entry_price:   float
    init_stop:     float         # stop at entry time (= or_mid)
    target_price:  float
    exit_price:    float
    exit_reason:   str           # "target" | "stop" | "be_stop" | "eod"
    qty:           int
    pnl_points:    float
    pnl_dollars:   float
    equity_before: float
    equity_after:  float
    be_triggered:  bool
    win:           bool


# ── Main simulation ───────────────────────────────────────────────────────────

def generate_trades(
    df: pd.DataFrame,
    config: ORBEMAConfig,
) -> list:
    """
    Simulate the 15-min ORB + EMA strategy against a 15-min OHLCV DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        15-min OHLCV bars with a tz-aware DatetimeIndex (US/Eastern).
        Required columns: Open, High, Low, Close, Volume.
    config : ORBEMAConfig

    Returns
    -------
    list[Trade]
    """
    # ── Pre-compute EMAs across the entire history (cumulative, not per-day) ──
    # adjust=False matches Pine Script's ta.ema() which uses the recursive formula:
    #   EMA_t = alpha * close_t + (1 - alpha) * EMA_(t-1),  alpha = 2 / (span + 1)
    df = df.copy()
    df["ema_f"] = df["Close"].ewm(span=config.fast_ema, adjust=False).mean()
    df["ema_s"] = df["Close"].ewm(span=config.slow_ema, adjust=False).mean()

    # Pre-parse time objects once
    or_start_t = pd.Timestamp(f"2000-01-01 {config.or_start}").time()
    or_end_t   = pd.Timestamp(f"2000-01-01 {config.or_end}").time()
    eod_t      = pd.Timestamp(f"2000-01-01 {config.eod_time}").time()

    trades: list = []
    equity = config.initial_capital

    dates = sorted(df.index.normalize().unique())

    for date in dates:
        date_obj = date.date() if hasattr(date, "date") else date
        day_df   = df[df.index.date == date_obj]

        # ── Build Opening Range ───────────────────────────────────────────────
        or_mask = (day_df.index.time >= or_start_t) & (day_df.index.time < or_end_t)
        or_bars = day_df[or_mask]
        if or_bars.empty:
            continue

        or_h    = float(or_bars["High"].max())
        or_l    = float(or_bars["Low"].min())
        or_mid  = (or_h + or_l) / 2.0
        half_or = (or_h - or_l) / 2.0

        if half_or <= 0:
            continue

        # ── Pre-compute all trade levels ──────────────────────────────────────
        # Long
        long_entry    = or_h
        long_sl       = or_mid
        long_tp       = or_h + config.reward_risk * half_or
        long_be_trig  = or_h + half_or          # 1R in the money → arm BE

        # Short
        short_entry   = or_l
        short_sl      = or_mid
        short_tp      = or_l - config.reward_risk * half_or
        short_be_trig = or_l - half_or           # 1R in the money → arm BE

        # ── Trade window: [or_end, eod) ───────────────────────────────────────
        tw_mask    = (day_df.index.time >= or_end_t) & (day_df.index.time < eod_t)
        trade_bars = day_df[tw_mask]
        if trade_bars.empty:
            continue

        # ── EOD bar (used for forced exit price) ──────────────────────────────
        eod_mask = day_df.index.time >= eod_t
        eod_bars = day_df[eod_mask]

        # ── Intraday simulation ───────────────────────────────────────────────
        position      = None     # None | "long" | "short"
        entry_price   = None
        entry_qty     = 0
        current_sl    = None
        be_triggered  = False
        trade_taken   = False    # True once a trade is actually placed (one_per_day)
        # Per-direction breach flags: once OR high (or low) is first crossed,
        # that side is permanently locked regardless of future EMA behaviour.
        # The OTHER side remains open until IT is also breached or a trade fills.
        # This prevents late entries from EMA flips while still allowing a
        # reversal (e.g. short breach skipped → long breach entered later).
        long_breach_seen  = False
        short_breach_seen = False
        eq_before     = equity

        for ts, bar in trade_bars.iterrows():
            high  = float(bar["High"])
            low   = float(bar["Low"])
            ema_f = float(df.at[ts, "ema_f"])
            ema_s = float(df.at[ts, "ema_s"])
            bullish = ema_f > ema_s
            bearish = ema_f < ema_s

            # ── ENTRY ─────────────────────────────────────────────────────────
            if position is None and not trade_taken:
                direction = None

                # Long side: check only if OR high not yet breached
                if not long_breach_seen and high >= long_entry:
                    long_breach_seen = True        # this side is now locked
                    if bullish:
                        direction = "long"

                # Short side: check only if OR low not yet breached.
                # Use separate `if` (not `elif`) so a same-bar double-breach
                # still gives the short a chance when the long EMA check fails.
                if direction is None and not short_breach_seen and low <= short_entry:
                    short_breach_seen = True       # this side is now locked
                    if bearish:
                        direction = "short"

                if direction is None:
                    continue   # neither side breached or neither EMA aligned

                # Position sizing
                risk_per_contract = half_or * config.point_value
                if config.size_mode == "% Risk" and risk_per_contract > 0:
                    dollar_risk = equity * (config.risk_pct / 100.0)
                    qty = min(
                        max(int(dollar_risk / risk_per_contract), 1),
                        config.max_contracts,
                    )
                else:
                    qty = config.fixed_qty

                if direction == "long":
                    position    = "long"
                    entry_price = long_entry
                    current_sl  = long_sl
                else:
                    position    = "short"
                    entry_price = short_entry
                    current_sl  = short_sl

                be_triggered = False
                trade_taken  = True
                entry_qty    = qty
                eq_before    = equity

                # Check same-bar exit (entry and exit in the same 15-min bar)
                if position == "long":
                    result = _same_bar_exit(
                        "long", high, low, entry_price, current_sl, long_tp,
                        long_be_trig, config.move_to_be,
                    )
                else:
                    result = _same_bar_exit(
                        "short", high, low, entry_price, current_sl, short_tp,
                        short_be_trig, config.move_to_be,
                    )

                if result is not None:
                    exit_p, exit_r, be_t = result
                    trade = _build_trade(
                        date, position, or_h, or_l, or_mid, half_or,
                        entry_price, current_sl,
                        long_tp if position == "long" else short_tp,
                        exit_p, exit_r, entry_qty, config.point_value,
                        eq_before, be_t,
                    )
                    equity += trade.pnl_dollars
                    trades.append(trade)
                    position = None
                continue

            # ── MANAGE OPEN POSITION ──────────────────────────────────────────
            if position == "long":
                # Check BE trigger first (may tighten stop for this same bar)
                if config.move_to_be and not be_triggered and high >= long_be_trig:
                    be_triggered = True

                eff_sl = entry_price if (config.move_to_be and be_triggered) else long_sl

                if low <= eff_sl:
                    exit_r = "be_stop" if be_triggered else "stop"
                    trade  = _build_trade(
                        date, "long", or_h, or_l, or_mid, half_or,
                        entry_price, long_sl, long_tp,
                        eff_sl, exit_r, entry_qty, config.point_value,
                        eq_before, be_triggered,
                    )
                    equity += trade.pnl_dollars
                    trades.append(trade)
                    position = None

                elif high >= long_tp:
                    trade = _build_trade(
                        date, "long", or_h, or_l, or_mid, half_or,
                        entry_price, long_sl, long_tp,
                        long_tp, "target", entry_qty, config.point_value,
                        eq_before, be_triggered,
                    )
                    equity += trade.pnl_dollars
                    trades.append(trade)
                    position = None

            elif position == "short":
                if config.move_to_be and not be_triggered and low <= short_be_trig:
                    be_triggered = True

                eff_sl = entry_price if (config.move_to_be and be_triggered) else short_sl

                if high >= eff_sl:
                    exit_r = "be_stop" if be_triggered else "stop"
                    trade  = _build_trade(
                        date, "short", or_h, or_l, or_mid, half_or,
                        entry_price, short_sl, short_tp,
                        eff_sl, exit_r, entry_qty, config.point_value,
                        eq_before, be_triggered,
                    )
                    equity += trade.pnl_dollars
                    trades.append(trade)
                    position = None

                elif low <= short_tp:
                    trade = _build_trade(
                        date, "short", or_h, or_l, or_mid, half_or,
                        entry_price, short_sl, short_tp,
                        short_tp, "target", entry_qty, config.point_value,
                        eq_before, be_triggered,
                    )
                    equity += trade.pnl_dollars
                    trades.append(trade)
                    position = None

        # ── EOD force-close ───────────────────────────────────────────────────
        if position is not None:
            if not eod_bars.empty:
                eod_exit = float(eod_bars.iloc[0]["Open"])
            elif not trade_bars.empty:
                eod_exit = float(trade_bars.iloc[-1]["Close"])
            else:
                eod_exit = entry_price  # shouldn't happen; fallback

            sl_ref = long_sl if position == "long" else short_sl
            tp_ref = long_tp if position == "long" else short_tp

            trade = _build_trade(
                date, position, or_h, or_l, or_mid, half_or,
                entry_price, sl_ref, tp_ref,
                eod_exit, "eod", entry_qty, config.point_value,
                eq_before, be_triggered,
            )
            equity += trade.pnl_dollars
            trades.append(trade)

    return trades


# ── Helpers ───────────────────────────────────────────────────────────────────

def _same_bar_exit(
    direction: str,
    high: float,
    low: float,
    entry: float,
    sl: float,
    tp: float,
    be_trig: float,
    move_to_be: bool,
) -> Optional[tuple]:
    """
    On the entry bar, determine if stop or target was also hit.
    Conservative: SL checked before TP.
    Returns (exit_price, exit_reason, be_triggered) or None if still open.
    """
    be_hit = move_to_be and (
        (direction == "long"  and high >= be_trig) or
        (direction == "short" and low  <= be_trig)
    )
    eff_sl = entry if (move_to_be and be_hit) else sl

    if direction == "long":
        if low <= eff_sl:
            return eff_sl, ("be_stop" if be_hit else "stop"), be_hit
        if high >= tp:
            return tp, "target", be_hit
    else:
        if high >= eff_sl:
            return eff_sl, ("be_stop" if be_hit else "stop"), be_hit
        if low <= tp:
            return tp, "target", be_hit

    return None


def _build_trade(
    date, direction, or_h, or_l, or_mid, half_or,
    entry, init_sl, tp,
    exit_price, exit_reason,
    qty, point_value,
    equity_before, be_triggered,
) -> Trade:
    pnl_pts = (exit_price - entry) if direction == "long" else (entry - exit_price)
    pnl_usd = pnl_pts * qty * point_value
    return Trade(
        date=date,
        direction=direction,
        or_high=or_h,
        or_low=or_l,
        or_mid=or_mid,
        half_or=half_or,
        entry_price=entry,
        init_stop=init_sl,
        target_price=tp,
        exit_price=exit_price,
        exit_reason=exit_reason,
        qty=qty,
        pnl_points=pnl_pts,
        pnl_dollars=pnl_usd,
        equity_before=equity_before,
        equity_after=equity_before + pnl_usd,
        be_triggered=be_triggered,
        win=pnl_usd > 0,
    )


# ── Output helpers ────────────────────────────────────────────────────────────

def trades_to_dataframe(trades: list) -> pd.DataFrame:
    """Convert list of Trade objects to a tidy DataFrame."""
    if not trades:
        return pd.DataFrame()
    return pd.DataFrame([t.__dict__ for t in trades])


def strategy_summary(trades: list) -> dict:
    """Return a summary dict of strategy performance statistics."""
    if not trades:
        return {}

    df    = trades_to_dataframe(trades)
    pnls  = df["pnl_dollars"].values
    wins  = pnls[pnls > 0]
    losses = pnls[pnls < 0]

    equity_curve = np.concatenate([[0.0], np.cumsum(pnls)])
    running_max  = np.maximum.accumulate(equity_curve)
    drawdowns    = running_max - equity_curve
    max_dd       = float(drawdowns.max())

    avg_win  = float(wins.mean())   if len(wins)   else 0.0
    avg_loss = float(losses.mean()) if len(losses) else 0.0
    rr_actual = abs(avg_win / avg_loss) if avg_loss != 0 else float("inf")

    counts = df["exit_reason"].value_counts()

    return {
        "n_trades":      len(df),
        "win_rate":      len(wins) / len(pnls),
        "avg_win":       avg_win,
        "avg_loss":      avg_loss,
        "actual_rr":     rr_actual,
        "ev_per_trade":  float(pnls.mean()),
        "total_pnl":     float(pnls.sum()),
        "profit_factor": wins.sum() / abs(losses.sum()) if losses.sum() != 0 else float("inf"),
        "max_drawdown":  max_dd,
        "sharpe":        (pnls.mean() / pnls.std() * np.sqrt(252)
                          if pnls.std() > 0 else 0.0),
        "target_pct":    counts.get("target",  0) / len(df),
        "stop_pct":      counts.get("stop",    0) / len(df),
        "be_stop_pct":   counts.get("be_stop", 0) / len(df),
        "eod_pct":       counts.get("eod",     0) / len(df),
        "final_equity":  float(df["equity_after"].iloc[-1]),
    }
