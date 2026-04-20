"""
15-Minute ORB + EMA Backtest  —  Databento / MNQ

Ports the Pine Script v5 strategy exactly:
  • 08:30–08:44 ET opening range on 15-min bars
  • EMA filter  (fast / slow crossover)
  • Break-of-range stop-order entry
  • OR-midpoint initial stop, fixed R:R target
  • Break-even stop after 1R in-the-money
  • EOD force-close at 14:55 ET
  • % risk position sizing (compounding)

Outputs
  trade_log_15m.csv              — every trade
  charts/orb_ema_15m_equity.png  — equity curve + drawdown
  charts/orb_ema_15m_monthly.png — monthly P&L bars
  charts/orb_ema_15m_mc.png      — Monte Carlo fan + distribution

Usage
─────
  export DATABENTO_API_KEY=your_key
  python backtest_15m_orb.py
  python backtest_15m_orb.py --symbol NQ --rr 1.5 --risk-pct 0.5 --capital 100000
  python backtest_15m_orb.py --start 2024-01-01 --end 2025-01-01 --force-refresh

Arguments
─────────
  --symbol       Databento symbol or alias: MNQ (default), NQ, ES, MES, or MNQ.c.0
  --start        Start date YYYY-MM-DD  (default: 1 year ago)
  --end          End date   YYYY-MM-DD  (default: today)
  --capital      Starting account size   (default: 50000)
  --risk-pct     % of equity risked per trade  (default: 1.0)
  --rr           Reward:Risk ratio             (default: 2.0)
  --point-val    $ per index point  MNQ=2, NQ=20, ES=50, MES=5  (default: 2)
  --fast-ema     Fast EMA period   (default: 9)
  --slow-ema     Slow EMA period   (default: 20)
  --no-be        Disable break-even stop
  --max-qty      Hard cap on contracts per trade  (default: 50)
  --mc-sims      Monte Carlo paths  (default: 5000)
  --force-refresh  Re-download data even if a cache file exists
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

sys.path.insert(0, os.path.dirname(__file__))

from src.data.databento_fetcher import fetch_and_cache
from src.strategies.orb_ema_15m import (
    ORBEMAConfig,
    generate_trades,
    trades_to_dataframe,
    strategy_summary,
)

os.makedirs("charts", exist_ok=True)

# ── CLI ───────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="15-Min ORB + EMA backtest via Databento")
parser.add_argument("--symbol",       default="MNQ.c.0")
parser.add_argument("--start",        default=None)
parser.add_argument("--end",          default=None)
parser.add_argument("--capital",      type=float, default=50_000.0)
parser.add_argument("--risk-pct",     type=float, default=1.0)
parser.add_argument("--rr",           type=float, default=1.0)
parser.add_argument("--point-val",    type=float, default=2.0)
parser.add_argument("--fast-ema",     type=int,   default=9)
parser.add_argument("--slow-ema",     type=int,   default=20)
parser.add_argument("--bar-mins",     type=int,   default=5,
                    help="Bar size for EMA computation in minutes (default: 5)")
parser.add_argument("--no-be",        action="store_true")
parser.add_argument("--max-qty",      type=int,   default=50)
parser.add_argument("--mc-sims",      type=int,   default=5_000)
parser.add_argument("--force-refresh",action="store_true")
args = parser.parse_args()

today      = datetime.now()
end_date   = args.end   or today.strftime("%Y-%m-%d")
start_date = args.start or (today - timedelta(days=365)).strftime("%Y-%m-%d")

print("=" * 62)
print("  ORB + EMA  —  Databento Backtest")
print(f"  Symbol   : {args.symbol}")
print(f"  Period   : {start_date}  →  {end_date}")
print(f"  Capital  : ${args.capital:>10,.0f}")
print(f"  Risk/trade: {args.risk_pct}%   R:R {args.rr}:1")
print(f"  EMA      : {args.fast_ema}/{args.slow_ema} on {args.bar_mins}-min bars   BE: {'off' if args.no_be else 'on'}")
print(f"  $/point  : {args.point_val}   max contracts: {args.max_qty}")
print("=" * 62)

# ── Fetch data ────────────────────────────────────────────────────────────────
df = fetch_and_cache(
    symbol=args.symbol,
    start=start_date,
    end=end_date,
    bar_minutes=args.bar_mins,
    force_refresh=args.force_refresh,
)
n_days = df.index.normalize().nunique()
print(f"\n  {len(df):,} {args.bar_mins}-min bars  |  {n_days} trading days\n")

# ── Run strategy ──────────────────────────────────────────────────────────────
config = ORBEMAConfig(
    fast_ema        = args.fast_ema,
    slow_ema        = args.slow_ema,
    reward_risk     = args.rr,
    move_to_be      = not args.no_be,
    size_mode       = "% Risk",
    risk_pct        = args.risk_pct,
    point_value     = args.point_val,
    max_contracts   = args.max_qty,
    initial_capital = args.capital,
)

print("Running backtest ...", flush=True)
trades = generate_trades(df, config)

if not trades:
    print("\n  No trades generated.  Check your symbol, date range, and session times.")
    sys.exit(1)

tdf   = trades_to_dataframe(trades)
stats = strategy_summary(trades)

# ── Print summary ─────────────────────────────────────────────────────────────
print(f"\n{'='*62}")
print(f"  {args.symbol}   {start_date} → {end_date}")
print(f"  R:R {args.rr}:1  |  Risk {args.risk_pct}%  |  "
      f"EMA {args.fast_ema}/{args.slow_ema}  |  BE {'on' if not args.no_be else 'off'}")
print(f"{'='*62}")
print(f"  Trades          : {stats['n_trades']}")
print(f"  Win Rate        : {stats['win_rate']:.1%}")
print(f"  Actual R:R      : {stats['actual_rr']:.2f}:1")
print(f"  EV / Trade      : ${stats['ev_per_trade']:+.2f}")
print(f"  Total P&L       : ${stats['total_pnl']:+,.2f}")
print(f"  Profit Factor   : {stats['profit_factor']:.2f}")
print(f"  Max Drawdown    : ${stats['max_drawdown']:,.2f}")
print(f"  Sharpe (ann.)   : {stats['sharpe']:.2f}")
print(f"  Final Equity    : ${stats['final_equity']:,.2f}")
print(f"{'─'*62}")
print(f"  Target exits    : {stats['target_pct']:.1%}")
print(f"  Stop exits      : {stats['stop_pct']:.1%}")
print(f"  BE-stop exits   : {stats['be_stop_pct']:.1%}")
print(f"  EOD exits       : {stats['eod_pct']:.1%}")
print(f"{'='*62}\n")

# ── Save trade log ────────────────────────────────────────────────────────────
tdf.to_csv("trade_log_15m.csv", index=False)
print("  [Saved] trade_log_15m.csv")

# ── Equity curve + drawdown ───────────────────────────────────────────────────
pnls         = tdf["pnl_dollars"].values
equity_curve = args.capital + np.cumsum(pnls)
trade_nums   = np.arange(1, len(equity_curve) + 1)

running_max  = np.maximum.accumulate(equity_curve)
drawdown     = equity_curve - running_max   # negative values

fig, axes = plt.subplots(2, 1, figsize=(13, 8),
                         gridspec_kw={"height_ratios": [3, 1]},
                         sharex=True)

axes[0].plot(trade_nums, equity_curve, color="#1f77b4", linewidth=1.5, zorder=3)
axes[0].axhline(args.capital, color="gray", linestyle="--", linewidth=1, alpha=0.6)
axes[0].fill_between(trade_nums, args.capital, equity_curve,
                     where=equity_curve >= args.capital,
                     alpha=0.18, color="#2ca02c", label="Profit")
axes[0].fill_between(trade_nums, args.capital, equity_curve,
                     where=equity_curve < args.capital,
                     alpha=0.18, color="#d62728", label="Loss")
axes[0].set_ylabel("Account Equity ($)")
axes[0].set_title(
    f"15-Min ORB + EMA  —  {args.symbol}  ({start_date} → {end_date})\n"
    f"R:R {args.rr}:1  |  Risk {args.risk_pct}%/trade  |  EMA {args.fast_ema}/{args.slow_ema}"
    f"  |  {'BE on' if not args.no_be else 'BE off'}",
    fontsize=11,
)
axes[0].legend(fontsize=9)

axes[1].fill_between(trade_nums, drawdown, 0, color="#d62728", alpha=0.55)
axes[1].set_ylabel("Drawdown ($)")
axes[1].set_xlabel("Trade #")

plt.tight_layout()
plt.savefig("charts/orb_ema_15m_equity.png", dpi=150)
print("  [Saved] charts/orb_ema_15m_equity.png")
plt.close()

# ── Monthly P&L bars ──────────────────────────────────────────────────────────
tdf["month"] = pd.to_datetime(tdf["date"]).dt.to_period("M")
monthly      = tdf.groupby("month")["pnl_dollars"].agg(["sum", "count"])
colors       = ["#2ca02c" if v >= 0 else "#d62728" for v in monthly["sum"]]

fig2, ax2 = plt.subplots(figsize=(13, 4))
bars = ax2.bar(monthly.index.astype(str), monthly["sum"],
               color=colors, edgecolor="black", linewidth=0.4)
for bar, (_, row) in zip(bars, monthly.iterrows()):
    ax2.text(bar.get_x() + bar.get_width() / 2,
             bar.get_height() + (abs(monthly["sum"]).max() * 0.01),
             f"n={int(row['count'])}", ha="center", va="bottom", fontsize=7)
ax2.axhline(0, color="black", linewidth=0.8)
ax2.set_xlabel("Month")
ax2.set_ylabel("P&L ($)")
ax2.set_title(f"Monthly P&L  —  15-Min ORB + EMA  ({args.symbol})")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.savefig("charts/orb_ema_15m_monthly.png", dpi=150)
print("  [Saved] charts/orb_ema_15m_monthly.png")
plt.close()

# ── Monte Carlo stress test ───────────────────────────────────────────────────
print(f"\nMonte Carlo stress test  ({args.mc_sims:,} paths × {len(pnls)} trades) ...",
      flush=True)

rng     = np.random.default_rng(42)
n       = len(pnls)
mc_mat  = rng.choice(pnls, size=(args.mc_sims, n), replace=True)
mc_eq   = np.cumsum(mc_mat, axis=1)              # shape (n_sims, n_trades)

final_pnl = mc_eq[:, -1]
max_dds   = np.array([(np.maximum.accumulate(row) - row).max() for row in mc_eq])

pct = lambda arr, p: float(np.percentile(arr, p))

print(f"\n{'='*62}")
print(f"  Monte Carlo Results  ({args.mc_sims:,} paths)")
print(f"{'='*62}")
print(f"  Final P&L  p5  / p50 / p95  : "
      f"${pct(final_pnl,5):+,.0f}  /  ${pct(final_pnl,50):+,.0f}  /  ${pct(final_pnl,95):+,.0f}")
print(f"  Max DD     p50 / p95         : "
      f"${pct(max_dds,50):,.0f}  /  ${pct(max_dds,95):,.0f}")
print(f"  % paths profitable           : {np.mean(final_pnl > 0):.1%}")
print(f"  % paths > 10% return         : {np.mean(final_pnl > args.capital * 0.10):.1%}")
print(f"  % paths > 20% drawdown       : {np.mean(max_dds > args.capital * 0.20):.1%}")
print(f"  % paths > 30% drawdown       : {np.mean(max_dds > args.capital * 0.30):.1%}")
print(f"{'='*62}\n")

# Monte Carlo charts
fig3, (ax_fan, ax_hist) = plt.subplots(1, 2, figsize=(14, 5))

# Fan chart: sample 300 curves + percentile bands
sample_idx = rng.choice(args.mc_sims, size=min(300, args.mc_sims), replace=False)
x          = np.arange(1, n + 1)

for i in sample_idx:
    clr = "#2ca02c" if mc_eq[i, -1] > 0 else "#d62728"
    ax_fan.plot(x, mc_eq[i], color=clr, alpha=0.04, linewidth=0.5)

bands = [(5, 95, "#aec7e8", "p5–p95"), (25, 75, "#1f77b4", "p25–p75")]
for lo_p, hi_p, col, lbl in bands:
    ax_fan.fill_between(x,
                        np.percentile(mc_eq, lo_p, axis=0),
                        np.percentile(mc_eq, hi_p, axis=0),
                        alpha=0.20, color=col, label=lbl)

ax_fan.plot(x, np.median(mc_eq, axis=0),
            color="black", linewidth=2, label="Median", zorder=5)
ax_fan.axhline(0, color="gray", linestyle="--", linewidth=1)
ax_fan.set_xlabel("Trade #")
ax_fan.set_ylabel("Cumulative P&L ($)")
ax_fan.set_title("Monte Carlo Equity Paths")
ax_fan.legend(fontsize=9)

# Final P&L histogram
ax_hist.hist(final_pnl, bins=70, color="#1f77b4",
             edgecolor="white", linewidth=0.3, alpha=0.85)
ax_hist.axvline(0,                   color="red",    linestyle="--", linewidth=1.5,
                label="Break-even")
ax_hist.axvline(np.median(final_pnl), color="#2ca02c", linewidth=2,
                label=f"Median: ${np.median(final_pnl):+,.0f}")
ax_hist.axvline(pct(final_pnl, 5),   color="orange",  linestyle=":", linewidth=1.5,
                label=f"p5: ${pct(final_pnl,5):+,.0f}")
ax_hist.set_xlabel("Final P&L ($)")
ax_hist.set_ylabel("Frequency")
ax_hist.set_title("Final P&L Distribution")
ax_hist.legend(fontsize=9)

plt.suptitle(
    f"Monte Carlo Stress Test  —  15-Min ORB + EMA  ({args.symbol})  "
    f"{start_date} → {end_date}",
    fontsize=11,
)
plt.tight_layout()
plt.savefig("charts/orb_ema_15m_mc.png", dpi=150)
print("  [Saved] charts/orb_ema_15m_mc.png")
plt.close()

print("\nDone.\n")
