"""
Opening Range Breakout backtest with prop firm rules applied.

Fetches ~2 years of hourly SPY data, runs the ORB strategy across multiple
TP multipliers (RR configurations), and shows:
  1. Strategy performance metrics per RR setting
  2. Pass rate / net EV per RR setting (bootstrapped from real trade P&Ls)
  3. Equity curve plots

The goal is to reproduce the video's claim that even a simple ORB with the
right risk geometry (high win rate / low RR) can achieve ~40-50% pass rate
and positive net expected value on Topstep challenges.

Usage:
    python backtest_orb.py [--rr 0.5] [--ticker SPY] [--risk 150]

Options:
    --rr FLOAT      TP multiplier (default: sweep 0.25, 0.5, 1.0, 2.0)
    --ticker STR    Data ticker (default: SPY)
    --risk FLOAT    Dollar risk per trade (default: 150)
    --no-sweep      Skip the RR sweep, just run with --rr value
"""

import os
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend (works without display)

sys.path.insert(0, os.path.dirname(__file__))

from src.data.fetcher import fetch_and_cache
from src.strategies.orb import ORBConfig, generate_trades, strategy_summary, trades_to_dataframe
from src.backtest.engine import plot_backtest_equity, simulate_n_attempts
from src.backtest.metrics import basic_metrics, print_summary, rr_sweep_report
from src.prop_firm import TOPSTEP_50K

os.makedirs("charts", exist_ok=True)

# --- CLI args ---
parser = argparse.ArgumentParser()
parser.add_argument("--rr", type=float, default=None)
parser.add_argument("--ticker", type=str, default="SPY")
parser.add_argument("--risk", type=float, default=150.0)
parser.add_argument("--no-sweep", action="store_true")
parser.add_argument("--bootstrap", type=int, default=2_000,
                    help="Bootstrap simulations for pass rate (default: 2000)")
args = parser.parse_args()

RULES = TOPSTEP_50K
TICKER = args.ticker
RISK = args.risk

print("=" * 60)
print("  ORB Strategy Backtest — Prop Firm Analysis")
print(f"  Ticker: {TICKER} | Risk/trade: ${RISK:.0f}")
print(f"  Prop firm: Topstep 50K")
print("=" * 60)

# --- Fetch data ---
print(f"\nFetching {TICKER} hourly data...", flush=True)
df = fetch_and_cache(TICKER, cache_path=f"{TICKER.lower()}_hourly_cache.parquet", period="2y")
print(f"  Loaded {len(df)} hourly bars across "
      f"{df.index.normalize().nunique()} trading days")

# --- RR sweep or single run ---
rr_values = [0.25, 0.33, 0.5, 1.0, 2.0] if not args.no_sweep else [args.rr or 0.5]

print(f"\nRunning ORB backtest for TP multipliers: {rr_values}")
print("(This may take a few minutes for bootstrap simulations)\n")

all_trades = {}

for rr in rr_values:
    config = ORBConfig(
        tp_multiplier=rr,
        risk_dollars=RISK,
        direction="both",
    )
    trades = generate_trades(df, config)
    all_trades[rr] = trades

    summary = strategy_summary(trades)
    label = f"ORB tp={rr:.2f}×  (win rate {summary.get('win_rate', 0):.1%}, RR {summary.get('actual_rr', 0):.2f}:1)"
    print_summary(trades, RULES, label=label)

# --- RR comparison table ---
if not args.no_sweep:
    print("\nBuilding RR comparison table (bootstrap pass rates)...", flush=True)
    report = rr_sweep_report(all_trades, RULES, n_bootstrap=args.bootstrap)
    print("\n  ORB Strategy — RR Sweep Results\n")
    # Drop private columns for display
    display_cols = [c for c in report.columns if not c.startswith("_")]
    print(report[display_cols].to_string(index=False))

    # Plot: pass rate vs RR
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    pass_rates = [float(r.strip("%")) / 100 for r in report["Pass Rate"]]
    net_evs = [float(r.replace(",", "").replace("$", "")) for r in report["Net EV ($)"]]

    axes[0].bar(report["TP Multiplier (RR)"].astype(str), [r * 100 for r in pass_rates],
                color=["#2ca02c" if r > 0.3 else "#ff7f0e" for r in pass_rates],
                edgecolor="black", linewidth=0.5)
    axes[0].axhline(12.4, color="red", linestyle="--", linewidth=1.5, label="Topstep avg (12.4%)")
    axes[0].set_xlabel("TP Multiplier")
    axes[0].set_ylabel("Pass Rate (%)")
    axes[0].set_title("Challenge Pass Rate vs. TP Multiplier")
    axes[0].legend()

    ev_colors = ["#2ca02c" if e > 0 else "#d62728" for e in net_evs]
    axes[1].bar(report["TP Multiplier (RR)"].astype(str), net_evs,
                color=ev_colors, edgecolor="black", linewidth=0.5)
    axes[1].axhline(0, color="black", linewidth=1)
    axes[1].set_xlabel("TP Multiplier")
    axes[1].set_ylabel("Net EV ($)")
    axes[1].set_title("Net Expected Value per Cycle vs. TP Multiplier")

    plt.suptitle(f"ORB on {TICKER} — Prop Firm Impact of Risk Geometry", fontsize=13)
    plt.tight_layout()
    plt.savefig("charts/orb_rr_sweep.png", dpi=150)
    print("\n  [Saved] charts/orb_rr_sweep.png")

# --- Equity curve for best single RR ---
best_rr = args.rr or 0.5
best_trades = all_trades.get(best_rr, list(all_trades.values())[0])

fig = plot_backtest_equity(
    best_trades,
    RULES,
    title=f"ORB on {TICKER} (tp_multiplier={best_rr}) — Full Backtest",
    save_path=f"charts/orb_equity_tp{best_rr}.png",
)
print(f"  [Saved] charts/orb_equity_tp{best_rr}.png")

# --- Final bootstrap simulation for best RR ---
print(f"\nBootstrap simulation for tp={best_rr} (n={args.bootstrap})...", flush=True)
sim = simulate_n_attempts(best_trades, RULES, n_attempts=args.bootstrap, seed=42)

print(f"\n  {'─'*45}")
print(f"  Bootstrap Prop Firm Results (tp={best_rr})")
print(f"  {'─'*45}")
print(f"  Pass rate per attempt:    {sim['pass_rate_per_attempt']:.1%}")
print(f"  Avg attempts to pass:     {sim['avg_attempts_to_pass']:.1f}")
print(f"  Avg cost to funded:       ${sim['avg_cost_to_funded']:,.0f}")
print(f"  Avg funded payout:        ${sim['avg_payout']:,.0f}")
print(f"  Avg net EV:               ${sim['avg_net_ev']:+,.0f}")
print(f"  % cycles net positive:    {sim['pct_net_positive']:.1%}")
print(f"  {'─'*45}\n")

# Histogram of net EV distribution
fig, ax = plt.subplots(figsize=(10, 5))
net_evs_dist = sim["net_evs"]
ax.hist(net_evs_dist, bins=50, color="#1f77b4", edgecolor="white", linewidth=0.5, alpha=0.8)
ax.axvline(0, color="red", linestyle="--", linewidth=1.5, label="Break-even")
ax.axvline(np.mean(net_evs_dist), color="#2ca02c", linestyle="-", linewidth=2,
           label=f"Mean: ${np.mean(net_evs_dist):+,.0f}")
ax.set_xlabel("Net EV per Full Cycle ($)", fontsize=11)
ax.set_ylabel("Frequency", fontsize=11)
ax.set_title(f"Distribution of Net EV — ORB tp={best_rr} on {TICKER}\n"
             f"(challenge cost + activation vs. funded payout)", fontsize=12)
ax.legend(fontsize=10)
plt.tight_layout()
plt.savefig(f"charts/orb_net_ev_dist_tp{best_rr}.png", dpi=150)
print(f"  [Saved] charts/orb_net_ev_dist_tp{best_rr}.png")
print("Done.")
