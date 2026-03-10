"""
Reproduces the risk geometry analysis from the YouTube video.

Runs a Monte Carlo simulation across 6 risk geometries (all zero expected value)
and prints/plots the pass rates, replicating the video's core finding:

    High win rate + low RR  →  better pass rate in prop firm challenges

Usage:
    python analyze_geometries.py

Output:
    - Console table of pass rates per geometry
    - charts/geometry_pass_rates.png
    - charts/equity_curves_high_rr.png
    - charts/equity_curves_low_rr.png
    - Net expected value analysis at the bottom
"""

import os
import sys
import numpy as np

# Make src importable from project root
sys.path.insert(0, os.path.dirname(__file__))

from src.prop_firm import TOPSTEP_50K
from src.monte_carlo import (
    SimStrategy,
    run_geometry_sweep,
    simulate_challenge,
    simulate_funded,
    net_expected_value,
    plot_geometry_sweep,
    plot_equity_curves,
)

os.makedirs("charts", exist_ok=True)

RULES = TOPSTEP_50K
N_SIMS = 20_000

# vol_target = per-trade standard deviation target (in dollars).
# With equal-variance sizing, risk_per_trade = vol_target / sqrt(rr).
# At 1:1 RR: risk = vol_target = $150, win = $150.
# At 4:1 RR: risk = $75, win = $300.
# At 0.33:1 RR: risk = $260, win = $86.
# All have identical per-trade std so the comparison is apples-to-apples.
# vol_target calibrated so 1:1 RR gives ~37% pass rate (matching video's baseline).
# At vol=400: 4:1 RR → 37%, 0.33:1 RR → 40% (trend correct; discrete sim is subtler than video's GBM)
VOL_TARGET = 400.0

print("=" * 60)
print("  Prop Firm Risk Geometry Analysis")
print(f"  Account: Topstep 50K | ${RULES.challenge_profit_target:,.0f} target")
print(f"  Max drawdown: ${RULES.challenge_max_trailing_drawdown:,.0f}")
print(f"  Daily loss limit: ${RULES.challenge_daily_loss_limit:,.0f}")
print(f"  Vol target: ${VOL_TARGET:.0f}/trade std (equal-variance sizing)")
print(f"  Simulations: {N_SIMS:,}")
print("=" * 60)

# --- Geometry sweep ---
print("\nRunning geometry sweep...", flush=True)
df = run_geometry_sweep(RULES, vol_target=VOL_TARGET, n_simulations=N_SIMS)

print("\n  Pass rates by risk geometry (all strategies: zero EV)\n")
display_cols = ["Strategy", "Win Rate", "RR", "Pass Rate (%)"]
print(df[display_cols].to_string(index=False))

# Plot bar chart
fig = plot_geometry_sweep(df, save_path="charts/geometry_pass_rates.png")
print("\n  [Saved] charts/geometry_pass_rates.png")

# --- Equity curve samples for two extreme geometries ---
high_rr = SimStrategy("4:1 RR (20% WR)", risk_reward=4.0, vol_target=VOL_TARGET)
low_rr = SimStrategy("0.33:1 RR (75% WR)", risk_reward=1/3, vol_target=VOL_TARGET)

print("\nCapturing equity curves for high-RR strategy (4:1)...", flush=True)
high_rr_result = simulate_challenge(high_rr, RULES, n_simulations=500,
                                     seed=42, capture_curves=True)
fig1 = plot_equity_curves(high_rr_result, n_curves=100, rules=RULES,
                           save_path="charts/equity_curves_high_rr.png")
print("  [Saved] charts/equity_curves_high_rr.png")

print("Capturing equity curves for low-RR strategy (0.33:1)...", flush=True)
low_rr_result = simulate_challenge(low_rr, RULES, n_simulations=500,
                                    seed=42, capture_curves=True)
fig2 = plot_equity_curves(low_rr_result, n_curves=100, rules=RULES,
                           save_path="charts/equity_curves_low_rr.png")
print("  [Saved] charts/equity_curves_low_rr.png")

# --- Net expected value analysis ---
print("\n" + "=" * 60)
print("  Net Expected Value Analysis")
print("  (Best geometry: 0.33:1 RR with 75% win rate)")
print("=" * 60)

best_geometry = low_rr
best_pass_rate = low_rr_result["pass_rate"]

# Simulate funded account with slightly better geometry (as video suggests)
funded_strategy = SimStrategy(
    "Funded phase (0.5:1 RR)",
    risk_reward=0.5,
    vol_target=VOL_TARGET * 1.5,  # Slightly larger size in funded phase
)
print("\nSimulating funded account phase...", flush=True)
funded_result = simulate_funded(funded_strategy, RULES, n_simulations=N_SIMS, seed=99)

ev = net_expected_value(
    pass_rate=best_pass_rate,
    avg_payout_per_passed_account=funded_result["avg_payout"],
    rules=RULES,
)

print(f"\n  Challenge pass rate:          {ev['pass_rate']:.1%}")
print(f"  Avg attempts to pass:         {ev['avg_attempts_to_pass']:.1f}")
print(f"  Avg challenge fees paid:      ${ev['avg_challenge_cost']:.2f}")
print(f"  Activation fee:               ${ev['activation_fee']:.2f}")
print(f"  Avg total cost to funded:     ${ev['avg_total_cost']:.2f}")
print(f"  Avg payout (funded):          ${ev['avg_payout_per_passed_account']:.2f}")
print(f"  ─────────────────────────────────")
print(f"  Net expected value:           ${ev['net_ev']:+,.2f}")
print()

if ev["net_ev"] > 0:
    print("  ✓ Positive net EV — the convex payoff structure works in our favor.")
else:
    print("  ✗ Negative net EV — adjust strategy or risk parameters.")

print("\n  Compare to Topstep 2024 avg pass rate: 12.4%")
print(f"  Our simulated pass rate:                {best_pass_rate:.1%}")
if best_pass_rate > 0.124:
    improvement = best_pass_rate / 0.124
    print(f"  That's {improvement:.1f}x better than the average Topstep trader.")

print()
