"""
Monte Carlo simulation of prop firm challenge equity paths.

Reproduces and extends the analysis from the video:
- Simulates random (zero expected value) strategies with different risk geometries
- Compares pass rates across win rate / risk-reward combinations
- Demonstrates why high win rate + low RR dominates in the barrier problem

Key insight: In a prop firm challenge you have an asymmetric payoff —
losses are capped at the challenge fee, gains are real. Strategies with
lower per-trade variance navigate the upper/lower barriers better.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Dict, List, Tuple

from src.prop_firm import PropFirmRules, AccountState, AccountPhase, AccountStatus, TOPSTEP_50K


@dataclass
class SimStrategy:
    """
    A simplified parameterized strategy for Monte Carlo simulation.
    All strategies here are zero expected value by construction:
        E[trade] = win_rate * (risk * rr) - (1 - win_rate) * risk = 0
    which gives: win_rate = 1 / (1 + rr)

    Position sizing: to compare strategies fairly across different RR values,
    we use equal-variance sizing: risk_per_trade = vol_target / sqrt(rr).

    Derivation: per-trade std = risk * sqrt(rr), so setting risk = vol_target / sqrt(rr)
    gives std = vol_target for all strategies. This matches GBM-based simulations
    where a fixed volatility is used regardless of the win/loss distribution shape.

    Pass vol_target instead of risk_per_trade to use equal-variance sizing.
    """
    label: str
    risk_reward: float       # TP size / SL size (e.g., 0.5 = TP is half the SL)
    risk_per_trade: float = 0.0   # Dollar amount risked (SL hit). Set by vol_target if 0.
    vol_target: float = 0.0       # Target per-trade std. Overrides risk_per_trade if > 0.
    trades_per_day: int = 2

    def __post_init__(self):
        if self.vol_target > 0:
            # Equal-variance sizing: risk = vol_target / sqrt(rr)
            self.risk_per_trade = self.vol_target / (self.risk_reward ** 0.5)
        elif self.risk_per_trade <= 0:
            raise ValueError("Provide either risk_per_trade or vol_target > 0")

    @property
    def win_rate(self) -> float:
        """Win rate that gives exactly zero expected value."""
        return 1.0 / (1.0 + self.risk_reward)

    @property
    def win_pnl(self) -> float:
        return self.risk_per_trade * self.risk_reward

    @property
    def loss_pnl(self) -> float:
        return -self.risk_per_trade

    @property
    def expected_value_per_trade(self) -> float:
        return self.win_rate * self.win_pnl + (1 - self.win_rate) * self.loss_pnl

    @property
    def std_per_trade(self) -> float:
        """Per-trade standard deviation."""
        return (self.win_rate * self.win_pnl**2 + (1 - self.win_rate) * self.loss_pnl**2) ** 0.5


def simulate_challenge(
    strategy: SimStrategy,
    rules: PropFirmRules,
    n_simulations: int = 10_000,
    seed: int = 42,
    capture_curves: bool = False,
) -> Dict:
    """
    Run n_simulations of a challenge attempt with the given strategy and rules.

    Returns a dict with:
        pass_rate, fail_rate, expired_rate
        avg_days_to_pass, avg_days_to_fail
        equity_curves (if capture_curves=True, list of lists)
    """
    rng = np.random.default_rng(seed)
    outcomes = []
    days_to_outcome = []
    equity_curves = [] if capture_curves else None

    for _ in range(n_simulations):
        state = AccountState(rules, phase=AccountPhase.CHALLENGE)
        curve = [0.0] if capture_curves else None

        while state.status == AccountStatus.ACTIVE:
            # Run trades for one day
            for _ in range(strategy.trades_per_day):
                if rng.random() < strategy.win_rate:
                    pnl = strategy.win_pnl
                else:
                    pnl = strategy.loss_pnl

                state.apply_trade(pnl)
                if capture_curves:
                    curve.append(state.cumulative_pnl)

                if state.status != AccountStatus.ACTIVE:
                    break

            state.end_of_day()

        outcomes.append(state.status)
        days_to_outcome.append(state.trading_days)
        if capture_curves:
            equity_curves.append(curve)

    passes = [o for o in outcomes if o == AccountStatus.PASSED]
    fails = [o for o in outcomes if o == AccountStatus.FAILED]
    expired = [o for o in outcomes if o == AccountStatus.EXPIRED]

    pass_indices = [i for i, o in enumerate(outcomes) if o == AccountStatus.PASSED]
    fail_indices = [i for i, o in enumerate(outcomes) if o == AccountStatus.FAILED]

    result = {
        "label": strategy.label,
        "win_rate": strategy.win_rate,
        "risk_reward": strategy.risk_reward,
        "risk_per_trade": strategy.risk_per_trade,
        "ev_per_trade": strategy.expected_value_per_trade,
        "pass_rate": len(passes) / n_simulations,
        "fail_rate": len(fails) / n_simulations,
        "expired_rate": len(expired) / n_simulations,
        "avg_days_to_pass": np.mean([days_to_outcome[i] for i in pass_indices]) if pass_indices else None,
        "avg_days_to_fail": np.mean([days_to_outcome[i] for i in fail_indices]) if fail_indices else None,
        "n_simulations": n_simulations,
    }
    if capture_curves:
        result["equity_curves"] = equity_curves

    return result


def simulate_funded(
    strategy: SimStrategy,
    rules: PropFirmRules,
    n_simulations: int = 10_000,
    max_payout_target: float = 9_000.0,
    seed: int = 99,
) -> Dict:
    """
    Simulate the funded account phase.
    Traders typically exit (take payout) once they hit a comfortable profit.
    We model this as: exit when cumulative P&L >= max_payout_target or account fails.
    """
    rng = np.random.default_rng(seed)
    payouts = []
    n_days_list = []

    for _ in range(n_simulations):
        state = AccountState(rules, phase=AccountPhase.FUNDED)

        while state.status == AccountStatus.ACTIVE:
            for _ in range(strategy.trades_per_day):
                if rng.random() < strategy.win_rate:
                    pnl = strategy.win_pnl
                else:
                    pnl = strategy.loss_pnl

                state.apply_trade(pnl)
                if state.status != AccountStatus.ACTIVE:
                    break

                # Take payout when we hit the target
                if state.cumulative_pnl >= max_payout_target:
                    state.status = AccountStatus.PASSED
                    break

            state.end_of_day()

        payouts.append(max(0.0, state.cumulative_pnl * rules.payout_split))
        n_days_list.append(state.trading_days)

    return {
        "avg_payout": np.mean(payouts),
        "median_payout": np.median(payouts),
        "pct_profitable": np.mean([p > 0 for p in payouts]),
        "payouts": payouts,
        "avg_days": np.mean(n_days_list),
    }


def net_expected_value(
    pass_rate: float,
    avg_payout_per_passed_account: float,
    rules: PropFirmRules,
    max_attempts: int = 20,
) -> Dict:
    """
    Calculate the net expected value per challenge attempt, accounting for:
    - Multiple challenge attempts until passing (geometric trials)
    - Challenge fees for each attempt
    - Activation fee paid once on pass
    - Expected payout on funded account

    Returns avg_total_cost, avg_payout, net_ev.
    """
    # Expected number of attempts to pass (geometric distribution)
    avg_attempts = 1.0 / pass_rate if pass_rate > 0 else float("inf")
    avg_challenge_cost = avg_attempts * rules.challenge_fee
    avg_total_cost = avg_challenge_cost + rules.activation_fee

    net_ev = avg_payout_per_passed_account - avg_total_cost

    return {
        "pass_rate": pass_rate,
        "avg_attempts_to_pass": avg_attempts,
        "avg_challenge_cost": avg_challenge_cost,
        "activation_fee": rules.activation_fee,
        "avg_total_cost": avg_total_cost,
        "avg_payout_per_passed_account": avg_payout_per_passed_account,
        "net_ev": net_ev,
    }


def run_geometry_sweep(
    rules: PropFirmRules = TOPSTEP_50K,
    vol_target: float = 150.0,
    n_simulations: int = 10_000,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Sweep across risk geometries matching the video's table.
    All strategies have zero expected value by construction.

    Uses equal-variance sizing (risk = vol_target / sqrt(rr)) so that all strategies
    have the same per-trade standard deviation. This is the correct apples-to-apples
    comparison and matches the GBM approach used in the video.

    vol_target: target per-trade std in dollars (at 1:1 RR, risk = vol_target)

    Returns a DataFrame with pass rates and stats for each geometry.
    """
    # RR values matching the video: 4:1, 3:1, 2:1, 1:1, 0.5:1, 0.33:1
    geometries = [
        SimStrategy("4:1 RR (20% WR)", risk_reward=4.0, vol_target=vol_target),
        SimStrategy("3:1 RR (25% WR)", risk_reward=3.0, vol_target=vol_target),
        SimStrategy("2:1 RR (33% WR)", risk_reward=2.0, vol_target=vol_target),
        SimStrategy("1:1 RR (50% WR)", risk_reward=1.0, vol_target=vol_target),
        SimStrategy("0.5:1 RR (67% WR)", risk_reward=0.5, vol_target=vol_target),
        SimStrategy("0.33:1 RR (75% WR)", risk_reward=1/3, vol_target=vol_target),
    ]

    rows = []
    for g in geometries:
        result = simulate_challenge(g, rules, n_simulations=n_simulations, seed=seed)
        rows.append({
            "Strategy": g.label,
            "Win Rate": f"{g.win_rate:.1%}",
            "RR": f"{g.risk_reward:.2f}",
            "Risk/Trade ($)": f"{g.risk_per_trade:.0f}",
            "Std/Trade ($)": f"{g.std_per_trade:.0f}",
            "EV/Trade ($)": f"{g.expected_value_per_trade:.2f}",
            "Pass Rate": result["pass_rate"],
            "Pass Rate (%)": f"{result['pass_rate']:.1%}",
            "Avg Days to Pass": result["avg_days_to_pass"],
            "risk_reward_raw": g.risk_reward,
            "win_rate_raw": g.win_rate,
        })

    return pd.DataFrame(rows)


def plot_geometry_sweep(df: pd.DataFrame, save_path: str = None):
    """Bar chart of pass rates across risk geometries."""
    fig, ax = plt.subplots(figsize=(10, 5))

    colors = ["#d62728" if r > 1.0 else "#2ca02c" for r in df["risk_reward_raw"]]
    bars = ax.bar(df["Strategy"], df["Pass Rate"] * 100, color=colors, edgecolor="black", linewidth=0.5)

    for bar, rate in zip(bars, df["Pass Rate"]):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.5,
            f"{rate:.1%}",
            ha="center", va="bottom", fontsize=10, fontweight="bold"
        )

    ax.axhline(12.4, color="orange", linestyle="--", linewidth=1.5, label="Topstep avg (12.4%)")
    ax.set_xlabel("Risk Geometry (zero EV strategies)", fontsize=11)
    ax.set_ylabel("Pass Rate (%)", fontsize=11)
    ax.set_title("Prop Firm Challenge Pass Rate by Risk Geometry\n(Monte Carlo, zero expected value strategies)", fontsize=12)
    ax.legend(fontsize=10)
    ax.set_ylim(0, max(df["Pass Rate"] * 100) * 1.2)
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
    return fig


def plot_equity_curves(
    result: Dict,
    n_curves: int = 50,
    rules: PropFirmRules = TOPSTEP_50K,
    save_path: str = None,
):
    """Plot a sample of equity curves from a simulation result."""
    curves = result.get("equity_curves", [])
    if not curves:
        raise ValueError("equity_curves not captured; re-run with capture_curves=True")

    fig, ax = plt.subplots(figsize=(12, 6))

    sample = curves[:n_curves]
    pass_color, fail_color = "#2ca02c", "#d62728"

    for curve in sample:
        final = curve[-1]
        color = pass_color if final >= rules.challenge_profit_target else fail_color
        ax.plot(curve, color=color, alpha=0.3, linewidth=0.8)

    ax.axhline(rules.challenge_profit_target, color=pass_color, linestyle="--",
               linewidth=2, label=f"Profit Target (+${rules.challenge_profit_target:,.0f})")
    ax.axhline(-rules.challenge_max_trailing_drawdown, color=fail_color, linestyle="--",
               linewidth=2, label=f"Max Drawdown (-${rules.challenge_max_trailing_drawdown:,.0f})")

    ax.set_xlabel("Trade Number", fontsize=11)
    ax.set_ylabel("Cumulative P&L ($)", fontsize=11)
    ax.set_title(
        f"Sample Equity Curves — {result['label']}\n"
        f"Win Rate: {result['win_rate']:.1%} | RR: {result['risk_reward']:.2f} | Pass Rate: {result['pass_rate']:.1%}",
        fontsize=12,
    )
    ax.legend(fontsize=10)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
    return fig
