"""
Prop firm backtest engine.

Takes a list of Trade objects (from a strategy) and simulates them
against prop firm rules, tracking account state, pass/fail events,
and expected value across many challenge attempts.

The engine handles:
  - Sequencing trades into days
  - Applying PropFirmRules via AccountState
  - Simulating multiple challenge attempts (Monte Carlo over real trade history)
  - Two-phase simulation: challenge → funded
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import List, Dict, Optional
from itertools import cycle

from src.prop_firm import PropFirmRules, AccountState, AccountPhase, AccountStatus, TOPSTEP_50K
from src.strategies.orb import Trade


def run_challenge(
    trades: List[Trade],
    rules: PropFirmRules,
    start_index: int = 0,
) -> Dict:
    """
    Simulate one challenge attempt using a sequential slice of trades.

    Trades are played in order. When the account hits a terminal state
    (passed/failed/expired) we stop and return the result.

    Args:
        trades:      Ordered list of Trade objects
        rules:       Prop firm rules to apply
        start_index: Index into trades to start from (for chaining attempts)

    Returns:
        dict with status, n_trades_used, final_pnl, equity_curve, end_index
    """
    state = AccountState(rules, phase=AccountPhase.CHALLENGE)
    equity_curve = [0.0]
    current_date = None
    i = start_index

    while i < len(trades) and state.status == AccountStatus.ACTIVE:
        trade = trades[i]

        # Detect day boundary to call end_of_day
        if current_date is not None and trade.date != current_date:
            state.end_of_day()
            if state.status != AccountStatus.ACTIVE:
                break

        current_date = trade.date
        state.apply_trade(trade.pnl_dollars)
        equity_curve.append(state.cumulative_pnl)
        i += 1

    # Close out final day
    if state.status == AccountStatus.ACTIVE:
        state.end_of_day()

    return {
        "status": state.status,
        "n_trades": i - start_index,
        "trading_days": state.trading_days,
        "final_pnl": state.cumulative_pnl,
        "high_water_mark": state.high_water_mark_pnl,
        "equity_curve": equity_curve,
        "end_index": i,
    }


def run_funded(
    trades: List[Trade],
    rules: PropFirmRules,
    start_index: int = 0,
    payout_target: float = 9_000.0,
) -> Dict:
    """
    Simulate the funded account phase using sequential trades.

    Trader exits (takes payout) when cumulative P&L hits payout_target.

    Returns:
        dict with status, payout, equity_curve, end_index
    """
    state = AccountState(rules, phase=AccountPhase.FUNDED)
    equity_curve = [0.0]
    current_date = None
    i = start_index

    while i < len(trades) and state.status == AccountStatus.ACTIVE:
        trade = trades[i]

        if current_date is not None and trade.date != current_date:
            state.end_of_day()
            if state.status != AccountStatus.ACTIVE:
                break

        current_date = trade.date
        state.apply_trade(trade.pnl_dollars)
        equity_curve.append(state.cumulative_pnl)

        if state.cumulative_pnl >= payout_target:
            state.status = AccountStatus.PASSED
            break

        i += 1

    if state.status == AccountStatus.ACTIVE:
        state.end_of_day()

    payout = max(0.0, state.cumulative_pnl) * rules.payout_split

    return {
        "status": state.status,
        "n_trades": i - start_index,
        "trading_days": state.trading_days,
        "final_pnl": state.cumulative_pnl,
        "payout": payout,
        "equity_curve": equity_curve,
        "end_index": i,
    }


def simulate_n_attempts(
    trades: List[Trade],
    rules: PropFirmRules,
    n_attempts: int = 500,
    payout_target: float = 9_000.0,
    seed: int = 42,
) -> Dict:
    """
    Simulate multiple challenge attempts by bootstrapping (sampling with replacement)
    from the historical trade list.

    This answers: "If I keep attempting challenges until I pass and then trade
    the funded account, what is my expected total payout and cost?"

    Returns a summary dict with:
        pass_rate, avg_cost_to_funded, avg_payout, net_ev, etc.
    """
    rng = np.random.default_rng(seed)
    trade_pnls = np.array([t.pnl_dollars for t in trades])
    trade_dates = [t.date for t in trades]

    # Estimate trades per challenge attempt (use ~45 trading days × avg trades/day)
    dates = sorted(set(trade_dates))
    trades_per_day = len(trades) / len(dates) if dates else 2
    trades_per_attempt = int(trades_per_day * rules.challenge_max_trading_days)

    total_costs = []
    payouts = []
    attempts_to_pass = []

    for _ in range(n_attempts):
        # Keep trying challenges until we pass
        challenge_attempts = 0
        passed = False

        while not passed:
            challenge_attempts += 1
            # Bootstrap sample trades for this attempt
            idx = rng.integers(0, len(trade_pnls), size=trades_per_attempt)
            sampled_pnls = trade_pnls[idx]

            # Simulate challenge with these P&Ls
            result = _simulate_challenge_from_pnls(sampled_pnls, rules)
            if result["passed"]:
                passed = True

        # Now simulate funded account
        funded_pnls = trade_pnls[rng.integers(0, len(trade_pnls), size=trades_per_attempt * 3)]
        funded_result = _simulate_funded_from_pnls(funded_pnls, rules, payout_target)

        total_cost = challenge_attempts * rules.challenge_fee + rules.activation_fee
        total_costs.append(total_cost)
        payouts.append(funded_result["payout"])
        attempts_to_pass.append(challenge_attempts)

    net_evs = [p - c for p, c in zip(payouts, total_costs)]

    return {
        "pass_rate_per_attempt": 1.0 / np.mean(attempts_to_pass),
        "avg_attempts_to_pass": np.mean(attempts_to_pass),
        "avg_cost_to_funded": np.mean(total_costs),
        "avg_payout": np.mean(payouts),
        "median_payout": np.median(payouts),
        "avg_net_ev": np.mean(net_evs),
        "pct_net_positive": np.mean([e > 0 for e in net_evs]),
        "net_evs": net_evs,
        "payouts": payouts,
        "total_costs": total_costs,
    }


def _simulate_challenge_from_pnls(pnls: np.ndarray, rules: PropFirmRules) -> Dict:
    """Simulate a challenge from a pre-sampled array of trade P&Ls."""
    cumulative = 0.0
    hwm = 0.0
    daily_pnl = 0.0
    days = 0

    # Assume fixed trades per day (round to nearest int from the array)
    # We'll treat every 2 elements as one "day"
    trades_per_day = 2

    for i, pnl in enumerate(pnls):
        cumulative += pnl
        daily_pnl += pnl
        hwm = max(hwm, cumulative)

        # Check profit target (after min days)
        if cumulative >= rules.challenge_profit_target and days >= rules.challenge_min_trading_days:
            return {"passed": True, "final_pnl": cumulative, "days": days}

        # Check trailing drawdown
        if hwm - cumulative >= rules.challenge_max_trailing_drawdown:
            return {"passed": False, "final_pnl": cumulative, "days": days}

        # Day boundary
        if (i + 1) % trades_per_day == 0:
            if daily_pnl <= -rules.challenge_daily_loss_limit:
                return {"passed": False, "final_pnl": cumulative, "days": days}
            daily_pnl = 0.0
            days += 1
            if days >= rules.challenge_max_trading_days:
                return {"passed": False, "final_pnl": cumulative, "days": days}

    return {"passed": False, "final_pnl": cumulative, "days": days}


def _simulate_funded_from_pnls(
    pnls: np.ndarray,
    rules: PropFirmRules,
    payout_target: float,
) -> Dict:
    """Simulate a funded account from a pre-sampled array of trade P&Ls."""
    cumulative = 0.0
    daily_pnl = 0.0
    start_balance = 0.0  # funded drawdown is from start
    trades_per_day = 2

    for i, pnl in enumerate(pnls):
        cumulative += pnl
        daily_pnl += pnl

        if cumulative >= payout_target:
            return {"payout": cumulative * rules.payout_split, "final_pnl": cumulative}

        if start_balance - cumulative >= rules.funded_max_loss:
            return {"payout": 0.0, "final_pnl": cumulative}

        if (i + 1) % trades_per_day == 0:
            if daily_pnl <= -rules.funded_daily_loss_limit:
                return {"payout": 0.0, "final_pnl": cumulative}
            daily_pnl = 0.0

    payout = max(0.0, cumulative) * rules.payout_split
    return {"payout": payout, "final_pnl": cumulative}


def plot_backtest_equity(
    trades: List[Trade],
    rules: PropFirmRules,
    title: str = "ORB Strategy — Prop Firm Backtest",
    save_path: Optional[str] = None,
):
    """Plot cumulative P&L from a sequential trade list with prop firm barriers."""
    pnls = [t.pnl_dollars for t in trades]
    equity = np.cumsum([0.0] + pnls)

    dates = [t.date for t in trades]
    x = list(range(len(equity)))

    fig, axes = plt.subplots(2, 1, figsize=(14, 8), gridspec_kw={"height_ratios": [3, 1]})

    # Top: equity curve
    ax = axes[0]
    ax.plot(x, equity, color="#1f77b4", linewidth=1.2, label="Cumulative P&L")
    ax.axhline(rules.challenge_profit_target, color="#2ca02c", linestyle="--",
               linewidth=1.5, label=f"Challenge Target (+${rules.challenge_profit_target:,.0f})")
    ax.axhline(-rules.challenge_max_trailing_drawdown, color="#d62728", linestyle="--",
               linewidth=1.5, label=f"Max Drawdown (-${rules.challenge_max_trailing_drawdown:,.0f})")
    ax.fill_between(x, equity, 0, where=[e > 0 for e in equity],
                    alpha=0.15, color="#2ca02c")
    ax.fill_between(x, equity, 0, where=[e < 0 for e in equity],
                    alpha=0.15, color="#d62728")
    ax.set_ylabel("Cumulative P&L ($)", fontsize=11)
    ax.set_title(title, fontsize=13)
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)

    # Bottom: per-trade bar chart
    ax2 = axes[1]
    colors = ["#2ca02c" if p > 0 else "#d62728" for p in pnls]
    ax2.bar(range(len(pnls)), pnls, color=colors, width=0.8, alpha=0.7)
    ax2.axhline(0, color="black", linewidth=0.5)
    ax2.set_xlabel("Trade #", fontsize=11)
    ax2.set_ylabel("Trade P&L ($)", fontsize=11)
    ax2.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    return fig
