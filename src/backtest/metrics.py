"""
Performance metrics for strategy evaluation.

Calculates standard trading metrics plus prop-firm-specific metrics
(pass rate, expected value per challenge attempt, etc.)
"""

import numpy as np
import pandas as pd
from typing import List, Dict

from src.strategies.orb import Trade
from src.prop_firm import PropFirmRules


def basic_metrics(trades: List[Trade]) -> Dict:
    """Standard strategy performance metrics."""
    if not trades:
        return {}

    pnls = [t.pnl_dollars for t in trades]
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p < 0]

    equity = np.cumsum([0.0] + pnls)
    running_max = np.maximum.accumulate(equity)
    drawdowns = running_max - equity
    max_dd = drawdowns.max()

    avg_win = np.mean(wins) if wins else 0.0
    avg_loss = np.mean(losses) if losses else 0.0
    win_rate = len(wins) / len(pnls)
    actual_rr = abs(avg_win / avg_loss) if avg_loss != 0 else float("inf")
    ev_per_trade = np.mean(pnls)

    # Approximate Sharpe (daily, assume 252 trading days)
    pnl_series = pd.Series(pnls)
    sharpe = (pnl_series.mean() / pnl_series.std() * np.sqrt(252)) if pnl_series.std() > 0 else 0.0

    # Profit factor
    gross_profit = sum(wins)
    gross_loss = abs(sum(losses))
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")

    return {
        "n_trades": len(pnls),
        "win_rate": win_rate,
        "loss_rate": 1 - win_rate,
        "actual_rr": actual_rr,
        "ev_per_trade": ev_per_trade,
        "total_pnl": sum(pnls),
        "gross_profit": gross_profit,
        "gross_loss": gross_loss,
        "profit_factor": profit_factor,
        "avg_win": avg_win,
        "avg_loss": avg_loss,
        "max_drawdown": max_dd,
        "sharpe_annualized": sharpe,
        "expectancy": win_rate * avg_win + (1 - win_rate) * avg_loss,
    }


def prop_firm_metrics(
    trades: List[Trade],
    rules: PropFirmRules,
    n_bootstrap: int = 5_000,
    payout_target: float = 9_000.0,
    seed: int = 42,
) -> Dict:
    """
    Prop-firm-specific metrics computed via bootstrap simulation.

    Bootstrap sampling lets us estimate pass rate from the real trade
    distribution rather than assuming a theoretical win rate.
    """
    from src.backtest.engine import simulate_n_attempts

    result = simulate_n_attempts(
        trades, rules,
        n_attempts=n_bootstrap,
        payout_target=payout_target,
        seed=seed,
    )

    return {
        "pass_rate_per_attempt": result["pass_rate_per_attempt"],
        "avg_attempts_to_pass": result["avg_attempts_to_pass"],
        "avg_cost_to_funded": result["avg_cost_to_funded"],
        "avg_payout": result["avg_payout"],
        "avg_net_ev": result["avg_net_ev"],
        "pct_net_positive": result["pct_net_positive"],
    }


def rr_sweep_report(
    trades_by_rr: Dict[float, List[Trade]],
    rules: PropFirmRules,
    n_bootstrap: int = 2_000,
) -> pd.DataFrame:
    """
    Generate a comparison table across multiple RR configurations.
    trades_by_rr: {rr_value: [Trade, ...]}
    """
    rows = []
    for rr, trades in trades_by_rr.items():
        bm = basic_metrics(trades)
        from src.backtest.engine import simulate_n_attempts
        sim = simulate_n_attempts(trades, rules, n_attempts=n_bootstrap, seed=42)

        rows.append({
            "TP Multiplier (RR)": rr,
            "Win Rate": f"{bm.get('win_rate', 0):.1%}",
            "Actual RR": f"{bm.get('actual_rr', 0):.2f}",
            "EV/Trade ($)": f"{bm.get('ev_per_trade', 0):.2f}",
            "Pass Rate": f"{sim['pass_rate_per_attempt']:.1%}",
            "Avg Cost to Funded ($)": f"{sim['avg_cost_to_funded']:.0f}",
            "Avg Payout ($)": f"{sim['avg_payout']:.0f}",
            "Net EV ($)": f"{sim['avg_net_ev']:.0f}",
            "% Net Positive": f"{sim['pct_net_positive']:.1%}",
            "_pass_rate_raw": sim["pass_rate_per_attempt"],
            "_net_ev_raw": sim["avg_net_ev"],
        })

    return pd.DataFrame(rows).sort_values("TP Multiplier (RR)")


def print_summary(trades: List[Trade], rules: PropFirmRules, label: str = "Strategy"):
    """Print a formatted summary table."""
    bm = basic_metrics(trades)
    if not bm:
        print("No trades to summarize.")
        return

    print(f"\n{'='*55}")
    print(f"  {label}")
    print(f"{'='*55}")
    print(f"  Trades:           {bm['n_trades']}")
    print(f"  Win Rate:         {bm['win_rate']:.1%}")
    print(f"  Actual RR:        {bm['actual_rr']:.2f}:1")
    print(f"  EV / Trade:       ${bm['ev_per_trade']:+.2f}")
    print(f"  Total P&L:        ${bm['total_pnl']:+,.2f}")
    print(f"  Profit Factor:    {bm['profit_factor']:.2f}")
    print(f"  Max Drawdown:     ${bm['max_drawdown']:,.2f}")
    print(f"  Sharpe (ann.):    {bm['sharpe_annualized']:.2f}")
    print(f"{'='*55}\n")
