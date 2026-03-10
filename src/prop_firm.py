"""
Prop firm account rules and account state tracking.

Models the Topstep 50K challenge and funded account rules
as described in the YouTube video analysis.
"""

from dataclasses import dataclass, field
from typing import Optional
from enum import Enum


class AccountPhase(Enum):
    CHALLENGE = "challenge"
    FUNDED = "funded"


class AccountStatus(Enum):
    ACTIVE = "active"
    PASSED = "passed"   # Hit profit target (challenge) or taking payout (funded)
    FAILED = "failed"   # Hit drawdown or daily loss limit
    EXPIRED = "expired" # Ran out of trading days


@dataclass
class PropFirmRules:
    """
    Rules governing a prop firm account challenge and funded phase.
    Defaults match Topstep 50K as described in the video.
    """
    # --- Account ---
    account_size: float = 50_000.0

    # --- Challenge phase ---
    challenge_profit_target: float = 3_000.0
    challenge_max_trailing_drawdown: float = 3_000.0  # Trails up with high water mark
    challenge_daily_loss_limit: float = 1_500.0
    challenge_min_trading_days: int = 5
    challenge_max_trading_days: int = 60

    # --- Funded phase ---
    funded_profit_target: Optional[float] = None      # None = no target, just maximize
    funded_max_loss: float = 3_000.0                  # Fixed from starting balance (not trailing)
    funded_daily_loss_limit: float = 1_500.0

    # --- Costs ---
    challenge_fee: float = 49.0
    activation_fee: float = 135.0

    # --- Payouts ---
    payout_split: float = 0.90  # Trader's share of profits


# Pre-built rule sets
TOPSTEP_50K = PropFirmRules(
    account_size=50_000,
    challenge_profit_target=3_000,
    challenge_max_trailing_drawdown=3_000,
    challenge_daily_loss_limit=1_500,
    challenge_min_trading_days=5,
    challenge_max_trading_days=60,
    funded_max_loss=3_000,
    funded_daily_loss_limit=1_500,
    challenge_fee=49,
    activation_fee=135,
    payout_split=0.90,
)


@dataclass
class AccountState:
    """
    Tracks the running state of a prop firm account.

    Usage:
        state = AccountState(rules, phase=AccountPhase.CHALLENGE)
        state.apply_trade(pnl=250.0)
        state.end_of_day()
        if state.status != AccountStatus.ACTIVE:
            # account is done
    """
    rules: PropFirmRules
    phase: AccountPhase = AccountPhase.CHALLENGE

    # Running P&L relative to account start
    cumulative_pnl: float = field(default=0.0, init=False)
    high_water_mark_pnl: float = field(default=0.0, init=False)
    daily_pnl: float = field(default=0.0, init=False)
    trading_days: int = field(default=0, init=False)
    status: AccountStatus = field(default=AccountStatus.ACTIVE, init=False)

    def _trailing_drawdown_from_peak(self) -> float:
        """Current drawdown from the high water mark."""
        return self.high_water_mark_pnl - self.cumulative_pnl

    def _fixed_drawdown_from_start(self) -> float:
        """Drawdown from starting balance (funded phase)."""
        return -self.cumulative_pnl if self.cumulative_pnl < 0 else 0.0

    def apply_trade(self, pnl: float) -> AccountStatus:
        """
        Apply a single trade's P&L and check for account termination.
        Returns current status after applying the trade.
        """
        if self.status != AccountStatus.ACTIVE:
            return self.status

        self.cumulative_pnl += pnl
        self.daily_pnl += pnl
        self.high_water_mark_pnl = max(self.high_water_mark_pnl, self.cumulative_pnl)

        self._check_termination()
        return self.status

    def end_of_day(self) -> AccountStatus:
        """
        Call at the end of each trading day to increment day counter
        and reset daily tracking. Returns current status.
        """
        if self.status != AccountStatus.ACTIVE:
            return self.status

        self.trading_days += 1
        self.daily_pnl = 0.0

        if self.phase == AccountPhase.CHALLENGE:
            if self.trading_days >= self.rules.challenge_max_trading_days:
                self.status = AccountStatus.EXPIRED

        return self.status

    def _check_termination(self):
        if self.phase == AccountPhase.CHALLENGE:
            self._check_challenge_termination()
        else:
            self._check_funded_termination()

    def _check_challenge_termination(self):
        # Profit target (can only pass after min trading days)
        if (self.cumulative_pnl >= self.rules.challenge_profit_target
                and self.trading_days >= self.rules.challenge_min_trading_days):
            self.status = AccountStatus.PASSED
            return

        # Trailing drawdown breach
        if self._trailing_drawdown_from_peak() >= self.rules.challenge_max_trailing_drawdown:
            self.status = AccountStatus.FAILED
            return

        # Daily loss limit
        if self.daily_pnl <= -self.rules.challenge_daily_loss_limit:
            self.status = AccountStatus.FAILED

    def _check_funded_termination(self):
        # Fixed drawdown from starting balance
        if self._fixed_drawdown_from_start() >= self.rules.funded_max_loss:
            self.status = AccountStatus.FAILED
            return

        # Daily loss limit
        if self.daily_pnl <= -self.rules.funded_daily_loss_limit:
            self.status = AccountStatus.FAILED

    @property
    def net_pnl(self) -> float:
        """P&L after payout split (for funded accounts)."""
        if self.phase == AccountPhase.FUNDED and self.cumulative_pnl > 0:
            return self.cumulative_pnl * self.rules.payout_split
        return self.cumulative_pnl
