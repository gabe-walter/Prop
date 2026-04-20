"""
ORB + EMA live strategy engine — tick-driven state machine.

This is the brain of the live bot. It receives raw trade ticks and fires
callbacks that the order manager converts to real API calls.

State machine
─────────────
  IDLE
    │  8:30:00 ET first tick
    ▼
  BUILDING_OR          ← tracks min/max of every tick 8:30–8:44:59
    │  8:45:00 ET first tick
    ▼
  WAITING_ENTRY        ← watches for price >= orH (if bullish) or <= orL (if bearish)
    │  first qualifying tick
    ▼
  IN_POSITION          ← monitors BE trigger; bracket orders handle stop/TP server-side
    │  EOD tick / fill confirmation / manual close
    ▼
  EOD_CLOSE / IDLE

Key design choices
──────────────────
• Entry fires on the FIRST tick that crosses the OR boundary (not bar close).
  This is the "maximum precision" the user asked for.
• Exits (stop / take-profit) are placed as server-side bracket orders so they
  execute at the exchange without a round-trip through this process.
• Break-even adjustment IS done from this engine: when we see a tick that
  crosses the BE trigger, we call on_move_stop() which modifies the stop order.
• The engine is thread-safe; the market stream runs on a different thread.
• EMA is updated on each 15-min bar CLOSE (not on every tick) to match the
  Pine Script logic. A separate BarBuilder feeds closed bars.
"""

import logging
import threading
from datetime import datetime, time as dtime
from enum import Enum, auto
from dataclasses import dataclass, field
from typing import Callable, Optional
from zoneinfo import ZoneInfo

log = logging.getLogger(__name__)
EASTERN = ZoneInfo("America/New_York")


# ── Config ────────────────────────────────────────────────────────────────────

@dataclass
class LiveConfig:
    # Session windows (Eastern time)
    or_start:  dtime = dtime(8, 30)
    or_end:    dtime = dtime(8, 45)   # OR locks at this moment
    eod_time:  dtime = dtime(14, 55)  # force-close at or after

    # EMA (applied to 15-min bar closes, adjust=False = Pine Script ta.ema)
    fast_ema:  int   = 9
    slow_ema:  int   = 20

    # Entry
    one_per_day: bool = True

    # Exit
    reward_risk: float = 1.0
    move_to_be:  bool  = True

    # Sizing
    risk_pct:       float = 1.0        # % of equity to risk per trade
    point_value:    float = 2.0        # $ per point  MNQ=2, NQ=20, ES=50, MES=5
    max_contracts:  int   = 50
    initial_equity: float = 50_000.0

    # OR sanity filters (skip degenerate days)
    min_or_points: float = 2.0         # ignore OR narrower than this
    max_or_points: float = 200.0       # ignore OR wider than this (holiday/gap)


# ── State ─────────────────────────────────────────────────────────────────────

class State(Enum):
    IDLE          = auto()
    BUILDING_OR   = auto()
    WAITING_ENTRY = auto()
    IN_POSITION   = auto()
    EOD_CLOSE     = auto()  # terminal for the day


@dataclass
class PositionInfo:
    direction:    str    # "long" | "short"
    entry_price:  float  # expected (may be updated on fill confirmation)
    or_high:      float
    or_low:       float
    or_mid:       float
    half_or:      float
    target:       float
    stop:         float  # current stop price (moves to BE)
    be_trigger:   float  # price at which 1R is in the money
    be_active:    bool   = False
    qty:          int    = 1
    stop_order_id: Optional[str] = None


# ── Engine ────────────────────────────────────────────────────────────────────

class StrategyEngine:
    """
    Call on_tick(price, ts) for every trade tick from the market stream.
    Call update_ema(close_price) when each 15-min bar closes.
    Call seed_ema(closes_list) once at startup with historical bar closes.

    Callbacks (set by the caller before calling start):
      on_enter_long(qty, stop, target)  → place market buy + brackets
      on_enter_short(qty, stop, target) → place market sell + brackets
      on_move_stop(new_stop_price)      → modify existing stop order
      on_eod_close()                    → cancel all orders + flatten position
    """

    def __init__(self, config: LiveConfig):
        self.cfg    = config
        self._lock  = threading.Lock()
        self._state = State.IDLE

        # OR tracking
        self._or_h: Optional[float] = None
        self._or_l: Optional[float] = None

        # EMA state (seeded from history then updated on bar closes)
        self._ema_f: Optional[float] = None
        self._ema_s: Optional[float] = None
        self._alpha_f = 2.0 / (config.fast_ema + 1)
        self._alpha_s = 2.0 / (config.slow_ema + 1)

        # Position
        self._pos: Optional[PositionInfo] = None
        self._traded_today = False
        self._equity = config.initial_equity

        # Day tracking (to detect new day and auto-reset)
        self._last_date: Optional[object] = None

        # Callbacks — wire these before starting
        self.on_enter_long:  Optional[Callable] = None  # (qty, stop, target)
        self.on_enter_short: Optional[Callable] = None  # (qty, stop, target)
        self.on_move_stop:   Optional[Callable] = None  # (new_stop)
        self.on_eod_close:   Optional[Callable] = None  # ()

    # ── Public API ─────────────────────────────────────────────────────────────

    def seed_ema(self, historical_closes: list):
        """
        Warm up EMA from historical 15-min bar closes (oldest → newest).
        Call this once at startup before the market opens.
        Uses the same recursive formula as Pine Script ta.ema().
        """
        with self._lock:
            for c in historical_closes:
                self._apply_ema(float(c))
        log.info(
            f"[Engine] EMA seeded from {len(historical_closes)} bars  "
            f"fast={self._ema_f:.2f}  slow={self._ema_s:.2f}"
        )

    def update_ema(self, bar_close: float):
        """Call when a 15-min bar closes. Updates EMA incrementally."""
        with self._lock:
            self._apply_ema(float(bar_close))

    def update_equity(self, equity: float):
        """Update running equity for position sizing (call after each fill)."""
        with self._lock:
            self._equity = equity

    def notify_fill(self, fill_price: float, stop_order_id: Optional[str] = None):
        """
        Call when the entry order fill is confirmed by the user hub.
        Updates entry price from expected to actual fill price.
        """
        with self._lock:
            if self._pos:
                self._pos.entry_price = fill_price
                if stop_order_id:
                    self._pos.stop_order_id = stop_order_id
                log.info(f"[Engine] Fill confirmed @ {fill_price:.2f}  stop_id={stop_order_id}")

    def notify_exit(self, reason: str = "external"):
        """
        Call when the user hub reports a position was closed externally
        (bracket orders hit, manual close, etc.) so state resets cleanly.
        """
        with self._lock:
            if self._pos:
                log.info(f"[Engine] Position closed externally  reason={reason}")
                self._pos = None
                self._state = State.IDLE

    def on_tick(self, price: float, ts: datetime):
        """
        Hot path — called for every trade tick from MarketStream.
        Thread-safe; returns immediately.
        """
        with self._lock:
            self._process(price, ts)

    def reset_for_new_day(self):
        """Force a full reset (called automatically on first tick of a new day)."""
        with self._lock:
            self._do_reset()

    # ── Internal state machine ────────────────────────────────────────────────

    def _process(self, price: float, ts: datetime):
        ts_et  = ts.astimezone(EASTERN)
        today  = ts_et.date()
        t      = ts_et.time()

        # Auto-reset at day boundary
        if self._last_date is not None and today != self._last_date:
            self._do_reset()
        self._last_date = today

        # ── Transitions ───────────────────────────────────────────────────────
        if self._state == State.IDLE:
            if self.cfg.or_start <= t < self.cfg.or_end:
                self._state = State.BUILDING_OR
                self._or_h = price
                self._or_l = price
                log.info(f"[Engine] OR building  {ts_et:%H:%M:%S ET}")

        elif self._state == State.BUILDING_OR:
            if t < self.cfg.or_end:
                self._or_h = max(self._or_h, price)
                self._or_l = min(self._or_l, price)
            else:
                # OR window closed — lock levels and move to entry watch
                or_range = self._or_h - self._or_l
                if or_range < self.cfg.min_or_points or or_range > self.cfg.max_or_points:
                    log.warning(
                        f"[Engine] OR range {or_range:.2f} pts outside filter "
                        f"[{self.cfg.min_or_points}, {self.cfg.max_or_points}] — skipping day"
                    )
                    self._state = State.EOD_CLOSE
                    return

                log.info(
                    f"[Engine] OR locked  H={self._or_h:.2f}  L={self._or_l:.2f}  "
                    f"range={or_range:.2f}  mid={(self._or_h+self._or_l)/2:.2f}"
                )
                self._state = State.WAITING_ENTRY
                self._check_entry(price, t)   # first tick after OR might already break

        elif self._state == State.WAITING_ENTRY:
            if t >= self.cfg.eod_time:
                log.info("[Engine] EOD reached without entry — standing down")
                self._state = State.EOD_CLOSE
            else:
                self._check_entry(price, t)

        elif self._state == State.IN_POSITION:
            if t >= self.cfg.eod_time:
                self._state = State.EOD_CLOSE
                log.info("[Engine] EOD — requesting force close")
                if self.on_eod_close:
                    self.on_eod_close()
                self._pos = None
            else:
                self._manage(price)

        # State.EOD_CLOSE is terminal until reset_for_new_day()

    def _check_entry(self, price: float, t: dtime):
        if self._traded_today:
            return
        if self._ema_f is None or self._ema_s is None:
            log.warning("[Engine] EMA not seeded — cannot evaluate entry")
            return

        or_h    = self._or_h
        or_l    = self._or_l
        or_mid  = (or_h + or_l) / 2.0
        half_or = (or_h - or_l) / 2.0
        bullish = self._ema_f > self._ema_s
        bearish = self._ema_f < self._ema_s

        if bullish and price >= or_h:
            qty    = self._calc_qty(half_or)
            target = or_h + self.cfg.reward_risk * half_or
            self._pos = PositionInfo(
                direction="long",
                entry_price=or_h,
                or_high=or_h, or_low=or_l, or_mid=or_mid, half_or=half_or,
                target=target, stop=or_mid,
                be_trigger=or_h + half_or,
                qty=qty,
            )
            self._state       = State.IN_POSITION
            self._traded_today = True
            log.info(
                f"[Engine] LONG signal  price={price:.2f}  entry≈{or_h:.2f}  "
                f"stop={or_mid:.2f}  target={target:.2f}  qty={qty}  "
                f"EMA {self._ema_f:.2f}/{self._ema_s:.2f}"
            )
            if self.on_enter_long:
                self.on_enter_long(qty, or_mid, target)

        elif bearish and price <= or_l:
            qty    = self._calc_qty(half_or)
            target = or_l - self.cfg.reward_risk * half_or
            self._pos = PositionInfo(
                direction="short",
                entry_price=or_l,
                or_high=or_h, or_low=or_l, or_mid=or_mid, half_or=half_or,
                target=target, stop=or_mid,
                be_trigger=or_l - half_or,
                qty=qty,
            )
            self._state       = State.IN_POSITION
            self._traded_today = True
            log.info(
                f"[Engine] SHORT signal  price={price:.2f}  entry≈{or_l:.2f}  "
                f"stop={or_mid:.2f}  target={target:.2f}  qty={qty}  "
                f"EMA {self._ema_f:.2f}/{self._ema_s:.2f}"
            )
            if self.on_enter_short:
                self.on_enter_short(qty, or_mid, target)

    def _manage(self, price: float):
        """
        Monitor for BE trigger.
        Stop and target exits are handled server-side by bracket orders —
        we only need to watch for the BE price to adjust the stop order.
        """
        pos = self._pos
        if pos is None:
            return

        if not self.cfg.move_to_be or pos.be_active:
            return

        triggered = (
            (pos.direction == "long"  and price >= pos.be_trigger) or
            (pos.direction == "short" and price <= pos.be_trigger)
        )
        if triggered:
            pos.be_active = True
            pos.stop      = pos.entry_price
            log.info(
                f"[Engine] BE triggered  price={price:.2f}  "
                f"moving stop → entry {pos.entry_price:.2f}"
            )
            if self.on_move_stop:
                self.on_move_stop(pos.entry_price)

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _apply_ema(self, close: float):
        if self._ema_f is None:
            self._ema_f = close
            self._ema_s = close
        else:
            self._ema_f = self._alpha_f * close + (1 - self._alpha_f) * self._ema_f
            self._ema_s = self._alpha_s * close + (1 - self._alpha_s) * self._ema_s

    def _calc_qty(self, half_or: float) -> int:
        risk_per_contract = half_or * self.cfg.point_value
        if risk_per_contract <= 0:
            return 1
        dollar_risk = self._equity * (self.cfg.risk_pct / 100.0)
        qty = int(dollar_risk / risk_per_contract)
        return max(1, min(qty, self.cfg.max_contracts))

    def _do_reset(self):
        self._state        = State.IDLE
        self._or_h         = None
        self._or_l         = None
        self._pos          = None
        self._traded_today = False
        log.info("[Engine] Reset for new trading day")

    # ── Diagnostic ────────────────────────────────────────────────────────────

    def status(self) -> dict:
        with self._lock:
            return {
                "state":       self._state.name,
                "or_high":     self._or_h,
                "or_low":      self._or_l,
                "ema_fast":    round(self._ema_f, 4) if self._ema_f else None,
                "ema_slow":    round(self._ema_s, 4) if self._ema_s else None,
                "bullish":     (self._ema_f or 0) > (self._ema_s or 0),
                "position":    self._pos.direction if self._pos else None,
                "traded_today":self._traded_today,
                "equity":      self._equity,
            }
