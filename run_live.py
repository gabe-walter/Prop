"""
15-Min ORB + EMA  —  Live Trading Bot for TopstepX / ProjectX

Wires together:
  MarketStream   → GatewayTrade tick stream (SignalR WebSocket)
  StrategyEngine → tick-driven ORB+EMA state machine
  OrderManager   → REST order placement + User Hub fill events

Architecture
────────────
  MarketStream thread
      every trade tick → StrategyEngine.on_tick()
                              │
                    on breakout tick:
                    on_enter_long / on_enter_short  ──→  OrderManager.enter_long/short()
                    on_move_stop (BE triggered)     ──→  OrderManager.move_stop()
                    on_eod_close                    ──→  OrderManager.cancel_and_flatten()
                              │
                    UserHub fill event → StrategyEngine.notify_fill()

  15-min bar close (timer thread)
      StrategyEngine.update_ema(close_price)

Precision notes
───────────────
• Entry fires on the first GatewayTrade tick that crosses the OR boundary —
  not on bar close. This is the highest granularity available from the API.
• Stop and take-profit are placed as server-side orders immediately after the
  market entry fills, so they execute at the exchange without round-trips.
• BE stop adjustment happens on the first tick that crosses the 1R trigger.
• The OR window (8:30–8:44 ET) is built from actual trade ticks, not bars.

Requirements
────────────
  pip install signalrcore requests
  pip install git+https://github.com/mceesincus/tsxapi4py.git  (optional, for helpers)

Environment variables (or .env file)
─────────────────────────────────────
  TOPSTEPX_USERNAME   your TopstepX login email
  TOPSTEPX_API_KEY    API key from TopstepX platform settings
  TOPSTEPX_ACCOUNT_ID numeric account ID
  TOPSTEPX_CONTRACT   contract ID, e.g. CON.F.US.MNQ.U25
                      (run with --list-contracts to find yours)

Usage
─────
  # Paper / sim account — recommended first
  python run_live.py --paper

  # Find the right contract ID for MNQ front month
  python run_live.py --list-contracts MNQ

  # Live (real money) — confirm with --confirm-live
  python run_live.py --confirm-live

  # Override any parameter
  python run_live.py --contract CON.F.US.MNQ.U25 --risk-pct 0.5 --rr 1.0 --paper
"""

import argparse
import logging
import os
import signal
import sys
import threading
import time
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

# Load .env if python-dotenv is available
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

sys.path.insert(0, os.path.dirname(__file__))

from src.live.market_stream  import MarketStream
from src.live.strategy_engine import StrategyEngine, LiveConfig
from src.live.order_manager  import OrderManager

EASTERN = ZoneInfo("America/New_York")

# ── Logging setup ─────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("live_bot.log", encoding="utf-8"),
    ],
)
log = logging.getLogger(__name__)

# ── CLI ───────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="15-Min ORB + EMA live trading bot for TopstepX")
parser.add_argument("--username",      default=os.getenv("TOPSTEPX_USERNAME"))
parser.add_argument("--api-key",       default=os.getenv("TOPSTEPX_API_KEY"))
parser.add_argument("--account-id",    type=int, default=int(os.getenv("TOPSTEPX_ACCOUNT_ID") or 0))
parser.add_argument("--contract",      default=os.getenv("TOPSTEPX_CONTRACT", "CON.F.US.MNQ.U25"))
parser.add_argument("--risk-pct",      type=float, default=1.0)
parser.add_argument("--rr",            type=float, default=1.0)
parser.add_argument("--point-val",     type=float, default=2.0,
                    help="$ per index point: MNQ=2, NQ=20, ES=50, MES=5")
parser.add_argument("--fast-ema",      type=int,   default=9)
parser.add_argument("--slow-ema",      type=int,   default=20)
parser.add_argument("--no-be",         action="store_true", help="Disable break-even stop")
parser.add_argument("--capital",       type=float, default=50_000.0,
                    help="Starting capital for position sizing (will update live from API)")
parser.add_argument("--max-contracts", type=int,   default=50)
parser.add_argument("--list-contracts",metavar="TEXT", default=None,
                    help="Search for contracts by text and exit")
parser.add_argument("--paper",         action="store_true",
                    help="Paper/sim mode — log orders but do NOT send to API")
parser.add_argument("--confirm-live",  action="store_true",
                    help="Required flag to trade with real money")
args = parser.parse_args()

# ── Safety gate ───────────────────────────────────────────────────────────────
PAPER_MODE = args.paper or not args.confirm_live

if not PAPER_MODE:
    print("\n" + "="*60)
    print("  ⚠  LIVE TRADING MODE  —  REAL MONEY")
    print("  Contract :", args.contract)
    print("  Account  :", args.account_id)
    print("  Risk/trade:", f"{args.risk_pct}%   R:R {args.rr}:1")
    print("="*60)
    ans = input("  Type YES to continue: ").strip()
    if ans != "YES":
        print("Aborted.")
        sys.exit(0)
else:
    log.info("*** PAPER / SIM MODE — orders will be logged but NOT sent ***")

# ── Validate credentials ──────────────────────────────────────────────────────
if not args.username or not args.api_key:
    log.error(
        "Missing TOPSTEPX_USERNAME or TOPSTEPX_API_KEY.\n"
        "Set them as environment variables or in a .env file."
    )
    sys.exit(1)

if not args.account_id:
    log.error(
        "Missing TOPSTEPX_ACCOUNT_ID.\n"
        "Run with --list-contracts MNQ to find your account/contract info."
    )
    sys.exit(1)

# ── Config ────────────────────────────────────────────────────────────────────
config = LiveConfig(
    fast_ema       = args.fast_ema,
    slow_ema       = args.slow_ema,
    reward_risk    = args.rr,
    move_to_be     = not args.no_be,
    risk_pct       = args.risk_pct,
    point_value    = args.point_val,
    max_contracts  = args.max_contracts,
    initial_equity = args.capital,
)

log.info("=" * 60)
log.info("  15-Min ORB + EMA  —  TopstepX Live Bot")
log.info(f"  Contract  : {args.contract}")
log.info(f"  Account   : {args.account_id}")
log.info(f"  R:R       : {args.rr}:1   Risk: {args.risk_pct}%/trade")
log.info(f"  EMA       : {args.fast_ema}/{args.slow_ema}   BE: {'on' if not args.no_be else 'off'}")
log.info(f"  Mode      : {'PAPER' if PAPER_MODE else 'LIVE'}")
log.info("=" * 60)

# ── Build components ──────────────────────────────────────────────────────────
engine = StrategyEngine(config)
mgr    = OrderManager(
    username    = args.username,
    api_key     = args.api_key,
    account_id  = args.account_id,
    contract_id = args.contract,
    point_value = args.point_val,
    on_fill     = engine.notify_fill,
    on_position_closed = lambda reason: engine.notify_exit(reason),
)

# ── Paper mode wrappers ───────────────────────────────────────────────────────
def _enter_long(qty, stop, target):
    log.info(f"[BOT] ENTER LONG   qty={qty}  stop={stop:.2f}  target={target:.2f}")
    if not PAPER_MODE:
        try:
            mgr.enter_long(qty, stop, target)
        except Exception as e:
            log.error(f"[BOT] enter_long failed: {e}")

def _enter_short(qty, stop, target):
    log.info(f"[BOT] ENTER SHORT  qty={qty}  stop={stop:.2f}  target={target:.2f}")
    if not PAPER_MODE:
        try:
            mgr.enter_short(qty, stop, target)
        except Exception as e:
            log.error(f"[BOT] enter_short failed: {e}")

def _move_stop(new_stop):
    log.info(f"[BOT] MOVE STOP → {new_stop:.2f}")
    if not PAPER_MODE:
        try:
            mgr.move_stop(new_stop)
        except Exception as e:
            log.error(f"[BOT] move_stop failed: {e}")

def _eod_close():
    log.info("[BOT] EOD — flattening all positions")
    if not PAPER_MODE:
        try:
            mgr.cancel_and_flatten()
        except Exception as e:
            log.error(f"[BOT] EOD close failed: {e}")

# Wire strategy callbacks
engine.on_enter_long  = _enter_long
engine.on_enter_short = _enter_short
engine.on_move_stop   = _move_stop
engine.on_eod_close   = _eod_close

# ── Connect order manager + list-contracts helper ─────────────────────────────
log.info("Connecting to TopstepX ...")
mgr.connect()

if args.list_contracts:
    log.info(f"Searching contracts for '{args.list_contracts}' ...")
    contracts = mgr.search_contracts(args.list_contracts)
    print(f"\n{'Contract ID':<30} {'Name':<40} {'Tick Size'}")
    print("-" * 80)
    for c in contracts:
        cid  = c.get("id") or c.get("contractId") or ""
        name = c.get("name") or c.get("description") or ""
        tick = c.get("tickSize") or ""
        print(f"{cid:<30} {name:<40} {tick}")
    mgr.disconnect()
    sys.exit(0)

# ── Seed EMA from historical bars ─────────────────────────────────────────────
log.info("Fetching historical 15-min bars to seed EMA ...")
try:
    hist_bars = mgr.get_historical_bars(
        contract_id=args.contract,
        bar_unit=4,    # minute bars (check ProjectX docs for exact enum)
        bar_size=15,
        count=100,
    )
    if hist_bars:
        closes = [float(b.get("close") or b.get("c") or 0) for b in hist_bars if b]
        closes = [c for c in closes if c > 0]
        engine.seed_ema(closes)
        log.info(f"  Seeded EMA from {len(closes)} historical closes")
    else:
        log.warning("  No historical bars returned — EMA not seeded. "
                    "Strategy will not enter until EMA warms up naturally.")
except Exception as e:
    log.warning(f"  Historical bar fetch failed: {e} — EMA not seeded")

# Fetch live equity
try:
    eq = mgr.get_account_equity()
    if eq:
        engine.update_equity(eq)
        log.info(f"  Account equity: ${eq:,.2f}")
except Exception as e:
    log.warning(f"  Could not fetch equity: {e} — using --capital value")

# ── Live price tick handler ───────────────────────────────────────────────────
_last_price: dict = {"v": None}

def on_trade_tick(price: float, size: int, ts: datetime):
    _last_price["v"] = price
    engine.on_tick(price, ts)

# ── 15-min bar close tracker (for EMA update) ─────────────────────────────────
_bar_tracker: dict = {
    "bar_start": None,
    "bar_close": None,
}

def _bar_close_monitor():
    """
    Background thread: detects 5-min bar boundaries in ET and calls
    engine.update_ema() on each close. EMA is updated on 5-min bar closes
    to match the strategy's EMA timeframe (finer than the 15-min OR window).
    """
    while not _shutdown.is_set():
        now_et = datetime.now(EASTERN)
        minute = (now_et.minute // 5) * 5
        bar_start = now_et.replace(minute=minute, second=0, microsecond=0)

        prev = _bar_tracker["bar_start"]
        if prev is not None and bar_start > prev:
            # New bar just opened — the old bar closed
            close = _last_price["v"]
            if close is not None:
                log.debug(f"[BarClose] {prev.strftime('%H:%M')} ET  close={close:.2f}")
                engine.update_ema(close)

        _bar_tracker["bar_start"] = bar_start
        time.sleep(5)  # check every 5 seconds

# ── Equity sync thread ────────────────────────────────────────────────────────
def _equity_sync():
    """Refresh account equity every 5 minutes so sizing stays accurate."""
    while not _shutdown.is_set():
        try:
            eq = mgr.get_account_equity()
            if eq:
                engine.update_equity(eq)
        except Exception:
            pass
        for _ in range(60):   # sleep 5 minutes in 5s increments
            if _shutdown.is_set():
                break
            time.sleep(5)

# ── Graceful shutdown ─────────────────────────────────────────────────────────
_shutdown = threading.Event()

def _handle_signal(sig, frame):
    log.info(f"Signal {sig} received — shutting down ...")
    _shutdown.set()

signal.signal(signal.SIGINT,  _handle_signal)
signal.signal(signal.SIGTERM, _handle_signal)

# ── Market stream ─────────────────────────────────────────────────────────────
stream = MarketStream(
    contract_id=args.contract,
    on_trade=on_trade_tick,
)

# ── Start everything ──────────────────────────────────────────────────────────
log.info("Starting market data stream ...")
stream.start(mgr.token)

threading.Thread(target=_bar_close_monitor, daemon=True, name="bar-monitor").start()
threading.Thread(target=_equity_sync,       daemon=True, name="equity-sync").start()

log.info("Bot running. Press Ctrl+C to stop.")
log.info(f"Watching {args.contract} for ORB signal after 08:30 ET ...")

# ── Main loop ─────────────────────────────────────────────────────────────────
HEARTBEAT_INTERVAL = 60  # seconds
last_heartbeat = time.time()

try:
    while not _shutdown.is_set():
        time.sleep(1)

        # Heartbeat log every minute
        if time.time() - last_heartbeat >= HEARTBEAT_INTERVAL:
            status = engine.status()
            price  = _last_price["v"]
            log.info(
                f"[Heartbeat]  state={status['state']:<15}  "
                f"price={f'{price:.2f}' if price else 'n/a':<10}  "
                f"EMA {status['ema_fast'] or 'n/a'}/{status['ema_slow'] or 'n/a'}  "
                f"OR {status['or_high'] or 'n/a'}/{status['or_low'] or 'n/a'}  "
                f"pos={status['position'] or 'flat'}"
            )
            last_heartbeat = time.time()

        # Refresh token for stream if needed (mgr handles REST token internally)
        stream.update_token(mgr.token)

except KeyboardInterrupt:
    pass

# ── Shutdown ──────────────────────────────────────────────────────────────────
log.info("Shutting down ...")
stream.stop()

# Safety: flatten any open position on exit
try:
    if engine.status()["position"]:
        log.info("Open position detected on shutdown — flattening ...")
        mgr.cancel_and_flatten()
except Exception as e:
    log.error(f"Shutdown flatten failed: {e}")

mgr.disconnect()
log.info("Bot stopped cleanly.")
