"""
TopstepX / ProjectX  —  Real-time market data via SignalR.

Connects directly to the ProjectX market hub WebSocket and subscribes to:
  • GatewayTrade  — every executed trade tick  (price, size, timestamp)
  • GatewayQuote  — best bid/ask updates

This is the most granular data available: individual trade prints as they
happen on the exchange, not aggregated bars.

Hub URL  : https://rtc.topstepx.com/hubs/market
Auth     : JWT passed as `access_token` query param (Bearer not supported on WS)
Transport: WebSocket-only (skip_negotiation=True required)

Events received by the hub:
  GatewayTrade(contractId: str, data: dict)
    data keys: price, volume/size, timestamp
  GatewayQuote(contractId: str, data: dict)
    data keys: bid, ask, bidSize, askSize, timestamp

Subscribe methods sent to hub:
  SubscribeContractTrades(contractId)
  SubscribeContractQuotes(contractId)
  UnsubscribeContractTrades(contractId)
  UnsubscribeContractQuotes(contractId)
"""

import logging
import threading
import time as _time
from datetime import datetime, timezone
from typing import Callable, Optional

log = logging.getLogger(__name__)

# Hub endpoint
_MARKET_HUB = "https://rtc.topstepx.com/hubs/market"

# Reconnect back-off: 2s, 4s, 8s, 16s, 30s cap
_BACKOFF = [2, 4, 8, 16, 30]


class MarketStream:
    """
    Real-time tick stream for a single TopstepX contract.

    Usage
    -----
    stream = MarketStream(
        contract_id="CON.F.US.MNQ.U25",
        on_trade=my_tick_handler,   # fn(price: float, size: int, ts: datetime)
        on_quote=my_quote_handler,  # fn(bid: float, ask: float, ts: datetime) — optional
    )
    stream.start(token)
    # ... run strategy ...
    stream.stop()
    stream.update_token(new_token)  # call every ~20h to keep alive
    """

    def __init__(
        self,
        contract_id: str,
        on_trade: Callable[[float, int, datetime], None],
        on_quote: Optional[Callable[[float, float, datetime], None]] = None,
    ):
        self.contract_id = contract_id
        self._on_trade   = on_trade
        self._on_quote   = on_quote

        self._token:      str       = ""
        self._conn                  = None
        self._stop_event            = threading.Event()
        self._connected             = threading.Event()
        self._lock                  = threading.Lock()
        self._reconnect_thread: Optional[threading.Thread] = None

    # ── Public ────────────────────────────────────────────────────────────────

    def start(self, token: str):
        """Connect to the market hub and begin streaming ticks."""
        self._token = token
        self._stop_event.clear()
        self._connect()

    def stop(self):
        """Disconnect and shut down background threads."""
        self._stop_event.set()
        self._disconnect()
        log.info("[MarketStream] Stopped.")

    def update_token(self, token: str):
        """Refresh the JWT. Reconnects automatically if the token changed."""
        if token == self._token:
            return
        log.info("[MarketStream] Token refreshed — reconnecting ...")
        self._token = token
        self._disconnect()
        if not self._stop_event.is_set():
            self._connect()

    @property
    def is_connected(self) -> bool:
        return self._connected.is_set()

    # ── Internal ──────────────────────────────────────────────────────────────

    def _connect(self):
        try:
            from signalrcore.hub_connection_builder import HubConnectionBuilder
            from signalrcore.transport.websockets.connection import WebsocketTransport
        except ImportError:
            raise ImportError(
                "signalrcore is required for live tick streaming.\n"
                "Install with: pip install signalrcore"
            )

        url = f"{_MARKET_HUB}?access_token={self._token}"

        self._conn = (
            HubConnectionBuilder()
            .with_url(url, options={
                "skip_negotiation": True,
                "transport_type": "WebSockets",
            })
            .with_automatic_reconnect({
                "type": "raw",
                "keep_alive_interval": 10,
                "reconnect_interval": 5,
                "max_attempts": 999,
            })
            .build()
        )

        # Wire event handlers
        self._conn.on("GatewayTrade", self._on_gateway_trade)
        self._conn.on("GatewayQuote", self._on_gateway_quote)
        self._conn.on_open(self._on_open)
        self._conn.on_close(self._on_close)
        self._conn.on_error(self._on_error)

        self._conn.start()

    def _disconnect(self):
        with self._lock:
            conn = self._conn
            self._conn = None
        if conn:
            try:
                conn.stop()
            except Exception:
                pass
        self._connected.clear()

    def _subscribe(self):
        """Send subscription messages after connection is established."""
        try:
            self._conn.send("SubscribeContractTrades", [self.contract_id])
            self._conn.send("SubscribeContractQuotes", [self.contract_id])
            log.info(f"[MarketStream] Subscribed to {self.contract_id} (trades + quotes)")
        except Exception as e:
            log.error(f"[MarketStream] Subscribe failed: {e}")

    # ── SignalR callbacks ─────────────────────────────────────────────────────

    def _on_open(self):
        log.info("[MarketStream] Connected to market hub")
        self._connected.set()
        self._subscribe()

    def _on_close(self):
        log.warning("[MarketStream] Disconnected from market hub")
        self._connected.clear()

    def _on_error(self, data):
        log.error(f"[MarketStream] Error: {data}")

    def _on_gateway_trade(self, args: list):
        """
        GatewayTrade(contractId, data)
        args[0] = contractId (str)
        args[1] = data (dict)  — keys: price, volume/size, timestamp
        """
        try:
            data     = args[1] if len(args) > 1 else args[0]
            price    = _extract_price(data, ("price", "tradePrice", "lastPrice"))
            size     = int(data.get("volume") or data.get("size") or data.get("quantity") or 1)
            ts       = _extract_ts(data)

            if price is None:
                return

            self._on_trade(price, size, ts)
        except Exception as e:
            log.debug(f"[MarketStream] GatewayTrade parse error: {e}  args={args}")

    def _on_gateway_quote(self, args: list):
        """
        GatewayQuote(contractId, data)
        args[0] = contractId (str)
        args[1] = data (dict)  — keys: bid, ask, bidSize, askSize, timestamp
        """
        if self._on_quote is None:
            return
        try:
            data = args[1] if len(args) > 1 else args[0]
            bid  = _extract_price(data, ("bid", "bidPrice"))
            ask  = _extract_price(data, ("ask", "askPrice"))
            ts   = _extract_ts(data)

            if bid is None or ask is None:
                return

            self._on_quote(bid, ask, ts)
        except Exception as e:
            log.debug(f"[MarketStream] GatewayQuote parse error: {e}  args={args}")


# ── Field extraction helpers ──────────────────────────────────────────────────

def _extract_price(data: dict, keys: tuple) -> Optional[float]:
    for k in keys:
        v = data.get(k)
        if v is not None:
            return float(v)
    return None


def _extract_ts(data: dict) -> datetime:
    raw = (
        data.get("timestamp")
        or data.get("ts")
        or data.get("dateTime")
        or data.get("time")
        or data.get("eventTime")
    )
    if raw is None:
        return datetime.now(timezone.utc)
    if isinstance(raw, (int, float)):
        # Nanosecond unix timestamp (Databento/ProjectX style)
        divisor = 1e9 if raw > 1e15 else (1e3 if raw > 1e12 else 1.0)
        return datetime.fromtimestamp(raw / divisor, tz=timezone.utc)
    return datetime.fromisoformat(str(raw).replace("Z", "+00:00"))
