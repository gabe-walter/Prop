"""
TopstepX / ProjectX  —  REST API client + User Hub stream.

Responsibilities
────────────────
1. Authenticate (username + API key → JWT)
2. Keep the JWT alive (auto-refresh every TOKEN_REFRESH_INTERVAL hours)
3. Place orders: market entry + linked stop and take-profit
4. Modify the stop order (BE adjustment)
5. Cancel all open orders + flatten position on EOD
6. Stream account/order/position events via the User Hub (SignalR)
7. Forward fill events back to the strategy engine

REST base URL : https://api.topstepx.com
User Hub URL  : https://rtc.topstepx.com/hubs/user

Order type enum (used in JSON body):
  1 = Limit    2 = Market    3 = Stop    4 = StopLimit    8 = TrailingStop

Side enum:
  0 = Buy     1 = Sell

All prices are in index points (MNQ tick = 0.25 pts).
"""

import logging
import threading
import time as _time
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from typing import Callable, Dict, List, Optional

import requests

log = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────
_BASE_URL   = "https://api.topstepx.com"
_USER_HUB   = "https://rtc.topstepx.com/hubs/user"
_AUTH_EP    = "/api/v1/Auth/loginKey"
_ACCOUNT_EP = "/api/v1/Account/search"
_CONTRACT_EP= "/api/v1/Contract/search"
_HIST_EP    = "/api/v1/History/retrieveBars"
_ORDER_EP   = "/api/v1/Order/place"
_CANCEL_EP  = "/api/v1/Order/cancel"
_MODIFY_EP  = "/api/v1/Order/modify"
_POS_EP     = "/api/v1/Position/search"
_CLOSE_EP   = "/api/v1/Position/close"

TOKEN_REFRESH_INTERVAL = 20   # hours (tokens live ~24h; refresh early)

ORDER_TYPE_LIMIT   = 1
ORDER_TYPE_MARKET  = 2
ORDER_TYPE_STOP    = 3

SIDE_BUY  = 0
SIDE_SELL = 1


# ── Data classes ──────────────────────────────────────────────────────────────

@dataclass
class PlacedOrders:
    """Tracks the IDs of all orders placed for a single trade."""
    entry_id:  Optional[str] = None
    stop_id:   Optional[str] = None
    target_id: Optional[str] = None


# ── Order manager ─────────────────────────────────────────────────────────────

class OrderManager:
    """
    Full lifecycle order manager for a single TopstepX account + contract.

    Typical usage
    -------------
    mgr = OrderManager(username="...", api_key="...",
                       account_id=12345, contract_id="CON.F.US.MNQ.U25",
                       on_fill=engine.notify_fill,
                       on_position_closed=engine.notify_exit)
    mgr.connect()
    ...
    ids = mgr.enter_long(qty=1, stop=21300.0, target=21400.0)
    ...
    mgr.cancel_and_flatten()   # EOD
    mgr.disconnect()
    """

    def __init__(
        self,
        username:    str,
        api_key:     str,
        account_id:  int,
        contract_id: str,
        point_value: float = 2.0,
        on_fill:     Optional[Callable] = None,  # fn(fill_price, stop_order_id)
        on_position_closed: Optional[Callable] = None,  # fn(reason)
    ):
        self.username    = username
        self.api_key     = api_key
        self.account_id  = account_id
        self.contract_id = contract_id
        self.point_value = point_value

        self._on_fill            = on_fill
        self._on_position_closed = on_position_closed

        self._token:          str = ""
        self._token_acquired: datetime = datetime.min.replace(tzinfo=timezone.utc)
        self._session = requests.Session()
        self._session.headers.update({"Content-Type": "application/json"})

        self._current_orders: PlacedOrders = PlacedOrders()
        self._lock = threading.Lock()

        # User hub SignalR connection
        self._user_hub = None
        self._refresh_thread: Optional[threading.Thread] = None
        self._stop_refresh = threading.Event()

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def connect(self):
        """Authenticate, open the user hub stream, start token refresh loop."""
        self._authenticate()
        self._open_user_hub()
        self._start_refresh_loop()
        log.info("[OrderManager] Ready.")

    def disconnect(self):
        """Close user hub and stop refresh loop cleanly."""
        self._stop_refresh.set()
        if self._user_hub:
            try:
                self._user_hub.stop()
            except Exception:
                pass
        log.info("[OrderManager] Disconnected.")

    @property
    def token(self) -> str:
        return self._token

    # ── Market / historical data ──────────────────────────────────────────────

    def search_contracts(self, text: str) -> list:
        """Find contracts by name fragment, e.g. 'MNQ' or 'NQ'."""
        resp = self._post(_CONTRACT_EP, {"searchText": text, "live": True})
        return resp.get("contracts", resp) if isinstance(resp, dict) else resp

    def get_historical_bars(
        self,
        contract_id: str,
        bar_unit: int     = 4,   # 4 = minute bars
        bar_size: int     = 15,  # 15-minute bars
        from_ts:  Optional[datetime] = None,
        to_ts:    Optional[datetime] = None,
        count:    int     = 50,
    ) -> list:
        """
        Fetch historical OHLCV bars for EMA seeding.

        bar_unit: 1=second 2=minute(?) 3=hour 4=minute (varies by API version)
                  Check ProjectX docs for the exact enum — 4 = minute is common.
        bar_size: number of bar_units per bar (15 for 15-min bars)

        Returns list of dicts: {open, high, low, close, volume, timestamp}
        """
        now = datetime.now(timezone.utc)
        payload = {
            "contractId": contract_id,
            "live":       True,
            "barUnit":    bar_unit,
            "barUnitNumber": bar_size,
            "fromTimestamp": (from_ts or now - timedelta(days=3)).isoformat(),
            "toTimestamp":   (to_ts   or now).isoformat(),
        }
        resp = self._post(_HIST_EP, payload)
        bars = resp.get("bars", resp) if isinstance(resp, dict) else resp
        return bars or []

    def get_account_equity(self) -> Optional[float]:
        """Return current account balance/equity."""
        try:
            resp = self._post(_ACCOUNT_EP, {"onlyActive": True})
            accounts = resp.get("accounts", []) if isinstance(resp, dict) else resp
            for acct in accounts:
                if acct.get("id") == self.account_id:
                    return float(acct.get("balance") or acct.get("equity") or 0)
        except Exception as e:
            log.warning(f"[OrderManager] get_account_equity failed: {e}")
        return None

    # ── Order actions ─────────────────────────────────────────────────────────

    def enter_long(self, qty: int, stop: float, target: float) -> PlacedOrders:
        """
        Place market BUY then immediately place stop-loss and take-profit orders.
        Returns PlacedOrders with IDs for later modification/cancellation.
        """
        log.info(f"[OrderManager] ENTER LONG  qty={qty}  stop={stop:.2f}  target={target:.2f}")
        ids = PlacedOrders()
        try:
            # 1. Market entry
            r = self._place_order(ORDER_TYPE_MARKET, SIDE_BUY, qty, stop_price=None, limit_price=None)
            ids.entry_id = str(r.get("orderId") or r.get("id") or "")
            log.info(f"[OrderManager] Long entry placed  id={ids.entry_id}")
        except Exception as e:
            log.error(f"[OrderManager] Failed to place long entry: {e}")
            raise

        # Place bracket orders (stop + target) — best effort
        ids.stop_id   = self._place_bracket_stop(   SIDE_SELL, qty, stop)
        ids.target_id = self._place_bracket_limit(  SIDE_SELL, qty, target)

        with self._lock:
            self._current_orders = ids
        return ids

    def enter_short(self, qty: int, stop: float, target: float) -> PlacedOrders:
        """Place market SELL then bracket stop-loss and take-profit."""
        log.info(f"[OrderManager] ENTER SHORT  qty={qty}  stop={stop:.2f}  target={target:.2f}")
        ids = PlacedOrders()
        try:
            r = self._place_order(ORDER_TYPE_MARKET, SIDE_SELL, qty, stop_price=None, limit_price=None)
            ids.entry_id = str(r.get("orderId") or r.get("id") or "")
            log.info(f"[OrderManager] Short entry placed  id={ids.entry_id}")
        except Exception as e:
            log.error(f"[OrderManager] Failed to place short entry: {e}")
            raise

        ids.stop_id   = self._place_bracket_stop(   SIDE_BUY, qty, stop)
        ids.target_id = self._place_bracket_limit(  SIDE_BUY, qty, target)

        with self._lock:
            self._current_orders = ids
        return ids

    def move_stop(self, new_stop: float):
        """Modify the existing stop order to a new price (BE adjustment)."""
        with self._lock:
            stop_id = self._current_orders.stop_id

        if not stop_id:
            log.warning("[OrderManager] move_stop: no stop order ID tracked — skipping")
            return

        log.info(f"[OrderManager] Moving stop → {new_stop:.2f}  order={stop_id}")
        try:
            self._post(_MODIFY_EP, {
                "accountId": self.account_id,
                "orderId":   stop_id,
                "stopPrice": new_stop,
            })
        except Exception as e:
            log.error(f"[OrderManager] Failed to modify stop: {e}")

    def cancel_and_flatten(self):
        """EOD: cancel all open orders then market-close the position."""
        log.info("[OrderManager] EOD — cancelling orders and flattening ...")
        with self._lock:
            ids = self._current_orders
            self._current_orders = PlacedOrders()

        for oid in [ids.stop_id, ids.target_id, ids.entry_id]:
            if oid:
                self._cancel_order(oid)

        # Close any remaining open position
        self._flatten_position()

    # ── Private REST helpers ──────────────────────────────────────────────────

    def _place_order(
        self,
        order_type: int,
        side: int,
        qty: int,
        stop_price: Optional[float],
        limit_price: Optional[float],
    ) -> dict:
        body = {
            "accountId":  self.account_id,
            "contractId": self.contract_id,
            "type":       order_type,
            "side":       side,
            "size":       qty,
        }
        if stop_price  is not None:
            body["stopPrice"]  = stop_price
        if limit_price is not None:
            body["limitPrice"] = limit_price

        return self._post(_ORDER_EP, body)

    def _place_bracket_stop(self, side: int, qty: int, stop: float) -> Optional[str]:
        """Place a stop order to close a position. Returns order ID or None."""
        try:
            r = self._place_order(ORDER_TYPE_STOP, side, qty, stop_price=stop, limit_price=None)
            oid = str(r.get("orderId") or r.get("id") or "")
            log.info(f"[OrderManager] Bracket stop placed  id={oid}  stop={stop:.2f}")
            return oid or None
        except Exception as e:
            log.error(f"[OrderManager] Failed to place bracket stop: {e}")
            return None

    def _place_bracket_limit(self, side: int, qty: int, limit: float) -> Optional[str]:
        """Place a limit order to close a position. Returns order ID or None."""
        try:
            r = self._place_order(ORDER_TYPE_LIMIT, side, qty, stop_price=None, limit_price=limit)
            oid = str(r.get("orderId") or r.get("id") or "")
            log.info(f"[OrderManager] Bracket target placed  id={oid}  limit={limit:.2f}")
            return oid or None
        except Exception as e:
            log.error(f"[OrderManager] Failed to place bracket target: {e}")
            return None

    def _cancel_order(self, order_id: str):
        try:
            self._post(_CANCEL_EP, {"accountId": self.account_id, "orderId": order_id})
            log.info(f"[OrderManager] Cancelled order {order_id}")
        except Exception as e:
            log.warning(f"[OrderManager] Failed to cancel {order_id}: {e}")

    def _flatten_position(self):
        try:
            resp     = self._post(_POS_EP, {"accountId": self.account_id})
            positions = resp.get("positions", []) if isinstance(resp, dict) else resp
            for pos in positions:
                if pos.get("contractId") == self.contract_id:
                    pid = pos.get("id")
                    self._post(_CLOSE_EP, {"accountId": self.account_id, "positionId": pid})
                    log.info(f"[OrderManager] Closed position {pid}")
        except Exception as e:
            log.error(f"[OrderManager] Failed to flatten position: {e}")

    def _post(self, endpoint: str, payload: dict) -> dict:
        url  = _BASE_URL + endpoint
        resp = self._session.post(url, json=payload, timeout=10)
        resp.raise_for_status()
        return resp.json()

    # ── Authentication ────────────────────────────────────────────────────────

    def _authenticate(self):
        log.info("[OrderManager] Authenticating ...")
        r = requests.post(
            _BASE_URL + _AUTH_EP,
            json={"userName": self.username, "apiKey": self.api_key},
            timeout=15,
        )
        r.raise_for_status()
        data  = r.json()
        token = data.get("token") or data.get("accessToken") or data.get("jwt")
        if not token:
            raise RuntimeError(f"Auth response had no token: {data}")
        self._token = token
        self._token_acquired = datetime.now(timezone.utc)
        self._session.headers["Authorization"] = f"Bearer {token}"
        log.info("[OrderManager] Authenticated.")

    def _start_refresh_loop(self):
        self._stop_refresh.clear()
        self._refresh_thread = threading.Thread(
            target=self._refresh_loop, daemon=True, name="token-refresh"
        )
        self._refresh_thread.start()

    def _refresh_loop(self):
        while not self._stop_refresh.is_set():
            age = (datetime.now(timezone.utc) - self._token_acquired).total_seconds() / 3600
            if age >= TOKEN_REFRESH_INTERVAL:
                log.info("[OrderManager] Token nearing expiry — refreshing ...")
                try:
                    self._authenticate()
                    if self._user_hub:
                        self._user_hub.update_token(self._token)
                except Exception as e:
                    log.error(f"[OrderManager] Token refresh failed: {e}")
            _time.sleep(300)  # check every 5 minutes

    # ── User Hub (order fill / position events) ───────────────────────────────

    def _open_user_hub(self):
        """Connect to the user hub to receive order fill notifications."""
        try:
            from signalrcore.hub_connection_builder import HubConnectionBuilder
        except ImportError:
            log.warning(
                "[OrderManager] signalrcore not installed — user hub disabled. "
                "Fill confirmations will not be forwarded to the strategy engine."
            )
            return

        url = f"{_USER_HUB}?access_token={self._token}"
        conn = (
            HubConnectionBuilder()
            .with_url(url, options={"skip_negotiation": True, "transport_type": "WebSockets"})
            .build()
        )
        conn.on("GatewayUserOrder",    self._on_order_event)
        conn.on("GatewayUserPosition", self._on_position_event)
        conn.on_open(lambda: (
            conn.send("SubscribeAccounts", [str(self.account_id)]),
            log.info("[OrderManager] User hub connected and subscribed")
        ))
        conn.on_error(lambda e: log.error(f"[OrderManager] User hub error: {e}"))
        conn.start()
        self._user_hub = conn

    def _on_order_event(self, args: list):
        """
        User hub fires GatewayUserOrder when order status changes.
        args[0] = account_id,  args[1] = order_dict
        """
        try:
            data   = args[1] if len(args) > 1 else args[0]
            status = str(data.get("status") or data.get("orderStatus") or "").upper()
            oid    = str(data.get("orderId") or data.get("id") or "")

            if status in ("FILLED", "PARTFILLED"):
                fill_price = float(
                    data.get("fillPrice")
                    or data.get("avgFillPrice")
                    or data.get("limitPrice")
                    or data.get("stopPrice")
                    or 0
                )
                log.info(f"[OrderManager] Fill event  id={oid}  fill={fill_price:.2f}")

                with self._lock:
                    stop_id = self._current_orders.stop_id

                if self._on_fill and fill_price:
                    self._on_fill(fill_price, stop_id)

            log.debug(f"[OrderManager] Order event  id={oid}  status={status}")
        except Exception as e:
            log.debug(f"[OrderManager] Order event parse error: {e}")

    def _on_position_event(self, args: list):
        """
        User hub fires GatewayUserPosition when a position closes.
        """
        try:
            data   = args[1] if len(args) > 1 else args[0]
            status = str(data.get("status") or "").upper()
            if status in ("CLOSED", "LIQUIDATED"):
                log.info(f"[OrderManager] Position closed externally  data={data}")
                if self._on_position_closed:
                    self._on_position_closed("external")
        except Exception as e:
            log.debug(f"[OrderManager] Position event parse error: {e}")
