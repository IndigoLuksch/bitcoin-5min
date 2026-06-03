#!/usr/bin/env python3
"""
Polymarket BTC 5-Min Up/Down — Live Trader v3
=============================================
v3 changes vs v2:
  - Dynamic profit target: PT = clip(PT_MULTIPLIER * entry_rel_edge, PT_FLOOR, PT_CEILING)
  - Stop loss: exit if unrealized return drops below -STOP_LOSS_PCT
  - Time-based exit: if neither PT nor SL fires within MAX_HOLD_SECONDS, exit at market
  - Removed MAX_VOL filter (no measurable effect in v2 live data)

Paper mode (default): logs virtual trades.
Live mode (--live): places real limit orders via the Polymarket CLOB.

Usage:
    python live_trader_v3.py              # paper trading (default)
    python live_trader_v3.py --live       # real trading via CLOB

Environment variables for --live mode:
    POLY_PRIVATE_KEY   — Polygon wallet private key
    POLY_API_KEY       — CLOB API key       (optional: auto-derived if missing)
    POLY_API_SECRET    — CLOB API secret     (optional: auto-derived if missing)
    POLY_API_PASSPHRASE — CLOB API passphrase (optional: auto-derived if missing)
    POLY_CHAIN_ID      — Chain ID (default: 137 for Polygon mainnet)
"""

import argparse
import csv
import json
import logging
import os
import signal
import sys
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import requests
from scipy.stats import norm

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
THRESHOLD = 0.24         # relative edge threshold (edge / token cost)
EXIT_THRESHOLD = -0.10   # close only on genuine edge reversal
TRADE_SIZE = 2.00        # USD per trade (paper mode)
MAX_LOSS = 50.00         # kill switch
VOL_LOOKBACK = 60        # minutes of trailing 1-min returns for vol
POLL_INTERVAL = 0.5      # seconds between checks
DASHBOARD_RENDER_SECS = 1.0   # throttle terminal redraw + status.json writes
OBS_LOG_SECS = 5.0            # throttle observation CSV appends
WINDOW_SECS = 300        # 5-minute windows
MOMENTUM_LOOKBACK = 5    # minutes for short-term drift estimate
MOMENTUM_EWMA_HALFLIFE = 1.0  # half-life (minutes) for exponential decay in momentum
MIN_HOLD_SECONDS = 120   # don't exit within this many seconds of entry
KELLY_FRACTION = 0.5     # half-Kelly for position sizing

# --- v3 signal filters ---
MIN_ELAPSED_MINUTES = 1.5   # don't enter in first N minutes (model=0.50 is useless)
MIN_MODEL_DEVIATION = 0.05  # model must deviate from 0.50 to have real conviction
ENTRY_PRICE_MIN = 0.10      # skip deep-OTM tokens (correctly priced, no edge)
ENTRY_PRICE_MAX = 0.90      # skip deep-ITM tokens
VOL_EWMA_HALFLIFE = 10      # minutes for EWMA vol halflife
DRIFT_SCALE = 0.25          # attenuate momentum's contribution to fair price

# --- v3 exit logic ---
# Dynamic profit target: PT = clip(PT_MULTIPLIER * entry_rel_edge, PT_FLOOR, PT_CEILING)
# v2 evidence: PT exits hit avg 85s with 100% WR; resolution holds were 4% WR.
# Idea: small-edge trades get smaller PT (exit faster), fat-edge trades can ride.
PT_MULTIPLIER = 1.0
PT_FLOOR = 0.30
PT_CEILING = 1.00
STOP_LOSS_PCT = 0.50        # exit if unrealized return drops below -50% of cost
MAX_HOLD_SECONDS = 70       # exit at market if no PT/SL fires within this window

GAMMA_BASE = "https://gamma-api.polymarket.com"
CLOB_BASE = "https://clob.polymarket.com"
BINANCE = "https://api.binance.com/api/v3"
FEE_RATE = 0.02

OUTPUT_DIR = Path("output")
OUTPUT_DIR.mkdir(exist_ok=True)

OBS_CSV = OUTPUT_DIR / "live_observations_v3.csv"
TRADE_CSV = OUTPUT_DIR / "live_trades_v3.csv"
LOG_FILE = OUTPUT_DIR / "live_trader_v3.log"
CONTROL_FILE = OUTPUT_DIR / "control.json"
STATUS_FILE = OUTPUT_DIR / "status.json"

OBS_FIELDS = [
    "timestamp", "slug", "window_start", "window_end",
    "btc_spot", "btc_strike", "sigma", "tau_min",
    "model_price", "market_price_gamma", "market_price_clob",
    "edge", "signal",
]
TRADE_FIELDS = [
    "timestamp", "slug", "window_start", "window_end",
    "side", "entry_price", "model_price", "edge",
    "rel_edge", "dynamic_pt",
    "btc_spot", "btc_strike", "sigma", "tau_min",
    "trade_size_usd", "resolved", "outcome", "pnl",
    "exit_price", "exit_reason", "exit_timestamp",
]

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(LOG_FILE, mode="a"),
    ],
)
log = logging.getLogger(__name__)

SESSION = requests.Session()
SESSION.headers.update({"Accept": "application/json"})

# ---------------------------------------------------------------------------
# Graceful shutdown
# ---------------------------------------------------------------------------
_shutdown = False

def _handle_signal(sig, frame):
    global _shutdown
    _shutdown = True
    log.info("Shutdown requested (signal %s)", sig)

signal.signal(signal.SIGINT, _handle_signal)
signal.signal(signal.SIGTERM, _handle_signal)

# ---------------------------------------------------------------------------
# HTTP helpers
# ---------------------------------------------------------------------------

def _get(url, params=None, timeout=10):
    try:
        r = SESSION.get(url, params=params, timeout=timeout)
        if r.status_code == 429:
            time.sleep(1)
            r = SESSION.get(url, params=params, timeout=timeout)
        r.raise_for_status()
        return r.json()
    except requests.exceptions.RequestException as e:
        log.debug("HTTP error %s: %s", url.split("?")[0], e)
        return None

# ---------------------------------------------------------------------------
# CLOB live trading engine
# ---------------------------------------------------------------------------

class ClobTrader:
    """Manages real order placement and lifecycle via py-clob-client."""

    def __init__(self, trade_size_usd, max_loss):
        from py_clob_client.client import ClobClient
        from py_clob_client.clob_types import ApiCreds

        private_key = os.environ.get("POLY_PRIVATE_KEY", "")
        if not private_key:
            raise RuntimeError("POLY_PRIVATE_KEY env var is required for --live mode")

        chain_id = int(os.environ.get("POLY_CHAIN_ID", "137"))

        api_key = os.environ.get("POLY_API_KEY", "")
        api_secret = os.environ.get("POLY_API_SECRET", "")
        api_passphrase = os.environ.get("POLY_API_PASSPHRASE", "")

        if api_key and api_secret and api_passphrase:
            creds = ApiCreds(
                api_key=api_key,
                api_secret=api_secret,
                api_passphrase=api_passphrase,
            )
        else:
            creds = None

        self.client = ClobClient(
            host=CLOB_BASE,
            chain_id=chain_id,
            key=private_key,
            creds=creds,
        )

        if creds is None:
            log.info("No API creds provided — deriving from wallet...")
            derived = self.client.create_or_derive_api_creds()
            self.client.set_api_creds(derived)
            log.info("API creds derived successfully")

        self.trade_size_usd = trade_size_usd
        self.max_loss = max_loss
        self.active_orders = {}  # window_start_ts -> order info dict

    def get_balance(self):
        try:
            from py_clob_client.clob_types import BalanceAllowanceParams, AssetType
            resp = self.client.get_balance_allowance(
                BalanceAllowanceParams(
                    asset_type=AssetType.COLLATERAL,
                    token_id="",
                    signature_type=0,
                )
            )
            if resp and "balance" in resp:
                return float(resp["balance"]) / 1e6  # USDC has 6 decimals
        except Exception as e:
            log.warning("Failed to fetch balance: %s", e)
        return None

    def place_limit_order(self, token_id, side_label, price, window_start_ts):
        """Place a limit order. side_label is 'BUY_UP' or 'BUY_DOWN'."""
        from py_clob_client.clob_types import OrderArgs, PartialCreateOrderOptions, OrderType

        tick_size = self.client.get_tick_size(token_id)
        neg_risk = self.client.get_neg_risk(token_id)

        size = round(self.trade_size_usd / price, 2)
        if size < 0.01:
            log.warning("Order size too small (%.4f), skipping", size)
            return None

        order_args = OrderArgs(
            token_id=token_id,
            price=round(price, 4),
            size=size,
            side="BUY",
        )
        options = PartialCreateOrderOptions(tick_size=tick_size, neg_risk=neg_risk)

        try:
            resp = self.client.create_and_post_order(order_args, options=options)
        except Exception as e:
            log.error("Failed to place order: %s", e)
            return None

        if not resp:
            log.error("Empty response from create_and_post_order")
            return None

        order_id = resp.get("orderID") or resp.get("order_id") or resp.get("id")
        if not order_id:
            log.error("No order ID in response: %s", resp)
            return None

        order_info = {
            "order_id": order_id,
            "token_id": token_id,
            "side_label": side_label,
            "price": price,
            "size": size,
            "window_start_ts": window_start_ts,
            "placed_at": time.time(),
            "filled": False,
            "fill_size": 0.0,
            "cancelled": False,
        }
        self.active_orders[window_start_ts] = order_info
        log.info("LIVE ORDER placed: %s  id=%s  price=%.4f  size=%.2f  token=%s…",
                 side_label, order_id[:12], price, size, token_id[:12])
        return order_info

    def check_order_status(self, window_start_ts):
        """Check if an active order has been filled."""
        info = self.active_orders.get(window_start_ts)
        if not info or info["cancelled"]:
            return info

        try:
            order = self.client.get_order(info["order_id"])
        except Exception as e:
            log.debug("Failed to check order %s: %s", info["order_id"][:12], e)
            return info

        if not order:
            return info

        status = order.get("status", "")
        size_matched = float(order.get("size_matched", 0))

        if status == "MATCHED" or size_matched >= info["size"] * 0.99:
            info["filled"] = True
            info["fill_size"] = size_matched
            log.info("LIVE ORDER FILLED: %s  id=%s  filled=%.2f",
                     info["side_label"], info["order_id"][:12], size_matched)
        elif status in ("CANCELLED", "EXPIRED"):
            info["cancelled"] = True
            info["fill_size"] = size_matched
            if size_matched > 0:
                info["filled"] = True  # partial fill
                log.info("LIVE ORDER PARTIAL FILL+CANCEL: %s  id=%s  filled=%.2f",
                         info["side_label"], info["order_id"][:12], size_matched)

        return info

    def cancel_order(self, window_start_ts):
        """Cancel an active order for a window."""
        info = self.active_orders.get(window_start_ts)
        if not info or info["cancelled"] or info["filled"]:
            return

        try:
            self.client.cancel(info["order_id"])
            info["cancelled"] = True
            log.info("LIVE ORDER CANCELLED: id=%s", info["order_id"][:12])
        except Exception as e:
            log.warning("Failed to cancel order %s: %s", info["order_id"][:12], e)

    def cancel_all_active(self):
        """Cancel all active unfilled orders."""
        for ws_ts in list(self.active_orders.keys()):
            self.cancel_order(ws_ts)

    def should_cancel_before_close(self, window_start_ts, now_ts):
        """Cancel unfilled orders 30 seconds before window closes."""
        info = self.active_orders.get(window_start_ts)
        if not info or info["filled"] or info["cancelled"]:
            return False
        window_end = window_start_ts + WINDOW_SECS
        return now_ts >= window_end - 30

    def cleanup_window(self, window_start_ts):
        """Remove order tracking for a completed window."""
        self.active_orders.pop(window_start_ts, None)


# ---------------------------------------------------------------------------
# BTC spot data — rolling 1-min candle buffer
# ---------------------------------------------------------------------------

class BTCFeed:
    # Cache the Binance ticker for 1s — spot doesn't move meaningfully in 500ms
    # and Binance latency (~400ms) was the loop bottleneck.
    TICKER_TTL = 1.0

    def __init__(self, lookback_minutes=120):
        self.lookback = lookback_minutes
        self.candles = deque(maxlen=lookback_minutes + 10)
        self.last_kline_fetch = 0
        self.last_ticker_fetch = 0
        self._live_spot = None

    def refresh(self):
        now = time.time()
        # Ticker: only refetch if cache is stale.
        if now - self.last_ticker_fetch >= self.TICKER_TTL:
            ticker = _get(f"{BINANCE}/ticker/price", params={"symbol": "BTCUSDT"})
            if ticker and "price" in ticker:
                self._live_spot = float(ticker["price"])
                self.last_ticker_fetch = now

        # Refresh klines less often (for vol calculation)
        if now - self.last_kline_fetch < 10:
            return
        data = _get(f"{BINANCE}/klines", params={
            "symbol": "BTCUSDT", "interval": "1m",
            "limit": self.lookback + 5,
        })
        if not data:
            return
        self.candles.clear()
        for k in data:
            self.candles.append({
                "open_time": k[0],
                "close": float(k[4]),
            })
        self.last_kline_fetch = now

    @property
    def spot(self):
        if self._live_spot is not None:
            return self._live_spot
        if not self.candles:
            return None
        return self.candles[-1]["close"]

    def price_at(self, ts_seconds):
        """Get the BTC price closest to a unix timestamp."""
        target_ms = int(ts_seconds * 1000)
        best = None
        best_dist = float("inf")
        for c in self.candles:
            dist = abs(c["open_time"] - target_ms)
            if dist < best_dist:
                best_dist = dist
                best = c["close"]
        return best

    def realized_vol(self, n_minutes=VOL_LOOKBACK):
        if len(self.candles) < n_minutes + 1:
            closes = [c["close"] for c in self.candles]
        else:
            closes = [c["close"] for c in list(self.candles)[-(n_minutes + 1):]]
        if len(closes) < 21:
            return 0.00046  # fallback from backtest
        log_rets = np.diff(np.log(closes))
        return float(np.std(log_rets, ddof=1))

    def momentum(self, n_minutes=MOMENTUM_LOOKBACK, halflife=MOMENTUM_EWMA_HALFLIFE):
        """EWMA-weighted drift: recent 1-min returns weighted exponentially over older ones."""
        if len(self.candles) < n_minutes + 1:
            return 0.0
        closes = [c["close"] for c in list(self.candles)[-(n_minutes + 1):]]
        log_rets = np.diff(np.log(closes))
        alpha = 1.0 - 0.5 ** (1.0 / halflife)
        n = len(log_rets)
        weights = np.array([(1.0 - alpha) ** i for i in range(n - 1, -1, -1)])
        weights /= weights.sum()
        return float(np.sum(weights * log_rets))

    def realized_vol_ewma(self, halflife=VOL_EWMA_HALFLIFE):
        """EWMA volatility — reacts faster to regime changes than flat lookback."""
        closes = [c["close"] for c in self.candles]
        if len(closes) < 21:
            return 0.00046
        log_rets = np.diff(np.log(closes))
        alpha = 1.0 - 0.5 ** (1.0 / halflife)
        n = len(log_rets)
        weights = np.array([(1.0 - alpha) ** i for i in range(n - 1, -1, -1)])
        weights /= weights.sum()
        weighted_var = float(np.sum(weights * log_rets ** 2))
        weighted_mean = float(np.sum(weights * log_rets))
        ewma_var = max(weighted_var - weighted_mean ** 2, 1e-14)
        return float(np.sqrt(ewma_var))

# ---------------------------------------------------------------------------
# Market discovery and pricing
# ---------------------------------------------------------------------------

def current_window_ts():
    """Return the start timestamp of the currently active 5-min window."""
    now = int(time.time())
    return (now // WINDOW_SECS) * WINDOW_SECS

def next_window_ts():
    return current_window_ts() + WINDOW_SECS

def fetch_market(window_start_ts):
    """Fetch market metadata from Gamma API."""
    slug = f"btc-updown-5m-{window_start_ts}"
    data = _get(f"{GAMMA_BASE}/events", params={"slug": slug})
    if not data or not isinstance(data, list) or len(data) == 0:
        return None

    ev = data[0]
    mk = ev.get("markets", [{}])[0] if ev.get("markets") else {}
    if not mk:
        return None

    prices_raw = mk.get("outcomePrices", "[]")
    if isinstance(prices_raw, str):
        try:
            prices_raw = json.loads(prices_raw)
        except json.JSONDecodeError:
            prices_raw = []

    clob_ids = mk.get("clobTokenIds", "[]")
    if isinstance(clob_ids, str):
        try:
            clob_ids = json.loads(clob_ids)
        except json.JSONDecodeError:
            clob_ids = []

    gamma_up_price = float(prices_raw[0]) if len(prices_raw) >= 1 else None

    return {
        "slug": slug,
        "condition_id": mk.get("conditionId", ""),
        "up_token_id": clob_ids[0] if len(clob_ids) > 0 else "",
        "down_token_id": clob_ids[1] if len(clob_ids) > 1 else "",
        "gamma_up_price": gamma_up_price,
        "closed": ev.get("closed", False),
        "window_start_ts": window_start_ts,
        "window_end_ts": window_start_ts + WINDOW_SECS,
    }

def fetch_clob_midprice(token_id):
    """Get mid-price from CLOB /midpoint endpoint."""
    if not token_id:
        return None
    data = _get(f"{CLOB_BASE}/midpoint", params={"token_id": token_id})
    if not data or "mid" not in data:
        return None
    mid = float(data["mid"])
    if mid <= 0.0 or mid >= 1.0:
        return None
    return mid

def check_resolution(window_start_ts):
    """Check if a market has resolved and return the outcome."""
    mkt = fetch_market(window_start_ts)
    if not mkt or not mkt["closed"]:
        return None
    prices_raw = None
    data = _get(f"{GAMMA_BASE}/events", params={"slug": mkt["slug"]})
    if data and len(data) > 0:
        mk = data[0].get("markets", [{}])[0]
        prices_raw = mk.get("outcomePrices", "[]")
        if isinstance(prices_raw, str):
            try:
                prices_raw = json.loads(prices_raw)
            except json.JSONDecodeError:
                return None
    if prices_raw and len(prices_raw) >= 2:
        return "Up" if float(prices_raw[0]) > 0.5 else "Down"
    return None

# ---------------------------------------------------------------------------
# Fair pricing model (from backtest.py)
# ---------------------------------------------------------------------------

def fair_price_up(S, S0, tau_min, sigma_1min, drift_per_min=0.0):
    if tau_min <= 0:
        return 1.0 if S >= S0 else 0.0
    if sigma_1min <= 0 or S <= 0 or S0 <= 0:
        return 0.5
    vol = sigma_1min * np.sqrt(tau_min)
    d2 = (np.log(S / S0) + drift_per_min * tau_min - 0.5 * sigma_1min**2 * tau_min) / vol
    return float(np.clip(norm.cdf(d2), 0.001, 0.999))

# ---------------------------------------------------------------------------
# CSV writers (append mode, create with headers if new)
# ---------------------------------------------------------------------------

def _ensure_csv(path, fields):
    if not path.exists():
        with open(path, "w", newline="") as f:
            csv.DictWriter(f, fieldnames=fields).writeheader()

def append_observation(row):
    _ensure_csv(OBS_CSV, OBS_FIELDS)
    with open(OBS_CSV, "a", newline="") as f:
        csv.DictWriter(f, fieldnames=OBS_FIELDS).writerow(row)

def append_trade(row):
    _ensure_csv(TRADE_CSV, TRADE_FIELDS)
    with open(TRADE_CSV, "a", newline="") as f:
        csv.DictWriter(f, fieldnames=TRADE_FIELDS).writerow(row)

def update_trade_resolution(slug, outcome, pnl):
    """Update the resolved/outcome/pnl columns for a trade in the CSV."""
    if not TRADE_CSV.exists():
        return
    rows = []
    with open(TRADE_CSV, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["slug"] == slug and row["resolved"] != "True":
                row["resolved"] = "True"
                row["outcome"] = outcome
                row["pnl"] = str(pnl)
            rows.append(row)
    with open(TRADE_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=TRADE_FIELDS)
        writer.writeheader()
        writer.writerows(rows)

def _update_trade_exit(slug, exit_price, exit_reason, exit_ts):
    """Write exit details into the trade CSV row."""
    if not TRADE_CSV.exists():
        return
    rows = []
    with open(TRADE_CSV, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["slug"] == slug and row.get("exit_reason", "") == "":
                row["exit_price"] = str(exit_price)
                row["exit_reason"] = exit_reason
                row["exit_timestamp"] = datetime.fromtimestamp(
                    exit_ts, tz=timezone.utc).isoformat()
            rows.append(row)
    with open(TRADE_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=TRADE_FIELDS)
        writer.writeheader()
        writer.writerows(rows)

def load_unresolved_trades():
    """Reload unresolved trades from CSV so restarts don't lose state."""
    pending = {}
    cum_pnl = 0.0
    n_trades = 0
    traded_windows = set()

    if not TRADE_CSV.exists():
        return pending, cum_pnl, n_trades, traded_windows

    with open(TRADE_CSV, "r", newline="") as f:
        for row in csv.DictReader(f):
            n_trades += 1
            slug = row["slug"]
            # Extract window_start_ts from slug
            parts = slug.rsplit("-", 1)
            if len(parts) == 2 and parts[1].isdigit():
                ws_ts = int(parts[1])
                traded_windows.add(ws_ts)

            if row["resolved"] == "True":
                try:
                    cum_pnl += float(row["pnl"])
                except (ValueError, TypeError):
                    pass
            else:
                ws_ts = int(slug.rsplit("-", 1)[1])
                # Backfill in-memory exit-trigger fields so PT/SL/time-exit work after restart.
                try:
                    row["_entry_ts"] = datetime.fromisoformat(row["timestamp"]).timestamp()
                except (KeyError, ValueError, TypeError):
                    row["_entry_ts"] = 0
                try:
                    row["_dynamic_pt"] = float(row.get("dynamic_pt") or 0) or None
                except (TypeError, ValueError):
                    row["_dynamic_pt"] = None
                pending[ws_ts] = row

    return pending, cum_pnl, n_trades, traded_windows

# ---------------------------------------------------------------------------
# Control & status files (for dashboard webapp)
# ---------------------------------------------------------------------------

def read_control():
    """Read control commands from the dashboard. Returns dict or empty."""
    if not CONTROL_FILE.exists():
        return {}
    try:
        with open(CONTROL_FILE) as f:
            ctrl = json.load(f)
        CONTROL_FILE.unlink(missing_ok=True)
        return ctrl
    except (json.JSONDecodeError, OSError):
        return {}

def write_status(state):
    """Write current bot status for the dashboard to read."""
    try:
        tmp = STATUS_FILE.with_suffix(".tmp")
        with open(tmp, "w") as f:
            json.dump(state, f)
        tmp.replace(STATUS_FILE)
    except OSError:
        pass

# ---------------------------------------------------------------------------
# Terminal dashboard
# ---------------------------------------------------------------------------

def render_dashboard(state):
    now_str = datetime.now(timezone.utc).strftime("%H:%M:%S UTC")
    lines = [
        "",
        f"  ╔══════════════════════════════════════════════════════════════╗",
        f"  ║  BTC 5-Min Trader  {state.get('mode', 'PAPER'):>6}       {now_str:>20}  ║",
        f"  ╠══════════════════════════════════════════════════════════════╣",
    ]

    w = state.get("window", "—")
    lines.append(f"  ║  Window:  {w:<49} ║")

    btc = state.get("btc_spot")
    btc_str = f"${btc:,.2f}" if btc else "—"
    strike = state.get("btc_strike")
    strike_str = f"${strike:,.2f}" if strike else "—"
    lines.append(f"  ║  BTC Spot: {btc_str:<17}  Strike: {strike_str:<20} ║")

    tau = state.get("tau_min")
    tau_str = f"{tau:.1f} min" if tau is not None else "—"
    vol = state.get("sigma")
    vol_str = f"{vol:.6f}" if vol else "—"
    lines.append(f"  ║  Time left: {tau_str:<16}  Vol(1m): {vol_str:<19} ║")

    mp = state.get("model_price")
    mp_str = f"{mp:.4f}" if mp is not None else "—"
    mkt = state.get("market_price")
    mkt_str = f"{mkt:.4f}" if mkt is not None else "—"
    lines.append(f"  ║  Model P(Up): {mp_str:<14}  Market P(Up): {mkt_str:<14} ║")

    edge = state.get("edge")
    edge_str = f"{edge:+.4f}" if edge is not None else "—"
    sig = state.get("signal", "NONE")
    lines.append(f"  ║  Edge: {edge_str:<12}  Signal: {sig:<27} ║")

    drift = state.get("drift")
    drift_str = f"{drift:+.8f}" if drift is not None else "—"
    elapsed = state.get("elapsed_min")
    elapsed_str = f"{elapsed:.1f} min" if elapsed is not None else "—"
    lines.append(f"  ║  Drift: {drift_str:<16}  Elapsed: {elapsed_str:<17} ║")

    filt = state.get("filter_reason")
    if filt:
        lines.append(f"  ║  Filter: {filt:<49} ║")

    lines.append(f"  ╠══════════════════════════════════════════════════════════════╣")
    n = state.get("n_trades", 0)
    pnl = state.get("cum_pnl", 0.0)
    pnl_color = "+" if pnl >= 0 else ""
    lines.append(f"  ║  Trades: {n:<8}  PnL: {pnl_color}{pnl:.2f} USD{' ' * 25} ║")
    pos = state.get("position", "—")
    lines.append(f"  ║  Position: {pos:<48} ║")
    lines.append(f"  ╚══════════════════════════════════════════════════════════════╝")

    os.system("clear" if os.name != "nt" else "cls")
    print("\n".join(lines))

# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Polymarket BTC 5-Min Paper Trader v3")
    parser.add_argument("--live", action="store_true", help="Enable real trading")
    parser.add_argument("--threshold", type=float, default=THRESHOLD)
    parser.add_argument("--size", type=float, default=TRADE_SIZE)
    parser.add_argument("--max-loss", type=float, default=MAX_LOSS)
    parser.add_argument("--kelly-fraction", type=float, default=KELLY_FRACTION,
                        help="Kelly fraction for position sizing (0=fixed, 0.5=half-Kelly)")
    parser.add_argument("--min-elapsed", type=float, default=MIN_ELAPSED_MINUTES,
                        help="Wait N minutes into window before entering")
    parser.add_argument("--min-model-dev", type=float, default=MIN_MODEL_DEVIATION,
                        help="Model must deviate from 0.50 by at least this")
    parser.add_argument("--pt-multiplier", type=float, default=PT_MULTIPLIER,
                        help="Dynamic PT multiplier on entry rel_edge")
    parser.add_argument("--pt-floor", type=float, default=PT_FLOOR,
                        help="Minimum profit target")
    parser.add_argument("--pt-ceiling", type=float, default=PT_CEILING,
                        help="Maximum profit target")
    parser.add_argument("--stop-loss", type=float, default=STOP_LOSS_PCT,
                        help="Stop loss as fraction of cost (e.g. 0.5 = exit at -50%%)")
    parser.add_argument("--max-hold", type=int, default=MAX_HOLD_SECONDS,
                        help="Max seconds to hold before time-based exit")
    args = parser.parse_args()

    live_mode = args.live
    threshold = args.threshold
    trade_size = args.size
    max_loss = args.max_loss
    kelly_fraction = args.kelly_fraction
    min_elapsed = args.min_elapsed
    min_model_dev = args.min_model_dev
    pt_multiplier = args.pt_multiplier
    pt_floor = args.pt_floor
    pt_ceiling = args.pt_ceiling
    stop_loss_pct = args.stop_loss
    max_hold_seconds = args.max_hold

    clob_trader = None
    if live_mode:
        try:
            clob_trader = ClobTrader(trade_size, max_loss)
        except Exception as e:
            log.error("Failed to initialize live trading: %s", e)
            sys.exit(1)

        balance = clob_trader.get_balance()
        if balance is not None:
            log.info("Wallet USDC balance: $%.2f", balance)
            if balance < trade_size:
                log.error("Insufficient balance ($%.2f) for trade size ($%.2f)", balance, trade_size)
                sys.exit(1)

    mode_str = "LIVE" if live_mode else "paper"
    log.info("Starting v3 %s trader — threshold=%.2f  size=$%.2f  max_loss=$%.2f",
             mode_str, threshold, trade_size, max_loss)
    log.info("v3 exit logic: PT=%.2f*rel_edge clipped to [%.2f,%.2f]  SL=-%.0f%%  max_hold=%ds",
             pt_multiplier, pt_floor, pt_ceiling, stop_loss_pct * 100, max_hold_seconds)

    btc_feed = BTCFeed(lookback_minutes=VOL_LOOKBACK + 10)

    # Reload state from previous runs
    pending_resolution, cum_pnl, n_trades, traded_windows = load_unresolved_trades()
    if pending_resolution:
        log.info("Resumed %d unresolved trades from previous run (cum_pnl=$%.2f, total_trades=%d)",
                 len(pending_resolution), cum_pnl, n_trades)
    dashboard = {}

    # Concurrent HTTP fetcher: BTC ticker and CLOB midpoint fire in parallel each tick.
    executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="fetch")
    # Per-window market metadata cache. fetch_market hits Gamma /events which only
    # changes when a new window opens, so caching saves ~120 req/min at 0.5s polling.
    mkt_cache = {}
    last_render_ts = 0.0
    last_obs_ts = 0.0

    global _shutdown
    paused = False

    while not _shutdown:
        loop_start = time.time()
        try:
            # --- Process control commands from dashboard ---
            ctrl = read_control()
            if ctrl.get("command") == "stop":
                log.info("Stop command received from dashboard")
                _shutdown = True
                break
            if ctrl.get("command") == "pause":
                paused = True
                log.info("Paused by dashboard")
            if ctrl.get("command") == "resume":
                paused = False
                log.info("Resumed by dashboard")
            if "threshold" in ctrl:
                threshold = float(ctrl["threshold"])
                log.info("Threshold updated to %.4f via dashboard", threshold)
            if "trade_size" in ctrl:
                trade_size = float(ctrl["trade_size"])
                log.info("Trade size updated to $%.2f via dashboard", trade_size)
            if "max_loss" in ctrl:
                max_loss = float(ctrl["max_loss"])
                log.info("Max loss updated to $%.2f via dashboard", max_loss)

            # --- Determine active window (needed before fetching market data) ---
            now_ts = time.time()
            win_start = current_window_ts()
            win_end = win_start + WINDOW_SECS
            tau_min = (win_end - now_ts) / 60.0

            win_start_str = datetime.fromtimestamp(win_start, tz=timezone.utc).strftime("%H:%M")
            win_end_str = datetime.fromtimestamp(win_end, tz=timezone.utc).strftime("%H:%M")

            # --- Market metadata: cached per window (only re-fetch when window changes
            #     or the previous attempt returned nothing) ---
            mkt_data = mkt_cache.get(win_start)
            if not mkt_data:
                mkt_data = fetch_market(win_start)
                if mkt_data:
                    mkt_cache = {win_start: mkt_data}  # drop stale windows
            up_token_id = mkt_data["up_token_id"] if mkt_data else None

            # --- Parallel HTTP: BTC refresh + CLOB midpoint fire concurrently ---
            btc_future = executor.submit(btc_feed.refresh)
            clob_future = executor.submit(fetch_clob_midprice, up_token_id) if up_token_id else None

            btc_future.result()
            spot = btc_feed.spot
            if spot is None:
                log.warning("No BTC spot data yet, retrying...")
                if clob_future is not None:
                    clob_future.result()  # drain future
                time.sleep(POLL_INTERVAL)
                continue

            clob_mid = clob_future.result() if clob_future is not None else None
            gamma_price = mkt_data["gamma_up_price"] if mkt_data else None

            # --- Get strike (BTC price at window start) ---
            strike = btc_feed.price_at(win_start)
            if strike is None:
                strike = spot  # fallback

            market_price = clob_mid or gamma_price

            # --- Compute model (v2: EWMA vol + momentum drift) ---
            sigma_ewma = btc_feed.realized_vol_ewma()
            drift = btc_feed.momentum()
            model_price = fair_price_up(spot, strike, tau_min, sigma_ewma, drift * DRIFT_SCALE) if tau_min > 0 else None
            edge = (model_price - market_price) if (model_price is not None and market_price is not None) else None
            elapsed_min = (WINDOW_SECS / 60.0) - tau_min

            # --- Determine signal (v2: relative edge + filters) ---
            signal = "NONE"
            rel_edge = None
            filter_reason = None
            if edge is not None and market_price is not None:
                if edge > 0 and market_price > 0.01:
                    rel_edge = edge / market_price
                    if rel_edge > threshold:
                        signal = "BUY_UP"
                elif edge < 0 and market_price < 0.99:
                    rel_edge = (-edge) / (1.0 - market_price)
                    if rel_edge > threshold:
                        signal = "BUY_DOWN"

            # --- v3 signal filters (MAX_VOL removed: no measurable effect in v2 live data) ---
            if signal != "NONE":
                if elapsed_min < min_elapsed:
                    filter_reason = f"too_early ({elapsed_min:.1f}m elapsed)"
                    signal = "NONE"
                elif model_price is not None and abs(model_price - 0.5) < min_model_dev:
                    filter_reason = f"model_near_0.50 ({model_price:.4f})"
                    signal = "NONE"
                elif market_price < ENTRY_PRICE_MIN or market_price > ENTRY_PRICE_MAX:
                    filter_reason = f"extreme_price ({market_price:.3f})"
                    signal = "NONE"
                elif drift is not None:
                    # Momentum must confirm direction
                    if signal == "BUY_UP" and drift <= 0:
                        filter_reason = f"no_momentum_confirm (drift={drift:.6f})"
                        signal = "NONE"
                    elif signal == "BUY_DOWN" and drift >= 0:
                        filter_reason = f"no_momentum_confirm (drift={drift:.6f})"
                        signal = "NONE"

                if filter_reason:
                    log.debug("Signal filtered: %s", filter_reason)

            # --- Log observation (throttled to OBS_LOG_SECS to limit CSV growth) ---
            if (model_price is not None and market_price is not None and mkt_data
                    and now_ts - last_obs_ts >= OBS_LOG_SECS):
                obs_row = {
                    "timestamp": datetime.fromtimestamp(now_ts, tz=timezone.utc).isoformat(),
                    "slug": mkt_data["slug"],
                    "window_start": datetime.fromtimestamp(win_start, tz=timezone.utc).isoformat(),
                    "window_end": datetime.fromtimestamp(win_end, tz=timezone.utc).isoformat(),
                    "btc_spot": round(spot, 2),
                    "btc_strike": round(strike, 2),
                    "sigma": round(sigma_ewma, 10),
                    "tau_min": round(tau_min, 3),
                    "model_price": round(model_price, 6),
                    "market_price_gamma": round(gamma_price, 6) if gamma_price else "",
                    "market_price_clob": round(clob_mid, 6) if clob_mid else "",
                    "edge": round(edge, 6) if edge else "",
                    "signal": signal,
                }
                append_observation(obs_row)
                last_obs_ts = now_ts

            # --- Execute trade ---
            if signal != "NONE" and win_start not in traded_windows and mkt_data and not paused:
                entry_price = market_price
                side = signal

                # Kelly-scaled position sizing
                if kelly_fraction > 0:
                    if side == "BUY_UP":
                        kelly_raw = (model_price - market_price) / (1.0 - market_price)
                    else:
                        kelly_raw = (market_price - model_price) / market_price
                    kelly_bet = max(0.25, min(kelly_raw * kelly_fraction, 1.0))
                    current_trade_size = round(trade_size * kelly_bet, 2)
                else:
                    current_trade_size = trade_size

                # v3 dynamic profit target: scales with entry rel_edge
                entry_rel_edge = rel_edge if rel_edge is not None else 0.0
                dynamic_pt = max(pt_floor, min(pt_multiplier * entry_rel_edge, pt_ceiling))

                # In live mode, place a limit order at the model fair price
                live_order_ok = True
                if live_mode and clob_trader:
                    token_id = mkt_data["up_token_id"] if side == "BUY_UP" else mkt_data["down_token_id"]
                    limit_price = model_price if side == "BUY_UP" else (1.0 - model_price)
                    order_info = clob_trader.place_limit_order(
                        token_id, side, limit_price, win_start)
                    if order_info is None:
                        live_order_ok = False
                    else:
                        entry_price = limit_price

                if live_order_ok:
                    traded_windows.add(win_start)
                    n_trades += 1

                    trade_row = {
                        "timestamp": datetime.fromtimestamp(now_ts, tz=timezone.utc).isoformat(),
                        "slug": mkt_data["slug"],
                        "window_start": datetime.fromtimestamp(win_start, tz=timezone.utc).isoformat(),
                        "window_end": datetime.fromtimestamp(win_end, tz=timezone.utc).isoformat(),
                        "side": side,
                        "entry_price": round(entry_price, 6),
                        "model_price": round(model_price, 6),
                        "edge": round(edge, 6),
                        "rel_edge": round(entry_rel_edge, 6),
                        "dynamic_pt": round(dynamic_pt, 4),
                        "btc_spot": round(spot, 2),
                        "btc_strike": round(strike, 2),
                        "sigma": round(sigma_ewma, 10),
                        "tau_min": round(tau_min, 3),
                        "trade_size_usd": current_trade_size,
                        "resolved": "False",
                        "outcome": "",
                        "pnl": "",
                        "exit_price": "",
                        "exit_reason": "",
                        "exit_timestamp": "",
                    }
                    append_trade(trade_row)
                    pending_resolution[win_start] = trade_row
                    trade_row["_entry_ts"] = now_ts
                    trade_row["_dynamic_pt"] = dynamic_pt
                    prefix = "LIVE TRADE" if live_mode else "PAPER TRADE"
                    log.info("%s: %s %s @ %.4f  edge=%.4f  rel=%.2f  PT=%.2f  kelly=$%.2f  BTC=$%.2f",
                             prefix, side, mkt_data["slug"], entry_price, edge,
                             entry_rel_edge, dynamic_pt, current_trade_size, spot)

            # --- v3 position management: dynamic PT, stop loss, time exit ---
            # Only process exits when we have a current market price (model_price is unused
            # for v3 exits, but we keep the guard to skip stale ticks).
            if win_start in pending_resolution and market_price is not None:
                trade = pending_resolution[win_start]
                side = trade["side"]
                entry = float(trade["entry_price"])
                entry_ts = trade.get("_entry_ts", 0)
                t_size = float(trade.get("trade_size_usd", trade_size))
                trade_pt = trade.get("_dynamic_pt")
                if trade_pt is None:
                    try:
                        trade_pt = float(trade.get("dynamic_pt") or pt_floor)
                    except (TypeError, ValueError):
                        trade_pt = pt_floor

                if side == "BUY_UP":
                    cost = entry
                    current_value = market_price
                else:
                    cost = 1.0 - entry
                    current_value = 1.0 - market_price

                unrealized_pct = (current_value - cost) / cost if cost > 0.001 else 0.0
                hold_secs = now_ts - entry_ts if entry_ts else 0
                exit_price = None
                reason = None

                # (1) Dynamic profit target
                if trade_pt > 0 and unrealized_pct >= trade_pt:
                    exit_price = current_value
                    reason = "profit_target"

                # (2) Stop loss
                elif stop_loss_pct > 0 and unrealized_pct <= -stop_loss_pct:
                    exit_price = current_value
                    reason = "stop_loss"

                # (3) Time-based exit
                elif max_hold_seconds > 0 and hold_secs >= max_hold_seconds:
                    exit_price = current_value
                    reason = "time_exit"

                if exit_price is not None and reason is not None:
                    gross = current_value - cost
                    fee = max(0, gross) * FEE_RATE
                    net = gross - fee
                    pnl_usd = net * t_size

                    cum_pnl += pnl_usd
                    update_trade_resolution(trade["slug"], f"EXIT:{reason}",
                                            round(pnl_usd, 4))
                    _update_trade_exit(trade["slug"], round(exit_price, 6),
                                       reason, now_ts)
                    del pending_resolution[win_start]

                    log.info("EXIT [%s]: %s %s  entry=%.4f  exit=%.4f  ret=%.0f%%  hold=%.0fs  pnl=$%.4f  cum=$%.2f",
                             reason, side, trade["slug"], entry,
                             exit_price, unrealized_pct * 100, hold_secs, pnl_usd, cum_pnl)

            # --- Live order lifecycle management ---
            if live_mode and clob_trader:
                for ws_ts in list(clob_trader.active_orders.keys()):
                    # Cancel unfilled orders 30s before window closes
                    if clob_trader.should_cancel_before_close(ws_ts, now_ts):
                        clob_trader.check_order_status(ws_ts)
                        info = clob_trader.active_orders.get(ws_ts)
                        if info and not info["filled"]:
                            clob_trader.cancel_order(ws_ts)
                            if info.get("fill_size", 0) == 0:
                                # Never filled — remove from pending resolution
                                if ws_ts in pending_resolution:
                                    del pending_resolution[ws_ts]
                                log.info("Order never filled for window %d, removed from tracking", ws_ts)
                    else:
                        clob_trader.check_order_status(ws_ts)

                    # Clean up completed windows
                    if now_ts > ws_ts + WINDOW_SECS + 300:
                        clob_trader.cleanup_window(ws_ts)

            # --- Check resolutions for past windows ---
            resolved_keys = []
            for pending_ws, trade in list(pending_resolution.items()):
                if now_ts < pending_ws + WINDOW_SECS + 120:
                    continue  # wait at least 2 min after window closes
                outcome = check_resolution(pending_ws)
                if outcome is None:
                    continue

                side = trade["side"]
                entry = float(trade["entry_price"])
                if side == "BUY_UP":
                    won = outcome == "Up"
                    gross = (1.0 - entry) if won else -entry
                else:
                    cost = 1.0 - entry
                    won = outcome == "Down"
                    gross = (1.0 - cost) if won else -cost

                fee = max(0, gross) * FEE_RATE
                net = gross - fee
                t_size = float(trade.get("trade_size_usd", trade_size))
                pnl_usd = net * t_size

                cum_pnl += pnl_usd
                update_trade_resolution(trade["slug"], outcome, round(pnl_usd, 4))
                resolved_keys.append(pending_ws)
                result = "WIN" if won else "LOSS"
                log.info("RESOLVED: %s %s → %s  pnl=$%.4f  cum=$%.2f",
                         trade["side"], trade["slug"], result, pnl_usd, cum_pnl)

            for k in resolved_keys:
                del pending_resolution[k]

            # --- Kill switch ---
            if cum_pnl <= -max_loss:
                log.error("KILL SWITCH: cumulative loss $%.2f exceeds max $%.2f — stopping",
                          abs(cum_pnl), max_loss)
                if live_mode and clob_trader:
                    clob_trader.cancel_all_active()
                break

            # --- Update dashboard ---
            position = "—"
            if win_start in traded_windows:
                for t in pending_resolution.values():
                    if t["slug"] == f"btc-updown-5m-{win_start}":
                        position = f"{t['side']} @ {t['entry_price']}"
                        break
                else:
                    position = "traded (resolved or resolving)"

            dashboard = {
                "mode": "LIVE" if live_mode else "PAPER",
                "window": f"{win_start_str}–{win_end_str} UTC",
                "btc_spot": spot,
                "btc_strike": strike,
                "tau_min": tau_min,
                "sigma": sigma_ewma,
                "model_price": model_price,
                "market_price": market_price,
                "edge": edge,
                "signal": signal,
                "filter_reason": filter_reason,
                "drift": drift,
                "elapsed_min": elapsed_min,
                "n_trades": n_trades,
                "cum_pnl": cum_pnl,
                "position": position,
                "paused": paused,
                "threshold": threshold,
                "trade_size": trade_size,
                "max_loss": max_loss,
                "version": "v3",
                "pt_multiplier": pt_multiplier,
                "pt_floor": pt_floor,
                "pt_ceiling": pt_ceiling,
                "stop_loss_pct": stop_loss_pct,
                "max_hold_seconds": max_hold_seconds,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
            # Terminal redraw + status.json write are throttled so 0.5s polling
            # doesn't flicker the terminal or thrash disk.
            if now_ts - last_render_ts >= DASHBOARD_RENDER_SECS:
                render_dashboard(dashboard)
                write_status(dashboard)
                last_render_ts = now_ts

        except Exception:
            log.exception("Error in main loop")

        # Sleep only the remainder of POLL_INTERVAL — if the parallel fetches
        # already burned most of the budget, sleep less (or not at all).
        elapsed = time.time() - loop_start
        time.sleep(max(0.0, POLL_INTERVAL - elapsed))

    if live_mode and clob_trader:
        log.info("Cancelling all active orders before shutdown...")
        clob_trader.cancel_all_active()

    executor.shutdown(wait=False)
    log.info("Shutting down. Total trades: %d  Cumulative PnL: $%.2f", n_trades, cum_pnl)
    log.info("Observations: %s", OBS_CSV)
    log.info("Trades: %s", TRADE_CSV)


if __name__ == "__main__":
    main()
