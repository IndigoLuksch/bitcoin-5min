#!/usr/bin/env python3
"""
Polymarket BTC 5-Minute Up/Down Backtester
============================================
Fetches real "Bitcoin Up or Down" 5-minute markets from Polymarket,
prices them with a Brownian motion fair-value model using Binance spot data,
and backtests a simple edge-based strategy.

Market mechanics:
  - Each market covers a 5-minute BTC window (e.g. 10:30-10:35 UTC).
  - Resolves "Up" if BTC price at window end >= price at window start.
  - Resolves "Down" otherwise.
  - Trading is most active during the window itself; pre-window price ~0.50.

Model:
  P(Up) = Φ(d2), where d2 = (ln(S/S0) - ½σ²τ) / (σ√τ)
  S  = current BTC spot price
  S0 = BTC price at the start of the window (the "strike")
  τ  = minutes remaining in the window
  σ  = rolling realized vol of 1-min BTC log returns

Data sources:
  - Gamma API:     event/market discovery via slug pattern
  - Data API:      historical trades per market (condition_id)
  - Binance:       BTC/USDT 1-minute candles for spot & volatility

Usage:
    python backtest.py
"""

import argparse
import json
import logging
import os
import re
import statistics
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone, timedelta
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
from scipy.stats import norm

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
CACHE_DIR = Path("cache")
OUTPUT_DIR = Path("output")
CACHE_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(OUTPUT_DIR / "backtest.log", mode="w"),
    ],
)
log = logging.getLogger(__name__)

GAMMA_BASE = "https://gamma-api.polymarket.com"
DATA_API = "https://data-api.polymarket.com"
BINANCE = "https://api.binance.com/api/v3"

THRESHOLDS = [0.03, 0.05, 0.10, 0.15]
FEE_RATE = 0.02
VOL_LOOKBACK = 60        # minutes of trailing 1-min returns for vol
BUCKET_SECONDS = 30       # aggregate trades into 30-second snapshots
NUM_DAYS = 14             # default days of history (use --days to override)
MARKETS_PER_BATCH = 50    # fetch this many markets per Gamma API call

SESSION = requests.Session()
SESSION.headers.update({"Accept": "application/json"})

# ---------------------------------------------------------------------------
# HTTP + caching
# ---------------------------------------------------------------------------

def _get(url, params=None, retries=3, backoff=1.0):
    for attempt in range(retries):
        try:
            r = SESSION.get(url, params=params, timeout=30)
            if r.status_code == 429:
                time.sleep(backoff * (2 ** attempt))
                continue
            r.raise_for_status()
            return r.json()
        except requests.exceptions.RequestException as e:
            if attempt == retries - 1:
                log.warning("Request failed: %s — %s", url.split("?")[0], e)
                return None
            time.sleep(backoff * (2 ** attempt))
    return None


def _cache_path(name):
    return CACHE_DIR / re.sub(r'[^\w\-.]', '_', name)


def _load_cache(name):
    p = _cache_path(name)
    if p.exists():
        with open(p) as f:
            return json.load(f)
    return None


def _save_cache(name, data):
    with open(_cache_path(name), "w") as f:
        json.dump(data, f)

# ---------------------------------------------------------------------------
# Step 1: Discover 5-minute BTC Up/Down markets
# ---------------------------------------------------------------------------

def discover_5m_markets(num_days=NUM_DAYS, anchor_ts=None):
    """
    Generate slugs for btc-updown-5m markets and fetch metadata.
    Markets exist every 5 minutes (300s) 24/7.
    """
    cache_name = f"btc_5m_markets_{num_days}d.json"
    cached = _load_cache(cache_name)
    if cached:
        log.info("Loaded %d 5-min markets from cache", len(cached))
        return cached

    if anchor_ts is None:
        now = datetime.now(timezone.utc)
        # Round down to the last completed 5-min boundary, minus a buffer
        anchor_ts = int(now.timestamp())
        anchor_ts = (anchor_ts // 300) * 300 - 600  # 10 minutes ago

    start_ts = anchor_ts - num_days * 86400
    # Generate all 5-min timestamps in the range
    timestamps = list(range(start_ts, anchor_ts, 300))
    log.info("Generating slugs for %d potential markets over %d days", len(timestamps), num_days)

    markets = []
    # Batch fetch via Gamma — query one at a time (Gamma has no batch slug lookup)
    for i, ts in enumerate(timestamps):
        slug = f"btc-updown-5m-{ts}"
        data = _get(f"{GAMMA_BASE}/events", params={"slug": slug})
        if not data or not isinstance(data, list) or len(data) == 0:
            continue

        ev = data[0]
        if not ev.get("closed"):
            continue

        mk = ev.get("markets", [{}])[0]
        if not mk:
            continue

        outcomes = mk.get("outcomes", [])
        prices_raw = mk.get("outcomePrices", "[]")
        if isinstance(prices_raw, str):
            try:
                prices_raw = json.loads(prices_raw)
            except json.JSONDecodeError:
                continue

        # Determine resolution
        up_won = False
        if len(prices_raw) >= 2:
            up_won = float(prices_raw[0]) > 0.5

        clob_ids = mk.get("clobTokenIds", "[]")
        if isinstance(clob_ids, str):
            try:
                clob_ids = json.loads(clob_ids)
            except json.JSONDecodeError:
                clob_ids = []

        markets.append({
            "slug": slug,
            "title": ev.get("title", ""),
            "condition_id": mk.get("conditionId", ""),
            "window_start_ts": ts,
            "window_end_ts": ts + 300,
            "up_won": up_won,
            "up_token_id": clob_ids[0] if len(clob_ids) > 0 else "",
            "down_token_id": clob_ids[1] if len(clob_ids) > 1 else "",
        })

        if (i + 1) % 100 == 0:
            log.info("  checked %d/%d timestamps, found %d markets",
                     i + 1, len(timestamps), len(markets))
        time.sleep(0.05)

    log.info("Discovered %d closed 5-min markets", len(markets))
    _save_cache(cache_name, markets)
    return markets


def discover_5m_markets_fast(num_days=NUM_DAYS, anchor_ts=None):
    """
    Faster discovery: assume all 5-min slots exist (they do 24/7).
    Fetch a sample to get condition_ids, then batch-verify via data-api trades.
    """
    cache_name = f"btc_5m_markets_fast_{num_days}d.json"
    cached = _load_cache(cache_name)
    if cached:
        log.info("Loaded %d 5-min markets from cache", len(cached))
        return cached

    if anchor_ts is None:
        now = datetime.now(timezone.utc)
        anchor_ts = (int(now.timestamp()) // 300) * 300 - 600

    start_ts = anchor_ts - num_days * 86400
    timestamps = list(range(start_ts, anchor_ts, 300))
    total = len(timestamps)
    log.info("Will fetch %d markets over %d days (this takes a while)...", total, num_days)

    markets = []
    for i, ts in enumerate(timestamps):
        slug = f"btc-updown-5m-{ts}"
        data = _get(f"{GAMMA_BASE}/events", params={"slug": slug})
        if not data or not data[0].get("closed"):
            continue

        ev = data[0]
        mk = ev["markets"][0] if ev.get("markets") else {}
        if not mk.get("conditionId"):
            continue

        prices_raw = mk.get("outcomePrices", "[]")
        if isinstance(prices_raw, str):
            prices_raw = json.loads(prices_raw)

        clob_ids = mk.get("clobTokenIds", "[]")
        if isinstance(clob_ids, str):
            clob_ids = json.loads(clob_ids)

        markets.append({
            "slug": slug,
            "title": ev.get("title", ""),
            "condition_id": mk["conditionId"],
            "window_start_ts": ts,
            "window_end_ts": ts + 300,
            "up_won": float(prices_raw[0]) > 0.5 if len(prices_raw) >= 2 else None,
            "up_token_id": clob_ids[0] if clob_ids else "",
            "down_token_id": clob_ids[1] if len(clob_ids) > 1 else "",
        })

        if (i + 1) % 50 == 0:
            pct = (i + 1) / total * 100
            log.info("  [%.0f%%] %d/%d checked, %d found", pct, i + 1, total, len(markets))
        time.sleep(0.05)

    log.info("Discovered %d 5-min markets", len(markets))
    _save_cache(cache_name, markets)
    return markets

# ---------------------------------------------------------------------------
# Step 2a: Fetch trades from Polymarket Data API
# ---------------------------------------------------------------------------

def fetch_trades(condition_id, max_pages=4):
    cache_name = f"trades5m_{condition_id[:20]}.json"
    cached = _load_cache(cache_name)
    if cached is not None:
        return cached

    all_trades = []
    for pg in range(max_pages):
        data = _get(f"{DATA_API}/trades", params={
            "market": condition_id, "limit": 500, "offset": pg * 500,
        }, retries=1, backoff=0.5)
        if not data or not isinstance(data, list) or len(data) == 0:
            break
        all_trades.extend(data)
        if len(data) < 500:
            break
        time.sleep(0.05)

    _save_cache(cache_name, all_trades)
    return all_trades

# ---------------------------------------------------------------------------
# Step 2b: Fetch BTC spot data from Binance
# ---------------------------------------------------------------------------

def fetch_btc_1m(start_dt, end_dt):
    """Fetch 1-minute BTC/USDT candles. Returns DataFrame with timestamp, close."""
    start_ms = int(start_dt.timestamp() * 1000)
    end_ms = int(end_dt.timestamp() * 1000)
    cache_name = f"btc1m_{start_ms}_{end_ms}.json"
    cached = _load_cache(cache_name)
    if cached is not None:
        klines = cached
    else:
        klines = []
        cur = start_ms
        while cur < end_ms:
            data = _get(f"{BINANCE}/klines", params={
                "symbol": "BTCUSDT", "interval": "1m",
                "startTime": cur, "endTime": end_ms, "limit": 1000,
            })
            if not data:
                break
            klines.extend(data)
            cur = data[-1][0] + 1
            if len(data) < 1000:
                break
            time.sleep(0.05)
        _save_cache(cache_name, klines)

    if not klines:
        return pd.DataFrame()

    df = pd.DataFrame(klines, columns=[
        "ot", "o", "h", "l", "c", "v", "ct", "qv", "n", "tbb", "tbq", "ign",
    ])
    df["timestamp"] = pd.to_datetime(df["ot"], unit="ms", utc=True)
    df["close"] = df["c"].astype(float)
    return df[["timestamp", "close"]].sort_values("timestamp").reset_index(drop=True)

# ---------------------------------------------------------------------------
# Step 3: Brownian motion fair pricing
# ---------------------------------------------------------------------------

def fair_price_up(S, S0, tau_min, sigma_1min):
    """
    P(BTC_end >= S0) given current spot S, with tau_min minutes left.
    Under GBM: d2 = (ln(S/S0) - 0.5*σ²*τ) / (σ*√τ)
    """
    if tau_min <= 0:
        return 1.0 if S >= S0 else 0.0
    if sigma_1min <= 0 or S <= 0 or S0 <= 0:
        return 0.5

    vol = sigma_1min * np.sqrt(tau_min)
    d2 = (np.log(S / S0) - 0.5 * sigma_1min**2 * tau_min) / vol
    return float(np.clip(norm.cdf(d2), 0.001, 0.999))

# ---------------------------------------------------------------------------
# Step 4: Build observations & run backtest
# ---------------------------------------------------------------------------

def build_observations(markets, btc_df):
    """
    For each market, merge trade snapshots with BTC spot to compute
    model price vs market price at each observation point.
    """
    if btc_df.empty:
        return []

    # Precompute BTC lookup and rolling vol
    btc_df = btc_df.copy()
    btc_df["ts_min"] = btc_df["timestamp"].dt.floor("min")
    btc_lookup = dict(zip(btc_df["ts_min"], btc_df["close"]))

    log_ret = np.log(btc_df["close"] / btc_df["close"].shift(1))
    vol_s = log_ret.rolling(VOL_LOOKBACK, min_periods=20).std()
    vol_lookup = dict(zip(btc_df["ts_min"], vol_s))
    fallback_vol = vol_s.dropna().median()

    all_obs = []
    mkts_used = 0

    for i, mkt in enumerate(markets):
        if (i + 1) % 50 == 0:
            log.info("  building obs: %d/%d markets, %d obs so far",
                     i + 1, len(markets), len(all_obs))

        cid = mkt["condition_id"]
        ws = mkt["window_start_ts"]
        we = mkt["window_end_ts"]
        outcome = 1 if mkt["up_won"] else 0

        # Get the BTC opening price (at window start)
        ws_dt = pd.Timestamp(datetime.fromtimestamp(ws, tz=timezone.utc)).floor("min")
        S0 = btc_lookup.get(ws_dt)
        if S0 is None:
            # Try nearby minutes
            for d in [-1, 1, -2, 2]:
                S0 = btc_lookup.get(ws_dt + pd.Timedelta(minutes=d))
                if S0 is not None:
                    break
        if S0 is None:
            continue

        # Fetch trades
        trades = fetch_trades(cid)
        if not trades or len(trades) < 5:
            continue

        # Filter to in-window trades and bucket into snapshots
        up_tid = mkt["up_token_id"]
        window_trades = []
        for t in trades:
            ts = t.get("timestamp")
            if ts is None:
                continue
            ts = int(ts)
            if ws <= ts <= we and t.get("outcome") == "Up":
                window_trades.append({"ts": ts, "price": float(t["price"])})

        if len(window_trades) < 3:
            continue

        # Bucket by BUCKET_SECONDS
        buckets = defaultdict(list)
        for t in window_trades:
            bk = (t["ts"] // BUCKET_SECONDS) * BUCKET_SECONDS
            buckets[bk].append(t["price"])

        for bk_ts in sorted(buckets):
            prices = buckets[bk_ts]
            market_price = statistics.median(prices)
            if market_price <= 0.01 or market_price >= 0.99:
                continue

            obs_dt = pd.Timestamp(datetime.fromtimestamp(bk_ts, tz=timezone.utc)).floor("min")

            # Current BTC spot
            S = btc_lookup.get(obs_dt)
            if S is None:
                for d in [-1, 1]:
                    S = btc_lookup.get(obs_dt + pd.Timedelta(minutes=d))
                    if S is not None:
                        break
            if S is None:
                continue

            tau = (we - bk_ts) / 60.0  # minutes remaining
            if tau <= 0:
                continue

            sigma = vol_lookup.get(obs_dt, fallback_vol)
            if sigma is None or np.isnan(sigma) or sigma <= 0:
                sigma = fallback_vol if (fallback_vol and not np.isnan(fallback_vol)) else 0.001

            model_price = fair_price_up(S, S0, tau, sigma)

            all_obs.append({
                "market_id": cid,
                "slug": mkt["slug"],
                "timestamp": datetime.fromtimestamp(bk_ts, tz=timezone.utc).isoformat(),
                "window_start": datetime.fromtimestamp(ws, tz=timezone.utc).isoformat(),
                "market_price": market_price,
                "model_price": model_price,
                "outcome": outcome,
                "S": S,
                "S0": S0,
                "sigma": sigma,
                "tau_min": tau,
                "n_trades": len(prices),
            })

        if any(o["market_id"] == cid for o in all_obs[-50:]):
            mkts_used += 1

    log.info("Built %d observations from %d markets", len(all_obs), mkts_used)
    return all_obs


def run_backtest(observations):
    results = {}
    for thresh in THRESHOLDS:
        trades = []
        for obs in observations:
            mp = obs["market_price"]
            fp = obs["model_price"]
            edge = fp - mp

            if edge > thresh:
                # Buy Up at mp — pays 1 if Up won
                won = obs["outcome"] == 1
                gross = (1.0 - mp) if won else -mp
                fee = max(0, gross) * FEE_RATE
                trades.append({**obs, "side": "BUY_UP", "edge": edge,
                               "gross_pnl": gross, "fee": fee, "net_pnl": gross - fee})
            elif edge < -thresh:
                # Buy Down at (1-mp) — pays 1 if Down won
                cost = 1.0 - mp
                won = obs["outcome"] == 0
                gross = (1.0 - cost) if won else -cost
                fee = max(0, gross) * FEE_RATE
                trades.append({**obs, "side": "BUY_DOWN", "edge": edge,
                               "gross_pnl": gross, "fee": fee, "net_pnl": gross - fee})

        if trades:
            df = pd.DataFrame(trades)
            cum = df["net_pnl"].cumsum()
            results[thresh] = {
                "n": len(trades),
                "win_rate": (df["net_pnl"] > 0).mean(),
                "total_pnl": df["net_pnl"].sum(),
                "avg_pnl": df["net_pnl"].mean(),
                "max_dd": (cum - cum.cummax()).min(),
                "sharpe": df["net_pnl"].mean() / df["net_pnl"].std() if df["net_pnl"].std() > 0 else 0,
                "df": df,
            }
        else:
            results[thresh] = {"n": 0, "win_rate": 0, "total_pnl": 0,
                               "avg_pnl": 0, "max_dd": 0, "sharpe": 0,
                               "df": pd.DataFrame()}
    return results

# ---------------------------------------------------------------------------
# Step 5: Plots and output
# ---------------------------------------------------------------------------

def make_plots(results, obs_df):
    if obs_df.empty:
        log.warning("No data to plot")
        return

    # Summary CSV
    rows = []
    for t in THRESHOLDS:
        r = results[t]
        rows.append({
            "threshold": t, "num_trades": r["n"],
            "win_rate": round(r["win_rate"], 4) if r["n"] else None,
            "total_pnl": round(r["total_pnl"], 4),
            "pnl_per_trade": round(r["avg_pnl"], 4) if r["n"] else None,
            "max_drawdown": round(r["max_dd"], 4),
        })
    pd.DataFrame(rows).to_csv(OUTPUT_DIR / "summary_table.csv", index=False)

    # 1. Equity curves
    fig, ax = plt.subplots(figsize=(12, 6))
    for t in THRESHOLDS:
        r = results[t]
        if r["n"] > 0:
            cum = r["df"]["net_pnl"].cumsum()
            ax.plot(range(len(cum)), cum.values,
                    label=f"thresh={t}  n={r['n']}  PnL={r['total_pnl']:.2f}")
    ax.set_xlabel("Trade #")
    ax.set_ylabel("Cumulative PnL (contract units)")
    ax.set_title("Equity Curves by Edge Threshold")
    ax.legend(fontsize=9)
    ax.axhline(0, c="black", lw=0.5, ls="--")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "equity_curves.png", dpi=150)
    plt.close(fig)

    # 2. Calibration
    fig, ax = plt.subplots(figsize=(8, 8))
    obs_df["bucket"] = pd.cut(obs_df["model_price"],
                               bins=np.arange(0, 1.05, 0.1), include_lowest=True)
    cal = obs_df.groupby("bucket", observed=True).agg(
        mp=("model_price", "mean"),
        freq=("outcome", "mean"),
        cnt=("outcome", "size"),
    ).dropna()
    if not cal.empty:
        ax.scatter(cal["mp"], cal["freq"], s=cal["cnt"] * 1.5, alpha=0.7,
                   edgecolors="black", zorder=5)
        ax.plot([0, 1], [0, 1], "r--", alpha=0.7, label="perfect calibration")
        for _, row in cal.iterrows():
            ax.annotate(f"n={int(row['cnt'])}", (row["mp"], row["freq"]),
                        textcoords="offset points", xytext=(5, 5), fontsize=7)
    ax.set_xlabel("Model Predicted P(Up)")
    ax.set_ylabel("Observed Fraction Up")
    ax.set_title("Model Calibration: Predicted vs Actual")
    ax.legend()
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "calibration.png", dpi=150)
    plt.close(fig)

    # 3. Edge vs PnL scatter (best threshold)
    best_t = max(THRESHOLDS, key=lambda t: results[t]["total_pnl"])
    bdf = results[best_t]["df"]
    if not bdf.empty:
        fig, ax = plt.subplots(figsize=(10, 6))
        colors = ["green" if p > 0 else "red" for p in bdf["net_pnl"]]
        ax.scatter(bdf["edge"], bdf["net_pnl"], c=colors, alpha=0.4, s=15)
        ax.axhline(0, c="black", lw=0.5, ls="--")
        ax.axvline(0, c="black", lw=0.5, ls="--")
        ax.set_xlabel("Edge (model − market)")
        ax.set_ylabel("Trade PnL")
        ax.set_title(f"Edge vs PnL  (threshold={best_t})")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(OUTPUT_DIR / "edge_vs_pnl.png", dpi=150)
        plt.close(fig)

    # 4. Edge distribution
    fig, ax = plt.subplots(figsize=(10, 6))
    edges = obs_df["model_price"] - obs_df["market_price"]
    ax.hist(edges, bins=60, alpha=0.7, color="steelblue", edgecolor="black", lw=0.3)
    ax.axvline(edges.mean(), c="red", ls="--", label=f"mean={edges.mean():.4f}")
    ax.axvline(edges.median(), c="orange", ls="--", label=f"median={edges.median():.4f}")
    ax.axvline(0, c="black", lw=0.5)
    ax.set_xlabel("Edge (model − market)")
    ax.set_ylabel("Count")
    ax.set_title("Distribution of Model−Market Edge (all observations)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "edge_distribution.png", dpi=150)
    plt.close(fig)

    log.info("All plots saved to %s/", OUTPUT_DIR)

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Polymarket BTC 5-min Up/Down Backtester")
    parser.add_argument("--days", type=int, default=NUM_DAYS,
                        help=f"Days of history to use (default: {NUM_DAYS}). "
                             "Run download_data.py first to prefetch the data.")
    args = parser.parse_args()
    num_days = args.days

    log.info("=" * 70)
    log.info("Polymarket BTC 5-Min Up/Down Backtester — %d days", num_days)
    log.info("=" * 70)

    # --- Step 1: Discover markets ---
    log.info("\n=== STEP 1: Discovering 5-minute BTC markets ===")
    markets = discover_5m_markets_fast(num_days=num_days)
    if not markets:
        log.error("No markets found!")
        return

    up_n = sum(1 for m in markets if m["up_won"])
    log.info("Found %d markets. Resolution: %d Up / %d Down (%.1f%% Up)",
             len(markets), up_n, len(markets) - up_n,
             up_n / len(markets) * 100 if markets else 0)

    # --- Step 2: Fetch BTC spot data for the full date range ---
    log.info("\n=== STEP 2: Fetching BTC spot data ===")
    min_ts = min(m["window_start_ts"] for m in markets)
    max_ts = max(m["window_end_ts"] for m in markets)
    btc_start = datetime.fromtimestamp(min_ts - 3600, tz=timezone.utc)  # 1h buffer for vol warmup
    btc_end = datetime.fromtimestamp(max_ts + 300, tz=timezone.utc)
    log.info("BTC data window: %s to %s", btc_start.isoformat(), btc_end.isoformat())

    btc_df = fetch_btc_1m(btc_start, btc_end)
    log.info("Fetched %d minutes of BTC/USDT data", len(btc_df))
    if btc_df.empty:
        log.error("No BTC data!")
        return

    # --- Step 3+4: Build observations & backtest ---
    log.info("\n=== STEP 3: Building observations (model vs market) ===")
    observations = build_observations(markets, btc_df)

    if not observations:
        log.error("No observations generated!")
        return

    obs_df = pd.DataFrame(observations)
    obs_df.to_csv(OUTPUT_DIR / "observations.csv", index=False)
    log.info("Saved %d observations (%d unique markets) to observations.csv",
             len(obs_df), obs_df["market_id"].nunique())

    edges = obs_df["model_price"] - obs_df["market_price"]
    log.info("  Market price:  [%.3f, %.3f]  mean=%.3f",
             obs_df["market_price"].min(), obs_df["market_price"].max(),
             obs_df["market_price"].mean())
    log.info("  Model price:   [%.3f, %.3f]  mean=%.3f",
             obs_df["model_price"].min(), obs_df["model_price"].max(),
             obs_df["model_price"].mean())
    log.info("  Edge:          [%.3f, %.3f]  mean=%.4f  std=%.4f",
             edges.min(), edges.max(), edges.mean(), edges.std())

    log.info("\n=== STEP 4: Running backtest ===")
    results = run_backtest(observations)

    # --- Step 5: Output ---
    log.info("\n=== STEP 5: Generating output ===")
    make_plots(results, obs_df)

    hdr = f"{'Thresh':>8} {'Trades':>7} {'WinRate':>8} {'TotPnL':>10} {'PnL/Tr':>10} {'MaxDD':>10} {'Sharpe':>8}"
    sep = "-" * 72
    print(f"\n{'=' * 72}")
    print("BACKTEST RESULTS — BTC 5-Min Up/Down Markets")
    print(f"{'=' * 72}")
    print(hdr)
    print(sep)
    for t in THRESHOLDS:
        r = results[t]
        wr = f"{r['win_rate']:.1%}" if r["n"] else "—"
        sh = f"{r['sharpe']:.3f}" if r["n"] else "—"
        print(f"{t:>8.2f} {r['n']:>7} {wr:>8} {r['total_pnl']:>10.4f} "
              f"{r['avg_pnl']:>10.4f} {r['max_dd']:>10.4f} {sh:>8}")
    print(sep)
    print(f"Observations: {len(observations)}  |  Markets: {obs_df['market_id'].nunique()}")
    print(f"Date range: {obs_df['window_start'].min()} → {obs_df['window_start'].max()}")
    print(f"Output: {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
