#!/usr/bin/env python3
"""
Backtest v2 — Applies v2 signal filters to historical observation data.

Uses the same cached market + BTC data as backtest.py, but adds:
  - EWMA volatility
  - Momentum drift in fair price
  - Relative edge threshold
  - Min elapsed time filter (skip first N minutes of window)
  - Min model deviation filter (skip when model ≈ 0.50)
  - Entry price range filter
  - Vol ceiling filter
  - Momentum confirmation
  - Profit-taking exit (simulated via intra-window price path)
  - Kelly position sizing
  - Comparison: v1 (absolute edge) vs v2 (all filters)

Usage:
    python backtest_v2.py
"""

import json
import logging
import os
import re
import statistics
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
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
        logging.FileHandler(OUTPUT_DIR / "backtest_v2.log", mode="w"),
    ],
)
log = logging.getLogger(__name__)

BINANCE = "https://api.binance.com/api/v3"
FEE_RATE = 0.02
WINDOW_SECS = 300
BUCKET_SECONDS = 30
VOL_LOOKBACK = 60
TRADE_SIZE = 2.00

# --- v1 baseline ---
V1_THRESHOLD = 0.24  # absolute edge

# --- v2 parameters ---
V2_THRESHOLD = 0.50          # relative edge
V2_MIN_ELAPSED = 2.0         # minutes
V2_MIN_MODEL_DEV = 0.05
V2_ENTRY_PRICE_MIN = 0.10
V2_ENTRY_PRICE_MAX = 0.90
V2_MAX_VOL = 0.0010
V2_PROFIT_TARGET = 0.50      # take profit at 50% return on cost
V2_EXIT_THRESHOLD = -0.10
V2_KELLY_FRACTION = 0.5
V2_MOMENTUM_LOOKBACK = 5
V2_VOL_EWMA_HALFLIFE = 10

import requests
SESSION = requests.Session()
SESSION.headers.update({"Accept": "application/json"})

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

# ---------------------------------------------------------------------------
# Data loading (reuse cached data from backtest.py)
# ---------------------------------------------------------------------------

def load_markets():
    cached = _load_cache("btc_5m_markets_fast_2d.json")
    if not cached:
        log.error("No cached markets found. Run backtest.py first.")
        sys.exit(1)
    log.info("Loaded %d markets from cache", len(cached))
    return cached

def fetch_btc_1m(start_dt, end_dt):
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
        from backtest import _save_cache
        _save_cache(cache_name, klines)

    if not klines:
        return pd.DataFrame()

    df = pd.DataFrame(klines, columns=[
        "ot", "o", "h", "l", "c", "v", "ct", "qv", "n", "tbb", "tbq", "ign",
    ])
    df["timestamp"] = pd.to_datetime(df["ot"], unit="ms", utc=True)
    df["close"] = df["c"].astype(float)
    return df[["timestamp", "close"]].sort_values("timestamp").reset_index(drop=True)

def fetch_trades(condition_id):
    cache_name = f"trades5m_{condition_id[:20]}.json"
    cached = _load_cache(cache_name)
    if cached is not None:
        return cached
    return []  # don't fetch new data, only use cache

# ---------------------------------------------------------------------------
# Model functions
# ---------------------------------------------------------------------------

def fair_price_up_v1(S, S0, tau_min, sigma):
    """v1: pure GBM, no drift."""
    if tau_min <= 0:
        return 1.0 if S >= S0 else 0.0
    if sigma <= 0 or S <= 0 or S0 <= 0:
        return 0.5
    vol = sigma * np.sqrt(tau_min)
    d2 = (np.log(S / S0) - 0.5 * sigma**2 * tau_min) / vol
    return float(np.clip(norm.cdf(d2), 0.001, 0.999))


def fair_price_up_v2(S, S0, tau_min, sigma, drift_per_min=0.0):
    """v2: GBM + momentum drift."""
    if tau_min <= 0:
        return 1.0 if S >= S0 else 0.0
    if sigma <= 0 or S <= 0 or S0 <= 0:
        return 0.5
    vol = sigma * np.sqrt(tau_min)
    d2 = (np.log(S / S0) + drift_per_min * tau_min - 0.5 * sigma**2 * tau_min) / vol
    return float(np.clip(norm.cdf(d2), 0.001, 0.999))


def compute_ewma_vol(log_rets_series, halflife=V2_VOL_EWMA_HALFLIFE):
    """Compute EWMA vol for each point in a series."""
    alpha = 1.0 - 0.5 ** (1.0 / halflife)
    n = len(log_rets_series)
    ewma_var = np.full(n, np.nan)
    var_est = log_rets_series[:20].var() if len(log_rets_series) >= 20 else 0.00046**2

    for i in range(n):
        r = log_rets_series.iloc[i]
        if np.isnan(r):
            continue
        var_est = alpha * r**2 + (1 - alpha) * var_est
        ewma_var[i] = var_est

    return pd.Series(np.sqrt(np.maximum(ewma_var, 1e-14)), index=log_rets_series.index)


def compute_momentum(btc_df, lookback=V2_MOMENTUM_LOOKBACK):
    """Compute per-minute momentum (log return over lookback window)."""
    shift = lookback
    mom = np.log(btc_df["close"] / btc_df["close"].shift(shift)) / shift
    return mom

# ---------------------------------------------------------------------------
# Build observations with both v1 and v2 signals
# ---------------------------------------------------------------------------

def build_observations(markets, btc_df):
    """Build per-snapshot observations with v1 and v2 model prices."""
    if btc_df.empty:
        return []

    btc_df = btc_df.copy()
    btc_df["ts_min"] = btc_df["timestamp"].dt.floor("min")
    btc_lookup = dict(zip(btc_df["ts_min"], btc_df["close"]))

    # v1 vol: rolling std
    log_ret = np.log(btc_df["close"] / btc_df["close"].shift(1))
    vol_s = log_ret.rolling(VOL_LOOKBACK, min_periods=20).std()
    vol_lookup = dict(zip(btc_df["ts_min"], vol_s))
    fallback_vol = vol_s.dropna().median()

    # v2 vol: EWMA
    ewma_vol_s = compute_ewma_vol(log_ret)
    ewma_vol_lookup = dict(zip(btc_df["ts_min"], ewma_vol_s))

    # v2 momentum
    momentum_s = compute_momentum(btc_df)
    momentum_lookup = dict(zip(btc_df["ts_min"], momentum_s))

    all_obs = []
    mkts_used = 0

    for i, mkt in enumerate(markets):
        if (i + 1) % 100 == 0:
            log.info("  building obs: %d/%d markets, %d obs so far",
                     i + 1, len(markets), len(all_obs))

        cid = mkt["condition_id"]
        ws = mkt["window_start_ts"]
        we = mkt["window_end_ts"]
        outcome = 1 if mkt["up_won"] else 0

        ws_dt = pd.Timestamp(datetime.fromtimestamp(ws, tz=timezone.utc)).floor("min")
        S0 = btc_lookup.get(ws_dt)
        if S0 is None:
            for d in [-1, 1, -2, 2]:
                S0 = btc_lookup.get(ws_dt + pd.Timedelta(minutes=d))
                if S0 is not None:
                    break
        if S0 is None:
            continue

        trades = fetch_trades(cid)
        if not trades or len(trades) < 5:
            continue

        up_tid = mkt.get("up_token_id", "")
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

        # Bucket
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

            S = btc_lookup.get(obs_dt)
            if S is None:
                for d in [-1, 1]:
                    S = btc_lookup.get(obs_dt + pd.Timedelta(minutes=d))
                    if S is not None:
                        break
            if S is None:
                continue

            tau = (we - bk_ts) / 60.0
            if tau <= 0:
                continue
            elapsed_min = (WINDOW_SECS / 60.0) - tau

            # v1 model
            sigma_v1 = vol_lookup.get(obs_dt, fallback_vol)
            if sigma_v1 is None or np.isnan(sigma_v1) or sigma_v1 <= 0:
                sigma_v1 = fallback_vol
            model_v1 = fair_price_up_v1(S, S0, tau, sigma_v1)
            edge_v1 = model_v1 - market_price

            # v2 model
            sigma_v2 = ewma_vol_lookup.get(obs_dt, fallback_vol)
            if sigma_v2 is None or np.isnan(sigma_v2) or sigma_v2 <= 0:
                sigma_v2 = fallback_vol
            drift = momentum_lookup.get(obs_dt, 0.0)
            if drift is None or np.isnan(drift):
                drift = 0.0
            model_v2 = fair_price_up_v2(S, S0, tau, sigma_v2, drift)
            edge_v2 = model_v2 - market_price

            all_obs.append({
                "market_id": cid,
                "slug": mkt["slug"],
                "timestamp": datetime.fromtimestamp(bk_ts, tz=timezone.utc).isoformat(),
                "window_start": datetime.fromtimestamp(ws, tz=timezone.utc).isoformat(),
                "window_start_ts": ws,
                "bk_ts": bk_ts,
                "market_price": market_price,
                "model_v1": model_v1,
                "model_v2": model_v2,
                "edge_v1": edge_v1,
                "edge_v2": edge_v2,
                "outcome": outcome,
                "S": S,
                "S0": S0,
                "sigma_v1": sigma_v1,
                "sigma_v2": sigma_v2,
                "drift": drift,
                "tau_min": tau,
                "elapsed_min": elapsed_min,
                "n_trades": len(prices),
            })

        if any(o["market_id"] == cid for o in all_obs[-50:]):
            mkts_used += 1

    log.info("Built %d observations from %d markets", len(all_obs), mkts_used)
    return all_obs

# ---------------------------------------------------------------------------
# Backtest strategies
# ---------------------------------------------------------------------------

def backtest_v1(observations):
    """v1: absolute edge threshold, every observation is a trade-to-resolution."""
    trades = []
    for obs in observations:
        mp = obs["market_price"]
        edge = obs["edge_v1"]

        if edge > V1_THRESHOLD:
            won = obs["outcome"] == 1
            gross = (1.0 - mp) if won else -mp
            fee = max(0, gross) * FEE_RATE
            trades.append({**obs, "side": "BUY_UP", "entry_price": mp,
                           "gross_pnl": gross, "fee": fee, "net_pnl": gross - fee,
                           "exit_reason": "resolution", "trade_size": TRADE_SIZE})
        elif edge < -V1_THRESHOLD:
            cost = 1.0 - mp
            won = obs["outcome"] == 0
            gross = (1.0 - cost) if won else -cost
            fee = max(0, gross) * FEE_RATE
            trades.append({**obs, "side": "BUY_DOWN", "entry_price": mp,
                           "gross_pnl": gross, "fee": fee, "net_pnl": gross - fee,
                           "exit_reason": "resolution", "trade_size": TRADE_SIZE})
    return pd.DataFrame(trades) if trades else pd.DataFrame()


def backtest_v2(observations):
    """
    v2: relative edge + all filters + intra-window profit-taking simulation.

    Groups observations by market window. Within each window:
    1. Find first observation that passes all filters → entry
    2. Scan subsequent observations for profit target or edge reversal
    3. If no exit triggered, hold to resolution
    """
    # Group observations by window
    by_window = defaultdict(list)
    for obs in observations:
        by_window[obs["window_start_ts"]].append(obs)

    # Sort each window's obs by timestamp
    for ws in by_window:
        by_window[ws].sort(key=lambda o: o["bk_ts"])

    trades = []

    for ws_ts, obs_list in sorted(by_window.items()):
        if len(obs_list) < 2:
            continue

        entry = None
        entry_obs = None
        entry_idx = None

        # --- Find entry point ---
        for idx, obs in enumerate(obs_list):
            mp = obs["market_price"]
            model = obs["model_v2"]
            edge = obs["edge_v2"]
            tau = obs["tau_min"]
            elapsed = obs["elapsed_min"]
            sigma = obs["sigma_v2"]
            drift = obs["drift"]

            # Relative edge
            if edge > 0 and mp > 0.01:
                rel_edge = edge / mp
                side = "BUY_UP"
            elif edge < 0 and mp < 0.99:
                rel_edge = (-edge) / (1.0 - mp)
                side = "BUY_DOWN"
            else:
                continue

            if rel_edge <= V2_THRESHOLD:
                continue

            # Filters
            if elapsed < V2_MIN_ELAPSED:
                continue
            if abs(model - 0.5) < V2_MIN_MODEL_DEV:
                continue
            if mp < V2_ENTRY_PRICE_MIN or mp > V2_ENTRY_PRICE_MAX:
                continue
            if V2_MAX_VOL > 0 and sigma > V2_MAX_VOL:
                continue
            # Momentum confirmation
            if side == "BUY_UP" and drift <= 0:
                continue
            if side == "BUY_DOWN" and drift >= 0:
                continue

            # Kelly sizing
            if side == "BUY_UP":
                kelly_raw = (model - mp) / (1.0 - mp)
            else:
                kelly_raw = (mp - model) / mp
            kelly_bet = max(0.25, min(kelly_raw * V2_KELLY_FRACTION, 1.0))
            current_size = round(TRADE_SIZE * kelly_bet, 2)

            entry = {
                "side": side,
                "entry_price": mp,
                "model_at_entry": model,
                "edge_at_entry": edge,
                "rel_edge": rel_edge,
                "kelly": kelly_bet,
                "trade_size": current_size,
                "entry_elapsed": elapsed,
                "entry_sigma": sigma,
                "entry_drift": drift,
            }
            entry_obs = obs
            entry_idx = idx
            break

        if entry is None:
            continue

        # --- Scan for exit within window ---
        side = entry["side"]
        ep = entry["entry_price"]
        cost = ep if side == "BUY_UP" else (1.0 - ep)
        exit_obs = None
        exit_reason = None
        exit_price = None

        for obs in obs_list[entry_idx + 1:]:
            mp_now = obs["market_price"]
            model_now = obs["model_v2"]

            if side == "BUY_UP":
                current_value = mp_now
                remaining_edge = model_now - mp_now
            else:
                current_value = 1.0 - mp_now
                remaining_edge = mp_now - model_now

            unrealized_pct = (current_value - cost) / cost if cost > 0.001 else 0

            # Profit target
            if V2_PROFIT_TARGET > 0 and unrealized_pct >= V2_PROFIT_TARGET:
                exit_obs = obs
                exit_reason = "profit_target"
                exit_price = current_value
                break

            # Edge reversal
            if remaining_edge < V2_EXIT_THRESHOLD:
                exit_obs = obs
                exit_reason = "edge_reversed"
                exit_price = current_value
                break

        # Compute PnL
        if exit_obs is not None:
            gross = exit_price - cost
            fee = max(0, gross) * FEE_RATE
            net = gross - fee
        else:
            # Hold to resolution
            exit_reason = "resolution"
            outcome = entry_obs["outcome"]
            if side == "BUY_UP":
                won = outcome == 1
                gross = (1.0 - ep) if won else -ep
            else:
                won = outcome == 0
                gross = ep if won else -(1.0 - ep)
            fee = max(0, gross) * FEE_RATE
            net = gross - fee

        pnl_usd = net * entry["trade_size"]

        trades.append({
            **entry_obs,
            "side": side,
            "entry_price": ep,
            "model_at_entry": entry["model_at_entry"],
            "edge_at_entry": entry["edge_at_entry"],
            "rel_edge": entry["rel_edge"],
            "kelly": entry["kelly"],
            "trade_size": entry["trade_size"],
            "entry_elapsed": entry["entry_elapsed"],
            "exit_reason": exit_reason,
            "gross_pnl": gross,
            "fee": fee,
            "net_pnl": net,
            "pnl_usd": pnl_usd,
        })

    return pd.DataFrame(trades) if trades else pd.DataFrame()

# ---------------------------------------------------------------------------
# Analysis and output
# ---------------------------------------------------------------------------

def analyze(label, df):
    """Print stats for a backtest run."""
    if df.empty:
        log.info("[%s] No trades", label)
        return {}

    n = len(df)
    wins = (df["net_pnl"] > 0).sum()
    wr = wins / n
    total = df["net_pnl"].sum()
    avg = df["net_pnl"].mean()
    cum = df["net_pnl"].cumsum()
    max_dd = (cum - cum.cummax()).min()
    sharpe = avg / df["net_pnl"].std() if df["net_pnl"].std() > 0 else 0

    pnl_usd_col = "pnl_usd" if "pnl_usd" in df.columns else None
    total_usd = df["pnl_usd"].sum() if pnl_usd_col else total * TRADE_SIZE

    stats = {
        "n": n, "wins": wins, "wr": wr, "total_pnl": total,
        "total_usd": total_usd, "avg_pnl": avg, "max_dd": max_dd, "sharpe": sharpe,
    }

    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")
    print(f"  Trades:     {n}")
    print(f"  Win rate:   {wr:.1%}  ({wins}/{n})")
    print(f"  Total PnL:  {total:+.4f}  (${total_usd:+.2f} @ ${TRADE_SIZE}/trade)")
    print(f"  Avg PnL:    {avg:+.4f}")
    print(f"  Max DD:     {max_dd:.4f}")
    print(f"  Sharpe:     {sharpe:.3f}")

    if "exit_reason" in df.columns:
        print(f"\n  Exit breakdown:")
        for reason, grp in df.groupby("exit_reason"):
            rn = len(grp)
            rw = (grp["net_pnl"] > 0).sum()
            rp = grp["net_pnl"].sum()
            ru = grp["pnl_usd"].sum() if "pnl_usd" in grp.columns else rp * TRADE_SIZE
            print(f"    {reason:20s}  n={rn:4d}  wr={rw/rn:.0%}  pnl={rp:+.4f}  (${ru:+.2f})")

    if "side" in df.columns:
        print(f"\n  By side:")
        for side, grp in df.groupby("side"):
            rn = len(grp)
            rw = (grp["net_pnl"] > 0).sum()
            rp = grp["net_pnl"].sum()
            print(f"    {side:10s}  n={rn:4d}  wr={rw/rn:.0%}  pnl={rp:+.4f}")

    if "entry_elapsed" in df.columns:
        print(f"\n  Entry timing:")
        print(f"    Mean elapsed: {df['entry_elapsed'].mean():.1f} min")
        print(f"    Min elapsed:  {df['entry_elapsed'].min():.1f} min")

    return stats


def make_plots(v1_df, v2_df, obs_df):
    """Generate comparison plots."""

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. Equity curves comparison
    ax = axes[0, 0]
    if not v1_df.empty:
        cum_v1 = v1_df["net_pnl"].cumsum()
        ax.plot(range(len(cum_v1)), cum_v1.values, label=f"v1 (n={len(v1_df)})", color="gray", alpha=0.7)
    if not v2_df.empty:
        pnl_col = "pnl_usd" if "pnl_usd" in v2_df.columns else "net_pnl"
        cum_v2 = v2_df[pnl_col].cumsum()
        ax.plot(range(len(cum_v2)), cum_v2.values, label=f"v2 (n={len(v2_df)})", color="black", linewidth=1.5)
    ax.axhline(0, c="red", lw=0.5, ls="--")
    ax.set_xlabel("Trade #")
    ax.set_ylabel("Cumulative PnL ($)")
    ax.set_title("Equity Curves: v1 vs v2")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 2. v2 PnL distribution
    ax = axes[0, 1]
    if not v2_df.empty:
        pnl_col = "pnl_usd" if "pnl_usd" in v2_df.columns else "net_pnl"
        colors = ["#16a34a" if p > 0 else "#dc2626" for p in v2_df[pnl_col]]
        ax.bar(range(len(v2_df)), v2_df[pnl_col].values, color=colors, width=1.0)
    ax.set_xlabel("Trade #")
    ax.set_ylabel("PnL ($)")
    ax.set_title("v2 Trade PnL")
    ax.axhline(0, c="black", lw=0.5)
    ax.grid(True, alpha=0.3)

    # 3. Model calibration: v1 vs v2
    ax = axes[1, 0]
    for label, col, color in [("v1", "model_v1", "gray"), ("v2", "model_v2", "black")]:
        obs_df["_bucket"] = pd.cut(obs_df[col], bins=np.arange(0, 1.05, 0.1), include_lowest=True)
        cal = obs_df.groupby("_bucket", observed=True).agg(
            mp=(col, "mean"), freq=("outcome", "mean"),
        ).dropna()
        if not cal.empty:
            ax.scatter(cal["mp"], cal["freq"], alpha=0.6, label=label, color=color, s=40)
    ax.plot([0, 1], [0, 1], "r--", alpha=0.5, label="perfect")
    ax.set_xlabel("Predicted P(Up)")
    ax.set_ylabel("Observed P(Up)")
    ax.set_title("Model Calibration")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 4. v2 exit reasons pie
    ax = axes[1, 1]
    if not v2_df.empty and "exit_reason" in v2_df.columns:
        reason_counts = v2_df["exit_reason"].value_counts()
        reason_pnl = v2_df.groupby("exit_reason")["net_pnl"].sum()
        labels = [f"{r}\nn={reason_counts[r]} pnl={reason_pnl[r]:+.2f}" for r in reason_counts.index]
        colors_pie = ["#16a34a" if reason_pnl[r] > 0 else "#dc2626" for r in reason_counts.index]
        ax.pie(reason_counts.values, labels=labels, colors=colors_pie, autopct="%1.0f%%",
               textprops={"fontsize": 8})
    ax.set_title("v2 Exit Reasons")

    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "backtest_v2_results.png", dpi=150)
    plt.close(fig)
    log.info("Plots saved to %s/backtest_v2_results.png", OUTPUT_DIR)

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    log.info("=" * 60)
    log.info("Backtest v2 — Signal Filter Analysis")
    log.info("=" * 60)

    # Load data
    markets = load_markets()
    min_ts = min(m["window_start_ts"] for m in markets)
    max_ts = max(m["window_end_ts"] for m in markets)

    btc_start = datetime.fromtimestamp(min_ts - 7200, tz=timezone.utc)  # 2h buffer for vol+momentum warmup
    btc_end = datetime.fromtimestamp(max_ts + 300, tz=timezone.utc)
    log.info("BTC data: %s → %s", btc_start.isoformat(), btc_end.isoformat())

    btc_df = fetch_btc_1m(btc_start, btc_end)
    log.info("BTC candles: %d", len(btc_df))
    if btc_df.empty:
        log.error("No BTC data!")
        return

    # Build observations
    log.info("Building observations...")
    observations = build_observations(markets, btc_df)
    if not observations:
        log.error("No observations!")
        return

    obs_df = pd.DataFrame(observations)
    log.info("Total observations: %d from %d markets",
             len(obs_df), obs_df["market_id"].nunique())
    log.info("Date range: %s → %s",
             obs_df["window_start"].min(), obs_df["window_start"].max())

    # Run backtests
    log.info("\nRunning v1 backtest (absolute edge > %.2f)...", V1_THRESHOLD)
    v1_df = backtest_v1(observations)
    v1_stats = analyze("v1: Absolute edge > 0.24, hold to resolution", v1_df)

    log.info("\nRunning v2 backtest (all filters)...")
    v2_df = backtest_v2(observations)
    v2_stats = analyze("v2: Relative edge + filters + profit-taking", v2_df)

    # --- Also run v2 without profit-taking for comparison ---
    log.info("\nRunning v2-noexit backtest (filters only, hold to resolution)...")
    old_pt = V2_PROFIT_TARGET
    old_et = V2_EXIT_THRESHOLD
    # Temporarily disable exits
    import backtest_v2 as self_mod
    self_mod.V2_PROFIT_TARGET = 0
    self_mod.V2_EXIT_THRESHOLD = -99
    v2_noexit_df = backtest_v2(observations)
    v2_noexit_stats = analyze("v2-noexit: Filters only, hold to resolution", v2_noexit_df)
    self_mod.V2_PROFIT_TARGET = old_pt
    self_mod.V2_EXIT_THRESHOLD = old_et

    # Comparison
    print(f"\n{'='*60}")
    print("  HEAD-TO-HEAD COMPARISON")
    print(f"{'='*60}")
    print(f"  {'':20s} {'v1':>12s} {'v2':>12s} {'v2-noexit':>12s}")
    print(f"  {'':20s} {'---':>12s} {'---':>12s} {'---':>12s}")
    for key, label in [("n", "Trades"), ("wr", "Win rate"),
                        ("total_usd", "Total PnL ($)"), ("avg_pnl", "Avg PnL"),
                        ("max_dd", "Max drawdown"), ("sharpe", "Sharpe")]:
        v1v = v1_stats.get(key, 0)
        v2v = v2_stats.get(key, 0)
        v2nv = v2_noexit_stats.get(key, 0)
        if key == "wr":
            print(f"  {label:20s} {v1v:>11.1%} {v2v:>11.1%} {v2nv:>11.1%}")
        elif key == "n":
            print(f"  {label:20s} {v1v:>12d} {v2v:>12d} {v2nv:>12d}")
        else:
            print(f"  {label:20s} {v1v:>12.4f} {v2v:>12.4f} {v2nv:>12.4f}")

    # Plots
    make_plots(v1_df, v2_df, obs_df)

    # Save v2 trades
    if not v2_df.empty:
        save_cols = [c for c in v2_df.columns if c not in ["market_id"]]
        v2_df[save_cols].to_csv(OUTPUT_DIR / "backtest_v2_trades.csv", index=False)
        log.info("v2 trades saved to %s/backtest_v2_trades.csv", OUTPUT_DIR)


if __name__ == "__main__":
    main()
