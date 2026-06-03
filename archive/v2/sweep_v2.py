#!/usr/bin/env python3
"""
Parameter sweep: threshold × profit_target for v2 model.
Uses cached data from backtest.py (576 markets, 2 days).
"""

import sys
import json
import re
import statistics
import logging
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

CACHE_DIR = Path("cache")
OUTPUT_DIR = Path("output")

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

# Fixed v2 params (structural filters — not swept)
MIN_ELAPSED = 2.0
MIN_MODEL_DEV = 0.05
ENTRY_PRICE_MIN = 0.10
ENTRY_PRICE_MAX = 0.90
MAX_VOL = 0.0010
EXIT_THRESHOLD = -0.10
KELLY_FRACTION = 0.5
MOMENTUM_LOOKBACK = 5
VOL_EWMA_HALFLIFE = 10
VOL_LOOKBACK = 60
BUCKET_SECONDS = 30
WINDOW_SECS = 300
FEE_RATE = 0.02
TRADE_SIZE = 2.00

# Sweep grid
THRESHOLDS = [0.20, 0.30, 0.40, 0.50, 0.60, 0.70]
PROFIT_TARGETS = [0.0, 0.25, 0.50, 0.75, 1.00]  # 0 = hold to resolution

import requests
SESSION = requests.Session()
SESSION.headers.update({"Accept": "application/json"})

BINANCE = "https://api.binance.com/api/v3"

def _cache_path(name):
    return CACHE_DIR / re.sub(r'[^\w\-.]', '_', name)

def _load_cache(name):
    p = _cache_path(name)
    if p.exists():
        with open(p) as f:
            return json.load(f)
    return None

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
                return None
            time.sleep(backoff * (2 ** attempt))
    return None

def load_markets():
    cached = _load_cache("btc_5m_markets_fast_2d.json")
    if not cached:
        sys.exit("No cached markets. Run backtest.py first.")
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
    return _load_cache(cache_name) or []

def fair_price_v2(S, S0, tau_min, sigma, drift=0.0):
    if tau_min <= 0:
        return 1.0 if S >= S0 else 0.0
    if sigma <= 0 or S <= 0 or S0 <= 0:
        return 0.5
    vol = sigma * np.sqrt(tau_min)
    d2 = (np.log(S / S0) + drift * tau_min - 0.5 * sigma**2 * tau_min) / vol
    return float(np.clip(norm.cdf(d2), 0.001, 0.999))

def compute_ewma_vol(log_rets, halflife=VOL_EWMA_HALFLIFE):
    alpha = 1.0 - 0.5 ** (1.0 / halflife)
    n = len(log_rets)
    result = np.full(n, np.nan)
    var_est = log_rets[:20].var() if len(log_rets) >= 20 else 0.00046**2
    for i in range(n):
        r = log_rets.iloc[i]
        if np.isnan(r):
            continue
        var_est = alpha * r**2 + (1 - alpha) * var_est
        result[i] = var_est
    return pd.Series(np.sqrt(np.maximum(result, 1e-14)), index=log_rets.index)

# ---------------------------------------------------------------------------
# Build observations once, then sweep parameters
# ---------------------------------------------------------------------------

def build_observations(markets, btc_df):
    btc_df = btc_df.copy()
    btc_df["ts_min"] = btc_df["timestamp"].dt.floor("min")
    btc_lookup = dict(zip(btc_df["ts_min"], btc_df["close"]))

    log_ret = np.log(btc_df["close"] / btc_df["close"].shift(1))
    ewma_vol_s = compute_ewma_vol(log_ret)
    ewma_vol_lookup = dict(zip(btc_df["ts_min"], ewma_vol_s))
    fallback_vol = ewma_vol_s.dropna().median()

    mom_s = np.log(btc_df["close"] / btc_df["close"].shift(MOMENTUM_LOOKBACK)) / MOMENTUM_LOOKBACK
    mom_lookup = dict(zip(btc_df["ts_min"], mom_s))

    all_obs = []
    for mkt in markets:
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

            sigma = ewma_vol_lookup.get(obs_dt, fallback_vol)
            if sigma is None or np.isnan(sigma) or sigma <= 0:
                sigma = fallback_vol
            drift = mom_lookup.get(obs_dt, 0.0)
            if drift is None or np.isnan(drift):
                drift = 0.0

            model = fair_price_v2(S, S0, tau, sigma, drift)

            all_obs.append({
                "market_id": cid,
                "window_start_ts": ws,
                "bk_ts": bk_ts,
                "market_price": market_price,
                "model": model,
                "edge": model - market_price,
                "outcome": outcome,
                "S": S, "S0": S0,
                "sigma": sigma, "drift": drift,
                "tau_min": tau,
                "elapsed_min": (WINDOW_SECS / 60.0) - tau,
            })

    log.info("Built %d observations from %d markets", len(all_obs), len(markets))
    return all_obs


def run_sweep_single(observations, threshold, profit_target):
    """Run one parameter combination."""
    by_window = defaultdict(list)
    for obs in observations:
        by_window[obs["window_start_ts"]].append(obs)
    for ws in by_window:
        by_window[ws].sort(key=lambda o: o["bk_ts"])

    trades = []
    for ws_ts, obs_list in sorted(by_window.items()):
        if len(obs_list) < 2:
            continue

        # Find entry
        entry = None
        entry_obs = None
        entry_idx = None

        for idx, obs in enumerate(obs_list):
            mp = obs["market_price"]
            model = obs["model"]
            edge = obs["edge"]
            elapsed = obs["elapsed_min"]
            sigma = obs["sigma"]
            drift = obs["drift"]

            if edge > 0 and mp > 0.01:
                rel_edge = edge / mp
                side = "BUY_UP"
            elif edge < 0 and mp < 0.99:
                rel_edge = (-edge) / (1.0 - mp)
                side = "BUY_DOWN"
            else:
                continue

            if rel_edge <= threshold:
                continue
            if elapsed < MIN_ELAPSED:
                continue
            if abs(model - 0.5) < MIN_MODEL_DEV:
                continue
            if mp < ENTRY_PRICE_MIN or mp > ENTRY_PRICE_MAX:
                continue
            if MAX_VOL > 0 and sigma > MAX_VOL:
                continue
            if side == "BUY_UP" and drift <= 0:
                continue
            if side == "BUY_DOWN" and drift >= 0:
                continue

            # Kelly
            if side == "BUY_UP":
                kelly_raw = (model - mp) / (1.0 - mp)
            else:
                kelly_raw = (mp - model) / mp
            kelly_bet = max(0.25, min(kelly_raw * KELLY_FRACTION, 1.0))

            entry = {"side": side, "entry_price": mp, "kelly": kelly_bet}
            entry_obs = obs
            entry_idx = idx
            break

        if entry is None:
            continue

        side = entry["side"]
        ep = entry["entry_price"]
        cost = ep if side == "BUY_UP" else (1.0 - ep)
        t_size = round(TRADE_SIZE * entry["kelly"], 2)
        exit_reason = None

        # Scan for exit
        if profit_target > 0:
            for obs in obs_list[entry_idx + 1:]:
                mp_now = obs["market_price"]
                model_now = obs["model"]
                cv = mp_now if side == "BUY_UP" else (1.0 - mp_now)
                re_ = (model_now - mp_now) if side == "BUY_UP" else (mp_now - model_now)
                unrealized_pct = (cv - cost) / cost if cost > 0.001 else 0

                if unrealized_pct >= profit_target:
                    gross = cv - cost
                    fee = max(0, gross) * FEE_RATE
                    net = gross - fee
                    exit_reason = "profit_target"
                    trades.append({"net_pnl": net, "pnl_usd": net * t_size,
                                   "exit_reason": exit_reason, "side": side})
                    break
                if re_ < EXIT_THRESHOLD:
                    gross = cv - cost
                    fee = max(0, gross) * FEE_RATE
                    net = gross - fee
                    exit_reason = "edge_reversed"
                    trades.append({"net_pnl": net, "pnl_usd": net * t_size,
                                   "exit_reason": exit_reason, "side": side})
                    break

        if exit_reason is None:
            outcome = entry_obs["outcome"]
            if side == "BUY_UP":
                won = outcome == 1
                gross = (1.0 - ep) if won else -ep
            else:
                won = outcome == 0
                gross = ep if won else -(1.0 - ep)
            fee = max(0, gross) * FEE_RATE
            net = gross - fee
            trades.append({"net_pnl": net, "pnl_usd": net * t_size,
                           "exit_reason": "resolution", "side": side})

    if not trades:
        return {"n": 0, "wr": 0, "total_usd": 0, "avg_pnl": 0,
                "max_dd": 0, "sharpe": 0, "pnl_per_trade_usd": 0}

    df = pd.DataFrame(trades)
    n = len(df)
    wins = (df["net_pnl"] > 0).sum()
    cum = df["pnl_usd"].cumsum()
    total_usd = df["pnl_usd"].sum()
    std = df["pnl_usd"].std()

    return {
        "n": n,
        "wr": wins / n,
        "total_usd": total_usd,
        "pnl_per_trade_usd": total_usd / n,
        "avg_pnl": df["net_pnl"].mean(),
        "max_dd": (cum - cum.cummax()).min(),
        "sharpe": df["pnl_usd"].mean() / std if std > 0 else 0,
    }


def main():
    log.info("Parameter sweep: threshold × profit_target")

    markets = load_markets()
    min_ts = min(m["window_start_ts"] for m in markets)
    max_ts = max(m["window_end_ts"] for m in markets)
    btc_df = fetch_btc_1m(
        datetime.fromtimestamp(min_ts - 7200, tz=timezone.utc),
        datetime.fromtimestamp(max_ts + 300, tz=timezone.utc),
    )
    log.info("BTC candles: %d", len(btc_df))

    observations = build_observations(markets, btc_df)

    # Run sweep
    results = []
    for thresh in THRESHOLDS:
        for pt in PROFIT_TARGETS:
            r = run_sweep_single(observations, thresh, pt)
            r["threshold"] = thresh
            r["profit_target"] = pt
            results.append(r)
            pt_label = f"{pt:.0%}" if pt > 0 else "hold"
            log.info("  thresh=%.2f  pt=%s  n=%d  wr=%.0f%%  pnl=$%.2f  sharpe=%.3f",
                     thresh, pt_label, r["n"], r["wr"] * 100, r["total_usd"], r["sharpe"])

    df = pd.DataFrame(results)
    df.to_csv(OUTPUT_DIR / "sweep_v2.csv", index=False)

    # --- Print table ---
    print(f"\n{'='*90}")
    print("PARAMETER SWEEP: threshold × profit_target")
    print(f"{'='*90}")
    print(f"{'Thresh':>8s} {'ProfTgt':>8s} {'Trades':>7s} {'WinRate':>8s} "
          f"{'PnL($)':>9s} {'$/Trade':>9s} {'MaxDD($)':>9s} {'Sharpe':>8s}")
    print("-" * 90)
    for _, r in df.iterrows():
        pt_label = f"{r['profit_target']:.0%}" if r["profit_target"] > 0 else "hold"
        wr = f"{r['wr']:.0%}" if r["n"] > 0 else "—"
        sh = f"{r['sharpe']:.3f}" if r["n"] > 0 else "—"
        print(f"{r['threshold']:>8.2f} {pt_label:>8s} {int(r['n']):>7d} {wr:>8s} "
              f"{r['total_usd']:>+9.2f} {r['pnl_per_trade_usd']:>+9.4f} "
              f"{r['max_dd']:>9.4f} {sh:>8s}")

    # --- Best by Sharpe ---
    viable = df[df["n"] >= 20].copy()
    if not viable.empty:
        best = viable.loc[viable["sharpe"].idxmax()]
        print(f"\nBest by Sharpe (n>=20):")
        pt_label = f"{best['profit_target']:.0%}" if best["profit_target"] > 0 else "hold"
        print(f"  threshold={best['threshold']:.2f}  profit_target={pt_label}")
        print(f"  n={int(best['n'])}  wr={best['wr']:.0%}  pnl=${best['total_usd']:+.2f}  "
              f"sharpe={best['sharpe']:.3f}")

        best_pnl = viable.loc[viable["total_usd"].idxmax()]
        print(f"\nBest by Total PnL (n>=20):")
        pt_label = f"{best_pnl['profit_target']:.0%}" if best_pnl["profit_target"] > 0 else "hold"
        print(f"  threshold={best_pnl['threshold']:.2f}  profit_target={pt_label}")
        print(f"  n={int(best_pnl['n'])}  wr={best_pnl['wr']:.0%}  pnl=${best_pnl['total_usd']:+.2f}  "
              f"sharpe={best_pnl['sharpe']:.3f}")

    # --- Heatmaps ---
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    for ax, metric, title, fmt in [
        (axes[0], "sharpe", "Sharpe Ratio", ".3f"),
        (axes[1], "total_usd", "Total PnL ($)", ".1f"),
        (axes[2], "wr", "Win Rate", ".0%"),
    ]:
        pivot = df.pivot(index="threshold", columns="profit_target", values=metric)
        pivot.columns = [f"{c:.0%}" if c > 0 else "hold" for c in pivot.columns]
        im = ax.imshow(pivot.values, cmap="RdYlGn", aspect="auto")
        ax.set_xticks(range(len(pivot.columns)))
        ax.set_xticklabels(pivot.columns, fontsize=9)
        ax.set_yticks(range(len(pivot.index)))
        ax.set_yticklabels([f"{v:.2f}" for v in pivot.index], fontsize=9)
        ax.set_xlabel("Profit Target")
        ax.set_ylabel("Threshold")
        ax.set_title(title)
        for i in range(len(pivot.index)):
            for j in range(len(pivot.columns)):
                val = pivot.values[i, j]
                if metric == "wr":
                    txt = f"{val:.0%}"
                elif metric == "total_usd":
                    txt = f"${val:.0f}"
                else:
                    txt = f"{val:.2f}"
                ax.text(j, i, txt, ha="center", va="center", fontsize=8,
                        color="white" if abs(val) > abs(pivot.values).max() * 0.7 else "black")
        plt.colorbar(im, ax=ax, shrink=0.8)

    fig.suptitle("v2 Parameter Sweep: threshold × profit_target (576 markets, 2 days)", fontsize=12)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "sweep_v2_heatmap.png", dpi=150)
    plt.close(fig)
    log.info("Heatmap saved to %s/sweep_v2_heatmap.png", OUTPUT_DIR)


if __name__ == "__main__":
    main()
