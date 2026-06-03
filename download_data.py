#!/usr/bin/env python3
"""
Bulk Polymarket Data Downloader
================================
Downloads BTC 5-min Up/Down market data incrementally, day by day.
Each day is cached separately so reruns only fetch what's missing.
Merges all day files into one combined market list for backtest.py.

Usage:
    python download_data.py            # last 14 days
    python download_data.py --days 30  # last 30 days
"""

import argparse
import json
import logging
import re
import sys
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path

import requests

CACHE_DIR = Path("cache")
CACHE_DIR.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)

GAMMA_BASE = "https://gamma-api.polymarket.com"
DATA_API   = "https://data-api.polymarket.com"
BINANCE    = "https://api.binance.com/api/v3"

SESSION = requests.Session()
SESSION.headers.update({"Accept": "application/json"})


# ---------------------------------------------------------------------------
# HTTP helpers
# ---------------------------------------------------------------------------

def _get(url, params=None, retries=3, backoff=1.0):
    for attempt in range(retries):
        try:
            r = SESSION.get(url, params=params, timeout=30)
            if r.status_code == 429:
                wait = backoff * (2 ** attempt)
                log.debug("Rate limited, sleeping %.1fs", wait)
                time.sleep(wait)
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
    return CACHE_DIR / re.sub(r"[^\w\-.]", "_", name)


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
# Step 1: Discover markets — one cache file per calendar day
# ---------------------------------------------------------------------------

def _discover_day(date: datetime) -> list:
    """
    Discover all closed 5-min BTC markets for one UTC calendar day.
    Cache key: markets_day_YYYYMMDD.json
    """
    date_str = date.strftime("%Y%m%d")
    cache_name = f"markets_day_{date_str}.json"
    cached = _load_cache(cache_name)
    if cached is not None:
        log.info("  %s: loaded %d markets from cache", date_str, len(cached))
        return cached

    # Generate all 5-min slots for this UTC day
    day_start = date.replace(hour=0, minute=0, second=0, microsecond=0)
    day_end   = day_start + timedelta(days=1)
    timestamps = list(range(int(day_start.timestamp()), int(day_end.timestamp()), 300))

    markets = []
    for i, ts in enumerate(timestamps):
        slug = f"btc-updown-5m-{ts}"
        data = _get(f"{GAMMA_BASE}/events", params={"slug": slug})
        if not data or not isinstance(data, list) or len(data) == 0:
            time.sleep(0.05)
            continue

        ev = data[0]
        if not ev.get("closed"):
            time.sleep(0.05)
            continue

        mk = ev.get("markets", [{}])[0] if ev.get("markets") else {}
        if not mk.get("conditionId"):
            time.sleep(0.05)
            continue

        prices_raw = mk.get("outcomePrices", "[]")
        if isinstance(prices_raw, str):
            try:
                prices_raw = json.loads(prices_raw)
            except Exception:
                prices_raw = []

        clob_ids = mk.get("clobTokenIds", "[]")
        if isinstance(clob_ids, str):
            try:
                clob_ids = json.loads(clob_ids)
            except Exception:
                clob_ids = []

        up_won = float(prices_raw[0]) > 0.5 if len(prices_raw) >= 2 else None

        markets.append({
            "slug": slug,
            "title": ev.get("title", ""),
            "condition_id": mk["conditionId"],
            "window_start_ts": ts,
            "window_end_ts": ts + 300,
            "up_won": up_won,
            "up_token_id": clob_ids[0] if clob_ids else "",
            "down_token_id": clob_ids[1] if len(clob_ids) > 1 else "",
        })

        time.sleep(0.05)

    log.info("  %s: discovered %d closed markets (of %d slots)",
             date_str, len(markets), len(timestamps))
    _save_cache(cache_name, markets)
    return markets


def discover_markets(num_days: int) -> list:
    """
    Discover closed markets for the last `num_days` calendar days.
    Each day is cached separately; returns a merged list sorted by window_start_ts.
    """
    now = datetime.now(timezone.utc)
    # Yesterday is the last fully-closed day; go back num_days from today
    days = []
    for d in range(num_days):
        day = (now - timedelta(days=d+1)).replace(
            hour=0, minute=0, second=0, microsecond=0
        )
        days.append(day)
    days.reverse()  # oldest first

    log.info("Discovering markets for %d days (%s to %s)",
             num_days,
             days[0].strftime("%Y-%m-%d"),
             days[-1].strftime("%Y-%m-%d"))

    all_markets = []
    for day in days:
        day_markets = _discover_day(day)
        all_markets.extend(day_markets)

    all_markets.sort(key=lambda m: m["window_start_ts"])
    log.info("Total: %d markets discovered", len(all_markets))
    return all_markets


# ---------------------------------------------------------------------------
# Step 2: Prefetch trades for all markets
# ---------------------------------------------------------------------------

def prefetch_trades(markets: list) -> dict:
    """
    Fetch and cache trades for every market. Returns summary stats.
    Markets whose trades are already cached are skipped instantly.
    """
    total = len(markets)
    fetched = 0
    skipped = 0
    empty   = 0

    log.info("Prefetching trades for %d markets...", total)

    for i, mkt in enumerate(markets):
        cid = mkt["condition_id"]
        cache_name = f"trades5m_{cid[:20]}.json"

        if _load_cache(cache_name) is not None:
            skipped += 1
            if (i + 1) % 200 == 0:
                log.info("  [%d/%d] %d fetched, %d cached, %d empty",
                         i + 1, total, fetched, skipped, empty)
            continue

        # Fetch up to 4 pages (4×500 = 2000 trades max)
        all_trades = []
        for pg in range(4):
            data = _get(f"{DATA_API}/trades", params={
                "market": cid, "limit": 500, "offset": pg * 500,
            }, retries=2, backoff=0.5)
            if not data or not isinstance(data, list) or len(data) == 0:
                break
            all_trades.extend(data)
            if len(data) < 500:
                break
            time.sleep(0.05)

        _save_cache(cache_name, all_trades)

        if len(all_trades) == 0:
            empty += 1
        else:
            fetched += 1

        if (i + 1) % 50 == 0:
            log.info("  [%d/%d] %d fetched, %d cached, %d empty",
                     i + 1, total, fetched, skipped, empty)

        time.sleep(0.02)

    log.info("Trades done: %d newly fetched, %d already cached, %d empty",
             fetched, skipped, empty)
    return {"fetched": fetched, "skipped": skipped, "empty": empty}


# ---------------------------------------------------------------------------
# Step 3: Prefetch BTC 1-minute candles
# ---------------------------------------------------------------------------

def prefetch_btc(markets: list):
    """Fetch BTC/USDT 1-min candles for the full market date range."""
    if not markets:
        return

    min_ts = min(m["window_start_ts"] for m in markets)
    max_ts = max(m["window_end_ts"]   for m in markets)

    # Add 1-hour warmup buffer for vol calculation
    start_ms = (min_ts - 3600) * 1000
    end_ms   = (max_ts + 300)  * 1000

    cache_name = f"btc1m_{start_ms}_{end_ms}.json"
    if _load_cache(cache_name) is not None:
        log.info("BTC 1m data already cached (%s)", cache_name)
        return

    log.info("Fetching BTC/USDT 1m candles from %s to %s",
             datetime.fromtimestamp(min_ts - 3600, tz=timezone.utc).strftime("%Y-%m-%d %H:%M"),
             datetime.fromtimestamp(max_ts + 300,  tz=timezone.utc).strftime("%Y-%m-%d %H:%M"))

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
    log.info("Fetched %d minutes of BTC/USDT 1m candles", len(klines))


# ---------------------------------------------------------------------------
# Step 4: Write merged market file for backtest.py
# ---------------------------------------------------------------------------

def write_merged(markets: list, num_days: int):
    """Write a combined market file that backtest.py can load directly."""
    merged_name = f"btc_5m_markets_fast_{num_days}d.json"
    _save_cache(merged_name, markets)
    log.info("Wrote merged file: cache/%s (%d markets)", merged_name, len(markets))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Download Polymarket data in bulk")
    parser.add_argument("--days", type=int, default=14,
                        help="Number of days of history to download (default: 14)")
    parser.add_argument("--skip-trades", action="store_true",
                        help="Skip trade prefetch (just do market discovery + BTC)")
    args = parser.parse_args()

    num_days = args.days

    log.info("=" * 60)
    log.info("Polymarket Bulk Data Downloader — %d days", num_days)
    log.info("=" * 60)

    # Step 1: Discover markets
    log.info("\n=== STEP 1: Market Discovery ===")
    markets = discover_markets(num_days)

    if not markets:
        log.error("No markets found — check API connectivity")
        sys.exit(1)

    up_n = sum(1 for m in markets if m.get("up_won"))
    log.info("Markets: %d total, %d Up / %d Down resolutions",
             len(markets), up_n, len(markets) - up_n)

    # Step 2: Prefetch trades
    if not args.skip_trades:
        log.info("\n=== STEP 2: Prefetching Trades ===")
        prefetch_trades(markets)

    # Step 3: Prefetch BTC data
    log.info("\n=== STEP 3: Prefetching BTC 1m Candles ===")
    prefetch_btc(markets)

    # Step 4: Write merged file for backtest.py
    log.info("\n=== STEP 4: Writing Merged Cache ===")
    write_merged(markets, num_days)

    log.info("\nDone! Run: python backtest.py --days %d", num_days)


if __name__ == "__main__":
    main()
