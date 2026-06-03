# Prompt: Build a Live Polymarket BTC 5-Min Trader

## Context

I have a backtested model for trading Polymarket's "Bitcoin Up or Down" 5-minute binary option markets. The backtest is in `/Users/indigo/Code/polymarket/backtest.py` — read it to understand the full system. The backtest ran on 576 real markets across 2 days and produced positive PnL at the optimal threshold (0.24). All output is in `output/`.

## What these markets are

Polymarket runs 24/7 "BTC Up or Down" markets, each covering a 5-minute window. They resolve "Up" if the BTC price at window end >= price at window start (per Chainlink BTC/USD oracle), otherwise "Down". New markets are created every 5 minutes.

### API details

**Discovery** — Each market has slug `btc-updown-5m-{unix_timestamp}` where the timestamp is the window start time (always a multiple of 300). Fetch metadata from:
```
GET https://gamma-api.polymarket.com/events?slug=btc-updown-5m-{ts}
```
Response includes `markets[0].conditionId`, `markets[0].clobTokenIds` (array: [Up_token_id, Down_token_id]), and `markets[0].outcomePrices`.

**Trading** — The Polymarket CLOB (Central Limit Order Book) is at `https://clob.polymarket.com`. Trading requires:
- A Polygon wallet (private key)
- The `py-clob-client` Python package (`pip install py-clob-client`)
- API key/secret/passphrase obtained via the CLOB `/auth/api-key` endpoint
- Signing orders with the wallet

Read the official docs: https://docs.polymarket.com/ and the Python client: https://github.com/Polymarket/py-clob-client

**BTC spot data** — Binance REST API:
```
GET https://api.binance.com/api/v3/klines?symbol=BTCUSDT&interval=1m&limit=120
```

## The pricing model

At each moment during a live 5-minute window:

```
S  = current BTC spot price (Binance)
S0 = BTC price at window start (the reference/strike price)
τ  = minutes remaining until window end
σ  = rolling realized volatility of trailing 60 × 1-min BTC log returns

d2 = (ln(S/S0) - 0.5 × σ² × τ) / (σ × √τ)
fair_P(Up) = Φ(d2)      # standard normal CDF

edge = fair_P(Up) - market_price_of_Up_token
```

Typical σ ≈ 0.00046 (per-minute std of log returns). The model is **well-calibrated** — see `output/calibration.png`.

## Trading rules

- **Threshold**: 0.24 — only trade when |edge| > 0.24
- If `edge > 0.24`: buy the Up token
- If `edge < -0.24`: buy the Down token (equivalent: the model says Up is overpriced)
- **Position size**: start with a fixed small size (e.g. $2 per trade) for virtual/paper mode
- **Fee**: Polymarket charges 2% on net winnings
- **One trade per market maximum** — don't average in

## What to build

### Phase 1: Paper trading (virtual)

Build `live_trader.py` that:

1. **Runs continuously** in a loop, checking for the current and upcoming 5-minute windows
2. **Fetches the current market** from Gamma API using the slug pattern
3. **Polls BTC price** from Binance every ~5 seconds during the active window
4. **Computes fair price** using the GBM model above, with trailing 60-min realized vol
5. **Computes edge** vs the Polymarket mid-price (from Gamma `outcomePrices` or from the CLOB orderbook)
6. **Logs virtual trades** when |edge| > 0.24 — record timestamp, market slug, side (Up/Down), entry price, model price, edge, and BTC spot
7. **Tracks resolution** — after each window closes, check if Up or Down won (query Gamma API), compute PnL
8. **Displays a live dashboard** in the terminal — current window, BTC price, model P(Up), market P(Up), edge, position, running PnL
9. **Saves a trade log** to CSV

### Phase 2: Real trading

Extend `live_trader.py` with a `--live` flag that:

1. Uses `py-clob-client` to authenticate with the CLOB
2. Places **limit orders** at the model's fair price (not market orders — we want to capture the edge, not give it away)
3. Manages order lifecycle — place, check fills, cancel unfilled orders before window closes
4. Handles errors gracefully — network failures, API rate limits, partial fills
5. Has a **kill switch** — if cumulative loss exceeds a configurable max (e.g. $50), stop trading
6. Logs everything

### Architecture

- Main loop: wake every 5 seconds
- Each iteration: determine which 5-min window is currently active, fetch BTC spot, compute model, check for trade signals
- Use async or threading for non-blocking API calls
- The BTC spot price cache should be a rolling buffer of the last 120 minutes of 1-min candles for vol computation
- Store the Polymarket mid-price by querying the CLOB orderbook (best bid + best ask / 2), not just the Gamma static prices

### Configuration

```python
THRESHOLD = 0.24
TRADE_SIZE = 2.00          # USD per trade (paper mode)
MAX_LOSS = 50.00           # kill switch
VOL_LOOKBACK = 60          # minutes
POLL_INTERVAL = 5          # seconds between checks
```

### Key files in this project

- `backtest.py` — the full backtesting system (read this first)
- `output/observations.csv` — 3,798 observation points with model vs market prices
- `output/calibration.png` — proves the model is well-calibrated
- `output/threshold_sweep.png` — shows threshold=0.24 is optimal
- `cache/` — cached API responses from the backtest

## Important notes

- The Polymarket CLOB requires a Polygon (MATIC) wallet with USDC for trading. I will provide wallet credentials when we go live — DO NOT hardcode any keys.
- For paper trading mode, no wallet is needed — just read market prices and log virtual trades.
- The market becomes active (tradeable) roughly 5-10 minutes before the window opens. Most volume happens during the window itself.
- Resolution is based on Chainlink oracle prices, not Binance. There may be small discrepancies.
- Start with paper trading. I will explicitly tell you when to switch to real money.
