# Bitcoin 5-Minute Up/Down Trader

A systematic trading bot for [Polymarket](https://polymarket.com) BTC 5-minute Up/Down prediction markets. Prices each market using a Geometric Brownian Motion (GBM) fair-value model calibrated with live Binance volatility data, then trades when a sufficient edge exists between the model and market prices.

## How it works

Polymarket runs continuous "Bitcoin Up or Down" markets, each covering a 5-minute window. A market resolves **Up** if BTC is at or above its opening price at the end of the window, and **Down** otherwise. These markets are listed every 5 minutes, 24/7.

### Pricing model

The fair probability of an Up resolution is computed using the GBM formula:

```
P(Up) = Φ(d₂)
d₂ = (ln(S / S₀) − ½σ²τ) / (σ√τ)
```

Where:
- `S` — current BTC spot price (from Binance)
- `S₀` — BTC price at window open (the "strike")
- `τ` — minutes remaining in the window
- `σ` — EWMA realized 1-minute volatility of BTC log returns
- A short-term momentum drift term is added to capture mean-reversion/trend effects

### Signal generation

The bot computes the **relative edge** — how much the model price deviates from the market price, expressed as a fraction of the token cost:

```
rel_edge = (model_price − market_price) / market_price   [for BUY_UP]
rel_edge = (market_price − model_price) / (1 − market_price)  [for BUY_DOWN]
```

A trade fires when `rel_edge > THRESHOLD` (default 0.24), subject to several filters:

| Filter | Default | Purpose |
|---|---|---|
| `MIN_ELAPSED_MINUTES` | 1.5 min | Skip the first N minutes of each window when the model is near 0.50 |
| `MIN_MODEL_DEVIATION` | 0.05 | Model must have real conviction (>5% from 0.50) |
| `ENTRY_PRICE_MIN/MAX` | 0.10 / 0.90 | Avoid deep OTM/ITM tokens |
| Momentum confirmation | — | Momentum drift must agree with trade direction |

### Exit logic (v3)

Trades are managed with three exit triggers, evaluated every 500ms:

1. **Dynamic profit target** — `PT = clip(PT_MULTIPLIER × rel_edge, PT_FLOOR, PT_CEILING)`. Scales the take-profit with entry conviction: small-edge trades exit quickly, large-edge trades can ride.
2. **Stop loss** — exits if unrealized return drops below `−STOP_LOSS_PCT × cost` (default −50%).
3. **Time exit** — exits at market after `MAX_HOLD_SECONDS` (default 70s) regardless of P&L.

If neither PT nor SL fires, the trade resolves at window close (binary 0 or 1 payout).

### Position sizing

Optional **half-Kelly** sizing scales trade size with estimated edge:

```
kelly_bet = clip(kelly_raw × KELLY_FRACTION, 0.25, 1.0)
```

Set `--kelly-fraction 0` for fixed-size trades.

---

## Scripts

| Script | Purpose |
|---|---|
| `live_trader_v3.py` | Main trading bot — paper or live via Polymarket CLOB |
| `backtest.py` | Historical backtest over real Polymarket trade data |
| `download_data.py` | Bulk market data downloader (run before backtesting) |
| `dashboard.py` | FastAPI web dashboard — monitors the bot and sends control commands |

---

## Quickstart

### 1. Install dependencies

```bash
pip install numpy scipy requests pandas matplotlib fastapi uvicorn
# For live trading only:
pip install py-clob-client
```

### 2. Download historical data (for backtesting)

```bash
python download_data.py            # last 14 days
python download_data.py --days 30  # last 30 days
```

Data is cached in `cache/` so subsequent runs skip already-fetched markets.

### 3. Run the backtester

```bash
python backtest.py
python backtest.py --days 7
```

Outputs to `output/`:
- `observations.csv` — every model vs market price snapshot
- `summary_table.csv` — P&L table across threshold values
- `equity_curves.png`, `calibration.png`, `edge_distribution.png`, `edge_vs_pnl.png`

### 4. Paper trade (default)

```bash
python live_trader_v3.py
```

Runs entirely in paper mode — no real money, no wallet required. Logs virtual trades to `output/live_trades_v3.csv` and observations to `output/live_observations_v3.csv`.

### 5. Live trade via Polymarket CLOB

Set environment variables for your Polygon wallet:

```bash
export POLY_PRIVATE_KEY="0x..."
export POLY_CHAIN_ID=137          # Polygon mainnet (default)
# Optional: auto-derived from wallet if omitted
export POLY_API_KEY="..."
export POLY_API_SECRET="..."
export POLY_API_PASSPHRASE="..."
```

Then:

```bash
python live_trader_v3.py --live
```

The bot checks your USDC balance before starting and refuses to run if it's insufficient.

### 6. Web dashboard

```bash
uvicorn dashboard:app --port 8050
```

Open `http://localhost:8050` to monitor live P&L, observations, and trade history. Supports pause/resume and live parameter updates (threshold, trade size, max loss) via the control API.

---

## Configuration

All parameters can be overridden from the command line:

```
python live_trader_v3.py \
  --threshold 0.30       # relative edge threshold (default: 0.24)
  --size 5.00            # USD per trade (default: 2.00)
  --max-loss 100.00      # kill switch (default: 50.00)
  --kelly-fraction 0.5   # half-Kelly sizing (default: 0.5; 0 = fixed)
  --min-elapsed 2.0      # min minutes into window before entry (default: 1.5)
  --min-model-dev 0.05   # min model deviation from 0.50 (default: 0.05)
  --pt-multiplier 1.0    # dynamic PT multiplier (default: 1.0)
  --pt-floor 0.30        # minimum profit target (default: 0.30)
  --pt-ceiling 1.00      # maximum profit target (default: 1.00)
  --stop-loss 0.50       # stop loss fraction of cost (default: 0.50)
  --max-hold 70          # max hold time in seconds (default: 70)
```

Key hardcoded constants in `live_trader_v3.py`:

| Constant | Default | Description |
|---|---|---|
| `VOL_LOOKBACK` | 60 min | Trailing window for realized vol |
| `VOL_EWMA_HALFLIFE` | 10 min | EWMA halflife for volatility |
| `MOMENTUM_LOOKBACK` | 5 min | Window for momentum drift |
| `MOMENTUM_EWMA_HALFLIFE` | 1.0 min | EWMA halflife for momentum |
| `DRIFT_SCALE` | 0.25 | Attenuates momentum's contribution to fair price |
| `POLL_INTERVAL` | 0.5 s | Main loop frequency |
| `MIN_HOLD_SECONDS` | 120 s | Minimum hold before PT/SL exit |
| `FEE_RATE` | 2% | Polymarket taker fee |

---

## Output files

All output is written to `output/` (excluded from git):

| File | Description |
|---|---|
| `live_trades_v3.csv` | Every trade with entry/exit prices, P&L, exit reason |
| `live_observations_v3.csv` | Per-tick model/market/edge log |
| `live_trader_v3.log` | Full bot log |
| `status.json` | Real-time bot state (read by dashboard) |
| `control.json` | Control commands written by dashboard, consumed by bot |

---

## Architecture

```
Binance API ──────────┐
                      ▼
              BTCFeed (spot + EWMA vol + momentum)
                      │
Gamma API ────────────┤
CLOB API ─────────────┤
                      ▼
              fair_price_up() → edge → signal
                      │
              [filters: elapsed, model_dev, price_range, momentum]
                      │
              [position sizing: half-Kelly]
                      │
              ┌───────┴───────┐
              │               │
           Paper          ClobTrader
           (CSV log)      (limit orders)
              │               │
              └───────┬───────┘
                      │
              [v3 exits: PT / SL / time]
                      │
              CSV append + status.json
                      │
              dashboard.py (FastAPI)
```

---

## Archived versions

`archive/` contains earlier iterations:

- `v1/` — original live trader
- `v2/` — added EWMA vol and momentum; removed MAX_VOL filter

---

## Disclaimer

This is experimental research software. Prediction market trading involves real financial risk. Paper trade thoroughly before enabling `--live`. Past backtest performance does not guarantee future results.
