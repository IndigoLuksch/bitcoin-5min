#!/usr/bin/env python3
"""
Dashboard for the Polymarket BTC 5-Min Trader.

Reads CSVs and status files written by live_trader.py.
Controls the bot via a shared control.json file.

Usage:
    python dashboard.py              # starts on http://localhost:8050
    python dashboard.py --port 9000  # custom port
"""

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse

OUTPUT_DIR = Path("output")
TRADE_CSV = OUTPUT_DIR / "live_trades_v3.csv"
OBS_CSV = OUTPUT_DIR / "live_observations_v3.csv"
STATUS_FILE = OUTPUT_DIR / "status.json"
CONTROL_FILE = OUTPUT_DIR / "control.json"
LOG_FILE = OUTPUT_DIR / "live_trader_v3.log"

app = FastAPI()


def read_trades():
    if not TRADE_CSV.exists():
        return []
    with open(TRADE_CSV) as f:
        return list(csv.DictReader(f))


def read_observations():
    if not OBS_CSV.exists():
        return []
    with open(OBS_CSV) as f:
        return list(csv.DictReader(f))


def read_status():
    if not STATUS_FILE.exists():
        return None
    try:
        with open(STATUS_FILE) as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return None


def send_control(data):
    with open(CONTROL_FILE, "w") as f:
        json.dump(data, f)


# ── API endpoints ──────────────────────────────────────────────────────────

@app.get("/api/status")
def api_status():
    status = read_status()
    if not status:
        return JSONResponse({"running": False})
    status["running"] = True
    # Check staleness — if status is older than 30s, bot is likely down
    ts = status.get("timestamp")
    if ts:
        try:
            age = (datetime.now(timezone.utc) - datetime.fromisoformat(ts)).total_seconds()
            if age > 30:
                status["running"] = False
                status["stale_seconds"] = round(age)
        except (ValueError, TypeError):
            pass
    return status


@app.get("/api/trades")
def api_trades():
    return read_trades()


@app.get("/api/observations")
def api_observations():
    rows = read_observations()
    # Downsample if huge — send last 2000 points
    if len(rows) > 2000:
        rows = rows[-2000:]
    return rows


@app.get("/api/log")
def api_log():
    if not LOG_FILE.exists():
        return {"lines": []}
    lines = LOG_FILE.read_text().splitlines()
    return {"lines": lines[-200:]}


@app.post("/api/control/{command}")
def api_control(command: str, threshold: float = None, trade_size: float = None, max_loss: float = None):
    data = {"command": command}
    if threshold is not None:
        data["threshold"] = threshold
    if trade_size is not None:
        data["trade_size"] = trade_size
    if max_loss is not None:
        data["max_loss"] = max_loss
    send_control(data)
    return {"ok": True, "sent": data}


@app.post("/api/update_params")
def api_update_params(threshold: float = None, trade_size: float = None, max_loss: float = None):
    data = {}
    if threshold is not None:
        data["threshold"] = threshold
    if trade_size is not None:
        data["trade_size"] = trade_size
    if max_loss is not None:
        data["max_loss"] = max_loss
    if data:
        send_control(data)
    return {"ok": True, "sent": data}


# ── Frontend ───────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
def index():
    return HTML_PAGE


HTML_PAGE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>BTC 5-Min Trader v3</title>
<style>
  :root {
    --bg: #ffffff; --surface: #f7f7f5; --border: #e9e9e7;
    --text: #111111; --muted: #9b9b9b; --green: #16a34a; --red: #dc2626;
  }
  * { margin:0; padding:0; box-sizing:border-box; }
  body { font-family: system-ui, -apple-system, 'Segoe UI', sans-serif; background:var(--bg); color:var(--text); font-size:14px; }
  .container { max-width:1100px; margin:0 auto; padding:24px 32px; }

  header { display:flex; align-items:baseline; justify-content:space-between; padding-bottom:16px; border-bottom:1px solid var(--border); margin-bottom:24px; }
  header h1 { font-size:16px; font-weight:600; letter-spacing:-0.01em; }
  .status-row { display:flex; align-items:center; gap:16px; font-size:12px; color:var(--muted); }
  .dot { width:7px; height:7px; border-radius:50%; display:inline-block; margin-right:4px; background:var(--muted); }
  .dot.running { background:var(--green); }
  .dot.stopped { background:var(--red); }
  .dot.paused { background:#f59e0b; }
  #statusText { font-size:12px; color:var(--muted); }
  #modeText { font-size:12px; font-weight:600; }

  .grid { display:grid; grid-template-columns: repeat(auto-fit, minmax(160px, 1fr)); gap:0; margin-bottom:24px; border:1px solid var(--border); }
  .card { padding:14px 16px; border-right:1px solid var(--border); }
  .card:last-child { border-right:none; }
  .card .label { font-size:11px; color:var(--muted); margin-bottom:6px; }
  .card .value { font-size:20px; font-weight:500; letter-spacing:-0.02em; }
  .card .value.green { color:var(--green); }
  .card .value.red { color:var(--red); }

  .section { margin-bottom:24px; }
  .section-title { font-size:11px; color:var(--muted); text-transform:uppercase; letter-spacing:0.06em; margin-bottom:10px; }

  .chart-row { display:grid; grid-template-columns:1fr 1fr; gap:1px; background:var(--border); border:1px solid var(--border); margin-bottom:24px; }
  @media (max-width:800px) { .chart-row { grid-template-columns:1fr; } }
  .chart-box { background:var(--bg); padding:16px; }
  .chart-box h3 { font-size:11px; color:var(--muted); margin-bottom:12px; text-transform:uppercase; letter-spacing:0.06em; }
  canvas { width:100% !important; height:180px !important; }

  .controls { padding:16px 0; margin-bottom:24px; border-top:1px solid var(--border); border-bottom:1px solid var(--border); }
  .ctrl-row { display:flex; gap:8px; align-items:center; flex-wrap:wrap; margin-bottom:8px; }
  .ctrl-row:last-child { margin-bottom:0; }
  .ctrl-row label { font-size:12px; color:var(--muted); min-width:80px; }
  .ctrl-row input { background:var(--bg); border:1px solid var(--border); color:var(--text); padding:5px 8px; width:90px; font-family:inherit; font-size:13px; outline:none; }
  .ctrl-row input:focus { border-color:#aaa; }
  button { background:var(--bg); color:var(--text); border:1px solid var(--border); padding:5px 12px; cursor:pointer; font-family:inherit; font-size:12px; }
  button:hover { background:var(--surface); }
  .btn-stop { color:var(--red); border-color:var(--red); }
  .btn-resume { color:var(--green); border-color:var(--green); }

  .trades-table { margin-bottom:24px; overflow-x:auto; }
  table { width:100%; border-collapse:collapse; font-size:13px; }
  th { text-align:left; color:var(--muted); font-weight:400; font-size:11px; text-transform:uppercase; letter-spacing:0.06em; padding:8px 10px; border-bottom:1px solid var(--border); }
  td { padding:8px 10px; border-bottom:1px solid var(--border); }
  tr:last-child td { border-bottom:none; }
  .win { color:var(--green); }
  .loss { color:var(--red); }

  .badge { display:inline-block; padding:2px 6px; font-size:10px; letter-spacing:0.04em; text-transform:uppercase; border:1px solid var(--border); border-radius:3px; font-weight:500; }
  .badge.pt { color:var(--green); border-color:var(--green); }
  .badge.sl { color:var(--red); border-color:var(--red); }
  .badge.time { color:#f59e0b; border-color:#f59e0b; }
  .badge.resolution { color:var(--muted); border-color:var(--border); }
  .badge.open { color:#2563eb; border-color:#2563eb; }

  .holdbar { position:relative; width:60px; height:4px; background:var(--border); border-radius:2px; overflow:hidden; display:inline-block; vertical-align:middle; margin-left:6px; }
  .holdbar > span { display:block; height:100%; background:#f59e0b; transition:width 0.5s linear; }
  .holdbar.danger > span { background:var(--red); }

  td.threshold { font-family:'SF Mono','Fira Code',monospace; font-size:12px; color:var(--muted); }
  td.threshold .arrow { color:var(--text); }
  td.threshold .pt { color:var(--green); }
  td.threshold .sl { color:var(--red); }

  .log-box { border-top:1px solid var(--border); padding-top:16px; }
  .log-box h3 { font-size:11px; color:var(--muted); text-transform:uppercase; letter-spacing:0.06em; margin-bottom:8px; }
  #logContent { height:180px; overflow-y:auto; font-size:11px; line-height:1.6; color:var(--muted); white-space:pre-wrap; word-break:break-all; font-family:'SF Mono','Fira Code',monospace; }
</style>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.7/dist/chart.umd.min.js"></script>
</head>
<body>
<div class="container">
  <header>
    <h1>BTC 5-Min Trader <span style="color:var(--muted);font-weight:400;font-size:13px">v3</span></h1>
    <div class="status-row">
      <span id="v3Config" style="font-family:'SF Mono',monospace;font-size:11px">—</span>
      <span id="modeText">—</span>
      <span><span class="dot stopped" id="statusDot"></span><span id="statusText">Offline</span></span>
    </div>
  </header>

  <div class="grid">
    <div class="card"><div class="label">BTC Spot</div><div class="value" id="btcSpot">—</div></div>
    <div class="card"><div class="label">Model P(Up)</div><div class="value" id="modelPrice">—</div></div>
    <div class="card"><div class="label">Market P(Up)</div><div class="value" id="marketPrice">—</div></div>
    <div class="card"><div class="label">Edge</div><div class="value" id="edge">—</div></div>
    <div class="card"><div class="label">Signal</div><div class="value" id="signal">—</div></div>
    <div class="card"><div class="label">Trades</div><div class="value" id="nTrades">—</div></div>
    <div class="card"><div class="label">Cumulative PnL</div><div class="value" id="cumPnl">—</div></div>
    <div class="card"><div class="label">Window</div><div class="value" style="font-size:15px" id="window">—</div></div>
  </div>

  <div class="chart-row">
    <div class="chart-box">
      <h3>Equity Curve (cumulative PnL)</h3>
      <canvas id="equityChart"></canvas>
    </div>
    <div class="chart-box">
      <h3>Edge over Time</h3>
      <canvas id="edgeChart"></canvas>
    </div>
  </div>
  <div class="chart-row">
    <div class="chart-box">
      <h3>Trade PnL Distribution</h3>
      <canvas id="pnlChart"></canvas>
    </div>
    <div class="chart-box">
      <h3>Model vs Market Price</h3>
      <canvas id="priceChart"></canvas>
    </div>
  </div>

  <div class="controls">
    <div class="ctrl-row">
      <button onclick="sendCmd('pause')">Pause</button>
      <button class="btn-resume" onclick="sendCmd('resume')">Resume</button>
      <button class="btn-stop" onclick="if(confirm('Stop the bot?')) sendCmd('stop')">Stop</button>
    </div>
    <div class="ctrl-row">
      <label>Threshold</label><input id="inThreshold" type="number" step="0.01" value="0.50">
      <label>Size ($)</label><input id="inSize" type="number" step="0.5" value="2.00">
      <label>Max Loss ($)</label><input id="inMaxLoss" type="number" step="5" value="50">
      <button onclick="updateParams()">Apply</button>
    </div>
  </div>

  <div class="trades-table">
    <div class="section-title">Recent Trades</div>
    <table>
      <thead><tr>
        <th>Time</th><th>Side</th><th>Entry</th>
        <th title="Profit target / Stop loss exit prices on the traded token">Targets</th>
        <th>Exit</th>
        <th title="Seconds held; orange bar shows progress toward 70s time-exit">Hold</th>
        <th>Reason</th>
        <th>PnL</th>
      </tr></thead>
      <tbody id="tradesBody"></tbody>
    </table>
  </div>

  <div class="log-box">
    <h3>Log</h3>
    <div id="logContent"></div>
  </div>
</div>

<script>
const $ = id => document.getElementById(id);

// ── Charts ──
const chartOpts = {
  responsive: true,
  animation: false,
  plugins: { legend: { display: false } },
  scales: {
    x: { display: false },
    y: { ticks: { color: '#9b9b9b', font: { size: 10, family: 'system-ui' } }, grid: { color: '#f0f0ee' }, border: { color: '#e9e9e7' } }
  }
};

const equityChart = new Chart($('equityChart'), {
  type: 'line',
  data: { labels: [], datasets: [{ data: [], borderColor: '#111111', borderWidth: 1.5, pointRadius: 0, fill: { target: 'origin', above: 'rgba(22,163,74,0.06)', below: 'rgba(220,38,38,0.06)' } }] },
  options: chartOpts
});

const edgeChart = new Chart($('edgeChart'), {
  type: 'line',
  data: { labels: [], datasets: [{ data: [], borderColor: '#111111', borderWidth: 1, pointRadius: 0 }] },
  options: { ...chartOpts, scales: { ...chartOpts.scales, y: { ...chartOpts.scales.y, suggestedMin: -0.5, suggestedMax: 0.5 } } }
});

const pnlChart = new Chart($('pnlChart'), {
  type: 'bar',
  data: { labels: [], datasets: [{ data: [], backgroundColor: [] }] },
  options: chartOpts
});

const priceChart = new Chart($('priceChart'), {
  type: 'line',
  data: {
    labels: [],
    datasets: [
      { label: 'Model', data: [], borderColor: '#16a34a', borderWidth: 1.2, pointRadius: 0 },
      { label: 'Market', data: [], borderColor: '#dc2626', borderWidth: 1.2, pointRadius: 0 }
    ]
  },
  options: { ...chartOpts, plugins: { legend: { display: true, labels: { color: '#9b9b9b', font: { size: 10, family: 'system-ui' } } } } }
});

// ── Polling ──
async function fetchStatus() {
  try {
    const r = await fetch('/api/status');
    const s = await r.json();
    const dot = $('statusDot');
    const statusText = $('statusText');
    const modeText = $('modeText');

    if (s.running) {
      if (s.paused) {
        statusText.textContent = 'Paused'; dot.className = 'dot paused';
      } else {
        statusText.textContent = 'Running'; dot.className = 'dot running';
      }
    } else {
      statusText.textContent = 'Offline'; dot.className = 'dot stopped';
    }

    modeText.textContent = s.mode || '—';
    modeText.style.color = s.mode === 'LIVE' ? '#dc2626' : '#111111';

    $('btcSpot').textContent = s.btc_spot ? '$' + Number(s.btc_spot).toLocaleString(undefined, {minimumFractionDigits:2}) : '—';
    $('modelPrice').textContent = s.model_price != null ? Number(s.model_price).toFixed(4) : '—';
    $('marketPrice').textContent = s.market_price != null ? Number(s.market_price).toFixed(4) : '—';

    const edge = s.edge;
    const edgeEl = $('edge');
    if (edge != null) {
      edgeEl.textContent = (edge >= 0 ? '+' : '') + Number(edge).toFixed(4);
      edgeEl.className = 'value ' + (Math.abs(edge) > 0.24 ? 'green' : '');
    } else {
      edgeEl.textContent = '—'; edgeEl.className = 'value';
    }

    $('signal').textContent = s.signal || '—';
    $('nTrades').textContent = s.n_trades != null ? s.n_trades : '—';

    const pnl = s.cum_pnl;
    const pnlEl = $('cumPnl');
    if (pnl != null) {
      pnlEl.textContent = (pnl >= 0 ? '+' : '') + Number(pnl).toFixed(2);
      pnlEl.className = 'value ' + (pnl >= 0 ? 'green' : 'red');
    }

    $('window').textContent = s.window || '—';

    if (s.threshold != null) $('inThreshold').value = s.threshold;
    if (s.trade_size != null) $('inSize').value = s.trade_size;
    if (s.max_loss != null) $('inMaxLoss').value = s.max_loss;

    // v3 exit config indicator: PT range / SL / max hold
    window._v3 = {
      pt_multiplier: s.pt_multiplier ?? 1.0,
      pt_floor: s.pt_floor ?? 0.30,
      pt_ceiling: s.pt_ceiling ?? 1.00,
      stop_loss_pct: s.stop_loss_pct ?? 0.50,
      max_hold_seconds: s.max_hold_seconds ?? 70,
    };
    if (s.pt_floor != null) {
      $('v3Config').textContent =
        `PT ${(s.pt_floor*100).toFixed(0)}–${(s.pt_ceiling*100).toFixed(0)}%` +
        ` · SL −${(s.stop_loss_pct*100).toFixed(0)}%` +
        ` · max ${s.max_hold_seconds}s`;
    }
  } catch(e) {}
}

async function fetchTrades() {
  try {
    const r = await fetch('/api/trades');
    const trades = await r.json();
    const tbody = $('tradesBody');
    tbody.innerHTML = '';

    // Equity curve from trades
    let cum = 0;
    const eqLabels = [], eqData = [];
    const pnlLabels = [], pnlData = [], pnlColors = [];

    trades.forEach((t, i) => {
      const pnl = parseFloat(t.pnl) || 0;
      cum += pnl;
      const ts = t.timestamp ? t.timestamp.split('T')[1]?.slice(0,8) : String(i);
      eqLabels.push(ts);
      eqData.push(cum.toFixed(2));
      pnlLabels.push(ts);
      pnlData.push(pnl.toFixed(4));
      pnlColors.push(pnl >= 0 ? '#16a34a' : '#dc2626');
    });

    equityChart.data.labels = eqLabels;
    equityChart.data.datasets[0].data = eqData;
    equityChart.update();

    pnlChart.data.labels = pnlLabels;
    pnlChart.data.datasets[0].data = pnlData;
    pnlChart.data.datasets[0].backgroundColor = pnlColors;
    pnlChart.update();

    renderTradesTable(trades);
  } catch(e) {}
}

function badgeFor(reason, resolved) {
  if (!reason && !resolved) return '<span class="badge open">open</span>';
  if (reason === 'profit_target') return '<span class="badge pt">PT</span>';
  if (reason === 'stop_loss') return '<span class="badge sl">SL</span>';
  if (reason === 'time_exit') return '<span class="badge time">TIME</span>';
  // resolution outcomes
  const r = (reason || resolved || '').toString();
  if (r === 'Up' || r === 'Down') return `<span class="badge resolution">→ ${r}</span>`;
  if (r.startsWith('EXIT:')) {
    const tag = r.slice(5);
    return badgeFor(tag, null);
  }
  return `<span class="badge resolution">${r}</span>`;
}

function renderTradesTable(trades) {
  const tbody = $('tradesBody');
  const cfg = window._v3 || { pt_floor: 0.30, pt_ceiling: 1.00, stop_loss_pct: 0.50, max_hold_seconds: 70 };
  const recent = trades.slice(-30).reverse();
  const nowMs = Date.now();
  let html = '';

  for (const t of recent) {
    const pnl = parseFloat(t.pnl);
    const pnlValid = !Number.isNaN(pnl);
    const cls = pnlValid ? (pnl >= 0 ? 'win' : 'loss') : '';
    const ts = t.timestamp ? t.timestamp.split('T')[1]?.slice(0,8) : '';
    const side = t.side || '';
    const entry = parseFloat(t.entry_price);
    const isBuyUp = side === 'BUY_UP';

    // Cost basis for the traded token, and price thresholds in token-price terms
    // BUY_UP: cost = entry; PT price = entry * (1+pt); SL price = entry * (1-sl)
    // BUY_DOWN: cost = 1 - entry; current token value = 1 - market_price
    //   For display we want the *market price of the UP token* at which we'd exit.
    //   PT for DOWN: cost * (1+pt) means down-token value = c*(1+pt), so up-price = 1 - c*(1+pt)
    //   SL for DOWN: down-token value = c*(1-sl), up-price = 1 - c*(1-sl)
    const pt = parseFloat(t.dynamic_pt);
    const ptUsed = !Number.isNaN(pt) ? pt : cfg.pt_floor;
    const sl = cfg.stop_loss_pct;
    let cost, ptPrice, slPrice;
    if (isBuyUp) {
      cost = entry;
      ptPrice = entry * (1 + ptUsed);
      slPrice = entry * (1 - sl);
    } else {
      cost = 1 - entry;
      ptPrice = 1 - cost * (1 + ptUsed);
      slPrice = 1 - cost * (1 - sl);
    }
    const ptStr = Number.isFinite(ptPrice) ? ptPrice.toFixed(3) : '—';
    const slStr = Number.isFinite(slPrice) ? slPrice.toFixed(3) : '—';
    const targetsCell = `<td class="threshold">
      <span class="pt">↑${ptStr}</span> <span class="arrow">·</span> <span class="sl">↓${slStr}</span>
      <div style="font-size:10px;color:var(--muted)">PT ${(ptUsed*100).toFixed(0)}% / SL −${(sl*100).toFixed(0)}%</div>
    </td>`;

    // Exit price + hold time
    let exitStr = '—';
    let holdCell = '—';
    let reasonCell = '—';

    if (t.exit_timestamp) {
      // Closed via PT/SL/time
      const entryMs = new Date(t.timestamp).getTime();
      const exitMs = new Date(t.exit_timestamp).getTime();
      const holdS = Math.max(0, Math.round((exitMs - entryMs) / 1000));
      const pct = Math.min(100, (holdS / cfg.max_hold_seconds) * 100);
      const danger = holdS >= cfg.max_hold_seconds ? 'danger' : '';
      exitStr = t.exit_price ? Number(t.exit_price).toFixed(3) : '—';
      holdCell = `${holdS}s <span class="holdbar ${danger}"><span style="width:${pct}%"></span></span>`;
      reasonCell = badgeFor(t.exit_reason, t.outcome);
    } else if (t.resolved === 'True' && t.outcome) {
      // Held to resolution (rare in v3 unless bot was offline)
      const entryMs = new Date(t.timestamp).getTime();
      const holdS = Math.max(0, Math.round((nowMs - entryMs) / 1000));
      exitStr = t.outcome;
      holdCell = `${holdS}s`;
      reasonCell = badgeFor(t.outcome, t.outcome);
    } else {
      // Still open — live counter
      const entryMs = new Date(t.timestamp).getTime();
      const holdS = Math.max(0, Math.round((nowMs - entryMs) / 1000));
      const pct = Math.min(100, (holdS / cfg.max_hold_seconds) * 100);
      const danger = holdS >= cfg.max_hold_seconds * 0.85 ? 'danger' : '';
      holdCell = `<span class="hold-live" data-entry="${entryMs}">${holdS}s</span> <span class="holdbar ${danger}"><span style="width:${pct}%"></span></span>`;
      reasonCell = badgeFor(null, null);
    }

    const pnlStr = pnlValid ? `${pnl >= 0 ? '+' : ''}${pnl.toFixed(4)}` : '—';

    html += `<tr>
      <td>${ts}</td>
      <td>${side}</td>
      <td>${Number.isFinite(entry) ? entry.toFixed(3) : '—'}</td>
      ${targetsCell}
      <td>${exitStr}</td>
      <td>${holdCell}</td>
      <td>${reasonCell}</td>
      <td class="${cls}">${pnlStr}</td>
    </tr>`;
  }
  tbody.innerHTML = html;
}

// Tick live-hold counters every second so open trades update without re-fetching.
setInterval(() => {
  const cfg = window._v3 || { max_hold_seconds: 70 };
  const now = Date.now();
  document.querySelectorAll('.hold-live').forEach(el => {
    const entryMs = parseInt(el.dataset.entry, 10);
    if (!entryMs) return;
    const holdS = Math.max(0, Math.round((now - entryMs) / 1000));
    el.textContent = holdS + 's';
    const bar = el.parentElement.querySelector('.holdbar > span');
    if (bar) {
      const pct = Math.min(100, (holdS / cfg.max_hold_seconds) * 100);
      bar.style.width = pct + '%';
    }
    const wrap = el.parentElement.querySelector('.holdbar');
    if (wrap) {
      if (holdS >= cfg.max_hold_seconds * 0.85) wrap.classList.add('danger');
    }
  });
}, 1000);

async function fetchObservations() {
  try {
    const r = await fetch('/api/observations');
    const obs = await r.json();
    // Last 500 for charts
    const recent = obs.slice(-500);
    const labels = recent.map(o => o.timestamp?.split('T')[1]?.slice(0,8) || '');
    const edges = recent.map(o => parseFloat(o.edge) || 0);
    const models = recent.map(o => parseFloat(o.model_price) || null);
    const markets = recent.map(o => {
      return parseFloat(o.market_price_clob) || parseFloat(o.market_price_gamma) || null;
    });

    edgeChart.data.labels = labels;
    edgeChart.data.datasets[0].data = edges;
    edgeChart.update();

    priceChart.data.labels = labels;
    priceChart.data.datasets[0].data = models;
    priceChart.data.datasets[1].data = markets;
    priceChart.update();
  } catch(e) {}
}

async function fetchLog() {
  try {
    const r = await fetch('/api/log');
    const d = await r.json();
    const el = $('logContent');
    el.textContent = (d.lines || []).join('\n');
    el.scrollTop = el.scrollHeight;
  } catch(e) {}
}

function sendCmd(cmd) {
  fetch('/api/control/' + cmd, { method: 'POST' });
}

function updateParams() {
  const params = new URLSearchParams();
  params.set('threshold', $('inThreshold').value);
  params.set('trade_size', $('inSize').value);
  params.set('max_loss', $('inMaxLoss').value);
  fetch('/api/update_params?' + params.toString(), { method: 'POST' });
}

// Poll intervals
setInterval(fetchStatus, 3000);
setInterval(fetchTrades, 5000);
setInterval(fetchObservations, 10000);
setInterval(fetchLog, 8000);

// Initial load
fetchStatus(); fetchTrades(); fetchObservations(); fetchLog();
</script>
</body>
</html>
"""

if __name__ == "__main__":
    import uvicorn
    parser = argparse.ArgumentParser(description="Trader Dashboard")
    parser.add_argument("--port", type=int, default=8050)
    parser.add_argument("--host", default="127.0.0.1")
    args = parser.parse_args()
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
