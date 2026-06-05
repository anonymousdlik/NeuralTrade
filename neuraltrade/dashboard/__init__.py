"""Web dashboard for NeuralTrade."""

from __future__ import annotations

import json
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles


def create_app() -> FastAPI:
    """Create the FastAPI dashboard app."""
    app = FastAPI(title="NeuralTrade Dashboard", version="1.0.0")

    @app.get("/", response_class=HTMLResponse)
    async def index():
        return DASHBOARD_HTML

    @app.get("/api/status")
    async def status():
        return {"status": "ok", "version": "1.0.0"}

    @app.post("/api/analyze")
    async def analyze(request: dict):
        symbol = request.get("symbol", "")
        if not symbol:
            raise HTTPException(status_code=400, detail="Symbol required")
        return {"symbol": symbol, "status": "analysis queued"}

    return app


DASHBOARD_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>NeuralTrade Dashboard</title>
<style>
* { margin: 0; padding: 0; box-sizing: border-box; }
body { font-family: 'Segoe UI', system-ui, sans-serif; background: #0d1117; color: #c9d1d9; min-height: 100vh; }
.header { background: linear-gradient(135deg, #161b22 0%, #1a1c2e 100%); border-bottom: 1px solid #30363d; padding: 20px 40px; display: flex; align-items: center; justify-content: space-between; }
.header h1 { font-size: 1.5em; background: linear-gradient(135deg, #00d4ff, #7b2ff7); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
.header .version { color: #484f58; font-size: 0.85em; }
.container { max-width: 1200px; margin: 0 auto; padding: 30px 20px; }
.card { background: #161b22; border: 1px solid #30363d; border-radius: 12px; padding: 24px; margin-bottom: 20px; }
.card h2 { color: #00d4ff; margin-bottom: 16px; font-size: 1.2em; }
.search-box { display: flex; gap: 12px; margin-bottom: 20px; }
.search-box input { flex: 1; padding: 12px 16px; background: #0d1117; border: 1px solid #30363d; border-radius: 8px; color: #c9d1d9; font-size: 1em; outline: none; }
.search-box input:focus { border-color: #00d4ff; }
.search-box button { padding: 12px 24px; background: linear-gradient(135deg, #00d4ff, #7b2ff7); border: none; border-radius: 8px; color: white; font-weight: 600; cursor: pointer; }
.search-box button:hover { opacity: 0.9; }
.grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap: 16px; }
.stat { text-align: center; padding: 16px; }
.stat .value { font-size: 2em; font-weight: 700; color: #00d4ff; }
.stat .label { color: #484f58; font-size: 0.85em; margin-top: 4px; }
.decision { text-align: center; padding: 30px; }
.decision .action { font-size: 3em; font-weight: 800; }
.decision.buy .action { color: #3fb950; }
.decision.sell .action { color: #f85149; }
.decision.hold .action { color: #d29922; }
.status-bar { display: flex; gap: 8px; flex-wrap: wrap; }
.status-badge { padding: 4px 12px; border-radius: 20px; font-size: 0.8em; background: #21262d; color: #8b949e; border: 1px solid #30363d; }
.status-badge.active { background: #0d2f1f; color: #3fb950; border-color: #238636; }
</style>
</head>
<body>
<div class="header">
  <h1>⚡ NeuralTrade</h1>
  <span class="version">v1.0.0 — Multi-Model AI Trading</span>
</div>
<div class="container">
  <div class="card">
    <h2>🔍 Analyze Asset</h2>
    <div class="search-box">
      <input type="text" id="symbol" placeholder="Enter symbol (e.g., AAPL, BTC-USD, TSLA)" />
      <button onclick="analyze()">Analyze</button>
    </div>
    <div class="status-bar">
      <span class="status-badge active">● Engine Ready</span>
      <span class="status-badge">Paper Trading</span>
      <span class="status-badge">Multi-Model</span>
    </div>
  </div>

  <div class="grid">
    <div class="card">
      <h2>📊 Portfolio</h2>
      <div class="stat">
        <div class="value" id="portfolio-value">$100,000</div>
        <div class="label">Total Value</div>
      </div>
    </div>
    <div class="card">
      <h2>📈 Today's P&L</h2>
      <div class="stat">
        <div class="value" id="pnl" style="color: #3fb950;">+$0.00</div>
        <div class="label">0.00%</div>
      </div>
    </div>
    <div class="card">
      <h2>🎯 Win Rate</h2>
      <div class="stat">
        <div class="value" id="win-rate">0%</div>
        <div class="label">0 Trades</div>
      </div>
    </div>
  </div>

  <div class="card" id="result-card" style="display: none;">
    <h2>🎯 Analysis Result</h2>
    <div id="result-content"></div>
  </div>
</div>

<script>
async function analyze() {
  const symbol = document.getElementById('symbol').value.trim();
  if (!symbol) return alert('Please enter a symbol');
  const btn = document.querySelector('.search-box button');
  btn.textContent = 'Analyzing...';
  btn.disabled = true;
  try {
    const res = await fetch('/api/analyze', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ symbol })
    });
    const data = await res.json();
    document.getElementById('result-card').style.display = 'block';
    document.getElementById('result-content').innerHTML =
      '<p>Analysis queued for <strong>' + data.symbol + '</strong></p>' +
      '<p style="color: #8b949e; margin-top: 8px;">Connect the Python backend for live analysis.</p>';
  } catch (e) {
    alert('Error: ' + e.message);
  }
  btn.textContent = 'Analyze';
  btn.disabled = false;
}
</script>
</body>
</html>"""
