<p align="center">
  <svg width="200" height="200" viewBox="0 0 200 200" xmlns="http://www.w3.org/2000/svg">
    <defs>
      <linearGradient id="bg" x1="0%" y1="0%" x2="100%" y2="100%">
        <stop offset="0%" style="stop-color:#2a2d3e"/>
        <stop offset="100%" style="stop-color:#1a1c2e"/>
      </linearGradient>
      <linearGradient id="glow" x1="0%" y1="0%" x2="100%" y2="0%">
        <stop offset="0%" style="stop-color:#00d4ff"/>
        <stop offset="100%" style="stop-color:#7b2ff7"/>
      </linearGradient>
      <radialGradient id="eye" cx="50%" cy="50%" r="50%">
        <stop offset="0%" style="stop-color:#00d4ff;stop-opacity:1"/>
        <stop offset="100%" style="stop-color:#004466;stop-opacity:0"/>
      </radialGradient>
      <filter id="fglow"><feGaussianBlur stdDeviation="3" result="b"/><feMerge><feMergeNode in="b"/><feMergeNode in="SourceGraphic"/></feMerge></filter>
    </defs>
    <circle cx="100" cy="100" r="95" fill="url(#bg)" stroke="#3a3d5e" stroke-width="2"/>
    <line x1="100" y1="28" x2="100" y2="15" stroke="#4a4d6e" stroke-width="3" stroke-linecap="round"/>
    <circle cx="100" cy="12" r="5" fill="url(#glow)" filter="url(#fglow)"/>
    <rect x="45" y="35" width="110" height="90" rx="20" fill="url(#bg)" stroke="#3a3d5e" stroke-width="2"/>
    <rect x="58" y="48" width="84" height="64" rx="12" fill="#12142a" stroke="#2a2d4e" stroke-width="1"/>
    <circle cx="78" cy="75" r="12" fill="#0a0c1e" stroke="#00d4ff" stroke-width="1.5" opacity="0.8"/>
    <circle cx="78" cy="75" r="7" fill="url(#eye)" filter="url(#fglow)"/>
    <circle cx="75" cy="72" r="2.5" fill="white" opacity="0.8"/>
    <circle cx="122" cy="75" r="12" fill="#0a0c1e" stroke="#00d4ff" stroke-width="1.5" opacity="0.8"/>
    <circle cx="122" cy="75" r="7" fill="url(#eye)" filter="url(#fglow)"/>
    <circle cx="119" cy="72" r="2.5" fill="white" opacity="0.8"/>
    <rect x="82" y="93" width="36" height="10" rx="5" fill="#0a0c1e" stroke="#2a2d4e" stroke-width="1"/>
    <line x1="88" y1="98" x2="92" y2="98" stroke="#00d4ff" stroke-width="1.5" opacity="0.6"/>
    <line x1="96" y1="98" x2="100" y2="98" stroke="#00d4ff" stroke-width="1.5" opacity="0.6"/>
    <line x1="104" y1="98" x2="108" y2="98" stroke="#00d4ff" stroke-width="1.5" opacity="0.6"/>
    <rect x="33" y="55" width="14" height="40" rx="5" fill="#1e2040" stroke="#3a3d5e" stroke-width="1.5"/>
    <rect x="153" y="55" width="14" height="40" rx="5" fill="#1e2040" stroke="#3a3d5e" stroke-width="1.5"/>
    <rect x="80" y="125" width="40" height="18" rx="4" fill="#1e2040" stroke="#2a2d4e" stroke-width="1"/>
    <circle cx="100" cy="155" r="4" fill="#64ffda" filter="url(#fglow)">
      <animate attributeName="opacity" values="1;0.3;1" dur="2s" repeatCount="indefinite"/>
    </circle>
    <text x="100" y="185" text-anchor="middle" fill="#484f58" font-family="monospace" font-size="11" letter-spacing="3">NEURALTRADE</text>
  </svg>
</p>

<h1 align="center">NeuralTrade</h1>
<p align="center"><em>Multi-Model AI Trading Framework — Autonomous agents for stock & crypto trading.</em></p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.12+-3776AB?style=flat-square&logo=python" />
  <img src="https://img.shields.io/badge/LangGraph-Agent%20Framework-1a1a2e?style=flat-square" />
  <img src="https://img.shields.io/badge/Apache-2.0-E8453C?style=flat-square" />
</p>

---

## Why NeuralTrading?

Most AI trading tools are either too simple or too rigid. NeuralTrade takes a different approach:

- **Multiple AI models, one framework** — use OpenAI, Anthropic, Google, Groq, or OpenRouter. Mix and match per agent.
- **Specialized agents** — fundamental analyst, sentiment analyst, news analyst, technical analyst, bull/bear researchers, trader, risk manager, and portfolio manager.
- **Built-in backtesting** — test strategies on historical data before risking real money.
- **Paper trading by default** — experiment safely before going live.
- **Web dashboard + CLI** — use it from terminal or browser.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     NeuralTrade Engine                       │
├──────────────┬──────────────┬───────────────┬───────────────┤
│   Analysts   │  Researchers  │    Trader     │  Risk Mgmt   │
│              │              │              │              │
│ Fundamental  │   Bull 🐂    │   Decision   │  Risk Score  │
│ Sentiment    │   Bear 🐻    │   Sizing     │  Position    │
│ News         │              │   Timing     │  Limits      │
│ Technical    │              │              │              │
├──────────────┴──────────────┴───────────────┴───────────────┤
│                  Multi-Model LLM Router                      │
│         openai/gpt-4o · anthropic/claude · groq/llama       │
├─────────────────────────────────────────────────────────────┤
│              Market Data · News · Sentiment                  │
│              yfinance · Alpha Vantage · RSS                 │
└─────────────────────────────────────────────────────────────┘
```

---

## Quick Start

```bash
# Install
pip install -e .

# Configure
cp .env.example .env
# Edit .env — add at least one API key

# Analyze a stock
neuraltrade analyze AAPL

# Analyze crypto
neuraltrade analyze BTC-USD --type crypto

# Use a specific model
neuraltrade analyze TSLA --model anthropic/claude-sonnet-4

# Backtest
neuraltrade backtest AAPL --period 1y --capital 50000

# See available models
neuraltrade models

# Start the dashboard
neuraltrade serve --port 8080
```

### Docker

```bash
docker build -t neuraltrade .
docker run -p 8080:8080 --env-file .env neuraltrade
```

---

## Multi-Model Routing

Each agent can use a different model. The router picks the best available model based on the task type:

| Task Type | Use Case | Example Models |
|-----------|----------|----------------|
| `QUICK` | Classification, extraction | gpt-4o-mini, gemini-2.0-flash |
| `STANDARD` | Analysis, reasoning | gpt-4o, claude-sonnet-4 |
| `COMPLEX` | Final decisions, risk assessment | o1, claude-opus-4, gpt-4o |

Configure per-agent overrides in `.env`:
```env
ANALYST_MODEL=openai/gpt-4o
TRADER_MODEL=anthropic/claude-sonnet-4
RISK_MANAGER_MODEL=openai/gpt-4o
```

---

## Agent Pipeline

1. **Analyst Team** (parallel)
   - Fundamental Analyst → financial health, valuation
   - Sentiment Analyst → news & social media mood
   - News Analyst → events, macro, catalysts
   - Technical Analyst → charts, indicators, patterns

2. **Researcher Team** (parallel)
   - Bull Researcher → argues FOR the trade
   - Bear Researcher → argues AGAINST the trade

3. **Decision Team** (sequential)
   - Trader → synthesizes research, proposes action
   - Risk Manager → evaluates risk, recommends size
   - Portfolio Manager → final approve/reject/modify

---

## Backtesting

```bash
# Test SMA crossover strategy
neuraltrade backtest MSFT --period 2y --capital 100000
```

```
╔══════════════════════════════════════════╗
║       NeuralTrade Backtest Results       ║
╠══════════════════════════════════════════╣
║ Symbol:          MSFT                    ║
║ Total Return:    +23.45%                ║
║ Sharpe Ratio:    1.82                   ║
║ Max Drawdown:    -8.32%                 ║
║ Win Rate:        62.5%                  ║
║ Total Trades:    48                     ║
╚══════════════════════════════════════════╝
```

---

## Roadmap

- [x] Multi-model LLM routing
- [x] 9 specialized agents
- [x] Backtesting engine
- [x] Web dashboard
- [x] CLI interface
- [ ] Live trading integration (Alpaca, Binance)
- [ ] Portfolio optimization
- [ ] IDX (Indonesia Stock Exchange) support
- [ ] Telegram/Discord bot alerts
- [ ] Custom strategy builder

---

## Disclaimer

**NeuralTrade is for research and educational purposes only.** It is not financial, investment, or trading advice. Past performance (including backtests) does not guarantee future results. Always do your own research and consult a qualified financial advisor before making investment decisions.

---

<p align="center">
  Built by <a href="https://github.com/anonymousdlik">Fadli Yeh</a> · Palembang, Indonesia
</p>
