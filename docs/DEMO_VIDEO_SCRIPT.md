# ERC-8004 Demo Video Script (3 Minutes)

> **Purpose**: Judge-facing walkthrough of the full ERC-8004 Autonomous Trading Agent pipeline.
> Target runtime: 3 minutes. Record with `demo_server.py` running on port 8084.
> **Sprint**: S54 | **Tests**: 6,300 passing

---

## OPENING HOOK (15s)

> *Screen: terminal, blank prompt*

**Narrator:**
> "What if 10 AI agents could trade better than one? ERC-8004 is an autonomous trading agent
> standard — where every agent has an on-chain identity, a stake-weighted vote, and a
> portfolio risk engine keeping them honest. Let me show you in 3 minutes."

---

## SECTION 1 — Architecture & Health (30s)

> *Screen: curl command running, then JSON output*

**Narrator:**
> "First, the health check — live state of the server: sprint S54, 6,300 tests, and what makes
> this version special."

```bash
curl -s http://localhost:8084/demo/health | python3 -m json.tool
```

**Expected output (highlight these fields):**
```json
{
  "status": "ok",
  "version": "S54",
  "sprint": "S54",
  "tests": 6300,
  "highlights": [
    "Portfolio risk engine (VaR 95/99%)",
    "10-agent swarm with 6 strategies",
    "Position sizing (Kelly/volatility/fixed)",
    "Exposure dashboard with concentration index",
    "Interactive HTML demo UI (/demo/ui)"
  ]
}
```

**Narrator:**
> "6,300 tests. A portfolio risk engine. 10 agents, 6 strategies."

---

## SECTION 2 — Judge Dashboard (45s)

> *Screen: browser at http://localhost:8084/demo/judge*

**Narrator:**
> "The judge dashboard is a live read — one page showing all the key metrics judges care about."

Navigate to: `http://localhost:8084/demo/judge`

**Point out these sections:**

**Swarm Agents (top section):**
> "Ten quant agents — quant-1 through quant-10 — each with a different strategy and stake weight.
> They vote on every tick. quant-1 uses momentum, quant-2 uses mean reversion, and so on."

**TA Signals Panel:**
> "RSI and MACD signals in real time for BTC-USD, ETH-USD, and SOL-USD. When RSI crosses 70,
> agents with momentum strategies get a SELL signal. When RSI drops below 30, they get BUY.
> MACD crossover adds a second confirmation layer."

**Portfolio Risk (VaR panel):**
> "VaR at 95% and 99% confidence, Sharpe ratio, max drawdown. The risk engine sets position
> sizing for every trade — so the swarm can never bet more than the math supports."

---

## SECTION 3 — TA Signals Endpoint (30s)

> *Screen: terminal, signals API*

**Narrator:**
> "The signals endpoint is machine-readable — designed for agents, not humans."

```bash
curl -s http://localhost:8084/api/v1/signals/latest | python3 -m json.tool
```

**Expected output:**
```json
{
  "signals": [
    {
      "symbol": "BTC-USD",
      "last_price": 67500.0,
      "rsi": 100.0,
      "rsi_signal": "SELL",
      "macd_signal": "NEUTRAL"
    },
    {
      "symbol": "ETH-USD",
      "rsi_signal": "SELL",
      "macd_signal": "NEUTRAL"
    },
    {
      "symbol": "SOL-USD",
      "rsi_signal": "SELL",
      "macd_signal": "NEUTRAL"
    }
  ],
  "version": "S54"
}
```

**Narrator:**
> "Three symbols, two signals each — RSI and MACD. Every agent in the swarm reads this before
> voting. It's what separates ERC-8004 from a simple random-vote system."

---

## SECTION 4 — Full Pipeline Showcase (45s)

> *Screen: single curl command, then walk through each step in the JSON*

**Narrator:**
> "This is the showcase endpoint — one POST, four pipeline stages, full transparency."

```bash
curl -s -X POST http://localhost:8084/api/v1/demo/showcase | python3 -m json.tool
```

**Walk through each step:**

**Step 1 — Live Price Tick:**
> "We fetch a live BTC-USD price tick with bid, ask, and 24h volume."

**Step 2 — 10-Agent Swarm Vote:**
> "Ten agents vote on whether to go LONG. We need 2/3 stake-weighted consensus before we act."

**Step 3 — Risk Engine:**
> "The risk engine computes VaR at 95% and 99% confidence, then sizes the position using
> volatility-based sizing — risk budget divided by daily volatility."

**Step 4 — Paper Trade Execution:**
> "The consensus decision executes as a paper trade — symbol, quantity, fill price, and fee
> all logged. Full real logic, no real money."

---

## SECTION 5 — Interactive Demo UI (15s)

> *Screen: browser at http://localhost:8084/demo/ui*

**Narrator:**
> "And there's a fully interactive HTML demo — judges can click each pipeline stage and see
> the live JSON response. No curl required."

Navigate to: `http://localhost:8084/demo/ui`

> "Click any panel — swarm vote, risk portfolio, TA signals, or the full showcase.
> Each button hits the live endpoint and renders the result."

---

## CLOSING (15s)

> *Screen: GitHub repo, test count badge*

**Narrator:**
> "ERC-8004: an open standard for autonomous trading agents — identity, reputation, risk,
> and consensus in one pipeline. 6,300 tests across 54 sprints. MIT licensed.
> Hit the showcase endpoint yourself."

```bash
# Try it:
curl -s -X POST http://localhost:8084/api/v1/demo/showcase | python3 -m json.tool
# Or explore the judge dashboard:
# http://localhost:8084/demo/judge
```

> **GitHub**: github.com/opspawn/erc8004-trading-agent
> **Unique features**: On-chain identity (ERC-8004), stake-weighted swarm consensus,
> RSI/MACD TA signals, Kelly/VaR risk engine, 6 strategy types, judge dashboard

---

## S54 Video Asset

- **Demo screenshots**: `docs/demo-screenshots/s54-*.png` (6 frames)
- **Demo video**: `docs/demo-video-s54.mp4` (~424KB, 18-second slideshow)
- Created with Playwright (headless Chromium) + ffmpeg

---

## Recording Checklist

- [ ] `demo_server.py` running on port 8084 (`python3 agent/demo_server.py`)
- [ ] Terminal font ≥ 16pt, high contrast theme
- [ ] Run each curl once before recording to warm the server
- [ ] Keep each section within its time budget
- [ ] Show the raw JSON first, then summarise verbally
- [ ] Demonstrate judge dashboard in browser (http://localhost:8084/demo/judge)
- [ ] Demonstrate interactive demo UI (http://localhost:8084/demo/ui)
- [ ] End with the GitHub URL visible on screen
