# ERC-8004 Demo Video Script (3 Minutes)

> **Purpose**: Judge-facing walkthrough of the full ERC-8004 Autonomous Trading Agent pipeline.
> Target runtime: 3 minutes. Record with `demo_server.py` running on port 8084.

---

## OPENING HOOK (15s)

> *Screen: terminal, blank prompt*

**Narrator:**
> "What if 10 AI agents could trade better than one? ERC-8004 is an autonomous trading agent
> standard — where every agent has an on-chain identity, a stake-weighted vote, and a
> portfolio risk engine keeping them honest. Let me show you in 3 minutes."

---

## SECTION 1 — Architecture & Health (45s)

> *Screen: curl command running, then JSON output side-by-side with a diagram*

**Narrator:**
> "First, the health check — this shows the live state of the server: sprint, test count, and
> what makes S46 special."

```bash
curl -s http://localhost:8084/demo/health | python3 -m json.tool
```

**Expected output (highlight these fields):**
```json
{
  "status": "ok",
  "version": "S46",
  "sprint": "S46",
  "tests": 6085,
  "highlights": [
    "Portfolio risk engine (VaR 95/99%)",
    "10-agent swarm with 6 strategies",
    "Position sizing (Kelly/volatility/fixed)",
    "Exposure dashboard with concentration index"
  ]
}
```

**Narrator:**
> "6,085 tests. A portfolio risk engine. 10 agents, 6 strategies. And today we're going to
> see all of it work together in a single call."

---

## SECTION 2 — Full Pipeline Showcase (60s)

> *Screen: single curl command, then walk through each step in the JSON*

**Narrator:**
> "This is our showcase endpoint — one POST, four pipeline stages, full transparency."

```bash
curl -s -X POST http://localhost:8084/api/v1/demo/showcase | python3 -m json.tool
```

**Walk through each step:**

**Step 1 — Live Price Tick:**
> "We fetch a live BTC-USD price tick with bid, ask, and 24h volume."

**Step 2 — 10-Agent Swarm Vote:**
> "Ten agents, each with a different strategy and stake weight, vote on whether to go LONG.
> We need 2/3 stake-weighted consensus before we act."

**Step 3 — Risk Engine:**
> "The risk engine computes VaR at 95% and 99% confidence, then sizes the position using
> volatility-based sizing — risk budget divided by daily volatility — so we never bet more
> than the math supports."

**Step 4 — Paper Trade Execution:**
> "The consensus decision executes as a paper trade — symbol, quantity, fill price, and fee
> all logged. No real money, full real logic."

**Narrator (pointing at summary block):**
> "The summary block tells the whole story in five lines: price, consensus, agents in
> agreement, VaR, and position size."

---

## SECTION 3 — Swarm Performance Leaderboard (45s)

> *Screen: terminal, swarm performance endpoint*

**Narrator:**
> "After the trade, you can see how each agent is performing. This is the swarm leaderboard —
> ranked by 24-hour PnL and Sharpe ratio."

```bash
curl -s http://localhost:8084/api/v1/swarm/performance | python3 -m json.tool
```

**Expected output (highlight these fields):**
```json
{
  "leaderboard": [
    {
      "rank": 1,
      "agent_id": "quant-7",
      "strategy": "momentum",
      "total_pnl_24h": 4.82,
      "sharpe_24h": 1.34,
      "win_rate": 0.75
    }
  ],
  "total_agents": 10,
  "portfolio_pnl_24h": 12.41
}
```

**Narrator:**
> "Every agent's Sharpe, win rate, and PnL — updated with each vote. Bad performers get
> out-voted by agents with higher stake. Natural selection, on-chain."

---

## SECTION 4 — Portfolio Risk Dashboard (30s)

> *Screen: risk/portfolio endpoint*

**Narrator:**
> "And the portfolio risk view — VaR across all four symbols, correlation matrix, and
> Calmar ratio."

```bash
curl -s http://localhost:8084/api/v1/risk/portfolio | python3 -m json.tool
```

**Expected output (highlight):**
```json
{
  "portfolio": {
    "var_95": 0.021,
    "var_99": 0.034,
    "sharpe_ratio": 0.84,
    "calmar_ratio": 2.1,
    "max_drawdown": 0.092
  },
  "symbols": ["BTC-USD", "ETH-USD", "SOL-USD", "MATIC-USD"]
}
```

**Narrator:**
> "A 9.2% max drawdown, 84% Sharpe. The risk engine tells each agent exactly how much
> capital to deploy — so the swarm can't blow up the portfolio."

---

## CLOSING (15s)

> *Screen: GitHub repo, test count badge*

**Narrator:**
> "ERC-8004: an open standard for autonomous trading agents — identity, reputation, risk,
> and consensus in one pipeline. 6,085 tests across 47 sprints. MIT licensed.
> Hit the showcase endpoint yourself."

```bash
# Try it:
curl -s -X POST http://localhost:8084/api/v1/demo/showcase | python3 -m json.tool
```

> **GitHub**: github.com/opspawn/erc8004-trading-agent
> **Unique features**: On-chain identity (ERC-8004), stake-weighted swarm consensus,
> Kelly/VaR risk engine, 6 strategy types, full paper trading pipeline

---

## Recording Checklist

- [ ] `demo_server.py` running on port 8084 (`python3 agent/demo_server.py`)
- [ ] Terminal font ≥ 16pt, high contrast theme
- [ ] Run each curl once before recording to warm the server
- [ ] Keep each section within its time budget
- [ ] Show the raw JSON first, then summarise verbally
- [ ] End with the GitHub URL visible on screen
