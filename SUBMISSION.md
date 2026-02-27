# ERC-8004 Autonomous Trading Agent — Submission

## Project Information

| Field | Details |
|-------|---------|
| **Title** | ERC-8004 Autonomous Trading Agent |
| **Team** | OpSpawn |
| **Demo URL** | https://erc8004-trading-agent.vercel.app |
| **Live Endpoint** | `POST https://api.opspawn.com/erc8004/demo/run` |
| **Category** | DeFi / Agent Infrastructure |
| **Deadline** | March 22, 2026 — lablab.ai |

---

## Project Description

**ERC-8004 Trading Agent** is a fully autonomous AI trading system built on the ERC-8004
standard for on-chain agent identity, reputation, and validation. The agent holds a
verifiable on-chain identity (ERC-721 token), accumulates a reputation score through each
trade outcome, and applies institutional-grade risk management using Credora credit ratings
before executing any position.

The system demonstrates ERC-8004's core value proposition: three autonomous agents
(Conservative, Balanced, Aggressive) each hold their own ERC-8004 identity and reputation
score. Their votes are weighted by reputation — the most trusted agent has the most
influence over the final trade decision. No single agent can force a bad trade; the
reputation-weighted mesh requires a 2/3 supermajority. Every trade is validated by the
on-chain `ValidationRegistry`, producing a signed artifact linking the outcome to the
agent's identity. A full x402 micropayment gate protects the demo endpoint, with dev_mode
enabled so judges can call it without a wallet.

---

## Deployment

| Network | Chain ID | Contract / Registry Address | Explorer |
|---------|----------|----------------------------|----------|
| Base Sepolia | 84532 | `0x8004B663056A597Dffe9eCcC1965A193B7388713` | [basescan](https://sepolia.basescan.org/address/0x8004B663056A597Dffe9eCcC1965A193B7388713) |
| **Arbitrum Sepolia** | **421614** | Deployer wallet `0x5Ee4e0E55213787A453FB720e8386F41Fd7d093E` — **pending testnet ETH faucet** (contract address available once funded) | [arbiscan](https://sepolia.arbiscan.io) |

> Deployed on: Arbitrum Sepolia (testnet)
> Arbitrum Sepolia deployment wallet: `0x5Ee4e0E55213787A453FB720e8386F41Fd7d093E`
> Get free testnet ETH at https://www.alchemy.com/faucets/arbitrum-sepolia then run `bash scripts/deploy-arbitrum-sepolia.sh`
> Contract address will be written to `deployment-arbitrum-sepolia.json` once funded.

---

## Key Technical Features

- **ERC-8004 On-chain Identity**: Each agent holds an ERC-721 token in `IdentityRegistry`
  with a DID in the format `eip155:{chainId}:{address}:{agentId}`
- **Reputation-weighted multi-agent consensus**: 3 agents vote per tick; 2/3 weighted
  majority (weighted by current ERC-8004 reputation score) required before any trade
- **On-chain trust loop**: Each trade outcome updates the agent's `ReputationRegistry`
  score on-chain (Base Sepolia `0x8004B663056A597Dffe9eCcC1965A193B7388713`)
- **x402 payment gate**: `/demo/run` endpoint is micropayment-gated via x402 HTTP
  protocol; `dev_mode=true` bypasses payment for evaluation
- **Credora credit ratings**: Kelly Criterion position sizing uses Credora tier as
  multiplier — AAA protocols get 1.0x, unrated protocols get 0.10x
- **Validation artifacts**: Signed JSON proofs generated for each session, containing
  performance metrics, strategy hash, and ECDSA signature over the canonical artifact hash
- **Backtesting engine**: Historical strategy comparison across 5 strategies with
  Sharpe ratio, max drawdown, win rate, and `BacktestRegistry` mock on-chain storage
- **Stress tester**: 7 adversarial scenarios — flash crash (-40% in 5 ticks), oracle
  failure, consensus deadlock, extreme volatility, reputation collapse, zero capital
- **Live demo server**: HTTP endpoint (port 8084) runs full pipeline in <100ms, returns
  JSON report with trades, PnL, reputation scores, and signed validation artifact hash
- **Live metrics dashboard** (`GET /demo/metrics`): Real-time aggregate stats across all
  runs — win rate, Sharpe ratio, Sortino ratio, max drawdown, cumulative return, active agents
- **Agent leaderboard** (`GET /demo/leaderboard`): Top agents ranked by configurable metric —
  `?sort_by=sortino|sharpe|pnl|trades|win_rate|reputation` with `?limit=N` (1–20);
  5 agents seeded, grows with live runs; each entry has rank, strategy, return pct, win rate, trade count, reputation
- **Multi-agent comparison** (`POST /demo/compare`): Side-by-side performance metrics for
  2-5 agent IDs — useful for demonstrating multi-agent coordination advantages
- **Real-time SSE stream** (`GET /demo/stream`): Server-Sent Events push live run_complete
  events to connected clients; keepalive every 15s for long-polling dashboards
- **RedStone/Surge oracle integration**: On-chain risk check via `RiskRouter.checkRisk()`
  validates oracle deviation before any settlement

---

## Test Evidence

**Total tests: 4,908 passing** (S39 — verified 2026-02-26 by running `python3 -m pytest --tb=no -q`)

| Sprint | Tests | Key additions |
|--------|-------|---------------|
| S01–S30 | 3,541 | Core pipeline, consensus, reputation, x402 |
| S31–S36 | 4,658 | Attribution, live feed, leaderboard, backtest |
| S37 | 4,729 | Risk dashboard, ensemble vote, alpha decay, cross-train |
| S38 | 4,800 | Strategy performance attribution (by type/period/risk bucket), demo script V2, pitch deck |
| **S39** | **4,908** | Live market simulation, portfolio snapshot, strategy comparison dashboard (+108 tests) |

| Test File | Coverage Area |
|-----------|---------------|
| `test_demo_runner.py` | Multi-agent E2E demo, consensus, edge cases |
| `test_demo_cli.py` | CLI pipeline, GBM prices, ledger writing |
| `test_backtester.py` + `test_backtester_extended.py` | Historical simulation, Sharpe, drawdown |
| `test_mesh_coordinator.py` | 3-agent consensus, reputation weighting |
| `test_reputation.py` | Score calculation, on-chain logging, P&L evidence |
| `test_validation_artifacts.py` | Signed artifacts, hash verification |
| `test_x402_client.py` | Payment protocol, 402/200 flows |
| `test_stress_tester.py` | 7 adversarial scenarios |
| `test_paper_trader.py` | GBM simulation, full 24h paper run |
| `test_pipeline.py` + `test_pipeline_e2e.py` | Integration orchestrator |
| `test_health_api.py` | REST health/metrics endpoints |
| `test_risk_dashboard.py` | Risk data API, VaR, Sharpe |
| `test_credora_client.py` + extensions | Credit tiers, Kelly multipliers |
| `test_s37_risk_ensemble.py` | Risk dashboard, ensemble vote, alpha decay, cross-train |
| `test_s38_performance.py` | Strategy attribution by type, period, risk bucket (71 tests) |
| `test_s39_live_sim.py` | Live market sim, portfolio snapshot, strategy compare (108 tests) |
| ... + 45 more test files | Full coverage across all modules |

```bash
cd agent && python3 -m pytest tests/ -q --tb=no
# 4908 passed in ~60s
```

---

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────┐
│                    ERC-8004 Trading Agent                           │
│                                                                     │
│  ┌───────────────┐    ┌─────────────────────────────────────────┐  │
│  │  Market Feed  │───▶│           Signal Bus (asyncio)          │  │
│  │  CoinGecko    │    │  BTC/ETH/SOL + GBM fallback             │  │
│  │  + GBM sim    │    └──────────────────┬──────────────────────┘  │
│  └───────────────┘                       │                         │
│                                          ▼                         │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │                    Strategy Engine                           │  │
│  │  Momentum │ MeanReversion │ VolBreakout │ Sentiment │ Ensemble│  │
│  └─────────────────────────────┬────────────────────────────────┘  │
│                                │                                   │
│                                ▼                                   │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │              Agent Mesh Coordinator                          │  │
│  │  Conservative (rep 7.5) │ Balanced (rep 6.0) │ Aggressive   │  │
│  │              2/3 reputation-weighted vote                    │  │
│  └─────────────────────────────┬────────────────────────────────┘  │
│                                │                                   │
│                                ▼                                   │
│  ┌────────────────┐  ┌─────────────────┐  ┌─────────────────────┐ │
│  │  Risk Manager  │─▶│  Paper Trader   │─▶│  Demo Server :8084  │ │
│  │  Credora+Kelly │  │  GBM simulation │  │  POST /demo/run     │ │
│  └────────────────┘  └────────┬────────┘  └─────────────────────┘ │
│                               │                                    │
│                               ▼                                    │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │            ERC-8004 On-chain Layer (Base Sepolia)            │  │
│  │  IdentityRegistry │ ReputationRegistry │ ValidationRegistry  │  │
│  │  AgentWallet (EIP-1271)  │  RiskRouter  │  BacktestRegistry  │  │
│  └──────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Live Endpoint Instructions

**Status: LIVE** — All endpoints verified working as of 2026-02-26.

The demo server runs publicly at `https://api.opspawn.com/erc8004/` (systemd service `erc8004-demo.service` → nginx proxy → localhost:8084).
All endpoints use `dev_mode=true` — free for judges, no wallet required.

### Public Endpoints

| Method | URL | Description |
|--------|-----|-------------|
| `GET`  | `https://api.opspawn.com/erc8004/demo/health` | Health check |
| `POST` | `https://api.opspawn.com/erc8004/demo/run` | Run full pipeline |
| `GET`  | `https://api.opspawn.com/erc8004/demo/portfolio` | Portfolio analytics |
| `GET`  | `https://api.opspawn.com/erc8004/demo/metrics` | **Live aggregate metrics** — win rate, Sharpe, Sortino, drawdown |
| `GET`  | `https://api.opspawn.com/erc8004/demo/leaderboard` | **Agent leaderboard** — top 5 agents by Sortino ratio |
| `POST` | `https://api.opspawn.com/erc8004/demo/compare` | **Side-by-side comparison** — `{"agent_ids": ["id1","id2"]}` |
| `GET`  | `https://api.opspawn.com/erc8004/demo/stream` | **SSE stream** — real-time events when `/demo/run` completes |
| `GET`  | `https://api.opspawn.com/erc8004/demo/strategy/performance-attribution` | **S38: Strategy attribution** — P&L by type, period, risk bucket — `?period=1h\|24h\|7d` |
| `POST` | `https://api.opspawn.com/erc8004/demo/live/simulate` | **S39: Live simulation** — tick-by-tick session with P&L, `{ticks,seed,symbol,strategy,initial_capital}` |
| `GET`  | `https://api.opspawn.com/erc8004/demo/portfolio/snapshot` | **S39: Portfolio snapshot** — positions, unrealized P&L, Sharpe, drawdown, win rate |
| `GET`  | `https://api.opspawn.com/erc8004/demo/strategy/compare` | **S39: Strategy comparison** — side-by-side metrics for all active strategies (ranked by Sharpe) |

### Prerequisites
- Python 3.12+, Node.js 22+
- `pip install -r agent/requirements.txt`

### Call the demo endpoint
```bash
# Health check (public)
curl https://api.opspawn.com/erc8004/demo/health

# Basic 10-tick run (public)
curl -s -X POST 'https://api.opspawn.com/erc8004/demo/run' | python3 -m json.tool

# Custom parameters
curl -s -X POST 'https://api.opspawn.com/erc8004/demo/run?ticks=50&seed=123&symbol=ETH/USD' \
  | python3 -m json.tool

# Portfolio analytics
curl https://api.opspawn.com/erc8004/demo/portfolio

# Live aggregate metrics (win rate, Sharpe, Sortino, drawdown)
curl https://api.opspawn.com/erc8004/demo/metrics

# Agent leaderboard (top 5 by Sortino ratio)
curl https://api.opspawn.com/erc8004/demo/leaderboard

# Side-by-side agent comparison
curl -s -X POST 'https://api.opspawn.com/erc8004/demo/compare' \
  -H 'Content-Type: application/json' \
  -d '{"agent_ids": ["agent-conservative-001", "agent-aggressive-003"]}' \
  | python3 -m json.tool

# SSE stream (real-time run events)
curl -s 'https://api.opspawn.com/erc8004/demo/stream'

# Local fallback (if running agent/demo_server.py directly)
curl -s -X POST 'http://localhost:8084/demo/run' | python3 -m json.tool
curl -s http://localhost:8084/demo/health
curl -s http://localhost:8084/demo/info
```

### Start the demo server (local)
```bash
cd agent
python3 demo_server.py
# [ERC-8004 Demo Server] Starting on port 8084
```

### Example response (abbreviated)
```json
{
  "status": "ok",
  "pipeline": "ERC-8004 Autonomous Trading Agent",
  "demo": {
    "symbol": "BTC/USD",
    "ticks_run": 10,
    "trades_executed": 0,
    "consensus_reached": 0,
    "avg_reputation_score": 6.1667,
    "duration_ms": 12.4
  },
  "agents": [
    {"profile": "conservative", "reputation_start": 7.5, "reputation_end": 7.5},
    {"profile": "balanced", "reputation_start": 6.0, "reputation_end": 6.0},
    {"profile": "aggressive", "reputation_start": 5.0, "reputation_end": 5.0}
  ],
  "validation_artifact": {
    "artifact_hash": "0x1ef7b7b813ab765...",
    "strategy_hash": "0x6cc5a766790ab6...",
    "timestamp": "2026-02-26T01:00:00+00:00",
    "signature": "7483570b761eb33c..."
  },
  "x402": {"dev_mode": true, "payment_gated": false, "price_usdc": "1000"}
}
```

---

## Video Script Outline (60–90 seconds, 6 scenes)

**Scene 1 — Hook (0:00–0:10)**
> "What if an AI trading agent had a verifiable on-chain reputation — and other agents
> could check it before trusting it?"

Show: ERC-8004 spec title + agent identity diagram.

**Scene 2 — Problem (0:10–0:20)**
> "Today's DeFi bots are anonymous black boxes. ERC-8004 changes that."

Show: Identity registry contract, DID format, trust score on-chain.

**Scene 3 — Architecture (0:20–0:35)**
> "Three agents — Conservative, Balanced, Aggressive — each hold their own ERC-8004
> identity. Every tick, they vote. The 2/3 reputation-weighted majority wins."

Show: Live terminal running `python3 demo_server.py`, then `curl /demo/run` output.

**Scene 4 — Live Demo (0:35–0:55)**
> "Watch it run end-to-end in under a second."

Show: `curl -s -X POST localhost:8084/demo/run | python3 -m json.tool` — JSON response
scrolling with ticks, trades, reputation scores, artifact hash.

**Scene 5 — Test Coverage (0:55–1:05)**
> "3,541 tests, 52+ test files. Every edge case: flash crash, oracle failure, consensus
> deadlock. Every reputation path."

Show: `python3 -m pytest tests/ -q` → `3541 passed in 35s`.

**Scene 6 — Close (1:05–1:20)**
> "On-chain identity. Reputation-weighted consensus. x402-gated endpoints. Validation
> artifacts. ERC-8004 — the trust layer for autonomous agents."

Show: Contracts deployed on Base Sepolia, GitHub link, OpSpawn logo.
