# ERC-8004 Trading Agent — Competition Submission

## Project Information

| Field | Details |
|-------|---------|
| **Title** | ERC-8004 Autonomous Trading Agent |
| **Team** | OpSpawn |
| **Demo URL** | https://erc8004-trading-agent.vercel.app |
| **Video URL** | TBD |
| **Category** | DeFi / Agent Infrastructure |
| **Deadline** | March 22, 2026 — lablab.ai |

---

## Description

**ERC-8004 Trading Agent** is a fully autonomous AI trading system built on the
ERC-8004 standard for on-chain agent identity, reputation, and validation. The
agent holds a verifiable DID (Decentralized Identifier), accumulates an on-chain
reputation score through each trade, and applies institutional-grade risk
management using Credora credit ratings before executing any position.

The system integrates five distinct data layers for intelligent decision-making:
(1) **Credora credit tiers** (AAA–CCC) as Kelly Criterion multipliers to scale
position sizes by protocol risk, (2) **RedStone/Surge oracle pricing** for
real-time market data and on-chain settlement, (3) an **ERC-8004
ReputationRegistry** that records every trade outcome on-chain and updates the
agent's trust score, (4) a **multi-agent mesh coordinator** that runs three
specialist agents simultaneously with different risk profiles and requires 2/3
consensus before executing — directly demonstrating ERC-8004's multi-agent
identity and trust model, and (5) a **sentiment signal layer** that aggregates
signals from multiple sources to modulate position sizing in real time.

Uniquely, the project implements `AgentCreditHistory` — an on-chain credit score
system for the agent itself, mirroring how Credora rates DeFi protocols but
applied to autonomous trading agents. As the agent accumulates wins in higher-rated
protocols, its own credit score improves, unlocking larger position sizes. This
aligns incentives for agents to trade responsibly: reckless behavior degrades their
own credit tier, reducing their capacity to take future positions.

The **multi-agent mesh** is the strongest proof of ERC-8004's value: three agents
(Conservative, Balanced, Aggressive) each hold their own on-chain ERC-8004 identity
and reputation score. Their votes are weighted by reputation — the most trusted
agent has the most influence. No single agent can cause a bad trade; it takes a
reputation-weighted majority.

---

## ERC-8004 Standard Integration

This project implements the ERC-8004 standard in full:

- **On-chain Agent Identity**: Each agent receives a unique DID in the format
  `eip155:{chainId}:{address}:{agentId}` stored in the `AgentRegistry` contract.
- **Reputation System**: The `ReputationRegistry` contract records trade outcomes
  and computes a weighted score (0–10) from validator feedback entries.
- **Trade Validation**: The `TradeValidator` contract enforces pre-trade checks
  (oracle deviation, position limits, agent reputation threshold) before any
  position can be opened.
- **Capability Registry**: Agent capabilities (trading, staking, LP) are registered
  on-chain and can be queried by other contracts or agents.

---

## Credora Integration

Credora provides institutional-grade risk ratings for DeFi protocols. Our
integration uses these ratings as Kelly Criterion multipliers:

| Credora Tier | Rating | Kelly Multiplier | Use Case |
|-------------|--------|-----------------|---------|
| AAA | Highest quality | 1.00 | ETH, BTC, USDC, Chainlink |
| AA  | Very high quality | 0.90 | Aave, Uniswap, Maker |
| A   | High quality | 0.80 | Compound, Curve, Lido |
| BBB | Upper medium grade | 0.65 | Balancer, Yearn, Synthetix |
| BB  | Speculative | 0.50 | GMX, dYdX |
| B   | High risk | 0.35 | Osmosis |
| CCC | Near-distressed | 0.20 | Unverified protocols |
| NR  | Not rated | 0.10 | Unknown protocols |

A minimum grade floor can be configured: e.g., `credora_min_grade=BBB` will
reject trades on any protocol rated below investment grade.

---

## Surge / RedStone Integration

- **SurgeRouterBase**: Abstracts over the Surge liquidity routing protocol for
  optimal trade execution paths.
- **OracleClient**: Fetches real-time pricing from RedStone oracles with
  configurable deviation thresholds.
- **On-chain Risk Check**: `RiskRouter.checkRisk()` validates oracle deviation
  before settling trades, mirroring the Python-side `check_oracle_risk()` logic.

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Smart Contracts | Solidity 0.8.x, Hardhat, OpenZeppelin |
| Agent Runtime | Python 3.12, asyncio |
| Risk Engine | Kelly Criterion + Credora credit ratings |
| Oracle | RedStone (via OracleClient), Surge routing |
| Dashboard | Next.js 15, React, Recharts, Tailwind CSS |
| Deployment | Vercel (frontend), Sepolia testnet (contracts) |
| Testing | pytest (1,903+ tests), Hardhat/Mocha (113 Solidity tests) |
| Standards | ERC-8004 (agent identity, reputation, validation) |

---

## Test Coverage

- **Python**: 1,903+ tests across 34 test files
- **Solidity**: 113 tests (AgentRegistry, ReputationRegistry, TradeValidator, RiskRouter)
- **Integration**: End-to-end tests simulating full trade lifecycle with Credora,
  oracle validation, on-chain reputation updates, multi-agent mesh consensus,
  and sentiment signal modulation

## S14: WebSocket Signal Server + Health API + Paper Trader (Latest)

Sprint 14 adds three live-demo-ready features that make the agent compelling
for judge evaluation with real-time data and simulation results:

### WebSocket Signal Server (`agent/signal_server.py`)
Live trade signal broadcast via asyncIO WebSockets:
- **Broadcasts** every 5 seconds: `{timestamp, signal_type, protocol, direction, confidence, agent_id, sequence, mesh_consensus}`
- **Signal types**: BUY, SELL, HOLD, REBALANCE — generated by real mesh coordinator
- **Multi-client**: handles multiple concurrent WebSocket connections
- **Zero frameworks**: pure `websockets` library + asyncio
- **Tests**: 43 unit tests covering message format, signal types, connection lifecycle, broadcast logic

### Agent Health API (`agent/health_api.py`)
REST health & metrics server using Python stdlib only (no FastAPI/Flask):
- **GET /health** → `{status, uptime_seconds, agents: [{id, credit_tier, reputation, last_trade, trades_today}]}`
- **GET /metrics** → `{total_trades, win_rate, sharpe_ratio, max_drawdown, portfolio_value, total_pnl}`
- **GET /agents** → full ERC-8004 identity roster (registry addresses, scores, chain)
- Background daemon thread, instant start/stop, CORS headers included
- **Tests**: 60 unit tests covering all endpoints, response format, error cases, routing

### Paper Trading Simulation (`agent/paper_trader.py`)
24-hour GBM-driven paper trading simulation running in ~1 second:
- **Geometric Brownian Motion** price generation (pure math, no numpy)
- Full loop: synthetic prices → mesh consensus → Kelly sizing → risk check → execute → update reputation
- **Report**: 24h P&L, Sharpe ratio, max drawdown, win rate, final portfolio value, per-trade audit trail
- 7 assets (ETH, BTC, AAVE, UNI, COMP, CRV, GMX) with asset-specific volatility
- **Tests**: 100 unit tests covering GBM, trade logic, stats computation, simulation run

**Sprint 14 total new tests: 203**

---

## S13: Backtesting Engine + Portfolio Optimizer

Sprint 13 adds two high-value modules for quantitative competition scoring:

### Backtesting Engine (`agent/backtester.py` — extended)
Historical simulation with ERC-8004 reputation-aware strategies:
- **Sharpe ratio, max drawdown, win rate, P&L** computed per strategy
- **`mesh_consensus` strategy**: aggregates trend + mean_reversion + momentum signals,
  requires 2/3 signal consensus — directly mirrors the ERC-8004 multi-agent voting model
- **`BacktestRegistry`**: on-chain analog (mock) that stores backtest results with
  unique IDs, supports query by strategy/token, and retrieves best/worst performers
- **CLI**: `python -m agent.backtester --days 30 --strategy mesh_consensus --compare`

### Portfolio Optimizer (`agent/portfolio_optimizer.py`)
Mean-variance optimization with ERC-8004 reputation weighting:
- **Credora-weighted optimization**: Kelly multiplier scales asset weights by credit tier
  (AAA→1.0x, NR→0.10x) — protocols with higher reputation receive higher allocation
- **Concentration caps by tier**: AAA max 40%, NR max 5% — directly tied to Credora ratings
- **5% drift trigger**: automatic rebalancing detection when any position drifts >5%
- **Integration**: `validate_rebalance_with_risk_manager()` passes orders through
  `RiskManager.validate_trade_with_sentiment()` — full risk pipeline for rebalances
- **Dashboard**: BacktestResults panel (Sharpe/drawdown table) + PortfolioWeights chart

### Dashboard Updates (`dashboard/components/AgentDashboard.tsx`)
- **BacktestResults panel**: strategy comparison table with Sharpe, max drawdown, win rate
- **PortfolioWeights panel**: visual weight bars with Credora tier badges,
  drift indicators, and rebalance status

---

## S12: Multi-Agent Mesh + Sentiment Signals

Sprint 12 added two major modules:

### Multi-Agent Mesh Coordinator (`agent/mesh_coordinator.py`)
Three specialist agents run simultaneously, each with ERC-8004 on-chain identity:
- **ConservativeAgent**: min grade A, Kelly 15%, max position 5% — starts at rep 7.0/10
- **BalancedAgent**: min grade BBB, Kelly 25%, max position 10% — starts at rep 6.5/10
- **AggressiveAgent**: min grade BB, Kelly 35%, max position 15% — starts at rep 5.5/10

The `MeshCoordinator` requires ≥2/3 agents to vote BUY or SELL before executing.
Votes are weighted by each agent's current ERC-8004 reputation score, creating
a trust-proportional consensus. After each trade, all participating agents receive
reputation updates — wins improve scores, losses reduce them.

### Sentiment Signal Layer (`agent/sentiment_signal.py`)
Aggregates market sentiment from multiple sources with confidence-weighted averaging:
- **Positive** (score > 0.3): +10% position size boost
- **Negative** (score < -0.3): -20% position size reduction
- **Extreme Negative** (score < -0.7): trade blocked entirely

`RiskManager.validate_trade_with_sentiment()` integrates sentiment into the core
risk pipeline. Stale signals (>1 hour) are automatically excluded.

---

## Repository Structure

```
erc8004-trading-agent/
├── agent/                    # Python trading agent
│   ├── risk_manager.py       # Pre-trade risk validation + Credora + sentiment
│   ├── credora_client.py     # Credora ratings + AgentCreditHistory
│   ├── mesh_coordinator.py   # Multi-agent mesh (Conservative/Balanced/Aggressive)
│   ├── sentiment_signal.py   # Sentiment signal aggregator + Kelly modifier
│   ├── trader.py             # Core trade execution
│   ├── strategy_runner.py    # Kelly Criterion strategies
│   ├── surge_router.py       # Surge liquidity routing
│   ├── oracle_client.py      # RedStone oracle client
│   ├── reputation.py         # On-chain reputation logger
│   ├── validator.py          # Trade validation
│   └── tests/                # 1,700+ pytest tests
├── contracts/                # Solidity ERC-8004 implementation
│   ├── AgentRegistry.sol
│   ├── ReputationRegistry.sol
│   ├── TradeValidator.sol
│   └── RiskRouter.sol
└── dashboard/                # Next.js live dashboard
    └── app/page.tsx          # Real-time agent monitoring UI
```

---

## How to Run

```bash
# Python agent tests
cd agent && python3 -m pytest --tb=short -q

# Solidity tests
npx hardhat test

# Dashboard (local)
npm run dev

# Dashboard (live)
https://erc8004-trading-agent.vercel.app
```

---

*Built by OpSpawn — autonomous AI agent infrastructure.*
*ERC-8004 standard: https://github.com/ethereum/EIPs (draft)*
