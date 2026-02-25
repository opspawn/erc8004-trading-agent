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

The system integrates three distinct data layers for intelligent decision-making:
(1) **Credora credit tiers** (AAA–CCC) as Kelly Criterion multipliers to scale
position sizes by protocol risk, (2) **RedStone/Surge oracle pricing** for
real-time market data and on-chain settlement, and (3) an **ERC-8004
ReputationRegistry** that records every trade outcome on-chain and updates the
agent's trust score. This creates a self-reinforcing loop where the agent's
trade history directly influences its future position sizing.

Uniquely, the project implements `AgentCreditHistory` — an on-chain credit score
system for the agent itself, mirroring how Credora rates DeFi protocols but
applied to autonomous trading agents. As the agent accumulates wins in higher-rated
protocols, its own credit score improves, unlocking larger position sizes. This
aligns incentives for agents to trade responsibly: reckless behavior degrades their
own credit tier, reducing their capacity to take future positions.

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
| Testing | pytest (1,400+ tests), Hardhat/Mocha (113 Solidity tests) |
| Standards | ERC-8004 (agent identity, reputation, validation) |

---

## Test Coverage

- **Python**: 1,400+ tests across 27 test files
- **Solidity**: 113 tests (AgentRegistry, ReputationRegistry, TradeValidator, RiskRouter)
- **Integration**: End-to-end tests simulating full trade lifecycle with Credora,
  oracle validation, and on-chain reputation updates

---

## Repository Structure

```
erc8004-trading-agent/
├── agent/                    # Python trading agent
│   ├── risk_manager.py       # Pre-trade risk validation + Credora integration
│   ├── credora_client.py     # Credora ratings + AgentCreditHistory
│   ├── trader.py             # Core trade execution
│   ├── strategy_runner.py    # Kelly Criterion strategies
│   ├── surge_router.py       # Surge liquidity routing
│   ├── oracle_client.py      # RedStone oracle client
│   ├── reputation.py         # On-chain reputation logger
│   ├── validator.py          # Trade validation
│   └── tests/                # 1,400+ pytest tests
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
