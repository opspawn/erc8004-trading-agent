# ERC-8004 Autonomous Trading Agent

> Multi-agent trading system with on-chain identity, reputation-weighted consensus, and x402 micropayment gating — built for the [lablab.ai ERC-8004 Hackathon](https://lablab.ai) (March 9–22, 2026, $50K USDC pool).

**Tests: 4,968 passing** | **Demo: live on port 8084** | **Contract: deployed on Base Sepolia**

---

## The Problem

DeFi trading agents are anonymous black boxes. There's no on-chain record of who ran a trade, whether the strategy was sound, or whether the agent has a track record of success. Clients can't evaluate agents. Agents can't prove their reputation.

## The Solution

ERC-8004 gives every agent a verifiable on-chain identity (ERC-721) and a reputation score updated by trade outcomes. Three specialist agents vote on every trade; a reputation-weighted 2/3 supermajority is required to execute. Bad actors get downvoted. Good agents build reputation over time.

---

## Key Features

| Feature | What it does |
|---------|-------------|
| **ERC-8004 Identity** | Each agent holds an on-chain ERC-721 token — tamper-proof, permanent record |
| **Reputation-Weighted Consensus** | 3 agents vote; higher-reputation agents carry more weight; requires 2/3 supermajority |
| **x402 Payment Gate** | Demo endpoint is micropayment-gated (`dev_mode=true` for judges — free) |
| **Credora Credit Ratings** | On-chain credit scores feed into position sizing and risk limits |
| **Backtester** | GBM historical simulation with Sharpe, drawdown, win rate |
| **Stress Tester** | 7 adversarial scenarios: flash crash, oracle failure, consensus deadlock |
| **Trade Ledger** | Every decision written to SQLite with deterministic tx_hash proof |

---

## 30-Second Quickstart

```bash
# 1. Health check (no setup needed)
curl https://api.opspawn.com/erc8004/demo/health

# 2. Run the full multi-agent demo pipeline
curl -s -X POST 'https://api.opspawn.com/erc8004/demo/run?ticks=10' | python3 -m json.tool

# 3. Portfolio snapshot
curl https://api.opspawn.com/erc8004/demo/portfolio/snapshot
```

**Or run locally:**
```bash
cd agent && python3 demo_server.py
# Server on http://localhost:8084
curl -s -X POST 'http://localhost:8084/demo/run?ticks=10' | python3 -m json.tool
```

---

## Architecture

```
┌──────────────────────────────────────────────────────┐
│              Multi-Agent Trading System               │
│                                                      │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  │
│  │ Conservative│  │  Balanced   │  │  Aggressive │  │
│  │   Agent     │  │   Agent     │  │   Agent     │  │
│  │ (ERC-8004)  │  │ (ERC-8004)  │  │ (ERC-8004)  │  │
│  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘  │
│         └────────────────┼────────────────┘          │
│                          ▼                           │
│              ┌───────────────────────┐               │
│              │  Reputation-Weighted  │               │
│              │  Consensus Engine     │               │
│              │  (2/3 supermajority)  │               │
│              └───────────┬───────────┘               │
│                          ▼                           │
│              ┌───────────────────────┐               │
│              │   x402 Payment Gate   │               │
│              │   Risk Manager        │               │
│              │   Trade Executor      │               │
│              └───────────┬───────────┘               │
└──────────────────────────┼───────────────────────────┘
                           ▼
┌──────────────────────────────────────────────────────┐
│              ERC-8004 Contracts (Base Sepolia)        │
│  ┌────────────────┐  ┌──────────────────┐            │
│  │ IdentityReg.   │  │ ReputationReg.   │            │
│  │ (ERC-721)      │  │ (on-chain scores)│            │
│  └────────────────┘  └──────────────────┘            │
│  ┌────────────────┐  ┌──────────────────┐            │
│  │ ValidationReg. │  │  AgentWallet     │            │
│  │ (trade proofs) │  │  (EIP-1271)      │            │
│  └────────────────┘  └──────────────────┘            │
└──────────────────────────────────────────────────────┘
```

---

## Smart Contracts

| Contract | Network | Address |
|----------|---------|---------|
| `IdentityRegistry.sol` | Base Sepolia | Deployed — see `contracts/deployment.json` |
| `ReputationRegistry.sol` | Base Sepolia | Deployed |
| `ValidationRegistry.sol` | Base Sepolia | Deployed |
| `AgentWallet.sol` | Base Sepolia | Deployed |

Also deployable to **Arbitrum Sepolia** (chain ID 421614) for Arbitrum Trailblazer 2.0 eligibility.

```bash
# Deploy to Arbitrum Sepolia
bash scripts/deploy-arbitrum-sepolia.sh

# Verify on Base Sepolia explorer
open https://sepolia.basescan.org
```

---

## API Endpoints

The demo server runs on **port 8084** (also proxied at `https://api.opspawn.com/erc8004/`).

| Method | Path | Description |
|--------|------|-------------|
| `GET`  | `/demo/health` | `{status:"ok", tests:6085, sprint:"S46", highlights:[...]}` |
| `GET`  | `/demo/info` | API index with all endpoints |
| `POST` | `/demo/run` | Run full 10-tick multi-agent demo |
| `GET`  | `/api/v1/risk/portfolio` | Portfolio VaR 95/99%, Sharpe, Sortino, Calmar, correlation matrix (S46) |
| `POST` | `/api/v1/risk/position-size` | Position sizing by risk budget + volatility: Kelly/volatility/fixed (S46) |
| `GET`  | `/api/v1/risk/exposure` | Per-symbol exposure + concentration risk HHI (S46) |
| `POST` | `/api/v1/swarm/vote` | 10-agent stake-weighted vote on trade signal (S46) |
| `GET`  | `/api/v1/swarm/performance` | 24h PnL + Sharpe leaderboard for all 10 swarm agents (S46) |
| `POST` | `/api/v1/demo/showcase` | **Judge showcase**: price tick + swarm vote + VaR + paper trade in one call (S47) |
| `GET`  | `/demo/leaderboard` | Agent leaderboard (`?sort_by=sharpe\|sortino\|pnl`) |
| `POST` | `/demo/backtest` | GBM historical backtest |
| `POST` | `/demo/consensus` | Multi-agent consensus vote |
| `GET`  | `/demo/stream` | Server-Sent Events live feed |

---

## Test Coverage

```bash
cd agent
pip install -r requirements.txt
python3 -m pytest tests/ -q --tb=short
# Expected: 6,088+ passed
```

**6,085+ tests across 47 sprints** covering:
- ERC-8004 identity registration and validation
- Reputation-weighted consensus algorithm
- x402 payment gate (dev and production mode)
- Credora credit rating integration
- Backtester: GBM, Sharpe, Sortino, drawdown
- Stress tester: flash crash, oracle failure, deadlock
- All HTTP endpoints (unit + integration)
- Strategy comparison and portfolio analytics

---

## Local Setup

```bash
# Agent (Python)
cd agent
pip install -r requirements.txt
python3 demo_server.py          # Demo API on :8084
python3 demo_cli.py --ticks 20  # CLI demo

# Contracts (Solidity)
cd contracts
npm install
npx hardhat test
npx hardhat run scripts/deploy.ts --network baseSepolia

# Dashboard (Next.js)
cd dashboard
npm install && npm run dev      # UI on :3000
```

---

## Demo Response (example)

```json
{
  "demo": {
    "ticks_run": 10,
    "trades_executed": 7,
    "avg_reputation_score": 7.24,
    "consensus_rate": 0.85
  },
  "agents": [
    {"profile": "conservative", "pnl_usd": 12.4, "reputation_delta": +0.1, "win_rate": 0.75},
    {"profile": "balanced",     "pnl_usd":  8.2, "reputation_delta": +0.05, "win_rate": 0.625},
    {"profile": "aggressive",   "pnl_usd": 15.1, "reputation_delta": +0.2, "win_rate": 0.70}
  ],
  "validation_artifact": {
    "artifact_hash": "0xabc123...",
    "win_rate": 0.72,
    "signed_at": 1740614400
  },
  "x402": {"dev_mode": true, "price_usdc": "1000"}
}
```

---

## License

MIT
