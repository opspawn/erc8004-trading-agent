# ERC-8004: On-Chain Trading Agent Identity Standard

> A Solidity standard for registering autonomous trading agents on-chain, with a reference implementation featuring a **10-agent swarm**, **portfolio risk engine**, and **live demo server**.

**6,170 tests passing** | **Sprint S50** | **Base Sepolia + Arbitrum Sepolia** | **lablab.ai Hackathon 2026**

---

## The Problem

DeFi trading agents are anonymous black boxes. There's no on-chain record of who ran a trade, whether the strategy was sound, or whether the agent has a track record of success.

## The Solution

**ERC-8004** gives every agent a verifiable on-chain identity (ERC-721 extension) with a reputation score updated by trade outcomes. A 10-agent swarm with stake-weighted consensus ensures no single agent can act unilaterally.

---

## Live Demo

```bash
# 1. Start the server
pip install -r requirements.txt
python agent/demo_server.py

# 2. Health check (see all 5 demo steps at once)
curl http://localhost:8084/demo/health
```

Or open [`docs/demo.html`](docs/demo.html) in any browser — shows all 5 demo steps with real captured outputs.

---

## Key Features

| Feature | Details |
|---------|---------|
| **10-Agent Swarm** | quant-1..quant-10 with 6 distinct strategies (momentum, mean-revert, arb, trend, contrarian, hybrid) |
| **Risk Engine** | VaR 95%/99% (historical simulation), Sharpe, Sortino, Calmar, max drawdown |
| **Position Sizing** | Volatility-based, Half-Kelly criterion, fixed-fraction |
| **Reputation Consensus** | Stake-weighted 2/3 supermajority required; bad agents lose voting weight |
| **x402 Payment Gate** | `/demo/run` is micropayment-gated; `dev_mode=true` for judges (free) |
| **Performance Tracking** | Win rate, PnL, drawdown across 10 agents |
| **6,170 Tests** | 100% pass rate across all components (run in ~2 min) |

---

## Demo Endpoints

| Endpoint | What it shows |
|----------|--------------|
| `GET /demo/health` | Server status, test count, version |
| `POST /api/v1/swarm/vote` | 10 agents vote on BTC-USD BUY/SELL signal |
| `GET /api/v1/risk/portfolio` | VaR 95/99%, per-symbol correlation matrix |
| `GET /api/v1/performance/summary` | Live performance metrics across all agents |
| `POST /api/v1/demo/showcase` | Full 4-step pipeline: price → vote → risk → execute |

```bash
# Run all 5 demo steps in sequence:
python scripts/record_demo.py
```

---

## Architecture

```
ERC-8004 On-Chain Registry (Base Sepolia)
    │
    ├── IdentityRegistry.sol   ← ERC-721 agent identity tokens
    ├── ReputationRegistry.sol ← on-chain reputation scores
    ├── ValidationRegistry.sol ← trade outcome proofs
    └── AgentWallet.sol        ← EIP-1271 smart contract wallet
              │
              ▼
    Swarm Coordinator (stake-weighted voting)
    ├── quant-1  (momentum)      stake: 12
    ├── quant-2  (mean_revert)   stake: 10
    ├── quant-3  (arb)           stake: 11
    ├── quant-4  (trend)         stake: 9
    ├── quant-5  (contrarian)    stake: 8
    ├── quant-6  (hybrid)        stake: 10
    ├── quant-7  (momentum)      stake: 11
    ├── quant-8  (trend)         stake: 9.5
    ├── quant-9  (mean_revert)   stake: 8.5
    └── quant-10 (contrarian)    stake: 10
              │
              ▼
    Risk Manager
    ├── VaR 95%/99% (historical simulation)
    ├── Kelly criterion position sizing
    └── Herfindahl concentration index
              │
              ▼
    Execution Engine (paper trading / DEX)
```

---

## Quick Evaluation

```bash
# Clone and run tests (2 min)
git clone https://github.com/opspawn/erc8004-trading-agent
cd erc8004-trading-agent
pip install -r requirements.txt
pytest agent/tests/ -q --tb=no
# → 6170 passed

# Start demo server
python agent/demo_server.py &

# Run all 5 demo steps
python scripts/record_demo.py
```

---

## Deployment

Configured for [Render.com](https://render.com) via `render.yaml`. Health check: `/demo/health`.

Live: `https://api.opspawn.com/erc8004/demo/health`

Contracts deployed on **Arbitrum Sepolia** (see `contracts/deployment-arbitrum-sepolia.json`). Deploy script: `scripts/deploy-arbitrum-sepolia.sh`.

---

## Submission

See [docs/SUBMISSION_CHECKLIST.md](docs/SUBMISSION_CHECKLIST.md) for full status.
See [JUDGE_DEMO.md](JUDGE_DEMO.md) for step-by-step 10-minute walkthrough.
