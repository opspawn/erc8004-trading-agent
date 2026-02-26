# ERC-8004 Trading Agent

An autonomous AI trading agent with on-chain identity, reputation, and validation — built on the ERC-8004 standard.

**Hackathon**: [lablab.ai ERC-8004 Trading Agents](https://lablab.ai) — March 9–22, 2026
**Prize**: $10,000 first place

## Overview

This project implements a fully autonomous trading agent that:
1. **Registers its identity** on-chain via ERC-8004 `IdentityRegistry` (ERC-721 extension)
2. **Executes trades** on prediction markets (Polymarket, Manifold)
3. **Submits validation requests** for each trade outcome
4. **Accumulates reputation** based on verified trade performance
5. **Exposes a smart contract wallet** (EIP-1271) for on-chain interactions

## Architecture

```
┌─────────────────────────────────────────┐
│          Trading Agent (Python)          │
│  ┌─────────┐  ┌─────────┐  ┌─────────┐ │
│  │ Trader  │  │Registry │  │Validator│ │
│  └────┬────┘  └────┬────┘  └────┬────┘ │
└───────┼────────────┼────────────┼───────┘
        │            │            │
        ▼            ▼            ▼
┌─────────────────────────────────────────┐
│         ERC-8004 Contracts (Sepolia)     │
│  ┌──────────────┐  ┌────────────────┐   │
│  │IdentityReg.  │  │ReputationReg.  │   │
│  └──────────────┘  └────────────────┘   │
│  ┌────────────────┐ ┌──────────────┐    │
│  │ValidationReg.  │ │ AgentWallet  │    │
│  └────────────────┘ └──────────────┘    │
└─────────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────────┐
│         Dashboard (Next.js)              │
│  Live agent stats, reputation, trades    │
└─────────────────────────────────────────┘
```

## Smart Contracts

| Contract | Description |
|----------|-------------|
| `IdentityRegistry.sol` | ERC-721 agent identity registry |
| `ReputationRegistry.sol` | On-chain feedback and reputation scoring |
| `ValidationRegistry.sol` | Trade outcome validation requests |
| `AgentWallet.sol` | EIP-1271 smart contract wallet |

## Supported Networks

| Network | Chain ID | RPC | Explorer |
|---------|----------|-----|----------|
| Ethereum Sepolia | 11155111 | `https://rpc.sepolia.org` | [etherscan](https://sepolia.etherscan.io) |
| Base Sepolia | 84532 | `https://sepolia.base.org` | [basescan](https://sepolia.basescan.org) |
| **Arbitrum Sepolia** | **421614** | `https://sepolia-rollup.arbitrum.io/rpc` | [arbiscan](https://sepolia.arbiscan.io) |

### Arbitrum Sepolia Support

ERC-8004 contracts are deployable to Arbitrum Sepolia (chain ID 421614), enabling eligibility for the [Arbitrum Trailblazer 2.0](https://arbitrum.io) grant program.

```bash
# Deploy to Arbitrum Sepolia
bash scripts/deploy-arbitrum-sepolia.sh
# Outputs: deployment-arbitrum-sepolia.json with contract address and tx hash
```

The Hardhat config (`contracts/hardhat.config.ts`) includes the `arbitrumSepolia` network using the public RPC endpoint `https://sepolia-rollup.arbitrum.io/rpc`. Override with `ARBITRUM_SEPOLIA_RPC_URL` in `.env` for a private RPC.

## Quickstart

### Prerequisites
- Node.js 22+, Python 3.12+
- MetaMask with Sepolia ETH (or Arbitrum Sepolia ETH from the [Alchemy faucet](https://www.alchemy.com/faucets/arbitrum-sepolia))

### Contracts
```bash
cd contracts
npm install
npx hardhat test          # Run tests
npx hardhat run scripts/deploy.ts --network sepolia          # Deploy to Sepolia
npx hardhat run scripts/deploy.ts --network arbitrumSepolia  # Deploy to Arbitrum Sepolia
# Or use the convenience script:
bash scripts/deploy-arbitrum-sepolia.sh
```

### Agent
```bash
cd agent
pip install -r requirements.txt
cp .env.example .env      # Fill in keys
python main.py
```

### Dashboard
```bash
cd dashboard
npm install
npm run dev
```

## 5-Minute Demo

See the full agent pipeline execute end-to-end in under a minute, with no external
dependencies. The dry-run fetches simulated market prices, generates trade signals,
validates them through the risk engine, simulates order execution, and writes every
decision to a local SQLite ledger with a deterministic `tx_hash` proof placeholder.

### Run the demo

```bash
cd agent
python3 demo_cli.py --ticks 20 --symbol BTC/USD --seed 42
```

**Expected output:**
```
========================================================================
  ERC-8004 Trading Agent — Pipeline Dry-Run
  Agent:   demo-agent-...
  Symbol:  BTC/USD
  Ticks:   20
  Capital: $10,000.00
========================================================================

TICK        PRICE SIGNAL RISK          SIZE     NOTIONAL TX_HASH
------------------------------------------------------------------------
1        65130.09 BUY    OK          0.0154      1000.00 0x566221c4b6ae85fa44...
2        65167.79 BUY    OK          0.0138       900.00 0x225a79958cfe35ee90...
3        65119.62 SELL   OK          0.0124       810.00 0xd7f28d8a8a4ff93ebc...
...
------------------------------------------------------------------------

Summary
-------
  Ticks processed  : 20
  Trades executed  : 17
  Trades rejected  : 3
  Total notional   : $13,150.60
  BUY / SELL       : 9 / 8
  Price start→end  : $65000.00 → $65074.63
  Price return     : +0.11%
  Final capital    : $8,304.70
  Elapsed          : 1.8ms

  Ledger written to: :memory:
========================================================================
```

### Pipeline steps (visible in the trace)

| Step | Module | What happens |
|------|--------|--------------|
| **Market fetch** | `demo_cli._gbm_prices` | GBM price ticks for the chosen symbol |
| **Signal** | `demo_cli._compute_signal` | Momentum signal: BUY / SELL / HOLD |
| **Risk check** | `demo_cli._risk_check` | Max position size + drawdown validation |
| **Order sim** | `demo_cli.DemoPipeline` | Paper trade: capital updated, position tracked |
| **Ledger write** | `trade_ledger.TradeLedger` | SQLite row: agent_id, market, side, size, price, tx_hash |

### Inspect the trade ledger

```bash
# Persist to file and inspect
python3 demo_cli.py --ticks 50 --db /tmp/trades.db

# Query via Python
python3 - <<'EOF'
from trade_ledger import TradeLedger
tl = TradeLedger("/tmp/trades.db")
for e in tl.get_entries(side="BUY"):
    print(e.tx_hash, e.market, e.price, e.notional)
print(tl.get_summary().to_dict())
tl.close()
EOF
```

### Machine-readable JSON output

```bash
python3 demo_cli.py --ticks 10 --json | python3 -m json.tool | head -30
```

### Run with different symbols or capital

```bash
python3 demo_cli.py --symbol ETH/USD --ticks 30 --capital 50000
python3 demo_cli.py --symbol SOL/USD --ticks 100 --seed 7
```

---

## Live Demo

**Project**: ERC-8004 Autonomous Trading Agent
**Tests**: 3,273 passing

Hit the live demo endpoint — no wallet, no setup:

```bash
curl -s -X POST 'http://localhost:8084/demo/run?ticks=10' | python3 -m json.tool
```

**Start the demo server:**
```bash
cd agent
python3 demo_server.py
# or via systemd:
# sudo systemctl start erc8004-demo
```

**Key features:**
- **On-chain trust loop** — each agent holds an ERC-8004 identity; every trade updates reputation on-chain
- **x402 payment gate** — endpoint is micropayment-gated; `dev_mode=true` for judges
- **Multi-agent consensus** — 3 specialist agents vote; 2/3 reputation-weighted majority required
- **Backtester** — historical strategy comparison with Sharpe, drawdown, win rate
- **Stress tester** — 7 adversarial scenarios (flash crash, oracle failure, consensus deadlock)
- **Validation artifact** — signed JSON proof of session performance, stored on disk

**Response includes:**
```json
{
  "demo": { "ticks_run": 10, "trades_executed": 4, "avg_reputation_score": 6.17 },
  "agents": [{"profile": "conservative", "pnl_usd": 12.4, "reputation_delta": 0.1}],
  "validation_artifact": { "artifact_hash": "0x...", "win_rate": 0.75 },
  "x402": { "dev_mode": true, "price_usdc": "1000" }
}
```

## Public Endpoints

The demo server runs on port 8084 and is proxied via nginx. All endpoints are
x402-gated with `dev_mode=True` (free for judges — no wallet required).

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET`  | `https://api.opspawn.com/erc8004/demo/health` | Server health check |
| `POST` | `https://api.opspawn.com/erc8004/demo/run` | Run full 10-tick demo pipeline |
| `GET`  | `https://api.opspawn.com/erc8004/demo/portfolio` | Portfolio analytics from last run |

**Quick test:**
```bash
# Health check
curl https://api.opspawn.com/erc8004/demo/health

# Run demo (optional params: ?ticks=10&seed=42&symbol=BTC%2FUSD)
curl -s -X POST 'https://api.opspawn.com/erc8004/demo/run?ticks=10' | python3 -m json.tool

# Portfolio analytics
curl https://api.opspawn.com/erc8004/demo/portfolio
```

**Portfolio endpoint response:**
```json
{
  "source": "live",
  "agent_profiles": [
    {"agent_id": "...", "strategy": "conservative", "win_rate": 0.75, "total_pnl": 12.4, "reputation_score": 7.82}
  ],
  "consensus_stats": {"avg_agreement_rate": 0.72, "supermajority_hits": 7, "veto_count": 1},
  "risk_metrics": {"max_drawdown": -0.042, "sharpe_estimate": 1.38, "volatility": 0.021}
}
```

---

## ERC-8004 Standard

ERC-8004 defines a standard for autonomous AI agent identity on Ethereum:
- **Identity**: ERC-721 token representing the agent
- **Reputation**: On-chain feedback from clients
- **Validation**: Third-party verification of agent actions
- **Wallet**: EIP-1271 smart contract wallet for agent transactions

## License

MIT
