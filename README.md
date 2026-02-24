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

## Quickstart

### Prerequisites
- Node.js 22+, Python 3.12+
- MetaMask with Sepolia ETH

### Contracts
```bash
cd contracts
npm install
npx hardhat test          # Run tests
npx hardhat run scripts/deploy.ts --network sepolia  # Deploy
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

## ERC-8004 Standard

ERC-8004 defines a standard for autonomous AI agent identity on Ethereum:
- **Identity**: ERC-721 token representing the agent
- **Reputation**: On-chain feedback from clients
- **Validation**: Third-party verification of agent actions
- **Wallet**: EIP-1271 smart contract wallet for agent transactions

## License

MIT
