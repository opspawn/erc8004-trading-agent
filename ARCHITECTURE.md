# ERC-8004 Autonomous Trading Agent — Architecture

## System Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                   ERC-8004 Trading Agent (S46)                  │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   ┌─────────────┐    ┌──────────────────────────────────────┐  │
│   │  Price Feed │    │         10-Agent Swarm               │  │
│   │  (S45 GBM)  │───►│  quant-1 … quant-10                  │  │
│   │             │    │  momentum · mean-revert · arb         │  │
│   │ BTC/ETH/SOL │    │  trend · contrarian · hybrid          │  │
│   │ MATIC ticks │    └──────────────┬───────────────────────┘  │
│   └─────────────┘                   │                           │
│                                     ▼                           │
│   ┌─────────────────────────────────────────────────────────┐  │
│   │                Consensus Engine                         │  │
│   │  stake-weighted vote → 2/3 supermajority threshold      │  │
│   │  POST /api/v1/swarm/vote  ·  GET /api/v1/swarm/perf.    │  │
│   └──────────────────────────┬──────────────────────────────┘  │
│                               │                                 │
│                               ▼                                 │
│   ┌─────────────────────────────────────────────────────────┐  │
│   │                Validation Layer                         │  │
│   │  ValidationRegistry · artifact_hash · on-chain proof   │  │
│   │  Credora credit rating → position limit adjustments     │  │
│   └──────────────────────────┬──────────────────────────────┘  │
│                               │                                 │
│                               ▼                                 │
│   ┌─────────────────────────────────────────────────────────┐  │
│   │             Portfolio Risk Engine (S46)                 │  │
│   │  VaR 95%/99% (historical simulation)                    │  │
│   │  Sharpe · Sortino · Calmar ratios                       │  │
│   │  Cross-symbol correlation matrix (4×4)                  │  │
│   │  HHI concentration index · exposure dashboard           │  │
│   │  Position sizing: volatility / Half-Kelly / fixed       │  │
│   │                                                         │  │
│   │  GET  /api/v1/risk/portfolio                            │  │
│   │  POST /api/v1/risk/position-size                        │  │
│   │  GET  /api/v1/risk/exposure                             │  │
│   └──────────────────────────┬──────────────────────────────┘  │
│                               │                                 │
│                               ▼                                 │
│   ┌─────────────────────────────────────────────────────────┐  │
│   │            ERC-8004 Identity + Reputation               │  │
│   │  IdentityRegistry.sol (ERC-721 extension, Base Sepolia) │  │
│   │  ReputationRegistry.sol (on-chain scores)               │  │
│   │  AgentWallet.sol (EIP-1271 smart contract wallet)        │  │
│   └─────────────────────────────────────────────────────────┘  │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

## Data Flow

```
Live Price Tick
      │
      ▼
 10 Agents evaluate signal (momentum/arb/trend/etc.)
      │
      ▼
 Swarm Vote  ──────────────────────────────────────────┐
 (stake-weighted, 2/3 threshold)                       │
      │                                                │
      ├─ consensus_reached=true  → EXECUTE TRADE        │
      └─ consensus_reached=false → HOLD                 │
                                                       │
                              ┌────────────────────────┘
                              ▼
                    Portfolio Risk Check
                    ├─ VaR budget OK?        → proceed
                    ├─ Concentration < 40%?  → proceed
                    └─ Position size calc    → submit
                              │
                              ▼
                    ValidationRegistry
                    (artifact_hash + proof)
```

## Key Endpoints by Category

### Health & Info
| Endpoint | Description |
|----------|-------------|
| `GET /health` | `{status, version: "S46", tests: 6085, highlights}` |
| `GET /` | Full endpoint catalog |

### Risk Management (S46)
| Endpoint | Description |
|----------|-------------|
| `GET /api/v1/risk/portfolio` | VaR 95/99, Sharpe/Sortino/Calmar, correlation matrix |
| `POST /api/v1/risk/position-size` | Recommended size (volatility / Half-Kelly / fixed) |
| `GET /api/v1/risk/exposure` | Per-symbol notional, portfolio %, HHI concentration |

### Multi-Agent Swarm (S46)
| Endpoint | Description |
|----------|-------------|
| `POST /api/v1/swarm/vote` | 10 agents vote, stake-weighted 2/3 supermajority |
| `GET /api/v1/swarm/performance` | 24h PnL + Sharpe leaderboard for all 10 agents |

### Live Market Data (S45)
| Endpoint | Description |
|----------|-------------|
| `GET /api/v1/market/price/{symbol}` | Live GBM price + 24h stats |
| `GET /api/v1/market/prices` | All 4 symbols |
| `WS /api/v1/ws/prices` | WebSocket 1-second price stream |
| `POST /api/v1/agents/{id}/auto-trade` | Agent auto-trades on live feed |

### Core Demo
| Endpoint | Description |
|----------|-------------|
| `POST /demo/run` | 10-tick multi-agent pipeline end-to-end |
| `GET /demo/portfolio/snapshot` | Live portfolio snapshot |
| `POST /demo/backtest` | GBM historical backtest |

## Technology Stack

- **Language**: Python 3.12 (stdlib only — no FastAPI dependency for demo)
- **HTTP**: `http.server.ThreadingHTTPServer` (zero external deps)
- **Risk math**: Pure Python + `math` (VaR, Sharpe, Sortino, Calmar, correlation)
- **Smart contracts**: Solidity 0.8, deployed to Base Sepolia testnet
- **x402 payment**: Optional gate on `/demo/run` (bypassed in dev mode)
- **Tests**: 6,085 passing (pytest, `cd agent && python3 -m pytest tests/ -q`)

## Sprint History

| Sprint | Feature |
|--------|---------|
| S01–S20 | Core pipeline: ERC-8004 identity, reputation, consensus, x402 gate |
| S21–S30 | Compliance, circuit breakers, coordination, trustless validation |
| S31–S40 | Market intelligence, attribution, live feed, health dashboard |
| S41–S43 | Monte Carlo, strategy compare, cross-agent coordination |
| S44 | Paper trading, agent leaderboard |
| S45 | Live GBM price feed, WebSocket streaming, auto-trade |
| **S46** | **Portfolio risk (VaR/Sharpe/Sortino/Calmar), 10-agent swarm** |
