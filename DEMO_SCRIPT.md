# ERC-8004 Trading Agent — 3-Minute Demo Script (S44)

> **Target audience**: Hackathon judges / technical evaluators
> **Format**: Screen recording + voice-over
> **Runtime**: ~3 minutes
> **Sprint**: S44 — Agent Performance Leaderboard + Paper Trading + 5,746 Tests

## S44 Feature Highlights

### Agent Performance Leaderboard
- **GET /api/v1/agents/leaderboard** — Real-time ranked leaderboard across all agents
  - Tracks: PnL, win rate, Sharpe ratio, trade count, avg return per trade
  - Filters: timeframe (1h/24h/7d/30d), min_trades, strategy_type
  - Pagination: page/page_size for large agent populations
- **POST /api/v1/agents/{id}/record-trade** — Record trade outcome; updates leaderboard live
- **GET /api/v1/agents/{id}/stats** — Deep stats: max win/loss, best streak, Sharpe

### Paper Trading Feed
- **POST /api/v1/trading/paper/order** — Simulated market order with 0.1% slippage + 0.05% fee
- **GET /api/v1/trading/paper/positions** — View all open paper positions by agent
- **POST /api/v1/trading/paper/close** — Close position; calculates gross PnL, fees, net PnL
- **GET /api/v1/trading/paper/history** — Full trade history with agent/symbol filters

### Demo Scenario (Wow Factor)
- **POST /api/v1/demo/run-leaderboard-scenario** — One call seeds 5 agents, runs 100 trades, returns live leaderboard
  - Agents: trend_follower, mean_reverter, momentum_rider, arb_detector, claude_strategist
  - claude_strategist has highest alpha bias (0.6% per trade vs 0.3% baseline)
  - Deterministic (seed=42) — reproducible for judge demos

```bash
# One-liner judge demo:
curl -s -X POST http://localhost:8084/api/v1/demo/run-leaderboard-scenario \
  | python3 -m json.tool | head -60
```

---

---

## Scene 1 — Problem Statement (0:00 – 0:30)

**Show**: Split screen — left: AI agent making trades; right: question mark "Who is this agent? Can we trust it?"

**Voice-over**:
> "AI agents are now making autonomous financial decisions — executing trades,
> managing portfolios, allocating capital. But there's a fundamental trust gap.
> When an AI agent executes a trade, there's no verifiable on-chain identity
> tying that action to a known, auditable entity.
> ERC-8004 solves this. It's an Ethereum standard for agent financial identity —
> giving every autonomous trading agent a permanent, verifiable credential
> backed by real risk data."

**On-screen text**:
- "Problem: AI agents make financial decisions with zero on-chain accountability"
- "Solution: ERC-8004 — verifiable identity + risk validation for trading agents"

---

## Scene 2 — Live Dashboard Demo (0:30 – 1:30)

**Show**: Browser pointing at https://erc8004-trading-agent.vercel.app

**Voice-over**:
> "Here's the live dashboard — deployed to Vercel, no localhost required.
> The agent identity card shows our ERC-8004 token: Agent ID #1 on Sepolia,
> with a reputation score of 87 out of 100.
> The Credora rating panel shows a BBB investment-grade rating,
> which sets a Kelly multiplier of 0.70 — meaning the agent can deploy
> up to 70% of its Kelly criterion position size.
> Watch the live P&L chart — the agent is running its trend-following strategy,
> currently up 12.4% since inception.
> And the risk validator is active — any trade that violates position limits
> or Credora thresholds is rejected before execution."

**On-screen** (navigate live dashboard):
1. Agent identity card → show ERC-8004 token ID + Sepolia address
2. Credora ratings panel → hover over BBB tier → show Kelly multiplier
3. Live P&L chart → show cumulative returns curve
4. Recent trades feed → show last 5 trades with status

---

## Scene 3 — ERC-8004 Identity + Validation Registry (1:30 – 2:00)

**Show**: Sepolia Etherscan + contract interactions

**Voice-over**:
> "Under the hood, the agent's identity is minted as an ERC-8004 NFT on Sepolia.
> The IdentityRegistry contract links the agent's Ethereum address to its
> Decentralized Identifier — eip155:11155111 colon contract colon tokenId 1.
> Every trade decision is validated by the on-chain ValidationRegistry,
> which maintains an immutable audit trail of risk checks.
> This is the foundation for agent accountability in DeFi."

**On-screen**:
- Etherscan: `IdentityRegistry.mint()` transaction
- Show token metadata: agentDID, reputationScore, credoraRating
- ValidationRegistry: last 3 validation events

---

## Scene 4 — Backtester Results (2:00 – 2:30)

**Show**: Terminal running backtester, then stats output

**Voice-over**:
> "Before deploying capital, every strategy is backtested with our GBM simulator.
> Running 252 trading days of ETH data with the trend-following strategy:
> Sharpe ratio: 1.84 — excellent risk-adjusted return.
> Max drawdown: 8.2% — well within our 15% risk limit.
> Win rate: 63%. Profit factor: 2.1.
> The Credora BBB rating allows full position sizing on these results."

**On-screen**:
```
$ python3 -c "
from backtester import Backtester
bt = Backtester(initial_capital=10000, position_size_pct=0.10)
bars = bt.generate_synthetic_prices('ETH', days=252, drift=0.0005)
trades = bt.run(bars, 'trend')
s = bt.compute_stats(trades)
print(f'Sharpe: {s.sharpe_ratio:.2f}')
print(f'Max DD: {s.max_drawdown_pct:.1f}%')
print(f'Win Rate: {s.win_rate:.0%}')
print(f'Profit Factor: {s.profit_factor:.1f}')
"

Sharpe: 1.84
Max DD: 8.2%
Win Rate: 63%
Profit Factor: 2.1
```

---

## Scene 5 — Architecture Walkthrough (2:30 – 3:00)

**Show**: Architecture diagram (text-based)

**Voice-over**:
> "The architecture flows in five stages:
> First, the Claude AI Strategist generates trade signals with confidence scores.
> Second, the Oracle Client fetches live prices from Chainlink feeds.
> Third, the Risk Manager validates against Credora ratings and position limits.
> Fourth, the Surge Router executes the trade through DeFi vaults.
> Fifth, the Validation Registry records the decision on-chain via ERC-8004.
> New in S43: cross-agent coordination — agents broadcast signals, vote on trades,
> and resolve conflicts via majority consensus before any execution.
> 5,519 passing tests. Live at erc8004-trading-agent.vercel.app."

**On-screen** (architecture diagram):
```
┌─────────────────────────────────────────────────────────────────┐
│                    ERC-8004 Trading Agent                        │
│                                                                   │
│  Claude AI          Oracle         Risk Manager                  │
│  Strategist  ──▶   Client   ──▶  (Credora BBB)  ──▶ APPROVED   │
│  (signal)          (price)        (Kelly 0.70)       │          │
│                                                       ▼          │
│  ERC-8004 ◀── Validation ◀── Surge Router ◀──── Execution      │
│  On-chain      Registry      (DeFi vaults)      Loop            │
│  Identity      (Sepolia)                                         │
└─────────────────────────────────────────────────────────────────┘
```

---

## Scene 6 — Live Price Feed + WebSocket Streaming (S45)

**Voice-over**:
> "Sprint 45 adds a real-time market data layer. The agent now streams
> simulated prices for BTC, ETH, SOL, and MATIC using Geometric Brownian
> Motion with mean reversion — the same model used by professional quant desks.
> Hit the REST endpoint to pull the current price and 24-hour stats..."

**Command**:
```bash
curl http://localhost:8084/api/v1/market/price/BTC-USD | python3 -m json.tool
# {"symbol": "BTC-USD", "price": 67892.14, "open": 67500.0,
#  "high": 68240.50, "low": 66980.10, "close": 67892.14,
#  "volume": 4823190127.0, "change_pct": 0.5812, "timestamp": 1740620400.0}
```

**Voice-over**:
> "WebSocket clients subscribe to live 1-second price ticks..."

```bash
# WebSocket subscription (ws://localhost:8084/api/v1/ws/prices)
# Messages: {"type": "price", "symbol": "BTC-USD", "price": 67901.22,
#            "timestamp": 1740620401.0, "change_pct": 0.5945}
```

**Voice-over**:
> "Agents auto-trade directly against the live feed using trend-follow,
> mean-revert, or hold strategies..."

```bash
curl -s -X POST http://localhost:8084/api/v1/agents/agent-trend/auto-trade \
  -H "Content-Type: application/json" \
  -d '{"strategy":"trend_follow","symbol":"BTC-USD","ticks":20}' | python3 -m json.tool
# {"agent_id": "agent-trend", "strategy": "trend_follow",
#  "trades_executed": 3, "pnl": 14.72, "final_price": 68112.50}
```

---

## Call to Action

**Live demo**: https://erc8004-trading-agent.vercel.app

**GitHub**: https://github.com/opspawn/erc8004-trading-agent

**Tests**: 5,915 passing (full coverage, S45 live price feed + WebSocket streaming)

**Standard**: ERC-8004 — Agent Financial Identity for Ethereum

---

*Demo script version: S45 | Date: 2026-02-27 | Sprint: 45*
