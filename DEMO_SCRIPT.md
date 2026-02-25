# ERC-8004 Trading Agent — 3-Minute Demo Script (S7)

> **Target audience**: Hackathon judges / technical evaluators
> **Format**: Screen recording + voice-over
> **Runtime**: ~3 minutes
> **Sprint**: S7 — Multi-Agent Coordinator + Live WebSocket Dashboard

---

## Scene 1 — Identity Registration (0:00 – 0:20)

**Show**: Etherscan TX for `IdentityRegistry.mint()`

**Voice-over**:
> "The agent registers its on-chain identity by minting an ERC-8004 NFT.
> This gives it a verifiable DID: `eip155:84532:<contract>:1`.
> Every action it takes is permanently tied to this identity."

**On-screen**:
- Etherscan / BaseScan transaction for `mint()` — show token ID, owner address, IPFS metadata URI
- Terminal: `agentId = 1`

---

## Scene 2 — Multi-Agent Coordinator Spinning Up (0:20 – 0:50)

**Show**: Terminal showing AgentPool initialization with 3 risk profiles

**Voice-over**:
> "Unlike single-agent systems, ERC-8004 supports a multi-agent coordinator.
> We launch three Claude strategist instances with different risk profiles:
> conservative, moderate, and aggressive.
> Before any trade executes, all three must reach a consensus threshold —
> at least 60% of weighted votes must agree."

**On-screen**:
```
[AgentCoordinator] Building pool: 3 agents
  agent-con: RiskProfile.CONSERVATIVE  max_pos=5%  min_conf=0.75
  agent-mod: RiskProfile.MODERATE      max_pos=10% min_conf=0.65
  agent-agg: RiskProfile.AGGRESSIVE    max_pos=20% min_conf=0.55
[AgentCoordinator] Consensus threshold: 60%
[AgentCoordinator] Pool ready. Dispatching signals...
```

---

## Scene 3 — Ensemble Signal + Consensus Vote (0:50 – 1:15)

**Show**: Live WebSocket dashboard at `http://localhost:8001` — Ensemble Signal card

**Voice-over**:
> "The coordinator queries all three agents concurrently.
> Each votes with a confidence-weighted signal.
> Two of three vote BUY with 78% and 82% confidence.
> The ensemble signal: BUY at 78% consensus — above threshold.
> The trade proceeds."

**On-screen**:
- Dashboard panel: **Ensemble Signal** → `BUY (78% consensus)`
- Agent votes: `agent-con: HOLD · agent-mod: BUY · agent-agg: BUY`
- Live Events Feed scrolling with `ensemble_signal` events
```
[Coordinator] Signals collected from 3 agents
[Coordinator] Vote weights: buy=2.1  sell=0.0  hold=1.0
[Coordinator] BUY wins at 67.7% weight (>60% threshold)
[EnsembleSignal] action=buy  confidence=0.80  size_pct=0.075
```

---

## Scene 4 — RedStone Oracle + RiskRouter Gate (1:15 – 1:40)

**Show**: Terminal output of oracle check and on-chain RiskRouter call

**Voice-over**:
> "Before the DEX swap, the agent validates the price against the RedStone
> oracle — and the on-chain RiskRouter contract confirms the deviation is safe.
> Watch the price update flow live into the WebSocket dashboard."

**On-screen**:
- Dashboard: **Price Feed** updating in real time (ETH, BTC)
- Terminal:
```
[OracleClient] ETH/USD = 3247.50
[RiskManager]  check_oracle_risk: PASS  deviation=0.08%
[RiskRouter]   RiskCheckPassed event emitted on-chain
[DEXExecutor]  simulate_swap: ETH→USDC  in=0.1  out=324.67
```
- Etherscan: `RiskCheckPassed(agentId=1, deviationBps=8)`

---

## Scene 5 — Trade Execution + P&L + Reputation (1:40 – 2:10)

**Show**: Trade result in terminal + dashboard updating + Etherscan TX

**Voice-over**:
> "Trade executes. The UniswapV3Executor swaps on-chain.
> The portfolio manager — using Kelly Criterion sizing — records a +$2.30 gain.
> The reputation module logs the outcome immutably to the ERC-8004 registry.
> Watch the dashboard P&L card update in real time."

**On-screen**:
- Dashboard: **Total PnL** ticking upward, **Reputation Score** updating
- Terminal:
```
[DEXExecutor]  executeSwap: ETH→USDC  tx=0xabc123...
[Portfolio]    Kelly size: 7.5% ($750 USDC)
[Portfolio]    Trade closed: PnL +$2.30  win_rate=0.67
[Reputation]   giveFeedback: score=90  tag=trading,defi-swap
[Reputation]   Aggregate: 8.5/10  (3 trades)
```
- Live Events Feed: `trade_executed`, `portfolio_update`, `agent_signal`

---

## Scene 6 — Performance Tracker + Weight Rebalancing (2:10 – 2:40)

**Show**: Dashboard Agent Pool card + terminal weight update

**Voice-over**:
> "After 10 trades, the performance tracker dynamically reweights agents.
> The moderate agent — with a 72% win rate — earns higher vote weight.
> The conservative agent, which avoided several bad trades, also gains weight.
> This self-improving ensemble gets smarter over time."

**On-screen**:
- Dashboard: **Agent Pool** card showing per-agent win rates
```
[PerformanceTracker] Rebalancing after 10 trades
  agent-con: weight 1.00 → 1.15  (win_rate=0.60, sharpe=0.31)
  agent-mod: weight 1.00 → 1.22  (win_rate=0.72, sharpe=0.55)
  agent-agg: weight 1.00 → 0.85  (win_rate=0.44, sharpe=-0.12)
```

---

## Scene 7 — Max Drawdown Halt + Risk Alert (2:40 – 3:00)

**Show**: Risk alert in dashboard + emergency stop terminal output

**Voice-over**:
> "Safety first. If daily drawdown exceeds 10%, the execution loop halts immediately.
> A risk alert fires to the WebSocket dashboard. No manual intervention needed.
> The agent protects its capital automatically."

**On-screen**:
- Dashboard: **Risk Status** → `HALTED` (red), alert in Live Events Feed
```
[RiskManager]  EMERGENCY STOP: daily drawdown 11.2% >= 10.0% threshold
[Dashboard]    broadcast risk_alert: severity=critical
[ExecutionLoop] is_running = False
```

---

## Key Metrics (Sprint 7 Complete)

| Metric | Value |
|--------|-------|
| Python tests | **688 passing** |
| Solidity tests | **113 passing** |
| **Total tests** | **801 passing** |
| Smart contracts | IdentityRegistry, ReputationRegistry, RiskRouter, RedStoneRiskOracle, UniswapV3Executor |
| New in S7 | Multi-Agent Coordinator, WebSocket Dashboard, Performance Tracker |
| Test files | 14 test files across Python + Solidity |

---

## Tips for Recording

- Start `uvicorn agent/dashboard_server:app --port 8001` before recording
- Open browser to `http://localhost:8001` — WebSocket connects automatically
- Use `scripts/deploy_sepolia.sh` to deploy fresh contracts before recording
- Set `DRY_RUN=false` in `.env` to submit real on-chain TXs
- Zoom into Etherscan event logs — they're the proof-of-work
- Keep terminal font large (18pt+) so judges can read output
- Use `/events/price` and `/events/trade` HTTP endpoints to simulate live events during demo

---

## WebSocket Demo Flow (for live demo)

```bash
# Terminal 1: Start dashboard
uvicorn agent/dashboard_server:app --port 8001 --reload

# Terminal 2: Push live events
curl -X POST http://localhost:8001/events/price \
  -H "Content-Type: application/json" \
  -d '{"symbol":"ETH","price":3247.50,"change_pct":0.0235}'

curl -X POST http://localhost:8001/events/trade \
  -H "Content-Type: application/json" \
  -d '{"market_id":"eth-usd","side":"buy","size_usdc":750,"tx_hash":"0xabc123","pnl":2.30}'

curl -X POST http://localhost:8001/events/risk-alert \
  -H "Content-Type: application/json" \
  -d '{"alert_type":"drawdown","message":"Daily drawdown 11.2% exceeded threshold","severity":"critical"}'
```

---

*ERC-8004 Trading Agent · Sprint 7 · lablab.ai Hackathon · Mar 2026*
