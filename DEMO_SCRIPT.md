# ERC-8004 Trading Agent — 2-Minute Demo Script

> **Target audience**: Hackathon judges / technical evaluators
> **Format**: Screen recording + voice-over
> **Runtime**: ~2 minutes

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

## Scene 2 — RedStone Price Data Flowing into Risk Check (0:20 – 0:45)

**Show**: Terminal output of `oracle_client.fetch_eth_price()` and on-chain oracle update

**Voice-over**:
> "Before every trade the agent pulls live ETH/USD and BTC/USD prices from
> the RedStone oracle gateway. The price — $3,247 — is validated on-chain
> by the `RedStoneRiskOracle` contract."

**On-screen**:
```
[OracleClient] ETH/USD = 3247.50
[OracleClient] BTC/USD = 67340.00
[RiskManager]  check_oracle_risk: PASS  amount=3250 oracle=3247.5 deviation=0.08%
```
- Etherscan: `setEthPrice()` TX — show price in 8-decimal fixed-point

---

## Scene 3 — Trade Executed Through RiskRouter (0:45 – 1:10)

**Show**: On-chain `RiskRouter.checkRisk()` call and `RiskCheckPassed` event

**Voice-over**:
> "The RiskRouter contract validates the trade on-chain before execution.
> It checks that the proposed amount is within 5% of the oracle price.
> The `RiskCheckPassed` event confirms the gate is open — trade proceeds."

**On-screen**:
- Etherscan event log: `RiskCheckPassed(agentId=1, amount=..., oraclePrice=..., deviationBps=8)`
- Terminal: strategist decision → risk pass → simulated DEX swap

---

## Scene 4 — Reputation Score Updated (1:10 – 1:35)

**Show**: `ReputationRegistry.giveFeedback()` TX and score before/after

**Voice-over**:
> "After every trade closes, the agent logs its outcome to the ERC-8004
> ReputationRegistry — an immutable on-chain track record.
> A win scores 90/100, a loss scores 60/100.
> Tags: `trading` / `defi-swap` let evaluators filter by category."

**On-screen**:
- Etherscan: `giveFeedback(agentId=1, score=90, decimals=2, tag1='trading', tag2='defi-swap')`
- Terminal:
```
[Reputation] give_feedback: outcome=WIN value=90 tag1=trading tag2=defi-swap
[Reputation] Aggregate score: 8.5 / 10  (3 trades)
```

---

## Scene 5 — Dashboard Showing Live Metrics (1:35 – 2:00)

**Show**: Web dashboard at `http://localhost:8080`

**Voice-over**:
> "The dashboard shows real-time agent state: portfolio value, open positions,
> daily P&L, oracle prices, and reputation score — all sourced from live
> contract reads and the execution loop."

**On-screen**:
- Dashboard panels:
  - **Identity**: Agent #1 · DID: `eip155:84532:...`
  - **Oracle**: ETH $3,247 · BTC $67,340
  - **Risk**: deviation 0.08% ✅ · 0 halts
  - **Trades**: 3 executed · 1 rejected · P&L +$2.31
  - **Reputation**: 8.50 / 10 · 3 feedbacks · 100% win rate

---

## Tips for Recording

- Use `scripts/deploy_sepolia.sh` to deploy fresh contracts before recording
- Set `DRY_RUN=false` in `.env` to submit real on-chain TXs
- Zoom into Etherscan event logs — they're the proof-of-work
- Keep terminal font large (18pt+) so judges can read output
- Total contract test coverage: **87 passing** | Python tests: **370+ passing**

---

*ERC-8004 Trading Agent · Sprint 5 · lablab.ai Hackathon · Mar 2026*
