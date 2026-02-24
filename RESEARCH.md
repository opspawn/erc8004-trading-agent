# ERC-8004 Trading Agents Hackathon — Research & Build Guide

**Hackathon**: AI Trading Agents with ERC-8004 | lablab.ai
**Dates**: March 9–22, 2026 (13 days)
**Prize Pool**: $50,000 USDC total
**1st Place**: $10,000 + fast-track to Trading Capital Program + Surge evaluation
**Researched**: 2026-02-24

---

## 1. Surge Platform

### What Is Surge?

Surge (@Surgexyz_) describes itself as "the onchain home for AI agents." The CEO is **Pawel Czech** (@czech_pawel). Surge is building **ACM (Agent Capital Markets)** infrastructure — enabling any agent to become a full market participant with:
- Multi-chain wallet creation
- Autonomous DEX trading
- Token launches via Surge Launchpad
- Zero gas fees (account abstraction)

**⚠️ CRITICAL FINDING**: The specific "Hackathon Capital Sandbox vault" and "Risk Router" are **NOT publicly documented** as of February 2026. These are hackathon-specific infrastructure resources provided by Surge. The details will be distributed through:
1. lablab.ai Discord (hackathon channel)
2. Direct contact with Surge team
3. Hackathon kick-off materials on March 9

### What We Know About the Risk Router

From the hackathon page:
> "DEX execution via whitelisted Risk Router contract"

This means:
- Surge will provide a **pre-deployed smart contract** that acts as a controlled DEX router
- Your agent must route all trades through this contract (not directly to Uniswap/etc.)
- The Risk Router enforces position limits, allowed assets, and safety guardrails
- This is how Surge controls drawdown risk when agents trade with real/sandbox capital

### Surge URLs & Contact

| Resource | URL/Handle |
|----------|-----------|
| Twitter/X | https://x.com/Surgexyz_ |
| Website | https://surge.xyz |
| Lablab | https://lablab.ai/ai-hackathons/surge-moltbook-hackathon |
| CEO | @czech_pawel on X |

### Immediate Action Required

**Sean must**:
1. Join the lablab.ai Discord and find the `#erc-8004-trading-agents` channel
2. Register on lablab.ai for the hackathon (if not already done)
3. DM @czech_pawel or @Surgexyz_ on X to get early access to sandbox docs
4. Ask in Discord for: sandbox vault address, Risk Router contract address, testnet/mainnet, API keys needed

---

## 2. ERC-8004 Reference Implementations

### Official ERC-8004 Registry Contracts

**Source**: https://github.com/erc-8004/erc-8004-contracts

These contracts use the same address across ALL supported testnets (deterministic deploy):

| Registry | Address |
|----------|---------|
| Identity Registry | `0x8004A818BFB912233c491871b3d84c89A494BD9e` |
| Reputation Registry | `0x8004B663056A597Dffe9eCcC1965A193B7388713` |

**Supported testnets** (all use above addresses):
- Ethereum Sepolia (ChainID: 11155111)
- Base Sepolia (ChainID: 84532)
- Arbitrum Sepolia
- Optimism Testnet
- Polygon Amoy
- Linea Sepolia
- Scroll Testnet
- Avalanche Testnet
- Celo Testnet
- Abstract Testnet
- Mantle Testnet
- Monad Testnet
- MegaETH Testnet
- BSC Testnet

**Note**: Validation Registry address not listed in official repo as of Feb 2026.

### ChaosChain Reference Implementation (Ethereum Sepolia Only)

**Source**: https://github.com/ChaosChain/trustless-agents-erc-ri
**SDK**: https://docs.chaoscha.in/sdk/installation

| Registry | Address (Ethereum Sepolia) |
|----------|---------------------------|
| Identity Registry | `0xf66e7CBdAE1Cb710fee7732E4e1f173624e137A7` |
| Reputation Registry | `0x6E2a285294B5c74CB76d76AB77C1ef15c2A9E407` |
| Validation Registry | `0xC26171A3c4e1d958cEA196A5e84B7418C58DCA2C` ✅ |

This is the **most complete implementation** (has Validation Registry + 74/74 tests passing).
Use the **official `0x8004...` addresses for submission** (more widely recognized by judges).

### EIP Specification

**URL**: https://eips.ethereum.org/EIPS/eip-8004
**Status**: Draft (as of Aug 2025); mainnet launch reported Jan 29, 2026

### Key Registry Interfaces

#### Identity Registry (ERC-721)
```solidity
function register(string agentURI) external returns (uint256 agentId)
function setAgentWallet(uint256 agentId, address newWallet, uint256 deadline, bytes signature) external
function setMetadata(uint256 agentId, string metadataKey, bytes metadataValue) external
```

#### Reputation Registry
```solidity
// Give feedback (value = fixed-point int128 + uint8 decimals)
// Example: 87/100 score → value=87, decimals=0
// Example: -3.2% yield → value=-32, decimals=1
function giveFeedback(
    uint256 agentId, int128 value, uint8 valueDecimals,
    string tag1, string tag2, string endpoint,
    string feedbackURI, bytes32 feedbackHash
) external

function getSummary(uint256 agentId, address[] clientAddresses, string tag1, string tag2)
    external view returns (uint64 count, int128 summaryValue, uint8 decimals)
```

#### Validation Registry
```solidity
function validationRequest(address validatorAddress, uint256 agentId, string requestURI, bytes32 requestHash) external
function validationResponse(bytes32 requestHash, uint8 response, string responseURI, bytes32 responseHash, string tag) external
// response is 0-100 scale
```

### Available SDKs

| SDK | Language | Repo |
|-----|----------|------|
| ChaosChain SDK | TypeScript | https://github.com/ChaosChain/chaoschain |
| erc-8004-js | TypeScript | https://github.com/tetratorus/erc-8004-js |
| erc-8004-py | Python | https://github.com/tetratorus/erc-8004-py |
| 0xgasless agent-sdk | TypeScript + LangChain | https://github.com/0xgasless/agent-sdk |
| agent0-ts | TypeScript | https://github.com/agent0lab/agent0-ts |
| chaoschain-sdk | Python (PyPI) | pip install chaoschain-sdk |

**For our Python stack**: Use `erc-8004-py` or `web3.py` with raw ABI calls.

---

## 3. Winning Strategy Analysis

### Judging Criteria (from lablab.ai)

1. **Application of Technology** — Using ERC-8004 correctly and creatively
2. **Originality and Creativity** — Novel approach, not just a tutorial clone
3. **Business Value** — Real-world applicability
4. **Presentation** — Demo video quality, clarity

### Prize Targets by Category

| Prize | $$ | Key to Win |
|-------|----|-----------|
| Best Trustless Trading Agent | $10,000 | Complete ERC-8004 integration (identity + rep + validation) |
| Best Risk-Adjusted Return | $5,000 | Sharpe ratio, drawdown control, not just PnL |
| Best Validation & Trust Model | $2,500 | Creative validation artifacts, verifiable decisions |
| Best Yield/Portfolio Agent | $2,500 | Multi-asset, rebalancing, yield optimization |
| Best Compliance & Risk Guardrails | $2,500 | On-chain enforcement, guardrails that work |

**Recommendation**: Target **1st place + "Best Validation & Trust Model"** = $12,500 potential.

### Competitive Gap Analysis

**What most teams will build** (based on prior ERC-8004 hackathons):
- Simple buy/sell bot that registers identity on ERC-8004
- CrewAI or LangChain agent with basic Uniswap integration
- Reputation as an afterthought (one `giveFeedback` call at the end)
- No real validation artifacts
- Demo with fake/mocked trades

**Our differentiation opportunities**:

1. **Continuous Reputation Accumulation**: Most teams log one reputation event. We log every trade decision with structured feedback (tag1=`"trading"`, tag2=`"momentum"`, value=realized_pnl_bps).

2. **x402 + ERC-8004 Combo** (our unique edge): Use x402 micropayments to purchase market data signals. The agent autonomously pays for data → executes → logs reputation. This is the only truly autonomous closed loop. References: https://hackernoon.com/not-a-lucid-web3-dream-anymore-x402-erc-8004-a2a-and-the-next-wave-of-ai-commerce

3. **Validation-as-Risk-Control**: Before executing a large trade, the agent requests validation from a second "risk validator" agent. The Risk Router only approves if validation score ≥ 70/100. This creates a trustless multi-agent trading system.

4. **On-Chain Audit Trail**: Every trade decision, data source, confidence score, and outcome is hash-logged to the Validation Registry. Judges can verify every decision on-chain.

5. **TEE Bonus (optional)**: Phala Network TEE agent for strategy confidentiality. See https://github.com/Phala-Network/erc-8004-tee-agent.

### Existing Reference Projects to Study

- **vistara-apps/erc-8004-example**: CrewAI + ERC-8004, market analysis (no actual trades)
- **Phala/erc-8004-tee-agent**: TEE-backed agent (bonus points territory)
- **0xgasless/agent-sdk**: x402 + ERC-8004 combined SDK (closest to what we want)

---

## 4. Technical Stack Recommendation

### Recommended Testnet: **Base Sepolia** (ChainID: 84532)

**Why Base Sepolia**:
- Active ecosystem, low latency, fast finality
- Uniswap V3 deployed (with maintained liquidity pools)
- ERC-8004 contracts at same `0x8004...` addresses
- Coinbase ecosystem → more faucet access
- Most hackathon judges are familiar with Base

### DEX for Mock Trades: Uniswap V3 on Base Sepolia

**Ethereum Sepolia SwapRouterV2**: `0x3bFA4769FB09eefC5a80d6E87c3B9C650f7Ae48E`
**Base mainnet V2 Router**: `0x4752ba5DBc23f44D87826276BF6Fd6b1C372aD24`
(Base Sepolia testnet addresses: check https://docs.base.org/base-chain/network-information/ecosystem-contracts)

**Important**: Surge's Risk Router may replace direct Uniswap calls. Wait for their sandbox docs. Design your agent to use an abstracted `execute_trade(token_in, token_out, amount)` interface so you can swap between Risk Router and Uniswap V3 easily.

### Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    OpSpawn Trading Agent                     │
├─────────────────────────────────────────────────────────────┤
│  Identity: ERC-8004 Identity Registry (0x8004A81...)        │
│  Reputation: Log every trade → Reputation Registry          │
│  Validation: Pre-trade risk check → Validation Registry     │
├─────────────────────────────────────────────────────────────┤
│  Strategy Layer (Python/Claude API)                         │
│  ├── Momentum detector                                      │
│  ├── Mean-reversion detector                                │
│  ├── Risk guardrail (max drawdown, position limits)         │
│  └── x402 data feed purchaser (market signals)             │
├─────────────────────────────────────────────────────────────┤
│  Execution Layer                                            │
│  ├── Surge Risk Router (primary, when available)            │
│  └── Uniswap V3 SwapRouter (fallback)                      │
├─────────────────────────────────────────────────────────────┤
│  Chain: Base Sepolia                                        │
└─────────────────────────────────────────────────────────────┘
```

### Python Tech Stack

```python
# Core dependencies
web3==6.x              # Blockchain interaction
eth-account==0.x       # EIP-712 signing
langchain==0.x         # Agent framework (OR use direct Claude API)
anthropic==0.x         # Claude for strategy decisions

# ERC-8004
# Option A: pip install chaoschain-sdk
# Option B: pip install erc-8004-py (tetratorus, PyPI)
# Option C: raw web3.py with ABI from github.com/erc-8004/erc-8004-contracts/abis/

# Data
ccxt==4.x              # CEX market data (no API key needed for public endpoints)
# OR x402-enabled price oracles (our secret weapon)
```

### Existing Asset Reuse

| Asset | How to Use |
|-------|-----------|
| `polymarket-bot/market_scanner.py` | Adapt as signal generator (price momentum detection) |
| `polymarket-bot/trade_executor.py` | Replace Polymarket API with Uniswap V3 / Risk Router |
| Polygon wallet `0x7483...` | Use for mainnet demo (post-hackathon), testnet for dev |
| x402 server knowledge | Integrate x402 for data feed payments (unique demo value) |

### Wallet Setup for Development

```bash
# Generate a testnet-only wallet for the trading agent
# Store private key in credentials/erc8004-agent.json
# Fund via Base Sepolia faucet: https://www.coinbase.com/faucets/base-ethereum-sepolia-faucet

# NEVER use main Polygon wallet private key in agent code
# Create separate hot wallet for agent autonomy
```

---

## 5. Five-Day Sprint Plan (March 9–13)

### Day 1 (March 9) — Foundation + Identity
**Goal**: Agent is registered on ERC-8004, can sign EIP-712 messages

- [ ] Set up Hardhat project (`erc8004-trading-agent/`)
- [ ] Create agent wallet (testnet hot wallet, fund from faucet)
- [ ] Call `IdentityRegistry.register(agentURI)` on Base Sepolia → get `agentId`
- [ ] Set agent wallet via `setAgentWallet()` with EIP-712 signature
- [ ] Store metadata: agent name, strategy description, version
- [ ] Join lablab.ai Discord → find Surge sandbox docs / Risk Router address
- [ ] Write `agent_identity.py` — manages on-chain identity

**Output**: Agent has verified ERC-8004 identity. Tx hash in README.

### Day 2 (March 10) — Trading Engine
**Goal**: Agent can execute mock trades through Surge Risk Router OR Uniswap V3

- [ ] Implement `strategy_engine.py`:
  - Momentum strategy (5m/1h EMA crossover)
  - Mean-reversion strategy (RSI < 30 / > 70)
  - Position sizing (max 20% of portfolio per trade)
- [ ] Implement `trade_executor.py`:
  - Surge Risk Router (primary, if sandbox available)
  - Uniswap V3 SwapRouter (fallback)
- [ ] Implement `risk_manager.py`:
  - Max daily drawdown: 5%
  - Max single trade size: $500 (sandbox)
  - Whitelist: ETH, USDC, WBTC only
- [ ] Dry-run test: 10 paper trades logged

**Output**: Agent executes trades, risk limits enforced.

### Day 3 (March 11) — Reputation + Validation Loop
**Goal**: Every trade creates an on-chain audit trail

- [ ] After each trade: call `ReputationRegistry.giveFeedback()`:
  - tag1: `"trading"`, tag2: strategy name
  - value: realized P&L in basis points (e.g., +42 bps = value=42, decimals=0)
  - feedbackURI: IPFS-pinned trade receipt JSON
- [ ] Build "Risk Validator" agent (second process):
  - Before large trades (>$200), agent requests validation
  - Validator scores the trade plan 0-100
  - Trade only executes if score ≥ 70
- [ ] Hook validation into trade execution: `validationRequest()` → wait → `validationResponse()` → execute
- [ ] Build `audit_logger.py`: generates IPFS-pinnable trade receipts

**Output**: 10+ on-chain reputation events. Live validation workflow.

### Day 4 (March 12) — Claude Intelligence Layer
**Goal**: Agent makes strategy decisions using Claude API

- [ ] Implement `claude_strategist.py`:
  - Fetches market data (CCXT or x402-paid oracle)
  - Prompts Claude: "Given current market conditions X, recommend position"
  - Claude outputs: direction, size, confidence, reasoning
  - Confidence < 60% → skip trade, log "abstained" to reputation
- [ ] Implement x402 data feed (if time allows):
  - Find x402-enabled price oracle (or stand up our own)
  - Agent pays <$0.01 per API call from its hot wallet
  - Demo: autonomous data purchasing → autonomous trading
- [ ] Add dashboard (simple HTML) showing:
  - Agent's ERC-8004 identity + reputation score
  - Portfolio PnL chart
  - Recent trade history with on-chain tx links

**Output**: Claude-driven trading with logged reasoning. Working dashboard.

### Day 5 (March 13) — Demo Polish + Submission Prep
**Goal**: Demo video recorded, README complete, GitHub ready

- [ ] Record 3-5 minute demo video:
  - Open: "This is an autonomous AI agent with an on-chain identity..."
  - Show: identity on Base Sepolia block explorer
  - Show: live trade execution → reputation update (in real-time)
  - Show: validation registry entries
  - Show: portfolio P&L dashboard
  - Close: Surge Risk Router integration (if available)
- [ ] Write project README with:
  - Architecture diagram
  - Contract addresses (ERC-8004 + agent identity tx hash)
  - How reputation accumulates
  - How validation prevents bad trades
  - How to run locally
- [ ] Deploy dashboard to Vercel (opspawn.com/trading-agent or new subdomain)
- [ ] Submit to lablab.ai

---

## 6. Critical Unknowns (Action Items Before March 9)

| Unknown | How to Resolve |
|---------|---------------|
| Surge sandbox vault address | Join lablab.ai Discord before March 9 |
| Risk Router contract address + ABI | Contact @czech_pawel or @Surgexyz_ on X |
| Hackathon capital amount (sandbox) | Ask in Discord |
| Whether Surge provides testnet USDC | Ask in Discord |
| Is Risk Router on testnet or mainnet? | Ask in Discord |
| Do we need API keys from Surge? | Ask in Discord |

---

## 7. Resources & Links

### Primary Sources
- Hackathon: https://lablab.ai/ai-hackathons/ai-trading-agents-erc-8004
- EIP-8004: https://eips.ethereum.org/EIPS/eip-8004
- Official Contracts: https://github.com/erc-8004/erc-8004-contracts
- ChaosChain RI: https://github.com/ChaosChain/trustless-agents-erc-ri
- Surge: https://x.com/Surgexyz_ | https://surge.xyz
- Awesome ERC-8004: https://github.com/sudeepb02/awesome-erc8004

### SDK Options
- Python: https://github.com/tetratorus/erc-8004-py
- TypeScript: https://github.com/agent0lab/agent0-ts
- x402 + ERC-8004: https://github.com/0xgasless/agent-sdk
- CrewAI Example: https://github.com/vistara-apps/erc-8004-example
- Phala TEE: https://github.com/Phala-Network/erc-8004-tee-agent

### DeFi Integration
- Uniswap V3 docs: https://docs.uniswap.org/contracts/v3/reference/deployments
- Base ecosystem contracts: https://docs.base.org/base-chain/network-information/ecosystem-contracts
- Uniswap on Sepolia: `0x3bFA4769FB09eefC5a80d6E87c3B9C650f7Ae48E`

### Community
- lablab.ai Discord: https://lablab.ai (find hackathon Discord link on event page)
- ERC-8004 Telegram: http://t.me/ERC8004
- ERC-8004 Builder Program: http://bit.ly/8004builderprogram

---

## Summary Assessment

**Feasibility**: HIGH. We have everything we need to win:
- Python trading bot (polymarket-bot) → adapt for crypto DEX
- x402 experience → unique differentiator that no other team has
- Autonomous agent identity (OpSpawn) → built-in demo narrative
- $100 USDC → enough for testnet gas (use faucets)
- Polygon wallet → testnet demo, then real trades for final demo if risk-appropriate

**Biggest Risk**: Surge sandbox not publicly available pre-hackathon. Mitigate by building with Uniswap V3 first and swapping in Risk Router on Day 2 when sandbox opens.

**Win Probability**: 35-50% for top 3 placement if execution is strong. The x402 + ERC-8004 combined demo is genuinely novel and judges haven't seen it before.
