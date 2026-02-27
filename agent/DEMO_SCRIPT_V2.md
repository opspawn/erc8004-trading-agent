# ERC-8004 Autonomous Trading Agent — Demo Video Script V2

**Duration**: 3 minutes
**Target audience**: lablab.ai hackathon judges
**Theme**: "AI agents that trade autonomously, verify their own decisions on-chain, and can be trusted"

---

## 0:00–0:30 — Hook: The Trust Problem

**[Screen: Dark background, bold text animation]**

**Voiceover / Title card:**
> "Autonomous AI agents are managing real money. But can you trust them?"

**[Screen: Flash news-style headlines — "AI bot loses $2M on bad trade", "Flash crash triggered by algorithm"]**

**Voiceover:**
> "Today, every autonomous trading agent is a black box. No accountability. No provable identity. No on-chain history of whether their past decisions were good or bad."

**[Screen: Transition to ERC-8004 logo]**

**Voiceover:**
> "ERC-8004 changes that. We built an autonomous trading agent that holds a verifiable on-chain identity, accumulates a reputation score through every trade outcome, and requires a 2/3 supermajority of trusted agents before any position is executed."

**[Screen: Dashboard URL animates in — https://erc8004-trading-agent.vercel.app]**

> "This is a live, working system. Let me show you."

---

## 0:30–1:30 — Live Demo Walkthrough

**[Screen: Browser opens to https://erc8004-trading-agent.vercel.app]**

**Voiceover:**
> "The dashboard shows our three ERC-8004 agents: Conservative, Balanced, and Aggressive. Each holds a unique on-chain identity — an ERC-721 token on Base Sepolia."

**[Screen: Click "Run Demo" button → POST /demo/run fires]**

**Voiceover:**
> "I'm triggering a live 10-tick trading run right now. Watch what happens."

**[Screen: Response JSON appears — agents, consensus, trades, PnL]**

**Voiceover:**
> "The conservative agent — with reputation score 7.82 — voted HOLD. The aggressive agent voted BUY. The balanced agent cast the deciding vote. Because reputation scores are weighted in the vote, the conservative agent's caution has the most influence."

**[Screen: Highlight consensus_rate field — "consensus_rate: 0.7143"]**

**Voiceover:**
> "71% consensus achieved across all ticks. No trade executes without a 2/3 supermajority — weighted by each agent's on-chain reputation."

**[Screen: GET /demo/leaderboard response — ranked agents]**

**Voiceover:**
> "The leaderboard shows cumulative performance. The conservative agent leads by Sortino ratio. Reputation rewards consistent, risk-adjusted performance — not just raw returns."

**[Screen: GET /demo/metrics → sharpe_ratio, sortino_ratio, max_drawdown]**

**Voiceover:**
> "Live aggregate metrics — 1.42 Sharpe, 1.87 Sortino, max drawdown under 4%. These update with every demo run."

---

## 1:30–2:00 — Technical Architecture

**[Screen: ASCII architecture diagram from SUBMISSION.md]**

**Voiceover:**
> "Here's how the system is built."

**[Screen: Highlight each layer as mentioned]**

**Voiceover:**
> "Market data flows in — real CoinGecko prices with a GBM simulation fallback. The strategy engine runs five strategies simultaneously: momentum, mean reversion, volatility breakout, sentiment, and ensemble."

**[Screen: Agent mesh diagram]**

**Voiceover:**
> "Three agents vote on each tick. Votes are weighted by their current ERC-8004 reputation score — stored on-chain at `0x8004B663` on Base Sepolia. If an agent makes bad trades, its reputation drops, and its future votes count less."

**[Screen: ERC-8004 contract on basescan]**

**Voiceover:**
> "Every trade outcome is validated by the on-chain ValidationRegistry, which produces a signed artifact — a cryptographic proof linking the session to the agent's identity."

**[Screen: x402 payment gate code]**

**Voiceover:**
> "The demo endpoint is protected by an x402 micropayment gate. For hackathon judges, dev_mode bypasses the payment. In production, every API call costs $0.001 USDC — direct agent-to-agent monetization."

---

## 2:00–2:30 — Results

**[Screen: Split view — pytest output + basescan transaction]**

**Voiceover:**
> "The numbers speak for themselves."

**[Screen: Animate in stats one by one]**

- **4,779+ tests passing** across 60+ test files
- **Base Sepolia contract** live at `0x8004B663056A597Dffe9eCcC1965A193B7388713`
- **Real x402 USDC settlements** on Base Sepolia
- **Sub-100ms end-to-end** pipeline execution
- **7 adversarial scenarios** tested — flash crash, oracle failure, consensus deadlock

**Voiceover:**
> "We tested seven adversarial scenarios: a flash crash wiping 40% of portfolio value in 5 ticks, oracle failures, consensus deadlock with 3-way tie, extreme volatility, reputation collapse, and zero-capital edge cases. All handled gracefully."

**[Screen: /demo/strategy/performance-attribution response]**

**Voiceover:**
> "Sprint 38 adds performance attribution — breaking down P&L by strategy type, time period, and risk bucket. Now judges can see exactly which strategies contributed alpha, which carried beta exposure, and how performance varies across low, medium, and high volatility environments."

---

## 2:30–3:00 — Call to Action

**[Screen: OpSpawn logo + ERC-8004 branding]**

**Voiceover:**
> "ERC-8004 is more than a trading agent. It's an infrastructure standard for autonomous agents that can be trusted."

**[Screen: Three value propositions animate in]**

1. **Identity** — every agent has a verifiable on-chain DID
2. **Reputation** — trust is earned through provable outcomes, not claimed
3. **Validation** — every decision produces a signed, auditable artifact

**[Screen: Live endpoint URL]**

**Voiceover:**
> "The demo is live right now. Hit `POST https://api.opspawn.com/erc8004/demo/run` and watch a real multi-agent consensus system execute trades and update on-chain reputation in under 100 milliseconds."

**[Screen: QR code to demo URL]**

**Voiceover:**
> "We're OpSpawn. We build agent infrastructure for the autonomous economy. ERC-8004 is where on-chain trust meets autonomous AI."

**[Screen: End card with links]**
- Demo: https://erc8004-trading-agent.vercel.app
- API: https://api.opspawn.com/erc8004/demo/health
- GitHub: github.com/opspawn
- lablab.ai submission: March 22, 2026

---

## Production Notes

- **Tone**: Confident, technical, demo-first. No hype without evidence.
- **Pacing**: 0:00–0:30 is fast (hook). 0:30–1:30 is deliberate (let the demo breathe).
- **Screen capture**: Record at 1920x1080, 30fps. Use browser zoom 110% for readability.
- **Captions**: Add auto-captions for judges watching without sound.
- **Background music**: Low ambient electronic, fade under voiceover.
- **B-roll**: Basescan transaction page, pytest output, architecture diagram.
