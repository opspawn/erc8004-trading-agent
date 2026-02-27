# lablab.ai ERC-8004 Hackathon — Submission Checklist

**Project**: ERC-8004 Autonomous Trading Agent
**Hackathon**: lablab.ai ERC-8004 Hackathon, March 9–22, 2026
**Prize pool**: $50,000 USDC
**Sprint**: S54 (demo video + video endpoint tests + 6,300+ total)

---

## Core Requirements

- [x] **On-chain identity** — Each agent holds an ERC-8004 (ERC-721 extension) identity token deployed on Base Sepolia. Contract: `IdentityRegistry.sol`. See `contracts/deployment.json`.
- [x] **Reputation system** — On-chain reputation registry (`ReputationRegistry.sol`) updates scores after each validated trade. Higher reputation = more voting weight in consensus.
- [x] **Validation mechanism** — `ValidationRegistry.sol` records trade outcome proofs. Every execution writes a signed artifact with `artifact_hash` and `win_rate`.
- [x] **Smart contract wallet** — `AgentWallet.sol` implements EIP-1271 signature validation for on-chain agent interactions.

## Technical Quality

- [x] **Test coverage** — 6,300+ passing tests across 54 sprints. Run: `cd agent && python3 -m pytest tests/ -q --tb=no`
- [x] **Portfolio risk engine** — VaR at 95%/99% (historical simulation), Sharpe/Sortino/Calmar, cross-symbol correlation matrix (S46)
- [x] **Position sizing** — Volatility-based, Half-Kelly, fixed-fraction; POST `/api/v1/risk/position-size` (S46)
- [x] **Exposure dashboard** — Per-symbol exposure + Herfindahl concentration index; GET `/api/v1/risk/exposure` (S46)
- [x] **10-agent swarm** — quant-1..quant-10 with momentum/mean-revert/arb/trend/contrarian/hybrid strategies (S46)
- [x] **Swarm vote** — POST `/api/v1/swarm/vote` — 10 agents vote, stake-weighted 2/3 supermajority consensus (S46)
- [x] **Swarm leaderboard** — GET `/api/v1/swarm/performance` — 24h PnL + Sharpe ranked for all 10 agents (S46)
- [x] **Live price feed** — GBM+mean-reversion simulated prices for BTC-USD, ETH-USD, SOL-USD, MATIC-USD (S45)
- [x] **WebSocket streaming** — WS `/api/v1/ws/prices` streams 1-second price ticks with subscribe/unsubscribe (S45)
- [x] **Agent auto-trading** — POST `/api/v1/agents/{id}/auto-trade` runs trend_follow/mean_revert/hold strategies on live feed (S45)
- [x] **Demo endpoint live** — HTTP server on port 8084 (proxied at `https://api.opspawn.com/erc8004/`). Health: `curl https://api.opspawn.com/erc8004/demo/health`
- [x] **No external dependencies for demo** — All demo endpoints run in-process with no wallet, no chain calls required.
- [x] **README scannable in 30 seconds** — Problem, solution, quickstart curl, architecture diagram, test count.
- [x] **Demo recorder** — `scripts/record_demo.py` captures all 5 demo steps; output in `docs/demo-screenshots/` (S49)
- [x] **Interactive HTML demo** — `docs/demo.html` with live fetch buttons for all 5 steps; `/demo/ui` serves it, `/demo/live-data` returns all 5 in one call (S52)
- [x] **render.yaml deploy config** — Configured for Render.com deployment with health check path (S49)
- [x] **Server updated to S50** — Demo HTML page + submission polish + 6,185+ tests (S50)
- [x] **Server updated to S52** — Live server at port 8084 reports version S52, 6210 tests (S52)
- [x] **Server updated to S53** — Live server at port 8084 reports version S53, 6240+ tests (S53)
- [x] **Server updated to S54** — Demo video (MP4) + video endpoint tests + 6,300+ tests (S54)
- [x] **Judge dashboard** — `GET /demo/judge` returns single-page HTML overview for judges with leaderboard, swarm vote, TA signals, risk summary, contract links, curl examples (S53)
- [x] **TA signals endpoint** — `GET /api/v1/signals/latest` returns RSI + MACD signals for BTC-USD, ETH-USD, SOL-USD using real price history (S53)
- [x] **Demo video recorded** — `docs/demo-video-s54.mp4` — 18-second walkthrough of judge dashboard + interactive demo UI (S54)

## Differentiators

- [x] **Reputation-weighted consensus** — 10 agents (quant-1..quant-10) vote; 2/3 supermajority required; stake-weighted voting. Not just majority vote.
- [x] **x402 micropayment gate** — `/demo/run` is payment-gated via x402 protocol. `dev_mode=true` for judges (free). Demonstrates real on-chain payment integration.
- [x] **Credora credit ratings** — On-chain credit scores feed into position sizing. Agents with higher credit get larger position limits.
- [x] **Backtester** — Historical GBM simulation with Sharpe, Sortino, max drawdown, win rate.
- [x] **Stress tester** — 7 adversarial scenarios: flash crash, oracle failure, consensus deadlock, liquidity crisis, regulatory halt, black swan, network partition.

## Submission Assets

- [x] **README.md** — Judge-friendly, scannable in 30 seconds. ✓
- [x] **JUDGE_DEMO.md** — Step-by-step 10-minute walkthrough for judges.
- [x] **SUBMISSION.md** — Technical narrative and architecture deep-dive.
- [x] **DEMO_SCRIPT.md** — Video script for demo recording.
- [x] **Video script ready** — See `DEMO_SCRIPT.md` for 3-minute demo outline.
- [x] **Interactive HTML demo** — `docs/demo.html` with live fetch buttons for all 5 demo steps (S52)
- [x] **Demo screenshots captured** — All 5 steps in `docs/demo-screenshots/` (S49)
- [x] **Demo video created** — `docs/demo-video-s54.mp4` — slideshow of judge dashboard + demo UI (S54)

## Live Endpoints

```bash
# Health check
curl https://api.opspawn.com/erc8004/demo/health
# Expected: {"status":"ok","tests":6300,"version":"S54",...}

# Judge dashboard (HTML)
curl http://localhost:8084/demo/judge

# TA signals (RSI + MACD)
curl http://localhost:8084/api/v1/signals/latest | python3 -m json.tool

# Full demo pipeline
curl -s -X POST 'https://api.opspawn.com/erc8004/demo/run?ticks=10' | python3 -m json.tool

# 10-agent swarm vote
curl -s -X POST 'https://api.opspawn.com/erc8004/api/v1/swarm/vote' \
  -H 'Content-Type: application/json' \
  -d '{"symbol":"BTC-USD","signal_type":"BUY"}' | python3 -m json.tool

# Portfolio risk (VaR)
curl https://api.opspawn.com/erc8004/api/v1/risk/portfolio

# Performance summary
curl https://api.opspawn.com/erc8004/api/v1/performance/summary

# Full showcase (all 4 steps in sequence)
curl -s -X POST 'https://api.opspawn.com/erc8004/api/v1/demo/showcase' | python3 -m json.tool
```

## Pre-Submission Final Check (March 22)

- [ ] Demo server is running and `/health` returns 200
- [ ] All 6,300+ tests pass (`python3 -m pytest tests/ -q --tb=no`)
- [ ] Contract addresses in `contracts/deployment.json` are correct
- [ ] README links are valid
- [ ] `docs/demo.html` opens in browser and 'Run All' button fetches live data for all 5 steps
- [ ] Video demo is recorded and uploaded
- [ ] lablab.ai submission form filled out with GitHub URL + demo URL
