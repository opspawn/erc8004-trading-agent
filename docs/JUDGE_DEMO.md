# Judge Demo Walkthrough

**Time required**: ~10 minutes
**Prerequisites**: `curl` and `python3` (for pretty-printing JSON). No wallet needed.

---

## Step 1 — Health Check (30 seconds)

Verify the server is live and see the current test count.

```bash
curl https://api.opspawn.com/erc8004/demo/health
```

**Expected response:**
```json
{
  "status": "ok",
  "service": "ERC-8004 Demo Server",
  "version": "S40",
  "tests": 4968,
  "uptime_s": 12345.6,
  "dev_mode": true
}
```

Key fields: `status: "ok"`, `tests: 4968`, `version: "S40"`.

**Or run locally:**
```bash
cd agent && python3 demo_server.py &
curl http://localhost:8084/health
```

---

## Step 2 — Run the Multi-Agent Demo Pipeline (2 minutes)

This is the core demo. Three agents with ERC-8004 identities vote on each trade tick using reputation-weighted consensus. The pipeline runs end-to-end in under 1 second.

```bash
curl -s -X POST 'https://api.opspawn.com/erc8004/demo/run?ticks=10' \
  | python3 -m json.tool
```

**What to look for in the response:**

```json
{
  "demo": {
    "ticks_run": 10,
    "trades_executed": 7,
    "avg_reputation_score": 7.24,   ← reputation-weighted consensus score
    "consensus_rate": 0.85           ← fraction of ticks reaching 2/3 agreement
  },
  "agents": [
    {
      "agent_id": "agent-conservative-001",
      "strategy": "conservative",
      "pnl_usd": 12.4,
      "reputation_delta": 0.1,       ← on-chain reputation updated after run
      "win_rate": 0.75,
      "erc8004_token_id": 1          ← on-chain identity token
    }
  ],
  "validation_artifact": {
    "artifact_hash": "0xabc123...",  ← signed proof of session performance
    "win_rate": 0.72,
    "signed_at": 1740614400
  },
  "x402": {
    "dev_mode": true,                ← payment gate bypassed for judges
    "price_usdc": "1000"             ← $0.001 USDC per call in production
  }
}
```

**The ERC-8004 story in the response:**
- Each agent has an `erc8004_token_id` — their permanent on-chain identity
- `reputation_delta` shows their score updating after each trade
- `validation_artifact.artifact_hash` is the signed proof stored on-chain

---

## Step 3 — Portfolio Snapshot (1 minute)

See the current portfolio state including risk metrics and Credora credit ratings.

```bash
curl -s https://api.opspawn.com/erc8004/demo/portfolio/snapshot \
  | python3 -m json.tool
```

**What to look for:**
```json
{
  "agents": [
    {
      "agent_id": "agent-conservative-001",
      "total_value_usd": 10847.20,
      "pnl_usd": 847.20,
      "credora_rating": "A+",        ← Credora credit score
      "max_drawdown": -0.042,
      "sharpe_ratio": 1.38
    }
  ],
  "portfolio_metrics": {
    "total_value_usd": 31200.50,
    "aggregate_sharpe": 1.24,
    "consensus_agreement_rate": 0.85
  }
}
```

---

## Step 4 — Strategy Comparison (1 minute)

Compare the three strategies side-by-side across risk-adjusted metrics.

```bash
curl -s https://api.opspawn.com/erc8004/demo/strategy/compare \
  | python3 -m json.tool
```

**What to look for:**
```json
{
  "strategies": [
    {"name": "conservative", "sharpe": 1.38, "sortino": 1.72, "win_rate": 0.75, "rank": 1},
    {"name": "balanced",     "sharpe": 1.21, "sortino": 1.48, "win_rate": 0.67, "rank": 2},
    {"name": "aggressive",   "sharpe": 0.94, "sortino": 1.21, "win_rate": 0.61, "rank": 3}
  ],
  "summary": {
    "best_sharpe": "conservative",
    "best_win_rate": "conservative",
    "recommended": "conservative"
  }
}
```

---

## Step 5 — Verify the On-Chain Contract (2 minutes)

The ERC-8004 contracts are deployed on Base Sepolia.

```bash
# View contract addresses
cat contracts/deployment.json
```

Navigate to [sepolia.basescan.org](https://sepolia.basescan.org) and search for the `IdentityRegistry` address. You'll see:
- ERC-721 token mints for each agent identity
- Reputation update transactions
- Validation proof records

---

## Step 6 — Run Tests Locally (2 minutes)

```bash
cd agent
pip install -r requirements.txt
python3 -m pytest tests/ -q --tb=no
# Expected: 4968 passed (or close — 1 flaky test in test_signal_server.py)
```

To run just the S40 tests:
```bash
python3 -m pytest tests/test_s40_health_judge.py -v
```

---

## Bonus: Agent Leaderboard

```bash
# Top agents by Sharpe ratio
curl 'https://api.opspawn.com/erc8004/demo/leaderboard?sort_by=sharpe&limit=5' \
  | python3 -m json.tool

# Top agents by PnL
curl 'https://api.opspawn.com/erc8004/demo/leaderboard?sort_by=pnl&limit=5' \
  | python3 -m json.tool
```

---

## S46: Portfolio Risk Management + 10-Agent Swarm

### Risk Management Endpoints (NEW in S46)

```bash
# Portfolio-level VaR, Sharpe/Sortino/Calmar, correlation matrix
curl https://api.opspawn.com/erc8004/api/v1/risk/portfolio | python3 -m json.tool

# Position sizing: volatility-based, Half-Kelly, or fixed-fraction
curl -s -X POST https://api.opspawn.com/erc8004/api/v1/risk/position-size \
  -H 'Content-Type: application/json' \
  -d '{"symbol":"BTC-USD","capital":100000,"risk_budget_pct":0.02,"method":"half_kelly"}' \
  | python3 -m json.tool

# Per-symbol exposure + Herfindahl concentration index
curl https://api.opspawn.com/erc8004/api/v1/risk/exposure | python3 -m json.tool
```

**What to look for in `/api/v1/risk/portfolio`:**
```json
{
  "portfolio": {
    "var_95": 0.012,          ← daily 95% Value at Risk
    "var_99": 0.021,          ← daily 99% VaR
    "sharpe_ratio": 1.42,
    "sortino_ratio": 2.14,
    "calmar_ratio": 3.5,
    "max_drawdown": 0.08
  },
  "correlation_matrix": { ... },  ← 4×4 cross-symbol correlation
  "per_symbol": { ... },
  "version": "S46"
}
```

### Multi-Agent Swarm Endpoints (NEW in S46)

The system now runs **10 agents** (quant-1 through quant-10) with diverse strategies:
momentum, mean-revert, arb, trend, contrarian, and hybrid.

```bash
# All 10 agents vote on a BUY signal — stake-weighted 2/3 supermajority
curl -s -X POST https://api.opspawn.com/erc8004/api/v1/swarm/vote \
  -H 'Content-Type: application/json' \
  -d '{"symbol":"BTC-USD","signal_type":"BUY"}' \
  | python3 -m json.tool

# 24h PnL + Sharpe leaderboard for all 10 agents
curl https://api.opspawn.com/erc8004/api/v1/swarm/performance | python3 -m json.tool
```

**What to look for in `/api/v1/swarm/vote`:**
```json
{
  "symbol": "BTC-USD",
  "signal_type": "BUY",
  "votes": [...],              ← 10 individual votes with confidence
  "vote_summary": {"BUY":7,"SELL":3},
  "weighted_agree_fraction": 0.72,
  "consensus_threshold": 0.667,
  "consensus_reached": true,
  "consensus_action": "BUY",
  "version": "S46"
}
```

---

## What Makes This Different

Most hackathon trading agent demos are just Python scripts with a `print()` statement.

**This project has:**
1. **Real on-chain identity** — each agent is an ERC-721 token with verifiable history
2. **Reputation as stake** — agents lose reputation for bad trades; bad agents get outvoted
3. **Payment gating** — the API costs real (micro) money in production; AI agents pay autonomously via x402
4. **6,085 tests** — not a toy; production-quality code
5. **Credora integration** — on-chain credit ratings feed into risk limits
6. **VaR risk engine** — 95%/99% historical-simulation VaR, Sharpe/Sortino/Calmar ratios
7. **10-agent swarm** — diverse strategy ensemble with stake-weighted consensus voting

---

## Troubleshooting

| Problem | Fix |
|---------|-----|
| `Connection refused` | Start server: `cd agent && python3 demo_server.py` |
| `404 Not found` | Check path — no trailing slash needed |
| Slow response | Add `?ticks=5` to reduce demo size |
| Import errors | Run `pip install -r requirements.txt` first |
