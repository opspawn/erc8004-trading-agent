# Judge Demo Walkthrough

**Time required**: ~10 minutes
**Prerequisites**: `curl` and `python3` (for pretty-printing JSON). No wallet needed.

---

## Step 1 — Health Check (30 seconds)

Verify the server is live and see the current test count.

```bash
curl http://localhost:8084/demo/health | python3 -m json.tool
```

**Expected response:**
```json
{
  "status": "ok",
  "service": "ERC-8004 Demo Server",
  "version": "S47",
  "sprint": "S47",
  "tests": 6121,
  "uptime_s": 12345.6,
  "dev_mode": true
}
```

Key fields: `status: "ok"`, `tests: 6121`, `version: "S47"`.

**Or run with public URL:**
```bash
curl https://api.opspawn.com/erc8004/demo/health | python3 -m json.tool
```

---

## Step 2 — 10-Agent Swarm Vote (1 minute)

The 10-agent swarm uses stake-weighted consensus to vote on a trade signal.

```bash
curl -s -X POST 'http://localhost:8084/api/v1/swarm/vote' \
  -H 'Content-Type: application/json' \
  -d '{"signal":"LONG","asset":"BTC-USD"}' \
  | python3 -m json.tool
```

**What to look for in the response:**

```json
{
  "symbol": "BTC-USD",
  "signal_type": "BUY",
  "votes": [...],
  "vote_summary": {"BUY": 7, "SELL": 3},
  "weighted_agree_fraction": 0.72,
  "consensus_threshold": 0.667,
  "consensus_reached": true,
  "consensus_action": "BUY",
  "version": "S46"
}
```

**The ERC-8004 story:**
- 10 agents with diverse strategies (momentum, mean-revert, arb, trend, contrarian, hybrid)
- Stake-weighted: high-reputation agents carry more weight
- 2/3 supermajority required for consensus

---

## Step 3 — Full Pipeline Showcase (1 minute)

Single call runs the entire pipeline: price tick → swarm vote → VaR → paper trade.

```bash
curl -s -X POST http://localhost:8084/api/v1/demo/showcase | python3 -m json.tool
```

**What to look for:**
```json
{
  "showcase": "ERC-8004 Full Pipeline",
  "version": "S47",
  "total_duration_ms": 12.4,
  "steps": [
    {"step": 1, "label": "Live Price Tick — BTC-USD", ...},
    {"step": 2, "label": "10-Agent Swarm Vote — LONG BTC-USD", ...},
    {"step": 3, "label": "Risk Engine — VaR 95/99% + Kelly Position Size", ...},
    {"step": 4, "label": "Paper Trade Execution — Consensus Decision", ...}
  ],
  "summary": {
    "btc_price": 68247.50,
    "swarm_consensus": "BUY",
    "consensus_agents": "7/10",
    "var_95": 0.012,
    "position_usd": 2000.0,
    "trade_executed": "s47-showcase-..."
  }
}
```

---

## Step 4 — Risk Portfolio (1 minute)

See the portfolio-level risk metrics: VaR, Sharpe/Sortino/Calmar, correlation.

```bash
curl -s http://localhost:8084/api/v1/risk/portfolio | python3 -m json.tool
```

**What to look for:**
```json
{
  "portfolio": {
    "var_95": 0.012,
    "var_99": 0.021,
    "sharpe_ratio": 1.42,
    "sortino_ratio": 2.14,
    "calmar_ratio": 3.5,
    "max_drawdown": 0.08
  },
  "correlation_matrix": { ... },
  "per_symbol": { ... },
  "version": "S46"
}
```

---

## Step 5 — Performance Summary (NEW in S48, 30 seconds)

Aggregate paper trading performance metrics across all sessions.

```bash
curl -s http://localhost:8084/api/v1/performance/summary | python3 -m json.tool
```

**What to look for:**
```json
{
  "total_paper_trades": 12,
  "total_pnl": 847.20,
  "win_rate": 66.7,
  "avg_trade_pnl": 70.6,
  "best_trade": 312.50,
  "worst_trade": -95.20,
  "sharpe_ratio": 1.38,
  "drawdown_pct": 4.2,
  "active_agents": 10,
  "version": "S48"
}
```

---

## Step 6 — Run Tests Locally (2 minutes)

```bash
cd agent
pip install -r requirements.txt
python3 -m pytest tests/ -q --tb=no
# Expected: 6121 passed
```

To run just the S48 tests:
```bash
python3 -m pytest tests/test_s48_performance.py -v
```

---

## What Makes This Different

Most hackathon trading agent demos are just Python scripts with a `print()` statement.

**This project has:**
1. **Real on-chain identity** — each agent is an ERC-721 token with verifiable history
2. **Reputation as stake** — agents lose reputation for bad trades; bad agents get outvoted
3. **Payment gating** — the API costs real (micro) money in production; AI agents pay autonomously via x402
4. **6,121 tests** — not a toy; production-quality code
5. **Credora integration** — on-chain credit ratings feed into risk limits
6. **VaR risk engine** — 95%/99% historical-simulation VaR, Sharpe/Sortino/Calmar ratios
7. **10-agent swarm** — diverse strategy ensemble with stake-weighted consensus voting
8. **Performance dashboard** — win rate, Sharpe ratio, max drawdown across all sessions

---

## Troubleshooting

| Problem | Fix |
|---------|-----|
| `Connection refused` | Start server: `cd agent && python3 demo_server.py` |
| `404 Not found` | Check path — no trailing slash needed |
| Slow response | Add `?ticks=5` to reduce demo size |
| Import errors | Run `pip install -r requirements.txt` first |
