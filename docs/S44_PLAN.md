# Sprint S44 — Live Market Data Paper Trading + One-Click Demo
**Target**: 5,519 → ~5,750 tests (+~230)
**Deadline**: March 5, 2026 (before hackathon opens March 9)
**Rationale**: Real CoinGecko prices wire into paper_trader.py; one-click demo lets judges run the full pipeline in < 5 min

## What's Already Done (Do NOT Rebuild)
- WebSocket real-time feed → S42 (`ws_server.py`)
- Agent-to-agent coordination, broadcast_signal, consensus voting → S43 (`mesh_coordinator.py`)
- Leaderboard endpoint → S42 (exists, will be wired into one-click demo)

## New Files to Create

### `agent/live_market_trader.py`
```python
class LiveMarketTrader:
    async def fetch_live_prices(self) -> dict        # BTC, ETH, SOL from CoinGecko (primary); GBM fallback
    async def run_trade_cycle(self, symbol: str) -> TradeCycleResult
    async def get_trade_history(self, limit: int = 20) -> list[dict]
    async def get_agent_performance(self) -> dict    # per-agent PnL, win rate, Sharpe
```

### `agent/one_click_demo.py`
```python
class OneClickDemo:
    async def run_bull_scenario(self) -> DemoResult       # BTC uptrend
    async def run_bear_scenario(self) -> DemoResult       # ETH flash crash
    async def run_consensus_deadlock(self) -> DemoResult  # 1-1-1 vote tie resolution
    async def get_demo_summary(self) -> dict              # shareable JSON for judges
```

### `agent/tests/test_s44_live_market_trading.py`  (~120 tests)
- CoinGecko price fetch (live + mock/timeout fallback)
- Trade cycle: price → signal → 3-agent vote → paper execute
- Trade ledger persistence
- Performance metrics (Sharpe, win rate, drawdown)
- Full pipeline integration

### `agent/tests/test_s44_one_click_demo.py`  (~80 tests)
- All 4 scenario types run end-to-end
- Demo summary JSON structure
- All new HTTP endpoint contracts
- Edge cases: timeout, empty history, deadlock resolution

### `agent/tests/test_s44_integration.py`  (~30 tests)
- CoinGecko price → on-chain reputation update (full pipeline)
- Demo server boots cleanly, all new endpoints respond
- WebSocket feed (S42) + live prices (S44) integrated

## Files to Modify

| File | Change |
|------|--------|
| `agent/market_feed.py` | Promote CoinGecko to primary; add retry + cache |
| `agent/demo_server.py` | Add 5 new endpoints (see below) |
| `agent/paper_trader.py` | Accept live price input (not just GBM output) |
| `agent/demo_runner.py` | Wire to `one_click_demo.py` |
| `README.md` | Add "One-Click Demo" section with curl examples |

## New API Endpoints

```
POST /demo/live-trade                → run one real trade cycle
GET  /demo/live-trade/history        → last 20 trades with real prices
GET  /demo/agent-performance         → per-agent Sharpe, win rate, drawdown
POST /demo/scenario/{name}           → preset: bull | bear | crash | deadlock
GET  /demo/scenario/{name}/result    → result of last scenario run
```

## Success Criteria
- [ ] `GET /demo/live-trade/history` returns real BTC/ETH prices (not GBM)
- [ ] `POST /demo/scenario/flash-crash` completes in < 30 seconds
- [ ] Per-agent PnL tracked individually (Momentum, Mean-Rev, Sentiment)
- [ ] On-chain reputation fires after each trade cycle (Base Sepolia)
- [ ] Full demo runs end-to-end in < 5 minutes (matches video limit)
- [ ] All 5,750+ tests pass
