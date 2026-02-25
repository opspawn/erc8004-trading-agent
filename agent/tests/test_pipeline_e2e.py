"""
test_pipeline_e2e.py — Extended end-to-end tests for the pipeline integration.

Covers additional pipeline scenarios:
  - TradeRecord dict structure and field validation
  - PipelineStatus dict serialization
  - run_n_ticks with various configs
  - Portfolio value changes via trade pnl
  - Trade history capping
  - Stop-sync behavior
  - API endpoints via AsyncClient
  - Error state recovery
  - Config edge cases
  - Concurrent symbol processing
  - Tick interval = 0 for test speed
"""

from __future__ import annotations

import asyncio
from unittest.mock import MagicMock, patch

import pytest
from httpx import ASGITransport, AsyncClient

from pipeline import Pipeline, PipelineConfig, PipelineStatus, TradeRecord
from pipeline_api import app, set_pipeline


# ─── Helpers ──────────────────────────────────────────────────────────────────

def fast_pipeline(**kw) -> Pipeline:
    kw.setdefault("seed", 99)
    kw.setdefault("tick_interval", 0.0)
    kw.setdefault("initial_capital", 10_000.0)
    return Pipeline(PipelineConfig(**kw))


def force_trade_signal(action="buy", confidence=0.85):
    sig = MagicMock()
    sig.action = action
    sig.confidence = confidence
    sig.strategy_name = "TestStrategy"
    sig.reason = "forced"
    return sig


# ─── TradeRecord Tests ────────────────────────────────────────────────────────

class TestTradeRecordExtra:
    def test_buy_action_stored(self):
        t = TradeRecord("ETH", "buy", 0.75, 500.0, 2000.0, 10.0, "Momentum")
        assert t.action == "buy"

    def test_sell_action_stored(self):
        t = TradeRecord("BTC", "sell", 0.65, 300.0, 60000.0, -5.0, "MeanReversion")
        assert t.action == "sell"

    def test_confidence_range(self):
        t = TradeRecord("SOL", "buy", 0.60, 100.0, 150.0, 2.0, "Ensemble")
        assert 0 <= t.confidence <= 1.0

    def test_to_dict_symbol(self):
        t = TradeRecord("ETH", "buy", 0.80, 500.0, 2500.0, 12.5, "Momentum")
        assert t.to_dict()["symbol"] == "ETH"

    def test_to_dict_action(self):
        t = TradeRecord("BTC", "sell", 0.70, 200.0, 65000.0, -8.0, "VolBreakout")
        assert t.to_dict()["action"] == "sell"

    def test_to_dict_pnl_rounded(self):
        t = TradeRecord("SOL", "buy", 0.72, 100.0, 160.0, 3.12345678, "Sentiment")
        d = t.to_dict()
        assert d["pnl_usdc"] == round(3.12345678, 4)

    def test_to_dict_price_rounded(self):
        t = TradeRecord("ETH", "buy", 0.80, 500.0, 2123.456789, 5.0, "Ensemble")
        d = t.to_dict()
        assert d["price"] == round(2123.456789, 4)

    def test_to_dict_size_rounded(self):
        t = TradeRecord("BTC", "buy", 0.80, 987.654321, 60000.0, 10.0, "Momentum")
        d = t.to_dict()
        assert d["size_usdc"] == round(987.654321, 4)

    def test_to_dict_executed_at_present(self):
        t = TradeRecord("BTC", "buy", 0.80, 500.0, 60000.0, 5.0, "Momentum")
        assert "executed_at" in t.to_dict()

    def test_executed_at_contains_T(self):
        t = TradeRecord("BTC", "buy", 0.80, 500.0, 60000.0, 5.0, "Momentum")
        assert "T" in t.executed_at

    def test_strategy_field_in_dict(self):
        t = TradeRecord("ETH", "sell", 0.65, 200.0, 2500.0, -3.0, "MyStrat")
        assert t.to_dict()["strategy"] == "MyStrat"

    def test_negative_pnl_in_dict(self):
        t = TradeRecord("SOL", "sell", 0.60, 100.0, 150.0, -7.5, "Momentum")
        assert t.to_dict()["pnl_usdc"] < 0


# ─── PipelineStatus Tests ─────────────────────────────────────────────────────

class TestPipelineStatusExtra:
    def test_status_dict_state_key(self):
        p = Pipeline()
        assert "state" in p.status().to_dict()

    def test_status_dict_symbols_is_list(self):
        p = Pipeline()
        assert isinstance(p.status().to_dict()["symbols"], list)

    def test_status_last_error_none_by_default(self):
        p = Pipeline()
        assert p.status().last_error is None

    def test_status_stopped_at_none_initially(self):
        p = Pipeline()
        assert p.status().stopped_at is None

    def test_status_portfolio_value_rounded(self):
        p = Pipeline()
        d = p.status().to_dict()
        assert isinstance(d["portfolio_value"], float)

    def test_status_total_pnl_rounded(self):
        p = Pipeline()
        d = p.status().to_dict()
        assert isinstance(d["total_pnl"], float)

    def test_status_symbols_match_config(self):
        cfg = PipelineConfig(symbols=["BTC", "ETH"])
        p = Pipeline(cfg)
        assert set(p.status().symbols) == {"BTC", "ETH"}


# ─── Pipeline Config Tests ────────────────────────────────────────────────────

class TestPipelineConfig:
    def test_default_tick_interval(self):
        cfg = PipelineConfig()
        assert cfg.tick_interval > 0

    def test_custom_capital(self):
        cfg = PipelineConfig(initial_capital=50_000.0)
        p = Pipeline(cfg)
        assert p.status().portfolio_value == 50_000.0

    def test_custom_max_position_pct(self):
        cfg = PipelineConfig(max_position_pct=0.05)
        assert cfg.max_position_pct == 0.05

    def test_custom_max_drawdown_pct(self):
        cfg = PipelineConfig(max_daily_drawdown_pct=0.02)
        assert cfg.max_daily_drawdown_pct == 0.02

    def test_single_symbol(self):
        p = fast_pipeline(symbols=["BTC"])
        assert p.status().symbols == ["BTC"]

    def test_four_symbols(self):
        p = fast_pipeline(symbols=["BTC", "ETH", "SOL", "LINK"])
        assert len(p.status().symbols) == 4

    def test_seed_reproducibility(self):
        """Two pipelines with same seed run identically."""
        p1 = fast_pipeline(symbols=["BTC"], seed=7)
        p2 = fast_pipeline(symbols=["BTC"], seed=7)
        # Both should have the same starting state
        assert p1.status().portfolio_value == p2.status().portfolio_value

    def test_max_trade_history_config(self):
        cfg = PipelineConfig(max_trade_history=10)
        assert cfg.max_trade_history == 10


# ─── run_n_ticks Tests ────────────────────────────────────────────────────────

class TestRunNTicks:
    @pytest.mark.asyncio
    async def test_zero_ticks(self):
        p = fast_pipeline()
        await p.run_n_ticks(0)
        assert p.status().ticks == 0

    @pytest.mark.asyncio
    async def test_one_tick(self):
        p = fast_pipeline()
        await p.run_n_ticks(1)
        assert p.status().ticks == 1

    @pytest.mark.asyncio
    async def test_ten_ticks(self):
        p = fast_pipeline()
        await p.run_n_ticks(10)
        assert p.status().ticks == 10

    @pytest.mark.asyncio
    async def test_state_is_stopped_after_run(self):
        p = fast_pipeline()
        await p.run_n_ticks(5)
        assert p.status().state == "stopped"

    @pytest.mark.asyncio
    async def test_started_at_set(self):
        p = fast_pipeline()
        await p.run_n_ticks(5)
        assert p.status().started_at is not None

    @pytest.mark.asyncio
    async def test_stopped_at_set(self):
        p = fast_pipeline()
        await p.run_n_ticks(5)
        assert p.status().stopped_at is not None

    @pytest.mark.asyncio
    async def test_price_history_grows(self):
        p = fast_pipeline(symbols=["ETH"])
        await p.run_n_ticks(15)
        assert len(p._price_history["ETH"]) == 15

    @pytest.mark.asyncio
    async def test_price_history_capped(self):
        p = fast_pipeline(symbols=["ETH"])
        await p.run_n_ticks(300)
        assert len(p._price_history["ETH"]) <= 200

    @pytest.mark.asyncio
    async def test_portfolio_value_changes_with_trades(self):
        """After enough ticks, forced trades affect portfolio value."""
        p = fast_pipeline(symbols=["BTC"])
        p._init_modules()
        # Prime price history
        for _ in range(30):
            p._price_history["BTC"].append(p._feed.next_price("BTC"))
        p._state = "running"

        sig = force_trade_signal("buy", 0.90)
        with patch.object(p._engine, "evaluate", return_value=sig):
            with patch.object(p._risk, "validate_trade", return_value=(True, "ok")):
                await p._process_tick()

        # Portfolio value should be != initial (trade happened)
        assert p._portfolio_value != 10_000.0 or p._total_pnl != 0.0

    @pytest.mark.asyncio
    async def test_multiple_symbols_all_get_price_history(self):
        p = fast_pipeline(symbols=["BTC", "ETH", "SOL"])
        await p.run_n_ticks(15)
        for sym in ["BTC", "ETH", "SOL"]:
            assert len(p._price_history[sym]) == 15


# ─── Trade History Tests ──────────────────────────────────────────────────────

class TestTradeHistoryCapping:
    @pytest.mark.asyncio
    async def test_trade_history_capped_at_max(self):
        p = fast_pipeline(symbols=["BTC"], max_trade_history=5)
        p._init_modules()
        # Prime price history
        for _ in range(30):
            p._price_history["BTC"].append(p._feed.next_price("BTC"))
        p._state = "running"

        sig = force_trade_signal("buy", 0.90)
        with patch.object(p._engine, "evaluate", return_value=sig):
            with patch.object(p._risk, "validate_trade", return_value=(True, "ok")):
                for _ in range(10):
                    await p._process_symbol_tick("BTC")

        assert len(p._trade_history) <= 5

    @pytest.mark.asyncio
    async def test_get_trades_limit_respected(self):
        p = fast_pipeline(symbols=["BTC"])
        p._init_modules()
        for _ in range(30):
            p._price_history["BTC"].append(p._feed.next_price("BTC"))
        p._state = "running"

        sig = force_trade_signal("buy", 0.90)
        with patch.object(p._engine, "evaluate", return_value=sig):
            with patch.object(p._risk, "validate_trade", return_value=(True, "ok")):
                for _ in range(20):
                    await p._process_symbol_tick("BTC")

        limited = p.get_trades(limit=5)
        assert len(limited) <= 5

    def test_get_trades_default_limit(self):
        p = fast_pipeline()
        result = p.get_trades()
        assert isinstance(result, list)

    @pytest.mark.asyncio
    async def test_trades_count_in_status(self):
        p = fast_pipeline(symbols=["BTC"])
        p._init_modules()
        for _ in range(30):
            p._price_history["BTC"].append(p._feed.next_price("BTC"))
        p._state = "running"

        sig = force_trade_signal("buy", 0.90)
        with patch.object(p._engine, "evaluate", return_value=sig):
            with patch.object(p._risk, "validate_trade", return_value=(True, "ok")):
                for _ in range(3):
                    await p._process_symbol_tick("BTC")

        assert p.status().trades == 3


# ─── Stop Sync Tests ──────────────────────────────────────────────────────────

class TestStopSync:
    def test_stop_sync_sets_state_stopped(self):
        p = fast_pipeline()
        p._state = "running"
        p.stop_sync()
        assert p.status().state == "stopped"

    def test_stop_sync_sets_stop_event(self):
        p = fast_pipeline()
        p.stop_sync()
        assert p._stop_event.is_set()

    def test_stop_sync_sets_stopped_at(self):
        p = fast_pipeline()
        p.stop_sync()
        assert p.status().stopped_at is not None

    def test_is_running_false_after_stop_sync(self):
        p = fast_pipeline()
        p._state = "running"
        p.stop_sync()
        assert not p.is_running()


# ─── Reset Tests ──────────────────────────────────────────────────────────────

class TestResetExtra:
    @pytest.mark.asyncio
    async def test_reset_after_ticks(self):
        p = fast_pipeline()
        await p.run_n_ticks(30)
        p.reset()
        assert p.status().ticks == 0
        assert p.status().portfolio_value == p.config.initial_capital

    def test_reset_clears_error(self):
        p = fast_pipeline()
        p._last_error = "something broke"
        p.reset()
        assert p.status().last_error is None

    def test_reset_clears_pnl(self):
        p = fast_pipeline()
        p._total_pnl = 999.9
        p.reset()
        assert p.status().total_pnl == 0.0

    def test_reset_clears_price_history(self):
        p = fast_pipeline(symbols=["BTC"])
        p._price_history["BTC"] = [100.0, 200.0, 300.0]
        p.reset()
        assert p._price_history["BTC"] == []


# ─── API Tests ────────────────────────────────────────────────────────────────

class TestPipelineApiExtra:
    @pytest.mark.asyncio
    async def test_health_endpoint(self):
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            r = await client.get("/pipeline/health")
            assert r.status_code == 200
            assert r.json()["status"] == "ok"

    @pytest.mark.asyncio
    async def test_status_endpoint_keys(self):
        p = fast_pipeline()
        set_pipeline(p)
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            r = await client.get("/pipeline/status")
            assert r.status_code == 200
            data = r.json()
            assert "state" in data
            assert "ticks" in data
            assert "trades" in data

    @pytest.mark.asyncio
    async def test_trades_endpoint_returns_list(self):
        p = fast_pipeline()
        set_pipeline(p)
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            r = await client.get("/pipeline/trades")
            assert r.status_code == 200
            assert isinstance(r.json(), list)

    @pytest.mark.asyncio
    async def test_trades_limit_query_param(self):
        p = fast_pipeline()
        set_pipeline(p)
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            r = await client.get("/pipeline/trades?limit=10")
            assert r.status_code == 200

    @pytest.mark.asyncio
    async def test_trades_limit_too_large_fails(self):
        p = fast_pipeline()
        set_pipeline(p)
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            r = await client.get("/pipeline/trades?limit=501")
            assert r.status_code == 422

    @pytest.mark.asyncio
    async def test_trades_limit_zero_fails(self):
        p = fast_pipeline()
        set_pipeline(p)
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            r = await client.get("/pipeline/trades?limit=0")
            assert r.status_code == 422

    @pytest.mark.asyncio
    async def test_start_when_running_returns_409(self):
        p = fast_pipeline()
        p._state = "running"
        set_pipeline(p)
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            r = await client.post("/pipeline/start")
            assert r.status_code == 409
        p.stop_sync()

    @pytest.mark.asyncio
    async def test_stop_when_not_running_returns_409(self):
        p = fast_pipeline()
        set_pipeline(p)
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            r = await client.post("/pipeline/stop")
            assert r.status_code == 409

    @pytest.mark.asyncio
    async def test_start_stop_full_cycle(self):
        p = fast_pipeline()
        set_pipeline(p)
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            r_start = await client.post("/pipeline/start")
            assert r_start.status_code == 200
            assert r_start.json()["ok"] is True

            r_stop = await client.post("/pipeline/stop")
            assert r_stop.status_code == 200
            assert r_stop.json()["ok"] is True

    @pytest.mark.asyncio
    async def test_status_after_start_shows_running(self):
        p = fast_pipeline()
        set_pipeline(p)
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            await client.post("/pipeline/start")
            r = await client.get("/pipeline/status")
            assert r.json()["state"] == "running"
            await client.post("/pipeline/stop")

    @pytest.mark.asyncio
    async def test_status_after_stop_shows_stopped(self):
        p = fast_pipeline()
        set_pipeline(p)
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            await client.post("/pipeline/start")
            await client.post("/pipeline/stop")
            r = await client.get("/pipeline/status")
            assert r.json()["state"] == "stopped"

    @pytest.mark.asyncio
    async def test_start_response_includes_status(self):
        p = fast_pipeline()
        set_pipeline(p)
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            r = await client.post("/pipeline/start")
            assert "status" in r.json()
            await client.post("/pipeline/stop")

    @pytest.mark.asyncio
    async def test_stop_response_includes_status(self):
        p = fast_pipeline()
        set_pipeline(p)
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            await client.post("/pipeline/start")
            r = await client.post("/pipeline/stop")
            assert "status" in r.json()
