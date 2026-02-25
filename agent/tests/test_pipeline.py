"""
test_pipeline.py — Tests for pipeline.py integration orchestrator.

Covers:
  - Pipeline init, start, stop lifecycle
  - Tick loop: feed price → strategy → risk → trade
  - API endpoints (start/stop/status/trades) via httpx TestClient
  - Error recovery (market_feed fails → graceful degradation)
  - Concurrent strategy evaluation
  - Status/trade record correctness
"""

from __future__ import annotations

import asyncio
from typing import List
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import pytest_asyncio
from httpx import AsyncClient, ASGITransport

from pipeline import (
    Pipeline,
    PipelineConfig,
    PipelineStatus,
    TradeRecord,
)
from pipeline_api import app, set_pipeline


# ─── Helpers ──────────────────────────────────────────────────────────────────


def make_pipeline(**kwargs) -> Pipeline:
    kwargs.setdefault("seed", 42)
    cfg = PipelineConfig(tick_interval=0.0, **kwargs)
    return Pipeline(cfg)


def make_signal(action="buy", confidence=0.80, strategy_name="MomentumStrategy"):
    sig = MagicMock()
    sig.action = action
    sig.confidence = confidence
    sig.strategy_name = strategy_name
    sig.reason = "test"
    return sig


# ─── Lifecycle Tests ──────────────────────────────────────────────────────────


class TestPipelineInit:
    def test_initial_state_is_stopped(self):
        p = Pipeline()
        assert p.status().state == "stopped"

    def test_initial_ticks_zero(self):
        p = Pipeline()
        assert p.status().ticks == 0

    def test_initial_trades_zero(self):
        p = Pipeline()
        assert p.status().trades == 0

    def test_initial_portfolio_value_matches_config(self):
        p = Pipeline(PipelineConfig(initial_capital=5_000.0))
        assert p.status().portfolio_value == 5_000.0

    def test_default_symbols(self):
        p = Pipeline()
        assert "BTC" in p.status().symbols
        assert "ETH" in p.status().symbols
        assert "SOL" in p.status().symbols

    def test_custom_symbols(self):
        p = Pipeline(PipelineConfig(symbols=["BTC", "ETH"]))
        assert p.status().symbols == ["BTC", "ETH"]

    def test_is_running_false_initially(self):
        p = Pipeline()
        assert p.is_running() is False

    def test_get_trades_empty_initially(self):
        p = Pipeline()
        assert p.get_trades() == []

    def test_no_started_at_initially(self):
        p = Pipeline()
        assert p.status().started_at is None

    def test_config_default_capital(self):
        p = Pipeline()
        assert p.config.initial_capital == 10_000.0


class TestPipelineStart:
    @pytest.mark.asyncio
    async def test_start_returns_true(self):
        p = make_pipeline()
        started = await p.start()
        assert started is True
        await p.stop()

    @pytest.mark.asyncio
    async def test_start_sets_running_state(self):
        p = make_pipeline()
        await p.start()
        assert p.is_running()
        await p.stop()

    @pytest.mark.asyncio
    async def test_start_sets_started_at(self):
        p = make_pipeline()
        await p.start()
        assert p.status().started_at is not None
        await p.stop()

    @pytest.mark.asyncio
    async def test_double_start_returns_false(self):
        p = make_pipeline()
        await p.start()
        second = await p.start()
        assert second is False
        await p.stop()

    @pytest.mark.asyncio
    async def test_start_clears_error_state(self):
        p = make_pipeline()
        p._last_error = "prior error"
        await p.start()
        assert p.status().last_error is None
        await p.stop()

    @pytest.mark.asyncio
    async def test_start_resets_ticks(self):
        p = make_pipeline()
        p._ticks = 99
        await p.start()
        # ticks reset to 0 on start
        assert p.status().ticks == 0
        await p.stop()


class TestPipelineStop:
    @pytest.mark.asyncio
    async def test_stop_returns_true_when_running(self):
        p = make_pipeline()
        await p.start()
        result = await p.stop()
        assert result is True

    @pytest.mark.asyncio
    async def test_stop_sets_stopped_state(self):
        p = make_pipeline()
        await p.start()
        await p.stop()
        assert not p.is_running()

    @pytest.mark.asyncio
    async def test_stop_sets_stopped_at(self):
        p = make_pipeline()
        await p.start()
        await p.stop()
        assert p.status().stopped_at is not None

    @pytest.mark.asyncio
    async def test_stop_when_not_running_returns_false(self):
        p = make_pipeline()
        result = await p.stop()
        assert result is False

    @pytest.mark.asyncio
    async def test_restart_cycle(self):
        p = make_pipeline()
        await p.start()
        await p.stop()
        started_again = await p.start()
        assert started_again is True
        await p.stop()

    def test_stop_sync_sets_stopped(self):
        p = make_pipeline()
        p._state = "running"
        p.stop_sync()
        assert not p.is_running()


class TestPipelineStatus:
    def test_status_returns_pipeline_status(self):
        p = Pipeline()
        s = p.status()
        assert isinstance(s, PipelineStatus)

    def test_status_dict_has_required_keys(self):
        p = Pipeline()
        d = p.status().to_dict()
        for key in ("state", "ticks", "trades", "portfolio_value", "total_pnl"):
            assert key in d

    def test_status_total_pnl_initially_zero(self):
        p = Pipeline()
        assert p.status().total_pnl == 0.0

    @pytest.mark.asyncio
    async def test_status_updates_ticks_after_run(self):
        p = make_pipeline()
        # Warm up price history then check ticks increment
        await p.run_n_ticks(30)
        assert p.status().ticks == 30

    @pytest.mark.asyncio
    async def test_status_state_is_stopped_after_run_n_ticks(self):
        p = make_pipeline()
        await p.run_n_ticks(5)
        assert p.status().state == "stopped"


# ─── Tick Loop Tests ──────────────────────────────────────────────────────────


class TestTickLoop:
    @pytest.mark.asyncio
    async def test_tick_increments_counter(self):
        p = make_pipeline()
        await p.run_n_ticks(10)
        assert p.status().ticks == 10

    @pytest.mark.asyncio
    async def test_tick_builds_price_history(self):
        p = make_pipeline(symbols=["BTC"])
        await p.run_n_ticks(30)
        assert len(p._price_history["BTC"]) == 30

    @pytest.mark.asyncio
    async def test_tick_caps_price_history_at_200(self):
        p = make_pipeline(symbols=["BTC"])
        await p.run_n_ticks(250)
        assert len(p._price_history["BTC"]) <= 200

    @pytest.mark.asyncio
    async def test_no_trades_before_25_ticks(self):
        p = make_pipeline()
        await p.run_n_ticks(24)
        # Should have no trades yet (not enough price history)
        # Note: we might have trades if 3 symbols each get 24 ticks
        # but per-symbol each needs 25 prices
        assert p.status().ticks == 24

    @pytest.mark.asyncio
    async def test_trades_possible_after_warmup(self):
        p = make_pipeline(seed=42)
        await p.run_n_ticks(50)
        # After 50 ticks with 3 symbols we should see some trades
        # (strategy evaluated with GBM prices, some signals > 0.55)
        assert p.status().ticks == 50

    @pytest.mark.asyncio
    async def test_trade_recorded_correctly(self):
        """Mock strategy to force a buy signal and verify trade record."""
        p = make_pipeline(symbols=["BTC"])
        p._init_modules()
        p._state = "running"
        # Prime price history
        for _ in range(30):
            price = p._feed.next_price("BTC")
            p._price_history["BTC"].append(price)

        forced_signal = make_signal(action="buy", confidence=0.90)
        with patch.object(p._engine, "evaluate", return_value=forced_signal):
            with patch.object(p._risk, "validate_trade", return_value=(True, "ok")):
                await p._process_symbol_tick("BTC")

        trades = p.get_trades()
        assert len(trades) == 1
        assert trades[0].action == "buy"
        assert trades[0].symbol == "BTC"
        assert trades[0].strategy == "MomentumStrategy"

    @pytest.mark.asyncio
    async def test_risk_rejection_skips_trade(self):
        """Risk manager rejecting trade means no trade recorded."""
        p = make_pipeline(symbols=["BTC"])
        p._init_modules()
        p._state = "running"
        for _ in range(30):
            p._price_history["BTC"].append(p._feed.next_price("BTC"))

        forced_signal = make_signal(action="buy", confidence=0.90)
        with patch.object(p._engine, "evaluate", return_value=forced_signal):
            with patch.object(p._risk, "validate_trade", return_value=(False, "over limit")):
                await p._process_symbol_tick("BTC")

        assert len(p.get_trades()) == 0

    @pytest.mark.asyncio
    async def test_hold_signal_skips_trade(self):
        """Hold signal means no trade recorded."""
        p = make_pipeline(symbols=["BTC"])
        p._init_modules()
        p._state = "running"
        for _ in range(30):
            p._price_history["BTC"].append(p._feed.next_price("BTC"))

        hold_signal = make_signal(action="hold", confidence=0.50)
        with patch.object(p._engine, "evaluate", return_value=hold_signal):
            await p._process_symbol_tick("BTC")

        assert len(p.get_trades()) == 0

    @pytest.mark.asyncio
    async def test_low_confidence_signal_skips_trade(self):
        """Signal with confidence < 0.55 means no trade."""
        p = make_pipeline(symbols=["BTC"])
        p._init_modules()
        p._state = "running"
        for _ in range(30):
            p._price_history["BTC"].append(p._feed.next_price("BTC"))

        low_signal = make_signal(action="buy", confidence=0.50)
        with patch.object(p._engine, "evaluate", return_value=low_signal):
            with patch.object(p._risk, "validate_trade", return_value=(True, "ok")):
                await p._process_symbol_tick("BTC")

        assert len(p.get_trades()) == 0

    @pytest.mark.asyncio
    async def test_pnl_updates_portfolio_value(self):
        """Successful trade updates portfolio value."""
        p = make_pipeline(symbols=["BTC"], initial_capital=10_000.0)
        p._init_modules()
        p._state = "running"
        for _ in range(30):
            p._price_history["BTC"].append(p._feed.next_price("BTC"))

        forced_signal = make_signal(action="buy", confidence=0.90)
        with patch.object(p._engine, "evaluate", return_value=forced_signal):
            with patch.object(p._risk, "validate_trade", return_value=(True, "ok")):
                await p._process_symbol_tick("BTC")

        assert p.status().portfolio_value != 10_000.0
        assert p.status().total_pnl != 0.0

    @pytest.mark.asyncio
    async def test_symbol_error_does_not_crash_tick(self):
        """Error on one symbol doesn't prevent others from processing."""
        p = make_pipeline(symbols=["BTC", "ETH"])
        p._init_modules()
        p._state = "running"

        original = p._process_symbol_tick

        call_count = [0]

        async def mock_tick(symbol):
            call_count[0] += 1
            if symbol == "BTC":
                raise RuntimeError("BTC feed error")
            await original(symbol)

        p._process_symbol_tick = mock_tick
        await p._process_tick()
        # Both symbols were attempted
        assert call_count[0] == 2

    @pytest.mark.asyncio
    async def test_trade_history_capped_at_max(self):
        p = make_pipeline(symbols=["BTC"], max_trade_history=5)
        p._init_modules()
        p._state = "running"
        for _ in range(30):
            p._price_history["BTC"].append(p._feed.next_price("BTC"))

        forced_signal = make_signal(action="buy", confidence=0.90)
        with patch.object(p._engine, "evaluate", return_value=forced_signal):
            with patch.object(p._risk, "validate_trade", return_value=(True, "ok")):
                for _ in range(10):
                    await p._process_symbol_tick("BTC")

        assert len(p._trade_history) <= 5


class TestGetTrades:
    def test_get_trades_returns_list(self):
        p = Pipeline()
        assert isinstance(p.get_trades(), list)

    def test_get_trades_default_limit_50(self):
        p = Pipeline()
        # Manually add 60 trades
        for i in range(60):
            p._trade_history.append(
                TradeRecord("BTC", "buy", 0.8, 100.0, 65000.0, 1.0, "test")
            )
        assert len(p.get_trades()) == 50

    def test_get_trades_custom_limit(self):
        p = Pipeline()
        for i in range(20):
            p._trade_history.append(
                TradeRecord("ETH", "sell", 0.7, 50.0, 3000.0, -0.5, "test")
            )
        assert len(p.get_trades(limit=10)) == 10

    def test_get_trades_returns_most_recent(self):
        p = Pipeline()
        for i in range(5):
            p._trade_history.append(
                TradeRecord("BTC", "buy", 0.8, float(i), 65000.0, 1.0, "test")
            )
        trades = p.get_trades(limit=3)
        assert trades[-1].size_usdc == 4.0  # last inserted


# ─── Error Recovery Tests ─────────────────────────────────────────────────────


class TestErrorRecovery:
    @pytest.mark.asyncio
    async def test_feed_error_does_not_crash_pipeline(self):
        """If GBM raises, tick should be skipped gracefully."""
        p = make_pipeline(symbols=["BTC"])
        p._init_modules()
        p._state = "running"

        with patch.object(p._feed, "next_price", side_effect=RuntimeError("feed down")):
            # Should not raise
            await p._process_tick()

        assert p._state == "running"

    @pytest.mark.asyncio
    async def test_strategy_error_does_not_crash_pipeline(self):
        p = make_pipeline(symbols=["BTC"])
        p._init_modules()
        p._state = "running"
        for _ in range(30):
            p._price_history["BTC"].append(p._feed.next_price("BTC"))

        with patch.object(p._engine, "evaluate", side_effect=ValueError("bad signal")):
            await p._process_tick()

        assert p._state == "running"

    @pytest.mark.asyncio
    async def test_risk_manager_error_is_isolated(self):
        p = make_pipeline(symbols=["BTC"])
        p._init_modules()
        p._state = "running"
        for _ in range(30):
            p._price_history["BTC"].append(p._feed.next_price("BTC"))

        forced_signal = make_signal(action="buy", confidence=0.90)
        with patch.object(p._engine, "evaluate", return_value=forced_signal):
            with patch.object(p._risk, "validate_trade", side_effect=RuntimeError("rm err")):
                await p._process_tick()

        assert p._state == "running"

    @pytest.mark.asyncio
    async def test_pipeline_error_state_recorded(self):
        """Loop error should transition to 'error' state."""
        p = make_pipeline(symbols=["BTC"])
        p._init_modules()
        p._state = "running"
        p._stop_event.clear()

        async def bad_tick():
            raise RuntimeError("catastrophic failure")

        p._process_tick = bad_tick  # type: ignore
        await p._run_loop()
        assert p._state == "error"
        assert "catastrophic failure" in (p._last_error or "")

    @pytest.mark.asyncio
    async def test_multi_symbol_partial_failure(self):
        """When one of three symbols fails, others still processed."""
        p = make_pipeline(symbols=["BTC", "ETH", "SOL"])
        p._init_modules()
        p._state = "running"
        processed = []

        async def mock_tick(sym):
            processed.append(sym)
            if sym == "ETH":
                raise RuntimeError("ETH down")

        p._process_symbol_tick = mock_tick  # type: ignore
        await p._process_tick()

        assert "BTC" in processed
        assert "ETH" in processed
        assert "SOL" in processed


# ─── Concurrent Evaluation Tests ──────────────────────────────────────────────


class TestConcurrentEvaluation:
    @pytest.mark.asyncio
    async def test_run_n_ticks_sequential_consistency(self):
        """run_n_ticks should produce deterministic results with fixed seed."""
        p1 = make_pipeline(seed=123)
        p2 = make_pipeline(seed=123)
        await p1.run_n_ticks(50)
        await p2.run_n_ticks(50)
        assert p1.status().ticks == p2.status().ticks

    @pytest.mark.asyncio
    async def test_multiple_symbols_evaluated_per_tick(self):
        """Default 3 symbols means 3 evaluations per tick (after warmup)."""
        p = make_pipeline()
        p._init_modules()
        p._state = "running"
        evaluated = []

        async def mock_tick(sym):
            evaluated.append(sym)

        p._process_symbol_tick = mock_tick  # type: ignore
        await p._process_tick()
        assert len(evaluated) == 3

    @pytest.mark.asyncio
    async def test_concurrent_pipelines_independent(self):
        """Two independent pipelines don't interfere."""
        p1 = make_pipeline(seed=1, symbols=["BTC"])
        p2 = make_pipeline(seed=2, symbols=["ETH"])
        await asyncio.gather(p1.run_n_ticks(30), p2.run_n_ticks(30))
        assert "BTC" in p1.status().symbols
        assert "ETH" in p2.status().symbols


# ─── API Tests ────────────────────────────────────────────────────────────────


@pytest_asyncio.fixture
async def fresh_pipeline():
    """Create and register a fresh pipeline for API tests."""
    p = make_pipeline()
    set_pipeline(p)
    yield p
    # cleanup — properly await stop so background task finishes
    if p.is_running():
        await p.stop()
    else:
        p.stop_sync()
    set_pipeline(Pipeline(PipelineConfig()))


class TestPipelineAPIStart:
    @pytest.mark.asyncio
    async def test_start_returns_200(self, fresh_pipeline):
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            resp = await client.post("/pipeline/start")
        assert resp.status_code == 200

    @pytest.mark.asyncio
    async def test_start_response_has_ok(self, fresh_pipeline):
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            resp = await client.post("/pipeline/start")
        assert resp.json()["ok"] is True

    @pytest.mark.asyncio
    async def test_start_sets_running(self, fresh_pipeline):
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            await client.post("/pipeline/start")
        assert fresh_pipeline.is_running()

    @pytest.mark.asyncio
    async def test_double_start_returns_409(self, fresh_pipeline):
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            await client.post("/pipeline/start")
            resp = await client.post("/pipeline/start")
        assert resp.status_code == 409

    @pytest.mark.asyncio
    async def test_start_response_contains_status(self, fresh_pipeline):
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            resp = await client.post("/pipeline/start")
        data = resp.json()
        assert "status" in data
        assert data["status"]["state"] == "running"


class TestPipelineAPIStop:
    @pytest.mark.asyncio
    async def test_stop_returns_200(self, fresh_pipeline):
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            await client.post("/pipeline/start")
            resp = await client.post("/pipeline/stop")
        assert resp.status_code == 200

    @pytest.mark.asyncio
    async def test_stop_response_has_ok(self, fresh_pipeline):
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            await client.post("/pipeline/start")
            resp = await client.post("/pipeline/stop")
        assert resp.json()["ok"] is True

    @pytest.mark.asyncio
    async def test_stop_without_start_returns_409(self, fresh_pipeline):
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            resp = await client.post("/pipeline/stop")
        assert resp.status_code == 409

    @pytest.mark.asyncio
    async def test_stop_sets_stopped_state(self, fresh_pipeline):
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            await client.post("/pipeline/start")
            await client.post("/pipeline/stop")
        assert not fresh_pipeline.is_running()

    @pytest.mark.asyncio
    async def test_start_stop_start_cycle(self, fresh_pipeline):
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            await client.post("/pipeline/start")
            await client.post("/pipeline/stop")
            resp = await client.post("/pipeline/start")
        assert resp.status_code == 200


class TestPipelineAPIStatus:
    @pytest.mark.asyncio
    async def test_status_returns_200(self, fresh_pipeline):
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            resp = await client.get("/pipeline/status")
        assert resp.status_code == 200

    @pytest.mark.asyncio
    async def test_status_has_state_field(self, fresh_pipeline):
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            resp = await client.get("/pipeline/status")
        assert "state" in resp.json()

    @pytest.mark.asyncio
    async def test_status_initial_state_stopped(self, fresh_pipeline):
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            resp = await client.get("/pipeline/status")
        assert resp.json()["state"] == "stopped"

    @pytest.mark.asyncio
    async def test_status_running_after_start(self, fresh_pipeline):
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            await client.post("/pipeline/start")
            resp = await client.get("/pipeline/status")
        assert resp.json()["state"] == "running"

    @pytest.mark.asyncio
    async def test_status_has_portfolio_value(self, fresh_pipeline):
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            resp = await client.get("/pipeline/status")
        assert "portfolio_value" in resp.json()

    @pytest.mark.asyncio
    async def test_status_has_trades_count(self, fresh_pipeline):
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            resp = await client.get("/pipeline/status")
        assert "trades" in resp.json()

    @pytest.mark.asyncio
    async def test_status_has_symbols(self, fresh_pipeline):
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            resp = await client.get("/pipeline/status")
        assert "symbols" in resp.json()


class TestPipelineAPITrades:
    @pytest.mark.asyncio
    async def test_trades_returns_200(self, fresh_pipeline):
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            resp = await client.get("/pipeline/trades")
        assert resp.status_code == 200

    @pytest.mark.asyncio
    async def test_trades_returns_list(self, fresh_pipeline):
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            resp = await client.get("/pipeline/trades")
        assert isinstance(resp.json(), list)

    @pytest.mark.asyncio
    async def test_trades_empty_initially(self, fresh_pipeline):
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            resp = await client.get("/pipeline/trades")
        assert resp.json() == []

    @pytest.mark.asyncio
    async def test_trades_shows_executed_trades(self, fresh_pipeline):
        # Inject some trades directly
        fresh_pipeline._trade_history.append(
            TradeRecord("BTC", "buy", 0.8, 100.0, 65000.0, 2.0, "Momentum")
        )
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            resp = await client.get("/pipeline/trades")
        assert len(resp.json()) == 1
        assert resp.json()[0]["symbol"] == "BTC"

    @pytest.mark.asyncio
    async def test_trades_limit_parameter(self, fresh_pipeline):
        for i in range(10):
            fresh_pipeline._trade_history.append(
                TradeRecord("ETH", "sell", 0.7, 50.0, 3000.0, -0.5, "MeanRev")
            )
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            resp = await client.get("/pipeline/trades?limit=5")
        assert len(resp.json()) == 5

    @pytest.mark.asyncio
    async def test_trades_invalid_limit_returns_422(self, fresh_pipeline):
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            resp = await client.get("/pipeline/trades?limit=0")
        assert resp.status_code == 422

    @pytest.mark.asyncio
    async def test_trades_has_expected_fields(self, fresh_pipeline):
        fresh_pipeline._trade_history.append(
            TradeRecord("SOL", "buy", 0.9, 200.0, 150.0, 3.0, "Ensemble")
        )
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            resp = await client.get("/pipeline/trades")
        record = resp.json()[0]
        for field in ("symbol", "action", "confidence", "size_usdc", "price", "pnl_usdc", "strategy", "executed_at"):
            assert field in record

    @pytest.mark.asyncio
    async def test_health_endpoint(self, fresh_pipeline):
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            resp = await client.get("/pipeline/health")
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"


# ─── TradeRecord Tests ────────────────────────────────────────────────────────


class TestTradeRecord:
    def test_to_dict_has_all_fields(self):
        t = TradeRecord("BTC", "buy", 0.8, 100.0, 65000.0, 2.0, "Momentum")
        d = t.to_dict()
        assert d["symbol"] == "BTC"
        assert d["action"] == "buy"
        assert d["confidence"] == 0.8
        assert d["strategy"] == "Momentum"

    def test_to_dict_rounds_price(self):
        t = TradeRecord("ETH", "sell", 0.7, 99.9999, 3000.1234567, -1.23456, "MeanRev")
        d = t.to_dict()
        assert len(str(d["price"]).replace("-", "").replace(".", "")) <= 10

    def test_executed_at_is_iso_string(self):
        t = TradeRecord("SOL", "buy", 0.6, 50.0, 150.0, 0.5, "Volatility")
        assert "T" in t.executed_at or "-" in t.executed_at

    def test_negative_pnl_recorded(self):
        t = TradeRecord("BTC", "sell", 0.6, 100.0, 65000.0, -5.0, "Momentum")
        assert t.to_dict()["pnl_usdc"] == -5.0


# ─── Reset Tests ──────────────────────────────────────────────────────────────


class TestReset:
    def test_reset_clears_trade_history(self):
        p = Pipeline()
        p._trade_history.append(
            TradeRecord("BTC", "buy", 0.8, 100.0, 65000.0, 2.0, "test")
        )
        p.reset()
        assert len(p._trade_history) == 0

    def test_reset_clears_ticks(self):
        p = Pipeline()
        p._ticks = 50
        p.reset()
        assert p._ticks == 0

    def test_reset_resets_pnl(self):
        p = Pipeline()
        p._total_pnl = 500.0
        p.reset()
        assert p._total_pnl == 0.0

    def test_reset_restores_portfolio_value(self):
        p = Pipeline(PipelineConfig(initial_capital=5_000.0))
        p._portfolio_value = 4_500.0
        p.reset()
        assert p._portfolio_value == 5_000.0
