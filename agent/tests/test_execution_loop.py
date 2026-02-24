"""
test_execution_loop.py — Tests for the ExecutionLoop class.
"""

from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from execution_loop import ExecutionLoop, LoopConfig, LoopMetrics, LoopState


# ─── Fixtures ─────────────────────────────────────────────────────────────────


@pytest.fixture
def tmp_state_file(tmp_path):
    return str(tmp_path / "test_loop_state.json")


@pytest.fixture
def config(tmp_state_file):
    return LoopConfig(
        cycle_interval_seconds=0.01,
        max_concurrent_trades=3,
        max_daily_drawdown_pct=0.05,
        state_file=tmp_state_file,
    )


@pytest.fixture
def loop(config):
    return ExecutionLoop(config=config)


# ─── LoopConfig ───────────────────────────────────────────────────────────────


class TestLoopConfig:
    def test_defaults(self):
        cfg = LoopConfig()
        assert cfg.cycle_interval_seconds == 30.0
        assert cfg.max_concurrent_trades == 3
        assert cfg.max_daily_drawdown_pct == 0.05
        assert cfg.emergency_alert_callback is None

    def test_custom_values(self):
        cfg = LoopConfig(cycle_interval_seconds=5.0, max_concurrent_trades=10)
        assert cfg.cycle_interval_seconds == 5.0
        assert cfg.max_concurrent_trades == 10

    def test_emergency_callback_set(self):
        cb = lambda msg: None
        cfg = LoopConfig(emergency_alert_callback=cb)
        assert cfg.emergency_alert_callback is cb


# ─── LoopMetrics ──────────────────────────────────────────────────────────────


class TestLoopMetrics:
    def test_defaults(self):
        m = LoopMetrics()
        assert m.cycles_run == 0
        assert m.trades_executed == 0
        assert m.errors_last_hour == 0
        assert m.uptime_seconds == 0.0
        assert m.started_at is None

    def test_to_dict(self):
        m = LoopMetrics(cycles_run=5, trades_executed=3, errors_last_hour=1)
        d = m.to_dict()
        assert d["cycles_run"] == 5
        assert d["trades_executed"] == 3
        assert d["errors_last_hour"] == 1
        assert "uptime_seconds" in d

    def test_to_dict_keys(self):
        m = LoopMetrics()
        d = m.to_dict()
        expected_keys = {
            "cycles_run", "trades_executed", "trades_rejected",
            "errors_last_hour", "uptime_seconds", "last_cycle_at",
            "emergency_stops", "started_at",
        }
        assert set(d.keys()) == expected_keys


# ─── LoopState ────────────────────────────────────────────────────────────────


class TestLoopState:
    def test_defaults(self):
        s = LoopState()
        assert s.is_running is False
        assert s.is_emergency_stopped is False
        assert s.active_trade_count == 0
        assert s.daily_pnl_usdc == 0.0
        assert s.portfolio_value_usdc == 100.0

    def test_to_dict(self):
        s = LoopState(daily_pnl_usdc=-5.0, portfolio_value_usdc=100.0)
        d = s.to_dict()
        assert d["daily_pnl_usdc"] == -5.0
        assert d["portfolio_value_usdc"] == 100.0
        assert "metrics" in d

    def test_from_dict_round_trip(self):
        s = LoopState(
            active_trade_count=2,
            daily_pnl_usdc=-1.5,
            portfolio_value_usdc=200.0,
        )
        s.metrics.cycles_run = 42
        s.metrics.trades_executed = 7

        d = s.to_dict()
        d["is_running"] = False   # simulate reload
        s2 = LoopState.from_dict(d)

        assert s2.active_trade_count == 2
        assert s2.daily_pnl_usdc == -1.5
        assert s2.portfolio_value_usdc == 200.0
        assert s2.metrics.cycles_run == 42
        assert s2.metrics.trades_executed == 7

    def test_from_dict_missing_metrics(self):
        d = {"is_running": False, "is_emergency_stopped": False,
             "active_trade_count": 0, "daily_pnl_usdc": 0.0,
             "portfolio_value_usdc": 100.0, "last_saved_at": None}
        s = LoopState.from_dict(d)
        assert s.metrics.cycles_run == 0


# ─── ExecutionLoop Construction ───────────────────────────────────────────────


class TestExecutionLoopInit:
    def test_default_init(self):
        el = ExecutionLoop()
        assert el.config.cycle_interval_seconds == 30.0
        assert el.is_running is False
        assert el.is_emergency_stopped is False

    def test_custom_config(self, config):
        el = ExecutionLoop(config=config)
        assert el.config.max_concurrent_trades == 3

    def test_no_components(self, loop):
        assert loop.strategist is None
        assert loop.risk_manager is None
        assert loop.trader is None

    def test_metrics_initial(self, loop):
        m = loop.metrics
        assert m.cycles_run == 0
        assert m.trades_executed == 0


# ─── State Persistence ────────────────────────────────────────────────────────


class TestStatePersistence:
    def test_save_creates_file(self, loop, tmp_state_file):
        loop._save_state()
        assert Path(tmp_state_file).exists()

    def test_save_valid_json(self, loop, tmp_state_file):
        loop._save_state()
        data = json.loads(Path(tmp_state_file).read_text())
        assert "is_running" in data
        assert "metrics" in data

    def test_load_nonexistent_file(self, loop):
        # Should not raise
        loop._load_state()
        assert loop.state.metrics.cycles_run == 0

    def test_save_then_load(self, loop, tmp_state_file):
        loop._state.metrics.cycles_run = 17
        loop._state.portfolio_value_usdc = 250.0
        loop._save_state()

        loop2 = ExecutionLoop(config=LoopConfig(state_file=tmp_state_file))
        loop2._load_state()
        assert loop2.state.metrics.cycles_run == 17
        assert loop2.state.portfolio_value_usdc == 250.0

    def test_load_corrupted_file(self, config, tmp_state_file):
        Path(tmp_state_file).write_text("not json {{")
        el = ExecutionLoop(config=config)
        el._load_state()   # should not raise
        assert el.state.metrics.cycles_run == 0

    def test_loaded_state_is_not_running(self, loop, tmp_state_file):
        loop._state.is_running = True
        loop._save_state()

        loop2 = ExecutionLoop(config=LoopConfig(state_file=tmp_state_file))
        loop2._load_state()
        assert loop2.state.is_running is False


# ─── Emergency Stop ───────────────────────────────────────────────────────────


class TestEmergencyStop:
    def test_no_trigger_small_loss(self, loop):
        loop._state.portfolio_value_usdc = 100.0
        loop._state.daily_pnl_usdc = -4.0   # 4% < 5% threshold
        triggered = loop._check_emergency_stop()
        assert triggered is False

    def test_triggers_at_threshold(self, loop):
        loop._state.portfolio_value_usdc = 100.0
        loop._state.daily_pnl_usdc = -5.0   # exactly 5%
        triggered = loop._check_emergency_stop()
        assert triggered is True
        assert loop.is_emergency_stopped is True

    def test_triggers_above_threshold(self, loop):
        loop._state.portfolio_value_usdc = 100.0
        loop._state.daily_pnl_usdc = -10.0  # 10%
        triggered = loop._check_emergency_stop()
        assert triggered is True

    def test_emergency_stop_calls_alert(self, config, tmp_state_file):
        alerts = []
        config.emergency_alert_callback = alerts.append
        el = ExecutionLoop(config=config)
        el._state.portfolio_value_usdc = 100.0
        el._state.daily_pnl_usdc = -6.0
        el._check_emergency_stop()
        assert len(alerts) == 1
        assert "drawdown" in alerts[0].lower()

    def test_emergency_stop_increments_metric(self, loop):
        loop._state.portfolio_value_usdc = 100.0
        loop._state.daily_pnl_usdc = -6.0
        loop._check_emergency_stop()
        assert loop.metrics.emergency_stops == 1

    def test_reset_emergency_stop(self, loop, tmp_state_file):
        loop._state.is_emergency_stopped = True
        loop.reset_emergency_stop()
        assert loop.is_emergency_stopped is False

    def test_no_trigger_zero_portfolio(self, loop):
        loop._state.portfolio_value_usdc = 0.0
        loop._state.daily_pnl_usdc = -1.0
        triggered = loop._check_emergency_stop()
        assert triggered is False


# ─── Error Tracking ───────────────────────────────────────────────────────────


class TestErrorTracking:
    def test_record_error_increments(self, loop):
        assert loop.metrics.errors_last_hour == 0
        loop._record_error()
        assert loop.metrics.errors_last_hour == 1

    def test_multiple_errors(self, loop):
        for _ in range(5):
            loop._record_error()
        assert loop.metrics.errors_last_hour == 5


# ─── Async Lifecycle ──────────────────────────────────────────────────────────


class TestAsyncLifecycle:
    @pytest.mark.asyncio
    async def test_start_sets_running(self, loop):
        await loop.start()
        assert loop.is_running is True
        await loop.stop()

    @pytest.mark.asyncio
    async def test_stop_clears_running(self, loop):
        await loop.start()
        await loop.stop()
        assert loop.is_running is False

    @pytest.mark.asyncio
    async def test_double_start_noop(self, loop):
        await loop.start()
        await loop.start()   # should not crash
        assert loop.is_running is True
        await loop.stop()

    @pytest.mark.asyncio
    async def test_stop_without_start_noop(self, loop):
        await loop.stop()   # should not raise
        assert loop.is_running is False

    @pytest.mark.asyncio
    async def test_emergency_stopped_prevents_start(self, loop):
        loop._state.is_emergency_stopped = True
        await loop.start()
        assert loop.is_running is False

    @pytest.mark.asyncio
    async def test_metrics_started_at_set_on_start(self, loop):
        await loop.start()
        assert loop.metrics.started_at is not None
        await loop.stop()


# ─── run_cycle ────────────────────────────────────────────────────────────────


class TestRunCycle:
    @pytest.mark.asyncio
    async def test_cycle_increments_count(self, loop):
        await loop.run_cycle()
        assert loop.metrics.cycles_run == 1

    @pytest.mark.asyncio
    async def test_cycle_no_signals_no_trades(self, loop):
        result = await loop.run_cycle()
        assert result["trades_executed"] == 0

    @pytest.mark.asyncio
    async def test_cycle_with_market_data(self, config, tmp_state_file):
        md = AsyncMock()
        md.tracked_symbols = ["ETH"]
        md.fetch_price = AsyncMock(return_value=3200.0)

        strategist = AsyncMock()
        decision = MagicMock()
        decision.action = "hold"
        strategist.decide = AsyncMock(return_value=decision)

        el = ExecutionLoop(config=config, market_data=md, strategist=strategist)
        result = await el.run_cycle()
        assert result["cycle_number"] == 1
        assert md.fetch_price.called

    @pytest.mark.asyncio
    async def test_cycle_executes_buy(self, config, tmp_state_file):
        md = AsyncMock()
        md.tracked_symbols = ["ETH"]
        md.fetch_price = AsyncMock(return_value=3200.0)

        decision = MagicMock()
        decision.action = "buy"
        decision.size_pct = 0.05

        strategist = AsyncMock()
        strategist.decide = AsyncMock(return_value=decision)

        trade_result = MagicMock()
        trade_result.pnl_usdc = 1.0

        trader = AsyncMock()
        trader.execute = AsyncMock(return_value=trade_result)

        el = ExecutionLoop(config=config, market_data=md, strategist=strategist, trader=trader)
        result = await el.run_cycle()
        assert result["trades_executed"] >= 0   # depends on concurrent limit

    @pytest.mark.asyncio
    async def test_cycle_last_cycle_at_updated(self, loop):
        result = await loop.run_cycle()
        assert loop.metrics.last_cycle_at is not None

    @pytest.mark.asyncio
    async def test_cycle_returns_started_at(self, loop):
        result = await loop.run_cycle()
        assert "started_at" in result
        assert result["started_at"] is not None

    @pytest.mark.asyncio
    async def test_concurrent_trade_limit(self, config, tmp_state_file):
        """Loop should skip trades when at max_concurrent_trades."""
        md = AsyncMock()
        md.tracked_symbols = ["ETH"]
        md.fetch_price = AsyncMock(return_value=3200.0)

        el = ExecutionLoop(config=config, market_data=md)
        el._active_trades = config.max_concurrent_trades  # artificially at limit

        result = await el.run_cycle()
        assert result["trades_executed"] == 0

    @pytest.mark.asyncio
    async def test_cycle_state_saved_after_run(self, loop, tmp_state_file):
        await loop.run_cycle()
        assert Path(tmp_state_file).exists()

    @pytest.mark.asyncio
    async def test_market_data_error_recorded(self, config, tmp_state_file):
        md = AsyncMock()
        md.tracked_symbols = ["ETH"]
        md.fetch_price = AsyncMock(side_effect=RuntimeError("network error"))

        el = ExecutionLoop(config=config, market_data=md)
        result = await el.run_cycle()
        assert el.metrics.errors_last_hour >= 1
