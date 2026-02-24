"""
test_integration.py — End-to-end integration tests for the ERC-8004 trading agent.

Tests the full pipeline: market_data → strategist → risk_manager → trader.
All external dependencies are mocked.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from execution_loop import ExecutionLoop, LoopConfig, LoopState
from market_data import MarketDataAdapter, OrderBook, OrderBookLevel


# ─── Shared Mocks ─────────────────────────────────────────────────────────────


def make_buy_decision(size_pct: float = 0.05):
    d = MagicMock()
    d.action = "buy"
    d.size_pct = size_pct
    d.confidence = 0.75
    d.reasoning = "Bullish signal"
    return d


def make_hold_decision():
    d = MagicMock()
    d.action = "hold"
    return d


def make_sell_decision(size_pct: float = 0.05):
    d = MagicMock()
    d.action = "sell"
    d.size_pct = size_pct
    d.confidence = 0.65
    d.reasoning = "Bearish signal"
    return d


def make_trade_result(pnl: float = 0.5):
    r = MagicMock()
    r.pnl_usdc = pnl
    r.market_id = "eth-usd"
    r.side = "buy"
    return r


def make_risk_manager(accept: bool = True, reason: str = "OK"):
    rm = MagicMock()
    rm.validate_trade.return_value = (accept, reason)
    return rm


# ─── Full Buy Cycle ───────────────────────────────────────────────────────────


class TestFullBuyCycle:
    @pytest.mark.asyncio
    async def test_full_buy_cycle_executes_trade(self, tmp_path):
        """market_data → strategist.buy → risk OK → trader.execute → reputation"""
        config = LoopConfig(state_file=str(tmp_path / "state.json"))
        md = MarketDataAdapter(use_synthetic=True, tracked_symbols=["ethereum"])
        strategist = AsyncMock()
        strategist.decide = AsyncMock(return_value=make_buy_decision())
        risk_mgr = make_risk_manager(accept=True)
        trader = AsyncMock()
        trader.execute = AsyncMock(return_value=make_trade_result(pnl=1.0))
        reputation = AsyncMock()
        reputation.log_trade = AsyncMock()

        el = ExecutionLoop(
            config=config,
            market_data=md,
            strategist=strategist,
            risk_manager=risk_mgr,
            trader=trader,
            reputation=reputation,
        )
        result = await el.run_cycle()

        assert result["trades_executed"] >= 1
        assert trader.execute.called
        assert reputation.log_trade.called

    @pytest.mark.asyncio
    async def test_buy_cycle_updates_metrics(self, tmp_path):
        config = LoopConfig(state_file=str(tmp_path / "state.json"))
        md = MarketDataAdapter(use_synthetic=True, tracked_symbols=["ethereum"])
        strategist = AsyncMock()
        strategist.decide = AsyncMock(return_value=make_buy_decision())
        risk_mgr = make_risk_manager(accept=True)
        trader = AsyncMock()
        trader.execute = AsyncMock(return_value=make_trade_result(pnl=1.5))

        el = ExecutionLoop(
            config=config, market_data=md, strategist=strategist,
            risk_manager=risk_mgr, trader=trader,
        )
        await el.run_cycle()
        assert el.metrics.cycles_run == 1
        assert el.metrics.trades_executed >= 1

    @pytest.mark.asyncio
    async def test_buy_cycle_pnl_tracked(self, tmp_path):
        config = LoopConfig(state_file=str(tmp_path / "state.json"))
        md = MarketDataAdapter(use_synthetic=True, tracked_symbols=["ethereum"])
        strategist = AsyncMock()
        strategist.decide = AsyncMock(return_value=make_buy_decision())
        risk_mgr = make_risk_manager(accept=True)
        trader = AsyncMock()
        trader.execute = AsyncMock(return_value=make_trade_result(pnl=2.0))

        el = ExecutionLoop(
            config=config, market_data=md, strategist=strategist,
            risk_manager=risk_mgr, trader=trader,
        )
        await el.run_cycle()
        assert el.state.daily_pnl_usdc == pytest.approx(2.0)


# ─── Full Sell Cycle ──────────────────────────────────────────────────────────


class TestFullSellCycle:
    @pytest.mark.asyncio
    async def test_sell_cycle_executes(self, tmp_path):
        config = LoopConfig(state_file=str(tmp_path / "state.json"))
        md = MarketDataAdapter(use_synthetic=True, tracked_symbols=["ethereum"])
        strategist = AsyncMock()
        strategist.decide = AsyncMock(return_value=make_sell_decision())
        risk_mgr = make_risk_manager(accept=True)
        trader = AsyncMock()
        trader.execute = AsyncMock(return_value=make_trade_result(pnl=-0.5))

        el = ExecutionLoop(
            config=config, market_data=md, strategist=strategist,
            risk_manager=risk_mgr, trader=trader,
        )
        result = await el.run_cycle()
        assert result["trades_executed"] >= 1

    @pytest.mark.asyncio
    async def test_sell_pnl_negative(self, tmp_path):
        config = LoopConfig(state_file=str(tmp_path / "state.json"))
        md = MarketDataAdapter(use_synthetic=True, tracked_symbols=["ethereum"])
        strategist = AsyncMock()
        strategist.decide = AsyncMock(return_value=make_sell_decision())
        risk_mgr = make_risk_manager(accept=True)
        trader = AsyncMock()
        trader.execute = AsyncMock(return_value=make_trade_result(pnl=-1.0))

        el = ExecutionLoop(
            config=config, market_data=md, strategist=strategist,
            risk_manager=risk_mgr, trader=trader,
        )
        await el.run_cycle()
        assert el.state.daily_pnl_usdc == pytest.approx(-1.0)


# ─── Risk Rejection Scenario ─────────────────────────────────────────────────


class TestRiskRejection:
    @pytest.mark.asyncio
    async def test_risk_rejected_trade_not_executed(self, tmp_path):
        config = LoopConfig(state_file=str(tmp_path / "state.json"))
        md = MarketDataAdapter(use_synthetic=True, tracked_symbols=["ethereum"])
        strategist = AsyncMock()
        strategist.decide = AsyncMock(return_value=make_buy_decision())
        risk_mgr = make_risk_manager(accept=False, reason="Position too large")
        trader = AsyncMock()

        el = ExecutionLoop(
            config=config, market_data=md, strategist=strategist,
            risk_manager=risk_mgr, trader=trader,
        )
        result = await el.run_cycle()
        assert result["trades_executed"] == 0
        assert result["trades_rejected"] >= 1
        trader.execute.assert_not_called()

    @pytest.mark.asyncio
    async def test_rejection_increments_metric(self, tmp_path):
        config = LoopConfig(state_file=str(tmp_path / "state.json"))
        md = MarketDataAdapter(use_synthetic=True, tracked_symbols=["ethereum"])
        strategist = AsyncMock()
        strategist.decide = AsyncMock(return_value=make_buy_decision())
        risk_mgr = make_risk_manager(accept=False, reason="Max drawdown")
        trader = AsyncMock()

        el = ExecutionLoop(
            config=config, market_data=md, strategist=strategist,
            risk_manager=risk_mgr, trader=trader,
        )
        await el.run_cycle()
        assert el.metrics.trades_rejected >= 1


# ─── Emergency Stop Scenario ─────────────────────────────────────────────────


class TestEmergencyStopScenario:
    @pytest.mark.asyncio
    async def test_emergency_stop_on_large_loss(self, tmp_path):
        alerts = []
        config = LoopConfig(
            state_file=str(tmp_path / "state.json"),
            max_daily_drawdown_pct=0.05,
            emergency_alert_callback=alerts.append,
        )
        md = MarketDataAdapter(use_synthetic=True, tracked_symbols=["ethereum"])
        strategist = AsyncMock()
        strategist.decide = AsyncMock(return_value=make_buy_decision())
        risk_mgr = make_risk_manager(accept=True)
        trader = AsyncMock()
        trader.execute = AsyncMock(return_value=make_trade_result(pnl=-10.0))

        el = ExecutionLoop(
            config=config, market_data=md, strategist=strategist,
            risk_manager=risk_mgr, trader=trader,
        )
        el._state.portfolio_value_usdc = 100.0
        result = await el.run_cycle()

        assert result["emergency_stop_triggered"] is True
        assert el.is_emergency_stopped is True
        assert len(alerts) >= 1

    @pytest.mark.asyncio
    async def test_emergency_stop_stops_loop(self, tmp_path):
        config = LoopConfig(
            state_file=str(tmp_path / "state.json"),
            max_daily_drawdown_pct=0.05,
        )
        md = MarketDataAdapter(use_synthetic=True, tracked_symbols=["ethereum"])
        strategist = AsyncMock()
        strategist.decide = AsyncMock(return_value=make_buy_decision())
        risk_mgr = make_risk_manager(accept=True)
        trader = AsyncMock()
        trader.execute = AsyncMock(return_value=make_trade_result(pnl=-10.0))

        el = ExecutionLoop(
            config=config, market_data=md, strategist=strategist,
            risk_manager=risk_mgr, trader=trader,
        )
        el._state.portfolio_value_usdc = 100.0
        await el.run_cycle()
        assert el.is_running is False


# ─── Dashboard Reflects Loop State ───────────────────────────────────────────


class TestDashboardState:
    @pytest.mark.asyncio
    async def test_metrics_exported_correctly(self, tmp_path):
        config = LoopConfig(state_file=str(tmp_path / "state.json"))
        md = MarketDataAdapter(use_synthetic=True, tracked_symbols=["ethereum"])
        strategist = AsyncMock()
        strategist.decide = AsyncMock(return_value=make_hold_decision())

        el = ExecutionLoop(config=config, market_data=md, strategist=strategist)
        await el.run_cycle()
        await el.run_cycle()
        await el.run_cycle()

        metrics = el.metrics
        d = metrics.to_dict()
        assert d["cycles_run"] == 3
        assert d["uptime_seconds"] >= 0

    @pytest.mark.asyncio
    async def test_state_persisted_after_cycle(self, tmp_path):
        state_file = str(tmp_path / "state.json")
        config = LoopConfig(state_file=state_file)
        md = MarketDataAdapter(use_synthetic=True, tracked_symbols=["ethereum"])

        el = ExecutionLoop(config=config, market_data=md)
        await el.run_cycle()

        saved = json.loads(Path(state_file).read_text())
        assert saved["metrics"]["cycles_run"] == 1

    @pytest.mark.asyncio
    async def test_hold_decision_no_trades(self, tmp_path):
        config = LoopConfig(state_file=str(tmp_path / "state.json"))
        md = MarketDataAdapter(use_synthetic=True, tracked_symbols=["ethereum"])
        strategist = AsyncMock()
        strategist.decide = AsyncMock(return_value=make_hold_decision())
        trader = AsyncMock()

        el = ExecutionLoop(
            config=config, market_data=md, strategist=strategist, trader=trader,
        )
        result = await el.run_cycle()
        assert result["trades_executed"] == 0
        trader.execute.assert_not_called()

    @pytest.mark.asyncio
    async def test_multi_symbol_market_data(self, tmp_path):
        config = LoopConfig(state_file=str(tmp_path / "state.json"))
        md = MarketDataAdapter(
            use_synthetic=True,
            tracked_symbols=["ethereum", "bitcoin", "solana"],
        )
        strategist = AsyncMock()
        strategist.decide = AsyncMock(return_value=make_hold_decision())

        el = ExecutionLoop(config=config, market_data=md, strategist=strategist)
        result = await el.run_cycle()
        # Three symbols → strategist called 3 times
        assert strategist.decide.call_count == 3

    @pytest.mark.asyncio
    async def test_market_data_and_execution_loop_state_consistent(self, tmp_path):
        """State after cycle should reflect actual outcomes."""
        config = LoopConfig(state_file=str(tmp_path / "state.json"))
        md = MarketDataAdapter(use_synthetic=True, tracked_symbols=["ethereum"])
        strategist = AsyncMock()
        strategist.decide = AsyncMock(return_value=make_buy_decision())
        risk_mgr = make_risk_manager(accept=True)
        trader = AsyncMock()
        trader.execute = AsyncMock(return_value=make_trade_result(pnl=0.0))
        reputation = AsyncMock()
        reputation.log_trade = AsyncMock()

        el = ExecutionLoop(
            config=config, market_data=md, strategist=strategist,
            risk_manager=risk_mgr, trader=trader, reputation=reputation,
        )
        await el.run_cycle()

        # State file should be written and consistent with metrics
        saved = json.loads(Path(config.state_file).read_text())
        assert saved["metrics"]["cycles_run"] == el.metrics.cycles_run
