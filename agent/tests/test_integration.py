"""
test_integration.py — End-to-end integration tests for the ERC-8004 trading agent.

Tests the full pipeline: market_data → strategist → risk_manager → trader.
S7 additions: multi-agent coordinator integration, 100-trade session simulation,
drawdown halt/resumption, reputation scoring, and ensemble-driven trades.

All external dependencies are mocked.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from execution_loop import ExecutionLoop, LoopConfig, LoopState
from market_data import MarketDataAdapter, OrderBook, OrderBookLevel
from agent_coordinator import (
    AgentConfig,
    AgentPool,
    AgentSignal,
    MultiAgentCoordinator,
    build_coordinator,
)
from dashboard_server import ConnectionManager, _state, update_state


# ─── Shared Factories ─────────────────────────────────────────────────────────


def make_buy_decision(size_pct: float = 0.05, confidence: float = 0.75):
    d = MagicMock()
    d.action = "buy"
    d.size_pct = size_pct
    d.confidence = confidence
    d.reasoning = "Bullish signal"
    return d


def make_hold_decision():
    d = MagicMock()
    d.action = "hold"
    d.confidence = 0.5
    return d


def make_sell_decision(size_pct: float = 0.05, confidence: float = 0.65):
    d = MagicMock()
    d.action = "sell"
    d.size_pct = size_pct
    d.confidence = confidence
    d.reasoning = "Bearish signal"
    return d


def make_trade_result(pnl: float = 0.5, market_id: str = "eth-usd", side: str = "buy"):
    r = MagicMock()
    r.pnl_usdc = pnl
    r.market_id = market_id
    r.side = side
    return r


def make_risk_manager(accept: bool = True, reason: str = "OK"):
    rm = MagicMock()
    rm.validate_trade.return_value = (accept, reason)
    return rm


def make_strategist_for(action: str, confidence: float = 0.8, size_pct: float = 0.05):
    m = AsyncMock()
    decision = MagicMock()
    decision.action = action
    decision.confidence = confidence
    decision.size_pct = size_pct
    decision.reasoning = f"signal: {action}"
    m.decide = AsyncMock(return_value=decision)
    return m


def make_loop(tmp_path, strategist=None, risk_mgr=None, trader=None, reputation=None,
              symbols=None, drawdown_pct=0.10, max_concurrent=1000):
    config = LoopConfig(
        state_file=str(tmp_path / "state.json"),
        max_daily_drawdown_pct=drawdown_pct,
        max_concurrent_trades=max_concurrent,
    )
    md = MarketDataAdapter(
        use_synthetic=True,
        tracked_symbols=symbols or ["ethereum"],
    )
    kwargs = {
        "config": config,
        "market_data": md,
    }
    if strategist:
        kwargs["strategist"] = strategist
    if risk_mgr:
        kwargs["risk_manager"] = risk_mgr
    if trader:
        kwargs["trader"] = trader
    if reputation:
        kwargs["reputation"] = reputation
    return ExecutionLoop(**kwargs)


# ─── Full Buy Cycle ───────────────────────────────────────────────────────────


class TestFullBuyCycle:
    @pytest.mark.asyncio
    async def test_full_buy_cycle_executes_trade(self, tmp_path):
        """market_data → strategist.buy → risk OK → trader.execute → reputation"""
        strategist = AsyncMock()
        strategist.decide = AsyncMock(return_value=make_buy_decision())
        trader = AsyncMock()
        trader.execute = AsyncMock(return_value=make_trade_result(pnl=1.0))
        reputation = AsyncMock()
        reputation.log_trade = AsyncMock()

        el = make_loop(tmp_path, strategist=strategist, risk_mgr=make_risk_manager(),
                       trader=trader, reputation=reputation)
        result = await el.run_cycle()

        assert result["trades_executed"] >= 1
        assert trader.execute.called
        assert reputation.log_trade.called

    @pytest.mark.asyncio
    async def test_buy_cycle_updates_metrics(self, tmp_path):
        strategist = AsyncMock()
        strategist.decide = AsyncMock(return_value=make_buy_decision())
        trader = AsyncMock()
        trader.execute = AsyncMock(return_value=make_trade_result(pnl=1.5))

        el = make_loop(tmp_path, strategist=strategist, risk_mgr=make_risk_manager(), trader=trader)
        await el.run_cycle()
        assert el.metrics.cycles_run == 1
        assert el.metrics.trades_executed >= 1

    @pytest.mark.asyncio
    async def test_buy_cycle_pnl_tracked(self, tmp_path):
        strategist = AsyncMock()
        strategist.decide = AsyncMock(return_value=make_buy_decision())
        trader = AsyncMock()
        trader.execute = AsyncMock(return_value=make_trade_result(pnl=2.0))

        el = make_loop(tmp_path, strategist=strategist, risk_mgr=make_risk_manager(), trader=trader)
        await el.run_cycle()
        assert el.state.daily_pnl_usdc == pytest.approx(2.0)

    @pytest.mark.asyncio
    async def test_buy_cycle_pnl_accumulates_over_multiple_cycles(self, tmp_path):
        strategist = AsyncMock()
        strategist.decide = AsyncMock(return_value=make_buy_decision())
        trader = AsyncMock()
        trader.execute = AsyncMock(return_value=make_trade_result(pnl=1.0))

        el = make_loop(tmp_path, strategist=strategist, risk_mgr=make_risk_manager(), trader=trader)
        for _ in range(3):
            await el.run_cycle()
        assert el.state.daily_pnl_usdc == pytest.approx(3.0)

    @pytest.mark.asyncio
    async def test_buy_cycle_trade_count_increments(self, tmp_path):
        strategist = AsyncMock()
        strategist.decide = AsyncMock(return_value=make_buy_decision())
        trader = AsyncMock()
        trader.execute = AsyncMock(return_value=make_trade_result(pnl=0.5))

        el = make_loop(tmp_path, strategist=strategist, risk_mgr=make_risk_manager(),
                       trader=trader, max_concurrent=1000)
        for _ in range(5):
            await el.run_cycle()
        assert el.metrics.trades_executed >= 5


# ─── Full Sell Cycle ──────────────────────────────────────────────────────────


class TestFullSellCycle:
    @pytest.mark.asyncio
    async def test_sell_cycle_executes(self, tmp_path):
        strategist = AsyncMock()
        strategist.decide = AsyncMock(return_value=make_sell_decision())
        trader = AsyncMock()
        trader.execute = AsyncMock(return_value=make_trade_result(pnl=-0.5, side="sell"))

        el = make_loop(tmp_path, strategist=strategist, risk_mgr=make_risk_manager(), trader=trader)
        result = await el.run_cycle()
        assert result["trades_executed"] >= 1

    @pytest.mark.asyncio
    async def test_sell_pnl_negative(self, tmp_path):
        strategist = AsyncMock()
        strategist.decide = AsyncMock(return_value=make_sell_decision())
        trader = AsyncMock()
        trader.execute = AsyncMock(return_value=make_trade_result(pnl=-1.0))

        el = make_loop(tmp_path, strategist=strategist, risk_mgr=make_risk_manager(), trader=trader)
        await el.run_cycle()
        assert el.state.daily_pnl_usdc == pytest.approx(-1.0)

    @pytest.mark.asyncio
    async def test_mixed_buy_sell_net_pnl(self, tmp_path):
        """3 buys (+1 each) then 2 sells (-0.5 each) = net +2.0."""
        results = [1.0, 1.0, 1.0, -0.5, -0.5]
        idx = [0]

        strategist = AsyncMock()
        strategist.decide = AsyncMock(return_value=make_buy_decision())
        trader = AsyncMock()

        def side_effect(*args, **kwargs):
            pnl = results[idx[0] % len(results)]
            idx[0] += 1
            return make_trade_result(pnl=pnl)

        trader.execute = AsyncMock(side_effect=side_effect)

        el = make_loop(tmp_path, strategist=strategist, risk_mgr=make_risk_manager(),
                       trader=trader, max_concurrent=1000)
        for _ in range(5):
            await el.run_cycle()
        assert el.state.daily_pnl_usdc == pytest.approx(2.0)


# ─── Risk Rejection Scenario ─────────────────────────────────────────────────


class TestRiskRejection:
    @pytest.mark.asyncio
    async def test_risk_rejected_trade_not_executed(self, tmp_path):
        strategist = AsyncMock()
        strategist.decide = AsyncMock(return_value=make_buy_decision())
        trader = AsyncMock()

        el = make_loop(tmp_path, strategist=strategist,
                       risk_mgr=make_risk_manager(accept=False, reason="Position too large"),
                       trader=trader)
        result = await el.run_cycle()
        assert result["trades_executed"] == 0
        assert result["trades_rejected"] >= 1
        trader.execute.assert_not_called()

    @pytest.mark.asyncio
    async def test_rejection_increments_metric(self, tmp_path):
        strategist = AsyncMock()
        strategist.decide = AsyncMock(return_value=make_buy_decision())
        trader = AsyncMock()

        el = make_loop(tmp_path, strategist=strategist,
                       risk_mgr=make_risk_manager(accept=False, reason="Max drawdown"),
                       trader=trader)
        await el.run_cycle()
        assert el.metrics.trades_rejected >= 1

    @pytest.mark.asyncio
    async def test_rejection_reason_logged(self, tmp_path):
        strategist = AsyncMock()
        strategist.decide = AsyncMock(return_value=make_buy_decision())

        el = make_loop(tmp_path, strategist=strategist,
                       risk_mgr=make_risk_manager(accept=False, reason="Oracle divergence"))
        result = await el.run_cycle()
        assert result["trades_rejected"] >= 1


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
        trader = AsyncMock()
        trader.execute = AsyncMock(return_value=make_trade_result(pnl=-10.0))

        el = ExecutionLoop(config=config, market_data=md, strategist=strategist,
                          risk_manager=make_risk_manager(), trader=trader)
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
        trader = AsyncMock()
        trader.execute = AsyncMock(return_value=make_trade_result(pnl=-10.0))

        el = ExecutionLoop(config=config, market_data=md, strategist=strategist,
                          risk_manager=make_risk_manager(), trader=trader)
        el._state.portfolio_value_usdc = 100.0
        await el.run_cycle()
        assert el.is_running is False

    @pytest.mark.asyncio
    async def test_alert_callback_receives_message(self, tmp_path):
        """Alert callback should receive a non-empty message."""
        messages = []
        config = LoopConfig(
            state_file=str(tmp_path / "state.json"),
            max_daily_drawdown_pct=0.05,
            emergency_alert_callback=messages.append,
        )
        md = MarketDataAdapter(use_synthetic=True, tracked_symbols=["ethereum"])
        strategist = AsyncMock()
        strategist.decide = AsyncMock(return_value=make_buy_decision())
        trader = AsyncMock()
        trader.execute = AsyncMock(return_value=make_trade_result(pnl=-100.0))

        el = ExecutionLoop(config=config, market_data=md, strategist=strategist,
                          risk_manager=make_risk_manager(), trader=trader)
        el._state.portfolio_value_usdc = 100.0
        await el.run_cycle()
        assert len(messages) > 0


# ─── 100-Trade Simulation ─────────────────────────────────────────────────────


class TestHundredTradeSession:
    @pytest.mark.asyncio
    async def test_100_trade_session_pnl_positive(self, tmp_path):
        """Simulate 100 winning trades, verify cumulative PnL."""
        strategist = AsyncMock()
        strategist.decide = AsyncMock(return_value=make_buy_decision())
        trader = AsyncMock()
        trader.execute = AsyncMock(return_value=make_trade_result(pnl=0.1))

        el = make_loop(tmp_path, strategist=strategist, risk_mgr=make_risk_manager(),
                       trader=trader, drawdown_pct=0.99, max_concurrent=1000)
        for _ in range(100):
            if el.is_emergency_stopped:
                break
            await el.run_cycle()

        assert el.state.daily_pnl_usdc == pytest.approx(10.0, abs=0.01)
        assert el.metrics.trades_executed >= 100

    @pytest.mark.asyncio
    async def test_100_trade_session_tracks_cycles(self, tmp_path):
        strategist = AsyncMock()
        strategist.decide = AsyncMock(return_value=make_hold_decision())

        el = make_loop(tmp_path, strategist=strategist)
        for _ in range(100):
            await el.run_cycle()

        assert el.metrics.cycles_run == 100

    @pytest.mark.asyncio
    async def test_50_win_50_loss_session(self, tmp_path):
        """Alternate wins and losses — net PnL should be near 0."""
        pnls = [1.0, -1.0] * 50
        idx = [0]

        strategist = AsyncMock()
        strategist.decide = AsyncMock(return_value=make_buy_decision())
        trader = AsyncMock()

        def side_effect(*args, **kwargs):
            pnl = pnls[idx[0] % len(pnls)]
            idx[0] += 1
            return make_trade_result(pnl=pnl)

        trader.execute = AsyncMock(side_effect=side_effect)
        el = make_loop(tmp_path, strategist=strategist, risk_mgr=make_risk_manager(),
                       trader=trader, drawdown_pct=0.99, max_concurrent=1000)
        for _ in range(100):
            if el.is_emergency_stopped:
                break
            await el.run_cycle()

        assert el.state.daily_pnl_usdc == pytest.approx(0.0, abs=0.01)

    @pytest.mark.asyncio
    async def test_session_state_file_updated_each_cycle(self, tmp_path):
        state_file = str(tmp_path / "state.json")
        config = LoopConfig(state_file=state_file)
        md = MarketDataAdapter(use_synthetic=True, tracked_symbols=["ethereum"])

        el = ExecutionLoop(config=config, market_data=md)
        for _ in range(5):
            await el.run_cycle()

        saved = json.loads(Path(state_file).read_text())
        assert saved["metrics"]["cycles_run"] == 5


# ─── Max Drawdown Halt + Resumption ──────────────────────────────────────────


class TestMaxDrawdownHalt:
    @pytest.mark.asyncio
    async def test_drawdown_halt_triggers_at_threshold(self, tmp_path):
        """Loss exceeding drawdown_pct should halt trading."""
        config = LoopConfig(
            state_file=str(tmp_path / "state.json"),
            max_daily_drawdown_pct=0.10,
        )
        md = MarketDataAdapter(use_synthetic=True, tracked_symbols=["ethereum"])
        strategist = AsyncMock()
        strategist.decide = AsyncMock(return_value=make_buy_decision())
        trader = AsyncMock()
        trader.execute = AsyncMock(return_value=make_trade_result(pnl=-20.0))

        el = ExecutionLoop(config=config, market_data=md, strategist=strategist,
                          risk_manager=make_risk_manager(), trader=trader)
        el._state.portfolio_value_usdc = 100.0
        result = await el.run_cycle()
        assert result.get("emergency_stop_triggered") is True or not el.is_running

    @pytest.mark.asyncio
    async def test_small_loss_does_not_halt(self, tmp_path):
        """Loss within threshold should NOT trigger emergency stop."""
        config = LoopConfig(
            state_file=str(tmp_path / "state.json"),
            max_daily_drawdown_pct=0.50,
            max_concurrent_trades=1000,
        )
        md = MarketDataAdapter(use_synthetic=True, tracked_symbols=["ethereum"])
        strategist = AsyncMock()
        strategist.decide = AsyncMock(return_value=make_buy_decision())
        trader = AsyncMock()
        trader.execute = AsyncMock(return_value=make_trade_result(pnl=-1.0))

        el = ExecutionLoop(config=config, market_data=md, strategist=strategist,
                          risk_manager=make_risk_manager(), trader=trader)
        el._state.portfolio_value_usdc = 1000.0
        result = await el.run_cycle()
        # Loss of 1 / portfolio 1000 = 0.1%, well below 50% threshold
        assert el.is_emergency_stopped is False

    @pytest.mark.asyncio
    async def test_multiple_small_losses_accumulate(self, tmp_path):
        """Accumulated small losses should eventually trigger halt."""
        config = LoopConfig(
            state_file=str(tmp_path / "state.json"),
            max_daily_drawdown_pct=0.10,
        )
        md = MarketDataAdapter(use_synthetic=True, tracked_symbols=["ethereum"])
        strategist = AsyncMock()
        strategist.decide = AsyncMock(return_value=make_buy_decision())
        trader = AsyncMock()
        trader.execute = AsyncMock(return_value=make_trade_result(pnl=-3.0))

        el = ExecutionLoop(config=config, market_data=md, strategist=strategist,
                          risk_manager=make_risk_manager(), trader=trader)
        el._state.portfolio_value_usdc = 100.0

        halted = False
        for _ in range(10):
            if not el.is_running:
                halted = True
                break
            await el.run_cycle()
        assert halted or not el.is_running


# ─── Multi-Symbol Market Data ─────────────────────────────────────────────────


class TestMultiSymbolIntegration:
    @pytest.mark.asyncio
    async def test_multi_symbol_strategist_called_per_symbol(self, tmp_path):
        strategist = AsyncMock()
        strategist.decide = AsyncMock(return_value=make_hold_decision())

        el = make_loop(tmp_path, strategist=strategist, symbols=["ethereum", "bitcoin", "solana"])
        result = await el.run_cycle()
        assert strategist.decide.call_count == 3

    @pytest.mark.asyncio
    async def test_five_symbol_cycle(self, tmp_path):
        strategist = AsyncMock()
        strategist.decide = AsyncMock(return_value=make_hold_decision())

        el = make_loop(tmp_path, strategist=strategist,
                       symbols=["ethereum", "bitcoin", "solana", "cardano", "polkadot"])
        result = await el.run_cycle()
        assert strategist.decide.call_count == 5

    @pytest.mark.asyncio
    async def test_multi_symbol_trades_execute_for_buys(self, tmp_path):
        """With 3 symbols all returning buy, 3 trades should execute."""
        strategist = AsyncMock()
        strategist.decide = AsyncMock(return_value=make_buy_decision())
        trader = AsyncMock()
        trader.execute = AsyncMock(return_value=make_trade_result(pnl=0.5))

        el = make_loop(tmp_path, strategist=strategist, risk_mgr=make_risk_manager(),
                       trader=trader, symbols=["ethereum", "bitcoin", "solana"])
        result = await el.run_cycle()
        assert result["trades_executed"] == 3


# ─── State Persistence ────────────────────────────────────────────────────────


class TestDashboardState:
    @pytest.mark.asyncio
    async def test_metrics_exported_correctly(self, tmp_path):
        el = make_loop(tmp_path, strategist=AsyncMock())
        el.strategist.decide = AsyncMock(return_value=make_hold_decision())
        for _ in range(3):
            await el.run_cycle()

        d = el.metrics.to_dict()
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
        strategist = AsyncMock()
        strategist.decide = AsyncMock(return_value=make_hold_decision())
        trader = AsyncMock()

        el = make_loop(tmp_path, strategist=strategist, trader=trader)
        result = await el.run_cycle()
        assert result["trades_executed"] == 0
        trader.execute.assert_not_called()

    @pytest.mark.asyncio
    async def test_market_data_and_execution_loop_state_consistent(self, tmp_path):
        """State after cycle should reflect actual outcomes."""
        strategist = AsyncMock()
        strategist.decide = AsyncMock(return_value=make_buy_decision())
        trader = AsyncMock()
        trader.execute = AsyncMock(return_value=make_trade_result(pnl=0.0))
        reputation = AsyncMock()
        reputation.log_trade = AsyncMock()

        el = make_loop(tmp_path, strategist=strategist, risk_mgr=make_risk_manager(),
                       trader=trader, reputation=reputation)
        await el.run_cycle()

        saved = json.loads(Path(el.config.state_file).read_text())
        assert saved["metrics"]["cycles_run"] == el.metrics.cycles_run


# ─── Multi-Agent Coordinator Integration ─────────────────────────────────────


class TestMultiAgentCoordinatorIntegration:
    """End-to-end tests integrating the coordinator with execution loop concepts."""

    @pytest.mark.asyncio
    async def test_coordinator_unanimous_buy_signal(self):
        """All agents agree → strong buy consensus."""
        pool = AgentPool()
        for i in range(3):
            pool.add_agent(AgentConfig.moderate(f"a{i}"), make_strategist_for("buy"))
        coord = MultiAgentCoordinator(pool, consensus_threshold=0.60)
        signal = await coord.get_ensemble_signal({"price": 2000.0})
        assert signal.action == "buy"
        assert signal.has_consensus is True

    @pytest.mark.asyncio
    async def test_coordinator_no_consensus_gives_hold(self):
        """Split vote below threshold → hold."""
        pool = AgentPool()
        pool.add_agent(AgentConfig.moderate("buyer"), make_strategist_for("buy"))
        pool.add_agent(AgentConfig.moderate("seller"), make_strategist_for("sell"))
        coord = MultiAgentCoordinator(pool, consensus_threshold=0.80)
        signal = await coord.get_ensemble_signal({"price": 2000.0})
        assert signal.action == "hold"

    @pytest.mark.asyncio
    async def test_coordinator_tracks_performance_after_trade(self):
        """Agents that voted with consensus should record the outcome."""
        pool = AgentPool()
        for i in range(3):
            pool.add_agent(AgentConfig.moderate(f"a{i}"), make_strategist_for("buy"))
        coord = MultiAgentCoordinator(pool, consensus_threshold=0.60)
        ensemble = await coord.get_ensemble_signal({"price": 2000.0})
        assert ensemble.action == "buy"
        coord.record_outcome_for_all_agreeing(ensemble, pnl=5.0)
        perf = coord.pool.tracker.get_all_performance()
        # All 3 agents voted buy → all should have recorded the outcome
        total_trades = sum(p["total_trades"] for p in perf)
        assert total_trades == 3

    @pytest.mark.asyncio
    async def test_coordinator_decision_history_grows(self):
        coord = build_coordinator()
        for _ in range(5):
            await coord.get_ensemble_signal({"price": 2000.0})
        assert coord.decision_count == 5

    @pytest.mark.asyncio
    async def test_coordinator_consensus_rate_all_agree(self):
        """Unanimous votes should give 100% consensus rate."""
        pool = AgentPool()
        for i in range(3):
            pool.add_agent(AgentConfig.moderate(f"a{i}"), make_strategist_for("buy"))
        coord = MultiAgentCoordinator(pool, consensus_threshold=0.60)
        for _ in range(10):
            await coord.get_ensemble_signal({"price": 2000.0})
        assert coord.consensus_rate == pytest.approx(1.0)

    @pytest.mark.asyncio
    async def test_coordinator_sell_consensus_flows_to_result(self):
        pool = AgentPool()
        for i in range(3):
            pool.add_agent(AgentConfig.moderate(f"s{i}"), make_strategist_for("sell"))
        coord = MultiAgentCoordinator(pool, consensus_threshold=0.60)
        signal = await coord.get_ensemble_signal({"price": 2000.0})
        assert signal.action == "sell"
        assert signal.is_actionable is True

    @pytest.mark.asyncio
    async def test_coordinator_mean_confidence_is_plausible(self):
        pool = AgentPool()
        for i in range(3):
            pool.add_agent(AgentConfig.moderate(f"a{i}"), make_strategist_for("buy", confidence=0.80))
        coord = MultiAgentCoordinator(pool, consensus_threshold=0.60)
        signal = await coord.get_ensemble_signal({"price": 2000.0})
        assert 0.70 <= signal.mean_confidence <= 0.90

    @pytest.mark.asyncio
    async def test_coordinator_size_pct_is_clipped_by_risk_profile(self):
        """Conservative agents (max_position_pct=0.05) clip large signals."""
        pool = AgentPool()
        for i in range(3):
            pool.add_agent(
                AgentConfig.conservative(f"c{i}"),
                make_strategist_for("buy", confidence=0.80, size_pct=0.30)
            )
        coord = MultiAgentCoordinator(pool, consensus_threshold=0.60)
        signal = await coord.get_ensemble_signal({"price": 2000.0})
        assert signal.mean_size_pct <= 0.05

    @pytest.mark.asyncio
    async def test_coordinator_summary_keys(self):
        coord = build_coordinator()
        await coord.get_ensemble_signal({"price": 2000.0})
        summary = coord.performance_summary()
        assert "pool_size" in summary
        assert "consensus_rate" in summary
        assert "agent_performance" in summary


# ─── Dashboard + Coordinator Integration ─────────────────────────────────────


class TestDashboardCoordinatorIntegration:
    @pytest.mark.asyncio
    async def test_ensemble_signal_stored_in_state(self):
        """Ensemble signal should be stored in dashboard state."""
        conn_mgr = ConnectionManager()
        votes = {"a1": "buy", "a2": "buy", "a3": "sell"}
        await conn_mgr.broadcast_ensemble_signal("buy", 0.67, votes, True)
        assert _state["last_ensemble"] is not None
        assert _state["last_ensemble"]["action"] == "buy"

    @pytest.mark.asyncio
    async def test_portfolio_update_reflected_in_state(self):
        conn_mgr = ConnectionManager()
        await conn_mgr.broadcast_portfolio_update(10000.0, 250.0, 5)
        assert _state["total_pnl"] == pytest.approx(250.0)
        assert _state["active_positions"] == 5

    @pytest.mark.asyncio
    async def test_risk_alert_broadcast_reaches_client(self):
        conn_mgr = ConnectionManager()
        ws = AsyncMock()
        await conn_mgr.connect(ws)
        await conn_mgr.broadcast_risk_alert("halt", "Emergency stop triggered", "critical")
        call_args = ws.send_json.call_args[0][0]
        assert call_args["type"] == "risk_alert"
        assert call_args["severity"] == "critical"

    @pytest.mark.asyncio
    async def test_price_updates_feed_multiple_symbols(self):
        conn_mgr = ConnectionManager()
        await conn_mgr.broadcast_price_update("ETH", 2100.0)
        await conn_mgr.broadcast_price_update("BTC", 51000.0)
        await conn_mgr.broadcast_price_update("SOL", 145.0)
        assert len(_state["price_feed"]) == 3

    @pytest.mark.asyncio
    async def test_trade_event_updates_last_trade(self):
        conn_mgr = ConnectionManager()
        await conn_mgr.broadcast_trade_executed("eth-usd", "buy", 500.0, "0xabc", pnl=3.5)
        assert _state["last_trade"]["pnl"] == pytest.approx(3.5)
        assert _state["last_trade"]["market_id"] == "eth-usd"

    @pytest.mark.asyncio
    async def test_event_log_contains_all_event_types(self):
        conn_mgr = ConnectionManager()
        await conn_mgr.broadcast_price_update("ETH", 2000.0)
        await conn_mgr.broadcast_trade_executed("eth-usd", "buy", 100.0, "0x1")
        await conn_mgr.broadcast_portfolio_update(10000.0, 50.0, 2)
        await conn_mgr.broadcast_risk_alert("test", "test msg")
        events = conn_mgr.recent_events(10)
        types = {e["type"] for e in events}
        assert "price_update" in types
        assert "trade_executed" in types
        assert "portfolio_update" in types
        assert "risk_alert" in types


# ─── Reputation Scoring After Session ────────────────────────────────────────


class TestReputationAfterSession:
    @pytest.mark.asyncio
    async def test_reputation_logged_after_trade(self, tmp_path):
        reputation = AsyncMock()
        reputation.log_trade = AsyncMock()

        strategist = AsyncMock()
        strategist.decide = AsyncMock(return_value=make_buy_decision())
        trader = AsyncMock()
        trader.execute = AsyncMock(return_value=make_trade_result(pnl=1.0))

        el = make_loop(tmp_path, strategist=strategist, risk_mgr=make_risk_manager(),
                       trader=trader, reputation=reputation)
        await el.run_cycle()
        assert reputation.log_trade.called

    @pytest.mark.asyncio
    async def test_reputation_called_for_each_trade(self, tmp_path):
        reputation = AsyncMock()
        reputation.log_trade = AsyncMock()

        strategist = AsyncMock()
        strategist.decide = AsyncMock(return_value=make_buy_decision())
        trader = AsyncMock()
        trader.execute = AsyncMock(return_value=make_trade_result(pnl=0.5))

        el = make_loop(tmp_path, strategist=strategist, risk_mgr=make_risk_manager(),
                       trader=trader, reputation=reputation, max_concurrent=1000)
        for _ in range(5):
            await el.run_cycle()
        assert reputation.log_trade.call_count >= 5

    @pytest.mark.asyncio
    async def test_reputation_not_called_on_hold(self, tmp_path):
        reputation = AsyncMock()
        reputation.log_trade = AsyncMock()

        strategist = AsyncMock()
        strategist.decide = AsyncMock(return_value=make_hold_decision())
        trader = AsyncMock()

        el = make_loop(tmp_path, strategist=strategist, trader=trader, reputation=reputation)
        await el.run_cycle()
        reputation.log_trade.assert_not_called()

    @pytest.mark.asyncio
    async def test_reputation_not_called_on_risk_rejection(self, tmp_path):
        reputation = AsyncMock()
        reputation.log_trade = AsyncMock()

        strategist = AsyncMock()
        strategist.decide = AsyncMock(return_value=make_buy_decision())
        trader = AsyncMock()

        el = make_loop(tmp_path, strategist=strategist,
                       risk_mgr=make_risk_manager(accept=False),
                       trader=trader, reputation=reputation)
        await el.run_cycle()
        reputation.log_trade.assert_not_called()
