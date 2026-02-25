"""
test_agent_coordinator.py — Tests for agent_coordinator.py.

65 tests covering:
  - AgentConfig validation and factory methods
  - AgentSignal validation
  - AgentPerformance tracking (win rate, PnL, Sharpe)
  - AgentPerformanceTracker weight rebalancing
  - AgentPool add/remove/collect_signals
  - EnsembleSignal properties
  - MultiAgentCoordinator consensus voting
  - Edge cases (no quorum, unanimous, ties, weight dynamics)
"""

from __future__ import annotations

import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from agent_coordinator import (
    AgentConfig,
    AgentPerformance,
    AgentPerformanceTracker,
    AgentPool,
    AgentSignal,
    EnsembleSignal,
    MultiAgentCoordinator,
    RiskProfile,
    build_coordinator,
    build_default_pool,
)


# ─── Fixtures ─────────────────────────────────────────────────────────────────


def make_signal(agent_id: str, action: str, confidence: float = 0.8, size_pct: float = 0.05) -> AgentSignal:
    return AgentSignal(
        agent_id=agent_id,
        action=action,
        confidence=confidence,
        size_pct=size_pct,
    )


def make_strategist(action: str, confidence: float = 0.8, size_pct: float = 0.05):
    """Return an async-mock strategist that decides with given params."""
    m = AsyncMock()
    decision = MagicMock()
    decision.action = action
    decision.confidence = confidence
    decision.size_pct = size_pct
    decision.reasoning = f"Test {action}"
    m.decide = AsyncMock(return_value=decision)
    return m


SNAPSHOT = {"price": 2000.0, "symbol": "ETH"}


# ─── AgentConfig Tests ────────────────────────────────────────────────────────


class TestAgentConfig:
    def test_default_construction(self):
        cfg = AgentConfig(agent_id="a1", risk_profile=RiskProfile.MODERATE)
        assert cfg.agent_id == "a1"
        assert cfg.risk_profile == RiskProfile.MODERATE
        assert cfg.initial_weight == 1.0

    def test_conservative_factory(self):
        cfg = AgentConfig.conservative("con")
        assert cfg.risk_profile == RiskProfile.CONSERVATIVE
        assert cfg.max_position_pct == 0.05
        assert cfg.min_confidence == 0.75

    def test_moderate_factory(self):
        cfg = AgentConfig.moderate("mod")
        assert cfg.risk_profile == RiskProfile.MODERATE
        assert cfg.max_position_pct == 0.10
        assert cfg.min_confidence == 0.65

    def test_aggressive_factory(self):
        cfg = AgentConfig.aggressive("agg")
        assert cfg.risk_profile == RiskProfile.AGGRESSIVE
        assert cfg.max_position_pct == 0.20
        assert cfg.min_confidence == 0.55

    def test_invalid_weight_raises(self):
        with pytest.raises(ValueError, match="initial_weight"):
            AgentConfig(agent_id="x", risk_profile=RiskProfile.MODERATE, initial_weight=0)

    def test_invalid_position_pct_raises(self):
        with pytest.raises(ValueError, match="max_position_pct"):
            AgentConfig(agent_id="x", risk_profile=RiskProfile.MODERATE, max_position_pct=1.5)

    def test_invalid_confidence_raises(self):
        with pytest.raises(ValueError, match="min_confidence"):
            AgentConfig(agent_id="x", risk_profile=RiskProfile.MODERATE, min_confidence=-0.1)

    def test_risk_profiles_are_enum_strings(self):
        assert RiskProfile.CONSERVATIVE == "conservative"
        assert RiskProfile.MODERATE == "moderate"
        assert RiskProfile.AGGRESSIVE == "aggressive"


# ─── AgentSignal Tests ────────────────────────────────────────────────────────


class TestAgentSignal:
    def test_valid_buy_signal(self):
        sig = make_signal("a1", "buy")
        assert sig.action == "buy"
        assert sig.is_directional is True

    def test_valid_sell_signal(self):
        sig = make_signal("a1", "sell")
        assert sig.is_directional is True

    def test_hold_not_directional(self):
        sig = make_signal("a1", "hold")
        assert sig.is_directional is False

    def test_invalid_action_raises(self):
        with pytest.raises(ValueError, match="Invalid action"):
            AgentSignal(agent_id="a1", action="maybe", confidence=0.5, size_pct=0.05)

    def test_confidence_out_of_range_raises(self):
        with pytest.raises(ValueError, match="confidence"):
            AgentSignal(agent_id="a1", action="buy", confidence=1.5, size_pct=0.05)

    def test_size_pct_out_of_range_raises(self):
        with pytest.raises(ValueError, match="size_pct"):
            AgentSignal(agent_id="a1", action="buy", confidence=0.8, size_pct=-0.1)

    def test_timestamp_auto_set(self):
        sig = make_signal("a1", "buy")
        assert sig.timestamp > 0


# ─── AgentPerformance Tests ───────────────────────────────────────────────────


class TestAgentPerformance:
    def test_initial_state(self):
        perf = AgentPerformance(agent_id="a1")
        assert perf.wins == 0
        assert perf.losses == 0
        assert perf.win_rate == 0.0
        assert perf.avg_pnl == 0.0

    def test_record_win(self):
        perf = AgentPerformance(agent_id="a1")
        perf.record_trade(pnl=1.5)
        assert perf.wins == 1
        assert perf.losses == 0
        assert perf.total_trades == 1
        assert perf.total_pnl == pytest.approx(1.5)

    def test_record_loss(self):
        perf = AgentPerformance(agent_id="a1")
        perf.record_trade(pnl=-0.5)
        assert perf.losses == 1
        assert perf.wins == 0

    def test_win_rate_calculation(self):
        perf = AgentPerformance(agent_id="a1")
        perf.record_trade(1.0)
        perf.record_trade(1.0)
        perf.record_trade(-1.0)
        assert perf.win_rate == pytest.approx(2 / 3)

    def test_avg_pnl(self):
        perf = AgentPerformance(agent_id="a1")
        perf.record_trade(1.0)
        perf.record_trade(3.0)
        assert perf.avg_pnl == pytest.approx(2.0)

    def test_recent_pnl_rolling_window(self):
        perf = AgentPerformance(agent_id="a1", _max_recent=5)
        for i in range(7):
            perf.record_trade(float(i))
        assert len(perf.recent_pnl) == 5
        assert perf.recent_pnl == [2.0, 3.0, 4.0, 5.0, 6.0]

    def test_sharpe_proxy_insufficient_data(self):
        perf = AgentPerformance(agent_id="a1")
        perf.record_trade(1.0)
        assert perf.sharpe_proxy == 0.0

    def test_sharpe_proxy_positive(self):
        perf = AgentPerformance(agent_id="a1")
        for _ in range(5):
            perf.record_trade(1.0)
        perf.record_trade(2.0)
        assert perf.sharpe_proxy >= 0  # positive mean, low variance

    def test_to_dict_keys(self):
        perf = AgentPerformance(agent_id="a1")
        d = perf.to_dict()
        assert "agent_id" in d
        assert "weight" in d
        assert "win_rate" in d
        assert "total_pnl" in d


# ─── AgentPerformanceTracker Tests ───────────────────────────────────────────


class TestAgentPerformanceTracker:
    def test_register_agent(self):
        tracker = AgentPerformanceTracker()
        tracker.register_agent("a1", initial_weight=2.0)
        assert tracker.get_weight("a1") == 2.0

    def test_get_weight_unknown_agent(self):
        tracker = AgentPerformanceTracker()
        assert tracker.get_weight("unknown") == 1.0

    def test_record_outcome_auto_registers(self):
        tracker = AgentPerformanceTracker()
        tracker.record_outcome("new_agent", pnl=1.0)
        perf = tracker.get_performance("new_agent")
        assert perf is not None
        assert perf.total_trades == 1

    def test_weights_rebalanced_after_interval(self):
        tracker = AgentPerformanceTracker(update_interval=5)
        tracker.register_agent("winner", 1.0)
        tracker.register_agent("loser", 1.0)
        # Record 5 wins for winner, 5 losses for loser
        for _ in range(5):
            tracker.record_outcome("winner", 1.0)
        for _ in range(5):
            tracker.record_outcome("loser", -1.0)
        # After 10 total trades (2 intervals), weights should differ
        w_winner = tracker.get_weight("winner")
        w_loser = tracker.get_weight("loser")
        assert w_winner >= w_loser

    def test_weights_clipped_to_min(self):
        tracker = AgentPerformanceTracker(min_weight=0.5, update_interval=3)
        tracker.register_agent("bad", 1.0)
        for _ in range(3):
            tracker.record_outcome("bad", -100.0)
        assert tracker.get_weight("bad") >= 0.5

    def test_weights_clipped_to_max(self):
        tracker = AgentPerformanceTracker(max_weight=2.0, update_interval=3)
        tracker.register_agent("good", 1.0)
        for _ in range(3):
            tracker.record_outcome("good", 100.0)
        assert tracker.get_weight("good") <= 2.0

    def test_top_performers(self):
        tracker = AgentPerformanceTracker()
        tracker.register_agent("a1")
        tracker.register_agent("a2")
        for _ in range(5):
            tracker.record_outcome("a1", 1.0)
            tracker.record_outcome("a2", -1.0)
        top = tracker.top_performers(n=1)
        assert top[0].agent_id == "a1"

    def test_worst_performer(self):
        tracker = AgentPerformanceTracker()
        tracker.register_agent("a1")
        tracker.register_agent("a2")
        tracker.record_outcome("a1", 10.0)
        tracker.record_outcome("a2", -10.0)
        worst = tracker.worst_performer()
        assert worst.agent_id == "a2"

    def test_get_all_weights(self):
        tracker = AgentPerformanceTracker()
        tracker.register_agent("a1", 1.5)
        tracker.register_agent("a2", 0.8)
        weights = tracker.get_all_weights()
        assert weights["a1"] == 1.5
        assert weights["a2"] == 0.8


# ─── AgentPool Tests ──────────────────────────────────────────────────────────


class TestAgentPool:
    def test_add_and_count(self):
        pool = AgentPool()
        cfg = AgentConfig.moderate("m1")
        pool.add_agent(cfg, AsyncMock())
        assert pool.size == 1
        assert "m1" in pool.agent_ids

    def test_remove_existing(self):
        pool = AgentPool()
        cfg = AgentConfig.moderate("m1")
        pool.add_agent(cfg, AsyncMock())
        assert pool.remove_agent("m1") is True
        assert pool.size == 0

    def test_remove_nonexistent(self):
        pool = AgentPool()
        assert pool.remove_agent("ghost") is False

    def test_get_config(self):
        pool = AgentPool()
        cfg = AgentConfig.conservative("c1")
        pool.add_agent(cfg, AsyncMock())
        assert pool.get_config("c1") == cfg
        assert pool.get_config("unknown") is None

    @pytest.mark.asyncio
    async def test_collect_signals_all_respond(self):
        pool = AgentPool()
        for i in range(3):
            pool.add_agent(AgentConfig.moderate(f"a{i}"), make_strategist("buy"))
        signals = await pool.collect_signals(SNAPSHOT)
        assert len(signals) == 3

    @pytest.mark.asyncio
    async def test_collect_signals_respects_min_confidence(self):
        """Conservative agent with confidence below min should emit hold."""
        pool = AgentPool()
        cfg = AgentConfig.conservative("c1")  # min_confidence=0.75
        # Strategist returns confidence=0.50 (below threshold)
        pool.add_agent(cfg, make_strategist("buy", confidence=0.50))
        signals = await pool.collect_signals(SNAPSHOT)
        assert signals[0].action == "hold"

    @pytest.mark.asyncio
    async def test_collect_signals_clips_size_to_max_position(self):
        """Conservative agent should clip position to 5% max."""
        pool = AgentPool()
        cfg = AgentConfig.conservative("c1")  # max_position_pct=0.05
        pool.add_agent(cfg, make_strategist("buy", confidence=0.80, size_pct=0.30))
        signals = await pool.collect_signals(SNAPSHOT)
        assert signals[0].size_pct <= 0.05

    @pytest.mark.asyncio
    async def test_collect_signals_skips_failing_agents(self):
        """If one agent raises, others still vote."""
        pool = AgentPool()
        good = make_strategist("buy")
        bad = AsyncMock()
        bad.decide = AsyncMock(side_effect=RuntimeError("connection failed"))
        pool.add_agent(AgentConfig.moderate("good"), good)
        pool.add_agent(AgentConfig.moderate("bad"), bad)
        signals = await pool.collect_signals(SNAPSHOT)
        assert len(signals) == 1
        assert signals[0].agent_id == "good"


# ─── EnsembleSignal Tests ─────────────────────────────────────────────────────


class TestEnsembleSignal:
    def test_is_actionable_true(self):
        es = EnsembleSignal(
            action="buy", consensus_weight=0.8, mean_confidence=0.75,
            mean_size_pct=0.05, agent_votes={}, dissenting_agents=[],
            has_consensus=True,
        )
        assert es.is_actionable is True

    def test_is_actionable_false_hold(self):
        es = EnsembleSignal(
            action="hold", consensus_weight=0.9, mean_confidence=0.5,
            mean_size_pct=0.0, agent_votes={}, dissenting_agents=[],
            has_consensus=True,
        )
        assert es.is_actionable is False

    def test_is_actionable_false_no_consensus(self):
        es = EnsembleSignal(
            action="buy", consensus_weight=0.4, mean_confidence=0.7,
            mean_size_pct=0.05, agent_votes={}, dissenting_agents=[],
            has_consensus=False,
        )
        assert es.is_actionable is False


# ─── MultiAgentCoordinator Tests ─────────────────────────────────────────────


class TestMultiAgentCoordinator:
    @pytest.mark.asyncio
    async def test_unanimous_buy_consensus(self):
        pool = AgentPool()
        for i in range(3):
            pool.add_agent(AgentConfig.moderate(f"a{i}"), make_strategist("buy"))
        coord = MultiAgentCoordinator(pool, consensus_threshold=0.60)
        signal = await coord.get_ensemble_signal(SNAPSHOT)
        assert signal.action == "buy"
        assert signal.has_consensus is True
        assert signal.consensus_weight == pytest.approx(1.0)

    @pytest.mark.asyncio
    async def test_split_vote_below_threshold_gives_hold(self):
        """2 buy vs 1 sell = 67%, above 60% threshold → buy wins."""
        pool = AgentPool()
        pool.add_agent(AgentConfig.moderate("a1"), make_strategist("buy"))
        pool.add_agent(AgentConfig.moderate("a2"), make_strategist("buy"))
        pool.add_agent(AgentConfig.moderate("a3"), make_strategist("sell"))
        coord = MultiAgentCoordinator(pool, consensus_threshold=0.80)
        signal = await coord.get_ensemble_signal(SNAPSHOT)
        # 67% < 80% threshold → hold
        assert signal.action == "hold"
        assert signal.has_consensus is False

    @pytest.mark.asyncio
    async def test_split_vote_above_threshold_passes(self):
        """2 buy vs 1 sell = 67% ≥ 60% threshold → buy."""
        pool = AgentPool()
        pool.add_agent(AgentConfig.moderate("a1"), make_strategist("buy"))
        pool.add_agent(AgentConfig.moderate("a2"), make_strategist("buy"))
        pool.add_agent(AgentConfig.moderate("a3"), make_strategist("sell"))
        coord = MultiAgentCoordinator(pool, consensus_threshold=0.60)
        signal = await coord.get_ensemble_signal(SNAPSHOT)
        assert signal.action == "buy"
        assert signal.has_consensus is True

    @pytest.mark.asyncio
    async def test_no_quorum_gives_hold(self):
        """Pool size < min_agents_required → no quorum."""
        pool = AgentPool()
        pool.add_agent(AgentConfig.moderate("a1"), make_strategist("buy"))
        coord = MultiAgentCoordinator(pool, min_agents_required=2)
        signal = await coord.get_ensemble_signal(SNAPSHOT)
        assert signal.action == "hold"
        assert signal.has_consensus is False

    @pytest.mark.asyncio
    async def test_dissenting_agents_listed(self):
        pool = AgentPool()
        pool.add_agent(AgentConfig.moderate("buyer"), make_strategist("buy"))
        pool.add_agent(AgentConfig.moderate("buyer2"), make_strategist("buy"))
        pool.add_agent(AgentConfig.moderate("seller"), make_strategist("sell"))
        coord = MultiAgentCoordinator(pool, consensus_threshold=0.60)
        signal = await coord.get_ensemble_signal(SNAPSHOT)
        assert "seller" in signal.dissenting_agents
        assert "buyer" not in signal.dissenting_agents

    @pytest.mark.asyncio
    async def test_decision_count_increments(self):
        coord = build_coordinator()
        for i in range(3):
            await coord.get_ensemble_signal(SNAPSHOT)
        assert coord.decision_count == 3

    @pytest.mark.asyncio
    async def test_consensus_rate(self):
        pool = AgentPool()
        for i in range(3):
            pool.add_agent(AgentConfig.moderate(f"a{i}"), make_strategist("buy"))
        coord = MultiAgentCoordinator(pool, consensus_threshold=0.60)
        for _ in range(4):
            await coord.get_ensemble_signal(SNAPSHOT)
        assert coord.consensus_rate == pytest.approx(1.0)

    @pytest.mark.asyncio
    async def test_on_consensus_callback(self):
        calls = []
        pool = AgentPool()
        for i in range(2):
            pool.add_agent(AgentConfig.moderate(f"a{i}"), make_strategist("buy"))
        coord = MultiAgentCoordinator(
            pool, consensus_threshold=0.60, on_consensus=calls.append
        )
        await coord.get_ensemble_signal(SNAPSHOT)
        assert len(calls) == 1

    @pytest.mark.asyncio
    async def test_performance_summary_keys(self):
        coord = build_coordinator()
        await coord.get_ensemble_signal(SNAPSHOT)
        summary = coord.performance_summary()
        assert "pool_size" in summary
        assert "decision_count" in summary
        assert "consensus_rate" in summary
        assert "agent_performance" in summary

    def test_record_trade_outcome(self):
        coord = build_coordinator()
        coord.record_trade_outcome("agent-mod", 1.5)
        perf = coord.pool.tracker.get_performance("agent-mod")
        assert perf.total_trades == 1

    @pytest.mark.asyncio
    async def test_record_outcome_for_all_agreeing(self):
        pool = AgentPool()
        pool.add_agent(AgentConfig.moderate("buyer1"), make_strategist("buy"))
        pool.add_agent(AgentConfig.moderate("buyer2"), make_strategist("buy"))
        pool.add_agent(AgentConfig.moderate("seller"), make_strategist("sell"))
        coord = MultiAgentCoordinator(pool, consensus_threshold=0.60)
        ensemble = await coord.get_ensemble_signal(SNAPSHOT)
        coord.record_outcome_for_all_agreeing(ensemble, pnl=2.0)
        p1 = coord.pool.tracker.get_performance("buyer1")
        p2 = coord.pool.tracker.get_performance("buyer2")
        assert p1.total_pnl == pytest.approx(2.0)
        assert p2.total_pnl == pytest.approx(2.0)

    @pytest.mark.asyncio
    async def test_weighted_vote_favors_high_weight_agent(self):
        """High-weight agent's vote should carry more influence."""
        pool = AgentPool()
        tracker = pool.tracker
        pool.add_agent(AgentConfig.moderate("heavy"), make_strategist("sell"))
        pool.add_agent(AgentConfig.moderate("light1"), make_strategist("buy"))
        pool.add_agent(AgentConfig.moderate("light2"), make_strategist("buy"))
        # Give heavy 10x weight by directly setting the record weight
        tracker._records["heavy"].weight = 10.0
        tracker._records["light1"].weight = 1.0
        tracker._records["light2"].weight = 1.0
        coord = MultiAgentCoordinator(pool, consensus_threshold=0.60)
        signal = await coord.get_ensemble_signal(SNAPSHOT)
        # heavy(10) > light1(1)+light2(1), so sell should win
        assert signal.action == "sell"


# ─── Build Helpers Tests ──────────────────────────────────────────────────────


class TestBuildHelpers:
    def test_build_default_pool(self):
        pool = build_default_pool()
        assert pool.size == 3
        assert "agent-con" in pool.agent_ids
        assert "agent-mod" in pool.agent_ids
        assert "agent-agg" in pool.agent_ids

    def test_build_coordinator_returns_coordinator(self):
        coord = build_coordinator()
        assert isinstance(coord, MultiAgentCoordinator)
        assert coord.pool.size == 3

    def test_build_coordinator_with_custom_threshold(self):
        coord = build_coordinator(consensus_threshold=0.75)
        assert coord.consensus_threshold == 0.75

    @pytest.mark.asyncio
    async def test_build_default_pool_with_factory(self):
        """Custom factory should be called for each agent."""
        calls = []

        def factory(agent_id, config):
            calls.append(agent_id)
            return make_strategist("hold")

        pool = build_default_pool(mock_strategist_factory=factory)
        assert len(calls) == 3
        signals = await pool.collect_signals(SNAPSHOT)
        assert len(signals) == 3


# ─── Extended AgentPerformance Tests ──────────────────────────────────────────


class TestAgentPerformanceExtended:
    def test_zero_trades_win_rate_is_zero(self):
        perf = AgentPerformance(agent_id="x")
        assert perf.win_rate == 0.0

    def test_all_wins(self):
        perf = AgentPerformance(agent_id="x")
        for _ in range(10):
            perf.record_trade(1.0)
        assert perf.win_rate == pytest.approx(1.0)

    def test_all_losses(self):
        perf = AgentPerformance(agent_id="x")
        for _ in range(10):
            perf.record_trade(-1.0)
        assert perf.win_rate == pytest.approx(0.0)
        assert perf.losses == 10

    def test_breakeven_trade_not_counted_as_win_or_loss(self):
        perf = AgentPerformance(agent_id="x")
        perf.record_trade(0.0)  # breakeven
        assert perf.wins == 0
        assert perf.losses == 0
        assert perf.total_trades == 1

    def test_recent_avg_pnl_empty(self):
        perf = AgentPerformance(agent_id="x")
        assert perf.recent_avg_pnl == 0.0

    def test_to_dict_roundtrips(self):
        perf = AgentPerformance(agent_id="x", weight=2.5)
        perf.record_trade(3.0)
        perf.record_trade(-1.0)
        d = perf.to_dict()
        assert d["agent_id"] == "x"
        assert d["weight"] == pytest.approx(2.5)
        assert d["total_trades"] == 2
        assert d["wins"] == 1
        assert d["losses"] == 1


# ─── Extended Tracker Tests ────────────────────────────────────────────────────


class TestAgentPerformanceTrackerExtended:
    def test_get_all_performance_empty(self):
        tracker = AgentPerformanceTracker()
        assert tracker.get_all_performance() == []

    def test_worst_performer_empty_returns_none(self):
        tracker = AgentPerformanceTracker()
        assert tracker.worst_performer() is None

    def test_top_performers_empty(self):
        tracker = AgentPerformanceTracker()
        result = tracker.top_performers()
        assert result == []

    def test_top_performers_returns_sorted(self):
        tracker = AgentPerformanceTracker()
        tracker.register_agent("a1")
        tracker.register_agent("a2")
        tracker.register_agent("a3")
        # a1 best, a3 worst
        tracker.record_outcome("a1", 10.0)
        tracker.record_outcome("a2", 1.0)
        tracker.record_outcome("a3", -10.0)
        top = tracker.top_performers(n=2)
        assert top[0].agent_id == "a1"

    def test_multiple_agents_rebalance_independently(self):
        tracker = AgentPerformanceTracker(update_interval=3)
        tracker.register_agent("a1")
        tracker.register_agent("a2")
        for _ in range(3):
            tracker.record_outcome("a1", 2.0)   # consistent wins
            tracker.record_outcome("a2", -2.0)  # consistent losses
        w1 = tracker.get_weight("a1")
        w2 = tracker.get_weight("a2")
        assert w1 > w2

    def test_register_agent_idempotent(self):
        tracker = AgentPerformanceTracker()
        tracker.register_agent("a1", 2.0)
        tracker.register_agent("a1", 5.0)  # should not overwrite
        assert tracker.get_weight("a1") == 2.0


# ─── Extended AgentPool Tests ─────────────────────────────────────────────────


class TestAgentPoolExtended:
    def test_size_zero_initially(self):
        pool = AgentPool()
        assert pool.size == 0

    def test_agent_ids_initially_empty(self):
        pool = AgentPool()
        assert pool.agent_ids == []

    def test_add_multiple_agents(self):
        pool = AgentPool()
        for i in range(5):
            pool.add_agent(AgentConfig.moderate(f"a{i}"), AsyncMock())
        assert pool.size == 5

    @pytest.mark.asyncio
    async def test_empty_pool_collect_signals_returns_empty(self):
        pool = AgentPool()
        signals = await pool.collect_signals(SNAPSHOT)
        assert signals == []

    @pytest.mark.asyncio
    async def test_aggressive_agent_larger_position(self):
        """Aggressive agent should allow larger position than conservative."""
        pool_agg = AgentPool()
        pool_con = AgentPool()
        pool_agg.add_agent(AgentConfig.aggressive("agg"), make_strategist("buy", size_pct=0.30))
        pool_con.add_agent(AgentConfig.conservative("con"), make_strategist("buy", size_pct=0.30))
        sigs_agg = await pool_agg.collect_signals(SNAPSHOT)
        sigs_con = await pool_con.collect_signals(SNAPSHOT)
        assert sigs_agg[0].size_pct > sigs_con[0].size_pct


# ─── Extended Coordinator Tests ────────────────────────────────────────────────


class TestMultiAgentCoordinatorExtended:
    @pytest.mark.asyncio
    async def test_all_hold_votes_gives_hold(self):
        pool = AgentPool()
        for i in range(3):
            pool.add_agent(AgentConfig.moderate(f"a{i}"), make_strategist("hold"))
        coord = MultiAgentCoordinator(pool, consensus_threshold=0.60)
        signal = await coord.get_ensemble_signal(SNAPSHOT)
        assert signal.action == "hold"

    @pytest.mark.asyncio
    async def test_ensemble_signal_not_actionable_when_hold(self):
        pool = AgentPool()
        for i in range(3):
            pool.add_agent(AgentConfig.moderate(f"a{i}"), make_strategist("hold"))
        coord = MultiAgentCoordinator(pool, consensus_threshold=0.60)
        signal = await coord.get_ensemble_signal(SNAPSHOT)
        assert signal.is_actionable is False

    @pytest.mark.asyncio
    async def test_agent_votes_all_recorded(self):
        pool = AgentPool()
        pool.add_agent(AgentConfig.moderate("a1"), make_strategist("buy"))
        pool.add_agent(AgentConfig.moderate("a2"), make_strategist("sell"))
        pool.add_agent(AgentConfig.moderate("a3"), make_strategist("hold"))
        coord = MultiAgentCoordinator(pool, consensus_threshold=0.99)
        signal = await coord.get_ensemble_signal(SNAPSHOT)
        assert len(signal.agent_votes) == 3
        assert "a1" in signal.agent_votes
        assert "a2" in signal.agent_votes
        assert "a3" in signal.agent_votes

    @pytest.mark.asyncio
    async def test_coordinator_min_agents_configurable(self):
        pool = AgentPool()
        pool.add_agent(AgentConfig.moderate("solo"), make_strategist("buy"))
        coord = MultiAgentCoordinator(pool, min_agents_required=1, consensus_threshold=0.60)
        signal = await coord.get_ensemble_signal(SNAPSHOT)
        # With min=1 and 1 agent responding, should reach consensus
        assert signal.action == "buy"
        assert signal.has_consensus is True

    @pytest.mark.asyncio
    async def test_on_consensus_not_called_when_no_consensus(self):
        calls = []
        pool = AgentPool()
        pool.add_agent(AgentConfig.moderate("a1"), make_strategist("buy"))
        pool.add_agent(AgentConfig.moderate("a2"), make_strategist("sell"))
        coord = MultiAgentCoordinator(
            pool, consensus_threshold=0.99, on_consensus=calls.append
        )
        await coord.get_ensemble_signal(SNAPSHOT)
        assert len(calls) == 0

    @pytest.mark.asyncio
    async def test_record_outcome_for_dissenting_agents(self):
        """Dissenting agents get positive credit when consensus loses (pnl < 0)."""
        pool = AgentPool()
        pool.add_agent(AgentConfig.moderate("buyer1"), make_strategist("buy"))
        pool.add_agent(AgentConfig.moderate("buyer2"), make_strategist("buy"))
        pool.add_agent(AgentConfig.moderate("seller"), make_strategist("sell"))
        coord = MultiAgentCoordinator(pool, consensus_threshold=0.60)
        ensemble = await coord.get_ensemble_signal(SNAPSHOT)
        # Consensus is buy; record a losing trade — dissenter (seller) gets credit
        coord.record_outcome_for_all_agreeing(ensemble, pnl=-10.0)
        seller_perf = coord.pool.tracker.get_performance("seller")
        # seller voted against consensus buy, was right (pnl<0) → gets +5.0
        assert seller_perf.total_pnl == pytest.approx(5.0)

    @pytest.mark.asyncio
    async def test_performance_summary_after_multiple_decisions(self):
        coord = build_coordinator()
        for _ in range(5):
            await coord.get_ensemble_signal(SNAPSHOT)
        summary = coord.performance_summary()
        assert summary["decision_count"] == 5
        assert summary["pool_size"] == 3
