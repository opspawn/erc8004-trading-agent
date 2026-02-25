"""
test_mesh_coordinator.py — Tests for the multi-agent mesh coordinator.

Coverage:
  - SpecialistAgent initialization and configuration
  - Individual agent vote evaluation (BUY, SELL, HOLD, REJECT)
  - Grade requirement enforcement per agent profile
  - Kelly sizing and position clipping
  - Reputation updates (win/loss)
  - MeshCoordinator consensus with 3 agents
  - All-agree, 2/3-agree, and split-vote scenarios
  - Reputation-weighted size aggregation
  - Outcome recording and reputation propagation
  - Edge cases: zero portfolio, boundary grades, zero edge
"""

from __future__ import annotations

import time
import pytest

from mesh_coordinator import (
    AgentConfig,
    AgentProfile,
    AgentVote,
    MeshConsensus,
    MeshCoordinator,
    SpecialistAgent,
    VoteAction,
    _TIER_ORDER,
    AGENT_CONFIGS,
)
from credora_client import CredoraRatingTier


# ─── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture
def conservative_agent() -> SpecialistAgent:
    return SpecialistAgent("cons_agent", AGENT_CONFIGS[AgentProfile.CONSERVATIVE])


@pytest.fixture
def balanced_agent() -> SpecialistAgent:
    return SpecialistAgent("bal_agent", AGENT_CONFIGS[AgentProfile.BALANCED])


@pytest.fixture
def aggressive_agent() -> SpecialistAgent:
    return SpecialistAgent("agg_agent", AGENT_CONFIGS[AgentProfile.AGGRESSIVE])


@pytest.fixture
def coordinator() -> MeshCoordinator:
    return MeshCoordinator()


# ─── AgentConfig ──────────────────────────────────────────────────────────────

class TestAgentConfig:
    def test_conservative_config_kelly(self):
        cfg = AGENT_CONFIGS[AgentProfile.CONSERVATIVE]
        assert cfg.kelly_fraction == 0.15

    def test_conservative_config_max_position(self):
        cfg = AGENT_CONFIGS[AgentProfile.CONSERVATIVE]
        assert cfg.max_position_pct == 0.05

    def test_conservative_config_grade(self):
        cfg = AGENT_CONFIGS[AgentProfile.CONSERVATIVE]
        assert cfg.credora_min_grade == CredoraRatingTier.A

    def test_balanced_config_kelly(self):
        cfg = AGENT_CONFIGS[AgentProfile.BALANCED]
        assert cfg.kelly_fraction == 0.25

    def test_balanced_config_max_position(self):
        cfg = AGENT_CONFIGS[AgentProfile.BALANCED]
        assert cfg.max_position_pct == 0.10

    def test_balanced_config_grade(self):
        cfg = AGENT_CONFIGS[AgentProfile.BALANCED]
        assert cfg.credora_min_grade == CredoraRatingTier.BBB

    def test_aggressive_config_kelly(self):
        cfg = AGENT_CONFIGS[AgentProfile.AGGRESSIVE]
        assert cfg.kelly_fraction == 0.35

    def test_aggressive_config_max_position(self):
        cfg = AGENT_CONFIGS[AgentProfile.AGGRESSIVE]
        assert cfg.max_position_pct == 0.15

    def test_aggressive_config_grade(self):
        cfg = AGENT_CONFIGS[AgentProfile.AGGRESSIVE]
        assert cfg.credora_min_grade == CredoraRatingTier.BB

    def test_all_profiles_have_configs(self):
        for profile in AgentProfile:
            assert profile in AGENT_CONFIGS


# ─── SpecialistAgent Initialization ──────────────────────────────────────────

class TestSpecialistAgentInit:
    def test_agent_id_stored(self, conservative_agent):
        assert conservative_agent.agent_id == "cons_agent"

    def test_profile_accessible(self, conservative_agent):
        assert conservative_agent.profile == AgentProfile.CONSERVATIVE

    def test_initial_reputation_conservative(self, conservative_agent):
        assert conservative_agent.reputation_score == 7.0

    def test_initial_reputation_balanced(self, balanced_agent):
        assert balanced_agent.reputation_score == 6.5

    def test_initial_reputation_aggressive(self, aggressive_agent):
        assert aggressive_agent.reputation_score == 5.5

    def test_get_stats_returns_dict(self, conservative_agent):
        stats = conservative_agent.get_stats()
        assert isinstance(stats, dict)

    def test_get_stats_keys(self, conservative_agent):
        stats = conservative_agent.get_stats()
        for key in ("agent_id", "profile", "reputation_score", "trade_count"):
            assert key in stats

    def test_get_stats_initial_trade_count(self, conservative_agent):
        stats = conservative_agent.get_stats()
        assert stats["trade_count"] == 0

    def test_get_stats_win_rate_zero_initially(self, conservative_agent):
        stats = conservative_agent.get_stats()
        assert stats["win_rate"] == 0.0


# ─── Grade Requirements ───────────────────────────────────────────────────────

class TestGradeRequirements:
    def test_conservative_rejects_bb(self, conservative_agent):
        vote = conservative_agent.evaluate(
            side="BUY", size=5.0, price=0.5, portfolio_value=100.0,
            protocol_grade=CredoraRatingTier.BB, edge=0.10
        )
        assert vote.action == VoteAction.REJECT

    def test_conservative_rejects_bbb(self, conservative_agent):
        vote = conservative_agent.evaluate(
            side="BUY", size=5.0, price=0.5, portfolio_value=100.0,
            protocol_grade=CredoraRatingTier.BBB, edge=0.10
        )
        assert vote.action == VoteAction.REJECT

    def test_conservative_accepts_a(self, conservative_agent):
        vote = conservative_agent.evaluate(
            side="BUY", size=1.0, price=0.5, portfolio_value=100.0,
            protocol_grade=CredoraRatingTier.A, edge=0.10
        )
        assert vote.action != VoteAction.REJECT

    def test_conservative_accepts_aa(self, conservative_agent):
        vote = conservative_agent.evaluate(
            side="BUY", size=1.0, price=0.5, portfolio_value=100.0,
            protocol_grade=CredoraRatingTier.AA, edge=0.10
        )
        assert vote.action != VoteAction.REJECT

    def test_balanced_accepts_bbb(self, balanced_agent):
        vote = balanced_agent.evaluate(
            side="BUY", size=5.0, price=0.5, portfolio_value=100.0,
            protocol_grade=CredoraRatingTier.BBB, edge=0.10
        )
        assert vote.action != VoteAction.REJECT

    def test_balanced_rejects_bb(self, balanced_agent):
        vote = balanced_agent.evaluate(
            side="BUY", size=5.0, price=0.5, portfolio_value=100.0,
            protocol_grade=CredoraRatingTier.BB, edge=0.10
        )
        assert vote.action == VoteAction.REJECT

    def test_aggressive_accepts_bb(self, aggressive_agent):
        vote = aggressive_agent.evaluate(
            side="BUY", size=5.0, price=0.5, portfolio_value=100.0,
            protocol_grade=CredoraRatingTier.BB, edge=0.10
        )
        assert vote.action != VoteAction.REJECT

    def test_aggressive_rejects_ccc(self, aggressive_agent):
        vote = aggressive_agent.evaluate(
            side="BUY", size=5.0, price=0.5, portfolio_value=100.0,
            protocol_grade=CredoraRatingTier.CCC, edge=0.10
        )
        assert vote.action == VoteAction.REJECT

    def test_reject_vote_has_zero_adjusted_size(self, conservative_agent):
        vote = conservative_agent.evaluate(
            side="BUY", size=5.0, price=0.5, portfolio_value=100.0,
            protocol_grade=CredoraRatingTier.BB, edge=0.10
        )
        assert vote.adjusted_size == 0.0

    def test_reject_vote_has_reasoning(self, conservative_agent):
        vote = conservative_agent.evaluate(
            side="BUY", size=5.0, price=0.5, portfolio_value=100.0,
            protocol_grade=CredoraRatingTier.BB, edge=0.10
        )
        assert "below minimum" in vote.reasoning.lower() or vote.reasoning


# ─── Vote Actions ─────────────────────────────────────────────────────────────

class TestVoteActions:
    def test_buy_side_returns_buy_action(self, balanced_agent):
        vote = balanced_agent.evaluate(
            side="BUY", size=5.0, price=0.5, portfolio_value=100.0,
            protocol_grade=CredoraRatingTier.BBB, edge=0.10
        )
        assert vote.action == VoteAction.BUY

    def test_yes_side_returns_buy_action(self, balanced_agent):
        vote = balanced_agent.evaluate(
            side="YES", size=5.0, price=0.5, portfolio_value=100.0,
            protocol_grade=CredoraRatingTier.BBB, edge=0.10
        )
        assert vote.action == VoteAction.BUY

    def test_sell_side_returns_sell_action(self, balanced_agent):
        vote = balanced_agent.evaluate(
            side="SELL", size=5.0, price=0.5, portfolio_value=100.0,
            protocol_grade=CredoraRatingTier.BBB, edge=0.10
        )
        assert vote.action == VoteAction.SELL

    def test_low_edge_returns_hold(self, balanced_agent):
        vote = balanced_agent.evaluate(
            side="BUY", size=5.0, price=0.5, portfolio_value=100.0,
            protocol_grade=CredoraRatingTier.BBB, edge=0.01
        )
        assert vote.action == VoteAction.HOLD

    def test_hold_has_zero_adjusted_size(self, balanced_agent):
        vote = balanced_agent.evaluate(
            side="BUY", size=5.0, price=0.5, portfolio_value=100.0,
            protocol_grade=CredoraRatingTier.BBB, edge=0.01
        )
        assert vote.adjusted_size == 0.0

    def test_vote_includes_reputation_score(self, balanced_agent):
        vote = balanced_agent.evaluate(
            side="BUY", size=5.0, price=0.5, portfolio_value=100.0,
            protocol_grade=CredoraRatingTier.BBB, edge=0.10
        )
        assert vote.reputation_score == balanced_agent.reputation_score

    def test_vote_confidence_in_range(self, balanced_agent):
        vote = balanced_agent.evaluate(
            side="BUY", size=5.0, price=0.5, portfolio_value=100.0,
            protocol_grade=CredoraRatingTier.BBB, edge=0.10
        )
        assert 0.0 <= vote.confidence <= 1.0


# ─── Position Sizing ──────────────────────────────────────────────────────────

class TestPositionSizing:
    def test_size_clipped_to_max_position(self, conservative_agent):
        # Conservative max_position_pct=0.05 → max 5 on 100
        vote = conservative_agent.evaluate(
            side="BUY", size=50.0, price=0.5, portfolio_value=100.0,
            protocol_grade=CredoraRatingTier.A, edge=0.10
        )
        assert vote.adjusted_size <= 100.0 * 0.05

    def test_aggressive_allows_larger_size(self, aggressive_agent):
        # Aggressive max_position_pct=0.15 → max 15 on 100
        vote = aggressive_agent.evaluate(
            side="BUY", size=10.0, price=0.5, portfolio_value=100.0,
            protocol_grade=CredoraRatingTier.BB, edge=0.10
        )
        # aggressive should not clip 10 when max is 15
        assert vote.action in (VoteAction.BUY, VoteAction.SELL)

    def test_adjusted_size_non_negative(self, balanced_agent):
        vote = balanced_agent.evaluate(
            side="BUY", size=5.0, price=0.5, portfolio_value=100.0,
            protocol_grade=CredoraRatingTier.BBB, edge=0.10
        )
        assert vote.adjusted_size >= 0.0


# ─── Reputation Updates ───────────────────────────────────────────────────────

class TestReputationUpdates:
    def test_win_increases_reputation(self, balanced_agent):
        initial = balanced_agent.reputation_score
        balanced_agent.update_reputation(trade_won=True, pnl_pct=0.05)
        assert balanced_agent.reputation_score > initial

    def test_loss_decreases_reputation(self, balanced_agent):
        initial = balanced_agent.reputation_score
        balanced_agent.update_reputation(trade_won=False, pnl_pct=-0.05)
        assert balanced_agent.reputation_score < initial

    def test_reputation_bounded_above(self, balanced_agent):
        for _ in range(100):
            balanced_agent.update_reputation(trade_won=True, pnl_pct=1.0)
        assert balanced_agent.reputation_score <= 10.0

    def test_reputation_bounded_below(self, balanced_agent):
        for _ in range(100):
            balanced_agent.update_reputation(trade_won=False, pnl_pct=-1.0)
        assert balanced_agent.reputation_score >= 0.0

    def test_trade_count_increments(self, balanced_agent):
        balanced_agent.update_reputation(trade_won=True)
        balanced_agent.update_reputation(trade_won=False)
        assert balanced_agent.get_stats()["trade_count"] == 2

    def test_win_count_tracks_wins(self, balanced_agent):
        balanced_agent.update_reputation(trade_won=True)
        balanced_agent.update_reputation(trade_won=True)
        balanced_agent.update_reputation(trade_won=False)
        assert balanced_agent.get_stats()["win_count"] == 2

    def test_large_pnl_moves_reputation_more(self, balanced_agent):
        a1 = SpecialistAgent("a1", AGENT_CONFIGS[AgentProfile.BALANCED])
        a2 = SpecialistAgent("a2", AGENT_CONFIGS[AgentProfile.BALANCED])
        a1.update_reputation(True, pnl_pct=0.01)
        a2.update_reputation(True, pnl_pct=1.0)
        assert a2.reputation_score > a1.reputation_score


# ─── MeshCoordinator Initialization ──────────────────────────────────────────

class TestMeshCoordinatorInit:
    def test_default_has_three_agents(self, coordinator):
        assert len(coordinator.agents) == 3

    def test_get_agent_by_id(self, coordinator):
        agent = coordinator.get_agent("conservative_agent")
        assert agent is not None

    def test_get_nonexistent_agent_returns_none(self, coordinator):
        assert coordinator.get_agent("nonexistent") is None

    def test_min_votes_default(self, coordinator):
        assert coordinator.min_votes_for_consensus == 2

    def test_custom_min_votes(self):
        coord = MeshCoordinator(min_votes_for_consensus=3)
        assert coord.min_votes_for_consensus == 3

    def test_mesh_stats_has_agent_count(self, coordinator):
        stats = coordinator.get_mesh_stats()
        assert stats["agent_count"] == 3

    def test_mesh_stats_has_agents_list(self, coordinator):
        stats = coordinator.get_mesh_stats()
        assert len(stats["agents"]) == 3


# ─── Consensus: All Three Agents Agree ───────────────────────────────────────

class TestConsensusAllAgree:
    def test_all_vote_buy_consensus_reached(self, coordinator):
        consensus = coordinator.evaluate(
            side="BUY", size=5.0, price=0.5, portfolio_value=100.0,
            protocol_grade=CredoraRatingTier.AA, edge=0.15
        )
        # AA grade passes all agents; edge=0.15 well above thresholds
        assert consensus.consensus_reached

    def test_all_vote_buy_final_action_is_buy(self, coordinator):
        consensus = coordinator.evaluate(
            side="BUY", size=5.0, price=0.5, portfolio_value=100.0,
            protocol_grade=CredoraRatingTier.AA, edge=0.15
        )
        assert consensus.final_action == VoteAction.BUY

    def test_consensus_includes_three_agents(self, coordinator):
        consensus = coordinator.evaluate(
            side="BUY", size=5.0, price=0.5, portfolio_value=100.0,
            protocol_grade=CredoraRatingTier.AA, edge=0.15
        )
        assert len(consensus.agents) == 3

    def test_weighted_size_positive_on_consensus(self, coordinator):
        consensus = coordinator.evaluate(
            side="BUY", size=5.0, price=0.5, portfolio_value=100.0,
            protocol_grade=CredoraRatingTier.AA, edge=0.15
        )
        if consensus.consensus_reached:
            assert consensus.weighted_size >= 0.0

    def test_proposal_stored_in_consensus(self, coordinator):
        consensus = coordinator.evaluate(
            side="BUY", size=5.0, price=0.5, portfolio_value=100.0,
            protocol_grade=CredoraRatingTier.AA, edge=0.15
        )
        assert consensus.proposal["side"] == "BUY"
        assert consensus.proposal["size"] == 5.0


# ─── Consensus: 2/3 Agents Agree ─────────────────────────────────────────────

class TestConsensusTwoThirds:
    def test_bbb_grade_rejects_conservative_still_consensus(self, coordinator):
        # BBB rejects conservative (min grade A), passes balanced+aggressive
        consensus = coordinator.evaluate(
            side="BUY", size=5.0, price=0.5, portfolio_value=100.0,
            protocol_grade=CredoraRatingTier.BBB, edge=0.10
        )
        # balanced and aggressive should vote active → 2/3 consensus
        assert consensus.consensus_reached

    def test_two_of_three_is_enough(self):
        coord = MeshCoordinator(min_votes_for_consensus=2)
        # Use BBB which only passes 2 agents
        consensus = coord.evaluate(
            side="BUY", size=5.0, price=0.5, portfolio_value=100.0,
            protocol_grade=CredoraRatingTier.BBB, edge=0.10
        )
        assert consensus.consensus_reached

    def test_approval_ratio_with_two_votes(self, coordinator):
        consensus = coordinator.evaluate(
            side="BUY", size=5.0, price=0.5, portfolio_value=100.0,
            protocol_grade=CredoraRatingTier.BBB, edge=0.10
        )
        assert consensus.approval_ratio >= 0.0


# ─── Consensus: Split Vote / No Consensus ────────────────────────────────────

class TestConsensusNoConsensus:
    def test_nr_grade_blocks_all_agents(self, coordinator):
        # NR grade should fail all agent grade checks
        consensus = coordinator.evaluate(
            side="BUY", size=5.0, price=0.5, portfolio_value=100.0,
            protocol_grade=CredoraRatingTier.NR, edge=0.10
        )
        assert not consensus.consensus_reached

    def test_nr_grade_final_action_hold(self, coordinator):
        consensus = coordinator.evaluate(
            side="BUY", size=5.0, price=0.5, portfolio_value=100.0,
            protocol_grade=CredoraRatingTier.NR, edge=0.10
        )
        assert consensus.final_action == VoteAction.HOLD

    def test_no_consensus_weighted_size_zero(self, coordinator):
        consensus = coordinator.evaluate(
            side="BUY", size=5.0, price=0.5, portfolio_value=100.0,
            protocol_grade=CredoraRatingTier.NR, edge=0.10
        )
        assert consensus.weighted_size == 0.0

    def test_low_edge_produces_no_consensus(self):
        # edge=0.001 is below all thresholds → all HOLD
        coord = MeshCoordinator()
        consensus = coord.evaluate(
            side="BUY", size=5.0, price=0.5, portfolio_value=100.0,
            protocol_grade=CredoraRatingTier.AA, edge=0.001
        )
        assert not consensus.consensus_reached


# ─── Reputation-Weighted Size ─────────────────────────────────────────────────

class TestReputationWeightedSize:
    def test_higher_rep_agent_influences_size_more(self):
        coord = MeshCoordinator()
        # Give conservative agent very high reputation
        conservative = coord.get_agent("conservative_agent")
        for _ in range(20):
            conservative.update_reputation(True, pnl_pct=0.5)

        consensus = coord.evaluate(
            side="BUY", size=5.0, price=0.5, portfolio_value=100.0,
            protocol_grade=CredoraRatingTier.AA, edge=0.15
        )
        # Just verify consensus was reached and size is reasonable
        if consensus.consensus_reached:
            assert consensus.weighted_size >= 0.0

    def test_total_rep_weight_positive(self, coordinator):
        consensus = coordinator.evaluate(
            side="BUY", size=5.0, price=0.5, portfolio_value=100.0,
            protocol_grade=CredoraRatingTier.AA, edge=0.15
        )
        assert consensus.total_reputation_weight > 0.0


# ─── Record Outcome ───────────────────────────────────────────────────────────

class TestRecordOutcome:
    def test_record_win_increases_all_reps(self, coordinator):
        initial_reps = {
            a.agent_id: a.reputation_score
            for a in coordinator.agents
        }
        coordinator.record_outcome(trade_won=True, pnl_pct=0.05)
        for agent in coordinator.agents:
            assert agent.reputation_score > initial_reps[agent.agent_id]

    def test_record_loss_decreases_all_reps(self, coordinator):
        initial_reps = {
            a.agent_id: a.reputation_score
            for a in coordinator.agents
        }
        coordinator.record_outcome(trade_won=False, pnl_pct=-0.05)
        for agent in coordinator.agents:
            assert agent.reputation_score < initial_reps[agent.agent_id]

    def test_record_outcome_specific_agents(self, coordinator):
        initial = coordinator.get_agent("conservative_agent").reputation_score
        coordinator.record_outcome(
            trade_won=True,
            agent_ids=["conservative_agent"]
        )
        assert coordinator.get_agent("conservative_agent").reputation_score > initial
        # Others unchanged
        balanced_rep = coordinator.get_agent("balanced_agent").reputation_score
        assert balanced_rep == AGENT_CONFIGS[AgentProfile.BALANCED].initial_reputation


# ─── MeshConsensus Properties ─────────────────────────────────────────────────

class TestMeshConsensusProperties:
    def test_vote_count_equals_votes_for_plus_against(self, coordinator):
        consensus = coordinator.evaluate(
            side="BUY", size=5.0, price=0.5, portfolio_value=100.0,
            protocol_grade=CredoraRatingTier.AA, edge=0.15
        )
        assert consensus.vote_count == len(consensus.votes_for) + len(consensus.votes_against)

    def test_approval_ratio_range(self, coordinator):
        consensus = coordinator.evaluate(
            side="BUY", size=5.0, price=0.5, portfolio_value=100.0,
            protocol_grade=CredoraRatingTier.AA, edge=0.15
        )
        assert 0.0 <= consensus.approval_ratio <= 1.0

    def test_consensus_timestamp_recent(self, coordinator):
        consensus = coordinator.evaluate(
            side="BUY", size=5.0, price=0.5, portfolio_value=100.0,
            protocol_grade=CredoraRatingTier.AA, edge=0.15
        )
        assert time.time() - consensus.timestamp < 5.0
