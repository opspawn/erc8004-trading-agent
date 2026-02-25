"""
test_credora_credit_history.py — Tests for AgentCreditHistory and
build_credora_api_response in credora_client.py.

Covers:
  - Credit history recording and retrieval
  - Credit score computation components
  - Tier mapping from score
  - Kelly multiplier from history
  - API response shape
  - Edge cases and boundary conditions
"""

from __future__ import annotations

import time
import math
import pytest

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from credora_client import (
    AgentCreditHistory,
    TradeRecord,
    CredoraRatingTier,
    CredoraRating,
    KELLY_MULTIPLIERS,
    INVESTMENT_GRADE,
    _TIER_SCORE_MID,
    build_credora_api_response,
)


# ─── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture
def empty_history():
    return AgentCreditHistory("agent-001")


@pytest.fixture
def good_history():
    """Agent with a solid track record."""
    h = AgentCreditHistory("agent-good")
    protocols = ["ethereum", "aave", "uniswap", "chainlink", "maker"]
    for i, proto in enumerate(protocols):
        h.record_trade(
            market_id=f"market-{i}",
            protocol=proto,
            size_usdc=10.0,
            outcome="WIN",
            pnl_usdc=2.5,
            credora_tier=CredoraRatingTier.AA,
        )
    return h


@pytest.fixture
def bad_history():
    """Agent with a poor track record."""
    h = AgentCreditHistory("agent-bad")
    for i in range(6):
        h.record_trade(
            market_id=f"bad-market-{i}",
            protocol="sushiswap",
            size_usdc=10.0,
            outcome="LOSS",
            pnl_usdc=-3.5,
            credora_tier=CredoraRatingTier.B,
        )
    return h


@pytest.fixture
def mixed_history():
    """Agent with 60% win rate and mixed protocols."""
    h = AgentCreditHistory("agent-mixed")
    outcomes = ["WIN", "WIN", "WIN", "LOSS", "WIN", "LOSS", "WIN", "LOSS", "WIN", "WIN"]
    protocols = ["aave", "uniswap", "compound", "gmx", "maker",
                 "balancer", "curve", "dydx", "ethereum", "wbtc"]
    pnls = [2.0, 1.5, 3.0, -2.0, 1.0, -1.5, 2.5, -1.0, 1.8, 2.2]
    tiers = [CredoraRatingTier.AA, CredoraRatingTier.AA, CredoraRatingTier.A,
             CredoraRatingTier.BB, CredoraRatingTier.AA, CredoraRatingTier.BBB,
             CredoraRatingTier.A, CredoraRatingTier.BB, CredoraRatingTier.AAA,
             CredoraRatingTier.AA]
    for i, (o, p, pnl, t) in enumerate(zip(outcomes, protocols, pnls, tiers)):
        h.record_trade(f"m-{i}", p, 10.0, o, pnl, t)
    return h


# ─── Initialization ───────────────────────────────────────────────────────────

class TestAgentCreditHistoryInit:
    def test_init_creates_empty_history(self, empty_history):
        assert empty_history.total_trades == 0

    def test_init_agent_id(self, empty_history):
        assert empty_history.agent_id == "agent-001"

    def test_init_settled_trades_empty(self, empty_history):
        assert empty_history.settled_trades == []

    def test_init_win_rate_default(self, empty_history):
        assert empty_history.win_rate == 0.5

    def test_init_total_pnl_zero(self, empty_history):
        assert empty_history.total_pnl == 0.0

    def test_init_protocol_diversity_zero(self, empty_history):
        assert empty_history.protocol_diversity == 0

    def test_default_credit_score_midrange(self, empty_history):
        score = empty_history.compute_credit_score()
        # With defaults: 50% win rate = 20 pts, 0 avg pnl = 15 pts, 0 diversity = 0,
        # avg tier = BBB = 70/100*15 = 10.5 → total ~45.5
        assert 0 <= score <= 100


# ─── Trade Recording ──────────────────────────────────────────────────────────

class TestTradeRecording:
    def test_record_win(self, empty_history):
        rec = empty_history.record_trade("m1", "aave", 10.0, "WIN", 2.0, CredoraRatingTier.AA)
        assert rec.outcome == "WIN"
        assert rec.pnl_usdc == 2.0

    def test_record_loss(self, empty_history):
        rec = empty_history.record_trade("m2", "aave", 5.0, "LOSS", -1.5, CredoraRatingTier.A)
        assert rec.outcome == "LOSS"
        assert rec.pnl_usdc == -1.5

    def test_record_pending(self, empty_history):
        rec = empty_history.record_trade("m3", "uniswap", 3.0, "PENDING", 0.0, CredoraRatingTier.AA)
        assert rec.outcome == "PENDING"

    def test_record_increments_total(self, empty_history):
        empty_history.record_trade("m1", "aave", 10.0, "WIN", 1.0, CredoraRatingTier.AA)
        empty_history.record_trade("m2", "uniswap", 5.0, "LOSS", -1.0, CredoraRatingTier.A)
        assert empty_history.total_trades == 2

    def test_invalid_outcome_raises(self, empty_history):
        with pytest.raises(ValueError, match="Invalid outcome"):
            empty_history.record_trade("m1", "aave", 10.0, "DRAW", 0.0)

    def test_record_returns_trade_record(self, empty_history):
        rec = empty_history.record_trade("m1", "aave", 10.0, "WIN", 2.0, CredoraRatingTier.AA)
        assert isinstance(rec, TradeRecord)

    def test_size_stored_as_abs(self, empty_history):
        rec = empty_history.record_trade("m1", "aave", -5.0, "WIN", 1.0, CredoraRatingTier.AA)
        assert rec.size_usdc == 5.0

    def test_timestamp_set(self, empty_history):
        before = time.time()
        rec = empty_history.record_trade("m1", "aave", 10.0, "WIN", 2.0, CredoraRatingTier.AA)
        after = time.time()
        assert before <= rec.timestamp <= after

    def test_trade_record_is_win_property(self, empty_history):
        rec = empty_history.record_trade("m1", "aave", 10.0, "WIN", 2.0, CredoraRatingTier.AA)
        assert rec.is_win is True
        assert rec.is_loss is False

    def test_trade_record_is_loss_property(self, empty_history):
        rec = empty_history.record_trade("m2", "aave", 5.0, "LOSS", -1.0, CredoraRatingTier.A)
        assert rec.is_loss is True
        assert rec.is_win is False


# ─── Update Outcome ───────────────────────────────────────────────────────────

class TestUpdateOutcome:
    def test_update_pending_to_win(self, empty_history):
        empty_history.record_trade("m1", "aave", 10.0, "PENDING", 0.0, CredoraRatingTier.AA)
        result = empty_history.update_outcome("m1", "WIN", 2.5)
        assert result is True
        assert empty_history._trades[0].outcome == "WIN"

    def test_update_pending_to_loss(self, empty_history):
        empty_history.record_trade("m1", "aave", 10.0, "PENDING", 0.0, CredoraRatingTier.AA)
        empty_history.update_outcome("m1", "LOSS", -2.0)
        assert empty_history._trades[0].pnl_usdc == -2.0

    def test_update_nonexistent_returns_false(self, empty_history):
        result = empty_history.update_outcome("nonexistent", "WIN", 1.0)
        assert result is False

    def test_update_settled_trade_not_changed(self, empty_history):
        empty_history.record_trade("m1", "aave", 10.0, "WIN", 2.0, CredoraRatingTier.AA)
        result = empty_history.update_outcome("m1", "LOSS", -1.0)
        assert result is False  # only updates PENDING


# ─── Settled Trades ───────────────────────────────────────────────────────────

class TestSettledTrades:
    def test_pending_excluded(self, empty_history):
        empty_history.record_trade("m1", "aave", 10.0, "PENDING", 0.0, CredoraRatingTier.AA)
        empty_history.record_trade("m2", "uniswap", 5.0, "WIN", 2.0, CredoraRatingTier.AA)
        assert len(empty_history.settled_trades) == 1

    def test_win_and_loss_included(self, empty_history):
        empty_history.record_trade("m1", "aave", 10.0, "WIN", 2.0, CredoraRatingTier.AA)
        empty_history.record_trade("m2", "uniswap", 5.0, "LOSS", -1.0, CredoraRatingTier.A)
        assert len(empty_history.settled_trades) == 2


# ─── Win Rate ─────────────────────────────────────────────────────────────────

class TestWinRate:
    def test_all_wins(self, good_history):
        assert good_history.win_rate == 1.0

    def test_all_losses(self, bad_history):
        assert bad_history.win_rate == 0.0

    def test_mixed_win_rate(self, mixed_history):
        assert abs(mixed_history.win_rate - 0.7) < 0.01

    def test_pending_excluded_from_win_rate(self, empty_history):
        empty_history.record_trade("m1", "aave", 10.0, "WIN", 2.0, CredoraRatingTier.AA)
        empty_history.record_trade("m2", "uniswap", 5.0, "PENDING", 0.0, CredoraRatingTier.AA)
        assert empty_history.win_rate == 1.0


# ─── Protocol Diversity ───────────────────────────────────────────────────────

class TestProtocolDiversity:
    def test_single_protocol(self, bad_history):
        assert bad_history.protocol_diversity == 1

    def test_five_protocols(self, good_history):
        assert good_history.protocol_diversity == 5

    def test_case_insensitive(self, empty_history):
        empty_history.record_trade("m1", "AAVE", 10.0, "WIN", 2.0, CredoraRatingTier.AA)
        empty_history.record_trade("m2", "aave", 5.0, "WIN", 1.0, CredoraRatingTier.AA)
        assert empty_history.protocol_diversity == 1


# ─── Credit Score ─────────────────────────────────────────────────────────────

class TestCreditScore:
    def test_good_history_high_score(self, good_history):
        score = good_history.compute_credit_score()
        assert score >= 70.0

    def test_bad_history_low_score(self, bad_history):
        score = bad_history.compute_credit_score()
        assert score <= 45.0

    def test_score_in_range(self, mixed_history):
        score = mixed_history.compute_credit_score()
        assert 0.0 <= score <= 100.0

    def test_score_is_float(self, empty_history):
        assert isinstance(empty_history.compute_credit_score(), float)

    def test_perfect_history_near_max(self):
        h = AgentCreditHistory("perfect")
        for i, proto in enumerate(["ethereum", "btc", "usdc", "chainlink", "aave"]):
            h.record_trade(f"m{i}", proto, 5.0, "WIN", 5.0, CredoraRatingTier.AAA)
        score = h.compute_credit_score()
        assert score >= 90.0


# ─── Credit Tier ──────────────────────────────────────────────────────────────

class TestCreditTier:
    def test_good_history_investment_grade(self, good_history):
        tier = good_history.get_credit_tier()
        assert tier in INVESTMENT_GRADE

    def test_bad_history_speculative_grade(self, bad_history):
        tier = bad_history.get_credit_tier()
        assert tier not in INVESTMENT_GRADE

    def test_tier_is_credora_tier(self, mixed_history):
        tier = mixed_history.get_credit_tier()
        assert isinstance(tier, CredoraRatingTier)

    def test_kelly_multiplier_matches_tier(self, mixed_history):
        tier = mixed_history.get_credit_tier()
        expected = KELLY_MULTIPLIERS[tier]
        assert mixed_history.get_kelly_multiplier() == expected


# ─── As CredoraRating ─────────────────────────────────────────────────────────

class TestAsCredoraRating:
    def test_returns_credora_rating(self, good_history):
        rating = good_history.as_credora_rating()
        assert isinstance(rating, CredoraRating)

    def test_source_is_credit_history(self, good_history):
        rating = good_history.as_credora_rating()
        assert rating.source == "credit_history"

    def test_protocol_contains_agent_id(self, good_history):
        rating = good_history.as_credora_rating()
        assert "agent-good" in rating.protocol

    def test_score_matches_credit_score(self, mixed_history):
        rating = mixed_history.as_credora_rating()
        assert rating.score == mixed_history.compute_credit_score()


# ─── Summary ──────────────────────────────────────────────────────────────────

class TestSummary:
    def test_summary_keys(self, mixed_history):
        s = mixed_history.get_summary()
        required = ["agent_id", "total_trades", "settled_trades", "win_rate",
                    "total_pnl_usdc", "credit_score", "credit_tier", "kelly_multiplier"]
        for key in required:
            assert key in s

    def test_summary_agent_id(self, mixed_history):
        assert mixed_history.get_summary()["agent_id"] == "agent-mixed"

    def test_summary_total_trades(self, mixed_history):
        assert mixed_history.get_summary()["total_trades"] == 10


# ─── Recent Trades ────────────────────────────────────────────────────────────

class TestRecentTrades:
    def test_returns_last_n(self, mixed_history):
        recent = mixed_history.recent_trades(3)
        assert len(recent) == 3

    def test_returns_all_if_less_than_n(self, empty_history):
        empty_history.record_trade("m1", "aave", 10.0, "WIN", 2.0, CredoraRatingTier.AA)
        assert len(empty_history.recent_trades(10)) == 1


# ─── Trades By Protocol ───────────────────────────────────────────────────────

class TestTradesByProtocol:
    def test_finds_by_protocol(self, mixed_history):
        trades = mixed_history.trades_by_protocol("aave")
        assert len(trades) == 1

    def test_case_insensitive_lookup(self, mixed_history):
        trades_lower = mixed_history.trades_by_protocol("aave")
        trades_upper = mixed_history.trades_by_protocol("AAVE")
        assert len(trades_lower) == len(trades_upper)

    def test_unknown_protocol_empty(self, mixed_history):
        assert mixed_history.trades_by_protocol("nonexistent_xyz") == []


# ─── Clear ────────────────────────────────────────────────────────────────────

class TestClear:
    def test_clear_removes_all(self, mixed_history):
        mixed_history.clear()
        assert mixed_history.total_trades == 0

    def test_clear_resets_pnl(self, mixed_history):
        mixed_history.clear()
        assert mixed_history.total_pnl == 0.0


# ─── API Response Shape ───────────────────────────────────────────────────────

class TestCredoraApiResponseShape:
    @pytest.mark.parametrize("tier", list(CredoraRatingTier))
    def test_build_response_all_tiers(self, tier):
        resp = build_credora_api_response("TestProtocol", tier)
        assert resp["rating"] == tier.value

    def test_response_has_required_keys(self):
        resp = build_credora_api_response("Aave", CredoraRatingTier.AA)
        required = ["protocol", "rating", "score", "outlook", "is_investment_grade",
                    "kelly_multiplier", "sub_scores", "methodology", "valid_until", "data_sources"]
        for key in required:
            assert key in resp

    def test_investment_grade_flag_aaa(self):
        resp = build_credora_api_response("Ethereum", CredoraRatingTier.AAA)
        assert resp["is_investment_grade"] is True

    def test_speculative_grade_flag_bb(self):
        resp = build_credora_api_response("GMX", CredoraRatingTier.BB)
        assert resp["is_investment_grade"] is False

    def test_kelly_multiplier_matches(self):
        for tier in CredoraRatingTier:
            resp = build_credora_api_response("Test", tier)
            assert resp["kelly_multiplier"] == KELLY_MULTIPLIERS[tier]

    def test_sub_scores_has_four_keys(self):
        resp = build_credora_api_response("Aave", CredoraRatingTier.A)
        assert set(resp["sub_scores"].keys()) == {"liquidity", "smart_contract", "governance", "market"}

    def test_valid_until_in_future(self):
        resp = build_credora_api_response("Compound", CredoraRatingTier.A)
        assert resp["valid_until"] > time.time()

    def test_stable_outlook_for_investment_grade(self):
        for tier in [CredoraRatingTier.AAA, CredoraRatingTier.AA,
                     CredoraRatingTier.A, CredoraRatingTier.BBB]:
            resp = build_credora_api_response("Proto", tier)
            assert resp["outlook"] == "Stable"

    def test_negative_outlook_for_speculative(self):
        for tier in [CredoraRatingTier.BB, CredoraRatingTier.B]:
            resp = build_credora_api_response("Proto", tier)
            assert resp["outlook"] == "Negative"
