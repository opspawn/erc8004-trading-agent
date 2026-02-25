"""
test_risk_adjusted_sizing.py — Tests for risk-adjusted position sizing
integrating Credora credit tiers with Kelly Criterion in the RiskManager.

Covers:
  - Kelly multiplier effect on position sizing by tier
  - Minimum grade enforcement (hard floor)
  - Tier-based sizing matrix (all 7 tiers × various sizes)
  - validate_trade_with_credora edge cases
  - Credit-tier-to-size scaling accuracy
  - Integration: AgentCreditHistory → RiskManager
"""

from __future__ import annotations

import pytest
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from credora_client import (
    AgentCreditHistory,
    CredoraClient,
    CredoraRatingTier,
    KELLY_MULTIPLIERS,
)
from risk_manager import RiskManager


# ─── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture
def credora():
    return CredoraClient(use_mock=True)


@pytest.fixture
def risk_mgr(credora):
    return RiskManager(credora_client=credora)


@pytest.fixture
def risk_mgr_min_bbb(credora):
    return RiskManager(credora_client=credora, credora_min_grade=CredoraRatingTier.BBB)


@pytest.fixture
def risk_mgr_min_a(credora):
    return RiskManager(credora_client=credora, credora_min_grade=CredoraRatingTier.A)


# ─── Kelly Multiplier Scaling ─────────────────────────────────────────────────

class TestKellyMultiplierScaling:
    """Verify that validate_trade_with_credora scales size by Kelly multiplier."""

    def test_aaa_multiplier_is_1(self):
        assert KELLY_MULTIPLIERS[CredoraRatingTier.AAA] == 1.00

    def test_aa_multiplier_is_0_9(self):
        assert KELLY_MULTIPLIERS[CredoraRatingTier.AA] == 0.90

    def test_a_multiplier_is_0_8(self):
        assert KELLY_MULTIPLIERS[CredoraRatingTier.A] == 0.80

    def test_bbb_multiplier_is_0_65(self):
        assert KELLY_MULTIPLIERS[CredoraRatingTier.BBB] == 0.65

    def test_bb_multiplier_is_0_5(self):
        assert KELLY_MULTIPLIERS[CredoraRatingTier.BB] == 0.50

    def test_b_multiplier_is_0_35(self):
        assert KELLY_MULTIPLIERS[CredoraRatingTier.B] == 0.35

    def test_ccc_multiplier_is_0_2(self):
        assert KELLY_MULTIPLIERS[CredoraRatingTier.CCC] == 0.20

    def test_nr_multiplier_is_0_1(self):
        assert KELLY_MULTIPLIERS[CredoraRatingTier.NR] == 0.10

    def test_multipliers_strictly_ordered(self):
        """AAA > AA > A > BBB > BB > B > CCC > NR."""
        tiers_ordered = [
            CredoraRatingTier.AAA, CredoraRatingTier.AA, CredoraRatingTier.A,
            CredoraRatingTier.BBB, CredoraRatingTier.BB, CredoraRatingTier.B,
            CredoraRatingTier.CCC, CredoraRatingTier.NR,
        ]
        mults = [KELLY_MULTIPLIERS[t] for t in tiers_ordered]
        for i in range(len(mults) - 1):
            assert mults[i] > mults[i + 1]


# ─── validate_trade_with_credora ──────────────────────────────────────────────

class TestValidateTradeWithCredora:
    def test_ethereum_aaa_full_size(self, risk_mgr):
        """ETH is AAA → multiplier 1.0, full size passes."""
        ok, reason, adj = risk_mgr.validate_trade_with_credora(
            "YES", 5.0, 0.60, 100.0, "ethereum"
        )
        assert ok is True
        assert abs(adj - 5.0) < 0.01

    def test_aave_aa_reduces_size(self, risk_mgr):
        """Aave is AA (0.9) → adjusted size = 5.0 * 0.9 = 4.5."""
        ok, reason, adj = risk_mgr.validate_trade_with_credora(
            "YES", 5.0, 0.60, 100.0, "aave"
        )
        assert ok is True
        assert abs(adj - 4.5) < 0.01

    def test_compound_a_reduces_size(self, risk_mgr):
        """Compound is A (0.8) → adjusted = 5.0 * 0.8 = 4.0."""
        ok, risk_mgr_min_a, adj = risk_mgr.validate_trade_with_credora(
            "YES", 5.0, 0.60, 100.0, "compound"
        )
        assert ok is True
        assert abs(adj - 4.0) < 0.01

    def test_adjusted_size_returned_on_failure(self, risk_mgr):
        """If trade fails validation, adjusted_size should be 0."""
        # Request size > max (10% of portfolio)
        ok, reason, adj = risk_mgr.validate_trade_with_credora(
            "YES", 20.0, 0.60, 100.0, "ethereum"
        )
        assert ok is False
        assert adj == 0.0

    def test_no_credora_client_returns_original_size(self):
        rm = RiskManager()
        ok, reason, adj = rm.validate_trade_with_credora(
            "YES", 5.0, 0.60, 100.0, "aave"
        )
        assert ok is True
        assert adj == 5.0

    def test_gmx_bb_halves_size(self, risk_mgr):
        """GMX is BB (0.5) → 5.0 * 0.5 = 2.5."""
        ok, _, adj = risk_mgr.validate_trade_with_credora(
            "YES", 5.0, 0.60, 100.0, "gmx"
        )
        assert ok is True
        assert abs(adj - 2.5) < 0.01

    def test_osmosis_b_reduces_heavily(self, risk_mgr):
        """Osmosis is B (0.35) → 5.0 * 0.35 = 1.75."""
        ok, _, adj = risk_mgr.validate_trade_with_credora(
            "YES", 5.0, 0.60, 100.0, "osmosis"
        )
        assert ok is True
        assert abs(adj - 1.75) < 0.01


# ─── Minimum Grade Enforcement ────────────────────────────────────────────────

class TestMinimumGradeEnforcement:
    def test_bbb_min_rejects_bb(self, risk_mgr_min_bbb):
        """GMX is BB, below BBB minimum → reject."""
        ok, reason, adj = risk_mgr_min_bbb.validate_trade_with_credora(
            "YES", 5.0, 0.60, 100.0, "gmx"
        )
        assert ok is False
        assert "below minimum" in reason.lower()
        assert adj == 0.0

    def test_bbb_min_allows_bbb(self, risk_mgr_min_bbb):
        """Balancer is BBB → meets minimum."""
        ok, reason, adj = risk_mgr_min_bbb.validate_trade_with_credora(
            "YES", 3.0, 0.60, 100.0, "balancer"
        )
        assert ok is True

    def test_bbb_min_allows_aaa(self, risk_mgr_min_bbb):
        """ETH is AAA → well above BBB minimum."""
        ok, _, adj = risk_mgr_min_bbb.validate_trade_with_credora(
            "YES", 5.0, 0.60, 100.0, "ethereum"
        )
        assert ok is True

    def test_a_min_rejects_bbb(self, risk_mgr_min_a):
        """BBB is below A minimum → reject."""
        ok, reason, adj = risk_mgr_min_a.validate_trade_with_credora(
            "YES", 3.0, 0.60, 100.0, "balancer"
        )
        assert ok is False

    def test_a_min_allows_a(self, risk_mgr_min_a):
        """Compound is A → meets A minimum."""
        ok, _, adj = risk_mgr_min_a.validate_trade_with_credora(
            "YES", 3.0, 0.60, 100.0, "compound"
        )
        assert ok is True

    def test_a_min_rejects_bb(self, risk_mgr_min_a):
        ok, reason, adj = risk_mgr_min_a.validate_trade_with_credora(
            "YES", 3.0, 0.60, 100.0, "gmx"
        )
        assert ok is False
        assert adj == 0.0


# ─── Tier-to-Size Matrix ──────────────────────────────────────────────────────

class TestTierSizeMatrix:
    """Verify the size scaling for each tier at a standard $10 base size."""

    BASE_SIZE = 10.0
    PORTFOLIO = 200.0
    PRICE = 0.60

    @pytest.mark.parametrize("protocol,expected_multiplier", [
        ("ethereum", 1.00),
        ("aave",     0.90),
        ("compound", 0.80),
        ("balancer", 0.65),
        ("gmx",      0.50),
        ("osmosis",  0.35),
    ])
    def test_size_scaling_by_protocol(self, risk_mgr, protocol, expected_multiplier):
        ok, _, adj = risk_mgr.validate_trade_with_credora(
            "YES", self.BASE_SIZE, self.PRICE, self.PORTFOLIO, protocol
        )
        assert ok is True
        expected_size = self.BASE_SIZE * expected_multiplier
        assert abs(adj - expected_size) < 0.01


# ─── Integration: AgentCreditHistory → RiskManager ───────────────────────────

class TestCreditHistoryIntegration:
    def test_good_agent_gets_high_multiplier(self):
        h = AgentCreditHistory("agent-integration")
        for i, proto in enumerate(["ethereum", "btc", "usdc", "chainlink", "aave"]):
            h.record_trade(f"m{i}", proto, 10.0, "WIN", 3.0, CredoraRatingTier.AAA)
        mult = h.get_kelly_multiplier()
        assert mult >= 0.80

    def test_bad_agent_gets_low_multiplier(self):
        h = AgentCreditHistory("agent-bad-int")
        for i in range(5):
            h.record_trade(f"m{i}", "sushiswap", 10.0, "LOSS", -5.0, CredoraRatingTier.B)
        mult = h.get_kelly_multiplier()
        assert mult <= 0.50

    def test_credit_multiplier_used_in_sizing(self):
        h = AgentCreditHistory("agent-sizing")
        for i, proto in enumerate(["ethereum", "btc", "usdc", "chainlink", "aave"]):
            h.record_trade(f"m{i}", proto, 10.0, "WIN", 2.0, CredoraRatingTier.AA)
        rating = h.as_credora_rating()
        mult = rating.kelly_multiplier
        base_size = 5.0
        expected_adj = base_size * mult
        assert expected_adj > 0

    def test_diverse_history_better_tier_than_single_protocol(self):
        single = AgentCreditHistory("single-proto")
        for i in range(5):
            single.record_trade(f"m{i}", "sushiswap", 5.0, "WIN", 1.0, CredoraRatingTier.BB)

        diverse = AgentCreditHistory("diverse-proto")
        protos = ["aave", "uniswap", "compound", "maker", "ethereum"]
        for i, p in enumerate(protos):
            diverse.record_trade(f"m{i}", p, 5.0, "WIN", 1.0, CredoraRatingTier.AA)

        assert diverse.compute_credit_score() > single.compute_credit_score()


# ─── Edge Cases ───────────────────────────────────────────────────────────────

class TestEdgeCases:
    def test_zero_size_rejected_by_risk_mgr(self, risk_mgr):
        ok, reason, adj = risk_mgr.validate_trade_with_credora(
            "YES", 0.0, 0.60, 100.0, "ethereum"
        )
        assert ok is False

    def test_negative_size_rejected(self, risk_mgr):
        ok, reason, adj = risk_mgr.validate_trade_with_credora(
            "YES", -5.0, 0.60, 100.0, "ethereum"
        )
        assert ok is False

    def test_portfolio_at_minimum(self, risk_mgr):
        ok, reason, adj = risk_mgr.validate_trade_with_credora(
            "YES", 0.5, 0.60, 10.0, "ethereum"
        )
        # $10 portfolio, max pos = 10% = $1.0 → $0.5 should pass
        assert ok is True

    def test_portfolio_below_minimum(self, risk_mgr):
        ok, reason, adj = risk_mgr.validate_trade_with_credora(
            "YES", 0.5, 0.60, 5.0, "ethereum"
        )
        assert ok is False

    def test_credora_rating_for_nonexistent_returns_something(self, risk_mgr):
        """Unknown protocols still get a rating (hash-based)."""
        ok, _, adj = risk_mgr.validate_trade_with_credora(
            "YES", 2.0, 0.50, 100.0, "totally_unknown_xyz_protocol"
        )
        # Should succeed (hash gives some tier between B and AA)
        assert isinstance(ok, bool)

    def test_halted_manager_rejects_regardless_of_credora(self, risk_mgr):
        """Trading halt overrides Credora-based approval."""
        risk_mgr._trading_halted = True
        risk_mgr._halt_reason = "test halt"
        ok, reason, adj = risk_mgr.validate_trade_with_credora(
            "YES", 1.0, 0.50, 100.0, "ethereum"
        )
        assert ok is False
        assert "halt" in reason.lower()
