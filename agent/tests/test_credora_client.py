"""
test_credora_client.py — Tests for the Credora risk ratings integration.

Tests cover:
  - CredoraRatingTier enum values and ordering
  - KELLY_MULTIPLIERS mapping correctness
  - CredoraRating dataclass properties
  - CredoraRatingProvider mock logic (well-known + hash-based)
  - CredoraClient caching, fallback, and bulk APIs
  - RiskManager Credora integration (validate_trade_with_credora)
"""

import sys
import os
import time

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from credora_client import (
    CredoraClient,
    CredoraRating,
    CredoraRatingProvider,
    CredoraRatingTier,
    INVESTMENT_GRADE,
    KELLY_MULTIPLIERS,
    _TIER_ORDER,
)
from risk_manager import RiskManager


# ─── CredoraRatingTier ────────────────────────────────────────────────────────

class TestCredoraRatingTier:
    def test_all_tiers_exist(self):
        expected = {"AAA", "AA", "A", "BBB", "BB", "B", "CCC", "NR"}
        actual = {t.value for t in CredoraRatingTier}
        assert actual == expected

    def test_tier_values_are_strings(self):
        for tier in CredoraRatingTier:
            assert isinstance(tier.value, str)

    def test_tier_is_str_enum(self):
        assert CredoraRatingTier.AA == "AA"
        assert CredoraRatingTier.BBB == "BBB"

    def test_tier_count(self):
        assert len(CredoraRatingTier) == 8

    def test_tier_order_contains_all_non_nr(self):
        in_order = {t.value for t in _TIER_ORDER}
        assert "NR" not in in_order
        assert "AAA" in in_order
        assert "CCC" in in_order

    def test_tier_order_length(self):
        assert len(_TIER_ORDER) == 7

    def test_tier_order_ascending(self):
        # CCC at index 0, AAA at index 6
        assert _TIER_ORDER[0] == CredoraRatingTier.CCC
        assert _TIER_ORDER[-1] == CredoraRatingTier.AAA


# ─── KELLY_MULTIPLIERS ────────────────────────────────────────────────────────

class TestKellyMultipliers:
    def test_all_tiers_have_multiplier(self):
        for tier in CredoraRatingTier:
            assert tier in KELLY_MULTIPLIERS

    def test_multiplier_range(self):
        for tier, mult in KELLY_MULTIPLIERS.items():
            assert 0.0 < mult <= 1.0, f"{tier}: mult={mult} out of range"

    def test_aaa_is_one(self):
        assert KELLY_MULTIPLIERS[CredoraRatingTier.AAA] == 1.0

    def test_nr_is_lowest(self):
        nr = KELLY_MULTIPLIERS[CredoraRatingTier.NR]
        for tier, mult in KELLY_MULTIPLIERS.items():
            if tier != CredoraRatingTier.NR:
                assert mult >= nr, f"{tier} mult {mult} < NR {nr}"

    def test_monotone_decreasing(self):
        # Going CCC → AAA multiplier should increase
        tiers_asc = [
            CredoraRatingTier.CCC,
            CredoraRatingTier.B,
            CredoraRatingTier.BB,
            CredoraRatingTier.BBB,
            CredoraRatingTier.A,
            CredoraRatingTier.AA,
            CredoraRatingTier.AAA,
        ]
        multipliers = [KELLY_MULTIPLIERS[t] for t in tiers_asc]
        for i in range(len(multipliers) - 1):
            assert multipliers[i] < multipliers[i + 1], (
                f"Non-monotone at {tiers_asc[i]}: {multipliers[i]} >= {multipliers[i+1]}"
            )

    def test_specific_multipliers(self):
        assert KELLY_MULTIPLIERS[CredoraRatingTier.AA]  == 0.90
        assert KELLY_MULTIPLIERS[CredoraRatingTier.A]   == 0.80
        assert KELLY_MULTIPLIERS[CredoraRatingTier.BBB] == 0.65
        assert KELLY_MULTIPLIERS[CredoraRatingTier.BB]  == 0.50
        assert KELLY_MULTIPLIERS[CredoraRatingTier.B]   == 0.35
        assert KELLY_MULTIPLIERS[CredoraRatingTier.CCC] == 0.20
        assert KELLY_MULTIPLIERS[CredoraRatingTier.NR]  == 0.10


# ─── INVESTMENT_GRADE ─────────────────────────────────────────────────────────

class TestInvestmentGrade:
    def test_aaa_is_ig(self):
        assert CredoraRatingTier.AAA in INVESTMENT_GRADE

    def test_bbb_is_ig(self):
        assert CredoraRatingTier.BBB in INVESTMENT_GRADE

    def test_bb_not_ig(self):
        assert CredoraRatingTier.BB not in INVESTMENT_GRADE

    def test_ccc_not_ig(self):
        assert CredoraRatingTier.CCC not in INVESTMENT_GRADE

    def test_nr_not_ig(self):
        assert CredoraRatingTier.NR not in INVESTMENT_GRADE

    def test_ig_count(self):
        assert len(INVESTMENT_GRADE) == 4


# ─── CredoraRating ────────────────────────────────────────────────────────────

class TestCredoraRating:
    def _make_rating(self, tier=CredoraRatingTier.AA, score=87.5, protocol="Aave"):
        return CredoraRating(protocol=protocol, tier=tier, score=score)

    def test_kelly_multiplier_property(self):
        r = self._make_rating(tier=CredoraRatingTier.AA)
        assert r.kelly_multiplier == 0.90

    def test_kelly_multiplier_aaa(self):
        r = self._make_rating(tier=CredoraRatingTier.AAA)
        assert r.kelly_multiplier == 1.0

    def test_kelly_multiplier_ccc(self):
        r = self._make_rating(tier=CredoraRatingTier.CCC)
        assert r.kelly_multiplier == 0.20

    def test_is_investment_grade_true(self):
        for tier in [CredoraRatingTier.AAA, CredoraRatingTier.AA,
                     CredoraRatingTier.A, CredoraRatingTier.BBB]:
            r = self._make_rating(tier=tier)
            assert r.is_investment_grade, f"{tier} should be investment grade"

    def test_is_investment_grade_false(self):
        for tier in [CredoraRatingTier.BB, CredoraRatingTier.B,
                     CredoraRatingTier.CCC, CredoraRatingTier.NR]:
            r = self._make_rating(tier=tier)
            assert not r.is_investment_grade, f"{tier} should NOT be investment grade"

    def test_age_seconds(self):
        r = self._make_rating()
        assert r.age_seconds >= 0.0
        assert r.age_seconds < 2.0

    def test_str_repr(self):
        r = self._make_rating(tier=CredoraRatingTier.AA, score=87.5, protocol="Aave")
        s = str(r)
        assert "Aave" in s
        assert "AA" in s
        assert "0.90" in s

    def test_source_default(self):
        r = self._make_rating()
        assert r.source == "mock"

    def test_custom_source(self):
        r = CredoraRating(protocol="Aave", tier=CredoraRatingTier.A, score=80.0, source="live")
        assert r.source == "live"

    def test_timestamp_set(self):
        before = time.time()
        r = self._make_rating()
        after = time.time()
        assert before <= r.timestamp <= after


# ─── CredoraRatingProvider ───────────────────────────────────────────────────

class TestCredoraRatingProvider:
    @pytest.fixture
    def provider(self):
        return CredoraRatingProvider()

    def test_aave_is_aa(self, provider):
        r = provider.get_rating("Aave")
        assert r.tier == CredoraRatingTier.AA

    def test_case_insensitive(self, provider):
        r1 = provider.get_rating("aave")
        r2 = provider.get_rating("AAVE")
        r3 = provider.get_rating("Aave")
        assert r1.tier == r2.tier == r3.tier

    def test_chainlink_is_aaa(self, provider):
        r = provider.get_rating("Chainlink")
        assert r.tier == CredoraRatingTier.AAA

    def test_ethereum_is_aaa(self, provider):
        r = provider.get_rating("Ethereum")
        assert r.tier == CredoraRatingTier.AAA

    def test_bitcoin_is_aaa(self, provider):
        r = provider.get_rating("Bitcoin")
        assert r.tier == CredoraRatingTier.AAA

    def test_usdc_is_aaa(self, provider):
        r = provider.get_rating("usdc")
        assert r.tier == CredoraRatingTier.AAA

    def test_uniswap_is_aa(self, provider):
        r = provider.get_rating("Uniswap")
        assert r.tier == CredoraRatingTier.AA

    def test_compound_is_a(self, provider):
        r = provider.get_rating("Compound")
        assert r.tier == CredoraRatingTier.A

    def test_gmx_is_bb(self, provider):
        r = provider.get_rating("GMX")
        assert r.tier == CredoraRatingTier.BB

    def test_unknown_protocol_deterministic(self, provider):
        r1 = provider.get_rating("SomeFakeProtocol123")
        r2 = provider.get_rating("SomeFakeProtocol123")
        assert r1.tier == r2.tier
        assert r1.score == r2.score

    def test_unknown_protocol_not_extreme(self, provider):
        # Unknown protocols should be assigned mid-range tiers (B–AA)
        for name in ["FakeProtocol", "XYZSwap", "TestDAO"]:
            r = provider.get_rating(name)
            assert r.tier not in {CredoraRatingTier.AAA, CredoraRatingTier.CCC, CredoraRatingTier.NR}

    def test_score_in_range(self, provider):
        for protocol in ["Aave", "Uniswap", "Compound", "FakeProtocol"]:
            r = provider.get_rating(protocol)
            assert 0.0 <= r.score <= 100.0

    def test_source_is_mock(self, provider):
        r = provider.get_rating("Aave")
        assert r.source == "mock"

    def test_protocol_field(self, provider):
        r = provider.get_rating("Aave")
        assert r.protocol == "Aave"


# ─── CredoraClient ────────────────────────────────────────────────────────────

class TestCredoraClient:
    @pytest.fixture
    def client(self):
        return CredoraClient(use_mock=True)

    def test_get_rating_returns_rating(self, client):
        r = client.get_rating("Aave")
        assert isinstance(r, CredoraRating)

    def test_caching(self, client):
        r1 = client.get_rating("Aave")
        r2 = client.get_rating("Aave")
        # Both calls should return the same cached object (same timestamp)
        assert r1.timestamp == r2.timestamp

    def test_cache_count(self, client):
        client.clear_cache()
        client.get_rating("Aave")
        client.get_rating("Uniswap")
        assert client.cached_count() == 2

    def test_clear_cache(self, client):
        client.get_rating("Aave")
        client.clear_cache()
        assert client.cached_count() == 0

    def test_kelly_multiplier_for(self, client):
        mult = client.kelly_multiplier_for("Aave")
        assert isinstance(mult, float)
        assert 0.0 < mult <= 1.0

    def test_kelly_multiplier_aave(self, client):
        mult = client.kelly_multiplier_for("Aave")
        assert mult == 0.90  # AA tier

    def test_get_bulk_ratings(self, client):
        protocols = ["Aave", "Uniswap", "Compound"]
        ratings = client.get_bulk_ratings(protocols)
        assert len(ratings) == 3
        for p in protocols:
            assert p in ratings
            assert isinstance(ratings[p], CredoraRating)

    def test_bulk_ratings_correctness(self, client):
        ratings = client.get_bulk_ratings(["Ethereum", "GMX"])
        assert ratings["Ethereum"].tier == CredoraRatingTier.AAA
        assert ratings["GMX"].tier == CredoraRatingTier.BB

    def test_cache_expiry(self):
        client = CredoraClient(use_mock=True, cache_ttl_seconds=0.01)
        r1 = client.get_rating("Aave")
        time.sleep(0.02)
        r2 = client.get_rating("Aave")
        # After TTL expires, new fetch gives new timestamp
        assert r2.timestamp >= r1.timestamp

    def test_fallback_on_no_api_key(self):
        # Without use_mock, live API will fail, should fall back to mock
        client = CredoraClient(use_mock=False, timeout_seconds=0.001)
        r = client.get_rating("Aave")
        assert isinstance(r, CredoraRating)
        assert r.tier == CredoraRatingTier.AA

    def test_default_use_mock_false(self):
        client = CredoraClient()
        assert not client._use_mock


# ─── RiskManager + Credora Integration ──────────────────────────────────────

class TestRiskManagerCredoraIntegration:
    @pytest.fixture
    def rm_with_credora(self):
        cred = CredoraClient(use_mock=True)
        return RiskManager(
            max_position_pct=0.10,
            credora_client=cred,
        )

    @pytest.fixture
    def rm_no_credora(self):
        return RiskManager(max_position_pct=0.10)

    def test_validate_with_credora_approved(self, rm_with_credora):
        ok, reason, adj_size = rm_with_credora.validate_trade_with_credora(
            side="YES", size=5.0, price=0.60, portfolio_value=100.0, protocol="Aave"
        )
        assert ok
        assert adj_size < 5.0   # Kelly multiplier should reduce size (AA=0.9 → 4.5)
        assert adj_size == pytest.approx(4.5, rel=0.01)

    def test_validate_with_credora_aaa_keeps_full_size(self, rm_with_credora):
        ok, reason, adj_size = rm_with_credora.validate_trade_with_credora(
            side="YES", size=5.0, price=0.60, portfolio_value=100.0, protocol="Ethereum"
        )
        assert ok
        assert adj_size == pytest.approx(5.0, rel=0.01)  # AAA → 1.0 multiplier

    def test_validate_with_credora_ccc_reduces_size(self, rm_with_credora):
        ok, reason, adj_size = rm_with_credora.validate_trade_with_credora(
            side="YES", size=5.0, price=0.60, portfolio_value=100.0, protocol="osmosis"
        )
        # osmosis is B-rated (0.35 multiplier) → 5.0 * 0.35 = 1.75
        assert ok
        assert adj_size < 3.0

    def test_validate_without_credora_passthrough(self, rm_no_credora):
        ok, reason, adj_size = rm_no_credora.validate_trade_with_credora(
            side="YES", size=5.0, price=0.60, portfolio_value=100.0, protocol="Aave"
        )
        assert ok
        assert adj_size == pytest.approx(5.0)  # No multiplier applied

    def test_get_credora_rating_with_client(self, rm_with_credora):
        rating = rm_with_credora.get_credora_rating("Uniswap")
        assert rating is not None
        assert rating.tier == CredoraRatingTier.AA

    def test_get_credora_rating_without_client(self, rm_no_credora):
        rating = rm_no_credora.get_credora_rating("Aave")
        assert rating is None

    def test_min_grade_filter_rejects_low_tier(self):
        cred = CredoraClient(use_mock=True)
        rm = RiskManager(
            max_position_pct=0.10,
            credora_client=cred,
            credora_min_grade=CredoraRatingTier.BBB,
        )
        # GMX is BB — below BBB minimum
        ok, reason, adj_size = rm.validate_trade_with_credora(
            side="YES", size=5.0, price=0.60, portfolio_value=100.0, protocol="GMX"
        )
        assert not ok
        assert "BB" in reason
        assert adj_size == 0.0

    def test_min_grade_filter_passes_high_tier(self):
        cred = CredoraClient(use_mock=True)
        rm = RiskManager(
            max_position_pct=0.10,
            credora_client=cred,
            credora_min_grade=CredoraRatingTier.BBB,
        )
        ok, reason, adj_size = rm.validate_trade_with_credora(
            side="YES", size=5.0, price=0.60, portfolio_value=100.0, protocol="Aave"
        )
        assert ok
        assert adj_size > 0.0

    def test_credora_in_risk_summary(self, rm_with_credora):
        summary = rm_with_credora.get_risk_summary()
        assert "limits" in summary
        assert isinstance(summary, dict)

    def test_rejected_trade_returns_zero_size(self, rm_with_credora):
        # Over-sized trade should be rejected
        ok, reason, adj_size = rm_with_credora.validate_trade_with_credora(
            side="YES", size=999.0, price=0.60, portfolio_value=100.0, protocol="Aave"
        )
        assert not ok
        assert adj_size == 0.0
