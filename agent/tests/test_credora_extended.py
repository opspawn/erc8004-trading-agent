"""
test_credora_extended.py — Extended tests for Credora + risk integration.

Covers edge cases, boundary conditions, bulk operations, and integration
scenarios between Credora ratings and the trading decision pipeline.
"""

import sys
import os
import time
import hashlib

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
    _WELL_KNOWN,
    _TIER_SCORE_MID,
)
from risk_manager import RiskManager


# ─── Additional CredoraRatingTier Tests ──────────────────────────────────────

class TestRatingTierExtended:
    def test_nr_in_kelly_multipliers(self):
        assert CredoraRatingTier.NR in KELLY_MULTIPLIERS

    def test_tier_from_string_aaa(self):
        t = CredoraRatingTier("AAA")
        assert t == CredoraRatingTier.AAA

    def test_tier_from_string_nr(self):
        t = CredoraRatingTier("NR")
        assert t == CredoraRatingTier.NR

    def test_invalid_tier_raises(self):
        with pytest.raises(ValueError):
            CredoraRatingTier("D")

    def test_investment_grade_is_frozenset(self):
        assert isinstance(INVESTMENT_GRADE, frozenset)

    def test_all_ig_tiers_have_high_multiplier(self):
        for tier in INVESTMENT_GRADE:
            assert KELLY_MULTIPLIERS[tier] >= 0.65, (
                f"{tier} is IG but multiplier {KELLY_MULTIPLIERS[tier]} < 0.65"
            )

    def test_non_ig_tiers_have_lower_multiplier(self):
        non_ig = set(CredoraRatingTier) - INVESTMENT_GRADE
        for tier in non_ig:
            assert KELLY_MULTIPLIERS[tier] <= 0.50, (
                f"{tier} is non-IG but multiplier {KELLY_MULTIPLIERS[tier]} > 0.50"
            )

    def test_tier_score_mids_reasonable(self):
        for tier, mid in _TIER_SCORE_MID.items():
            assert 0 <= mid <= 100, f"{tier}: mid={mid} out of range"

    def test_score_mids_monotone(self):
        tiers = [
            CredoraRatingTier.CCC, CredoraRatingTier.B, CredoraRatingTier.BB,
            CredoraRatingTier.BBB, CredoraRatingTier.A, CredoraRatingTier.AA,
            CredoraRatingTier.AAA,
        ]
        mids = [_TIER_SCORE_MID[t] for t in tiers]
        for i in range(len(mids) - 1):
            assert mids[i] < mids[i + 1], (
                f"Score mids non-monotone at {tiers[i]}: {mids[i]} >= {mids[i+1]}"
            )


# ─── Well-Known Protocol Coverage ────────────────────────────────────────────

class TestWellKnownProtocols:
    @pytest.fixture
    def provider(self):
        return CredoraRatingProvider()

    @pytest.mark.parametrize("protocol,expected_tier", [
        ("aave", CredoraRatingTier.AA),
        ("uniswap", CredoraRatingTier.AA),
        ("compound", CredoraRatingTier.A),
        ("maker", CredoraRatingTier.AA),
        ("curve", CredoraRatingTier.A),
        ("lido", CredoraRatingTier.A),
        ("chainlink", CredoraRatingTier.AAA),
        ("ethereum", CredoraRatingTier.AAA),
        ("eth", CredoraRatingTier.AAA),
        ("bitcoin", CredoraRatingTier.AAA),
        ("btc", CredoraRatingTier.AAA),
        ("wbtc", CredoraRatingTier.AA),
        ("usdc", CredoraRatingTier.AAA),
        ("usdt", CredoraRatingTier.AA),
        ("dai", CredoraRatingTier.AA),
        ("balancer", CredoraRatingTier.BBB),
        ("yearn", CredoraRatingTier.BBB),
        ("convex", CredoraRatingTier.BBB),
        ("frax", CredoraRatingTier.BBB),
        ("gmx", CredoraRatingTier.BB),
        ("dydx", CredoraRatingTier.BB),
        ("synthetix", CredoraRatingTier.BBB),
        ("mantle", CredoraRatingTier.BBB),
        ("sushiswap", CredoraRatingTier.BB),
        ("pancakeswap", CredoraRatingTier.BB),
        ("osmosis", CredoraRatingTier.B),
    ])
    def test_well_known_rating(self, provider, protocol, expected_tier):
        r = provider.get_rating(protocol)
        assert r.tier == expected_tier, (
            f"{protocol}: expected {expected_tier}, got {r.tier}"
        )

    def test_well_known_count(self):
        assert len(_WELL_KNOWN) >= 25

    def test_all_well_known_have_valid_tier(self):
        for protocol, tier in _WELL_KNOWN.items():
            assert tier in CredoraRatingTier, f"{protocol}: invalid tier {tier}"


# ─── CredoraRating Edge Cases ────────────────────────────────────────────────

class TestCredoraRatingEdgeCases:
    def test_rating_with_score_zero(self):
        r = CredoraRating(protocol="test", tier=CredoraRatingTier.NR, score=0.0)
        assert r.kelly_multiplier == 0.10

    def test_rating_with_score_100(self):
        r = CredoraRating(protocol="test", tier=CredoraRatingTier.AAA, score=100.0)
        assert r.is_investment_grade

    def test_rating_timestamp_is_float(self):
        r = CredoraRating(protocol="test", tier=CredoraRatingTier.A, score=80.0)
        assert isinstance(r.timestamp, float)

    def test_all_tiers_produce_valid_kelly(self):
        for tier in CredoraRatingTier:
            r = CredoraRating(protocol="test", tier=tier, score=50.0)
            assert 0 < r.kelly_multiplier <= 1.0

    def test_rating_equality_not_same_object(self):
        r1 = CredoraRating(protocol="Aave", tier=CredoraRatingTier.AA, score=87.5)
        r2 = CredoraRating(protocol="Aave", tier=CredoraRatingTier.AA, score=87.5)
        # Dataclasses compare by value
        assert r1.tier == r2.tier
        assert r1.protocol == r2.protocol

    def test_str_contains_kelly(self):
        r = CredoraRating(protocol="Test", tier=CredoraRatingTier.BBB, score=70.0)
        assert "0.65" in str(r)

    def test_nr_tier_not_investment_grade(self):
        r = CredoraRating(protocol="Unknown", tier=CredoraRatingTier.NR, score=5.0)
        assert not r.is_investment_grade

    def test_age_increases_over_time(self):
        r = CredoraRating(protocol="test", tier=CredoraRatingTier.A, score=80.0)
        age1 = r.age_seconds
        time.sleep(0.01)
        age2 = r.age_seconds
        assert age2 >= age1


# ─── CredoraClient Advanced Tests ────────────────────────────────────────────

class TestCredoraClientAdvanced:
    @pytest.fixture
    def client(self):
        return CredoraClient(use_mock=True)

    def test_cache_is_dict(self, client):
        assert isinstance(client._cache, dict)

    def test_multiple_protocols_cached_independently(self, client):
        client.clear_cache()
        client.get_rating("Aave")
        client.get_rating("Uniswap")
        client.get_rating("Compound")
        assert client.cached_count() == 3

    def test_case_insensitive_cache(self, client):
        client.clear_cache()
        client.get_rating("Aave")
        client.get_rating("aave")  # Should hit cache from same normalized key
        assert client.cached_count() == 1

    def test_whitespace_normalized(self, client):
        client.clear_cache()
        client.get_rating("Aave")
        client.get_rating("  Aave  ")  # Same key after strip
        assert client.cached_count() == 1

    def test_bulk_all_have_valid_kelly(self, client):
        protocols = ["Aave", "Uniswap", "GMX", "Osmosis", "FakeProtocol"]
        ratings = client.get_bulk_ratings(protocols)
        for p, r in ratings.items():
            assert 0 < r.kelly_multiplier <= 1.0

    def test_bulk_empty_list(self, client):
        result = client.get_bulk_ratings([])
        assert result == {}

    def test_bulk_single_protocol(self, client):
        result = client.get_bulk_ratings(["Ethereum"])
        assert len(result) == 1
        assert result["Ethereum"].tier == CredoraRatingTier.AAA

    def test_kelly_multiplier_for_is_float(self, client):
        mult = client.kelly_multiplier_for("Aave")
        assert isinstance(mult, float)

    def test_kelly_multiplier_for_all_in_range(self, client):
        protocols = ["Aave", "Bitcoin", "GMX", "SushiSwap", "UnknownFoo"]
        for p in protocols:
            mult = client.kelly_multiplier_for(p)
            assert 0 < mult <= 1.0, f"{p}: {mult}"

    def test_short_cache_ttl_causes_refetch(self):
        client = CredoraClient(use_mock=True, cache_ttl_seconds=0.005)
        r1 = client.get_rating("Aave")
        time.sleep(0.01)
        r2 = client.get_rating("Aave")
        # Both should be valid ratings
        assert r1.tier == r2.tier  # Same tier (deterministic mock)

    def test_long_cache_ttl_uses_cache(self):
        client = CredoraClient(use_mock=True, cache_ttl_seconds=3600)
        r1 = client.get_rating("Aave")
        r2 = client.get_rating("Aave")
        assert r1 is r2  # Same object from cache

    def test_protocol_field_preserved(self, client):
        r = client.get_rating("MyProtocol")
        assert r.protocol == "MyProtocol"

    def test_double_clear_cache(self, client):
        client.get_rating("Aave")
        client.clear_cache()
        client.clear_cache()  # Should not raise
        assert client.cached_count() == 0

    def test_source_is_mock_for_mock_client(self, client):
        r = client.get_rating("Aave")
        assert r.source == "mock"


# ─── RiskManager Credora Deep Integration ────────────────────────────────────

class TestRiskManagerCredoraDeep:
    @pytest.fixture
    def rm(self):
        return RiskManager(
            max_position_pct=0.20,
            credora_client=CredoraClient(use_mock=True),
        )

    def test_chainlink_aaa_no_reduction(self, rm):
        ok, _, size = rm.validate_trade_with_credora(
            "YES", 10.0, 0.5, 100.0, "Chainlink"
        )
        assert ok
        assert size == pytest.approx(10.0, rel=0.01)

    def test_uniswap_aa_10pct_reduction(self, rm):
        ok, _, size = rm.validate_trade_with_credora(
            "YES", 10.0, 0.5, 100.0, "Uniswap"
        )
        assert ok
        assert size == pytest.approx(9.0, rel=0.01)

    def test_compound_a_20pct_reduction(self, rm):
        ok, _, size = rm.validate_trade_with_credora(
            "YES", 10.0, 0.5, 100.0, "Compound"
        )
        assert ok
        assert size == pytest.approx(8.0, rel=0.01)

    def test_yearn_bbb_35pct_reduction(self, rm):
        ok, _, size = rm.validate_trade_with_credora(
            "YES", 10.0, 0.5, 100.0, "Yearn"
        )
        assert ok
        assert size == pytest.approx(6.5, rel=0.01)

    def test_gmx_bb_50pct_reduction(self, rm):
        ok, _, size = rm.validate_trade_with_credora(
            "YES", 10.0, 0.5, 100.0, "GMX"
        )
        assert ok
        assert size == pytest.approx(5.0, rel=0.01)

    def test_osmosis_b_65pct_reduction(self, rm):
        ok, _, size = rm.validate_trade_with_credora(
            "YES", 10.0, 0.5, 100.0, "Osmosis"
        )
        assert ok
        assert size == pytest.approx(3.5, rel=0.01)

    def test_reject_reason_on_min_grade_failure(self):
        cred = CredoraClient(use_mock=True)
        rm = RiskManager(
            credora_client=cred,
            credora_min_grade=CredoraRatingTier.A,
        )
        ok, reason, _ = rm.validate_trade_with_credora(
            "YES", 5.0, 0.5, 100.0, "GMX"  # GMX is BB
        )
        assert not ok
        assert "below minimum" in reason.lower() or "minimum" in reason.lower()

    def test_no_credora_client_no_adjustment(self):
        rm = RiskManager(max_position_pct=0.20)
        ok, _, size = rm.validate_trade_with_credora(
            "YES", 10.0, 0.5, 100.0, "Aave"
        )
        assert ok
        assert size == pytest.approx(10.0)

    def test_credora_does_not_bypass_position_limit(self):
        rm = RiskManager(
            max_position_pct=0.10,
            credora_client=CredoraClient(use_mock=True),
        )
        # 50 USDC is 50% of 100 — even after credora reduction it should exceed 10% limit
        ok, reason, _ = rm.validate_trade_with_credora(
            "YES", 50.0, 0.5, 100.0, "Aave"  # 50 * 0.9 = 45 > 10 max
        )
        assert not ok
        assert "max position" in reason.lower() or "exceeds" in reason.lower()

    def test_credora_rating_cached_in_client(self):
        cred = CredoraClient(use_mock=True)
        cred.clear_cache()
        rm = RiskManager(credora_client=cred)
        rm.validate_trade_with_credora("YES", 5.0, 0.5, 100.0, "Aave")
        rm.validate_trade_with_credora("YES", 5.0, 0.5, 100.0, "Aave")
        # Should be cached after first call
        assert cred.cached_count() >= 1

    def test_adjusted_size_never_negative(self):
        cred = CredoraClient(use_mock=True)
        rm = RiskManager(credora_client=cred)
        for protocol in ["Aave", "GMX", "Osmosis", "UnknownProto"]:
            _, _, size = rm.validate_trade_with_credora(
                "YES", 5.0, 0.5, 100.0, protocol
            )
            assert size >= 0.0

    def test_both_sides_apply_credora(self):
        cred = CredoraClient(use_mock=True)
        rm = RiskManager(max_position_pct=0.20, credora_client=cred)
        ok_yes, _, size_yes = rm.validate_trade_with_credora(
            "YES", 10.0, 0.5, 100.0, "Aave"
        )
        ok_no, _, size_no = rm.validate_trade_with_credora(
            "NO", 10.0, 0.5, 100.0, "Aave"
        )
        assert ok_yes and ok_no
        assert size_yes == pytest.approx(size_no)

    def test_get_rating_returns_correct_type(self):
        cred = CredoraClient(use_mock=True)
        rm = RiskManager(credora_client=cred)
        r = rm.get_credora_rating("Aave")
        assert isinstance(r, CredoraRating)

    def test_risk_summary_unchanged_by_credora(self):
        cred = CredoraClient(use_mock=True)
        rm1 = RiskManager(max_position_pct=0.10)
        rm2 = RiskManager(max_position_pct=0.10, credora_client=cred)
        s1 = rm1.get_risk_summary()
        s2 = rm2.get_risk_summary()
        # Both should have the same limit keys
        assert set(s1["limits"].keys()) == set(s2["limits"].keys())


# ─── Credora Provider Hash Stability ─────────────────────────────────────────

class TestProviderHashStability:
    @pytest.fixture
    def provider(self):
        return CredoraRatingProvider()

    def test_hash_deterministic_cross_calls(self, provider):
        results = [provider.get_rating("DeadBeefProtocol").tier for _ in range(5)]
        assert all(r == results[0] for r in results)

    def test_different_names_can_produce_different_tiers(self, provider):
        # Not all unknown protocols should map to the same tier
        tiers = set()
        for name in ["AlphaProtocol", "BetaDAO", "GammaSwap", "DeltaVault",
                     "EpsilonBridge", "ZetaLend", "EtaFarm"]:
            tiers.add(provider.get_rating(name).tier)
        # Should produce at least 2 distinct tiers across 7 protocols
        assert len(tiers) >= 2

    def test_score_deterministic(self, provider):
        s1 = provider.get_rating("RandomFoo").score
        s2 = provider.get_rating("RandomFoo").score
        assert s1 == s2
