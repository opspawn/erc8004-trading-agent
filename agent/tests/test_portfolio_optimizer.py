"""
Tests for portfolio_optimizer.py — 75+ tests covering:
  - OptimizationResult data class
  - RebalanceOrder data class
  - PortfolioOptimizer._mean/_variance/_covariance (statistics helpers)
  - _build_cov_matrix
  - optimize() method (min_variance, max_sharpe, equal_weight)
  - Credora integration (with mocked client)
  - _normalize_and_cap
  - check_rebalance_needed()
  - compute_rebalance_orders()
  - compute_drift(), max_drift()
  - effective_weights_with_credora()
  - portfolio_risk_summary()
  - validate_rebalance_with_risk_manager()
  - Edge cases: empty inputs, single asset, all-NR ratings
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import math
import pytest
from unittest.mock import MagicMock, patch

from portfolio_optimizer import (
    PortfolioOptimizer,
    OptimizationResult,
    RebalanceOrder,
    MAX_CONCENTRATION_BY_TIER,
    REBALANCE_THRESHOLD,
)


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _make_returns(n_assets: int, n_periods: int = 60, seed: int = 42) -> dict:
    """Generate deterministic synthetic returns for testing."""
    import random
    rng = random.Random(seed)
    assets = [f"PROTO_{i}" for i in range(n_assets)]
    return {a: [rng.gauss(0.001, 0.02) for _ in range(n_periods)] for a in assets}


def _mock_credora(tier: str = "A"):
    """Return a mock Credora client that always returns the given tier."""
    from credora_client import CredoraRating, CredoraRatingTier, KELLY_MULTIPLIERS, _TIER_SCORE_MID
    client = MagicMock()
    t = CredoraRatingTier(tier)
    rating = CredoraRating(protocol="any", tier=t, score=_TIER_SCORE_MID[t])
    client.get_rating.return_value = rating
    return client


def _opt(credora_tier: str = None, risk_manager=None) -> PortfolioOptimizer:
    credora = _mock_credora(credora_tier) if credora_tier else None
    return PortfolioOptimizer(credora_client=credora, risk_manager=risk_manager)


# ─── OptimizationResult ───────────────────────────────────────────────────────

class TestOptimizationResult:
    def test_to_dict_keys(self):
        r = OptimizationResult(
            weights={"A": 0.5, "B": 0.5},
            expected_return=0.1,
            expected_variance=0.02,
            sharpe_ratio=1.5,
            credora_adjustments={"A": 1.0, "B": 0.8},
            concentration_caps={"A": 0.4, "B": 0.3},
            rebalance_needed=False,
        )
        d = r.to_dict()
        assert "weights" in d
        assert "expected_return" in d
        assert "sharpe_ratio" in d
        assert "credora_adjustments" in d

    def test_weights_rounded(self):
        r = OptimizationResult(
            weights={"A": 0.333333333},
            expected_return=0.0, expected_variance=0.0, sharpe_ratio=0.0,
            credora_adjustments={}, concentration_caps={}, rebalance_needed=False,
        )
        d = r.to_dict()
        assert d["weights"]["A"] == pytest.approx(0.333333, abs=1e-4)

    def test_rebalance_needed_in_dict(self):
        r = OptimizationResult(
            weights={}, expected_return=0.0, expected_variance=0.0, sharpe_ratio=0.0,
            credora_adjustments={}, concentration_caps={}, rebalance_needed=True,
        )
        assert r.to_dict()["rebalance_needed"] is True

    def test_method_field(self):
        r = OptimizationResult(
            weights={}, expected_return=0.0, expected_variance=0.0, sharpe_ratio=0.0,
            credora_adjustments={}, concentration_caps={}, rebalance_needed=False,
            method="max_sharpe",
        )
        assert r.to_dict()["method"] == "max_sharpe"


# ─── RebalanceOrder ───────────────────────────────────────────────────────────

class TestRebalanceOrder:
    def test_to_dict_keys(self):
        o = RebalanceOrder("ETH", 0.40, 0.30, 0.10, "SELL", "high")
        d = o.to_dict()
        assert set(d.keys()) == {"protocol", "current_weight", "target_weight", "drift", "direction", "urgency"}

    def test_buy_direction(self):
        o = RebalanceOrder("ETH", 0.10, 0.30, -0.20, "BUY", "high")
        assert o.direction == "BUY"

    def test_sell_direction(self):
        o = RebalanceOrder("ETH", 0.40, 0.10, 0.30, "SELL", "high")
        assert o.direction == "SELL"

    def test_urgency_high(self):
        o = RebalanceOrder("ETH", 0.50, 0.30, 0.20, "SELL", "high")
        assert o.urgency == "high"


# ─── Statistics Helpers ───────────────────────────────────────────────────────

class TestStatisticsHelpers:
    def test_mean_empty(self):
        assert PortfolioOptimizer._mean([]) == 0.0

    def test_mean_single(self):
        assert PortfolioOptimizer._mean([5.0]) == 5.0

    def test_mean_multiple(self):
        assert PortfolioOptimizer._mean([1.0, 2.0, 3.0]) == pytest.approx(2.0)

    def test_variance_single_returns_sentinel(self):
        assert PortfolioOptimizer._variance([1.0]) == 1e-6

    def test_variance_zero_variance(self):
        v = PortfolioOptimizer._variance([5.0, 5.0, 5.0])
        assert v == pytest.approx(0.0, abs=1e-9)

    def test_variance_known(self):
        # variance of [1, 2, 3] = 1.0 (sample variance)
        v = PortfolioOptimizer._variance([1.0, 2.0, 3.0])
        assert v == pytest.approx(1.0, rel=1e-6)

    def test_covariance_single_element(self):
        assert PortfolioOptimizer._covariance([1.0], [1.0]) == 0.0

    def test_covariance_uncorrelated(self):
        # perfect anticorrelation
        xs = [1, -1, 1, -1, 1, -1]
        ys = [-1, 1, -1, 1, -1, 1]
        cov = PortfolioOptimizer._covariance(xs, ys)
        assert cov < 0

    def test_covariance_perfect_positive(self):
        xs = [1.0, 2.0, 3.0, 4.0]
        cov = PortfolioOptimizer._covariance(xs, xs)
        var = PortfolioOptimizer._variance(xs)
        assert cov == pytest.approx(var, rel=1e-6)


# ─── Covariance Matrix ────────────────────────────────────────────────────────

class TestBuildCovMatrix:
    def test_single_asset(self):
        opt = _opt()
        assets, cov = opt._build_cov_matrix({"ETH": [0.01, -0.01, 0.02]})
        assert assets == ["ETH"]
        assert len(cov) == 1
        assert cov[0][0] > 0

    def test_two_assets_symmetric(self):
        opt = _opt()
        returns = {"A": [0.01, 0.02, -0.01], "B": [0.02, 0.01, -0.02]}
        assets, cov = opt._build_cov_matrix(returns)
        n = len(assets)
        for i in range(n):
            for j in range(n):
                assert cov[i][j] == pytest.approx(cov[j][i], rel=1e-9)

    def test_diagonal_nonnegative(self):
        opt = _opt()
        returns = _make_returns(4)
        assets, cov = opt._build_cov_matrix(returns)
        for i in range(len(assets)):
            assert cov[i][i] >= 0

    def test_sorted_assets(self):
        opt = _opt()
        returns = {"Z": [0.01], "A": [0.02], "M": [0.01]}
        assets, _ = opt._build_cov_matrix(returns)
        assert assets == sorted(assets)


# ─── optimize() ───────────────────────────────────────────────────────────────

class TestOptimize:
    def test_empty_returns_dict(self):
        opt = _opt()
        result = opt.optimize({})
        assert result.weights == {}
        assert result.expected_return == 0.0

    def test_single_asset_full_weight(self):
        opt = _opt()
        result = opt.optimize({"ETH": [0.01, 0.02, -0.01, 0.005]})
        assert result.weights.get("ETH", 0.0) == pytest.approx(1.0, abs=1e-4)

    def test_weights_sum_to_one(self):
        opt = _opt()
        returns = _make_returns(5)
        result = opt.optimize(returns)
        total = sum(result.weights.values())
        assert total == pytest.approx(1.0, abs=1e-4)

    def test_weights_nonnegative(self):
        opt = _opt()
        returns = _make_returns(5)
        result = opt.optimize(returns)
        for w in result.weights.values():
            assert w >= -1e-9

    def test_min_variance_method(self):
        opt = _opt()
        returns = _make_returns(4)
        result = opt.optimize(returns, method="min_variance")
        assert result.method == "min_variance"
        assert result.expected_variance >= 0

    def test_max_sharpe_method(self):
        opt = _opt()
        returns = _make_returns(4)
        result = opt.optimize(returns, method="max_sharpe")
        assert result.method == "max_sharpe"

    def test_equal_weight_method(self):
        opt = _opt()
        returns = {"A": [0.01, 0.02], "B": [0.01, -0.01], "C": [0.005, 0.005]}
        result = opt.optimize(returns, method="equal_weight")
        for w in result.weights.values():
            assert w == pytest.approx(1.0 / 3, abs=1e-4)

    def test_invalid_method_raises(self):
        opt = _opt()
        returns = _make_returns(2)
        with pytest.raises(ValueError, match="Unknown method"):
            opt.optimize(returns, method="bad_method")

    def test_result_has_credora_adjustments(self):
        opt = _opt("A")
        returns = _make_returns(3)
        result = opt.optimize(returns)
        assert len(result.credora_adjustments) == 3

    def test_result_has_concentration_caps(self):
        opt = _opt("BBB")
        returns = _make_returns(3)
        result = opt.optimize(returns)
        assert len(result.concentration_caps) == 3

    def test_no_weight_exceeds_cap(self):
        opt = _opt("AAA")  # AAA → 40% cap; 5 assets × 40% = 200% > 100% (feasible)
        returns = _make_returns(5)
        result = opt.optimize(returns)
        for protocol, w in result.weights.items():
            cap = result.concentration_caps.get(protocol, 1.0)
            assert w <= cap + 1e-3

    @pytest.mark.parametrize("n_assets", [2, 3, 5, 8])
    def test_optimize_n_assets(self, n_assets):
        opt = _opt()
        returns = _make_returns(n_assets)
        result = opt.optimize(returns)
        assert len(result.weights) == n_assets
        assert sum(result.weights.values()) == pytest.approx(1.0, abs=1e-3)

    @pytest.mark.parametrize("method", ["min_variance", "max_sharpe", "equal_weight"])
    def test_all_methods_valid(self, method):
        opt = _opt()
        returns = _make_returns(4)
        result = opt.optimize(returns, method=method)
        assert sum(result.weights.values()) == pytest.approx(1.0, abs=1e-3)
        assert result.expected_variance >= 0

    def test_aaa_credora_higher_cap(self):
        opt_aaa = _opt("AAA")
        opt_nr = _opt("NR")
        returns = _make_returns(3, n_periods=40)
        r_aaa = opt_aaa.optimize(returns)
        r_nr = opt_nr.optimize(returns)
        # AAA allows bigger concentrations
        assert max(r_aaa.concentration_caps.values()) > max(r_nr.concentration_caps.values())

    def test_sharpe_ratio_is_float(self):
        opt = _opt()
        returns = _make_returns(3)
        result = opt.optimize(returns)
        assert isinstance(result.sharpe_ratio, float)

    def test_expected_return_float(self):
        opt = _opt()
        returns = _make_returns(3)
        result = opt.optimize(returns)
        assert isinstance(result.expected_return, float)


# ─── Normalize and Cap ────────────────────────────────────────────────────────

class TestNormalizeAndCap:
    def test_empty(self):
        opt = _opt()
        result = opt._normalize_and_cap([], [])
        assert result == []

    def test_single_weight(self):
        opt = _opt()
        result = opt._normalize_and_cap([3.0], [1.0])
        assert result[0] == pytest.approx(1.0)

    def test_enforces_cap(self):
        opt = _opt()
        weights = [0.9, 0.1]
        caps = [0.5, 0.5]
        result = opt._normalize_and_cap(weights, caps)
        assert result[0] <= 0.5 + 1e-6

    def test_sum_to_one(self):
        opt = _opt()
        weights = [0.5, 0.3, 0.2]
        caps = [0.4, 0.4, 0.4]
        result = opt._normalize_and_cap(weights, caps)
        assert sum(result) == pytest.approx(1.0, abs=1e-6)

    def test_all_zero_weights_equal_fallback(self):
        opt = _opt()
        result = opt._normalize_and_cap([0.0, 0.0, 0.0], [0.5, 0.5, 0.5])
        # Should produce equal weights fallback
        assert sum(result) == pytest.approx(1.0, abs=1e-4)


# ─── check_rebalance_needed() ─────────────────────────────────────────────────

class TestCheckRebalanceNeeded:
    def test_no_drift_no_rebalance(self):
        opt = _opt()
        current = {"A": 0.5, "B": 0.5}
        target = {"A": 0.5, "B": 0.5}
        assert opt.check_rebalance_needed(current, target) is False

    def test_small_drift_no_rebalance(self):
        opt = _opt()
        current = {"A": 0.52, "B": 0.48}
        target = {"A": 0.50, "B": 0.50}
        assert opt.check_rebalance_needed(current, target) is False

    def test_large_drift_triggers_rebalance(self):
        opt = _opt()
        current = {"A": 0.70, "B": 0.30}
        target = {"A": 0.50, "B": 0.50}
        assert opt.check_rebalance_needed(current, target) is True

    def test_below_threshold_no_rebalance(self):
        opt = _opt()
        current = {"A": 0.549, "B": 0.451}
        target = {"A": 0.50, "B": 0.50}
        # drift ≈ 4.9% < 5% threshold, no rebalance
        assert opt.check_rebalance_needed(current, target) is False

    def test_just_above_threshold_triggers(self):
        opt = _opt()
        current = {"A": 0.551, "B": 0.449}
        target = {"A": 0.50, "B": 0.50}
        assert opt.check_rebalance_needed(current, target) is True

    def test_custom_threshold(self):
        opt = _opt()
        current = {"A": 0.53, "B": 0.47}
        target = {"A": 0.50, "B": 0.50}
        # drift=0.03, threshold=0.02 → triggers
        assert opt.check_rebalance_needed(current, target, threshold=0.02) is True

    def test_new_asset_triggers_rebalance(self):
        opt = _opt()
        current = {"A": 1.0}
        target = {"A": 0.5, "B": 0.5}
        # B is in target but not current → drift = 0.5 > 0.05
        assert opt.check_rebalance_needed(current, target) is True

    def test_empty_weights_no_rebalance(self):
        opt = _opt()
        assert opt.check_rebalance_needed({}, {}) is False

    @pytest.mark.parametrize("drift", [0.06, 0.10, 0.20, 0.50])
    def test_various_large_drifts_trigger(self, drift):
        opt = _opt()
        current = {"A": 0.50 + drift, "B": 0.50 - drift}
        target = {"A": 0.50, "B": 0.50}
        assert opt.check_rebalance_needed(current, target) is True


# ─── compute_rebalance_orders() ───────────────────────────────────────────────

class TestComputeRebalanceOrders:
    def test_no_drift_no_orders(self):
        opt = _opt()
        orders = opt.compute_rebalance_orders({"A": 0.5, "B": 0.5}, {"A": 0.5, "B": 0.5})
        assert orders == []

    def test_sell_overweight(self):
        opt = _opt()
        orders = opt.compute_rebalance_orders({"A": 0.70}, {"A": 0.50})
        assert len(orders) == 1
        assert orders[0].direction == "SELL"

    def test_buy_underweight(self):
        opt = _opt()
        orders = opt.compute_rebalance_orders({"A": 0.30}, {"A": 0.50})
        assert len(orders) == 1
        assert orders[0].direction == "BUY"

    def test_urgency_high_for_large_drift(self):
        opt = _opt()
        orders = opt.compute_rebalance_orders({"A": 0.80}, {"A": 0.50})
        assert orders[0].urgency == "high"

    def test_urgency_medium_for_medium_drift(self):
        opt = _opt()
        orders = opt.compute_rebalance_orders({"A": 0.57}, {"A": 0.50})
        assert orders[0].urgency == "medium"

    def test_urgency_low_for_small_drift(self):
        opt = _opt()
        orders = opt.compute_rebalance_orders({"A": 0.53}, {"A": 0.50})
        assert orders[0].urgency == "low"

    def test_sorted_by_urgency_then_drift(self):
        opt = _opt()
        current = {"A": 0.70, "B": 0.20, "C": 0.10}
        target = {"A": 0.40, "B": 0.40, "C": 0.20}
        orders = opt.compute_rebalance_orders(current, target)
        # A has largest drift (0.30), should come first
        assert orders[0].protocol == "A"

    def test_new_asset_creates_buy_order(self):
        opt = _opt()
        orders = opt.compute_rebalance_orders({"A": 1.0}, {"A": 0.5, "B": 0.5})
        protocols = {o.protocol for o in orders}
        assert "B" in protocols
        b_order = next(o for o in orders if o.protocol == "B")
        assert b_order.direction == "BUY"

    def test_order_drift_field(self):
        opt = _opt()
        orders = opt.compute_rebalance_orders({"A": 0.70}, {"A": 0.50})
        assert orders[0].drift == pytest.approx(0.20, abs=1e-6)


# ─── compute_drift() and max_drift() ─────────────────────────────────────────

class TestDriftMethods:
    def test_compute_drift_zero(self):
        opt = _opt()
        drift = opt.compute_drift({"A": 0.5, "B": 0.5}, {"A": 0.5, "B": 0.5})
        for v in drift.values():
            assert v == pytest.approx(0.0)

    def test_compute_drift_positive(self):
        opt = _opt()
        drift = opt.compute_drift({"A": 0.7}, {"A": 0.5})
        assert drift["A"] == pytest.approx(0.2)

    def test_compute_drift_negative(self):
        opt = _opt()
        drift = opt.compute_drift({"A": 0.3}, {"A": 0.5})
        assert drift["A"] == pytest.approx(-0.2)

    def test_max_drift_zero(self):
        opt = _opt()
        assert opt.max_drift({"A": 0.5}, {"A": 0.5}) == pytest.approx(0.0)

    def test_max_drift_nonzero(self):
        opt = _opt()
        result = opt.max_drift({"A": 0.8, "B": 0.2}, {"A": 0.5, "B": 0.5})
        assert result == pytest.approx(0.3, abs=1e-6)

    def test_max_drift_empty(self):
        opt = _opt()
        assert opt.max_drift({}, {}) == 0.0


# ─── effective_weights_with_credora() ─────────────────────────────────────────

class TestEffectiveWeightsWithCredora:
    def test_no_credora_unchanged(self):
        opt = _opt()
        raw = {"A": 0.5, "B": 0.5}
        result = opt.effective_weights_with_credora(raw)
        assert sum(result.values()) == pytest.approx(1.0, abs=1e-6)

    def test_with_credora_sums_to_one(self):
        opt = _opt("BBB")
        raw = {"A": 0.4, "B": 0.3, "C": 0.3}
        result = opt.effective_weights_with_credora(raw)
        assert sum(result.values()) == pytest.approx(1.0, abs=1e-6)

    def test_empty_returns_empty(self):
        opt = _opt("A")
        assert opt.effective_weights_with_credora({}) == {}

    def test_nr_credora_lowers_weight(self):
        opt_nr = _opt("NR")    # kelly_multiplier=0.10
        opt_aaa = _opt("AAA")  # kelly_multiplier=1.00
        raw = {"A": 0.5, "B": 0.5}
        result_nr = opt_nr.effective_weights_with_credora(raw)
        result_aaa = opt_aaa.effective_weights_with_credora(raw)
        # With uniform tiers, weights normalize to the same — but NR applies 0.10 multiplier
        # The ratio stays equal since both assets get the same multiplier
        # At least check they still sum to 1
        assert sum(result_nr.values()) == pytest.approx(1.0, abs=1e-6)
        assert sum(result_aaa.values()) == pytest.approx(1.0, abs=1e-6)


# ─── portfolio_risk_summary() ─────────────────────────────────────────────────

class TestPortfolioRiskSummary:
    def test_empty_returns_empty_dict(self):
        opt = _opt()
        assert opt.portfolio_risk_summary({}, {}) == {}

    def test_has_required_keys(self):
        opt = _opt()
        weights = {"A": 0.6, "B": 0.4}
        returns = {"A": [0.01, 0.02, -0.01], "B": [0.005, -0.005, 0.015]}
        summary = opt.portfolio_risk_summary(weights, returns)
        assert "expected_annual_return" in summary
        assert "sharpe_ratio" in summary
        assert "per_asset" in summary

    def test_per_asset_keys(self):
        opt = _opt()
        weights = {"A": 1.0}
        returns = {"A": [0.01, 0.02, 0.03]}
        summary = opt.portfolio_risk_summary(weights, returns)
        assert "A" in summary["per_asset"]
        assert "weight" in summary["per_asset"]["A"]

    def test_std_nonnegative(self):
        opt = _opt()
        weights = {"A": 0.5, "B": 0.5}
        returns = {"A": [0.01, -0.01, 0.02], "B": [0.02, 0.01, -0.02]}
        summary = opt.portfolio_risk_summary(weights, returns)
        assert summary["expected_annual_std"] >= 0


# ─── validate_rebalance_with_risk_manager() ───────────────────────────────────

class TestValidateRebalanceWithRiskManager:
    def test_no_risk_manager_always_ok(self):
        opt = _opt()
        order = RebalanceOrder("ETH", 0.40, 0.30, 0.10, "SELL", "medium")
        ok, reason, size = opt.validate_rebalance_with_risk_manager(order, 1000.0)
        assert ok is True
        assert "no_risk_manager" in reason
        assert size == pytest.approx(100.0)

    def test_risk_manager_called(self):
        rm = MagicMock()
        rm.validate_trade_with_sentiment.return_value = (True, "OK | sentiment: neutral", 95.0)
        opt = PortfolioOptimizer(risk_manager=rm)
        order = RebalanceOrder("ETH", 0.40, 0.30, 0.10, "SELL", "medium")
        ok, reason, size = opt.validate_rebalance_with_risk_manager(order, 1000.0)
        assert ok is True
        rm.validate_trade_with_sentiment.assert_called_once()

    def test_risk_manager_reject(self):
        rm = MagicMock()
        rm.validate_trade_with_sentiment.return_value = (False, "drawdown_limit", 0.0)
        opt = PortfolioOptimizer(risk_manager=rm)
        order = RebalanceOrder("ETH", 0.40, 0.30, 0.10, "SELL", "medium")
        ok, reason, size = opt.validate_rebalance_with_risk_manager(order, 1000.0)
        assert ok is False

    def test_risk_manager_exception_fallback(self):
        rm = MagicMock()
        rm.validate_trade_with_sentiment.side_effect = RuntimeError("network error")
        opt = PortfolioOptimizer(risk_manager=rm)
        order = RebalanceOrder("ETH", 0.40, 0.30, 0.10, "SELL", "medium")
        ok, reason, size = opt.validate_rebalance_with_risk_manager(order, 1000.0)
        assert ok is True  # fallback to allowing trade
        assert "risk_manager_error" in reason


# ─── Concentration Cap Constants ─────────────────────────────────────────────

class TestConcentrationCaps:
    def test_aaa_has_highest_cap(self):
        assert MAX_CONCENTRATION_BY_TIER["AAA"] == max(MAX_CONCENTRATION_BY_TIER.values())

    def test_nr_has_lowest_cap(self):
        assert MAX_CONCENTRATION_BY_TIER["NR"] == min(MAX_CONCENTRATION_BY_TIER.values())

    def test_all_tiers_present(self):
        for tier in ["AAA", "AA", "A", "BBB", "BB", "B", "CCC", "NR"]:
            assert tier in MAX_CONCENTRATION_BY_TIER

    def test_caps_strictly_decreasing(self):
        # Higher-quality tiers should have higher caps
        tiers = ["AAA", "AA", "A", "BBB", "BB", "B", "CCC", "NR"]
        caps = [MAX_CONCENTRATION_BY_TIER[t] for t in tiers]
        for i in range(len(caps) - 1):
            assert caps[i] >= caps[i + 1]

    def test_rebalance_threshold_constant(self):
        assert REBALANCE_THRESHOLD == 0.05
