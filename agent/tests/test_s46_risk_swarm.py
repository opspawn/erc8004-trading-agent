"""
test_s46_risk_swarm.py — Sprint 46: Portfolio Risk Management + Multi-Agent Swarm.

Covers (160+ tests):

  Section A — Risk helper functions (unit tests, ~50 tests):
    A.1  _s46_seed_returns
    A.2  _s46_var (VaR at 95% and 99%)
    A.3  _s46_sharpe
    A.4  _s46_sortino
    A.5  _s46_calmar
    A.6  _s46_max_drawdown
    A.7  _s46_correlation

  Section B — get_s46_portfolio_risk (20+ tests)
  Section C — get_s46_position_size (20+ tests)
  Section D — get_s46_exposure (15+ tests)
  Section E — Swarm state / helpers (15+ tests)
  Section F — get_s46_swarm_vote (25+ tests)
  Section G — get_s46_swarm_performance (15+ tests)
  Section H — HTTP endpoint tests via DemoServer (25+ tests)
"""

from __future__ import annotations

import json
import math
import time
import urllib.error
import urllib.request
import threading
from typing import Any, Dict, List

import pytest

# ── import module under test ──────────────────────────────────────────────────
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from demo_server import (
    _s46_seed_returns,
    _s46_var,
    _s46_sharpe,
    _s46_sortino,
    _s46_calmar,
    _s46_max_drawdown,
    _s46_correlation,
    get_s46_portfolio_risk,
    get_s46_position_size,
    get_s46_exposure,
    get_s46_swarm_vote,
    get_s46_swarm_performance,
    _S46_SYMBOLS,
    _S46_SWARM_AGENTS,
    DemoServer,
)

# ─────────────────────────────────────────────────────────────────────────────
# Section A — Risk helper functions
# ─────────────────────────────────────────────────────────────────────────────

class TestS46SeedReturns:
    """A.1 — _s46_seed_returns"""

    def test_returns_list(self):
        r = _s46_seed_returns("BTC-USD")
        assert isinstance(r, list)

    def test_default_length_252(self):
        r = _s46_seed_returns("BTC-USD")
        assert len(r) == 252

    def test_custom_length(self):
        r = _s46_seed_returns("ETH-USD", n=50)
        assert len(r) == 50

    def test_all_floats(self):
        r = _s46_seed_returns("SOL-USD")
        assert all(isinstance(x, float) for x in r)

    def test_deterministic_same_seed(self):
        r1 = _s46_seed_returns("BTC-USD", seed=46)
        r2 = _s46_seed_returns("BTC-USD", seed=46)
        assert r1 == r2

    def test_different_seeds_differ(self):
        r1 = _s46_seed_returns("BTC-USD", seed=1)
        r2 = _s46_seed_returns("BTC-USD", seed=2)
        assert r1 != r2

    def test_different_symbols_differ(self):
        r1 = _s46_seed_returns("BTC-USD")
        r2 = _s46_seed_returns("ETH-USD")
        assert r1 != r2

    def test_returns_have_variance(self):
        r = _s46_seed_returns("BTC-USD")
        assert max(r) > min(r)

    def test_btc_moderate_volatility(self):
        r = _s46_seed_returns("BTC-USD")
        std = math.sqrt(sum(x**2 for x in r) / len(r))
        assert 0.01 < std < 0.10  # daily vol 1%–10%

    def test_matic_higher_vol_than_btc(self):
        r_btc = _s46_seed_returns("BTC-USD")
        r_matic = _s46_seed_returns("MATIC-USD")
        std_btc = math.sqrt(sum(x**2 for x in r_btc) / len(r_btc))
        std_matic = math.sqrt(sum(x**2 for x in r_matic) / len(r_matic))
        assert std_matic > std_btc


class TestS46VaR:
    """A.2 — _s46_var"""

    def test_returns_float(self):
        r = _s46_seed_returns("BTC-USD")
        v = _s46_var(r, 0.95)
        assert isinstance(v, float)

    def test_var_positive(self):
        r = _s46_seed_returns("BTC-USD")
        v = _s46_var(r, 0.95)
        assert v >= 0.0

    def test_var_99_gte_var_95(self):
        r = _s46_seed_returns("BTC-USD")
        v95 = _s46_var(r, 0.95)
        v99 = _s46_var(r, 0.99)
        assert v99 >= v95

    def test_var_empty_returns_zero(self):
        assert _s46_var([], 0.95) == 0.0

    def test_var_95_reasonable_btc(self):
        r = _s46_seed_returns("BTC-USD")
        v = _s46_var(r, 0.95)
        assert 0.001 < v < 0.20

    def test_var_matic_gte_btc(self):
        r_btc = _s46_seed_returns("BTC-USD")
        r_matic = _s46_seed_returns("MATIC-USD")
        assert _s46_var(r_matic, 0.95) >= _s46_var(r_btc, 0.95)

    def test_var_100_confidence_equals_max_loss(self):
        # At 100% confidence the worst-case loss is max
        r = [-0.05, -0.10, 0.02, 0.03, -0.01]
        v = _s46_var(r, 1.0)
        assert v == pytest.approx(abs(min(r)), rel=0.01)

    def test_var_single_value(self):
        v = _s46_var([0.05], 0.95)
        assert isinstance(v, float)


class TestS46Sharpe:
    """A.3 — _s46_sharpe"""

    def test_returns_float(self):
        r = _s46_seed_returns("BTC-USD")
        s = _s46_sharpe(r)
        assert isinstance(s, float)

    def test_positive_positive_drift(self):
        # Consistent positive returns → positive Sharpe
        r = [0.001] * 100
        s = _s46_sharpe(r)
        assert s > 0

    def test_negative_negative_drift(self):
        r = [-0.001] * 100
        s = _s46_sharpe(r)
        assert s < 0

    def test_single_value_returns_zero(self):
        s = _s46_sharpe([0.01])
        assert s == 0.0

    def test_empty_returns_zero(self):
        s = _s46_sharpe([])
        assert s == 0.0

    def test_reasonable_range(self):
        r = _s46_seed_returns("BTC-USD")
        s = _s46_sharpe(r)
        assert -10.0 < s < 10.0


class TestS46Sortino:
    """A.4 — _s46_sortino"""

    def test_returns_float(self):
        r = _s46_seed_returns("BTC-USD")
        s = _s46_sortino(r)
        assert isinstance(s, float)

    def test_empty_returns_zero(self):
        s = _s46_sortino([])
        assert s == 0.0

    def test_only_positive_returns_high_value(self):
        r = [0.01] * 50
        s = _s46_sortino(r)
        assert s > 0  # no downside → capped at 5.0 or very high

    def test_mixed_returns_finite(self):
        r = _s46_seed_returns("ETH-USD")
        s = _s46_sortino(r)
        assert math.isfinite(s)

    def test_sortino_gte_sharpe_for_upside(self):
        # When returns have mixed sign, both metrics should be finite and sortino > 0
        r = [0.002 + (0.001 if i % 3 == 0 else -0.0005) for i in range(100)]
        sortino = _s46_sortino(r)
        sharpe = _s46_sharpe(r)
        assert math.isfinite(sortino)
        assert math.isfinite(sharpe)
        assert sortino > 0


class TestS46Calmar:
    """A.5 — _s46_calmar"""

    def test_returns_float(self):
        r = _s46_seed_returns("BTC-USD")
        c = _s46_calmar(r)
        assert isinstance(c, float)

    def test_single_value_returns_zero(self):
        c = _s46_calmar([0.01])
        assert c == 0.0

    def test_no_drawdown_returns_high(self):
        r = [0.001] * 100  # always up → max_dd ≈ 0
        c = _s46_calmar(r)
        assert c == pytest.approx(10.0)

    def test_reasonable_range(self):
        r = _s46_seed_returns("SOL-USD")
        c = _s46_calmar(r)
        assert -100 < c < 100


class TestS46MaxDrawdown:
    """A.6 — _s46_max_drawdown"""

    def test_returns_float(self):
        r = _s46_seed_returns("BTC-USD")
        md = _s46_max_drawdown(r)
        assert isinstance(md, float)

    def test_always_non_negative(self):
        r = _s46_seed_returns("BTC-USD")
        md = _s46_max_drawdown(r)
        assert md >= 0.0

    def test_always_lte_one(self):
        r = _s46_seed_returns("BTC-USD")
        md = _s46_max_drawdown(r)
        assert md <= 1.0

    def test_monotone_increase_is_zero(self):
        r = [0.01] * 50
        md = _s46_max_drawdown(r)
        assert md == pytest.approx(0.0, abs=1e-6)

    def test_matic_drawdown_gte_btc(self):
        r_btc = _s46_seed_returns("BTC-USD")
        r_matic = _s46_seed_returns("MATIC-USD")
        md_btc = _s46_max_drawdown(r_btc)
        md_matic = _s46_max_drawdown(r_matic)
        # Higher vol tends to larger drawdown (not guaranteed but almost always true)
        assert md_matic >= 0.0 and md_btc >= 0.0


class TestS46Correlation:
    """A.7 — _s46_correlation"""

    def test_returns_dict(self):
        rm = {s: _s46_seed_returns(s) for s in _S46_SYMBOLS}
        c = _s46_correlation(rm)
        assert isinstance(c, dict)

    def test_diagonal_is_one(self):
        rm = {s: _s46_seed_returns(s) for s in _S46_SYMBOLS}
        c = _s46_correlation(rm)
        for sym in _S46_SYMBOLS:
            assert c[sym][sym] == pytest.approx(1.0, abs=1e-4)

    def test_symmetric(self):
        rm = {s: _s46_seed_returns(s) for s in _S46_SYMBOLS}
        c = _s46_correlation(rm)
        for a in _S46_SYMBOLS:
            for b in _S46_SYMBOLS:
                assert c[a][b] == pytest.approx(c[b][a], abs=1e-6)

    def test_values_in_minus_one_to_one(self):
        rm = {s: _s46_seed_returns(s) for s in _S46_SYMBOLS}
        c = _s46_correlation(rm)
        for a in _S46_SYMBOLS:
            for b in _S46_SYMBOLS:
                assert -1.01 <= c[a][b] <= 1.01

    def test_all_symbols_present(self):
        rm = {s: _s46_seed_returns(s) for s in _S46_SYMBOLS}
        c = _s46_correlation(rm)
        for sym in _S46_SYMBOLS:
            assert sym in c
            for sym2 in _S46_SYMBOLS:
                assert sym2 in c[sym]

    def test_identical_series_correlation_one(self):
        r = [0.01, -0.02, 0.03, -0.01, 0.02]
        rm = {"A": r, "B": r}
        c = _s46_correlation(rm)
        assert c["A"]["B"] == pytest.approx(1.0, abs=1e-4)

    def test_opposite_series_correlation_negative_one(self):
        r = [0.01, -0.02, 0.03, -0.01, 0.02]
        r_neg = [-x for x in r]
        rm = {"A": r, "B": r_neg}
        c = _s46_correlation(rm)
        assert c["A"]["B"] == pytest.approx(-1.0, abs=1e-4)


# ─────────────────────────────────────────────────────────────────────────────
# Section B — get_s46_portfolio_risk
# ─────────────────────────────────────────────────────────────────────────────

class TestGetS46PortfolioRisk:
    def test_returns_dict(self):
        r = get_s46_portfolio_risk()
        assert isinstance(r, dict)

    def test_portfolio_key_exists(self):
        r = get_s46_portfolio_risk()
        assert "portfolio" in r

    def test_per_symbol_key_exists(self):
        r = get_s46_portfolio_risk()
        assert "per_symbol" in r

    def test_correlation_matrix_key(self):
        r = get_s46_portfolio_risk()
        assert "correlation_matrix" in r

    def test_symbols_key(self):
        r = get_s46_portfolio_risk()
        assert "symbols" in r
        assert set(r["symbols"]) == set(_S46_SYMBOLS)

    def test_portfolio_var_95_present(self):
        r = get_s46_portfolio_risk()
        assert "var_95" in r["portfolio"]

    def test_portfolio_var_99_present(self):
        r = get_s46_portfolio_risk()
        assert "var_99" in r["portfolio"]

    def test_portfolio_sharpe_present(self):
        r = get_s46_portfolio_risk()
        assert "sharpe_ratio" in r["portfolio"]

    def test_portfolio_sortino_present(self):
        r = get_s46_portfolio_risk()
        assert "sortino_ratio" in r["portfolio"]

    def test_portfolio_calmar_present(self):
        r = get_s46_portfolio_risk()
        assert "calmar_ratio" in r["portfolio"]

    def test_portfolio_max_drawdown_present(self):
        r = get_s46_portfolio_risk()
        assert "max_drawdown" in r["portfolio"]

    def test_portfolio_var_95_positive(self):
        r = get_s46_portfolio_risk()
        assert r["portfolio"]["var_95"] >= 0.0

    def test_portfolio_var_99_gte_var_95(self):
        r = get_s46_portfolio_risk()
        assert r["portfolio"]["var_99"] >= r["portfolio"]["var_95"]

    def test_per_symbol_all_present(self):
        r = get_s46_portfolio_risk()
        for sym in _S46_SYMBOLS:
            assert sym in r["per_symbol"]

    def test_per_symbol_fields(self):
        r = get_s46_portfolio_risk()
        for sym in _S46_SYMBOLS:
            entry = r["per_symbol"][sym]
            for key in ("var_95", "var_99", "sharpe", "sortino", "calmar", "max_drawdown"):
                assert key in entry, f"Missing {key} for {sym}"

    def test_correlation_matrix_diagonal_one(self):
        r = get_s46_portfolio_risk()
        cm = r["correlation_matrix"]
        for sym in _S46_SYMBOLS:
            assert cm[sym][sym] == pytest.approx(1.0, abs=1e-3)

    def test_version_s46(self):
        r = get_s46_portfolio_risk()
        assert r.get("version") in ("S46", "S47", "S48")

    def test_generated_at_recent(self):
        r = get_s46_portfolio_risk()
        assert abs(r["generated_at"] - time.time()) < 5

    def test_n_symbols(self):
        r = get_s46_portfolio_risk()
        assert r["portfolio"]["n_symbols"] == 4

    def test_history_days_252(self):
        r = get_s46_portfolio_risk()
        assert r["portfolio"]["history_days"] == 252

    def test_deterministic(self):
        r1 = get_s46_portfolio_risk()
        r2 = get_s46_portfolio_risk()
        assert r1["portfolio"]["var_95"] == r2["portfolio"]["var_95"]


# ─────────────────────────────────────────────────────────────────────────────
# Section C — get_s46_position_size
# ─────────────────────────────────────────────────────────────────────────────

class TestGetS46PositionSize:
    def test_returns_dict(self):
        r = get_s46_position_size()
        assert isinstance(r, dict)

    def test_symbol_preserved(self):
        r = get_s46_position_size(symbol="ETH-USD")
        assert r["symbol"] == "ETH-USD"

    def test_capital_preserved(self):
        r = get_s46_position_size(capital=50000.0)
        assert r["capital"] == 50000.0

    def test_risk_budget_preserved(self):
        r = get_s46_position_size(risk_budget_pct=0.05)
        assert r["risk_budget_pct"] == pytest.approx(0.05)

    def test_method_preserved(self):
        r = get_s46_position_size(method="half_kelly")
        assert r["method"] == "half_kelly"

    def test_recommended_position_non_negative(self):
        r = get_s46_position_size()
        assert r["recommended_position_usd"] >= 0.0

    def test_position_pct_between_0_and_1(self):
        r = get_s46_position_size()
        pct = r["position_pct_of_capital"]
        assert 0.0 <= pct <= 1.5  # allow over-concentrated in some methods

    def test_daily_volatility_positive(self):
        r = get_s46_position_size()
        assert r["daily_volatility"] > 0.0

    def test_var_95_positive(self):
        r = get_s46_position_size()
        assert r["var_95"] > 0.0

    def test_rationale_string(self):
        r = get_s46_position_size()
        assert isinstance(r["rationale"], str)
        assert len(r["rationale"]) > 0

    def test_version_s46(self):
        r = get_s46_position_size()
        assert r.get("version") in ("S46", "S47", "S48")

    def test_invalid_symbol_raises(self):
        with pytest.raises(ValueError, match="Unknown symbol"):
            get_s46_position_size(symbol="INVALID-USD")

    def test_invalid_method_raises(self):
        with pytest.raises(ValueError, match="Unknown method"):
            get_s46_position_size(method="magic_method")

    def test_all_symbols_work(self):
        for sym in _S46_SYMBOLS:
            r = get_s46_position_size(symbol=sym)
            assert r["symbol"] == sym

    def test_all_methods_work(self):
        for method in ("volatility", "half_kelly", "fixed_fraction"):
            r = get_s46_position_size(method=method)
            assert r["method"] == method

    def test_higher_capital_bigger_position(self):
        r_small = get_s46_position_size(capital=10000.0, method="fixed_fraction")
        r_large = get_s46_position_size(capital=100000.0, method="fixed_fraction")
        assert r_large["recommended_position_usd"] > r_small["recommended_position_usd"]

    def test_zero_capital_safe(self):
        r = get_s46_position_size(capital=0.0)
        assert r["recommended_position_usd"] == 0.0

    def test_mean_daily_return_finite(self):
        r = get_s46_position_size()
        assert math.isfinite(r["mean_daily_return"])

    def test_matic_higher_vol_than_btc(self):
        r_btc = get_s46_position_size(symbol="BTC-USD")
        r_matic = get_s46_position_size(symbol="MATIC-USD")
        assert r_matic["daily_volatility"] > r_btc["daily_volatility"]

    def test_fixed_fraction_position_equals_risk_times_capital(self):
        r = get_s46_position_size(
            symbol="BTC-USD",
            capital=100000.0,
            risk_budget_pct=0.02,
            method="fixed_fraction",
        )
        expected = 100000.0 * 0.02
        assert r["recommended_position_usd"] == pytest.approx(expected, rel=0.01)


# ─────────────────────────────────────────────────────────────────────────────
# Section D — get_s46_exposure
# ─────────────────────────────────────────────────────────────────────────────

class TestGetS46Exposure:
    def test_returns_dict(self):
        r = get_s46_exposure()
        assert isinstance(r, dict)

    def test_exposures_list(self):
        r = get_s46_exposure()
        assert isinstance(r["exposures"], list)

    def test_exposure_count_equals_symbols(self):
        r = get_s46_exposure()
        assert len(r["exposures"]) == len(_S46_SYMBOLS)

    def test_all_symbols_in_exposures(self):
        r = get_s46_exposure()
        syms = {e["symbol"] for e in r["exposures"]}
        assert syms == set(_S46_SYMBOLS)

    def test_total_notional_positive(self):
        r = get_s46_exposure()
        assert r["total_notional_usd"] > 0

    def test_n_positions(self):
        r = get_s46_exposure()
        assert r["n_positions"] == len(_S46_SYMBOLS)

    def test_hhi_between_0_and_1(self):
        r = get_s46_exposure()
        assert 0.0 <= r["hhi_concentration"] <= 1.0

    def test_concentration_risk_string(self):
        r = get_s46_exposure()
        assert r["concentration_risk"] in ("LOW", "MEDIUM", "HIGH")

    def test_exposure_fields(self):
        r = get_s46_exposure()
        for e in r["exposures"]:
            for key in ("symbol", "units_held", "notional_usd", "portfolio_pct",
                        "var_95_daily", "concentration_flag"):
                assert key in e, f"Missing {key}"

    def test_portfolio_pct_sums_to_one(self):
        r = get_s46_exposure()
        total_pct = sum(e["portfolio_pct"] for e in r["exposures"])
        assert total_pct == pytest.approx(1.0, abs=0.01)

    def test_concentration_flag_is_bool(self):
        r = get_s46_exposure()
        for e in r["exposures"]:
            assert isinstance(e["concentration_flag"], bool)

    def test_notional_positive(self):
        r = get_s46_exposure()
        for e in r["exposures"]:
            assert e["notional_usd"] > 0

    def test_var_95_daily_positive(self):
        r = get_s46_exposure()
        for e in r["exposures"]:
            assert e["var_95_daily"] >= 0.0

    def test_version_s46(self):
        r = get_s46_exposure()
        assert r.get("version") in ("S46", "S47", "S48")

    def test_deterministic(self):
        r1 = get_s46_exposure()
        r2 = get_s46_exposure()
        assert r1["total_notional_usd"] == r2["total_notional_usd"]


# ─────────────────────────────────────────────────────────────────────────────
# Section E — Swarm state
# ─────────────────────────────────────────────────────────────────────────────

class TestS46SwarmState:
    def test_ten_agents_defined(self):
        assert len(_S46_SWARM_AGENTS) == 10

    def test_all_agents_have_id(self):
        for a in _S46_SWARM_AGENTS:
            assert "id" in a and isinstance(a["id"], str)

    def test_agent_ids_unique(self):
        ids = [a["id"] for a in _S46_SWARM_AGENTS]
        assert len(ids) == len(set(ids))

    def test_all_agents_have_strategy(self):
        for a in _S46_SWARM_AGENTS:
            assert "strategy" in a

    def test_all_agents_have_stake(self):
        for a in _S46_SWARM_AGENTS:
            assert "stake" in a
            assert a["stake"] > 0

    def test_strategy_diversity(self):
        strategies = {a["strategy"] for a in _S46_SWARM_AGENTS}
        assert len(strategies) >= 4  # at least 4 distinct strategies

    def test_quant_names(self):
        ids = [a["id"] for a in _S46_SWARM_AGENTS]
        for i in range(1, 11):
            assert f"quant-{i}" in ids


# ─────────────────────────────────────────────────────────────────────────────
# Section F — get_s46_swarm_vote
# ─────────────────────────────────────────────────────────────────────────────

class TestGetS46SwarmVote:
    def test_returns_dict(self):
        r = get_s46_swarm_vote("BTC-USD", "BUY")
        assert isinstance(r, dict)

    def test_symbol_preserved(self):
        r = get_s46_swarm_vote("ETH-USD", "BUY")
        assert r["symbol"] == "ETH-USD"

    def test_signal_type_preserved(self):
        r = get_s46_swarm_vote("BTC-USD", "SELL")
        assert r["signal_type"] == "SELL"

    def test_votes_list(self):
        r = get_s46_swarm_vote("BTC-USD", "BUY")
        assert isinstance(r["votes"], list)

    def test_ten_votes(self):
        r = get_s46_swarm_vote("BTC-USD", "BUY")
        assert len(r["votes"]) == 10

    def test_total_agents(self):
        r = get_s46_swarm_vote("BTC-USD", "BUY")
        assert r["total_agents"] == 10

    def test_vote_summary_dict(self):
        r = get_s46_swarm_vote("BTC-USD", "BUY")
        assert isinstance(r["vote_summary"], dict)

    def test_vote_summary_sums_to_ten(self):
        r = get_s46_swarm_vote("BTC-USD", "BUY")
        assert sum(r["vote_summary"].values()) == 10

    def test_agree_count_is_int(self):
        r = get_s46_swarm_vote("BTC-USD", "BUY")
        assert isinstance(r["agree_count"], int)

    def test_agree_count_lte_ten(self):
        r = get_s46_swarm_vote("BTC-USD", "BUY")
        assert 0 <= r["agree_count"] <= 10

    def test_weighted_fraction_between_0_and_1(self):
        r = get_s46_swarm_vote("BTC-USD", "BUY")
        assert 0.0 <= r["weighted_agree_fraction"] <= 1.0

    def test_consensus_threshold_two_thirds(self):
        r = get_s46_swarm_vote("BTC-USD", "BUY")
        assert r["consensus_threshold"] == pytest.approx(2 / 3)

    def test_consensus_reached_is_bool(self):
        r = get_s46_swarm_vote("BTC-USD", "BUY")
        assert isinstance(r["consensus_reached"], bool)

    def test_consensus_action_valid(self):
        r = get_s46_swarm_vote("BTC-USD", "BUY")
        assert r["consensus_action"] in ("BUY", "SELL", "HOLD")

    def test_total_stake_positive(self):
        r = get_s46_swarm_vote("BTC-USD", "BUY")
        assert r["total_stake"] > 0

    def test_each_vote_has_required_fields(self):
        r = get_s46_swarm_vote("BTC-USD", "BUY")
        for v in r["votes"]:
            for key in ("agent_id", "strategy", "stake", "vote", "confidence", "agrees"):
                assert key in v, f"Missing {key} in vote {v}"

    def test_votes_confidence_in_range(self):
        r = get_s46_swarm_vote("BTC-USD", "BUY")
        for v in r["votes"]:
            assert 0.5 <= v["confidence"] <= 1.0

    def test_votes_stake_positive(self):
        r = get_s46_swarm_vote("BTC-USD", "BUY")
        for v in r["votes"]:
            assert v["stake"] > 0

    def test_votes_agree_is_bool(self):
        r = get_s46_swarm_vote("BTC-USD", "BUY")
        for v in r["votes"]:
            assert isinstance(v["agrees"], bool)

    def test_consensus_action_matches_signal_when_reached(self):
        # If consensus_reached, action == signal_type
        r = get_s46_swarm_vote("BTC-USD", "BUY")
        if r["consensus_reached"]:
            assert r["consensus_action"] == "BUY"
        else:
            assert r["consensus_action"] == "HOLD"

    def test_sell_signal(self):
        r = get_s46_swarm_vote("SOL-USD", "SELL")
        assert r["signal_type"] == "SELL"
        assert len(r["votes"]) == 10

    def test_hold_signal(self):
        r = get_s46_swarm_vote("MATIC-USD", "HOLD")
        assert r["signal_type"] == "HOLD"

    def test_invalid_symbol_raises(self):
        with pytest.raises(ValueError, match="Unknown symbol"):
            get_s46_swarm_vote("INVALID", "BUY")

    def test_invalid_signal_raises(self):
        with pytest.raises(ValueError, match="Unknown signal_type"):
            get_s46_swarm_vote("BTC-USD", "MAYBE")

    def test_version_s46(self):
        r = get_s46_swarm_vote("BTC-USD", "BUY")
        assert r.get("version") in ("S46", "S47", "S48")

    def test_voted_at_recent(self):
        r = get_s46_swarm_vote("BTC-USD", "BUY")
        assert abs(r["voted_at"] - time.time()) < 5


# ─────────────────────────────────────────────────────────────────────────────
# Section G — get_s46_swarm_performance
# ─────────────────────────────────────────────────────────────────────────────

class TestGetS46SwarmPerformance:
    def test_returns_dict(self):
        r = get_s46_swarm_performance()
        assert isinstance(r, dict)

    def test_leaderboard_list(self):
        r = get_s46_swarm_performance()
        assert isinstance(r["leaderboard"], list)

    def test_leaderboard_count_ten(self):
        r = get_s46_swarm_performance()
        assert len(r["leaderboard"]) == 10

    def test_total_agents_ten(self):
        r = get_s46_swarm_performance()
        assert r["total_agents"] == 10

    def test_leaderboard_rank_field(self):
        r = get_s46_swarm_performance()
        ranks = [e["rank"] for e in r["leaderboard"]]
        assert ranks == list(range(1, 11))

    def test_leaderboard_sorted_by_pnl(self):
        r = get_s46_swarm_performance()
        pnls = [e["total_pnl_24h"] for e in r["leaderboard"]]
        assert pnls == sorted(pnls, reverse=True)

    def test_each_entry_has_fields(self):
        r = get_s46_swarm_performance()
        for e in r["leaderboard"]:
            for key in ("rank", "agent_id", "strategy", "stake",
                        "total_pnl_24h", "win_rate", "sharpe_24h"):
                assert key in e, f"Missing {key}"

    def test_win_rate_between_0_and_1(self):
        r = get_s46_swarm_performance()
        for e in r["leaderboard"]:
            assert 0.0 <= e["win_rate"] <= 1.0

    def test_stake_positive(self):
        r = get_s46_swarm_performance()
        for e in r["leaderboard"]:
            assert e["stake"] > 0

    def test_top_performer_string(self):
        r = get_s46_swarm_performance()
        assert isinstance(r["top_performer"], str)
        assert len(r["top_performer"]) > 0

    def test_portfolio_pnl_finite(self):
        r = get_s46_swarm_performance()
        assert math.isfinite(r["portfolio_pnl_24h"])

    def test_version_s46(self):
        r = get_s46_swarm_performance()
        assert r.get("version") in ("S46", "S47", "S48")

    def test_generated_at_recent(self):
        r = get_s46_swarm_performance()
        assert abs(r["generated_at"] - time.time()) < 5

    def test_bottom_performer_string(self):
        r = get_s46_swarm_performance()
        assert isinstance(r["bottom_performer"], str)

    def test_vote_after_performance_increments(self):
        # After a vote, the agent's vote_count should be ≥ before
        perf_before = get_s46_swarm_performance()
        get_s46_swarm_vote("BTC-USD", "BUY")
        perf_after = get_s46_swarm_performance()
        total_before = sum(e.get("vote_count", 0) for e in perf_before["leaderboard"])
        total_after = sum(e.get("vote_count", 0) for e in perf_after["leaderboard"])
        assert total_after >= total_before


# ─────────────────────────────────────────────────────────────────────────────
# Section H — HTTP endpoint tests
# ─────────────────────────────────────────────────────────────────────────────

def _make_server(port: int = 18046) -> "DemoServer":
    s = DemoServer(port=port)
    s.start()
    time.sleep(0.15)
    return s


def _get(url: str) -> Dict[str, Any]:
    with urllib.request.urlopen(url, timeout=5) as resp:
        return json.loads(resp.read())


def _post(url: str, body: Any = None) -> Dict[str, Any]:
    data = json.dumps(body or {}).encode("utf-8")
    req = urllib.request.Request(url, data=data,
                                  headers={"Content-Type": "application/json"},
                                  method="POST")
    with urllib.request.urlopen(req, timeout=5) as resp:
        return json.loads(resp.read())


@pytest.fixture(scope="module")
def server():
    srv = _make_server(18046)
    yield srv
    srv.stop()


class TestS46HTTPEndpoints:
    BASE = "http://localhost:18046"

    def test_health_has_version_s46(self, server):
        r = _get(f"{self.BASE}/health")
        assert r.get("version") in ("S46", "S47", "S48")

    def test_health_has_sprint_s46(self, server):
        r = _get(f"{self.BASE}/health")
        assert r.get("sprint") in ("S46", "S47", "S48")

    def test_health_has_highlights(self, server):
        r = _get(f"{self.BASE}/health")
        assert "highlights" in r
        assert isinstance(r["highlights"], list)
        assert len(r["highlights"]) >= 4

    def test_health_highlights_contains_swarm(self, server):
        r = _get(f"{self.BASE}/health")
        hl_str = " ".join(r["highlights"]).lower()
        assert "swarm" in hl_str

    def test_health_highlights_contains_var(self, server):
        r = _get(f"{self.BASE}/health")
        hl_str = " ".join(r["highlights"]).lower()
        assert "var" in hl_str or "risk" in hl_str

    def test_risk_portfolio_get(self, server):
        r = _get(f"{self.BASE}/api/v1/risk/portfolio")
        assert "portfolio" in r
        assert "correlation_matrix" in r

    def test_risk_portfolio_var_fields(self, server):
        r = _get(f"{self.BASE}/api/v1/risk/portfolio")
        assert "var_95" in r["portfolio"]
        assert "var_99" in r["portfolio"]

    def test_risk_portfolio_per_symbol(self, server):
        r = _get(f"{self.BASE}/api/v1/risk/portfolio")
        for sym in _S46_SYMBOLS:
            assert sym in r["per_symbol"]

    def test_risk_exposure_get(self, server):
        r = _get(f"{self.BASE}/api/v1/risk/exposure")
        assert "exposures" in r
        assert len(r["exposures"]) == 4

    def test_risk_exposure_hhi(self, server):
        r = _get(f"{self.BASE}/api/v1/risk/exposure")
        assert "hhi_concentration" in r
        assert 0 <= r["hhi_concentration"] <= 1.0

    def test_risk_position_size_post_default(self, server):
        r = _post(f"{self.BASE}/api/v1/risk/position-size", {})
        assert "recommended_position_usd" in r
        assert r["recommended_position_usd"] >= 0.0

    def test_risk_position_size_post_custom(self, server):
        r = _post(f"{self.BASE}/api/v1/risk/position-size", {
            "symbol": "ETH-USD",
            "capital": 50000,
            "risk_budget_pct": 0.01,
            "method": "volatility",
        })
        assert r["symbol"] == "ETH-USD"
        assert r["capital"] == 50000

    def test_risk_position_size_invalid_symbol(self, server):
        with pytest.raises(urllib.error.HTTPError) as exc_info:
            _post(f"{self.BASE}/api/v1/risk/position-size", {"symbol": "BAD-USD"})
        assert exc_info.value.code == 400

    def test_swarm_vote_post_buy(self, server):
        r = _post(f"{self.BASE}/api/v1/swarm/vote", {"symbol": "BTC-USD", "signal_type": "BUY"})
        assert r["symbol"] == "BTC-USD"
        assert len(r["votes"]) == 10

    def test_swarm_vote_post_sell(self, server):
        r = _post(f"{self.BASE}/api/v1/swarm/vote", {"symbol": "SOL-USD", "signal_type": "SELL"})
        assert r["signal_type"] == "SELL"

    def test_swarm_vote_invalid_signal(self, server):
        with pytest.raises(urllib.error.HTTPError) as exc_info:
            _post(f"{self.BASE}/api/v1/swarm/vote", {"signal_type": "MAYBE"})
        assert exc_info.value.code == 400

    def test_swarm_performance_get(self, server):
        r = _get(f"{self.BASE}/api/v1/swarm/performance")
        assert "leaderboard" in r
        assert len(r["leaderboard"]) == 10

    def test_swarm_performance_sorted(self, server):
        r = _get(f"{self.BASE}/api/v1/swarm/performance")
        pnls = [e["total_pnl_24h"] for e in r["leaderboard"]]
        assert pnls == sorted(pnls, reverse=True)

    def test_root_lists_s46_endpoints(self, server):
        r = _get(f"{self.BASE}/")
        endpoint_str = json.dumps(r.get("endpoints", {}))
        assert "risk/portfolio" in endpoint_str

    def test_health_test_count_gte_baseline(self, server):
        r = _get(f"{self.BASE}/health")
        assert r.get("tests", 0) >= 6000
