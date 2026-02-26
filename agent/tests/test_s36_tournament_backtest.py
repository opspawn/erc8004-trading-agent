"""
test_s36_tournament_backtest.py — Sprint 36: Multi-Agent Tournament + Strategy Backtester UI.

Covers:
  - run_tournament(): bracket structure, winner, rankings, stats, determinism,
    edge cases (2 agents, many agents, extreme params, bad inputs)
  - run_backtest_ui(): equity curve, metrics schema, determinism, all strategy types,
    param variations, edge cases (bad inputs)
  - get_market_regime(): schema, regime validity, history, determinism
  - get_agent_attribution(): schema, component sum, suggestions, determinism
  - HTTP integration: POST /demo/tournament/run, POST /demo/backtest/run,
    GET /demo/market/regime, GET /demo/agents/{id}/attribution
  - Error cases: missing fields, invalid types, bad JSON, 404 routing
"""

from __future__ import annotations

import json
import math
import socket
import sys
import os
import time
import threading
from http.client import HTTPConnection
from typing import Any, Dict
from urllib.request import urlopen, Request
from urllib.error import HTTPError

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from demo_server import (
    # S36 functions
    run_tournament,
    run_backtest_ui,
    get_market_regime,
    get_agent_attribution,
    _simulate_match,
    _agent_strategy_for_tournament,
    # Server
    DemoServer,
)


# ── Helpers ────────────────────────────────────────────────────────────────────

def _get(url: str) -> dict:
    with urlopen(url, timeout=5) as resp:
        return json.loads(resp.read())


def _post(url: str, data: dict, expect_error: bool = False):
    body = json.dumps(data).encode("utf-8")
    req = Request(url, data=body, headers={"Content-Type": "application/json"})
    try:
        with urlopen(req, timeout=10) as resp:
            return json.loads(resp.read())
    except HTTPError as exc:
        if expect_error:
            return json.loads(exc.read())
        raise


def _post_raw(url: str, data: bytes, headers: Dict[str, str]) -> tuple[int, dict]:
    req = Request(url, data=data, headers=headers)
    try:
        with urlopen(req, timeout=10) as resp:
            return resp.status, json.loads(resp.read())
    except HTTPError as exc:
        return exc.code, json.loads(exc.read())


# ── Fixtures ───────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def server():
    """Start a DemoServer on a free port and yield (base_url, port)."""
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    sock.close()

    srv = DemoServer(port=port)
    srv.start()
    time.sleep(0.2)
    base = f"http://127.0.0.1:{port}"
    yield base, port
    srv.stop()


# ═══════════════════════════════════════════════════════════════════════════════
# 1. _agent_strategy_for_tournament
# ═══════════════════════════════════════════════════════════════════════════════

class TestAgentStrategyForTournament:
    def test_returns_string(self):
        s = _agent_strategy_for_tournament("agent-x")
        assert isinstance(s, str)

    def test_returns_valid_strategy(self):
        valid = {"momentum", "mean_reversion", "arbitrage", "ml_hybrid",
                 "conservative", "balanced", "aggressive"}
        s = _agent_strategy_for_tournament("agent-x")
        assert s in valid

    def test_deterministic(self):
        s1 = _agent_strategy_for_tournament("agent-alpha")
        s2 = _agent_strategy_for_tournament("agent-alpha")
        assert s1 == s2

    def test_different_agents_may_differ(self):
        strategies = {_agent_strategy_for_tournament(f"agent-{i}") for i in range(20)}
        # Should have more than one distinct strategy among 20 agents
        assert len(strategies) > 1

    def test_empty_agent_id(self):
        s = _agent_strategy_for_tournament("")
        assert isinstance(s, str)


# ═══════════════════════════════════════════════════════════════════════════════
# 2. _simulate_match
# ═══════════════════════════════════════════════════════════════════════════════

class TestSimulateMatch:
    def _default_mc(self):
        return {"volatility": 0.02, "trend": 0.0}

    def test_returns_dict(self):
        result = _simulate_match("a1", "a2", 1.0, self._default_mc(), 42)
        assert isinstance(result, dict)

    def test_schema(self):
        result = _simulate_match("a1", "a2", 1.0, self._default_mc(), 42)
        required = {"agent_a", "agent_b", "strategy_a", "strategy_b",
                    "agent_a_pnl", "agent_b_pnl", "trades_a", "trades_b",
                    "winner", "loser", "winner_sharpe", "duration_hours"}
        assert required.issubset(result.keys())

    def test_winner_is_one_of_agents(self):
        result = _simulate_match("alpha", "beta", 2.0, self._default_mc(), 99)
        assert result["winner"] in ("alpha", "beta")

    def test_loser_is_other_agent(self):
        result = _simulate_match("alpha", "beta", 2.0, self._default_mc(), 99)
        if result["winner"] == "alpha":
            assert result["loser"] == "beta"
        else:
            assert result["loser"] == "alpha"

    def test_winner_loser_disjoint(self):
        result = _simulate_match("x", "y", 1.0, self._default_mc(), 7)
        assert result["winner"] != result["loser"]

    def test_trades_positive(self):
        result = _simulate_match("x", "y", 1.0, self._default_mc(), 7)
        assert result["trades_a"] > 0
        assert result["trades_b"] > 0

    def test_duration_hours_preserved(self):
        result = _simulate_match("x", "y", 3.5, self._default_mc(), 1)
        assert result["duration_hours"] == 3.5

    def test_deterministic_same_seed(self):
        r1 = _simulate_match("x", "y", 2.0, self._default_mc(), 123)
        r2 = _simulate_match("x", "y", 2.0, self._default_mc(), 123)
        assert r1["winner"] == r2["winner"]
        assert r1["agent_a_pnl"] == r2["agent_a_pnl"]

    def test_different_seeds_can_differ(self):
        r1 = _simulate_match("x", "y", 2.0, self._default_mc(), 1)
        r2 = _simulate_match("x", "y", 2.0, self._default_mc(), 999)
        # Different seeds — at least one value should differ
        assert r1 != r2

    def test_high_volatility(self):
        result = _simulate_match("x", "y", 1.0, {"volatility": 0.5, "trend": 0.0}, 42)
        # Should still produce valid output
        assert result["winner"] in ("x", "y")

    def test_positive_trend_favors_long_pnl(self):
        # Trending market: both agents should generally have higher pnl than crisis
        result_up = _simulate_match("x", "y", 10.0, {"volatility": 0.01, "trend": 1.0}, 42)
        result_dn = _simulate_match("x", "y", 10.0, {"volatility": 0.01, "trend": -1.0}, 42)
        # At least one agent benefits from positive trend
        avg_pnl_up = (result_up["agent_a_pnl"] + result_up["agent_b_pnl"]) / 2
        avg_pnl_dn = (result_dn["agent_a_pnl"] + result_dn["agent_b_pnl"]) / 2
        assert avg_pnl_up > avg_pnl_dn


# ═══════════════════════════════════════════════════════════════════════════════
# 3. run_tournament
# ═══════════════════════════════════════════════════════════════════════════════

AGENTS_3 = ["agent-alpha", "agent-beta", "agent-gamma"]
AGENTS_6 = [f"agent-{i}" for i in range(6)]
MC_DEFAULT = {"volatility": 0.02, "trend": 0.1}


class TestRunTournament:
    def test_returns_dict(self):
        result = run_tournament(AGENTS_3, 1.0, MC_DEFAULT)
        assert isinstance(result, dict)

    def test_top_level_keys(self):
        result = run_tournament(AGENTS_3, 1.0, MC_DEFAULT)
        required = {"bracket", "winner", "winner_strategy", "final_rankings",
                    "tournament_stats", "market_conditions", "generated_at"}
        assert required.issubset(result.keys())

    def test_winner_is_one_of_agents(self):
        result = run_tournament(AGENTS_3, 1.0, MC_DEFAULT)
        assert result["winner"] in AGENTS_3

    def test_winner_strategy_valid(self):
        valid = {"momentum", "mean_reversion", "arbitrage", "ml_hybrid",
                 "conservative", "balanced", "aggressive"}
        result = run_tournament(AGENTS_3, 1.0, MC_DEFAULT)
        assert result["winner_strategy"] in valid

    def test_final_rankings_count(self):
        result = run_tournament(AGENTS_3, 1.0, MC_DEFAULT)
        assert len(result["final_rankings"]) == len(AGENTS_3)

    def test_final_rankings_schema(self):
        result = run_tournament(AGENTS_3, 1.0, MC_DEFAULT)
        for entry in result["final_rankings"]:
            assert "rank" in entry
            assert "agent_id" in entry
            assert "strategy" in entry
            assert "qualifying_wins" in entry
            assert "total_pnl" in entry
            assert "total_trades" in entry

    def test_final_rankings_sorted_by_rank(self):
        result = run_tournament(AGENTS_3, 1.0, MC_DEFAULT)
        ranks = [e["rank"] for e in result["final_rankings"]]
        assert ranks == sorted(ranks)

    def test_bracket_has_qualifying(self):
        result = run_tournament(AGENTS_3, 1.0, MC_DEFAULT)
        assert "qualifying" in result["bracket"]
        assert isinstance(result["bracket"]["qualifying"], list)
        assert len(result["bracket"]["qualifying"]) > 0

    def test_bracket_has_semifinals(self):
        result = run_tournament(AGENTS_3, 1.0, MC_DEFAULT)
        assert "semifinals" in result["bracket"]
        assert isinstance(result["bracket"]["semifinals"], list)

    def test_bracket_has_final(self):
        result = run_tournament(AGENTS_3, 1.0, MC_DEFAULT)
        assert "final" in result["bracket"]
        final = result["bracket"]["final"]
        assert isinstance(final, dict)
        assert "winner" in final

    def test_tournament_stats_schema(self):
        result = run_tournament(AGENTS_3, 1.0, MC_DEFAULT)
        stats = result["tournament_stats"]
        assert "total_matches" in stats
        assert "total_trades" in stats
        assert "avg_pnl" in stats
        assert "volatility" in stats
        assert "duration_hours" in stats
        assert "agent_count" in stats

    def test_tournament_stats_agent_count(self):
        result = run_tournament(AGENTS_3, 1.0, MC_DEFAULT)
        assert result["tournament_stats"]["agent_count"] == len(AGENTS_3)

    def test_tournament_stats_total_trades_positive(self):
        result = run_tournament(AGENTS_3, 1.0, MC_DEFAULT)
        assert result["tournament_stats"]["total_trades"] > 0

    def test_market_conditions_preserved(self):
        result = run_tournament(AGENTS_3, 1.0, MC_DEFAULT)
        assert result["market_conditions"] == MC_DEFAULT

    def test_deterministic_same_inputs(self):
        r1 = run_tournament(AGENTS_3, 1.0, MC_DEFAULT)
        r2 = run_tournament(AGENTS_3, 1.0, MC_DEFAULT)
        assert r1["winner"] == r2["winner"]
        assert r1["final_rankings"] == r2["final_rankings"]

    def test_two_agents_minimum(self):
        result = run_tournament(["a1", "a2"], 1.0, {})
        assert result["winner"] in ("a1", "a2")

    def test_six_agents(self):
        result = run_tournament(AGENTS_6, 2.0, MC_DEFAULT)
        assert result["winner"] in AGENTS_6
        assert len(result["final_rankings"]) == 6

    def test_deduplication_of_agent_ids(self):
        # Duplicate agents should be deduplicated
        result = run_tournament(["a", "b", "a", "b"], 1.0, {})
        assert result["tournament_stats"]["agent_count"] == 2

    def test_invalid_agent_ids_not_list(self):
        with pytest.raises(ValueError, match="agent_ids"):
            run_tournament("agent-alpha", 1.0, {})

    def test_invalid_agent_ids_too_few(self):
        with pytest.raises(ValueError, match="agent_ids"):
            run_tournament(["only-one"], 1.0, {})

    def test_invalid_duration_zero(self):
        with pytest.raises(ValueError, match="duration_hours"):
            run_tournament(AGENTS_3, 0.0, {})

    def test_invalid_duration_negative(self):
        with pytest.raises(ValueError, match="duration_hours"):
            run_tournament(AGENTS_3, -1.0, {})

    def test_invalid_market_conditions_not_dict(self):
        with pytest.raises(ValueError, match="market_conditions"):
            run_tournament(AGENTS_3, 1.0, "not-a-dict")

    def test_generated_at_is_float(self):
        result = run_tournament(AGENTS_3, 1.0, MC_DEFAULT)
        assert isinstance(result["generated_at"], float)

    def test_empty_market_conditions(self):
        result = run_tournament(AGENTS_3, 1.0, {})
        assert result["winner"] in AGENTS_3

    def test_qualifying_match_count_3_agents(self):
        # C(3,2) = 3 qualifying matches
        result = run_tournament(AGENTS_3, 1.0, MC_DEFAULT)
        assert len(result["bracket"]["qualifying"]) == 3

    def test_qualifying_match_count_6_agents(self):
        # C(6,2) = 15 qualifying matches
        result = run_tournament(AGENTS_6, 1.0, MC_DEFAULT)
        assert len(result["bracket"]["qualifying"]) == 15

    def test_duration_hours_in_stats(self):
        result = run_tournament(AGENTS_3, 3.5, MC_DEFAULT)
        assert result["tournament_stats"]["duration_hours"] == 3.5


# ═══════════════════════════════════════════════════════════════════════════════
# 4. run_backtest_ui
# ═══════════════════════════════════════════════════════════════════════════════

class TestRunBacktestUi:
    def _default_config(self):
        return {"type": "momentum", "params": {"window": 10, "threshold": 0.01, "leverage": 1.0}}

    def test_returns_dict(self):
        result = run_backtest_ui(self._default_config(), 90, ["BTC/USD"])
        assert isinstance(result, dict)

    def test_top_level_keys(self):
        result = run_backtest_ui(self._default_config(), 90, ["BTC/USD"])
        required = {"strategy_config", "lookback_days", "assets", "equity_curve",
                    "sharpe_ratio", "max_drawdown_pct", "win_rate", "profit_factor",
                    "total_return_pct", "benchmark_return_pct", "alpha", "beta",
                    "total_trades", "generated_at"}
        assert required.issubset(result.keys())

    def test_equity_curve_is_list(self):
        result = run_backtest_ui(self._default_config(), 30, ["BTC/USD"])
        assert isinstance(result["equity_curve"], list)
        assert len(result["equity_curve"]) >= 2

    def test_equity_curve_starts_near_10000(self):
        result = run_backtest_ui(self._default_config(), 30, ["BTC/USD"])
        # First value should be close to initial capital 10000
        assert 9000 <= result["equity_curve"][0] <= 11000

    def test_equity_curve_values_positive(self):
        result = run_backtest_ui(self._default_config(), 90, ["BTC/USD"])
        for v in result["equity_curve"]:
            assert v > 0

    def test_sharpe_ratio_is_float(self):
        result = run_backtest_ui(self._default_config(), 90, ["BTC/USD"])
        assert isinstance(result["sharpe_ratio"], float)

    def test_max_drawdown_non_negative(self):
        result = run_backtest_ui(self._default_config(), 90, ["BTC/USD"])
        assert result["max_drawdown_pct"] >= 0

    def test_win_rate_in_range(self):
        result = run_backtest_ui(self._default_config(), 90, ["BTC/USD"])
        assert 0.0 <= result["win_rate"] <= 1.0

    def test_profit_factor_positive(self):
        result = run_backtest_ui(self._default_config(), 90, ["BTC/USD"])
        assert result["profit_factor"] > 0

    def test_total_trades_positive(self):
        result = run_backtest_ui(self._default_config(), 90, ["BTC/USD"])
        assert result["total_trades"] > 0

    def test_benchmark_return_pct_is_float(self):
        result = run_backtest_ui(self._default_config(), 90, ["BTC/USD"])
        assert isinstance(result["benchmark_return_pct"], float)

    def test_alpha_is_float(self):
        result = run_backtest_ui(self._default_config(), 90, ["BTC/USD"])
        assert isinstance(result["alpha"], float)

    def test_beta_is_float(self):
        result = run_backtest_ui(self._default_config(), 90, ["BTC/USD"])
        assert isinstance(result["beta"], float)

    def test_strategy_types_all_work(self):
        for strat in ["momentum", "mean_reversion", "arbitrage", "ml_hybrid"]:
            cfg = {"type": strat, "params": {"window": 10, "threshold": 0.01, "leverage": 1.0}}
            result = run_backtest_ui(cfg, 30, ["BTC/USD"])
            assert result["strategy_config"]["type"] == strat

    def test_deterministic_same_inputs(self):
        cfg = self._default_config()
        r1 = run_backtest_ui(cfg, 60, ["ETH/USD"])
        r2 = run_backtest_ui(cfg, 60, ["ETH/USD"])
        assert r1["total_return_pct"] == r2["total_return_pct"]
        assert r1["equity_curve"] == r2["equity_curve"]

    def test_different_strategy_different_result(self):
        c1 = {"type": "momentum", "params": {}}
        c2 = {"type": "arbitrage", "params": {}}
        r1 = run_backtest_ui(c1, 60, ["BTC/USD"])
        r2 = run_backtest_ui(c2, 60, ["BTC/USD"])
        assert r1["total_return_pct"] != r2["total_return_pct"]

    def test_higher_leverage_changes_result(self):
        c1 = {"type": "momentum", "params": {"leverage": 1.0}}
        c2 = {"type": "momentum", "params": {"leverage": 3.0}}
        r1 = run_backtest_ui(c1, 60, ["BTC/USD"])
        r2 = run_backtest_ui(c2, 60, ["BTC/USD"])
        # Should differ due to different leverage
        assert r1["total_return_pct"] != r2["total_return_pct"]

    def test_assets_field_preserved(self):
        assets = ["BTC/USD", "ETH/USD"]
        result = run_backtest_ui(self._default_config(), 30, assets)
        assert result["assets"] == assets

    def test_lookback_days_preserved(self):
        result = run_backtest_ui(self._default_config(), 42, ["BTC/USD"])
        assert result["lookback_days"] == 42

    def test_strategy_config_in_response(self):
        result = run_backtest_ui(self._default_config(), 30, ["BTC/USD"])
        assert result["strategy_config"]["type"] == "momentum"

    def test_invalid_strategy_type(self):
        cfg = {"type": "does_not_exist", "params": {}}
        with pytest.raises(ValueError, match="Unknown strategy"):
            run_backtest_ui(cfg, 30, ["BTC/USD"])

    def test_invalid_lookback_days_zero(self):
        with pytest.raises(ValueError, match="lookback_days"):
            run_backtest_ui(self._default_config(), 0, ["BTC/USD"])

    def test_invalid_assets_empty(self):
        with pytest.raises(ValueError, match="assets"):
            run_backtest_ui(self._default_config(), 30, [])

    def test_params_default_values(self):
        cfg = {"type": "momentum", "params": {}}
        result = run_backtest_ui(cfg, 30, ["BTC/USD"])
        params = result["strategy_config"]["params"]
        assert params["window"] >= 2
        assert params["leverage"] >= 0.1

    def test_leverage_capped_at_10(self):
        cfg = {"type": "momentum", "params": {"leverage": 999.0}}
        result = run_backtest_ui(cfg, 30, ["BTC/USD"])
        assert result["strategy_config"]["params"]["leverage"] <= 10.0

    def test_long_lookback(self):
        result = run_backtest_ui(self._default_config(), 500, ["BTC/USD"])
        assert result["lookback_days"] == 500
        assert len(result["equity_curve"]) >= 2


# ═══════════════════════════════════════════════════════════════════════════════
# 5. get_market_regime
# ═══════════════════════════════════════════════════════════════════════════════

class TestGetMarketRegime:
    def test_returns_dict(self):
        result = get_market_regime()
        assert isinstance(result, dict)

    def test_top_level_keys(self):
        result = get_market_regime()
        required = {"current_regime", "regime_confidence_pct",
                    "regime_duration_hours", "optimal_strategies",
                    "regime_history", "generated_at"}
        assert required.issubset(result.keys())

    def test_current_regime_valid(self):
        valid = {"trending", "ranging", "volatile", "crisis"}
        result = get_market_regime()
        assert result["current_regime"] in valid

    def test_confidence_in_range(self):
        result = get_market_regime()
        assert 0.0 < result["regime_confidence_pct"] <= 100.0

    def test_duration_hours_positive(self):
        result = get_market_regime()
        assert result["regime_duration_hours"] > 0

    def test_optimal_strategies_is_list(self):
        result = get_market_regime()
        assert isinstance(result["optimal_strategies"], list)
        assert len(result["optimal_strategies"]) > 0

    def test_optimal_strategies_strings(self):
        result = get_market_regime()
        for s in result["optimal_strategies"]:
            assert isinstance(s, str)

    def test_regime_history_length(self):
        result = get_market_regime()
        assert len(result["regime_history"]) == 7

    def test_regime_history_schema(self):
        result = get_market_regime()
        for entry in result["regime_history"]:
            assert "date" in entry
            assert "regime" in entry
            assert "return_pct" in entry

    def test_regime_history_regimes_valid(self):
        valid = {"trending", "ranging", "volatile", "crisis"}
        result = get_market_regime()
        for entry in result["regime_history"]:
            assert entry["regime"] in valid

    def test_regime_history_return_pct_float(self):
        result = get_market_regime()
        for entry in result["regime_history"]:
            assert isinstance(entry["return_pct"], float)

    def test_regime_history_dates_sorted(self):
        result = get_market_regime()
        dates = [e["date"] for e in result["regime_history"]]
        assert dates == sorted(dates)

    def test_generated_at_is_float(self):
        result = get_market_regime()
        assert isinstance(result["generated_at"], float)

    def test_confidence_reasonable_range(self):
        result = get_market_regime()
        # Should be in 55–95% range based on seeding logic
        assert 50 <= result["regime_confidence_pct"] <= 100

    def test_duration_reasonable_range(self):
        result = get_market_regime()
        # Should be in 2–24 hours range
        assert 1 <= result["regime_duration_hours"] <= 30

    def test_optimal_strategies_matches_regime(self):
        from demo_server import _REGIME_OPTIMAL_STRATEGIES
        result = get_market_regime()
        regime = result["current_regime"]
        expected = _REGIME_OPTIMAL_STRATEGIES[regime]
        assert result["optimal_strategies"] == expected


# ═══════════════════════════════════════════════════════════════════════════════
# 6. get_agent_attribution
# ═══════════════════════════════════════════════════════════════════════════════

class TestGetAgentAttribution:
    def test_returns_dict(self):
        result = get_agent_attribution("agent-alpha")
        assert isinstance(result, dict)

    def test_top_level_keys(self):
        result = get_agent_attribution("agent-alpha")
        required = {"agent_id", "total_pnl_usd", "portfolio_value",
                    "contribution_analysis", "improvement_suggestions", "generated_at"}
        assert required.issubset(result.keys())

    def test_agent_id_preserved(self):
        result = get_agent_attribution("my-agent")
        assert result["agent_id"] == "my-agent"

    def test_contribution_analysis_keys(self):
        result = get_agent_attribution("agent-alpha")
        ca = result["contribution_analysis"]
        assert "strategy_alpha_pct" in ca
        assert "market_beta_pct" in ca
        assert "timing_pct" in ca
        assert "risk_management_pct" in ca
        assert "description" in ca

    def test_component_descriptions_are_strings(self):
        result = get_agent_attribution("agent-alpha")
        desc = result["contribution_analysis"]["description"]
        for key, val in desc.items():
            assert isinstance(val, str)

    def test_improvement_suggestions_list(self):
        result = get_agent_attribution("agent-alpha")
        assert isinstance(result["improvement_suggestions"], list)
        assert len(result["improvement_suggestions"]) >= 1

    def test_improvement_suggestions_strings(self):
        result = get_agent_attribution("agent-alpha")
        for s in result["improvement_suggestions"]:
            assert isinstance(s, str)

    def test_portfolio_value_positive(self):
        result = get_agent_attribution("agent-alpha")
        assert result["portfolio_value"] > 0

    def test_total_pnl_is_float(self):
        result = get_agent_attribution("agent-alpha")
        assert isinstance(result["total_pnl_usd"], float)

    def test_deterministic_same_agent(self):
        r1 = get_agent_attribution("agent-beta")
        r2 = get_agent_attribution("agent-beta")
        assert r1["total_pnl_usd"] == r2["total_pnl_usd"]
        assert r1["contribution_analysis"]["strategy_alpha_pct"] == \
               r2["contribution_analysis"]["strategy_alpha_pct"]

    def test_different_agents_different_results(self):
        r1 = get_agent_attribution("agent-one")
        r2 = get_agent_attribution("agent-two")
        assert r1["total_pnl_usd"] != r2["total_pnl_usd"]

    def test_generated_at_is_float(self):
        result = get_agent_attribution("agent-alpha")
        assert isinstance(result["generated_at"], float)

    def test_empty_agent_id(self):
        result = get_agent_attribution("")
        assert result["agent_id"] == ""

    def test_components_are_floats(self):
        result = get_agent_attribution("agent-alpha")
        ca = result["contribution_analysis"]
        assert isinstance(ca["strategy_alpha_pct"], float)
        assert isinstance(ca["market_beta_pct"], float)
        assert isinstance(ca["timing_pct"], float)
        assert isinstance(ca["risk_management_pct"], float)


# ═══════════════════════════════════════════════════════════════════════════════
# 7. HTTP Integration Tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestTournamentHTTP:
    def test_post_tournament_run_ok(self, server):
        base, _ = server
        result = _post(f"{base}/demo/tournament/run", {
            "agent_ids": ["alpha", "beta", "gamma"],
            "duration_hours": 1.0,
            "market_conditions": {"volatility": 0.02, "trend": 0.0},
        })
        assert "winner" in result
        assert result["winner"] in ("alpha", "beta", "gamma")

    def test_post_tournament_returns_bracket(self, server):
        base, _ = server
        result = _post(f"{base}/demo/tournament/run", {
            "agent_ids": ["x", "y", "z"],
            "duration_hours": 2.0,
            "market_conditions": {},
        })
        assert "bracket" in result
        assert "qualifying" in result["bracket"]

    def test_post_tournament_returns_final_rankings(self, server):
        base, _ = server
        result = _post(f"{base}/demo/tournament/run", {
            "agent_ids": ["a", "b", "c"],
            "duration_hours": 1.0,
            "market_conditions": {},
        })
        assert "final_rankings" in result
        assert len(result["final_rankings"]) == 3

    def test_post_tournament_returns_stats(self, server):
        base, _ = server
        result = _post(f"{base}/demo/tournament/run", {
            "agent_ids": ["a", "b"],
            "duration_hours": 1.0,
            "market_conditions": {},
        })
        assert "tournament_stats" in result
        assert result["tournament_stats"]["agent_count"] == 2

    def test_post_tournament_missing_agent_ids(self, server):
        base, _ = server
        result = _post(f"{base}/demo/tournament/run", {
            "duration_hours": 1.0,
            "market_conditions": {},
        }, expect_error=True)
        assert "error" in result

    def test_post_tournament_single_agent_error(self, server):
        base, _ = server
        result = _post(f"{base}/demo/tournament/run", {
            "agent_ids": ["only"],
            "duration_hours": 1.0,
            "market_conditions": {},
        }, expect_error=True)
        assert "error" in result

    def test_post_tournament_invalid_duration(self, server):
        base, _ = server
        result = _post(f"{base}/demo/tournament/run", {
            "agent_ids": ["a", "b"],
            "duration_hours": -1.0,
            "market_conditions": {},
        }, expect_error=True)
        assert "error" in result

    def test_post_tournament_invalid_json(self, server):
        base, _ = server
        status, body = _post_raw(
            f"{base}/demo/tournament/run",
            b"not valid json",
            {"Content-Type": "application/json"},
        )
        assert status == 400
        assert "error" in body


class TestBacktestRunHTTP:
    def test_post_backtest_run_ok(self, server):
        base, _ = server
        result = _post(f"{base}/demo/backtest/run", {
            "strategy_config": {"type": "momentum", "params": {"window": 10}},
            "lookback_days": 60,
            "assets": ["BTC/USD"],
        })
        assert "equity_curve" in result
        assert "sharpe_ratio" in result
        assert "total_return_pct" in result

    def test_post_backtest_run_all_fields(self, server):
        base, _ = server
        result = _post(f"{base}/demo/backtest/run", {
            "strategy_config": {"type": "arbitrage", "params": {"leverage": 1.5}},
            "lookback_days": 30,
            "assets": ["ETH/USD", "BTC/USD"],
        })
        required = {"equity_curve", "sharpe_ratio", "max_drawdown_pct", "win_rate",
                    "profit_factor", "total_return_pct", "benchmark_return_pct", "alpha", "beta"}
        assert required.issubset(result.keys())

    def test_post_backtest_missing_strategy_config(self, server):
        base, _ = server
        result = _post(f"{base}/demo/backtest/run", {
            "lookback_days": 30,
            "assets": ["BTC/USD"],
        }, expect_error=True)
        assert "error" in result

    def test_post_backtest_invalid_strategy_type(self, server):
        base, _ = server
        result = _post(f"{base}/demo/backtest/run", {
            "strategy_config": {"type": "does_not_exist"},
            "lookback_days": 30,
            "assets": ["BTC/USD"],
        }, expect_error=True)
        assert "error" in result

    def test_post_backtest_empty_assets(self, server):
        base, _ = server
        result = _post(f"{base}/demo/backtest/run", {
            "strategy_config": {"type": "momentum"},
            "lookback_days": 30,
            "assets": [],
        }, expect_error=True)
        assert "error" in result

    def test_post_backtest_invalid_json(self, server):
        base, _ = server
        status, body = _post_raw(
            f"{base}/demo/backtest/run",
            b"bad json!!",
            {"Content-Type": "application/json"},
        )
        assert status == 400

    def test_post_backtest_equity_curve_is_list(self, server):
        base, _ = server
        result = _post(f"{base}/demo/backtest/run", {
            "strategy_config": {"type": "mean_reversion"},
            "lookback_days": 30,
            "assets": ["BTC/USD"],
        })
        assert isinstance(result["equity_curve"], list)
        assert len(result["equity_curve"]) >= 2


class TestMarketRegimeHTTP:
    def test_get_market_regime_ok(self, server):
        base, _ = server
        result = _get(f"{base}/demo/market/regime")
        assert "current_regime" in result
        valid = {"trending", "ranging", "volatile", "crisis"}
        assert result["current_regime"] in valid

    def test_get_market_regime_fields(self, server):
        base, _ = server
        result = _get(f"{base}/demo/market/regime")
        required = {"current_regime", "regime_confidence_pct",
                    "regime_duration_hours", "optimal_strategies", "regime_history"}
        assert required.issubset(result.keys())

    def test_get_market_regime_history_length(self, server):
        base, _ = server
        result = _get(f"{base}/demo/market/regime")
        assert len(result["regime_history"]) == 7

    def test_get_market_regime_optimal_strategies_list(self, server):
        base, _ = server
        result = _get(f"{base}/demo/market/regime")
        assert isinstance(result["optimal_strategies"], list)
        assert len(result["optimal_strategies"]) > 0


class TestAgentAttributionHTTP:
    def test_get_attribution_ok(self, server):
        base, _ = server
        result = _get(f"{base}/demo/agents/agent-alpha/attribution")
        assert "agent_id" in result
        assert result["agent_id"] == "agent-alpha"

    def test_get_attribution_fields(self, server):
        base, _ = server
        result = _get(f"{base}/demo/agents/agent-beta/attribution")
        required = {"agent_id", "total_pnl_usd", "portfolio_value",
                    "contribution_analysis", "improvement_suggestions"}
        assert required.issubset(result.keys())

    def test_get_attribution_contribution_analysis(self, server):
        base, _ = server
        result = _get(f"{base}/demo/agents/agent-gamma/attribution")
        ca = result["contribution_analysis"]
        assert "strategy_alpha_pct" in ca
        assert "market_beta_pct" in ca
        assert "timing_pct" in ca
        assert "risk_management_pct" in ca

    def test_get_attribution_suggestions_list(self, server):
        base, _ = server
        result = _get(f"{base}/demo/agents/test-agent/attribution")
        assert isinstance(result["improvement_suggestions"], list)
        assert len(result["improvement_suggestions"]) >= 1

    def test_get_attribution_404_bad_path(self, server):
        base, _ = server
        # Path without agent_id segment should return 404
        try:
            _get(f"{base}/demo/agents//attribution")
            assert False, "Should have raised"
        except HTTPError as exc:
            assert exc.code == 404

    def test_get_attribution_different_agents(self, server):
        base, _ = server
        r1 = _get(f"{base}/demo/agents/agent-one/attribution")
        r2 = _get(f"{base}/demo/agents/agent-two/attribution")
        assert r1["total_pnl_usd"] != r2["total_pnl_usd"]


# ═══════════════════════════════════════════════════════════════════════════════
# 8. Edge Cases and 404 routing
# ═══════════════════════════════════════════════════════════════════════════════

class TestEdgeCasesAndRouting:
    def test_get_nonexistent_route(self, server):
        base, _ = server
        try:
            _get(f"{base}/demo/tournament/nonexistent")
            assert False, "Should return 404"
        except HTTPError as exc:
            assert exc.code == 404

    def test_post_nonexistent_route(self, server):
        base, _ = server
        result = _post(f"{base}/demo/nonexistent", {}, expect_error=True)
        # Should be a 404 error
        assert "error" in result

    def test_tournament_with_market_conditions_volatility(self, server):
        base, _ = server
        result = _post(f"{base}/demo/tournament/run", {
            "agent_ids": ["a", "b", "c"],
            "duration_hours": 1.0,
            "market_conditions": {"volatility": 0.05, "trend": 0.5},
        })
        assert result["winner"] in ("a", "b", "c")

    def test_backtest_all_strategy_types_http(self, server):
        base, _ = server
        for strat in ["momentum", "mean_reversion", "arbitrage", "ml_hybrid"]:
            result = _post(f"{base}/demo/backtest/run", {
                "strategy_config": {"type": strat, "params": {}},
                "lookback_days": 20,
                "assets": ["BTC/USD"],
            })
            assert result["strategy_config"]["type"] == strat

    def test_tournament_large_agent_list(self, server):
        base, _ = server
        agents = [f"agent-{i}" for i in range(8)]
        result = _post(f"{base}/demo/tournament/run", {
            "agent_ids": agents,
            "duration_hours": 0.5,
            "market_conditions": {},
        })
        assert result["winner"] in agents

    def test_regime_history_dates_are_strings(self, server):
        base, _ = server
        result = _get(f"{base}/demo/market/regime")
        for entry in result["regime_history"]:
            assert isinstance(entry["date"], str)

    def test_backtest_default_lookback(self, server):
        base, _ = server
        # Missing lookback_days defaults gracefully
        result = _post(f"{base}/demo/backtest/run", {
            "strategy_config": {"type": "momentum"},
            "assets": ["BTC/USD"],
        })
        assert "equity_curve" in result

    def test_attribution_generated_at_recent(self, server):
        base, _ = server
        before = time.time() - 1
        result = _get(f"{base}/demo/agents/myagent/attribution")
        assert result["generated_at"] >= before
