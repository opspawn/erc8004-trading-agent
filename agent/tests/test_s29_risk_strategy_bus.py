"""
test_s29_risk_strategy_bus.py — Sprint 29: Risk Dashboard, Strategy Comparison,
                                  and Agent Message Bus tests.

110 tests covering:
  - _mc_var: Monte Carlo VaR computation
  - _compute_beta: beta calculation
  - _compute_liquidity_score: liquidity metric
  - build_risk_dashboard: consolidated risk endpoint
  - _run_strategy_sim: per-strategy simulation
  - build_strategy_comparison: strategy comparison table
  - broadcast_message: inter-agent message dispatch
  - get_bus_messages: message retrieval
  - clear_bus_messages: bus reset
  - HTTP GET /demo/risk/dashboard
  - HTTP GET /demo/strategies/compare
  - HTTP GET /demo/agents/messages
  - HTTP POST /demo/agents/broadcast
  - Edge cases and error handling
"""

from __future__ import annotations

import json
import math
import threading
import time
from urllib.request import urlopen, Request
from urllib.error import HTTPError

import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from demo_server import (
    # Risk dashboard
    _mc_var,
    _compute_beta,
    _compute_liquidity_score,
    build_risk_dashboard,
    # Strategy comparison
    _run_strategy_sim,
    _COMPARE_STRATEGIES,
    _COMPARE_N_DAYS,
    _COMPARE_CAPITAL,
    _COMPARE_SEED_BASE,
    build_strategy_comparison,
    # Message bus
    broadcast_message,
    get_bus_messages,
    clear_bus_messages,
    _MSG_BUS_CAPACITY,
    _msg_bus,
    # Shared helpers
    _gbm_price_series,
    DemoServer,
    DEFAULT_PORT,
)

# Use a unique port for S29 HTTP tests
S29_PORT = 18099


# ─── _mc_var ──────────────────────────────────────────────────────────────────

class TestMcVar:
    def test_returns_four_keys(self):
        result = _mc_var(10_000.0, mu=0.0003, sigma=0.02)
        assert set(result.keys()) == {"var_95", "var_99", "cvar_95", "cvar_99"}

    def test_var_99_greater_or_equal_var_95(self):
        result = _mc_var(10_000.0, mu=0.0003, sigma=0.02, seed=1)
        assert result["var_99"] >= result["var_95"]

    def test_cvar_95_greater_or_equal_var_95(self):
        result = _mc_var(10_000.0, mu=0.0003, sigma=0.02, seed=2)
        assert result["cvar_95"] >= result["var_95"]

    def test_cvar_99_greater_or_equal_var_99(self):
        result = _mc_var(10_000.0, mu=0.0003, sigma=0.02, seed=3)
        assert result["cvar_99"] >= result["var_99"]

    def test_all_values_positive(self):
        # VaR/CVaR are expressed as positive dollar losses
        result = _mc_var(10_000.0, mu=-0.001, sigma=0.03, seed=5)
        assert result["var_95"] > 0
        assert result["var_99"] > 0
        assert result["cvar_95"] > 0
        assert result["cvar_99"] > 0

    def test_deterministic_with_same_seed(self):
        r1 = _mc_var(5_000.0, mu=0.0, sigma=0.02, seed=99)
        r2 = _mc_var(5_000.0, mu=0.0, sigma=0.02, seed=99)
        assert r1 == r2

    def test_different_seeds_differ(self):
        r1 = _mc_var(5_000.0, mu=0.0, sigma=0.02, seed=10)
        r2 = _mc_var(5_000.0, mu=0.0, sigma=0.02, seed=11)
        assert r1 != r2

    def test_higher_sigma_increases_var(self):
        low = _mc_var(10_000.0, mu=0.0, sigma=0.01, n_paths=1000, seed=42)
        high = _mc_var(10_000.0, mu=0.0, sigma=0.05, n_paths=1000, seed=42)
        assert high["var_95"] > low["var_95"]

    def test_larger_portfolio_larger_var(self):
        small = _mc_var(1_000.0, mu=0.0, sigma=0.02, seed=42)
        large = _mc_var(100_000.0, mu=0.0, sigma=0.02, seed=42)
        assert large["var_95"] > small["var_95"]

    def test_horizon_affects_var(self):
        short = _mc_var(10_000.0, mu=0.0, sigma=0.02, horizon=1, seed=42)
        long_ = _mc_var(10_000.0, mu=0.0, sigma=0.02, horizon=30, seed=42)
        assert long_["var_95"] > short["var_95"]

    def test_n_paths_param(self):
        result = _mc_var(10_000.0, mu=0.0, sigma=0.02, n_paths=500, seed=42)
        assert result["var_95"] > 0

    def test_returns_floats(self):
        result = _mc_var(10_000.0, mu=0.0, sigma=0.02, seed=42)
        for v in result.values():
            assert isinstance(v, float)

    def test_values_rounded_to_2dp(self):
        result = _mc_var(10_000.0, mu=0.0, sigma=0.02, seed=42)
        for v in result.values():
            assert round(v, 2) == v


# ─── _compute_beta ────────────────────────────────────────────────────────────

class TestComputeBeta:
    def test_beta_of_identical_returns_is_one(self):
        returns = [0.01, -0.02, 0.03, -0.01, 0.02]
        beta = _compute_beta(returns, returns)
        assert abs(beta - 1.0) < 1e-6

    def test_beta_of_negative_correlated_is_negative(self):
        pos = [0.01, 0.02, 0.03]
        neg = [-0.01, -0.02, -0.03]
        beta = _compute_beta(pos, neg)
        assert beta < 0

    def test_beta_zero_market_variance_returns_one(self):
        constant = [0.01, 0.01, 0.01]
        beta = _compute_beta([0.01, 0.02, 0.03], constant)
        assert beta == 1.0

    def test_beta_with_empty_returns(self):
        beta = _compute_beta([], [])
        assert beta == 1.0

    def test_beta_with_single_element(self):
        beta = _compute_beta([0.01], [0.01])
        assert beta == 1.0

    def test_beta_returns_float(self):
        beta = _compute_beta([0.01, 0.02, -0.01], [0.005, 0.015, 0.0])
        assert isinstance(beta, float)

    def test_beta_rounded_to_4dp(self):
        beta = _compute_beta([0.01, 0.02, -0.01], [0.005, 0.015, 0.0])
        assert round(beta, 4) == beta

    def test_beta_uses_shorter_series_length(self):
        # Should not crash when lengths differ
        beta = _compute_beta([0.01, 0.02, 0.03, 0.04], [0.01, 0.02])
        assert isinstance(beta, float)

    def test_beta_leveraged_strategy(self):
        # Double the market return → beta ≈ 2
        market = [0.01, -0.02, 0.03, -0.01, 0.02]
        agent = [0.02, -0.04, 0.06, -0.02, 0.04]
        beta = _compute_beta(agent, market)
        assert abs(beta - 2.0) < 0.01


# ─── _compute_liquidity_score ─────────────────────────────────────────────────

class TestComputeLiquidityScore:
    def test_returns_float(self):
        score = _compute_liquidity_score(10_000.0, 4)
        assert isinstance(score, float)

    def test_score_in_range_0_to_100(self):
        for value in [100.0, 1_000.0, 10_000.0, 1_000_000.0]:
            for agents in [1, 3, 5, 10]:
                score = _compute_liquidity_score(value, agents)
                assert 0.0 <= score <= 100.0, f"{score} out of range"

    def test_zero_agents_still_works(self):
        score = _compute_liquidity_score(10_000.0, 0)
        assert 0.0 <= score <= 100.0

    def test_higher_value_higher_score(self):
        low = _compute_liquidity_score(1_000.0, 1)
        high = _compute_liquidity_score(100_000.0, 1)
        assert high >= low

    def test_more_agents_higher_score(self):
        few = _compute_liquidity_score(10_000.0, 1)
        many = _compute_liquidity_score(10_000.0, 5)
        assert many >= few

    def test_max_score_capped_at_100(self):
        score = _compute_liquidity_score(1_000_000.0, 100)
        assert score == 100.0

    def test_rounded_to_2dp(self):
        score = _compute_liquidity_score(12345.0, 3)
        assert round(score, 2) == score


# ─── build_risk_dashboard ─────────────────────────────────────────────────────

class TestBuildRiskDashboard:
    def test_returns_dict(self):
        result = build_risk_dashboard()
        assert isinstance(result, dict)

    def test_required_top_level_keys(self):
        result = build_risk_dashboard()
        required = {
            "portfolio_value_usd", "n_agents", "var", "cvar",
            "beta_vs_market", "drawdown", "liquidity_score",
            "per_agent", "generated_at",
        }
        assert required.issubset(result.keys())

    def test_var_structure(self):
        var = build_risk_dashboard()["var"]
        assert "confidence_95_pct" in var
        assert "confidence_99_pct" in var
        assert "horizon_days" in var
        assert "method" in var
        assert "n_paths" in var

    def test_cvar_structure(self):
        cvar = build_risk_dashboard()["cvar"]
        assert "expected_shortfall_95" in cvar
        assert "expected_shortfall_99" in cvar

    def test_var_method_is_monte_carlo(self):
        result = build_risk_dashboard()
        assert "Monte Carlo" in result["var"]["method"]

    def test_var_99_gte_var_95(self):
        result = build_risk_dashboard()
        assert result["var"]["confidence_99_pct"] >= result["var"]["confidence_95_pct"]

    def test_cvar_gte_var(self):
        result = build_risk_dashboard()
        assert result["cvar"]["expected_shortfall_95"] >= result["var"]["confidence_95_pct"]

    def test_portfolio_value_positive(self):
        result = build_risk_dashboard()
        assert result["portfolio_value_usd"] > 0

    def test_n_agents_positive(self):
        result = build_risk_dashboard()
        assert result["n_agents"] >= 1

    def test_liquidity_score_in_range(self):
        result = build_risk_dashboard()
        assert 0.0 <= result["liquidity_score"] <= 100.0

    def test_beta_is_float(self):
        result = build_risk_dashboard()
        assert isinstance(result["beta_vs_market"], float)

    def test_drawdown_structure(self):
        dd = build_risk_dashboard()["drawdown"]
        assert "current_avg_pct" in dd
        assert "current_max_pct" in dd
        assert "historical_max_pct" in dd
        assert "drawdown_ratio" in dd

    def test_drawdown_ratio_non_negative(self):
        result = build_risk_dashboard()
        assert result["drawdown"]["drawdown_ratio"] >= 0.0

    def test_per_agent_is_list(self):
        result = build_risk_dashboard()
        assert isinstance(result["per_agent"], list)

    def test_per_agent_has_required_fields(self):
        result = build_risk_dashboard()
        for agent in result["per_agent"]:
            assert "agent_id" in agent
            assert "drawdown_pct" in agent
            assert "win_rate_30d" in agent
            assert "beta" in agent

    def test_generated_at_is_recent(self):
        result = build_risk_dashboard()
        assert abs(result["generated_at"] - time.time()) < 10

    def test_var_n_paths_is_1000(self):
        result = build_risk_dashboard()
        assert result["var"]["n_paths"] == 1000

    def test_deterministic_output(self):
        r1 = build_risk_dashboard()
        r2 = build_risk_dashboard()
        assert r1["var"]["confidence_95_pct"] == r2["var"]["confidence_95_pct"]


# ─── _run_strategy_sim ────────────────────────────────────────────────────────

class TestRunStrategySim:
    @pytest.fixture
    def prices(self):
        return _gbm_price_series(seed=42, n_days=60)

    def test_returns_required_keys(self, prices):
        result = _run_strategy_sim("momentum", prices, 10_000.0)
        assert set(result.keys()) == {"return_pct", "sharpe", "max_drawdown", "win_rate", "n_trades"}

    @pytest.mark.parametrize("strategy", ["momentum", "mean_reversion", "arbitrage", "market_making"])
    def test_all_strategies_run(self, prices, strategy):
        result = _run_strategy_sim(strategy, prices, 10_000.0)
        assert isinstance(result, dict)

    def test_win_rate_in_01(self, prices):
        for s in _COMPARE_STRATEGIES:
            result = _run_strategy_sim(s, prices, 10_000.0)
            assert 0.0 <= result["win_rate"] <= 1.0

    def test_max_drawdown_non_negative(self, prices):
        for s in _COMPARE_STRATEGIES:
            result = _run_strategy_sim(s, prices, 10_000.0)
            assert result["max_drawdown"] >= 0.0

    def test_return_pct_is_float(self, prices):
        result = _run_strategy_sim("momentum", prices, 10_000.0)
        assert isinstance(result["return_pct"], float)

    def test_n_trades_non_negative(self, prices):
        for s in _COMPARE_STRATEGIES:
            result = _run_strategy_sim(s, prices, 10_000.0)
            assert result["n_trades"] >= 0

    def test_market_making_many_trades(self, prices):
        result = _run_strategy_sim("market_making", prices, 10_000.0)
        assert result["n_trades"] > 0

    def test_sharpe_is_float(self, prices):
        result = _run_strategy_sim("momentum", prices, 10_000.0)
        assert isinstance(result["sharpe"], float)

    def test_deterministic(self, prices):
        r1 = _run_strategy_sim("arbitrage", prices, 10_000.0)
        r2 = _run_strategy_sim("arbitrage", prices, 10_000.0)
        assert r1 == r2


# ─── build_strategy_comparison ────────────────────────────────────────────────

class TestBuildStrategyComparison:
    def test_returns_dict(self):
        result = build_strategy_comparison()
        assert isinstance(result, dict)

    def test_required_keys(self):
        result = build_strategy_comparison()
        required = {"comparison", "strategies_compared", "backtest_params", "best_strategy", "generated_at"}
        assert required.issubset(result.keys())

    def test_comparison_has_four_entries(self):
        result = build_strategy_comparison()
        assert len(result["comparison"]) == 4

    def test_all_strategies_present(self):
        result = build_strategy_comparison()
        strategies = {r["strategy"] for r in result["comparison"]}
        assert strategies == set(_COMPARE_STRATEGIES)

    def test_ranks_assigned(self):
        result = build_strategy_comparison()
        ranks = sorted(r["rank"] for r in result["comparison"])
        assert ranks == [1, 2, 3, 4]

    def test_rank_1_has_best_sharpe(self):
        result = build_strategy_comparison()
        by_rank = sorted(result["comparison"], key=lambda x: x["rank"])
        sharpes = [r["sharpe"] for r in by_rank]
        assert sharpes[0] >= sharpes[1] >= sharpes[2] >= sharpes[3]

    def test_each_entry_has_required_fields(self):
        result = build_strategy_comparison()
        for entry in result["comparison"]:
            assert "strategy" in entry
            assert "return_pct" in entry
            assert "sharpe" in entry
            assert "max_drawdown" in entry
            assert "win_rate" in entry
            assert "rank" in entry

    def test_best_strategy_is_rank_1(self):
        result = build_strategy_comparison()
        rank1 = next(r for r in result["comparison"] if r["rank"] == 1)
        assert result["best_strategy"] == rank1["strategy"]

    def test_backtest_params_structure(self):
        result = build_strategy_comparison()
        params = result["backtest_params"]
        assert "n_days" in params
        assert "initial_capital" in params
        assert "price_model" in params
        assert "seed" in params

    def test_price_model_is_gbm(self):
        result = build_strategy_comparison()
        assert result["backtest_params"]["price_model"] == "GBM"

    def test_default_n_days(self):
        result = build_strategy_comparison()
        assert result["backtest_params"]["n_days"] == _COMPARE_N_DAYS

    def test_default_capital(self):
        result = build_strategy_comparison()
        assert result["backtest_params"]["initial_capital"] == _COMPARE_CAPITAL

    def test_custom_n_days(self):
        result = build_strategy_comparison(n_days=30)
        assert result["backtest_params"]["n_days"] == 30

    def test_custom_seed(self):
        r1 = build_strategy_comparison(seed=1)
        r2 = build_strategy_comparison(seed=2)
        # Different seeds → different results
        assert r1["comparison"] != r2["comparison"]

    def test_deterministic_same_seed(self):
        r1 = build_strategy_comparison(seed=77)
        r2 = build_strategy_comparison(seed=77)
        assert r1["comparison"] == r2["comparison"]

    def test_win_rates_in_01(self):
        result = build_strategy_comparison()
        for entry in result["comparison"]:
            assert 0.0 <= entry["win_rate"] <= 1.0

    def test_max_drawdown_non_negative(self):
        result = build_strategy_comparison()
        for entry in result["comparison"]:
            assert entry["max_drawdown"] >= 0.0

    def test_generated_at_recent(self):
        result = build_strategy_comparison()
        assert abs(result["generated_at"] - time.time()) < 10

    def test_strategies_compared_list(self):
        result = build_strategy_comparison()
        assert isinstance(result["strategies_compared"], list)
        assert len(result["strategies_compared"]) == 4


# ─── broadcast_message ────────────────────────────────────────────────────────

class TestBroadcastMessage:
    def setup_method(self):
        clear_bus_messages()

    def test_returns_dict(self):
        msg = broadcast_message("heartbeat", {})
        assert isinstance(msg, dict)

    def test_required_fields(self):
        msg = broadcast_message("signal", {"action": "buy"})
        assert "msg_id" in msg
        assert "type" in msg
        assert "from_agent" in msg
        assert "payload" in msg
        assert "timestamp" in msg
        assert "recipients" in msg
        assert "recipient_count" in msg

    def test_type_preserved(self):
        msg = broadcast_message("risk_update", {})
        assert msg["type"] == "risk_update"

    def test_payload_preserved(self):
        payload = {"symbol": "BTC/USD", "value": 42}
        msg = broadcast_message("signal", payload)
        assert msg["payload"] == payload

    def test_from_agent_preserved(self):
        msg = broadcast_message("heartbeat", {}, from_agent="orchestrator")
        assert msg["from_agent"] == "orchestrator"

    def test_default_from_agent(self):
        msg = broadcast_message("heartbeat", {})
        assert msg["from_agent"] == "main"

    def test_msg_id_is_unique(self):
        ids = {broadcast_message("heartbeat", {})["msg_id"] for _ in range(5)}
        assert len(ids) == 5

    def test_timestamp_is_recent(self):
        msg = broadcast_message("heartbeat", {})
        assert abs(msg["timestamp"] - time.time()) < 5

    def test_message_added_to_bus(self):
        initial_count = get_bus_messages()["count"]
        broadcast_message("info", {"note": "test"})
        assert get_bus_messages()["count"] == initial_count + 1

    def test_empty_type_raises(self):
        with pytest.raises(ValueError):
            broadcast_message("", {})

    def test_long_type_raises(self):
        with pytest.raises(ValueError):
            broadcast_message("x" * 65, {})

    def test_from_agent_truncated_at_64(self):
        msg = broadcast_message("info", {}, from_agent="a" * 100)
        assert len(msg["from_agent"]) <= 64

    def test_recipient_count_matches_recipients(self):
        msg = broadcast_message("heartbeat", {})
        assert msg["recipient_count"] == len(msg["recipients"])

    def test_bus_capacity_respected(self):
        # Fill beyond capacity
        for i in range(_MSG_BUS_CAPACITY + 10):
            broadcast_message("heartbeat", {"i": i})
        result = get_bus_messages()
        assert result["count"] <= _MSG_BUS_CAPACITY

    def test_custom_type_accepted(self):
        msg = broadcast_message("custom_rebalance", {"target": 0.5})
        assert msg["type"] == "custom_rebalance"

    def test_nested_payload(self):
        payload = {"level1": {"level2": [1, 2, 3]}}
        msg = broadcast_message("info", payload)
        assert msg["payload"] == payload


# ─── get_bus_messages ─────────────────────────────────────────────────────────

class TestGetBusMessages:
    def setup_method(self):
        clear_bus_messages()

    def test_returns_dict(self):
        result = get_bus_messages()
        assert isinstance(result, dict)

    def test_required_keys(self):
        result = get_bus_messages()
        assert "messages" in result
        assert "count" in result
        assert "capacity" in result
        assert "retrieved_at" in result

    def test_messages_is_list(self):
        result = get_bus_messages()
        assert isinstance(result["messages"], list)

    def test_count_matches_messages_length(self):
        result = get_bus_messages()
        assert result["count"] == len(result["messages"])

    def test_capacity_is_50(self):
        result = get_bus_messages()
        assert result["capacity"] == 50

    def test_limit_parameter(self):
        for _ in range(10):
            broadcast_message("heartbeat", {})
        result = get_bus_messages(limit=3)
        assert result["count"] <= 3

    def test_limit_min_1(self):
        result = get_bus_messages(limit=0)
        assert result["count"] >= 1

    def test_limit_max_capacity(self):
        result = get_bus_messages(limit=1000)
        assert result["count"] <= _MSG_BUS_CAPACITY

    def test_messages_have_required_fields(self):
        broadcast_message("signal", {"action": "buy"})
        result = get_bus_messages()
        for msg in result["messages"]:
            assert "msg_id" in msg
            assert "type" in msg
            assert "timestamp" in msg

    def test_seed_messages_present_at_startup(self):
        result = get_bus_messages()
        # Should have at least the 3 seed messages
        assert result["count"] >= 3

    def test_retrieved_at_is_recent(self):
        result = get_bus_messages()
        assert abs(result["retrieved_at"] - time.time()) < 5

    def test_returns_latest_messages(self):
        clear_bus_messages()
        broadcast_message("first", {"n": 1})
        broadcast_message("last", {"n": 2})
        result = get_bus_messages(limit=1)
        assert result["messages"][-1]["type"] == "last"


# ─── clear_bus_messages ───────────────────────────────────────────────────────

class TestClearBusMessages:
    def test_returns_int(self):
        count = clear_bus_messages()
        assert isinstance(count, int)

    def test_returns_nonzero_when_messages_exist(self):
        broadcast_message("heartbeat", {})
        count = clear_bus_messages()
        assert count >= 1

    def test_seeds_restored_after_clear(self):
        clear_bus_messages()
        result = get_bus_messages()
        # Seed messages should still be there
        assert result["count"] >= 3

    def test_extra_messages_cleared(self):
        clear_bus_messages()
        initial = get_bus_messages()["count"]
        for _ in range(5):
            broadcast_message("info", {})
        clear_bus_messages()
        after = get_bus_messages()["count"]
        assert after == initial


# ─── HTTP Endpoint Tests ──────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def s29_server():
    """Start a DemoServer on S29_PORT for HTTP tests."""
    srv = DemoServer(port=S29_PORT)
    srv.start()
    time.sleep(0.3)
    yield srv
    srv.stop()


def _get(path: str, port: int = S29_PORT) -> dict:
    url = f"http://localhost:{port}{path}"
    with urlopen(Request(url, method="GET"), timeout=10) as resp:
        return json.loads(resp.read())


def _post(path: str, body: dict, port: int = S29_PORT) -> dict:
    url = f"http://localhost:{port}{path}"
    data = json.dumps(body).encode("utf-8")
    req = Request(url, data=data, method="POST",
                  headers={"Content-Type": "application/json",
                           "Content-Length": str(len(data))})
    with urlopen(req, timeout=10) as resp:
        return json.loads(resp.read())


class TestHttpRiskDashboard:
    def test_200_ok(self, s29_server):
        result = _get("/demo/risk/dashboard")
        assert "var" in result

    def test_required_fields(self, s29_server):
        result = _get("/demo/risk/dashboard")
        assert "portfolio_value_usd" in result
        assert "n_agents" in result
        assert "cvar" in result
        assert "liquidity_score" in result

    def test_var_structure(self, s29_server):
        result = _get("/demo/risk/dashboard")
        assert "confidence_95_pct" in result["var"]
        assert "confidence_99_pct" in result["var"]

    def test_per_agent_list(self, s29_server):
        result = _get("/demo/risk/dashboard")
        assert isinstance(result["per_agent"], list)

    def test_content_type_json(self, s29_server):
        url = f"http://localhost:{S29_PORT}/demo/risk/dashboard"
        with urlopen(Request(url), timeout=10) as resp:
            ct = resp.headers.get("Content-Type", "")
        assert "application/json" in ct

    def test_trailing_slash_works(self, s29_server):
        result = _get("/demo/risk/dashboard/")
        assert "var" in result


class TestHttpStrategiesCompare:
    def test_200_ok(self, s29_server):
        result = _get("/demo/strategies/compare")
        assert "comparison" in result

    def test_four_strategies(self, s29_server):
        result = _get("/demo/strategies/compare")
        assert len(result["comparison"]) == 4

    def test_best_strategy_present(self, s29_server):
        result = _get("/demo/strategies/compare")
        assert result["best_strategy"] in _COMPARE_STRATEGIES

    def test_custom_n_days_param(self, s29_server):
        result = _get("/demo/strategies/compare?n_days=30")
        assert result["backtest_params"]["n_days"] == 30

    def test_custom_seed_param(self, s29_server):
        r1 = _get("/demo/strategies/compare?seed=1")
        r2 = _get("/demo/strategies/compare?seed=2")
        assert r1["comparison"] != r2["comparison"]

    def test_invalid_capital_returns_400(self, s29_server):
        try:
            _get("/demo/strategies/compare?capital=-100")
            pytest.fail("Expected HTTPError 400")
        except HTTPError as e:
            assert e.code == 400

    def test_invalid_n_days_clamped(self, s29_server):
        # n_days=5 < min 10 → clamped to 10
        result = _get("/demo/strategies/compare?n_days=5")
        assert result["backtest_params"]["n_days"] == 10

    def test_strategy_entries_have_rank(self, s29_server):
        result = _get("/demo/strategies/compare")
        for entry in result["comparison"]:
            assert "rank" in entry

    def test_trailing_slash_works(self, s29_server):
        result = _get("/demo/strategies/compare/")
        assert "comparison" in result


class TestHttpAgentsMessages:
    def test_200_ok(self, s29_server):
        result = _get("/demo/agents/messages")
        assert "messages" in result

    def test_messages_list(self, s29_server):
        result = _get("/demo/agents/messages")
        assert isinstance(result["messages"], list)

    def test_count_field(self, s29_server):
        result = _get("/demo/agents/messages")
        assert "count" in result

    def test_custom_limit(self, s29_server):
        result = _get("/demo/agents/messages?limit=2")
        assert result["count"] <= 2

    def test_capacity_field(self, s29_server):
        result = _get("/demo/agents/messages")
        assert result["capacity"] == 50

    def test_trailing_slash_works(self, s29_server):
        result = _get("/demo/agents/messages/")
        assert "messages" in result


class TestHttpAgentsBroadcast:
    def test_200_ok(self, s29_server):
        result = _post("/demo/agents/broadcast", {"type": "heartbeat", "payload": {}})
        assert result["status"] == "broadcast"

    def test_message_in_response(self, s29_server):
        result = _post("/demo/agents/broadcast", {"type": "signal", "payload": {"action": "buy"}})
        assert "message" in result
        assert result["message"]["type"] == "signal"

    def test_bus_size_in_response(self, s29_server):
        result = _post("/demo/agents/broadcast", {"type": "heartbeat", "payload": {}})
        assert "bus_size" in result

    def test_missing_type_returns_400(self, s29_server):
        try:
            _post("/demo/agents/broadcast", {"payload": {}})
            pytest.fail("Expected HTTPError 400")
        except HTTPError as e:
            assert e.code == 400

    def test_custom_from_agent(self, s29_server):
        result = _post("/demo/agents/broadcast", {
            "type": "risk_update",
            "payload": {"drawdown": 5.0},
            "from_agent": "risk-module",
        })
        assert result["message"]["from_agent"] == "risk-module"

    def test_message_visible_in_get(self, s29_server):
        _post("/demo/agents/broadcast", {"type": "test_visibility", "payload": {"id": 999}})
        msgs = _get("/demo/agents/messages")
        types = [m["type"] for m in msgs["messages"]]
        assert "test_visibility" in types

    def test_trailing_slash_works(self, s29_server):
        result = _post("/demo/agents/broadcast/", {"type": "info", "payload": {}})
        assert result["status"] == "broadcast"

    def test_invalid_json_returns_400(self, s29_server):
        url = f"http://localhost:{S29_PORT}/demo/agents/broadcast"
        data = b"not json"
        req = Request(url, data=data, method="POST",
                      headers={"Content-Type": "application/json",
                               "Content-Length": str(len(data))})
        try:
            urlopen(req, timeout=10)
            pytest.fail("Expected HTTPError 400")
        except HTTPError as e:
            assert e.code == 400
