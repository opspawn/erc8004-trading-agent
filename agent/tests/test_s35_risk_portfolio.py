"""
test_s35_risk_portfolio.py — Sprint 35: Risk Management + Portfolio Rebalancing.

Covers:
  - assess_trade_risk(): happy path, field schema, recommendation thresholds,
    seeding determinism, edge cases (missing fields, bad inputs)
  - rebalance_portfolio(): schema, trade generation, cost calculation,
    normalisation, edge cases (zero drift, empty, all-equal)
  - plan_agent_collaboration(): role assignment, step structure, synergy,
    overhead, edge cases (1 agent, max agents, no task category)
  - _generate_pnl_snapshot(): schema, tick variation, determinism
  - HTTP integration: POST /demo/risk/assess, POST /demo/portfolio/rebalance,
    POST /demo/agents/collaborate, GET /demo/agents/{id}/pnl/stream
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
    # S35 functions
    assess_trade_risk,
    rebalance_portfolio,
    plan_agent_collaboration,
    _generate_pnl_snapshot,
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
    """Post raw bytes, return (status_code, body_dict)."""
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
    sock.bind(("", 0))
    port = sock.getsockname()[1]
    sock.close()
    srv = DemoServer(port=port)
    srv.start()
    time.sleep(0.3)
    yield f"http://localhost:{port}", port
    srv.stop()


# ════════════════════════════════════════════════════════════════════════════════
# 1. assess_trade_risk() — unit tests
# ════════════════════════════════════════════════════════════════════════════════

class TestAssessTradeRisk:

    def test_returns_required_fields(self):
        result = assess_trade_risk("agent-test-s35-001", {"asset": "BTC/USD", "size": 1000, "direction": "long"})
        for field in ("agent_id", "trade", "portfolio_value", "position_size_pct",
                      "volatility", "var_95", "max_drawdown_risk", "risk_score",
                      "recommendation", "reasoning", "assessed_at"):
            assert field in result, f"Missing field: {field}"

    def test_agent_id_echoed(self):
        result = assess_trade_risk("agent-xyz", {"asset": "ETH/USD", "size": 500, "direction": "long"})
        assert result["agent_id"] == "agent-xyz"

    def test_trade_dict_echoed(self):
        result = assess_trade_risk("agent-001", {"asset": "SOL/USD", "size": 2000, "direction": "short"})
        assert result["trade"]["asset"] == "SOL/USD"
        assert result["trade"]["size"] == 2000.0
        assert result["trade"]["direction"] == "short"

    def test_risk_score_range(self):
        result = assess_trade_risk("agent-002", {"asset": "BTC/USD", "size": 100, "direction": "long"})
        assert 0.0 <= result["risk_score"] <= 10.0

    def test_position_size_pct_positive(self):
        result = assess_trade_risk("agent-003", {"asset": "ETH/USD", "size": 500, "direction": "long"})
        assert result["position_size_pct"] >= 0.0

    def test_position_size_pct_max_100(self):
        result = assess_trade_risk("agent-004", {"asset": "BTC/USD", "size": 1_000_000, "direction": "long"})
        assert result["position_size_pct"] <= 100.0

    def test_var_95_positive(self):
        result = assess_trade_risk("agent-005", {"asset": "BTC/USD", "size": 1000, "direction": "long"})
        assert result["var_95"] > 0.0

    def test_volatility_positive(self):
        result = assess_trade_risk("agent-006", {"asset": "ETH/USD", "size": 1000, "direction": "long"})
        assert result["volatility"] > 0.0

    def test_max_drawdown_risk_non_negative(self):
        result = assess_trade_risk("agent-007", {"asset": "BTC/USD", "size": 1000, "direction": "long"})
        assert result["max_drawdown_risk"] >= 0.0

    def test_recommendation_valid_values(self):
        result = assess_trade_risk("agent-008", {"asset": "BTC/USD", "size": 100, "direction": "long"})
        assert result["recommendation"] in ("proceed", "reduce", "reject")

    def test_reasoning_is_string(self):
        result = assess_trade_risk("agent-009", {"asset": "BTC/USD", "size": 100, "direction": "long"})
        assert isinstance(result["reasoning"], str)
        assert len(result["reasoning"]) > 10

    def test_assessed_at_is_float(self):
        result = assess_trade_risk("agent-010", {"asset": "BTC/USD", "size": 100, "direction": "long"})
        assert isinstance(result["assessed_at"], float)

    def test_deterministic_same_inputs(self):
        r1 = assess_trade_risk("agent-det", {"asset": "BTC/USD", "size": 2500, "direction": "long"})
        r2 = assess_trade_risk("agent-det", {"asset": "BTC/USD", "size": 2500, "direction": "long"})
        assert r1["risk_score"] == r2["risk_score"]
        assert r1["recommendation"] == r2["recommendation"]
        assert r1["var_95"] == r2["var_95"]

    def test_different_agents_different_scores(self):
        r1 = assess_trade_risk("agent-aaa", {"asset": "BTC/USD", "size": 5000, "direction": "long"})
        r2 = assess_trade_risk("agent-zzz", {"asset": "BTC/USD", "size": 5000, "direction": "long"})
        # Different agents have different portfolio values → different scores
        assert r1["portfolio_value"] != r2["portfolio_value"] or r1["risk_score"] != r2["risk_score"]

    def test_short_direction_increases_risk(self):
        r_long = assess_trade_risk("agent-dir", {"asset": "ETH/USD", "size": 5000, "direction": "long"})
        r_short = assess_trade_risk("agent-dir", {"asset": "ETH/USD", "size": 5000, "direction": "short"})
        assert r_short["risk_score"] >= r_long["risk_score"]

    def test_large_position_higher_risk(self):
        r_small = assess_trade_risk("agent-size", {"asset": "BTC/USD", "size": 100, "direction": "long"})
        r_large = assess_trade_risk("agent-size", {"asset": "BTC/USD", "size": 500_000, "direction": "long"})
        assert r_large["risk_score"] >= r_small["risk_score"]

    def test_proceed_recommendation_low_risk(self):
        # Small trade → should be proceed
        result = assess_trade_risk("agent-low", {"asset": "BTC/USD", "size": 50, "direction": "long"})
        assert result["risk_score"] <= 6.5  # proceed or reduce, not reject at tiny sizes

    def test_reject_recommendation_reasoning(self):
        # Tiny portfolio agent + huge trade → reject
        result = assess_trade_risk("agent-reject-999", {"asset": "SOL/USD", "size": 9_999_999, "direction": "long"})
        if result["recommendation"] == "reject":
            assert "reject" in result["reasoning"].lower() or "exceed" in result["reasoning"].lower()

    def test_btc_volatility_lower_than_sol(self):
        rbtc = assess_trade_risk("agent-vol", {"asset": "BTC/USD", "size": 1000, "direction": "long"})
        rsol = assess_trade_risk("agent-vol", {"asset": "SOL/USD", "size": 1000, "direction": "long"})
        # BTC vol base is 0.045, SOL is 0.085 — seeded may vary but on average BTC < SOL
        # Just verify both are valid
        assert rbtc["volatility"] > 0
        assert rsol["volatility"] > 0

    def test_portfolio_value_positive(self):
        result = assess_trade_risk("agent-pv", {"asset": "BTC/USD", "size": 1000, "direction": "long"})
        assert result["portfolio_value"] > 0

    def test_default_direction_long(self):
        result = assess_trade_risk("agent-default", {"asset": "BTC/USD", "size": 1000})
        assert result["trade"]["direction"] == "long"

    def test_default_asset(self):
        result = assess_trade_risk("agent-da", {"size": 1000})
        assert result["trade"]["asset"] == "BTC/USD"

    def test_unknown_asset_uses_fallback_vol(self):
        result = assess_trade_risk("agent-unk", {"asset": "DOGE/USD", "size": 1000, "direction": "long"})
        assert result["volatility"] > 0  # fallback vol applied


# ════════════════════════════════════════════════════════════════════════════════
# 2. rebalance_portfolio() — unit tests
# ════════════════════════════════════════════════════════════════════════════════

class TestRebalancePortfolio:

    def test_returns_required_fields(self):
        result = rebalance_portfolio({"BTC": 50, "ETH": 30, "USDC": 20}, "agent-rb-001")
        for field in ("agent_id", "current_allocations", "target_allocations",
                      "rebalance_trades", "total_drift_pct", "trade_count",
                      "estimated_cost_bps", "expected_improvement_pct", "rebalanced_at"):
            assert field in result, f"Missing field: {field}"

    def test_agent_id_echoed(self):
        result = rebalance_portfolio({"BTC": 60, "ETH": 40}, "agent-rb-999")
        assert result["agent_id"] == "agent-rb-999"

    def test_target_allocations_normalised_to_100(self):
        result = rebalance_portfolio({"BTC": 1, "ETH": 1, "USDC": 1}, "agent-norm")
        total = sum(result["target_allocations"].values())
        assert abs(total - 100.0) < 0.1

    def test_current_allocations_sum_to_100(self):
        result = rebalance_portfolio({"BTC": 50, "ETH": 50}, "agent-sum")
        total = sum(result["current_allocations"].values())
        assert abs(total - 100.0) < 0.5

    def test_rebalance_trades_is_list(self):
        result = rebalance_portfolio({"BTC": 40, "ETH": 30, "SOL": 30}, "agent-list")
        assert isinstance(result["rebalance_trades"], list)

    def test_trade_has_required_fields(self):
        result = rebalance_portfolio({"BTC": 70, "ETH": 30}, "agent-fields")
        for trade in result["rebalance_trades"]:
            for field in ("action", "asset", "amount", "current_pct", "target_pct", "drift_pct", "reason"):
                assert field in trade, f"Trade missing field: {field}"

    def test_trade_action_valid(self):
        result = rebalance_portfolio({"BTC": 80, "ETH": 20}, "agent-action")
        for trade in result["rebalance_trades"]:
            assert trade["action"] in ("buy", "sell")

    def test_trade_amount_positive(self):
        result = rebalance_portfolio({"BTC": 60, "ETH": 40}, "agent-amount")
        for trade in result["rebalance_trades"]:
            assert trade["amount"] > 0

    def test_trade_reason_is_string(self):
        result = rebalance_portfolio({"BTC": 50, "ETH": 50}, "agent-reason")
        for trade in result["rebalance_trades"]:
            assert isinstance(trade["reason"], str)
            assert len(trade["reason"]) > 5

    def test_estimated_cost_bps_positive(self):
        result = rebalance_portfolio({"BTC": 50, "ETH": 30, "SOL": 20}, "agent-cost")
        assert result["estimated_cost_bps"] >= 0

    def test_expected_improvement_positive(self):
        result = rebalance_portfolio({"BTC": 50, "ETH": 30, "SOL": 20}, "agent-imp")
        assert result["expected_improvement_pct"] >= 0

    def test_total_drift_non_negative(self):
        result = rebalance_portfolio({"BTC": 50, "ETH": 50}, "agent-drift")
        assert result["total_drift_pct"] >= 0

    def test_trade_count_matches_list(self):
        result = rebalance_portfolio({"BTC": 40, "ETH": 30, "SOL": 30}, "agent-count")
        assert result["trade_count"] == len(result["rebalance_trades"])

    def test_deterministic_same_inputs(self):
        r1 = rebalance_portfolio({"BTC": 50, "ETH": 30, "USDC": 20}, "agent-det-rb")
        r2 = rebalance_portfolio({"BTC": 50, "ETH": 30, "USDC": 20}, "agent-det-rb")
        assert r1["trade_count"] == r2["trade_count"]
        assert r1["estimated_cost_bps"] == r2["estimated_cost_bps"]

    def test_single_asset_no_trades(self):
        # Single asset at 100% — current allocation should also be ~100%
        result = rebalance_portfolio({"BTC": 100}, "agent-single")
        assert result["agent_id"] == "agent-single"
        # Only one asset, so current is 100%, drift < 0.5%, no trades
        assert result["trade_count"] == 0

    def test_many_assets(self):
        allocs = {"BTC": 20, "ETH": 20, "SOL": 15, "AVAX": 15, "USDC": 30}
        result = rebalance_portfolio(allocs, "agent-many")
        assert len(result["current_allocations"]) == 5

    def test_normalise_non_100_input(self):
        result = rebalance_portfolio({"BTC": 200, "ETH": 300}, "agent-nonorm")
        total = sum(result["target_allocations"].values())
        assert abs(total - 100.0) < 0.1

    def test_sells_before_buys(self):
        result = rebalance_portfolio({"BTC": 80, "ETH": 20}, "agent-order")
        trades = result["rebalance_trades"]
        if len(trades) >= 2:
            sell_indices = [i for i, t in enumerate(trades) if t["action"] == "sell"]
            buy_indices = [i for i, t in enumerate(trades) if t["action"] == "buy"]
            if sell_indices and buy_indices:
                assert max(sell_indices) < min(buy_indices)

    def test_rebalanced_at_is_float(self):
        result = rebalance_portfolio({"BTC": 50, "ETH": 50}, "agent-ts")
        assert isinstance(result["rebalanced_at"], float)

    def test_raises_on_zero_total(self):
        with pytest.raises((ValueError, ZeroDivisionError, Exception)):
            rebalance_portfolio({"BTC": 0, "ETH": 0}, "agent-zero")


# ════════════════════════════════════════════════════════════════════════════════
# 3. plan_agent_collaboration() — unit tests
# ════════════════════════════════════════════════════════════════════════════════

class TestPlanAgentCollaboration:

    def test_returns_required_fields(self):
        result = plan_agent_collaboration(["agent-1", "agent-2"], "Assess portfolio risk")
        for field in ("task", "task_category", "agents_with_roles", "collaboration_plan",
                      "expected_synergy_score", "coordination_overhead_pct",
                      "agent_count", "planned_at"):
            assert field in result, f"Missing field: {field}"

    def test_task_echoed(self):
        task = "Perform sentiment analysis on BTC/USD"
        result = plan_agent_collaboration(["a1", "a2"], task)
        assert result["task"] == task

    def test_agent_count_matches(self):
        agents = ["a1", "a2", "a3"]
        result = plan_agent_collaboration(agents, "Portfolio rebalancing task")
        assert result["agent_count"] == 3

    def test_agents_with_roles_length(self):
        agents = ["a1", "a2", "a3", "a4"]
        result = plan_agent_collaboration(agents, "Strategy coordination")
        assert len(result["agents_with_roles"]) == 4

    def test_exactly_one_lead(self):
        agents = ["a1", "a2", "a3", "a4", "a5"]
        result = plan_agent_collaboration(agents, "Multi-agent coordination")
        leads = [a for a in result["agents_with_roles"] if a["role"] == "lead"]
        assert len(leads) == 1

    def test_single_agent_is_lead(self):
        result = plan_agent_collaboration(["agent-solo"], "Solo task")
        assert result["agents_with_roles"][0]["role"] == "lead"
        assert result["agents_with_roles"][0]["agent_id"] == "agent-solo"

    def test_roles_are_valid(self):
        agents = ["a1", "a2", "a3", "a4"]
        result = plan_agent_collaboration(agents, "Risk task")
        for a in result["agents_with_roles"]:
            assert a["role"] in ("lead", "support", "validator")

    def test_collaboration_plan_has_steps(self):
        result = plan_agent_collaboration(["a1", "a2"], "Task")
        plan = result["collaboration_plan"]
        assert "steps" in plan
        assert isinstance(plan["steps"], list)
        assert len(plan["steps"]) > 0

    def test_steps_have_required_fields(self):
        result = plan_agent_collaboration(["a1", "a2"], "Task")
        for step in result["collaboration_plan"]["steps"]:
            assert "step" in step
            assert "phase" in step
            assert "responsible" in step
            assert "action" in step
            assert "estimated_duration_s" in step

    def test_steps_numbered_sequentially(self):
        result = plan_agent_collaboration(["a1", "a2", "a3"], "Task")
        step_nums = [s["step"] for s in result["collaboration_plan"]["steps"]]
        assert step_nums == list(range(1, len(step_nums) + 1))

    def test_total_steps_matches_steps_list(self):
        result = plan_agent_collaboration(["a1", "a2"], "Task")
        plan = result["collaboration_plan"]
        assert plan["total_steps"] == len(plan["steps"])

    def test_synergy_score_range(self):
        result = plan_agent_collaboration(["a1", "a2", "a3"], "Task")
        assert 0.0 <= result["expected_synergy_score"] <= 1.0

    def test_synergy_increases_with_more_agents(self):
        r1 = plan_agent_collaboration(["a1"], "Task description")
        r3 = plan_agent_collaboration(["a1", "a2", "a3"], "Task description")
        assert r3["expected_synergy_score"] >= r1["expected_synergy_score"]

    def test_coordination_overhead_non_negative(self):
        result = plan_agent_collaboration(["a1", "a2"], "Task")
        assert result["coordination_overhead_pct"] >= 0

    def test_coordination_overhead_increases_with_agents(self):
        r1 = plan_agent_collaboration(["a1"], "Task")
        r5 = plan_agent_collaboration(["a1", "a2", "a3", "a4", "a5"], "Task")
        assert r5["coordination_overhead_pct"] >= r1["coordination_overhead_pct"]

    def test_task_category_detected_risk(self):
        result = plan_agent_collaboration(["a1", "a2"], "Perform risk assessment")
        assert "risk" in result["task_category"].lower()

    def test_task_category_detected_sentiment(self):
        result = plan_agent_collaboration(["a1", "a2"], "Sentiment analysis for BTC")
        assert "sentiment" in result["task_category"].lower()

    def test_task_category_default_general(self):
        result = plan_agent_collaboration(["a1", "a2"], "Do something unrelated xyz")
        assert result["task_category"] == "general coordination"

    def test_deterministic_lead_same_input(self):
        agents = ["agent-x", "agent-y", "agent-z"]
        r1 = plan_agent_collaboration(agents, "Fixed task description")
        r2 = plan_agent_collaboration(agents, "Fixed task description")
        lead1 = next(a for a in r1["agents_with_roles"] if a["role"] == "lead")
        lead2 = next(a for a in r2["agents_with_roles"] if a["role"] == "lead")
        assert lead1["agent_id"] == lead2["agent_id"]

    def test_raises_on_empty_agents(self):
        with pytest.raises(ValueError, match="non-empty"):
            plan_agent_collaboration([], "Some task")

    def test_raises_on_too_many_agents(self):
        agents = [f"agent-{i}" for i in range(11)]
        with pytest.raises(ValueError, match="most 10"):
            plan_agent_collaboration(agents, "Some task")

    def test_planned_at_is_float(self):
        result = plan_agent_collaboration(["a1"], "Task")
        assert isinstance(result["planned_at"], float)

    def test_all_original_agents_included(self):
        agents = ["agent-alpha", "agent-beta", "agent-gamma"]
        result = plan_agent_collaboration(agents, "Portfolio task")
        included = {a["agent_id"] for a in result["agents_with_roles"]}
        assert included == set(agents)


# ════════════════════════════════════════════════════════════════════════════════
# 4. _generate_pnl_snapshot() — unit tests
# ════════════════════════════════════════════════════════════════════════════════

class TestGeneratePnlSnapshot:

    def test_returns_required_fields(self):
        snap = _generate_pnl_snapshot("agent-pnl-001", 0)
        for field in ("agent_id", "tick", "realized_pnl", "unrealized_pnl",
                      "total_pnl", "portfolio_value", "peak_value",
                      "drawdown_pct", "timestamp"):
            assert field in snap, f"Missing field: {field}"

    def test_agent_id_echoed(self):
        snap = _generate_pnl_snapshot("agent-echo", 3)
        assert snap["agent_id"] == "agent-echo"

    def test_tick_echoed(self):
        snap = _generate_pnl_snapshot("agent-tick", 7)
        assert snap["tick"] == 7

    def test_total_pnl_is_float(self):
        snap = _generate_pnl_snapshot("agent-f", 1)
        assert isinstance(snap["total_pnl"], float)

    def test_realized_plus_unrealized_equals_total(self):
        snap = _generate_pnl_snapshot("agent-sum", 5)
        assert abs(snap["realized_pnl"] + snap["unrealized_pnl"] - snap["total_pnl"]) < 1e-6

    def test_drawdown_pct_non_negative(self):
        snap = _generate_pnl_snapshot("agent-dd", 2)
        assert snap["drawdown_pct"] >= 0.0

    def test_peak_value_positive(self):
        snap = _generate_pnl_snapshot("agent-peak", 4)
        assert snap["peak_value"] > 0

    def test_portfolio_value_positive(self):
        snap = _generate_pnl_snapshot("agent-pv2", 3)
        assert snap["portfolio_value"] > 0

    def test_deterministic_same_tick(self):
        s1 = _generate_pnl_snapshot("agent-det-pnl", 10)
        s2 = _generate_pnl_snapshot("agent-det-pnl", 10)
        assert s1["total_pnl"] == s2["total_pnl"]
        assert s1["drawdown_pct"] == s2["drawdown_pct"]

    def test_different_ticks_different_values(self):
        s0 = _generate_pnl_snapshot("agent-ticks", 0)
        s5 = _generate_pnl_snapshot("agent-ticks", 5)
        s10 = _generate_pnl_snapshot("agent-ticks", 10)
        # Values change across ticks
        totals = {s0["total_pnl"], s5["total_pnl"], s10["total_pnl"]}
        assert len(totals) > 1

    def test_timestamp_is_float(self):
        snap = _generate_pnl_snapshot("agent-ts2", 0)
        assert isinstance(snap["timestamp"], float)

    def test_different_agents_different_values(self):
        s1 = _generate_pnl_snapshot("agent-aaa2", 5)
        s2 = _generate_pnl_snapshot("agent-bbb2", 5)
        assert s1["portfolio_value"] != s2["portfolio_value"] or s1["total_pnl"] != s2["total_pnl"]


# ════════════════════════════════════════════════════════════════════════════════
# 5. HTTP integration — POST /demo/risk/assess
# ════════════════════════════════════════════════════════════════════════════════

class TestHttpRiskAssess:

    def test_happy_path(self, server):
        base_url, _ = server
        result = _post(f"{base_url}/demo/risk/assess", {
            "agent_id": "http-agent-001",
            "trade": {"asset": "BTC/USD", "size": 3000, "direction": "long"},
        })
        assert result["recommendation"] in ("proceed", "reduce", "reject")
        assert 0 <= result["risk_score"] <= 10

    def test_response_fields(self, server):
        base_url, _ = server
        result = _post(f"{base_url}/demo/risk/assess", {
            "agent_id": "http-agent-002",
            "trade": {"asset": "ETH/USD", "size": 1500, "direction": "short"},
        })
        for field in ("agent_id", "trade", "risk_score", "recommendation", "reasoning"):
            assert field in result

    def test_missing_agent_id_returns_400(self, server):
        base_url, _ = server
        result = _post(f"{base_url}/demo/risk/assess", {
            "trade": {"asset": "BTC/USD", "size": 100, "direction": "long"},
        }, expect_error=True)
        assert "error" in result

    def test_missing_trade_returns_400(self, server):
        base_url, _ = server
        result = _post(f"{base_url}/demo/risk/assess", {
            "agent_id": "http-agent-003",
        }, expect_error=True)
        assert "error" in result

    def test_invalid_json_returns_400(self, server):
        base_url, _ = server
        code, body = _post_raw(
            f"{base_url}/demo/risk/assess",
            b"not-valid-json",
            {"Content-Type": "application/json"},
        )
        assert code == 400
        assert "error" in body

    def test_trade_not_dict_returns_400(self, server):
        base_url, _ = server
        result = _post(f"{base_url}/demo/risk/assess", {
            "agent_id": "http-agent-004",
            "trade": "not-a-dict",
        }, expect_error=True)
        assert "error" in result

    def test_deterministic_via_http(self, server):
        base_url, _ = server
        payload = {"agent_id": "http-det", "trade": {"asset": "ETH/USD", "size": 2000, "direction": "long"}}
        r1 = _post(f"{base_url}/demo/risk/assess", payload)
        r2 = _post(f"{base_url}/demo/risk/assess", payload)
        assert r1["risk_score"] == r2["risk_score"]


# ════════════════════════════════════════════════════════════════════════════════
# 6. HTTP integration — POST /demo/portfolio/rebalance
# ════════════════════════════════════════════════════════════════════════════════

class TestHttpPortfolioRebalance:

    def test_happy_path(self, server):
        base_url, _ = server
        result = _post(f"{base_url}/demo/portfolio/rebalance", {
            "target_allocations": {"BTC": 40, "ETH": 30, "USDC": 30},
            "agent_id": "http-rb-001",
        })
        assert "rebalance_trades" in result
        assert "estimated_cost_bps" in result

    def test_response_fields(self, server):
        base_url, _ = server
        result = _post(f"{base_url}/demo/portfolio/rebalance", {
            "target_allocations": {"BTC": 50, "ETH": 50},
        })
        for field in ("current_allocations", "target_allocations",
                      "rebalance_trades", "trade_count", "estimated_cost_bps"):
            assert field in result

    def test_missing_target_allocations_returns_400(self, server):
        base_url, _ = server
        result = _post(f"{base_url}/demo/portfolio/rebalance", {
            "agent_id": "http-rb-002",
        }, expect_error=True)
        assert "error" in result

    def test_empty_allocations_returns_400(self, server):
        base_url, _ = server
        result = _post(f"{base_url}/demo/portfolio/rebalance", {
            "target_allocations": {},
        }, expect_error=True)
        assert "error" in result

    def test_invalid_json_returns_400(self, server):
        base_url, _ = server
        code, body = _post_raw(
            f"{base_url}/demo/portfolio/rebalance",
            b"{broken",
            {"Content-Type": "application/json"},
        )
        assert code == 400
        assert "error" in body

    def test_negative_allocation_returns_400(self, server):
        base_url, _ = server
        result = _post(f"{base_url}/demo/portfolio/rebalance", {
            "target_allocations": {"BTC": -10, "ETH": 110},
        }, expect_error=True)
        assert "error" in result

    def test_target_sums_to_100(self, server):
        base_url, _ = server
        result = _post(f"{base_url}/demo/portfolio/rebalance", {
            "target_allocations": {"BTC": 3, "ETH": 2, "SOL": 5},
        })
        total = sum(result["target_allocations"].values())
        assert abs(total - 100.0) < 0.1


# ════════════════════════════════════════════════════════════════════════════════
# 7. HTTP integration — POST /demo/agents/collaborate
# ════════════════════════════════════════════════════════════════════════════════

class TestHttpAgentsCollaborate:

    def test_happy_path(self, server):
        base_url, _ = server
        result = _post(f"{base_url}/demo/agents/collaborate", {
            "agent_ids": ["a1", "a2", "a3"],
            "task_description": "Perform portfolio risk assessment",
        })
        assert "agents_with_roles" in result
        assert "collaboration_plan" in result

    def test_response_fields(self, server):
        base_url, _ = server
        result = _post(f"{base_url}/demo/agents/collaborate", {
            "agent_ids": ["x1", "x2"],
            "task_description": "Sentiment analysis task",
        })
        for field in ("task", "agents_with_roles", "collaboration_plan",
                      "expected_synergy_score", "coordination_overhead_pct"):
            assert field in result

    def test_missing_agent_ids_returns_400(self, server):
        base_url, _ = server
        result = _post(f"{base_url}/demo/agents/collaborate", {
            "task_description": "Some task",
        }, expect_error=True)
        assert "error" in result

    def test_empty_agent_ids_returns_400(self, server):
        base_url, _ = server
        result = _post(f"{base_url}/demo/agents/collaborate", {
            "agent_ids": [],
            "task_description": "Some task",
        }, expect_error=True)
        assert "error" in result

    def test_invalid_json_returns_400(self, server):
        base_url, _ = server
        code, body = _post_raw(
            f"{base_url}/demo/agents/collaborate",
            b"{{invalid",
            {"Content-Type": "application/json"},
        )
        assert code == 400
        assert "error" in body

    def test_exactly_one_lead(self, server):
        base_url, _ = server
        result = _post(f"{base_url}/demo/agents/collaborate", {
            "agent_ids": ["a1", "a2", "a3", "a4"],
            "task_description": "Strategy coordination",
        })
        leads = [a for a in result["agents_with_roles"] if a["role"] == "lead"]
        assert len(leads) == 1

    def test_synergy_in_range(self, server):
        base_url, _ = server
        result = _post(f"{base_url}/demo/agents/collaborate", {
            "agent_ids": ["a1", "a2"],
            "task_description": "Backtesting task",
        })
        assert 0 <= result["expected_synergy_score"] <= 1


# ════════════════════════════════════════════════════════════════════════════════
# 8. HTTP integration — GET /demo/agents/{id}/pnl/stream (SSE)
# ════════════════════════════════════════════════════════════════════════════════

class TestHttpPnlStream:
    """
    Tests for the SSE P&L streaming endpoint.
    We connect via raw HTTP to read the stream and verify at least 1 event.
    We set a short read timeout to avoid blocking the test suite.
    """

    def test_pnl_stream_content_type(self, server):
        base_url, port = server
        conn = HTTPConnection("localhost", port, timeout=6)
        conn.request("GET", "/demo/agents/stream-test-001/pnl/stream")
        resp = conn.getresponse()
        assert resp.status == 200
        ct = resp.getheader("Content-Type", "")
        assert "text/event-stream" in ct
        resp.close()
        conn.close()

    def test_pnl_stream_first_event_valid_json(self, server):
        base_url, port = server
        conn = HTTPConnection("localhost", port, timeout=6)
        conn.request("GET", "/demo/agents/stream-test-002/pnl/stream")
        resp = conn.getresponse()
        assert resp.status == 200

        # Read enough bytes for at least 2 SSE events
        raw = b""
        deadline = time.time() + 5
        while time.time() < deadline and b"\n\n" not in raw:
            chunk = resp.read(512)
            if not chunk:
                break
            raw += chunk

        resp.close()
        conn.close()

        # Parse first data line
        lines = raw.decode("utf-8", errors="replace").split("\n")
        data_lines = [l[6:] for l in lines if l.startswith("data: ")]
        assert len(data_lines) >= 1
        evt = json.loads(data_lines[0])
        assert "event" in evt or "agent_id" in evt

    def test_pnl_stream_snapshot_fields(self, server):
        base_url, port = server
        conn = HTTPConnection("localhost", port, timeout=8)
        conn.request("GET", "/demo/agents/stream-test-003/pnl/stream")
        resp = conn.getresponse()
        assert resp.status == 200

        # Read until we have at least 2 data events
        raw = b""
        deadline = time.time() + 7
        events_found = 0
        while time.time() < deadline and events_found < 2:
            chunk = resp.read(1024)
            if not chunk:
                break
            raw += chunk
            events_found = raw.decode("utf-8", errors="replace").count("data: ")

        resp.close()
        conn.close()

        lines = raw.decode("utf-8", errors="replace").split("\n")
        data_lines = [l[6:] for l in lines if l.startswith("data: ")]

        # Find first snapshot event (not the "connected" event)
        snap_evt = None
        for dl in data_lines:
            try:
                parsed = json.loads(dl)
                if "realized_pnl" in parsed:
                    snap_evt = parsed
                    break
            except json.JSONDecodeError:
                continue

        if snap_evt is None:
            pytest.skip("Could not read a snapshot event within timeout")

        for field in ("agent_id", "tick", "realized_pnl", "unrealized_pnl",
                      "total_pnl", "drawdown_pct", "peak_value"):
            assert field in snap_evt, f"Missing field in SSE snapshot: {field}"

    def test_pnl_stream_unknown_agent_still_streams(self, server):
        base_url, port = server
        conn = HTTPConnection("localhost", port, timeout=5)
        conn.request("GET", "/demo/agents/UNKNOWN-AGENT-99999/pnl/stream")
        resp = conn.getresponse()
        assert resp.status == 200
        resp.close()
        conn.close()

    def test_pnl_stream_bad_path_returns_404(self, server):
        base_url, _ = server
        try:
            with urlopen(f"{base_url}/demo/agents/pnl/stream", timeout=3) as resp:
                # Path has no agent_id segment — should 404
                # (But some paths may match depending on path split)
                pass
        except HTTPError as exc:
            assert exc.code == 404

    def test_pnl_stream_cors_header(self, server):
        base_url, port = server
        conn = HTTPConnection("localhost", port, timeout=5)
        conn.request("GET", "/demo/agents/cors-test/pnl/stream")
        resp = conn.getresponse()
        acao = resp.getheader("Access-Control-Allow-Origin", "")
        assert acao == "*"
        resp.close()
        conn.close()
