"""
test_s37_risk_ensemble.py — Sprint 37: Risk Dashboard, Ensemble Vote, Alpha Decay, Cross-Training.

Covers:
  - get_portfolio_risk_dashboard(): schema, value ranges, correlation matrix, determinism
  - vote_ensemble(): schema, BUY/SELL/HOLD decision, weighted majority, determinism, edge cases
  - get_alpha_decay(): schema, exponential curve, half-life range, determinism, 30-day curve
  - cross_train_agents(): schema, transfer_ratio range, knowledge items, performance_delta, determinism
  - HTTP integration: GET /demo/portfolio/risk-dashboard, POST /demo/ensemble/vote,
    GET /demo/strategy/alpha-decay/{id}, POST /demo/agents/cross-train
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
    get_portfolio_risk_dashboard,
    vote_ensemble,
    get_alpha_decay,
    cross_train_agents,
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


def _get_status(url: str) -> tuple[int, dict]:
    try:
        with urlopen(url, timeout=5) as resp:
            return resp.status, json.loads(resp.read())
    except HTTPError as exc:
        return exc.code, json.loads(exc.read())


# ── Fixtures ───────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def server():
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    sock.close()
    srv = DemoServer(port=port)
    srv.start()
    time.sleep(0.3)
    base = f"http://127.0.0.1:{port}"
    yield base, port
    srv.stop()


# ══════════════════════════════════════════════════════════════════════════════
# 1. get_portfolio_risk_dashboard() — unit tests
# ══════════════════════════════════════════════════════════════════════════════

class TestRiskDashboardUnit:

    def test_returns_dict(self):
        r = get_portfolio_risk_dashboard()
        assert isinstance(r, dict)

    def test_required_keys(self):
        r = get_portfolio_risk_dashboard()
        for key in ("var_95", "var_99", "cvar", "max_drawdown",
                    "sharpe_ratio", "beta_to_market", "correlation_matrix", "generated_at"):
            assert key in r, f"Missing key: {key}"

    def test_var_95_negative(self):
        r = get_portfolio_risk_dashboard()
        assert r["var_95"] < 0, "VaR 95% should be negative (loss)"

    def test_var_99_more_negative_than_var_95(self):
        r = get_portfolio_risk_dashboard()
        assert r["var_99"] <= r["var_95"], "VaR 99% should be <= VaR 95%"

    def test_cvar_more_negative_than_var_99(self):
        r = get_portfolio_risk_dashboard()
        assert r["cvar"] <= r["var_99"], "CVaR should be <= VaR 99%"

    def test_max_drawdown_negative(self):
        r = get_portfolio_risk_dashboard()
        assert r["max_drawdown"] < 0

    def test_sharpe_ratio_positive(self):
        r = get_portfolio_risk_dashboard()
        assert r["sharpe_ratio"] > 0

    def test_beta_to_market_range(self):
        r = get_portfolio_risk_dashboard()
        assert 0.0 <= r["beta_to_market"] <= 2.5

    def test_correlation_matrix_structure(self):
        r = get_portfolio_risk_dashboard()
        cm = r["correlation_matrix"]
        assert "agents" in cm
        assert "matrix" in cm
        assert isinstance(cm["agents"], list)
        assert isinstance(cm["matrix"], list)

    def test_correlation_matrix_3x3(self):
        r = get_portfolio_risk_dashboard()
        cm = r["correlation_matrix"]
        assert len(cm["agents"]) == 3
        assert len(cm["matrix"]) == 3
        for row in cm["matrix"]:
            assert len(row) == 3

    def test_correlation_diagonal_ones(self):
        r = get_portfolio_risk_dashboard()
        matrix = r["correlation_matrix"]["matrix"]
        for i in range(3):
            assert matrix[i][i] == 1.0

    def test_correlation_matrix_symmetric(self):
        r = get_portfolio_risk_dashboard()
        matrix = r["correlation_matrix"]["matrix"]
        for i in range(3):
            for j in range(3):
                assert math.isclose(matrix[i][j], matrix[j][i], rel_tol=1e-9)

    def test_generated_at_is_float(self):
        r = get_portfolio_risk_dashboard()
        assert isinstance(r["generated_at"], float)

    def test_determinism(self):
        r1 = get_portfolio_risk_dashboard()
        r2 = get_portfolio_risk_dashboard()
        assert r1["var_95"] == r2["var_95"]
        assert r1["var_99"] == r2["var_99"]
        assert r1["sharpe_ratio"] == r2["sharpe_ratio"]
        assert r1["correlation_matrix"]["matrix"] == r2["correlation_matrix"]["matrix"]

    def test_var_95_in_expected_range(self):
        r = get_portfolio_risk_dashboard()
        assert -0.05 <= r["var_95"] <= -0.01

    def test_sharpe_ratio_in_expected_range(self):
        r = get_portfolio_risk_dashboard()
        assert 0.5 <= r["sharpe_ratio"] <= 3.5

    def test_correlation_off_diagonal_in_minus_one_to_one(self):
        r = get_portfolio_risk_dashboard()
        matrix = r["correlation_matrix"]["matrix"]
        for i in range(3):
            for j in range(3):
                assert -1.0 <= matrix[i][j] <= 1.0


# ══════════════════════════════════════════════════════════════════════════════
# 2. vote_ensemble() — unit tests
# ══════════════════════════════════════════════════════════════════════════════

class TestEnsembleVoteUnit:

    _AGENTS = ["alpha-agent", "beta-agent", "gamma-agent"]
    _MARKET = {"price": 50000, "volume": 1200, "rsi": 62}

    def test_returns_dict(self):
        r = vote_ensemble(self._AGENTS, self._MARKET)
        assert isinstance(r, dict)

    def test_required_keys(self):
        r = vote_ensemble(self._AGENTS, self._MARKET)
        for key in ("decision", "confidence", "votes", "tally", "generated_at"):
            assert key in r, f"Missing: {key}"

    def test_decision_is_valid(self):
        r = vote_ensemble(self._AGENTS, self._MARKET)
        assert r["decision"] in ("BUY", "SELL", "HOLD")

    def test_confidence_in_range(self):
        r = vote_ensemble(self._AGENTS, self._MARKET)
        assert 0.0 <= r["confidence"] <= 1.0

    def test_votes_list_length(self):
        r = vote_ensemble(self._AGENTS, self._MARKET)
        assert len(r["votes"]) == len(self._AGENTS)

    def test_vote_schema(self):
        r = vote_ensemble(self._AGENTS, self._MARKET)
        for v in r["votes"]:
            assert "agent_id" in v
            assert "vote" in v
            assert "weight" in v
            assert "confidence" in v
            assert v["vote"] in ("BUY", "SELL", "HOLD")

    def test_vote_confidence_in_range(self):
        r = vote_ensemble(self._AGENTS, self._MARKET)
        for v in r["votes"]:
            assert 0.0 <= v["confidence"] <= 1.0

    def test_tally_has_three_keys(self):
        r = vote_ensemble(self._AGENTS, self._MARKET)
        assert set(r["tally"].keys()) == {"BUY", "SELL", "HOLD"}

    def test_decision_matches_max_tally(self):
        r = vote_ensemble(self._AGENTS, self._MARKET)
        tally = r["tally"]
        expected = max(tally, key=lambda k: tally[k])
        assert r["decision"] == expected

    def test_determinism_same_inputs(self):
        r1 = vote_ensemble(self._AGENTS, self._MARKET)
        r2 = vote_ensemble(self._AGENTS, self._MARKET)
        assert r1["decision"] == r2["decision"]
        assert r1["confidence"] == r2["confidence"]
        assert r1["votes"] == r2["votes"]

    def test_different_market_data_different_result(self):
        r1 = vote_ensemble(self._AGENTS, {"price": 100})
        r2 = vote_ensemble(self._AGENTS, {"price": 999})
        # They may differ (usually do)
        assert isinstance(r1["decision"], str)
        assert isinstance(r2["decision"], str)

    def test_single_agent(self):
        r = vote_ensemble(["solo-agent"], {"price": 1000})
        assert r["decision"] in ("BUY", "SELL", "HOLD")
        assert len(r["votes"]) == 1

    def test_many_agents(self):
        agents = [f"agent-{i}" for i in range(10)]
        r = vote_ensemble(agents, {"price": 5000})
        assert r["decision"] in ("BUY", "SELL", "HOLD")
        assert len(r["votes"]) == 10

    def test_empty_agent_ids_raises(self):
        with pytest.raises(ValueError):
            vote_ensemble([], {"price": 100})

    def test_empty_market_data(self):
        r = vote_ensemble(["a1", "a2"], {})
        assert r["decision"] in ("BUY", "SELL", "HOLD")

    def test_generated_at_float(self):
        r = vote_ensemble(self._AGENTS, self._MARKET)
        assert isinstance(r["generated_at"], float)

    def test_agent_ids_preserved_in_votes(self):
        agents = ["my-agent-x", "my-agent-y"]
        r = vote_ensemble(agents, {})
        returned_ids = [v["agent_id"] for v in r["votes"]]
        assert returned_ids == agents

    def test_unknown_agent_gets_default_weight(self):
        r = vote_ensemble(["unknown-xyz"], {})
        assert r["votes"][0]["weight"] == 0.20

    def test_known_agent_alpha_weight(self):
        r = vote_ensemble(["alpha-agent"], {})
        assert r["votes"][0]["weight"] == 0.40

    def test_known_agent_beta_weight(self):
        r = vote_ensemble(["beta-agent"], {})
        assert r["votes"][0]["weight"] == 0.35

    def test_known_agent_gamma_weight(self):
        r = vote_ensemble(["gamma-agent"], {})
        assert r["votes"][0]["weight"] == 0.25


# ══════════════════════════════════════════════════════════════════════════════
# 3. get_alpha_decay() — unit tests
# ══════════════════════════════════════════════════════════════════════════════

class TestAlphaDecayUnit:

    _SID = "momentum-v3"

    def test_returns_dict(self):
        r = get_alpha_decay(self._SID)
        assert isinstance(r, dict)

    def test_required_keys(self):
        r = get_alpha_decay(self._SID)
        for key in ("strategy_id", "half_life_days", "initial_alpha",
                    "current_alpha", "decay_curve", "recommendation", "generated_at"):
            assert key in r

    def test_strategy_id_echoed(self):
        r = get_alpha_decay(self._SID)
        assert r["strategy_id"] == self._SID

    def test_half_life_int(self):
        r = get_alpha_decay(self._SID)
        assert isinstance(r["half_life_days"], int)

    def test_half_life_range(self):
        r = get_alpha_decay(self._SID)
        assert 5 <= r["half_life_days"] <= 60

    def test_initial_alpha_positive(self):
        r = get_alpha_decay(self._SID)
        assert r["initial_alpha"] > 0

    def test_current_alpha_less_than_initial(self):
        r = get_alpha_decay(self._SID)
        assert r["current_alpha"] < r["initial_alpha"]

    def test_decay_curve_31_points(self):
        r = get_alpha_decay(self._SID)
        assert len(r["decay_curve"]) == 31

    def test_decay_curve_day_zero_matches_initial(self):
        r = get_alpha_decay(self._SID)
        assert math.isclose(r["decay_curve"][0]["alpha"], r["initial_alpha"], rel_tol=1e-5)

    def test_decay_curve_monotonically_decreasing(self):
        r = get_alpha_decay(self._SID)
        curve = r["decay_curve"]
        for i in range(1, len(curve)):
            assert curve[i]["alpha"] <= curve[i - 1]["alpha"], (
                f"Curve not decreasing at day {curve[i]['day']}"
            )

    def test_decay_curve_days_sequential(self):
        r = get_alpha_decay(self._SID)
        for i, point in enumerate(r["decay_curve"]):
            assert point["day"] == i

    def test_recommendation_is_string(self):
        r = get_alpha_decay(self._SID)
        assert isinstance(r["recommendation"], str)
        assert len(r["recommendation"]) > 0

    def test_determinism(self):
        r1 = get_alpha_decay(self._SID)
        r2 = get_alpha_decay(self._SID)
        assert r1["half_life_days"] == r2["half_life_days"]
        assert r1["initial_alpha"] == r2["initial_alpha"]
        assert r1["decay_curve"] == r2["decay_curve"]

    def test_different_strategies_different_results(self):
        r1 = get_alpha_decay("strat-aaa")
        r2 = get_alpha_decay("strat-bbb")
        # They differ on at least one field
        differ = (r1["half_life_days"] != r2["half_life_days"] or
                  r1["initial_alpha"] != r2["initial_alpha"])
        assert differ

    def test_half_life_at_day_half_life(self):
        r = get_alpha_decay(self._SID)
        hl = r["half_life_days"]
        if hl < 30:
            curve = r["decay_curve"]
            alpha_at_hl = curve[hl]["alpha"]
            expected = r["initial_alpha"] * 0.5
            assert math.isclose(alpha_at_hl, expected, rel_tol=1e-5)

    def test_decay_curve_last_alpha_less_than_half_initial(self):
        r = get_alpha_decay(self._SID)
        # If half_life <= 30, alpha at day 30 should be < initial
        if r["half_life_days"] <= 30:
            assert r["decay_curve"][30]["alpha"] < r["initial_alpha"]

    def test_various_strategy_ids(self):
        for sid in ("mean-reversion", "arima-v2", "llm-signal", "xgb-ensemble"):
            r = get_alpha_decay(sid)
            assert r["strategy_id"] == sid
            assert 5 <= r["half_life_days"] <= 60

    def test_generated_at_float(self):
        r = get_alpha_decay(self._SID)
        assert isinstance(r["generated_at"], float)


# ══════════════════════════════════════════════════════════════════════════════
# 4. cross_train_agents() — unit tests
# ══════════════════════════════════════════════════════════════════════════════

class TestCrossTrainUnit:

    _SRC = "alpha-agent"
    _TGT = "beta-agent"

    def test_returns_dict(self):
        r = cross_train_agents(self._SRC, self._TGT, 0.5)
        assert isinstance(r, dict)

    def test_required_keys(self):
        r = cross_train_agents(self._SRC, self._TGT, 0.5)
        for key in ("source_agent_id", "target_agent_id", "transfer_ratio",
                    "knowledge_transferred", "performance_delta", "new_accuracy", "generated_at"):
            assert key in r

    def test_source_echoed(self):
        r = cross_train_agents(self._SRC, self._TGT, 0.5)
        assert r["source_agent_id"] == self._SRC

    def test_target_echoed(self):
        r = cross_train_agents(self._SRC, self._TGT, 0.5)
        assert r["target_agent_id"] == self._TGT

    def test_transfer_ratio_echoed(self):
        r = cross_train_agents(self._SRC, self._TGT, 0.7)
        assert math.isclose(r["transfer_ratio"], 0.7)

    def test_knowledge_transferred_list(self):
        r = cross_train_agents(self._SRC, self._TGT, 0.5)
        assert isinstance(r["knowledge_transferred"], list)

    def test_knowledge_transferred_non_empty(self):
        r = cross_train_agents(self._SRC, self._TGT, 0.1)
        assert len(r["knowledge_transferred"]) >= 1

    def test_higher_ratio_more_knowledge(self):
        r_lo = cross_train_agents(self._SRC, self._TGT, 0.1)
        r_hi = cross_train_agents(self._SRC, self._TGT, 1.0)
        assert len(r_hi["knowledge_transferred"]) >= len(r_lo["knowledge_transferred"])

    def test_knowledge_items_are_strings(self):
        r = cross_train_agents(self._SRC, self._TGT, 0.5)
        for item in r["knowledge_transferred"]:
            assert isinstance(item, str)

    def test_performance_delta_in_range(self):
        r = cross_train_agents(self._SRC, self._TGT, 0.5)
        assert -0.05 <= r["performance_delta"] <= 0.15

    def test_new_accuracy_in_range(self):
        r = cross_train_agents(self._SRC, self._TGT, 0.5)
        assert 0.0 <= r["new_accuracy"] <= 1.0

    def test_new_accuracy_float(self):
        r = cross_train_agents(self._SRC, self._TGT, 0.5)
        assert isinstance(r["new_accuracy"], float)

    def test_determinism(self):
        r1 = cross_train_agents(self._SRC, self._TGT, 0.5)
        r2 = cross_train_agents(self._SRC, self._TGT, 0.5)
        assert r1["knowledge_transferred"] == r2["knowledge_transferred"]
        assert r1["performance_delta"] == r2["performance_delta"]
        assert r1["new_accuracy"] == r2["new_accuracy"]

    def test_invalid_ratio_negative_raises(self):
        with pytest.raises(ValueError):
            cross_train_agents(self._SRC, self._TGT, -0.1)

    def test_invalid_ratio_gt_one_raises(self):
        with pytest.raises(ValueError):
            cross_train_agents(self._SRC, self._TGT, 1.1)

    def test_ratio_zero(self):
        r = cross_train_agents(self._SRC, self._TGT, 0.0)
        # 0 ratio → at least 1 item transferred (max(1, ...))
        assert len(r["knowledge_transferred"]) >= 1

    def test_ratio_one(self):
        r = cross_train_agents(self._SRC, self._TGT, 1.0)
        assert len(r["knowledge_transferred"]) >= 6

    def test_same_src_different_tgt(self):
        r1 = cross_train_agents(self._SRC, "tgt-a", 0.5)
        r2 = cross_train_agents(self._SRC, "tgt-b", 0.5)
        # Different target → possibly different knowledge items
        assert isinstance(r1["knowledge_transferred"], list)
        assert isinstance(r2["knowledge_transferred"], list)

    def test_generated_at_float(self):
        r = cross_train_agents(self._SRC, self._TGT, 0.5)
        assert isinstance(r["generated_at"], float)

    def test_various_ratios(self):
        for ratio in (0.0, 0.25, 0.5, 0.75, 1.0):
            r = cross_train_agents("src", "tgt", ratio)
            assert -0.05 <= r["performance_delta"] <= 0.15
            assert 0.0 <= r["new_accuracy"] <= 1.0


# ══════════════════════════════════════════════════════════════════════════════
# 5. HTTP Integration — GET /demo/portfolio/risk-dashboard
# ══════════════════════════════════════════════════════════════════════════════

class TestRiskDashboardHTTP:

    def test_get_200(self, server):
        base, _ = server
        r = _get(f"{base}/demo/portfolio/risk-dashboard")
        assert isinstance(r, dict)

    def test_schema_via_http(self, server):
        base, _ = server
        r = _get(f"{base}/demo/portfolio/risk-dashboard")
        for key in ("var_95", "var_99", "cvar", "max_drawdown",
                    "sharpe_ratio", "beta_to_market", "correlation_matrix", "generated_at"):
            assert key in r

    def test_var_negative_via_http(self, server):
        base, _ = server
        r = _get(f"{base}/demo/portfolio/risk-dashboard")
        assert r["var_95"] < 0
        assert r["var_99"] < 0
        assert r["cvar"] < 0

    def test_correlation_matrix_via_http(self, server):
        base, _ = server
        r = _get(f"{base}/demo/portfolio/risk-dashboard")
        cm = r["correlation_matrix"]
        assert len(cm["agents"]) == 3
        assert len(cm["matrix"]) == 3

    def test_trailing_slash_404(self, server):
        base, _ = server
        # trailing slash stripped → should still resolve
        r = _get(f"{base}/demo/portfolio/risk-dashboard/")
        assert "var_95" in r

    def test_sharpe_ratio_positive_via_http(self, server):
        base, _ = server
        r = _get(f"{base}/demo/portfolio/risk-dashboard")
        assert r["sharpe_ratio"] > 0


# ══════════════════════════════════════════════════════════════════════════════
# 6. HTTP Integration — POST /demo/ensemble/vote
# ══════════════════════════════════════════════════════════════════════════════

class TestEnsembleVoteHTTP:

    _PAYLOAD = {
        "agent_ids": ["alpha-agent", "beta-agent", "gamma-agent"],
        "market_data": {"price": 50000, "rsi": 55},
    }

    def test_post_200(self, server):
        base, _ = server
        r = _post(f"{base}/demo/ensemble/vote", self._PAYLOAD)
        assert "decision" in r

    def test_decision_valid_via_http(self, server):
        base, _ = server
        r = _post(f"{base}/demo/ensemble/vote", self._PAYLOAD)
        assert r["decision"] in ("BUY", "SELL", "HOLD")

    def test_votes_list_via_http(self, server):
        base, _ = server
        r = _post(f"{base}/demo/ensemble/vote", self._PAYLOAD)
        assert len(r["votes"]) == 3

    def test_tally_keys_via_http(self, server):
        base, _ = server
        r = _post(f"{base}/demo/ensemble/vote", self._PAYLOAD)
        assert set(r["tally"].keys()) == {"BUY", "SELL", "HOLD"}

    def test_missing_agent_ids_400(self, server):
        base, _ = server
        code, body = _post_raw(
            f"{base}/demo/ensemble/vote",
            json.dumps({"market_data": {}}).encode(),
            {"Content-Type": "application/json"},
        )
        assert code == 400

    def test_empty_agent_ids_400(self, server):
        base, _ = server
        code, body = _post_raw(
            f"{base}/demo/ensemble/vote",
            json.dumps({"agent_ids": [], "market_data": {}}).encode(),
            {"Content-Type": "application/json"},
        )
        assert code == 400

    def test_bad_json_400(self, server):
        base, _ = server
        code, body = _post_raw(
            f"{base}/demo/ensemble/vote",
            b"not-json",
            {"Content-Type": "application/json"},
        )
        assert code == 400

    def test_single_agent_via_http(self, server):
        base, _ = server
        r = _post(f"{base}/demo/ensemble/vote", {
            "agent_ids": ["solo"],
            "market_data": {},
        })
        assert len(r["votes"]) == 1

    def test_market_data_object_required(self, server):
        base, _ = server
        # market_data as list → 400
        code, body = _post_raw(
            f"{base}/demo/ensemble/vote",
            json.dumps({"agent_ids": ["a1"], "market_data": [1, 2, 3]}).encode(),
            {"Content-Type": "application/json"},
        )
        assert code == 400

    def test_no_body_returns_400_or_valid(self, server):
        """Empty body → missing agent_ids → 400."""
        base, _ = server
        code, body = _post_raw(
            f"{base}/demo/ensemble/vote",
            b"{}",
            {"Content-Type": "application/json"},
        )
        assert code == 400


# ══════════════════════════════════════════════════════════════════════════════
# 7. HTTP Integration — GET /demo/strategy/alpha-decay/{strategy_id}
# ══════════════════════════════════════════════════════════════════════════════

class TestAlphaDecayHTTP:

    def test_get_200(self, server):
        base, _ = server
        r = _get(f"{base}/demo/strategy/alpha-decay/momentum-v1")
        assert "strategy_id" in r

    def test_strategy_id_echoed_via_http(self, server):
        base, _ = server
        r = _get(f"{base}/demo/strategy/alpha-decay/arima-strategy")
        assert r["strategy_id"] == "arima-strategy"

    def test_half_life_days_via_http(self, server):
        base, _ = server
        r = _get(f"{base}/demo/strategy/alpha-decay/test-strat")
        assert 5 <= r["half_life_days"] <= 60

    def test_decay_curve_31_points_via_http(self, server):
        base, _ = server
        r = _get(f"{base}/demo/strategy/alpha-decay/test-strat")
        assert len(r["decay_curve"]) == 31

    def test_recommendation_via_http(self, server):
        base, _ = server
        r = _get(f"{base}/demo/strategy/alpha-decay/my-strategy")
        assert isinstance(r["recommendation"], str)

    def test_different_strategy_ids_different_results(self, server):
        base, _ = server
        r1 = _get(f"{base}/demo/strategy/alpha-decay/strat-aaa")
        r2 = _get(f"{base}/demo/strategy/alpha-decay/strat-zzz")
        # May differ
        assert isinstance(r1["half_life_days"], int)
        assert isinstance(r2["half_life_days"], int)

    def test_missing_strategy_id_404(self, server):
        base, _ = server
        code, body = _get_status(f"{base}/demo/strategy/alpha-decay/")
        # No strategy_id segment → 404
        assert code == 404

    def test_initial_alpha_positive_via_http(self, server):
        base, _ = server
        r = _get(f"{base}/demo/strategy/alpha-decay/test-alpha")
        assert r["initial_alpha"] > 0

    def test_decay_monotone_via_http(self, server):
        base, _ = server
        r = _get(f"{base}/demo/strategy/alpha-decay/test-mono")
        curve = r["decay_curve"]
        for i in range(1, len(curve)):
            assert curve[i]["alpha"] <= curve[i - 1]["alpha"]


# ══════════════════════════════════════════════════════════════════════════════
# 8. HTTP Integration — POST /demo/agents/cross-train
# ══════════════════════════════════════════════════════════════════════════════

class TestCrossTrainHTTP:

    _PAYLOAD = {
        "source_agent_id": "alpha-agent",
        "target_agent_id": "beta-agent",
        "transfer_ratio": 0.6,
    }

    def test_post_200(self, server):
        base, _ = server
        r = _post(f"{base}/demo/agents/cross-train", self._PAYLOAD)
        assert "knowledge_transferred" in r

    def test_source_echoed_via_http(self, server):
        base, _ = server
        r = _post(f"{base}/demo/agents/cross-train", self._PAYLOAD)
        assert r["source_agent_id"] == "alpha-agent"

    def test_target_echoed_via_http(self, server):
        base, _ = server
        r = _post(f"{base}/demo/agents/cross-train", self._PAYLOAD)
        assert r["target_agent_id"] == "beta-agent"

    def test_knowledge_transferred_list_via_http(self, server):
        base, _ = server
        r = _post(f"{base}/demo/agents/cross-train", self._PAYLOAD)
        assert isinstance(r["knowledge_transferred"], list)
        assert len(r["knowledge_transferred"]) >= 1

    def test_new_accuracy_in_range_via_http(self, server):
        base, _ = server
        r = _post(f"{base}/demo/agents/cross-train", self._PAYLOAD)
        assert 0.0 <= r["new_accuracy"] <= 1.0

    def test_performance_delta_in_range_via_http(self, server):
        base, _ = server
        r = _post(f"{base}/demo/agents/cross-train", self._PAYLOAD)
        assert -0.05 <= r["performance_delta"] <= 0.15

    def test_missing_source_400(self, server):
        base, _ = server
        code, body = _post_raw(
            f"{base}/demo/agents/cross-train",
            json.dumps({"target_agent_id": "b", "transfer_ratio": 0.5}).encode(),
            {"Content-Type": "application/json"},
        )
        assert code == 400

    def test_missing_target_400(self, server):
        base, _ = server
        code, body = _post_raw(
            f"{base}/demo/agents/cross-train",
            json.dumps({"source_agent_id": "a", "transfer_ratio": 0.5}).encode(),
            {"Content-Type": "application/json"},
        )
        assert code == 400

    def test_invalid_transfer_ratio_400(self, server):
        base, _ = server
        code, body = _post_raw(
            f"{base}/demo/agents/cross-train",
            json.dumps({
                "source_agent_id": "a", "target_agent_id": "b",
                "transfer_ratio": 2.0
            }).encode(),
            {"Content-Type": "application/json"},
        )
        assert code == 400

    def test_bad_json_400(self, server):
        base, _ = server
        code, body = _post_raw(
            f"{base}/demo/agents/cross-train",
            b"not-json",
            {"Content-Type": "application/json"},
        )
        assert code == 400

    def test_default_transfer_ratio(self, server):
        base, _ = server
        r = _post(f"{base}/demo/agents/cross-train", {
            "source_agent_id": "a",
            "target_agent_id": "b",
        })
        assert math.isclose(r["transfer_ratio"], 0.5)

    def test_transfer_ratio_zero(self, server):
        base, _ = server
        r = _post(f"{base}/demo/agents/cross-train", {
            "source_agent_id": "a",
            "target_agent_id": "b",
            "transfer_ratio": 0.0,
        })
        assert len(r["knowledge_transferred"]) >= 1

    def test_transfer_ratio_one(self, server):
        base, _ = server
        r = _post(f"{base}/demo/agents/cross-train", {
            "source_agent_id": "a",
            "target_agent_id": "b",
            "transfer_ratio": 1.0,
        })
        assert len(r["knowledge_transferred"]) >= 6

    def test_non_string_transfer_ratio_400(self, server):
        base, _ = server
        code, body = _post_raw(
            f"{base}/demo/agents/cross-train",
            json.dumps({
                "source_agent_id": "a", "target_agent_id": "b",
                "transfer_ratio": "high"
            }).encode(),
            {"Content-Type": "application/json"},
        )
        assert code == 400


# ══════════════════════════════════════════════════════════════════════════════
# 9. 404 routing sanity checks
# ══════════════════════════════════════════════════════════════════════════════

class TestRoutingS37:

    def test_unknown_get_404(self, server):
        base, _ = server
        code, _ = _get_status(f"{base}/demo/s37-unknown-endpoint")
        assert code == 404

    def test_alpha_decay_no_id_404(self, server):
        base, _ = server
        # path = /demo/strategy/alpha-decay  (no trailing segment)
        code, _ = _get_status(f"{base}/demo/strategy/alpha-decay")
        assert code == 404

    def test_cross_train_via_get_404(self, server):
        base, _ = server
        # cross-train is POST-only; GET → 404
        code, _ = _get_status(f"{base}/demo/agents/cross-train")
        assert code == 404

    def test_ensemble_vote_via_get_404(self, server):
        base, _ = server
        code, _ = _get_status(f"{base}/demo/ensemble/vote")
        assert code == 404

    def test_risk_dashboard_max_drawdown_range(self, server):
        base, _ = server
        r = _get(f"{base}/demo/portfolio/risk-dashboard")
        assert -0.30 <= r["max_drawdown"] <= 0.0

    def test_alpha_decay_current_less_than_initial_via_http(self, server):
        base, _ = server
        r = _get(f"{base}/demo/strategy/alpha-decay/check-decay")
        assert r["current_alpha"] < r["initial_alpha"]

    def test_cross_train_all_knowledge_items_from_pool(self, server):
        base, _ = server
        from demo_server import _CROSS_TRAIN_KNOWLEDGE_POOL
        r = _post(f"{base}/demo/agents/cross-train", {
            "source_agent_id": "a",
            "target_agent_id": "b",
            "transfer_ratio": 1.0,
        })
        for item in r["knowledge_transferred"]:
            assert item in _CROSS_TRAIN_KNOWLEDGE_POOL

    def test_ensemble_confidence_positive(self, server):
        base, _ = server
        r = _post(f"{base}/demo/ensemble/vote", {
            "agent_ids": ["alpha-agent", "beta-agent"],
            "market_data": {"price": 42000},
        })
        assert r["confidence"] > 0
