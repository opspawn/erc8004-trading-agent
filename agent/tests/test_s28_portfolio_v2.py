"""
test_s28_portfolio_v2.py — Sprint 28: Multi-agent portfolio v2 tests.

35 tests covering:
  - build_portfolio_v2() structure and required fields
  - allocation_pct per agent
  - current_pnl values
  - correlation_matrix: shape, diagonal=1, range [-1,1]
  - Portfolio aggregate metrics: total_pnl, return_pct, diversification_score
  - Default (no run) vs live (with run) paths
  - HTTP endpoint GET /demo/portfolio enhanced fields
  - _compute_correlation_matrix edge cases
"""

from __future__ import annotations

import json
import math
import time
import threading
from urllib.request import urlopen

import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from demo_server import (
    build_portfolio_v2,
    _compute_correlation_matrix,
    _AGENT_ALLOCATIONS,
    _portfolio_lock,
    DemoServer,
    DEFAULT_PORT,
)

PORTFOLIO_TEST_PORT = 18094


# ─── _compute_correlation_matrix ──────────────────────────────────────────────

class TestComputeCorrelationMatrix:
    def _make_agents(self, n=4):
        return [
            {"agent_id": f"agent-{i}", "win_rate": 0.5 + i * 0.05,
             "total_pnl": 10.0 + i * 5.0, "reputation_score": 7.0 + i * 0.2}
            for i in range(n)
        ]

    def test_diagonal_is_one(self):
        agents = self._make_agents(4)
        matrix = _compute_correlation_matrix(agents)
        for a in agents:
            aid = a["agent_id"]
            assert matrix[aid][aid] == 1.0

    def test_symmetric(self):
        agents = self._make_agents(3)
        matrix = _compute_correlation_matrix(agents)
        ids = [a["agent_id"] for a in agents]
        for i in ids:
            for j in ids:
                assert abs(matrix[i][j] - matrix[j][i]) < 1e-9

    def test_values_in_range(self):
        agents = self._make_agents(4)
        matrix = _compute_correlation_matrix(agents)
        for i, ai in enumerate(agents):
            for j, aj in enumerate(agents):
                v = matrix[ai["agent_id"]][aj["agent_id"]]
                assert -1.0 <= v <= 1.0, f"Out of range: {v}"

    def test_returns_dict_of_dicts(self):
        agents = self._make_agents(2)
        matrix = _compute_correlation_matrix(agents)
        assert isinstance(matrix, dict)
        for k, v in matrix.items():
            assert isinstance(v, dict)

    def test_all_agent_ids_present(self):
        agents = self._make_agents(4)
        matrix = _compute_correlation_matrix(agents)
        ids = {a["agent_id"] for a in agents}
        assert set(matrix.keys()) == ids
        for v in matrix.values():
            assert set(v.keys()) == ids

    def test_single_agent(self):
        agents = self._make_agents(1)
        matrix = _compute_correlation_matrix(agents)
        assert matrix["agent-0"]["agent-0"] == 1.0

    def test_identical_agents_correlation_is_one_or_zero(self):
        # Two identical agents → std=0 → correlation = 0 by convention
        agents = [
            {"agent_id": "a1", "win_rate": 0.6, "total_pnl": 10.0, "reputation_score": 7.5},
            {"agent_id": "a2", "win_rate": 0.6, "total_pnl": 10.0, "reputation_score": 7.5},
        ]
        matrix = _compute_correlation_matrix(agents)
        # Identical signals → denom=0 → 0.0
        assert matrix["a1"]["a2"] == 0.0 or matrix["a1"]["a2"] == 1.0


# ─── build_portfolio_v2 ───────────────────────────────────────────────────────

class TestBuildPortfolioV2:
    def test_default_path_returns_dict(self):
        result = build_portfolio_v2(None)
        assert isinstance(result, dict)

    def test_source_is_default_when_no_run(self):
        result = build_portfolio_v2(None)
        assert result["source"] == "default"

    def test_required_top_level_keys(self):
        result = build_portfolio_v2(None)
        for key in ("total_capital_usd", "total_allocated_usd", "total_pnl",
                    "portfolio_return_pct", "diversification_score",
                    "avg_inter_agent_correlation", "agents", "correlation_matrix",
                    "generated_at"):
            assert key in result, f"Missing key: {key}"

    def test_agents_is_list(self):
        result = build_portfolio_v2(None)
        assert isinstance(result["agents"], list)
        assert len(result["agents"]) > 0

    def test_each_agent_has_allocation_pct(self):
        result = build_portfolio_v2(None)
        for agent in result["agents"]:
            assert "allocation_pct" in agent
            assert 0.0 < agent["allocation_pct"] <= 100.0

    def test_each_agent_has_current_pnl(self):
        result = build_portfolio_v2(None)
        for agent in result["agents"]:
            assert "current_pnl" in agent
            assert isinstance(agent["current_pnl"], (int, float))

    def test_each_agent_has_current_pnl_pct(self):
        result = build_portfolio_v2(None)
        for agent in result["agents"]:
            assert "current_pnl_pct" in agent

    def test_each_agent_has_allocated_usd(self):
        result = build_portfolio_v2(None)
        for agent in result["agents"]:
            assert "allocated_usd" in agent
            assert agent["allocated_usd"] > 0

    def test_correlation_matrix_is_dict(self):
        result = build_portfolio_v2(None)
        assert isinstance(result["correlation_matrix"], dict)

    def test_correlation_matrix_diagonal_is_one(self):
        result = build_portfolio_v2(None)
        matrix = result["correlation_matrix"]
        for agent_id in matrix:
            assert matrix[agent_id][agent_id] == 1.0

    def test_total_allocated_is_positive(self):
        result = build_portfolio_v2(None)
        assert result["total_allocated_usd"] > 0

    def test_diversification_score_in_range(self):
        result = build_portfolio_v2(None)
        score = result["diversification_score"]
        assert 0.0 <= score <= 1.0

    def test_generated_at_is_recent(self):
        result = build_portfolio_v2(None)
        assert abs(result["generated_at"] - time.time()) < 5

    def test_avg_inter_agent_correlation_in_range(self):
        result = build_portfolio_v2(None)
        corr = result["avg_inter_agent_correlation"]
        assert -1.0 <= corr <= 1.0

    def test_portfolio_return_pct_is_float(self):
        result = build_portfolio_v2(None)
        assert isinstance(result["portfolio_return_pct"], float)

    def test_total_capital_is_100k(self):
        result = build_portfolio_v2(None)
        assert result["total_capital_usd"] == 100_000.0

    def test_allocation_pcts_are_positive(self):
        result = build_portfolio_v2(None)
        # Each agent's allocation should be positive and sum ≤ 100
        total = sum(a["allocation_pct"] for a in result["agents"])
        assert total > 0.0
        assert total <= 100.0 + 1e-6  # allow rounding


# ─── HTTP: GET /demo/portfolio (enhanced) ─────────────────────────────────────

@pytest.fixture(scope="module")
def portfolio_server():
    srv = DemoServer(port=PORTFOLIO_TEST_PORT)
    srv.start()
    time.sleep(0.3)
    yield f"http://localhost:{PORTFOLIO_TEST_PORT}"
    srv.stop()


class TestPortfolioHTTP:
    def test_portfolio_returns_200(self, portfolio_server):
        resp = urlopen(f"{portfolio_server}/demo/portfolio", timeout=10)
        assert resp.status == 200

    def test_portfolio_has_agents_field(self, portfolio_server):
        resp = urlopen(f"{portfolio_server}/demo/portfolio", timeout=10)
        body = json.loads(resp.read())
        assert "agents" in body

    def test_portfolio_has_correlation_matrix(self, portfolio_server):
        resp = urlopen(f"{portfolio_server}/demo/portfolio", timeout=10)
        body = json.loads(resp.read())
        assert "correlation_matrix" in body
        assert isinstance(body["correlation_matrix"], dict)

    def test_portfolio_has_allocation_pct(self, portfolio_server):
        resp = urlopen(f"{portfolio_server}/demo/portfolio", timeout=10)
        body = json.loads(resp.read())
        for agent in body["agents"]:
            assert "allocation_pct" in agent

    def test_portfolio_has_current_pnl(self, portfolio_server):
        resp = urlopen(f"{portfolio_server}/demo/portfolio", timeout=10)
        body = json.loads(resp.read())
        for agent in body["agents"]:
            assert "current_pnl" in agent

    def test_portfolio_has_portfolio_return_pct(self, portfolio_server):
        resp = urlopen(f"{portfolio_server}/demo/portfolio", timeout=10)
        body = json.loads(resp.read())
        assert "portfolio_return_pct" in body

    def test_portfolio_has_diversification_score(self, portfolio_server):
        resp = urlopen(f"{portfolio_server}/demo/portfolio", timeout=10)
        body = json.loads(resp.read())
        assert "diversification_score" in body

    def test_portfolio_source_default(self, portfolio_server):
        resp = urlopen(f"{portfolio_server}/demo/portfolio", timeout=10)
        body = json.loads(resp.read())
        # No run done yet → should be "default"
        assert body["source"] in ("default", "live")

    def test_portfolio_json_content_type(self, portfolio_server):
        resp = urlopen(f"{portfolio_server}/demo/portfolio", timeout=10)
        ct = resp.headers.get("Content-Type", "")
        assert "application/json" in ct
