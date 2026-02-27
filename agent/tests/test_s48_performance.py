"""
test_s48_performance.py — Sprint 48: Performance summary endpoint.

Covers:
  test_performance_summary_returns_200        — HTTP GET returns 200 via live server
  test_performance_summary_has_required_fields — all required keys present
  test_performance_summary_win_rate_between_0_and_100 — win_rate in [0, 100]
  test_performance_summary_active_agents_positive     — active_agents >= 1
"""

from __future__ import annotations

import json
import sys
import os
import time
import urllib.request
import urllib.error

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from demo_server import get_s48_performance_summary, DemoServer


# ── Required fields ────────────────────────────────────────────────────────────

REQUIRED_FIELDS = [
    "total_paper_trades",
    "win_rate",
    "sharpe_ratio",
    "total_pnl",
    "avg_trade_pnl",
    "best_trade",
    "worst_trade",
    "drawdown_pct",
    "active_agents",
]


# ── Unit tests ────────────────────────────────────────────────────────────────

class TestPerformanceSummaryHasRequiredFields:
    """test_performance_summary_has_required_fields — all keys present in result."""

    def test_total_paper_trades_present(self):
        result = get_s48_performance_summary()
        assert "total_paper_trades" in result

    def test_win_rate_present(self):
        result = get_s48_performance_summary()
        assert "win_rate" in result

    def test_sharpe_ratio_present(self):
        result = get_s48_performance_summary()
        assert "sharpe_ratio" in result

    def test_total_pnl_present(self):
        result = get_s48_performance_summary()
        assert "total_pnl" in result

    def test_avg_trade_pnl_present(self):
        result = get_s48_performance_summary()
        assert "avg_trade_pnl" in result

    def test_best_trade_present(self):
        result = get_s48_performance_summary()
        assert "best_trade" in result

    def test_worst_trade_present(self):
        result = get_s48_performance_summary()
        assert "worst_trade" in result

    def test_drawdown_pct_present(self):
        result = get_s48_performance_summary()
        assert "drawdown_pct" in result

    def test_active_agents_present(self):
        result = get_s48_performance_summary()
        assert "active_agents" in result

    def test_all_required_fields_at_once(self):
        result = get_s48_performance_summary()
        missing = [f for f in REQUIRED_FIELDS if f not in result]
        assert missing == [], f"Missing fields: {missing}"

    def test_version_field_present(self):
        result = get_s48_performance_summary()
        assert "version" in result
        assert result["version"] == "S48"


class TestPerformanceSummaryWinRateBetween0And100:
    """test_performance_summary_win_rate_between_0_and_100 — win_rate in valid range."""

    def test_win_rate_is_numeric(self):
        result = get_s48_performance_summary()
        assert isinstance(result["win_rate"], (int, float))

    def test_win_rate_not_negative(self):
        result = get_s48_performance_summary()
        assert result["win_rate"] >= 0

    def test_win_rate_not_above_100(self):
        result = get_s48_performance_summary()
        assert result["win_rate"] <= 100

    def test_win_rate_in_range(self):
        result = get_s48_performance_summary()
        assert 0 <= result["win_rate"] <= 100

    def test_drawdown_pct_not_negative(self):
        result = get_s48_performance_summary()
        assert result["drawdown_pct"] >= 0

    def test_total_paper_trades_non_negative(self):
        result = get_s48_performance_summary()
        assert result["total_paper_trades"] >= 0


class TestPerformanceSummaryActiveAgentsPositive:
    """test_performance_summary_active_agents_positive — active_agents >= 1."""

    def test_active_agents_is_integer(self):
        result = get_s48_performance_summary()
        assert isinstance(result["active_agents"], int)

    def test_active_agents_at_least_one(self):
        result = get_s48_performance_summary()
        assert result["active_agents"] >= 1

    def test_active_agents_matches_swarm_size(self):
        from demo_server import _S46_SWARM_AGENTS
        result = get_s48_performance_summary()
        assert result["active_agents"] == len(_S46_SWARM_AGENTS)


# ── HTTP integration test ─────────────────────────────────────────────────────

class TestPerformanceSummaryReturns200:
    """test_performance_summary_returns_200 — GET /api/v1/performance/summary via HTTP."""

    @pytest.fixture(scope="class")
    def server(self):
        import socket
        with socket.socket() as s:
            s.bind(("", 0))
            port = s.getsockname()[1]
        srv = DemoServer(port=port)
        srv.start()
        time.sleep(0.15)
        yield port
        srv.stop()

    def test_returns_200(self, server):
        port = server
        req = urllib.request.urlopen(
            f"http://localhost:{port}/api/v1/performance/summary", timeout=5
        )
        assert req.status == 200

    def test_returns_json(self, server):
        port = server
        req = urllib.request.urlopen(
            f"http://localhost:{port}/api/v1/performance/summary", timeout=5
        )
        body = json.loads(req.read().decode())
        assert isinstance(body, dict)

    def test_http_response_has_required_fields(self, server):
        port = server
        req = urllib.request.urlopen(
            f"http://localhost:{port}/api/v1/performance/summary", timeout=5
        )
        body = json.loads(req.read().decode())
        missing = [f for f in REQUIRED_FIELDS if f not in body]
        assert missing == [], f"Missing fields via HTTP: {missing}"

    def test_http_win_rate_in_range(self, server):
        port = server
        req = urllib.request.urlopen(
            f"http://localhost:{port}/api/v1/performance/summary", timeout=5
        )
        body = json.loads(req.read().decode())
        assert 0 <= body["win_rate"] <= 100

    def test_http_active_agents_positive(self, server):
        port = server
        req = urllib.request.urlopen(
            f"http://localhost:{port}/api/v1/performance/summary", timeout=5
        )
        body = json.loads(req.read().decode())
        assert body["active_agents"] >= 1
