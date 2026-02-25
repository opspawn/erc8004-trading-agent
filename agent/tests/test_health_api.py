"""
Tests for health_api.py — Agent Health & Metrics HTTP API.

All tests use mock HTTP requests or direct method calls.
No actual HTTP server is started during tests.
"""

from __future__ import annotations

import io
import json
import sys
import os
import time
from unittest.mock import MagicMock, patch, call
from http.server import BaseHTTPRequestHandler

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from health_api import HealthAPIServer, _HealthHandler


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _make_mock_coordinator(num_agents: int = 3):
    """Create a mock MeshCoordinator with realistic agent data."""
    coord = MagicMock()
    agents = []
    profiles = [("conservative_agent", "A"), ("balanced_agent", "BBB"),
                ("aggressive_agent", "BB")][:num_agents]
    for agent_id, tier in profiles:
        ag = MagicMock()
        ag.agent_id = agent_id
        ag.get_stats.return_value = {
            "profile": agent_id.split("_")[0],
            "credora_min_grade": tier,
            "reputation_score": 6.5,
            "last_trade_ts": time.time() - 300,
            "trades_today": 5,
            "wins": 3,
            "losses": 2,
        }
        agents.append(ag)
    coord.agents = agents
    coord.get_agent = lambda aid: next((a for a in agents if a.agent_id == aid), None)
    return coord


def _make_mock_backtester():
    bt = MagicMock()
    stats = MagicMock()
    stats.win_rate = 0.62
    stats.sharpe_ratio = 1.42
    stats.max_drawdown_pct = 0.085
    stats.final_capital = 11_200.0
    bt.compute_stats.return_value = stats
    return bt


# ─── HealthAPIServer Initialisation Tests ─────────────────────────────────────

class TestHealthAPIServerInit:
    def test_creates_with_defaults(self):
        api = HealthAPIServer()
        assert api is not None

    def test_coordinator_stored(self):
        coord = _make_mock_coordinator()
        api = HealthAPIServer(coordinator=coord)
        assert api._coordinator is coord

    def test_backtester_stored(self):
        bt = _make_mock_backtester()
        api = HealthAPIServer(backtester=bt)
        assert api._backtester is bt

    def test_start_time_defaults_to_now(self):
        before = time.time()
        api = HealthAPIServer()
        after = time.time()
        assert before <= api._start_time <= after

    def test_start_time_can_be_set(self):
        t0 = time.time() - 3600
        api = HealthAPIServer(start_time=t0)
        assert api._start_time == t0

    def test_trade_history_starts_empty(self):
        api = HealthAPIServer()
        assert api.trade_history == []

    def test_server_not_running_initially(self):
        api = HealthAPIServer()
        assert not api.running

    def test_port_zero_before_start(self):
        api = HealthAPIServer()
        assert api.port == 0


# ─── /health Endpoint Tests ────────────────────────────────────────────────────

class TestHealthEndpoint:
    def _api(self, **kwargs) -> HealthAPIServer:
        return HealthAPIServer(**kwargs)

    def test_health_returns_dict(self):
        api = self._api()
        result = api.health()
        assert isinstance(result, dict)

    def test_health_status_ok(self):
        api = self._api()
        result = api.health()
        assert result["status"] == "ok"

    def test_health_has_uptime(self):
        api = self._api(start_time=time.time() - 120)
        result = api.health()
        assert "uptime_seconds" in result
        assert result["uptime_seconds"] >= 120

    def test_health_has_timestamp(self):
        api = self._api()
        result = api.health()
        assert "timestamp" in result
        assert isinstance(result["timestamp"], float)

    def test_health_has_agents(self):
        api = self._api()
        result = api.health()
        assert "agents" in result
        assert isinstance(result["agents"], list)

    def test_health_default_three_agents(self):
        api = self._api()  # no coordinator → default agents
        result = api.health()
        assert len(result["agents"]) == 3

    def test_health_agents_have_required_fields(self):
        api = self._api()
        result = api.health()
        required = ["id", "credit_tier", "reputation", "last_trade", "trades_today"]
        for agent in result["agents"]:
            for f in required:
                assert f in agent, f"Missing field {f} in agent {agent}"

    def test_health_with_coordinator_uses_real_agents(self):
        coord = _make_mock_coordinator(3)
        api = self._api(coordinator=coord)
        result = api.health()
        ids = [a["id"] for a in result["agents"]]
        assert "conservative_agent" in ids

    def test_health_with_coordinator_agent_credit_tier(self):
        coord = _make_mock_coordinator()
        api = self._api(coordinator=coord)
        result = api.health()
        for ag in result["agents"]:
            assert ag["credit_tier"] in {"AAA", "AA", "A", "BBB", "BB", "B", "CCC", "NR"}

    def test_health_with_coordinator_reputation_float(self):
        coord = _make_mock_coordinator()
        api = self._api(coordinator=coord)
        result = api.health()
        for ag in result["agents"]:
            assert isinstance(ag["reputation"], float)

    def test_health_version_field(self):
        api = self._api()
        result = api.health()
        assert "version" in result

    def test_health_uptime_increases(self):
        api = self._api(start_time=time.time() - 10)
        r1 = api.health()
        time.sleep(0.01)
        r2 = api.health()
        assert r2["uptime_seconds"] >= r1["uptime_seconds"]

    def test_health_wins_losses_in_agents(self):
        coord = _make_mock_coordinator()
        api = self._api(coordinator=coord)
        result = api.health()
        for ag in result["agents"]:
            assert "wins" in ag
            assert "losses" in ag

    def test_health_trades_today_non_negative(self):
        coord = _make_mock_coordinator()
        api = self._api(coordinator=coord)
        result = api.health()
        for ag in result["agents"]:
            assert ag["trades_today"] >= 0


# ─── /metrics Endpoint Tests ───────────────────────────────────────────────────

class TestMetricsEndpoint:
    def _api(self, **kwargs) -> HealthAPIServer:
        return HealthAPIServer(**kwargs)

    def test_metrics_returns_dict(self):
        api = self._api()
        result = api.metrics()
        assert isinstance(result, dict)

    def test_metrics_has_total_trades(self):
        api = self._api()
        result = api.metrics()
        assert "total_trades" in result

    def test_metrics_has_win_rate(self):
        api = self._api()
        result = api.metrics()
        assert "win_rate" in result

    def test_metrics_has_sharpe_ratio(self):
        api = self._api()
        result = api.metrics()
        assert "sharpe_ratio" in result

    def test_metrics_has_max_drawdown(self):
        api = self._api()
        result = api.metrics()
        assert "max_drawdown" in result

    def test_metrics_has_portfolio_value(self):
        api = self._api()
        result = api.metrics()
        assert "portfolio_value" in result

    def test_metrics_total_trades_zero_when_empty(self):
        api = self._api()
        result = api.metrics()
        assert result["total_trades"] == 0

    def test_metrics_win_rate_zero_when_empty(self):
        api = self._api()
        result = api.metrics()
        assert result["win_rate"] == 0.0

    def test_metrics_with_trade_history(self):
        api = self._api()
        api.trade_history = [
            {"pnl": 50.0}, {"pnl": -20.0}, {"pnl": 30.0},
        ]
        result = api.metrics()
        assert result["total_trades"] == 3

    def test_metrics_win_rate_calculated(self):
        api = self._api()
        api.trade_history = [
            {"pnl": 50.0}, {"pnl": -20.0}, {"pnl": 30.0}, {"pnl": -5.0},
        ]
        result = api.metrics()
        assert result["win_rate"] == pytest.approx(0.5, abs=0.01)

    def test_metrics_total_pnl(self):
        api = self._api()
        api.trade_history = [{"pnl": 100.0}, {"pnl": -50.0}]
        result = api.metrics()
        assert result["total_pnl"] == pytest.approx(50.0, abs=0.01)

    def test_metrics_portfolio_value_updated(self):
        api = self._api()
        api.trade_history = [{"pnl": 500.0}]
        result = api.metrics()
        assert result["portfolio_value"] == pytest.approx(10_500.0, abs=1.0)

    def test_metrics_has_profit_factor(self):
        api = self._api()
        result = api.metrics()
        assert "profit_factor" in result

    def test_metrics_has_total_pnl(self):
        api = self._api()
        result = api.metrics()
        assert "total_pnl" in result

    def test_metrics_all_numeric(self):
        api = self._api()
        result = api.metrics()
        for key in ["total_trades", "win_rate", "sharpe_ratio",
                    "max_drawdown", "portfolio_value"]:
            assert isinstance(result[key], (int, float))


# ─── /agents Endpoint Tests ────────────────────────────────────────────────────

class TestAgentsEndpoint:
    def _api(self, **kwargs) -> HealthAPIServer:
        return HealthAPIServer(**kwargs)

    def test_agents_returns_dict(self):
        api = self._api()
        result = api.agents_list()
        assert isinstance(result, dict)

    def test_agents_has_agents_key(self):
        api = self._api()
        result = api.agents_list()
        assert "agents" in result

    def test_agents_has_count_key(self):
        api = self._api()
        result = api.agents_list()
        assert "count" in result

    def test_agents_empty_without_coordinator(self):
        api = self._api()
        result = api.agents_list()
        assert result["agents"] == []
        assert result["count"] == 0

    def test_agents_returns_all_agents(self):
        coord = _make_mock_coordinator(3)
        api = self._api(coordinator=coord)
        result = api.agents_list()
        assert result["count"] == 3

    def test_agents_have_erc8004_identity(self):
        coord = _make_mock_coordinator()
        api = self._api(coordinator=coord)
        result = api.agents_list()
        for ag in result["agents"]:
            assert "erc8004" in ag

    def test_agents_erc8004_has_agent_id(self):
        coord = _make_mock_coordinator()
        api = self._api(coordinator=coord)
        result = api.agents_list()
        for ag in result["agents"]:
            assert "agent_id" in ag["erc8004"]

    def test_agents_erc8004_has_registry_address(self):
        coord = _make_mock_coordinator()
        api = self._api(coordinator=coord)
        result = api.agents_list()
        for ag in result["agents"]:
            assert "reputation_registry" in ag["erc8004"]
            assert ag["erc8004"]["reputation_registry"].startswith("0x")

    def test_agents_have_status(self):
        coord = _make_mock_coordinator()
        api = self._api(coordinator=coord)
        result = api.agents_list()
        for ag in result["agents"]:
            assert "status" in ag

    def test_agents_status_has_active(self):
        coord = _make_mock_coordinator()
        api = self._api(coordinator=coord)
        result = api.agents_list()
        for ag in result["agents"]:
            assert "active" in ag["status"]
            assert ag["status"]["active"] is True

    def test_agents_erc8004_chain_base_sepolia(self):
        coord = _make_mock_coordinator()
        api = self._api(coordinator=coord)
        result = api.agents_list()
        for ag in result["agents"]:
            assert ag["erc8004"]["chain"] == "Base Sepolia"

    def test_agents_erc8004_score_numeric(self):
        coord = _make_mock_coordinator()
        api = self._api(coordinator=coord)
        result = api.agents_list()
        for ag in result["agents"]:
            assert isinstance(ag["erc8004"]["score"], (int, float))

    def test_agents_status_reputation_float(self):
        coord = _make_mock_coordinator()
        api = self._api(coordinator=coord)
        result = api.agents_list()
        for ag in result["agents"]:
            assert isinstance(ag["status"]["reputation"], float)

    def test_agents_count_matches_list_length(self):
        coord = _make_mock_coordinator(2)
        api = self._api(coordinator=coord)
        result = api.agents_list()
        assert result["count"] == len(result["agents"])


# ─── Handler Routing Tests ─────────────────────────────────────────────────────

class TestHandlerRouting:
    """Test the HTTP handler routing logic without starting a server."""

    def _make_handler(self, path: str, api: HealthAPIServer) -> _HealthHandler:
        """Create a handler that processes a fake GET request."""
        handler_cls = type("_BoundHandler", (_HealthHandler,), {"_api": api})
        rfile = io.BytesIO(b"")
        # Minimal socket mock
        sock = MagicMock()
        sock.makefile.return_value = rfile
        handler = handler_cls.__new__(handler_cls)
        handler._api = api
        handler.path = path
        handler.headers = {}
        handler.wfile = io.BytesIO()
        handler.requestline = f"GET {path} HTTP/1.1"
        handler.server = MagicMock()
        handler.command = "GET"
        return handler

    def test_send_json_writes_body(self):
        api = HealthAPIServer()
        handler = self._make_handler("/health", api)
        handler.send_response = MagicMock()
        handler.send_header = MagicMock()
        handler.end_headers = MagicMock()
        handler.wfile = io.BytesIO()
        handler._send_json(200, {"status": "ok"})
        handler.wfile.seek(0)
        content = handler.wfile.read()
        assert b"status" in content

    def test_send_error_json_includes_message(self):
        api = HealthAPIServer()
        handler = self._make_handler("/bad", api)
        handler.send_response = MagicMock()
        handler.send_header = MagicMock()
        handler.end_headers = MagicMock()
        handler.wfile = io.BytesIO()
        handler._send_error_json(404, "Not found")
        handler.wfile.seek(0)
        content = handler.wfile.read()
        assert b"error" in content

    def test_log_message_suppressed(self):
        api = HealthAPIServer()
        handler = self._make_handler("/health", api)
        # Should not raise
        handler.log_message("test %s", "arg")

    def test_unknown_path_returns_error(self):
        api = HealthAPIServer()
        handler = self._make_handler("/unknown", api)
        handler.send_response = MagicMock()
        handler.send_header = MagicMock()
        handler.end_headers = MagicMock()
        handler.wfile = io.BytesIO()
        handler.do_GET()
        handler.send_response.assert_called_with(404)

    def test_health_path_returns_200(self):
        api = HealthAPIServer()
        handler = self._make_handler("/health", api)
        handler.send_response = MagicMock()
        handler.send_header = MagicMock()
        handler.end_headers = MagicMock()
        handler.wfile = io.BytesIO()
        handler.do_GET()
        handler.send_response.assert_called_with(200)

    def test_metrics_path_returns_200(self):
        api = HealthAPIServer()
        handler = self._make_handler("/metrics", api)
        handler.send_response = MagicMock()
        handler.send_header = MagicMock()
        handler.end_headers = MagicMock()
        handler.wfile = io.BytesIO()
        handler.do_GET()
        handler.send_response.assert_called_with(200)

    def test_agents_path_returns_200(self):
        api = HealthAPIServer()
        handler = self._make_handler("/agents", api)
        handler.send_response = MagicMock()
        handler.send_header = MagicMock()
        handler.end_headers = MagicMock()
        handler.wfile = io.BytesIO()
        handler.do_GET()
        handler.send_response.assert_called_with(200)

    def test_head_health_returns_200(self):
        api = HealthAPIServer()
        handler = self._make_handler("/health", api)
        handler.send_response = MagicMock()
        handler.send_header = MagicMock()
        handler.end_headers = MagicMock()
        handler.wfile = io.BytesIO()
        handler.do_HEAD()
        handler.send_response.assert_called_with(200)

    def test_head_unknown_returns_404(self):
        api = HealthAPIServer()
        handler = self._make_handler("/unknown", api)
        handler.send_response = MagicMock()
        handler.send_header = MagicMock()
        handler.end_headers = MagicMock()
        handler.wfile = io.BytesIO()
        handler.do_HEAD()
        handler.send_response.assert_called_with(404)

    def test_trailing_slash_stripped(self):
        api = HealthAPIServer()
        handler = self._make_handler("/health/", api)
        handler.send_response = MagicMock()
        handler.send_header = MagicMock()
        handler.end_headers = MagicMock()
        handler.wfile = io.BytesIO()
        handler.do_GET()
        handler.send_response.assert_called_with(200)


# ─── Server Lifecycle Tests ────────────────────────────────────────────────────

class TestServerLifecycle:
    def test_get_base_url_format(self):
        api = HealthAPIServer()
        api._port = 8080
        url = api.get_base_url()
        assert url.startswith("http://")
        assert "8080" in url

    def test_running_false_before_start(self):
        api = HealthAPIServer()
        assert api.running is False

    def test_start_and_stop(self):
        api = HealthAPIServer()
        api.start(port=0)  # port 0 = random free port
        assert api.running
        assert api.port > 0
        api.stop()
        assert not api.running

    def test_stop_idempotent(self):
        api = HealthAPIServer()
        api.stop()  # Should not raise
        api.stop()  # Should not raise

    def test_start_sets_port(self):
        api = HealthAPIServer()
        api.start(port=0)
        port = api.port
        api.stop()
        assert port > 0
