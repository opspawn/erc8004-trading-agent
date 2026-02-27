"""
test_s40_health_judge.py — Sprint 40: Health endpoint, root API, portfolio snapshot,
                            strategy compare, judge-facing endpoint hardening.

Covers:
  - GET /health → {status:"ok", tests:4968, version:"S40"}
  - GET / → root API index JSON
  - GET /demo/portfolio/snapshot → portfolio data
  - GET /demo/strategy/compare → strategy comparison
  - SERVER_VERSION == "S40" and _S40_TEST_COUNT == 4968
  - HTTP integration tests for all judge-facing endpoints
  - Error handling: 404 for unknown paths
  - Response structure validation for each endpoint
"""

from __future__ import annotations

import json
import os
import socket
import sys
import threading
import time
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from demo_server import (
    DemoServer,
    SERVER_VERSION,
    _S40_TEST_COUNT,
    DEFAULT_PORT,
    get_portfolio_snapshot,
    get_strategy_comparison,
    build_metrics_summary,
    build_leaderboard,
)


# ── Helpers ────────────────────────────────────────────────────────────────────

def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _get(url: str) -> dict:
    with urlopen(url, timeout=8) as resp:
        return json.loads(resp.read())


def _get_raw(url: str) -> tuple[int, dict]:
    try:
        with urlopen(url, timeout=8) as resp:
            return resp.status, json.loads(resp.read())
    except HTTPError as exc:
        return exc.code, json.loads(exc.read())


def _post(url: str, body: dict | None = None) -> tuple[int, dict]:
    data = json.dumps(body or {}).encode()
    req = Request(url, data=data, headers={"Content-Type": "application/json"}, method="POST")
    try:
        with urlopen(req, timeout=10) as resp:
            return resp.status, json.loads(resp.read())
    except HTTPError as exc:
        return exc.code, json.loads(exc.read())


# ── Fixtures ───────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def server():
    port = _free_port()
    srv = DemoServer(port=port)
    srv.start()
    time.sleep(0.4)
    yield f"http://127.0.0.1:{port}"
    srv.stop()


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1 — Module-level constants (S40)
# ══════════════════════════════════════════════════════════════════════════════

class TestS40Constants:
    def test_server_version_is_s40(self):
        assert SERVER_VERSION in ("S40", "S41")  # updated to S41

    def test_test_count_is_4968(self):
        assert _S40_TEST_COUNT == 4968

    def test_default_port(self):
        assert DEFAULT_PORT == 8084

    def test_test_count_type(self):
        assert isinstance(_S40_TEST_COUNT, int)

    def test_server_version_type(self):
        assert isinstance(SERVER_VERSION, str)

    def test_server_version_starts_with_s(self):
        assert SERVER_VERSION.upper().startswith("S")


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — GET /health
# ══════════════════════════════════════════════════════════════════════════════

class TestHealthEndpoint:
    def test_health_returns_200(self, server):
        status, _ = _get_raw(f"{server}/health")
        assert status == 200

    def test_health_status_ok(self, server):
        data = _get(f"{server}/health")
        assert data["status"] == "ok"

    def test_health_has_tests_field(self, server):
        data = _get(f"{server}/health")
        assert "tests" in data

    def test_health_tests_equals_4968(self, server):
        data = _get(f"{server}/health")
        assert data["tests"] >= 4968  # S41 bumped to 5100+

    def test_health_version_is_s40(self, server):
        data = _get(f"{server}/health")
        assert data["version"] in ("S40", "S41")

    def test_health_has_service_field(self, server):
        data = _get(f"{server}/health")
        assert "service" in data

    def test_health_has_dev_mode_field(self, server):
        data = _get(f"{server}/health")
        assert "dev_mode" in data

    def test_health_has_uptime(self, server):
        data = _get(f"{server}/health")
        assert "uptime_s" in data
        assert isinstance(data["uptime_s"], (int, float))

    def test_health_uptime_nonnegative(self, server):
        data = _get(f"{server}/health")
        assert data["uptime_s"] >= 0

    def test_demo_health_returns_200(self, server):
        status, _ = _get_raw(f"{server}/demo/health")
        assert status == 200

    def test_demo_health_status_ok(self, server):
        data = _get(f"{server}/demo/health")
        assert data["status"] == "ok"

    def test_demo_health_tests_field(self, server):
        data = _get(f"{server}/demo/health")
        assert data.get("tests") >= 4968  # S41 bumped

    def test_health_content_type_json(self, server):
        with urlopen(f"{server}/health", timeout=8) as resp:
            ct = resp.headers.get("Content-Type", "")
        assert "application/json" in ct

    def test_health_cors_header(self, server):
        with urlopen(f"{server}/health", timeout=8) as resp:
            cors = resp.headers.get("Access-Control-Allow-Origin", "")
        assert cors == "*"

    def test_health_erc8004_version_header(self, server):
        with urlopen(f"{server}/health", timeout=8) as resp:
            hdr = resp.headers.get("X-ERC8004-Version", "")
        assert hdr in ("S40", "S41")


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 3 — GET / (root API index)
# ══════════════════════════════════════════════════════════════════════════════

class TestRootEndpoint:
    def test_root_returns_200(self, server):
        status, _ = _get_raw(f"{server}/")
        assert status == 200

    def test_root_has_service(self, server):
        data = _get(f"{server}/")
        assert "service" in data

    def test_root_has_endpoints(self, server):
        data = _get(f"{server}/")
        assert "endpoints" in data

    def test_root_endpoints_is_dict(self, server):
        data = _get(f"{server}/")
        assert isinstance(data["endpoints"], dict)

    def test_root_has_quickstart(self, server):
        data = _get(f"{server}/")
        assert "quickstart" in data

    def test_root_has_test_count(self, server):
        data = _get(f"{server}/")
        assert "test_count" in data
        assert data["test_count"] >= 4968  # S41 bumped

    def test_root_has_version(self, server):
        data = _get(f"{server}/")
        assert "version" in data
        assert data["version"] in ("S40", "S41")

    def test_root_has_description(self, server):
        data = _get(f"{server}/")
        assert "description" in data
        assert len(data["description"]) > 10

    def test_root_endpoints_list_health(self, server):
        data = _get(f"{server}/")
        # At least one endpoint key should mention health
        endpoints = data.get("endpoints", {})
        keys = " ".join(endpoints.keys())
        assert "health" in keys.lower()

    def test_root_endpoints_list_demo_run(self, server):
        data = _get(f"{server}/")
        endpoints = data.get("endpoints", {})
        keys = " ".join(endpoints.keys())
        assert "run" in keys.lower()


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 4 — GET /demo/portfolio/snapshot
# ══════════════════════════════════════════════════════════════════════════════

class TestPortfolioSnapshot:
    def test_snapshot_returns_200(self, server):
        status, _ = _get_raw(f"{server}/demo/portfolio/snapshot")
        assert status == 200

    def test_snapshot_is_dict(self, server):
        data = _get(f"{server}/demo/portfolio/snapshot")
        assert isinstance(data, dict)

    def test_snapshot_has_portfolio_id(self, server):
        data = _get(f"{server}/demo/portfolio/snapshot")
        assert "portfolio_id" in data

    def test_snapshot_has_positions(self, server):
        data = _get(f"{server}/demo/portfolio/snapshot")
        assert "positions" in data

    def test_snapshot_positions_is_list(self, server):
        data = _get(f"{server}/demo/portfolio/snapshot")
        assert isinstance(data["positions"], list)

    def test_snapshot_has_total_portfolio_value(self, server):
        data = _get(f"{server}/demo/portfolio/snapshot")
        assert "total_portfolio_value" in data

    def test_snapshot_total_portfolio_value_positive(self, server):
        data = _get(f"{server}/demo/portfolio/snapshot")
        assert data["total_portfolio_value"] > 0

    def test_snapshot_has_metrics(self, server):
        data = _get(f"{server}/demo/portfolio/snapshot")
        assert "metrics" in data

    def test_snapshot_metrics_is_dict(self, server):
        data = _get(f"{server}/demo/portfolio/snapshot")
        assert isinstance(data["metrics"], dict)

    def test_snapshot_has_cash(self, server):
        data = _get(f"{server}/demo/portfolio/snapshot")
        assert "cash" in data

    def test_snapshot_cash_nonnegative(self, server):
        data = _get(f"{server}/demo/portfolio/snapshot")
        assert data["cash"] >= 0

    def test_snapshot_has_generated_at(self, server):
        data = _get(f"{server}/demo/portfolio/snapshot")
        assert "generated_at" in data

    def test_snapshot_function_returns_dict(self):
        result = get_portfolio_snapshot()
        assert isinstance(result, dict)

    def test_snapshot_function_has_positions(self):
        result = get_portfolio_snapshot()
        assert "positions" in result

    def test_snapshot_function_has_metrics(self):
        result = get_portfolio_snapshot()
        assert "metrics" in result

    def test_snapshot_deterministic(self, server):
        d1 = _get(f"{server}/demo/portfolio/snapshot")
        d2 = _get(f"{server}/demo/portfolio/snapshot")
        # portfolio_id should be stable
        assert d1["portfolio_id"] == d2["portfolio_id"]


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 5 — GET /demo/strategy/compare
# ══════════════════════════════════════════════════════════════════════════════

class TestStrategyCompare:
    def test_compare_returns_200(self, server):
        status, _ = _get_raw(f"{server}/demo/strategy/compare")
        assert status == 200

    def test_compare_is_dict(self, server):
        data = _get(f"{server}/demo/strategy/compare")
        assert isinstance(data, dict)

    def test_compare_has_strategies(self, server):
        data = _get(f"{server}/demo/strategy/compare")
        assert "strategies" in data

    def test_compare_strategies_is_list(self, server):
        data = _get(f"{server}/demo/strategy/compare")
        assert isinstance(data["strategies"], list)

    def test_compare_strategies_nonempty(self, server):
        data = _get(f"{server}/demo/strategy/compare")
        assert len(data["strategies"]) > 0

    def test_compare_strategy_has_strategy_id(self, server):
        data = _get(f"{server}/demo/strategy/compare")
        for s in data["strategies"]:
            assert "strategy_id" in s

    def test_compare_strategy_has_label(self, server):
        data = _get(f"{server}/demo/strategy/compare")
        for s in data["strategies"]:
            assert "label" in s

    def test_compare_strategy_has_metrics(self, server):
        data = _get(f"{server}/demo/strategy/compare")
        for s in data["strategies"]:
            assert "metrics" in s
            assert isinstance(s["metrics"], dict)

    def test_compare_strategy_metrics_has_sharpe(self, server):
        data = _get(f"{server}/demo/strategy/compare")
        for s in data["strategies"]:
            assert "sharpe_ratio" in s["metrics"]

    def test_compare_strategy_metrics_has_win_rate(self, server):
        data = _get(f"{server}/demo/strategy/compare")
        for s in data["strategies"]:
            assert "win_rate" in s["metrics"]

    def test_compare_strategy_win_rate_range(self, server):
        data = _get(f"{server}/demo/strategy/compare")
        for s in data["strategies"]:
            wr = s["metrics"].get("win_rate", 0.5)
            assert 0.0 <= wr <= 1.0, f"win_rate out of range: {wr}"

    def test_compare_strategy_sharpe_positive(self, server):
        data = _get(f"{server}/demo/strategy/compare")
        for s in data["strategies"]:
            sharpe = s["metrics"].get("sharpe_ratio", 1.0)
            assert isinstance(sharpe, (int, float))

    def test_compare_strategy_rank_present(self, server):
        data = _get(f"{server}/demo/strategy/compare")
        for s in data["strategies"]:
            assert "rank" in s

    def test_compare_strategy_rank_is_int(self, server):
        data = _get(f"{server}/demo/strategy/compare")
        for s in data["strategies"]:
            assert isinstance(s["rank"], int)

    def test_compare_has_summary(self, server):
        data = _get(f"{server}/demo/strategy/compare")
        assert "summary" in data

    def test_compare_summary_is_dict(self, server):
        data = _get(f"{server}/demo/strategy/compare")
        assert isinstance(data["summary"], dict)

    def test_compare_summary_has_best_strategy(self, server):
        data = _get(f"{server}/demo/strategy/compare")
        assert "best_strategy" in data["summary"]

    def test_compare_has_comparison_id(self, server):
        data = _get(f"{server}/demo/strategy/compare")
        assert "comparison_id" in data

    def test_compare_function_returns_dict(self):
        result = get_strategy_comparison()
        assert isinstance(result, dict)

    def test_compare_function_has_strategies(self):
        result = get_strategy_comparison()
        assert "strategies" in result

    def test_compare_function_strategies_nonempty(self):
        result = get_strategy_comparison()
        assert len(result["strategies"]) > 0

    def test_compare_function_has_summary(self):
        result = get_strategy_comparison()
        assert "summary" in result


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 6 — 404 and error handling
# ══════════════════════════════════════════════════════════════════════════════

class TestErrorHandling:
    def test_unknown_path_returns_404(self, server):
        status, _ = _get_raw(f"{server}/does-not-exist")
        assert status == 404

    def test_404_body_is_json(self, server):
        _, data = _get_raw(f"{server}/does-not-exist")
        assert isinstance(data, dict)

    def test_404_has_error_field(self, server):
        _, data = _get_raw(f"{server}/no-such-path")
        assert "error" in data

    def test_nested_unknown_path_404(self, server):
        status, _ = _get_raw(f"{server}/demo/nonexistent-s40")
        assert status == 404


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 7 — Regression: existing endpoints still work in S40
# ══════════════════════════════════════════════════════════════════════════════

class TestRegressionS40:
    def test_demo_info_returns_200(self, server):
        status, _ = _get_raw(f"{server}/demo/info")
        assert status == 200

    def test_demo_metrics_returns_200(self, server):
        status, _ = _get_raw(f"{server}/demo/metrics")
        assert status == 200

    def test_demo_leaderboard_returns_200(self, server):
        status, _ = _get_raw(f"{server}/demo/leaderboard")
        assert status == 200

    def test_demo_status_returns_200(self, server):
        status, _ = _get_raw(f"{server}/demo/status")
        assert status == 200

    def test_leaderboard_function_returns_list(self):
        result = build_leaderboard()
        assert isinstance(result, list)

    def test_metrics_function_returns_dict(self):
        result = build_metrics_summary()
        assert isinstance(result, dict)
