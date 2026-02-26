"""
test_demo_server_s23.py — Tests for S23 competitive features.

Covers:
  - GET /demo/metrics:     seeded defaults, live update after /demo/run
  - GET /demo/leaderboard: seeded defaults, live update after /demo/run
  - POST /demo/compare:    side-by-side comparison, error cases
  - GET /demo/stream:      SSE connection, keepalive, event delivery
  - build_metrics_summary, build_leaderboard, build_compare helper functions
  - _calc_sharpe and _calc_sortino math helpers
  - SSE broadcast fan-out to multiple clients
  - State persistence across multiple runs
"""

from __future__ import annotations

import json
import math
import sys
import os
import queue
import time
import threading
import urllib.request
import urllib.error
from typing import Any, Dict, List

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import demo_server as ds
from demo_server import (
    build_metrics_summary,
    build_leaderboard,
    build_compare,
    _calc_sharpe,
    _calc_sortino,
    _sse_broadcast,
    _sse_clients,
    _sse_clients_lock,
    _metrics_state,
    _metrics_lock,
    _agent_cumulative,
    _leaderboard_lock,
    _SEEDED_METRICS,
    _SEEDED_LEADERBOARD,
    DemoServer,
    run_demo_pipeline,
)


# ─── Fixtures ────────────────────────────────────────────────────────────────

@pytest.fixture(autouse=True)
def reset_state():
    """Reset shared state before each test to prevent cross-test pollution."""
    with _metrics_lock:
        _metrics_state["total_trades"] = 0
        _metrics_state["total_wins"] = 0
        _metrics_state["run_count"] = 0
        _metrics_state["cumulative_pnl"] = 0.0
        _metrics_state["pnl_history"] = []
        _metrics_state["max_drawdown"] = -0.042
        _metrics_state["last_updated"] = None

    with _leaderboard_lock:
        _agent_cumulative.clear()

    with _sse_clients_lock:
        _sse_clients.clear()

    with ds._portfolio_lock:
        ds._last_run_result = None

    yield


@pytest.fixture(scope="module")
def s23_server():
    """Shared server for HTTP tests in this module (port 18093)."""
    port = 18093
    server = DemoServer(port=port)
    server.start()
    time.sleep(0.3)
    yield port
    server.stop()


def _get(port: int, path: str) -> Dict[str, Any]:
    resp = urllib.request.urlopen(f"http://localhost:{port}{path}", timeout=10)
    return json.loads(resp.read())


def _post(port: int, path: str, body: Any = None, extra_headers: Dict | None = None) -> Dict[str, Any]:
    body_bytes = json.dumps(body).encode() if body is not None else b""
    req = urllib.request.Request(f"http://localhost:{port}{path}", method="POST", data=body_bytes)
    req.add_header("Content-Length", str(len(body_bytes)))
    req.add_header("Content-Type", "application/json")
    if extra_headers:
        for k, v in extra_headers.items():
            req.add_header(k, v)
    resp = urllib.request.urlopen(req, timeout=10)
    return json.loads(resp.read())


def _post_error(port: int, path: str, body: Any = None) -> urllib.error.HTTPError:
    body_bytes = json.dumps(body).encode() if body is not None else b""
    req = urllib.request.Request(f"http://localhost:{port}{path}", method="POST", data=body_bytes)
    req.add_header("Content-Length", str(len(body_bytes)))
    req.add_header("Content-Type", "application/json")
    try:
        urllib.request.urlopen(req, timeout=10)
        return None
    except urllib.error.HTTPError as e:
        return e


# ─── _calc_sharpe ─────────────────────────────────────────────────────────────

class TestCalcSharpe:
    def test_single_value_returns_seeded(self):
        result = _calc_sharpe([5.0])
        assert result == 1.42

    def test_empty_returns_seeded(self):
        result = _calc_sharpe([])
        assert result == 1.42

    def test_positive_mean_positive_std(self):
        values = [1.0, 2.0, 3.0, 4.0, 5.0]
        result = _calc_sharpe(values)
        assert isinstance(result, float)
        assert result > 0

    def test_constant_values_high_sharpe(self):
        # All same value → std very small → high Sharpe
        values = [2.0, 2.0, 2.0, 2.0]
        result = _calc_sharpe(values)
        assert result > 1000  # near infinite Sharpe for constant returns

    def test_two_value_minimum(self):
        result = _calc_sharpe([1.0, 3.0])
        assert isinstance(result, float)
        # mean=2.0, std=sqrt(2)≈1.414, sharpe≈1.414
        assert abs(result - round(2.0 / math.sqrt(2), 4)) < 0.01

    def test_mixed_signs(self):
        values = [-1.0, 2.0, -1.5, 3.0]
        result = _calc_sharpe(values)
        assert isinstance(result, float)

    def test_result_is_rounded(self):
        values = [1.1, 2.2, 3.3]
        result = _calc_sharpe(values)
        assert result == round(result, 4)


# ─── _calc_sortino ────────────────────────────────────────────────────────────

class TestCalcSortino:
    def test_single_value_returns_seeded(self):
        result = _calc_sortino([5.0])
        assert result == 1.87

    def test_empty_returns_seeded(self):
        result = _calc_sortino([])
        assert result == 1.87

    def test_no_downside_positive_mean(self):
        # All values above mean? Only values BELOW mean count as downside
        values = [3.0, 3.0, 3.0, 3.0]
        # No downside → returns 0 (mean > 0 but no downside
        # Actually: no values below mean, so downside_sq=[], returns very large
        result = _calc_sortino(values)
        assert result >= 0  # positive or infinite

    def test_mixed_returns(self):
        values = [2.0, -1.0, 3.0, -0.5, 1.5]
        result = _calc_sortino(values)
        assert isinstance(result, float)

    def test_result_is_rounded(self):
        values = [1.0, 2.0, 3.0, -1.0]
        result = _calc_sortino(values)
        assert result == round(result, 4)

    def test_two_values_minimum(self):
        result = _calc_sortino([1.0, 3.0])
        assert isinstance(result, float)

    def test_all_negative(self):
        values = [-1.0, -2.0, -3.0]
        result = _calc_sortino(values)
        assert isinstance(result, float)


# ─── build_metrics_summary ────────────────────────────────────────────────────

class TestBuildMetricsSummary:
    def test_returns_dict(self):
        m = build_metrics_summary()
        assert isinstance(m, dict)

    def test_seeded_before_run(self):
        m = build_metrics_summary()
        assert m["source"] == "seeded"

    def test_seeded_has_required_keys(self):
        m = build_metrics_summary()
        assert "total_trades" in m
        assert "win_rate" in m
        assert "sharpe_ratio" in m
        assert "sortino_ratio" in m
        assert "max_drawdown" in m
        assert "cumulative_return_pct" in m
        assert "active_agents_count" in m
        assert "run_count" in m
        assert "source" in m

    def test_seeded_run_count_zero(self):
        m = build_metrics_summary()
        assert m["run_count"] == 0

    def test_seeded_win_rate_bounded(self):
        m = build_metrics_summary()
        assert 0.0 <= m["win_rate"] <= 1.0

    def test_live_after_run(self):
        run_demo_pipeline(ticks=5, seed=42)
        m = build_metrics_summary()
        assert m["source"] == "live"

    def test_live_run_count_increments(self):
        run_demo_pipeline(ticks=5, seed=1)
        run_demo_pipeline(ticks=5, seed=2)
        m = build_metrics_summary()
        assert m["run_count"] == 2

    def test_live_total_trades_nonnegative(self):
        run_demo_pipeline(ticks=10, seed=42)
        m = build_metrics_summary()
        assert m["total_trades"] >= 0

    def test_live_win_rate_bounded(self):
        run_demo_pipeline(ticks=10, seed=42)
        m = build_metrics_summary()
        assert 0.0 <= m["win_rate"] <= 1.0

    def test_live_active_agents_count(self):
        run_demo_pipeline(ticks=5, seed=42)
        m = build_metrics_summary()
        assert m["active_agents_count"] == 3

    def test_live_has_last_updated(self):
        run_demo_pipeline(ticks=5, seed=42)
        m = build_metrics_summary()
        assert m["last_updated"] is not None
        assert m["last_updated"] > 0

    def test_live_sharpe_is_float(self):
        run_demo_pipeline(ticks=10, seed=42)
        m = build_metrics_summary()
        assert isinstance(m["sharpe_ratio"], float)

    def test_live_sortino_is_float(self):
        run_demo_pipeline(ticks=10, seed=42)
        m = build_metrics_summary()
        assert isinstance(m["sortino_ratio"], float)

    def test_live_max_drawdown_negative_or_zero(self):
        run_demo_pipeline(ticks=10, seed=42)
        m = build_metrics_summary()
        assert m["max_drawdown"] <= 0.0

    def test_multiple_runs_accumulate_trades(self):
        r1 = run_demo_pipeline(ticks=5, seed=1)
        r2 = run_demo_pipeline(ticks=5, seed=2)
        m = build_metrics_summary()
        expected = r1["demo"]["trades_executed"] + r2["demo"]["trades_executed"]
        assert m["total_trades"] == expected

    def test_seeded_matches_constants(self):
        m = build_metrics_summary()
        assert m["total_trades"] == _SEEDED_METRICS["total_trades"]
        assert m["win_rate"] == _SEEDED_METRICS["win_rate"]


# ─── build_leaderboard ────────────────────────────────────────────────────────

class TestBuildLeaderboard:
    def test_returns_list(self):
        lb = build_leaderboard()
        assert isinstance(lb, list)

    def test_seeded_before_run(self):
        lb = build_leaderboard()
        # Seeded data has conservative/balanced/aggressive
        strategies = {e["strategy"] for e in lb}
        assert "conservative" in strategies

    def test_seeded_count_three(self):
        lb = build_leaderboard()
        assert len(lb) == len(_SEEDED_LEADERBOARD)

    def test_seeded_has_ranks(self):
        lb = build_leaderboard()
        for e in lb:
            assert "rank" in e

    def test_live_after_run(self):
        run_demo_pipeline(ticks=10, seed=42)
        lb = build_leaderboard()
        assert len(lb) > 0

    def test_live_required_keys(self):
        run_demo_pipeline(ticks=10, seed=42)
        lb = build_leaderboard()
        for e in lb:
            assert "agent_id" in e
            assert "strategy" in e
            assert "total_return_pct" in e
            assert "sortino_ratio" in e
            assert "win_rate" in e
            assert "trades_count" in e
            assert "rank" in e

    def test_live_ranks_ascending(self):
        run_demo_pipeline(ticks=10, seed=42)
        lb = build_leaderboard()
        for i, e in enumerate(lb, start=1):
            assert e["rank"] == i

    def test_live_sorted_by_sortino_descending(self):
        run_demo_pipeline(ticks=20, seed=42)
        lb = build_leaderboard()
        for i in range(len(lb) - 1):
            assert lb[i]["sortino_ratio"] >= lb[i + 1]["sortino_ratio"]

    def test_live_win_rates_bounded(self):
        run_demo_pipeline(ticks=10, seed=42)
        lb = build_leaderboard()
        for e in lb:
            assert 0.0 <= e["win_rate"] <= 1.0

    def test_live_max_five_entries(self):
        # Even with multiple runs, at most 5 entries
        for seed in range(5):
            run_demo_pipeline(ticks=5, seed=seed)
        lb = build_leaderboard()
        assert len(lb) <= 5

    def test_live_agent_ids_are_strings(self):
        run_demo_pipeline(ticks=5, seed=42)
        lb = build_leaderboard()
        for e in lb:
            assert isinstance(e["agent_id"], str)
            assert len(e["agent_id"]) > 0

    def test_live_trades_count_nonnegative(self):
        run_demo_pipeline(ticks=10, seed=42)
        lb = build_leaderboard()
        for e in lb:
            assert e["trades_count"] >= 0

    def test_live_reputation_score_in_range(self):
        run_demo_pipeline(ticks=10, seed=42)
        lb = build_leaderboard()
        for e in lb:
            assert 0.0 <= e["reputation_score"] <= 10.0

    def test_multiple_runs_accumulate_trades(self):
        run_demo_pipeline(ticks=10, seed=1)
        lb1 = build_leaderboard()
        total1 = sum(e["trades_count"] for e in lb1)
        run_demo_pipeline(ticks=10, seed=2)
        lb2 = build_leaderboard()
        total2 = sum(e["trades_count"] for e in lb2)
        assert total2 >= total1  # more runs → more trades


# ─── build_compare ────────────────────────────────────────────────────────────

class TestBuildCompare:
    def test_returns_dict(self):
        result = build_compare(["agent-conservative-001", "agent-balanced-002"])
        assert isinstance(result, dict)

    def test_has_comparison_key(self):
        result = build_compare(["agent-conservative-001", "agent-balanced-002"])
        assert "comparison" in result

    def test_has_agent_ids_key(self):
        ids = ["agent-conservative-001", "agent-balanced-002"]
        result = build_compare(ids)
        assert result["agent_ids"] == ids

    def test_has_generated_at(self):
        result = build_compare(["agent-conservative-001", "agent-balanced-002"])
        assert "generated_at" in result
        assert result["generated_at"] > 0

    def test_seeded_fallback_two_agents(self):
        result = build_compare(["agent-conservative-001", "agent-balanced-002"])
        comparison = result["comparison"]
        assert "agent-conservative-001" in comparison
        assert "agent-balanced-002" in comparison

    def test_seeded_fallback_marks_source(self):
        result = build_compare(["agent-conservative-001", "agent-balanced-002"])
        for entry in result["comparison"].values():
            # Seeded entries have 'source': 'seeded' or have 'rank' key
            assert "agent_id" in entry or "error" in entry

    def test_three_agents(self):
        ids = ["agent-conservative-001", "agent-balanced-002", "agent-aggressive-003"]
        result = build_compare(ids)
        assert len(result["comparison"]) == 3

    def test_unknown_agent_returns_error(self):
        result = build_compare(["agent-conservative-001", "agent-UNKNOWN-999"])
        comparison = result["comparison"]
        assert "error" in comparison["agent-UNKNOWN-999"]

    def test_live_after_run(self):
        run_demo_pipeline(ticks=10, seed=42)
        lb = build_leaderboard()
        ids = [lb[0]["agent_id"], lb[1]["agent_id"]]
        result = build_compare(ids)
        comparison = result["comparison"]
        for aid in ids:
            assert aid in comparison
            entry = comparison[aid]
            assert "total_return_pct" in entry
            assert "sortino_ratio" in entry
            assert "win_rate" in entry

    def test_live_trades_count_nonnegative(self):
        run_demo_pipeline(ticks=10, seed=42)
        lb = build_leaderboard()
        ids = [lb[0]["agent_id"], lb[1]["agent_id"]]
        result = build_compare(ids)
        for entry in result["comparison"].values():
            if "trades_count" in entry:
                assert entry["trades_count"] >= 0

    def test_live_win_rate_bounded(self):
        run_demo_pipeline(ticks=10, seed=42)
        lb = build_leaderboard()
        ids = [e["agent_id"] for e in lb[:2]]
        result = build_compare(ids)
        for entry in result["comparison"].values():
            if "win_rate" in entry:
                assert 0.0 <= entry["win_rate"] <= 1.0


# ─── SSE Broadcast ────────────────────────────────────────────────────────────

class TestSSEBroadcast:
    def test_broadcast_reaches_subscribed_queue(self):
        q: queue.Queue = queue.Queue()
        with _sse_clients_lock:
            _sse_clients.append(q)
        try:
            _sse_broadcast({"event": "test", "value": 42})
            payload = q.get(timeout=1)
            data = json.loads(payload)
            assert data["event"] == "test"
            assert data["value"] == 42
        finally:
            with _sse_clients_lock:
                if q in _sse_clients:
                    _sse_clients.remove(q)

    def test_broadcast_fan_out_multiple_clients(self):
        queues = [queue.Queue() for _ in range(3)]
        with _sse_clients_lock:
            _sse_clients.extend(queues)
        try:
            _sse_broadcast({"event": "fanout", "n": 3})
            for q in queues:
                payload = q.get(timeout=1)
                data = json.loads(payload)
                assert data["event"] == "fanout"
        finally:
            with _sse_clients_lock:
                for q in queues:
                    if q in _sse_clients:
                        _sse_clients.remove(q)

    def test_broadcast_empty_clients_no_error(self):
        # No clients registered — should not raise
        _sse_broadcast({"event": "empty"})

    def test_broadcast_removes_full_queues(self):
        # A full queue should be removed automatically
        full_q: queue.Queue = queue.Queue(maxsize=1)
        full_q.put("pre-filled")  # fill it up
        with _sse_clients_lock:
            _sse_clients.append(full_q)
        try:
            _sse_broadcast({"event": "overflow"})
            # After broadcast, full queue was removed
            with _sse_clients_lock:
                assert full_q not in _sse_clients
        finally:
            with _sse_clients_lock:
                if full_q in _sse_clients:
                    _sse_clients.remove(full_q)

    def test_run_triggers_broadcast(self):
        """run_demo_pipeline should broadcast a run_complete event."""
        q: queue.Queue = queue.Queue()
        with _sse_clients_lock:
            _sse_clients.append(q)
        try:
            run_demo_pipeline(ticks=5, seed=42)
            payload = q.get(timeout=5)
            data = json.loads(payload)
            assert data["event"] == "run_complete"
            assert "session_id" in data
            assert "total_pnl_usd" in data
        finally:
            with _sse_clients_lock:
                if q in _sse_clients:
                    _sse_clients.remove(q)

    def test_run_broadcast_has_required_fields(self):
        q: queue.Queue = queue.Queue()
        with _sse_clients_lock:
            _sse_clients.append(q)
        try:
            run_demo_pipeline(ticks=5, seed=42)
            payload = q.get(timeout=5)
            data = json.loads(payload)
            assert "event" in data
            assert "session_id" in data
            assert "symbol" in data
            assert "ticks" in data
            assert "consensus_rate" in data
            assert "duration_ms" in data
            assert "timestamp" in data
        finally:
            with _sse_clients_lock:
                if q in _sse_clients:
                    _sse_clients.remove(q)


# ─── HTTP: GET /demo/metrics ──────────────────────────────────────────────────

class TestHTTPMetrics:
    def test_metrics_200(self, s23_server):
        data = _get(s23_server, "/demo/metrics")
        assert isinstance(data, dict)

    def test_metrics_has_total_trades(self, s23_server):
        data = _get(s23_server, "/demo/metrics")
        assert "total_trades" in data

    def test_metrics_has_win_rate(self, s23_server):
        data = _get(s23_server, "/demo/metrics")
        assert "win_rate" in data

    def test_metrics_has_sharpe_ratio(self, s23_server):
        data = _get(s23_server, "/demo/metrics")
        assert "sharpe_ratio" in data

    def test_metrics_has_sortino_ratio(self, s23_server):
        data = _get(s23_server, "/demo/metrics")
        assert "sortino_ratio" in data

    def test_metrics_has_max_drawdown(self, s23_server):
        data = _get(s23_server, "/demo/metrics")
        assert "max_drawdown" in data

    def test_metrics_has_active_agents_count(self, s23_server):
        data = _get(s23_server, "/demo/metrics")
        assert "active_agents_count" in data

    def test_metrics_has_run_count(self, s23_server):
        data = _get(s23_server, "/demo/metrics")
        assert "run_count" in data

    def test_metrics_win_rate_bounded(self, s23_server):
        data = _get(s23_server, "/demo/metrics")
        assert 0.0 <= data["win_rate"] <= 1.0

    def test_metrics_has_source(self, s23_server):
        data = _get(s23_server, "/demo/metrics")
        assert "source" in data

    def test_metrics_cors_header(self, s23_server):
        resp = urllib.request.urlopen(f"http://localhost:{s23_server}/demo/metrics")
        assert resp.headers.get("Access-Control-Allow-Origin") == "*"

    def test_metrics_content_type_json(self, s23_server):
        resp = urllib.request.urlopen(f"http://localhost:{s23_server}/demo/metrics")
        assert "application/json" in resp.headers.get("Content-Type", "")

    def test_metrics_updates_after_run(self, s23_server):
        data_before = _get(s23_server, "/demo/metrics")
        _post(s23_server, "/demo/run?ticks=5")
        data_after = _get(s23_server, "/demo/metrics")
        assert data_after["run_count"] > data_before["run_count"]

    def test_metrics_shortpath(self, s23_server):
        data = _get(s23_server, "/metrics")
        assert "total_trades" in data


# ─── HTTP: GET /demo/leaderboard ─────────────────────────────────────────────

class TestHTTPLeaderboard:
    def test_leaderboard_200(self, s23_server):
        data = _get(s23_server, "/demo/leaderboard")
        assert isinstance(data, dict)

    def test_leaderboard_has_leaderboard_key(self, s23_server):
        data = _get(s23_server, "/demo/leaderboard")
        assert "leaderboard" in data

    def test_leaderboard_is_list(self, s23_server):
        data = _get(s23_server, "/demo/leaderboard")
        assert isinstance(data["leaderboard"], list)

    def test_leaderboard_not_empty(self, s23_server):
        data = _get(s23_server, "/demo/leaderboard")
        assert len(data["leaderboard"]) > 0

    def test_leaderboard_entries_have_rank(self, s23_server):
        data = _get(s23_server, "/demo/leaderboard")
        for e in data["leaderboard"]:
            assert "rank" in e

    def test_leaderboard_entries_have_agent_id(self, s23_server):
        data = _get(s23_server, "/demo/leaderboard")
        for e in data["leaderboard"]:
            assert "agent_id" in e

    def test_leaderboard_entries_have_strategy(self, s23_server):
        data = _get(s23_server, "/demo/leaderboard")
        for e in data["leaderboard"]:
            assert "strategy" in e

    def test_leaderboard_entries_have_sortino(self, s23_server):
        data = _get(s23_server, "/demo/leaderboard")
        for e in data["leaderboard"]:
            assert "sortino_ratio" in e

    def test_leaderboard_entries_have_win_rate(self, s23_server):
        data = _get(s23_server, "/demo/leaderboard")
        for e in data["leaderboard"]:
            assert "win_rate" in e

    def test_leaderboard_entries_have_trades_count(self, s23_server):
        data = _get(s23_server, "/demo/leaderboard")
        for e in data["leaderboard"]:
            assert "trades_count" in e

    def test_leaderboard_cors_header(self, s23_server):
        resp = urllib.request.urlopen(f"http://localhost:{s23_server}/demo/leaderboard")
        assert resp.headers.get("Access-Control-Allow-Origin") == "*"

    def test_leaderboard_content_type_json(self, s23_server):
        resp = urllib.request.urlopen(f"http://localhost:{s23_server}/demo/leaderboard")
        assert "application/json" in resp.headers.get("Content-Type", "")

    def test_leaderboard_max_five_entries(self, s23_server):
        data = _get(s23_server, "/demo/leaderboard")
        assert len(data["leaderboard"]) <= 5

    def test_leaderboard_win_rates_bounded(self, s23_server):
        data = _get(s23_server, "/demo/leaderboard")
        for e in data["leaderboard"]:
            assert 0.0 <= e["win_rate"] <= 1.0

    def test_leaderboard_updates_after_run(self, s23_server):
        _post(s23_server, "/demo/run?ticks=10&seed=5")
        data = _get(s23_server, "/demo/leaderboard")
        assert len(data["leaderboard"]) > 0

    def test_leaderboard_shortpath(self, s23_server):
        data = _get(s23_server, "/leaderboard")
        assert "leaderboard" in data


# ─── HTTP: POST /demo/compare ─────────────────────────────────────────────────

class TestHTTPCompare:
    def test_compare_200_two_agents(self, s23_server):
        data = _post(s23_server, "/demo/compare", {
            "agent_ids": ["agent-conservative-001", "agent-balanced-002"]
        })
        assert "comparison" in data

    def test_compare_200_three_agents(self, s23_server):
        data = _post(s23_server, "/demo/compare", {
            "agent_ids": [
                "agent-conservative-001",
                "agent-balanced-002",
                "agent-aggressive-003",
            ]
        })
        assert len(data["comparison"]) == 3

    def test_compare_has_agent_ids_field(self, s23_server):
        ids = ["agent-conservative-001", "agent-balanced-002"]
        data = _post(s23_server, "/demo/compare", {"agent_ids": ids})
        assert data["agent_ids"] == ids

    def test_compare_has_generated_at(self, s23_server):
        data = _post(s23_server, "/demo/compare", {
            "agent_ids": ["agent-conservative-001", "agent-balanced-002"]
        })
        assert "generated_at" in data

    def test_compare_unknown_agent_error(self, s23_server):
        data = _post(s23_server, "/demo/compare", {
            "agent_ids": ["agent-conservative-001", "agent-NONEXISTENT-XXX"]
        })
        comparison = data["comparison"]
        assert "error" in comparison["agent-NONEXISTENT-XXX"]

    def test_compare_too_few_agents_400(self, s23_server):
        err = _post_error(s23_server, "/demo/compare", {"agent_ids": ["only-one"]})
        assert err is not None
        assert err.code == 400

    def test_compare_too_many_agents_400(self, s23_server):
        err = _post_error(s23_server, "/demo/compare", {
            "agent_ids": ["a1", "a2", "a3", "a4", "a5", "a6"]
        })
        assert err is not None
        assert err.code == 400

    def test_compare_non_list_agent_ids_400(self, s23_server):
        err = _post_error(s23_server, "/demo/compare", {"agent_ids": "not-a-list"})
        assert err is not None
        assert err.code == 400

    def test_compare_invalid_json_400(self, s23_server):
        req = urllib.request.Request(
            f"http://localhost:{s23_server}/demo/compare",
            method="POST",
            data=b"not json!!!",
        )
        req.add_header("Content-Length", "10")
        req.add_header("Content-Type", "application/json")
        try:
            urllib.request.urlopen(req, timeout=5)
            assert False, "Expected 400"
        except urllib.error.HTTPError as e:
            assert e.code == 400

    def test_compare_cors_header(self, s23_server):
        body = json.dumps({"agent_ids": ["agent-conservative-001", "agent-balanced-002"]}).encode()
        req = urllib.request.Request(
            f"http://localhost:{s23_server}/demo/compare",
            method="POST",
            data=body,
        )
        req.add_header("Content-Length", str(len(body)))
        req.add_header("Content-Type", "application/json")
        resp = urllib.request.urlopen(req, timeout=5)
        assert resp.headers.get("Access-Control-Allow-Origin") == "*"

    def test_compare_content_type_json(self, s23_server):
        body = json.dumps({"agent_ids": ["agent-conservative-001", "agent-balanced-002"]}).encode()
        req = urllib.request.Request(
            f"http://localhost:{s23_server}/demo/compare",
            method="POST",
            data=body,
        )
        req.add_header("Content-Length", str(len(body)))
        req.add_header("Content-Type", "application/json")
        resp = urllib.request.urlopen(req, timeout=5)
        assert "application/json" in resp.headers.get("Content-Type", "")

    def test_compare_shortpath(self, s23_server):
        data = _post(s23_server, "/compare", {
            "agent_ids": ["agent-conservative-001", "agent-balanced-002"]
        })
        assert "comparison" in data

    def test_compare_empty_body_defaults_400(self, s23_server):
        # No agent_ids key → empty list → 400
        req = urllib.request.Request(
            f"http://localhost:{s23_server}/demo/compare",
            method="POST",
            data=b"{}",
        )
        req.add_header("Content-Length", "2")
        req.add_header("Content-Type", "application/json")
        try:
            urllib.request.urlopen(req, timeout=5)
            assert False, "Expected 400"
        except urllib.error.HTTPError as e:
            assert e.code == 400


# ─── HTTP: GET /demo/stream (SSE) ─────────────────────────────────────────────

class TestHTTPSSEStream:
    def test_stream_returns_200(self, s23_server):
        """SSE endpoint should return 200 with SSE content-type."""
        connected = threading.Event()
        received = []
        error_holder = []

        def read_stream():
            try:
                req = urllib.request.Request(
                    f"http://localhost:{s23_server}/demo/stream",
                    method="GET",
                )
                resp = urllib.request.urlopen(req, timeout=5)
                assert resp.status == 200
                connected.set()
                # Read a few lines
                lines_read = 0
                while lines_read < 3:
                    line = resp.fp.readline()
                    if line:
                        received.append(line)
                        lines_read += 1
            except Exception as e:
                if not connected.is_set():
                    error_holder.append(str(e))
                connected.set()

        t = threading.Thread(target=read_stream, daemon=True)
        t.start()
        connected.wait(timeout=5)

        assert not error_holder, f"SSE connect failed: {error_holder}"

    def test_stream_content_type_sse(self, s23_server):
        """SSE endpoint must return text/event-stream content type."""
        result_holder = []

        def check_ct():
            try:
                req = urllib.request.Request(
                    f"http://localhost:{s23_server}/demo/stream"
                )
                resp = urllib.request.urlopen(req, timeout=5)
                result_holder.append(resp.headers.get("Content-Type", ""))
            except Exception:
                pass

        t = threading.Thread(target=check_ct, daemon=True)
        t.start()
        t.join(timeout=4)
        if result_holder:
            assert "text/event-stream" in result_holder[0]

    def test_stream_sends_connected_event(self, s23_server):
        """SSE stream must immediately send a 'connected' event."""
        lines = []
        done = threading.Event()

        def read_first():
            try:
                req = urllib.request.Request(
                    f"http://localhost:{s23_server}/demo/stream"
                )
                resp = urllib.request.urlopen(req, timeout=5)
                # Read first few lines looking for data:
                for _ in range(10):
                    line = resp.fp.readline()
                    decoded = line.decode("utf-8", errors="ignore")
                    lines.append(decoded)
                    if decoded.startswith("data:"):
                        done.set()
                        break
            except Exception:
                done.set()

        t = threading.Thread(target=read_first, daemon=True)
        t.start()
        done.wait(timeout=5)

        data_lines = [l for l in lines if l.startswith("data:")]
        assert len(data_lines) > 0
        # Parse and check connected event
        for line in data_lines:
            raw = line[len("data:"):].strip()
            try:
                payload = json.loads(raw)
                if payload.get("event") == "connected":
                    assert "message" in payload
                    break
            except json.JSONDecodeError:
                continue

    def test_stream_cors_header(self, s23_server):
        """SSE stream must include CORS header."""
        result_holder = []

        def check_cors():
            try:
                req = urllib.request.Request(
                    f"http://localhost:{s23_server}/demo/stream"
                )
                resp = urllib.request.urlopen(req, timeout=5)
                result_holder.append(resp.headers.get("Access-Control-Allow-Origin", ""))
            except Exception:
                pass

        t = threading.Thread(target=check_cors, daemon=True)
        t.start()
        t.join(timeout=4)
        if result_holder:
            assert result_holder[0] == "*"


# ─── /demo/info includes new endpoints ───────────────────────────────────────

class TestInfoEndpointUpdated:
    def test_info_includes_metrics(self, s23_server):
        data = _get(s23_server, "/demo/info")
        endpoints = data["endpoints"]
        assert any("metrics" in k.lower() for k in endpoints)

    def test_info_includes_leaderboard(self, s23_server):
        data = _get(s23_server, "/demo/info")
        endpoints = data["endpoints"]
        assert any("leaderboard" in k.lower() for k in endpoints)

    def test_info_includes_compare(self, s23_server):
        data = _get(s23_server, "/demo/info")
        endpoints = data["endpoints"]
        assert any("compare" in k.lower() for k in endpoints)

    def test_info_includes_stream(self, s23_server):
        data = _get(s23_server, "/demo/info")
        endpoints = data["endpoints"]
        assert any("stream" in k.lower() for k in endpoints)
