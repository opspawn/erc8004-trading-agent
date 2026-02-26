"""
test_s27_health_dashboard.py — Sprint 27: Agent health dashboard tests.

27 tests covering:
  - build_detailed_health() structure and field presence
  - Seeded data fallback (no runs yet)
  - _compute_health_score logic
  - _update_agent_health accumulation
  - Sort order (health_score descending)
  - System status classification
  - HTTP endpoint GET /demo/health/detailed
  - _agent_health state management
"""

from __future__ import annotations

import json
import time
import threading
from urllib.request import urlopen

import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from demo_server import (
    build_detailed_health,
    _compute_health_score,
    _update_agent_health,
    _agent_health,
    _agent_health_lock,
    DemoServer,
)


# ─── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture(autouse=True)
def clear_agent_health():
    """Reset _agent_health before each test."""
    with _agent_health_lock:
        _agent_health.clear()
    yield
    with _agent_health_lock:
        _agent_health.clear()


# ─── _compute_health_score ────────────────────────────────────────────────────

class TestComputeHealthScore:
    def test_perfect_inputs(self):
        score = _compute_health_score(1.0, 0.0, 30)
        assert 0.9 <= score <= 1.0

    def test_zero_inputs(self):
        score = _compute_health_score(0.0, 0.0, 0)
        assert 0.0 <= score <= 1.0

    def test_high_drawdown_lowers_score(self):
        good = _compute_health_score(0.6, -1.0, 20)
        bad = _compute_health_score(0.6, -18.0, 20)
        assert good > bad

    def test_high_win_rate_raises_score(self):
        low = _compute_health_score(0.3, -2.0, 20)
        high = _compute_health_score(0.8, -2.0, 20)
        assert high > low

    def test_score_in_range(self):
        for wr in [0.0, 0.3, 0.5, 0.7, 1.0]:
            for dd in [0.0, -5.0, -15.0]:
                for tr in [0, 10, 30, 100]:
                    s = _compute_health_score(wr, dd, tr)
                    assert 0.0 <= s <= 1.0

    def test_more_trades_higher_score(self):
        low = _compute_health_score(0.55, -2.0, 0)
        high = _compute_health_score(0.55, -2.0, 30)
        assert high >= low


# ─── build_detailed_health (seeded) ──────────────────────────────────────────

class TestDetailedHealthSeeded:
    def test_top_level_fields(self):
        h = build_detailed_health()
        for field in ["system_status", "total_agents", "active_agents",
                       "avg_health_score", "agents", "generated_at"]:
            assert field in h

    def test_agents_is_list(self):
        h = build_detailed_health()
        assert isinstance(h["agents"], list)

    def test_seeded_has_agents(self):
        h = build_detailed_health()
        assert h["total_agents"] > 0
        assert len(h["agents"]) == h["total_agents"]

    def test_active_agents_lte_total(self):
        h = build_detailed_health()
        assert h["active_agents"] <= h["total_agents"]

    def test_avg_health_in_range(self):
        h = build_detailed_health()
        assert 0.0 <= h["avg_health_score"] <= 1.0

    def test_system_status_valid(self):
        h = build_detailed_health()
        assert h["system_status"] in ("healthy", "degraded", "critical")

    def test_agent_fields_present(self):
        h = build_detailed_health()
        for agent in h["agents"]:
            for field in ["agent_id", "strategy", "status", "last_signal_ts",
                           "win_rate_30d", "drawdown_pct", "health_score",
                           "trades_30d", "pnl_30d_usd"]:
                assert field in agent, f"Agent missing field: {field}"

    def test_sorted_by_health_score_desc(self):
        h = build_detailed_health()
        scores = [a["health_score"] for a in h["agents"]]
        assert scores == sorted(scores, reverse=True)

    def test_generated_at_is_recent(self):
        before = time.time()
        h = build_detailed_health()
        assert h["generated_at"] >= before

    def test_last_signal_ts_updated(self):
        h = build_detailed_health()
        now = time.time()
        for a in h["agents"]:
            assert a["last_signal_ts"] <= now


# ─── _update_agent_health ─────────────────────────────────────────────────────

class TestUpdateAgentHealth:
    def _make_run_result(self, agent_id="test_agent", win_rate=0.6, trades=10,
                          pnl=100.0, drawdown=-0.05):
        return {
            "agents": [{
                "id": agent_id,
                "profile": "momentum",
                "win_rate": win_rate,
                "trades": trades,
                "pnl_usd": pnl,
                "max_drawdown": drawdown,
                "reputation_end": 0.8,
                "current_position": None,
            }]
        }

    def test_creates_agent_entry(self):
        _update_agent_health(self._make_run_result())
        assert "test_agent" in _agent_health

    def test_agent_entry_fields(self):
        _update_agent_health(self._make_run_result())
        a = _agent_health["test_agent"]
        for field in ["agent_id", "strategy", "status", "last_signal_ts",
                       "win_rate_30d", "drawdown_pct", "health_score",
                       "trades_30d", "pnl_30d_usd", "reputation_score"]:
            assert field in a

    def test_trades_accumulate(self):
        _update_agent_health(self._make_run_result(trades=10))
        _update_agent_health(self._make_run_result(trades=5))
        assert _agent_health["test_agent"]["trades_30d"] == 15

    def test_pnl_accumulates(self):
        _update_agent_health(self._make_run_result(pnl=100.0))
        _update_agent_health(self._make_run_result(pnl=50.0))
        assert abs(_agent_health["test_agent"]["pnl_30d_usd"] - 150.0) < 0.01

    def test_live_data_overrides_seeded(self):
        _update_agent_health(self._make_run_result(agent_id="live_agent"))
        h = build_detailed_health()
        ids = [a["agent_id"] for a in h["agents"]]
        assert "live_agent" in ids

    def test_empty_agents_list_no_error(self):
        _update_agent_health({"agents": []})  # should not raise

    def test_health_score_is_float(self):
        _update_agent_health(self._make_run_result())
        score = _agent_health["test_agent"]["health_score"]
        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0


# ─── HTTP Endpoint ────────────────────────────────────────────────────────────

class TestHealthDetailedHTTP:
    @pytest.fixture(scope="class")
    def server(self):
        import socket
        s = socket.socket()
        s.bind(("", 0))
        port = s.getsockname()[1]
        s.close()
        srv = DemoServer(port=port)
        srv.start()
        time.sleep(0.3)
        yield port
        srv.stop()

    def test_endpoint_returns_200(self, server):
        url = f"http://localhost:{server}/demo/health/detailed"
        with urlopen(url) as r:
            assert r.status == 200

    def test_response_is_json(self, server):
        url = f"http://localhost:{server}/demo/health/detailed"
        with urlopen(url) as r:
            data = json.loads(r.read())
        assert "agents" in data
        assert "system_status" in data

    def test_response_has_agents(self, server):
        url = f"http://localhost:{server}/demo/health/detailed"
        with urlopen(url) as r:
            data = json.loads(r.read())
        assert len(data["agents"]) > 0
