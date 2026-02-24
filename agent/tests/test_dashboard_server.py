"""
Tests for dashboard_server.py — 18 tests covering FastAPI endpoints.

Uses httpx.AsyncClient (ASGI transport) to test the app without running a server.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import pytest
import pytest_asyncio
from datetime import datetime, timezone

import httpx
from fastapi.testclient import TestClient

from dashboard_server import app, _state, update_state


# ─── Sync test client (for synchronous tests) ─────────────────────────────────

@pytest.fixture
def client():
    return TestClient(app)


@pytest.fixture(autouse=True)
def reset_state():
    """Reset global state between tests."""
    _state["tests_passing"] = 0
    _state["active_positions"] = 0
    _state["total_pnl"] = 0.0
    _state["last_trade"] = None
    _state["risk_summary"] = {}
    _state["trades"] = []
    _state["reputation"] = {
        "agent_id": 0,
        "aggregate_score": 0.0,
        "total_feedback": 0,
        "on_chain_count": 0,
        "win_count": 0,
        "loss_count": 0,
    }
    _state["strategist_stats"] = {}
    yield


# ─── Health ───────────────────────────────────────────────────────────────────

class TestHealth:

    def test_health_ok(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert "timestamp" in data

    def test_health_has_uptime(self, client):
        resp = client.get("/health")
        data = resp.json()
        assert "uptime_since" in data


# ─── Status Endpoint ──────────────────────────────────────────────────────────

class TestStatus:

    def test_status_default(self, client):
        resp = client.get("/status")
        assert resp.status_code == 200
        data = resp.json()
        assert data["tests_passing"] == 0
        assert data["active_positions"] == 0
        assert data["total_pnl"] == 0.0
        assert data["last_trade"] is None

    def test_status_reflects_state(self, client):
        update_state("tests_passing", 166)
        update_state("total_pnl", 12.50)
        update_state("active_positions", 3)

        resp = client.get("/status")
        data = resp.json()
        assert data["tests_passing"] == 166
        assert data["total_pnl"] == 12.50
        assert data["active_positions"] == 3

    def test_status_with_last_trade(self, client):
        trade = {
            "market_id": "m1",
            "question": "Will BTC hit 100k?",
            "side": "YES",
            "size_usdc": 5.0,
            "executed_at": datetime.now(timezone.utc).isoformat(),
            "confidence": 0.72,
            "reasoning": "YES appears cheap",
        }
        update_state("last_trade", trade)

        resp = client.get("/status")
        data = resp.json()
        assert data["last_trade"] is not None
        assert data["last_trade"]["question"] == "Will BTC hit 100k?"

    def test_status_has_timestamp(self, client):
        resp = client.get("/status")
        data = resp.json()
        assert "timestamp" in data

    def test_status_trading_halted_field(self, client):
        update_state("risk_summary", {"trading_halted": True, "halt_reason": "drawdown"})
        resp = client.get("/status")
        data = resp.json()
        assert data["trading_halted"] is True


# ─── Trades Endpoint ──────────────────────────────────────────────────────────

class TestTrades:

    def test_trades_empty_default(self, client):
        resp = client.get("/trades")
        assert resp.status_code == 200
        data = resp.json()
        assert isinstance(data, list)
        assert len(data) == 0

    def test_trades_returns_state_trades(self, client):
        trades = [
            {"market_id": "m1", "side": "YES", "size_usdc": 5.0, "pnl_usdc": 0.0, "outcome": "PENDING"},
            {"market_id": "m2", "side": "NO", "size_usdc": 3.0, "pnl_usdc": 1.5, "outcome": "WIN"},
        ]
        update_state("trades", trades)

        resp = client.get("/trades")
        data = resp.json()
        assert len(data) == 2

    def test_trades_limit_param(self, client):
        trades = [{"market_id": f"m{i}", "side": "YES", "size_usdc": 1.0} for i in range(30)]
        update_state("trades", trades)

        resp = client.get("/trades?limit=5")
        data = resp.json()
        assert len(data) <= 5

    def test_trades_limit_capped_at_100(self, client):
        # Requesting more than 100 should not error
        resp = client.get("/trades?limit=999")
        assert resp.status_code == 200


# ─── Reputation Endpoint ──────────────────────────────────────────────────────

class TestReputation:

    def test_reputation_default(self, client):
        resp = client.get("/reputation")
        assert resp.status_code == 200
        data = resp.json()
        assert data["agent_id"] == 0
        assert data["aggregate_score"] == 0.0
        assert data["total_feedback"] == 0

    def test_reputation_reflects_state(self, client):
        update_state("reputation", {
            "agent_id": 42,
            "aggregate_score": 8.5,
            "total_feedback": 15,
            "on_chain_count": 5,
            "win_count": 10,
            "loss_count": 5,
        })
        resp = client.get("/reputation")
        data = resp.json()
        assert data["agent_id"] == 42
        assert data["aggregate_score"] == 8.5
        assert data["total_feedback"] == 15
        assert data["win_count"] == 10

    def test_reputation_win_rate_calculated(self, client):
        update_state("reputation", {
            "agent_id": 1,
            "aggregate_score": 7.0,
            "total_feedback": 10,
            "on_chain_count": 0,
            "win_count": 7,
            "loss_count": 3,
        })
        resp = client.get("/reputation")
        data = resp.json()
        assert abs(data["win_rate"] - 0.7) < 0.01

    def test_reputation_has_timestamp(self, client):
        resp = client.get("/reputation")
        data = resp.json()
        assert "timestamp" in data


# ─── State Update Endpoint ────────────────────────────────────────────────────

class TestStateUpdate:

    def test_update_state_endpoint(self, client):
        payload = {"tests_passing": 250, "total_pnl": 7.33}
        resp = client.post("/state/update", json=payload)
        assert resp.status_code == 200
        data = resp.json()
        assert data["ok"] is True
        assert "tests_passing" in data["updated"]

        # Verify state was updated
        status_resp = client.get("/status")
        status = status_resp.json()
        assert status["tests_passing"] == 250

    def test_update_unknown_key_ignored(self, client):
        # Unknown keys should not break anything
        payload = {"nonexistent_key": "value"}
        resp = client.post("/state/update", json=payload)
        assert resp.status_code == 200


# ─── Root / Dashboard ─────────────────────────────────────────────────────────

class TestRoot:

    def test_root_returns_html(self, client):
        resp = client.get("/")
        assert resp.status_code == 200
        assert "html" in resp.headers["content-type"].lower()
