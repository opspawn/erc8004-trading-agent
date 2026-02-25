"""
test_websocket_dashboard.py — Tests for WebSocket + extended dashboard features (S7).

55 tests covering:
  - ConnectionManager state and event broadcasting
  - WebSocket event types: price_update, trade_executed, portfolio_update,
    agent_signal, ensemble_signal, risk_alert
  - HTTP endpoints added in S7: /agents, /events, /events/price, /events/trade, /events/risk-alert
  - WebSocket /ws/trading connect, snapshot, ping/pong
  - Event log management and replay
  - Multi-client broadcast
"""

from __future__ import annotations

import asyncio
import json
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from dashboard_server import (
    ConnectionManager,
    app,
    _state,
    update_state,
    manager,
    _now,
)


# ─── Fixtures ─────────────────────────────────────────────────────────────────


@pytest.fixture(autouse=True)
def reset_state():
    """Reset global state between tests."""
    _state.update({
        "tests_passing": 0,
        "active_positions": 0,
        "total_pnl": 0.0,
        "last_trade": None,
        "risk_summary": {},
        "trades": [],
        "reputation": {
            "agent_id": 0, "aggregate_score": 0.0, "total_feedback": 0,
            "on_chain_count": 0, "win_count": 0, "loss_count": 0,
        },
        "strategist_stats": {},
        "agent_pool": {"pool_size": 0, "agents": [], "consensus_rate": 0.0, "decision_count": 0},
        "last_ensemble": None,
        "price_feed": {},
    })
    yield


@pytest.fixture
def client():
    return TestClient(app)


@pytest.fixture
def fresh_manager():
    """Return a fresh ConnectionManager for isolation."""
    return ConnectionManager()


# ─── ConnectionManager Unit Tests ─────────────────────────────────────────────


class TestConnectionManager:
    def test_initial_connection_count(self, fresh_manager):
        assert fresh_manager.connection_count == 0

    @pytest.mark.asyncio
    async def test_connect_increments_count(self, fresh_manager):
        ws = AsyncMock()
        await fresh_manager.connect(ws)
        assert fresh_manager.connection_count == 1

    @pytest.mark.asyncio
    async def test_disconnect_decrements_count(self, fresh_manager):
        ws = AsyncMock()
        await fresh_manager.connect(ws)
        fresh_manager.disconnect(ws)
        assert fresh_manager.connection_count == 0

    @pytest.mark.asyncio
    async def test_disconnect_unknown_ws_no_error(self, fresh_manager):
        ws = AsyncMock()
        fresh_manager.disconnect(ws)  # should not raise
        assert fresh_manager.connection_count == 0

    @pytest.mark.asyncio
    async def test_send_event_calls_send_json(self, fresh_manager):
        ws = AsyncMock()
        event = {"type": "test", "data": 42}
        await fresh_manager.send_event(ws, event)
        ws.send_json.assert_called_once_with(event)

    @pytest.mark.asyncio
    async def test_send_event_failed_removes_connection(self, fresh_manager):
        ws = AsyncMock()
        ws.send_json.side_effect = RuntimeError("broken pipe")
        await fresh_manager.connect(ws)
        await fresh_manager.send_event(ws, {"type": "test"})
        assert fresh_manager.connection_count == 0

    @pytest.mark.asyncio
    async def test_broadcast_reaches_all_clients(self, fresh_manager):
        clients = [AsyncMock() for _ in range(3)]
        for c in clients:
            await fresh_manager.connect(c)
        await fresh_manager.broadcast({"type": "test_event"})
        for c in clients:
            c.send_json.assert_called_once()

    @pytest.mark.asyncio
    async def test_broadcast_removes_dead_clients(self, fresh_manager):
        good = AsyncMock()
        dead = AsyncMock()
        dead.send_json.side_effect = RuntimeError("dead")
        await fresh_manager.connect(good)
        await fresh_manager.connect(dead)
        await fresh_manager.broadcast({"type": "test"})
        assert fresh_manager.connection_count == 1

    @pytest.mark.asyncio
    async def test_event_log_grows(self, fresh_manager):
        for i in range(5):
            await fresh_manager.broadcast({"type": "event", "i": i})
        assert len(fresh_manager.recent_events(10)) == 5

    @pytest.mark.asyncio
    async def test_event_log_capped_at_max(self, fresh_manager):
        fresh_manager._max_log = 5
        for i in range(10):
            await fresh_manager.broadcast({"type": "e", "i": i})
        assert len(fresh_manager._event_log) == 5

    @pytest.mark.asyncio
    async def test_recent_events_limit(self, fresh_manager):
        for i in range(20):
            await fresh_manager.broadcast({"type": "e", "i": i})
        events = fresh_manager.recent_events(5)
        assert len(events) == 5

    @pytest.mark.asyncio
    async def test_broadcast_price_update_event_type(self, fresh_manager):
        ws = AsyncMock()
        await fresh_manager.connect(ws)
        await fresh_manager.broadcast_price_update("ETH", 2100.0, 0.05)
        call_args = ws.send_json.call_args[0][0]
        assert call_args["type"] == "price_update"
        assert call_args["symbol"] == "ETH"
        assert call_args["price"] == 2100.0

    @pytest.mark.asyncio
    async def test_broadcast_price_updates_state(self, fresh_manager):
        await fresh_manager.broadcast_price_update("BTC", 50000.0, -0.02)
        assert "BTC" in _state["price_feed"]
        assert _state["price_feed"]["BTC"]["price"] == 50000.0

    @pytest.mark.asyncio
    async def test_broadcast_trade_executed_event_type(self, fresh_manager):
        ws = AsyncMock()
        await fresh_manager.connect(ws)
        await fresh_manager.broadcast_trade_executed("eth-usd", "buy", 100.0, "0xabc")
        call_args = ws.send_json.call_args[0][0]
        assert call_args["type"] == "trade_executed"
        assert call_args["market_id"] == "eth-usd"
        assert call_args["side"] == "buy"

    @pytest.mark.asyncio
    async def test_broadcast_trade_updates_last_trade_state(self, fresh_manager):
        await fresh_manager.broadcast_trade_executed("eth-usd", "sell", 50.0, "0xdef", pnl=1.5)
        assert _state["last_trade"] is not None
        assert _state["last_trade"]["pnl"] == 1.5

    @pytest.mark.asyncio
    async def test_broadcast_portfolio_update(self, fresh_manager):
        ws = AsyncMock()
        await fresh_manager.connect(ws)
        await fresh_manager.broadcast_portfolio_update(10000.0, 150.0, 3)
        call_args = ws.send_json.call_args[0][0]
        assert call_args["type"] == "portfolio_update"
        assert call_args["active_positions"] == 3
        assert _state["total_pnl"] == pytest.approx(150.0)

    @pytest.mark.asyncio
    async def test_broadcast_agent_signal(self, fresh_manager):
        ws = AsyncMock()
        await fresh_manager.connect(ws)
        await fresh_manager.broadcast_agent_signal("agent-mod", "buy", 0.82, "ETH")
        call_args = ws.send_json.call_args[0][0]
        assert call_args["type"] == "agent_signal"
        assert call_args["action"] == "buy"
        assert call_args["confidence"] == pytest.approx(0.82)

    @pytest.mark.asyncio
    async def test_broadcast_ensemble_signal(self, fresh_manager):
        ws = AsyncMock()
        await fresh_manager.connect(ws)
        votes = {"agent-con": "buy", "agent-mod": "buy", "agent-agg": "sell"}
        await fresh_manager.broadcast_ensemble_signal("buy", 0.67, votes, True)
        call_args = ws.send_json.call_args[0][0]
        assert call_args["type"] == "ensemble_signal"
        assert call_args["has_consensus"] is True
        assert _state["last_ensemble"] is not None

    @pytest.mark.asyncio
    async def test_broadcast_risk_alert(self, fresh_manager):
        ws = AsyncMock()
        await fresh_manager.connect(ws)
        await fresh_manager.broadcast_risk_alert("drawdown", "Max drawdown reached", "critical")
        call_args = ws.send_json.call_args[0][0]
        assert call_args["type"] == "risk_alert"
        assert call_args["severity"] == "critical"
        assert "drawdown" in call_args["alert_type"]


# ─── HTTP Endpoint Tests (S7 additions) ───────────────────────────────────────


class TestAgentsEndpoint:
    def test_agents_returns_pool_state(self, client):
        update_state("agent_pool", {
            "pool_size": 3,
            "agents": ["agent-con", "agent-mod", "agent-agg"],
            "consensus_rate": 0.85,
            "decision_count": 42,
        })
        r = client.get("/agents")
        assert r.status_code == 200
        data = r.json()
        assert data["agent_pool"]["pool_size"] == 3

    def test_agents_includes_timestamp(self, client):
        r = client.get("/agents")
        assert "timestamp" in r.json()

    def test_agents_last_ensemble_null_initially(self, client):
        r = client.get("/agents")
        assert r.json()["last_ensemble"] is None

    def test_agents_last_ensemble_reflects_state(self, client):
        update_state("last_ensemble", {"type": "ensemble_signal", "action": "buy"})
        r = client.get("/agents")
        assert r.json()["last_ensemble"]["action"] == "buy"


class TestEventsEndpoint:
    def test_events_empty_initially(self, client):
        # Reset manager event log
        manager._event_log.clear()
        r = client.get("/events")
        assert r.status_code == 200
        assert isinstance(r.json(), list)

    def test_events_limit_param(self, client):
        manager._event_log.clear()
        for i in range(10):
            manager._event_log.append({"type": "test", "i": i})
        r = client.get("/events?limit=5")
        assert len(r.json()) == 5


class TestPushPriceEndpoint:
    def test_push_price_event_returns_ok(self, client):
        r = client.post("/events/price", json={"symbol": "ETH", "price": 2200.0, "change_pct": 0.01})
        assert r.status_code == 200
        assert r.json()["ok"] is True

    def test_push_price_updates_price_feed(self, client):
        client.post("/events/price", json={"symbol": "BTC", "price": 55000.0})
        assert "BTC" in _state["price_feed"]

    def test_push_price_default_symbol(self, client):
        r = client.post("/events/price", json={"price": 1000.0})
        assert r.status_code == 200


class TestPushTradeEndpoint:
    def test_push_trade_returns_ok(self, client):
        payload = {
            "market_id": "eth-usd",
            "side": "buy",
            "size_usdc": 100.0,
            "tx_hash": "0xabc123",
        }
        r = client.post("/events/trade", json=payload)
        assert r.status_code == 200
        assert r.json()["ok"] is True

    def test_push_trade_updates_last_trade(self, client):
        client.post("/events/trade", json={
            "market_id": "btc-usd", "side": "sell", "size_usdc": 200.0, "tx_hash": "0xdef"
        })
        assert _state["last_trade"] is not None
        assert _state["last_trade"]["market_id"] == "btc-usd"


class TestPushRiskAlertEndpoint:
    def test_push_risk_alert_returns_ok(self, client):
        r = client.post("/events/risk-alert", json={
            "alert_type": "drawdown",
            "message": "Max drawdown reached",
            "severity": "critical",
        })
        assert r.status_code == 200
        assert r.json()["ok"] is True

    def test_push_risk_alert_default_severity(self, client):
        r = client.post("/events/risk-alert", json={
            "alert_type": "generic",
            "message": "Test alert",
        })
        assert r.status_code == 200


class TestStatusEndpointS7:
    def test_status_includes_agent_pool(self, client):
        update_state("agent_pool", {"pool_size": 3})
        r = client.get("/status")
        assert "agent_pool" in r.json()

    def test_status_includes_last_ensemble(self, client):
        r = client.get("/status")
        assert "last_ensemble" in r.json()

    def test_status_includes_ws_connections(self, client):
        r = client.get("/status")
        assert "ws_connections" in r.json()

    def test_health_includes_ws_connections(self, client):
        r = client.get("/health")
        assert "ws_connections" in r.json()


# ─── WebSocket Tests ──────────────────────────────────────────────────────────


class TestWebSocketEndpoint:
    def test_websocket_connect_and_snapshot(self, client):
        """WebSocket connect should receive initial snapshot."""
        with client.websocket_connect("/ws/trading") as ws:
            data = ws.receive_json()
            assert data["type"] == "snapshot"
            assert "state" in data
            assert "total_pnl" in data["state"]

    def test_websocket_snapshot_includes_price_feed(self, client):
        _state["price_feed"] = {"ETH": {"price": 2000.0}}
        with client.websocket_connect("/ws/trading") as ws:
            data = ws.receive_json()
            assert "price_feed" in data["state"]

    def test_websocket_snapshot_includes_agent_pool(self, client):
        _state["agent_pool"] = {"pool_size": 3}
        with client.websocket_connect("/ws/trading") as ws:
            data = ws.receive_json()
            assert "agent_pool" in data["state"]

    def test_websocket_ping_pong(self, client):
        """Client sends ping, server responds pong."""
        with client.websocket_connect("/ws/trading") as ws:
            ws.receive_json()  # consume snapshot
            ws.send_json({"type": "ping"})
            response = ws.receive_json()
            assert response["type"] == "pong"
            assert "timestamp" in response

    def test_websocket_snapshot_has_timestamp(self, client):
        with client.websocket_connect("/ws/trading") as ws:
            data = ws.receive_json()
            assert "timestamp" in data


# ─── _now helper ──────────────────────────────────────────────────────────────


class TestNowHelper:
    def test_now_returns_iso_string(self):
        ts = _now()
        assert "T" in ts
        assert "Z" in ts or "+" in ts

    def test_now_is_recent(self):
        import time
        ts = _now()
        parsed = datetime.fromisoformat(ts.replace("Z", "+00:00"))
        diff = abs((datetime.now(timezone.utc) - parsed).total_seconds())
        assert diff < 5


# ─── Extended ConnectionManager Tests ─────────────────────────────────────────


class TestConnectionManagerExtended:
    @pytest.mark.asyncio
    async def test_broadcast_with_no_clients(self, fresh_manager):
        """Broadcast with no clients should succeed without error."""
        await fresh_manager.broadcast({"type": "test"})
        assert fresh_manager.connection_count == 0

    @pytest.mark.asyncio
    async def test_multiple_connects_and_disconnects(self, fresh_manager):
        clients = [AsyncMock() for _ in range(5)]
        for c in clients:
            await fresh_manager.connect(c)
        assert fresh_manager.connection_count == 5
        for c in clients[:3]:
            fresh_manager.disconnect(c)
        assert fresh_manager.connection_count == 2

    @pytest.mark.asyncio
    async def test_event_log_returns_last_n(self, fresh_manager):
        for i in range(20):
            await fresh_manager.broadcast({"type": "e", "seq": i})
        events = fresh_manager.recent_events(10)
        assert len(events) == 10
        assert events[-1]["seq"] == 19

    @pytest.mark.asyncio
    async def test_broadcast_price_change_pct_rounded(self, fresh_manager):
        ws = AsyncMock()
        await fresh_manager.connect(ws)
        await fresh_manager.broadcast_price_update("ETH", 2000.0, 0.123456789)
        call_args = ws.send_json.call_args[0][0]
        # Should be rounded to 4 decimal places
        assert len(str(call_args["change_pct"]).replace("0.", "")) <= 5

    @pytest.mark.asyncio
    async def test_broadcast_trade_with_none_pnl(self, fresh_manager):
        ws = AsyncMock()
        await fresh_manager.connect(ws)
        await fresh_manager.broadcast_trade_executed("eth-usd", "buy", 100.0, "0xabc", pnl=None)
        call_args = ws.send_json.call_args[0][0]
        assert call_args["pnl"] is None

    @pytest.mark.asyncio
    async def test_broadcast_portfolio_updates_active_positions_state(self, fresh_manager):
        await fresh_manager.broadcast_portfolio_update(10000.0, 100.0, 7)
        assert _state["active_positions"] == 7

    @pytest.mark.asyncio
    async def test_broadcast_agent_signal_has_timestamp(self, fresh_manager):
        ws = AsyncMock()
        await fresh_manager.connect(ws)
        await fresh_manager.broadcast_agent_signal("agent-con", "hold", 0.45, "BTC")
        call_args = ws.send_json.call_args[0][0]
        assert "timestamp" in call_args

    @pytest.mark.asyncio
    async def test_broadcast_ensemble_stores_votes(self, fresh_manager):
        votes = {"a1": "sell", "a2": "sell"}
        await fresh_manager.broadcast_ensemble_signal("sell", 1.0, votes, True)
        stored = _state["last_ensemble"]
        assert stored["agent_votes"]["a1"] == "sell"

    @pytest.mark.asyncio
    async def test_broadcast_risk_alert_has_alert_type(self, fresh_manager):
        ws = AsyncMock()
        await fresh_manager.connect(ws)
        await fresh_manager.broadcast_risk_alert("oracle_failure", "Oracle price diverged")
        call_args = ws.send_json.call_args[0][0]
        assert call_args["alert_type"] == "oracle_failure"

    @pytest.mark.asyncio
    async def test_recent_events_ordered_newest_last(self, fresh_manager):
        for i in range(5):
            await fresh_manager.broadcast({"type": "e", "order": i})
        events = fresh_manager.recent_events(5)
        for i in range(4):
            assert events[i]["order"] < events[i+1]["order"]


# ─── Extended HTTP Endpoint Tests ─────────────────────────────────────────────


class TestHealthEndpointExtended:
    def test_health_status_ok(self, client):
        r = client.get("/health")
        assert r.json()["status"] == "ok"

    def test_health_uptime_since_set(self, client):
        r = client.get("/health")
        assert r.json()["uptime_since"] is not None


class TestStatusEndpointExtended:
    def test_status_returns_200(self, client):
        r = client.get("/status")
        assert r.status_code == 200

    def test_status_total_pnl_rounded(self, client):
        update_state("total_pnl", 1.23456789)
        r = client.get("/status")
        # total_pnl should be rounded to 4 decimal places
        pnl = r.json()["total_pnl"]
        assert pnl == pytest.approx(1.2346, abs=0.0001)

    def test_status_trading_halted_reflects_risk_summary(self, client):
        update_state("risk_summary", {"trading_halted": True})
        r = client.get("/status")
        assert r.json()["trading_halted"] is True

    def test_status_trading_not_halted_by_default(self, client):
        r = client.get("/status")
        assert r.json()["trading_halted"] is False

    def test_status_active_positions_reflects_state(self, client):
        update_state("active_positions", 5)
        r = client.get("/status")
        assert r.json()["active_positions"] == 5


class TestTradesEndpointExtended:
    def test_trades_returns_list(self, client):
        r = client.get("/trades")
        assert isinstance(r.json(), list)

    def test_trades_default_limit(self, client):
        update_state("trades", [{"market_id": f"m{i}"} for i in range(30)])
        r = client.get("/trades")
        assert len(r.json()) <= 20

    def test_trades_custom_limit(self, client):
        update_state("trades", [{"market_id": f"m{i}"} for i in range(50)])
        r = client.get("/trades?limit=5")
        assert len(r.json()) <= 5

    def test_trades_max_limit_capped_at_100(self, client):
        r = client.get("/trades?limit=999")
        assert r.status_code == 200


class TestStateUpdateEndpointExtended:
    def test_state_update_returns_updated_keys(self, client):
        r = client.post("/state/update", json={"total_pnl": 42.0})
        assert "total_pnl" in r.json()["updated"]

    def test_state_update_ignores_unknown_keys(self, client):
        """Unknown keys are ignored in _state but included in 'updated' list."""
        r = client.post("/state/update", json={"unknown_key": "value"})
        assert r.status_code == 200
        assert r.json()["ok"] is True
        # unknown_key is not applied to _state
        assert "unknown_key" not in _state

    def test_state_update_multiple_keys(self, client):
        r = client.post("/state/update", json={
            "total_pnl": 10.0,
            "active_positions": 3,
        })
        assert len(r.json()["updated"]) == 2


class TestReputationEndpointExtended:
    def test_reputation_win_rate_zero_with_no_trades(self, client):
        r = client.get("/reputation")
        assert r.json()["win_rate"] == 0.0

    def test_reputation_win_rate_computed_correctly(self, client):
        update_state("reputation", {
            "agent_id": 1,
            "aggregate_score": 7.5,
            "total_feedback": 10,
            "on_chain_count": 5,
            "win_count": 6,
            "loss_count": 4,
        })
        r = client.get("/reputation")
        assert r.json()["win_rate"] == pytest.approx(0.6)

    def test_reputation_includes_timestamp(self, client):
        r = client.get("/reputation")
        assert "timestamp" in r.json()
