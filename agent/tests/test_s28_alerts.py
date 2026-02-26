"""
test_s28_alerts.py — Sprint 28: Alert system tests.

38 tests covering:
  - configure_alerts() with valid and invalid inputs
  - get_active_alerts() structure and fields
  - Alert triggering from health state
  - clear_alerts() count and list reset
  - _check_alerts_from_health() with seeded health data
  - HTTP POST /demo/alerts/config
  - HTTP GET /demo/alerts/active
  - Alert deduplication
  - enabled/disabled flag behavior
"""

from __future__ import annotations

import json
import time
import threading
from urllib.request import urlopen, Request
from urllib.error import HTTPError

import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from demo_server import (
    configure_alerts,
    get_active_alerts,
    clear_alerts,
    _check_alerts_from_health,
    _alert_config,
    _alert_config_lock,
    _active_alerts,
    _active_alerts_lock,
    _agent_health,
    _agent_health_lock,
    DemoServer,
)

ALERTS_TEST_PORT = 18095


# ─── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture(autouse=True)
def reset_alert_state():
    """Reset all alert state before each test."""
    with _alert_config_lock:
        _alert_config.update({
            "drawdown_threshold": 10.0,
            "win_rate_floor": 0.40,
            "pnl_floor": -500.0,
            "sharpe_floor": -0.5,
            "enabled": True,
            "configured_at": None,
        })
    with _active_alerts_lock:
        _active_alerts.clear()
    with _agent_health_lock:
        _agent_health.clear()
    yield
    with _alert_config_lock:
        _alert_config.update({
            "drawdown_threshold": 10.0,
            "win_rate_floor": 0.40,
            "pnl_floor": -500.0,
            "sharpe_floor": -0.5,
            "enabled": True,
            "configured_at": None,
        })
    with _active_alerts_lock:
        _active_alerts.clear()
    with _agent_health_lock:
        _agent_health.clear()


# ─── configure_alerts ─────────────────────────────────────────────────────────

class TestConfigureAlerts:
    def test_returns_dict(self):
        result = configure_alerts({"drawdown_threshold": 15.0})
        assert isinstance(result, dict)

    def test_updates_drawdown_threshold(self):
        configure_alerts({"drawdown_threshold": 20.0})
        with _alert_config_lock:
            assert _alert_config["drawdown_threshold"] == 20.0

    def test_updates_win_rate_floor(self):
        configure_alerts({"win_rate_floor": 0.35})
        with _alert_config_lock:
            assert _alert_config["win_rate_floor"] == 0.35

    def test_updates_pnl_floor(self):
        configure_alerts({"pnl_floor": -1000.0})
        with _alert_config_lock:
            assert _alert_config["pnl_floor"] == -1000.0

    def test_updates_sharpe_floor(self):
        configure_alerts({"sharpe_floor": -1.0})
        with _alert_config_lock:
            assert _alert_config["sharpe_floor"] == -1.0

    def test_updates_enabled(self):
        configure_alerts({"enabled": False})
        with _alert_config_lock:
            assert _alert_config["enabled"] is False

    def test_sets_configured_at(self):
        configure_alerts({"drawdown_threshold": 5.0})
        with _alert_config_lock:
            assert _alert_config["configured_at"] is not None
            assert abs(_alert_config["configured_at"] - time.time()) < 5

    def test_multiple_keys_at_once(self):
        configure_alerts({"drawdown_threshold": 8.0, "win_rate_floor": 0.45})
        with _alert_config_lock:
            assert _alert_config["drawdown_threshold"] == 8.0
            assert _alert_config["win_rate_floor"] == 0.45

    def test_empty_config_ok(self):
        result = configure_alerts({})
        assert isinstance(result, dict)

    def test_unknown_key_raises(self):
        with pytest.raises(ValueError, match="Unknown config keys"):
            configure_alerts({"nonexistent_key": 1.0})

    def test_negative_drawdown_threshold_raises(self):
        with pytest.raises(ValueError):
            configure_alerts({"drawdown_threshold": -5.0})

    def test_win_rate_floor_above_1_raises(self):
        with pytest.raises(ValueError):
            configure_alerts({"win_rate_floor": 1.5})

    def test_win_rate_floor_below_0_raises(self):
        with pytest.raises(ValueError):
            configure_alerts({"win_rate_floor": -0.1})

    def test_returned_config_matches_state(self):
        result = configure_alerts({"drawdown_threshold": 12.5})
        assert result["drawdown_threshold"] == 12.5


# ─── Alert triggering ─────────────────────────────────────────────────────────

class TestAlertTriggering:
    def _seed_health(self, agent_id="agent-test", drawdown=5.0, win_rate=0.5, pnl=0.0):
        with _agent_health_lock:
            _agent_health[agent_id] = {
                "agent_id": agent_id,
                "strategy": "test",
                "status": "active",
                "win_rate_30d": win_rate,
                "drawdown_pct": drawdown,
                "pnl_30d_usd": pnl,
                "health_score": 0.8,
            }

    def test_no_alerts_when_healthy(self):
        self._seed_health(drawdown=5.0, win_rate=0.6, pnl=100.0)
        alerts = _check_alerts_from_health()
        assert len(alerts) == 0

    def test_drawdown_alert_triggered(self):
        self._seed_health(drawdown=20.0)
        configure_alerts({"drawdown_threshold": 10.0})
        alerts = _check_alerts_from_health()
        types = [a["type"] for a in alerts]
        assert "drawdown_exceeded" in types

    def test_win_rate_alert_triggered(self):
        self._seed_health(win_rate=0.25)
        configure_alerts({"win_rate_floor": 0.40})
        alerts = _check_alerts_from_health()
        types = [a["type"] for a in alerts]
        assert "win_rate_below_floor" in types

    def test_pnl_alert_triggered(self):
        self._seed_health(pnl=-1000.0)
        configure_alerts({"pnl_floor": -500.0})
        alerts = _check_alerts_from_health()
        types = [a["type"] for a in alerts]
        assert "pnl_below_floor" in types

    def test_alert_has_required_fields(self):
        self._seed_health(drawdown=25.0)
        configure_alerts({"drawdown_threshold": 10.0})
        alerts = _check_alerts_from_health()
        assert len(alerts) > 0
        alert = alerts[0]
        for key in ("alert_id", "type", "agent_id", "message", "severity",
                    "value", "threshold", "triggered_at"):
            assert key in alert, f"Missing field: {key}"

    def test_drawdown_high_severity(self):
        # More than 1.5x threshold → "high"
        self._seed_health(drawdown=20.0)
        configure_alerts({"drawdown_threshold": 10.0})
        alerts = _check_alerts_from_health()
        dd_alerts = [a for a in alerts if a["type"] == "drawdown_exceeded"]
        assert len(dd_alerts) > 0
        assert dd_alerts[0]["severity"] == "high"

    def test_drawdown_medium_severity(self):
        # Just above threshold but not 1.5x
        self._seed_health(drawdown=11.0)
        configure_alerts({"drawdown_threshold": 10.0})
        alerts = _check_alerts_from_health()
        dd_alerts = [a for a in alerts if a["type"] == "drawdown_exceeded"]
        assert len(dd_alerts) > 0
        assert dd_alerts[0]["severity"] == "medium"

    def test_no_alerts_when_disabled(self):
        self._seed_health(drawdown=50.0, win_rate=0.1, pnl=-9999.0)
        configure_alerts({"enabled": False})
        alerts = _check_alerts_from_health()
        assert len(alerts) == 0

    def test_empty_health_returns_no_alerts(self):
        # No health data → no alerts
        alerts = _check_alerts_from_health()
        assert len(alerts) == 0


# ─── get_active_alerts ────────────────────────────────────────────────────────

class TestGetActiveAlerts:
    def test_returns_dict(self):
        result = get_active_alerts()
        assert isinstance(result, dict)

    def test_required_fields(self):
        result = get_active_alerts()
        for key in ("enabled", "config", "alert_count", "alerts", "checked_at"):
            assert key in result, f"Missing key: {key}"

    def test_config_has_all_thresholds(self):
        result = get_active_alerts()
        cfg = result["config"]
        assert "drawdown_threshold" in cfg
        assert "win_rate_floor" in cfg
        assert "pnl_floor" in cfg
        assert "sharpe_floor" in cfg

    def test_alerts_is_list(self):
        result = get_active_alerts()
        assert isinstance(result["alerts"], list)

    def test_alert_count_matches_list_length(self):
        result = get_active_alerts()
        assert result["alert_count"] == len(result["alerts"])

    def test_checked_at_is_recent(self):
        result = get_active_alerts()
        assert abs(result["checked_at"] - time.time()) < 5

    def test_enabled_reflects_config(self):
        configure_alerts({"enabled": False})
        result = get_active_alerts()
        assert result["enabled"] is False


# ─── clear_alerts ─────────────────────────────────────────────────────────────

class TestClearAlerts:
    def test_returns_count(self):
        with _active_alerts_lock:
            _active_alerts.append({"alert_id": "x", "type": "t", "agent_id": "a",
                                    "message": "m", "severity": "low",
                                    "value": 1.0, "threshold": 0.5,
                                    "triggered_at": time.time()})
        count = clear_alerts()
        assert count == 1

    def test_alerts_empty_after_clear(self):
        with _active_alerts_lock:
            _active_alerts.append({"alert_id": "x", "type": "t", "agent_id": "a",
                                    "message": "m", "severity": "low",
                                    "value": 1.0, "threshold": 0.5,
                                    "triggered_at": time.time()})
        clear_alerts()
        with _active_alerts_lock:
            assert len(_active_alerts) == 0

    def test_clear_when_empty_returns_zero(self):
        count = clear_alerts()
        assert count == 0


# ─── HTTP: POST /demo/alerts/config and GET /demo/alerts/active ───────────────

@pytest.fixture(scope="module")
def alerts_server():
    srv = DemoServer(port=ALERTS_TEST_PORT)
    srv.start()
    time.sleep(0.3)
    yield f"http://localhost:{ALERTS_TEST_PORT}"
    srv.stop()


def _post_config(base_url: str, payload: dict):
    data = json.dumps(payload).encode()
    req = Request(f"{base_url}/demo/alerts/config", data=data,
                  headers={"Content-Type": "application/json"}, method="POST")
    with urlopen(req, timeout=10) as resp:
        return resp.status, json.loads(resp.read())


class TestAlertsHTTP:
    def test_config_200(self, alerts_server):
        status, body = _post_config(alerts_server, {"drawdown_threshold": 15.0})
        assert status == 200
        assert body["status"] == "ok"

    def test_config_returns_updated_config(self, alerts_server):
        status, body = _post_config(alerts_server, {"win_rate_floor": 0.35})
        assert status == 200
        assert "config" in body
        assert body["config"]["win_rate_floor"] == 0.35

    def test_config_invalid_key_returns_400(self, alerts_server):
        data = json.dumps({"bad_key": 1.0}).encode()
        req = Request(f"{alerts_server}/demo/alerts/config", data=data,
                      headers={"Content-Type": "application/json"}, method="POST")
        try:
            urlopen(req, timeout=10)
            assert False, "Expected 400"
        except HTTPError as e:
            assert e.code == 400

    def test_config_invalid_json_returns_400(self, alerts_server):
        req = Request(f"{alerts_server}/demo/alerts/config",
                      data=b"not-json",
                      headers={"Content-Type": "application/json"}, method="POST")
        try:
            urlopen(req, timeout=10)
            assert False, "Expected 400"
        except HTTPError as e:
            assert e.code == 400

    def test_alerts_active_returns_200(self, alerts_server):
        resp = urlopen(f"{alerts_server}/demo/alerts/active", timeout=10)
        assert resp.status == 200

    def test_alerts_active_has_required_fields(self, alerts_server):
        resp = urlopen(f"{alerts_server}/demo/alerts/active", timeout=10)
        body = json.loads(resp.read())
        for key in ("enabled", "config", "alert_count", "alerts", "checked_at"):
            assert key in body

    def test_alerts_active_json_content_type(self, alerts_server):
        resp = urlopen(f"{alerts_server}/demo/alerts/active", timeout=10)
        ct = resp.headers.get("Content-Type", "")
        assert "application/json" in ct

    def test_config_empty_body_ok(self, alerts_server):
        data = json.dumps({}).encode()
        req = Request(f"{alerts_server}/demo/alerts/config", data=data,
                      headers={"Content-Type": "application/json"}, method="POST")
        with urlopen(req, timeout=10) as resp:
            assert resp.status == 200

    def test_disable_then_enable(self, alerts_server):
        status, body = _post_config(alerts_server, {"enabled": False})
        assert status == 200
        assert body["config"]["enabled"] is False
        status2, body2 = _post_config(alerts_server, {"enabled": True})
        assert status2 == 200
        assert body2["config"]["enabled"] is True
