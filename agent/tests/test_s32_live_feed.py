"""
test_s32_live_feed.py — Sprint 32: Live Feed, Scenario Orchestrator, Leaderboard, Health tests.

Covers:
  - get_live_feed: buffer, last=N, event structure
  - run_scenario: all 4 scenarios, circuit breaker in bear_crash, timeline structure
  - build_agents_leaderboard: composite score sort, rank_change, extended fields
  - get_demo_status: feature summary structure
  - HTTP GET /demo/live/feed, /demo/agents/leaderboard, /demo/status, /health, /demo/health
  - HTTP POST /demo/scenario/run: valid scenarios, invalid inputs, circuit breaker
"""

from __future__ import annotations

import json
import threading
import time
from urllib.request import urlopen, Request
from urllib.error import HTTPError

import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from demo_server import (
    # Live feed
    get_live_feed,
    _LIVE_FEED_BUFFER,
    _LIVE_FEED_LOCK,
    _generate_feed_event,
    _push_feed_event,
    _FEED_EVENT_TYPES,
    _FEED_AGENT_IDS,
    _FEED_SYMBOLS,
    # Scenario orchestrator
    run_scenario,
    _VALID_SCENARIOS,
    _SCENARIO_CONFIGS,
    # Leaderboard
    build_agents_leaderboard,
    _EXTENDED_LEADERBOARD,
    # Status
    get_demo_status,
    # Server
    DemoServer,
    SERVER_VERSION,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def server():
    import socket, time as _time
    # Find a free port
    sock = socket.socket()
    sock.bind(("", 0))
    port = sock.getsockname()[1]
    sock.close()
    srv = DemoServer(port=port)
    srv.start()
    _time.sleep(0.2)
    yield f"http://localhost:{port}"
    srv.stop()


def _get(url: str) -> dict:
    with urlopen(url, timeout=5) as resp:
        return json.loads(resp.read())


def _post(url: str, data: dict) -> dict:
    body = json.dumps(data).encode("utf-8")
    req = Request(url, data=body, headers={"Content-Type": "application/json"})
    with urlopen(req, timeout=10) as resp:
        return json.loads(resp.read())


def _post_expecting_error(url: str, data: dict) -> tuple[int, dict]:
    body = json.dumps(data).encode("utf-8")
    req = Request(url, data=body, headers={"Content-Type": "application/json"})
    try:
        with urlopen(req, timeout=5) as resp:
            return resp.status, json.loads(resp.read())
    except HTTPError as e:
        return e.code, json.loads(e.read())


# ── Unit Tests: get_live_feed ─────────────────────────────────────────────────

class TestLiveFeed:

    def test_get_live_feed_returns_dict(self):
        result = get_live_feed(last=5)
        assert isinstance(result, dict)

    def test_get_live_feed_has_events_key(self):
        result = get_live_feed(last=5)
        assert "events" in result

    def test_get_live_feed_has_count_key(self):
        result = get_live_feed(last=5)
        assert "count" in result

    def test_get_live_feed_count_equals_events_length(self):
        result = get_live_feed(last=5)
        assert result["count"] == len(result["events"])

    def test_get_live_feed_last_param_limits_results(self):
        result = get_live_feed(last=3)
        assert result["count"] <= 3

    def test_get_live_feed_last_100_returns_at_most_100(self):
        result = get_live_feed(last=100)
        assert result["count"] <= 100

    def test_get_live_feed_has_feed_seq(self):
        result = get_live_feed(last=5)
        assert "feed_seq" in result
        assert isinstance(result["feed_seq"], int)

    def test_get_live_feed_has_total_buffered(self):
        result = get_live_feed(last=5)
        assert "total_buffered" in result

    def test_get_live_feed_has_generated_at(self):
        result = get_live_feed(last=5)
        assert "generated_at" in result

    def test_get_live_feed_events_have_required_fields(self):
        result = get_live_feed(last=10)
        for evt in result["events"]:
            assert "event" in evt
            assert "agent_id" in evt
            assert "timestamp" in evt

    def test_get_live_feed_event_types_are_valid(self):
        result = get_live_feed(last=20)
        valid = set(_FEED_EVENT_TYPES)
        for evt in result["events"]:
            assert evt["event"] in valid

    def test_generate_feed_event_returns_dict(self):
        evt = _generate_feed_event(event_type="agent_vote", seed_offset=0)
        assert isinstance(evt, dict)

    def test_generate_feed_event_agent_vote_has_action(self):
        evt = _generate_feed_event(event_type="agent_vote", seed_offset=0)
        assert "action" in evt
        assert evt["action"] in ("BUY", "SELL", "HOLD")

    def test_generate_feed_event_trade_executed_has_price(self):
        evt = _generate_feed_event(event_type="trade_executed", seed_offset=0)
        assert "price" in evt
        assert evt["price"] > 0

    def test_generate_feed_event_consensus_reached_has_agreement_rate(self):
        evt = _generate_feed_event(event_type="consensus_reached", seed_offset=0)
        assert "agreement_rate" in evt
        assert 0.0 <= evt["agreement_rate"] <= 1.0

    def test_generate_feed_event_reputation_updated_has_scores(self):
        evt = _generate_feed_event(event_type="reputation_updated", seed_offset=0)
        assert "old_score" in evt
        assert "new_score" in evt

    def test_push_feed_event_adds_to_buffer(self):
        before = get_live_feed(last=200)["total_buffered"]
        _push_feed_event({"event": "test_event", "agent_id": "test", "timestamp": 0})
        after = get_live_feed(last=200)["total_buffered"]
        assert after >= before  # may be equal if buffer is at maxlen


# ── Unit Tests: run_scenario ──────────────────────────────────────────────────

class TestScenarioOrchestrator:

    def test_run_scenario_bull_run(self):
        result = run_scenario("bull_run", seed=32)
        assert result["scenario"] == "bull_run"
        assert result["ticks"] == 20

    def test_run_scenario_bear_crash(self):
        result = run_scenario("bear_crash", seed=32)
        assert result["scenario"] == "bear_crash"

    def test_run_scenario_volatile_chop(self):
        result = run_scenario("volatile_chop", seed=32)
        assert result["scenario"] == "volatile_chop"

    def test_run_scenario_stable_trend(self):
        result = run_scenario("stable_trend", seed=32)
        assert result["scenario"] == "stable_trend"

    def test_run_scenario_invalid_raises_value_error(self):
        with pytest.raises(ValueError):
            run_scenario("moon_rocket", seed=32)

    def test_run_scenario_returns_20_ticks(self):
        result = run_scenario("bull_run", seed=32)
        assert len(result["timeline"]) == 20

    def test_run_scenario_bear_crash_fires_circuit_breaker(self):
        result = run_scenario("bear_crash", seed=32)
        assert result["circuit_breaker_fired"] is True

    def test_run_scenario_bear_crash_circuit_breaker_has_tick(self):
        result = run_scenario("bear_crash", seed=32)
        assert result["circuit_breaker_tick"] is not None
        assert 0 <= result["circuit_breaker_tick"] < 20

    def test_run_scenario_bull_run_does_not_fire_circuit_breaker(self):
        result = run_scenario("bull_run", seed=32)
        assert result["circuit_breaker_fired"] is False

    def test_run_scenario_stable_trend_does_not_fire_circuit_breaker(self):
        result = run_scenario("stable_trend", seed=32)
        assert result["circuit_breaker_fired"] is False

    def test_run_scenario_timeline_has_tick_numbers(self):
        result = run_scenario("bull_run", seed=32)
        ticks = [t["tick"] for t in result["timeline"]]
        assert ticks == list(range(1, 21))

    def test_run_scenario_timeline_has_agent_votes(self):
        result = run_scenario("bull_run", seed=32)
        for tick in result["timeline"]:
            assert "agent_votes" in tick
            assert len(tick["agent_votes"]) > 0

    def test_run_scenario_timeline_has_consensus(self):
        result = run_scenario("bull_run", seed=32)
        for tick in result["timeline"]:
            assert "consensus" in tick
            assert "action" in tick["consensus"]
            assert "agreement_rate" in tick["consensus"]

    def test_run_scenario_timeline_has_trade(self):
        result = run_scenario("bull_run", seed=32)
        for tick in result["timeline"]:
            assert "trade" in tick
            assert "executed" in tick["trade"]
            assert "pnl" in tick["trade"]

    def test_run_scenario_timeline_has_circuit_breaker_field(self):
        result = run_scenario("bear_crash", seed=32)
        for tick in result["timeline"]:
            assert "circuit_breaker" in tick
            assert "active" in tick["circuit_breaker"]

    def test_run_scenario_circuit_breaker_active_after_firing(self):
        result = run_scenario("bear_crash", seed=32)
        fired_tick = result["circuit_breaker_tick"]
        for tick in result["timeline"]:
            if tick["tick"] > fired_tick + 1:
                assert tick["circuit_breaker"]["active"] is True

    def test_run_scenario_has_agent_summary(self):
        result = run_scenario("bull_run", seed=32)
        assert "agent_summary" in result
        assert len(result["agent_summary"]) > 0

    def test_run_scenario_agent_summary_has_fields(self):
        result = run_scenario("bull_run", seed=32)
        for ag in result["agent_summary"]:
            assert "agent_id" in ag
            assert "final_reputation" in ag
            assert "total_pnl" in ag
            assert "win_rate" in ag
            assert "total_trades" in ag

    def test_run_scenario_is_deterministic(self):
        r1 = run_scenario("bull_run", seed=42)
        r2 = run_scenario("bull_run", seed=42)
        assert r1["cumulative_pnl"] == r2["cumulative_pnl"]
        assert r1["final_price"] == r2["final_price"]

    def test_run_scenario_different_seeds_different_results(self):
        r1 = run_scenario("bull_run", seed=1)
        r2 = run_scenario("bull_run", seed=99)
        # Different seeds should produce different results
        assert r1["cumulative_pnl"] != r2["cumulative_pnl"] or r1["final_price"] != r2["final_price"]

    def test_valid_scenarios_set(self):
        assert "bull_run" in _VALID_SCENARIOS
        assert "bear_crash" in _VALID_SCENARIOS
        assert "volatile_chop" in _VALID_SCENARIOS
        assert "stable_trend" in _VALID_SCENARIOS

    def test_scenario_configs_have_descriptions(self):
        for scenario, cfg in _SCENARIO_CONFIGS.items():
            assert "description" in cfg
            assert len(cfg["description"]) > 0


# ── Unit Tests: build_agents_leaderboard ──────────────────────────────────────

class TestAgentsLeaderboard:

    def test_build_agents_leaderboard_returns_dict(self):
        result = build_agents_leaderboard(limit=5)
        assert isinstance(result, dict)

    def test_build_agents_leaderboard_has_leaderboard_key(self):
        result = build_agents_leaderboard(limit=5)
        assert "leaderboard" in result

    def test_build_agents_leaderboard_respects_limit(self):
        result = build_agents_leaderboard(limit=3)
        assert len(result["leaderboard"]) <= 3

    def test_build_agents_leaderboard_max_20(self):
        result = build_agents_leaderboard(limit=100)
        assert len(result["leaderboard"]) <= 20

    def test_build_agents_leaderboard_has_sort_by(self):
        result = build_agents_leaderboard()
        assert "sort_by" in result
        assert "composite" in result["sort_by"].lower()

    def test_build_agents_leaderboard_entries_have_rank(self):
        result = build_agents_leaderboard(limit=5)
        for i, e in enumerate(result["leaderboard"], start=1):
            assert e["rank"] == i

    def test_build_agents_leaderboard_entries_have_win_rate(self):
        result = build_agents_leaderboard(limit=5)
        for e in result["leaderboard"]:
            assert "win_rate" in e
            assert 0.0 <= e["win_rate"] <= 1.0

    def test_build_agents_leaderboard_entries_have_total_trades(self):
        result = build_agents_leaderboard(limit=5)
        for e in result["leaderboard"]:
            assert "total_trades" in e
            assert isinstance(e["total_trades"], int)

    def test_build_agents_leaderboard_entries_have_avg_position_size(self):
        result = build_agents_leaderboard(limit=5)
        for e in result["leaderboard"]:
            assert "avg_position_size" in e

    def test_build_agents_leaderboard_entries_have_sharpe_ratio(self):
        result = build_agents_leaderboard(limit=5)
        for e in result["leaderboard"]:
            assert "sharpe_ratio" in e

    def test_build_agents_leaderboard_entries_have_composite_score(self):
        result = build_agents_leaderboard(limit=5)
        for e in result["leaderboard"]:
            assert "composite_score" in e

    def test_build_agents_leaderboard_entries_have_rank_change(self):
        result = build_agents_leaderboard(limit=5)
        for e in result["leaderboard"]:
            assert "rank_change" in e

    def test_build_agents_leaderboard_sorted_by_composite(self):
        result = build_agents_leaderboard(limit=5)
        scores = [e["composite_score"] for e in result["leaderboard"]]
        assert scores == sorted(scores, reverse=True)

    def test_build_agents_leaderboard_has_generated_at(self):
        result = build_agents_leaderboard()
        assert "generated_at" in result


# ── Unit Tests: get_demo_status ───────────────────────────────────────────────

class TestDemoStatus:

    def test_get_demo_status_returns_dict(self):
        result = get_demo_status()
        assert isinstance(result, dict)

    def test_get_demo_status_ok(self):
        result = get_demo_status()
        assert result["status"] == "ok"

    def test_get_demo_status_has_version(self):
        result = get_demo_status()
        assert "version" in result
        assert result["version"] == SERVER_VERSION

    def test_get_demo_status_has_uptime(self):
        result = get_demo_status()
        assert "uptime_s" in result
        assert result["uptime_s"] >= 0

    def test_get_demo_status_has_features(self):
        result = get_demo_status()
        assert "features" in result

    def test_get_demo_status_live_feed_feature(self):
        result = get_demo_status()
        assert "live_feed" in result["features"]
        assert result["features"]["live_feed"]["enabled"] is True

    def test_get_demo_status_scenario_orchestrator_feature(self):
        result = get_demo_status()
        assert "scenario_orchestrator" in result["features"]
        assert result["features"]["scenario_orchestrator"]["enabled"] is True

    def test_get_demo_status_agents_leaderboard_feature(self):
        result = get_demo_status()
        assert "agents_leaderboard" in result["features"]

    def test_get_demo_status_websocket_feature(self):
        result = get_demo_status()
        assert "websocket_feed" in result["features"]

    def test_get_demo_status_has_test_count(self):
        result = get_demo_status()
        assert "test_count" in result
        assert result["test_count"] >= 150


# ── HTTP Integration Tests ────────────────────────────────────────────────────

class TestHTTPLiveFeed:

    def test_get_live_feed_http(self, server):
        result = _get(f"{server}/demo/live/feed")
        assert result["count"] >= 0

    def test_get_live_feed_last_param(self, server):
        result = _get(f"{server}/demo/live/feed?last=5")
        assert result["count"] <= 5

    def test_get_live_feed_returns_200(self, server):
        import urllib.request
        with urllib.request.urlopen(f"{server}/demo/live/feed", timeout=5) as resp:
            assert resp.status == 200


class TestHTTPScenario:

    def test_post_scenario_bull_run(self, server):
        result = _post(f"{server}/demo/scenario/run", {"scenario": "bull_run"})
        assert result["scenario"] == "bull_run"
        assert result["ticks"] == 20

    def test_post_scenario_bear_crash_fires_circuit_breaker(self, server):
        result = _post(f"{server}/demo/scenario/run", {"scenario": "bear_crash"})
        assert result["circuit_breaker_fired"] is True

    def test_post_scenario_volatile_chop(self, server):
        result = _post(f"{server}/demo/scenario/run", {"scenario": "volatile_chop"})
        assert result["scenario"] == "volatile_chop"

    def test_post_scenario_stable_trend(self, server):
        result = _post(f"{server}/demo/scenario/run", {"scenario": "stable_trend"})
        assert result["circuit_breaker_fired"] is False

    def test_post_scenario_missing_field_returns_400(self, server):
        code, body = _post_expecting_error(f"{server}/demo/scenario/run", {})
        assert code == 400
        assert "error" in body

    def test_post_scenario_invalid_scenario_returns_400(self, server):
        code, body = _post_expecting_error(f"{server}/demo/scenario/run", {"scenario": "moon_shot"})
        assert code == 400
        assert "valid_scenarios" in body

    def test_post_scenario_timeline_length(self, server):
        result = _post(f"{server}/demo/scenario/run", {"scenario": "bull_run"})
        assert len(result["timeline"]) == 20

    def test_post_scenario_custom_seed(self, server):
        r1 = _post(f"{server}/demo/scenario/run", {"scenario": "bull_run", "seed": 99})
        r2 = _post(f"{server}/demo/scenario/run", {"scenario": "bull_run", "seed": 99})
        # Same seed = same result
        assert r1["cumulative_pnl"] == r2["cumulative_pnl"]


class TestHTTPLeaderboard:

    def test_get_agents_leaderboard(self, server):
        result = _get(f"{server}/demo/agents/leaderboard")
        assert "leaderboard" in result
        assert len(result["leaderboard"]) > 0

    def test_get_agents_leaderboard_has_composite_score(self, server):
        result = _get(f"{server}/demo/agents/leaderboard")
        for e in result["leaderboard"]:
            assert "composite_score" in e

    def test_get_agents_leaderboard_has_rank_change(self, server):
        result = _get(f"{server}/demo/agents/leaderboard")
        for e in result["leaderboard"]:
            assert "rank_change" in e

    def test_get_agents_leaderboard_limit_param(self, server):
        result = _get(f"{server}/demo/agents/leaderboard?limit=3")
        assert len(result["leaderboard"]) <= 3


class TestHTTPHealth:

    def test_get_health_endpoint(self, server):
        result = _get(f"{server}/health")
        assert result["status"] == "ok"

    def test_get_health_has_uptime(self, server):
        result = _get(f"{server}/health")
        assert "uptime_s" in result

    def test_get_health_has_test_count(self, server):
        result = _get(f"{server}/health")
        assert "test_count" in result

    def test_get_health_has_version(self, server):
        result = _get(f"{server}/health")
        assert "version" in result

    def test_get_demo_health_endpoint(self, server):
        result = _get(f"{server}/demo/health")
        assert result["status"] == "ok"

    def test_get_demo_status_endpoint(self, server):
        result = _get(f"{server}/demo/status")
        assert result["status"] == "ok"
        assert "features" in result

    def test_get_demo_status_has_scenario_list(self, server):
        result = _get(f"{server}/demo/status")
        scenarios = result["features"]["scenario_orchestrator"]["scenarios"]
        assert "bear_crash" in scenarios
        assert "bull_run" in scenarios
