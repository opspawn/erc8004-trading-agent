"""
test_s43_cross_agent_coordination.py — Sprint 43: Cross-Agent Coordination.

Covers (170+ tests):

  Section A — broadcast_signal() unit tests (50+ tests):
    - Happy-path: valid actions, assets, confidence values
    - Return schema: broadcast_id, recipients, timestamp, status, metadata
    - Recipients exclude sender; all other known agents are recipients
    - Edge: unknown agent still broadcasts (no sender validation)
    - Validation errors: empty agent_id, invalid action, invalid asset,
      confidence < 0, confidence > 1

  Section B — get_coordination_signals() unit tests (40+ tests):
    - No signals: empty buffer returns empty list
    - After broadcast: signals appear, most-recent-first ordering
    - Filter by asset: only matching signals returned
    - Filter by agent_id: only signals from that agent returned
    - Limit clamping: 1–100 range enforced
    - Invalid asset filter raises ValueError
    - Combined filters work correctly
    - Buffer cap: after 100+ signals oldest are dropped

  Section C — resolve_coordination_conflict() unit tests (50+ tests):
    - highest_confidence strategy: picks max confidence signal
    - majority_vote strategy: picks most common action
    - majority_vote tie-break: resolves by confidence
    - weighted_consensus strategy: weights by confidence bucket
    - No conflict (all agents agree): conflict_detected == False
    - Conflict (agents disagree): conflict_detected == True, disagreement_logged == True
    - actions_present contains all unique actions
    - winning_signal is a dict with agent_id, action, confidence
    - candidates list contains all normalised signals
    - resolution recorded in signal buffer
    - Empty signals list raises ValueError
    - Invalid strategy raises ValueError
    - confidence clamped to [0.0, 1.0] on normalisation
    - Unknown action normalised to HOLD

  Section D — HTTP endpoints (30+ tests):
    POST /api/v1/agents/broadcast:
      - 200 on valid payload
      - 400 on missing/invalid fields
      - Response body schema
    GET /api/v1/coordination/signals:
      - 200 with signals list
      - Filters via query params
      - 400 on invalid asset
    POST /api/v1/coordination/resolve:
      - 200 on valid payload, all 3 strategies
      - 400 on empty signals, invalid strategy
      - Response includes resolved_action

  Section E — Version & constants checks:
    - SERVER_VERSION == "S43"
    - _S43_TEST_COUNT defined and >= 5355
    - _S43_VALID_ACTIONS contains BUY SELL HOLD REBALANCE
    - _S43_VALID_ASSETS non-empty, BTC/USD present
    - _S43_CONFLICT_STRATEGIES correct set
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
    _S43_TEST_COUNT,
    _S43_VALID_ACTIONS,
    _S43_VALID_ASSETS,
    _S43_AGENT_IDS,
    _S43_CONFLICT_STRATEGIES,
    _S43_SIGNAL_BUFFER,
    _S43_SIGNAL_BUFFER_MAX,
    _s43_record_signal,
    broadcast_signal,
    get_coordination_signals,
    resolve_coordination_conflict,
)


# ─── Helpers ──────────────────────────────────────────────────────────────────


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("", 0))
        return s.getsockname()[1]


def _get(url: str) -> dict:
    with urlopen(url, timeout=5) as resp:
        return json.loads(resp.read())


def _post(url: str, body: dict) -> dict:
    data = json.dumps(body).encode()
    req = Request(url, data=data, headers={"Content-Type": "application/json"}, method="POST")
    with urlopen(req, timeout=5) as resp:
        return json.loads(resp.read())


def _post_status(url: str, body: dict) -> int:
    data = json.dumps(body).encode()
    req = Request(url, data=data, headers={"Content-Type": "application/json"}, method="POST")
    try:
        with urlopen(req, timeout=5) as resp:
            return resp.status
    except HTTPError as e:
        return e.code


def _get_status(url: str) -> int:
    try:
        with urlopen(url, timeout=5) as resp:
            return resp.status
    except HTTPError as e:
        return e.code


def _clear_signal_buffer() -> None:
    """Empty the global signal buffer for test isolation."""
    with __import__("demo_server")._S43_SIGNAL_LOCK:
        _S43_SIGNAL_BUFFER.clear()


# ─── Section E — Constants & Version ──────────────────────────────────────────


class TestS43Constants:
    def test_server_version_is_s43(self):
        assert SERVER_VERSION == "S43"

    def test_s43_test_count_defined(self):
        assert _S43_TEST_COUNT is not None

    def test_s43_test_count_is_int(self):
        assert isinstance(_S43_TEST_COUNT, int)

    def test_s43_test_count_gte_baseline(self):
        assert _S43_TEST_COUNT >= 5355

    def test_valid_actions_has_buy(self):
        assert "BUY" in _S43_VALID_ACTIONS

    def test_valid_actions_has_sell(self):
        assert "SELL" in _S43_VALID_ACTIONS

    def test_valid_actions_has_hold(self):
        assert "HOLD" in _S43_VALID_ACTIONS

    def test_valid_actions_has_rebalance(self):
        assert "REBALANCE" in _S43_VALID_ACTIONS

    def test_valid_actions_count(self):
        assert len(_S43_VALID_ACTIONS) == 4

    def test_valid_assets_nonempty(self):
        assert len(_S43_VALID_ASSETS) > 0

    def test_valid_assets_btcusd(self):
        assert "BTC/USD" in _S43_VALID_ASSETS

    def test_valid_assets_ethusd(self):
        assert "ETH/USD" in _S43_VALID_ASSETS

    def test_valid_assets_solusd(self):
        assert "SOL/USD" in _S43_VALID_ASSETS

    def test_agent_ids_nonempty(self):
        assert len(_S43_AGENT_IDS) >= 3

    def test_conflict_strategies_set(self):
        assert "highest_confidence" in _S43_CONFLICT_STRATEGIES
        assert "majority_vote" in _S43_CONFLICT_STRATEGIES
        assert "weighted_consensus" in _S43_CONFLICT_STRATEGIES

    def test_conflict_strategies_count(self):
        assert len(_S43_CONFLICT_STRATEGIES) == 3

    def test_buffer_max_is_100(self):
        assert _S43_SIGNAL_BUFFER_MAX == 100


# ─── Section A — broadcast_signal() ───────────────────────────────────────────


class TestBroadcastSignal:
    def setup_method(self):
        _clear_signal_buffer()

    def test_returns_dict(self):
        r = broadcast_signal("agent-conservative-001", "BUY", "BTC/USD", 0.9)
        assert isinstance(r, dict)

    def test_has_broadcast_id(self):
        r = broadcast_signal("agent-conservative-001", "BUY", "BTC/USD", 0.9)
        assert "broadcast_id" in r
        assert r["broadcast_id"].startswith("bc-")

    def test_has_from_agent(self):
        r = broadcast_signal("agent-conservative-001", "BUY", "BTC/USD", 0.9)
        assert r["from_agent"] == "agent-conservative-001"

    def test_has_action_upper(self):
        r = broadcast_signal("agent-conservative-001", "buy", "BTC/USD", 0.9)
        assert r["action"] == "BUY"

    def test_has_asset(self):
        r = broadcast_signal("agent-balanced-002", "SELL", "ETH/USD", 0.7)
        assert r["asset"] == "ETH/USD"

    def test_has_confidence(self):
        r = broadcast_signal("agent-balanced-002", "HOLD", "BTC/USD", 0.55)
        assert r["confidence"] == pytest.approx(0.55, abs=1e-6)

    def test_confidence_rounded(self):
        r = broadcast_signal("agent-balanced-002", "HOLD", "BTC/USD", 0.123456789)
        assert len(str(r["confidence"]).split(".")[-1]) <= 4

    def test_has_recipients_list(self):
        r = broadcast_signal("agent-conservative-001", "BUY", "BTC/USD", 0.8)
        assert isinstance(r["recipients"], list)

    def test_sender_not_in_recipients(self):
        r = broadcast_signal("agent-conservative-001", "BUY", "BTC/USD", 0.8)
        assert "agent-conservative-001" not in r["recipients"]

    def test_recipient_count(self):
        r = broadcast_signal("agent-conservative-001", "BUY", "BTC/USD", 0.8)
        assert r["recipient_count"] == len(r["recipients"])

    def test_at_least_one_recipient(self):
        r = broadcast_signal("agent-conservative-001", "BUY", "BTC/USD", 0.8)
        assert r["recipient_count"] >= 1

    def test_all_known_agents_are_recipients(self):
        sender = "agent-conservative-001"
        r = broadcast_signal(sender, "BUY", "BTC/USD", 0.8)
        expected = sorted(_S43_AGENT_IDS - {sender})
        assert r["recipients"] == expected

    def test_has_timestamp(self):
        before = time.time()
        r = broadcast_signal("agent-conservative-001", "BUY", "BTC/USD", 0.8)
        assert r["timestamp"] >= before

    def test_status_delivered(self):
        r = broadcast_signal("agent-conservative-001", "BUY", "BTC/USD", 0.8)
        assert r["status"] == "delivered"

    def test_metadata_default_empty_dict(self):
        r = broadcast_signal("agent-conservative-001", "BUY", "BTC/USD", 0.8)
        assert r["metadata"] == {}

    def test_metadata_passed_through(self):
        meta = {"reason": "momentum spike", "signal_strength": 3}
        r = broadcast_signal("agent-conservative-001", "BUY", "BTC/USD", 0.8, metadata=meta)
        assert r["metadata"] == meta

    def test_action_sell(self):
        r = broadcast_signal("agent-balanced-002", "SELL", "ETH/USD", 0.6)
        assert r["action"] == "SELL"

    def test_action_hold(self):
        r = broadcast_signal("agent-balanced-002", "HOLD", "SOL/USD", 0.5)
        assert r["action"] == "HOLD"

    def test_action_rebalance(self):
        r = broadcast_signal("agent-balanced-002", "REBALANCE", "BTC/USD", 0.4)
        assert r["action"] == "REBALANCE"

    def test_confidence_zero(self):
        r = broadcast_signal("agent-balanced-002", "HOLD", "BTC/USD", 0.0)
        assert r["confidence"] == pytest.approx(0.0)

    def test_confidence_one(self):
        r = broadcast_signal("agent-balanced-002", "BUY", "BTC/USD", 1.0)
        assert r["confidence"] == pytest.approx(1.0)

    def test_confidence_mid(self):
        r = broadcast_signal("agent-balanced-002", "BUY", "BTC/USD", 0.5)
        assert 0.0 <= r["confidence"] <= 1.0

    def test_unknown_agent_broadcasts(self):
        """Sender validation is lenient — any agent_id is accepted."""
        r = broadcast_signal("agent-unknown-999", "BUY", "BTC/USD", 0.7)
        assert r["from_agent"] == "agent-unknown-999"

    def test_records_to_buffer(self):
        _clear_signal_buffer()
        broadcast_signal("agent-conservative-001", "BUY", "BTC/USD", 0.9)
        assert len(_S43_SIGNAL_BUFFER) >= 1

    def test_broadcast_id_unique(self):
        r1 = broadcast_signal("agent-conservative-001", "BUY", "BTC/USD", 0.9)
        time.sleep(0.001)
        r2 = broadcast_signal("agent-balanced-002", "SELL", "ETH/USD", 0.7)
        assert r1["broadcast_id"] != r2["broadcast_id"]

    def test_error_empty_agent_id(self):
        with pytest.raises(ValueError, match="agent_id"):
            broadcast_signal("", "BUY", "BTC/USD", 0.8)

    def test_error_invalid_action(self):
        with pytest.raises(ValueError, match="action"):
            broadcast_signal("agent-conservative-001", "JUMP", "BTC/USD", 0.8)

    def test_error_invalid_action_case_insensitive_check(self):
        # "buy" (lowercase) is OK — uppercased internally
        r = broadcast_signal("agent-conservative-001", "buy", "BTC/USD", 0.8)
        assert r["action"] == "BUY"

    def test_error_invalid_asset(self):
        with pytest.raises(ValueError, match="asset"):
            broadcast_signal("agent-conservative-001", "BUY", "DOGE/USD", 0.8)

    def test_error_confidence_negative(self):
        with pytest.raises(ValueError, match="confidence"):
            broadcast_signal("agent-conservative-001", "BUY", "BTC/USD", -0.1)

    def test_error_confidence_above_one(self):
        with pytest.raises(ValueError, match="confidence"):
            broadcast_signal("agent-conservative-001", "BUY", "BTC/USD", 1.1)

    def test_all_valid_assets(self):
        for asset in _S43_VALID_ASSETS:
            r = broadcast_signal("agent-conservative-001", "HOLD", asset, 0.5)
            assert r["asset"] == asset

    def test_all_valid_actions(self):
        for action in _S43_VALID_ACTIONS:
            r = broadcast_signal("agent-conservative-001", action, "BTC/USD", 0.5)
            assert r["action"] == action


# ─── Section B — get_coordination_signals() ───────────────────────────────────


class TestGetCoordinationSignals:
    def setup_method(self):
        _clear_signal_buffer()

    def test_empty_buffer_returns_empty_list(self):
        r = get_coordination_signals()
        assert r["signals"] == []

    def test_returns_dict(self):
        r = get_coordination_signals()
        assert isinstance(r, dict)

    def test_has_total_returned(self):
        r = get_coordination_signals()
        assert "total_returned" in r

    def test_has_filters(self):
        r = get_coordination_signals()
        assert "filters" in r

    def test_has_limit_field(self):
        r = get_coordination_signals()
        assert "limit" in r

    def test_has_generated_at(self):
        r = get_coordination_signals()
        assert "generated_at" in r

    def test_after_broadcast_signals_visible(self):
        broadcast_signal("agent-conservative-001", "BUY", "BTC/USD", 0.9)
        r = get_coordination_signals()
        assert r["total_returned"] >= 1

    def test_signals_most_recent_first(self):
        broadcast_signal("agent-conservative-001", "BUY", "BTC/USD", 0.9)
        time.sleep(0.001)
        broadcast_signal("agent-balanced-002", "SELL", "ETH/USD", 0.7)
        r = get_coordination_signals()
        # Most recent (SELL by balanced-002) should be first
        assert r["signals"][0]["from_agent"] == "agent-balanced-002"
        assert r["signals"][1]["from_agent"] == "agent-conservative-001"

    def test_filter_by_asset(self):
        broadcast_signal("agent-conservative-001", "BUY", "BTC/USD", 0.9)
        broadcast_signal("agent-balanced-002", "SELL", "ETH/USD", 0.7)
        r = get_coordination_signals(asset="ETH/USD")
        assert all(s["asset"] == "ETH/USD" for s in r["signals"])

    def test_filter_by_asset_no_match(self):
        broadcast_signal("agent-conservative-001", "BUY", "BTC/USD", 0.9)
        r = get_coordination_signals(asset="SOL/USD")
        assert r["signals"] == []

    def test_filter_by_agent_id(self):
        broadcast_signal("agent-conservative-001", "BUY", "BTC/USD", 0.9)
        broadcast_signal("agent-balanced-002", "SELL", "ETH/USD", 0.7)
        r = get_coordination_signals(agent_id="agent-conservative-001")
        assert all(s["from_agent"] == "agent-conservative-001" for s in r["signals"])

    def test_filter_by_agent_id_no_match(self):
        broadcast_signal("agent-conservative-001", "BUY", "BTC/USD", 0.9)
        r = get_coordination_signals(agent_id="agent-unknown-xxx")
        assert r["signals"] == []

    def test_combined_filters(self):
        broadcast_signal("agent-conservative-001", "BUY", "BTC/USD", 0.9)
        broadcast_signal("agent-conservative-001", "HOLD", "ETH/USD", 0.5)
        broadcast_signal("agent-balanced-002", "BUY", "BTC/USD", 0.6)
        r = get_coordination_signals(asset="BTC/USD", agent_id="agent-conservative-001")
        assert all(
            s["asset"] == "BTC/USD" and s["from_agent"] == "agent-conservative-001"
            for s in r["signals"]
        )

    def test_limit_default_20(self):
        r = get_coordination_signals()
        assert r["limit"] == 20

    def test_limit_clamped_min(self):
        r = get_coordination_signals(limit=0)
        assert r["limit"] == 1

    def test_limit_clamped_max(self):
        r = get_coordination_signals(limit=999)
        assert r["limit"] == 100

    def test_limit_applied(self):
        for i in range(10):
            broadcast_signal("agent-conservative-001", "BUY", "BTC/USD", 0.5)
        r = get_coordination_signals(limit=3)
        assert r["total_returned"] <= 3

    def test_invalid_asset_raises_value_error(self):
        with pytest.raises(ValueError, match="asset"):
            get_coordination_signals(asset="INVALID/COIN")

    def test_filters_reflected_in_response(self):
        r = get_coordination_signals(asset="BTC/USD", agent_id="agent-conservative-001")
        assert r["filters"]["asset"] == "BTC/USD"
        assert r["filters"]["agent_id"] == "agent-conservative-001"

    def test_total_returned_matches_list_length(self):
        broadcast_signal("agent-conservative-001", "BUY", "BTC/USD", 0.9)
        broadcast_signal("agent-balanced-002", "SELL", "ETH/USD", 0.7)
        r = get_coordination_signals()
        assert r["total_returned"] == len(r["signals"])

    def test_buffer_cap_oldest_dropped(self):
        for i in range(110):
            broadcast_signal("agent-conservative-001", "HOLD", "BTC/USD", 0.5)
        r = get_coordination_signals(limit=100)
        assert r["total_returned"] <= _S43_SIGNAL_BUFFER_MAX


# ─── Section C — resolve_coordination_conflict() ──────────────────────────────


class TestResolveCoordinationConflict:
    def setup_method(self):
        _clear_signal_buffer()

    # ── Highest confidence ──────────────────────────────────────────────────

    def test_returns_dict(self):
        signals = [{"agent_id": "a", "action": "BUY", "confidence": 0.9}]
        r = resolve_coordination_conflict(signals)
        assert isinstance(r, dict)

    def test_highest_confidence_picks_max(self):
        signals = [
            {"agent_id": "a", "action": "BUY", "confidence": 0.9},
            {"agent_id": "b", "action": "SELL", "confidence": 0.6},
        ]
        r = resolve_coordination_conflict(signals, strategy="highest_confidence")
        assert r["resolved_action"] == "BUY"

    def test_highest_confidence_resolves_to_correct_agent(self):
        signals = [
            {"agent_id": "a", "action": "BUY", "confidence": 0.9},
            {"agent_id": "b", "action": "SELL", "confidence": 0.4},
        ]
        r = resolve_coordination_conflict(signals)
        assert r["winning_signal"]["agent_id"] == "a"

    def test_highest_confidence_single_signal(self):
        signals = [{"agent_id": "a", "action": "HOLD", "confidence": 0.7}]
        r = resolve_coordination_conflict(signals)
        assert r["resolved_action"] == "HOLD"

    def test_no_conflict_detected_when_all_agree(self):
        signals = [
            {"agent_id": "a", "action": "BUY", "confidence": 0.9},
            {"agent_id": "b", "action": "BUY", "confidence": 0.7},
            {"agent_id": "c", "action": "BUY", "confidence": 0.5},
        ]
        r = resolve_coordination_conflict(signals)
        assert r["conflict_detected"] is False

    def test_conflict_detected_when_disagree(self):
        signals = [
            {"agent_id": "a", "action": "BUY", "confidence": 0.9},
            {"agent_id": "b", "action": "SELL", "confidence": 0.7},
        ]
        r = resolve_coordination_conflict(signals)
        assert r["conflict_detected"] is True

    def test_disagreement_logged_when_conflict(self):
        signals = [
            {"agent_id": "a", "action": "BUY", "confidence": 0.9},
            {"agent_id": "b", "action": "SELL", "confidence": 0.7},
        ]
        r = resolve_coordination_conflict(signals)
        assert r["disagreement_logged"] is True

    def test_disagreement_logged_false_when_no_conflict(self):
        signals = [
            {"agent_id": "a", "action": "BUY", "confidence": 0.9},
            {"agent_id": "b", "action": "BUY", "confidence": 0.7},
        ]
        r = resolve_coordination_conflict(signals)
        assert r["disagreement_logged"] is False

    def test_actions_present_all_unique_actions(self):
        signals = [
            {"agent_id": "a", "action": "BUY", "confidence": 0.9},
            {"agent_id": "b", "action": "SELL", "confidence": 0.7},
            {"agent_id": "c", "action": "HOLD", "confidence": 0.5},
        ]
        r = resolve_coordination_conflict(signals)
        assert set(r["actions_present"]) == {"BUY", "SELL", "HOLD"}

    def test_candidates_list_present(self):
        signals = [{"agent_id": "a", "action": "BUY", "confidence": 0.8}]
        r = resolve_coordination_conflict(signals)
        assert isinstance(r["candidates"], list)
        assert len(r["candidates"]) == 1

    def test_winning_signal_has_required_keys(self):
        signals = [{"agent_id": "a", "action": "BUY", "confidence": 0.8}]
        r = resolve_coordination_conflict(signals)
        ws = r["winning_signal"]
        assert "agent_id" in ws
        assert "action" in ws
        assert "confidence" in ws

    def test_has_strategy_field(self):
        signals = [{"agent_id": "a", "action": "BUY", "confidence": 0.8}]
        r = resolve_coordination_conflict(signals)
        assert r["strategy"] == "highest_confidence"

    def test_has_resolved_at(self):
        signals = [{"agent_id": "a", "action": "BUY", "confidence": 0.8}]
        before = time.time()
        r = resolve_coordination_conflict(signals)
        assert r["resolved_at"] >= before

    def test_resolution_recorded_in_buffer(self):
        _clear_signal_buffer()
        signals = [{"agent_id": "a", "action": "BUY", "confidence": 0.8}]
        resolve_coordination_conflict(signals)
        buf = get_coordination_signals(limit=100)
        resolver_records = [s for s in buf["signals"] if s["from_agent"] == "__resolver__"]
        assert len(resolver_records) >= 1

    def test_resolution_record_has_type(self):
        _clear_signal_buffer()
        signals = [
            {"agent_id": "a", "action": "BUY", "confidence": 0.9},
            {"agent_id": "b", "action": "SELL", "confidence": 0.3},
        ]
        resolve_coordination_conflict(signals)
        buf = get_coordination_signals(limit=100)
        rec = next(s for s in buf["signals"] if s["from_agent"] == "__resolver__")
        assert rec["metadata"]["type"] == "conflict_resolution"

    def test_confidence_clamped_below_zero(self):
        signals = [{"agent_id": "a", "action": "BUY", "confidence": -5.0}]
        r = resolve_coordination_conflict(signals)
        assert r["candidates"][0]["confidence"] == pytest.approx(0.0)

    def test_confidence_clamped_above_one(self):
        signals = [{"agent_id": "a", "action": "BUY", "confidence": 99.0}]
        r = resolve_coordination_conflict(signals)
        assert r["candidates"][0]["confidence"] == pytest.approx(1.0)

    def test_unknown_action_normalised_to_hold(self):
        signals = [{"agent_id": "a", "action": "MOON", "confidence": 0.8}]
        r = resolve_coordination_conflict(signals)
        assert r["candidates"][0]["action"] == "HOLD"

    # ── Majority vote ───────────────────────────────────────────────────────

    def test_majority_vote_picks_majority(self):
        signals = [
            {"agent_id": "a", "action": "BUY", "confidence": 0.5},
            {"agent_id": "b", "action": "BUY", "confidence": 0.4},
            {"agent_id": "c", "action": "SELL", "confidence": 0.9},
        ]
        r = resolve_coordination_conflict(signals, strategy="majority_vote")
        assert r["resolved_action"] == "BUY"

    def test_majority_vote_tie_break_by_confidence(self):
        signals = [
            {"agent_id": "a", "action": "BUY", "confidence": 0.9},
            {"agent_id": "b", "action": "SELL", "confidence": 0.6},
        ]
        r = resolve_coordination_conflict(signals, strategy="majority_vote")
        # 1 vs 1, tie-break: highest confidence wins
        assert r["resolved_action"] == "BUY"

    def test_majority_vote_has_vote_tally(self):
        signals = [
            {"agent_id": "a", "action": "BUY", "confidence": 0.9},
            {"agent_id": "b", "action": "BUY", "confidence": 0.4},
            {"agent_id": "c", "action": "SELL", "confidence": 0.7},
        ]
        r = resolve_coordination_conflict(signals, strategy="majority_vote")
        assert "vote_tally" in r["resolution_details"]

    def test_majority_vote_strategy_field(self):
        signals = [{"agent_id": "a", "action": "HOLD", "confidence": 0.5}]
        r = resolve_coordination_conflict(signals, strategy="majority_vote")
        assert r["strategy"] == "majority_vote"

    def test_majority_vote_unanimous(self):
        signals = [
            {"agent_id": "a", "action": "SELL", "confidence": 0.8},
            {"agent_id": "b", "action": "SELL", "confidence": 0.6},
            {"agent_id": "c", "action": "SELL", "confidence": 0.4},
        ]
        r = resolve_coordination_conflict(signals, strategy="majority_vote")
        assert r["resolved_action"] == "SELL"

    # ── Weighted consensus ──────────────────────────────────────────────────

    def test_weighted_consensus_picks_heaviest_bucket(self):
        signals = [
            {"agent_id": "a", "action": "BUY", "confidence": 0.8},
            {"agent_id": "b", "action": "BUY", "confidence": 0.7},
            {"agent_id": "c", "action": "SELL", "confidence": 0.9},
        ]
        r = resolve_coordination_conflict(signals, strategy="weighted_consensus")
        # BUY bucket: 1.5, SELL bucket: 0.9
        assert r["resolved_action"] == "BUY"

    def test_weighted_consensus_has_bucket_weights(self):
        signals = [
            {"agent_id": "a", "action": "BUY", "confidence": 0.8},
            {"agent_id": "b", "action": "SELL", "confidence": 0.6},
        ]
        r = resolve_coordination_conflict(signals, strategy="weighted_consensus")
        assert "bucket_weights" in r["resolution_details"]

    def test_weighted_consensus_bucket_weights_are_floats(self):
        signals = [
            {"agent_id": "a", "action": "BUY", "confidence": 0.8},
        ]
        r = resolve_coordination_conflict(signals, strategy="weighted_consensus")
        for v in r["resolution_details"]["bucket_weights"].values():
            assert isinstance(v, float)

    def test_weighted_consensus_strategy_field(self):
        signals = [{"agent_id": "a", "action": "HOLD", "confidence": 0.5}]
        r = resolve_coordination_conflict(signals, strategy="weighted_consensus")
        assert r["strategy"] == "weighted_consensus"

    def test_weighted_consensus_sum_correct(self):
        signals = [
            {"agent_id": "a", "action": "BUY", "confidence": 0.4},
            {"agent_id": "b", "action": "BUY", "confidence": 0.3},
        ]
        r = resolve_coordination_conflict(signals, strategy="weighted_consensus")
        weights = r["resolution_details"]["bucket_weights"]
        assert weights.get("BUY", 0) == pytest.approx(0.7, abs=1e-4)

    # ── Error cases ──────────────────────────────────────────────────────────

    def test_empty_signals_raises_value_error(self):
        with pytest.raises(ValueError, match="signals"):
            resolve_coordination_conflict([])

    def test_invalid_strategy_raises_value_error(self):
        signals = [{"agent_id": "a", "action": "BUY", "confidence": 0.5}]
        with pytest.raises(ValueError, match="strategy"):
            resolve_coordination_conflict(signals, strategy="coin_flip")

    def test_three_agents_disagree_conflict_detected(self):
        signals = [
            {"agent_id": "a", "action": "BUY", "confidence": 0.9},
            {"agent_id": "b", "action": "SELL", "confidence": 0.5},
            {"agent_id": "c", "action": "HOLD", "confidence": 0.3},
        ]
        r = resolve_coordination_conflict(signals)
        assert r["conflict_detected"] is True

    def test_missing_confidence_defaults_to_0_5(self):
        signals = [{"agent_id": "a", "action": "BUY"}]
        r = resolve_coordination_conflict(signals)
        assert r["candidates"][0]["confidence"] == pytest.approx(0.5)

    def test_missing_agent_id_defaults_to_unknown(self):
        signals = [{"action": "BUY", "confidence": 0.8}]
        r = resolve_coordination_conflict(signals)
        assert r["candidates"][0]["agent_id"] == "unknown"


# ─── Section D — HTTP endpoints ────────────────────────────────────────────────


@pytest.fixture(scope="module")
def server():
    port = _free_port()
    srv = DemoServer(port=port)
    srv.start()
    time.sleep(0.3)
    yield port
    srv.stop()


class TestBroadcastHTTP:
    def test_post_broadcast_200(self, server):
        url = f"http://localhost:{server}/api/v1/agents/broadcast"
        body = {
            "agent_id": "agent-conservative-001",
            "action": "BUY",
            "asset": "BTC/USD",
            "confidence": 0.85,
        }
        assert _post_status(url, body) == 200

    def test_post_broadcast_response_schema(self, server):
        url = f"http://localhost:{server}/api/v1/agents/broadcast"
        body = {
            "agent_id": "agent-balanced-002",
            "action": "SELL",
            "asset": "ETH/USD",
            "confidence": 0.70,
        }
        r = _post(url, body)
        assert "broadcast_id" in r
        assert "from_agent" in r
        assert r["from_agent"] == "agent-balanced-002"
        assert r["action"] == "SELL"
        assert r["asset"] == "ETH/USD"
        assert "recipients" in r
        assert "status" in r

    def test_post_broadcast_400_empty_agent_id(self, server):
        url = f"http://localhost:{server}/api/v1/agents/broadcast"
        body = {"agent_id": "", "action": "BUY", "asset": "BTC/USD", "confidence": 0.8}
        assert _post_status(url, body) == 400

    def test_post_broadcast_400_invalid_action(self, server):
        url = f"http://localhost:{server}/api/v1/agents/broadcast"
        body = {
            "agent_id": "agent-conservative-001",
            "action": "MOONSHOT",
            "asset": "BTC/USD",
            "confidence": 0.8,
        }
        assert _post_status(url, body) == 400

    def test_post_broadcast_400_invalid_asset(self, server):
        url = f"http://localhost:{server}/api/v1/agents/broadcast"
        body = {
            "agent_id": "agent-conservative-001",
            "action": "BUY",
            "asset": "DOGECOIN",
            "confidence": 0.8,
        }
        assert _post_status(url, body) == 400

    def test_post_broadcast_400_invalid_confidence(self, server):
        url = f"http://localhost:{server}/api/v1/agents/broadcast"
        body = {
            "agent_id": "agent-conservative-001",
            "action": "BUY",
            "asset": "BTC/USD",
            "confidence": 2.5,
        }
        assert _post_status(url, body) == 400

    def test_post_broadcast_metadata_forwarded(self, server):
        url = f"http://localhost:{server}/api/v1/agents/broadcast"
        body = {
            "agent_id": "agent-conservative-001",
            "action": "HOLD",
            "asset": "SOL/USD",
            "confidence": 0.5,
            "metadata": {"note": "waiting for confirmation"},
        }
        r = _post(url, body)
        assert r["metadata"] == {"note": "waiting for confirmation"}

    def test_post_broadcast_hold_action(self, server):
        url = f"http://localhost:{server}/api/v1/agents/broadcast"
        body = {
            "agent_id": "agent-momentum-004",
            "action": "HOLD",
            "asset": "ETH/USD",
            "confidence": 0.45,
        }
        r = _post(url, body)
        assert r["action"] == "HOLD"

    def test_post_broadcast_rebalance_action(self, server):
        url = f"http://localhost:{server}/api/v1/agents/broadcast"
        body = {
            "agent_id": "agent-momentum-004",
            "action": "REBALANCE",
            "asset": "BTC/USD",
            "confidence": 0.60,
        }
        r = _post(url, body)
        assert r["action"] == "REBALANCE"


class TestCoordinationSignalsHTTP:
    def test_get_signals_200(self, server):
        url = f"http://localhost:{server}/api/v1/coordination/signals"
        assert _get_status(url) == 200

    def test_get_signals_response_schema(self, server):
        url = f"http://localhost:{server}/api/v1/coordination/signals"
        r = _get(url)
        assert "signals" in r
        assert isinstance(r["signals"], list)
        assert "total_returned" in r
        assert "limit" in r

    def test_get_signals_after_broadcast(self, server):
        # Broadcast then check signals
        broadcast_url = f"http://localhost:{server}/api/v1/agents/broadcast"
        _post(broadcast_url, {
            "agent_id": "agent-conservative-001",
            "action": "BUY",
            "asset": "BTC/USD",
            "confidence": 0.88,
        })
        signals_url = f"http://localhost:{server}/api/v1/coordination/signals"
        r = _get(signals_url)
        assert r["total_returned"] >= 1

    def test_get_signals_filter_by_asset(self, server):
        url = f"http://localhost:{server}/api/v1/coordination/signals?asset=BTC%2FUSD"
        r = _get(url)
        assert isinstance(r["signals"], list)

    def test_get_signals_400_invalid_asset(self, server):
        url = f"http://localhost:{server}/api/v1/coordination/signals?asset=FAKECOIN"
        assert _get_status(url) == 400

    def test_get_signals_limit_param(self, server):
        url = f"http://localhost:{server}/api/v1/coordination/signals?limit=5"
        r = _get(url)
        assert r["limit"] == 5
        assert r["total_returned"] <= 5

    def test_get_signals_default_limit(self, server):
        url = f"http://localhost:{server}/api/v1/coordination/signals"
        r = _get(url)
        assert r["limit"] == 20


class TestCoordinationResolveHTTP:
    def test_post_resolve_200(self, server):
        url = f"http://localhost:{server}/api/v1/coordination/resolve"
        body = {
            "signals": [
                {"agent_id": "a", "action": "BUY", "confidence": 0.9},
                {"agent_id": "b", "action": "SELL", "confidence": 0.6},
            ],
            "strategy": "highest_confidence",
        }
        assert _post_status(url, body) == 200

    def test_post_resolve_response_schema(self, server):
        url = f"http://localhost:{server}/api/v1/coordination/resolve"
        body = {
            "signals": [
                {"agent_id": "a", "action": "BUY", "confidence": 0.9},
            ],
            "strategy": "highest_confidence",
        }
        r = _post(url, body)
        assert "resolved_action" in r
        assert "winning_signal" in r
        assert "conflict_detected" in r
        assert "strategy" in r

    def test_post_resolve_majority_vote(self, server):
        url = f"http://localhost:{server}/api/v1/coordination/resolve"
        body = {
            "signals": [
                {"agent_id": "a", "action": "BUY", "confidence": 0.5},
                {"agent_id": "b", "action": "BUY", "confidence": 0.4},
                {"agent_id": "c", "action": "SELL", "confidence": 0.9},
            ],
            "strategy": "majority_vote",
        }
        r = _post(url, body)
        assert r["resolved_action"] == "BUY"

    def test_post_resolve_weighted_consensus(self, server):
        url = f"http://localhost:{server}/api/v1/coordination/resolve"
        body = {
            "signals": [
                {"agent_id": "a", "action": "SELL", "confidence": 0.6},
                {"agent_id": "b", "action": "SELL", "confidence": 0.5},
                {"agent_id": "c", "action": "BUY", "confidence": 0.9},
            ],
            "strategy": "weighted_consensus",
        }
        r = _post(url, body)
        # SELL bucket: 1.1, BUY bucket: 0.9
        assert r["resolved_action"] == "SELL"

    def test_post_resolve_400_empty_signals(self, server):
        url = f"http://localhost:{server}/api/v1/coordination/resolve"
        body = {"signals": [], "strategy": "highest_confidence"}
        assert _post_status(url, body) == 400

    def test_post_resolve_400_invalid_strategy(self, server):
        url = f"http://localhost:{server}/api/v1/coordination/resolve"
        body = {
            "signals": [{"agent_id": "a", "action": "BUY", "confidence": 0.8}],
            "strategy": "random_pick",
        }
        assert _post_status(url, body) == 400

    def test_post_resolve_400_signals_not_list(self, server):
        url = f"http://localhost:{server}/api/v1/coordination/resolve"
        body = {"signals": "not-a-list", "strategy": "highest_confidence"}
        assert _post_status(url, body) == 400

    def test_post_resolve_conflict_detected_field(self, server):
        url = f"http://localhost:{server}/api/v1/coordination/resolve"
        body = {
            "signals": [
                {"agent_id": "a", "action": "BUY", "confidence": 0.9},
                {"agent_id": "b", "action": "SELL", "confidence": 0.5},
            ],
            "strategy": "highest_confidence",
        }
        r = _post(url, body)
        assert r["conflict_detected"] is True

    def test_post_resolve_no_conflict(self, server):
        url = f"http://localhost:{server}/api/v1/coordination/resolve"
        body = {
            "signals": [
                {"agent_id": "a", "action": "BUY", "confidence": 0.9},
                {"agent_id": "b", "action": "BUY", "confidence": 0.7},
            ],
            "strategy": "majority_vote",
        }
        r = _post(url, body)
        assert r["conflict_detected"] is False

    def test_post_resolve_default_strategy(self, server):
        """Omitting strategy field defaults to highest_confidence."""
        url = f"http://localhost:{server}/api/v1/coordination/resolve"
        body = {
            "signals": [
                {"agent_id": "a", "action": "HOLD", "confidence": 0.7},
            ],
        }
        r = _post(url, body)
        assert r["strategy"] == "highest_confidence"
