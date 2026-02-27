"""
test_s43_coordination_edge_cases.py — Sprint 43 supplemental: coordination edge cases.

Additional coverage for broadcast_signal, get_coordination_signals, and
resolve_coordination_conflict to push test suite past 5,500 total.
"""
from __future__ import annotations

import os
import sys
import time

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from demo_server import (
    _S43_SIGNAL_BUFFER,
    _S43_VALID_ACTIONS,
    _S43_VALID_ASSETS,
    broadcast_signal,
    get_coordination_signals,
    resolve_coordination_conflict,
)


def _clear():
    import demo_server
    with demo_server._S43_SIGNAL_LOCK:
        _S43_SIGNAL_BUFFER.clear()


# ─── broadcast_signal edge cases ───────────────────────────────────────────────


class TestBroadcastEdgeCases:
    def setup_method(self):
        _clear()

    def test_action_case_insensitive_sell(self):
        r = broadcast_signal("agent-balanced-002", "sell", "ETH/USD", 0.7)
        assert r["action"] == "SELL"

    def test_action_case_insensitive_hold(self):
        r = broadcast_signal("agent-balanced-002", "hold", "BTC/USD", 0.5)
        assert r["action"] == "HOLD"

    def test_action_case_insensitive_rebalance(self):
        r = broadcast_signal("agent-balanced-002", "rebalance", "BTC/USD", 0.4)
        assert r["action"] == "REBALANCE"

    def test_multiple_broadcasts_accumulate_in_buffer(self):
        _clear()
        for i in range(5):
            broadcast_signal("agent-conservative-001", "HOLD", "BTC/USD", 0.5)
        r = get_coordination_signals(limit=10)
        assert r["total_returned"] >= 5

    def test_recipient_count_equals_agents_minus_one(self):
        """For a known agent_id in _S43_AGENT_IDS, recipients = len(_S43_AGENT_IDS) - 1."""
        from demo_server import _S43_AGENT_IDS
        sender = "agent-conservative-001"
        r = broadcast_signal(sender, "BUY", "BTC/USD", 0.8)
        assert r["recipient_count"] == len(_S43_AGENT_IDS) - 1

    def test_unknown_sender_gets_all_known_as_recipients(self):
        """Unknown sender is not in _S43_AGENT_IDS so all known agents are recipients."""
        from demo_server import _S43_AGENT_IDS
        r = broadcast_signal("agent-unknown-xyz", "BUY", "BTC/USD", 0.8)
        assert r["recipient_count"] == len(_S43_AGENT_IDS)

    def test_confidence_boundary_0_0(self):
        r = broadcast_signal("agent-conservative-001", "HOLD", "BTC/USD", 0.0)
        assert r["confidence"] == pytest.approx(0.0)

    def test_confidence_boundary_1_0(self):
        r = broadcast_signal("agent-conservative-001", "BUY", "BTC/USD", 1.0)
        assert r["confidence"] == pytest.approx(1.0)

    def test_broadcast_id_contains_agent_prefix(self):
        r = broadcast_signal("agent-conservative-001", "BUY", "BTC/USD", 0.8)
        # broadcast_id format: bc-{ts}-{agent[:8]}
        assert "agent-co" in r["broadcast_id"]

    def test_btcusd_valid(self):
        r = broadcast_signal("agent-balanced-002", "BUY", "BTC/USD", 0.7)
        assert r["asset"] == "BTC/USD"

    def test_ethusd_valid(self):
        r = broadcast_signal("agent-balanced-002", "SELL", "ETH/USD", 0.6)
        assert r["asset"] == "ETH/USD"

    def test_solusd_valid(self):
        r = broadcast_signal("agent-balanced-002", "HOLD", "SOL/USD", 0.5)
        assert r["asset"] == "SOL/USD"

    def test_metadata_none_becomes_empty_dict(self):
        r = broadcast_signal("agent-conservative-001", "BUY", "BTC/USD", 0.8, metadata=None)
        assert r["metadata"] == {}

    def test_confidence_error_exactly_neg(self):
        with pytest.raises(ValueError):
            broadcast_signal("agent-conservative-001", "BUY", "BTC/USD", -0.001)

    def test_confidence_error_exactly_over_one(self):
        with pytest.raises(ValueError):
            broadcast_signal("agent-conservative-001", "BUY", "BTC/USD", 1.001)


# ─── get_coordination_signals edge cases ───────────────────────────────────────


class TestGetSignalsEdgeCases:
    def setup_method(self):
        _clear()

    def test_no_filter_returns_all_recent(self):
        for i in range(3):
            broadcast_signal("agent-conservative-001", "HOLD", "BTC/USD", 0.5)
        r = get_coordination_signals()
        assert r["total_returned"] >= 3

    def test_limit_1_returns_one(self):
        for i in range(5):
            broadcast_signal("agent-conservative-001", "BUY", "BTC/USD", 0.8)
        r = get_coordination_signals(limit=1)
        assert r["total_returned"] == 1

    def test_filter_none_values_ignored(self):
        broadcast_signal("agent-conservative-001", "BUY", "BTC/USD", 0.9)
        r = get_coordination_signals(asset=None, agent_id=None)
        assert r["total_returned"] >= 1

    def test_signals_are_dicts(self):
        broadcast_signal("agent-balanced-002", "SELL", "ETH/USD", 0.7)
        r = get_coordination_signals()
        for sig in r["signals"]:
            assert isinstance(sig, dict)

    def test_signal_has_broadcast_id(self):
        broadcast_signal("agent-balanced-002", "SELL", "ETH/USD", 0.7)
        r = get_coordination_signals(limit=1)
        assert "broadcast_id" in r["signals"][0]

    def test_signal_has_from_agent(self):
        broadcast_signal("agent-balanced-002", "SELL", "ETH/USD", 0.7)
        r = get_coordination_signals(limit=1)
        assert "from_agent" in r["signals"][0]

    def test_signal_has_action(self):
        broadcast_signal("agent-balanced-002", "SELL", "ETH/USD", 0.7)
        r = get_coordination_signals(limit=1)
        assert "action" in r["signals"][0]

    def test_signal_has_confidence(self):
        broadcast_signal("agent-balanced-002", "SELL", "ETH/USD", 0.7)
        r = get_coordination_signals(limit=1)
        assert "confidence" in r["signals"][0]

    def test_filters_both_none_in_response(self):
        r = get_coordination_signals()
        assert r["filters"]["asset"] is None
        assert r["filters"]["agent_id"] is None

    def test_generated_at_is_float(self):
        r = get_coordination_signals()
        assert isinstance(r["generated_at"], float)


# ─── resolve_coordination_conflict edge cases ──────────────────────────────────


class TestResolveEdgeCases:
    def setup_method(self):
        _clear()

    def test_single_signal_no_conflict(self):
        signals = [{"agent_id": "a", "action": "BUY", "confidence": 0.9}]
        r = resolve_coordination_conflict(signals)
        assert r["conflict_detected"] is False

    def test_two_same_action_no_conflict(self):
        signals = [
            {"agent_id": "a", "action": "SELL", "confidence": 0.7},
            {"agent_id": "b", "action": "SELL", "confidence": 0.5},
        ]
        r = resolve_coordination_conflict(signals)
        assert r["conflict_detected"] is False

    def test_resolved_action_is_string(self):
        signals = [{"agent_id": "a", "action": "HOLD", "confidence": 0.6}]
        r = resolve_coordination_conflict(signals)
        assert isinstance(r["resolved_action"], str)

    def test_resolved_action_is_valid_action(self):
        signals = [{"agent_id": "a", "action": "REBALANCE", "confidence": 0.5}]
        r = resolve_coordination_conflict(signals)
        assert r["resolved_action"] in _S43_VALID_ACTIONS

    def test_actions_present_is_sorted(self):
        signals = [
            {"agent_id": "a", "action": "SELL", "confidence": 0.7},
            {"agent_id": "b", "action": "BUY", "confidence": 0.5},
        ]
        r = resolve_coordination_conflict(signals)
        assert r["actions_present"] == sorted(r["actions_present"])

    def test_candidates_count_matches_input(self):
        signals = [
            {"agent_id": "a", "action": "BUY", "confidence": 0.9},
            {"agent_id": "b", "action": "SELL", "confidence": 0.7},
            {"agent_id": "c", "action": "HOLD", "confidence": 0.5},
        ]
        r = resolve_coordination_conflict(signals)
        assert len(r["candidates"]) == 3

    def test_resolution_details_is_dict(self):
        signals = [{"agent_id": "a", "action": "BUY", "confidence": 0.8}]
        r = resolve_coordination_conflict(signals)
        assert isinstance(r["resolution_details"], dict)

    def test_highest_confidence_three_way(self):
        signals = [
            {"agent_id": "a", "action": "BUY", "confidence": 0.5},
            {"agent_id": "b", "action": "SELL", "confidence": 0.8},
            {"agent_id": "c", "action": "HOLD", "confidence": 0.3},
        ]
        r = resolve_coordination_conflict(signals, strategy="highest_confidence")
        assert r["resolved_action"] == "SELL"

    def test_majority_three_agree_one_disagree(self):
        signals = [
            {"agent_id": "a", "action": "HOLD", "confidence": 0.5},
            {"agent_id": "b", "action": "HOLD", "confidence": 0.4},
            {"agent_id": "c", "action": "HOLD", "confidence": 0.6},
            {"agent_id": "d", "action": "BUY", "confidence": 0.9},
        ]
        r = resolve_coordination_conflict(signals, strategy="majority_vote")
        assert r["resolved_action"] == "HOLD"

    def test_weighted_consensus_all_same(self):
        signals = [
            {"agent_id": "a", "action": "BUY", "confidence": 0.8},
            {"agent_id": "b", "action": "BUY", "confidence": 0.7},
        ]
        r = resolve_coordination_conflict(signals, strategy="weighted_consensus")
        assert r["resolved_action"] == "BUY"
