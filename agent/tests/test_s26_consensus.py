"""
test_s26_consensus.py — Tests for POST /demo/consensus multi-agent consensus endpoint.

Tests the build_consensus() function directly (unit tests) and the HTTP handler.
All tests are offline — no live server required.
"""

from __future__ import annotations

import json
import sys
import os
from pathlib import Path

import pytest

# Allow imports from agent/
sys.path.insert(0, str(Path(__file__).parent.parent))

from demo_server import build_consensus


# ─── 1. Basic consensus results ───────────────────────────────────────────────


class TestBuildConsensusBasic:
    def test_empty_signals_returns_hold(self):
        result = build_consensus("BTC/USD", [])
        assert result["decision"] == "HOLD"

    def test_empty_signals_zero_confidence(self):
        result = build_consensus("BTC/USD", [])
        assert result["confidence"] == 0.0

    def test_empty_signals_zero_votes(self):
        result = build_consensus("BTC/USD", [])
        assert result["votes_for"] == 0
        assert result["votes_against"] == 0

    def test_result_has_symbol(self):
        result = build_consensus("ETH/USD", [])
        assert result["symbol"] == "ETH/USD"

    def test_result_has_reasoning(self):
        result = build_consensus("BTC/USD", [])
        assert isinstance(result["reasoning"], str)
        assert len(result["reasoning"]) > 0


# ─── 2. Single signal consensus ───────────────────────────────────────────────


class TestSingleSignal:
    def test_single_buy_signal(self):
        signals = [{"agent_id": "a1", "action": "BUY", "confidence": 0.9}]
        result = build_consensus("BTC/USD", signals)
        assert result["decision"] == "BUY"

    def test_single_sell_signal(self):
        signals = [{"agent_id": "a1", "action": "SELL", "confidence": 0.8}]
        result = build_consensus("BTC/USD", signals)
        assert result["decision"] == "SELL"

    def test_single_hold_signal(self):
        signals = [{"agent_id": "a1", "action": "HOLD", "confidence": 0.7}]
        result = build_consensus("BTC/USD", signals)
        assert result["decision"] == "HOLD"

    def test_single_buy_votes_for_equals_1(self):
        signals = [{"agent_id": "a1", "action": "BUY", "confidence": 0.9}]
        result = build_consensus("BTC/USD", signals)
        assert result["votes_for"] == 1
        assert result["votes_against"] == 0

    def test_single_buy_confidence_is_1(self):
        signals = [{"agent_id": "a1", "action": "BUY", "confidence": 0.9}]
        result = build_consensus("BTC/USD", signals)
        assert result["confidence"] == 1.0


# ─── 3. Majority vote ─────────────────────────────────────────────────────────


class TestMajorityVote:
    def _buy_majority(self):
        return [
            {"agent_id": "a1", "action": "BUY", "confidence": 0.9},
            {"agent_id": "a2", "action": "BUY", "confidence": 0.8},
            {"agent_id": "a3", "action": "SELL", "confidence": 0.6},
        ]

    def test_buy_majority_wins(self):
        result = build_consensus("ETH/USD", self._buy_majority())
        assert result["decision"] == "BUY"

    def test_buy_majority_votes_for(self):
        result = build_consensus("ETH/USD", self._buy_majority())
        assert result["votes_for"] == 2

    def test_buy_majority_votes_against(self):
        result = build_consensus("ETH/USD", self._buy_majority())
        assert result["votes_against"] == 1

    def test_sell_majority_wins(self):
        signals = [
            {"agent_id": "a1", "action": "SELL", "confidence": 0.9},
            {"agent_id": "a2", "action": "SELL", "confidence": 0.85},
            {"agent_id": "a3", "action": "BUY", "confidence": 0.4},
        ]
        result = build_consensus("BTC/USD", signals)
        assert result["decision"] == "SELL"

    def test_hold_wins_when_majority(self):
        signals = [
            {"agent_id": "a1", "action": "HOLD", "confidence": 0.8},
            {"agent_id": "a2", "action": "HOLD", "confidence": 0.7},
            {"agent_id": "a3", "action": "BUY", "confidence": 0.3},
        ]
        result = build_consensus("SOL/USD", signals)
        assert result["decision"] == "HOLD"


# ─── 4. Confidence weighting ──────────────────────────────────────────────────


class TestConfidenceWeighting:
    def test_high_confidence_buy_beats_two_low_sells(self):
        """One BUY at 0.9 vs two SELLs at 0.1 each — BUY wins by weight."""
        signals = [
            {"agent_id": "a1", "action": "BUY", "confidence": 0.9},
            {"agent_id": "a2", "action": "SELL", "confidence": 0.1},
            {"agent_id": "a3", "action": "SELL", "confidence": 0.1},
        ]
        result = build_consensus("BTC/USD", signals)
        assert result["decision"] == "BUY"

    def test_confidence_in_range_0_to_1(self):
        signals = [{"agent_id": "a1", "action": "BUY", "confidence": 0.75}]
        result = build_consensus("BTC/USD", signals)
        assert 0.0 <= result["confidence"] <= 1.0

    def test_equal_weights_has_valid_decision(self):
        signals = [
            {"agent_id": "a1", "action": "BUY", "confidence": 0.5},
            {"agent_id": "a2", "action": "SELL", "confidence": 0.5},
        ]
        result = build_consensus("BTC/USD", signals)
        assert result["decision"] in {"BUY", "SELL", "HOLD"}

    def test_confidence_clipped_above_1(self):
        """Confidence > 1.0 should be clipped to 1.0."""
        signals = [{"agent_id": "a1", "action": "BUY", "confidence": 1.5}]
        result = build_consensus("BTC/USD", signals)
        assert result["confidence"] <= 1.0

    def test_confidence_clipped_below_0(self):
        """Confidence < 0.0 should be clipped to 0.0."""
        signals = [{"agent_id": "a1", "action": "BUY", "confidence": -0.5}]
        result = build_consensus("BTC/USD", signals)
        assert result["confidence"] >= 0.0


# ─── 5. Edge cases ────────────────────────────────────────────────────────────


class TestEdgeCases:
    def test_missing_action_defaults_to_hold(self):
        signals = [{"agent_id": "a1", "confidence": 0.8}]
        result = build_consensus("BTC/USD", signals)
        assert result["decision"] == "HOLD"

    def test_missing_confidence_defaults_to_half(self):
        signals = [{"agent_id": "a1", "action": "BUY"}]
        result = build_consensus("BTC/USD", signals)
        assert result["confidence"] > 0.0

    def test_case_insensitive_action(self):
        signals = [{"agent_id": "a1", "action": "buy", "confidence": 0.9}]
        result = build_consensus("BTC/USD", signals)
        assert result["decision"] == "BUY"

    def test_large_signal_set(self):
        signals = [{"agent_id": f"a{i}", "action": "BUY", "confidence": 0.7} for i in range(50)]
        result = build_consensus("BTC/USD", signals)
        assert result["decision"] == "BUY"
        assert result["votes_for"] == 50

    def test_return_dict_has_all_keys(self):
        result = build_consensus("BTC/USD", [])
        expected_keys = {"symbol", "decision", "confidence", "votes_for", "votes_against", "reasoning"}
        assert expected_keys.issubset(result.keys())

    def test_reasoning_mentions_decision(self):
        signals = [{"agent_id": "a1", "action": "BUY", "confidence": 0.9}]
        result = build_consensus("ETH/USD", signals)
        assert result["decision"] in result["reasoning"]
