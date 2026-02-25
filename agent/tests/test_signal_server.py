"""
Tests for signal_server.py — WebSocket Signal Server.

All tests are unit/mock-based. No actual WebSocket server is started.
"""

from __future__ import annotations

import json
import sys
import os
import time
from unittest.mock import MagicMock, patch, AsyncMock

import pytest

# Ensure agent/ is on path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from signal_server import (
    SignalType,
    TradeSignal,
    SignalGenerator,
    SignalServer,
    PROTOCOLS,
)


# ─── SignalType Tests ──────────────────────────────────────────────────────────

class TestSignalType:
    def test_signal_type_values(self):
        assert SignalType.BUY == "BUY"
        assert SignalType.SELL == "SELL"
        assert SignalType.HOLD == "HOLD"
        assert SignalType.REBALANCE == "REBALANCE"

    def test_signal_type_is_str(self):
        for st in SignalType:
            assert isinstance(st.value, str)

    def test_all_signal_types_present(self):
        values = {st.value for st in SignalType}
        assert "BUY" in values
        assert "SELL" in values
        assert "HOLD" in values
        assert "REBALANCE" in values

    def test_signal_type_count(self):
        assert len(SignalType) == 4

    def test_signal_type_iteration(self):
        types = list(SignalType)
        assert len(types) == 4


# ─── TradeSignal Tests ─────────────────────────────────────────────────────────

class TestTradeSignal:
    def _make_signal(self, **kwargs) -> TradeSignal:
        defaults = dict(
            timestamp=time.time(),
            signal_type="BUY",
            protocol="Aave",
            direction="LONG",
            confidence=0.75,
            agent_id="test_agent_v1",
            sequence=1,
            mesh_consensus=True,
            weighted_confidence=0.75,
        )
        defaults.update(kwargs)
        return TradeSignal(**defaults)

    def test_to_json_returns_string(self):
        sig = self._make_signal()
        assert isinstance(sig.to_json(), str)

    def test_to_json_is_valid_json(self):
        sig = self._make_signal()
        data = json.loads(sig.to_json())
        assert isinstance(data, dict)

    def test_to_json_required_fields(self):
        sig = self._make_signal()
        data = json.loads(sig.to_json())
        required = ["timestamp", "signal_type", "protocol", "direction",
                    "confidence", "agent_id"]
        for key in required:
            assert key in data, f"Missing field: {key}"

    def test_to_json_signal_type_field(self):
        sig = self._make_signal(signal_type="SELL")
        data = json.loads(sig.to_json())
        assert data["signal_type"] == "SELL"

    def test_to_json_protocol_field(self):
        sig = self._make_signal(protocol="Uniswap")
        data = json.loads(sig.to_json())
        assert data["protocol"] == "Uniswap"

    def test_to_json_direction_field(self):
        sig = self._make_signal(direction="SHORT")
        data = json.loads(sig.to_json())
        assert data["direction"] == "SHORT"

    def test_to_json_confidence_rounded(self):
        sig = self._make_signal(confidence=0.123456789)
        data = json.loads(sig.to_json())
        assert len(str(data["confidence"]).split(".")[-1]) <= 4

    def test_to_json_agent_id_field(self):
        sig = self._make_signal(agent_id="mesh_agent_42")
        data = json.loads(sig.to_json())
        assert data["agent_id"] == "mesh_agent_42"

    def test_to_json_sequence_field(self):
        sig = self._make_signal(sequence=99)
        data = json.loads(sig.to_json())
        assert data["sequence"] == 99

    def test_to_json_mesh_consensus_field(self):
        sig = self._make_signal(mesh_consensus=True)
        data = json.loads(sig.to_json())
        assert data["mesh_consensus"] is True

    def test_to_json_mesh_consensus_false(self):
        sig = self._make_signal(mesh_consensus=False)
        data = json.loads(sig.to_json())
        assert data["mesh_consensus"] is False

    def test_from_dict_roundtrip(self):
        sig = self._make_signal()
        data = json.loads(sig.to_json())
        restored = TradeSignal.from_dict(data)
        assert restored.signal_type == sig.signal_type
        assert restored.protocol == sig.protocol
        assert restored.agent_id == sig.agent_id

    def test_from_dict_required_fields(self):
        d = {
            "timestamp": time.time(),
            "signal_type": "HOLD",
            "protocol": "Compound",
            "direction": "NEUTRAL",
            "confidence": 0.30,
            "agent_id": "conservative_agent",
            "sequence": 5,
        }
        sig = TradeSignal.from_dict(d)
        assert sig.signal_type == "HOLD"
        assert sig.protocol == "Compound"

    def test_confidence_range_in_json(self):
        for conf in [0.0, 0.5, 1.0]:
            sig = self._make_signal(confidence=conf)
            data = json.loads(sig.to_json())
            assert 0.0 <= data["confidence"] <= 1.0

    def test_all_signal_types_serialisable(self):
        for st in SignalType:
            sig = self._make_signal(signal_type=st.value)
            data = json.loads(sig.to_json())
            assert data["signal_type"] == st.value

    def test_timestamp_is_float(self):
        sig = self._make_signal(timestamp=1000000.5)
        data = json.loads(sig.to_json())
        assert isinstance(data["timestamp"], (int, float))

    def test_weighted_confidence_in_json(self):
        sig = self._make_signal(weighted_confidence=0.82)
        data = json.loads(sig.to_json())
        assert "weighted_confidence" in data


# ─── SignalGenerator Tests ─────────────────────────────────────────────────────

class TestSignalGenerator:
    def test_generator_creates_coordinator_by_default(self):
        gen = SignalGenerator()
        assert gen.coordinator is not None

    def test_generator_accepts_custom_coordinator(self):
        mock_coord = MagicMock()
        gen = SignalGenerator(coordinator=mock_coord)
        assert gen.coordinator is mock_coord

    def test_generator_agent_id(self):
        gen = SignalGenerator(agent_id="my_agent")
        assert gen.agent_id == "my_agent"

    def test_sequence_starts_at_zero(self):
        gen = SignalGenerator()
        assert gen.sequence == 0

    def test_sequence_increments(self):
        gen = SignalGenerator()
        gen.next_signal()
        assert gen.sequence == 1
        gen.next_signal()
        assert gen.sequence == 2

    def test_next_signal_returns_trade_signal(self):
        gen = SignalGenerator()
        sig = gen.next_signal()
        assert isinstance(sig, TradeSignal)

    def test_next_signal_has_valid_signal_type(self):
        gen = SignalGenerator()
        valid_types = {st.value for st in SignalType}
        for _ in range(10):
            sig = gen.next_signal()
            assert sig.signal_type in valid_types

    def test_next_signal_has_valid_protocol(self):
        gen = SignalGenerator()
        for _ in range(10):
            sig = gen.next_signal()
            assert sig.protocol in PROTOCOLS

    def test_next_signal_confidence_in_range(self):
        gen = SignalGenerator()
        for _ in range(20):
            sig = gen.next_signal()
            assert 0.0 <= sig.confidence <= 1.0

    def test_next_signal_direction_valid(self):
        gen = SignalGenerator()
        valid = {"LONG", "SHORT", "NEUTRAL"}
        for _ in range(20):
            sig = gen.next_signal()
            assert sig.direction in valid

    def test_next_signal_has_agent_id(self):
        gen = SignalGenerator(agent_id="test_v1")
        sig = gen.next_signal()
        assert sig.agent_id == "test_v1"

    def test_next_signal_has_timestamp(self):
        gen = SignalGenerator()
        before = time.time()
        sig = gen.next_signal()
        after = time.time()
        assert before <= sig.timestamp <= after

    def test_generate_batch_count(self):
        gen = SignalGenerator()
        batch = gen.generate_batch(5)
        assert len(batch) == 5

    def test_generate_batch_all_trade_signals(self):
        gen = SignalGenerator()
        batch = gen.generate_batch(10)
        for sig in batch:
            assert isinstance(sig, TradeSignal)

    def test_generate_batch_sequence_monotonic(self):
        gen = SignalGenerator()
        batch = gen.generate_batch(5)
        seqs = [s.sequence for s in batch]
        assert seqs == sorted(seqs)
        assert seqs[0] < seqs[-1]

    def test_rebalance_signal_generated_eventually(self):
        """Every 7th signal is forced to REBALANCE."""
        gen = SignalGenerator()
        types_seen = set()
        for _ in range(20):
            sig = gen.next_signal()
            types_seen.add(sig.signal_type)
        assert "REBALANCE" in types_seen

    def test_protocols_cycle_through(self):
        gen = SignalGenerator()
        batch = gen.generate_batch(len(PROTOCOLS) * 2)
        protocols_seen = {s.protocol for s in batch}
        assert len(protocols_seen) > 1


# ─── SignalServer Tests ────────────────────────────────────────────────────────

class TestSignalServer:
    def test_server_creates_with_defaults(self):
        srv = SignalServer()
        assert srv.host == "localhost"
        assert srv.port == 8765
        assert srv.broadcast_interval == 5.0

    def test_server_custom_params(self):
        srv = SignalServer(host="0.0.0.0", port=9000, broadcast_interval=1.0)
        assert srv.host == "0.0.0.0"
        assert srv.port == 9000
        assert srv.broadcast_interval == 1.0

    def test_server_not_running_initially(self):
        srv = SignalServer()
        assert not srv._running

    def test_server_zero_clients_initially(self):
        srv = SignalServer()
        assert srv.connected_clients == 0

    def test_server_broadcast_count_zero_initially(self):
        srv = SignalServer()
        assert srv.broadcast_count == 0

    def test_server_last_signal_none_initially(self):
        srv = SignalServer()
        assert srv.last_signal is None

    def test_server_has_generator(self):
        srv = SignalServer()
        assert isinstance(srv.generator, SignalGenerator)

    def test_server_get_status_not_running(self):
        srv = SignalServer()
        status = srv.get_status()
        assert status["running"] is False
        assert status["connected_clients"] == 0
        assert status["broadcast_count"] == 0

    def test_server_get_status_keys(self):
        srv = SignalServer()
        status = srv.get_status()
        expected = ["running", "host", "port", "connected_clients",
                    "broadcast_count", "broadcast_interval", "last_signal"]
        for key in expected:
            assert key in status

    def test_server_accepts_coordinator(self):
        mock_coord = MagicMock()
        srv = SignalServer(coordinator=mock_coord)
        assert srv.generator.coordinator is mock_coord

    def test_server_accepts_agent_id(self):
        srv = SignalServer(agent_id="custom_agent")
        assert srv.generator.agent_id == "custom_agent"

    def test_server_clients_set_initially_empty(self):
        srv = SignalServer()
        assert len(srv._clients) == 0

    def test_generator_produces_valid_signal(self):
        srv = SignalServer()
        sig = srv.generator.next_signal()
        assert isinstance(sig, TradeSignal)
        assert sig.signal_type in {st.value for st in SignalType}
