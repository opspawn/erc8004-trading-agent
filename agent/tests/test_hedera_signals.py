"""
test_hedera_signals.py — Unit and integration tests for HederaSignalBus (HCS-10).

All tests use mock mode (HEDERA_TESTNET_MODE=mock) — no network calls.
Target: ~80 tests covering normal operation, edge cases, and error paths.
"""

from __future__ import annotations

import pytest
from unittest.mock import MagicMock

from hedera_signals import (
    HederaSignalBus,
    HCSSignal,
    HCSNetworkError,
    HCSValidationError,
    _MOCK_STORE,
)


# ─── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture(autouse=True)
def clear_mock_store():
    """Reset global mock store before each test."""
    _MOCK_STORE.clear()
    yield
    _MOCK_STORE.clear()


@pytest.fixture
def bus():
    return HederaSignalBus(agent_id="test-agent", mode="mock")


@pytest.fixture
def bus2():
    return HederaSignalBus(agent_id="test-agent-2", mode="mock", topic_id="0.0.9999")


@pytest.fixture
def btc_buy_signal(bus):
    return bus.make_signal("BUY", "BTC", 0.85)


@pytest.fixture
def eth_sell_signal(bus):
    return bus.make_signal("SELL", "ETH", 0.70)


# ─── HCSSignal Validation ─────────────────────────────────────────────────────

class TestHCSSignalValidation:
    def test_valid_buy_signal(self):
        sig = HCSSignal(agent_id="a", signal_type="BUY", ticker="BTC", confidence=0.8)
        sig.validate()   # should not raise

    def test_valid_sell_signal(self):
        sig = HCSSignal(agent_id="a", signal_type="SELL", ticker="ETH", confidence=0.5)
        sig.validate()

    def test_valid_hold_signal(self):
        sig = HCSSignal(agent_id="a", signal_type="HOLD", ticker="SOL", confidence=0.0)
        sig.validate()

    def test_invalid_signal_type(self):
        sig = HCSSignal(agent_id="a", signal_type="LONG", ticker="BTC", confidence=0.8)
        with pytest.raises(ValueError, match="Invalid signal_type"):
            sig.validate()

    def test_confidence_above_one(self):
        sig = HCSSignal(agent_id="a", signal_type="BUY", ticker="BTC", confidence=1.5)
        with pytest.raises(ValueError, match="confidence"):
            sig.validate()

    def test_confidence_below_zero(self):
        sig = HCSSignal(agent_id="a", signal_type="BUY", ticker="BTC", confidence=-0.1)
        with pytest.raises(ValueError, match="confidence"):
            sig.validate()

    def test_confidence_boundary_zero(self):
        sig = HCSSignal(agent_id="a", signal_type="HOLD", ticker="BTC", confidence=0.0)
        sig.validate()   # 0.0 is valid

    def test_confidence_boundary_one(self):
        sig = HCSSignal(agent_id="a", signal_type="BUY", ticker="BTC", confidence=1.0)
        sig.validate()   # 1.0 is valid

    def test_empty_ticker(self):
        sig = HCSSignal(agent_id="a", signal_type="BUY", ticker="", confidence=0.8)
        with pytest.raises(ValueError, match="ticker"):
            sig.validate()

    def test_empty_agent_id(self):
        sig = HCSSignal(agent_id="", signal_type="BUY", ticker="BTC", confidence=0.8)
        with pytest.raises(ValueError, match="agent_id"):
            sig.validate()

    def test_to_dict_roundtrip(self):
        sig = HCSSignal(agent_id="a", signal_type="BUY", ticker="BTC", confidence=0.8)
        d = sig.to_dict()
        assert d["signal_type"] == "BUY"
        assert d["ticker"] == "BTC"
        assert d["confidence"] == 0.8

    def test_to_json_roundtrip(self):
        sig = HCSSignal(agent_id="a", signal_type="SELL", ticker="ETH", confidence=0.6)
        json_str = sig.to_json()
        restored = HCSSignal.from_json(json_str)
        assert restored.signal_type == "SELL"
        assert restored.ticker == "ETH"
        assert restored.confidence == 0.6

    def test_from_dict(self):
        d = {
            "agent_id": "x",
            "signal_type": "HOLD",
            "ticker": "SOL",
            "confidence": 0.5,
        }
        sig = HCSSignal.from_dict(d)
        assert sig.agent_id == "x"
        assert sig.signal_type == "HOLD"

    def test_signal_has_unique_message_id(self):
        sig1 = HCSSignal(agent_id="a", signal_type="BUY", ticker="BTC", confidence=0.8)
        sig2 = HCSSignal(agent_id="a", signal_type="BUY", ticker="BTC", confidence=0.8)
        assert sig1.message_id != sig2.message_id

    def test_signal_has_timestamp(self):
        sig = HCSSignal(agent_id="a", signal_type="BUY", ticker="BTC", confidence=0.8)
        assert sig.timestamp
        assert "T" in sig.timestamp


# ─── HederaSignalBus Init ─────────────────────────────────────────────────────

class TestHederaSignalBusInit:
    def test_default_mode_is_mock(self):
        bus = HederaSignalBus()
        assert bus.mode == "mock"

    def test_custom_agent_id(self):
        bus = HederaSignalBus(agent_id="my-agent")
        assert bus.agent_id == "my-agent"

    def test_custom_topic_id(self):
        bus = HederaSignalBus(topic_id="0.0.12345")
        assert bus.topic_id == "0.0.12345"

    def test_default_topic_id(self):
        bus = HederaSignalBus()
        assert bus.topic_id == HederaSignalBus.DEFAULT_TOPIC_ID

    def test_initial_published_count(self, bus):
        assert bus.get_published_count() == 0


# ─── Publish ──────────────────────────────────────────────────────────────────

class TestPublishSignal:
    def test_publish_buy_signal(self, bus, btc_buy_signal):
        result = bus.publish_signal(btc_buy_signal)
        assert result.signal_type == "BUY"
        assert result.ticker == "BTC"
        assert result.sequence_number == 0

    def test_publish_increments_count(self, bus, btc_buy_signal):
        bus.publish_signal(btc_buy_signal)
        assert bus.get_published_count() == 1

    def test_publish_multiple_increments(self, bus):
        for i in range(5):
            s = bus.make_signal("BUY", "BTC", 0.7 + i * 0.01)
            bus.publish_signal(s)
        assert bus.get_published_count() == 5

    def test_publish_sets_sequence_number(self, bus):
        s1 = bus.make_signal("BUY", "BTC", 0.8)
        s2 = bus.make_signal("SELL", "ETH", 0.7)
        r1 = bus.publish_signal(s1)
        r2 = bus.publish_signal(s2)
        assert r1.sequence_number == 0
        assert r2.sequence_number == 1

    def test_publish_sets_topic_id(self, bus, btc_buy_signal):
        result = bus.publish_signal(btc_buy_signal)
        assert result.topic_id == bus.topic_id

    def test_publish_invalid_signal_raises(self, bus):
        bad = HCSSignal(agent_id="a", signal_type="INVALID", ticker="BTC", confidence=0.8)
        with pytest.raises(HCSValidationError, match="Invalid signal_type"):
            bus.publish_signal(bad)

    def test_publish_network_failure(self, bus, btc_buy_signal):
        bus.set_network_failure(True)
        with pytest.raises(HCSNetworkError):
            bus.publish_signal(btc_buy_signal)

    def test_publish_network_failure_does_not_increment_count(self, bus, btc_buy_signal):
        bus.set_network_failure(True)
        with pytest.raises(HCSNetworkError):
            bus.publish_signal(btc_buy_signal)
        assert bus.get_published_count() == 0

    def test_publish_after_network_recovery(self, bus, btc_buy_signal):
        bus.set_network_failure(True)
        with pytest.raises(HCSNetworkError):
            bus.publish_signal(btc_buy_signal)
        bus.set_network_failure(False)
        s2 = bus.make_signal("BUY", "ETH", 0.75)
        result = bus.publish_signal(s2)
        assert result.sequence_number == 0


# ─── Subscribe ────────────────────────────────────────────────────────────────

class TestSubscribeSignals:
    def test_subscribe_callback_called_on_publish(self, bus):
        received = []
        bus.subscribe_signals(bus.topic_id, received.append)
        s = bus.make_signal("BUY", "BTC", 0.8)
        bus.publish_signal(s)
        assert len(received) == 1
        assert received[0].signal_type == "BUY"

    def test_subscribe_callback_called_multiple_times(self, bus):
        received = []
        bus.subscribe_signals(bus.topic_id, received.append)
        for _ in range(3):
            bus.publish_signal(bus.make_signal("SELL", "ETH", 0.6))
        assert len(received) == 3

    def test_multiple_subscribers(self, bus):
        cb1 = MagicMock()
        cb2 = MagicMock()
        bus.subscribe_signals(bus.topic_id, cb1)
        bus.subscribe_signals(bus.topic_id, cb2)
        bus.publish_signal(bus.make_signal("BUY", "BTC", 0.7))
        cb1.assert_called_once()
        cb2.assert_called_once()

    def test_unsubscribe_all(self, bus):
        received = []
        bus.subscribe_signals(bus.topic_id, received.append)
        bus.unsubscribe_all()
        bus.publish_signal(bus.make_signal("BUY", "BTC", 0.8))
        assert len(received) == 0

    def test_subscriber_exception_does_not_break_publish(self, bus):
        def bad_cb(sig):
            raise RuntimeError("callback error")

        received = []
        bus.subscribe_signals(bus.topic_id, bad_cb)
        bus.subscribe_signals(bus.topic_id, received.append)
        bus.publish_signal(bus.make_signal("BUY", "BTC", 0.8))
        # Second callback should still be called
        assert len(received) == 1


# ─── Signal History ───────────────────────────────────────────────────────────

class TestGetSignalHistory:
    def test_empty_history(self, bus):
        history = bus.get_signal_history()
        assert history == []

    def test_history_after_publish(self, bus):
        bus.publish_signal(bus.make_signal("BUY", "BTC", 0.8))
        history = bus.get_signal_history()
        assert len(history) == 1

    def test_history_limit(self, bus):
        for i in range(10):
            bus.publish_signal(bus.make_signal("BUY", "BTC", 0.8))
        history = bus.get_signal_history(limit=5)
        assert len(history) == 5

    def test_history_returns_latest(self, bus):
        for i in range(5):
            bus.publish_signal(bus.make_signal("BUY", "BTC", 0.5 + i * 0.1))
        history = bus.get_signal_history(limit=3)
        # Last 3 should have sequence_numbers 2, 3, 4
        seqs = [s.sequence_number for s in history]
        assert seqs == [2, 3, 4]

    def test_history_different_topic(self, bus):
        # bus uses default topic; bus2 uses 0.0.9999
        bus.publish_signal(bus.make_signal("BUY", "BTC", 0.8))
        history_other = bus.get_signal_history(topic_id="0.0.9999")
        assert len(history_other) == 0

    def test_history_network_failure(self, bus):
        bus.set_network_failure(True)
        with pytest.raises(HCSNetworkError):
            bus.get_signal_history()

    def test_history_preserves_signal_data(self, bus):
        s = bus.make_signal("SELL", "ETH", 0.65)
        bus.publish_signal(s)
        history = bus.get_signal_history()
        assert history[0].signal_type == "SELL"
        assert history[0].ticker == "ETH"
        assert history[0].confidence == 0.65

    def test_history_from_explicit_topic_id(self, bus):
        bus.publish_signal(bus.make_signal("BUY", "BTC", 0.8))
        history = bus.get_signal_history(topic_id=bus.topic_id)
        assert len(history) == 1


# ─── Make Signal ──────────────────────────────────────────────────────────────

class TestMakeSignal:
    def test_make_signal_sets_agent_id(self, bus):
        s = bus.make_signal("BUY", "BTC", 0.8)
        assert s.agent_id == bus.agent_id

    def test_make_signal_fields(self, bus):
        s = bus.make_signal("SELL", "SOL", 0.6)
        assert s.signal_type == "SELL"
        assert s.ticker == "SOL"
        assert s.confidence == 0.6

    def test_make_multiple_signals_unique_ids(self, bus):
        s1 = bus.make_signal("BUY", "BTC", 0.8)
        s2 = bus.make_signal("BUY", "BTC", 0.8)
        assert s1.message_id != s2.message_id


# ─── Real Mode Guard ──────────────────────────────────────────────────────────

class TestRealModeGuard:
    def test_real_mode_publish_raises_not_implemented(self):
        bus = HederaSignalBus(mode="real")
        s = bus.make_signal("BUY", "BTC", 0.8)
        with pytest.raises(NotImplementedError):
            bus.publish_signal(s)

    def test_real_mode_history_raises_not_implemented(self):
        bus = HederaSignalBus(mode="real")
        with pytest.raises(NotImplementedError):
            bus.get_signal_history()


# ─── Reset Mock Store ─────────────────────────────────────────────────────────

class TestResetMockStore:
    def test_reset_clears_history(self, bus):
        bus.publish_signal(bus.make_signal("BUY", "BTC", 0.8))
        bus.reset_mock_store()
        assert bus.get_signal_history() == []

    def test_reset_resets_sequence(self, bus):
        bus.publish_signal(bus.make_signal("BUY", "BTC", 0.8))
        bus.reset_mock_store()
        s2 = bus.make_signal("SELL", "ETH", 0.7)
        result = bus.publish_signal(s2)
        assert result.sequence_number == 0
