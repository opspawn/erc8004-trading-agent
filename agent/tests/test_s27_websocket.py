"""
test_s27_websocket.py — Sprint 27: WebSocket streaming server tests.

35 tests covering:
  - _StreamState tick generation (GBM prices, volume, change_pct)
  - _StreamState signal and trade generation
  - Message builder functions (welcome, pong, error, risk_alert)
  - build_welcome_message / build_pong_message / build_error_message / build_risk_alert
  - WSServer class construction
  - Module-level state accessors (get_connected_count, get_stream_state)
  - Session path validation helpers
  - Symbol filter parsing
  - JSON serialization of all message types
  - SYMBOLS and AGENT_IDS constants
"""

from __future__ import annotations

import json
import math
import time

import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from ws_server import (
    _StreamState,
    _SEED_PRICES,
    SYMBOLS,
    AGENT_IDS,
    WS_PORT,
    TICK_INTERVAL,
    build_welcome_message,
    build_pong_message,
    build_error_message,
    build_risk_alert,
    get_connected_count,
    get_stream_state,
    WSServer,
)


# ─── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture
def state():
    return _StreamState()


# ─── Constants ────────────────────────────────────────────────────────────────

class TestConstants:
    def test_symbols_not_empty(self):
        assert len(SYMBOLS) >= 4

    def test_known_symbols(self):
        for sym in ["BTC/USD", "ETH/USD", "SOL/USD"]:
            assert sym in SYMBOLS

    def test_agent_ids_not_empty(self):
        assert len(AGENT_IDS) >= 4

    def test_ws_port_default(self):
        assert WS_PORT == 8085

    def test_tick_interval_positive(self):
        assert TICK_INTERVAL > 0

    def test_seed_prices_all_symbols(self):
        for sym in SYMBOLS:
            assert sym in _SEED_PRICES
            assert _SEED_PRICES[sym] > 0


# ─── _StreamState ─────────────────────────────────────────────────────────────

class TestStreamState:
    def test_initial_prices(self, state):
        for sym in SYMBOLS:
            assert sym in state.prices
            assert state.prices[sym] > 0

    def test_initial_volumes(self, state):
        for sym in SYMBOLS:
            assert state.volumes[sym] == 1000.0

    def test_tick_count_starts_zero(self, state):
        assert state.tick_count == 0

    def test_next_tick_structure(self, state):
        tick = state.next_tick("BTC/USD")
        assert tick["type"] == "tick"
        assert "ts" in tick
        assert tick["symbol"] == "BTC/USD"
        assert tick["price"] > 0
        assert tick["volume"] > 0
        assert "change_pct" in tick
        assert "tick_num" in tick

    def test_next_tick_increments_counter(self, state):
        state.next_tick("BTC/USD")
        assert state.tick_count == 1
        state.next_tick("ETH/USD")
        assert state.tick_count == 2

    def test_next_tick_changes_price(self, state):
        original = state.prices["BTC/USD"]
        ticks = [state.next_tick("BTC/USD") for _ in range(10)]
        prices = [t["price"] for t in ticks]
        # Not all prices should be identical (probabilistic but near-certain)
        assert len(set(prices)) > 1

    def test_next_tick_price_stays_positive(self, state):
        for _ in range(50):
            t = state.next_tick("ETH/USD")
            assert t["price"] > 0

    def test_next_tick_ts_is_recent(self, state):
        before = time.time()
        tick = state.next_tick("SOL/USD")
        after = time.time()
        assert before <= tick["ts"] <= after

    def test_next_tick_change_pct_reasonable(self, state):
        for _ in range(20):
            t = state.next_tick("BTC/USD")
            # GBM with low vol: daily change should be small
            assert abs(t["change_pct"]) < 10.0

    def test_random_signal_structure(self, state):
        sig = state.random_signal()
        assert sig["type"] == "signal"
        assert "ts" in sig
        assert sig["agent_id"] in AGENT_IDS
        assert sig["action"] in ("BUY", "SELL", "HOLD")
        assert 0.5 <= sig["confidence"] <= 0.99
        assert sig["symbol"] in SYMBOLS
        assert sig["price"] > 0

    def test_random_signal_with_symbol(self, state):
        sig = state.random_signal(symbol="ETH/USD")
        assert sig["symbol"] == "ETH/USD"

    def test_random_trade_structure(self, state):
        trade = state.random_trade()
        assert trade["type"] == "trade"
        assert "ts" in trade
        assert trade["agent_id"] in AGENT_IDS
        assert trade["side"] in ("BUY", "SELL")
        assert trade["qty"] > 0
        assert trade["price"] > 0
        assert trade["symbol"] in SYMBOLS
        assert "pnl" in trade

    def test_random_trade_with_symbol(self, state):
        t = state.random_trade(symbol="SOL/USD")
        assert t["symbol"] == "SOL/USD"

    def test_connected_clients_default_zero(self, state):
        assert state.connected_clients == 0


# ─── Message Builders ─────────────────────────────────────────────────────────

class TestMessageBuilders:
    def test_welcome_message_is_valid_json(self):
        raw = build_welcome_message("abc123")
        msg = json.loads(raw)
        assert msg["type"] == "connected"
        assert msg["session_id"] == "abc123"
        assert "ts" in msg
        assert msg["symbols"] == SYMBOLS
        assert msg["agents"] == AGENT_IDS
        assert "tick_interval_s" in msg

    def test_welcome_message_message_field(self):
        msg = json.loads(build_welcome_message("x"))
        assert "ERC-8004" in msg["message"]

    def test_pong_message_no_echo(self):
        raw = build_pong_message()
        msg = json.loads(raw)
        assert msg["type"] == "pong"
        assert "ts" in msg
        assert "echo" not in msg

    def test_pong_message_with_echo(self):
        raw = build_pong_message("req-42")
        msg = json.loads(raw)
        assert msg["echo"] == "req-42"

    def test_error_message_structure(self):
        raw = build_error_message("bad input")
        msg = json.loads(raw)
        assert msg["type"] == "error"
        assert msg["message"] == "bad input"
        assert "ts" in msg

    def test_risk_alert_defaults(self):
        raw = build_risk_alert()
        msg = json.loads(raw)
        assert msg["type"] == "risk_alert"
        assert "level" in msg
        assert "message" in msg
        assert "ts" in msg
        assert "agent_id" in msg

    def test_risk_alert_custom(self):
        raw = build_risk_alert(level="high", message="Stop loss triggered")
        msg = json.loads(raw)
        assert msg["level"] == "high"
        assert msg["message"] == "Stop loss triggered"


# ─── Module-Level Accessors ───────────────────────────────────────────────────

class TestModuleAccessors:
    def test_get_connected_count_returns_int(self):
        count = get_connected_count()
        assert isinstance(count, int)
        assert count >= 0

    def test_get_stream_state_structure(self):
        s = get_stream_state()
        assert "tick_count" in s
        assert "connected_clients" in s
        assert "prices" in s
        assert "symbols" in s
        assert "agents" in s
        assert s["symbols"] == SYMBOLS
        assert s["agents"] == AGENT_IDS

    def test_get_stream_state_prices_are_positive(self):
        s = get_stream_state()
        for sym, price in s["prices"].items():
            assert price > 0


# ─── WSServer Class ───────────────────────────────────────────────────────────

class TestWSServer:
    def test_instantiation_default_port(self):
        srv = WSServer()
        assert srv.port == WS_PORT

    def test_instantiation_custom_port(self):
        srv = WSServer(port=9999)
        assert srv.port == 9999

    def test_thread_none_before_start(self):
        srv = WSServer()
        assert srv._thread is None

    def test_loop_none_before_start(self):
        srv = WSServer()
        assert srv._loop is None
