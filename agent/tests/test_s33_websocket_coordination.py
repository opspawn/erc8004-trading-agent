"""
test_s33_websocket_coordination.py — Sprint 33: WebSocket streaming + cross-agent coordination.

Covers:
  - WebSocket /demo/ws/feed: handshake, welcome frame, event streaming
  - POST /demo/agents/coordinate: all three strategies, validation, vote tallies
  - GET /demo/agents/{id}/history: pagination, content, unknown agents
  - Unit tests for coordinate_agents, get_agent_history, _ws_accept_key, _ws_send_text
  - Integration tests via HTTP server
"""

from __future__ import annotations

import base64
import hashlib
import json
import socket
import struct
import threading
import time
from urllib.request import urlopen, Request
from urllib.error import HTTPError

import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from demo_server import (
    # S33 functions
    coordinate_agents,
    get_agent_history,
    _ws_accept_key,
    _ws_send_text,
    _ws_broadcast,
    _ws_clients,
    _ws_clients_lock,
    _COORD_STRATEGIES,
    # Feed buffer
    _push_feed_event,
    _generate_feed_event,
    _FEED_AGENT_IDS,
    _FEED_EVENT_TYPES,
    _AGENT_HISTORY,
    _AGENT_HISTORY_LOCK,
    # Server
    DemoServer,
    SERVER_VERSION,
)

import queue


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def server():
    """Start a DemoServer on a free port and yield its base URL."""
    sock = socket.socket()
    sock.bind(("", 0))
    port = sock.getsockname()[1]
    sock.close()
    srv = DemoServer(port=port)
    srv.start()
    time.sleep(0.25)
    yield f"http://localhost:{port}", port
    srv.stop()


def _get(url: str) -> dict:
    with urlopen(url, timeout=5) as resp:
        return json.loads(resp.read())


def _post(url: str, data: dict, expect_error: bool = False) -> dict:
    body = json.dumps(data).encode("utf-8")
    req = Request(url, data=body, headers={"Content-Type": "application/json"})
    try:
        with urlopen(req, timeout=10) as resp:
            return json.loads(resp.read())
    except HTTPError as exc:
        if expect_error:
            return json.loads(exc.read())
        raise


# ── WebSocket helpers ─────────────────────────────────────────────────────────

def _ws_connect(host: str, port: int, path: str = "/demo/ws/feed"):
    """Open a raw TCP socket and perform WebSocket handshake. Returns (sock, key, headers).

    Reads HTTP headers byte-by-byte to avoid consuming any subsequent WS frame data.
    """
    sock = socket.create_connection((host, port), timeout=5)
    key_bytes = base64.b64encode(b"s33-test-key-12345").decode()
    handshake = (
        f"GET {path} HTTP/1.1\r\n"
        f"Host: {host}:{port}\r\n"
        f"Upgrade: websocket\r\n"
        f"Connection: Upgrade\r\n"
        f"Sec-WebSocket-Key: {key_bytes}\r\n"
        f"Sec-WebSocket-Version: 13\r\n"
        f"\r\n"
    )
    sock.sendall(handshake.encode())
    # Read HTTP response headers byte-by-byte so we don't consume WS frame data
    response = b""
    while not response.endswith(b"\r\n\r\n"):
        b = sock.recv(1)
        if not b:
            break
        response += b
    return sock, key_bytes, response.decode(errors="replace")


def _ws_read_frame(sock) -> str:
    """Read one WebSocket text frame from socket. Returns decoded text."""
    header = b""
    while len(header) < 2:
        header += sock.recv(2 - len(header))

    opcode = header[0] & 0x0F
    length = header[1] & 0x7F

    if length == 126:
        ext = b""
        while len(ext) < 2:
            ext += sock.recv(2 - len(ext))
        length = struct.unpack(">H", ext)[0]
    elif length == 127:
        ext = b""
        while len(ext) < 8:
            ext += sock.recv(8 - len(ext))
        length = struct.unpack(">Q", ext)[0]

    data = b""
    while len(data) < length:
        chunk = sock.recv(min(4096, length - len(data)))
        if not chunk:
            break
        data += chunk

    return data.decode("utf-8")


# ── Unit Tests: _ws_accept_key ─────────────────────────────────────────────────

class TestWsAcceptKey:
    def test_produces_string(self):
        result = _ws_accept_key("dGhlIHNhbXBsZSBub25jZQ==")
        assert isinstance(result, str)

    def test_rfc_6455_known_value(self):
        # RFC 6455 Section 1.3 example
        client_key = "dGhlIHNhbXBsZSBub25jZQ=="
        expected = "s3pPLMBiTxaQ9kYGzzhZRbK+xOo="
        assert _ws_accept_key(client_key) == expected

    def test_different_keys_produce_different_accepts(self):
        a = _ws_accept_key("key1")
        b = _ws_accept_key("key2")
        assert a != b

    def test_base64_encoded(self):
        result = _ws_accept_key("test-key")
        # Should be valid base64
        decoded = base64.b64decode(result)
        assert len(decoded) == 20  # SHA-1 produces 20 bytes

    def test_deterministic(self):
        key = "s33-unit-test-key=="
        assert _ws_accept_key(key) == _ws_accept_key(key)


# ── Unit Tests: coordinate_agents ─────────────────────────────────────────────

class TestCoordinateAgents:
    def _agents(self, n: int = 3):
        return [f"agent-test-{i:03d}" for i in range(n)]

    def test_consensus_returns_decision(self):
        result = coordinate_agents(
            agent_ids=self._agents(3),
            strategy="consensus",
            market_conditions={"symbol": "BTC/USD"},
        )
        assert result["decision"] in ("BUY", "SELL", "HOLD")
        assert result["strategy"] == "consensus"

    def test_independent_strategy(self):
        result = coordinate_agents(
            agent_ids=self._agents(4),
            strategy="independent",
            market_conditions={"symbol": "ETH/USD"},
        )
        assert result["strategy"] == "independent"
        assert result["decision"] in ("BUY", "SELL", "HOLD")

    def test_hierarchical_strategy(self):
        result = coordinate_agents(
            agent_ids=self._agents(5),
            strategy="hierarchical",
            market_conditions={"symbol": "SOL/USD"},
        )
        assert result["strategy"] == "hierarchical"
        assert result["decision"] in ("BUY", "SELL", "HOLD")

    def test_vote_tally_sums_to_total_agents(self):
        agents = self._agents(5)
        result = coordinate_agents(
            agent_ids=agents,
            strategy="consensus",
            market_conditions={},
        )
        tally = result["vote_tally"]
        assert tally["BUY"] + tally["SELL"] + tally["HOLD"] == len(agents)

    def test_agent_votes_count_matches(self):
        agents = self._agents(4)
        result = coordinate_agents(
            agent_ids=agents,
            strategy="consensus",
            market_conditions={},
        )
        assert len(result["agent_votes"]) == len(agents)

    def test_agent_votes_have_required_fields(self):
        result = coordinate_agents(
            agent_ids=self._agents(3),
            strategy="consensus",
            market_conditions={},
        )
        for vote in result["agent_votes"]:
            assert "agent_id" in vote
            assert "vote" in vote
            assert vote["vote"] in ("BUY", "SELL", "HOLD")
            assert "confidence" in vote
            assert 0.0 <= vote["confidence"] <= 1.0

    def test_confidence_score_in_range(self):
        result = coordinate_agents(
            agent_ids=self._agents(3),
            strategy="consensus",
            market_conditions={},
        )
        assert 0.0 <= result["confidence_score"] <= 1.0

    def test_rationale_is_string(self):
        result = coordinate_agents(
            agent_ids=self._agents(3),
            strategy="consensus",
            market_conditions={},
        )
        assert isinstance(result["rationale"], str)
        assert len(result["rationale"]) > 0

    def test_invalid_strategy_raises(self):
        with pytest.raises(ValueError):
            coordinate_agents(
                agent_ids=self._agents(3),
                strategy="magic",
                market_conditions={},
            )

    def test_empty_agent_ids_raises(self):
        with pytest.raises(ValueError):
            coordinate_agents(
                agent_ids=[],
                strategy="consensus",
                market_conditions={},
            )

    def test_too_many_agents_raises(self):
        with pytest.raises(ValueError):
            coordinate_agents(
                agent_ids=[f"agent-{i}" for i in range(11)],
                strategy="consensus",
                market_conditions={},
            )

    def test_market_conditions_passed_through(self):
        mc = {"symbol": "SOL/USD", "trend": "bullish", "volatility": 0.03}
        result = coordinate_agents(
            agent_ids=self._agents(3),
            strategy="consensus",
            market_conditions=mc,
        )
        assert result["market_conditions"] == mc

    def test_symbol_in_result(self):
        result = coordinate_agents(
            agent_ids=self._agents(2),
            strategy="independent",
            market_conditions={"symbol": "ETH/USD"},
        )
        assert result["symbol"] == "ETH/USD"

    def test_total_agents_field(self):
        agents = self._agents(4)
        result = coordinate_agents(
            agent_ids=agents,
            strategy="hierarchical",
            market_conditions={},
        )
        assert result["total_agents"] == 4

    def test_has_generated_at(self):
        result = coordinate_agents(
            agent_ids=self._agents(2),
            strategy="consensus",
            market_conditions={},
        )
        assert "generated_at" in result
        assert result["generated_at"] > 0

    def test_all_three_strategies_valid(self):
        for strat in ("consensus", "independent", "hierarchical"):
            result = coordinate_agents(
                agent_ids=self._agents(3),
                strategy=strat,
                market_conditions={"symbol": "BTC/USD"},
            )
            assert result["decision"] in ("BUY", "SELL", "HOLD")

    def test_coord_strategies_set_contains_all(self):
        assert "consensus" in _COORD_STRATEGIES
        assert "independent" in _COORD_STRATEGIES
        assert "hierarchical" in _COORD_STRATEGIES

    def test_single_agent_consensus(self):
        result = coordinate_agents(
            agent_ids=["solo-agent-001"],
            strategy="consensus",
            market_conditions={},
        )
        assert result["total_agents"] == 1
        assert result["decision"] in ("BUY", "SELL", "HOLD")


# ── Unit Tests: get_agent_history ─────────────────────────────────────────────

class TestGetAgentHistory:
    def _prime_history(self, agent_id: str, n: int = 25) -> None:
        """Push N fake events for an agent into the system."""
        for i in range(n):
            evt = _generate_feed_event(event_type="agent_vote", seed_offset=i * 7 + 999)
            evt["agent_id"] = agent_id
            _push_feed_event(evt)

    def test_returns_dict_structure(self):
        self._prime_history("agent-hist-001", 5)
        result = get_agent_history("agent-hist-001")
        assert "history" in result
        assert "page" in result
        assert "limit" in result
        assert "total_events" in result
        assert "total_pages" in result

    def test_pagination_page_1(self):
        self._prime_history("agent-hist-002", 30)
        result = get_agent_history("agent-hist-002", page=1, limit=10)
        assert result["page"] == 1
        assert len(result["history"]) == 10

    def test_pagination_page_2(self):
        self._prime_history("agent-hist-003", 30)
        result = get_agent_history("agent-hist-003", page=2, limit=10)
        assert result["page"] == 2
        assert len(result["history"]) <= 10

    def test_has_next_when_more_pages(self):
        self._prime_history("agent-hist-004", 25)
        result = get_agent_history("agent-hist-004", page=1, limit=10)
        assert result["has_next"] is True

    def test_no_next_on_last_page(self):
        self._prime_history("agent-hist-005", 5)
        result = get_agent_history("agent-hist-005", page=1, limit=20)
        assert result["has_next"] is False

    def test_has_prev_on_page_2(self):
        self._prime_history("agent-hist-006", 30)
        result = get_agent_history("agent-hist-006", page=2, limit=10)
        assert result["has_prev"] is True

    def test_no_prev_on_page_1(self):
        self._prime_history("agent-hist-007", 10)
        result = get_agent_history("agent-hist-007", page=1, limit=10)
        assert result["has_prev"] is False

    def test_unknown_agent_returns_empty_not_error(self):
        result = get_agent_history("agent-does-not-exist-xyz")
        assert "history" in result
        assert isinstance(result["history"], list)
        assert result["total_events"] >= 0

    def test_agent_id_in_result(self):
        self._prime_history("agent-hist-008", 5)
        result = get_agent_history("agent-hist-008")
        assert result["agent_id"] == "agent-hist-008"

    def test_generated_at_present(self):
        result = get_agent_history("agent-hist-000")
        assert "generated_at" in result


# ── Integration: POST /demo/agents/coordinate ─────────────────────────────────

class TestCoordinateEndpoint:
    def test_basic_consensus(self, server):
        base_url, _ = server
        result = _post(f"{base_url}/demo/agents/coordinate", {
            "agent_ids": ["agent-a", "agent-b", "agent-c"],
            "strategy": "consensus",
            "market_conditions": {"symbol": "BTC/USD"},
        })
        assert result["decision"] in ("BUY", "SELL", "HOLD")
        assert result["strategy"] == "consensus"

    def test_independent_strategy_http(self, server):
        base_url, _ = server
        result = _post(f"{base_url}/demo/agents/coordinate", {
            "agent_ids": ["a1", "a2", "a3", "a4"],
            "strategy": "independent",
            "market_conditions": {},
        })
        assert "vote_tally" in result
        assert result["total_agents"] == 4

    def test_hierarchical_strategy_http(self, server):
        base_url, _ = server
        result = _post(f"{base_url}/demo/agents/coordinate", {
            "agent_ids": ["a1", "a2"],
            "strategy": "hierarchical",
            "market_conditions": {"trend": "bearish"},
        })
        assert result["strategy"] == "hierarchical"

    def test_invalid_strategy_returns_400(self, server):
        base_url, _ = server
        try:
            _post(f"{base_url}/demo/agents/coordinate", {
                "agent_ids": ["a1"],
                "strategy": "magic",
                "market_conditions": {},
            })
            assert False, "Should have raised"
        except HTTPError as e:
            assert e.code == 400

    def test_empty_agent_ids_returns_400(self, server):
        base_url, _ = server
        try:
            _post(f"{base_url}/demo/agents/coordinate", {
                "agent_ids": [],
                "strategy": "consensus",
                "market_conditions": {},
            })
            assert False, "Should have raised"
        except HTTPError as e:
            assert e.code == 400

    def test_missing_agent_ids_returns_400(self, server):
        base_url, _ = server
        try:
            _post(f"{base_url}/demo/agents/coordinate", {
                "strategy": "consensus",
                "market_conditions": {},
            })
            assert False, "Should have raised"
        except HTTPError as e:
            assert e.code == 400


# ── Integration: GET /demo/agents/{id}/history ────────────────────────────────

class TestAgentHistoryEndpoint:
    def test_known_agent_returns_200(self, server):
        base_url, _ = server
        agent_id = _FEED_AGENT_IDS[0]
        result = _get(f"{base_url}/demo/agents/{agent_id}/history")
        assert "history" in result
        assert result["agent_id"] == agent_id

    def test_unknown_agent_returns_empty(self, server):
        base_url, _ = server
        result = _get(f"{base_url}/demo/agents/no-such-agent-xyz/history")
        assert result["total_events"] == 0
        assert result["history"] == []

    def test_pagination_limit_param(self, server):
        base_url, _ = server
        agent_id = _FEED_AGENT_IDS[0]
        result = _get(f"{base_url}/demo/agents/{agent_id}/history?limit=5")
        assert result["limit"] == 5
        assert len(result["history"]) <= 5

    def test_page_param(self, server):
        base_url, _ = server
        agent_id = _FEED_AGENT_IDS[0]
        result = _get(f"{base_url}/demo/agents/{agent_id}/history?page=1&limit=5")
        assert result["page"] == 1

    def test_total_pages_computed(self, server):
        base_url, _ = server
        agent_id = _FEED_AGENT_IDS[0]
        result = _get(f"{base_url}/demo/agents/{agent_id}/history?limit=5")
        assert result["total_pages"] >= 1


# ── Integration: WebSocket /demo/ws/feed ──────────────────────────────────────

class TestWebSocketFeed:
    def test_non_ws_request_returns_426(self, server):
        """Non-WebSocket GET to /demo/ws/feed should get 426."""
        base_url, _ = server
        try:
            _get(f"{base_url}/demo/ws/feed")
            assert False, "Should have raised"
        except HTTPError as e:
            assert e.code == 426

    def test_ws_handshake_returns_101(self, server):
        """WebSocket handshake should succeed with 101."""
        _, port = server
        sock, key, response = _ws_connect("localhost", port, "/demo/ws/feed")
        try:
            assert "101" in response
            assert "Switching Protocols" in response
        finally:
            sock.close()

    def test_ws_accept_header_correct(self, server):
        """Server should send correct Sec-WebSocket-Accept header."""
        _, port = server
        sock, key, response = _ws_connect("localhost", port, "/demo/ws/feed")
        try:
            expected_accept = _ws_accept_key(key)
            assert expected_accept in response
        finally:
            sock.close()

    def test_ws_receives_welcome_frame(self, server):
        """Server should send a welcome event immediately after handshake."""
        _, port = server
        sock, key, response = _ws_connect("localhost", port, "/demo/ws/feed")
        try:
            sock.settimeout(5.0)
            frame_text = _ws_read_frame(sock)
            data = json.loads(frame_text)
            assert data["event"] == "connected"
            assert "message" in data
        finally:
            sock.close()

    def test_ws_welcome_contains_event_types(self, server):
        """Welcome frame should list the feed event types."""
        _, port = server
        sock, key, response = _ws_connect("localhost", port, "/demo/ws/feed")
        try:
            sock.settimeout(5.0)
            frame_text = _ws_read_frame(sock)
            data = json.loads(frame_text)
            assert "events" in data
            assert isinstance(data["events"], list)
            assert len(data["events"]) > 0
        finally:
            sock.close()

    def test_ws_broadcast_delivers_to_client(self, server):
        """Events pushed to the feed should be delivered over WS."""
        _, port = server
        sock, key, response = _ws_connect("localhost", port, "/demo/ws/feed")
        try:
            sock.settimeout(5.0)
            # Read welcome frame
            _ws_read_frame(sock)
            # Push an event
            test_evt = {
                "event": "agent_vote",
                "agent_id": "agent-ws-test-001",
                "seq": 99999,
                "symbol": "BTC/USD",
                "timestamp": time.time(),
            }
            _push_feed_event(test_evt)
            time.sleep(0.1)
            # Read next frame
            frame_text = _ws_read_frame(sock)
            data = json.loads(frame_text)
            assert data.get("agent_id") == "agent-ws-test-001"
        finally:
            sock.close()

    def test_ws_broadcast_function_queues_to_clients(self):
        """_ws_broadcast should put payload in all connected client queues."""
        test_q: queue.Queue = queue.Queue()
        with _ws_clients_lock:
            _ws_clients.append(test_q)
        try:
            _ws_broadcast('{"test": true}')
            payload = test_q.get(timeout=1.0)
            assert json.loads(payload) == {"test": True}
        finally:
            with _ws_clients_lock:
                if test_q in _ws_clients:
                    _ws_clients.remove(test_q)
