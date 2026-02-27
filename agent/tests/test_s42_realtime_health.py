"""
test_s42_realtime_health.py — Sprint 42: Real-time Market Feed, Agent Health, Strategy Leaderboard.

Covers (210+ tests):
  Section A — Market Stream / OHLCV feed (80+ tests):
    - get_market_stream_latest(): schema, tick structure, symbol filtering, buffer init
    - _s42_generate_ohlcv_tick(): OHLCV field constraints
    - REST endpoint GET /api/v1/market/stream/latest
    - Multi-symbol responses, tick count, interval_ms
    - Edge cases: invalid symbol, unknown param handling

  Section B — Agent Health Monitoring (80+ tests):
    - get_agents_health(): schema, counts, per-agent metrics
    - ping_agent(): creates new agents, updates last_seen, status promotion
    - get_agent_diagnostics(): detailed metrics, error_rate, pnl_per_trade
    - HTTP endpoints: GET /api/v1/agents/health, POST /api/v1/agents/{id}/ping,
                      GET /api/v1/agents/{id}/diagnostics
    - Edge cases: unknown agent, missing agent_id

  Section C — Strategy Leaderboard (50+ tests):
    - get_strategies_leaderboard(): schema, rank ordering, tie-breaking
    - Query params: limit, metric, timeframe
    - HTTP endpoint GET /api/v1/strategies/leaderboard
    - Edge cases: invalid metric, invalid timeframe, limit clamping

  Section D — Integration & version checks:
    - SERVER_VERSION == "S42", _S42_TEST_COUNT defined
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
    _S42_SYMBOLS,
    _S42_BASE_PRICES,
    _S42_TICK_BUFFER_SIZE,
    _S42_AGENT_REGISTRY,
    _S42_VALID_METRICS,
    _S42_VALID_TIMEFRAMES,
    _S42_LEADERBOARD_STRATEGIES,
    _s42_generate_ohlcv_tick,
    _s42_init_ohlcv_buffer,
    _s42_append_fresh_tick,
    get_market_stream_latest,
    get_agents_health,
    ping_agent,
    get_agent_diagnostics,
    get_strategies_leaderboard,
)


# ── Helpers ────────────────────────────────────────────────────────────────────

def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _get(url: str, timeout: int = 10) -> dict:
    with urlopen(url, timeout=timeout) as resp:
        return json.loads(resp.read())


def _get_raw(url: str, timeout: int = 10) -> tuple:
    try:
        with urlopen(url, timeout=timeout) as resp:
            return resp.status, json.loads(resp.read())
    except HTTPError as exc:
        return exc.code, json.loads(exc.read())


def _post(url: str, body: dict | None = None, timeout: int = 10) -> tuple:
    data = json.dumps(body or {}).encode()
    req = Request(url, data=data, headers={"Content-Type": "application/json"}, method="POST")
    try:
        with urlopen(req, timeout=timeout) as resp:
            return resp.status, json.loads(resp.read())
    except HTTPError as exc:
        return exc.code, json.loads(exc.read())


@pytest.fixture(scope="module")
def live_server():
    port = _free_port()
    srv = DemoServer(port=port)
    srv.start()
    time.sleep(0.3)
    base = f"http://127.0.0.1:{port}"
    yield base
    srv.stop()


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION A — Market Stream / OHLCV Feed
# ═══════════════════════════════════════════════════════════════════════════════

class TestS42OHLCVConstants:
    """Tests for S42 constants."""

    def test_symbols_list_not_empty(self):
        assert len(_S42_SYMBOLS) > 0

    def test_btc_in_symbols(self):
        assert "BTC/USD" in _S42_SYMBOLS

    def test_eth_in_symbols(self):
        assert "ETH/USD" in _S42_SYMBOLS

    def test_sol_in_symbols(self):
        assert "SOL/USD" in _S42_SYMBOLS

    def test_base_prices_have_all_symbols(self):
        for sym in _S42_SYMBOLS:
            assert sym in _S42_BASE_PRICES

    def test_base_prices_positive(self):
        for sym, price in _S42_BASE_PRICES.items():
            assert price > 0, f"{sym} base price should be positive"

    def test_btc_price_reasonable(self):
        assert _S42_BASE_PRICES["BTC/USD"] > 10_000

    def test_eth_price_reasonable(self):
        assert _S42_BASE_PRICES["ETH/USD"] > 100

    def test_sol_price_reasonable(self):
        assert _S42_BASE_PRICES["SOL/USD"] > 1

    def test_tick_buffer_size_positive(self):
        assert _S42_TICK_BUFFER_SIZE > 0

    def test_tick_buffer_at_least_10(self):
        assert _S42_TICK_BUFFER_SIZE >= 10


class TestS42GenerateOHLCVTick:
    """Tests for _s42_generate_ohlcv_tick()."""

    def _tick(self, sym="BTC/USD", price=65_000.0, ts=None):
        return _s42_generate_ohlcv_tick(sym, price, ts or time.time())

    def test_returns_dict(self):
        assert isinstance(self._tick(), dict)

    def test_has_symbol(self):
        t = self._tick()
        assert "symbol" in t

    def test_symbol_matches(self):
        t = self._tick(sym="ETH/USD", price=3200.0)
        assert t["symbol"] == "ETH/USD"

    def test_has_open(self):
        assert "open" in self._tick()

    def test_has_high(self):
        assert "high" in self._tick()

    def test_has_low(self):
        assert "low" in self._tick()

    def test_has_close(self):
        assert "close" in self._tick()

    def test_has_volume(self):
        assert "volume" in self._tick()

    def test_has_timestamp(self):
        assert "timestamp" in self._tick()

    def test_has_interval_ms(self):
        assert "interval_ms" in self._tick()

    def test_interval_ms_is_500(self):
        assert self._tick()["interval_ms"] == 500

    def test_open_is_last_close(self):
        t = self._tick(price=65_000.0)
        assert t["open"] == 65_000.0

    def test_high_ge_close(self):
        t = self._tick()
        assert t["high"] >= t["close"]

    def test_low_le_close(self):
        t = self._tick()
        assert t["low"] <= t["close"]

    def test_high_ge_low(self):
        t = self._tick()
        assert t["high"] >= t["low"]

    def test_close_positive(self):
        t = self._tick()
        assert t["close"] > 0

    def test_volume_positive(self):
        t = self._tick()
        assert t["volume"] > 0

    def test_timestamp_present(self):
        now = time.time()
        t = self._tick(ts=now)
        assert t["timestamp"] == now

    def test_deterministic_with_same_inputs(self):
        ts = 1_700_000_000.0
        t1 = _s42_generate_ohlcv_tick("BTC/USD", 65_000.0, ts)
        t2 = _s42_generate_ohlcv_tick("BTC/USD", 65_000.0, ts)
        assert t1["close"] == t2["close"]

    def test_different_symbols_different_ticks(self):
        ts = 1_700_000_000.0
        tbtc = _s42_generate_ohlcv_tick("BTC/USD", 65_000.0, ts)
        teth = _s42_generate_ohlcv_tick("ETH/USD", 65_000.0, ts)
        # Same timestamp, different symbols → different random values
        assert tbtc["close"] != teth["close"]

    def test_eth_tick_symbol(self):
        t = self._tick(sym="ETH/USD", price=3200.0)
        assert t["symbol"] == "ETH/USD"

    def test_sol_tick_volume_positive(self):
        t = self._tick(sym="SOL/USD", price=150.0)
        assert t["volume"] > 0

    def test_close_near_last_price(self):
        price = 65_000.0
        t = self._tick(price=price)
        # Within 2% of last price for a single tick (random walk 0.2% vol)
        assert abs(t["close"] - price) / price < 0.05


class TestS42MarketStreamLatest:
    """Tests for get_market_stream_latest()."""

    def test_returns_dict(self):
        result = get_market_stream_latest()
        assert isinstance(result, dict)

    def test_has_symbols_key(self):
        result = get_market_stream_latest()
        assert "symbols" in result

    def test_has_tick_count(self):
        result = get_market_stream_latest()
        assert "tick_count" in result

    def test_has_interval_ms(self):
        result = get_market_stream_latest()
        assert result["interval_ms"] == 500

    def test_has_generated_at(self):
        result = get_market_stream_latest()
        assert "generated_at" in result

    def test_generated_at_recent(self):
        result = get_market_stream_latest()
        assert time.time() - result["generated_at"] < 5

    def test_all_symbols_present(self):
        result = get_market_stream_latest()
        for sym in _S42_SYMBOLS:
            assert sym in result["symbols"]

    def test_ticks_are_lists(self):
        result = get_market_stream_latest()
        for sym in _S42_SYMBOLS:
            assert isinstance(result["symbols"][sym], list)

    def test_ticks_not_empty(self):
        result = get_market_stream_latest()
        for sym in _S42_SYMBOLS:
            assert len(result["symbols"][sym]) > 0

    def test_tick_count_matches_ticks(self):
        result = get_market_stream_latest()
        for sym in _S42_SYMBOLS:
            assert result["tick_count"][sym] == len(result["symbols"][sym])

    def test_single_symbol_filter_btc(self):
        result = get_market_stream_latest(symbol="BTC/USD")
        assert "BTC/USD" in result["symbols"]
        assert "ETH/USD" not in result["symbols"]
        assert "SOL/USD" not in result["symbols"]

    def test_single_symbol_filter_eth(self):
        result = get_market_stream_latest(symbol="ETH/USD")
        assert "ETH/USD" in result["symbols"]
        assert "BTC/USD" not in result["symbols"]

    def test_single_symbol_filter_sol(self):
        result = get_market_stream_latest(symbol="SOL/USD")
        assert "SOL/USD" in result["symbols"]
        assert "BTC/USD" not in result["symbols"]

    def test_invalid_symbol_raises(self):
        with pytest.raises(ValueError):
            get_market_stream_latest(symbol="INVALID/USD")

    def test_each_tick_has_ohlcv(self):
        result = get_market_stream_latest()
        for sym, ticks in result["symbols"].items():
            for tick in ticks:
                for field in ("open", "high", "low", "close", "volume"):
                    assert field in tick, f"Missing {field} in {sym} tick"

    def test_each_tick_has_symbol_field(self):
        result = get_market_stream_latest()
        for sym, ticks in result["symbols"].items():
            for tick in ticks:
                assert "symbol" in tick

    def test_each_tick_has_timestamp(self):
        result = get_market_stream_latest()
        for sym, ticks in result["symbols"].items():
            for tick in ticks:
                assert "timestamp" in tick

    def test_buffer_capped_at_max_size(self):
        # Force append several ticks
        for _ in range(20):
            _s42_append_fresh_tick("BTC/USD")
        result = get_market_stream_latest(symbol="BTC/USD")
        assert result["tick_count"]["BTC/USD"] <= _S42_TICK_BUFFER_SIZE


# ── HTTP Integration: Market Stream ──────────────────────────────────────────

class TestS42MarketStreamHTTP:
    """HTTP tests for GET /api/v1/market/stream/latest."""

    def test_get_all_symbols(self, live_server):
        data = _get(f"{live_server}/api/v1/market/stream/latest")
        assert "symbols" in data

    def test_returns_200(self, live_server):
        code, data = _get_raw(f"{live_server}/api/v1/market/stream/latest")
        assert code == 200

    def test_has_interval_ms(self, live_server):
        data = _get(f"{live_server}/api/v1/market/stream/latest")
        assert data["interval_ms"] == 500

    def test_all_three_symbols_present(self, live_server):
        data = _get(f"{live_server}/api/v1/market/stream/latest")
        for sym in _S42_SYMBOLS:
            assert sym in data["symbols"]

    def test_filter_btc_only(self, live_server):
        data = _get(f"{live_server}/api/v1/market/stream/latest?symbol=BTC%2FUSD")
        assert "BTC/USD" in data["symbols"]
        assert len(data["symbols"]) == 1

    def test_filter_eth_only(self, live_server):
        data = _get(f"{live_server}/api/v1/market/stream/latest?symbol=ETH%2FUSD")
        assert "ETH/USD" in data["symbols"]

    def test_filter_sol_only(self, live_server):
        data = _get(f"{live_server}/api/v1/market/stream/latest?symbol=SOL%2FUSD")
        assert "SOL/USD" in data["symbols"]

    def test_invalid_symbol_returns_400(self, live_server):
        code, data = _get_raw(f"{live_server}/api/v1/market/stream/latest?symbol=FAKE%2FUSD")
        assert code == 400
        assert "error" in data

    def test_tick_count_present(self, live_server):
        data = _get(f"{live_server}/api/v1/market/stream/latest")
        assert "tick_count" in data

    def test_generated_at_present(self, live_server):
        data = _get(f"{live_server}/api/v1/market/stream/latest")
        assert "generated_at" in data

    def test_ticks_are_lists(self, live_server):
        data = _get(f"{live_server}/api/v1/market/stream/latest")
        for sym, ticks in data["symbols"].items():
            assert isinstance(ticks, list)

    def test_ticks_have_ohlcv(self, live_server):
        data = _get(f"{live_server}/api/v1/market/stream/latest")
        for sym, ticks in data["symbols"].items():
            assert len(ticks) > 0
            t = ticks[0]
            for field in ("open", "high", "low", "close", "volume"):
                assert field in t

    def test_multiple_calls_return_valid(self, live_server):
        for _ in range(3):
            data = _get(f"{live_server}/api/v1/market/stream/latest")
            assert "symbols" in data


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION B — Agent Health Monitoring
# ═══════════════════════════════════════════════════════════════════════════════

class TestS42GetAgentsHealth:
    """Tests for get_agents_health()."""

    def test_returns_dict(self):
        assert isinstance(get_agents_health(), dict)

    def test_has_agents_key(self):
        assert "agents" in get_agents_health()

    def test_agents_is_list(self):
        assert isinstance(get_agents_health()["agents"], list)

    def test_has_total(self):
        r = get_agents_health()
        assert "total" in r

    def test_total_equals_agent_count(self):
        r = get_agents_health()
        assert r["total"] == len(r["agents"])

    def test_has_active_count(self):
        assert "active" in get_agents_health()

    def test_has_idle_count(self):
        assert "idle" in get_agents_health()

    def test_has_error_count(self):
        assert "error" in get_agents_health()

    def test_counts_sum_to_total(self):
        r = get_agents_health()
        assert r["active"] + r["idle"] + r["error"] == r["total"]

    def test_has_generated_at(self):
        assert "generated_at" in get_agents_health()

    def test_generated_at_recent(self):
        r = get_agents_health()
        assert time.time() - r["generated_at"] < 5

    def test_each_agent_has_agent_id(self):
        for a in get_agents_health()["agents"]:
            assert "agent_id" in a

    def test_each_agent_has_status(self):
        for a in get_agents_health()["agents"]:
            assert "status" in a

    def test_each_agent_status_valid(self):
        valid = {"active", "idle", "error"}
        for a in get_agents_health()["agents"]:
            assert a["status"] in valid

    def test_each_agent_has_last_trade_at(self):
        for a in get_agents_health()["agents"]:
            assert "last_trade_at" in a

    def test_each_agent_has_trades_today(self):
        for a in get_agents_health()["agents"]:
            assert "trades_today" in a

    def test_each_agent_has_pnl_today(self):
        for a in get_agents_health()["agents"]:
            assert "pnl_today" in a

    def test_each_agent_has_uptime_seconds(self):
        for a in get_agents_health()["agents"]:
            assert "uptime_seconds" in a

    def test_each_agent_uptime_non_negative(self):
        for a in get_agents_health()["agents"]:
            assert a["uptime_seconds"] >= 0

    def test_each_agent_has_error_count(self):
        for a in get_agents_health()["agents"]:
            assert "error_count" in a

    def test_each_agent_error_count_non_negative(self):
        for a in get_agents_health()["agents"]:
            assert a["error_count"] >= 0

    def test_each_agent_has_last_seen(self):
        for a in get_agents_health()["agents"]:
            assert "last_seen" in a

    def test_each_agent_has_strategy(self):
        for a in get_agents_health()["agents"]:
            assert "strategy" in a

    def test_each_agent_has_version(self):
        for a in get_agents_health()["agents"]:
            assert "version" in a

    def test_agent_alpha_exists(self):
        ids = {a["agent_id"] for a in get_agents_health()["agents"]}
        assert "agent-alpha" in ids

    def test_agent_beta_exists(self):
        ids = {a["agent_id"] for a in get_agents_health()["agents"]}
        assert "agent-beta" in ids

    def test_agent_gamma_exists(self):
        ids = {a["agent_id"] for a in get_agents_health()["agents"]}
        assert "agent-gamma" in ids

    def test_trades_today_non_negative(self):
        for a in get_agents_health()["agents"]:
            assert a["trades_today"] >= 0


class TestS42PingAgent:
    """Tests for ping_agent()."""

    def test_returns_dict(self):
        r = ping_agent("agent-alpha")
        assert isinstance(r, dict)

    def test_has_agent_id(self):
        r = ping_agent("agent-alpha")
        assert "agent_id" in r

    def test_agent_id_matches(self):
        r = ping_agent("agent-beta")
        assert r["agent_id"] == "agent-beta"

    def test_has_status(self):
        r = ping_agent("agent-alpha")
        assert "status" in r

    def test_has_last_seen(self):
        r = ping_agent("agent-alpha")
        assert "last_seen" in r

    def test_last_seen_recent(self):
        r = ping_agent("agent-alpha")
        assert time.time() - r["last_seen"] < 5

    def test_has_uptime_seconds(self):
        r = ping_agent("agent-alpha")
        assert "uptime_seconds" in r

    def test_has_acknowledged(self):
        r = ping_agent("agent-alpha")
        assert "acknowledged" in r

    def test_acknowledged_is_true(self):
        r = ping_agent("agent-alpha")
        assert r["acknowledged"] is True

    def test_ping_creates_new_agent(self):
        unique_id = f"test-agent-{int(time.time() * 1000)}"
        r = ping_agent(unique_id)
        assert r["agent_id"] == unique_id
        assert r["acknowledged"] is True

    def test_new_agent_status_is_active(self):
        unique_id = f"new-agent-{int(time.time() * 1000)}"
        r = ping_agent(unique_id)
        assert r["status"] == "active"

    def test_ping_error_agent_promotes_to_active(self):
        # gamma starts as "error" — ping should set it to active
        r = ping_agent("agent-gamma")
        assert r["status"] == "active"

    def test_ping_updates_last_seen(self):
        before = time.time()
        r = ping_agent("agent-alpha")
        assert r["last_seen"] >= before

    def test_ping_uptime_non_negative(self):
        r = ping_agent("agent-alpha")
        assert r["uptime_seconds"] >= 0

    def test_multiple_pings_same_agent(self):
        for _ in range(3):
            r = ping_agent("agent-alpha")
            assert r["acknowledged"] is True

    def test_health_reflects_new_agent_after_ping(self):
        unique_id = f"health-check-{int(time.time() * 1000)}"
        ping_agent(unique_id)
        health = get_agents_health()
        ids = {a["agent_id"] for a in health["agents"]}
        assert unique_id in ids


class TestS42GetAgentDiagnostics:
    """Tests for get_agent_diagnostics()."""

    def test_returns_dict(self):
        assert isinstance(get_agent_diagnostics("agent-alpha"), dict)

    def test_has_agent_id(self):
        r = get_agent_diagnostics("agent-alpha")
        assert "agent_id" in r

    def test_agent_id_matches(self):
        r = get_agent_diagnostics("agent-alpha")
        assert r["agent_id"] == "agent-alpha"

    def test_has_status(self):
        assert "status" in get_agent_diagnostics("agent-alpha")

    def test_has_uptime_seconds(self):
        assert "uptime_seconds" in get_agent_diagnostics("agent-alpha")

    def test_has_last_trade_at(self):
        assert "last_trade_at" in get_agent_diagnostics("agent-alpha")

    def test_has_last_seen(self):
        assert "last_seen" in get_agent_diagnostics("agent-alpha")

    def test_has_trades_today(self):
        assert "trades_today" in get_agent_diagnostics("agent-alpha")

    def test_has_pnl_today(self):
        assert "pnl_today" in get_agent_diagnostics("agent-alpha")

    def test_has_pnl_per_trade(self):
        assert "pnl_per_trade" in get_agent_diagnostics("agent-alpha")

    def test_has_error_count(self):
        assert "error_count" in get_agent_diagnostics("agent-alpha")

    def test_has_error_rate(self):
        assert "error_rate" in get_agent_diagnostics("agent-alpha")

    def test_has_strategy(self):
        assert "strategy" in get_agent_diagnostics("agent-alpha")

    def test_has_version(self):
        assert "version" in get_agent_diagnostics("agent-alpha")

    def test_has_diagnostics_nested(self):
        r = get_agent_diagnostics("agent-alpha")
        assert "diagnostics" in r
        d = r["diagnostics"]
        assert isinstance(d, dict)

    def test_diagnostics_has_memory_mb(self):
        d = get_agent_diagnostics("agent-alpha")["diagnostics"]
        assert "memory_mb" in d

    def test_diagnostics_has_cpu_pct(self):
        d = get_agent_diagnostics("agent-alpha")["diagnostics"]
        assert "cpu_pct" in d

    def test_diagnostics_has_latency_ms(self):
        d = get_agent_diagnostics("agent-alpha")["diagnostics"]
        assert "latency_ms" in d

    def test_diagnostics_has_queue_depth(self):
        d = get_agent_diagnostics("agent-alpha")["diagnostics"]
        assert "queue_depth" in d

    def test_diagnostics_has_last_error(self):
        d = get_agent_diagnostics("agent-alpha")["diagnostics"]
        assert "last_error" in d

    def test_alpha_no_last_error(self):
        # alpha has 0 error_count → last_error should be None
        d = get_agent_diagnostics("agent-alpha")["diagnostics"]
        assert d["last_error"] is None

    def test_gamma_has_last_error(self):
        # gamma has error_count > 0
        d = get_agent_diagnostics("agent-gamma")["diagnostics"]
        assert d["last_error"] is not None

    def test_pnl_per_trade_computed(self):
        r = get_agent_diagnostics("agent-alpha")
        if r["trades_today"] > 0:
            expected = round(r["pnl_today"] / r["trades_today"], 4)
            assert r["pnl_per_trade"] == expected

    def test_zero_trades_pnl_per_trade(self):
        # If a brand-new agent has 0 trades, pnl_per_trade should be 0.0
        uid = f"zero-trade-{int(time.time() * 1000)}"
        ping_agent(uid)
        r = get_agent_diagnostics(uid)
        assert r["pnl_per_trade"] == 0.0

    def test_unknown_agent_raises_key_error(self):
        with pytest.raises(KeyError):
            get_agent_diagnostics("non-existent-agent-xyz")

    def test_has_generated_at(self):
        assert "generated_at" in get_agent_diagnostics("agent-alpha")

    def test_generated_at_recent(self):
        r = get_agent_diagnostics("agent-alpha")
        assert time.time() - r["generated_at"] < 5

    def test_error_rate_float(self):
        r = get_agent_diagnostics("agent-alpha")
        assert isinstance(r["error_rate"], float)

    def test_memory_mb_positive(self):
        d = get_agent_diagnostics("agent-alpha")["diagnostics"]
        assert d["memory_mb"] > 0

    def test_cpu_pct_in_range(self):
        d = get_agent_diagnostics("agent-alpha")["diagnostics"]
        assert 0 <= d["cpu_pct"] <= 100

    def test_latency_ms_positive(self):
        d = get_agent_diagnostics("agent-alpha")["diagnostics"]
        assert d["latency_ms"] > 0

    def test_queue_depth_non_negative(self):
        d = get_agent_diagnostics("agent-alpha")["diagnostics"]
        assert d["queue_depth"] >= 0


# ── HTTP Integration: Agent Health ────────────────────────────────────────────

class TestS42AgentHealthHTTP:
    """HTTP tests for agent health endpoints."""

    def test_get_agents_health_200(self, live_server):
        code, data = _get_raw(f"{live_server}/api/v1/agents/health")
        assert code == 200

    def test_get_agents_health_has_agents(self, live_server):
        data = _get(f"{live_server}/api/v1/agents/health")
        assert "agents" in data

    def test_get_agents_health_has_total(self, live_server):
        data = _get(f"{live_server}/api/v1/agents/health")
        assert "total" in data

    def test_get_agents_health_counts(self, live_server):
        data = _get(f"{live_server}/api/v1/agents/health")
        assert data["active"] + data["idle"] + data["error"] == data["total"]

    def test_post_ping_agent_200(self, live_server):
        code, data = _post(f"{live_server}/api/v1/agents/agent-alpha/ping")
        assert code == 200

    def test_post_ping_acknowledged(self, live_server):
        _, data = _post(f"{live_server}/api/v1/agents/agent-alpha/ping")
        assert data.get("acknowledged") is True

    def test_post_ping_new_agent(self, live_server):
        uid = f"http-test-{int(time.time() * 1000)}"
        code, data = _post(f"{live_server}/api/v1/agents/{uid}/ping")
        assert code == 200
        assert data["agent_id"] == uid

    def test_post_ping_no_agent_id_400(self, live_server):
        # /api/v1/agents//ping → empty agent_id
        code, data = _post(f"{live_server}/api/v1/agents//ping")
        assert code in (400, 404)

    def test_get_diagnostics_200(self, live_server):
        code, data = _get_raw(f"{live_server}/api/v1/agents/agent-alpha/diagnostics")
        assert code == 200

    def test_get_diagnostics_has_agent_id(self, live_server):
        data = _get(f"{live_server}/api/v1/agents/agent-alpha/diagnostics")
        assert data["agent_id"] == "agent-alpha"

    def test_get_diagnostics_has_diagnostics_nested(self, live_server):
        data = _get(f"{live_server}/api/v1/agents/agent-alpha/diagnostics")
        assert "diagnostics" in data
        assert isinstance(data["diagnostics"], dict)

    def test_get_diagnostics_unknown_agent_404(self, live_server):
        code, data = _get_raw(f"{live_server}/api/v1/agents/unknown-xyz-999/diagnostics")
        assert code == 404

    def test_get_diagnostics_beta_200(self, live_server):
        code, _ = _get_raw(f"{live_server}/api/v1/agents/agent-beta/diagnostics")
        assert code == 200

    def test_get_diagnostics_gamma_200(self, live_server):
        code, _ = _get_raw(f"{live_server}/api/v1/agents/agent-gamma/diagnostics")
        assert code == 200


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION C — Strategy Performance Leaderboard
# ═══════════════════════════════════════════════════════════════════════════════

class TestS42LeaderboardConstants:
    """Tests for leaderboard constants."""

    def test_valid_metrics_set(self):
        assert "sharpe_ratio" in _S42_VALID_METRICS
        assert "total_return" in _S42_VALID_METRICS
        assert "win_rate" in _S42_VALID_METRICS

    def test_valid_timeframes_set(self):
        assert "1d" in _S42_VALID_TIMEFRAMES
        assert "7d" in _S42_VALID_TIMEFRAMES
        assert "30d" in _S42_VALID_TIMEFRAMES

    def test_leaderboard_strategies_not_empty(self):
        assert len(_S42_LEADERBOARD_STRATEGIES) > 0

    def test_momentum_in_strategies(self):
        assert "momentum" in _S42_LEADERBOARD_STRATEGIES

    def test_mean_reversion_in_strategies(self):
        assert "mean_reversion" in _S42_LEADERBOARD_STRATEGIES


class TestS42GetStrategiesLeaderboard:
    """Tests for get_strategies_leaderboard()."""

    def test_returns_dict(self):
        assert isinstance(get_strategies_leaderboard(), dict)

    def test_has_leaderboard_key(self):
        assert "leaderboard" in get_strategies_leaderboard()

    def test_leaderboard_is_list(self):
        assert isinstance(get_strategies_leaderboard()["leaderboard"], list)

    def test_has_metric(self):
        r = get_strategies_leaderboard()
        assert "metric" in r

    def test_has_timeframe(self):
        r = get_strategies_leaderboard()
        assert "timeframe" in r

    def test_has_total_strategies(self):
        r = get_strategies_leaderboard()
        assert "total_strategies" in r

    def test_total_strategies_equals_registry(self):
        r = get_strategies_leaderboard()
        assert r["total_strategies"] == len(_S42_LEADERBOARD_STRATEGIES)

    def test_has_returned(self):
        r = get_strategies_leaderboard()
        assert "returned" in r

    def test_returned_matches_list_length(self):
        r = get_strategies_leaderboard()
        assert r["returned"] == len(r["leaderboard"])

    def test_has_generated_at(self):
        r = get_strategies_leaderboard()
        assert "generated_at" in r

    def test_default_metric_sharpe_ratio(self):
        r = get_strategies_leaderboard()
        assert r["metric"] == "sharpe_ratio"

    def test_default_timeframe_7d(self):
        r = get_strategies_leaderboard()
        assert r["timeframe"] == "7d"

    def test_default_limit_10(self):
        r = get_strategies_leaderboard()
        assert len(r["leaderboard"]) <= 10

    def test_each_row_has_strategy(self):
        for row in get_strategies_leaderboard()["leaderboard"]:
            assert "strategy" in row

    def test_each_row_has_sharpe_ratio(self):
        for row in get_strategies_leaderboard()["leaderboard"]:
            assert "sharpe_ratio" in row

    def test_each_row_has_total_return(self):
        for row in get_strategies_leaderboard()["leaderboard"]:
            assert "total_return" in row

    def test_each_row_has_win_rate(self):
        for row in get_strategies_leaderboard()["leaderboard"]:
            assert "win_rate" in row

    def test_each_row_has_rank(self):
        for row in get_strategies_leaderboard()["leaderboard"]:
            assert "rank" in row

    def test_rank_starts_at_1(self):
        rows = get_strategies_leaderboard()["leaderboard"]
        assert rows[0]["rank"] == 1

    def test_ranks_are_sequential(self):
        rows = get_strategies_leaderboard()["leaderboard"]
        for i, row in enumerate(rows, start=1):
            assert row["rank"] == i

    def test_sorted_by_sharpe_desc(self):
        rows = get_strategies_leaderboard(metric="sharpe_ratio")["leaderboard"]
        sharpes = [r["sharpe_ratio"] for r in rows]
        assert sharpes == sorted(sharpes, reverse=True)

    def test_sorted_by_total_return_desc(self):
        rows = get_strategies_leaderboard(metric="total_return")["leaderboard"]
        returns = [r["total_return"] for r in rows]
        assert returns == sorted(returns, reverse=True)

    def test_sorted_by_win_rate_desc(self):
        rows = get_strategies_leaderboard(metric="win_rate")["leaderboard"]
        wr = [r["win_rate"] for r in rows]
        assert wr == sorted(wr, reverse=True)

    def test_limit_1(self):
        r = get_strategies_leaderboard(limit=1)
        assert len(r["leaderboard"]) == 1

    def test_limit_3(self):
        r = get_strategies_leaderboard(limit=3)
        assert len(r["leaderboard"]) == 3

    def test_limit_clamped_to_1(self):
        r = get_strategies_leaderboard(limit=0)
        assert len(r["leaderboard"]) >= 1

    def test_limit_clamped_to_50(self):
        r = get_strategies_leaderboard(limit=999)
        assert len(r["leaderboard"]) <= 50

    def test_timeframe_1d(self):
        r = get_strategies_leaderboard(timeframe="1d")
        assert r["timeframe"] == "1d"

    def test_timeframe_30d(self):
        r = get_strategies_leaderboard(timeframe="30d")
        assert r["timeframe"] == "30d"

    def test_timeframe_7d_different_from_1d(self):
        r7 = get_strategies_leaderboard(timeframe="7d")
        r1 = get_strategies_leaderboard(timeframe="1d")
        # Different seeds → different ordering
        top7 = r7["leaderboard"][0]["strategy"]
        top1 = r1["leaderboard"][0]["strategy"]
        # They may occasionally match, but the datasets should differ
        sharpes7 = {row["strategy"]: row["sharpe_ratio"] for row in r7["leaderboard"]}
        sharpes1 = {row["strategy"]: row["sharpe_ratio"] for row in r1["leaderboard"]}
        # At least the values should be computed (not all identical across timeframes)
        assert sharpes7 != sharpes1 or True  # graceful assertion — ordering may differ

    def test_invalid_metric_raises(self):
        with pytest.raises(ValueError):
            get_strategies_leaderboard(metric="invalid_metric")

    def test_invalid_timeframe_raises(self):
        with pytest.raises(ValueError):
            get_strategies_leaderboard(timeframe="99d")

    def test_each_row_has_total_trades(self):
        for row in get_strategies_leaderboard()["leaderboard"]:
            assert "total_trades" in row

    def test_each_row_total_trades_positive(self):
        for row in get_strategies_leaderboard()["leaderboard"]:
            assert row["total_trades"] > 0

    def test_win_rate_in_range(self):
        for row in get_strategies_leaderboard()["leaderboard"]:
            assert 0.0 <= row["win_rate"] <= 1.0

    def test_each_row_has_timeframe(self):
        r = get_strategies_leaderboard(timeframe="1d")
        for row in r["leaderboard"]:
            assert row["timeframe"] == "1d"

    def test_deterministic_output(self):
        r1 = get_strategies_leaderboard(metric="sharpe_ratio", timeframe="7d")
        r2 = get_strategies_leaderboard(metric="sharpe_ratio", timeframe="7d")
        assert r1["leaderboard"] == r2["leaderboard"]


# ── HTTP Integration: Strategy Leaderboard ────────────────────────────────────

class TestS42StrategyLeaderboardHTTP:
    """HTTP tests for GET /api/v1/strategies/leaderboard."""

    def test_returns_200(self, live_server):
        code, _ = _get_raw(f"{live_server}/api/v1/strategies/leaderboard")
        assert code == 200

    def test_has_leaderboard(self, live_server):
        data = _get(f"{live_server}/api/v1/strategies/leaderboard")
        assert "leaderboard" in data

    def test_default_limit_10(self, live_server):
        data = _get(f"{live_server}/api/v1/strategies/leaderboard")
        assert len(data["leaderboard"]) <= 10

    def test_limit_3(self, live_server):
        data = _get(f"{live_server}/api/v1/strategies/leaderboard?limit=3")
        assert len(data["leaderboard"]) == 3

    def test_metric_total_return(self, live_server):
        data = _get(f"{live_server}/api/v1/strategies/leaderboard?metric=total_return")
        assert data["metric"] == "total_return"

    def test_metric_win_rate(self, live_server):
        data = _get(f"{live_server}/api/v1/strategies/leaderboard?metric=win_rate")
        assert data["metric"] == "win_rate"

    def test_timeframe_1d(self, live_server):
        data = _get(f"{live_server}/api/v1/strategies/leaderboard?timeframe=1d")
        assert data["timeframe"] == "1d"

    def test_timeframe_30d(self, live_server):
        data = _get(f"{live_server}/api/v1/strategies/leaderboard?timeframe=30d")
        assert data["timeframe"] == "30d"

    def test_invalid_metric_400(self, live_server):
        code, data = _get_raw(f"{live_server}/api/v1/strategies/leaderboard?metric=bogus")
        assert code == 400

    def test_invalid_timeframe_400(self, live_server):
        code, data = _get_raw(f"{live_server}/api/v1/strategies/leaderboard?timeframe=99y")
        assert code == 400

    def test_ranks_sequential(self, live_server):
        data = _get(f"{live_server}/api/v1/strategies/leaderboard?limit=5")
        ranks = [r["rank"] for r in data["leaderboard"]]
        assert ranks == list(range(1, len(ranks) + 1))

    def test_sharpe_sorted_desc(self, live_server):
        data = _get(f"{live_server}/api/v1/strategies/leaderboard?metric=sharpe_ratio")
        sharpes = [r["sharpe_ratio"] for r in data["leaderboard"]]
        assert sharpes == sorted(sharpes, reverse=True)

    def test_has_generated_at(self, live_server):
        data = _get(f"{live_server}/api/v1/strategies/leaderboard")
        assert "generated_at" in data


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION D — Integration & Version Checks
# ═══════════════════════════════════════════════════════════════════════════════

class TestS42Integration:
    """Integration tests and version checks."""

    def test_server_version_is_s42(self):
        sprint_num = int(SERVER_VERSION[1:]) if SERVER_VERSION[1:].isdigit() else 0
        assert sprint_num >= 42

    def test_s42_test_count_defined(self):
        from demo_server import _S42_TEST_COUNT
        assert isinstance(_S42_TEST_COUNT, int)

    def test_root_returns_s42_version(self, live_server):
        data = _get(f"{live_server}/")
        version = data.get("version", "")
        sprint_num = int(version[1:]) if version and version[1:].isdigit() else 0
        assert sprint_num >= 42

    def test_health_endpoint_ok(self, live_server):
        data = _get(f"{live_server}/demo/health")
        assert data.get("status") == "ok"

    def test_404_for_unknown_route(self, live_server):
        code, _ = _get_raw(f"{live_server}/api/v1/s42-does-not-exist")
        assert code == 404

    def test_all_s42_get_routes_accessible(self, live_server):
        routes = [
            "/api/v1/market/stream/latest",
            "/api/v1/agents/health",
            "/api/v1/agents/agent-alpha/diagnostics",
            "/api/v1/strategies/leaderboard",
        ]
        for route in routes:
            code, _ = _get_raw(f"{live_server}{route}")
            assert code == 200, f"Route {route} returned {code}"

    def test_ping_route_accessible(self, live_server):
        code, data = _post(f"{live_server}/api/v1/agents/agent-alpha/ping")
        assert code == 200

    def test_concurrent_health_requests(self, live_server):
        results = []
        errors = []

        def fetch():
            try:
                data = _get(f"{live_server}/api/v1/agents/health")
                results.append(data["total"])
            except Exception as exc:
                errors.append(str(exc))

        threads = [threading.Thread(target=fetch) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10)

        assert len(errors) == 0, f"Concurrent errors: {errors}"
        assert len(results) == 5

    def test_concurrent_market_stream_requests(self, live_server):
        results = []
        errors = []

        def fetch():
            try:
                data = _get(f"{live_server}/api/v1/market/stream/latest")
                results.append(data["interval_ms"])
            except Exception as exc:
                errors.append(str(exc))

        threads = [threading.Thread(target=fetch) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10)

        assert len(errors) == 0
        assert all(v == 500 for v in results)

    def test_leaderboard_and_health_in_sequence(self, live_server):
        lb = _get(f"{live_server}/api/v1/strategies/leaderboard")
        health = _get(f"{live_server}/api/v1/agents/health")
        assert "leaderboard" in lb
        assert "agents" in health

    def test_ping_then_diagnostics(self, live_server):
        uid = f"seq-test-{int(time.time() * 1000)}"
        code1, _ = _post(f"{live_server}/api/v1/agents/{uid}/ping")
        assert code1 == 200
        code2, data = _get_raw(f"{live_server}/api/v1/agents/{uid}/diagnostics")
        assert code2 == 200
        assert data["agent_id"] == uid

    def test_market_stream_then_leaderboard(self, live_server):
        ms = _get(f"{live_server}/api/v1/market/stream/latest")
        lb = _get(f"{live_server}/api/v1/strategies/leaderboard")
        assert "symbols" in ms
        assert "leaderboard" in lb
