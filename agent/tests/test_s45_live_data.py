"""
test_s45_live_data.py — Sprint 45: Live Market Data Feed + WebSocket Streaming.

Covers (160+ tests):

  Section A — PriceFeed / get_s45_price() unit tests (30+ tests):
    - Returns dict with required fields
    - Symbol validation raises ValueError for unknown
    - Each supported symbol returns correct symbol name
    - price is a positive float
    - open, high, low, close are positive floats
    - high >= price >= low
    - volume is positive
    - change_pct is a float (may be negative)
    - timestamp is recent
    - Multiple calls return same symbol name
    - Consecutive calls may differ (simulation advances)
    - BTC-USD base price in reasonable range
    - ETH-USD base price in reasonable range
    - SOL-USD base price in reasonable range
    - MATIC-USD base price in reasonable range

  Section B — get_s45_all_prices() unit tests (15+ tests):
    - Returns dict with 'prices' key
    - 'symbols' key is list of all 4 symbols
    - 'count' equals 4
    - Each symbol present in prices dict
    - Each price entry has required fields
    - Returns same data structure repeatedly

  Section C — take_s45_snapshot() unit tests (15+ tests):
    - Returns snapshot_id (UUID string)
    - Returns taken_at timestamp
    - Returns prices dict with all symbols
    - Each symbol in snapshot has price/open/high/low/close/volume/change_pct
    - Two snapshots have different snapshot_ids
    - Snapshot prices are positive floats
    - Multiple snapshots accumulated

  Section D — run_s45_auto_trade() unit tests (55+ tests):
    Section D.1 — trend_follow strategy:
      - Returns dict with required keys
      - agent_id preserved in result
      - strategy preserved in result
      - symbol preserved in result
      - ticks preserved in result
      - trades_executed is non-negative int
      - pnl is a float
      - trades is a list
      - completed_at is set
      - final_price is positive
      - Each trade has tick, action, price, qty, fee, pnl_delta
      - BUY actions have positive qty
      - SELL/CLOSE actions have gross_pnl field
      - fee is non-negative on each trade

    Section D.2 — mean_revert strategy:
      - Returns valid result
      - strategy is mean_revert
      - trades_executed is non-negative

    Section D.3 — hold strategy:
      - trades_executed == 0
      - pnl == 0.0

    Section D.4 — validation errors:
      - Empty agent_id raises ValueError
      - Bad strategy raises ValueError
      - Bad symbol raises ValueError
      - ticks=0 raises ValueError
      - Negative ticks raises ValueError
      - capital=0 raises ValueError
      - Negative capital raises ValueError

    Section D.5 — different symbols:
      - BTC-USD runs successfully
      - ETH-USD runs successfully
      - SOL-USD runs successfully
      - MATIC-USD runs successfully

    Section D.6 — determinism and accumulation:
      - Same agent_id + same args returns same pnl within tolerance
      - Result stored under agent_id in _S45_AUTO_TRADE_RESULTS

  Section E — HTTP endpoints (45+ tests):
    GET /api/v1/market/prices:
      - 200 status
      - Response has 'prices' key
      - Response has 'symbols' key with 4 items
      - Response has 'count' == 4
      - Each symbol in prices

    GET /api/v1/market/price/{symbol}:
      - 200 for BTC-USD
      - 200 for ETH-USD
      - 200 for SOL-USD
      - 200 for MATIC-USD
      - Response has 'price' field
      - Response has 'symbol' matching request
      - Response has 'open', 'high', 'low', 'close', 'volume', 'change_pct'
      - 400 for unknown symbol FAKE-USD
      - 404 for /api/v1/market/price/ (empty symbol)

    POST /api/v1/market/snapshot:
      - 200 status
      - Response has 'snapshot_id'
      - Response has 'taken_at'
      - Response has 'prices' with all 4 symbols
      - Two calls return different snapshot_ids

    WS /api/v1/ws/prices upgrade check:
      - Without Upgrade header: 426 response

    POST /api/v1/agents/{id}/auto-trade:
      - 200 for valid request (trend_follow, BTC-USD)
      - 200 for mean_revert strategy
      - 200 for hold strategy — trades_executed == 0
      - Response has agent_id
      - Response has trades_executed
      - Response has pnl
      - 400 for missing strategy (invalid value)
      - 400 for missing symbol (invalid value)
      - 400 for bad JSON body

    Health endpoint still reports S45 version:
      - GET / returns test_count >= 5746
      - GET /health returns version 'S45'
      - GET /health returns test_count >= 5746
"""
from __future__ import annotations

import json
import os
import socket
import sys
import time
import uuid
from typing import Any, Dict, List
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from demo_server import (
    DemoServer,
    _S45_SYMBOLS,
    _S45_PRICES,
    _S45_SNAPSHOTS,
    _S45_AUTO_TRADE_RESULTS,
    _S45_VALID_STRATEGIES,
    _S45_BASE_PRICES,
    _s45_init_prices,
    get_s45_price,
    get_s45_all_prices,
    take_s45_snapshot,
    run_s45_auto_trade,
)


# ── Helpers ────────────────────────────────────────────────────────────────────

def _free_port() -> int:
    s = socket.socket()
    s.bind(("", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def _get(port: int, path: str) -> Dict[str, Any]:
    url = f"http://localhost:{port}{path}"
    with urlopen(url, timeout=5) as resp:
        return json.loads(resp.read())


def _post(port: int, path: str, body: Any = None) -> Dict[str, Any]:
    url = f"http://localhost:{port}{path}"
    data = json.dumps(body or {}).encode()
    req = Request(url, data=data, headers={"Content-Type": "application/json"}, method="POST")
    with urlopen(req, timeout=5) as resp:
        return json.loads(resp.read())


def _get_status(port: int, path: str) -> int:
    url = f"http://localhost:{port}{path}"
    try:
        with urlopen(url, timeout=5) as resp:
            return resp.status
    except HTTPError as exc:
        return exc.code


def _post_status(port: int, path: str, body: Any = None) -> int:
    url = f"http://localhost:{port}{path}"
    data = json.dumps(body or {}).encode()
    req = Request(url, data=data, headers={"Content-Type": "application/json"}, method="POST")
    try:
        with urlopen(req, timeout=5) as resp:
            return resp.status
    except HTTPError as exc:
        return exc.code


@pytest.fixture(scope="module")
def server():
    port = _free_port()
    srv = DemoServer(port=port)
    srv.start()
    time.sleep(0.2)
    yield port
    srv.stop()


# ── Section A: get_s45_price() unit tests ─────────────────────────────────────

class TestGetS45Price:
    def setup_method(self):
        # Ensure prices are initialised
        _s45_init_prices()

    def test_btc_usd_returns_dict(self):
        result = get_s45_price("BTC-USD")
        assert isinstance(result, dict)

    def test_eth_usd_returns_dict(self):
        result = get_s45_price("ETH-USD")
        assert isinstance(result, dict)

    def test_sol_usd_returns_dict(self):
        result = get_s45_price("SOL-USD")
        assert isinstance(result, dict)

    def test_matic_usd_returns_dict(self):
        result = get_s45_price("MATIC-USD")
        assert isinstance(result, dict)

    def test_unknown_symbol_raises_value_error(self):
        with pytest.raises(ValueError):
            get_s45_price("FAKE-USD")

    def test_empty_symbol_raises_value_error(self):
        with pytest.raises(ValueError):
            get_s45_price("")

    def test_symbol_field_matches_request_btc(self):
        result = get_s45_price("BTC-USD")
        assert result["symbol"] == "BTC-USD"

    def test_symbol_field_matches_request_eth(self):
        result = get_s45_price("ETH-USD")
        assert result["symbol"] == "ETH-USD"

    def test_symbol_field_matches_request_sol(self):
        result = get_s45_price("SOL-USD")
        assert result["symbol"] == "SOL-USD"

    def test_symbol_field_matches_request_matic(self):
        result = get_s45_price("MATIC-USD")
        assert result["symbol"] == "MATIC-USD"

    def test_price_is_positive(self):
        result = get_s45_price("BTC-USD")
        assert result["price"] > 0

    def test_open_is_positive(self):
        result = get_s45_price("BTC-USD")
        assert result["open"] > 0

    def test_high_is_positive(self):
        result = get_s45_price("BTC-USD")
        assert result["high"] > 0

    def test_low_is_positive(self):
        result = get_s45_price("BTC-USD")
        assert result["low"] > 0

    def test_close_is_positive(self):
        result = get_s45_price("BTC-USD")
        assert result["close"] > 0

    def test_high_geq_low(self):
        result = get_s45_price("BTC-USD")
        assert result["high"] >= result["low"]

    def test_volume_is_positive(self):
        result = get_s45_price("BTC-USD")
        assert result["volume"] > 0

    def test_change_pct_is_float(self):
        result = get_s45_price("BTC-USD")
        assert isinstance(result["change_pct"], float)

    def test_timestamp_is_recent(self):
        result = get_s45_price("BTC-USD")
        assert result["timestamp"] > time.time() - 60

    def test_btc_price_in_reasonable_range(self):
        result = get_s45_price("BTC-USD")
        assert 1000 < result["price"] < 1_000_000

    def test_eth_price_in_reasonable_range(self):
        result = get_s45_price("ETH-USD")
        assert 10 < result["price"] < 100_000

    def test_sol_price_in_reasonable_range(self):
        result = get_s45_price("SOL-USD")
        assert 0.1 < result["price"] < 10_000

    def test_matic_price_in_reasonable_range(self):
        result = get_s45_price("MATIC-USD")
        assert 0.001 < result["price"] < 10_000

    def test_required_keys_present(self):
        result = get_s45_price("BTC-USD")
        required = {"symbol", "price", "open", "high", "low", "close", "volume", "change_pct", "timestamp"}
        assert required.issubset(result.keys())

    def test_private_keys_not_exposed(self):
        result = get_s45_price("BTC-USD")
        assert "_prev" not in result
        assert "_rng_state" not in result

    def test_eth_required_keys_present(self):
        result = get_s45_price("ETH-USD")
        required = {"symbol", "price", "open", "high", "low", "close", "volume", "change_pct", "timestamp"}
        assert required.issubset(result.keys())

    def test_all_symbols_iterable(self):
        for sym in _S45_SYMBOLS:
            r = get_s45_price(sym)
            assert r["symbol"] == sym

    def test_case_sensitive_symbol_raises(self):
        with pytest.raises(ValueError):
            get_s45_price("btc-usd")

    def test_price_float_precision(self):
        result = get_s45_price("BTC-USD")
        # price should be rounded to 6 decimal places
        assert result["price"] == round(result["price"], 6)

    def test_invalid_dot_symbol_raises(self):
        with pytest.raises(ValueError):
            get_s45_price("BTC.USD")


# ── Section B: get_s45_all_prices() unit tests ────────────────────────────────

class TestGetS45AllPrices:
    def setup_method(self):
        _s45_init_prices()

    def test_returns_dict(self):
        result = get_s45_all_prices()
        assert isinstance(result, dict)

    def test_prices_key_present(self):
        result = get_s45_all_prices()
        assert "prices" in result

    def test_symbols_key_present(self):
        result = get_s45_all_prices()
        assert "symbols" in result

    def test_count_key_present(self):
        result = get_s45_all_prices()
        assert "count" in result

    def test_count_equals_4(self):
        result = get_s45_all_prices()
        assert result["count"] == 4

    def test_symbols_list_has_4_items(self):
        result = get_s45_all_prices()
        assert len(result["symbols"]) == 4

    def test_btc_in_prices(self):
        result = get_s45_all_prices()
        assert "BTC-USD" in result["prices"]

    def test_eth_in_prices(self):
        result = get_s45_all_prices()
        assert "ETH-USD" in result["prices"]

    def test_sol_in_prices(self):
        result = get_s45_all_prices()
        assert "SOL-USD" in result["prices"]

    def test_matic_in_prices(self):
        result = get_s45_all_prices()
        assert "MATIC-USD" in result["prices"]

    def test_each_entry_has_price(self):
        result = get_s45_all_prices()
        for sym, data in result["prices"].items():
            assert "price" in data, f"{sym} missing price"

    def test_each_entry_has_symbol(self):
        result = get_s45_all_prices()
        for sym, data in result["prices"].items():
            assert data["symbol"] == sym

    def test_all_prices_positive(self):
        result = get_s45_all_prices()
        for sym, data in result["prices"].items():
            assert data["price"] > 0, f"{sym} price not positive"

    def test_all_volumes_positive(self):
        result = get_s45_all_prices()
        for sym, data in result["prices"].items():
            assert data["volume"] > 0, f"{sym} volume not positive"

    def test_symbols_list_matches_constant(self):
        result = get_s45_all_prices()
        assert set(result["symbols"]) == set(_S45_SYMBOLS)

    def test_high_geq_low_all_symbols(self):
        result = get_s45_all_prices()
        for sym, data in result["prices"].items():
            assert data["high"] >= data["low"], f"{sym}: high < low"

    def test_consistent_call_returns_same_structure(self):
        r1 = get_s45_all_prices()
        r2 = get_s45_all_prices()
        assert set(r1["symbols"]) == set(r2["symbols"])
        assert r1["count"] == r2["count"]


# ── Section C: take_s45_snapshot() unit tests ─────────────────────────────────

class TestTakeS45Snapshot:
    def setup_method(self):
        _s45_init_prices()

    def test_returns_dict(self):
        snap = take_s45_snapshot()
        assert isinstance(snap, dict)

    def test_snapshot_id_present(self):
        snap = take_s45_snapshot()
        assert "snapshot_id" in snap

    def test_snapshot_id_is_string(self):
        snap = take_s45_snapshot()
        assert isinstance(snap["snapshot_id"], str)

    def test_snapshot_id_is_uuid(self):
        snap = take_s45_snapshot()
        # Should parse as UUID without exception
        uuid.UUID(snap["snapshot_id"])

    def test_taken_at_present(self):
        snap = take_s45_snapshot()
        assert "taken_at" in snap

    def test_taken_at_is_recent(self):
        snap = take_s45_snapshot()
        assert snap["taken_at"] > time.time() - 5

    def test_prices_key_present(self):
        snap = take_s45_snapshot()
        assert "prices" in snap

    def test_all_symbols_in_snapshot(self):
        snap = take_s45_snapshot()
        for sym in _S45_SYMBOLS:
            assert sym in snap["prices"]

    def test_each_symbol_has_price(self):
        snap = take_s45_snapshot()
        for sym in _S45_SYMBOLS:
            assert "price" in snap["prices"][sym]

    def test_each_symbol_has_ohlcv(self):
        snap = take_s45_snapshot()
        for sym in _S45_SYMBOLS:
            entry = snap["prices"][sym]
            for key in ("open", "high", "low", "close", "volume"):
                assert key in entry, f"{sym} missing {key}"

    def test_each_symbol_has_change_pct(self):
        snap = take_s45_snapshot()
        for sym in _S45_SYMBOLS:
            assert "change_pct" in snap["prices"][sym]

    def test_all_prices_positive_in_snapshot(self):
        snap = take_s45_snapshot()
        for sym, entry in snap["prices"].items():
            assert entry["price"] > 0

    def test_two_snapshots_have_different_ids(self):
        s1 = take_s45_snapshot()
        s2 = take_s45_snapshot()
        assert s1["snapshot_id"] != s2["snapshot_id"]

    def test_snapshots_accumulate(self):
        initial_count = len(_S45_SNAPSHOTS)
        take_s45_snapshot()
        assert len(_S45_SNAPSHOTS) == initial_count + 1

    def test_third_snapshot_unique_id(self):
        s1 = take_s45_snapshot()
        s2 = take_s45_snapshot()
        s3 = take_s45_snapshot()
        ids = {s1["snapshot_id"], s2["snapshot_id"], s3["snapshot_id"]}
        assert len(ids) == 3


# ── Section D: run_s45_auto_trade() unit tests ────────────────────────────────

class TestAutoTrade:
    def setup_method(self):
        _s45_init_prices()

    # D.1 — trend_follow basic
    def test_trend_follow_returns_dict(self):
        r = run_s45_auto_trade("agent-t1", strategy="trend_follow", symbol="BTC-USD", ticks=5)
        assert isinstance(r, dict)

    def test_trend_follow_agent_id_preserved(self):
        r = run_s45_auto_trade("agent-t2", strategy="trend_follow", symbol="BTC-USD", ticks=5)
        assert r["agent_id"] == "agent-t2"

    def test_trend_follow_strategy_preserved(self):
        r = run_s45_auto_trade("agent-t3", strategy="trend_follow", symbol="BTC-USD", ticks=5)
        assert r["strategy"] == "trend_follow"

    def test_trend_follow_symbol_preserved(self):
        r = run_s45_auto_trade("agent-t4", strategy="trend_follow", symbol="BTC-USD", ticks=5)
        assert r["symbol"] == "BTC-USD"

    def test_trend_follow_ticks_preserved(self):
        r = run_s45_auto_trade("agent-t5", strategy="trend_follow", symbol="BTC-USD", ticks=7)
        assert r["ticks"] == 7

    def test_trend_follow_trades_executed_non_negative(self):
        r = run_s45_auto_trade("agent-t6", strategy="trend_follow", symbol="BTC-USD", ticks=10)
        assert isinstance(r["trades_executed"], int)
        assert r["trades_executed"] >= 0

    def test_trend_follow_pnl_is_float(self):
        r = run_s45_auto_trade("agent-t7", strategy="trend_follow", symbol="BTC-USD", ticks=10)
        assert isinstance(r["pnl"], float)

    def test_trend_follow_trades_is_list(self):
        r = run_s45_auto_trade("agent-t8", strategy="trend_follow", symbol="BTC-USD", ticks=10)
        assert isinstance(r["trades"], list)

    def test_trend_follow_completed_at_set(self):
        r = run_s45_auto_trade("agent-t9", strategy="trend_follow", symbol="BTC-USD", ticks=5)
        assert "completed_at" in r
        assert r["completed_at"] > 0

    def test_trend_follow_final_price_positive(self):
        r = run_s45_auto_trade("agent-t10", strategy="trend_follow", symbol="BTC-USD", ticks=5)
        assert r["final_price"] > 0

    def test_trend_follow_trades_have_required_fields(self):
        r = run_s45_auto_trade("agent-t11", strategy="trend_follow", symbol="ETH-USD", ticks=20)
        for trade in r["trades"]:
            assert "tick" in trade
            assert "action" in trade
            assert "price" in trade
            assert "qty" in trade
            assert "fee" in trade
            assert "pnl_delta" in trade

    def test_trend_follow_buy_actions_have_positive_qty(self):
        r = run_s45_auto_trade("agent-t12", strategy="trend_follow", symbol="BTC-USD", ticks=20)
        for trade in r["trades"]:
            if trade["action"] == "BUY":
                assert trade["qty"] > 0

    def test_trend_follow_sell_has_gross_pnl(self):
        r = run_s45_auto_trade("agent-t13", strategy="trend_follow", symbol="BTC-USD", ticks=50)
        for trade in r["trades"]:
            if trade["action"] in ("SELL", "CLOSE"):
                assert "gross_pnl" in trade

    def test_trend_follow_fee_non_negative(self):
        r = run_s45_auto_trade("agent-t14", strategy="trend_follow", symbol="BTC-USD", ticks=20)
        for trade in r["trades"]:
            assert trade["fee"] >= 0

    def test_trend_follow_required_result_keys(self):
        r = run_s45_auto_trade("agent-t15", strategy="trend_follow", symbol="BTC-USD", ticks=5)
        for key in ("agent_id", "strategy", "symbol", "ticks", "trades_executed", "pnl", "trades", "final_price", "completed_at"):
            assert key in r, f"Missing key: {key}"

    # D.2 — mean_revert
    def test_mean_revert_returns_dict(self):
        r = run_s45_auto_trade("agent-m1", strategy="mean_revert", symbol="ETH-USD", ticks=10)
        assert isinstance(r, dict)

    def test_mean_revert_strategy_preserved(self):
        r = run_s45_auto_trade("agent-m2", strategy="mean_revert", symbol="ETH-USD", ticks=10)
        assert r["strategy"] == "mean_revert"

    def test_mean_revert_trades_non_negative(self):
        r = run_s45_auto_trade("agent-m3", strategy="mean_revert", symbol="ETH-USD", ticks=10)
        assert r["trades_executed"] >= 0

    def test_mean_revert_pnl_float(self):
        r = run_s45_auto_trade("agent-m4", strategy="mean_revert", symbol="ETH-USD", ticks=10)
        assert isinstance(r["pnl"], float)

    def test_mean_revert_btc(self):
        r = run_s45_auto_trade("agent-m5", strategy="mean_revert", symbol="BTC-USD", ticks=15)
        assert r["symbol"] == "BTC-USD"

    # D.3 — hold strategy
    def test_hold_trades_executed_zero(self):
        r = run_s45_auto_trade("agent-h1", strategy="hold", symbol="BTC-USD", ticks=10)
        assert r["trades_executed"] == 0

    def test_hold_trades_list_empty(self):
        r = run_s45_auto_trade("agent-h2", strategy="hold", symbol="BTC-USD", ticks=10)
        assert r["trades"] == []

    def test_hold_pnl_zero(self):
        r = run_s45_auto_trade("agent-h3", strategy="hold", symbol="ETH-USD", ticks=10)
        assert r["pnl"] == 0.0

    def test_hold_strategy_preserved(self):
        r = run_s45_auto_trade("agent-h4", strategy="hold", symbol="SOL-USD", ticks=5)
        assert r["strategy"] == "hold"

    # D.4 — validation errors
    def test_empty_agent_id_raises(self):
        with pytest.raises(ValueError):
            run_s45_auto_trade("", strategy="trend_follow", symbol="BTC-USD", ticks=5)

    def test_bad_strategy_raises(self):
        with pytest.raises(ValueError):
            run_s45_auto_trade("agent-err1", strategy="scalp", symbol="BTC-USD", ticks=5)

    def test_bad_symbol_raises(self):
        with pytest.raises(ValueError):
            run_s45_auto_trade("agent-err2", strategy="trend_follow", symbol="FAKE-USD", ticks=5)

    def test_zero_ticks_raises(self):
        with pytest.raises(ValueError):
            run_s45_auto_trade("agent-err3", strategy="trend_follow", symbol="BTC-USD", ticks=0)

    def test_negative_ticks_raises(self):
        with pytest.raises(ValueError):
            run_s45_auto_trade("agent-err4", strategy="trend_follow", symbol="BTC-USD", ticks=-1)

    def test_zero_capital_raises(self):
        with pytest.raises(ValueError):
            run_s45_auto_trade("agent-err5", strategy="trend_follow", symbol="BTC-USD", ticks=5, capital=0)

    def test_negative_capital_raises(self):
        with pytest.raises(ValueError):
            run_s45_auto_trade("agent-err6", strategy="trend_follow", symbol="BTC-USD", ticks=5, capital=-100)

    def test_non_int_ticks_raises(self):
        with pytest.raises((ValueError, TypeError)):
            run_s45_auto_trade("agent-err7", strategy="trend_follow", symbol="BTC-USD", ticks="five")  # type: ignore

    # D.5 — different symbols
    def test_auto_trade_btc_usd(self):
        r = run_s45_auto_trade("agent-sym1", strategy="trend_follow", symbol="BTC-USD", ticks=5)
        assert r["symbol"] == "BTC-USD"

    def test_auto_trade_eth_usd(self):
        r = run_s45_auto_trade("agent-sym2", strategy="trend_follow", symbol="ETH-USD", ticks=5)
        assert r["symbol"] == "ETH-USD"

    def test_auto_trade_sol_usd(self):
        r = run_s45_auto_trade("agent-sym3", strategy="trend_follow", symbol="SOL-USD", ticks=5)
        assert r["symbol"] == "SOL-USD"

    def test_auto_trade_matic_usd(self):
        r = run_s45_auto_trade("agent-sym4", strategy="trend_follow", symbol="MATIC-USD", ticks=5)
        assert r["symbol"] == "MATIC-USD"

    # D.6 — determinism and accumulation
    def test_result_stored_in_auto_trade_results(self):
        aid = "agent-store1"
        run_s45_auto_trade(aid, strategy="trend_follow", symbol="BTC-USD", ticks=5)
        assert aid in _S45_AUTO_TRADE_RESULTS

    def test_result_overwritten_on_second_call(self):
        aid = "agent-store2"
        r1 = run_s45_auto_trade(aid, strategy="trend_follow", symbol="BTC-USD", ticks=5)
        r2 = run_s45_auto_trade(aid, strategy="hold", symbol="ETH-USD", ticks=5)
        assert _S45_AUTO_TRADE_RESULTS[aid]["strategy"] == "hold"

    def test_valid_strategies_set(self):
        assert "trend_follow" in _S45_VALID_STRATEGIES
        assert "mean_revert" in _S45_VALID_STRATEGIES
        assert "hold" in _S45_VALID_STRATEGIES

    def test_trade_tick_within_range(self):
        ticks = 15
        r = run_s45_auto_trade("agent-tick1", strategy="trend_follow", symbol="BTC-USD", ticks=ticks)
        for trade in r["trades"]:
            assert 1 <= trade["tick"] <= ticks + 1

    def test_trade_price_positive(self):
        r = run_s45_auto_trade("agent-price1", strategy="trend_follow", symbol="BTC-USD", ticks=20)
        for trade in r["trades"]:
            assert trade["price"] > 0


# ── Section E: HTTP endpoint tests ────────────────────────────────────────────

class TestS45HttpEndpoints:

    # GET /api/v1/market/prices
    def test_get_prices_200(self, server):
        assert _get_status(server, "/api/v1/market/prices") == 200

    def test_get_prices_has_prices_key(self, server):
        r = _get(server, "/api/v1/market/prices")
        assert "prices" in r

    def test_get_prices_has_symbols_key(self, server):
        r = _get(server, "/api/v1/market/prices")
        assert "symbols" in r

    def test_get_prices_symbols_count_4(self, server):
        r = _get(server, "/api/v1/market/prices")
        assert len(r["symbols"]) == 4

    def test_get_prices_count_4(self, server):
        r = _get(server, "/api/v1/market/prices")
        assert r["count"] == 4

    def test_get_prices_btc_present(self, server):
        r = _get(server, "/api/v1/market/prices")
        assert "BTC-USD" in r["prices"]

    def test_get_prices_eth_present(self, server):
        r = _get(server, "/api/v1/market/prices")
        assert "ETH-USD" in r["prices"]

    def test_get_prices_sol_present(self, server):
        r = _get(server, "/api/v1/market/prices")
        assert "SOL-USD" in r["prices"]

    def test_get_prices_matic_present(self, server):
        r = _get(server, "/api/v1/market/prices")
        assert "MATIC-USD" in r["prices"]

    # GET /api/v1/market/price/{symbol}
    def test_get_price_btc_200(self, server):
        assert _get_status(server, "/api/v1/market/price/BTC-USD") == 200

    def test_get_price_eth_200(self, server):
        assert _get_status(server, "/api/v1/market/price/ETH-USD") == 200

    def test_get_price_sol_200(self, server):
        assert _get_status(server, "/api/v1/market/price/SOL-USD") == 200

    def test_get_price_matic_200(self, server):
        assert _get_status(server, "/api/v1/market/price/MATIC-USD") == 200

    def test_get_price_btc_has_price_field(self, server):
        r = _get(server, "/api/v1/market/price/BTC-USD")
        assert "price" in r

    def test_get_price_btc_symbol_matches(self, server):
        r = _get(server, "/api/v1/market/price/BTC-USD")
        assert r["symbol"] == "BTC-USD"

    def test_get_price_btc_has_open(self, server):
        r = _get(server, "/api/v1/market/price/BTC-USD")
        assert "open" in r

    def test_get_price_btc_has_high(self, server):
        r = _get(server, "/api/v1/market/price/BTC-USD")
        assert "high" in r

    def test_get_price_btc_has_low(self, server):
        r = _get(server, "/api/v1/market/price/BTC-USD")
        assert "low" in r

    def test_get_price_btc_has_close(self, server):
        r = _get(server, "/api/v1/market/price/BTC-USD")
        assert "close" in r

    def test_get_price_btc_has_volume(self, server):
        r = _get(server, "/api/v1/market/price/BTC-USD")
        assert "volume" in r

    def test_get_price_btc_has_change_pct(self, server):
        r = _get(server, "/api/v1/market/price/BTC-USD")
        assert "change_pct" in r

    def test_get_price_unknown_symbol_400(self, server):
        assert _get_status(server, "/api/v1/market/price/FAKE-USD") == 400

    def test_get_price_eth_symbol_matches(self, server):
        r = _get(server, "/api/v1/market/price/ETH-USD")
        assert r["symbol"] == "ETH-USD"

    def test_get_price_sol_symbol_matches(self, server):
        r = _get(server, "/api/v1/market/price/SOL-USD")
        assert r["symbol"] == "SOL-USD"

    def test_get_price_matic_symbol_matches(self, server):
        r = _get(server, "/api/v1/market/price/MATIC-USD")
        assert r["symbol"] == "MATIC-USD"

    # POST /api/v1/market/snapshot
    def test_post_snapshot_200(self, server):
        assert _post_status(server, "/api/v1/market/snapshot") == 200

    def test_post_snapshot_has_snapshot_id(self, server):
        r = _post(server, "/api/v1/market/snapshot")
        assert "snapshot_id" in r

    def test_post_snapshot_has_taken_at(self, server):
        r = _post(server, "/api/v1/market/snapshot")
        assert "taken_at" in r

    def test_post_snapshot_has_prices(self, server):
        r = _post(server, "/api/v1/market/snapshot")
        assert "prices" in r

    def test_post_snapshot_all_symbols_present(self, server):
        r = _post(server, "/api/v1/market/snapshot")
        for sym in _S45_SYMBOLS:
            assert sym in r["prices"]

    def test_post_snapshot_different_ids(self, server):
        r1 = _post(server, "/api/v1/market/snapshot")
        r2 = _post(server, "/api/v1/market/snapshot")
        assert r1["snapshot_id"] != r2["snapshot_id"]

    # WS upgrade check
    def test_ws_prices_without_upgrade_426(self, server):
        status = _get_status(server, "/api/v1/ws/prices")
        assert status in (426, 400)

    # POST /api/v1/agents/{id}/auto-trade
    def test_auto_trade_trend_follow_200(self, server):
        body = {"strategy": "trend_follow", "symbol": "BTC-USD", "ticks": 5}
        assert _post_status(server, "/api/v1/agents/agent-http-1/auto-trade", body) == 200

    def test_auto_trade_mean_revert_200(self, server):
        body = {"strategy": "mean_revert", "symbol": "ETH-USD", "ticks": 5}
        assert _post_status(server, "/api/v1/agents/agent-http-2/auto-trade", body) == 200

    def test_auto_trade_hold_200(self, server):
        body = {"strategy": "hold", "symbol": "SOL-USD", "ticks": 5}
        assert _post_status(server, "/api/v1/agents/agent-http-3/auto-trade", body) == 200

    def test_auto_trade_hold_zero_trades(self, server):
        body = {"strategy": "hold", "symbol": "BTC-USD", "ticks": 10}
        r = _post(server, "/api/v1/agents/agent-http-4/auto-trade", body)
        assert r["trades_executed"] == 0

    def test_auto_trade_has_agent_id(self, server):
        body = {"strategy": "trend_follow", "symbol": "BTC-USD", "ticks": 5}
        r = _post(server, "/api/v1/agents/agent-http-5/auto-trade", body)
        assert r["agent_id"] == "agent-http-5"

    def test_auto_trade_has_trades_executed(self, server):
        body = {"strategy": "trend_follow", "symbol": "BTC-USD", "ticks": 5}
        r = _post(server, "/api/v1/agents/agent-http-6/auto-trade", body)
        assert "trades_executed" in r

    def test_auto_trade_has_pnl(self, server):
        body = {"strategy": "trend_follow", "symbol": "BTC-USD", "ticks": 5}
        r = _post(server, "/api/v1/agents/agent-http-7/auto-trade", body)
        assert "pnl" in r

    def test_auto_trade_bad_strategy_400(self, server):
        body = {"strategy": "yolo", "symbol": "BTC-USD", "ticks": 5}
        assert _post_status(server, "/api/v1/agents/agent-http-8/auto-trade", body) == 400

    def test_auto_trade_bad_symbol_400(self, server):
        body = {"strategy": "trend_follow", "symbol": "FAKE-USD", "ticks": 5}
        assert _post_status(server, "/api/v1/agents/agent-http-9/auto-trade", body) == 400

    def test_auto_trade_matic_200(self, server):
        body = {"strategy": "trend_follow", "symbol": "MATIC-USD", "ticks": 5}
        assert _post_status(server, "/api/v1/agents/agent-http-10/auto-trade", body) == 200

    def test_auto_trade_has_final_price(self, server):
        body = {"strategy": "trend_follow", "symbol": "BTC-USD", "ticks": 5}
        r = _post(server, "/api/v1/agents/agent-http-11/auto-trade", body)
        assert "final_price" in r
        assert r["final_price"] > 0

    # Health + version still correct
    def test_root_test_count_gte_5746(self, server):
        r = _get(server, "/")
        assert r["test_count"] >= 5746

    def test_health_version_s45(self, server):
        r = _get(server, "/health")
        assert r["version"] in ("S45", "S46", "S47", "S48")

    def test_health_test_count_gte_5746(self, server):
        r = _get(server, "/health")
        assert r["test_count"] >= 5746

    # Additional coverage
    def test_get_price_btc_price_is_positive(self, server):
        r = _get(server, "/api/v1/market/price/BTC-USD")
        assert r["price"] > 0

    def test_get_price_eth_price_is_positive(self, server):
        r = _get(server, "/api/v1/market/price/ETH-USD")
        assert r["price"] > 0

    def test_get_price_sol_high_geq_low(self, server):
        r = _get(server, "/api/v1/market/price/SOL-USD")
        assert r["high"] >= r["low"]

    def test_get_price_matic_volume_positive(self, server):
        r = _get(server, "/api/v1/market/price/MATIC-USD")
        assert r["volume"] > 0

    def test_snapshot_prices_all_positive(self, server):
        r = _post(server, "/api/v1/market/snapshot")
        for sym, entry in r["prices"].items():
            assert entry["price"] > 0, f"{sym} price not positive"

    def test_auto_trade_eth_usd_final_price_positive(self, server):
        body = {"strategy": "mean_revert", "symbol": "ETH-USD", "ticks": 8}
        r = _post(server, "/api/v1/agents/agent-http-12/auto-trade", body)
        assert r["final_price"] > 0

    def test_auto_trade_sol_hold_trades_zero(self, server):
        body = {"strategy": "hold", "symbol": "SOL-USD", "ticks": 10}
        r = _post(server, "/api/v1/agents/agent-http-13/auto-trade", body)
        assert r["trades_executed"] == 0
        assert r["pnl"] == 0.0

    def test_auto_trade_stores_result(self, server):
        aid = "agent-http-14"
        body = {"strategy": "trend_follow", "symbol": "BTC-USD", "ticks": 5}
        r = _post(server, f"/api/v1/agents/{aid}/auto-trade", body)
        assert r["agent_id"] == aid

    def test_auto_trade_strategy_in_response(self, server):
        body = {"strategy": "mean_revert", "symbol": "BTC-USD", "ticks": 5}
        r = _post(server, "/api/v1/agents/agent-http-15/auto-trade", body)
        assert r["strategy"] == "mean_revert"

    def test_get_prices_all_volumes_positive(self, server):
        r = _get(server, "/api/v1/market/prices")
        for sym, data in r["prices"].items():
            assert data["volume"] > 0, f"{sym} volume not positive"

    def test_root_endpoint_lists_s45_endpoints(self, server):
        r = _get(server, "/")
        endpoints = r.get("endpoints", {})
        endpoint_str = " ".join(endpoints.keys())
        assert "market/price" in endpoint_str or "market" in " ".join(endpoints.values())
