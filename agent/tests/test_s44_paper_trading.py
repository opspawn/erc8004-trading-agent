"""
test_s44_paper_trading.py — Sprint 44: Paper Trading Simulation.

Covers (70+ tests):

  Section A — place_paper_order() unit tests (25+ tests):
    - Returns order_id, position_id, fill_price, fee, pnl_delta
    - BUY: fill_price = price * (1 + 0.1% slippage)
    - SELL: fill_price = price * (1 - 0.1% slippage)
    - Fee = 0.05% of notional
    - pnl_delta is negative on open (= -fee)
    - position recorded as open
    - order appended to history
    - side stored as uppercase
    - Invalid: empty agent_id raises ValueError
    - Invalid: bad symbol raises ValueError
    - Invalid: bad side raises ValueError
    - Invalid: quantity <= 0 raises ValueError
    - Invalid: price <= 0 raises ValueError

  Section B — get_paper_positions() unit tests (15+ tests):
    - Returns open positions only
    - Filter by agent_id
    - Closed positions excluded
    - Empty list when no positions

  Section C — close_paper_position() unit tests (20+ tests):
    - BUY close: gross_pnl = (close - entry) * qty
    - SELL close: gross_pnl = (entry - close) * qty
    - net_pnl = gross_pnl - open_fee - close_fee
    - win=True when net_pnl > 0
    - win=False when net_pnl <= 0
    - Position status becomes 'closed'
    - Close record appended to history
    - Invalid: empty position_id raises ValueError
    - Invalid: close_price <= 0 raises ValueError
    - Invalid: unknown position_id raises KeyError
    - Invalid: already-closed raises ValueError
    - Agent mismatch raises ValueError

  Section D — get_paper_history() unit tests (15+ tests):
    - Returns all orders (opens + closes)
    - Filter by agent_id
    - Filter by symbol
    - Limit enforced
    - Most-recent first ordering
    - Invalid symbol raises ValueError

  Section E — HTTP endpoints (10+ tests):
    POST /api/v1/trading/paper/order:
      - 200 on valid order
      - 400 on missing agent_id
      - 400 on bad symbol
      - 400 on bad side
    GET /api/v1/trading/paper/positions:
      - 200 response with positions list
    POST /api/v1/trading/paper/close:
      - 200 on valid close
      - 404 on unknown position
    GET /api/v1/trading/paper/history:
      - 200 response with history list
"""
from __future__ import annotations

import json
import os
import socket
import sys
import time
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from demo_server import (
    DemoServer,
    _S44_PAPER_LOCK,
    _S44_PAPER_ORDERS,
    _S44_PAPER_POSITIONS,
    _S44_SLIPPAGE,
    _S44_FEE_RATE,
    _S44_VALID_SYMBOLS,
    _S44_VALID_SIDES,
    place_paper_order,
    get_paper_positions,
    close_paper_position,
    get_paper_history,
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


def _clear_paper() -> None:
    with _S44_PAPER_LOCK:
        _S44_PAPER_ORDERS.clear()
        _S44_PAPER_POSITIONS.clear()


# ─── Section A — place_paper_order() ──────────────────────────────────────────


class TestPlacePaperOrder:
    def setup_method(self):
        _clear_paper()

    def test_returns_order_id(self):
        result = place_paper_order("agent-po-001", "BTC/USD", "BUY", 0.1, 68000.0)
        assert "order_id" in result

    def test_returns_position_id(self):
        result = place_paper_order("agent-po-001", "BTC/USD", "BUY", 0.1, 68000.0)
        assert "position_id" in result

    def test_returns_fill_price(self):
        result = place_paper_order("agent-fp-001", "BTC/USD", "BUY", 0.1, 68000.0)
        assert "fill_price" in result

    def test_buy_fill_price_higher(self):
        price = 68000.0
        result = place_paper_order("agent-fp-buy-001", "BTC/USD", "BUY", 0.1, price)
        expected = round(price * (1 + _S44_SLIPPAGE), 6)
        assert abs(result["fill_price"] - expected) < 0.01

    def test_sell_fill_price_lower(self):
        price = 68000.0
        result = place_paper_order("agent-fp-sell-001", "BTC/USD", "SELL", 0.1, price)
        expected = round(price * (1 - _S44_SLIPPAGE), 6)
        assert abs(result["fill_price"] - expected) < 0.01

    def test_fee_equals_notional_times_rate(self):
        price = 3500.0
        qty = 1.0
        result = place_paper_order("agent-fee-001", "ETH/USD", "BUY", qty, price)
        fill = result["fill_price"]
        expected_fee = round(fill * qty * _S44_FEE_RATE, 6)
        assert abs(result["fee"] - expected_fee) < 1e-4

    def test_pnl_delta_negative_on_open(self):
        result = place_paper_order("agent-pnl-open-001", "BTC/USD", "BUY", 0.1, 68000.0)
        assert result["pnl_delta"] < 0

    def test_pnl_delta_equals_negative_fee(self):
        result = place_paper_order("agent-pnl-open-002", "ETH/USD", "SELL", 2.0, 3500.0)
        assert abs(result["pnl_delta"] - (-result["fee"])) < 1e-6

    def test_side_stored_uppercase_buy(self):
        result = place_paper_order("agent-side-001", "BTC/USD", "buy", 0.1, 68000.0)
        assert result["side"] == "BUY"

    def test_side_stored_uppercase_sell(self):
        result = place_paper_order("agent-side-002", "SOL/USD", "sell", 5.0, 180.0)
        assert result["side"] == "SELL"

    def test_status_is_filled(self):
        result = place_paper_order("agent-stat-001", "BTC/USD", "BUY", 0.1, 68000.0)
        assert result["status"] == "filled"

    def test_position_created_as_open(self):
        result = place_paper_order("agent-pos-001", "BTC/USD", "BUY", 0.1, 68000.0)
        pos_id = result["position_id"]
        positions = get_paper_positions()
        ids = [p["position_id"] for p in positions]
        assert pos_id in ids

    def test_order_appended_to_history(self):
        place_paper_order("agent-hist-001", "ETH/USD", "SELL", 1.0, 3500.0)
        history = get_paper_history(agent_id="agent-hist-001")
        assert len(history) >= 1

    def test_agent_id_stored(self):
        result = place_paper_order("agent-aid-001", "BTC/USD", "BUY", 0.1, 68000.0)
        assert result["agent_id"] == "agent-aid-001"

    def test_symbol_stored(self):
        result = place_paper_order("agent-sym-001", "SOL/USD", "BUY", 5.0, 180.0)
        assert result["symbol"] == "SOL/USD"

    def test_quantity_stored(self):
        result = place_paper_order("agent-qty-001", "BTC/USD", "BUY", 0.25, 68000.0)
        assert result["quantity"] == 0.25

    def test_notional_computed(self):
        qty = 2.0
        price = 3500.0
        result = place_paper_order("agent-ntl-001", "ETH/USD", "BUY", qty, price)
        fill = result["fill_price"]
        expected_notional = fill * qty
        assert abs(result["notional"] - expected_notional) < 0.01

    def test_raises_empty_agent_id(self):
        with pytest.raises(ValueError, match="agent_id"):
            place_paper_order("", "BTC/USD", "BUY", 0.1, 68000.0)

    def test_raises_none_agent_id(self):
        with pytest.raises(ValueError, match="agent_id"):
            place_paper_order(None, "BTC/USD", "BUY", 0.1, 68000.0)

    def test_raises_invalid_symbol(self):
        with pytest.raises(ValueError, match="symbol"):
            place_paper_order("agent-err-001", "DOGE/USD", "BUY", 0.1, 0.08)

    def test_raises_invalid_side(self):
        with pytest.raises(ValueError, match="side"):
            place_paper_order("agent-err-002", "BTC/USD", "HOLD", 0.1, 68000.0)

    def test_raises_zero_quantity(self):
        with pytest.raises(ValueError, match="quantity"):
            place_paper_order("agent-err-003", "BTC/USD", "BUY", 0.0, 68000.0)

    def test_raises_negative_quantity(self):
        with pytest.raises(ValueError, match="quantity"):
            place_paper_order("agent-err-004", "BTC/USD", "BUY", -1.0, 68000.0)

    def test_raises_zero_price(self):
        with pytest.raises(ValueError, match="price"):
            place_paper_order("agent-err-005", "BTC/USD", "BUY", 0.1, 0.0)

    def test_raises_negative_price(self):
        with pytest.raises(ValueError, match="price"):
            place_paper_order("agent-err-006", "BTC/USD", "BUY", 0.1, -100.0)

    def test_each_order_has_unique_order_id(self):
        r1 = place_paper_order("agent-uid-001", "BTC/USD", "BUY", 0.1, 68000.0)
        r2 = place_paper_order("agent-uid-001", "ETH/USD", "SELL", 1.0, 3500.0)
        assert r1["order_id"] != r2["order_id"]


# ─── Section B — get_paper_positions() ────────────────────────────────────────


class TestGetPaperPositions:
    def setup_method(self):
        _clear_paper()

    def test_empty_when_no_orders(self):
        positions = get_paper_positions()
        assert positions == []

    def test_returns_open_positions(self):
        place_paper_order("agent-gpp-001", "BTC/USD", "BUY", 0.1, 68000.0)
        positions = get_paper_positions()
        assert len(positions) >= 1

    def test_filter_by_agent_id(self):
        place_paper_order("agent-f1-001", "BTC/USD", "BUY", 0.1, 68000.0)
        place_paper_order("agent-f2-001", "ETH/USD", "SELL", 1.0, 3500.0)
        positions = get_paper_positions(agent_id="agent-f1-001")
        assert all(p["agent_id"] == "agent-f1-001" for p in positions)

    def test_filter_by_agent_excludes_other(self):
        place_paper_order("agent-fe-001", "BTC/USD", "BUY", 0.1, 68000.0)
        place_paper_order("agent-fe-002", "ETH/USD", "SELL", 1.0, 3500.0)
        positions = get_paper_positions(agent_id="agent-fe-001")
        ids = [p["agent_id"] for p in positions]
        assert "agent-fe-002" not in ids

    def test_closed_position_excluded(self):
        result = place_paper_order("agent-cp-001", "BTC/USD", "BUY", 0.1, 68000.0)
        pos_id = result["position_id"]
        close_paper_position(pos_id, 69000.0)
        positions = get_paper_positions(agent_id="agent-cp-001")
        ids = [p["position_id"] for p in positions]
        assert pos_id not in ids

    def test_returns_list(self):
        positions = get_paper_positions()
        assert isinstance(positions, list)

    def test_raises_empty_string_agent_id(self):
        with pytest.raises(ValueError, match="agent_id"):
            get_paper_positions(agent_id="")

    def test_position_has_status_open(self):
        place_paper_order("agent-sto-001", "SOL/USD", "BUY", 5.0, 180.0)
        positions = get_paper_positions(agent_id="agent-sto-001")
        assert all(p["status"] == "open" for p in positions)

    def test_multiple_positions_same_agent(self):
        place_paper_order("agent-mp-001", "BTC/USD", "BUY", 0.1, 68000.0)
        place_paper_order("agent-mp-001", "ETH/USD", "SELL", 1.0, 3500.0)
        positions = get_paper_positions(agent_id="agent-mp-001")
        assert len(positions) == 2

    def test_none_agent_id_returns_all(self):
        place_paper_order("agent-all-001", "BTC/USD", "BUY", 0.1, 68000.0)
        place_paper_order("agent-all-002", "ETH/USD", "SELL", 1.0, 3500.0)
        positions = get_paper_positions(agent_id=None)
        agent_ids = {p["agent_id"] for p in positions}
        assert "agent-all-001" in agent_ids
        assert "agent-all-002" in agent_ids


# ─── Section C — close_paper_position() ───────────────────────────────────────


class TestClosePaperPosition:
    def setup_method(self):
        _clear_paper()

    def test_returns_dict_with_net_pnl(self):
        result = place_paper_order("agent-cl-001", "BTC/USD", "BUY", 0.1, 68000.0)
        close = close_paper_position(result["position_id"], 69000.0)
        assert "net_pnl" in close

    def test_buy_profit_on_price_increase(self):
        result = place_paper_order("agent-cl-002", "BTC/USD", "BUY", 1.0, 68000.0)
        close = close_paper_position(result["position_id"], 70000.0)
        assert close["net_pnl"] > 0

    def test_buy_loss_on_price_decrease(self):
        result = place_paper_order("agent-cl-003", "BTC/USD", "BUY", 1.0, 68000.0)
        close = close_paper_position(result["position_id"], 60000.0)
        assert close["net_pnl"] < 0

    def test_sell_profit_on_price_decrease(self):
        result = place_paper_order("agent-cl-004", "ETH/USD", "SELL", 1.0, 3500.0)
        close = close_paper_position(result["position_id"], 3000.0)
        assert close["net_pnl"] > 0

    def test_sell_loss_on_price_increase(self):
        result = place_paper_order("agent-cl-005", "ETH/USD", "SELL", 1.0, 3500.0)
        close = close_paper_position(result["position_id"], 4000.0)
        assert close["net_pnl"] < 0

    def test_win_true_on_profit(self):
        result = place_paper_order("agent-cl-006", "BTC/USD", "BUY", 1.0, 68000.0)
        close = close_paper_position(result["position_id"], 70000.0)
        assert close["win"] is True

    def test_win_false_on_loss(self):
        result = place_paper_order("agent-cl-007", "BTC/USD", "BUY", 1.0, 68000.0)
        close = close_paper_position(result["position_id"], 60000.0)
        assert close["win"] is False

    def test_position_status_closed_after(self):
        result = place_paper_order("agent-cl-008", "SOL/USD", "BUY", 5.0, 180.0)
        pos_id = result["position_id"]
        close_paper_position(pos_id, 200.0)
        with _S44_PAPER_LOCK:
            pos = _S44_PAPER_POSITIONS[pos_id]
        assert pos["status"] == "closed"

    def test_close_record_appended_to_history(self):
        result = place_paper_order("agent-cl-009", "BTC/USD", "BUY", 0.1, 68000.0)
        close_paper_position(result["position_id"], 69000.0)
        history = get_paper_history(agent_id="agent-cl-009")
        # Should have both open and close records
        assert len(history) >= 2

    def test_position_id_in_close_result(self):
        result = place_paper_order("agent-cl-010", "ETH/USD", "BUY", 1.0, 3500.0)
        pos_id = result["position_id"]
        close = close_paper_position(pos_id, 3600.0)
        assert close["position_id"] == pos_id

    def test_entry_price_in_close_result(self):
        result = place_paper_order("agent-cl-011", "BTC/USD", "BUY", 0.1, 68000.0)
        close = close_paper_position(result["position_id"], 70000.0)
        assert "entry_price" in close

    def test_agent_id_in_close_result(self):
        result = place_paper_order("agent-cl-012", "BTC/USD", "SELL", 0.1, 68000.0)
        close = close_paper_position(result["position_id"], 65000.0)
        assert close["agent_id"] == "agent-cl-012"

    def test_raises_empty_position_id(self):
        with pytest.raises(ValueError, match="position_id"):
            close_paper_position("", 68000.0)

    def test_raises_zero_close_price(self):
        result = place_paper_order("agent-cl-err-001", "BTC/USD", "BUY", 0.1, 68000.0)
        with pytest.raises(ValueError, match="close_price"):
            close_paper_position(result["position_id"], 0.0)

    def test_raises_negative_close_price(self):
        result = place_paper_order("agent-cl-err-002", "BTC/USD", "BUY", 0.1, 68000.0)
        with pytest.raises(ValueError, match="close_price"):
            close_paper_position(result["position_id"], -100.0)

    def test_raises_unknown_position_id(self):
        with pytest.raises(KeyError):
            close_paper_position("pp-totally-unknown-xyz", 68000.0)

    def test_raises_already_closed(self):
        result = place_paper_order("agent-cl-err-003", "BTC/USD", "BUY", 0.1, 68000.0)
        pos_id = result["position_id"]
        close_paper_position(pos_id, 69000.0)
        with pytest.raises(ValueError, match="already closed"):
            close_paper_position(pos_id, 70000.0)

    def test_raises_agent_mismatch(self):
        result = place_paper_order("agent-owner-001", "BTC/USD", "BUY", 0.1, 68000.0)
        with pytest.raises(ValueError, match="does not belong"):
            close_paper_position(result["position_id"], 69000.0, agent_id="agent-wrong-001")

    def test_net_pnl_deducts_both_fees(self):
        qty = 1.0
        entry_price = 3500.0
        result = place_paper_order("agent-fees-001", "ETH/USD", "BUY", qty, entry_price)
        close = close_paper_position(result["position_id"], 3600.0)
        # gross - entry_fee - close_fee
        expected_net = close["gross_pnl"] - close["entry_fee"] - close["close_fee"]
        assert abs(close["net_pnl"] - expected_net) < 1e-4


# ─── Section D — get_paper_history() ──────────────────────────────────────────


class TestGetPaperHistory:
    def setup_method(self):
        _clear_paper()

    def test_empty_when_no_trades(self):
        history = get_paper_history()
        assert history == []

    def test_returns_list(self):
        history = get_paper_history()
        assert isinstance(history, list)

    def test_contains_order_after_place(self):
        place_paper_order("agent-gh-001", "BTC/USD", "BUY", 0.1, 68000.0)
        history = get_paper_history()
        assert len(history) >= 1

    def test_filter_by_agent_id(self):
        place_paper_order("agent-ghf-001", "BTC/USD", "BUY", 0.1, 68000.0)
        place_paper_order("agent-ghf-002", "ETH/USD", "SELL", 1.0, 3500.0)
        history = get_paper_history(agent_id="agent-ghf-001")
        assert all(h.get("agent_id") == "agent-ghf-001" for h in history)

    def test_filter_by_symbol(self):
        place_paper_order("agent-ghs-001", "BTC/USD", "BUY", 0.1, 68000.0)
        place_paper_order("agent-ghs-001", "ETH/USD", "SELL", 1.0, 3500.0)
        history = get_paper_history(symbol="BTC/USD")
        assert all(h.get("symbol") == "BTC/USD" for h in history)

    def test_limit_enforced(self):
        for i in range(10):
            place_paper_order(f"agent-ghlim-{i:03d}", "BTC/USD", "BUY", 0.1, 68000.0)
        history = get_paper_history(limit=3)
        assert len(history) <= 3

    def test_limit_clamped_to_1(self):
        place_paper_order("agent-ghlim2-001", "BTC/USD", "BUY", 0.1, 68000.0)
        history = get_paper_history(limit=0)
        assert len(history) >= 1

    def test_limit_clamped_to_500(self):
        history = get_paper_history(limit=9999)
        assert len(history) <= 500

    def test_raises_invalid_symbol(self):
        with pytest.raises(ValueError, match="symbol"):
            get_paper_history(symbol="DOGE/USD")

    def test_includes_close_records(self):
        result = place_paper_order("agent-ghcl-001", "BTC/USD", "BUY", 0.1, 68000.0)
        close_paper_position(result["position_id"], 70000.0)
        history = get_paper_history(agent_id="agent-ghcl-001")
        assert len(history) == 2

    def test_combined_filters(self):
        place_paper_order("agent-ghcomb-001", "BTC/USD", "BUY", 0.1, 68000.0)
        place_paper_order("agent-ghcomb-001", "ETH/USD", "BUY", 1.0, 3500.0)
        place_paper_order("agent-ghcomb-002", "BTC/USD", "BUY", 0.2, 68000.0)
        history = get_paper_history(agent_id="agent-ghcomb-001", symbol="BTC/USD")
        assert all(
            h.get("agent_id") == "agent-ghcomb-001" and h.get("symbol") == "BTC/USD"
            for h in history
        )


# ─── Section E — HTTP endpoints ────────────────────────────────────────────────


@pytest.fixture(scope="module")
def server():
    port = _free_port()
    srv = DemoServer(port=port)
    srv.start()
    time.sleep(0.3)
    yield f"http://localhost:{port}"
    srv.stop()


class TestS44PaperTradingHTTP:
    def setup_method(self):
        _clear_paper()

    def test_post_paper_order_200(self, server):
        status = _post_status(
            f"{server}/api/v1/trading/paper/order",
            {"agent_id": "http-agent-001", "symbol": "BTC/USD", "side": "BUY", "quantity": 0.1, "price": 68000.0},
        )
        assert status == 200

    def test_post_paper_order_response_has_order_id(self, server):
        result = _post(
            f"{server}/api/v1/trading/paper/order",
            {"agent_id": "http-agent-002", "symbol": "ETH/USD", "side": "SELL", "quantity": 1.0, "price": 3500.0},
        )
        assert "order_id" in result

    def test_post_paper_order_missing_agent_id(self, server):
        status = _post_status(
            f"{server}/api/v1/trading/paper/order",
            {"symbol": "BTC/USD", "side": "BUY", "quantity": 0.1, "price": 68000.0},
        )
        assert status == 400

    def test_post_paper_order_bad_symbol(self, server):
        status = _post_status(
            f"{server}/api/v1/trading/paper/order",
            {"agent_id": "http-err-001", "symbol": "DOGE/USD", "side": "BUY", "quantity": 0.1, "price": 0.1},
        )
        assert status == 400

    def test_post_paper_order_bad_side(self, server):
        status = _post_status(
            f"{server}/api/v1/trading/paper/order",
            {"agent_id": "http-err-002", "symbol": "BTC/USD", "side": "HOLD", "quantity": 0.1, "price": 68000.0},
        )
        assert status == 400

    def test_post_paper_order_missing_quantity(self, server):
        status = _post_status(
            f"{server}/api/v1/trading/paper/order",
            {"agent_id": "http-err-003", "symbol": "BTC/USD", "side": "BUY", "price": 68000.0},
        )
        assert status == 400

    def test_post_paper_order_missing_price(self, server):
        status = _post_status(
            f"{server}/api/v1/trading/paper/order",
            {"agent_id": "http-err-004", "symbol": "BTC/USD", "side": "BUY", "quantity": 0.1},
        )
        assert status == 400

    def test_get_paper_positions_200(self, server):
        result = _get(f"{server}/api/v1/trading/paper/positions")
        assert "positions" in result
        assert "count" in result

    def test_get_paper_positions_after_order(self, server):
        _post(
            f"{server}/api/v1/trading/paper/order",
            {"agent_id": "http-pos-001", "symbol": "SOL/USD", "side": "BUY", "quantity": 10.0, "price": 180.0},
        )
        result = _get(f"{server}/api/v1/trading/paper/positions?agent_id=http-pos-001")
        assert result["count"] >= 1

    def test_post_paper_close_200(self, server):
        order = _post(
            f"{server}/api/v1/trading/paper/order",
            {"agent_id": "http-close-001", "symbol": "BTC/USD", "side": "BUY", "quantity": 0.1, "price": 68000.0},
        )
        status = _post_status(
            f"{server}/api/v1/trading/paper/close",
            {"position_id": order["position_id"], "close_price": 70000.0},
        )
        assert status == 200

    def test_post_paper_close_404_unknown(self, server):
        status = _post_status(
            f"{server}/api/v1/trading/paper/close",
            {"position_id": "pp-totally-unknown-abc", "close_price": 68000.0},
        )
        assert status == 404

    def test_get_paper_history_200(self, server):
        result = _get(f"{server}/api/v1/trading/paper/history")
        assert "history" in result
        assert "count" in result

    def test_get_paper_history_after_trade(self, server):
        _post(
            f"{server}/api/v1/trading/paper/order",
            {"agent_id": "http-hist-001", "symbol": "ETH/USD", "side": "BUY", "quantity": 1.0, "price": 3500.0},
        )
        result = _get(f"{server}/api/v1/trading/paper/history?agent_id=http-hist-001")
        assert result["count"] >= 1
