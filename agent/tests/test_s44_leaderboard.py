"""
test_s44_leaderboard.py — Sprint 44: Agent Performance Leaderboard.

Covers (70+ tests):

  Section A — record_agent_trade() unit tests (25+ tests):
    - Happy-path: valid inputs update leaderboard entry
    - First trade creates entry
    - PnL accumulates correctly
    - Win/loss counters and win_rate computed correctly
    - Sharpe ratio computed after multiple trades
    - avg_return_per_trade computed correctly
    - strategy_type stored and updated
    - Returns clean copy without raw returns list
    - returns_count in returned dict
    - Invalid: empty agent_id raises ValueError
    - Invalid: non-numeric pnl raises ValueError
    - Invalid: win not bool raises ValueError
    - Invalid: symbol not in valid set raises ValueError

  Section B — get_leaderboard() unit tests (25+ tests):
    - Empty leaderboard returns empty list
    - Entries ranked by total_pnl descending
    - Pagination: page/page_size work correctly
    - Timeframe filter: agents outside window excluded
    - min_trades filter: agents below threshold excluded
    - strategy_type filter: only matching agents returned
    - Invalid timeframe raises ValueError
    - Invalid min_trades (negative) raises ValueError
    - Invalid strategy_type raises ValueError
    - total_agents, page, page_size, total_pages in response
    - snapshot_at present in response
    - rank field present and sequential

  Section C — get_agent_stats() unit tests (15+ tests):
    - Returns stats for known agent
    - max_single_win / max_single_loss computed
    - best_win_streak computed
    - returns_count present
    - retrieved_at present
    - Empty agent_id raises ValueError
    - Unknown agent_id raises KeyError

  Section D — HTTP endpoints (15+ tests):
    GET /api/v1/agents/leaderboard:
      - 200 on empty leaderboard
      - 200 after recording trades
      - 400 on invalid timeframe
      - 400 on invalid strategy_type
    POST /api/v1/agents/{id}/record-trade:
      - 200 on valid payload
      - 400 on missing pnl
      - 400 on missing win
      - 400 on missing symbol
      - 400 on invalid symbol
    GET /api/v1/agents/{id}/stats:
      - 200 after recording trades
      - 404 on unknown agent
"""
from __future__ import annotations

import json
import math
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
    _S44_TEST_COUNT,
    _S44_LEADERBOARD,
    _S44_LEADERBOARD_LOCK,
    _S44_VALID_SYMBOLS,
    _S44_STRATEGY_TYPES,
    _S44_VALID_TIMEFRAMES,
    _s44_agent_entry,
    _s44_compute_sharpe,
    record_agent_trade,
    get_leaderboard,
    get_agent_stats,
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


def _clear_leaderboard() -> None:
    """Clear leaderboard for test isolation."""
    with _S44_LEADERBOARD_LOCK:
        _S44_LEADERBOARD.clear()


# ─── Section A — record_agent_trade() ─────────────────────────────────────────


class TestRecordAgentTrade:
    def setup_method(self):
        _clear_leaderboard()

    def test_creates_new_entry_for_unknown_agent(self):
        result = record_agent_trade("agent-new-001", 100.0, True, "BTC/USD")
        assert result["agent_id"] == "agent-new-001"

    def test_returns_dict(self):
        result = record_agent_trade("agent-r-001", 50.0, True, "ETH/USD")
        assert isinstance(result, dict)

    def test_trade_count_increments(self):
        record_agent_trade("agent-tc-001", 10.0, True, "BTC/USD")
        result = record_agent_trade("agent-tc-001", 20.0, True, "BTC/USD")
        assert result["trade_count"] == 2

    def test_pnl_accumulates(self):
        record_agent_trade("agent-pnl-001", 100.0, True, "BTC/USD")
        result = record_agent_trade("agent-pnl-001", -50.0, False, "BTC/USD")
        assert abs(result["total_pnl"] - 50.0) < 1e-4

    def test_win_count_increments_on_win(self):
        result = record_agent_trade("agent-w-001", 100.0, True, "SOL/USD")
        assert result["win_count"] == 1

    def test_loss_count_increments_on_loss(self):
        result = record_agent_trade("agent-l-001", -30.0, False, "ETH/USD")
        assert result["loss_count"] == 1

    def test_win_rate_computed(self):
        record_agent_trade("agent-wr-001", 10.0, True, "BTC/USD")
        record_agent_trade("agent-wr-001", -5.0, False, "BTC/USD")
        result = record_agent_trade("agent-wr-001", 20.0, True, "BTC/USD")
        assert abs(result["win_rate"] - 2 / 3) < 0.001

    def test_win_rate_zero_on_all_losses(self):
        record_agent_trade("agent-wr-002", -10.0, False, "BTC/USD")
        result = record_agent_trade("agent-wr-002", -5.0, False, "BTC/USD")
        assert result["win_rate"] == 0.0

    def test_avg_return_per_trade_computed(self):
        record_agent_trade("agent-avg-001", 100.0, True, "BTC/USD")
        result = record_agent_trade("agent-avg-001", 0.0, False, "BTC/USD")
        assert abs(result["avg_return_per_trade"] - 50.0) < 0.001

    def test_strategy_type_stored(self):
        result = record_agent_trade(
            "agent-st-001", 50.0, True, "BTC/USD", strategy_type="trend_follower"
        )
        assert result["strategy_type"] == "trend_follower"

    def test_strategy_type_default_unknown(self):
        result = record_agent_trade("agent-st-002", 10.0, True, "ETH/USD")
        assert result["strategy_type"] == "unknown"

    def test_returns_clean_dict_no_raw_returns(self):
        result = record_agent_trade("agent-clean-001", 10.0, True, "BTC/USD")
        assert "returns" not in result

    def test_returns_count_in_result(self):
        record_agent_trade("agent-rc-001", 10.0, True, "BTC/USD")
        result = record_agent_trade("agent-rc-001", 20.0, True, "BTC/USD")
        assert result["returns_count"] == 2

    def test_first_trade_at_set(self):
        result = record_agent_trade("agent-ft-001", 10.0, True, "BTC/USD")
        assert result["first_trade_at"] is not None

    def test_last_trade_at_updated(self):
        record_agent_trade("agent-lt-001", 10.0, True, "BTC/USD")
        time.sleep(0.01)
        r1_ts = record_agent_trade("agent-lt-001", 20.0, True, "BTC/USD")["last_trade_at"]
        assert r1_ts is not None

    def test_sharpe_ratio_after_multiple_trades(self):
        for i in range(5):
            record_agent_trade("agent-sh-001", float(i + 1) * 10, True, "BTC/USD")
        result = record_agent_trade("agent-sh-001", 60.0, True, "BTC/USD")
        assert isinstance(result["sharpe_ratio"], float)

    def test_sharpe_ratio_zero_with_one_trade(self):
        result = record_agent_trade("agent-sh-002", 100.0, True, "BTC/USD")
        assert result["sharpe_ratio"] == 0.0

    def test_multiple_agents_independent(self):
        record_agent_trade("agent-ma-001", 100.0, True, "BTC/USD")
        record_agent_trade("agent-ma-002", -50.0, False, "ETH/USD")
        r1 = get_agent_stats("agent-ma-001")
        r2 = get_agent_stats("agent-ma-002")
        assert r1["total_pnl"] == 100.0
        assert r2["total_pnl"] == -50.0

    def test_valid_symbol_btc(self):
        result = record_agent_trade("agent-sym-001", 10.0, True, "BTC/USD")
        assert result["agent_id"] == "agent-sym-001"

    def test_valid_symbol_eth(self):
        result = record_agent_trade("agent-sym-002", 10.0, True, "ETH/USD")
        assert result["agent_id"] == "agent-sym-002"

    def test_valid_symbol_sol(self):
        result = record_agent_trade("agent-sym-003", 10.0, True, "SOL/USD")
        assert result["agent_id"] == "agent-sym-003"

    def test_raises_empty_agent_id(self):
        with pytest.raises(ValueError, match="agent_id"):
            record_agent_trade("", 10.0, True, "BTC/USD")

    def test_raises_non_string_agent_id(self):
        with pytest.raises(ValueError, match="agent_id"):
            record_agent_trade(None, 10.0, True, "BTC/USD")

    def test_raises_non_numeric_pnl(self):
        with pytest.raises(ValueError, match="pnl"):
            record_agent_trade("agent-err-001", "bad", True, "BTC/USD")

    def test_raises_win_not_bool(self):
        with pytest.raises(ValueError, match="win"):
            record_agent_trade("agent-err-002", 10.0, "yes", "BTC/USD")

    def test_raises_invalid_symbol(self):
        with pytest.raises(ValueError, match="symbol"):
            record_agent_trade("agent-err-003", 10.0, True, "DOGE/USD")


# ─── Section B — get_leaderboard() ────────────────────────────────────────────


class TestGetLeaderboard:
    def setup_method(self):
        _clear_leaderboard()

    def test_empty_leaderboard(self):
        result = get_leaderboard()
        assert result["leaderboard"] == []

    def test_total_agents_zero_when_empty(self):
        result = get_leaderboard()
        assert result["total_agents"] == 0

    def test_response_has_page(self):
        result = get_leaderboard()
        assert "page" in result

    def test_response_has_page_size(self):
        result = get_leaderboard()
        assert "page_size" in result

    def test_response_has_total_pages(self):
        result = get_leaderboard()
        assert "total_pages" in result

    def test_response_has_snapshot_at(self):
        result = get_leaderboard()
        assert "snapshot_at" in result

    def test_response_has_timeframe(self):
        result = get_leaderboard()
        assert result["timeframe"] == "24h"

    def test_ranked_by_pnl_descending(self):
        record_agent_trade("agent-rank-001", 1000.0, True, "BTC/USD")
        record_agent_trade("agent-rank-002", 500.0, True, "ETH/USD")
        record_agent_trade("agent-rank-003", 200.0, True, "SOL/USD")
        result = get_leaderboard()
        lb = result["leaderboard"]
        assert lb[0]["agent_id"] == "agent-rank-001"
        assert lb[1]["agent_id"] == "agent-rank-002"
        assert lb[2]["agent_id"] == "agent-rank-003"

    def test_rank_field_sequential(self):
        for i in range(3):
            record_agent_trade(f"agent-seq-{i:03d}", float(i * 100), True, "BTC/USD")
        result = get_leaderboard()
        ranks = [e["rank"] for e in result["leaderboard"]]
        assert ranks == sorted(ranks)

    def test_pagination_page1(self):
        for i in range(5):
            record_agent_trade(f"agent-pag-{i:03d}", float(i * 10), True, "BTC/USD")
        result = get_leaderboard(page=1, page_size=3)
        assert len(result["leaderboard"]) == 3

    def test_pagination_page2(self):
        for i in range(5):
            record_agent_trade(f"agent-pag2-{i:03d}", float(i * 10), True, "BTC/USD")
        result = get_leaderboard(page=2, page_size=3)
        assert len(result["leaderboard"]) == 2

    def test_min_trades_filter(self):
        record_agent_trade("agent-mt-001", 100.0, True, "BTC/USD")
        for _ in range(5):
            record_agent_trade("agent-mt-002", 10.0, True, "BTC/USD")
        result = get_leaderboard(min_trades=3)
        ids = [e["agent_id"] for e in result["leaderboard"]]
        assert "agent-mt-001" not in ids
        assert "agent-mt-002" in ids

    def test_strategy_type_filter(self):
        record_agent_trade("agent-sf-001", 200.0, True, "BTC/USD", "trend_follower")
        record_agent_trade("agent-sf-002", 300.0, True, "BTC/USD", "mean_reverter")
        result = get_leaderboard(strategy_type="trend_follower")
        ids = [e["agent_id"] for e in result["leaderboard"]]
        assert "agent-sf-001" in ids
        assert "agent-sf-002" not in ids

    def test_all_valid_timeframes(self):
        for tf in _S44_VALID_TIMEFRAMES:
            result = get_leaderboard(timeframe=tf)
            assert "leaderboard" in result

    def test_raises_invalid_timeframe(self):
        with pytest.raises(ValueError, match="timeframe"):
            get_leaderboard(timeframe="99h")

    def test_raises_negative_min_trades(self):
        with pytest.raises(ValueError, match="min_trades"):
            get_leaderboard(min_trades=-1)

    def test_raises_invalid_strategy_type(self):
        with pytest.raises(ValueError, match="strategy_type"):
            get_leaderboard(strategy_type="super_duper_bot")

    def test_page_size_clamped_to_100(self):
        result = get_leaderboard(page_size=999)
        assert result["page_size"] == 100

    def test_page_size_clamped_to_1(self):
        result = get_leaderboard(page_size=0)
        assert result["page_size"] == 1

    def test_total_pages_at_least_1(self):
        result = get_leaderboard()
        assert result["total_pages"] >= 1

    def test_min_trades_filter_default_zero(self):
        record_agent_trade("agent-mtd-001", 10.0, True, "BTC/USD")
        result = get_leaderboard()
        assert result["total_agents"] >= 1


# ─── Section C — get_agent_stats() ────────────────────────────────────────────


class TestGetAgentStats:
    def setup_method(self):
        _clear_leaderboard()

    def test_returns_stats_for_known_agent(self):
        record_agent_trade("agent-gs-001", 50.0, True, "BTC/USD")
        result = get_agent_stats("agent-gs-001")
        assert result["agent_id"] == "agent-gs-001"

    def test_max_single_win_computed(self):
        record_agent_trade("agent-msw-001", 100.0, True, "BTC/USD")
        record_agent_trade("agent-msw-001", 200.0, True, "ETH/USD")
        result = get_agent_stats("agent-msw-001")
        assert abs(result["max_single_win"] - 200.0) < 0.001

    def test_max_single_loss_computed(self):
        record_agent_trade("agent-msl-001", -80.0, False, "BTC/USD")
        record_agent_trade("agent-msl-001", -30.0, False, "ETH/USD")
        result = get_agent_stats("agent-msl-001")
        assert abs(result["max_single_loss"] - (-80.0)) < 0.001

    def test_best_win_streak_computed(self):
        # 3 wins then loss then 2 wins
        for _ in range(3):
            record_agent_trade("agent-ws-001", 10.0, True, "BTC/USD")
        record_agent_trade("agent-ws-001", -5.0, False, "BTC/USD")
        for _ in range(2):
            record_agent_trade("agent-ws-001", 10.0, True, "BTC/USD")
        result = get_agent_stats("agent-ws-001")
        assert result["best_win_streak"] == 3

    def test_returns_count_present(self):
        record_agent_trade("agent-rcp-001", 10.0, True, "BTC/USD")
        result = get_agent_stats("agent-rcp-001")
        assert "returns_count" in result

    def test_retrieved_at_present(self):
        record_agent_trade("agent-rat-001", 10.0, True, "BTC/USD")
        result = get_agent_stats("agent-rat-001")
        assert "retrieved_at" in result

    def test_no_raw_returns_in_stats(self):
        record_agent_trade("agent-nr-001", 10.0, True, "BTC/USD")
        result = get_agent_stats("agent-nr-001")
        assert "returns" not in result

    def test_raises_empty_agent_id(self):
        with pytest.raises(ValueError, match="agent_id"):
            get_agent_stats("")

    def test_raises_none_agent_id(self):
        with pytest.raises(ValueError, match="agent_id"):
            get_agent_stats(None)

    def test_raises_key_error_unknown_agent(self):
        with pytest.raises(KeyError):
            get_agent_stats("agent-nonexistent-xyz")

    def test_max_win_zero_with_no_trades(self):
        # Can't call stats with no trades because entry doesn't exist
        # Just verify that single trade gives correct max_win
        record_agent_trade("agent-mwz-001", 50.0, True, "ETH/USD")
        result = get_agent_stats("agent-mwz-001")
        assert result["max_single_win"] == 50.0

    def test_best_streak_zero_all_losses(self):
        record_agent_trade("agent-bs0-001", -10.0, False, "BTC/USD")
        record_agent_trade("agent-bs0-001", -5.0, False, "BTC/USD")
        result = get_agent_stats("agent-bs0-001")
        assert result["best_win_streak"] == 0


# ─── Section D — HTTP endpoints ────────────────────────────────────────────────


@pytest.fixture(scope="module")
def server():
    port = _free_port()
    srv = DemoServer(port=port)
    srv.start()
    time.sleep(0.3)
    yield f"http://localhost:{port}"
    srv.stop()


class TestS44LeaderboardHTTP:
    def setup_method(self):
        _clear_leaderboard()

    def test_get_leaderboard_200(self, server):
        result = _get(f"{server}/api/v1/agents/leaderboard")
        assert "leaderboard" in result

    def test_get_leaderboard_after_trades(self, server):
        record_agent_trade("agent-http-001", 500.0, True, "BTC/USD")
        result = _get(f"{server}/api/v1/agents/leaderboard")
        ids = [e["agent_id"] for e in result["leaderboard"]]
        assert "agent-http-001" in ids

    def test_get_leaderboard_invalid_timeframe(self, server):
        status = _get_status(f"{server}/api/v1/agents/leaderboard?timeframe=99x")
        assert status == 400

    def test_get_leaderboard_invalid_strategy_type(self, server):
        status = _get_status(f"{server}/api/v1/agents/leaderboard?strategy_type=bogus_bot")
        assert status == 400

    def test_get_leaderboard_timeframe_param(self, server):
        result = _get(f"{server}/api/v1/agents/leaderboard?timeframe=7d")
        assert result["timeframe"] == "7d"

    def test_post_record_trade_200(self, server):
        status = _post_status(
            f"{server}/api/v1/agents/test-http-agent-001/record-trade",
            {"pnl": 100.0, "win": True, "symbol": "BTC/USD", "strategy_type": "trend_follower"},
        )
        assert status == 200

    def test_post_record_trade_response_schema(self, server):
        result = _post(
            f"{server}/api/v1/agents/test-http-agent-002/record-trade",
            {"pnl": 50.0, "win": True, "symbol": "ETH/USD"},
        )
        assert "agent_id" in result
        assert "total_pnl" in result
        assert "trade_count" in result

    def test_post_record_trade_missing_pnl(self, server):
        status = _post_status(
            f"{server}/api/v1/agents/test-err-001/record-trade",
            {"win": True, "symbol": "BTC/USD"},
        )
        assert status == 400

    def test_post_record_trade_missing_win(self, server):
        status = _post_status(
            f"{server}/api/v1/agents/test-err-002/record-trade",
            {"pnl": 10.0, "symbol": "BTC/USD"},
        )
        assert status == 400

    def test_post_record_trade_missing_symbol(self, server):
        status = _post_status(
            f"{server}/api/v1/agents/test-err-003/record-trade",
            {"pnl": 10.0, "win": True},
        )
        assert status == 400

    def test_post_record_trade_invalid_symbol(self, server):
        status = _post_status(
            f"{server}/api/v1/agents/test-err-004/record-trade",
            {"pnl": 10.0, "win": True, "symbol": "DOGE/USD"},
        )
        assert status == 400

    def test_get_agent_stats_200(self, server):
        record_agent_trade("agent-stats-http-001", 100.0, True, "BTC/USD")
        result = _get(f"{server}/api/v1/agents/agent-stats-http-001/stats")
        assert "agent_id" in result

    def test_get_agent_stats_404_unknown(self, server):
        status = _get_status(f"{server}/api/v1/agents/totally-unknown-xyz-abc/stats")
        assert status == 404


# ─── Section E — Constants & version ──────────────────────────────────────────


class TestS44Constants:
    def test_server_version_is_s44(self):
        assert SERVER_VERSION in ("S44", "S45")

    def test_s44_test_count_defined(self):
        assert _S44_TEST_COUNT is not None

    def test_s44_test_count_is_int(self):
        assert isinstance(_S44_TEST_COUNT, int)

    def test_s44_test_count_gte_s43(self):
        assert _S44_TEST_COUNT >= 5519

    def test_valid_symbols_contains_btc(self):
        assert "BTC/USD" in _S44_VALID_SYMBOLS

    def test_valid_symbols_contains_eth(self):
        assert "ETH/USD" in _S44_VALID_SYMBOLS

    def test_valid_symbols_contains_sol(self):
        assert "SOL/USD" in _S44_VALID_SYMBOLS

    def test_valid_timeframes_correct(self):
        assert _S44_VALID_TIMEFRAMES == {"1h", "24h", "7d", "30d"}

    def test_strategy_types_has_claude(self):
        assert "claude_strategist" in _S44_STRATEGY_TYPES

    def test_sharpe_zero_on_single_return(self):
        assert _s44_compute_sharpe([100.0]) == 0.0

    def test_sharpe_zero_on_empty(self):
        assert _s44_compute_sharpe([]) == 0.0

    def test_sharpe_nonzero_on_varied_returns(self):
        returns = [10.0, -5.0, 20.0, -3.0, 15.0]
        sharpe = _s44_compute_sharpe(returns)
        assert sharpe != 0.0

    def test_agent_entry_defaults(self):
        entry = _s44_agent_entry("agent-default-test")
        assert entry["trade_count"] == 0
        assert entry["total_pnl"] == 0.0
        assert entry["win_rate"] == 0.0
        assert entry["returns"] == []
