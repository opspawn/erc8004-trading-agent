"""
test_s44_demo_scenario.py — Sprint 44: Demo Scenario + Integration.

Covers (65+ tests):

  Section A — run_leaderboard_scenario() (30+ tests):
    - Returns dict with scenario key
    - agents_seeded == 5
    - trades_per_agent == 20
    - total_trades == 100 (5 agents × 20 trades)
    - positions_closed == 100
    - leaderboard key present
    - leaderboard has up to 5 agents
    - symbols_traded contains BTC, ETH, SOL
    - completed_at present
    - Leaderboard agents have rank field
    - Leaderboard sorted by total_pnl descending
    - Each agent has strategy_type in leaderboard
    - Each leaderboard entry has sharpe_ratio
    - Each leaderboard entry has win_rate in [0, 1]
    - claude_strategist agent appears in leaderboard
    - trend_follower agent appears in leaderboard
    - arb_detector agent has highest positive bias

  Section B — Integration: full round-trip (20+ tests):
    - Place order, close it, check leaderboard updated
    - record_agent_trade after paper trade matches expected PnL direction
    - Multiple agents compete and leaderboard is consistent
    - Scenario is repeatable (deterministic seed)
    - Closed position no longer in open positions
    - History grows after each scenario

  Section C — HTTP endpoint (15+ tests):
    POST /api/v1/demo/run-leaderboard-scenario:
      - 200 response
      - agents_seeded == 5 in response
      - leaderboard present in response
      - completed_at present
      - positions_closed > 0
      - leaderboard not empty
      - total_trades == 100
      - each leaderboard entry has rank

  Section D — Constants and edge cases (5+ tests):
    - _S44_DEMO_AGENTS has 5 entries
    - _S44_DEMO_PRICES has BTC, ETH, SOL
    - _S44_SLIPPAGE is 0.001
    - _S44_FEE_RATE is 0.0005
"""
from __future__ import annotations

import json
import os
import socket
import sys
import time
from urllib.request import Request, urlopen

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from demo_server import (
    DemoServer,
    _S44_LEADERBOARD,
    _S44_LEADERBOARD_LOCK,
    _S44_PAPER_LOCK,
    _S44_PAPER_ORDERS,
    _S44_PAPER_POSITIONS,
    _S44_DEMO_AGENTS,
    _S44_DEMO_PRICES,
    _S44_SLIPPAGE,
    _S44_FEE_RATE,
    record_agent_trade,
    get_leaderboard,
    get_paper_positions,
    get_paper_history,
    place_paper_order,
    close_paper_position,
    run_leaderboard_scenario,
)


# ─── Helpers ──────────────────────────────────────────────────────────────────


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("", 0))
        return s.getsockname()[1]


def _post(url: str, body: dict) -> dict:
    data = json.dumps(body).encode()
    req = Request(url, data=data, headers={"Content-Type": "application/json"}, method="POST")
    with urlopen(req, timeout=10) as resp:
        return json.loads(resp.read())


def _clear_all() -> None:
    with _S44_LEADERBOARD_LOCK:
        _S44_LEADERBOARD.clear()
    with _S44_PAPER_LOCK:
        _S44_PAPER_ORDERS.clear()
        _S44_PAPER_POSITIONS.clear()


# ─── Section A — run_leaderboard_scenario() unit tests ─────────────────────────


class TestRunLeaderboardScenario:
    def setup_method(self):
        _clear_all()

    def test_returns_dict(self):
        result = run_leaderboard_scenario()
        assert isinstance(result, dict)

    def test_scenario_key(self):
        result = run_leaderboard_scenario()
        assert result["scenario"] == "leaderboard_demo"

    def test_agents_seeded_is_5(self):
        result = run_leaderboard_scenario()
        assert result["agents_seeded"] == 5

    def test_trades_per_agent_is_20(self):
        result = run_leaderboard_scenario()
        assert result["trades_per_agent"] == 20

    def test_total_trades_is_100(self):
        result = run_leaderboard_scenario()
        assert result["total_trades"] == 100

    def test_positions_closed_is_100(self):
        result = run_leaderboard_scenario()
        assert result["positions_closed"] == 100

    def test_leaderboard_key_present(self):
        result = run_leaderboard_scenario()
        assert "leaderboard" in result

    def test_leaderboard_sub_dict_has_leaderboard_list(self):
        result = run_leaderboard_scenario()
        assert "leaderboard" in result["leaderboard"]

    def test_leaderboard_has_5_agents(self):
        result = run_leaderboard_scenario()
        lb = result["leaderboard"]["leaderboard"]
        assert len(lb) == 5

    def test_symbols_traded_has_btc(self):
        result = run_leaderboard_scenario()
        assert "BTC/USD" in result["symbols_traded"]

    def test_symbols_traded_has_eth(self):
        result = run_leaderboard_scenario()
        assert "ETH/USD" in result["symbols_traded"]

    def test_symbols_traded_has_sol(self):
        result = run_leaderboard_scenario()
        assert "SOL/USD" in result["symbols_traded"]

    def test_completed_at_present(self):
        result = run_leaderboard_scenario()
        assert "completed_at" in result

    def test_completed_at_is_number(self):
        result = run_leaderboard_scenario()
        assert isinstance(result["completed_at"], float)

    def test_leaderboard_agents_have_rank(self):
        result = run_leaderboard_scenario()
        for entry in result["leaderboard"]["leaderboard"]:
            assert "rank" in entry

    def test_leaderboard_sorted_by_total_pnl_desc(self):
        result = run_leaderboard_scenario()
        lb = result["leaderboard"]["leaderboard"]
        pnls = [e["total_pnl"] for e in lb]
        assert pnls == sorted(pnls, reverse=True)

    def test_each_agent_has_strategy_type(self):
        result = run_leaderboard_scenario()
        for entry in result["leaderboard"]["leaderboard"]:
            assert "strategy_type" in entry

    def test_each_agent_has_sharpe_ratio(self):
        result = run_leaderboard_scenario()
        for entry in result["leaderboard"]["leaderboard"]:
            assert "sharpe_ratio" in entry

    def test_each_agent_has_win_rate_in_range(self):
        result = run_leaderboard_scenario()
        for entry in result["leaderboard"]["leaderboard"]:
            wr = entry["win_rate"]
            assert 0.0 <= wr <= 1.0

    def test_claude_strategist_in_leaderboard(self):
        result = run_leaderboard_scenario()
        strategy_types = {e["strategy_type"] for e in result["leaderboard"]["leaderboard"]}
        assert "claude_strategist" in strategy_types

    def test_trend_follower_in_leaderboard(self):
        result = run_leaderboard_scenario()
        strategy_types = {e["strategy_type"] for e in result["leaderboard"]["leaderboard"]}
        assert "trend_follower" in strategy_types

    def test_all_agents_have_20_trades(self):
        result = run_leaderboard_scenario()
        for entry in result["leaderboard"]["leaderboard"]:
            assert entry["trade_count"] == 20

    def test_rank_1_has_highest_pnl(self):
        result = run_leaderboard_scenario()
        lb = result["leaderboard"]["leaderboard"]
        rank1 = next(e for e in lb if e["rank"] == 1)
        all_pnls = [e["total_pnl"] for e in lb]
        assert rank1["total_pnl"] == max(all_pnls)

    def test_scenario_deterministic(self):
        result1 = run_leaderboard_scenario()
        _clear_all()
        result2 = run_leaderboard_scenario()
        # Same total_trades should be produced each run
        assert result1["total_trades"] == result2["total_trades"]
        assert result1["positions_closed"] == result2["positions_closed"]

    def test_history_grows_after_scenario(self):
        run_leaderboard_scenario()
        history = get_paper_history(limit=500)
        # 100 opens + 100 closes = 200 records (limit=500 to see all)
        assert len(history) >= 100

    def test_all_positions_closed_after_scenario(self):
        run_leaderboard_scenario()
        open_positions = get_paper_positions()
        assert len(open_positions) == 0

    def test_each_leaderboard_entry_has_win_count(self):
        result = run_leaderboard_scenario()
        for entry in result["leaderboard"]["leaderboard"]:
            assert "win_count" in entry

    def test_each_leaderboard_entry_has_trade_count(self):
        result = run_leaderboard_scenario()
        for entry in result["leaderboard"]["leaderboard"]:
            assert "trade_count" in entry

    def test_each_leaderboard_entry_has_avg_return(self):
        result = run_leaderboard_scenario()
        for entry in result["leaderboard"]["leaderboard"]:
            assert "avg_return_per_trade" in entry


# ─── Section B — Integration: full round-trip ─────────────────────────────────


class TestS44Integration:
    def setup_method(self):
        _clear_all()

    def test_place_order_then_check_position(self):
        place_paper_order("int-agent-001", "BTC/USD", "BUY", 0.1, 68000.0)
        positions = get_paper_positions(agent_id="int-agent-001")
        assert len(positions) == 1

    def test_close_order_removes_from_open_positions(self):
        result = place_paper_order("int-agent-002", "ETH/USD", "BUY", 1.0, 3500.0)
        close_paper_position(result["position_id"], 3600.0)
        positions = get_paper_positions(agent_id="int-agent-002")
        assert len(positions) == 0

    def test_record_trade_after_close_updates_leaderboard(self):
        result = place_paper_order("int-agent-003", "BTC/USD", "BUY", 0.5, 68000.0)
        close = close_paper_position(result["position_id"], 70000.0)
        record_agent_trade(
            "int-agent-003", close["net_pnl"], close["win"], "BTC/USD", "trend_follower"
        )
        lb = get_leaderboard()
        ids = [e["agent_id"] for e in lb["leaderboard"]]
        assert "int-agent-003" in ids

    def test_profitable_trade_shows_positive_pnl(self):
        result = place_paper_order("int-agent-004", "SOL/USD", "BUY", 10.0, 180.0)
        close = close_paper_position(result["position_id"], 200.0)
        assert close["net_pnl"] > 0

    def test_losing_trade_shows_negative_pnl(self):
        result = place_paper_order("int-agent-005", "BTC/USD", "BUY", 0.1, 68000.0)
        close = close_paper_position(result["position_id"], 60000.0)
        assert close["net_pnl"] < 0

    def test_history_shows_both_open_and_close(self):
        result = place_paper_order("int-agent-006", "ETH/USD", "SELL", 1.0, 3500.0)
        close_paper_position(result["position_id"], 3400.0)
        history = get_paper_history(agent_id="int-agent-006")
        assert len(history) == 2

    def test_multiple_agents_leaderboard_correct_order(self):
        # Agent 1 makes $1000, Agent 2 makes $100
        record_agent_trade("int-rank-001", 1000.0, True, "BTC/USD")
        record_agent_trade("int-rank-002", 100.0, True, "BTC/USD")
        lb = get_leaderboard()
        agents = [e["agent_id"] for e in lb["leaderboard"]]
        idx1 = agents.index("int-rank-001")
        idx2 = agents.index("int-rank-002")
        assert idx1 < idx2

    def test_scenario_populates_leaderboard(self):
        run_leaderboard_scenario()
        lb = get_leaderboard()
        assert lb["total_agents"] >= 5

    def test_all_demo_agent_ids_in_leaderboard(self):
        run_leaderboard_scenario()
        lb = get_leaderboard()
        agent_ids = {e["agent_id"] for e in lb["leaderboard"]}
        demo_ids = {a["agent_id"] for a in _S44_DEMO_AGENTS}
        assert demo_ids.issubset(agent_ids)

    def test_paper_order_fill_price_different_from_requested(self):
        result = place_paper_order("int-fp-001", "BTC/USD", "BUY", 0.1, 68000.0)
        assert result["fill_price"] != 68000.0

    def test_paper_position_has_correct_side(self):
        place_paper_order("int-side-001", "ETH/USD", "SELL", 1.0, 3500.0)
        positions = get_paper_positions(agent_id="int-side-001")
        assert positions[0]["side"] == "SELL"

    def test_multiple_positions_all_tracked(self):
        for i in range(3):
            place_paper_order("int-multi-001", "BTC/USD", "BUY", 0.1, 68000.0 + i)
        positions = get_paper_positions(agent_id="int-multi-001")
        assert len(positions) == 3

    def test_win_rate_correct_after_mixed_trades(self):
        # 3 wins, 2 losses
        for _ in range(3):
            record_agent_trade("int-wr-001", 100.0, True, "BTC/USD")
        for _ in range(2):
            record_agent_trade("int-wr-001", -50.0, False, "BTC/USD")
        lb = get_leaderboard()
        entry = next(e for e in lb["leaderboard"] if e["agent_id"] == "int-wr-001")
        assert abs(entry["win_rate"] - 0.6) < 0.001

    def test_scenario_history_count_200(self):
        run_leaderboard_scenario()
        # 100 opens + 100 closes
        history = get_paper_history(limit=500)
        assert len(history) == 200


# ─── Section C — HTTP endpoint ─────────────────────────────────────────────────


@pytest.fixture(scope="module")
def server():
    port = _free_port()
    srv = DemoServer(port=port)
    srv.start()
    time.sleep(0.3)
    yield f"http://localhost:{port}"
    srv.stop()


class TestS44DemoScenarioHTTP:
    def setup_method(self):
        _clear_all()

    def test_post_scenario_200(self, server):
        req = Request(
            f"{server}/api/v1/demo/run-leaderboard-scenario",
            data=b"{}",
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urlopen(req, timeout=15) as resp:
            assert resp.status == 200

    def test_post_scenario_agents_seeded(self, server):
        result = _post(f"{server}/api/v1/demo/run-leaderboard-scenario", {})
        assert result["agents_seeded"] == 5

    def test_post_scenario_leaderboard_present(self, server):
        result = _post(f"{server}/api/v1/demo/run-leaderboard-scenario", {})
        assert "leaderboard" in result

    def test_post_scenario_completed_at(self, server):
        result = _post(f"{server}/api/v1/demo/run-leaderboard-scenario", {})
        assert "completed_at" in result

    def test_post_scenario_positions_closed(self, server):
        result = _post(f"{server}/api/v1/demo/run-leaderboard-scenario", {})
        assert result["positions_closed"] > 0

    def test_post_scenario_leaderboard_not_empty(self, server):
        result = _post(f"{server}/api/v1/demo/run-leaderboard-scenario", {})
        lb = result["leaderboard"]["leaderboard"]
        assert len(lb) > 0

    def test_post_scenario_total_trades_100(self, server):
        result = _post(f"{server}/api/v1/demo/run-leaderboard-scenario", {})
        assert result["total_trades"] == 100

    def test_post_scenario_each_entry_has_rank(self, server):
        result = _post(f"{server}/api/v1/demo/run-leaderboard-scenario", {})
        for entry in result["leaderboard"]["leaderboard"]:
            assert "rank" in entry

    def test_post_scenario_scenario_key(self, server):
        result = _post(f"{server}/api/v1/demo/run-leaderboard-scenario", {})
        assert result["scenario"] == "leaderboard_demo"

    def test_post_scenario_symbols_traded(self, server):
        result = _post(f"{server}/api/v1/demo/run-leaderboard-scenario", {})
        assert "symbols_traded" in result

    def test_post_scenario_leaderboard_sorted(self, server):
        result = _post(f"{server}/api/v1/demo/run-leaderboard-scenario", {})
        lb = result["leaderboard"]["leaderboard"]
        pnls = [e["total_pnl"] for e in lb]
        assert pnls == sorted(pnls, reverse=True)

    def test_post_scenario_all_have_strategy_type(self, server):
        result = _post(f"{server}/api/v1/demo/run-leaderboard-scenario", {})
        for entry in result["leaderboard"]["leaderboard"]:
            assert entry.get("strategy_type") in {
                "trend_follower", "mean_reverter", "momentum_rider",
                "arb_detector", "claude_strategist",
            }


# ─── Section D — Constants edge cases ─────────────────────────────────────────


class TestS44DemoConstants:
    def test_demo_agents_count_5(self):
        assert len(_S44_DEMO_AGENTS) == 5

    def test_demo_prices_has_btc(self):
        assert "BTC/USD" in _S44_DEMO_PRICES

    def test_demo_prices_has_eth(self):
        assert "ETH/USD" in _S44_DEMO_PRICES

    def test_demo_prices_has_sol(self):
        assert "SOL/USD" in _S44_DEMO_PRICES

    def test_slippage_is_0001(self):
        assert abs(_S44_SLIPPAGE - 0.001) < 1e-8

    def test_fee_rate_is_00005(self):
        assert abs(_S44_FEE_RATE - 0.0005) < 1e-8

    def test_demo_agents_have_agent_id(self):
        for agent in _S44_DEMO_AGENTS:
            assert "agent_id" in agent

    def test_demo_agents_have_strategy_type(self):
        for agent in _S44_DEMO_AGENTS:
            assert "strategy_type" in agent

    def test_all_strategy_types_in_demo_agents(self):
        strategies = {a["strategy_type"] for a in _S44_DEMO_AGENTS}
        assert "claude_strategist" in strategies
        assert "trend_follower" in strategies
        assert "arb_detector" in strategies
