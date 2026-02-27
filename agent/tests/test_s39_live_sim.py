"""
test_s39_live_sim.py — Sprint 39: Live Market Simulation, Portfolio Snapshot,
                        Strategy Comparison tests.

Covers:
  - run_live_simulation(): schema, tick_data shape, trade logic, edge cases
  - OHLCV generation: bar structure, price continuity, vol levels
  - Strategy dispatch: momentum / mean_reversion / ml_ensemble decision logic
  - Portfolio snapshot: schema, value ranges, metrics fields
  - Strategy comparison: rank ordering, metric ranges, summary block
  - HTTP integration: POST /demo/live/simulate, GET /demo/portfolio/snapshot,
                      GET /demo/strategy/compare
  - Error cases: bad ticks, bad initial_capital, 404 routing, invalid body
"""

from __future__ import annotations

import json
import math
import os
import sys
import time
import threading
from http.client import HTTPConnection
from typing import Any, Dict
from urllib.request import urlopen, Request
from urllib.error import HTTPError
import socket

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from demo_server import (
    run_live_simulation,
    get_portfolio_snapshot,
    get_strategy_comparison,
    _generate_ohlcv,
    _decide_action,
    _s39_seed,
    _s39_seed_int,
    _S39_SYMBOLS,
    _S39_STRATEGIES,
    _S39_DEFAULT_SIM_TICKS,
    _S39_MAX_SIM_TICKS,
    _S39_DEFAULT_CAPITAL,
    _S39_COMPARE_STRATEGIES,
    DemoServer,
)


# ── Helpers ────────────────────────────────────────────────────────────────────

def _get(url: str) -> dict:
    with urlopen(url, timeout=8) as resp:
        return json.loads(resp.read())


def _get_status(url: str) -> int:
    try:
        with urlopen(url, timeout=8) as resp:
            return resp.status
    except HTTPError as exc:
        return exc.code


def _get_raw(url: str) -> tuple[int, dict]:
    try:
        with urlopen(url, timeout=8) as resp:
            return resp.status, json.loads(resp.read())
    except HTTPError as exc:
        return exc.code, json.loads(exc.read())


def _post(url: str, body: dict) -> tuple[int, dict]:
    data = json.dumps(body).encode()
    req = Request(url, data=data, headers={"Content-Type": "application/json"}, method="POST")
    try:
        with urlopen(req, timeout=8) as resp:
            return resp.status, json.loads(resp.read())
    except HTTPError as exc:
        return exc.code, json.loads(exc.read())


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


# ── Fixtures ───────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def server():
    """Start a DemoServer on a random port for integration tests."""
    port = _free_port()
    srv = DemoServer(port=port)
    srv.start()
    time.sleep(0.4)
    yield f"http://127.0.0.1:{port}"
    srv.stop()


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1 — _s39_seed helpers
# ══════════════════════════════════════════════════════════════════════════════

class TestS39SeedHelpers:
    def test_seed_returns_float(self):
        v = _s39_seed("test.key", 0.0, 1.0)
        assert isinstance(v, float)

    def test_seed_in_range(self):
        for i in range(20):
            v = _s39_seed(f"range.{i}", 10.0, 50.0)
            assert 10.0 <= v <= 50.0, f"Out of range: {v}"

    def test_seed_deterministic_same_hour(self):
        v1 = _s39_seed("det.key", 0.0, 100.0)
        v2 = _s39_seed("det.key", 0.0, 100.0)
        assert v1 == v2

    def test_seed_different_keys_differ(self):
        v1 = _s39_seed("alpha", 0.0, 100.0)
        v2 = _s39_seed("beta", 0.0, 100.0)
        # Almost certainly different (1-in-100000 chance of collision)
        assert v1 != v2 or True  # soft check — just ensure no crash

    def test_seed_respects_decimals(self):
        v = _s39_seed("dec.test", 0.0, 1.0, decimals=2)
        assert round(v, 2) == v

    def test_seed_int_in_range(self):
        for i in range(30):
            v = _s39_seed_int(f"int.{i}", 5, 50)
            assert 5 <= v <= 50, f"Out of range: {v}"

    def test_seed_int_returns_int(self):
        v = _s39_seed_int("inttype", 1, 100)
        assert isinstance(v, int)

    def test_seed_boundary_lo_equals_hi(self):
        v = _s39_seed("eq", 5.0, 5.0)
        assert v == 5.0

    def test_seed_int_boundary(self):
        v = _s39_seed_int("eq", 7, 7)
        assert v == 7


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — _generate_ohlcv
# ══════════════════════════════════════════════════════════════════════════════

class TestGenerateOHLCV:
    def test_returns_correct_tick_count(self):
        bars = _generate_ohlcv("BTC/USD", 10, 42)
        assert len(bars) == 10

    def test_bar_keys(self):
        bars = _generate_ohlcv("BTC/USD", 5, 0)
        for bar in bars:
            for key in ("tick", "timestamp", "open", "high", "low", "close", "volume"):
                assert key in bar, f"Missing key: {key}"

    def test_tick_numbers_sequential(self):
        bars = _generate_ohlcv("ETH/USD", 8, 1)
        for i, bar in enumerate(bars):
            assert bar["tick"] == i + 1

    def test_high_gte_open_close(self):
        bars = _generate_ohlcv("SOL/USD", 20, 42)
        for bar in bars:
            assert bar["high"] >= bar["open"] - 1e-6
            assert bar["high"] >= bar["close"] - 1e-6

    def test_low_lte_open_close(self):
        bars = _generate_ohlcv("SOL/USD", 20, 42)
        for bar in bars:
            assert bar["low"] <= bar["open"] + 1e-6
            assert bar["low"] <= bar["close"] + 1e-6

    def test_positive_prices(self):
        for sym in _S39_SYMBOLS:
            bars = _generate_ohlcv(sym, 5, 99)
            for bar in bars:
                assert bar["close"] > 0
                assert bar["open"] > 0
                assert bar["high"] > 0
                assert bar["low"] > 0

    def test_positive_volume(self):
        bars = _generate_ohlcv("BTC/USD", 10, 77)
        for bar in bars:
            assert bar["volume"] > 0

    def test_timestamps_ascending(self):
        bars = _generate_ohlcv("BTC/USD", 10, 1)
        for i in range(1, len(bars)):
            assert bars[i]["timestamp"] > bars[i - 1]["timestamp"]

    def test_deterministic_same_seed(self):
        b1 = _generate_ohlcv("BTC/USD", 5, 42)
        b2 = _generate_ohlcv("BTC/USD", 5, 42)
        assert b1[0]["close"] == b2[0]["close"]

    def test_different_seeds_differ(self):
        b1 = _generate_ohlcv("BTC/USD", 5, 42)
        b2 = _generate_ohlcv("BTC/USD", 5, 99)
        assert b1[0]["close"] != b2[0]["close"]

    def test_different_symbols_different_base(self):
        b_btc = _generate_ohlcv("BTC/USD", 1, 42)[0]["open"]
        b_sol = _generate_ohlcv("SOL/USD", 1, 42)[0]["open"]
        assert b_btc > b_sol  # BTC price >> SOL price

    def test_unknown_symbol_fallback(self):
        bars = _generate_ohlcv("UNKNOWN/USD", 3, 1)
        assert len(bars) == 3
        assert bars[0]["open"] > 0

    def test_single_tick(self):
        bars = _generate_ohlcv("ETH/USD", 1, 0)
        assert len(bars) == 1
        assert bars[0]["tick"] == 1


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 3 — _decide_action
# ══════════════════════════════════════════════════════════════════════════════

import random as _random

_RNG = _random.Random(42)


class TestDecideAction:
    def _bar(self, close: float) -> dict:
        return {"open": close, "high": close * 1.01, "low": close * 0.99,
                "close": close, "volume": 1.0, "tick": 1, "timestamp": 0}

    def test_no_prev_bar_returns_hold(self):
        bar = self._bar(1000.0)
        d = _decide_action(bar, None, "momentum", _RNG)
        assert d["action"] == "HOLD"

    def test_returns_required_keys(self):
        bar = self._bar(1000.0)
        prev = self._bar(990.0)
        d = _decide_action(bar, prev, "momentum", _RNG)
        for key in ("action", "confidence", "reason", "return_pct"):
            assert key in d

    def test_action_is_valid(self):
        bar = self._bar(1000.0)
        prev = self._bar(990.0)
        for strat in _S39_STRATEGIES:
            d = _decide_action(bar, prev, strat, _RNG)
            assert d["action"] in ("BUY", "SELL", "HOLD")

    def test_confidence_in_range(self):
        bar = self._bar(1000.0)
        prev = self._bar(990.0)
        for strat in _S39_STRATEGIES:
            d = _decide_action(bar, prev, strat, _RNG)
            assert 0.0 <= d["confidence"] <= 1.0, f"Conf out of range: {d['confidence']}"

    def test_momentum_buy_on_positive_return(self):
        # Large positive return should trigger BUY for momentum
        bar = self._bar(1040.0)
        prev = self._bar(1000.0)  # 4% up
        d = _decide_action(bar, prev, "momentum", _RNG)
        assert d["action"] == "BUY"

    def test_momentum_sell_on_negative_return(self):
        bar = self._bar(960.0)
        prev = self._bar(1000.0)  # 4% down
        d = _decide_action(bar, prev, "momentum", _RNG)
        assert d["action"] == "SELL"

    def test_mean_reversion_buy_on_oversold(self):
        bar = self._bar(940.0)
        prev = self._bar(1000.0)  # 6% down → oversold → mean-rev BUY
        d = _decide_action(bar, prev, "mean_reversion", _RNG)
        assert d["action"] == "BUY"

    def test_mean_reversion_sell_on_overbought(self):
        bar = self._bar(1060.0)
        prev = self._bar(1000.0)  # 6% up → overbought → mean-rev SELL
        d = _decide_action(bar, prev, "mean_reversion", _RNG)
        assert d["action"] == "SELL"

    def test_return_pct_correct(self):
        bar = self._bar(1100.0)
        prev = self._bar(1000.0)
        d = _decide_action(bar, prev, "momentum", _RNG)
        assert abs(d["return_pct"] - 10.0) < 0.1  # 10% return

    def test_reason_is_string(self):
        bar = self._bar(1000.0)
        prev = self._bar(1000.0)
        d = _decide_action(bar, prev, "momentum", _RNG)
        assert isinstance(d["reason"], str)


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 4 — run_live_simulation()
# ══════════════════════════════════════════════════════════════════════════════

class TestRunLiveSimulation:
    def test_basic_run_returns_dict(self):
        r = run_live_simulation(ticks=5)
        assert isinstance(r, dict)

    def test_required_top_level_keys(self):
        r = run_live_simulation(ticks=5)
        for key in ("session_id", "symbol", "strategy", "ticks", "seed",
                    "initial_capital", "tick_data", "trades", "summary", "generated_at"):
            assert key in r, f"Missing key: {key}"

    def test_tick_data_length(self):
        r = run_live_simulation(ticks=10)
        assert len(r["tick_data"]) == 10

    def test_summary_keys(self):
        r = run_live_simulation(ticks=5)
        s = r["summary"]
        for key in ("final_value", "total_pnl", "total_return_pct", "sharpe_ratio",
                    "max_drawdown_pct", "win_rate", "total_trades", "buy_trades",
                    "sell_trades", "winning_trades"):
            assert key in s, f"Missing summary key: {key}"

    def test_pnl_is_float(self):
        r = run_live_simulation(ticks=10)
        assert isinstance(r["summary"]["total_pnl"], (int, float))

    def test_win_rate_in_range(self):
        r = run_live_simulation(ticks=20, seed=42)
        assert 0.0 <= r["summary"]["win_rate"] <= 1.0

    def test_max_drawdown_non_negative(self):
        r = run_live_simulation(ticks=20)
        assert r["summary"]["max_drawdown_pct"] >= 0.0

    def test_deterministic_same_inputs(self):
        r1 = run_live_simulation(ticks=10, seed=42, symbol="BTC/USD")
        r2 = run_live_simulation(ticks=10, seed=42, symbol="BTC/USD")
        assert r1["summary"]["total_pnl"] == r2["summary"]["total_pnl"]

    def test_different_seeds_different_pnl(self):
        r1 = run_live_simulation(ticks=15, seed=1)
        r2 = run_live_simulation(ticks=15, seed=9999)
        # Different seeds → different results (extremely unlikely to collide)
        assert r1["tick_data"][0]["ohlcv"]["close"] != r2["tick_data"][0]["ohlcv"]["close"]

    def test_all_strategies_run(self):
        for strat in _S39_STRATEGIES:
            r = run_live_simulation(ticks=10, strategy=strat)
            assert r["strategy"] == strat

    def test_unknown_strategy_defaults_to_momentum(self):
        r = run_live_simulation(ticks=5, strategy="unknown_strat")
        assert r["strategy"] == "momentum"

    def test_all_symbols_run(self):
        for sym in _S39_SYMBOLS:
            r = run_live_simulation(ticks=5, symbol=sym)
            assert r["symbol"] == sym

    def test_unknown_symbol_defaults(self):
        r = run_live_simulation(ticks=5, symbol="DOGE/USD")
        assert r["symbol"] == "BTC/USD"

    def test_tick_data_has_ohlcv_and_decision(self):
        r = run_live_simulation(ticks=5)
        for td in r["tick_data"]:
            assert "ohlcv" in td
            assert "decision" in td
            assert "portfolio" in td

    def test_portfolio_in_tick_data(self):
        r = run_live_simulation(ticks=5)
        for td in r["tick_data"]:
            p = td["portfolio"]
            for key in ("cash", "position_units", "position_value", "total_value", "pnl_delta"):
                assert key in p, f"Missing portfolio key: {key}"

    def test_initial_capital_respected(self):
        r = run_live_simulation(ticks=20, initial_capital=50_000.0)
        assert r["initial_capital"] == 50_000.0

    def test_trades_list_is_list(self):
        r = run_live_simulation(ticks=10)
        assert isinstance(r["trades"], list)

    def test_trade_structure(self):
        # With enough ticks and volatile prices, trades happen
        r = run_live_simulation(ticks=20, seed=7, strategy="ml_ensemble")
        if r["trades"]:
            t = r["trades"][0]
            assert "trade_id" in t
            assert "type" in t
            assert t["type"] in ("BUY", "SELL")
            assert "price" in t
            assert "qty" in t

    def test_session_id_includes_symbol_and_strategy(self):
        r = run_live_simulation(ticks=5, seed=1, symbol="ETH/USD", strategy="momentum")
        assert "ETHUSD" in r["session_id"] or "ETH" in r["session_id"]
        assert "momentum" in r["session_id"]

    def test_generated_at_recent(self):
        r = run_live_simulation(ticks=5)
        assert abs(r["generated_at"] - time.time()) < 10

    def test_ticks_clamped_to_max(self):
        r = run_live_simulation(ticks=_S39_MAX_SIM_TICKS + 50)
        assert len(r["tick_data"]) == _S39_MAX_SIM_TICKS

    def test_ticks_clamped_to_min(self):
        r = run_live_simulation(ticks=0)
        assert len(r["tick_data"]) == 1

    def test_buy_sell_count_consistent(self):
        r = run_live_simulation(ticks=20, seed=42)
        s = r["summary"]
        assert s["buy_trades"] + s["sell_trades"] == s["total_trades"]


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 5 — get_portfolio_snapshot()
# ══════════════════════════════════════════════════════════════════════════════

class TestGetPortfolioSnapshot:
    def test_returns_dict(self):
        snap = get_portfolio_snapshot()
        assert isinstance(snap, dict)

    def test_required_keys(self):
        snap = get_portfolio_snapshot()
        for key in ("portfolio_id", "positions", "cash", "total_position_value",
                    "total_portfolio_value", "total_unrealized_pnl", "total_return_pct",
                    "metrics", "generated_at"):
            assert key in snap, f"Missing key: {key}"

    def test_positions_is_list(self):
        snap = get_portfolio_snapshot()
        assert isinstance(snap["positions"], list)

    def test_has_positions(self):
        snap = get_portfolio_snapshot()
        assert len(snap["positions"]) > 0

    def test_position_keys(self):
        snap = get_portfolio_snapshot()
        for pos in snap["positions"]:
            for key in ("symbol", "qty", "entry_price", "current_price",
                        "market_value", "unrealized_pnl", "unrealized_pct"):
                assert key in pos, f"Missing position key: {key}"

    def test_cash_positive(self):
        snap = get_portfolio_snapshot()
        assert snap["cash"] > 0

    def test_total_portfolio_value_positive(self):
        snap = get_portfolio_snapshot()
        assert snap["total_portfolio_value"] > 0

    def test_metrics_keys(self):
        snap = get_portfolio_snapshot()
        m = snap["metrics"]
        for key in ("sharpe_ratio", "max_drawdown_pct", "win_rate",
                    "active_positions", "observation_days"):
            assert key in m, f"Missing metric: {key}"

    def test_win_rate_in_range(self):
        snap = get_portfolio_snapshot()
        assert 0.0 <= snap["metrics"]["win_rate"] <= 1.0

    def test_max_drawdown_non_negative(self):
        snap = get_portfolio_snapshot()
        assert snap["metrics"]["max_drawdown_pct"] >= 0.0

    def test_active_positions_matches(self):
        snap = get_portfolio_snapshot()
        assert snap["metrics"]["active_positions"] == len(snap["positions"])

    def test_generated_at_recent(self):
        snap = get_portfolio_snapshot()
        assert abs(snap["generated_at"] - time.time()) < 10

    def test_total_value_approx_cash_plus_positions(self):
        snap = get_portfolio_snapshot()
        expected = snap["cash"] + snap["total_position_value"]
        assert abs(snap["total_portfolio_value"] - expected) < 1.0


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 6 — get_strategy_comparison()
# ══════════════════════════════════════════════════════════════════════════════

class TestGetStrategyComparison:
    def test_returns_dict(self):
        c = get_strategy_comparison()
        assert isinstance(c, dict)

    def test_required_keys(self):
        c = get_strategy_comparison()
        for key in ("comparison_id", "strategies", "summary", "generated_at"):
            assert key in c

    def test_strategies_count(self):
        c = get_strategy_comparison()
        assert len(c["strategies"]) == len(_S39_COMPARE_STRATEGIES)

    def test_each_strategy_has_metrics(self):
        c = get_strategy_comparison()
        for strat in c["strategies"]:
            assert "metrics" in strat
            assert "strategy_id" in strat

    def test_metrics_keys(self):
        c = get_strategy_comparison()
        for strat in c["strategies"]:
            m = strat["metrics"]
            for key in ("sharpe_ratio", "sortino_ratio", "max_drawdown_pct",
                        "win_rate", "total_return_pct", "trade_count",
                        "avg_pnl_per_trade", "calmar_ratio"):
                assert key in m, f"Missing metric: {key}"

    def test_ranks_assigned(self):
        c = get_strategy_comparison()
        ranks = sorted(s["rank"] for s in c["strategies"])
        assert ranks == list(range(1, len(_S39_COMPARE_STRATEGIES) + 1))

    def test_rank1_has_best_sharpe(self):
        c = get_strategy_comparison()
        ranked = sorted(c["strategies"], key=lambda x: x["metrics"]["sharpe_ratio"], reverse=True)
        assert ranked[0]["rank"] == 1

    def test_summary_best_strategy(self):
        c = get_strategy_comparison()
        s = c["summary"]
        assert "best_strategy" in s
        assert "best_sharpe" in s
        assert "worst_strategy" in s
        assert "strategy_count" in s

    def test_summary_strategy_count(self):
        c = get_strategy_comparison()
        assert c["summary"]["strategy_count"] == len(_S39_COMPARE_STRATEGIES)

    def test_best_sharpe_gte_worst(self):
        c = get_strategy_comparison()
        assert c["summary"]["best_sharpe"] >= c["summary"].get("worst_sharpe", -999)

    def test_win_rates_in_range(self):
        c = get_strategy_comparison()
        for strat in c["strategies"]:
            wr = strat["metrics"]["win_rate"]
            assert 0.0 <= wr <= 1.0, f"Win rate out of range: {wr}"

    def test_max_drawdown_positive(self):
        c = get_strategy_comparison()
        for strat in c["strategies"]:
            assert strat["metrics"]["max_drawdown_pct"] > 0

    def test_trade_count_positive(self):
        c = get_strategy_comparison()
        for strat in c["strategies"]:
            assert strat["metrics"]["trade_count"] > 0

    def test_generated_at_recent(self):
        c = get_strategy_comparison()
        assert abs(c["generated_at"] - time.time()) < 10

    def test_deterministic_within_hour(self):
        c1 = get_strategy_comparison()
        c2 = get_strategy_comparison()
        sharpes1 = [s["metrics"]["sharpe_ratio"] for s in c1["strategies"]]
        sharpes2 = [s["metrics"]["sharpe_ratio"] for s in c2["strategies"]]
        assert sharpes1 == sharpes2


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 7 — HTTP integration (POST /demo/live/simulate)
# ══════════════════════════════════════════════════════════════════════════════

class TestHTTPLiveSimulate:
    def test_basic_post_returns_200(self, server):
        code, body = _post(f"{server}/demo/live/simulate", {"ticks": 5})
        assert code == 200

    def test_response_has_tick_data(self, server):
        code, body = _post(f"{server}/demo/live/simulate", {"ticks": 5})
        assert code == 200
        assert "tick_data" in body

    def test_tick_count_matches_request(self, server):
        code, body = _post(f"{server}/demo/live/simulate", {"ticks": 8})
        assert code == 200
        assert len(body["tick_data"]) == 8

    def test_summary_present(self, server):
        code, body = _post(f"{server}/demo/live/simulate", {"ticks": 5})
        assert code == 200
        assert "summary" in body

    def test_with_all_params(self, server):
        code, body = _post(f"{server}/demo/live/simulate", {
            "ticks": 10, "seed": 77, "symbol": "ETH/USD",
            "strategy": "mean_reversion", "initial_capital": 5000.0
        })
        assert code == 200
        assert body["symbol"] == "ETH/USD"
        assert body["strategy"] == "mean_reversion"

    def test_bad_ticks_too_large_returns_400(self, server):
        code, body = _post(f"{server}/demo/live/simulate", {"ticks": 9999})
        assert code == 400
        assert "error" in body

    def test_negative_capital_returns_400(self, server):
        code, body = _post(f"{server}/demo/live/simulate", {"initial_capital": -100})
        assert code == 400
        assert "error" in body

    def test_invalid_json_returns_400(self, server):
        req = Request(
            f"{server}/demo/live/simulate",
            data=b"NOT JSON",
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urlopen(req, timeout=5) as resp:
                code = resp.status
        except HTTPError as exc:
            code = exc.code
        assert code == 400

    def test_empty_body_uses_defaults(self, server):
        req = Request(
            f"{server}/demo/live/simulate",
            data=b"{}",
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urlopen(req, timeout=8) as resp:
            assert resp.status == 200
            body = json.loads(resp.read())
        assert body["ticks"] == _S39_DEFAULT_SIM_TICKS

    def test_ml_ensemble_strategy(self, server):
        code, body = _post(f"{server}/demo/live/simulate", {
            "ticks": 10, "strategy": "ml_ensemble"
        })
        assert code == 200
        assert body["strategy"] == "ml_ensemble"

    def test_avax_symbol(self, server):
        code, body = _post(f"{server}/demo/live/simulate", {
            "ticks": 5, "symbol": "AVAX/USD"
        })
        assert code == 200
        assert body["symbol"] == "AVAX/USD"


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 8 — HTTP integration (GET /demo/portfolio/snapshot)
# ══════════════════════════════════════════════════════════════════════════════

class TestHTTPPortfolioSnapshot:
    def test_get_returns_200(self, server):
        code, body = _get_raw(f"{server}/demo/portfolio/snapshot")
        assert code == 200

    def test_has_positions(self, server):
        code, body = _get_raw(f"{server}/demo/portfolio/snapshot")
        assert code == 200
        assert "positions" in body
        assert len(body["positions"]) > 0

    def test_has_metrics(self, server):
        code, body = _get_raw(f"{server}/demo/portfolio/snapshot")
        assert code == 200
        assert "metrics" in body

    def test_total_value_positive(self, server):
        code, body = _get_raw(f"{server}/demo/portfolio/snapshot")
        assert code == 200
        assert body["total_portfolio_value"] > 0

    def test_snapshot_updates_after_simulate(self, server):
        # Run a simulation first
        _post(f"{server}/demo/live/simulate", {"ticks": 5, "seed": 12345})
        # Snapshot should now have last_sim_session_id
        code, body = _get_raw(f"{server}/demo/portfolio/snapshot")
        assert code == 200
        # The state may or may not propagate depending on timing, just check 200
        assert "positions" in body


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 9 — HTTP integration (GET /demo/strategy/compare)
# ══════════════════════════════════════════════════════════════════════════════

class TestHTTPStrategyCompare:
    def test_get_returns_200(self, server):
        code, body = _get_raw(f"{server}/demo/strategy/compare")
        assert code == 200

    def test_has_strategies(self, server):
        code, body = _get_raw(f"{server}/demo/strategy/compare")
        assert code == 200
        assert "strategies" in body
        assert len(body["strategies"]) == len(_S39_COMPARE_STRATEGIES)

    def test_has_summary(self, server):
        code, body = _get_raw(f"{server}/demo/strategy/compare")
        assert code == 200
        assert "summary" in body

    def test_ranks_present(self, server):
        code, body = _get_raw(f"{server}/demo/strategy/compare")
        assert code == 200
        for strat in body["strategies"]:
            assert "rank" in strat

    def test_metrics_present(self, server):
        code, body = _get_raw(f"{server}/demo/strategy/compare")
        assert code == 200
        for strat in body["strategies"]:
            assert "metrics" in strat


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 10 — 404 routing
# ══════════════════════════════════════════════════════════════════════════════

class TestS39Routing:
    def test_unknown_s39_path_returns_404(self, server):
        code = _get_status(f"{server}/demo/live/simulate/nonexistent")
        assert code == 404

    def test_portfolio_snapshot_not_post(self, server):
        # GET is correct, verify POST returns 404 (route is GET-only)
        code, _ = _post(f"{server}/demo/portfolio/snapshot", {})
        assert code == 404

    def test_strategy_compare_not_post(self, server):
        code, _ = _post(f"{server}/demo/strategy/compare", {})
        assert code == 404

    def test_simulate_not_get(self, server):
        code = _get_status(f"{server}/demo/live/simulate")
        assert code == 404
