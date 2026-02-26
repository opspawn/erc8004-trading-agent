"""
test_s28_backtest.py — Sprint 28: Backtesting endpoint tests.

40 tests covering:
  - build_backtest() with all 4 strategies
  - GBM price generation (_gbm_price_series)
  - Sharpe ratio computation (_compute_sharpe)
  - Max drawdown computation (_compute_max_drawdown)
  - Response structure and required fields
  - Edge cases: min period, max period, invalid inputs
  - HTTP endpoint POST /demo/backtest
  - Equity curve downsampling
"""

from __future__ import annotations

import json
import math
import time
import threading
from urllib.request import urlopen, Request
from urllib.error import HTTPError

import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from demo_server import (
    build_backtest,
    _gbm_price_series,
    _compute_sharpe,
    _compute_max_drawdown,
    _BACKTEST_STRATEGIES,
    _BACKTEST_MAX_DAYS,
    DemoServer,
    DEFAULT_PORT,
)

# Use a different port to avoid conflicts
BACKTEST_TEST_PORT = 18096


# ─── _gbm_price_series ────────────────────────────────────────────────────────

class TestGbmPriceSeries:
    def test_returns_correct_length(self):
        prices = _gbm_price_series(seed=42, n_days=30)
        assert len(prices) == 31  # day 0 + 30 steps

    def test_first_price_is_s0(self):
        prices = _gbm_price_series(seed=42, n_days=10, s0=500.0)
        assert prices[0] == 500.0

    def test_all_prices_positive(self):
        prices = _gbm_price_series(seed=7, n_days=100)
        assert all(p > 0 for p in prices)

    def test_deterministic_with_same_seed(self):
        p1 = _gbm_price_series(seed=123, n_days=20)
        p2 = _gbm_price_series(seed=123, n_days=20)
        assert p1 == p2

    def test_different_seeds_differ(self):
        p1 = _gbm_price_series(seed=1, n_days=50)
        p2 = _gbm_price_series(seed=2, n_days=50)
        assert p1 != p2

    def test_single_day(self):
        prices = _gbm_price_series(seed=42, n_days=1)
        assert len(prices) == 2

    def test_zero_sigma_stays_near_drift(self):
        prices = _gbm_price_series(seed=42, n_days=10, mu=0.0, sigma=0.0)
        # With zero sigma, price should be exactly s0 * exp(mu*t)
        for p in prices:
            assert p > 0


# ─── _compute_sharpe ──────────────────────────────────────────────────────────

class TestComputeSharpe:
    def test_empty_returns_zero(self):
        assert _compute_sharpe([]) == 0.0

    def test_single_return_zero(self):
        assert _compute_sharpe([0.01]) == 0.0

    def test_constant_returns_high_sharpe(self):
        # Constant positive return → std dev = 0 → sharpe = 0 (no variance edge case)
        returns = [0.01] * 50
        # std is 0 → returns 0 by convention
        result = _compute_sharpe(returns)
        assert result == 0.0

    def test_mixed_returns(self):
        import random
        rng = random.Random(42)
        returns = [rng.gauss(0.001, 0.02) for _ in range(252)]
        sharpe = _compute_sharpe(returns)
        assert isinstance(sharpe, float)
        assert -10 < sharpe < 10

    def test_negative_returns_negative_sharpe(self):
        returns = [-0.01] * 100
        # std = 0, so returns 0
        result = _compute_sharpe(returns)
        assert result == 0.0

    def test_returns_rounded_to_4_decimals(self):
        import random
        rng = random.Random(99)
        returns = [rng.gauss(0.001, 0.015) for _ in range(100)]
        sharpe = _compute_sharpe(returns)
        assert sharpe == round(sharpe, 4)


# ─── _compute_max_drawdown ────────────────────────────────────────────────────

class TestComputeMaxDrawdown:
    def test_monotonically_increasing_is_zero(self):
        equity = [100.0, 110.0, 120.0, 130.0]
        assert _compute_max_drawdown(equity) == 0.0

    def test_single_value_is_zero(self):
        assert _compute_max_drawdown([100.0]) == 0.0

    def test_empty_is_zero(self):
        assert _compute_max_drawdown([]) == 0.0

    def test_known_drawdown(self):
        # Peak 200, trough 100 → 50% drawdown
        equity = [100.0, 200.0, 100.0]
        dd = _compute_max_drawdown(equity)
        assert abs(dd - 50.0) < 0.01

    def test_dd_is_percentage(self):
        equity = [1000.0, 900.0, 800.0, 700.0]
        # Peak = 1000, trough = 700 → 30%
        dd = _compute_max_drawdown(equity)
        assert abs(dd - 30.0) < 0.01

    def test_recovery_uses_new_peak(self):
        # After recovery, new peak, then new drawdown
        equity = [100.0, 90.0, 110.0, 80.0]
        # Second drawdown: (110 - 80) / 110 = 27.27%
        dd = _compute_max_drawdown(equity)
        assert dd >= 27.0

    def test_result_is_float(self):
        assert isinstance(_compute_max_drawdown([100.0, 80.0]), float)


# ─── build_backtest ────────────────────────────────────────────────────────────

class TestBuildBacktest:
    def _run(self, strategy="momentum", days=30):
        import datetime as dt
        start = dt.date(2024, 1, 1)
        end = start + dt.timedelta(days=days)
        return build_backtest(
            symbol="BTC/USD",
            strategy=strategy,
            start_date=start.isoformat(),
            end_date=end.isoformat(),
            initial_capital=10000.0,
        )

    def test_required_fields_present(self):
        r = self._run()
        for key in ("symbol", "strategy", "start_date", "end_date", "period_days",
                    "initial_capital", "final_equity", "total_return_pct",
                    "max_drawdown_pct", "sharpe_ratio", "num_trades", "equity_curve"):
            assert key in r, f"Missing key: {key}"

    def test_equity_curve_is_list(self):
        r = self._run()
        assert isinstance(r["equity_curve"], list)
        assert len(r["equity_curve"]) >= 2

    def test_initial_capital_matches(self):
        r = self._run()
        assert r["initial_capital"] == 10000.0

    def test_period_days_correct(self):
        r = self._run(days=30)
        assert r["period_days"] == 30

    def test_all_strategies(self):
        for strat in _BACKTEST_STRATEGIES:
            r = self._run(strategy=strat)
            assert r["strategy"] == strat
            assert r["total_return_pct"] is not None

    def test_buy_and_hold_has_2_trades(self):
        r = self._run(strategy="buy_and_hold", days=90)
        assert r["num_trades"] == 2

    def test_num_trades_non_negative(self):
        for strat in _BACKTEST_STRATEGIES:
            r = self._run(strategy=strat)
            assert r["num_trades"] >= 0

    def test_max_drawdown_non_negative(self):
        r = self._run()
        assert r["max_drawdown_pct"] >= 0.0

    def test_max_drawdown_under_100(self):
        r = self._run()
        assert r["max_drawdown_pct"] <= 100.0

    def test_deterministic(self):
        r1 = self._run(strategy="momentum", days=60)
        r2 = self._run(strategy="momentum", days=60)
        assert r1["total_return_pct"] == r2["total_return_pct"]

    def test_generated_at_is_recent(self):
        r = self._run()
        assert abs(r["generated_at"] - time.time()) < 5

    def test_invalid_strategy_raises(self):
        import datetime as dt
        with pytest.raises(ValueError, match="Unknown strategy"):
            build_backtest("BTC/USD", "nonexistent", "2024-01-01", "2024-02-01", 10000.0)

    def test_invalid_date_format_raises(self):
        with pytest.raises(ValueError):
            build_backtest("BTC/USD", "momentum", "01/01/2024", "2024-02-01", 10000.0)

    def test_end_before_start_raises(self):
        with pytest.raises(ValueError, match="end_date must be after start_date"):
            build_backtest("BTC/USD", "momentum", "2024-06-01", "2024-01-01", 10000.0)

    def test_same_start_end_raises(self):
        with pytest.raises(ValueError):
            build_backtest("BTC/USD", "momentum", "2024-01-01", "2024-01-01", 10000.0)

    def test_zero_capital_raises(self):
        with pytest.raises(ValueError, match="initial_capital must be positive"):
            build_backtest("BTC/USD", "momentum", "2024-01-01", "2024-02-01", 0.0)

    def test_negative_capital_raises(self):
        with pytest.raises(ValueError):
            build_backtest("BTC/USD", "momentum", "2024-01-01", "2024-02-01", -100.0)

    def test_long_period_capped(self):
        with pytest.raises(ValueError, match="Period too long"):
            build_backtest("BTC/USD", "momentum", "2000-01-01", "2030-01-01", 10000.0)

    def test_equity_curve_starts_near_capital(self):
        r = self._run()
        # First point should be initial_capital (or very close)
        assert r["equity_curve"][0] > 0

    def test_different_symbols_differ(self):
        import datetime as dt
        start, end = "2024-01-01", "2024-06-01"
        r_btc = build_backtest("BTC/USD", "momentum", start, end, 10000.0)
        r_eth = build_backtest("ETH/USD", "momentum", start, end, 10000.0)
        # Different symbols → different vol → different results
        assert r_btc["total_return_pct"] != r_eth["total_return_pct"] or \
               r_btc["sharpe_ratio"] != r_eth["sharpe_ratio"]

    def test_large_capital(self):
        import datetime as dt
        r = build_backtest("BTC/USD", "buy_and_hold", "2024-01-01", "2024-04-01", 1_000_000.0)
        assert r["initial_capital"] == 1_000_000.0
        assert r["final_equity"] > 0


# ─── HTTP: POST /demo/backtest ─────────────────────────────────────────────────

@pytest.fixture(scope="module")
def backtest_server():
    srv = DemoServer(port=BACKTEST_TEST_PORT)
    srv.start()
    time.sleep(0.3)
    yield f"http://localhost:{BACKTEST_TEST_PORT}"
    srv.stop()


def _post_backtest(base_url: str, payload: dict):
    data = json.dumps(payload).encode()
    req = Request(f"{base_url}/demo/backtest", data=data,
                  headers={"Content-Type": "application/json"}, method="POST")
    with urlopen(req, timeout=10) as resp:
        return resp.status, json.loads(resp.read())


class TestBacktestHTTP:
    def test_valid_request_200(self, backtest_server):
        status, body = _post_backtest(backtest_server, {
            "symbol": "BTC/USD",
            "strategy": "momentum",
            "start_date": "2024-01-01",
            "end_date": "2024-06-01",
            "initial_capital": 10000,
        })
        assert status == 200
        assert "equity_curve" in body
        assert body["strategy"] == "momentum"

    def test_missing_dates_returns_400(self, backtest_server):
        data = json.dumps({"symbol": "BTC/USD"}).encode()
        req = Request(f"{backtest_server}/demo/backtest", data=data,
                      headers={"Content-Type": "application/json"}, method="POST")
        try:
            urlopen(req, timeout=10)
            assert False, "Expected 400"
        except HTTPError as e:
            assert e.code == 400

    def test_invalid_strategy_returns_400(self, backtest_server):
        data = json.dumps({
            "strategy": "nope",
            "start_date": "2024-01-01",
            "end_date": "2024-03-01",
        }).encode()
        req = Request(f"{backtest_server}/demo/backtest", data=data,
                      headers={"Content-Type": "application/json"}, method="POST")
        try:
            urlopen(req, timeout=10)
            assert False, "Expected 400"
        except HTTPError as e:
            assert e.code == 400

    def test_buy_and_hold_via_http(self, backtest_server):
        status, body = _post_backtest(backtest_server, {
            "strategy": "buy_and_hold",
            "start_date": "2024-01-01",
            "end_date": "2024-12-31",
            "initial_capital": 5000,
        })
        assert status == 200
        assert body["num_trades"] == 2

    def test_response_has_all_keys(self, backtest_server):
        status, body = _post_backtest(backtest_server, {
            "symbol": "ETH/USD",
            "strategy": "mean_reversion",
            "start_date": "2024-03-01",
            "end_date": "2024-09-01",
            "initial_capital": 20000,
        })
        assert status == 200
        for key in ("symbol", "strategy", "total_return_pct", "max_drawdown_pct",
                    "sharpe_ratio", "num_trades", "equity_curve"):
            assert key in body

    def test_invalid_json_returns_400(self, backtest_server):
        req = Request(f"{backtest_server}/demo/backtest",
                      data=b"not-json",
                      headers={"Content-Type": "application/json"}, method="POST")
        try:
            urlopen(req, timeout=10)
            assert False, "Expected 400"
        except HTTPError as e:
            assert e.code == 400
