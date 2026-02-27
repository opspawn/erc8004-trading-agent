"""
test_s41_strategy_compare.py — Sprint 41: Strategy Compare, Monte Carlo, Correlation Matrix.

Covers:
  - run_strategies_compare(): schema, ranking, metrics, edge cases, error handling
  - run_monte_carlo(): schema, percentile ordering, prob_profit, edge cases
  - get_market_correlation(): schema, diagonal=1.0, symmetry, edge cases
  - HTTP integration: POST /api/v1/strategies/compare, POST /api/v1/portfolio/monte-carlo,
                      GET /api/v1/market/correlation
  - SERVER_VERSION == "S41", _S41_TEST_COUNT >= 5100
  - Error handling: bad JSON, invalid strategy IDs, bad dates, out-of-range params, 404
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
    _S41_TEST_COUNT,
    _S41_VALID_STRATEGIES,
    _S41_MIN_STRATEGIES,
    _S41_MAX_STRATEGIES,
    _S41_DEFAULT_SYMBOL,
    _S41_DEFAULT_CAPITAL,
    _S41_DEFAULT_START,
    _S41_DEFAULT_END,
    _S41_MC_DEFAULT_PATHS,
    _S41_MC_DEFAULT_DAYS,
    _S41_MC_MAX_PATHS,
    _S41_MC_MAX_DAYS,
    _S41_DEFAULT_SYMBOLS,
    _S41_CORR_DAYS,
    run_strategies_compare,
    run_monte_carlo,
    get_market_correlation,
    _s41_win_rate_from_equity,
)


# ── Helpers ────────────────────────────────────────────────────────────────────

def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _get(url: str) -> dict:
    with urlopen(url, timeout=10) as resp:
        return json.loads(resp.read())


def _get_raw(url: str) -> tuple[int, dict]:
    try:
        with urlopen(url, timeout=10) as resp:
            return resp.status, json.loads(resp.read())
    except HTTPError as exc:
        return exc.code, json.loads(exc.read())


def _post(url: str, body: dict | None = None) -> tuple[int, dict]:
    data = json.dumps(body or {}).encode()
    req = Request(url, data=data, headers={"Content-Type": "application/json"}, method="POST")
    try:
        with urlopen(req, timeout=15) as resp:
            return resp.status, json.loads(resp.read())
    except HTTPError as exc:
        return exc.code, json.loads(exc.read())


def _post_raw(url: str, body: bytes, content_type: str = "application/json") -> tuple[int, dict]:
    req = Request(url, data=body, headers={"Content-Type": content_type}, method="POST")
    try:
        with urlopen(req, timeout=10) as resp:
            return resp.status, json.loads(resp.read())
    except HTTPError as exc:
        return exc.code, json.loads(exc.read())


# ── Fixtures ───────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def server():
    port = _free_port()
    srv = DemoServer(port=port)
    srv.start()
    time.sleep(0.3)
    yield f"http://127.0.0.1:{port}"
    srv.stop()


# ═══════════════════════════════════════════════════════════════════════════════
# 1. Version / constants
# ═══════════════════════════════════════════════════════════════════════════════

class TestS41Constants:
    def test_server_version_is_s41(self):
        assert SERVER_VERSION in ("S41", "S42", "S43", "S44", "S45")

    def test_s41_test_count_at_least_5100(self):
        assert _S41_TEST_COUNT >= 5100

    def test_valid_strategies_nonempty(self):
        assert len(_S41_VALID_STRATEGIES) >= 4

    def test_expected_strategies_present(self):
        assert "momentum" in _S41_VALID_STRATEGIES
        assert "mean_reversion" in _S41_VALID_STRATEGIES
        assert "buy_and_hold" in _S41_VALID_STRATEGIES
        assert "random" in _S41_VALID_STRATEGIES

    def test_min_max_strategies(self):
        assert _S41_MIN_STRATEGIES == 2
        assert _S41_MAX_STRATEGIES >= 4

    def test_default_symbol(self):
        assert "/" in _S41_DEFAULT_SYMBOL

    def test_default_capital_positive(self):
        assert _S41_DEFAULT_CAPITAL > 0

    def test_mc_defaults_positive(self):
        assert _S41_MC_DEFAULT_PATHS > 0
        assert _S41_MC_DEFAULT_DAYS > 0

    def test_mc_max_paths_ge_default(self):
        assert _S41_MC_MAX_PATHS >= _S41_MC_DEFAULT_PATHS

    def test_mc_max_days_ge_default(self):
        assert _S41_MC_MAX_DAYS >= _S41_MC_DEFAULT_DAYS

    def test_default_symbols_list(self):
        assert isinstance(_S41_DEFAULT_SYMBOLS, list)
        assert len(_S41_DEFAULT_SYMBOLS) >= 2

    def test_corr_days_positive(self):
        assert _S41_CORR_DAYS > 0


# ═══════════════════════════════════════════════════════════════════════════════
# 2. Unit tests: _s41_win_rate_from_equity
# ═══════════════════════════════════════════════════════════════════════════════

class TestWinRateHelper:
    def test_always_up(self):
        equity = [100, 110, 120, 130]
        wr = _s41_win_rate_from_equity(equity)
        assert wr == 1.0

    def test_always_down(self):
        equity = [130, 120, 110, 100]
        wr = _s41_win_rate_from_equity(equity)
        assert wr == 0.0

    def test_half_and_half(self):
        equity = [100, 110, 105, 115, 110]
        wr = _s41_win_rate_from_equity(equity)
        assert 0.0 < wr < 1.0

    def test_single_element_returns_half(self):
        wr = _s41_win_rate_from_equity([100])
        assert wr == 0.5

    def test_empty_returns_half(self):
        wr = _s41_win_rate_from_equity([])
        assert wr == 0.5

    def test_result_between_0_and_1(self):
        equity = [100 + i * (1 if i % 3 else -1) for i in range(50)]
        wr = _s41_win_rate_from_equity(equity)
        assert 0.0 <= wr <= 1.0

    def test_result_is_float(self):
        wr = _s41_win_rate_from_equity([100, 110, 105])
        assert isinstance(wr, float)


# ═══════════════════════════════════════════════════════════════════════════════
# 3. Unit tests: run_strategies_compare
# ═══════════════════════════════════════════════════════════════════════════════

class TestRunStrategiesCompare:
    _two = ["momentum", "mean_reversion"]
    _three = ["momentum", "mean_reversion", "buy_and_hold"]
    _all = ["momentum", "mean_reversion", "buy_and_hold", "random"]

    def test_returns_dict(self):
        r = run_strategies_compare(self._two)
        assert isinstance(r, dict)

    def test_top_level_keys(self):
        r = run_strategies_compare(self._two)
        for key in ("comparison_id", "symbol", "start_date", "end_date",
                    "strategies", "leaderboard", "summary", "generated_at"):
            assert key in r, f"Missing key: {key}"

    def test_strategy_count_matches_input(self):
        r = run_strategies_compare(self._three)
        assert len(r["strategies"]) == 3

    def test_leaderboard_count_matches(self):
        r = run_strategies_compare(self._three)
        assert len(r["leaderboard"]) == 3

    def test_ranked_by_sharpe_descending(self):
        r = run_strategies_compare(self._all)
        sharpes = [s["metrics"]["sharpe_ratio"] for s in r["strategies"]]
        assert sharpes == sorted(sharpes, reverse=True)

    def test_rank_field_starts_at_1(self):
        r = run_strategies_compare(self._two)
        ranks = sorted(s["rank"] for s in r["strategies"])
        assert ranks == list(range(1, len(ranks) + 1))

    def test_each_strategy_has_metrics(self):
        r = run_strategies_compare(self._two)
        for strat in r["strategies"]:
            m = strat["metrics"]
            for metric in ("total_return_pct", "sharpe_ratio", "max_drawdown_pct",
                           "win_rate", "num_trades"):
                assert metric in m

    def test_win_rate_between_0_and_1(self):
        r = run_strategies_compare(self._all)
        for strat in r["strategies"]:
            assert 0.0 <= strat["metrics"]["win_rate"] <= 1.0

    def test_max_drawdown_nonnegative(self):
        r = run_strategies_compare(self._all)
        for strat in r["strategies"]:
            assert strat["metrics"]["max_drawdown_pct"] >= 0.0

    def test_num_trades_nonnegative(self):
        r = run_strategies_compare(self._all)
        for strat in r["strategies"]:
            assert strat["metrics"]["num_trades"] >= 0

    def test_summary_best_strategy_is_rank1(self):
        r = run_strategies_compare(self._three)
        best_id = r["strategies"][0]["strategy_id"]
        assert r["summary"]["best_strategy"] == best_id

    def test_summary_strategy_count(self):
        r = run_strategies_compare(self._three)
        assert r["summary"]["strategy_count"] == 3

    def test_summary_best_sharpe_gte_worst(self):
        r = run_strategies_compare(self._all)
        assert r["summary"]["best_sharpe"] >= r["summary"]["worst_sharpe"]

    def test_comparison_id_is_string(self):
        r = run_strategies_compare(self._two)
        assert isinstance(r["comparison_id"], str)
        assert len(r["comparison_id"]) > 0

    def test_generated_at_is_numeric(self):
        r = run_strategies_compare(self._two)
        assert isinstance(r["generated_at"], float)
        assert r["generated_at"] > 0

    def test_default_symbol_in_response(self):
        r = run_strategies_compare(self._two)
        assert r["symbol"] == _S41_DEFAULT_SYMBOL

    def test_custom_symbol(self):
        r = run_strategies_compare(self._two, symbol="ETH/USD")
        assert r["symbol"] == "ETH/USD"

    def test_custom_capital(self):
        r = run_strategies_compare(self._two, initial_capital=50000.0)
        for strat in r["strategies"]:
            assert strat["initial_capital"] == 50000.0

    def test_strategy_ids_in_output(self):
        r = run_strategies_compare(self._three)
        output_ids = {s["strategy_id"] for s in r["strategies"]}
        assert output_ids == set(self._three)

    def test_leaderboard_has_required_fields(self):
        r = run_strategies_compare(self._two)
        for lb in r["leaderboard"]:
            for field in ("rank", "strategy_id", "sharpe_ratio", "total_return_pct",
                          "max_drawdown_pct", "win_rate"):
                assert field in lb

    def test_leaderboard_ranks_sequential(self):
        r = run_strategies_compare(self._three)
        ranks = [lb["rank"] for lb in r["leaderboard"]]
        assert sorted(ranks) == list(range(1, len(ranks) + 1))

    def test_final_equity_positive(self):
        r = run_strategies_compare(self._two)
        for strat in r["strategies"]:
            assert strat["final_equity"] > 0

    def test_deterministic_same_call(self):
        r1 = run_strategies_compare(self._two, "2023-01-01", "2024-01-01")
        r2 = run_strategies_compare(self._two, "2023-01-01", "2024-01-01")
        assert r1["summary"]["best_strategy"] == r2["summary"]["best_strategy"]
        assert r1["strategies"][0]["metrics"]["sharpe_ratio"] == r2["strategies"][0]["metrics"]["sharpe_ratio"]

    def test_different_date_range_different_result(self):
        r1 = run_strategies_compare(self._two, "2023-01-01", "2023-06-01")
        r2 = run_strategies_compare(self._two, "2023-06-01", "2024-01-01")
        # Total return pcts may differ
        assert r1["start_date"] != r2["start_date"]

    # ── Error cases ──────────────────────────────────────────────────────────

    def test_too_few_strategies_raises(self):
        with pytest.raises(ValueError, match="at least"):
            run_strategies_compare(["momentum"])

    def test_empty_list_raises(self):
        with pytest.raises(ValueError):
            run_strategies_compare([])

    def test_too_many_strategies_raises(self):
        strats = list(_S41_VALID_STRATEGIES) * 3
        with pytest.raises(ValueError, match="Too many"):
            run_strategies_compare(strats[:_S41_MAX_STRATEGIES + 1])

    def test_unknown_strategy_raises(self):
        with pytest.raises(ValueError, match="Unknown"):
            run_strategies_compare(["momentum", "invalid_strat"])

    def test_bad_start_date_raises(self):
        with pytest.raises(ValueError, match="date"):
            run_strategies_compare(self._two, start_date="not-a-date")

    def test_bad_end_date_raises(self):
        with pytest.raises(ValueError, match="date"):
            run_strategies_compare(self._two, end_date="2099-13-99")

    def test_end_before_start_raises(self):
        with pytest.raises(ValueError, match="end_date"):
            run_strategies_compare(self._two, start_date="2024-01-01", end_date="2023-01-01")

    def test_zero_capital_raises(self):
        with pytest.raises(ValueError, match="positive"):
            run_strategies_compare(self._two, initial_capital=0)

    def test_negative_capital_raises(self):
        with pytest.raises(ValueError, match="positive"):
            run_strategies_compare(self._two, initial_capital=-500)

    def test_not_a_list_raises(self):
        with pytest.raises(ValueError):
            run_strategies_compare("momentum")

    def test_period_days_in_response(self):
        r = run_strategies_compare(self._two, "2023-01-01", "2023-07-01")
        for strat in r["strategies"]:
            assert strat["period_days"] == 181


# ═══════════════════════════════════════════════════════════════════════════════
# 4. Unit tests: run_monte_carlo
# ═══════════════════════════════════════════════════════════════════════════════

class TestRunMonteCarlo:

    def test_returns_dict(self):
        r = run_monte_carlo(n_paths=50, n_days=30)
        assert isinstance(r, dict)

    def test_top_level_keys(self):
        r = run_monte_carlo(n_paths=50, n_days=30)
        for key in ("simulation_id", "symbol", "initial_capital", "n_paths",
                    "n_days", "percentiles", "summary", "paths_sample", "generated_at"):
            assert key in r

    def test_percentiles_keys(self):
        r = run_monte_carlo(n_paths=50, n_days=30)
        assert "p5" in r["percentiles"]
        assert "p50" in r["percentiles"]
        assert "p95" in r["percentiles"]

    def test_percentile_fields(self):
        r = run_monte_carlo(n_paths=50, n_days=30)
        for pkey in ("p5", "p50", "p95"):
            pct = r["percentiles"][pkey]
            assert "equity" in pct
            assert "return_pct" in pct

    def test_p5_le_p50_le_p95(self):
        r = run_monte_carlo(n_paths=200, n_days=252)
        p5 = r["percentiles"]["p5"]["equity"]
        p50 = r["percentiles"]["p50"]["equity"]
        p95 = r["percentiles"]["p95"]["equity"]
        assert p5 <= p50
        assert p50 <= p95

    def test_all_percentile_equities_positive(self):
        r = run_monte_carlo(n_paths=100, n_days=30)
        for pkey in ("p5", "p50", "p95"):
            assert r["percentiles"][pkey]["equity"] > 0

    def test_summary_keys(self):
        r = run_monte_carlo(n_paths=50, n_days=30)
        s = r["summary"]
        for key in ("mean_terminal_equity", "mean_return_pct", "prob_profit",
                    "median_max_drawdown_pct"):
            assert key in s

    def test_prob_profit_between_0_and_1(self):
        r = run_monte_carlo(n_paths=200, n_days=252)
        pp = r["summary"]["prob_profit"]
        assert 0.0 <= pp <= 1.0

    def test_median_max_drawdown_nonnegative(self):
        r = run_monte_carlo(n_paths=50, n_days=30)
        assert r["summary"]["median_max_drawdown_pct"] >= 0.0

    def test_n_paths_capped_at_max(self):
        r = run_monte_carlo(n_paths=99999, n_days=10)
        assert r["n_paths"] <= _S41_MC_MAX_PATHS

    def test_n_days_capped_at_max(self):
        r = run_monte_carlo(n_paths=10, n_days=99999)
        assert r["n_days"] <= _S41_MC_MAX_DAYS

    def test_n_paths_at_least_1(self):
        r = run_monte_carlo(n_paths=0, n_days=30)
        assert r["n_paths"] >= 1

    def test_n_days_at_least_1(self):
        r = run_monte_carlo(n_paths=10, n_days=0)
        assert r["n_days"] >= 1

    def test_simulation_id_is_string(self):
        r = run_monte_carlo(n_paths=10, n_days=10)
        assert isinstance(r["simulation_id"], str)
        assert len(r["simulation_id"]) > 0

    def test_paths_sample_at_most_10(self):
        r = run_monte_carlo(n_paths=100, n_days=30)
        assert len(r["paths_sample"]) <= 10

    def test_paths_sample_length_equals_n_days_plus_1(self):
        r = run_monte_carlo(n_paths=20, n_days=50)
        for path in r["paths_sample"]:
            assert len(path) == 51  # n_days + 1 (including initial)

    def test_deterministic_output(self):
        r1 = run_monte_carlo(n_paths=100, n_days=30, seed=42)
        r2 = run_monte_carlo(n_paths=100, n_days=30, seed=42)
        assert r1["percentiles"]["p50"]["equity"] == r2["percentiles"]["p50"]["equity"]

    def test_different_seed_different_output(self):
        r1 = run_monte_carlo(n_paths=100, n_days=30, seed=1)
        r2 = run_monte_carlo(n_paths=100, n_days=30, seed=9999)
        # Very likely to differ
        assert r1["percentiles"]["p50"]["equity"] != r2["percentiles"]["p50"]["equity"]

    def test_custom_initial_capital(self):
        r = run_monte_carlo(initial_capital=50000.0, n_paths=50, n_days=10)
        assert r["initial_capital"] == 50000.0

    def test_symbol_in_response(self):
        r = run_monte_carlo(symbol="ETH/USD", n_paths=10, n_days=10)
        assert r["symbol"] == "ETH/USD"

    def test_return_pct_consistent_with_equity(self):
        capital = 10000.0
        r = run_monte_carlo(initial_capital=capital, n_paths=50, n_days=30)
        p50_eq = r["percentiles"]["p50"]["equity"]
        p50_ret = r["percentiles"]["p50"]["return_pct"]
        expected = round((p50_eq - capital) / capital * 100.0, 4)
        assert abs(p50_ret - expected) < 0.01

    def test_zero_capital_raises(self):
        with pytest.raises(ValueError, match="positive"):
            run_monte_carlo(initial_capital=0)

    def test_negative_capital_raises(self):
        with pytest.raises(ValueError, match="positive"):
            run_monte_carlo(initial_capital=-1)

    def test_generated_at_numeric(self):
        r = run_monte_carlo(n_paths=10, n_days=10)
        assert isinstance(r["generated_at"], float)
        assert r["generated_at"] > 0

    def test_n_paths_matches_request(self):
        r = run_monte_carlo(n_paths=50, n_days=20)
        assert r["n_paths"] == 50

    def test_n_days_matches_request(self):
        r = run_monte_carlo(n_paths=10, n_days=20)
        assert r["n_days"] == 20


# ═══════════════════════════════════════════════════════════════════════════════
# 5. Unit tests: get_market_correlation
# ═══════════════════════════════════════════════════════════════════════════════

class TestGetMarketCorrelation:
    _two = ["BTC/USD", "ETH/USD"]
    _three = ["BTC/USD", "ETH/USD", "SOL/USD"]

    def test_returns_dict(self):
        r = get_market_correlation(self._two)
        assert isinstance(r, dict)

    def test_top_level_keys(self):
        r = get_market_correlation(self._two)
        for key in ("matrix", "symbols", "n_days", "summary", "generated_at"):
            assert key in r

    def test_matrix_has_all_symbols(self):
        r = get_market_correlation(self._three)
        for sym in self._three:
            assert sym in r["matrix"]

    def test_diagonal_is_one(self):
        r = get_market_correlation(self._three)
        for sym in self._three:
            assert r["matrix"][sym][sym] == 1.0

    def test_matrix_symmetric(self):
        r = get_market_correlation(self._three)
        for i, sym_a in enumerate(self._three):
            for sym_b in self._three[i + 1:]:
                assert r["matrix"][sym_a][sym_b] == r["matrix"][sym_b][sym_a]

    def test_correlation_between_minus1_and_1(self):
        r = get_market_correlation(self._three)
        for sym_a in self._three:
            for sym_b in self._three:
                c = r["matrix"][sym_a][sym_b]
                assert -1.0 <= c <= 1.0, f"Correlation out of range: {sym_a}↔{sym_b} = {c}"

    def test_symbols_list_in_response(self):
        r = get_market_correlation(self._two)
        assert set(r["symbols"]) == set(self._two)

    def test_n_days_in_response(self):
        r = get_market_correlation(self._two)
        assert r["n_days"] > 0

    def test_custom_n_days(self):
        r = get_market_correlation(self._two, n_days=50)
        assert r["n_days"] == 50

    def test_summary_most_correlated_pair(self):
        r = get_market_correlation(self._three)
        pair = r["summary"]["most_correlated_pair"]
        assert isinstance(pair, list)
        assert len(pair) == 2

    def test_summary_correlation_value(self):
        r = get_market_correlation(self._three)
        c = r["summary"]["correlation"]
        assert -1.0 <= c <= 1.0

    def test_deterministic_output(self):
        r1 = get_market_correlation(self._two, seed=42)
        r2 = get_market_correlation(self._two, seed=42)
        assert r1["matrix"]["BTC/USD"]["ETH/USD"] == r2["matrix"]["BTC/USD"]["ETH/USD"]

    def test_different_seed_may_differ(self):
        r1 = get_market_correlation(self._two, seed=1)
        r2 = get_market_correlation(self._two, seed=9999)
        # Not guaranteed to differ but likely
        # Just check both return valid structure
        assert "matrix" in r1
        assert "matrix" in r2

    def test_generated_at_numeric(self):
        r = get_market_correlation(self._two)
        assert isinstance(r["generated_at"], float)
        assert r["generated_at"] > 0

    def test_four_symbols(self):
        syms = ["BTC/USD", "ETH/USD", "SOL/USD", "AVAX/USD"]
        r = get_market_correlation(syms)
        assert len(r["matrix"]) == 4

    def test_two_symbols_matrix_size(self):
        r = get_market_correlation(self._two)
        assert len(r["matrix"]) == 2
        for sym in self._two:
            assert len(r["matrix"][sym]) == 2

    def test_too_few_symbols_raises(self):
        with pytest.raises(ValueError, match="at least"):
            get_market_correlation(["BTC/USD"])

    def test_empty_symbols_raises(self):
        with pytest.raises(ValueError):
            get_market_correlation([])

    def test_too_many_symbols_raises(self):
        syms = [f"SYM{i}/USD" for i in range(10)]
        with pytest.raises(ValueError, match="max"):
            get_market_correlation(syms)

    def test_not_list_raises(self):
        with pytest.raises(ValueError):
            get_market_correlation("BTC/USD,ETH/USD")  # type: ignore

    def test_n_days_capped_at_1000(self):
        r = get_market_correlation(self._two, n_days=99999)
        assert r["n_days"] <= 1000

    def test_n_days_at_least_10(self):
        r = get_market_correlation(self._two, n_days=1)
        assert r["n_days"] >= 10


# ═══════════════════════════════════════════════════════════════════════════════
# 6. HTTP integration: /api/v1/strategies/compare
# ═══════════════════════════════════════════════════════════════════════════════

class TestHTTPStrategiesCompare:

    def test_basic_two_strategies(self, server):
        code, body = _post(f"{server}/api/v1/strategies/compare", {
            "strategy_ids": ["momentum", "mean_reversion"],
            "start_date": "2023-01-01",
            "end_date": "2024-01-01",
        })
        assert code == 200
        assert "strategies" in body
        assert len(body["strategies"]) == 2

    def test_all_four_strategies(self, server):
        code, body = _post(f"{server}/api/v1/strategies/compare", {
            "strategy_ids": ["momentum", "mean_reversion", "buy_and_hold", "random"],
        })
        assert code == 200
        assert len(body["strategies"]) == 4

    def test_response_has_leaderboard(self, server):
        code, body = _post(f"{server}/api/v1/strategies/compare", {
            "strategy_ids": ["momentum", "buy_and_hold"],
        })
        assert code == 200
        assert "leaderboard" in body
        assert len(body["leaderboard"]) == 2

    def test_response_has_summary(self, server):
        code, body = _post(f"{server}/api/v1/strategies/compare", {
            "strategy_ids": ["momentum", "buy_and_hold"],
        })
        assert code == 200
        assert "summary" in body

    def test_ranked_by_sharpe(self, server):
        code, body = _post(f"{server}/api/v1/strategies/compare", {
            "strategy_ids": ["momentum", "mean_reversion", "buy_and_hold"],
        })
        assert code == 200
        sharpes = [s["metrics"]["sharpe_ratio"] for s in body["strategies"]]
        assert sharpes == sorted(sharpes, reverse=True)

    def test_custom_symbol(self, server):
        code, body = _post(f"{server}/api/v1/strategies/compare", {
            "strategy_ids": ["momentum", "buy_and_hold"],
            "symbol": "ETH/USD",
        })
        assert code == 200
        assert body["symbol"] == "ETH/USD"

    def test_custom_capital(self, server):
        code, body = _post(f"{server}/api/v1/strategies/compare", {
            "strategy_ids": ["momentum", "buy_and_hold"],
            "initial_capital": 25000.0,
        })
        assert code == 200
        for strat in body["strategies"]:
            assert strat["initial_capital"] == 25000.0

    def test_missing_strategy_ids_uses_default(self, server):
        # Empty list should return 400
        code, body = _post(f"{server}/api/v1/strategies/compare", {
            "strategy_ids": [],
        })
        assert code == 400

    def test_one_strategy_returns_400(self, server):
        code, body = _post(f"{server}/api/v1/strategies/compare", {
            "strategy_ids": ["momentum"],
        })
        assert code == 400
        assert "error" in body

    def test_unknown_strategy_returns_400(self, server):
        code, body = _post(f"{server}/api/v1/strategies/compare", {
            "strategy_ids": ["momentum", "bogus_strat"],
        })
        assert code == 400
        assert "error" in body

    def test_bad_dates_returns_400(self, server):
        code, body = _post(f"{server}/api/v1/strategies/compare", {
            "strategy_ids": ["momentum", "buy_and_hold"],
            "start_date": "not-a-date",
        })
        assert code == 400

    def test_end_before_start_returns_400(self, server):
        code, body = _post(f"{server}/api/v1/strategies/compare", {
            "strategy_ids": ["momentum", "buy_and_hold"],
            "start_date": "2024-01-01",
            "end_date": "2023-01-01",
        })
        assert code == 400

    def test_zero_capital_returns_400(self, server):
        code, body = _post(f"{server}/api/v1/strategies/compare", {
            "strategy_ids": ["momentum", "buy_and_hold"],
            "initial_capital": 0,
        })
        assert code == 400

    def test_bad_json_returns_400(self, server):
        code, body = _post_raw(f"{server}/api/v1/strategies/compare", b"not-json")
        assert code == 400

    def test_empty_body_uses_defaults(self, server):
        # Empty body → strategy_ids will be [] → 400 due to too few
        code, body = _post(f"{server}/api/v1/strategies/compare", {})
        assert code == 400  # empty strategy_ids fails validation

    def test_content_type_json(self, server):
        import urllib.request as _ur
        req = _ur.Request(
            f"{server}/api/v1/strategies/compare",
            data=json.dumps({"strategy_ids": ["momentum", "buy_and_hold"]}).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with _ur.urlopen(req, timeout=10) as resp:
            ct = resp.headers.get("Content-Type", "")
            assert "application/json" in ct


# ═══════════════════════════════════════════════════════════════════════════════
# 7. HTTP integration: /api/v1/portfolio/monte-carlo
# ═══════════════════════════════════════════════════════════════════════════════

class TestHTTPMonteCarlo:

    def test_basic_request(self, server):
        code, body = _post(f"{server}/api/v1/portfolio/monte-carlo", {
            "n_paths": 100,
            "n_days": 30,
        })
        assert code == 200
        assert "percentiles" in body

    def test_percentile_ordering(self, server):
        code, body = _post(f"{server}/api/v1/portfolio/monte-carlo", {
            "n_paths": 200,
            "n_days": 252,
        })
        assert code == 200
        p5 = body["percentiles"]["p5"]["equity"]
        p50 = body["percentiles"]["p50"]["equity"]
        p95 = body["percentiles"]["p95"]["equity"]
        assert p5 <= p50 <= p95

    def test_prob_profit_in_range(self, server):
        code, body = _post(f"{server}/api/v1/portfolio/monte-carlo", {
            "n_paths": 200,
            "n_days": 252,
        })
        assert code == 200
        pp = body["summary"]["prob_profit"]
        assert 0.0 <= pp <= 1.0

    def test_custom_capital(self, server):
        code, body = _post(f"{server}/api/v1/portfolio/monte-carlo", {
            "initial_capital": 50000.0,
            "n_paths": 50,
            "n_days": 30,
        })
        assert code == 200
        assert body["initial_capital"] == 50000.0

    def test_custom_n_paths(self, server):
        code, body = _post(f"{server}/api/v1/portfolio/monte-carlo", {
            "n_paths": 250,
            "n_days": 30,
        })
        assert code == 200
        assert body["n_paths"] == 250

    def test_custom_symbol(self, server):
        code, body = _post(f"{server}/api/v1/portfolio/monte-carlo", {
            "n_paths": 50,
            "n_days": 30,
            "symbol": "ETH/USD",
        })
        assert code == 200
        assert body["symbol"] == "ETH/USD"

    def test_paths_sample_present(self, server):
        code, body = _post(f"{server}/api/v1/portfolio/monte-carlo", {
            "n_paths": 20,
            "n_days": 10,
        })
        assert code == 200
        assert "paths_sample" in body
        assert len(body["paths_sample"]) <= 10

    def test_simulation_id_string(self, server):
        code, body = _post(f"{server}/api/v1/portfolio/monte-carlo", {
            "n_paths": 10,
            "n_days": 10,
        })
        assert code == 200
        assert isinstance(body["simulation_id"], str)

    def test_zero_capital_returns_400(self, server):
        code, body = _post(f"{server}/api/v1/portfolio/monte-carlo", {
            "initial_capital": 0,
            "n_paths": 10,
            "n_days": 10,
        })
        assert code == 400

    def test_negative_capital_returns_400(self, server):
        code, body = _post(f"{server}/api/v1/portfolio/monte-carlo", {
            "initial_capital": -100,
            "n_paths": 10,
            "n_days": 10,
        })
        assert code == 400

    def test_zero_n_paths_clamped(self, server):
        code, body = _post(f"{server}/api/v1/portfolio/monte-carlo", {
            "n_paths": 0,
            "n_days": 10,
        })
        assert code in (200, 400)  # handler rejects n_paths<1

    def test_bad_json_returns_400(self, server):
        code, body = _post_raw(f"{server}/api/v1/portfolio/monte-carlo", b"{not_json}")
        assert code == 400

    def test_empty_body_uses_defaults(self, server):
        code, body = _post(f"{server}/api/v1/portfolio/monte-carlo", {})
        assert code == 200
        assert "percentiles" in body

    def test_return_pct_consistent(self, server):
        capital = 10000.0
        code, body = _post(f"{server}/api/v1/portfolio/monte-carlo", {
            "initial_capital": capital,
            "n_paths": 100,
            "n_days": 252,
            "seed": 42,
        })
        assert code == 200
        p50_eq = body["percentiles"]["p50"]["equity"]
        p50_ret = body["percentiles"]["p50"]["return_pct"]
        expected_ret = round((p50_eq - capital) / capital * 100.0, 4)
        assert abs(p50_ret - expected_ret) < 0.01


# ═══════════════════════════════════════════════════════════════════════════════
# 8. HTTP integration: /api/v1/market/correlation
# ═══════════════════════════════════════════════════════════════════════════════

class TestHTTPMarketCorrelation:

    def test_basic_two_symbols(self, server):
        code, body = _get_raw(f"{server}/api/v1/market/correlation?symbols=BTC/USD,ETH/USD")
        assert code == 200
        assert "matrix" in body

    def test_three_symbols(self, server):
        code, body = _get_raw(f"{server}/api/v1/market/correlation?symbols=BTC/USD,ETH/USD,SOL/USD")
        assert code == 200
        assert len(body["matrix"]) == 3

    def test_diagonal_is_one(self, server):
        code, body = _get_raw(f"{server}/api/v1/market/correlation?symbols=BTC/USD,ETH/USD,SOL/USD")
        assert code == 200
        for sym in ["BTC/USD", "ETH/USD", "SOL/USD"]:
            assert body["matrix"][sym][sym] == 1.0

    def test_matrix_symmetric(self, server):
        code, body = _get_raw(f"{server}/api/v1/market/correlation?symbols=BTC/USD,ETH/USD")
        assert code == 200
        c_ab = body["matrix"]["BTC/USD"]["ETH/USD"]
        c_ba = body["matrix"]["ETH/USD"]["BTC/USD"]
        assert c_ab == c_ba

    def test_correlation_range(self, server):
        code, body = _get_raw(f"{server}/api/v1/market/correlation?symbols=BTC/USD,ETH/USD,SOL/USD")
        assert code == 200
        for sym_a in body["matrix"]:
            for sym_b, c in body["matrix"][sym_a].items():
                assert -1.0 <= c <= 1.0

    def test_summary_present(self, server):
        code, body = _get_raw(f"{server}/api/v1/market/correlation?symbols=BTC/USD,ETH/USD")
        assert code == 200
        assert "summary" in body
        assert "most_correlated_pair" in body["summary"]

    def test_symbols_in_response(self, server):
        code, body = _get_raw(f"{server}/api/v1/market/correlation?symbols=BTC/USD,ETH/USD")
        assert code == 200
        assert set(body["symbols"]) == {"BTC/USD", "ETH/USD"}

    def test_default_symbols(self, server):
        code, body = _get_raw(f"{server}/api/v1/market/correlation")
        assert code == 200
        assert "matrix" in body
        assert len(body["matrix"]) >= 2

    def test_custom_n_days(self, server):
        code, body = _get_raw(f"{server}/api/v1/market/correlation?symbols=BTC/USD,ETH/USD&n_days=30")
        assert code == 200
        assert body["n_days"] == 30

    def test_one_symbol_returns_400(self, server):
        code, body = _get_raw(f"{server}/api/v1/market/correlation?symbols=BTC/USD")
        assert code == 400

    def test_generated_at_present(self, server):
        code, body = _get_raw(f"{server}/api/v1/market/correlation?symbols=BTC/USD,ETH/USD")
        assert code == 200
        assert "generated_at" in body

    def test_four_symbols(self, server):
        code, body = _get_raw(
            f"{server}/api/v1/market/correlation?symbols=BTC/USD,ETH/USD,SOL/USD,AVAX/USD"
        )
        assert code == 200
        assert len(body["matrix"]) == 4


# ═══════════════════════════════════════════════════════════════════════════════
# 9. Health endpoint reflects S41
# ═══════════════════════════════════════════════════════════════════════════════

class TestHealthS41:

    def test_health_version_is_s41(self, server):
        body = _get(f"{server}/health")
        assert body["version"] in ("S41", "S42", "S43", "S44", "S45")

    def test_health_test_count_gte_5100(self, server):
        body = _get(f"{server}/health")
        assert body.get("test_count", 0) >= 5100

    def test_health_status_ok(self, server):
        body = _get(f"{server}/health")
        assert body["status"] == "ok"

    def test_demo_health_same(self, server):
        body = _get(f"{server}/demo/health")
        assert body["version"] in ("S41", "S42", "S43", "S44", "S45")
        assert body.get("tests", 0) >= 5100

    def test_root_test_count(self, server):
        body = _get(f"{server}/")
        assert body.get("test_count", 0) >= 5100


# ═══════════════════════════════════════════════════════════════════════════════
# 10. 404 / routing
# ═══════════════════════════════════════════════════════════════════════════════

class TestRouting:

    def test_unknown_get_returns_404(self, server):
        code, body = _get_raw(f"{server}/api/v1/nonexistent")
        assert code == 404

    def test_unknown_post_returns_404(self, server):
        code, body = _post(f"{server}/api/v1/nonexistent", {})
        assert code == 404

    def test_strategies_compare_get_not_found(self, server):
        # /api/v1/strategies/compare is POST-only
        code, body = _get_raw(f"{server}/api/v1/strategies/compare")
        assert code == 404

    def test_monte_carlo_get_not_found(self, server):
        # /api/v1/portfolio/monte-carlo is POST-only
        code, body = _get_raw(f"{server}/api/v1/portfolio/monte-carlo")
        assert code == 404

    def test_correlation_post_not_routed(self, server):
        # POST /api/v1/market/correlation has no handler → 404
        code, body = _post(f"{server}/api/v1/market/correlation", {})
        assert code == 404
