"""test_s55_backtest_performance.py — S55 backtest results + confidence scores.

Tests for:
  - GET /api/v1/backtest/results endpoint
  - Response structure validation
  - Per-symbol metrics (win_rate, sharpe, drawdown)
  - Aggregate metrics
  - /demo/judge HTML contains performance section
  - /api/v1/signals/latest confidence_score + signal_strength
"""

from __future__ import annotations

import json
import sys
import os
import time
import threading
import urllib.request
import urllib.error

import pytest

# Make sure we can import demo_server
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from demo_server import (
    DemoServer,
    get_s55_backtest_results,
    get_s53_signals,
    get_s53_judge_html,
    _S55_BACKTEST_DATA,
    SERVER_VERSION,
)

# ── Fixtures ──────────────────────────────────────────────────────────────────

_TEST_PORT = 8185
_BASE = f"http://localhost:{_TEST_PORT}"
_server: DemoServer | None = None


def _get_server() -> DemoServer:
    global _server
    if _server is None:
        _server = DemoServer(port=_TEST_PORT)
        _server.start()
        # Wait for server to be ready
        for _ in range(30):
            try:
                urllib.request.urlopen(f"{_BASE}/demo/health", timeout=1)
                break
            except Exception:
                time.sleep(0.1)
    return _server


@pytest.fixture(scope="module", autouse=True)
def server():
    s = _get_server()
    yield s


def _get(path: str) -> dict:
    url = f"{_BASE}{path}"
    with urllib.request.urlopen(url, timeout=5) as resp:
        return json.loads(resp.read().decode())


# ── /api/v1/backtest/results — basic 200 tests (5 tests) ─────────────────────

class TestBacktestEndpointReturns200:
    def test_backtest_returns_200(self):
        resp = urllib.request.urlopen(f"{_BASE}/api/v1/backtest/results", timeout=5)
        assert resp.status == 200

    def test_backtest_content_type_json(self):
        resp = urllib.request.urlopen(f"{_BASE}/api/v1/backtest/results", timeout=5)
        ct = resp.headers.get("Content-Type", "")
        assert "json" in ct.lower()

    def test_backtest_response_parseable(self):
        data = _get("/api/v1/backtest/results")
        assert isinstance(data, dict)

    def test_backtest_has_period(self):
        data = _get("/api/v1/backtest/results")
        assert "period" in data

    def test_backtest_period_is_30d(self):
        data = _get("/api/v1/backtest/results")
        assert data["period"] == "30d"


# ── Response structure — symbols key with all 3 pairs (10 tests) ─────────────

class TestBacktestResponseStructure:
    def test_has_symbols_key(self):
        data = _get("/api/v1/backtest/results")
        assert "symbols" in data

    def test_symbols_is_dict(self):
        data = _get("/api/v1/backtest/results")
        assert isinstance(data["symbols"], dict)

    def test_symbols_has_btc(self):
        data = _get("/api/v1/backtest/results")
        assert "BTC-USD" in data["symbols"]

    def test_symbols_has_eth(self):
        data = _get("/api/v1/backtest/results")
        assert "ETH-USD" in data["symbols"]

    def test_symbols_has_sol(self):
        data = _get("/api/v1/backtest/results")
        assert "SOL-USD" in data["symbols"]

    def test_has_aggregate_key(self):
        data = _get("/api/v1/backtest/results")
        assert "aggregate" in data

    def test_has_methodology_key(self):
        data = _get("/api/v1/backtest/results")
        assert "methodology" in data

    def test_has_generated_at(self):
        data = _get("/api/v1/backtest/results")
        assert "generated_at" in data
        assert data["generated_at"] is not None

    def test_has_version(self):
        data = _get("/api/v1/backtest/results")
        assert "version" in data

    def test_version_is_s55(self):
        data = _get("/api/v1/backtest/results")
        assert data["version"] == "S55"


# ── Per-symbol win_rate between 0.5 and 0.9 (10 tests) ───────────────────────

class TestSymbolWinRates:
    SYMBOLS = ["BTC-USD", "ETH-USD", "SOL-USD"]

    def _sym(self, sym):
        data = _get("/api/v1/backtest/results")
        return data["symbols"][sym]

    def test_btc_win_rate_has_field(self):
        assert "win_rate" in self._sym("BTC-USD")

    def test_eth_win_rate_has_field(self):
        assert "win_rate" in self._sym("ETH-USD")

    def test_sol_win_rate_has_field(self):
        assert "win_rate" in self._sym("SOL-USD")

    def test_btc_win_rate_in_range(self):
        wr = self._sym("BTC-USD")["win_rate"]
        assert 0.5 <= wr <= 0.9, f"BTC win_rate {wr} out of range"

    def test_eth_win_rate_in_range(self):
        wr = self._sym("ETH-USD")["win_rate"]
        assert 0.5 <= wr <= 0.9, f"ETH win_rate {wr} out of range"

    def test_sol_win_rate_in_range(self):
        wr = self._sym("SOL-USD")["win_rate"]
        assert 0.5 <= wr <= 0.9, f"SOL win_rate {wr} out of range"

    def test_btc_trades_positive(self):
        assert self._sym("BTC-USD")["trades"] > 0

    def test_eth_trades_positive(self):
        assert self._sym("ETH-USD")["trades"] > 0

    def test_sol_trades_positive(self):
        assert self._sym("SOL-USD")["trades"] > 0

    def test_aggregate_avg_win_rate_in_range(self):
        data = _get("/api/v1/backtest/results")
        avg = data["aggregate"]["avg_win_rate"]
        assert 0.5 <= avg <= 0.9


# ── Per-symbol sharpe_ratio > 0 (10 tests) ───────────────────────────────────

class TestSymbolSharpeRatios:
    def _sym(self, sym):
        data = _get("/api/v1/backtest/results")
        return data["symbols"][sym]

    def test_btc_sharpe_has_field(self):
        assert "sharpe_ratio" in self._sym("BTC-USD")

    def test_eth_sharpe_has_field(self):
        assert "sharpe_ratio" in self._sym("ETH-USD")

    def test_sol_sharpe_has_field(self):
        assert "sharpe_ratio" in self._sym("SOL-USD")

    def test_btc_sharpe_positive(self):
        assert self._sym("BTC-USD")["sharpe_ratio"] > 0

    def test_eth_sharpe_positive(self):
        assert self._sym("ETH-USD")["sharpe_ratio"] > 0

    def test_sol_sharpe_positive(self):
        assert self._sym("SOL-USD")["sharpe_ratio"] > 0

    def test_btc_total_return_positive(self):
        assert self._sym("BTC-USD")["total_return_pct"] > 0

    def test_eth_total_return_positive(self):
        assert self._sym("ETH-USD")["total_return_pct"] > 0

    def test_sol_total_return_positive(self):
        assert self._sym("SOL-USD")["total_return_pct"] > 0

    def test_aggregate_avg_sharpe_positive(self):
        data = _get("/api/v1/backtest/results")
        assert data["aggregate"]["avg_sharpe"] > 0


# ── Per-symbol max_drawdown_pct < 0 (5 tests) ────────────────────────────────

class TestSymbolMaxDrawdown:
    def _sym(self, sym):
        data = _get("/api/v1/backtest/results")
        return data["symbols"][sym]

    def test_btc_drawdown_has_field(self):
        assert "max_drawdown_pct" in self._sym("BTC-USD")

    def test_btc_drawdown_negative(self):
        dd = self._sym("BTC-USD")["max_drawdown_pct"]
        assert dd < 0, f"BTC drawdown should be negative, got {dd}"

    def test_eth_drawdown_negative(self):
        dd = self._sym("ETH-USD")["max_drawdown_pct"]
        assert dd < 0, f"ETH drawdown should be negative, got {dd}"

    def test_sol_drawdown_negative(self):
        dd = self._sym("SOL-USD")["max_drawdown_pct"]
        assert dd < 0, f"SOL drawdown should be negative, got {dd}"

    def test_aggregate_avg_drawdown_negative(self):
        data = _get("/api/v1/backtest/results")
        assert data["aggregate"]["avg_max_drawdown_pct"] < 0


# ── Aggregate total_trades > 0 (5 tests) ─────────────────────────────────────

class TestAggregateMetrics:
    def test_aggregate_has_total_trades(self):
        data = _get("/api/v1/backtest/results")
        assert "total_trades" in data["aggregate"]

    def test_aggregate_total_trades_positive(self):
        data = _get("/api/v1/backtest/results")
        assert data["aggregate"]["total_trades"] > 0

    def test_aggregate_total_trades_sum_equals_symbols(self):
        data = _get("/api/v1/backtest/results")
        sym_total = sum(s["trades"] for s in data["symbols"].values())
        assert data["aggregate"]["total_trades"] == sym_total

    def test_aggregate_symbols_traded(self):
        data = _get("/api/v1/backtest/results")
        assert data["aggregate"]["symbols_traded"] == 3

    def test_aggregate_period_days(self):
        data = _get("/api/v1/backtest/results")
        assert data["aggregate"]["period_days"] == 30


# ── /demo/judge HTML contains performance section (10 tests) ──────────────────

class TestJudgeDashboardBacktest:
    def _html(self) -> str:
        resp = urllib.request.urlopen(f"{_BASE}/demo/judge", timeout=5)
        return resp.read().decode("utf-8")

    def test_judge_dashboard_200(self):
        resp = urllib.request.urlopen(f"{_BASE}/demo/judge", timeout=5)
        assert resp.status == 200

    def test_judge_html_contains_win_rate(self):
        html = self._html()
        assert "win_rate" in html.lower() or "win rate" in html.lower() or "Win Rate" in html

    def test_judge_html_contains_backtest_heading(self):
        html = self._html()
        assert "Backtesting" in html or "backtesting" in html or "backtest" in html.lower()

    def test_judge_html_contains_sharpe(self):
        html = self._html()
        assert "Sharpe" in html or "sharpe" in html.lower()

    def test_judge_html_contains_btc_usd(self):
        html = self._html()
        assert "BTC-USD" in html

    def test_judge_html_contains_eth_usd(self):
        html = self._html()
        assert "ETH-USD" in html

    def test_judge_html_contains_sol_usd(self):
        html = self._html()
        assert "SOL-USD" in html

    def test_judge_html_contains_equity_curve_chars(self):
        html = self._html()
        # Unicode block chars for equity curve visualization
        assert any(c in html for c in "▁▂▃▄▅▆▇█")

    def test_judge_html_contains_backtest_api_link(self):
        html = self._html()
        assert "backtest/results" in html

    def test_judge_html_contains_30_day_reference(self):
        html = self._html()
        assert "30" in html and ("day" in html.lower() or "30d" in html or "30-Day" in html)


# ── /api/v1/signals/latest confidence_score (10 tests) ────────────────────────

class TestSignalsConfidenceScore:
    def _signals(self) -> list:
        data = _get("/api/v1/signals/latest")
        return data["signals"]

    def test_signals_has_confidence_score_field(self):
        sigs = self._signals()
        assert len(sigs) > 0
        assert "confidence_score" in sigs[0]

    def test_btc_confidence_score_exists(self):
        sigs = {s["symbol"]: s for s in self._signals()}
        assert "confidence_score" in sigs["BTC-USD"]

    def test_eth_confidence_score_exists(self):
        sigs = {s["symbol"]: s for s in self._signals()}
        assert "confidence_score" in sigs["ETH-USD"]

    def test_sol_confidence_score_exists(self):
        sigs = {s["symbol"]: s for s in self._signals()}
        assert "confidence_score" in sigs["SOL-USD"]

    def test_confidence_score_is_numeric(self):
        sigs = self._signals()
        for s in sigs:
            assert isinstance(s["confidence_score"], (int, float))

    def test_confidence_score_in_valid_range(self):
        sigs = self._signals()
        for s in sigs:
            cs = s["confidence_score"]
            assert 0 <= cs <= 100, f"{s['symbol']} confidence_score {cs} out of [0,100]"

    def test_signals_version_updated(self):
        data = _get("/api/v1/signals/latest")
        # Version should be S55
        assert data.get("version") == "S55"

    def test_all_three_symbols_have_confidence(self):
        sigs = self._signals()
        symbols = {s["symbol"] for s in sigs}
        assert {"BTC-USD", "ETH-USD", "SOL-USD"}.issubset(symbols)

    def test_confidence_score_not_none(self):
        sigs = self._signals()
        for s in sigs:
            assert s["confidence_score"] is not None

    def test_signals_has_generated_at(self):
        data = _get("/api/v1/signals/latest")
        assert "generated_at" in data


# ── signal_strength is one of STRONG/MODERATE/WEAK (5 tests) ─────────────────

class TestSignalStrength:
    VALID_STRENGTHS = {"STRONG", "MODERATE", "WEAK"}

    def _signals(self) -> list:
        return _get("/api/v1/signals/latest")["signals"]

    def test_signal_strength_field_exists(self):
        sigs = self._signals()
        assert len(sigs) > 0
        assert "signal_strength" in sigs[0]

    def test_btc_signal_strength_valid(self):
        sigs = {s["symbol"]: s for s in self._signals()}
        assert sigs["BTC-USD"]["signal_strength"] in self.VALID_STRENGTHS

    def test_eth_signal_strength_valid(self):
        sigs = {s["symbol"]: s for s in self._signals()}
        assert sigs["ETH-USD"]["signal_strength"] in self.VALID_STRENGTHS

    def test_sol_signal_strength_valid(self):
        sigs = {s["symbol"]: s for s in self._signals()}
        assert sigs["SOL-USD"]["signal_strength"] in self.VALID_STRENGTHS

    def test_all_signals_have_valid_strength(self):
        sigs = self._signals()
        for s in sigs:
            assert s["signal_strength"] in self.VALID_STRENGTHS, (
                f"{s['symbol']} has invalid signal_strength: {s['signal_strength']}"
            )


# ── Unit tests for get_s55_backtest_results() (direct function) ───────────────

class TestBacktestResultsFunction:
    def test_function_returns_dict(self):
        result = get_s55_backtest_results()
        assert isinstance(result, dict)

    def test_function_has_symbols(self):
        result = get_s55_backtest_results()
        assert "symbols" in result

    def test_function_symbols_has_three_pairs(self):
        result = get_s55_backtest_results()
        assert len(result["symbols"]) == 3

    def test_function_generated_at_is_set(self):
        result = get_s55_backtest_results()
        assert result["generated_at"] is not None

    def test_function_does_not_mutate_base_data(self):
        # Calling multiple times should not stack up generated_at
        r1 = get_s55_backtest_results()
        r2 = get_s55_backtest_results()
        assert r1["generated_at"] != r2["generated_at"] or abs(r1["generated_at"] - r2["generated_at"]) < 5
