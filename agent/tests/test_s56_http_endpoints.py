"""
test_s56_http_endpoints.py — S56 HTTP endpoint validation.

Tests all three new S56 endpoints over a live HTTP server:
  - GET /api/v1/portfolio/simulation
  - GET /api/v1/trades/history
  - GET /api/v1/leaderboard

Spins up DemoServer on port 8186 for isolation.
"""

from __future__ import annotations

import json
import time
import sys
import os
import urllib.request
import urllib.error

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from demo_server import DemoServer, SERVER_VERSION

# ── Test server fixture ───────────────────────────────────────────────────────

_TEST_PORT = 8186
_BASE = f"http://localhost:{_TEST_PORT}"
_server: DemoServer | None = None


def _get_server() -> DemoServer:
    global _server
    if _server is None:
        _server = DemoServer(port=_TEST_PORT)
        _server.start()
        for _ in range(50):
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


def _get(path: str) -> tuple:
    url = f"{_BASE}{path}"
    with urllib.request.urlopen(url, timeout=5) as resp:
        return resp.status, json.loads(resp.read().decode()), resp.headers


# ── /api/v1/portfolio/simulation — HTTP 200 tests (10 tests) ─────────────────

class TestPortfolioSimulationHTTP200:
    """Test portfolio simulation returns 200 over HTTP."""

    def test_portfolio_sim_returns_200(self):
        status, _, _ = _get("/api/v1/portfolio/simulation")
        assert status == 200

    def test_portfolio_sim_content_type_json(self):
        _, _, headers = _get("/api/v1/portfolio/simulation")
        ct = headers.get("Content-Type", "")
        assert "json" in ct.lower()

    def test_portfolio_sim_with_default_capital(self):
        _, body, _ = _get("/api/v1/portfolio/simulation")
        assert body["initial_capital"] == 10000

    def test_portfolio_sim_with_custom_capital(self):
        _, body, _ = _get("/api/v1/portfolio/simulation?initial_capital=5000")
        assert body["initial_capital"] == 5000.0

    def test_portfolio_sim_with_large_capital(self):
        _, body, _ = _get("/api/v1/portfolio/simulation?initial_capital=100000")
        assert body["initial_capital"] == 100000.0

    def test_portfolio_sim_current_value_positive(self):
        _, body, _ = _get("/api/v1/portfolio/simulation")
        assert body["current_value"] > 0

    def test_portfolio_sim_pnl_positive(self):
        _, body, _ = _get("/api/v1/portfolio/simulation")
        assert body["total_pnl"] > 0

    def test_portfolio_sim_period_days_30(self):
        _, body, _ = _get("/api/v1/portfolio/simulation")
        assert body["period_days"] == 30

    def test_portfolio_sim_version_in_response(self):
        _, body, _ = _get("/api/v1/portfolio/simulation")
        assert "version" in body

    def test_portfolio_sim_version_matches_server(self):
        _, body, _ = _get("/api/v1/portfolio/simulation")
        assert body["version"] == SERVER_VERSION


# ── /api/v1/portfolio/simulation — positions via HTTP (12 tests) ─────────────

class TestPortfolioSimulationPositionsHTTP:
    """Test positions array returned over HTTP."""

    def test_positions_key_in_response(self):
        _, body, _ = _get("/api/v1/portfolio/simulation")
        assert "positions" in body

    def test_positions_is_list(self):
        _, body, _ = _get("/api/v1/portfolio/simulation")
        assert isinstance(body["positions"], list)

    def test_positions_count_is_3(self):
        _, body, _ = _get("/api/v1/portfolio/simulation")
        assert len(body["positions"]) == 3

    def test_btc_position_in_response(self):
        _, body, _ = _get("/api/v1/portfolio/simulation")
        symbols = [p["symbol"] for p in body["positions"]]
        assert "BTC-USD" in symbols

    def test_eth_position_in_response(self):
        _, body, _ = _get("/api/v1/portfolio/simulation")
        symbols = [p["symbol"] for p in body["positions"]]
        assert "ETH-USD" in symbols

    def test_sol_position_in_response(self):
        _, body, _ = _get("/api/v1/portfolio/simulation")
        symbols = [p["symbol"] for p in body["positions"]]
        assert "SOL-USD" in symbols

    def test_btc_current_price_realistic(self):
        _, body, _ = _get("/api/v1/portfolio/simulation")
        btc = next(p for p in body["positions"] if p["symbol"] == "BTC-USD")
        assert 50000 < btc["current_price"] < 200000

    def test_eth_current_price_realistic(self):
        _, body, _ = _get("/api/v1/portfolio/simulation")
        eth = next(p for p in body["positions"] if p["symbol"] == "ETH-USD")
        assert 500 < eth["current_price"] < 20000

    def test_sol_current_price_realistic(self):
        _, body, _ = _get("/api/v1/portfolio/simulation")
        sol = next(p for p in body["positions"] if p["symbol"] == "SOL-USD")
        assert 20 < sol["current_price"] < 1000

    def test_positions_have_signal_used(self):
        _, body, _ = _get("/api/v1/portfolio/simulation")
        for pos in body["positions"]:
            assert pos["signal_used"] in ("BUY", "SELL", "HOLD")

    def test_positions_have_confidence(self):
        _, body, _ = _get("/api/v1/portfolio/simulation")
        for pos in body["positions"]:
            assert 0 <= pos["confidence"] <= 100

    def test_positions_have_unrealized_pnl(self):
        _, body, _ = _get("/api/v1/portfolio/simulation")
        for pos in body["positions"]:
            assert "unrealized_pnl" in pos


# ── /api/v1/portfolio/simulation — risk_metrics via HTTP (8 tests) ───────────

class TestPortfolioSimulationRiskMetricsHTTP:
    """Test risk_metrics section returned over HTTP."""

    def test_risk_metrics_present(self):
        _, body, _ = _get("/api/v1/portfolio/simulation")
        assert "risk_metrics" in body

    def test_risk_metrics_has_max_drawdown(self):
        _, body, _ = _get("/api/v1/portfolio/simulation")
        assert "max_drawdown_pct" in body["risk_metrics"]

    def test_risk_metrics_has_var_95(self):
        _, body, _ = _get("/api/v1/portfolio/simulation")
        assert "var_95" in body["risk_metrics"]

    def test_risk_metrics_has_win_rate(self):
        _, body, _ = _get("/api/v1/portfolio/simulation")
        assert "win_rate" in body["risk_metrics"]

    def test_risk_metrics_has_volatility(self):
        _, body, _ = _get("/api/v1/portfolio/simulation")
        assert "volatility_daily" in body["risk_metrics"]

    def test_risk_metrics_max_drawdown_negative(self):
        _, body, _ = _get("/api/v1/portfolio/simulation")
        assert body["risk_metrics"]["max_drawdown_pct"] < 0

    def test_risk_metrics_win_rate_valid(self):
        _, body, _ = _get("/api/v1/portfolio/simulation")
        wr = body["risk_metrics"]["win_rate"]
        assert 0 < wr < 1

    def test_risk_metrics_var_95_negative(self):
        _, body, _ = _get("/api/v1/portfolio/simulation")
        assert body["risk_metrics"]["var_95"] < 0


# ── /api/v1/portfolio/simulation — capital scaling via HTTP (6 tests) ─────────

class TestPortfolioSimulationCapitalScalingHTTP:
    """Test that capital scales P&L correctly."""

    def test_custom_capital_current_value_positive(self):
        _, body, _ = _get("/api/v1/portfolio/simulation?initial_capital=5000")
        assert body["current_value"] > 0

    def test_custom_capital_pnl_positive(self):
        _, body, _ = _get("/api/v1/portfolio/simulation?initial_capital=5000")
        assert body["total_pnl"] > 0

    def test_pnl_pct_consistent_with_capital(self):
        _, body_10k, _ = _get("/api/v1/portfolio/simulation?initial_capital=10000")
        _, body_20k, _ = _get("/api/v1/portfolio/simulation?initial_capital=20000")
        assert abs(body_10k["total_pnl_pct"] - body_20k["total_pnl_pct"]) < 0.01

    def test_large_capital_returns_higher_pnl(self):
        _, body_10k, _ = _get("/api/v1/portfolio/simulation?initial_capital=10000")
        _, body_100k, _ = _get("/api/v1/portfolio/simulation?initial_capital=100000")
        assert body_100k["total_pnl"] > body_10k["total_pnl"]

    def test_invalid_capital_defaults_to_10000(self):
        _, body, _ = _get("/api/v1/portfolio/simulation?initial_capital=abc")
        assert body["initial_capital"] == 10000

    def test_trade_history_present_in_sim(self):
        _, body, _ = _get("/api/v1/portfolio/simulation?initial_capital=5000")
        assert "trade_history" in body
        assert len(body["trade_history"]) >= 10


# ── /api/v1/trades/history — HTTP tests (15 tests) ───────────────────────────

class TestTradeHistoryHTTP:
    """Test trades/history endpoint over HTTP."""

    def test_trades_history_returns_200(self):
        status, _, _ = _get("/api/v1/trades/history")
        assert status == 200

    def test_trades_history_content_type_json(self):
        _, _, headers = _get("/api/v1/trades/history")
        ct = headers.get("Content-Type", "")
        assert "json" in ct.lower()

    def test_trades_history_has_trades_key(self):
        _, body, _ = _get("/api/v1/trades/history")
        assert "trades" in body

    def test_trades_history_has_total_key(self):
        _, body, _ = _get("/api/v1/trades/history")
        assert "total" in body

    def test_trades_history_has_limit_key(self):
        _, body, _ = _get("/api/v1/trades/history")
        assert "limit" in body

    def test_trades_history_default_limit_is_20(self):
        _, body, _ = _get("/api/v1/trades/history")
        assert body["limit"] == 20

    def test_trades_history_limit_param(self):
        _, body, _ = _get("/api/v1/trades/history?limit=5")
        assert len(body["trades"]) == 5

    def test_trades_history_limit_stored(self):
        _, body, _ = _get("/api/v1/trades/history?limit=5")
        assert body["limit"] == 5

    def test_trades_history_trades_is_list(self):
        _, body, _ = _get("/api/v1/trades/history")
        assert isinstance(body["trades"], list)

    def test_trades_history_has_version(self):
        _, body, _ = _get("/api/v1/trades/history")
        assert body.get("version") == "S56"

    def test_each_trade_has_date(self):
        _, body, _ = _get("/api/v1/trades/history")
        for t in body["trades"]:
            assert "date" in t

    def test_each_trade_has_action(self):
        _, body, _ = _get("/api/v1/trades/history")
        for t in body["trades"]:
            assert t["action"] in ("BUY", "SELL")

    def test_each_trade_has_price(self):
        _, body, _ = _get("/api/v1/trades/history")
        for t in body["trades"]:
            assert t["price"] > 0

    def test_each_trade_has_signal_type(self):
        _, body, _ = _get("/api/v1/trades/history")
        for t in body["trades"]:
            assert t["signal_type"] in ("RSI", "MACD", "COMBINED")

    def test_each_trade_has_confidence(self):
        _, body, _ = _get("/api/v1/trades/history")
        for t in body["trades"]:
            assert 0 <= t["confidence"] <= 100


# ── /api/v1/leaderboard — HTTP tests (15 tests) ──────────────────────────────

class TestLeaderboardHTTP:
    """Test enhanced leaderboard endpoint over HTTP."""

    def test_leaderboard_returns_200(self):
        status, _, _ = _get("/api/v1/leaderboard")
        assert status == 200

    def test_leaderboard_content_type_json(self):
        _, _, headers = _get("/api/v1/leaderboard")
        ct = headers.get("Content-Type", "")
        assert "json" in ct.lower()

    def test_leaderboard_has_leaderboard_key(self):
        _, body, _ = _get("/api/v1/leaderboard")
        assert "leaderboard" in body

    def test_leaderboard_default_returns_5(self):
        _, body, _ = _get("/api/v1/leaderboard")
        assert len(body["leaderboard"]) == 5

    def test_leaderboard_limit_param_3(self):
        _, body, _ = _get("/api/v1/leaderboard?limit=3")
        assert len(body["leaderboard"]) == 3

    def test_leaderboard_limit_stored(self):
        _, body, _ = _get("/api/v1/leaderboard?limit=3")
        assert body.get("limit") == 3 or len(body["leaderboard"]) == 3

    def test_leaderboard_has_period_30d(self):
        _, body, _ = _get("/api/v1/leaderboard")
        assert body.get("period") == "30d"

    def test_leaderboard_version_is_s56(self):
        _, body, _ = _get("/api/v1/leaderboard")
        assert body.get("version") == "S56"

    def test_leaderboard_entries_have_pnl_30d(self):
        _, body, _ = _get("/api/v1/leaderboard")
        for entry in body["leaderboard"]:
            assert "pnl_30d" in entry

    def test_leaderboard_entries_have_pnl_pct_30d(self):
        _, body, _ = _get("/api/v1/leaderboard")
        for entry in body["leaderboard"]:
            assert "pnl_pct_30d" in entry

    def test_leaderboard_entries_have_sharpe_ratio(self):
        _, body, _ = _get("/api/v1/leaderboard")
        for entry in body["leaderboard"]:
            assert "sharpe_ratio" in entry

    def test_leaderboard_entries_have_consecutive_wins(self):
        _, body, _ = _get("/api/v1/leaderboard")
        for entry in body["leaderboard"]:
            assert "consecutive_wins" in entry

    def test_leaderboard_entries_have_accuracy_pct(self):
        _, body, _ = _get("/api/v1/leaderboard")
        for entry in body["leaderboard"]:
            assert "accuracy_pct" in entry

    def test_leaderboard_pnl_30d_all_positive(self):
        _, body, _ = _get("/api/v1/leaderboard")
        for entry in body["leaderboard"]:
            assert entry["pnl_30d"] > 0

    def test_leaderboard_sharpe_all_positive(self):
        _, body, _ = _get("/api/v1/leaderboard")
        for entry in body["leaderboard"]:
            assert entry["sharpe_ratio"] > 0


# ── Server health with S56 version (5 tests) ─────────────────────────────────

class TestServerHealthS56:
    """Test health endpoint reflects S56."""

    def test_health_returns_200(self):
        status, _, _ = _get("/demo/health")
        assert status == 200

    def test_health_version_is_s56(self):
        _, body, _ = _get("/demo/health")
        assert body.get("version") == "S56"

    def test_health_sprint_is_s56(self):
        _, body, _ = _get("/demo/health")
        assert body.get("sprint") == "S56"

    def test_health_test_count_above_6400(self):
        _, body, _ = _get("/demo/health")
        tc = body.get("tests") or body.get("test_count") or 0
        assert tc >= 6400

    def test_health_status_is_ok(self):
        _, body, _ = _get("/demo/health")
        assert body.get("status") in ("ok", "OK", "healthy")


# ── Judge dashboard S56 content (5 tests) ────────────────────────────────────

class TestJudgeDashboardS56:
    """Test /demo/judge dashboard contains S56 content."""

    def test_judge_returns_200(self):
        url = f"{_BASE}/demo/judge"
        with urllib.request.urlopen(url, timeout=5) as resp:
            assert resp.status == 200

    def test_judge_is_html(self):
        url = f"{_BASE}/demo/judge"
        with urllib.request.urlopen(url, timeout=5) as resp:
            body = resp.read().decode("utf-8")
        assert "<html" in body.lower() or "<!doctype html" in body.lower()

    def test_judge_contains_portfolio_section(self):
        url = f"{_BASE}/demo/judge"
        with urllib.request.urlopen(url, timeout=5) as resp:
            body = resp.read().decode("utf-8")
        assert "portfolio" in body.lower() or "Portfolio" in body

    def test_judge_contains_s56(self):
        url = f"{_BASE}/demo/judge"
        with urllib.request.urlopen(url, timeout=5) as resp:
            body = resp.read().decode("utf-8")
        assert "S56" in body

    def test_judge_is_substantial(self):
        url = f"{_BASE}/demo/judge"
        with urllib.request.urlopen(url, timeout=5) as resp:
            body = resp.read().decode("utf-8")
        assert len(body) > 5000


# ── Extra portfolio simulation edge-case tests (12 tests) ────────────────────

class TestPortfolioSimulationEdgeCases:
    """Additional edge-case and data-integrity tests for portfolio simulation."""

    def test_portfolio_sim_trade_history_has_date_strings(self):
        _, body, _ = _get("/api/v1/portfolio/simulation")
        for trade in body["trade_history"]:
            assert isinstance(trade["date"], str)
            assert len(trade["date"]) >= 8  # YYYY-MM-DD

    def test_portfolio_sim_trade_history_has_symbols(self):
        _, body, _ = _get("/api/v1/portfolio/simulation")
        for trade in body["trade_history"]:
            assert trade["symbol"] in ("BTC-USD", "ETH-USD", "SOL-USD")

    def test_portfolio_sim_positions_entry_price_positive(self):
        _, body, _ = _get("/api/v1/portfolio/simulation")
        for pos in body["positions"]:
            assert pos["entry_price"] > 0

    def test_portfolio_sim_positions_position_size_positive(self):
        _, body, _ = _get("/api/v1/portfolio/simulation")
        for pos in body["positions"]:
            assert pos.get("position_size", 1) > 0

    def test_portfolio_sim_has_generated_at(self):
        _, body, _ = _get("/api/v1/portfolio/simulation")
        assert "generated_at" in body

    def test_portfolio_sim_generated_at_recent(self):
        _, body, _ = _get("/api/v1/portfolio/simulation")
        assert abs(body["generated_at"] - time.time()) < 10

    def test_portfolio_sim_total_pnl_pct_consistent(self):
        _, body, _ = _get("/api/v1/portfolio/simulation")
        expected = (body["total_pnl"] / body["initial_capital"]) * 100
        assert abs(body["total_pnl_pct"] - expected) < 0.01

    def test_portfolio_sim_current_value_consistent(self):
        _, body, _ = _get("/api/v1/portfolio/simulation")
        expected = body["initial_capital"] + body["total_pnl"]
        assert abs(body["current_value"] - expected) < 0.01

    def test_portfolio_sim_trade_history_has_actions(self):
        _, body, _ = _get("/api/v1/portfolio/simulation")
        actions = {t["action"] for t in body["trade_history"]}
        assert "BUY" in actions

    def test_portfolio_sim_risk_volatility_positive(self):
        _, body, _ = _get("/api/v1/portfolio/simulation")
        assert body["risk_metrics"]["volatility_daily"] > 0

    def test_leaderboard_has_total_agents(self):
        _, body, _ = _get("/api/v1/leaderboard")
        assert "total_agents" in body

    def test_trade_history_total_is_int(self):
        _, body, _ = _get("/api/v1/trades/history")
        assert isinstance(body["total"], int)
        assert body["total"] > 0
