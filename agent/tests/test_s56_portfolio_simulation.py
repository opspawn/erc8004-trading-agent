"""
S56 Tests: Portfolio P&L Simulation, Trade History, and Enhanced Leaderboard

Tests for:
  - GET /api/v1/portfolio/simulation
  - GET /api/v1/trades/history
  - GET /api/v1/leaderboard  (enhanced with P&L)
  - server version bump to S56
"""

import json
import time
import threading
import urllib.request
import urllib.error
from http.server import HTTPServer

import pytest

# ─── import the module under test ─────────────────────────────────────────────
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from demo_server import (
    get_s56_portfolio_simulation,
    get_s56_trade_history,
    get_s56_leaderboard,
    _S56_PORTFOLIO_BASE,
    _S56_TRADE_HISTORY,
    _S56_LEADERBOARD_DATA,
    SERVER_VERSION,
    _S56_TEST_COUNT_CONST,
)


# ─── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def portfolio_sim():
    return get_s56_portfolio_simulation()


@pytest.fixture(scope="module")
def portfolio_sim_50k():
    return get_s56_portfolio_simulation(50000.0)


@pytest.fixture(scope="module")
def trade_history_default():
    return get_s56_trade_history()


@pytest.fixture(scope="module")
def trade_history_5():
    return get_s56_trade_history(5)


@pytest.fixture(scope="module")
def leaderboard_default():
    return get_s56_leaderboard()


@pytest.fixture(scope="module")
def leaderboard_3():
    return get_s56_leaderboard(3)


# ─── Test /api/v1/portfolio/simulation returns 200 ────────────────────────────

class TestPortfolioSimulationStructure:
    """Test portfolio simulation response structure."""

    def test_portfolio_sim_returns_dict(self, portfolio_sim):
        assert isinstance(portfolio_sim, dict)

    def test_portfolio_sim_has_initial_capital(self, portfolio_sim):
        assert "initial_capital" in portfolio_sim

    def test_portfolio_sim_has_current_value(self, portfolio_sim):
        assert "current_value" in portfolio_sim

    def test_portfolio_sim_has_total_pnl(self, portfolio_sim):
        assert "total_pnl" in portfolio_sim

    def test_portfolio_sim_has_total_pnl_pct(self, portfolio_sim):
        assert "total_pnl_pct" in portfolio_sim


class TestPortfolioSimulationCapital:
    """Test initial_capital preserved and scaled correctly."""

    def test_default_initial_capital_is_10000(self, portfolio_sim):
        assert portfolio_sim["initial_capital"] == 10000

    def test_initial_capital_preserved_at_50k(self, portfolio_sim_50k):
        assert portfolio_sim_50k["initial_capital"] == 50000.0

    def test_pnl_scales_with_capital(self, portfolio_sim_50k):
        # 5x capital → 5x P&L
        base_pnl = _S56_PORTFOLIO_BASE["total_pnl"]
        assert abs(portfolio_sim_50k["total_pnl"] - base_pnl * 5) < 0.01

    def test_current_value_scales_with_capital(self, portfolio_sim_50k):
        assert portfolio_sim_50k["current_value"] > 50000.0

    def test_pnl_pct_same_regardless_of_capital(self, portfolio_sim, portfolio_sim_50k):
        assert abs(portfolio_sim["total_pnl_pct"] - portfolio_sim_50k["total_pnl_pct"]) < 0.001


class TestPortfolioSimulationPositive:
    """Test that P&L is positive (agents are profitable)."""

    def test_current_value_greater_than_initial(self, portfolio_sim):
        assert portfolio_sim["current_value"] > portfolio_sim["initial_capital"]

    def test_total_pnl_positive(self, portfolio_sim):
        assert portfolio_sim["total_pnl"] > 0

    def test_total_pnl_pct_positive(self, portfolio_sim):
        assert portfolio_sim["total_pnl_pct"] > 0

    def test_current_value_consistent(self, portfolio_sim):
        expected = portfolio_sim["initial_capital"] + portfolio_sim["total_pnl"]
        assert abs(portfolio_sim["current_value"] - expected) < 0.01

    def test_pnl_pct_consistent(self, portfolio_sim):
        expected_pct = (portfolio_sim["total_pnl"] / portfolio_sim["initial_capital"]) * 100
        assert abs(portfolio_sim["total_pnl_pct"] - expected_pct) < 0.01


class TestPortfolioSimulationPositions:
    """Test positions array has 3 entries with required fields."""

    def test_positions_key_present(self, portfolio_sim):
        assert "positions" in portfolio_sim

    def test_positions_is_list(self, portfolio_sim):
        assert isinstance(portfolio_sim["positions"], list)

    def test_positions_has_3_entries(self, portfolio_sim):
        assert len(portfolio_sim["positions"]) == 3

    def test_btc_position_present(self, portfolio_sim):
        symbols = [p["symbol"] for p in portfolio_sim["positions"]]
        assert "BTC-USD" in symbols

    def test_eth_position_present(self, portfolio_sim):
        symbols = [p["symbol"] for p in portfolio_sim["positions"]]
        assert "ETH-USD" in symbols

    def test_sol_position_present(self, portfolio_sim):
        symbols = [p["symbol"] for p in portfolio_sim["positions"]]
        assert "SOL-USD" in symbols


class TestPortfolioSimulationPositionFields:
    """Test each position has required fields."""

    def test_all_positions_have_symbol(self, portfolio_sim):
        for pos in portfolio_sim["positions"]:
            assert "symbol" in pos

    def test_all_positions_have_entry_price(self, portfolio_sim):
        for pos in portfolio_sim["positions"]:
            assert "entry_price" in pos

    def test_all_positions_have_current_price(self, portfolio_sim):
        for pos in portfolio_sim["positions"]:
            assert "current_price" in pos

    def test_all_positions_have_unrealized_pnl(self, portfolio_sim):
        for pos in portfolio_sim["positions"]:
            assert "unrealized_pnl" in pos

    def test_all_positions_have_signal_used(self, portfolio_sim):
        for pos in portfolio_sim["positions"]:
            assert "signal_used" in pos

    def test_all_positions_have_confidence(self, portfolio_sim):
        for pos in portfolio_sim["positions"]:
            assert "confidence" in pos

    def test_all_positions_have_positive_entry_price(self, portfolio_sim):
        for pos in portfolio_sim["positions"]:
            assert pos["entry_price"] > 0

    def test_all_positions_have_positive_current_price(self, portfolio_sim):
        for pos in portfolio_sim["positions"]:
            assert pos["current_price"] > 0

    def test_all_positions_have_valid_signal(self, portfolio_sim):
        valid_signals = {"BUY", "SELL", "HOLD"}
        for pos in portfolio_sim["positions"]:
            assert pos["signal_used"] in valid_signals

    def test_all_positions_have_confidence_in_range(self, portfolio_sim):
        for pos in portfolio_sim["positions"]:
            assert 0 <= pos["confidence"] <= 100


class TestPortfolioSimulationTradeHistory:
    """Test trade_history has >= 10 trades."""

    def test_trade_history_key_present(self, portfolio_sim):
        assert "trade_history" in portfolio_sim

    def test_trade_history_is_list(self, portfolio_sim):
        assert isinstance(portfolio_sim["trade_history"], list)

    def test_trade_history_has_at_least_10_trades(self, portfolio_sim):
        assert len(portfolio_sim["trade_history"]) >= 10

    def test_each_trade_has_date(self, portfolio_sim):
        for trade in portfolio_sim["trade_history"]:
            assert "date" in trade

    def test_each_trade_has_symbol(self, portfolio_sim):
        for trade in portfolio_sim["trade_history"]:
            assert "symbol" in trade

    def test_each_trade_has_action(self, portfolio_sim):
        for trade in portfolio_sim["trade_history"]:
            assert "action" in trade

    def test_each_trade_has_price(self, portfolio_sim):
        for trade in portfolio_sim["trade_history"]:
            assert "price" in trade

    def test_each_trade_action_is_buy_or_sell(self, portfolio_sim):
        for trade in portfolio_sim["trade_history"]:
            assert trade["action"] in ("BUY", "SELL")

    def test_each_trade_price_is_positive(self, portfolio_sim):
        for trade in portfolio_sim["trade_history"]:
            assert trade["price"] > 0


class TestPortfolioSimulationRiskMetrics:
    """Test risk_metrics has required fields."""

    def test_risk_metrics_key_present(self, portfolio_sim):
        assert "risk_metrics" in portfolio_sim

    def test_risk_metrics_has_max_drawdown_pct(self, portfolio_sim):
        assert "max_drawdown_pct" in portfolio_sim["risk_metrics"]

    def test_risk_metrics_has_var_95(self, portfolio_sim):
        assert "var_95" in portfolio_sim["risk_metrics"]

    def test_risk_metrics_has_win_rate(self, portfolio_sim):
        assert "win_rate" in portfolio_sim["risk_metrics"]

    def test_risk_metrics_has_volatility_daily(self, portfolio_sim):
        assert "volatility_daily" in portfolio_sim["risk_metrics"]

    def test_max_drawdown_is_negative(self, portfolio_sim):
        assert portfolio_sim["risk_metrics"]["max_drawdown_pct"] < 0

    def test_win_rate_in_valid_range(self, portfolio_sim):
        wr = portfolio_sim["risk_metrics"]["win_rate"]
        assert 0 < wr < 1

    def test_var_95_is_negative(self, portfolio_sim):
        assert portfolio_sim["risk_metrics"]["var_95"] < 0

    def test_volatility_daily_is_positive(self, portfolio_sim):
        assert portfolio_sim["risk_metrics"]["volatility_daily"] > 0


class TestPortfolioSimulationMeta:
    """Test metadata fields."""

    def test_has_period_days(self, portfolio_sim):
        assert "period_days" in portfolio_sim

    def test_period_days_is_30(self, portfolio_sim):
        assert portfolio_sim["period_days"] == 30

    def test_has_generated_at(self, portfolio_sim):
        assert "generated_at" in portfolio_sim

    def test_generated_at_is_recent(self, portfolio_sim):
        assert abs(portfolio_sim["generated_at"] - time.time()) < 5

    def test_version_is_s56(self, portfolio_sim):
        assert portfolio_sim["version"] == "S56"


# ─── Test /api/v1/trades/history ──────────────────────────────────────────────

class TestTradeHistoryStructure:
    """Test trades/history endpoint returns 200 with correct structure."""

    def test_trade_history_returns_dict(self, trade_history_default):
        assert isinstance(trade_history_default, dict)

    def test_trade_history_has_trades_key(self, trade_history_default):
        assert "trades" in trade_history_default

    def test_trade_history_has_total(self, trade_history_default):
        assert "total" in trade_history_default

    def test_trade_history_has_limit(self, trade_history_default):
        assert "limit" in trade_history_default

    def test_trade_history_trades_is_list(self, trade_history_default):
        assert isinstance(trade_history_default["trades"], list)


class TestTradeHistoryLimit:
    """Test trades history limit parameter working."""

    def test_default_limit_is_20(self, trade_history_default):
        assert trade_history_default["limit"] == 20

    def test_limit_5_returns_5_trades(self, trade_history_5):
        assert len(trade_history_5["trades"]) == 5

    def test_limit_5_stored_in_response(self, trade_history_5):
        assert trade_history_5["limit"] == 5

    def test_total_reflects_all_trades(self, trade_history_default):
        assert trade_history_default["total"] == len(_S56_TRADE_HISTORY)

    def test_default_returns_at_most_20(self, trade_history_default):
        assert len(trade_history_default["trades"]) <= 20


class TestTradeHistoryFields:
    """Test each trade has required fields."""

    def test_all_trades_have_date(self, trade_history_default):
        for t in trade_history_default["trades"]:
            assert "date" in t

    def test_all_trades_have_symbol(self, trade_history_default):
        for t in trade_history_default["trades"]:
            assert "symbol" in t

    def test_all_trades_have_action(self, trade_history_default):
        for t in trade_history_default["trades"]:
            assert "action" in t

    def test_all_trades_have_price(self, trade_history_default):
        for t in trade_history_default["trades"]:
            assert "price" in t

    def test_all_trades_have_signal_type(self, trade_history_default):
        for t in trade_history_default["trades"]:
            assert "signal_type" in t

    def test_all_trades_have_confidence(self, trade_history_default):
        for t in trade_history_default["trades"]:
            assert "confidence" in t

    def test_all_trades_action_buy_or_sell(self, trade_history_default):
        for t in trade_history_default["trades"]:
            assert t["action"] in ("BUY", "SELL")

    def test_all_trades_signal_type_valid(self, trade_history_default):
        valid = {"RSI", "MACD", "COMBINED"}
        for t in trade_history_default["trades"]:
            assert t["signal_type"] in valid

    def test_all_trades_confidence_in_range(self, trade_history_default):
        for t in trade_history_default["trades"]:
            assert 0 <= t["confidence"] <= 100

    def test_all_trades_price_positive(self, trade_history_default):
        for t in trade_history_default["trades"]:
            assert t["price"] > 0

    def test_has_version(self, trade_history_default):
        assert trade_history_default.get("version") == "S56"


# ─── Test /api/v1/leaderboard enhanced ────────────────────────────────────────

class TestLeaderboardStructure:
    """Test enhanced leaderboard structure."""

    def test_leaderboard_returns_dict(self, leaderboard_default):
        assert isinstance(leaderboard_default, dict)

    def test_leaderboard_has_leaderboard_key(self, leaderboard_default):
        assert "leaderboard" in leaderboard_default

    def test_leaderboard_is_list(self, leaderboard_default):
        assert isinstance(leaderboard_default["leaderboard"], list)

    def test_default_returns_5_agents(self, leaderboard_default):
        assert len(leaderboard_default["leaderboard"]) == 5

    def test_limit_3_returns_3_agents(self, leaderboard_3):
        assert len(leaderboard_3["leaderboard"]) == 3


class TestLeaderboardPnLFields:
    """Test each leaderboard entry has P&L fields."""

    def test_all_entries_have_pnl_30d(self, leaderboard_default):
        for entry in leaderboard_default["leaderboard"]:
            assert "pnl_30d" in entry

    def test_all_entries_have_pnl_pct_30d(self, leaderboard_default):
        for entry in leaderboard_default["leaderboard"]:
            assert "pnl_pct_30d" in entry

    def test_all_entries_have_sharpe_ratio(self, leaderboard_default):
        for entry in leaderboard_default["leaderboard"]:
            assert "sharpe_ratio" in entry

    def test_all_entries_have_consecutive_wins(self, leaderboard_default):
        for entry in leaderboard_default["leaderboard"]:
            assert "consecutive_wins" in entry

    def test_all_entries_have_signals_generated(self, leaderboard_default):
        for entry in leaderboard_default["leaderboard"]:
            assert "signals_generated" in entry

    def test_all_entries_have_accuracy_pct(self, leaderboard_default):
        for entry in leaderboard_default["leaderboard"]:
            assert "accuracy_pct" in entry

    def test_pnl_30d_positive_for_all(self, leaderboard_default):
        for entry in leaderboard_default["leaderboard"]:
            assert entry["pnl_30d"] > 0

    def test_sharpe_ratio_positive_for_all(self, leaderboard_default):
        for entry in leaderboard_default["leaderboard"]:
            assert entry["sharpe_ratio"] > 0

    def test_accuracy_pct_in_valid_range(self, leaderboard_default):
        for entry in leaderboard_default["leaderboard"]:
            assert 0 < entry["accuracy_pct"] <= 100

    def test_has_period_key(self, leaderboard_default):
        assert leaderboard_default.get("period") == "30d"

    def test_has_version_key(self, leaderboard_default):
        assert leaderboard_default.get("version") == "S56"


# ─── Test server version and constants ────────────────────────────────────────

class TestServerVersion:
    """Test server version constants."""

    def test_server_version_is_s56(self):
        assert SERVER_VERSION == "S56"

    def test_s56_test_count_const_at_least_6480(self):
        assert _S56_TEST_COUNT_CONST >= 6480

    def test_s56_test_count_const_is_int(self):
        assert isinstance(_S56_TEST_COUNT_CONST, int)


# ─── Test portfolio data integrity ────────────────────────────────────────────

class TestPortfolioDataIntegrity:
    """Test portfolio base data integrity."""

    def test_btc_price_realistic(self):
        btc = next(p for p in _S56_PORTFOLIO_BASE["positions"] if p["symbol"] == "BTC-USD")
        assert 80000 < btc["current_price"] < 120000

    def test_eth_price_realistic(self):
        eth = next(p for p in _S56_PORTFOLIO_BASE["positions"] if p["symbol"] == "ETH-USD")
        assert 1000 < eth["current_price"] < 10000

    def test_sol_price_realistic(self):
        sol = next(p for p in _S56_PORTFOLIO_BASE["positions"] if p["symbol"] == "SOL-USD")
        assert 50 < sol["current_price"] < 500

    def test_trade_history_has_both_buy_and_sell(self):
        actions = {t["action"] for t in _S56_PORTFOLIO_BASE["trade_history"]}
        assert "BUY" in actions
        assert "SELL" in actions

    def test_leaderboard_data_has_5_entries(self):
        assert len(_S56_LEADERBOARD_DATA) == 5

    def test_leaderboard_ranks_are_1_through_5(self):
        ranks = [e["rank"] for e in _S56_LEADERBOARD_DATA]
        assert sorted(ranks) == [1, 2, 3, 4, 5]
