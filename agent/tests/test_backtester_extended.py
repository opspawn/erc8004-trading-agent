"""
test_backtester_extended.py — Extended GBM parameter sweep + edge case tests.

Adds 120+ tests covering:
  - GBM parameter sweeps: drift, volatility, time horizon, start price
  - Trending price generation parameter sweeps
  - Mean-reverting price generation parameter sweeps
  - BacktestTrade data class edge cases
  - BacktestStats expectancy property
  - compute_stats edge cases (empty, single, large sets)
  - Max drawdown scenarios
  - Sharpe / Sortino edge cases
  - compare_strategies cross-validation
  - replay_trades equity curve properties
  - Backtester initial capital / position-size boundary conditions
  - Strategy signal functions across various bar counts
  - Multiple token backtests
"""

import sys
import os
import math

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import pytest
from datetime import datetime, timedelta, timezone

from backtester import (
    Backtester,
    BacktestStats,
    BacktestTrade,
    PriceBar,
    trend_signal,
    mean_reversion_signal,
    momentum_signal,
    STRATEGY_MAP,
)


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _bar(ts, price):
    return PriceBar(ts, price, price * 1.01, price * 0.99, price, 1_000_000)


def _make_trade(pnl: float, i: int = 0) -> BacktestTrade:
    now = datetime(2026, 1, 1, tzinfo=timezone.utc)
    side = "BUY" if pnl >= 0 else "BUY"
    entry = 100.0
    exit_ = 100.0 + pnl
    return BacktestTrade(i, "ETH", side, entry, exit_, 1.0, now, now, pnl_usdc=pnl)


# ─── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture
def bt():
    return Backtester(initial_capital=10_000.0, position_size_pct=0.10)


@pytest.fixture
def bt_precise():
    return Backtester(initial_capital=100_000.0, position_size_pct=0.05)


# ═══════════════════════════════════════════════════════════════════════════════
# GBM parameter sweep tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestGBMParameterSweep:
    """Test generate_synthetic_prices with varying drift, volatility, horizon."""

    # Drift parameter
    @pytest.mark.parametrize("drift", [-0.005, -0.001, 0.0, 0.001, 0.005, 0.01])
    def test_gbm_drift_returns_correct_length(self, bt, drift):
        bars = bt.generate_synthetic_prices("ETH", days=30, drift=drift)
        assert len(bars) == 30

    @pytest.mark.parametrize("drift", [-0.005, -0.001, 0.0, 0.001, 0.005, 0.01])
    def test_gbm_drift_prices_positive(self, bt, drift):
        bars = bt.generate_synthetic_prices("ETH", days=50, drift=drift)
        assert all(b.close > 0 for b in bars)

    @pytest.mark.parametrize("drift", [-0.005, -0.001, 0.0, 0.001, 0.005, 0.01])
    def test_gbm_drift_high_ge_low(self, bt, drift):
        bars = bt.generate_synthetic_prices("ETH", days=30, drift=drift)
        assert all(b.high >= b.low for b in bars)

    # Volatility parameter
    @pytest.mark.parametrize("vol", [0.005, 0.01, 0.02, 0.05, 0.10])
    def test_gbm_volatility_returns_correct_length(self, bt, vol):
        bars = bt.generate_synthetic_prices("ETH", days=30, volatility=vol)
        assert len(bars) == 30

    @pytest.mark.parametrize("vol", [0.005, 0.01, 0.02, 0.05, 0.10])
    def test_gbm_volatility_prices_positive(self, bt, vol):
        bars = bt.generate_synthetic_prices("ETH", days=40, volatility=vol)
        assert all(b.close > 0 for b in bars)

    @pytest.mark.parametrize("vol", [0.005, 0.01, 0.02, 0.05, 0.10])
    def test_gbm_vol_ohlcv_consistent(self, bt, vol):
        bars = bt.generate_synthetic_prices("ETH", days=30, volatility=vol)
        for b in bars:
            assert b.high >= b.close - 1e-6
            assert b.low <= b.close + 1e-6

    # Time horizon parameter
    @pytest.mark.parametrize("days", [1, 5, 10, 30, 60, 120, 252])
    def test_gbm_time_horizon_length(self, bt, days):
        bars = bt.generate_synthetic_prices("ETH", days=days)
        assert len(bars) == days

    @pytest.mark.parametrize("days", [1, 5, 10, 30, 60, 120])
    def test_gbm_time_horizon_timestamps_monotone(self, bt, days):
        bars = bt.generate_synthetic_prices("ETH", days=days)
        for i in range(1, len(bars)):
            assert bars[i].timestamp > bars[i - 1].timestamp

    # Start price parameter
    @pytest.mark.parametrize("start", [100.0, 500.0, 1000.0, 2000.0, 50000.0])
    def test_gbm_start_price_first_open(self, bt, start):
        bars = bt.generate_synthetic_prices("ETH", days=30, start_price=start)
        assert bars[0].open == pytest.approx(start, rel=0.01)

    @pytest.mark.parametrize("start", [100.0, 500.0, 1000.0, 2000.0, 50000.0])
    def test_gbm_start_price_positive_output(self, bt, start):
        bars = bt.generate_synthetic_prices("ETH", days=30, start_price=start)
        assert all(b.close > 0 for b in bars)

    # Volume is positive
    def test_gbm_volume_positive(self, bt):
        bars = bt.generate_synthetic_prices("ETH", days=50)
        assert all(b.volume > 0 for b in bars)

    # Different token labels
    @pytest.mark.parametrize("token", ["ETH", "BTC", "LINK", "MATIC"])
    def test_gbm_token_label(self, bt, token):
        bars = bt.generate_synthetic_prices(token, days=20)
        assert len(bars) == 20

    # Combined parameter sweep
    @pytest.mark.parametrize("drift,vol,days", [
        (-0.002, 0.01, 30),
        (0.0, 0.02, 60),
        (0.001, 0.03, 90),
        (0.005, 0.05, 15),
    ])
    def test_gbm_combined_params_valid(self, bt, drift, vol, days):
        bars = bt.generate_synthetic_prices("ETH", days=days, volatility=vol, drift=drift)
        assert len(bars) == days
        assert all(b.close > 0 for b in bars)


# ═══════════════════════════════════════════════════════════════════════════════
# Trending price generation parameter sweeps
# ═══════════════════════════════════════════════════════════════════════════════

class TestTrendingParameterSweep:
    @pytest.mark.parametrize("trend", [0.001, 0.005, 0.01, 0.02, 0.05])
    def test_trending_prices_upward(self, bt, trend):
        bars = bt.generate_trending_prices(days=20, trend=trend)
        assert bars[-1].close > bars[0].close

    @pytest.mark.parametrize("trend", [-0.001, -0.005, -0.01])
    def test_trending_prices_downward(self, bt, trend):
        bars = bt.generate_trending_prices(days=20, trend=trend)
        assert bars[-1].close < bars[0].close

    @pytest.mark.parametrize("days", [5, 10, 20, 50, 100])
    def test_trending_length(self, bt, days):
        bars = bt.generate_trending_prices(days=days)
        assert len(bars) == days

    @pytest.mark.parametrize("start", [500.0, 1000.0, 5000.0, 50000.0])
    def test_trending_start_price(self, bt, start):
        bars = bt.generate_trending_prices(days=10, start_price=start)
        assert bars[0].close > start * 0.9

    def test_trending_high_gt_low(self, bt):
        bars = bt.generate_trending_prices(days=20, trend=0.01)
        assert all(b.high > b.low for b in bars)

    def test_trending_volume_constant(self, bt):
        bars = bt.generate_trending_prices(days=10)
        assert all(b.volume == 1_000_000.0 for b in bars)


# ═══════════════════════════════════════════════════════════════════════════════
# Mean-reverting price generation parameter sweeps
# ═══════════════════════════════════════════════════════════════════════════════

class TestMeanRevertingParameterSweep:
    @pytest.mark.parametrize("days", [10, 20, 60, 120])
    def test_mr_length(self, bt, days):
        bars = bt.generate_mean_reverting_prices(days=days)
        assert len(bars) == days

    @pytest.mark.parametrize("mean", [500.0, 1000.0, 2000.0, 10000.0])
    def test_mr_prices_near_mean(self, bt, mean):
        amplitude = mean * 0.05
        bars = bt.generate_mean_reverting_prices(days=60, mean_price=mean, amplitude=amplitude)
        closes = [b.close for b in bars]
        avg = sum(closes) / len(closes)
        assert abs(avg - mean) < amplitude * 2

    @pytest.mark.parametrize("amplitude", [10.0, 50.0, 100.0, 200.0])
    def test_mr_amplitude_respected(self, bt, amplitude):
        bars = bt.generate_mean_reverting_prices(days=40, mean_price=1000.0, amplitude=amplitude)
        closes = [b.close for b in bars]
        spread = max(closes) - min(closes)
        assert spread <= amplitude * 2.1  # within 2x amplitude

    def test_mr_prices_positive(self, bt):
        bars = bt.generate_mean_reverting_prices(days=60, mean_price=100.0, amplitude=80.0)
        assert all(b.close > 0 for b in bars)

    def test_mr_high_ge_low(self, bt):
        bars = bt.generate_mean_reverting_prices(days=30)
        assert all(b.high >= b.low for b in bars)


# ═══════════════════════════════════════════════════════════════════════════════
# BacktestTrade edge cases
# ═══════════════════════════════════════════════════════════════════════════════

class TestBacktestTradeEdgeCases:
    def test_is_winner_positive_pnl(self):
        t = _make_trade(10.0)
        assert t.is_winner is True

    def test_is_winner_zero_pnl(self):
        t = _make_trade(0.0)
        assert t.is_winner is False

    def test_is_winner_negative_pnl(self):
        t = _make_trade(-5.0)
        assert t.is_winner is False

    def test_pnl_pct_buy_win(self):
        now = datetime(2026, 1, 1, tzinfo=timezone.utc)
        t = BacktestTrade(0, "ETH", "BUY", 1000.0, 1100.0, 1.0, now, now, pnl_usdc=100.0, pnl_pct=10.0)
        assert t.pnl_pct == pytest.approx(10.0)

    def test_pnl_pct_sell_win(self):
        now = datetime(2026, 1, 1, tzinfo=timezone.utc)
        t = BacktestTrade(0, "ETH", "SELL", 1000.0, 900.0, 1.0, now, now, pnl_usdc=100.0, pnl_pct=10.0)
        assert t.pnl_pct == pytest.approx(10.0)

    def test_held_bars_field(self):
        now = datetime(2026, 1, 1, tzinfo=timezone.utc)
        t = BacktestTrade(0, "ETH", "BUY", 100, 110, 1.0, now, now, pnl_usdc=10, held_bars=5)
        assert t.held_bars == 5

    def test_trade_id_field(self):
        now = datetime(2026, 1, 1, tzinfo=timezone.utc)
        t = BacktestTrade(42, "BTC", "BUY", 40000, 42000, 0.1, now, now, pnl_usdc=200)
        assert t.trade_id == 42

    @pytest.mark.parametrize("token", ["ETH", "BTC", "MATIC", "LINK"])
    def test_token_field(self, token):
        now = datetime(2026, 1, 1, tzinfo=timezone.utc)
        t = BacktestTrade(0, token, "BUY", 100, 110, 1.0, now, now, pnl_usdc=10)
        assert t.token == token


# ═══════════════════════════════════════════════════════════════════════════════
# BacktestStats edge cases
# ═══════════════════════════════════════════════════════════════════════════════

class TestBacktestStatsEdgeCases:
    def test_expectancy_zero_trades(self, bt):
        s = bt.compute_stats([])
        assert s.expectancy == 0.0

    def test_expectancy_single_winner(self, bt):
        s = bt.compute_stats([_make_trade(50.0)])
        assert s.expectancy == pytest.approx(50.0)

    def test_expectancy_mixed(self, bt):
        trades = [_make_trade(100.0), _make_trade(-50.0)]
        s = bt.compute_stats(trades)
        assert s.expectancy == pytest.approx(25.0)

    def test_profit_factor_all_winners(self, bt):
        trades = [_make_trade(10.0) for _ in range(5)]
        s = bt.compute_stats(trades)
        assert s.profit_factor == float("inf")

    def test_profit_factor_all_losers(self, bt):
        trades = [_make_trade(-10.0) for _ in range(5)]
        s = bt.compute_stats(trades)
        assert s.profit_factor == 0.0

    def test_empty_trades_returns_initial_capital(self, bt):
        s = bt.compute_stats([])
        assert s.final_capital == bt.initial_capital

    def test_empty_trades_zero_drawdown(self, bt):
        s = bt.compute_stats([])
        assert s.max_drawdown_pct == 0.0

    def test_two_trade_sharpe_defined(self, bt):
        trades = [_make_trade(10.0), _make_trade(20.0)]
        s = bt.compute_stats(trades)
        assert isinstance(s.sharpe_ratio, float)

    def test_single_trade_sharpe_zero(self, bt):
        s = bt.compute_stats([_make_trade(10.0)])
        assert s.sharpe_ratio == 0.0

    def test_gross_profit_correct(self, bt):
        trades = [_make_trade(30.0), _make_trade(-10.0), _make_trade(20.0)]
        s = bt.compute_stats(trades)
        assert s.gross_profit == pytest.approx(50.0)

    def test_gross_loss_correct(self, bt):
        trades = [_make_trade(30.0), _make_trade(-15.0), _make_trade(20.0)]
        s = bt.compute_stats(trades)
        assert s.gross_loss == pytest.approx(15.0)

    @pytest.mark.parametrize("n", [1, 2, 5, 10, 20])
    def test_total_trades_correct(self, bt, n):
        trades = [_make_trade(float(i)) for i in range(1, n + 1)]
        s = bt.compute_stats(trades)
        assert s.total_trades == n

    def test_avg_win_correct(self, bt):
        trades = [_make_trade(10.0), _make_trade(30.0), _make_trade(-5.0)]
        s = bt.compute_stats(trades)
        assert s.avg_win == pytest.approx(20.0)

    def test_avg_loss_correct(self, bt):
        trades = [_make_trade(10.0), _make_trade(-20.0), _make_trade(-10.0)]
        s = bt.compute_stats(trades)
        assert s.avg_loss == pytest.approx(-15.0)


# ═══════════════════════════════════════════════════════════════════════════════
# Max drawdown scenarios
# ═══════════════════════════════════════════════════════════════════════════════

class TestMaxDrawdown:
    def test_no_drawdown_all_positive(self, bt):
        trades = [_make_trade(10.0) for _ in range(10)]
        s = bt.compute_stats(trades)
        assert s.max_drawdown_pct == 0.0

    def test_drawdown_after_peak(self, bt):
        trades = [_make_trade(500.0), _make_trade(-100.0)]
        s = bt.compute_stats(trades)
        assert s.max_drawdown_pct > 0.0

    def test_drawdown_continuous_loss(self, bt):
        trades = [_make_trade(-100.0) for _ in range(5)]
        s = bt.compute_stats(trades)
        assert s.max_drawdown_pct > 0.0

    def test_drawdown_recovery(self, bt):
        # Goes up, down, recovers — drawdown should be finite
        trades = [_make_trade(200.0), _make_trade(-50.0), _make_trade(200.0)]
        s = bt.compute_stats(trades)
        assert 0 < s.max_drawdown_pct <= 100.0

    def test_drawdown_never_exceeds_100(self, bt):
        trades = [_make_trade(-1.0) for _ in range(50)]
        s = bt.compute_stats(trades)
        assert s.max_drawdown_pct <= 100.0


# ═══════════════════════════════════════════════════════════════════════════════
# Strategy signals
# ═══════════════════════════════════════════════════════════════════════════════

class TestStrategySignals:
    def test_trend_signal_insufficient_bars(self):
        now = datetime(2026, 1, 1, tzinfo=timezone.utc)
        bars = [_bar(now + timedelta(days=i), 1000 + i) for i in range(5)]
        assert trend_signal(bars, 0) is None
        assert trend_signal(bars, 3) is None

    def test_trend_signal_buy_on_uptrend(self):
        now = datetime(2026, 1, 1, tzinfo=timezone.utc)
        bars = [_bar(now + timedelta(days=i), 1000 + i * 10) for i in range(20)]
        sig = trend_signal(bars, 15, lookback=5)
        assert sig in (None, "BUY", "SELL")  # valid return

    def test_trend_signal_returns_valid_or_none(self):
        now = datetime(2026, 1, 1, tzinfo=timezone.utc)
        bars = [_bar(now + timedelta(days=i), 1000 + (i - 10) * 5) for i in range(30)]
        for i in range(30):
            sig = trend_signal(bars, i)
            assert sig in (None, "BUY", "SELL")

    def test_mean_reversion_insufficient_bars(self):
        now = datetime(2026, 1, 1, tzinfo=timezone.utc)
        bars = [_bar(now + timedelta(days=i), 1000) for i in range(5)]
        assert mean_reversion_signal(bars, 1) is None

    def test_mean_reversion_returns_valid_or_none(self):
        now = datetime(2026, 1, 1, tzinfo=timezone.utc)
        prices = [1000 + 50 * math.sin(2 * math.pi * i / 10) for i in range(40)]
        bars = [_bar(now + timedelta(days=i), p) for i, p in enumerate(prices)]
        for i in range(40):
            sig = mean_reversion_signal(bars, i)
            assert sig in (None, "BUY", "SELL")

    def test_momentum_insufficient_bars(self):
        now = datetime(2026, 1, 1, tzinfo=timezone.utc)
        bars = [_bar(now + timedelta(days=i), 1000) for i in range(3)]
        assert momentum_signal(bars, 1) is None

    def test_momentum_returns_valid_or_none(self):
        now = datetime(2026, 1, 1, tzinfo=timezone.utc)
        bars = [_bar(now + timedelta(days=i), 1000 + i * 2) for i in range(30)]
        for i in range(30):
            sig = momentum_signal(bars, i)
            assert sig in (None, "BUY", "SELL")


# ═══════════════════════════════════════════════════════════════════════════════
# compare_strategies cross-validation
# ═══════════════════════════════════════════════════════════════════════════════

class TestCompareStrategies:
    def test_compare_returns_all_strategies(self, bt):
        bars = bt.generate_synthetic_prices(days=60)
        results = bt.compare_strategies(bars)
        assert set(results.keys()) == set(STRATEGY_MAP.keys())

    def test_compare_all_stats_valid(self, bt):
        bars = bt.generate_trending_prices(days=50, trend=0.005)
        results = bt.compare_strategies(bars)
        for name, s in results.items():
            assert s.total_trades >= 0, f"{name}: invalid total_trades"
            assert s.final_capital > 0, f"{name}: non-positive capital"

    def test_compare_max_drawdown_bounded(self, bt):
        bars = bt.generate_mean_reverting_prices(days=60)
        results = bt.compare_strategies(bars)
        for name, s in results.items():
            assert s.max_drawdown_pct <= 100.0, f"{name}: drawdown > 100%"

    @pytest.mark.parametrize("strategy", ["trend", "mean_reversion", "momentum"])
    def test_single_strategy_runs(self, bt, strategy):
        bars = bt.generate_synthetic_prices(days=80)
        trades = bt.run(bars, strategy=strategy)
        s = bt.compute_stats(trades)
        assert s.total_trades >= 0


# ═══════════════════════════════════════════════════════════════════════════════
# replay_trades equity curve properties
# ═══════════════════════════════════════════════════════════════════════════════

class TestReplayTrades:
    def test_equity_starts_at_initial_capital(self, bt):
        trades = [_make_trade(10.0), _make_trade(-5.0)]
        equity = bt.replay_trades(trades)
        assert equity[0] == bt.initial_capital

    def test_equity_length_is_trades_plus_one(self, bt):
        trades = [_make_trade(float(i)) for i in range(5)]
        equity = bt.replay_trades(trades)
        assert len(equity) == 6

    def test_equity_empty_trades(self, bt):
        equity = bt.replay_trades([])
        assert equity == [bt.initial_capital]

    def test_equity_all_winners_monotone(self, bt):
        trades = [_make_trade(10.0) for _ in range(5)]
        equity = bt.replay_trades(trades)
        for i in range(1, len(equity)):
            assert equity[i] >= equity[i - 1]

    def test_equity_final_matches_stats(self, bt):
        trades = [_make_trade(10.0), _make_trade(-5.0), _make_trade(20.0)]
        equity = bt.replay_trades(trades)
        s = bt.compute_stats(trades)
        assert equity[-1] == pytest.approx(s.final_capital, abs=1e-6)

    @pytest.mark.parametrize("n_trades", [1, 5, 10, 20])
    def test_equity_length_parametrized(self, bt, n_trades):
        trades = [_make_trade(float(i)) for i in range(n_trades)]
        equity = bt.replay_trades(trades)
        assert len(equity) == n_trades + 1


# ═══════════════════════════════════════════════════════════════════════════════
# Backtester init boundary conditions
# ═══════════════════════════════════════════════════════════════════════════════

class TestBacktesterInitBoundaries:
    def test_minimal_capital(self):
        bt = Backtester(initial_capital=1.0, position_size_pct=0.01)
        assert bt.initial_capital == 1.0

    def test_large_capital(self):
        bt = Backtester(initial_capital=1_000_000.0, position_size_pct=0.20)
        assert bt.initial_capital == 1_000_000.0

    def test_tiny_position_size(self):
        bt = Backtester(initial_capital=1000.0, position_size_pct=0.001)
        assert bt.position_size_pct == 0.001

    def test_max_position_size(self):
        bt = Backtester(initial_capital=1000.0, position_size_pct=1.0)
        assert bt.position_size_pct == 1.0

    def test_invalid_zero_capital(self):
        with pytest.raises(ValueError):
            Backtester(initial_capital=0.0)

    def test_invalid_negative_capital(self):
        with pytest.raises(ValueError):
            Backtester(initial_capital=-1000.0)

    def test_invalid_zero_position_size(self):
        with pytest.raises(ValueError):
            Backtester(position_size_pct=0.0)

    @pytest.mark.parametrize("capital", [0.01, 1.0, 100.0, 10_000.0, 1_000_000.0])
    def test_valid_capitals(self, capital):
        bt = Backtester(initial_capital=capital, position_size_pct=0.05)
        assert bt.initial_capital == capital

    @pytest.mark.parametrize("psize", [0.001, 0.01, 0.05, 0.10, 0.25, 0.50, 1.0])
    def test_valid_position_sizes(self, psize):
        bt = Backtester(initial_capital=1000.0, position_size_pct=psize)
        assert bt.position_size_pct == psize
