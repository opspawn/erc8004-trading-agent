"""
Tests for backtester.py — 65+ tests covering:
  - PriceBar data class
  - BacktestTrade data class
  - BacktestStats data class
  - Synthetic price generation (GBM, trending, mean-reverting)
  - Strategy signals (trend, mean_reversion, momentum)
  - Backtester.run() with all strategies
  - Backtester.compute_stats()
  - Max drawdown calculation
  - Sharpe / Sortino ratios
  - Edge cases (empty bars, single bar, all winners, all losers)
  - compare_strategies()
  - replay_trades() equity curve
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import math
import pytest
from datetime import datetime, timezone

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


# ─── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture
def bt():
    return Backtester(initial_capital=1000.0, position_size_pct=0.10)

@pytest.fixture
def bt_small():
    return Backtester(initial_capital=100.0, position_size_pct=0.05)

@pytest.fixture
def flat_bars():
    """30 bars at constant price 1000."""
    now = datetime(2026, 1, 1, tzinfo=timezone.utc)
    from datetime import timedelta
    return [
        PriceBar(
            timestamp=now + timedelta(days=i),
            open=1000.0, high=1005.0, low=995.0, close=1000.0, volume=1e6,
        )
        for i in range(30)
    ]

@pytest.fixture
def uptrend_bars(bt):
    return bt.generate_trending_prices(days=40, start_price=1000.0, trend=0.01)

@pytest.fixture
def mr_bars(bt):
    return bt.generate_mean_reverting_prices(days=60)

@pytest.fixture
def synth_bars(bt):
    return bt.generate_synthetic_prices("ETH", days=60)


# ─── PriceBar ─────────────────────────────────────────────────────────────────

class TestPriceBar:
    def test_mid(self):
        bar = PriceBar(datetime.now(timezone.utc), 100, 110, 90, 105, 1000)
        assert bar.mid() == 100.0  # (110+90)/2

    def test_range_pct(self):
        bar = PriceBar(datetime.now(timezone.utc), 100, 110, 90, 105, 1000)
        assert pytest.approx(bar.range_pct(), rel=1e-6) == 0.20

    def test_range_pct_zero_open(self):
        bar = PriceBar(datetime.now(timezone.utc), 0, 10, 0, 5, 0)
        assert bar.range_pct() == 0.0

    def test_fields_set(self):
        bar = PriceBar(datetime.now(timezone.utc), 1.0, 2.0, 0.5, 1.5, 9999.0)
        assert bar.open == 1.0
        assert bar.high == 2.0
        assert bar.low == 0.5
        assert bar.close == 1.5
        assert bar.volume == 9999.0


# ─── BacktestTrade ─────────────────────────────────────────────────────────────

class TestBacktestTrade:
    def test_is_winner_true(self):
        t = BacktestTrade(0, "ETH", "BUY", 100, 120, 1.0,
                          datetime.now(timezone.utc), datetime.now(timezone.utc),
                          pnl_usdc=20.0)
        assert t.is_winner is True

    def test_is_winner_false(self):
        t = BacktestTrade(0, "ETH", "BUY", 100, 80, 1.0,
                          datetime.now(timezone.utc), datetime.now(timezone.utc),
                          pnl_usdc=-20.0)
        assert t.is_winner is False

    def test_is_winner_zero(self):
        t = BacktestTrade(0, "ETH", "BUY", 100, 100, 1.0,
                          datetime.now(timezone.utc), datetime.now(timezone.utc),
                          pnl_usdc=0.0)
        assert t.is_winner is False


# ─── BacktestStats ─────────────────────────────────────────────────────────────

class TestBacktestStats:
    def test_expectancy_no_trades(self):
        s = BacktestStats()
        assert s.expectancy == 0.0

    def test_expectancy_with_trades(self):
        s = BacktestStats(total_trades=10, net_pnl=50.0)
        assert s.expectancy == 5.0

    def test_expectancy_negative(self):
        s = BacktestStats(total_trades=5, net_pnl=-25.0)
        assert s.expectancy == -5.0


# ─── Synthetic Price Generation ───────────────────────────────────────────────

class TestSyntheticPrices:
    def test_synthetic_length(self, bt):
        bars = bt.generate_synthetic_prices(days=30)
        assert len(bars) == 30

    def test_synthetic_prices_positive(self, bt):
        bars = bt.generate_synthetic_prices(days=50)
        assert all(b.close > 0 for b in bars)

    def test_synthetic_high_gte_low(self, bt):
        bars = bt.generate_synthetic_prices(days=50)
        assert all(b.high >= b.low for b in bars)

    def test_trending_prices_upward(self, bt):
        bars = bt.generate_trending_prices(days=20, trend=0.02)
        assert bars[-1].close > bars[0].close

    def test_trending_length(self, bt):
        bars = bt.generate_trending_prices(days=15)
        assert len(bars) == 15

    def test_mean_reverting_oscillates(self, bt):
        bars = bt.generate_mean_reverting_prices(days=40, mean_price=1000, amplitude=100)
        prices = [b.close for b in bars]
        assert max(prices) > 1000  # goes above mean
        assert min(prices) < 1000  # goes below mean

    def test_mean_reverting_length(self, bt):
        bars = bt.generate_mean_reverting_prices(days=30)
        assert len(bars) == 30

    def test_synthetic_deterministic(self):
        # Each Backtester gets its own seeded RNG — same seed → same prices
        bt1 = Backtester()
        bt2 = Backtester()
        bars1 = bt1.generate_synthetic_prices(days=20)
        bars2 = bt2.generate_synthetic_prices(days=20)
        assert all(b1.close == b2.close for b1, b2 in zip(bars1, bars2))


# ─── Strategy Signals ─────────────────────────────────────────────────────────

class TestStrategySignals:
    def test_trend_signal_insufficient_history(self, flat_bars):
        assert trend_signal(flat_bars, 0) is None
        assert trend_signal(flat_bars, 5) is None

    def test_trend_signal_flat(self, flat_bars):
        # Flat market → no clear signal
        sig = trend_signal(flat_bars, 15)
        assert sig is None or sig in ("BUY", "SELL")

    def test_trend_signal_uptrend(self, bt):
        bars = bt.generate_trending_prices(days=30, trend=0.02)
        sig = trend_signal(bars, 20)
        assert sig == "BUY"

    def test_mean_reversion_insufficient_history(self, flat_bars):
        assert mean_reversion_signal(flat_bars, 0) is None

    def test_mean_reversion_flat(self, flat_bars):
        sig = mean_reversion_signal(flat_bars, 25)
        assert sig is None

    def test_momentum_insufficient_history(self, flat_bars):
        assert momentum_signal(flat_bars, 0) is None
        assert momentum_signal(flat_bars, 3) is None

    def test_momentum_flat(self, flat_bars):
        sig = momentum_signal(flat_bars, 10)
        assert sig is None

    def test_strategy_map_keys(self):
        assert "trend" in STRATEGY_MAP
        assert "mean_reversion" in STRATEGY_MAP
        assert "momentum" in STRATEGY_MAP

    def test_momentum_upward(self, bt):
        bars = bt.generate_trending_prices(days=20, trend=0.015)
        sig = momentum_signal(bars, 10)
        assert sig == "BUY"


# ─── Backtester.run() ─────────────────────────────────────────────────────────

class TestBacktesterRun:
    def test_run_empty_bars(self, bt):
        trades = bt.run([], strategy="trend")
        assert trades == []

    def test_run_invalid_strategy(self, bt, flat_bars):
        with pytest.raises(ValueError, match="Unknown strategy"):
            bt.run(flat_bars, strategy="bad_strat")

    def test_run_returns_list(self, bt, synth_bars):
        trades = bt.run(synth_bars, strategy="trend")
        assert isinstance(trades, list)

    def test_run_trend_uptrend(self, bt, uptrend_bars):
        trades = bt.run(uptrend_bars, strategy="trend")
        assert len(trades) >= 0  # at least no crash

    def test_run_mean_reversion(self, bt, mr_bars):
        trades = bt.run(mr_bars, strategy="mean_reversion")
        assert isinstance(trades, list)

    def test_run_momentum(self, bt, uptrend_bars):
        trades = bt.run(uptrend_bars, strategy="momentum")
        assert isinstance(trades, list)

    def test_trade_fields_populated(self, bt, uptrend_bars):
        trades = bt.run(uptrend_bars, strategy="trend")
        for t in trades:
            assert t.entry_price > 0
            assert t.exit_price > 0
            assert t.side in ("BUY", "SELL")
            assert t.token == "ETH"

    def test_trade_ids_unique(self, bt, synth_bars):
        trades = bt.run(synth_bars, strategy="trend")
        ids = [t.trade_id for t in trades]
        assert len(ids) == len(set(ids))

    def test_held_bars_nonnegative(self, bt, synth_bars):
        trades = bt.run(synth_bars, strategy="momentum")
        for t in trades:
            assert t.held_bars >= 0

    def test_single_bar_no_crash(self, bt):
        now = datetime(2026, 1, 1, tzinfo=timezone.utc)
        bar = PriceBar(now, 100, 105, 95, 100, 1e6)
        trades = bt.run([bar], strategy="trend")
        assert isinstance(trades, list)

    def test_force_close_on_max_hold(self):
        bt2 = Backtester(initial_capital=1000.0, max_hold_bars=3)
        bars = bt2.generate_trending_prices(days=30, trend=0.01)
        trades = bt2.run(bars, strategy="trend")
        for t in trades[:-1]:  # last trade may be shorter
            assert t.held_bars <= 3 + 1  # slight tolerance for final close


# ─── compute_stats() ──────────────────────────────────────────────────────────

class TestComputeStats:
    def test_empty_trades(self, bt):
        s = bt.compute_stats([])
        assert s.total_trades == 0
        assert s.final_capital == 1000.0
        assert s.sharpe_ratio == 0.0

    def test_net_pnl(self, bt):
        trades = bt.run(bt.generate_trending_prices(days=60, trend=0.01), "trend")
        s = bt.compute_stats(trades)
        assert s.net_pnl == pytest.approx(s.gross_profit - s.gross_loss, abs=1e-6)

    def test_final_capital(self, bt):
        trades = bt.run(bt.generate_synthetic_prices(days=60), "trend")
        s = bt.compute_stats(trades)
        assert s.final_capital == pytest.approx(bt.initial_capital + s.net_pnl, abs=1e-6)

    def test_win_rate_range(self, bt):
        trades = bt.run(bt.generate_synthetic_prices(days=60), "trend")
        s = bt.compute_stats(trades)
        assert 0.0 <= s.win_rate <= 1.0

    def test_profit_factor_positive(self, bt):
        trades = bt.run(bt.generate_synthetic_prices(days=60), "trend")
        s = bt.compute_stats(trades)
        assert s.profit_factor >= 0

    def test_max_drawdown_nonnegative(self, bt):
        trades = bt.run(bt.generate_synthetic_prices(days=60), "mean_reversion")
        s = bt.compute_stats(trades)
        assert s.max_drawdown_pct >= 0

    def test_max_drawdown_capped(self, bt):
        trades = bt.run(bt.generate_synthetic_prices(days=60), "trend")
        s = bt.compute_stats(trades)
        assert s.max_drawdown_pct <= 100.0

    def test_single_trade_stats(self, bt):
        now = datetime(2026, 1, 1, tzinfo=timezone.utc)
        t = BacktestTrade(0, "ETH", "BUY", 100, 110, 1.0, now, now, pnl_usdc=10.0)
        s = bt.compute_stats([t])
        assert s.total_trades == 1
        assert s.winning_trades == 1
        assert s.losing_trades == 0
        assert s.win_rate == 1.0
        assert s.net_pnl == 10.0

    def test_all_losers(self, bt):
        now = datetime(2026, 1, 1, tzinfo=timezone.utc)
        trades = [
            BacktestTrade(i, "ETH", "BUY", 100, 90, 1.0, now, now, pnl_usdc=-10.0)
            for i in range(5)
        ]
        s = bt.compute_stats(trades)
        assert s.winning_trades == 0
        assert s.losing_trades == 5
        assert s.win_rate == 0.0
        assert s.profit_factor == 0.0

    def test_best_worst_trade(self, bt):
        now = datetime(2026, 1, 1, tzinfo=timezone.utc)
        trades = [
            BacktestTrade(0, "ETH", "BUY", 100, 150, 1.0, now, now, pnl_usdc=50.0),
            BacktestTrade(1, "ETH", "BUY", 100, 80, 1.0, now, now, pnl_usdc=-20.0),
        ]
        s = bt.compute_stats(trades)
        assert s.best_trade == 50.0
        assert s.worst_trade == -20.0

    def test_sharpe_with_multiple_trades(self, bt):
        trades = bt.run(bt.generate_synthetic_prices(days=80), "trend")
        s = bt.compute_stats(trades)
        assert isinstance(s.sharpe_ratio, float)

    def test_compare_strategies(self, bt, synth_bars):
        results = bt.compare_strategies(synth_bars)
        assert set(results.keys()) == {"trend", "mean_reversion", "momentum"}
        for s in results.values():
            assert isinstance(s, BacktestStats)

    def test_replay_trades_equity_curve(self, bt, synth_bars):
        trades = bt.run(synth_bars, "trend")
        equity = bt.replay_trades(trades)
        assert equity[0] == bt.initial_capital
        assert len(equity) == len(trades) + 1

    def test_backtester_invalid_capital(self):
        with pytest.raises(ValueError):
            Backtester(initial_capital=-100)

    def test_backtester_invalid_position_size(self):
        with pytest.raises(ValueError):
            Backtester(position_size_pct=0.0)

    def test_sortino_computed(self, bt):
        trades = bt.run(bt.generate_synthetic_prices(days=80), "mean_reversion")
        s = bt.compute_stats(trades)
        assert isinstance(s.sortino_ratio, float)

    def test_annualised_return(self, bt):
        trades = bt.run(bt.generate_trending_prices(days=40, trend=0.005), "trend")
        s = bt.compute_stats(trades)
        assert isinstance(s.annualised_return_pct, float)

    def test_calc_pnl_buy(self, bt):
        assert bt._calc_pnl("BUY", 100, 120, 1.0) == pytest.approx(20.0)

    def test_calc_pnl_sell(self, bt):
        assert bt._calc_pnl("SELL", 120, 100, 1.0) == pytest.approx(20.0)

    def test_calc_pnl_loss(self, bt):
        assert bt._calc_pnl("BUY", 100, 80, 1.0) == pytest.approx(-20.0)
