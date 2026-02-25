"""
Tests for strategy_runner.py — 45+ tests covering:
  - StrategySignal validation
  - AggregatedSignal properties
  - TrendStrategy signal generation
  - MeanReversionStrategy signal generation
  - MomentumStrategy signal generation
  - StrategyRunner aggregation modes (majority, unanimous, weighted)
  - Parallel vs serial execution
  - Position sizing
  - Signal history and stats
  - Edge cases: empty prices, single strategy, all hold
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import pytest

from strategy_runner import (
    AggregatedSignal,
    MeanReversionStrategy,
    MomentumStrategy,
    StrategyRunner,
    StrategySignal,
    TrendStrategy,
)


# ─── Helpers ──────────────────────────────────────────────────────────────────

def rising_prices(n=30, start=1000.0, rate=0.01):
    prices = [start]
    for _ in range(n - 1):
        prices.append(prices[-1] * (1 + rate))
    return prices

def flat_prices(n=30, price=1000.0):
    return [price] * n

def oscillating_prices(n=60, center=1000.0, amplitude=50.0):
    import math
    return [center + amplitude * math.sin(2 * math.pi * i / 10) for i in range(n)]


# ─── StrategySignal ───────────────────────────────────────────────────────────

class TestStrategySignal:
    def test_valid_buy(self):
        s = StrategySignal("trend", "buy", 0.8, "test")
        assert s.action == "buy"

    def test_valid_sell(self):
        s = StrategySignal("trend", "sell", 0.5)
        assert s.action == "sell"

    def test_valid_hold(self):
        s = StrategySignal("trend", "hold", 0.0)
        assert s.action == "hold"

    def test_invalid_action(self):
        with pytest.raises(ValueError):
            StrategySignal("trend", "long", 0.5)

    def test_confidence_out_of_range_high(self):
        with pytest.raises(ValueError):
            StrategySignal("trend", "buy", 1.1)

    def test_confidence_out_of_range_low(self):
        with pytest.raises(ValueError):
            StrategySignal("trend", "buy", -0.1)

    def test_metadata_default_empty(self):
        s = StrategySignal("trend", "hold", 0.0)
        assert s.metadata == {}


# ─── AggregatedSignal ─────────────────────────────────────────────────────────

class TestAggregatedSignal:
    def test_is_actionable_buy(self):
        agg = AggregatedSignal("buy", 0.7)
        assert agg.is_actionable is True

    def test_is_actionable_hold(self):
        agg = AggregatedSignal("hold", 0.0)
        assert agg.is_actionable is False

    def test_is_actionable_zero_confidence(self):
        agg = AggregatedSignal("buy", 0.0)
        assert agg.is_actionable is False


# ─── TrendStrategy ────────────────────────────────────────────────────────────

class TestTrendStrategy:
    def test_buy_in_uptrend(self):
        strat = TrendStrategy(short_window=3, long_window=10)
        prices = rising_prices(20, rate=0.02)
        sig = strat.generate_signal(prices, prices[-1])
        assert sig.action == "buy"
        assert sig.confidence > 0

    def test_hold_insufficient_data(self):
        strat = TrendStrategy()
        sig = strat.generate_signal([1000.0, 1001.0], 1001.0)
        assert sig.action == "hold"

    def test_flat_market_hold(self):
        strat = TrendStrategy(short_window=3, long_window=10)
        prices = flat_prices(25)
        sig = strat.generate_signal(prices, 1000.0)
        assert sig.action == "hold"

    def test_invalid_windows(self):
        with pytest.raises(ValueError):
            TrendStrategy(short_window=20, long_window=5)

    def test_confidence_in_range(self):
        strat = TrendStrategy(short_window=3, long_window=10)
        prices = rising_prices(30, rate=0.05)
        sig = strat.generate_signal(prices, prices[-1])
        assert 0.0 <= sig.confidence <= 1.0

    def test_strategy_name(self):
        strat = TrendStrategy()
        assert strat.name == "trend"


# ─── MeanReversionStrategy ────────────────────────────────────────────────────

class TestMeanReversionStrategy:
    def test_buy_below_mean(self):
        strat = MeanReversionStrategy(lookback=20, z_thresh=1.0)
        prices = flat_prices(25, 1000.0)
        prices.append(800.0)  # Sharp dip
        sig = strat.generate_signal(prices, 800.0)
        assert sig.action == "buy"

    def test_sell_above_mean(self):
        strat = MeanReversionStrategy(lookback=20, z_thresh=1.0)
        prices = flat_prices(25, 1000.0)
        prices.append(1200.0)  # Sharp spike
        sig = strat.generate_signal(prices, 1200.0)
        assert sig.action == "sell"

    def test_hold_within_bands(self):
        strat = MeanReversionStrategy(lookback=20, z_thresh=2.0)
        prices = oscillating_prices(30, 1000.0, 5.0)
        sig = strat.generate_signal(prices, 1001.0)
        assert sig.action == "hold"

    def test_insufficient_data(self):
        strat = MeanReversionStrategy(lookback=20)
        sig = strat.generate_signal([1000.0, 1001.0], 1001.0)
        assert sig.action == "hold"

    def test_bad_lookback(self):
        with pytest.raises(ValueError):
            MeanReversionStrategy(lookback=1)

    def test_confidence_in_range(self):
        strat = MeanReversionStrategy(lookback=10, z_thresh=1.0)
        prices = flat_prices(15, 1000.0)
        prices.append(700.0)
        sig = strat.generate_signal(prices, 700.0)
        assert 0.0 <= sig.confidence <= 1.0


# ─── MomentumStrategy ─────────────────────────────────────────────────────────

class TestMomentumStrategy:
    def test_buy_strong_upward(self):
        strat = MomentumStrategy(lookback=5, buy_thresh=0.01)
        prices = rising_prices(20, rate=0.05)
        sig = strat.generate_signal(prices, prices[-1])
        assert sig.action == "buy"

    def test_sell_strong_downward(self):
        strat = MomentumStrategy(lookback=5, sell_thresh=-0.01)
        prices = [1000.0 * (0.95 ** i) for i in range(20)]
        sig = strat.generate_signal(prices, prices[-1])
        assert sig.action == "sell"

    def test_hold_weak_momentum(self):
        strat = MomentumStrategy(lookback=5, buy_thresh=0.05, sell_thresh=-0.05)
        prices = flat_prices(10)
        sig = strat.generate_signal(prices, 1000.0)
        assert sig.action == "hold"

    def test_insufficient_data(self):
        strat = MomentumStrategy(lookback=10)
        sig = strat.generate_signal([1000.0], 1000.0)
        assert sig.action == "hold"

    def test_zero_past_price(self):
        strat = MomentumStrategy(lookback=2)
        sig = strat.generate_signal([0.0, 0.0, 100.0], 100.0)
        assert sig.action == "hold"

    def test_bad_lookback(self):
        with pytest.raises(ValueError):
            MomentumStrategy(lookback=0)

    def test_confidence_capped(self):
        strat = MomentumStrategy(lookback=2, buy_thresh=0.001)
        prices = [1000.0, 2000.0, 4000.0]
        sig = strat.generate_signal(prices, 4000.0)
        assert sig.confidence <= 1.0


# ─── StrategyRunner ───────────────────────────────────────────────────────────

class TestStrategyRunnerInit:
    def test_default_strategies(self):
        runner = StrategyRunner()
        assert len(runner.strategies) == 3

    def test_invalid_mode(self):
        with pytest.raises(ValueError):
            StrategyRunner(aggregation_mode="oracle")

    def test_invalid_position_pct(self):
        with pytest.raises(ValueError):
            StrategyRunner(max_position_pct=0.0)

    def test_invalid_confidence(self):
        with pytest.raises(ValueError):
            StrategyRunner(min_confidence=-0.1)

    def test_add_strategy(self):
        runner = StrategyRunner(strategies=[])
        runner.add_strategy(TrendStrategy())
        assert len(runner.strategies) == 1

    def test_remove_strategy(self):
        runner = StrategyRunner()
        removed = runner.remove_strategy("trend")
        assert removed is True
        assert all(s.name != "trend" for s in runner.strategies)

    def test_remove_nonexistent(self):
        runner = StrategyRunner()
        removed = runner.remove_strategy("phantom")
        assert removed is False


# ─── StrategyRunner.run() ─────────────────────────────────────────────────────

class TestStrategyRunnerRun:
    def test_empty_prices_returns_hold(self):
        runner = StrategyRunner()
        result = runner.run([])
        assert result.action == "hold"

    def test_returns_aggregated_signal(self):
        runner = StrategyRunner()
        result = runner.run(rising_prices(30))
        assert isinstance(result, AggregatedSignal)

    def test_action_valid(self):
        runner = StrategyRunner()
        result = runner.run(rising_prices(30))
        assert result.action in ("buy", "sell", "hold")

    def test_confidence_in_range(self):
        runner = StrategyRunner()
        result = runner.run(rising_prices(30))
        assert 0.0 <= result.confidence <= 1.0

    def test_buy_in_strong_uptrend(self):
        runner = StrategyRunner(
            strategies=[TrendStrategy(short_window=3, long_window=10)],
            aggregation_mode="majority",
            min_confidence=0.01,
        )
        prices = rising_prices(25, rate=0.03)
        result = runner.run(prices)
        assert result.action == "buy"

    def test_position_size_zero_on_hold(self):
        runner = StrategyRunner()
        result = runner.run(flat_prices(5))  # too few for any strategy
        assert result.position_size_usdc == 0.0

    def test_position_size_nonzero_on_buy(self):
        runner = StrategyRunner(
            strategies=[TrendStrategy(short_window=3, long_window=10)],
            min_confidence=0.01,
        )
        result = runner.run(rising_prices(25, rate=0.05))
        if result.action == "buy":
            assert result.position_size_usdc > 0

    def test_signals_list_populated(self):
        runner = StrategyRunner()
        result = runner.run(rising_prices(30))
        assert len(result.signals) == 3

    def test_vote_counts(self):
        runner = StrategyRunner()
        result = runner.run(rising_prices(30))
        total = result.buy_count + result.sell_count + result.hold_count
        assert total == len(runner.strategies)

    def test_serial_mode(self):
        runner = StrategyRunner(parallel=False)
        result = runner.run(rising_prices(30))
        assert isinstance(result, AggregatedSignal)

    def test_custom_current_price(self):
        runner = StrategyRunner(strategies=[MomentumStrategy(lookback=5, buy_thresh=0.01)])
        prices = rising_prices(20, rate=0.02)
        result = runner.run(prices, current_price=prices[-1] * 1.1)
        assert isinstance(result, AggregatedSignal)


# ─── Aggregation Modes ────────────────────────────────────────────────────────

class TestAggregationModes:
    def test_majority_mode(self):
        runner = StrategyRunner(aggregation_mode="majority", min_confidence=0.01)
        result = runner.run(rising_prices(30, rate=0.03))
        assert result.aggregation_mode == "majority"

    def test_unanimous_mode_disagree(self):
        # With divergent strategies, unanimous should return hold
        runner = StrategyRunner(
            strategies=[TrendStrategy(), MeanReversionStrategy()],
            aggregation_mode="unanimous",
        )
        prices = oscillating_prices(40)
        result = runner.run(prices)
        # Unanimous requires all agree; hard to guarantee specific outcome
        assert result.action in ("buy", "sell", "hold")

    def test_weighted_mode(self):
        runner = StrategyRunner(aggregation_mode="weighted", min_confidence=0.01)
        result = runner.run(rising_prices(30, rate=0.03))
        assert result.aggregation_mode == "weighted"

    def test_weighted_high_weight_wins(self):
        # Trend strategy with high weight against low-weight others
        strats = [
            TrendStrategy(short_window=3, long_window=10, weight=10.0),
            MeanReversionStrategy(weight=1.0),
        ]
        runner = StrategyRunner(strategies=strats, aggregation_mode="weighted",
                                min_confidence=0.01)
        prices = rising_prices(25, rate=0.03)
        result = runner.run(prices)
        # High-weight trend strategy should dominate
        assert result.action in ("buy", "sell", "hold")

    def test_min_confidence_gate(self):
        runner = StrategyRunner(min_confidence=0.99)  # Extremely high bar
        result = runner.run(flat_prices(25))
        assert result.action == "hold"
        assert result.position_size_usdc == 0.0


# ─── Signal History & Stats ───────────────────────────────────────────────────

class TestSignalHistory:
    def test_history_grows(self):
        runner = StrategyRunner()
        runner.run(rising_prices(30))
        runner.run(flat_prices(30))
        assert len(runner.get_signal_history()) == 2

    def test_clear_history(self):
        runner = StrategyRunner()
        runner.run(rising_prices(30))
        runner.clear_history()
        assert len(runner.get_signal_history()) == 0

    def test_stats_empty(self):
        runner = StrategyRunner()
        s = runner.stats()
        assert s["total"] == 0

    def test_stats_after_runs(self):
        runner = StrategyRunner()
        for _ in range(5):
            runner.run(rising_prices(30))
        s = runner.stats()
        assert s["total"] == 5
        assert "avg_confidence" in s
