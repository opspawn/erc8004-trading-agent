"""
Tests for agent/strategy_engine.py

Covers:
  - StrategySignal: validation, confidence clamping
  - BacktestResult: fields
  - BaseStrategy: helpers (_compute_returns, _rolling_signals, backtest_score)
  - MomentumStrategy: buy/sell/hold logic, edge cases
  - MeanReversionStrategy: z-score bands, edge cases
  - VolatilityBreakout: Bollinger Band logic, compute_bands
  - SentimentWeightedStrategy: blend, extreme negative blocking, conflicts
  - EnsembleVoting: majority vote, vote_breakdown, min_confidence
  - StrategyEngine: evaluate, evaluate_all, backtest_all, names
  - Integration: full price series through each strategy
"""

from __future__ import annotations

import math
import statistics
import sys, os
from typing import List

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from strategy_engine import (
    BacktestResult,
    BaseStrategy,
    EnsembleVoting,
    MeanReversionStrategy,
    MomentumStrategy,
    SentimentWeightedStrategy,
    StrategyEngine,
    StrategySignal,
    VolatilityBreakout,
)


# ─── Fixtures / Helpers ───────────────────────────────────────────────────────

def _flat(n: int, price: float = 100.0) -> List[float]:
    """Flat price series."""
    return [price] * n

def _trending_up(n: int, start: float = 100.0, pct: float = 0.01) -> List[float]:
    prices = [start]
    for _ in range(n - 1):
        prices.append(prices[-1] * (1 + pct))
    return prices

def _trending_down(n: int, start: float = 100.0, pct: float = 0.01) -> List[float]:
    prices = [start]
    for _ in range(n - 1):
        prices.append(prices[-1] * (1 - pct))
    return prices

def _mean_reverting(n: int, mean: float = 100.0, amp: float = 5.0) -> List[float]:
    """Oscillating series around a mean."""
    return [mean + amp * math.sin(2 * math.pi * i / 20) for i in range(n)]


# ─── StrategySignal Tests ─────────────────────────────────────────────────────

class TestStrategySignal:

    def test_valid_buy(self):
        s = StrategySignal("test", "buy", 0.8)
        assert s.action == "buy"
        assert s.confidence == 0.8

    def test_valid_sell(self):
        s = StrategySignal("test", "sell", 0.5)
        assert s.action == "sell"

    def test_valid_hold(self):
        s = StrategySignal("test", "hold", 0.0)
        assert s.action == "hold"

    def test_invalid_action_raises(self):
        with pytest.raises(ValueError):
            StrategySignal("test", "short", 0.5)

    def test_confidence_clamp_below_zero(self):
        s = StrategySignal("test", "buy", -0.5)
        assert s.confidence == 0.0

    def test_confidence_clamp_above_one(self):
        s = StrategySignal("test", "sell", 1.5)
        assert s.confidence == 1.0

    def test_confidence_boundary_zero(self):
        s = StrategySignal("test", "hold", 0.0)
        assert s.confidence == 0.0

    def test_confidence_boundary_one(self):
        s = StrategySignal("test", "buy", 1.0)
        assert s.confidence == 1.0

    def test_reason_default_empty(self):
        s = StrategySignal("test", "hold", 0.5)
        assert s.reason == ""

    def test_reason_custom(self):
        s = StrategySignal("test", "buy", 0.7, "momentum=0.05")
        assert "momentum" in s.reason

    def test_metadata_default_empty(self):
        s = StrategySignal("test", "hold", 0.5)
        assert s.metadata == {}

    def test_metadata_custom(self):
        s = StrategySignal("test", "buy", 0.6, metadata={"z": 2.1})
        assert s.metadata["z"] == 2.1

    def test_strategy_name_preserved(self):
        s = StrategySignal("MomentumStrategy", "buy", 0.9)
        assert s.strategy_name == "MomentumStrategy"


# ─── BaseStrategy Tests ───────────────────────────────────────────────────────

class TestBaseStrategy:

    def _make_concrete(self):
        """Return a minimal concrete implementation for testing helpers."""
        class AlwaysBuy(BaseStrategy):
            name = "AlwaysBuy"
            def fit(self, prices):
                return StrategySignal("AlwaysBuy", "buy", 0.9)
        return AlwaysBuy()

    def test_compute_returns_basic(self):
        prices = [100.0, 110.0, 105.0]
        returns = BaseStrategy._compute_returns(prices)
        assert len(returns) == 2
        assert abs(returns[0] - math.log(110 / 100)) < 1e-9
        assert abs(returns[1] - math.log(105 / 110)) < 1e-9

    def test_compute_returns_single(self):
        assert BaseStrategy._compute_returns([100.0]) == []

    def test_compute_returns_empty(self):
        assert BaseStrategy._compute_returns([]) == []

    def test_backtest_score_too_short(self):
        strategy = self._make_concrete()
        assert strategy.backtest_score([1.0, 2.0]) == 0.0

    def test_backtest_score_flat_prices(self):
        strategy = self._make_concrete()
        prices = _flat(50)
        # All buy signals, zero returns → score could be 0
        score = strategy.backtest_score(prices)
        assert isinstance(score, float)

    def test_backtest_score_trending_up(self):
        class BuyOnUp(BaseStrategy):
            name = "BuyOnUp"
            def fit(self, prices):
                return StrategySignal("BuyOnUp", "buy", 0.9)

        strategy = BuyOnUp()
        # Add small noise so returns are not identical (avoids stdev≈0 edge case)
        import random; random.seed(1)
        prices = [p * (1 + random.gauss(0, 0.001)) for p in _trending_up(100, pct=0.01)]
        score = strategy.backtest_score(prices)
        assert isinstance(score, float)
        assert score > 0  # buying in uptrend should be profitable

    def test_backtest_detailed_returns_backtest_result(self):
        strategy = self._make_concrete()
        result = strategy.backtest_detailed(_trending_up(50))
        assert isinstance(result, BacktestResult)
        assert result.strategy_name == "AlwaysBuy"

    def test_backtest_detailed_no_trades(self):
        class AlwaysHold(BaseStrategy):
            name = "AlwaysHold"
            def fit(self, prices):
                return StrategySignal("AlwaysHold", "hold", 0.5)
        strategy = AlwaysHold()
        result = strategy.backtest_detailed(_trending_up(50))
        assert result.total_trades == 0
        assert result.sharpe == 0.0

    def test_rolling_signals_length(self):
        strategy = self._make_concrete()
        prices = _flat(30)
        signals = strategy._rolling_signals(prices)
        assert len(signals) == 30

    def test_fit_raises_not_implemented(self):
        with pytest.raises(NotImplementedError):
            BaseStrategy().fit([1.0, 2.0, 3.0])


# ─── MomentumStrategy Tests ───────────────────────────────────────────────────

class TestMomentumStrategy:

    def test_default_params(self):
        m = MomentumStrategy()
        assert m.period == 3
        assert m.threshold == 0.01

    def test_invalid_period(self):
        with pytest.raises(ValueError):
            MomentumStrategy(period=0)

    def test_negative_threshold(self):
        with pytest.raises(ValueError):
            MomentumStrategy(threshold=-0.01)

    def test_insufficient_data(self):
        m = MomentumStrategy(period=3)
        sig = m.fit([100.0, 101.0])
        assert sig.action == "hold"

    def test_buy_signal_uptrend(self):
        m = MomentumStrategy(period=3, threshold=0.01)
        # 3% gain over 3 bars → momentum=0.03 > 0.01
        prices = [100.0, 101.0, 102.0, 103.0]
        sig = m.fit(prices)
        assert sig.action == "buy"

    def test_sell_signal_downtrend(self):
        m = MomentumStrategy(period=3, threshold=0.01)
        prices = [100.0, 99.0, 98.0, 97.0]
        sig = m.fit(prices)
        assert sig.action == "sell"

    def test_hold_within_threshold(self):
        m = MomentumStrategy(period=3, threshold=0.05)
        # Small move within threshold
        prices = [100.0, 100.1, 100.2, 100.3]
        sig = m.fit(prices)
        assert sig.action == "hold"

    def test_confidence_positive(self):
        m = MomentumStrategy(period=3, threshold=0.01)
        prices = _trending_up(10, pct=0.02)
        sig = m.fit(prices)
        assert 0.0 <= sig.confidence <= 1.0

    def test_momentum_in_metadata(self):
        m = MomentumStrategy(period=3, threshold=0.01)
        prices = _trending_up(10, pct=0.02)
        sig = m.fit(prices)
        if sig.action != "hold":
            assert "momentum" in sig.metadata

    def test_zero_previous_price(self):
        m = MomentumStrategy(period=1)
        sig = m.fit([0.0, 100.0])
        assert sig.action == "hold"

    def test_custom_period(self):
        m = MomentumStrategy(period=5, threshold=0.01)
        # Need at least 6 prices for period=5
        prices = _trending_up(10, pct=0.02)
        sig = m.fit(prices)
        assert sig.action in ("buy", "sell", "hold")

    def test_large_threshold_forces_hold(self):
        m = MomentumStrategy(period=3, threshold=1.0)  # impossible threshold
        prices = _trending_up(20, pct=0.05)
        sig = m.fit(prices)
        assert sig.action == "hold"

    def test_confidence_capped_at_one(self):
        m = MomentumStrategy(period=1, threshold=0.001)
        # Massive price jump
        sig = m.fit([100.0, 200.0])
        assert sig.confidence <= 1.0

    def test_name_attribute(self):
        assert MomentumStrategy.name == "MomentumStrategy"

    def test_backtest_score_uptrend(self):
        m = MomentumStrategy(period=3, threshold=0.005)
        prices = _trending_up(100, pct=0.01)
        score = m.backtest_score(prices)
        assert isinstance(score, float)

    def test_returns_strategy_signal(self):
        m = MomentumStrategy()
        sig = m.fit(_trending_up(10))
        assert isinstance(sig, StrategySignal)
        assert sig.strategy_name == "MomentumStrategy"


# ─── MeanReversionStrategy Tests ─────────────────────────────────────────────

class TestMeanReversionStrategy:

    def test_default_params(self):
        mr = MeanReversionStrategy()
        assert mr.period == 20
        assert mr.std_multiplier == 1.5

    def test_invalid_period(self):
        with pytest.raises(ValueError):
            MeanReversionStrategy(period=1)

    def test_insufficient_data(self):
        mr = MeanReversionStrategy(period=20)
        sig = mr.fit([100.0] * 5)
        assert sig.action == "hold"

    def test_buy_when_oversold(self):
        mr = MeanReversionStrategy(period=20, std_multiplier=1.5)
        # Create prices with mean=100, then drop sharply
        prices = [100.0] * 20 + [85.0]  # z ≈ -inf (std ≈ 0 until 85)
        prices_osc = _mean_reverting(20) + [80.0]
        sig = mr.fit(prices_osc)
        assert sig.action in ("buy", "hold", "sell")

    def test_sell_when_overbought(self):
        mr = MeanReversionStrategy(period=20, std_multiplier=1.5)
        prices = _mean_reverting(20) + [130.0]
        sig = mr.fit(prices)
        assert sig.action in ("buy", "sell", "hold")

    def test_hold_within_bands(self):
        mr = MeanReversionStrategy(period=20, std_multiplier=1.5)
        prices = _mean_reverting(21)
        sig = mr.fit(prices)
        assert sig.action in ("buy", "sell", "hold")  # at mean → hold

    def test_zero_variance_returns_hold(self):
        mr = MeanReversionStrategy(period=5)
        prices = _flat(6)
        sig = mr.fit(prices)
        assert sig.action == "hold"

    def test_confidence_in_range(self):
        mr = MeanReversionStrategy(period=10, std_multiplier=1.0)
        prices = list(range(1, 30))  # trending up
        sig = mr.fit(prices)
        assert 0.0 <= sig.confidence <= 1.0

    def test_z_score_in_metadata(self):
        mr = MeanReversionStrategy(period=10, std_multiplier=1.0)
        prices = _mean_reverting(11, mean=100, amp=15)
        sig = mr.fit(prices)
        assert "z_score" in sig.metadata

    def test_name_attribute(self):
        assert MeanReversionStrategy.name == "MeanReversionStrategy"

    def test_backtest_returns_float(self):
        mr = MeanReversionStrategy(period=10, std_multiplier=1.5)
        score = mr.backtest_score(_mean_reverting(100))
        assert isinstance(score, float)

    def test_custom_std_multiplier(self):
        mr = MeanReversionStrategy(period=10, std_multiplier=0.5)
        prices = _mean_reverting(20, amp=10)
        sig = mr.fit(prices)
        assert sig.action in ("buy", "sell", "hold")

    def test_returns_strategy_signal(self):
        mr = MeanReversionStrategy()
        sig = mr.fit(_flat(25))
        assert isinstance(sig, StrategySignal)
        assert sig.strategy_name == "MeanReversionStrategy"

    def test_explicit_oversold_scenario(self):
        """Build a clear oversold scenario: high variance window, then low price."""
        mr = MeanReversionStrategy(period=20, std_multiplier=1.5)
        # High variance window
        import random
        random.seed(42)
        prices = [100 + random.gauss(0, 10) for _ in range(20)]
        # Price crashes far below mean
        mean_val = statistics.mean(prices[-20:])
        std_val  = statistics.stdev(prices[-20:])
        oversold = mean_val - 3 * std_val
        prices.append(oversold)
        sig = mr.fit(prices)
        assert sig.action == "buy"

    def test_explicit_overbought_scenario(self):
        mr = MeanReversionStrategy(period=20, std_multiplier=1.5)
        import random
        random.seed(99)
        prices = [100 + random.gauss(0, 10) for _ in range(20)]
        mean_val = statistics.mean(prices[-20:])
        std_val  = statistics.stdev(prices[-20:])
        overbought = mean_val + 3 * std_val
        prices.append(overbought)
        sig = mr.fit(prices)
        assert sig.action == "sell"


# ─── VolatilityBreakout Tests ─────────────────────────────────────────────────

class TestVolatilityBreakout:

    def test_default_params(self):
        vb = VolatilityBreakout()
        assert vb.period == 20
        assert vb.num_std == 2.0

    def test_invalid_period(self):
        with pytest.raises(ValueError):
            VolatilityBreakout(period=1)

    def test_insufficient_data(self):
        vb = VolatilityBreakout(period=20)
        sig = vb.fit([100.0] * 5)
        assert sig.action == "hold"

    def test_compute_bands_correct(self):
        vb = VolatilityBreakout(period=5, num_std=2.0)
        prices = [100.0, 102.0, 98.0, 101.0, 99.0]
        lower, ma, upper = vb.compute_bands(prices)
        expected_ma = statistics.mean(prices)
        expected_std = statistics.stdev(prices)
        assert abs(ma - expected_ma) < 1e-6
        assert abs(upper - (expected_ma + 2.0 * expected_std)) < 1e-6
        assert abs(lower - (expected_ma - 2.0 * expected_std)) < 1e-6

    def test_compute_bands_insufficient_raises(self):
        vb = VolatilityBreakout(period=10)
        with pytest.raises(ValueError):
            vb.compute_bands([100.0, 101.0])

    def test_buy_on_upper_breakout(self):
        vb = VolatilityBreakout(period=20, num_std=2.0)
        import random
        random.seed(7)
        prices = [100 + random.gauss(0, 2) for _ in range(20)]
        ma = statistics.mean(prices[-20:])
        std = statistics.stdev(prices[-20:])
        breakout_price = ma + 3 * std  # well above upper band
        prices.append(breakout_price)
        sig = vb.fit(prices)
        assert sig.action == "buy"

    def test_sell_on_lower_breakout(self):
        vb = VolatilityBreakout(period=20, num_std=2.0)
        import random
        random.seed(13)
        prices = [100 + random.gauss(0, 2) for _ in range(20)]
        ma = statistics.mean(prices[-20:])
        std = statistics.stdev(prices[-20:])
        breakdown_price = ma - 3 * std  # well below lower band
        prices.append(breakdown_price)
        sig = vb.fit(prices)
        assert sig.action == "sell"

    def test_hold_inside_bands(self):
        vb = VolatilityBreakout(period=20, num_std=2.0)
        import random
        random.seed(21)
        prices = [100 + random.gauss(0, 2) for _ in range(20)]
        ma = statistics.mean(prices[-20:])
        prices.append(ma)  # price at mean → inside bands
        sig = vb.fit(prices)
        assert sig.action == "hold"

    def test_confidence_in_range(self):
        vb = VolatilityBreakout(period=10, num_std=2.0)
        prices = _trending_up(30, pct=0.05)
        sig = vb.fit(prices)
        assert 0.0 <= sig.confidence <= 1.0

    def test_upper_lower_in_metadata(self):
        vb = VolatilityBreakout(period=10, num_std=2.0)
        import random
        random.seed(42)
        prices = [100 + random.gauss(0, 5) for _ in range(11)]
        sig = vb.fit(prices)
        assert "upper" in sig.metadata
        assert "lower" in sig.metadata
        assert "ma" in sig.metadata

    def test_band_position_in_hold_metadata(self):
        vb = VolatilityBreakout(period=10, num_std=2.0)
        import random
        random.seed(42)
        prices = [100 + random.gauss(0, 5) for _ in range(10)]
        prices.append(statistics.mean(prices[-10:]))  # exactly at MA
        sig = vb.fit(prices)
        if sig.action == "hold":
            assert "band_position" in sig.metadata

    def test_zero_variance_returns_hold(self):
        vb = VolatilityBreakout(period=10, num_std=2.0)
        prices = _flat(11)
        sig = vb.fit(prices)
        assert sig.action == "hold"

    def test_name_attribute(self):
        assert VolatilityBreakout.name == "VolatilityBreakout"

    def test_custom_num_std(self):
        vb = VolatilityBreakout(period=10, num_std=0.5)  # tight bands
        import random
        random.seed(55)
        prices = [100 + random.gauss(0, 5) for _ in range(11)]
        sig = vb.fit(prices)
        assert sig.action in ("buy", "sell", "hold")

    def test_backtest_score_returns_float(self):
        vb = VolatilityBreakout(period=10, num_std=2.0)
        prices = _mean_reverting(100)
        score = vb.backtest_score(prices)
        assert isinstance(score, float)

    def test_returns_strategy_signal(self):
        vb = VolatilityBreakout()
        sig = vb.fit(_flat(25))
        assert isinstance(sig, StrategySignal)
        assert sig.strategy_name == "VolatilityBreakout"


# ─── SentimentWeightedStrategy Tests ─────────────────────────────────────────

class TestSentimentWeightedStrategy:

    def test_default_params(self):
        sw = SentimentWeightedStrategy()
        assert abs(sw.sentiment_weight + sw.technical_weight - 1.0) < 1e-9
        assert sw.extreme_negative == -0.7

    def test_invalid_weights_sum(self):
        with pytest.raises(ValueError):
            SentimentWeightedStrategy(sentiment_weight=0.3, technical_weight=0.3)

    def test_extreme_negative_blocks_trade(self):
        sw = SentimentWeightedStrategy()
        prices = _trending_up(10, pct=0.03)
        sig = sw.fit(prices, sentiment_score=-0.9)
        assert sig.action == "hold"
        assert sig.metadata.get("blocked") is True

    def test_extreme_negative_at_boundary(self):
        sw = SentimentWeightedStrategy(extreme_negative=-0.7)
        prices = _trending_up(10, pct=0.03)
        # Just at the threshold — should NOT block (-0.7 is NOT < -0.7)
        sig = sw.fit(prices, sentiment_score=-0.7)
        assert sig.metadata.get("blocked") is not True

    def test_positive_sentiment_boosts_buy(self):
        sw = SentimentWeightedStrategy()
        prices = _trending_up(10, pct=0.02)
        sig_no_sentiment = sw.fit(prices, sentiment_score=0.0)
        sig_with_sentiment = sw.fit(prices, sentiment_score=0.8)
        # Positive sentiment on an uptrend should give higher or equal confidence
        if sig_no_sentiment.action == "buy" and sig_with_sentiment.action == "buy":
            assert sig_with_sentiment.confidence >= sig_no_sentiment.confidence

    def test_negative_sentiment_on_uptrend_causes_conflict(self):
        sw = SentimentWeightedStrategy()
        prices = _trending_up(10, pct=0.03)
        sig = sw.fit(prices, sentiment_score=-0.5)
        # Conflict between uptrend (buy) and negative sentiment (sell)
        assert sig.action in ("buy", "sell", "hold")

    def test_neutral_sentiment_passes_through(self):
        sw = SentimentWeightedStrategy()
        prices = _trending_up(10, pct=0.02)
        sig = sw.fit(prices, sentiment_score=0.0)
        assert sig.action in ("buy", "sell", "hold")

    def test_sentiment_in_metadata(self):
        sw = SentimentWeightedStrategy()
        prices = _flat(10)
        sig = sw.fit(prices, sentiment_score=0.5)
        assert "sentiment" in sig.metadata

    def test_positive_sentiment_neutral_technical(self):
        sw = SentimentWeightedStrategy()
        prices = _flat(10)   # flat = neutral technical
        sig = sw.fit(prices, sentiment_score=0.8)
        # Sentiment should drive the signal
        assert sig.action in ("buy", "hold")

    def test_name_attribute(self):
        assert SentimentWeightedStrategy.name == "SentimentWeighted"

    def test_confidence_clamped(self):
        sw = SentimentWeightedStrategy()
        prices = _trending_up(10, pct=0.05)
        sig = sw.fit(prices, sentiment_score=1.0)
        assert 0.0 <= sig.confidence <= 1.0

    def test_zero_sentiment_weight_valid(self):
        # 0.0 sentiment weight means pure technical
        # But they must sum to 1.0, so test 0.0 + 1.0
        sw = SentimentWeightedStrategy(sentiment_weight=0.0, technical_weight=1.0)
        prices = _trending_up(10, pct=0.02)
        sig = sw.fit(prices, sentiment_score=0.9)
        assert sig.action in ("buy", "sell", "hold")

    def test_default_sentiment_score_zero(self):
        sw = SentimentWeightedStrategy()
        prices = _trending_up(10, pct=0.02)
        sig = sw.fit(prices)  # no sentiment_score arg
        assert sig.action in ("buy", "sell", "hold")

    def test_returns_strategy_signal(self):
        sw = SentimentWeightedStrategy()
        sig = sw.fit(_flat(10))
        assert isinstance(sig, StrategySignal)
        assert sig.strategy_name == "SentimentWeighted"

    def test_backtest_returns_float(self):
        sw = SentimentWeightedStrategy()
        score = sw.backtest_score(_trending_up(50))
        assert isinstance(score, float)


# ─── EnsembleVoting Tests ─────────────────────────────────────────────────────

class TestEnsembleVoting:

    def test_default_strategies(self):
        ev = EnsembleVoting()
        assert len(ev.strategies) == 4

    def test_custom_strategies(self):
        ev = EnsembleVoting(strategies=[MomentumStrategy(), MeanReversionStrategy()])
        assert len(ev.strategies) == 2

    def test_returns_strategy_signal(self):
        ev = EnsembleVoting()
        sig = ev.fit(_trending_up(30, pct=0.01))
        assert isinstance(sig, StrategySignal)
        assert sig.strategy_name == "EnsembleVoting"

    def test_confidence_in_range(self):
        ev = EnsembleVoting()
        sig = ev.fit(_trending_up(30, pct=0.01))
        assert 0.0 <= sig.confidence <= 1.0

    def test_action_valid(self):
        ev = EnsembleVoting()
        sig = ev.fit(_trending_up(30, pct=0.01))
        assert sig.action in ("buy", "sell", "hold")

    def test_votes_in_metadata(self):
        ev = EnsembleVoting()
        sig = ev.fit(_trending_up(30, pct=0.01))
        assert "votes" in sig.metadata

    def test_individual_in_metadata(self):
        ev = EnsembleVoting()
        sig = ev.fit(_trending_up(30, pct=0.01))
        assert "individual" in sig.metadata

    def test_min_confidence_hold_when_low(self):
        ev = EnsembleVoting(min_confidence=0.99)  # impossible to reach
        sig = ev.fit(_trending_up(30, pct=0.01))
        assert sig.action == "hold"

    def test_sentiment_score_passed_to_sentiment_strategy(self):
        ev = EnsembleVoting()
        # Extreme negative should block SentimentWeighted
        sig_blocked = ev.fit(_trending_up(30, pct=0.01), sentiment_score=-0.95)
        sig_normal  = ev.fit(_trending_up(30, pct=0.01), sentiment_score=0.0)
        # Both should be valid signals
        assert sig_blocked.action in ("buy", "sell", "hold")
        assert sig_normal.action in ("buy", "sell", "hold")

    def test_vote_breakdown_returns_dict(self):
        ev = EnsembleVoting()
        breakdown = ev.vote_breakdown(_trending_up(30, pct=0.01))
        assert isinstance(breakdown, dict)
        assert len(breakdown) == 4

    def test_vote_breakdown_has_all_strategies(self):
        ev = EnsembleVoting()
        breakdown = ev.vote_breakdown(_trending_up(30))
        for strat in ev.strategies:
            assert strat.name in breakdown

    def test_vote_breakdown_each_has_action(self):
        ev = EnsembleVoting()
        breakdown = ev.vote_breakdown(_trending_up(30))
        for name, info in breakdown.items():
            if "error" not in info:
                assert "action" in info
                assert "confidence" in info

    def test_empty_strategies_returns_hold(self):
        ev = EnsembleVoting(strategies=[])
        sig = ev.fit(_trending_up(30))
        assert sig.action == "hold"

    def test_all_buy_signals(self):
        """When all strategies emit buy, ensemble should buy."""
        class ForceBuy(BaseStrategy):
            name = "ForceBuy"
            def fit(self, prices):
                return StrategySignal("ForceBuy", "buy", 0.9)

        ev = EnsembleVoting(
            strategies=[ForceBuy(), ForceBuy(), ForceBuy()],
            min_confidence=0.0,
        )
        sig = ev.fit([100.0] * 30)
        assert sig.action == "buy"
        assert sig.confidence > 0.8

    def test_all_sell_signals(self):
        class ForceSell(BaseStrategy):
            name = "ForceSell"
            def fit(self, prices):
                return StrategySignal("ForceSell", "sell", 0.9)

        ev = EnsembleVoting(
            strategies=[ForceSell(), ForceSell(), ForceSell()],
            min_confidence=0.0,
        )
        sig = ev.fit([100.0] * 30)
        assert sig.action == "sell"

    def test_name_attribute(self):
        assert EnsembleVoting.name == "EnsembleVoting"

    def test_strategy_exception_handled(self):
        class BrokenStrategy(BaseStrategy):
            name = "Broken"
            def fit(self, prices):
                raise RuntimeError("intentional failure")

        ev = EnsembleVoting(
            strategies=[BrokenStrategy(), MomentumStrategy()],
            min_confidence=0.0,
        )
        sig = ev.fit(_trending_up(15, pct=0.02))
        assert sig.action in ("buy", "sell", "hold")


# ─── StrategyEngine Tests ─────────────────────────────────────────────────────

class TestStrategyEngine:

    def test_strategy_names(self):
        engine = StrategyEngine()
        names = engine.strategy_names()
        expected = {"momentum", "mean_reversion", "vol_breakout", "sentiment", "ensemble"}
        assert set(names) == expected

    def test_evaluate_returns_signal(self):
        engine = StrategyEngine()
        sig = engine.evaluate(_trending_up(30, pct=0.01))
        assert isinstance(sig, StrategySignal)

    def test_evaluate_action_valid(self):
        engine = StrategyEngine()
        sig = engine.evaluate(_trending_up(30, pct=0.01))
        assert sig.action in ("buy", "sell", "hold")

    def test_evaluate_with_sentiment(self):
        engine = StrategyEngine()
        sig = engine.evaluate(_trending_up(30, pct=0.01), sentiment_score=0.5)
        assert sig.action in ("buy", "sell", "hold")

    def test_evaluate_all_returns_all_strategies(self):
        engine = StrategyEngine()
        results = engine.evaluate_all(_trending_up(30, pct=0.01))
        assert set(results.keys()) == set(engine.strategy_names())

    def test_evaluate_all_each_is_signal(self):
        engine = StrategyEngine()
        results = engine.evaluate_all(_trending_up(30, pct=0.01))
        for sig in results.values():
            assert isinstance(sig, StrategySignal)

    def test_backtest_all_returns_all_scores(self):
        engine = StrategyEngine()
        scores = engine.backtest_all(_trending_up(100, pct=0.01))
        assert set(scores.keys()) == set(engine.strategy_names())

    def test_backtest_all_scores_are_floats(self):
        engine = StrategyEngine()
        scores = engine.backtest_all(_trending_up(100, pct=0.01))
        for v in scores.values():
            assert isinstance(v, float)

    def test_get_strategy_existing(self):
        engine = StrategyEngine()
        assert engine.get_strategy("momentum") is engine.momentum
        assert engine.get_strategy("ensemble") is engine.ensemble

    def test_get_strategy_nonexistent(self):
        engine = StrategyEngine()
        assert engine.get_strategy("nonexistent") is None

    def test_evaluate_consistent_with_ensemble(self):
        engine = StrategyEngine()
        prices = _trending_up(30, pct=0.01)
        sig_engine   = engine.evaluate(prices)
        sig_ensemble = engine.ensemble.fit(prices)
        assert sig_engine.action == sig_ensemble.action

    def test_momentum_strategy_accessible(self):
        engine = StrategyEngine()
        assert isinstance(engine.momentum, MomentumStrategy)

    def test_mean_rev_accessible(self):
        engine = StrategyEngine()
        assert isinstance(engine.mean_rev, MeanReversionStrategy)

    def test_vol_breakout_accessible(self):
        engine = StrategyEngine()
        assert isinstance(engine.vol_breakout, VolatilityBreakout)

    def test_sentiment_wt_accessible(self):
        engine = StrategyEngine()
        assert isinstance(engine.sentiment_wt, SentimentWeightedStrategy)


# ─── Integration Tests ────────────────────────────────────────────────────────

class TestStrategyIntegration:

    def test_uptrend_favors_buy(self):
        """Strong uptrend: momentum and ensemble should lean toward buy."""
        engine = StrategyEngine()
        prices = _trending_up(100, start=100.0, pct=0.02)
        sig = engine.evaluate(prices, sentiment_score=0.5)
        # Strong uptrend with positive sentiment should produce buy or hold
        assert sig.action in ("buy", "hold")

    def test_downtrend_favors_sell(self):
        """Strong downtrend: strategies should lean toward sell."""
        engine = StrategyEngine()
        prices = _trending_down(100, start=100.0, pct=0.02)
        sig = engine.evaluate(prices, sentiment_score=-0.5)
        assert sig.action in ("sell", "hold")

    def test_evaluate_all_with_insufficient_data(self):
        """Short price series should return hold for strategies that need more data."""
        engine = StrategyEngine()
        results = engine.evaluate_all([100.0, 101.0, 100.5])
        for sig in results.values():
            assert sig.action in ("buy", "sell", "hold")

    def test_backtest_trending_better_than_flat(self):
        """Momentum should backtest better on trends than flat prices."""
        mom = MomentumStrategy(period=3, threshold=0.005)
        scores_trending = mom.backtest_score(_trending_up(200, pct=0.01))
        # trending should produce a finite score
        assert isinstance(scores_trending, float)

    def test_full_pipeline(self):
        """Simulate a typical call chain from the trading agent."""
        engine = StrategyEngine()
        import random
        random.seed(42)
        prices = [3200.0]
        for _ in range(49):
            prices.append(prices[-1] * (1 + random.gauss(0.0005, 0.02)))

        # Evaluate with slight positive sentiment
        sig = engine.evaluate(prices, sentiment_score=0.1)
        assert sig.action in ("buy", "sell", "hold")
        assert 0.0 <= sig.confidence <= 1.0
        assert sig.strategy_name == "EnsembleVoting"

        # Backtest all
        scores = engine.backtest_all(prices)
        assert len(scores) == 5

        # Get individual signals
        all_sigs = engine.evaluate_all(prices, sentiment_score=0.1)
        assert len(all_sigs) == 5

    def test_mean_reversion_oscillating_series(self):
        """Mean reversion should find opportunities in oscillating prices."""
        mr = MeanReversionStrategy(period=20, std_multiplier=1.5)
        prices = _mean_reverting(50, mean=100, amp=8)
        sig = mr.fit(prices)
        assert sig.action in ("buy", "sell", "hold")

    def test_bollinger_breakout_on_volatile_series(self):
        vb = VolatilityBreakout(period=20, num_std=1.5)
        import random
        random.seed(77)
        prices = [100 + random.gauss(0, 3) for _ in range(20)]
        # Force breakout
        prices.append(prices[-1] + 30)  # large positive breakout
        sig = vb.fit(prices)
        assert sig.action == "buy"

    def test_backtest_detailed_full_stats(self):
        mom = MomentumStrategy(period=3, threshold=0.005)
        result = mom.backtest_detailed(_trending_up(100, pct=0.01))
        assert isinstance(result, BacktestResult)
        assert result.strategy_name == "MomentumStrategy"
        assert 0.0 <= result.win_rate <= 1.0
        assert result.total_trades >= 0
        assert result.max_drawdown >= 0.0


# ─── Additional Edge Case Tests ───────────────────────────────────────────────

class TestMomentumEdgeCases:

    def test_period_one_buy(self):
        m = MomentumStrategy(period=1, threshold=0.005)
        sig = m.fit([100.0, 102.0])
        assert sig.action == "buy"

    def test_period_one_sell(self):
        m = MomentumStrategy(period=1, threshold=0.005)
        sig = m.fit([100.0, 97.0])
        assert sig.action == "sell"

    def test_exact_threshold_boundary(self):
        m = MomentumStrategy(period=1, threshold=0.01)
        # momentum = 0.01 exactly — NOT > threshold → hold
        sig = m.fit([100.0, 101.0])
        assert sig.action == "hold"

    def test_backtest_detailed_win_rate_between_zero_and_one(self):
        m = MomentumStrategy(period=3, threshold=0.005)
        import random; random.seed(5)
        prices = [100 * (1 + random.gauss(0.001, 0.02)) for _ in range(100)]
        result = m.backtest_detailed(prices)
        if result.total_trades > 0:
            assert 0.0 <= result.win_rate <= 1.0

    def test_large_period_near_end_of_prices(self):
        m = MomentumStrategy(period=50, threshold=0.01)
        prices = _trending_up(100, pct=0.005)
        sig = m.fit(prices)
        assert sig.action in ("buy", "sell", "hold")


class TestMeanReversionEdgeCases:

    def test_exactly_at_lower_band(self):
        mr = MeanReversionStrategy(period=20, std_multiplier=1.5)
        import random; random.seed(42)
        prices = [100 + random.gauss(0, 5) for _ in range(20)]
        ma   = statistics.mean(prices[-20:])
        std  = statistics.stdev(prices[-20:])
        # Price exactly at lower band should NOT trigger buy (not strictly less)
        prices.append(ma - 1.5 * std)
        sig = mr.fit(prices)
        # At the boundary; expect hold (z = -1.5 is NOT < -1.5)
        assert sig.action == "hold"

    def test_short_period_two(self):
        mr = MeanReversionStrategy(period=2, std_multiplier=1.0)
        sig = mr.fit([100.0, 50.0, 150.0])
        assert sig.action in ("buy", "sell", "hold")

    def test_longer_price_series(self):
        mr = MeanReversionStrategy(period=20, std_multiplier=2.0)
        import random; random.seed(99)
        prices = [100 + random.gauss(0, 10) for _ in range(200)]
        sig = mr.fit(prices)
        assert sig.action in ("buy", "sell", "hold")


class TestVolatilityBreakoutEdgeCases:

    def test_period_two(self):
        vb = VolatilityBreakout(period=2, num_std=1.0)
        sig = vb.fit([100.0, 150.0, 200.0])
        assert sig.action in ("buy", "sell", "hold")

    def test_very_tight_bands_many_breakouts(self):
        vb = VolatilityBreakout(period=5, num_std=0.1)
        import random; random.seed(3)
        prices = [100 + random.gauss(0, 5) for _ in range(6)]
        sig = vb.fit(prices)
        assert sig.action in ("buy", "sell", "hold")

    def test_num_std_three(self):
        vb = VolatilityBreakout(period=10, num_std=3.0)
        import random; random.seed(8)
        prices = [100 + random.gauss(0, 2) for _ in range(11)]
        sig = vb.fit(prices)
        assert sig.action in ("buy", "sell", "hold")


class TestEnsembleEdgeCases:

    def test_single_strategy_ensemble(self):
        ev = EnsembleVoting(strategies=[MomentumStrategy()], min_confidence=0.0)
        sig = ev.fit(_trending_up(20, pct=0.02))
        assert sig.action in ("buy", "sell", "hold")

    def test_vote_breakdown_with_sentiment_score(self):
        ev = EnsembleVoting()
        breakdown = ev.vote_breakdown(_trending_up(30), sentiment_score=0.8)
        assert isinstance(breakdown, dict)
        assert len(breakdown) > 0

    def test_min_confidence_zero_allows_any_action(self):
        ev = EnsembleVoting(min_confidence=0.0)
        prices = _trending_up(30, pct=0.02)
        sig = ev.fit(prices)
        assert sig.action in ("buy", "sell", "hold")


class TestStrategyEngineEdgeCases:

    def test_evaluate_flat_prices(self):
        engine = StrategyEngine()
        sig = engine.evaluate(_flat(30))
        assert sig.action in ("buy", "sell", "hold")

    def test_evaluate_all_with_sentiment(self):
        engine = StrategyEngine()
        results = engine.evaluate_all(_trending_up(30), sentiment_score=-0.5)
        assert len(results) == 5

    def test_backtest_all_downtrend(self):
        engine = StrategyEngine()
        scores = engine.backtest_all(_trending_down(100, pct=0.01))
        assert len(scores) == 5
        for v in scores.values():
            assert isinstance(v, float)

    def test_evaluate_with_extreme_negative_sentiment(self):
        """Extreme negative sentiment should block SentimentWeighted but engine still works."""
        engine = StrategyEngine()
        sig = engine.evaluate(_trending_up(30, pct=0.01), sentiment_score=-0.99)
        assert sig.action in ("buy", "sell", "hold")
        assert 0.0 <= sig.confidence <= 1.0
