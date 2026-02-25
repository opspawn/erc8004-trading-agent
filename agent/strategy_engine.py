"""
strategy_engine.py — Multi-Strategy Engine for ERC-8004 Trading Agent.

Implements five distinct named trading strategies plus an ensemble meta-strategy.
Each strategy exposes a consistent interface:

    strategy.fit(prices: List[float]) -> StrategySignal
    strategy.backtest_score(prices: List[float]) -> float   # Sharpe-like metric

Strategies:
  1. MomentumStrategy       — buy when 3-period momentum > threshold
  2. MeanReversionStrategy  — buy when price < 20-period MA minus 1.5 std
  3. VolatilityBreakout      — trade on Bollinger Band breakouts
  4. SentimentWeighted       — blend technical signals with sentiment scores
  5. EnsembleVoting          — majority vote across all base strategies

Usage:
    engine = StrategyEngine()
    signal = engine.evaluate(prices, sentiment_score=0.2)
    print(signal.action, signal.confidence)

    # Use individual strategy:
    mom = MomentumStrategy(threshold=0.02)
    sig = mom.fit(prices)
    score = mom.backtest_score(prices)
"""

from __future__ import annotations

import math
import statistics
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from loguru import logger


# ─── Data Classes ─────────────────────────────────────────────────────────────

@dataclass
class StrategySignal:
    """Signal emitted by a single strategy instance."""
    strategy_name: str
    action:        str      # "buy" | "sell" | "hold"
    confidence:    float    # 0.0 – 1.0
    reason:        str      = ""
    metadata:      Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.action not in ("buy", "sell", "hold"):
            raise ValueError(f"Invalid action '{self.action}'")
        self.confidence = max(0.0, min(1.0, self.confidence))


@dataclass
class BacktestResult:
    """Summary of a strategy's historical performance on a price series."""
    strategy_name:  str
    sharpe:         float       # Sharpe-like ratio
    win_rate:       float       # fraction of profitable trades
    total_trades:   int
    avg_return:     float       # average trade return
    max_drawdown:   float       # worst peak-to-trough drawdown


# ─── Base Class ───────────────────────────────────────────────────────────────

class BaseStrategy:
    """Abstract base for all trading strategies."""

    name: str = "base"

    def fit(self, prices: List[float]) -> StrategySignal:
        """
        Generate a trading signal from a price history.

        Args:
            prices: list of closing prices, oldest first, length >= 2

        Returns:
            StrategySignal with action and confidence
        """
        raise NotImplementedError

    def backtest_score(self, prices: List[float]) -> float:
        """
        Compute a quality score for this strategy on a price history.

        Returns a Sharpe-like ratio: positive = good, negative = bad.
        Scores above 1.0 are excellent; below 0.0 means the strategy loses money.
        """
        if len(prices) < 10:
            return 0.0

        returns = self._compute_returns(prices)
        signals = self._rolling_signals(prices)

        trade_returns: List[float] = []
        for i, action in enumerate(signals):
            if action == "buy" and i + 1 < len(returns):
                trade_returns.append(returns[i + 1])
            elif action == "sell" and i + 1 < len(returns):
                trade_returns.append(-returns[i + 1])

        if not trade_returns:
            return 0.0

        avg = statistics.mean(trade_returns)
        if len(trade_returns) < 2:
            return avg / 0.01  # not enough data
        std = statistics.stdev(trade_returns)
        if std < 1e-10:
            return 0.0
        return avg / std

    def backtest_detailed(self, prices: List[float]) -> BacktestResult:
        """Run a detailed backtest and return full statistics."""
        returns = self._compute_returns(prices)
        signals = self._rolling_signals(prices)

        trade_returns: List[float] = []
        for i, action in enumerate(signals):
            if action == "buy" and i + 1 < len(returns):
                trade_returns.append(returns[i + 1])
            elif action == "sell" and i + 1 < len(returns):
                trade_returns.append(-returns[i + 1])

        if not trade_returns:
            return BacktestResult(
                strategy_name=self.name, sharpe=0.0, win_rate=0.0,
                total_trades=0, avg_return=0.0, max_drawdown=0.0,
            )

        wins = sum(1 for r in trade_returns if r > 0)
        avg  = statistics.mean(trade_returns)
        std  = statistics.stdev(trade_returns) if len(trade_returns) > 1 else 1e-9
        sharpe = avg / std if std > 1e-10 else 0.0

        # Compute drawdown on cumulative returns
        cum_ret = 0.0
        peak    = 0.0
        max_dd  = 0.0
        for r in trade_returns:
            cum_ret += r
            if cum_ret > peak:
                peak = cum_ret
            dd = (peak - cum_ret)
            if dd > max_dd:
                max_dd = dd

        return BacktestResult(
            strategy_name=self.name,
            sharpe=sharpe,
            win_rate=wins / len(trade_returns),
            total_trades=len(trade_returns),
            avg_return=avg,
            max_drawdown=max_dd,
        )

    # ── Helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _compute_returns(prices: List[float]) -> List[float]:
        """Compute log returns from a price series."""
        returns = []
        for i in range(1, len(prices)):
            if prices[i - 1] > 0 and prices[i] > 0:
                returns.append(math.log(prices[i] / prices[i - 1]))
        return returns

    def _rolling_signals(self, prices: List[float]) -> List[str]:
        """
        Generate one signal per price bar by calling fit() on rolling windows.
        Used internally by backtest methods.
        """
        min_window = getattr(self, "_min_window", 5)
        signals = ["hold"] * len(prices)
        for i in range(min_window, len(prices)):
            window = prices[: i + 1]
            try:
                sig = self.fit(window)
                signals[i] = sig.action
            except Exception:
                signals[i] = "hold"
        return signals


# ─── Strategy 1: Momentum ─────────────────────────────────────────────────────

class MomentumStrategy(BaseStrategy):
    """
    Momentum Strategy — buy when 3-period momentum exceeds the threshold.

    Momentum = (price[t] - price[t-n]) / price[t-n]

    A positive momentum above `threshold` generates a BUY signal; negative
    momentum below `-threshold` generates a SELL.
    """

    name = "MomentumStrategy"
    _min_window = 4

    def __init__(self, period: int = 3, threshold: float = 0.01) -> None:
        """
        Args:
            period:    look-back window for momentum (bars)
            threshold: minimum momentum to generate a directional signal
        """
        if period < 1:
            raise ValueError("period must be >= 1")
        if threshold < 0:
            raise ValueError("threshold must be >= 0")
        self.period    = period
        self.threshold = threshold

    def fit(self, prices: List[float]) -> StrategySignal:
        if len(prices) < self.period + 1:
            return StrategySignal(self.name, "hold", 0.0, "insufficient data")

        current  = prices[-1]
        previous = prices[-(self.period + 1)]

        if previous <= 0:
            return StrategySignal(self.name, "hold", 0.0, "zero price")

        momentum = (current - previous) / previous
        abs_mom  = abs(momentum)

        if momentum > self.threshold:
            confidence = min(abs_mom / (self.threshold * 5), 1.0)
            return StrategySignal(
                self.name, "buy", confidence,
                f"momentum={momentum:.4f} > {self.threshold}",
                {"momentum": momentum},
            )
        elif momentum < -self.threshold:
            confidence = min(abs_mom / (self.threshold * 5), 1.0)
            return StrategySignal(
                self.name, "sell", confidence,
                f"momentum={momentum:.4f} < -{self.threshold}",
                {"momentum": momentum},
            )
        else:
            return StrategySignal(
                self.name, "hold", 0.3,
                f"momentum={momentum:.4f} within threshold",
                {"momentum": momentum},
            )


# ─── Strategy 2: Mean Reversion ───────────────────────────────────────────────

class MeanReversionStrategy(BaseStrategy):
    """
    Mean Reversion Strategy — trade when price deviates significantly from MA.

    BUY  when price < MA - k * std   (oversold)
    SELL when price > MA + k * std   (overbought)
    """

    name = "MeanReversionStrategy"
    _min_window = 21

    def __init__(self, period: int = 20, std_multiplier: float = 1.5) -> None:
        """
        Args:
            period:         look-back for moving average (bars)
            std_multiplier: number of standard deviations from MA to trigger
        """
        if period < 2:
            raise ValueError("period must be >= 2")
        self.period         = period
        self.std_multiplier = std_multiplier

    def fit(self, prices: List[float]) -> StrategySignal:
        if len(prices) < self.period + 1:
            return StrategySignal(self.name, "hold", 0.0, "insufficient data")

        window  = prices[-self.period:]
        ma      = statistics.mean(window)
        std     = statistics.stdev(window) if len(window) > 1 else 0.0
        current = prices[-1]

        if std < 1e-10:
            return StrategySignal(self.name, "hold", 0.5, "zero variance")

        z_score = (current - ma) / std
        band    = self.std_multiplier

        if z_score < -band:
            # Price below lower band → expect reversion upward → BUY
            confidence = min(abs(z_score) / (band * 2), 1.0)
            return StrategySignal(
                self.name, "buy", confidence,
                f"z={z_score:.2f} < -{band} (oversold)",
                {"z_score": z_score, "ma": ma, "std": std},
            )
        elif z_score > band:
            # Price above upper band → expect reversion downward → SELL
            confidence = min(abs(z_score) / (band * 2), 1.0)
            return StrategySignal(
                self.name, "sell", confidence,
                f"z={z_score:.2f} > {band} (overbought)",
                {"z_score": z_score, "ma": ma, "std": std},
            )
        else:
            return StrategySignal(
                self.name, "hold", 0.4,
                f"z={z_score:.2f} within bands",
                {"z_score": z_score},
            )


# ─── Strategy 3: Volatility Breakout (Bollinger Bands) ───────────────────────

class VolatilityBreakout(BaseStrategy):
    """
    Volatility Breakout — trade when price breaks outside Bollinger Bands.

    Bollinger Bands:
        Upper = MA + k * std
        Lower = MA - k * std

    BUY  signal: price closes above upper band (upward breakout)
    SELL signal: price closes below lower band (downward breakout)
    HOLD when price is within the bands.
    """

    name = "VolatilityBreakout"
    _min_window = 21

    def __init__(self, period: int = 20, num_std: float = 2.0) -> None:
        """
        Args:
            period:  Bollinger Band period (bars)
            num_std: band width in standard deviations
        """
        if period < 2:
            raise ValueError("period must be >= 2")
        self.period  = period
        self.num_std = num_std

    def compute_bands(self, prices: List[float]) -> Tuple[float, float, float]:
        """
        Return (lower_band, middle_band, upper_band) from recent price history.
        """
        if len(prices) < self.period:
            raise ValueError("Not enough prices for Bollinger Bands")
        window = prices[-self.period:]
        ma     = statistics.mean(window)
        std    = statistics.stdev(window) if len(window) > 1 else 0.0
        upper  = ma + self.num_std * std
        lower  = ma - self.num_std * std
        return lower, ma, upper

    def fit(self, prices: List[float]) -> StrategySignal:
        if len(prices) < self.period + 1:
            return StrategySignal(self.name, "hold", 0.0, "insufficient data")

        try:
            lower, ma, upper = self.compute_bands(prices)
        except Exception as exc:
            return StrategySignal(self.name, "hold", 0.0, str(exc))

        current = prices[-1]
        band_width = upper - lower

        if band_width < 1e-10:
            return StrategySignal(self.name, "hold", 0.5, "zero band width")

        if current > upper:
            # Breakout to the upside
            pct_above = (current - upper) / band_width
            confidence = min(0.5 + pct_above * 2, 1.0)
            return StrategySignal(
                self.name, "buy", confidence,
                f"price {current:.2f} above upper band {upper:.2f}",
                {"upper": upper, "lower": lower, "ma": ma},
            )
        elif current < lower:
            # Breakout to the downside
            pct_below = (lower - current) / band_width
            confidence = min(0.5 + pct_below * 2, 1.0)
            return StrategySignal(
                self.name, "sell", confidence,
                f"price {current:.2f} below lower band {lower:.2f}",
                {"upper": upper, "lower": lower, "ma": ma},
            )
        else:
            # Inside bands — no breakout
            position = (current - lower) / band_width  # 0.0 = at lower, 1.0 = at upper
            return StrategySignal(
                self.name, "hold", 0.3,
                f"price within bands ({position:.0%} of width)",
                {"upper": upper, "lower": lower, "ma": ma, "band_position": position},
            )


# ─── Strategy 4: Sentiment Weighted ──────────────────────────────────────────

class SentimentWeightedStrategy(BaseStrategy):
    """
    Sentiment Weighted Strategy — blend technical signals with sentiment.

    Computes a base technical signal from momentum, then adjusts the action
    and confidence based on the provided sentiment score (-1.0 to +1.0).

    - Strong positive sentiment boosts BUY signals and suppresses SELL
    - Strong negative sentiment boosts SELL signals and suppresses BUY
    - Extreme negative sentiment (< -0.7) blocks all trades
    """

    name = "SentimentWeighted"
    _min_window = 4

    def __init__(
        self,
        sentiment_weight: float = 0.3,
        technical_weight: float = 0.7,
        extreme_negative: float = -0.7,
    ) -> None:
        """
        Args:
            sentiment_weight: weight assigned to sentiment vs technical signal
            technical_weight: weight assigned to technical signal
            extreme_negative: sentiment below this blocks all trading
        """
        total = sentiment_weight + technical_weight
        if abs(total - 1.0) > 0.01:
            raise ValueError("sentiment_weight + technical_weight must sum to 1.0")
        self.sentiment_weight = sentiment_weight
        self.technical_weight = technical_weight
        self.extreme_negative = extreme_negative
        self._momentum = MomentumStrategy(period=3, threshold=0.005)

    def fit(
        self,
        prices: List[float],
        sentiment_score: float = 0.0,
    ) -> StrategySignal:
        """
        Args:
            prices:          price history (oldest first)
            sentiment_score: aggregated sentiment [-1.0, +1.0]
        """
        # Guard: extreme negative sentiment blocks all trades
        if sentiment_score < self.extreme_negative:
            return StrategySignal(
                self.name, "hold", 1.0,
                f"trade blocked: extreme negative sentiment={sentiment_score:.2f}",
                {"sentiment": sentiment_score, "blocked": True},
            )

        # Technical signal from momentum
        tech_signal = self._momentum.fit(prices)
        tech_action = tech_signal.action
        tech_conf   = tech_signal.confidence

        # Convert sentiment to a directional bias
        # Positive sentiment → bias toward buy; negative → bias toward sell
        if sentiment_score > 0.3:
            sentiment_action = "buy"
        elif sentiment_score < -0.3:
            sentiment_action = "sell"
        else:
            sentiment_action = "hold"

        # Blend signals
        if tech_action == sentiment_action:
            # Both agree — combine confidences
            blended_conf = (
                self.technical_weight * tech_conf
                + self.sentiment_weight * abs(sentiment_score)
            )
            return StrategySignal(
                self.name, tech_action, min(blended_conf, 1.0),
                f"tech={tech_action}({tech_conf:.2f}) + sentiment={sentiment_score:.2f} agree",
                {"sentiment": sentiment_score, "tech_action": tech_action},
            )
        elif tech_action == "hold":
            # Technical is neutral, let sentiment lead with lower confidence
            if sentiment_action != "hold":
                return StrategySignal(
                    self.name, sentiment_action,
                    self.sentiment_weight * abs(sentiment_score),
                    f"sentiment={sentiment_score:.2f} drives signal",
                    {"sentiment": sentiment_score},
                )
            return StrategySignal(
                self.name, "hold", 0.3, "both signals neutral",
                {"sentiment": sentiment_score},
            )
        elif sentiment_action == "hold":
            # Sentiment neutral, technical leads
            return StrategySignal(
                self.name, tech_action, tech_conf * self.technical_weight,
                f"tech={tech_action} with neutral sentiment",
                {"sentiment": sentiment_score, "tech_action": tech_action},
            )
        else:
            # Signals conflict — default to hold with low confidence
            return StrategySignal(
                self.name, "hold", 0.2,
                f"conflict: tech={tech_action} vs sentiment={sentiment_action}",
                {"sentiment": sentiment_score, "tech_action": tech_action, "conflict": True},
            )


# ─── Strategy 5: Ensemble Voting ─────────────────────────────────────────────

class EnsembleVoting(BaseStrategy):
    """
    Ensemble Voting — meta-strategy that aggregates signals from base strategies.

    Runs all component strategies, collects their votes (buy/sell/hold), and
    uses confidence-weighted majority voting to produce a final signal.

    This is the top-level strategy used by the StrategyEngine by default.
    """

    name = "EnsembleVoting"
    _min_window = 21

    def __init__(
        self,
        strategies: Optional[List[BaseStrategy]] = None,
        min_confidence: float = 0.4,
    ) -> None:
        """
        Args:
            strategies:     list of base strategies to vote; defaults to all four
            min_confidence: minimum aggregate confidence to generate non-hold signal
        """
        if strategies is None:
            strategies = [
                MomentumStrategy(),
                MeanReversionStrategy(),
                VolatilityBreakout(),
                SentimentWeightedStrategy(),
            ]
        self.strategies     = strategies
        self.min_confidence = min_confidence

    def fit(
        self,
        prices: List[float],
        sentiment_score: float = 0.0,
    ) -> StrategySignal:
        """
        Run all strategies and return majority-vote signal.

        Confidence-weighted: strategies with higher confidence get more weight.
        """
        votes: Dict[str, float] = {"buy": 0.0, "sell": 0.0, "hold": 0.0}
        individual: List[StrategySignal] = []

        for strategy in self.strategies:
            try:
                if isinstance(strategy, SentimentWeightedStrategy):
                    sig = strategy.fit(prices, sentiment_score=sentiment_score)
                else:
                    sig = strategy.fit(prices)
                votes[sig.action] += sig.confidence
                individual.append(sig)
            except Exception as exc:
                logger.warning(f"Ensemble: strategy {strategy.name} failed: {exc}")

        if not individual:
            return StrategySignal(self.name, "hold", 0.0, "all strategies failed")

        # Determine winner by weighted vote
        total_weight = sum(votes.values())
        if total_weight < 1e-10:
            return StrategySignal(self.name, "hold", 0.0, "zero total weight")

        best_action = max(votes, key=lambda a: votes[a])
        best_weight = votes[best_action]
        confidence  = best_weight / total_weight

        if confidence < self.min_confidence:
            return StrategySignal(
                self.name, "hold", confidence,
                f"majority insufficient: {confidence:.0%} for {best_action}",
                {"votes": votes, "individual": [s.action for s in individual]},
            )

        return StrategySignal(
            self.name, best_action, confidence,
            f"ensemble vote: {best_action} with {confidence:.0%} confidence",
            {"votes": votes, "individual": [s.action for s in individual]},
        )

    def vote_breakdown(self, prices: List[float], sentiment_score: float = 0.0) -> Dict[str, Any]:
        """
        Return detailed vote breakdown for inspection / debugging.
        """
        breakdown = {}
        for strategy in self.strategies:
            try:
                if isinstance(strategy, SentimentWeightedStrategy):
                    sig = strategy.fit(prices, sentiment_score=sentiment_score)
                else:
                    sig = strategy.fit(prices)
                breakdown[strategy.name] = {
                    "action":     sig.action,
                    "confidence": sig.confidence,
                    "reason":     sig.reason,
                }
            except Exception as exc:
                breakdown[strategy.name] = {"error": str(exc)}
        return breakdown


# ─── Strategy Engine ──────────────────────────────────────────────────────────

class StrategyEngine:
    """
    Top-level strategy engine used by the trading agent.

    Maintains all strategies, provides evaluation interface, and tracks
    running performance metrics.
    """

    def __init__(self) -> None:
        self.momentum      = MomentumStrategy()
        self.mean_rev      = MeanReversionStrategy()
        self.vol_breakout  = VolatilityBreakout()
        self.sentiment_wt  = SentimentWeightedStrategy()
        self.ensemble      = EnsembleVoting([
            self.momentum,
            self.mean_rev,
            self.vol_breakout,
            self.sentiment_wt,
        ])

        self._strategies: Dict[str, BaseStrategy] = {
            "momentum":       self.momentum,
            "mean_reversion": self.mean_rev,
            "vol_breakout":   self.vol_breakout,
            "sentiment":      self.sentiment_wt,
            "ensemble":       self.ensemble,
        }

    def evaluate(
        self,
        prices: List[float],
        sentiment_score: float = 0.0,
    ) -> StrategySignal:
        """
        Run the ensemble and return a final trading signal.

        Args:
            prices:          price history, oldest first (min 21 bars)
            sentiment_score: aggregated sentiment [-1.0, +1.0]

        Returns:
            StrategySignal from the EnsembleVoting meta-strategy
        """
        return self.ensemble.fit(prices, sentiment_score=sentiment_score)

    def evaluate_all(
        self,
        prices: List[float],
        sentiment_score: float = 0.0,
    ) -> Dict[str, StrategySignal]:
        """Run every strategy and return a mapping of name → signal."""
        results: Dict[str, StrategySignal] = {}
        for name, strategy in self._strategies.items():
            try:
                if isinstance(strategy, (SentimentWeightedStrategy, EnsembleVoting)):
                    results[name] = strategy.fit(prices, sentiment_score=sentiment_score)
                else:
                    results[name] = strategy.fit(prices)
            except Exception as exc:
                logger.warning(f"StrategyEngine: {name} failed: {exc}")
        return results

    def backtest_all(self, prices: List[float]) -> Dict[str, float]:
        """Run backtest_score() on every strategy and return name → score."""
        scores: Dict[str, float] = {}
        for name, strategy in self._strategies.items():
            try:
                scores[name] = strategy.backtest_score(prices)
            except Exception as exc:
                logger.warning(f"StrategyEngine backtest: {name} failed: {exc}")
                scores[name] = 0.0
        return scores

    def strategy_names(self) -> List[str]:
        return list(self._strategies.keys())

    def get_strategy(self, name: str) -> Optional[BaseStrategy]:
        return self._strategies.get(name)
