"""
strategy_runner.py — Multi-strategy runner for the ERC-8004 Trading Agent.

Runs multiple strategies in parallel, aggregates their signals, and produces
a single consolidated position-sizing decision.

Strategies supported (pluggable):
  - TrendStrategy: SMA crossover
  - MeanReversionStrategy: Z-score bands
  - MomentumStrategy: Rate-of-change

Aggregation modes:
  - majority: signal must appear in ≥50% of strategies
  - unanimous: all strategies must agree
  - weighted: weighted sum of confidences, winner takes all

Usage:
    runner = StrategyRunner([TrendStrategy(), MomentumStrategy()])
    result = runner.run(price_history, current_price)
    if result.action != "hold":
        size = result.position_size_usdc
"""

from __future__ import annotations

import math
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

from loguru import logger


# ─── Data Classes ─────────────────────────────────────────────────────────────

@dataclass
class StrategySignal:
    """Signal emitted by a single strategy."""
    strategy_name: str
    action: str             # "buy", "sell", "hold"
    confidence: float       # 0.0 – 1.0
    reason: str = ""
    metadata: Dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.action not in ("buy", "sell", "hold"):
            raise ValueError(f"Invalid action '{self.action}'. Must be buy/sell/hold.")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(f"confidence must be in [0, 1], got {self.confidence}")


@dataclass
class AggregatedSignal:
    """Consolidated output from all strategies."""
    action: str             # "buy", "sell", "hold"
    confidence: float       # 0.0 – 1.0 (weighted average of agreeing strategies)
    position_size_usdc: float = 0.0
    signals: List[StrategySignal] = field(default_factory=list)
    buy_count: int = 0
    sell_count: int = 0
    hold_count: int = 0
    aggregation_mode: str = "majority"
    computed_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    @property
    def is_actionable(self) -> bool:
        return self.action in ("buy", "sell") and self.confidence > 0.0


# ─── Base Strategy ─────────────────────────────────────────────────────────────

class BaseStrategy:
    """Abstract base class for trading strategies."""

    def __init__(self, name: str, weight: float = 1.0) -> None:
        if weight <= 0:
            raise ValueError("weight must be positive")
        self.name = name
        self.weight = weight

    def generate_signal(
        self,
        prices: List[float],
        current_price: float,
    ) -> StrategySignal:
        raise NotImplementedError

    def _safe_mean(self, values: List[float]) -> float:
        return sum(values) / len(values) if values else 0.0

    def _safe_std(self, values: List[float]) -> float:
        if len(values) < 2:
            return 1e-9
        mean = self._safe_mean(values)
        var = sum((v - mean) ** 2 for v in values) / (len(values) - 1)
        return math.sqrt(var) if var > 0 else 1e-9


# ─── Concrete Strategies ───────────────────────────────────────────────────────

class TrendStrategy(BaseStrategy):
    """
    Simple Moving Average crossover.
    BUY when short MA > long MA by threshold; SELL when below.
    """

    def __init__(
        self,
        short_window: int = 5,
        long_window: int = 20,
        threshold_pct: float = 0.005,
        weight: float = 1.0,
    ) -> None:
        super().__init__("trend", weight)
        if short_window >= long_window:
            raise ValueError("short_window must be less than long_window")
        self.short_window = short_window
        self.long_window = long_window
        self.threshold_pct = threshold_pct

    def generate_signal(self, prices: List[float], current_price: float) -> StrategySignal:
        if len(prices) < self.long_window:
            return StrategySignal(self.name, "hold", 0.0, "Insufficient history")

        short_ma = self._safe_mean(prices[-self.short_window:])
        long_ma = self._safe_mean(prices[-self.long_window:])

        if long_ma == 0:
            return StrategySignal(self.name, "hold", 0.0, "Zero long MA")

        diff_pct = (short_ma - long_ma) / long_ma
        confidence = min(abs(diff_pct) / (self.threshold_pct * 5), 1.0)

        if diff_pct > self.threshold_pct:
            return StrategySignal(self.name, "buy", confidence,
                                  f"Short MA {short_ma:.2f} > Long MA {long_ma:.2f} (+{diff_pct:.3%})",
                                  {"short_ma": short_ma, "long_ma": long_ma})
        elif diff_pct < -self.threshold_pct:
            return StrategySignal(self.name, "sell", confidence,
                                  f"Short MA {short_ma:.2f} < Long MA {long_ma:.2f} ({diff_pct:.3%})",
                                  {"short_ma": short_ma, "long_ma": long_ma})
        return StrategySignal(self.name, "hold", 0.0, "No significant crossover")


class MeanReversionStrategy(BaseStrategy):
    """
    Z-score band mean reversion.
    BUY when price is z_thresh std below mean; SELL when above.
    """

    def __init__(
        self,
        lookback: int = 20,
        z_thresh: float = 1.5,
        weight: float = 1.0,
    ) -> None:
        super().__init__("mean_reversion", weight)
        if lookback < 2:
            raise ValueError("lookback must be >= 2")
        self.lookback = lookback
        self.z_thresh = z_thresh

    def generate_signal(self, prices: List[float], current_price: float) -> StrategySignal:
        if len(prices) < self.lookback:
            return StrategySignal(self.name, "hold", 0.0, "Insufficient history")

        window = prices[-self.lookback:]
        mean = self._safe_mean(window)
        std = self._safe_std(window)
        z = (current_price - mean) / std
        confidence = min(abs(z) / (self.z_thresh * 3), 1.0)

        if z < -self.z_thresh:
            return StrategySignal(self.name, "buy", confidence,
                                  f"Price below mean by {abs(z):.2f}σ (buy dip)",
                                  {"z_score": z, "mean": mean})
        elif z > self.z_thresh:
            return StrategySignal(self.name, "sell", confidence,
                                  f"Price above mean by {z:.2f}σ (sell spike)",
                                  {"z_score": z, "mean": mean})
        return StrategySignal(self.name, "hold", 0.0, f"Within bands (z={z:.2f})")


class MomentumStrategy(BaseStrategy):
    """
    Rate-of-change momentum.
    BUY if ROC > buy_thresh; SELL if ROC < sell_thresh.
    """

    def __init__(
        self,
        lookback: int = 10,
        buy_thresh: float = 0.02,
        sell_thresh: float = -0.02,
        weight: float = 1.0,
    ) -> None:
        super().__init__("momentum", weight)
        if lookback < 1:
            raise ValueError("lookback must be >= 1")
        self.lookback = lookback
        self.buy_thresh = buy_thresh
        self.sell_thresh = sell_thresh

    def generate_signal(self, prices: List[float], current_price: float) -> StrategySignal:
        if len(prices) < self.lookback:
            return StrategySignal(self.name, "hold", 0.0, "Insufficient history")

        past_price = prices[-self.lookback]
        if past_price == 0:
            return StrategySignal(self.name, "hold", 0.0, "Zero past price")

        roc = (current_price - past_price) / past_price
        confidence = min(abs(roc) / (max(abs(self.buy_thresh), abs(self.sell_thresh)) * 5), 1.0)

        if roc > self.buy_thresh:
            return StrategySignal(self.name, "buy", confidence,
                                  f"Positive momentum: ROC={roc:.3%}",
                                  {"roc": roc})
        elif roc < self.sell_thresh:
            return StrategySignal(self.name, "sell", confidence,
                                  f"Negative momentum: ROC={roc:.3%}",
                                  {"roc": roc})
        return StrategySignal(self.name, "hold", 0.0, f"Weak momentum (ROC={roc:.3%})")


# ─── Strategy Runner ──────────────────────────────────────────────────────────

class StrategyRunner:
    """
    Runs multiple strategies in parallel and aggregates their signals.

    Args:
        strategies: List of BaseStrategy instances.
        aggregation_mode: "majority", "unanimous", or "weighted".
        capital_usdc: Total capital available for position sizing.
        max_position_pct: Max fraction of capital per trade.
        min_confidence: Minimum confidence threshold to act.
        parallel: If True, run strategies in parallel threads.
    """

    VALID_MODES = ("majority", "unanimous", "weighted")

    def __init__(
        self,
        strategies: Optional[List[BaseStrategy]] = None,
        aggregation_mode: str = "majority",
        capital_usdc: float = 1000.0,
        max_position_pct: float = 0.10,
        min_confidence: float = 0.3,
        parallel: bool = True,
    ) -> None:
        if aggregation_mode not in self.VALID_MODES:
            raise ValueError(f"aggregation_mode must be one of {self.VALID_MODES}")
        if not 0 < max_position_pct <= 1:
            raise ValueError("max_position_pct must be in (0, 1]")
        if not 0 <= min_confidence <= 1:
            raise ValueError("min_confidence must be in [0, 1]")

        self.strategies = strategies if strategies is not None else [
            TrendStrategy(),
            MeanReversionStrategy(),
            MomentumStrategy(),
        ]
        self.aggregation_mode = aggregation_mode
        self.capital_usdc = capital_usdc
        self.max_position_pct = max_position_pct
        self.min_confidence = min_confidence
        self.parallel = parallel
        self._signal_history: List[AggregatedSignal] = []

    def add_strategy(self, strategy: BaseStrategy) -> None:
        self.strategies.append(strategy)

    def remove_strategy(self, name: str) -> bool:
        before = len(self.strategies)
        self.strategies = [s for s in self.strategies if s.name != name]
        return len(self.strategies) < before

    # ── Signal collection ──────────────────────────────────────────────────────

    def _collect_signals_serial(
        self,
        prices: List[float],
        current_price: float,
    ) -> List[StrategySignal]:
        return [s.generate_signal(prices, current_price) for s in self.strategies]

    def _collect_signals_parallel(
        self,
        prices: List[float],
        current_price: float,
    ) -> List[StrategySignal]:
        results: List[StrategySignal] = [None] * len(self.strategies)  # type: ignore
        with ThreadPoolExecutor(max_workers=min(len(self.strategies), 4)) as ex:
            futures = {
                ex.submit(s.generate_signal, prices, current_price): i
                for i, s in enumerate(self.strategies)
            }
            for future in as_completed(futures):
                idx = futures[future]
                try:
                    results[idx] = future.result()
                except Exception as e:
                    logger.warning(f"Strategy {self.strategies[idx].name} failed: {e}")
                    results[idx] = StrategySignal(
                        self.strategies[idx].name, "hold", 0.0, f"Error: {e}"
                    )
        return results

    # ── Aggregation ────────────────────────────────────────────────────────────

    def _aggregate_majority(self, signals: List[StrategySignal]) -> Tuple[str, float]:
        counts: Dict[str, int] = {"buy": 0, "sell": 0, "hold": 0}
        conf_sums: Dict[str, float] = {"buy": 0.0, "sell": 0.0, "hold": 0.0}
        for sig in signals:
            counts[sig.action] += 1
            conf_sums[sig.action] += sig.confidence

        total = len(signals)
        for action in ("buy", "sell"):
            if counts[action] / total >= 0.5:
                avg_conf = conf_sums[action] / counts[action] if counts[action] > 0 else 0
                return action, avg_conf
        return "hold", 0.0

    def _aggregate_unanimous(self, signals: List[StrategySignal]) -> Tuple[str, float]:
        actions = {s.action for s in signals}
        if len(actions) == 1 and list(actions)[0] != "hold":
            action = list(actions)[0]
            avg_conf = sum(s.confidence for s in signals) / len(signals)
            return action, avg_conf
        return "hold", 0.0

    def _aggregate_weighted(self, signals: List[StrategySignal]) -> Tuple[str, float]:
        weights = {s.strategy_name: 1.0 for s in signals}
        for strat in self.strategies:
            weights[strat.name] = strat.weight

        scores: Dict[str, float] = {"buy": 0.0, "sell": 0.0}
        total_weight = sum(weights.get(s.strategy_name, 1.0) for s in signals)

        for sig in signals:
            w = weights.get(sig.strategy_name, 1.0)
            if sig.action in scores:
                scores[sig.action] += sig.confidence * w

        if total_weight == 0:
            return "hold", 0.0

        buy_score = scores["buy"] / total_weight
        sell_score = scores["sell"] / total_weight

        if buy_score > sell_score and buy_score > 0:
            return "buy", buy_score
        elif sell_score > buy_score and sell_score > 0:
            return "sell", sell_score
        return "hold", 0.0

    def _compute_position_size(self, confidence: float) -> float:
        """Kelly-inspired position sizing: size = capital × max_pct × confidence."""
        return self.capital_usdc * self.max_position_pct * confidence

    # ── Main entry point ───────────────────────────────────────────────────────

    def run(
        self,
        prices: List[float],
        current_price: Optional[float] = None,
    ) -> AggregatedSignal:
        """
        Run all strategies and return an aggregated signal.

        Args:
            prices: Historical price list (most recent last).
            current_price: Current price (defaults to prices[-1]).

        Returns:
            AggregatedSignal with action, confidence, and position sizing.
        """
        if not prices:
            return AggregatedSignal(
                action="hold", confidence=0.0, signals=[],
                aggregation_mode=self.aggregation_mode,
            )

        cp = current_price if current_price is not None else prices[-1]

        if self.parallel and len(self.strategies) > 1:
            signals = self._collect_signals_parallel(prices, cp)
        else:
            signals = self._collect_signals_serial(prices, cp)

        # Count votes
        counts: Dict[str, int] = {"buy": 0, "sell": 0, "hold": 0}
        for s in signals:
            counts[s.action] += 1

        # Aggregate
        if self.aggregation_mode == "majority":
            action, confidence = self._aggregate_majority(signals)
        elif self.aggregation_mode == "unanimous":
            action, confidence = self._aggregate_unanimous(signals)
        else:
            action, confidence = self._aggregate_weighted(signals)

        # Apply min_confidence gate
        if confidence < self.min_confidence:
            action = "hold"
            confidence = 0.0

        position_size = self._compute_position_size(confidence) if action != "hold" else 0.0

        agg = AggregatedSignal(
            action=action,
            confidence=confidence,
            position_size_usdc=position_size,
            signals=signals,
            buy_count=counts["buy"],
            sell_count=counts["sell"],
            hold_count=counts["hold"],
            aggregation_mode=self.aggregation_mode,
        )
        self._signal_history.append(agg)
        logger.debug(f"Aggregated signal: {action} ({confidence:.2%}) size={position_size:.2f}")
        return agg

    def get_signal_history(self) -> List[AggregatedSignal]:
        return list(self._signal_history)

    def clear_history(self) -> None:
        self._signal_history.clear()

    def stats(self) -> Dict:
        """Summarise signal history."""
        history = self._signal_history
        if not history:
            return {"total": 0, "buy": 0, "sell": 0, "hold": 0}
        counts: Dict[str, int] = {"buy": 0, "sell": 0, "hold": 0}
        for sig in history:
            counts[sig.action] = counts.get(sig.action, 0) + 1
        avg_conf = sum(s.confidence for s in history) / len(history)
        return {
            "total": len(history),
            "buy": counts["buy"],
            "sell": counts["sell"],
            "hold": counts["hold"],
            "avg_confidence": avg_conf,
        }
