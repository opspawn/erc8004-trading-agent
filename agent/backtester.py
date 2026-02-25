"""
backtester.py — Historical price simulation and backtest engine for ERC-8004.

Replays historical price data against a trading strategy and computes:
  - Per-trade P&L
  - Cumulative returns
  - Sharpe ratio (annualised)
  - Max drawdown
  - Win rate / profit factor

Usage:
    bt = Backtester(initial_capital=1000.0)
    prices = bt.generate_synthetic_prices("ETH", days=30)
    trades = bt.run(prices, strategy="trend")
    stats  = bt.compute_stats(trades)
    print(stats.sharpe_ratio, stats.max_drawdown_pct)
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Callable, Dict, List, Optional, Tuple

from loguru import logger


# ─── Data Classes ─────────────────────────────────────────────────────────────

@dataclass
class PriceBar:
    """OHLCV bar for a single time period."""
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float = 0.0

    def mid(self) -> float:
        return (self.high + self.low) / 2.0

    def range_pct(self) -> float:
        if self.open == 0:
            return 0.0
        return (self.high - self.low) / self.open


@dataclass
class BacktestTrade:
    """A single simulated trade produced by the backtester."""
    trade_id: int
    token: str
    side: str           # "BUY" or "SELL"
    entry_price: float
    exit_price: float
    size: float         # position size in base currency
    entry_ts: datetime
    exit_ts: datetime
    pnl_usdc: float = 0.0
    pnl_pct: float = 0.0
    held_bars: int = 0

    @property
    def is_winner(self) -> bool:
        return self.pnl_usdc > 0


@dataclass
class BacktestStats:
    """Aggregate performance statistics over a backtest run."""
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    gross_profit: float = 0.0
    gross_loss: float = 0.0
    net_pnl: float = 0.0
    final_capital: float = 0.0
    max_drawdown_pct: float = 0.0
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    best_trade: float = 0.0
    worst_trade: float = 0.0
    total_return_pct: float = 0.0
    annualised_return_pct: float = 0.0
    num_bars: int = 0

    @property
    def expectancy(self) -> float:
        """Expected P&L per trade."""
        if self.total_trades == 0:
            return 0.0
        return self.net_pnl / self.total_trades


# ─── Strategy Signals ─────────────────────────────────────────────────────────

def trend_signal(bars: List[PriceBar], idx: int, lookback: int = 10) -> Optional[str]:
    """Simple moving-average crossover: BUY when price > MA, SELL when below."""
    if idx < lookback:
        return None
    window = bars[max(0, idx - lookback): idx]
    ma = sum(b.close for b in window) / len(window)
    current = bars[idx].close
    if current > ma * 1.005:
        return "BUY"
    elif current < ma * 0.995:
        return "SELL"
    return None


def mean_reversion_signal(bars: List[PriceBar], idx: int, lookback: int = 20,
                           z_thresh: float = 1.5) -> Optional[str]:
    """Z-score mean reversion: BUY when price is 1.5σ below mean."""
    if idx < lookback:
        return None
    window = [b.close for b in bars[max(0, idx - lookback): idx]]
    mean = sum(window) / len(window)
    variance = sum((x - mean) ** 2 for x in window) / len(window)
    std = math.sqrt(variance) if variance > 0 else 1e-9
    z = (bars[idx].close - mean) / std
    if z < -z_thresh:
        return "BUY"
    elif z > z_thresh:
        return "SELL"
    return None


def momentum_signal(bars: List[PriceBar], idx: int, lookback: int = 5) -> Optional[str]:
    """Rate-of-change momentum: BUY if price up >2% over lookback bars."""
    if idx < lookback:
        return None
    past_close = bars[idx - lookback].close
    current = bars[idx].close
    if past_close == 0:
        return None
    roc = (current - past_close) / past_close
    if roc > 0.02:
        return "BUY"
    elif roc < -0.02:
        return "SELL"
    return None


STRATEGY_MAP: Dict[str, Callable] = {
    "trend": trend_signal,
    "mean_reversion": mean_reversion_signal,
    "momentum": momentum_signal,
}


# ─── Backtester ───────────────────────────────────────────────────────────────

class Backtester:
    """
    Simulates trading a strategy against historical price bars.

    Args:
        initial_capital: Starting USDC capital.
        position_size_pct: Fraction of capital to allocate per trade (0–1).
        max_hold_bars: Force-close position after this many bars.
        slippage_pct: Simulated slippage on each fill.
        fee_pct: Taker fee per trade leg.
    """

    def __init__(
        self,
        initial_capital: float = 1000.0,
        position_size_pct: float = 0.10,
        max_hold_bars: int = 10,
        slippage_pct: float = 0.001,
        fee_pct: float = 0.002,
    ) -> None:
        if initial_capital <= 0:
            raise ValueError("initial_capital must be positive")
        if not 0 < position_size_pct <= 1:
            raise ValueError("position_size_pct must be in (0, 1]")
        self.initial_capital = initial_capital
        self.position_size_pct = position_size_pct
        self.max_hold_bars = max_hold_bars
        self.slippage_pct = slippage_pct
        self.fee_pct = fee_pct
        self._rng = random.Random(42)

    # ── Synthetic data generation ──────────────────────────────────────────────

    def generate_synthetic_prices(
        self,
        token: str = "ETH",
        days: int = 30,
        start_price: float = 2000.0,
        volatility: float = 0.02,
        drift: float = 0.0005,
    ) -> List[PriceBar]:
        """Generate GBM price bars for testing."""
        bars: List[PriceBar] = []
        price = start_price
        now = datetime(2026, 1, 1, tzinfo=timezone.utc)
        for i in range(days):
            ret = self._rng.gauss(drift, volatility)
            close = max(price * math.exp(ret), 0.01)
            high = close * (1 + abs(self._rng.gauss(0, volatility / 2)))
            low = close * (1 - abs(self._rng.gauss(0, volatility / 2)))
            bars.append(PriceBar(
                timestamp=now + timedelta(days=i),
                open=price,
                high=max(price, close, high),
                low=min(price, close, low),
                close=close,
                volume=self._rng.uniform(1e5, 1e7),
            ))
            price = close
        logger.debug(f"Generated {len(bars)} bars for {token}")
        return bars

    def generate_trending_prices(
        self,
        days: int = 30,
        start_price: float = 1000.0,
        trend: float = 0.01,
    ) -> List[PriceBar]:
        """Deterministic uptrend for reproducible tests."""
        bars: List[PriceBar] = []
        price = start_price
        now = datetime(2026, 1, 1, tzinfo=timezone.utc)
        for i in range(days):
            price = price * (1 + trend)
            bars.append(PriceBar(
                timestamp=now + timedelta(days=i),
                open=price / (1 + trend),
                high=price * 1.005,
                low=price * 0.995,
                close=price,
                volume=1_000_000.0,
            ))
        return bars

    def generate_mean_reverting_prices(
        self,
        days: int = 60,
        mean_price: float = 1000.0,
        amplitude: float = 50.0,
    ) -> List[PriceBar]:
        """Sinusoidal oscillation for mean-reversion testing."""
        bars: List[PriceBar] = []
        now = datetime(2026, 1, 1, tzinfo=timezone.utc)
        for i in range(days):
            price = mean_price + amplitude * math.sin(2 * math.pi * i / 10)
            bars.append(PriceBar(
                timestamp=now + timedelta(days=i),
                open=price,
                high=price * 1.003,
                low=price * 0.997,
                close=price,
                volume=500_000.0,
            ))
        return bars

    # ── Core backtest loop ─────────────────────────────────────────────────────

    def run(
        self,
        bars: List[PriceBar],
        strategy: str = "trend",
        token: str = "ETH",
    ) -> List[BacktestTrade]:
        """
        Replay bars with the given strategy.

        Returns list of completed trades.
        """
        if strategy not in STRATEGY_MAP:
            raise ValueError(f"Unknown strategy '{strategy}'. Choose: {list(STRATEGY_MAP)}")
        if not bars:
            return []

        signal_fn = STRATEGY_MAP[strategy]
        trades: List[BacktestTrade] = []
        capital = self.initial_capital
        trade_id = 0
        in_position: Optional[Tuple[str, float, datetime, int]] = None  # (side, entry, ts, bar_idx)

        for idx, bar in enumerate(bars):
            # Force-close if held too long
            if in_position and (idx - in_position[3]) >= self.max_hold_bars:
                side, entry_price, entry_ts, entry_idx = in_position
                exit_price = bar.close * (1 - self.slippage_pct)
                size = capital * self.position_size_pct / entry_price
                pnl = self._calc_pnl(side, entry_price, exit_price, size)
                fee = entry_price * size * self.fee_pct + exit_price * size * self.fee_pct
                net = pnl - fee
                capital += net
                trades.append(BacktestTrade(
                    trade_id=trade_id,
                    token=token,
                    side=side,
                    entry_price=entry_price,
                    exit_price=exit_price,
                    size=size,
                    entry_ts=entry_ts,
                    exit_ts=bar.timestamp,
                    pnl_usdc=net,
                    pnl_pct=net / (entry_price * size) if entry_price * size > 0 else 0,
                    held_bars=idx - entry_idx,
                ))
                trade_id += 1
                in_position = None

            signal = signal_fn(bars, idx)
            if signal and not in_position:
                entry_price = bar.close * (1 + self.slippage_pct)
                in_position = (signal, entry_price, bar.timestamp, idx)

        # Close any open position at final bar
        if in_position and len(bars) > 0:
            side, entry_price, entry_ts, entry_idx = in_position
            bar = bars[-1]
            exit_price = bar.close * (1 - self.slippage_pct)
            size = capital * self.position_size_pct / entry_price
            pnl = self._calc_pnl(side, entry_price, exit_price, size)
            fee = entry_price * size * self.fee_pct + exit_price * size * self.fee_pct
            net = pnl - fee
            capital += net
            trades.append(BacktestTrade(
                trade_id=trade_id,
                token=token,
                side=side,
                entry_price=entry_price,
                exit_price=exit_price,
                size=size,
                entry_ts=entry_ts,
                exit_ts=bar.timestamp,
                pnl_usdc=net,
                pnl_pct=net / (entry_price * size) if entry_price * size > 0 else 0,
                held_bars=len(bars) - 1 - entry_idx,
            ))

        logger.info(f"Backtest complete: {len(trades)} trades, strategy={strategy}")
        return trades

    def _calc_pnl(self, side: str, entry: float, exit_: float, size: float) -> float:
        if side == "BUY":
            return (exit_ - entry) * size
        else:
            return (entry - exit_) * size

    # ── Statistics ────────────────────────────────────────────────────────────

    def compute_stats(
        self,
        trades: List[BacktestTrade],
        periods_per_year: int = 252,
    ) -> BacktestStats:
        """Compute aggregate performance statistics from a list of trades."""
        stats = BacktestStats(total_trades=len(trades))
        if not trades:
            stats.final_capital = self.initial_capital
            return stats

        pnls = [t.pnl_usdc for t in trades]
        winners = [p for p in pnls if p > 0]
        losers = [p for p in pnls if p <= 0]

        stats.winning_trades = len(winners)
        stats.losing_trades = len(losers)
        stats.gross_profit = sum(winners)
        stats.gross_loss = abs(sum(losers))
        stats.net_pnl = sum(pnls)
        stats.final_capital = self.initial_capital + stats.net_pnl
        stats.win_rate = len(winners) / len(trades)
        stats.profit_factor = stats.gross_profit / stats.gross_loss if stats.gross_loss > 0 else float("inf")
        stats.avg_win = sum(winners) / len(winners) if winners else 0.0
        stats.avg_loss = sum(losers) / len(losers) if losers else 0.0
        stats.best_trade = max(pnls)
        stats.worst_trade = min(pnls)
        stats.total_return_pct = stats.net_pnl / self.initial_capital * 100
        stats.annualised_return_pct = stats.total_return_pct * (periods_per_year / max(len(trades), 1))

        # Sharpe ratio
        if len(pnls) > 1:
            mean_r = sum(pnls) / len(pnls)
            variance = sum((p - mean_r) ** 2 for p in pnls) / (len(pnls) - 1)
            std_r = math.sqrt(variance) if variance > 0 else 1e-9
            stats.sharpe_ratio = (mean_r / std_r) * math.sqrt(periods_per_year)
        else:
            stats.sharpe_ratio = 0.0

        # Sortino ratio (downside deviation only)
        downside = [p for p in pnls if p < 0]
        if downside and len(pnls) > 1:
            mean_r = sum(pnls) / len(pnls)
            down_var = sum(p ** 2 for p in downside) / len(pnls)
            down_std = math.sqrt(down_var) if down_var > 0 else 1e-9
            stats.sortino_ratio = (mean_r / down_std) * math.sqrt(periods_per_year)

        # Max drawdown
        stats.max_drawdown_pct = self._max_drawdown(pnls)

        return stats

    def _max_drawdown(self, pnls: List[float]) -> float:
        """Maximum peak-to-trough drawdown as a percentage of peak capital."""
        capital = self.initial_capital
        peak = capital
        max_dd = 0.0
        for p in pnls:
            capital += p
            if capital > peak:
                peak = capital
            dd = (peak - capital) / peak * 100 if peak > 0 else 0
            if dd > max_dd:
                max_dd = dd
        return max_dd

    def compare_strategies(
        self,
        bars: List[PriceBar],
        token: str = "ETH",
    ) -> Dict[str, BacktestStats]:
        """Run all strategies against the same bars and return stats."""
        results: Dict[str, BacktestStats] = {}
        for name in STRATEGY_MAP:
            trades = self.run(bars, strategy=name, token=token)
            results[name] = self.compute_stats(trades)
        return results

    def replay_trades(
        self,
        trades: List[BacktestTrade],
    ) -> List[float]:
        """Return the equity curve (running capital) after each trade."""
        equity = [self.initial_capital]
        capital = self.initial_capital
        for t in trades:
            capital += t.pnl_usdc
            equity.append(capital)
        return equity
