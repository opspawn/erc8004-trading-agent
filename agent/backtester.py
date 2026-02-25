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


def mesh_consensus_signal(bars: List[PriceBar], idx: int, lookback: int = 10) -> Optional[str]:
    """
    Mesh consensus signal — aggregates trend, mean_reversion, and momentum signals.

    Inspired by ERC-8004 mesh coordination: multiple signal sources vote,
    consensus above a threshold triggers a trade.

    Returns "BUY" if 2+ signals agree on BUY, "SELL" if 2+ agree on SELL,
    None if signals are split or insufficient history.
    """
    if idx < max(lookback, 20):   # need enough bars for all sub-signals
        return None

    trend = trend_signal(bars, idx, lookback=lookback)
    mr = mean_reversion_signal(bars, idx, lookback=lookback)
    mom = momentum_signal(bars, idx, lookback=min(lookback, 5))

    votes = [trend, mr, mom]
    buys = sum(1 for v in votes if v == "BUY")
    sells = sum(1 for v in votes if v == "SELL")

    if buys >= 2:
        return "BUY"
    elif sells >= 2:
        return "SELL"
    return None


STRATEGY_MAP: Dict[str, Callable] = {
    "trend": trend_signal,
    "mean_reversion": mean_reversion_signal,
    "momentum": momentum_signal,
    "mesh_consensus": mesh_consensus_signal,
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


# ─── Backtest Registry ────────────────────────────────────────────────────────

@dataclass
class BacktestRecord:
    """Immutable record of a completed backtest run (on-chain analog)."""
    record_id: str
    strategy: str
    token: str
    days: int
    initial_capital: float
    final_capital: float
    sharpe_ratio: float
    max_drawdown_pct: float
    win_rate: float
    total_trades: int
    net_pnl: float
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> dict:
        return {
            "record_id": self.record_id,
            "strategy": self.strategy,
            "token": self.token,
            "days": self.days,
            "initial_capital": round(self.initial_capital, 4),
            "final_capital": round(self.final_capital, 4),
            "sharpe_ratio": round(self.sharpe_ratio, 4),
            "max_drawdown_pct": round(self.max_drawdown_pct, 4),
            "win_rate": round(self.win_rate, 4),
            "total_trades": self.total_trades,
            "net_pnl": round(self.net_pnl, 4),
            "timestamp": self.timestamp,
        }


class BacktestRegistry:
    """
    In-memory registry for backtest results — on-chain analog (mock).

    Mimics the ERC-8004 BacktestRegistry smart contract interface:
      - store(record) → records result with unique ID
      - get(record_id) → retrieve by ID
      - query(strategy=..., token=...) → filter records
      - best(metric="sharpe_ratio") → find top performer

    In a production deployment this would write to a smart contract on Hedera/EVM.
    """

    def __init__(self) -> None:
        self._records: Dict[str, BacktestRecord] = {}
        self._counter: int = 0
        logger.info("BacktestRegistry initialized (mock on-chain registry)")

    def store(
        self,
        backtester: "Backtester",
        trades: List[BacktestTrade],
        stats: BacktestStats,
        strategy: str,
        token: str,
        days: int,
    ) -> BacktestRecord:
        """Store a backtest result and return the registry record."""
        self._counter += 1
        record_id = f"BT-{self._counter:06d}"
        record = BacktestRecord(
            record_id=record_id,
            strategy=strategy,
            token=token,
            days=days,
            initial_capital=backtester.initial_capital,
            final_capital=stats.final_capital,
            sharpe_ratio=stats.sharpe_ratio,
            max_drawdown_pct=stats.max_drawdown_pct,
            win_rate=stats.win_rate,
            total_trades=stats.total_trades,
            net_pnl=stats.net_pnl,
        )
        self._records[record_id] = record
        logger.info(
            f"BacktestRegistry: stored {record_id} "
            f"strategy={strategy} sharpe={stats.sharpe_ratio:.3f}"
        )
        return record

    def get(self, record_id: str) -> Optional[BacktestRecord]:
        """Retrieve a backtest record by ID."""
        return self._records.get(record_id)

    def query(
        self,
        strategy: Optional[str] = None,
        token: Optional[str] = None,
    ) -> List[BacktestRecord]:
        """Filter registry records by strategy and/or token."""
        results = list(self._records.values())
        if strategy:
            results = [r for r in results if r.strategy == strategy]
        if token:
            results = [r for r in results if r.token == token]
        return results

    def best(
        self,
        metric: str = "sharpe_ratio",
        strategy: Optional[str] = None,
    ) -> Optional[BacktestRecord]:
        """Return the record with the highest value for the given metric."""
        records = self.query(strategy=strategy)
        if not records:
            return None
        valid_metrics = {
            "sharpe_ratio", "win_rate", "net_pnl",
            "final_capital", "total_trades",
        }
        if metric not in valid_metrics:
            raise ValueError(f"Unknown metric {metric!r}. Choose: {valid_metrics}")
        return max(records, key=lambda r: getattr(r, metric))

    def worst(
        self,
        metric: str = "max_drawdown_pct",
        strategy: Optional[str] = None,
    ) -> Optional[BacktestRecord]:
        """Return the record with the highest drawdown (or worst value)."""
        records = self.query(strategy=strategy)
        if not records:
            return None
        return max(records, key=lambda r: getattr(r, metric))

    def all_records(self) -> List[BacktestRecord]:
        """Return all stored records sorted by timestamp descending."""
        return sorted(self._records.values(), key=lambda r: r.timestamp, reverse=True)

    def count(self) -> int:
        """Total number of stored records."""
        return len(self._records)

    def clear(self) -> None:
        """Clear all records (used for testing)."""
        self._records.clear()
        self._counter = 0
        logger.info("BacktestRegistry: cleared all records")

    def summary(self) -> Dict[str, float]:
        """Return aggregate statistics across all stored records."""
        records = list(self._records.values())
        if not records:
            return {"count": 0}
        sharpes = [r.sharpe_ratio for r in records]
        drawdowns = [r.max_drawdown_pct for r in records]
        win_rates = [r.win_rate for r in records]
        return {
            "count": len(records),
            "avg_sharpe": sum(sharpes) / len(sharpes),
            "max_sharpe": max(sharpes),
            "avg_drawdown": sum(drawdowns) / len(drawdowns),
            "max_drawdown": max(drawdowns),
            "avg_win_rate": sum(win_rates) / len(win_rates),
        }


# ─── CLI Entry Point ──────────────────────────────────────────────────────────

def _cli_main() -> None:
    """
    Command-line interface for running backtests.

    Usage:
        python -m agent.backtester --days 30 --strategy mesh_consensus
        python -m agent.backtester --days 60 --strategy trend --token BTC --capital 5000
    """
    import argparse

    parser = argparse.ArgumentParser(
        description="ERC-8004 Trading Agent — Backtester CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--days", type=int, default=30, help="Backtest window in days")
    parser.add_argument(
        "--strategy",
        default="mesh_consensus",
        choices=list(STRATEGY_MAP.keys()),
        help="Trading strategy to backtest",
    )
    parser.add_argument("--token", default="ETH", help="Token symbol")
    parser.add_argument("--capital", type=float, default=1000.0, help="Initial capital (USDC)")
    parser.add_argument("--position-size", type=float, default=0.10, help="Position size fraction")
    parser.add_argument("--volatility", type=float, default=0.02, help="Simulated volatility")
    parser.add_argument("--drift", type=float, default=0.0005, help="Simulated price drift")
    parser.add_argument(
        "--mode",
        default="gbm",
        choices=["gbm", "trending", "mean_reverting"],
        help="Price simulation mode",
    )
    parser.add_argument("--register", action="store_true", help="Store result in BacktestRegistry")
    parser.add_argument("--compare", action="store_true", help="Compare all strategies")

    args = parser.parse_args()

    bt = Backtester(
        initial_capital=args.capital,
        position_size_pct=args.position_size,
    )

    # Generate price data
    if args.mode == "trending":
        bars = bt.generate_trending_prices(days=args.days)
    elif args.mode == "mean_reverting":
        bars = bt.generate_mean_reverting_prices(days=args.days)
    else:
        bars = bt.generate_synthetic_prices(
            token=args.token,
            days=args.days,
            volatility=args.volatility,
            drift=args.drift,
        )

    if args.compare:
        # Compare all strategies
        print(f"\n{'='*60}")
        print(f"Strategy Comparison | {args.token} | {args.days} days")
        print(f"{'='*60}")
        results = bt.compare_strategies(bars, token=args.token)
        for name, stats in results.items():
            print(
                f"{name:20s}  sharpe={stats.sharpe_ratio:6.3f}  "
                f"dd={stats.max_drawdown_pct:5.1f}%  "
                f"wr={stats.win_rate:4.1%}  "
                f"pnl=${stats.net_pnl:+8.2f}"
            )
        print()
        return

    # Single strategy backtest
    trades = bt.run(bars, strategy=args.strategy, token=args.token)
    stats = bt.compute_stats(trades)

    print(f"\n{'='*60}")
    print(f"Backtest Results | strategy={args.strategy} | {args.token} | {args.days}d")
    print(f"{'='*60}")
    print(f"  Initial capital  : ${stats.final_capital - stats.net_pnl:,.2f}")
    print(f"  Final capital    : ${stats.final_capital:,.2f}")
    print(f"  Net P&L          : ${stats.net_pnl:+,.2f}")
    print(f"  Total return     : {stats.total_return_pct:+.2f}%")
    print(f"  Sharpe ratio     : {stats.sharpe_ratio:.4f}")
    print(f"  Max drawdown     : {stats.max_drawdown_pct:.2f}%")
    print(f"  Win rate         : {stats.win_rate:.1%}")
    print(f"  Total trades     : {stats.total_trades}")
    print(f"  Profit factor    : {stats.profit_factor:.2f}")
    print()

    if args.register:
        registry = BacktestRegistry()
        record = registry.store(bt, trades, stats, args.strategy, args.token, args.days)
        print(f"Stored in BacktestRegistry as {record.record_id}")


if __name__ == "__main__":
    _cli_main()
