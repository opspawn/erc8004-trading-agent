"""
pipeline.py — Integration Orchestrator for ERC-8004 Trading Agent.

Wires together all modules into a single runnable tick loop:
    market_feed → strategy_engine → risk_manager → paper_trader

The Pipeline class manages lifecycle (start/stop/status) and exposes
the last N trades and current P&L for the API layer.

Usage:
    pipeline = Pipeline()
    await pipeline.start()
    # ... runs indefinitely ...
    await pipeline.stop()
    status = pipeline.status()
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from loguru import logger

from market_feed import GBMSimulator, PriceTick
from strategy_engine import StrategyEngine, StrategySignal
from risk_manager import RiskManager, RiskConfig
from paper_trader import PaperTrader, PaperTrade


# ─── State & Config ───────────────────────────────────────────────────────────

SYMBOLS = ["BTC", "ETH", "SOL"]
MAX_TRADE_HISTORY = 200
TICK_INTERVAL_SECONDS = 1.0     # seconds between pipeline ticks


@dataclass
class PipelineConfig:
    """Configuration for the integration pipeline."""
    symbols: List[str] = field(default_factory=lambda: list(SYMBOLS))
    tick_interval: float = TICK_INTERVAL_SECONDS
    initial_capital: float = 10_000.0
    max_trade_history: int = MAX_TRADE_HISTORY
    # Risk params
    max_position_pct: float = 0.10
    max_daily_drawdown_pct: float = 0.05
    # Strategy params
    sentiment_score: float = 0.0
    seed: Optional[int] = None


@dataclass
class TradeRecord:
    """A single executed trade captured in the pipeline history."""
    symbol: str
    action: str                 # "buy" | "sell"
    confidence: float
    size_usdc: float
    price: float
    pnl_usdc: float
    strategy: str
    executed_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "symbol": self.symbol,
            "action": self.action,
            "confidence": self.confidence,
            "size_usdc": round(self.size_usdc, 4),
            "price": round(self.price, 4),
            "pnl_usdc": round(self.pnl_usdc, 4),
            "strategy": self.strategy,
            "executed_at": self.executed_at,
        }


@dataclass
class PipelineStatus:
    """Current observable state of the pipeline."""
    state: str                      # "stopped" | "running" | "error"
    ticks: int = 0
    trades: int = 0
    portfolio_value: float = 0.0
    total_pnl: float = 0.0
    started_at: Optional[str] = None
    stopped_at: Optional[str] = None
    last_error: Optional[str] = None
    symbols: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "state": self.state,
            "ticks": self.ticks,
            "trades": self.trades,
            "portfolio_value": round(self.portfolio_value, 2),
            "total_pnl": round(self.total_pnl, 2),
            "started_at": self.started_at,
            "stopped_at": self.stopped_at,
            "last_error": self.last_error,
            "symbols": self.symbols,
        }


# ─── Pipeline ─────────────────────────────────────────────────────────────────

class Pipeline:
    """
    Integration orchestrator that runs the tick loop.

    Lifecycle:
        await pipeline.start()   # spawns background task
        await pipeline.stop()    # graceful shutdown
        pipeline.status()        # returns PipelineStatus
        pipeline.get_trades()    # returns last N TradeRecords
    """

    def __init__(self, config: Optional[PipelineConfig] = None) -> None:
        self.config = config or PipelineConfig()

        # Sub-modules
        self._feed: Optional[GBMSimulator] = None
        self._engine: Optional[StrategyEngine] = None
        self._risk: Optional[RiskManager] = None
        self._trader: Optional[PaperTrader] = None

        # Runtime state
        self._task: Optional[asyncio.Task] = None
        self._stop_event = asyncio.Event()
        self._trade_history: List[TradeRecord] = []
        self._portfolio_value: float = self.config.initial_capital
        self._total_pnl: float = 0.0
        self._ticks: int = 0
        self._started_at: Optional[str] = None
        self._stopped_at: Optional[str] = None
        self._last_error: Optional[str] = None
        self._state: str = "stopped"

        # Price history for strategy evaluation (per symbol)
        self._price_history: Dict[str, List[float]] = {
            sym: [] for sym in self.config.symbols
        }

        self._lock = asyncio.Lock()

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    async def start(self) -> bool:
        """Start the pipeline. Returns True on success, False if already running."""
        async with self._lock:
            if self._state == "running":
                logger.warning("Pipeline already running")
                return False

            self._init_modules()
            self._stop_event.clear()
            self._state = "running"
            self._started_at = datetime.now(timezone.utc).isoformat()
            self._stopped_at = None
            self._last_error = None
            self._ticks = 0

            self._task = asyncio.create_task(self._run_loop(), name="pipeline-loop")
            self._task.add_done_callback(self._on_task_done)
            logger.info("Pipeline started with symbols={}", self.config.symbols)
            return True

    async def stop(self) -> bool:
        """Stop the pipeline gracefully. Returns True if stopped, False if not running."""
        async with self._lock:
            if self._state != "running":
                logger.warning("Pipeline is not running (state={})", self._state)
                return False

            self._stop_event.set()

        # Wait for task outside the lock
        if self._task and not self._task.done():
            try:
                await asyncio.wait_for(asyncio.shield(self._task), timeout=5.0)
            except (asyncio.TimeoutError, asyncio.CancelledError):
                self._task.cancel()
                logger.warning("Pipeline loop forcefully cancelled")

        async with self._lock:
            self._state = "stopped"
            self._stopped_at = datetime.now(timezone.utc).isoformat()
            logger.info("Pipeline stopped at {}", self._stopped_at)
            return True

    def stop_sync(self) -> None:
        """Synchronous stop signal (sets stop event without awaiting)."""
        self._stop_event.set()
        self._state = "stopped"
        self._stopped_at = datetime.now(timezone.utc).isoformat()

    def is_running(self) -> bool:
        return self._state == "running"

    def status(self) -> PipelineStatus:
        """Return current pipeline status snapshot."""
        return PipelineStatus(
            state=self._state,
            ticks=self._ticks,
            trades=len(self._trade_history),
            portfolio_value=self._portfolio_value,
            total_pnl=self._total_pnl,
            started_at=self._started_at,
            stopped_at=self._stopped_at,
            last_error=self._last_error,
            symbols=list(self.config.symbols),
        )

    def get_trades(self, limit: int = 50) -> List[TradeRecord]:
        """Return last `limit` trades (most recent last)."""
        return self._trade_history[-limit:]

    # ── Internal ──────────────────────────────────────────────────────────────

    def _init_modules(self) -> None:
        """Instantiate sub-modules fresh for each pipeline run."""
        self._feed = GBMSimulator()
        self._engine = StrategyEngine()
        self._risk = RiskManager(
            RiskConfig(
                max_position_pct=self.config.max_position_pct,
                max_daily_drawdown_pct=self.config.max_daily_drawdown_pct,
            )
        )
        self._trader = PaperTrader(
            initial_capital=self.config.initial_capital,
            seed=self.config.seed,
        )
        self._portfolio_value = self.config.initial_capital
        self._total_pnl = 0.0
        self._price_history = {sym: [] for sym in self.config.symbols}
        self._trade_history.clear()

    async def _run_loop(self) -> None:
        """Main tick loop. Runs until stop event set or error."""
        logger.debug("Pipeline tick loop started")
        try:
            while not self._stop_event.is_set():
                tick_start = time.monotonic()

                await self._process_tick()

                elapsed = time.monotonic() - tick_start
                sleep_time = max(0.0, self.config.tick_interval - elapsed)
                if sleep_time > 0:
                    try:
                        await asyncio.wait_for(
                            self._stop_event.wait(), timeout=sleep_time
                        )
                    except asyncio.TimeoutError:
                        pass  # normal — just means tick interval elapsed
                else:
                    # Always yield at least once per tick so other tasks
                    # (e.g. stop()) can acquire locks and run.
                    await asyncio.sleep(0)

        except Exception as exc:
            logger.exception("Pipeline loop error: {}", exc)
            self._last_error = str(exc)
            self._state = "error"

        logger.debug("Pipeline tick loop exited (state={})", self._state)

    async def _process_tick(self) -> None:
        """Single tick: feed → strategy → risk → trade."""
        self._ticks += 1

        for symbol in self.config.symbols:
            if self._stop_event.is_set():
                break

            try:
                await self._process_symbol_tick(symbol)
            except Exception as exc:
                # Per-symbol errors are isolated; log and continue
                logger.warning("Tick error for {}: {}", symbol, exc)

    async def _process_symbol_tick(self, symbol: str) -> None:
        """Process a single symbol for one tick."""
        assert self._feed is not None
        assert self._engine is not None
        assert self._risk is not None

        # 1. Feed: get next simulated price
        price = self._feed.next_price(symbol)

        # 2. Accumulate price history
        history = self._price_history[symbol]
        history.append(price)
        if len(history) > 200:
            history.pop(0)

        # Need at least 25 prices before evaluating
        if len(history) < 25:
            return

        # 3. Strategy Engine: evaluate signals
        signal: StrategySignal = self._engine.evaluate(
            history, sentiment_score=self.config.sentiment_score
        )

        if signal.action == "hold" or signal.confidence <= 0.55:
            return

        # 4. Risk check
        trade_size = self._portfolio_value * self.config.max_position_pct
        ok, reason = self._risk.validate_trade(
            side=signal.action.upper(),
            size=trade_size,
            price=price,
            portfolio_value=self._portfolio_value,
        )

        if not ok:
            logger.debug("Risk rejected {}/{}: {}", symbol, signal.action, reason)
            return

        # 5. Simulate trade outcome (simple ±price move)
        # In a paper trading context we approximate PnL on the next tick
        pnl_pct = (signal.confidence - 0.5) * 0.02  # scaled confidence → pnl
        pnl_usdc = trade_size * pnl_pct

        # Update portfolio
        self._portfolio_value += pnl_usdc
        self._total_pnl += pnl_usdc

        # Record trade
        record = TradeRecord(
            symbol=symbol,
            action=signal.action,
            confidence=signal.confidence,
            size_usdc=trade_size,
            price=price,
            pnl_usdc=pnl_usdc,
            strategy=signal.strategy_name,
        )

        self._trade_history.append(record)
        if len(self._trade_history) > self.config.max_trade_history:
            self._trade_history.pop(0)

        # Record P&L in risk manager
        self._risk.record_trade_pnl(pnl_usdc)

        logger.debug(
            "Executed {}/{} @ ${:.2f} | pnl={:.4f} | strategy={}",
            symbol, signal.action, price, pnl_usdc, signal.strategy_name,
        )

    def _on_task_done(self, task: asyncio.Task) -> None:
        """Callback when the background task finishes."""
        try:
            exc = task.exception()
            if exc:
                logger.error("Pipeline task raised: {}", exc)
                self._last_error = str(exc)
                self._state = "error"
        except asyncio.CancelledError:
            pass

    # ── Convenience ───────────────────────────────────────────────────────────

    async def run_n_ticks(self, n: int) -> None:
        """Run exactly N ticks synchronously (useful for tests)."""
        self._init_modules()
        self._state = "running"
        self._started_at = datetime.now(timezone.utc).isoformat()
        for _ in range(n):
            await self._process_tick()
        self._state = "stopped"
        self._stopped_at = datetime.now(timezone.utc).isoformat()

    def reset(self) -> None:
        """Reset pipeline to initial state (stops if running)."""
        self._stop_event.set()
        self._state = "stopped"
        self._ticks = 0
        self._trade_history.clear()
        self._total_pnl = 0.0
        self._portfolio_value = self.config.initial_capital
        self._last_error = None
        self._started_at = None
        self._stopped_at = None
        self._price_history = {sym: [] for sym in self.config.symbols}
