"""
execution_loop.py — Production-ready async trading loop for the ERC-8004 Trading Agent.

Orchestrates the full cycle:
  fetch market signals → strategist.analyze() → risk_manager.validate_trade()
  → trader.execute() → reputation update

Features:
- Configurable cycle interval (default 30s)
- Max concurrent trades guard
- State persistence (JSON file)
- Emergency stop on daily drawdown > 5%
- Metrics: cycles_run, trades_executed, errors_last_hour, uptime_seconds
"""

from __future__ import annotations

import asyncio
import json
import time
from collections import deque
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Deque, Optional

from loguru import logger


# ─── Data Classes ─────────────────────────────────────────────────────────────


@dataclass
class LoopConfig:
    """Configurable execution loop parameters."""
    cycle_interval_seconds: float = 30.0
    max_concurrent_trades: int = 3
    max_daily_drawdown_pct: float = 0.05    # 5% → emergency stop
    state_file: str = "execution_loop_state.json"
    emergency_alert_callback: Optional[Callable[[str], None]] = None


@dataclass
class LoopMetrics:
    """Runtime metrics for the execution loop."""
    cycles_run: int = 0
    trades_executed: int = 0
    trades_rejected: int = 0
    errors_last_hour: int = 0
    uptime_seconds: float = 0.0
    last_cycle_at: Optional[str] = None
    emergency_stops: int = 0
    started_at: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "cycles_run": self.cycles_run,
            "trades_executed": self.trades_executed,
            "trades_rejected": self.trades_rejected,
            "errors_last_hour": self.errors_last_hour,
            "uptime_seconds": round(self.uptime_seconds, 2),
            "last_cycle_at": self.last_cycle_at,
            "emergency_stops": self.emergency_stops,
            "started_at": self.started_at,
        }


@dataclass
class LoopState:
    """Persistent state saved/loaded across runs."""
    is_running: bool = False
    is_emergency_stopped: bool = False
    active_trade_count: int = 0
    daily_pnl_usdc: float = 0.0
    portfolio_value_usdc: float = 100.0
    last_saved_at: Optional[str] = None
    metrics: LoopMetrics = field(default_factory=LoopMetrics)

    def to_dict(self) -> dict:
        return {
            "is_running": self.is_running,
            "is_emergency_stopped": self.is_emergency_stopped,
            "active_trade_count": self.active_trade_count,
            "daily_pnl_usdc": self.daily_pnl_usdc,
            "portfolio_value_usdc": self.portfolio_value_usdc,
            "last_saved_at": self.last_saved_at,
            "metrics": self.metrics.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: dict) -> "LoopState":
        metrics_data = data.pop("metrics", {})
        state = cls(**{k: v for k, v in data.items() if k != "metrics"})
        state.metrics = LoopMetrics(**{
            k: v for k, v in metrics_data.items()
            if k in LoopMetrics.__dataclass_fields__
        })
        return state


# ─── Execution Loop ────────────────────────────────────────────────────────────


class ExecutionLoop:
    """
    Production-ready async trading loop.

    Usage:
        loop = ExecutionLoop(config, strategist, risk_manager, trader, reputation)
        await loop.start()
        # ... later ...
        await loop.stop()
    """

    def __init__(
        self,
        config: Optional[LoopConfig] = None,
        strategist=None,
        risk_manager=None,
        trader=None,
        reputation=None,
        market_data=None,
    ):
        self.config = config or LoopConfig()
        self.strategist = strategist
        self.risk_manager = risk_manager
        self.trader = trader
        self.reputation = reputation
        self.market_data = market_data

        self._state = LoopState()
        self._running = False
        self._task: Optional[asyncio.Task] = None
        self._start_time: Optional[float] = None
        self._error_timestamps: Deque[float] = deque(maxlen=1000)
        self._active_trades: int = 0

    # ─── Lifecycle ────────────────────────────────────────────────────────────

    async def start(self) -> None:
        """Start the execution loop."""
        if self._running:
            logger.warning("ExecutionLoop already running")
            return

        self._load_state()

        if self._state.is_emergency_stopped:
            logger.error("Cannot start: emergency stop is active. Call reset_emergency_stop() first.")
            return

        self._running = True
        self._state.is_running = True
        self._start_time = time.monotonic()
        self._state.metrics.started_at = datetime.now(timezone.utc).isoformat()
        logger.info("ExecutionLoop starting (interval={}s, max_trades={})",
                    self.config.cycle_interval_seconds, self.config.max_concurrent_trades)

        self._task = asyncio.create_task(self._loop())

    async def stop(self) -> None:
        """Gracefully stop the execution loop."""
        if not self._running:
            return

        logger.info("ExecutionLoop stopping…")
        self._running = False
        self._state.is_running = False

        if self._task and not self._task.done():
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass

        self._save_state()
        logger.info("ExecutionLoop stopped. Metrics: {}", self._state.metrics.to_dict())

    def reset_emergency_stop(self) -> None:
        """Clear emergency stop flag so the loop can restart."""
        self._state.is_emergency_stopped = False
        self._save_state()
        logger.info("Emergency stop cleared")

    # ─── Main Loop ────────────────────────────────────────────────────────────

    async def _loop(self) -> None:
        """Inner async loop."""
        while self._running:
            try:
                await self.run_cycle()
            except asyncio.CancelledError:
                break
            except Exception as exc:
                logger.exception("Unhandled error in cycle: {}", exc)
                self._record_error()

            if self._running:
                await asyncio.sleep(self.config.cycle_interval_seconds)

    async def run_cycle(self) -> dict[str, Any]:
        """
        Execute one trading cycle.

        Returns dict with cycle results for inspection/testing.
        """
        cycle_start = time.monotonic()
        result: dict[str, Any] = {
            "cycle_number": self._state.metrics.cycles_run + 1,
            "started_at": datetime.now(timezone.utc).isoformat(),
            "trades_attempted": 0,
            "trades_executed": 0,
            "trades_rejected": 0,
            "errors": [],
            "emergency_stop_triggered": False,
        }

        try:
            # ── 1. Fetch market signals ──────────────────────────────────────
            market_signals = await self._fetch_market_signals()
            if not market_signals:
                logger.debug("No market signals available this cycle")
                return result

            # ── 2. Check concurrent trade limit ─────────────────────────────
            if self._active_trades >= self.config.max_concurrent_trades:
                logger.debug("At max concurrent trades ({}), skipping", self._active_trades)
                return result

            # ── 3. Strategist analysis ───────────────────────────────────────
            for signal in market_signals:
                try:
                    decision = await self._analyze_signal(signal)
                    if decision is None or getattr(decision, "action", "hold") == "hold":
                        continue

                    result["trades_attempted"] += 1

                    # ── 4. Risk validation ────────────────────────────────────
                    if self.risk_manager:
                        size_pct = getattr(decision, "size_pct", 0.05)
                        size_usdc = self._state.portfolio_value_usdc * size_pct
                        price = signal.get("price", 0.5)
                        valid, reason = self.risk_manager.validate_trade(
                            side=getattr(decision, "action", "YES").upper(),
                            size=size_usdc,
                            price=price,
                            portfolio_value=self._state.portfolio_value_usdc,
                        )
                        if not valid:
                            logger.info("Trade rejected by risk manager: {}", reason)
                            result["trades_rejected"] += 1
                            self._state.metrics.trades_rejected += 1
                            continue

                    # ── 5. Execute trade ──────────────────────────────────────
                    trade_result = await self._execute_trade(decision, signal)
                    if trade_result:
                        result["trades_executed"] += 1
                        self._state.metrics.trades_executed += 1
                        self._state.active_trade_count += 1
                        self._active_trades += 1

                        # ── 6. Update PnL + check drawdown ───────────────────
                        pnl = getattr(trade_result, "pnl_usdc", 0.0)
                        self._state.daily_pnl_usdc += pnl
                        drawdown_triggered = self._check_emergency_stop()
                        if drawdown_triggered:
                            result["emergency_stop_triggered"] = True
                            await self.stop()
                            return result

                        # ── 7. Reputation update ──────────────────────────────
                        await self._update_reputation(trade_result)

                except Exception as exc:
                    logger.exception("Error processing signal {}: {}", signal, exc)
                    self._record_error()
                    result["errors"].append(str(exc))

        finally:
            self._state.metrics.cycles_run += 1
            self._update_uptime()
            self._state.metrics.last_cycle_at = result["started_at"]
            self._save_state()

        cycle_duration = time.monotonic() - cycle_start
        logger.debug("Cycle {} complete in {:.2f}s: {} trades executed",
                     result["cycle_number"], cycle_duration, result["trades_executed"])
        return result

    # ─── Internal Helpers ─────────────────────────────────────────────────────

    async def _fetch_market_signals(self) -> list[dict]:
        """Fetch market signals via market_data adapter or fallback."""
        if self.market_data:
            try:
                symbols = getattr(self.market_data, "tracked_symbols", ["ETH", "BTC"])
                signals = []
                for symbol in symbols:
                    price = await self.market_data.fetch_price(symbol)
                    signals.append({"symbol": symbol, "price": price})
                return signals
            except Exception as exc:
                logger.warning("market_data fetch failed: {}", exc)
                self._record_error()
                return []
        return []

    async def _analyze_signal(self, signal: dict) -> Any:
        """Run strategist analysis on a signal."""
        if self.strategist:
            try:
                return await self.strategist.decide(signal)
            except Exception as exc:
                logger.warning("Strategist error: {}", exc)
                self._record_error()
        return None

    async def _execute_trade(self, decision: Any, signal: dict) -> Any:
        """Execute a trade via trader."""
        if self.trader:
            try:
                return await self.trader.execute(decision, signal)
            except Exception as exc:
                logger.warning("Trader error: {}", exc)
                self._record_error()
        return None

    async def _update_reputation(self, trade_result: Any) -> None:
        """Push trade result to on-chain reputation."""
        if self.reputation:
            try:
                await self.reputation.log_trade(trade_result)
            except Exception as exc:
                logger.warning("Reputation update error: {}", exc)

    def _check_emergency_stop(self) -> bool:
        """
        Check if daily drawdown exceeds threshold.
        Returns True if emergency stop was triggered.
        """
        if self._state.portfolio_value_usdc <= 0:
            return False

        drawdown_pct = -self._state.daily_pnl_usdc / self._state.portfolio_value_usdc
        if drawdown_pct >= self.config.max_daily_drawdown_pct:
            logger.error(
                "EMERGENCY STOP: daily drawdown {:.1%} >= {:.1%} threshold",
                drawdown_pct, self.config.max_daily_drawdown_pct,
            )
            self._state.is_emergency_stopped = True
            self._state.is_running = False
            self._running = False
            self._state.metrics.emergency_stops += 1

            msg = f"Emergency stop: drawdown {drawdown_pct:.1%}"
            if self.config.emergency_alert_callback:
                try:
                    self.config.emergency_alert_callback(msg)
                except Exception:
                    pass
            return True
        return False

    def _record_error(self) -> None:
        """Track error timestamp for errors_last_hour metric."""
        now = time.monotonic()
        self._error_timestamps.append(now)
        cutoff = now - 3600.0
        # Recount errors within last hour
        count = sum(1 for t in self._error_timestamps if t >= cutoff)
        self._state.metrics.errors_last_hour = count

    def _update_uptime(self) -> None:
        """Update uptime_seconds metric."""
        if self._start_time is not None:
            self._state.metrics.uptime_seconds = time.monotonic() - self._start_time

    # ─── State Persistence ────────────────────────────────────────────────────

    def _save_state(self) -> None:
        """Persist loop state to JSON file."""
        self._state.last_saved_at = datetime.now(timezone.utc).isoformat()
        state_path = Path(self.config.state_file)
        try:
            state_path.write_text(json.dumps(self._state.to_dict(), indent=2))
        except Exception as exc:
            logger.warning("Could not save loop state: {}", exc)

    def _load_state(self) -> None:
        """Load previously persisted state if available."""
        state_path = Path(self.config.state_file)
        if not state_path.exists():
            logger.debug("No state file found, starting fresh")
            return
        try:
            data = json.loads(state_path.read_text())
            # Reset runtime fields on load
            data["is_running"] = False
            self._state = LoopState.from_dict(data)
            logger.info("Loaded state: {} cycles run previously",
                        self._state.metrics.cycles_run)
        except Exception as exc:
            logger.warning("Could not load state file: {}", exc)

    # ─── Public Accessors ─────────────────────────────────────────────────────

    @property
    def metrics(self) -> LoopMetrics:
        self._update_uptime()
        return self._state.metrics

    @property
    def state(self) -> LoopState:
        return self._state

    @property
    def is_running(self) -> bool:
        return self._running

    @property
    def is_emergency_stopped(self) -> bool:
        return self._state.is_emergency_stopped
