"""
paper_trader.py — Paper Trading Simulation for ERC-8004 Trading Agent.

Runs a 24-hour simulation of the full agent trading loop using synthetic
price data (Geometric Brownian Motion) in ~1 second of wall clock time.

Pipeline per tick:
  1. Generate synthetic price bar (GBM)
  2. Run mesh coordinator consensus check
  3. Apply risk manager constraints
  4. Execute paper trade (no real funds)
  5. Update agent reputation
  6. Record P&L

At the end produces a SimulationReport with Sharpe ratio, drawdown, win
rate, number of trades, and final portfolio value.

Usage:
    sim = PaperTrader(initial_capital=10_000.0, seed=42)
    report = sim.run()
    print(report.summary())
"""

from __future__ import annotations

import math
import random
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional


# ─── GBM Price Generator ──────────────────────────────────────────────────────

class GBMPriceGenerator:
    """
    Geometric Brownian Motion price generator.

    dS = mu*S*dt + sigma*S*dW
    where dW ~ N(0, sqrt(dt))

    Pure math — no numpy required.
    """

    def __init__(
        self,
        initial_price: float = 1.0,
        mu: float = 0.05,       # annualised drift (5%)
        sigma: float = 0.20,    # annualised volatility (20%)
        dt: float = 1 / 8760,   # one hour in years (24*365=8760)
        seed: Optional[int] = None,
    ) -> None:
        self.price = initial_price
        self.mu = mu
        self.sigma = sigma
        self.dt = dt
        self._rng = random.Random(seed)
        self._step = 0

    def _normal(self) -> float:
        """Box-Muller transform for standard normal variate."""
        while True:
            u = self._rng.uniform(0.0, 1.0)
            v = self._rng.uniform(0.0, 1.0)
            if u > 0.0:
                break
        mag = math.sqrt(-2.0 * math.log(u))
        return mag * math.cos(2.0 * math.pi * v)

    def next_price(self) -> float:
        """Advance one time step using GBM formula."""
        z = self._normal()
        drift = (self.mu - 0.5 * self.sigma ** 2) * self.dt
        diffusion = self.sigma * math.sqrt(self.dt) * z
        self.price *= math.exp(drift + diffusion)
        self._step += 1
        return self.price

    def generate(self, n: int) -> list[float]:
        """Generate n price steps."""
        return [self.next_price() for _ in range(n)]

    @property
    def steps(self) -> int:
        return self._step


# ─── Trade Record ─────────────────────────────────────────────────────────────

@dataclass
class PaperTrade:
    """A single paper trade with full audit trail."""
    trade_id: int
    token: str
    side: str           # "BUY" or "SELL"
    entry_price: float
    exit_price: float
    size: float         # position size (units of token)
    size_usdc: float    # position value in USDC
    pnl: float          # realised P&L in USDC
    pnl_pct: float      # P&L as fraction of position
    entry_ts: float     # unix timestamp (simulated)
    exit_ts: float
    mesh_consensus: bool
    agent_id: str
    protocol: str
    confidence: float

    @property
    def is_winner(self) -> bool:
        return self.pnl > 0.0

    def to_dict(self) -> dict:
        return {
            "trade_id": self.trade_id,
            "token": self.token,
            "side": self.side,
            "entry_price": round(self.entry_price, 6),
            "exit_price": round(self.exit_price, 6),
            "size": round(self.size, 6),
            "size_usdc": round(self.size_usdc, 2),
            "pnl": round(self.pnl, 4),
            "pnl_pct": round(self.pnl_pct, 6),
            "entry_ts": self.entry_ts,
            "exit_ts": self.exit_ts,
            "mesh_consensus": self.mesh_consensus,
            "agent_id": self.agent_id,
            "protocol": self.protocol,
            "confidence": round(self.confidence, 4),
            "is_winner": self.is_winner,
        }


# ─── Simulation Report ────────────────────────────────────────────────────────

@dataclass
class SimulationReport:
    """Aggregated results of a 24-hour paper trading simulation."""
    initial_capital: float
    final_portfolio_value: float
    total_pnl: float
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    sharpe_ratio: float
    max_drawdown_pct: float
    ticks_simulated: int
    simulation_time_secs: float
    trades: list[PaperTrade] = field(default_factory=list)
    agent_reputation_updates: list[dict] = field(default_factory=list)

    @property
    def total_return_pct(self) -> float:
        if self.initial_capital == 0:
            return 0.0
        return (self.final_portfolio_value - self.initial_capital) / self.initial_capital

    def summary(self) -> str:
        lines = [
            "=" * 60,
            "  ERC-8004 Paper Trading Simulation — 24h Report",
            "=" * 60,
            f"  Initial Capital:    ${self.initial_capital:,.2f}",
            f"  Final Portfolio:    ${self.final_portfolio_value:,.2f}",
            f"  Total P&L:          ${self.total_pnl:+.2f}",
            f"  Return:             {self.total_return_pct:.2%}",
            f"  Total Trades:       {self.total_trades}",
            f"  Win Rate:           {self.win_rate:.2%}",
            f"  Sharpe Ratio:       {self.sharpe_ratio:.4f}",
            f"  Max Drawdown:       {self.max_drawdown_pct:.2%}",
            f"  Ticks Simulated:    {self.ticks_simulated}",
            f"  Sim Time (wall):    {self.simulation_time_secs:.3f}s",
            "=" * 60,
        ]
        return "\n".join(lines)

    def to_dict(self) -> dict:
        return {
            "initial_capital": self.initial_capital,
            "final_portfolio_value": round(self.final_portfolio_value, 2),
            "total_pnl": round(self.total_pnl, 4),
            "total_return_pct": round(self.total_return_pct, 6),
            "total_trades": self.total_trades,
            "winning_trades": self.winning_trades,
            "losing_trades": self.losing_trades,
            "win_rate": round(self.win_rate, 4),
            "sharpe_ratio": round(self.sharpe_ratio, 4),
            "max_drawdown_pct": round(self.max_drawdown_pct, 4),
            "ticks_simulated": self.ticks_simulated,
            "simulation_time_secs": round(self.simulation_time_secs, 4),
            "reputation_updates": len(self.agent_reputation_updates),
        }


# ─── Paper Trader ─────────────────────────────────────────────────────────────

PROTOCOLS = ["ETH", "BTC", "AAVE", "UNI", "COMP", "CRV", "GMX"]

class PaperTrader:
    """
    24-hour paper trading simulation with synthetic GBM prices.

    Runs the full ERC-8004 trading loop:
      market_data → mesh consensus → risk check → execute → update reputation

    The simulation runs in ~1 second: no asyncio, no real network calls,
    no sleep. Synthetic ticks replace real market data.
    """

    TICKS_PER_24H = 288       # one tick every 5 minutes → 288 ticks/day
    TICK_INTERVAL_SECS = 300  # 5 minutes in seconds (for Sharpe annualisation)

    def __init__(
        self,
        initial_capital: float = 10_000.0,
        coordinator=None,
        seed: Optional[int] = None,
        position_size_pct: float = 0.05,  # 5% of portfolio per trade
        min_confidence: float = 0.55,     # minimum mesh confidence to trade
    ) -> None:
        self.initial_capital = initial_capital
        self.portfolio_value = initial_capital
        self.position_size_pct = position_size_pct
        self.min_confidence = min_confidence
        self._seed = seed
        self._rng = random.Random(seed)

        # Lazy import to avoid circular deps
        if coordinator is None:
            from mesh_coordinator import MeshCoordinator
            self._coordinator = MeshCoordinator()
        else:
            self._coordinator = coordinator

        self._trades: list[PaperTrade] = []
        self._pnl_series: list[float] = []     # per-trade P&L for Sharpe
        self._portfolio_series: list[float] = []  # portfolio value series for drawdown
        self._reputation_updates: list[dict] = []
        self._trade_id = 0

    # ─── Internal Helpers ─────────────────────────────────────────────────────

    def _kelly_size(self, confidence: float) -> float:
        """Conservative Kelly sizing: f* = confidence - (1-confidence)."""
        kelly_full = max(0.0, 2 * confidence - 1.0)
        kelly_fraction = kelly_full * 0.25  # quarter-Kelly
        return min(self.position_size_pct, kelly_fraction)

    def _compute_pnl(
        self, side: str, entry: float, exit_: float, size_usdc: float
    ) -> float:
        """Compute realised P&L for a trade (fees = 0.1% round-trip)."""
        if side == "BUY":
            gross = (exit_ - entry) / entry * size_usdc
        else:  # SELL / SHORT
            gross = (entry - exit_) / entry * size_usdc
        fees = size_usdc * 0.001  # 0.1% total fee
        return gross - fees

    def _sharpe_ratio(self, pnl_series: list[float]) -> float:
        """Annualised Sharpe ratio from a P&L series."""
        n = len(pnl_series)
        if n < 2:
            return 0.0
        mean = sum(pnl_series) / n
        variance = sum((x - mean) ** 2 for x in pnl_series) / (n - 1)
        std = math.sqrt(variance) if variance > 0 else 0.0
        if std == 0.0:
            return 0.0
        # Annualise: ticks per year = 365*24*60/5 = 105120
        ticks_per_year = 105_120
        return (mean / std) * math.sqrt(ticks_per_year)

    def _max_drawdown(self, portfolio_series: list[float]) -> float:
        """Maximum peak-to-trough drawdown as a fraction."""
        if not portfolio_series:
            return 0.0
        peak = portfolio_series[0]
        max_dd = 0.0
        for v in portfolio_series:
            if v > peak:
                peak = v
            dd = (peak - v) / peak if peak > 0 else 0.0
            if dd > max_dd:
                max_dd = dd
        return max_dd

    def _update_reputation(self, agent_id: str, won: bool, pnl_pct: float) -> dict:
        """Update agent reputation (simplified on-chain analogue)."""
        score = 750 if won else 350
        entry = {
            "agent_id": agent_id,
            "trade_id": self._trade_id,
            "won": won,
            "pnl_pct": round(pnl_pct, 6),
            "score": score,
            "timestamp": time.time(),
        }
        self._reputation_updates.append(entry)
        # Propagate to coordinator agent if possible
        agent = self._coordinator.get_agent(agent_id)
        if agent and hasattr(agent, "update_reputation"):
            agent.update_reputation(trade_won=won, pnl_pct=pnl_pct)
        return entry

    # ─── Main Simulation ──────────────────────────────────────────────────────

    def run(self, ticks: Optional[int] = None) -> SimulationReport:
        """
        Run the full paper trading simulation.

        Args:
            ticks: Number of ticks to simulate (default: 288 = 24h at 5min).

        Returns:
            SimulationReport with full statistics.
        """
        ticks = ticks or self.TICKS_PER_24H
        wall_start = time.perf_counter()

        # One GBM generator per protocol
        generators = {
            proto: GBMPriceGenerator(
                initial_price=self._initial_price(proto),
                mu=0.08,
                sigma=self._volatility(proto),
                dt=self.TICK_INTERVAL_SECS / (365 * 24 * 3600),
                seed=self._rng.randint(0, 2**31),
            )
            for proto in PROTOCOLS
        }

        simulated_ts = time.time() - ticks * self.TICK_INTERVAL_SECS
        self._portfolio_series.append(self.portfolio_value)

        from credora_client import CredoraRatingTier
        grades = list(CredoraRatingTier)

        for tick in range(ticks):
            simulated_ts += self.TICK_INTERVAL_SECS

            # Advance all prices
            prices = {proto: gen.next_price() for proto, gen in generators.items()}

            # Try to generate a trade signal each tick
            proto = PROTOCOLS[tick % len(PROTOCOLS)]
            price = prices[proto]
            side = "BUY" if self._rng.random() > 0.5 else "SELL"

            position_usdc = self.portfolio_value * self.position_size_pct
            edge = self._rng.uniform(0.01, 0.15)
            grade = self._rng.choice(grades[:5])  # stick to investment grade

            consensus = self._coordinator.evaluate(
                side=side,
                size=position_usdc / price if price > 0 else 1.0,
                price=price,
                portfolio_value=self.portfolio_value,
                protocol_grade=grade,
                edge=edge,
            )

            if not consensus.consensus_reached:
                continue

            confidence = consensus.approval_ratio
            if confidence < self.min_confidence:
                continue

            # Risk check: don't trade if portfolio down >15%
            max_loss_pct = (self.initial_capital - self.portfolio_value) / self.initial_capital
            if max_loss_pct > 0.15:
                continue

            # Size trade using conservative Kelly
            size_pct = self._kelly_size(confidence)
            size_usdc = self.portfolio_value * size_pct
            if size_usdc < 10.0:
                continue

            # Simulate holding for 1–3 ticks then exit
            hold_ticks = self._rng.randint(1, 3)
            exit_tick_idx = min(tick + hold_ticks, ticks - 1)
            exit_price = prices[proto]
            for _ in range(hold_ticks):
                exit_price = generators[proto].next_price()

            pnl = self._compute_pnl(side, price, exit_price, size_usdc)
            pnl_pct = pnl / size_usdc if size_usdc > 0 else 0.0
            self.portfolio_value += pnl

            self._trade_id += 1
            agent_id = consensus.votes_for[0].agent_id if consensus.votes_for else "mesh_agent"

            trade = PaperTrade(
                trade_id=self._trade_id,
                token=proto,
                side=side,
                entry_price=price,
                exit_price=exit_price,
                size=size_usdc / price if price > 0 else 1.0,
                size_usdc=size_usdc,
                pnl=pnl,
                pnl_pct=pnl_pct,
                entry_ts=simulated_ts,
                exit_ts=simulated_ts + hold_ticks * self.TICK_INTERVAL_SECS,
                mesh_consensus=True,
                agent_id=agent_id,
                protocol=proto,
                confidence=confidence,
            )
            self._trades.append(trade)
            self._pnl_series.append(pnl)
            self._portfolio_series.append(self.portfolio_value)

            # Update reputation
            self._update_reputation(agent_id, pnl > 0, pnl_pct)

        wall_elapsed = time.perf_counter() - wall_start

        # Build report
        total = len(self._trades)
        winners = [t for t in self._trades if t.is_winner]
        win_rate = len(winners) / total if total > 0 else 0.0
        total_pnl = sum(t.pnl for t in self._trades)

        return SimulationReport(
            initial_capital=self.initial_capital,
            final_portfolio_value=round(self.portfolio_value, 4),
            total_pnl=round(total_pnl, 4),
            total_trades=total,
            winning_trades=len(winners),
            losing_trades=total - len(winners),
            win_rate=round(win_rate, 4),
            sharpe_ratio=round(self._sharpe_ratio(self._pnl_series), 4),
            max_drawdown_pct=round(self._max_drawdown(self._portfolio_series), 4),
            ticks_simulated=ticks,
            simulation_time_secs=round(wall_elapsed, 4),
            trades=self._trades,
            agent_reputation_updates=self._reputation_updates,
        )

    # ─── Protocol Configuration ───────────────────────────────────────────────

    @staticmethod
    def _initial_price(protocol: str) -> float:
        defaults = {
            "ETH": 2500.0, "BTC": 45000.0, "AAVE": 120.0,
            "UNI": 8.0, "COMP": 60.0, "CRV": 0.55, "GMX": 40.0,
        }
        return defaults.get(protocol, 1.0)

    @staticmethod
    def _volatility(protocol: str) -> float:
        # Annualised volatility
        vols = {
            "ETH": 0.75, "BTC": 0.65, "AAVE": 1.20,
            "UNI": 1.10, "COMP": 1.30, "CRV": 1.40, "GMX": 1.00,
        }
        return vols.get(protocol, 0.80)


# ─── Entrypoint ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    trader = PaperTrader(initial_capital=10_000.0, seed=42)
    report = trader.run()
    print(report.summary())
