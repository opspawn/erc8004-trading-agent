"""
demo_cli.py — ERC-8004 Pipeline Dry-Run Demo CLI.

Executes a complete dry-run of the trading pipeline:
  Step 1: Market fetch   — GBM price ticks for BTC/ETH/SOL
  Step 2: Signal         — Strategy engine produces BUY/SELL/HOLD signal
  Step 3: Risk check     — Risk manager validates position size
  Step 4: Order sim      — Paper trader executes (simulated)
  Step 5: Ledger write   — SQLite ledger records decision with tx_hash

Outputs a clean human-readable trace of every step.

Usage (CLI):
    python3 demo_cli.py                  # default 10 ticks, BTC
    python3 demo_cli.py --ticks 20 --symbol ETH
    python3 demo_cli.py --json           # machine-readable JSON output

Usage (programmatic):
    from demo_cli import DemoPipeline
    result = DemoPipeline(ticks=5).run()
    print(result.trace)
"""

from __future__ import annotations

import argparse
import json
import math
import random
import time
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from trade_ledger import TradeLedger, LedgerEntry, LedgerSummary


# ─── Minimal market simulation ────────────────────────────────────────────────


def _gbm_prices(
    initial: float,
    n: int,
    mu: float = 0.05,
    sigma: float = 0.20,
    dt: float = 1 / 8760,
    seed: Optional[int] = None,
) -> List[float]:
    """Generate n GBM price ticks starting from initial."""
    rng = random.Random(seed)
    prices = [initial]
    for _ in range(n):
        z = _box_muller(rng)
        drift = (mu - 0.5 * sigma ** 2) * dt
        diffusion = sigma * math.sqrt(dt) * z
        prices.append(prices[-1] * math.exp(drift + diffusion))
    return prices


def _box_muller(rng: random.Random) -> float:
    while True:
        u = rng.uniform(1e-10, 1.0)
        v = rng.uniform(0.0, 1.0)
        if u > 0.0:
            break
    return math.sqrt(-2.0 * math.log(u)) * math.cos(2.0 * math.pi * v)


# ─── Signal generation ────────────────────────────────────────────────────────


def _compute_signal(price: float, prev_price: float, volatility: float) -> str:
    """
    Simple momentum signal:
      returns > threshold → BUY
      returns < -threshold → SELL
      otherwise → HOLD

    Uses a fixed 0.05% minimum threshold to produce signals
    on realistic intraday GBM price moves.
    """
    if prev_price <= 0:
        return "HOLD"
    ret = (price - prev_price) / prev_price
    # Use half the realised tick volatility, min 0.05%
    threshold = max(volatility * 0.5, 0.0005)
    if ret > threshold:
        return "BUY"
    elif ret < -threshold:
        return "SELL"
    return "HOLD"


# ─── Risk check ───────────────────────────────────────────────────────────────


@dataclass
class RiskCheck:
    """Result of a risk validation."""
    approved: bool
    reason: str
    adjusted_size: float


def _risk_check(
    signal: str,
    price: float,
    capital: float,
    current_position: float,
    max_position_pct: float = 0.10,
    max_drawdown_pct: float = 0.20,
) -> RiskCheck:
    """
    Validate trade size against risk limits.

    Returns RiskCheck(approved=True) if within limits,
    RiskCheck(approved=False) with reason if rejected.
    """
    if signal == "HOLD":
        return RiskCheck(approved=False, reason="HOLD signal — no trade", adjusted_size=0.0)

    max_notional = capital * max_position_pct
    size = max_notional / price if price > 0 else 0.0

    # Check absolute drawdown
    unrealized_loss = 0.0
    if current_position < 0 and signal == "BUY":
        unrealized_loss = abs(current_position) * price
    if unrealized_loss / (capital + 1e-9) > max_drawdown_pct:
        return RiskCheck(
            approved=False,
            reason=f"Drawdown limit exceeded ({unrealized_loss/capital*100:.1f}% > {max_drawdown_pct*100:.0f}%)",
            adjusted_size=0.0,
        )

    return RiskCheck(approved=True, reason="Within risk limits", adjusted_size=size)


# ─── Step records ─────────────────────────────────────────────────────────────


@dataclass
class PipelineStep:
    """Record of one pipeline step for a single tick."""
    tick: int
    symbol: str
    price: float
    prev_price: float
    volatility: float
    signal: str
    risk_approved: bool
    risk_reason: str
    size: float
    notional: float
    ledger_entry: Optional[Dict[str, Any]]
    elapsed_ms: float


@dataclass
class DemoResult:
    """Full result of a demo dry-run."""
    agent_id: str
    symbol: str
    ticks: int
    initial_capital: float
    steps: List[PipelineStep]
    summary: Dict[str, Any]
    trace: str
    elapsed_total_ms: float

    def to_json(self) -> str:
        d = {
            "agent_id": self.agent_id,
            "symbol": self.symbol,
            "ticks": self.ticks,
            "initial_capital": self.initial_capital,
            "elapsed_total_ms": self.elapsed_total_ms,
            "summary": self.summary,
            "steps": [
                {
                    "tick": s.tick,
                    "symbol": s.symbol,
                    "price": s.price,
                    "signal": s.signal,
                    "risk_approved": s.risk_approved,
                    "risk_reason": s.risk_reason,
                    "size": s.size,
                    "notional": s.notional,
                    "ledger_entry": s.ledger_entry,
                    "elapsed_ms": s.elapsed_ms,
                }
                for s in self.steps
            ],
        }
        return json.dumps(d, indent=2)


# ─── Demo Pipeline ────────────────────────────────────────────────────────────


class DemoPipeline:
    """
    End-to-end dry-run demo pipeline.

    Runs market fetch → signal → risk check → order simulation → ledger write
    for the specified number of ticks and symbol.

    Parameters
    ----------
    ticks           : Number of price ticks to simulate.
    symbol          : Market symbol (e.g. "BTC/USD").
    initial_price   : Starting price.
    initial_capital : Starting capital.
    agent_id        : Agent identifier.
    seed            : RNG seed (default 42 for reproducibility).
    db_path         : SQLite path (":memory:" for ephemeral runs).
    """

    DEFAULT_PRICES = {
        "BTC": 65_000.0,
        "ETH": 3_200.0,
        "SOL": 180.0,
        "BTC/USD": 65_000.0,
        "ETH/USD": 3_200.0,
        "SOL/USD": 180.0,
    }

    def __init__(
        self,
        ticks: int = 10,
        symbol: str = "BTC/USD",
        initial_price: Optional[float] = None,
        initial_capital: float = 10_000.0,
        agent_id: Optional[str] = None,
        seed: Optional[int] = 42,
        db_path: str = ":memory:",
    ) -> None:
        self.ticks = ticks
        self.symbol = symbol
        base = symbol.split("/")[0] if "/" in symbol else symbol
        self.initial_price = initial_price or self.DEFAULT_PRICES.get(base, 100.0)
        self.initial_capital = initial_capital
        self.agent_id = agent_id or f"demo-agent-{uuid.uuid4().hex[:8]}"
        self.seed = seed
        self.db_path = db_path
        self._ledger: Optional[TradeLedger] = None

    @property
    def ledger(self) -> TradeLedger:
        if self._ledger is None:
            self._ledger = TradeLedger(self.db_path)
        return self._ledger

    def run(self) -> DemoResult:
        """
        Execute the full dry-run pipeline.

        Returns a DemoResult containing step-by-step trace and summary.
        """
        t0 = time.monotonic()
        prices = _gbm_prices(
            initial=self.initial_price,
            n=self.ticks,
            seed=self.seed,
        )

        capital = self.initial_capital
        position = 0.0
        steps: List[PipelineStep] = []
        trades_executed = 0
        trades_rejected = 0

        for i in range(self.ticks):
            tick_t0 = time.monotonic()
            price = prices[i + 1]
            prev_price = prices[i]
            vol = abs(price - prev_price) / prev_price if prev_price > 0 else 0.001

            # Step 2: Signal
            signal = _compute_signal(price, prev_price, vol)

            # Step 3: Risk check
            risk = _risk_check(
                signal=signal,
                price=price,
                capital=capital,
                current_position=position,
            )

            # Step 4 & 5: Order sim + ledger
            entry_dict: Optional[Dict[str, Any]] = None
            notional = 0.0

            if risk.approved:
                ts = datetime.now(timezone.utc).isoformat()
                ledger_entry = self.ledger.log_trade(
                    agent_id=self.agent_id,
                    market=self.symbol,
                    side=signal,
                    size=risk.adjusted_size,
                    price=price,
                    timestamp=ts,
                )
                notional = ledger_entry.notional

                # Update simulated position
                if signal == "BUY":
                    position += risk.adjusted_size
                    capital -= notional
                else:
                    position -= risk.adjusted_size
                    capital += notional

                entry_dict = ledger_entry.to_dict()
                trades_executed += 1
            else:
                trades_rejected += 1

            elapsed_ms = (time.monotonic() - tick_t0) * 1000

            steps.append(PipelineStep(
                tick=i + 1,
                symbol=self.symbol,
                price=price,
                prev_price=prev_price,
                volatility=vol,
                signal=signal,
                risk_approved=risk.approved,
                risk_reason=risk.reason,
                size=risk.adjusted_size,
                notional=notional,
                ledger_entry=entry_dict,
                elapsed_ms=elapsed_ms,
            ))

        elapsed_total = (time.monotonic() - t0) * 1000
        ledger_summary = self.ledger.get_summary()

        summary = {
            "ticks": self.ticks,
            "trades_executed": trades_executed,
            "trades_rejected": trades_rejected,
            "total_notional": ledger_summary.total_notional,
            "buy_count": ledger_summary.buy_count,
            "sell_count": ledger_summary.sell_count,
            "final_capital": capital,
            "price_start": prices[0],
            "price_end": prices[-1],
            "price_return_pct": (prices[-1] - prices[0]) / prices[0] * 100,
        }

        trace = self._build_trace(steps, summary, elapsed_total)

        return DemoResult(
            agent_id=self.agent_id,
            symbol=self.symbol,
            ticks=self.ticks,
            initial_capital=self.initial_capital,
            steps=steps,
            summary=summary,
            trace=trace,
            elapsed_total_ms=elapsed_total,
        )

    def _build_trace(
        self,
        steps: List[PipelineStep],
        summary: Dict[str, Any],
        elapsed_ms: float,
    ) -> str:
        """Build a human-readable trace of the pipeline run."""
        lines = [
            "=" * 72,
            "  ERC-8004 Trading Agent — Pipeline Dry-Run",
            f"  Agent:   {self.agent_id}",
            f"  Symbol:  {self.symbol}",
            f"  Ticks:   {self.ticks}",
            f"  Capital: ${self.initial_capital:,.2f}",
            "=" * 72,
            "",
            f"{'TICK':<6} {'PRICE':>10} {'SIGNAL':<6} {'RISK':<9} "
            f"{'SIZE':>8} {'NOTIONAL':>12} {'TX_HASH'}",
            "-" * 72,
        ]

        for s in steps:
            if s.risk_approved and s.ledger_entry:
                tx = s.ledger_entry["tx_hash"][:20] + "..."
                lines.append(
                    f"{s.tick:<6} {s.price:>10.2f} {s.signal:<6} "
                    f"{'OK':<9} {s.size:>8.4f} {s.notional:>12.2f} {tx}"
                )
            else:
                reason = s.risk_reason[:20] if s.risk_reason else "rejected"
                lines.append(
                    f"{s.tick:<6} {s.price:>10.2f} {s.signal:<6} "
                    f"{'SKIP':<9} {'—':>8} {'—':>12} ({reason})"
                )

        lines += [
            "-" * 72,
            "",
            "Summary",
            "-------",
            f"  Ticks processed  : {summary['ticks']}",
            f"  Trades executed  : {summary['trades_executed']}",
            f"  Trades rejected  : {summary['trades_rejected']}",
            f"  Total notional   : ${summary['total_notional']:,.2f}",
            f"  BUY / SELL       : {summary['buy_count']} / {summary['sell_count']}",
            f"  Price start→end  : ${summary['price_start']:.2f} → ${summary['price_end']:.2f}",
            f"  Price return     : {summary['price_return_pct']:+.2f}%",
            f"  Final capital    : ${summary['final_capital']:,.2f}",
            f"  Elapsed          : {elapsed_ms:.1f}ms",
            "",
            f"  Ledger written to: {self.db_path}",
            "=" * 72,
        ]
        return "\n".join(lines)


# ─── CLI entry point ──────────────────────────────────────────────────────────


def _parse_args(argv: Optional[list] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="demo_cli",
        description="ERC-8004 Pipeline Dry-Run Demo",
    )
    parser.add_argument("--ticks", type=int, default=10, help="Number of price ticks")
    parser.add_argument("--symbol", type=str, default="BTC/USD", help="Market symbol")
    parser.add_argument("--capital", type=float, default=10_000.0, help="Initial capital")
    parser.add_argument("--seed", type=int, default=42, help="RNG seed")
    parser.add_argument("--json", action="store_true", dest="as_json",
                        help="Output machine-readable JSON")
    parser.add_argument("--db", type=str, default=":memory:",
                        help="SQLite path (default: :memory:)")
    return parser.parse_args(argv)


def main(argv: Optional[list] = None) -> DemoResult:
    """Run the demo CLI. Returns DemoResult (useful for testing)."""
    args = _parse_args(argv)
    pipeline = DemoPipeline(
        ticks=args.ticks,
        symbol=args.symbol,
        initial_capital=args.capital,
        seed=args.seed,
        db_path=args.db,
    )
    result = pipeline.run()
    if args.as_json:
        print(result.to_json())
    else:
        print(result.trace)
    return result


if __name__ == "__main__":
    main()
