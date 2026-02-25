"""
portfolio_manager.py — Portfolio management layer for the ERC-8004 Trading Agent.

Integrates:
  - Kelly Criterion position sizing (optimal bet sizing given edge + confidence)
  - Max drawdown enforcement (10% hard limit — halts trading if breached)
  - Position tracking (open/close/PnL per trade)
  - claude_strategist integration for trade signals
  - check_oracle_risk() validation before any trade

Usage:
    pm = PortfolioManager(capital_usdc=100.0)
    signal = await pm.get_signal(market_data, oracle_price)
    if signal.action != "hold":
        size = pm.compute_kelly_size(signal.confidence, signal.edge)
        ok = pm.open_position(market_id, signal.action, size, oracle_price)
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional, TYPE_CHECKING

from loguru import logger

if TYPE_CHECKING:
    from claude_strategist import TradeDecision


# ─── Data Classes ─────────────────────────────────────────────────────────────

@dataclass
class Position:
    """A single open or closed position."""
    position_id: str
    market_id: str
    side: str               # "BUY" or "SELL"
    size_usdc: float        # Capital allocated
    entry_price: float      # Oracle price at entry
    opened_at: float = field(default_factory=time.time)
    closed_at: Optional[float] = None
    exit_price: Optional[float] = None
    pnl_usdc: float = 0.0
    is_open: bool = True

    def close(self, exit_price: float, pnl_usdc: float) -> None:
        """Close this position and record P&L."""
        self.exit_price = exit_price
        self.pnl_usdc = pnl_usdc
        self.closed_at = time.time()
        self.is_open = False

    def to_dict(self) -> dict:
        return {
            "position_id": self.position_id,
            "market_id": self.market_id,
            "side": self.side,
            "size_usdc": self.size_usdc,
            "entry_price": self.entry_price,
            "exit_price": self.exit_price,
            "pnl_usdc": round(self.pnl_usdc, 6),
            "is_open": self.is_open,
            "opened_at": self.opened_at,
            "closed_at": self.closed_at,
        }


@dataclass
class TradeSignal:
    """Signal output from portfolio manager's decision process."""
    action: str              # "BUY", "SELL", "HOLD"
    confidence: float        # 0.0–1.0
    edge: float              # Estimated edge (expected return above market)
    kelly_fraction: float    # Kelly-recommended fraction of capital
    recommended_size: float  # Concrete USDC size to trade
    reasoning: str
    oracle_price: float
    risk_score: float        # Oracle risk score (deviation from fair price)
    blocked: bool = False    # True if trade was blocked by risk check
    block_reason: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "action": self.action,
            "confidence": round(self.confidence, 4),
            "edge": round(self.edge, 4),
            "kelly_fraction": round(self.kelly_fraction, 4),
            "recommended_size": round(self.recommended_size, 4),
            "reasoning": self.reasoning,
            "oracle_price": self.oracle_price,
            "risk_score": round(self.risk_score, 4),
            "blocked": self.blocked,
            "block_reason": self.block_reason,
        }


@dataclass
class PortfolioStats:
    """Snapshot of current portfolio state."""
    capital_usdc: float
    peak_capital: float
    current_drawdown_pct: float
    open_positions: int
    total_trades: int
    winning_trades: int
    total_pnl_usdc: float
    drawdown_halted: bool

    @property
    def win_rate(self) -> float:
        closed = self.total_trades - self.open_positions
        return self.winning_trades / closed if closed > 0 else 0.0

    def to_dict(self) -> dict:
        return {
            "capital_usdc": round(self.capital_usdc, 4),
            "peak_capital": round(self.peak_capital, 4),
            "current_drawdown_pct": round(self.current_drawdown_pct, 4),
            "open_positions": self.open_positions,
            "total_trades": self.total_trades,
            "winning_trades": self.winning_trades,
            "total_pnl_usdc": round(self.total_pnl_usdc, 4),
            "win_rate": round(self.win_rate, 4),
            "drawdown_halted": self.drawdown_halted,
        }


# ─── Kelly Criterion ──────────────────────────────────────────────────────────

def kelly_fraction(
    win_prob: float,
    win_return: float = 1.0,
    loss_return: float = 1.0,
) -> float:
    """
    Compute the Kelly Criterion fraction of capital to bet.

    Kelly formula: f* = (p * b - q) / b
    where:
      p = probability of winning
      q = 1 - p = probability of losing
      b = net odds received (win_return / loss_return)

    Returns fraction clamped to [0, 1]. Negative Kelly → don't bet.

    Args:
        win_prob:    Estimated probability of winning (0.0–1.0)
        win_return:  Net return on win (default 1.0 = doubling, i.e. 100% return)
        loss_return: Loss on loss (default 1.0 = losing entire stake)

    Returns:
        Kelly fraction (0.0–1.0)
    """
    if win_prob <= 0.0 or win_prob >= 1.0:
        return 0.0
    if loss_return <= 0.0:
        return 0.0

    q = 1.0 - win_prob
    b = win_return / loss_return

    if b <= 0.0:
        return 0.0

    f = (win_prob * b - q) / b
    return max(0.0, min(1.0, f))


def fractional_kelly(
    win_prob: float,
    fraction: float = 0.25,
    win_return: float = 1.0,
    loss_return: float = 1.0,
) -> float:
    """
    Apply fractional Kelly (more conservative sizing).

    Fractional Kelly (typically 1/4 Kelly) reduces variance at the cost
    of slightly lower expected growth. Preferred for real-money trading.

    Args:
        win_prob: Win probability
        fraction: Kelly multiplier (0.25 = quarter Kelly)
        win_return: Net return on win
        loss_return: Loss on loss

    Returns:
        Fractional Kelly fraction (0.0–1.0)
    """
    full_kelly = kelly_fraction(win_prob, win_return, loss_return)
    return full_kelly * fraction


# ─── Portfolio Manager ────────────────────────────────────────────────────────

class PortfolioManager:
    """
    Portfolio manager integrating Kelly sizing, drawdown protection,
    and AI-driven signals.

    Wires together:
      - ClaudeStrategist → trade signals
      - RiskManager.check_oracle_risk() → oracle risk gate
      - Kelly Criterion → position sizing
      - 10% max drawdown → hard halt

    Usage:
        pm = PortfolioManager(capital_usdc=100.0, strategist=claude)
        signal = await pm.get_signal(market_data, oracle_price=3000.0)
        if not signal.blocked and signal.action != "HOLD":
            pm.open_position(market_id, signal.action, signal.recommended_size, oracle_price)
    """

    MAX_DRAWDOWN_PCT = 0.10          # 10% max drawdown → halt
    MAX_POSITION_PCT = 0.20          # 20% max single position
    MIN_CONFIDENCE = 0.55            # Minimum confidence to take a trade
    KELLY_MULTIPLIER = 0.25          # Quarter Kelly (conservative)

    def __init__(
        self,
        capital_usdc: float = 100.0,
        strategist=None,
        risk_manager=None,
        max_drawdown_pct: float = MAX_DRAWDOWN_PCT,
        kelly_multiplier: float = KELLY_MULTIPLIER,
    ) -> None:
        self.capital_usdc = capital_usdc
        self.strategist = strategist
        self.risk_manager = risk_manager
        self.max_drawdown_pct = max_drawdown_pct
        self.kelly_multiplier = kelly_multiplier

        self._peak_capital = capital_usdc
        self._positions: dict[str, Position] = {}
        self._closed_positions: list[Position] = []
        self._total_pnl: float = 0.0
        self._winning_trades: int = 0
        self._drawdown_halted: bool = False
        self._position_counter: int = 0

        logger.info(
            f"PortfolioManager initialized: capital=${capital_usdc:.2f} "
            f"max_dd={max_drawdown_pct:.1%} kelly={kelly_multiplier:.2f}x"
        )

    # ─── Signal Generation ────────────────────────────────────────────────────

    async def get_signal(
        self, market_data: dict, oracle_price: float
    ) -> TradeSignal:
        """
        Get a trade signal integrating oracle risk score and AI strategy.

        Flow:
          1. Compute oracle risk score (deviation from expected range)
          2. Call ClaudeStrategist.decide() for AI signal
          3. Check risk gate — block if oracle deviation too high
          4. Apply Kelly sizing to compute recommended position size

        Args:
            market_data: Market state dict (yes_price, volume, question, etc.)
            oracle_price: Current oracle price (e.g. ETH/USD)

        Returns:
            TradeSignal with action, sizing, and risk assessment
        """
        # 1. Oracle risk score: how far is market price from oracle?
        market_price = market_data.get("yes_price", market_data.get("price", 0.5))
        risk_score = self._compute_risk_score(market_price, oracle_price)

        # 2. Check drawdown halt
        if self._drawdown_halted:
            return TradeSignal(
                action="HOLD", confidence=0.0, edge=0.0,
                kelly_fraction=0.0, recommended_size=0.0,
                reasoning="Trading halted: max drawdown exceeded",
                oracle_price=oracle_price, risk_score=risk_score,
                blocked=True, block_reason="drawdown_halt",
            )

        # 3. Get AI signal (or use fallback)
        if self.strategist is not None:
            try:
                decision = await self.strategist.decide(market_data)
                action, confidence, reasoning = self._normalize_decision(decision)
            except Exception as e:
                logger.warning(f"PortfolioManager: strategist failed ({e}), using fallback")
                action, confidence, reasoning = self._fallback_signal(market_price)
        else:
            action, confidence, reasoning = self._fallback_signal(market_price)

        # 4. Risk gate: check oracle risk
        if self.risk_manager is not None:
            try:
                oracle_ok = self.risk_manager.check_oracle_risk(
                    amount=market_price * oracle_price,  # notional value
                    oracle_price=oracle_price,
                )
                if not oracle_ok:
                    return TradeSignal(
                        action="HOLD", confidence=confidence, edge=0.0,
                        kelly_fraction=0.0, recommended_size=0.0,
                        reasoning=reasoning,
                        oracle_price=oracle_price, risk_score=risk_score,
                        blocked=True, block_reason="oracle_risk_check_failed",
                    )
            except Exception as e:
                logger.warning(f"PortfolioManager: oracle risk check failed ({e})")

        # 5. Minimum confidence gate
        if confidence < self.MIN_CONFIDENCE or action == "HOLD":
            return TradeSignal(
                action="HOLD", confidence=confidence, edge=0.0,
                kelly_fraction=0.0, recommended_size=0.0,
                reasoning=reasoning,
                oracle_price=oracle_price, risk_score=risk_score,
            )

        # 6. Kelly sizing
        edge = max(confidence - 0.5, 0.0) * 2  # edge 0–1 based on confidence above 50%
        kf = fractional_kelly(confidence, self.kelly_multiplier)
        max_size = self.capital_usdc * self.MAX_POSITION_PCT
        recommended_size = min(self.capital_usdc * kf, max_size)
        recommended_size = round(recommended_size, 4)

        return TradeSignal(
            action=action,
            confidence=confidence,
            edge=edge,
            kelly_fraction=kf,
            recommended_size=recommended_size,
            reasoning=reasoning,
            oracle_price=oracle_price,
            risk_score=risk_score,
        )

    def _normalize_decision(self, decision) -> tuple[str, float, str]:
        """Normalize a TradeDecision from ClaudeStrategist to (action, confidence, reasoning)."""
        action_map = {
            "buy": "BUY", "buy_yes": "BUY",
            "sell": "SELL", "buy_no": "SELL",
            "hold": "HOLD",
        }
        action = action_map.get(decision.action.lower(), "HOLD")
        confidence = max(0.0, min(1.0, decision.confidence))
        reasoning = decision.reasoning
        return action, confidence, reasoning

    def _fallback_signal(self, price: float) -> tuple[str, float, str]:
        """Simple fallback signal when strategist is unavailable."""
        if price < 0.35:
            return "BUY", 0.60, f"Fallback: price {price:.2f} below 0.35 threshold"
        elif price > 0.75:
            return "SELL", 0.60, f"Fallback: price {price:.2f} above 0.75 threshold"
        else:
            return "HOLD", 0.50, f"Fallback: no edge at price {price:.2f}"

    def _compute_risk_score(self, market_price: float, oracle_price: float) -> float:
        """
        Compute oracle risk score (0.0–1.0, lower = safer).
        Measures how much the market price deviates from oracle-implied fair value.
        """
        if oracle_price <= 0:
            return 1.0  # Maximum risk if oracle unavailable
        # Normalize: deviation as fraction of oracle price, capped at 1.0
        deviation = abs(market_price - (oracle_price % 1.0)) / max(oracle_price % 1.0, 0.01)
        return min(deviation, 1.0)

    # ─── Position Management ──────────────────────────────────────────────────

    def open_position(
        self,
        market_id: str,
        side: str,
        size_usdc: float,
        entry_price: float,
    ) -> Optional[Position]:
        """
        Open a new position.

        Validates:
          - Capital available
          - Max position size
          - Drawdown halt

        Returns the Position if opened, None if blocked.
        """
        if self._drawdown_halted:
            logger.warning("PortfolioManager: cannot open position — drawdown halt")
            return None

        if size_usdc <= 0:
            logger.warning(f"PortfolioManager: invalid size {size_usdc}")
            return None

        max_size = self.capital_usdc * self.MAX_POSITION_PCT
        if size_usdc > max_size:
            logger.warning(
                f"PortfolioManager: size ${size_usdc:.2f} exceeds max ${max_size:.2f}"
            )
            size_usdc = max_size  # Clip to max

        if market_id in self._positions:
            logger.warning(f"PortfolioManager: position already open for {market_id}")
            return None

        self._position_counter += 1
        pos = Position(
            position_id=f"pos-{self._position_counter:04d}",
            market_id=market_id,
            side=side.upper(),
            size_usdc=size_usdc,
            entry_price=entry_price,
        )
        self._positions[market_id] = pos

        logger.info(
            f"PortfolioManager: opened {pos.side} {market_id} "
            f"${size_usdc:.2f} @ {entry_price:.4f}"
        )
        return pos

    def close_position(
        self,
        market_id: str,
        exit_price: float,
        pnl_usdc: float,
    ) -> Optional[Position]:
        """
        Close an open position and record P&L.

        Updates:
          - Capital balance
          - Peak capital (for drawdown calculation)
          - Drawdown check → may halt trading

        Returns the closed Position, or None if not found.
        """
        pos = self._positions.pop(market_id, None)
        if pos is None:
            logger.warning(f"PortfolioManager: no open position for {market_id}")
            return None

        pos.close(exit_price, pnl_usdc)
        self._closed_positions.append(pos)

        # Update capital
        self.capital_usdc += pnl_usdc
        self._total_pnl += pnl_usdc

        if pnl_usdc > 0:
            self._winning_trades += 1

        # Update peak
        if self.capital_usdc > self._peak_capital:
            self._peak_capital = self.capital_usdc

        # Check drawdown
        self._check_drawdown_halt()

        logger.info(
            f"PortfolioManager: closed {market_id} "
            f"P&L=${pnl_usdc:+.4f} capital=${self.capital_usdc:.4f}"
        )
        return pos

    def _check_drawdown_halt(self) -> None:
        """Check if max drawdown has been breached and halt if so."""
        if self._peak_capital <= 0:
            return
        drawdown = (self._peak_capital - self.capital_usdc) / self._peak_capital
        if drawdown >= self.max_drawdown_pct:
            self._drawdown_halted = True
            logger.warning(
                f"PortfolioManager: DRAWDOWN HALT "
                f"drawdown={drawdown:.2%} >= max={self.max_drawdown_pct:.2%}"
            )

    def lift_drawdown_halt(self) -> None:
        """Manually lift the drawdown halt (operator override)."""
        self._drawdown_halted = False
        logger.info("PortfolioManager: drawdown halt lifted by operator")

    # ─── Stats & Reporting ────────────────────────────────────────────────────

    def get_stats(self) -> PortfolioStats:
        """Return current portfolio statistics."""
        drawdown = 0.0
        if self._peak_capital > 0:
            drawdown = max(
                0.0,
                (self._peak_capital - self.capital_usdc) / self._peak_capital
            )

        total_trades = len(self._positions) + len(self._closed_positions)

        return PortfolioStats(
            capital_usdc=self.capital_usdc,
            peak_capital=self._peak_capital,
            current_drawdown_pct=drawdown,
            open_positions=len(self._positions),
            total_trades=total_trades,
            winning_trades=self._winning_trades,
            total_pnl_usdc=self._total_pnl,
            drawdown_halted=self._drawdown_halted,
        )

    def get_open_positions(self) -> list[dict]:
        """Return all open positions as dicts."""
        return [p.to_dict() for p in self._positions.values()]

    def get_closed_positions(self) -> list[dict]:
        """Return all closed positions as dicts."""
        return [p.to_dict() for p in self._closed_positions]

    def compute_kelly_size(
        self,
        confidence: float,
        edge: Optional[float] = None,
    ) -> float:
        """
        Compute recommended position size using fractional Kelly Criterion.

        Args:
            confidence: Win probability (0.0–1.0)
            edge: Optional explicit edge. If None, derived from confidence.

        Returns:
            Recommended USDC position size, capped at MAX_POSITION_PCT.
        """
        kf = fractional_kelly(confidence, self.kelly_multiplier)
        size = self.capital_usdc * kf
        max_size = self.capital_usdc * self.MAX_POSITION_PCT
        return min(round(size, 4), max_size)
