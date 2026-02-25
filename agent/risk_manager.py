"""
risk_manager.py — Pre-trade risk validation for the ERC-8004 Trading Agent.

Validates each proposed trade against configurable risk limits:
  - Position size limits (max % of portfolio per trade)
  - Stop-loss / daily drawdown limits
  - Leverage caps
  - Exposure limits (max concurrent open positions)

Usage:
    rm = RiskManager(max_position_pct=0.10, max_daily_drawdown_pct=0.05)
    ok, reason = rm.validate_trade(side="YES", size=5.0, price=0.65, portfolio_value=100.0)
    if not ok:
        logger.warning(f"Trade rejected: {reason}")
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from typing import Optional

from loguru import logger

from credora_client import CredoraClient, CredoraRating, CredoraRatingTier  # noqa: E402


# ─── Configuration ────────────────────────────────────────────────────────────

@dataclass
class RiskConfig:
    """Configurable risk parameters."""
    # Maximum single-trade size as fraction of portfolio
    max_position_pct: float = 0.10          # 10% of portfolio per trade
    # Maximum total exposure as fraction of portfolio (all open positions)
    max_exposure_pct: float = 0.30          # 30% of portfolio in open positions
    # Maximum concurrent open positions
    max_open_positions: int = 5
    # Per-trade stop-loss (loss as fraction of trade size)
    stop_loss_pct: float = 0.50             # 50% loss on any single trade → close
    # Daily drawdown limit (loss as fraction of portfolio)
    max_daily_drawdown_pct: float = 0.05    # 5% daily drawdown → halt trading
    # Maximum effective leverage (size / collateral)
    max_leverage: float = 3.0
    # Minimum portfolio value to allow trading
    min_portfolio_value: float = 10.0       # $10 minimum


# ─── Position Tracking ────────────────────────────────────────────────────────

@dataclass
class OpenPosition:
    """Represents an open (unresolved) position."""
    market_id: str
    side: str
    size_usdc: float
    entry_price: float
    opened_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    current_pnl: float = 0.0

    @property
    def notional_value(self) -> float:
        return self.size_usdc / max(self.entry_price, 0.01)


# ─── Risk Manager ─────────────────────────────────────────────────────────────

class RiskManager:
    """
    Pre-trade and ongoing risk validation.

    Integrates with trader.py execute_trade() — call validate_trade() before
    executing any order.  If validation fails, the trade is skipped with a
    reason string for logging.
    """

    def __init__(
        self,
        config: Optional[RiskConfig] = None,
        *,
        max_position_pct: float = 0.10,
        max_exposure_pct: float = 0.30,
        max_open_positions: int = 5,
        stop_loss_pct: float = 0.50,
        max_daily_drawdown_pct: float = 0.05,
        max_leverage: float = 3.0,
        min_portfolio_value: float = 10.0,
        credora_client: Optional[CredoraClient] = None,
        credora_min_grade: Optional[CredoraRatingTier] = None,
    ) -> None:
        if config is not None:
            self.config = config
        else:
            self.config = RiskConfig(
                max_position_pct=max_position_pct,
                max_exposure_pct=max_exposure_pct,
                max_open_positions=max_open_positions,
                stop_loss_pct=stop_loss_pct,
                max_daily_drawdown_pct=max_daily_drawdown_pct,
                max_leverage=max_leverage,
                min_portfolio_value=min_portfolio_value,
            )

        self._open_positions: list[OpenPosition] = []
        self._daily_pnl: float = 0.0         # cumulative P&L today
        self._last_reset_date: date = date.today()
        self._trading_halted: bool = False
        self._halt_reason: str = ""

        # Optional Credora integration
        self._credora: Optional[CredoraClient] = credora_client
        # Minimum acceptable Credora tier (None = no filter)
        self._credora_min_grade: Optional[CredoraRatingTier] = credora_min_grade

        logger.info(
            f"RiskManager initialized: max_pos={max_position_pct:.0%} "
            f"max_exp={max_exposure_pct:.0%} "
            f"max_dd={max_daily_drawdown_pct:.0%} "
            f"credora={'enabled' if credora_client else 'disabled'}"
        )

    # ─── Daily Reset ──────────────────────────────────────────────────────────

    def _maybe_reset_daily(self) -> None:
        """Reset daily P&L tracker at start of each trading day."""
        today = date.today()
        if today != self._last_reset_date:
            logger.info(f"RiskManager: new trading day — resetting daily P&L (was ${self._daily_pnl:.2f})")
            self._daily_pnl = 0.0
            self._last_reset_date = today
            # Lift halt if it was due to drawdown (new day = fresh start)
            if self._trading_halted and "drawdown" in self._halt_reason.lower():
                self._trading_halted = False
                self._halt_reason = ""
                logger.info("RiskManager: daily drawdown halt lifted for new trading day")

    # ─── Core Validation ──────────────────────────────────────────────────────

    def validate_trade(
        self,
        side: str,
        size: float,
        price: float,
        portfolio_value: float,
    ) -> tuple[bool, str]:
        """
        Validate a proposed trade against all risk limits.

        Args:
            side: "YES" or "NO" (market side)
            size: Trade size in USDC
            price: Current market price (0.0–1.0)
            portfolio_value: Total portfolio value in USDC

        Returns:
            (allowed: bool, reason: str)
            If allowed=False, reason explains why the trade was rejected.
        """
        self._maybe_reset_daily()

        # 1. Halt check
        if self._trading_halted:
            return False, f"Trading halted: {self._halt_reason}"

        # 2. Portfolio minimum
        if portfolio_value < self.config.min_portfolio_value:
            return False, (
                f"Portfolio value ${portfolio_value:.2f} below minimum "
                f"${self.config.min_portfolio_value:.2f}"
            )

        # 3. Position size limit
        max_size = portfolio_value * self.config.max_position_pct
        if size > max_size:
            return False, (
                f"Trade size ${size:.2f} exceeds max position "
                f"${max_size:.2f} ({self.config.max_position_pct:.0%} of portfolio)"
            )

        # 4. Zero / negative size
        if size <= 0:
            return False, f"Invalid trade size: ${size:.2f}"

        # 5. Price sanity (prediction market prices are 0–1)
        if not (0.0 < price < 1.0):
            return False, f"Invalid price {price:.4f} (must be 0 < price < 1)"

        # 6. Leverage check
        # In prediction markets, notional = size / price (shares purchased)
        if price > 0:
            leverage = size / (price * portfolio_value) if portfolio_value > 0 else float("inf")
            if leverage > self.config.max_leverage:
                return False, (
                    f"Effective leverage {leverage:.2f}x exceeds max "
                    f"{self.config.max_leverage:.1f}x"
                )

        # 7. Open position count
        if len(self._open_positions) >= self.config.max_open_positions:
            return False, (
                f"Max concurrent positions reached: "
                f"{len(self._open_positions)}/{self.config.max_open_positions}"
            )

        # 8. Total exposure check
        current_exposure = sum(p.size_usdc for p in self._open_positions)
        new_exposure = current_exposure + size
        max_exposure = portfolio_value * self.config.max_exposure_pct
        if new_exposure > max_exposure:
            return False, (
                f"Total exposure ${new_exposure:.2f} would exceed max "
                f"${max_exposure:.2f} ({self.config.max_exposure_pct:.0%} of portfolio)"
            )

        logger.debug(
            f"RiskManager: trade approved — side={side} size=${size:.2f} "
            f"price={price:.3f} portfolio=${portfolio_value:.2f}"
        )
        return True, "OK"

    def validate_trade_with_credora(
        self,
        side: str,
        size: float,
        price: float,
        portfolio_value: float,
        protocol: str,
    ) -> tuple[bool, str, float]:
        """
        Validate a trade and apply Credora Kelly multiplier to the size.

        Args:
            side: "YES" or "NO"
            size: Proposed trade size in USDC
            price: Market price (0–1)
            portfolio_value: Total portfolio value in USDC
            protocol: Protocol/asset name for Credora rating lookup

        Returns:
            (allowed: bool, reason: str, adjusted_size: float)
            adjusted_size = size * credora_kelly_multiplier if allowed, else 0.0
        """
        if self._credora is None:
            ok, reason = self.validate_trade(side, size, price, portfolio_value)
            return ok, reason, size if ok else 0.0

        rating = self._credora.get_rating(protocol)
        multiplier = rating.kelly_multiplier

        logger.debug(
            f"RiskManager: Credora rating for {protocol!r}: "
            f"{rating.tier.value} (kelly_mult={multiplier:.2f})"
        )

        # Optional hard floor — reject if below minimum grade
        if self._credora_min_grade is not None:
            from credora_client import _TIER_ORDER
            min_idx = _TIER_ORDER.index(self._credora_min_grade)
            rating_idx = _TIER_ORDER.index(rating.tier) if rating.tier in _TIER_ORDER else -1
            if rating_idx < min_idx:
                return (
                    False,
                    f"Protocol {protocol!r} rated {rating.tier.value} is below "
                    f"minimum {self._credora_min_grade.value}",
                    0.0,
                )

        # Apply Kelly multiplier to effective size before standard checks
        adjusted_size = size * multiplier
        ok, reason = self.validate_trade(side, adjusted_size, price, portfolio_value)
        return ok, reason, adjusted_size if ok else 0.0

    def get_credora_rating(self, protocol: str) -> Optional[CredoraRating]:
        """Fetch Credora rating for a protocol (if Credora client is configured)."""
        if self._credora is None:
            return None
        return self._credora.get_rating(protocol)

    # ─── Drawdown Tracking ────────────────────────────────────────────────────

    def record_trade_pnl(self, pnl_usdc: float) -> None:
        """
        Record realized P&L from a completed trade.
        Call this after a trade settles to update daily drawdown tracking.
        """
        self._maybe_reset_daily()
        self._daily_pnl += pnl_usdc
        logger.debug(f"RiskManager: daily P&L updated to ${self._daily_pnl:.4f}")

    def check_drawdown(self, portfolio_value: float) -> bool:
        """
        Check whether daily drawdown limit has been breached.

        Returns True if trading should continue, False if halt is triggered.
        Also sets internal halt flag — subsequent validate_trade() calls will fail.

        Args:
            portfolio_value: Current portfolio value in USDC
        """
        self._maybe_reset_daily()

        if portfolio_value <= 0:
            return True  # Can't compute drawdown without portfolio value

        drawdown_pct = abs(min(self._daily_pnl, 0)) / portfolio_value
        limit = self.config.max_daily_drawdown_pct

        if drawdown_pct >= limit:
            self._trading_halted = True
            self._halt_reason = (
                f"daily drawdown {drawdown_pct:.2%} >= limit {limit:.2%} "
                f"(P&L=${self._daily_pnl:.2f})"
            )
            logger.warning(f"RiskManager: HALT triggered — {self._halt_reason}")
            return False

        logger.debug(
            f"RiskManager: drawdown check OK — "
            f"{drawdown_pct:.2%} < {limit:.2%} limit"
        )
        return True

    # ─── Position Management ──────────────────────────────────────────────────

    def open_position(self, market_id: str, side: str, size: float, price: float) -> None:
        """Register an opened position for exposure tracking."""
        pos = OpenPosition(
            market_id=market_id,
            side=side,
            size_usdc=size,
            entry_price=price,
        )
        self._open_positions.append(pos)
        logger.debug(f"RiskManager: opened position {market_id} {side} ${size:.2f}")

    def close_position(self, market_id: str, pnl_usdc: float = 0.0) -> bool:
        """
        Remove a closed position from tracking and record its P&L.
        Returns True if position was found and removed.
        """
        for i, pos in enumerate(self._open_positions):
            if pos.market_id == market_id:
                self._open_positions.pop(i)
                self.record_trade_pnl(pnl_usdc)
                logger.debug(
                    f"RiskManager: closed position {market_id} "
                    f"P&L=${pnl_usdc:.4f}"
                )
                return True
        return False

    def check_stop_loss(self, position: OpenPosition) -> bool:
        """
        Check if a position should be stopped out.
        Returns True if stop-loss is triggered (position should be closed).
        """
        if position.size_usdc <= 0:
            return False
        loss_pct = abs(min(position.current_pnl, 0)) / position.size_usdc
        if loss_pct >= self.config.stop_loss_pct:
            logger.warning(
                f"RiskManager: stop-loss triggered on {position.market_id} "
                f"loss={loss_pct:.2%} >= {self.config.stop_loss_pct:.2%}"
            )
            return True
        return False

    # ─── Summary ──────────────────────────────────────────────────────────────

    def get_risk_summary(self) -> dict:
        """
        Return current risk state for the dashboard and logging.

        Returns a dict with all key risk metrics:
          - trading_halted: bool
          - halt_reason: str
          - daily_pnl: float (today's realized P&L)
          - open_positions: int
          - total_exposure_usdc: float
          - limits: dict of configured limits
        """
        self._maybe_reset_daily()
        total_exposure = sum(p.size_usdc for p in self._open_positions)

        return {
            "trading_halted": self._trading_halted,
            "halt_reason": self._halt_reason,
            "daily_pnl_usdc": round(self._daily_pnl, 4),
            "open_positions": len(self._open_positions),
            "total_exposure_usdc": round(total_exposure, 4),
            "positions": [
                {
                    "market_id": p.market_id,
                    "side": p.side,
                    "size_usdc": p.size_usdc,
                    "entry_price": p.entry_price,
                    "current_pnl": p.current_pnl,
                }
                for p in self._open_positions
            ],
            "limits": {
                "max_position_pct": self.config.max_position_pct,
                "max_exposure_pct": self.config.max_exposure_pct,
                "max_open_positions": self.config.max_open_positions,
                "stop_loss_pct": self.config.stop_loss_pct,
                "max_daily_drawdown_pct": self.config.max_daily_drawdown_pct,
                "max_leverage": self.config.max_leverage,
            },
        }

    def reset_halt(self) -> None:
        """Manually clear trading halt (operator override)."""
        self._trading_halted = False
        self._halt_reason = ""
        logger.info("RiskManager: trading halt cleared by operator")

    # ─── Oracle Risk Check ────────────────────────────────────────────────────

    def check_oracle_risk(
        self,
        amount: float,
        oracle_price: float,
        max_deviation_pct: float = 0.05,
    ) -> bool:
        """
        Validate a trade amount against an oracle price using a deviation threshold.

        Mirrors the on-chain RiskRouter.checkRisk() logic in Python so that the
        agent can pre-screen trades before broadcasting them on-chain.

        Args:
            amount:            Proposed trade amount (in the same units as oracle_price)
            oracle_price:      Current oracle price (e.g. from OracleClient.fetch_eth_price())
            max_deviation_pct: Maximum allowed deviation from oracle price (default 5%)

        Returns:
            True if the trade passes the oracle risk check, False otherwise.
        """
        if oracle_price <= 0:
            logger.warning("RiskManager.check_oracle_risk: invalid oracle_price")
            return False

        if amount <= 0:
            logger.warning("RiskManager.check_oracle_risk: non-positive amount rejected")
            return False

        deviation = abs(amount - oracle_price) / oracle_price
        allowed = deviation <= max_deviation_pct

        if allowed:
            logger.debug(
                f"RiskManager.check_oracle_risk: PASS "
                f"amount={amount} oracle={oracle_price} deviation={deviation:.2%}"
            )
        else:
            logger.info(
                f"RiskManager.check_oracle_risk: FAIL "
                f"amount={amount} oracle={oracle_price} "
                f"deviation={deviation:.2%} > max={max_deviation_pct:.2%}"
            )

        return allowed
