"""
surge_router.py — Surge Risk Router abstraction for ERC-8004 Trading Agent.

Provides a clean interface to the Surge decentralised risk market protocol.
A mock implementation is included for development/testing (real ABI ships March 9).

Surge protocol allows agents to:
  - Execute trades routing through Surge vaults
  - Query vault balances and open positions
  - Fetch real-time pricing for tokens

Usage:
    router = MockSurgeRouter()
    result = await router.execute_trade("USDC", "ETH", 100.0, min_out=0.049)
    balance = await router.get_vault_balance()
    pos     = await router.get_position("ETH")
"""

from __future__ import annotations

import asyncio
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from loguru import logger


# ─── Data Classes ─────────────────────────────────────────────────────────────

@dataclass
class TradeResult:
    """Result of a trade execution through Surge."""
    success: bool
    trade_id: str
    token_in: str
    token_out: str
    amount_in: float
    amount_out: float
    executed_price: float
    slippage_pct: float
    fee_usdc: float
    tx_hash: Optional[str] = None
    error: Optional[str] = None
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    @property
    def effective_rate(self) -> float:
        """token_out per unit token_in."""
        if self.amount_in == 0:
            return 0.0
        return self.amount_out / self.amount_in


@dataclass
class VaultBalance:
    """Snapshot of vault holdings."""
    vault_address: str
    usdc_balance: float
    eth_balance: float
    btc_balance: float
    total_value_usdc: float
    last_updated: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def get_balance(self, token: str) -> float:
        token = token.upper()
        if token == "USDC":
            return self.usdc_balance
        elif token == "ETH":
            return self.eth_balance
        elif token == "BTC":
            return self.btc_balance
        return 0.0


@dataclass
class Position:
    """An open position held by the agent through Surge."""
    position_id: str
    token: str
    size: float             # Amount of token held
    entry_price: float      # Price at which position was opened
    current_price: float    # Latest mark price
    pnl_usdc: float = 0.0
    opened_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    is_open: bool = True

    @property
    def pnl_pct(self) -> float:
        if self.entry_price == 0:
            return 0.0
        return (self.current_price - self.entry_price) / self.entry_price * 100

    @property
    def value_usdc(self) -> float:
        return self.size * self.current_price


# ─── Abstract Interface ────────────────────────────────────────────────────────

class SurgeRouterBase:
    """
    Abstract base class defining the Surge router interface.

    All methods are async to support both mock (in-memory) and
    real on-chain implementations without interface changes.
    """

    async def execute_trade(
        self,
        token_in: str,
        token_out: str,
        amount: float,
        min_out: float = 0.0,
    ) -> TradeResult:
        raise NotImplementedError

    async def get_vault_balance(self, vault_address: Optional[str] = None) -> VaultBalance:
        raise NotImplementedError

    async def get_position(self, token: str) -> Optional[Position]:
        raise NotImplementedError

    async def list_positions(self) -> List[Position]:
        raise NotImplementedError

    async def close_position(self, position_id: str) -> TradeResult:
        raise NotImplementedError

    async def get_token_price(self, token: str) -> float:
        raise NotImplementedError

    async def get_supported_tokens(self) -> List[str]:
        raise NotImplementedError


# ─── Mock Implementation ──────────────────────────────────────────────────────

# Simulated token prices (USD)
_MOCK_PRICES: Dict[str, float] = {
    "ETH": 2000.0,
    "BTC": 45000.0,
    "USDC": 1.0,
    "SOL": 120.0,
    "MATIC": 0.90,
}

_SUPPORTED_TOKENS = list(_MOCK_PRICES.keys())


class MockSurgeRouter(SurgeRouterBase):
    """
    In-memory mock of the Surge Risk Router.

    Simulates trade execution with configurable slippage and fees.
    Used for development and testing before the real Surge ABI ships (March 9).

    Args:
        initial_usdc: Starting USDC balance in the mock vault.
        slippage_pct: Simulated slippage on each fill.
        fee_pct: Trading fee per leg.
        latency_ms: Simulated network latency.
    """

    DEFAULT_VAULT = "0xMockSurgeVault0000000000000000000000000"

    def __init__(
        self,
        initial_usdc: float = 10_000.0,
        slippage_pct: float = 0.001,
        fee_pct: float = 0.003,
        latency_ms: float = 0.0,
    ) -> None:
        if initial_usdc < 0:
            raise ValueError("initial_usdc must be non-negative")
        if not 0 <= slippage_pct < 1:
            raise ValueError("slippage_pct must be in [0, 1)")
        if not 0 <= fee_pct < 0.1:
            raise ValueError("fee_pct must be in [0, 0.1)")

        self._usdc = initial_usdc
        self._eth = 0.0
        self._btc = 0.0
        self._slippage_pct = slippage_pct
        self._fee_pct = fee_pct
        self._latency_ms = latency_ms
        self._positions: Dict[str, Position] = {}
        self._trade_count = 0
        self._prices = dict(_MOCK_PRICES)
        logger.debug(f"MockSurgeRouter init: USDC={initial_usdc}")

    # ── Internal helpers ───────────────────────────────────────────────────────

    def _get_price(self, token: str) -> float:
        token = token.upper()
        if token not in self._prices:
            raise ValueError(f"Unsupported token: {token}")
        return self._prices[token]

    def _new_trade_id(self) -> str:
        self._trade_count += 1
        return f"mock-trade-{self._trade_count:05d}-{uuid.uuid4().hex[:8]}"

    async def _simulate_latency(self) -> None:
        if self._latency_ms > 0:
            await asyncio.sleep(self._latency_ms / 1000)

    def _adjust_balance(self, token: str, delta: float) -> None:
        token = token.upper()
        if token == "USDC":
            self._usdc += delta
        elif token == "ETH":
            self._eth += delta
        elif token == "BTC":
            self._btc += delta

    def _get_balance(self, token: str) -> float:
        token = token.upper()
        if token == "USDC":
            return self._usdc
        elif token == "ETH":
            return self._eth
        elif token == "BTC":
            return self._btc
        return 0.0

    # ── Public API ─────────────────────────────────────────────────────────────

    async def execute_trade(
        self,
        token_in: str,
        token_out: str,
        amount: float,
        min_out: float = 0.0,
    ) -> TradeResult:
        """
        Execute a swap from token_in to token_out.

        Args:
            token_in: Source token symbol (e.g. "USDC").
            token_out: Destination token symbol (e.g. "ETH").
            amount: Amount of token_in to swap.
            min_out: Minimum acceptable amount of token_out (slippage guard).

        Returns:
            TradeResult with execution details.
        """
        await self._simulate_latency()

        tid = self._new_trade_id()

        # Validation
        if amount <= 0:
            return TradeResult(
                success=False, trade_id=tid, token_in=token_in,
                token_out=token_out, amount_in=amount, amount_out=0,
                executed_price=0, slippage_pct=0, fee_usdc=0,
                error="amount must be positive",
            )

        available = self._get_balance(token_in)
        if available < amount:
            return TradeResult(
                success=False, trade_id=tid, token_in=token_in,
                token_out=token_out, amount_in=amount, amount_out=0,
                executed_price=0, slippage_pct=0, fee_usdc=0,
                error=f"Insufficient {token_in}: have {available:.4f}, need {amount:.4f}",
            )

        price_in = self._get_price(token_in)
        price_out = self._get_price(token_out)
        value_usdc = amount * price_in
        fee_usdc = value_usdc * self._fee_pct

        # Apply slippage (adverse)
        slippage = self._slippage_pct
        amount_out_gross = (value_usdc - fee_usdc) / price_out
        amount_out = amount_out_gross * (1 - slippage)

        if amount_out < min_out:
            return TradeResult(
                success=False, trade_id=tid, token_in=token_in,
                token_out=token_out, amount_in=amount, amount_out=0,
                executed_price=0, slippage_pct=slippage, fee_usdc=fee_usdc,
                error=f"Slippage guard: got {amount_out:.6f}, min_out={min_out:.6f}",
            )

        # Update balances
        self._adjust_balance(token_in, -amount)
        self._adjust_balance(token_out, amount_out)

        executed_price = price_out * (1 + slippage)
        logger.info(f"Trade {tid}: {amount} {token_in} → {amount_out:.6f} {token_out}")

        return TradeResult(
            success=True,
            trade_id=tid,
            token_in=token_in.upper(),
            token_out=token_out.upper(),
            amount_in=amount,
            amount_out=amount_out,
            executed_price=executed_price,
            slippage_pct=slippage,
            fee_usdc=fee_usdc,
            tx_hash=f"0x{uuid.uuid4().hex}",
        )

    async def get_vault_balance(self, vault_address: Optional[str] = None) -> VaultBalance:
        await self._simulate_latency()
        eth_price = self._prices["ETH"]
        btc_price = self._prices["BTC"]
        total = self._usdc + self._eth * eth_price + self._btc * btc_price
        return VaultBalance(
            vault_address=vault_address or self.DEFAULT_VAULT,
            usdc_balance=self._usdc,
            eth_balance=self._eth,
            btc_balance=self._btc,
            total_value_usdc=total,
        )

    async def get_position(self, token: str) -> Optional[Position]:
        await self._simulate_latency()
        token = token.upper()
        for pos in self._positions.values():
            if pos.token == token and pos.is_open:
                # Refresh mark price
                pos.current_price = self._get_price(token)
                pos.pnl_usdc = (pos.current_price - pos.entry_price) * pos.size
                return pos
        return None

    async def list_positions(self) -> List[Position]:
        await self._simulate_latency()
        result = []
        for pos in self._positions.values():
            if pos.is_open:
                pos.current_price = self._get_price(pos.token)
                pos.pnl_usdc = (pos.current_price - pos.entry_price) * pos.size
                result.append(pos)
        return result

    async def close_position(self, position_id: str) -> TradeResult:
        await self._simulate_latency()
        pos = self._positions.get(position_id)
        if not pos or not pos.is_open:
            tid = self._new_trade_id()
            return TradeResult(
                success=False, trade_id=tid, token_in=pos.token if pos else "?",
                token_out="USDC", amount_in=0, amount_out=0,
                executed_price=0, slippage_pct=0, fee_usdc=0,
                error=f"Position {position_id} not found or already closed",
            )
        result = await self.execute_trade(pos.token, "USDC", pos.size)
        if result.success:
            pos.is_open = False
        return result

    async def get_token_price(self, token: str) -> float:
        await self._simulate_latency()
        return self._get_price(token)

    async def get_supported_tokens(self) -> List[str]:
        await self._simulate_latency()
        return list(self._prices.keys())

    def set_price(self, token: str, price: float) -> None:
        """Test helper: override mock price."""
        self._prices[token.upper()] = price

    def open_position(self, token: str, size: float, entry_price: float) -> Position:
        """Test helper: directly add a position to the mock state."""
        pos = Position(
            position_id=str(uuid.uuid4()),
            token=token.upper(),
            size=size,
            entry_price=entry_price,
            current_price=entry_price,
        )
        self._positions[pos.position_id] = pos
        self._adjust_balance(token, size)
        return pos
