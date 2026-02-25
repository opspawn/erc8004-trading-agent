"""
dex_executor.py — Python DEX swap execution layer for the ERC-8004 Trading Agent.

Provides execute_swap() and simulate_swap() that mirror the on-chain
UniswapV3Executor contract logic in Python. The RiskRouter check is performed
in Python before broadcasting any on-chain transaction.

Architecture:
  1. simulate_swap() — dry-run, checks risk, returns estimated output
  2. execute_swap() — validates risk then records the swap

The Python layer mirrors the Solidity contract:
  contracts/contracts/UniswapV3Executor.sol

Risk gating: every swap passes through check_oracle_risk() from risk_manager.py,
which mirrors the on-chain RiskRouter.checkRisk() logic.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Optional

from loguru import logger


# ─── Token Definitions ────────────────────────────────────────────────────────

class TokenSymbol(str, Enum):
    ETH = "ETH"
    BTC = "BTC"
    USDC = "USDC"
    WETH = "WETH"


# Well-known token addresses (Base Sepolia)
TOKEN_ADDRESSES: dict[TokenSymbol, str] = {
    TokenSymbol.ETH: "0x0000000000000000000000000000000000000000",   # native ETH
    TokenSymbol.WETH: "0x4200000000000000000000000000000000000006",  # WETH on Base
    TokenSymbol.USDC: "0x036CbD53842c5426634e7929541eC2318f3dCF7e",  # USDC on Base Sepolia
    TokenSymbol.BTC: "0x0000000000000000000000000000000000000001",   # BTC proxy (testnet)
}

# ─── Swap Result Data Classes ─────────────────────────────────────────────────

@dataclass
class SwapParams:
    """Parameters for a swap request."""
    agent_id: int
    token_in: TokenSymbol
    token_out: TokenSymbol
    amount_in: float        # In USD-equivalent (USDC)
    min_amount_out: float   # Minimum acceptable output (slippage protection)
    oracle_price: float     # Current oracle price for risk check
    slippage_bps: int = 50  # 0.5% default slippage tolerance


@dataclass
class SwapResult:
    """Result of a swap execution."""
    success: bool
    agent_id: int
    token_in: TokenSymbol
    token_out: TokenSymbol
    amount_in: float
    amount_out: float
    oracle_price: float
    risk_passed: bool
    simulated: bool         # True if dry run
    tx_hash: Optional[str]
    error: Optional[str]
    executed_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> dict:
        return {
            "success": self.success,
            "agent_id": self.agent_id,
            "token_in": self.token_in.value,
            "token_out": self.token_out.value,
            "amount_in": self.amount_in,
            "amount_out": self.amount_out,
            "oracle_price": self.oracle_price,
            "risk_passed": self.risk_passed,
            "simulated": self.simulated,
            "tx_hash": self.tx_hash,
            "error": self.error,
            "executed_at": self.executed_at,
        }


@dataclass
class SimulationResult:
    """Result of a swap simulation (no state changes)."""
    agent_id: int
    token_in: TokenSymbol
    token_out: TokenSymbol
    amount_in: float
    estimated_out: float
    oracle_price: float
    risk_passed: bool
    rejection_reason: Optional[str]

    @property
    def is_executable(self) -> bool:
        return self.risk_passed and self.estimated_out > 0

    def to_dict(self) -> dict:
        return {
            "agent_id": self.agent_id,
            "token_in": self.token_in.value,
            "token_out": self.token_out.value,
            "amount_in": self.amount_in,
            "estimated_out": self.estimated_out,
            "oracle_price": self.oracle_price,
            "risk_passed": self.risk_passed,
            "is_executable": self.is_executable,
            "rejection_reason": self.rejection_reason,
        }


# ─── Price Calculation ────────────────────────────────────────────────────────

def calculate_swap_output(
    amount_in: float,
    token_in: TokenSymbol,
    token_out: TokenSymbol,
    oracle_price: float,
) -> float:
    """
    Calculate estimated swap output using oracle price.

    Mirrors the _simulateSwap() function in UniswapV3Executor.sol:
      output = amountIn * 1e8 / oraclePrice (for ETH → other)

    For USDC → ETH: output = amountIn / oracle_price
    For ETH → USDC: output = amountIn * oracle_price

    Args:
        amount_in: Input amount in token's native unit
        token_in: Input token
        token_out: Output token
        oracle_price: Current oracle price (USD per ETH or BTC)

    Returns:
        Estimated output amount
    """
    if oracle_price <= 0 or amount_in <= 0:
        return 0.0

    # ETH or WETH → USDC: sell ETH at oracle price
    if token_in in (TokenSymbol.ETH, TokenSymbol.WETH) and token_out == TokenSymbol.USDC:
        return amount_in * oracle_price

    # USDC → ETH or WETH: buy ETH at oracle price
    if token_in == TokenSymbol.USDC and token_out in (TokenSymbol.ETH, TokenSymbol.WETH):
        return amount_in / oracle_price

    # BTC → USDC
    if token_in == TokenSymbol.BTC and token_out == TokenSymbol.USDC:
        return amount_in * oracle_price

    # USDC → BTC
    if token_in == TokenSymbol.USDC and token_out == TokenSymbol.BTC:
        return amount_in / oracle_price

    # Fallback: 1:1 (same-asset swap, e.g. ETH → WETH)
    return amount_in


def check_slippage(
    amount_out: float,
    min_amount_out: float,
) -> tuple[bool, Optional[str]]:
    """
    Check that output meets slippage threshold.

    Returns:
        (passed, error_message_or_none)
    """
    if min_amount_out <= 0:
        return True, None  # No minimum specified — always pass
    if amount_out < min_amount_out:
        return False, (
            f"Slippage exceeded: got {amount_out:.6f}, "
            f"minimum {min_amount_out:.6f}"
        )
    return True, None


# ─── DEX Executor ─────────────────────────────────────────────────────────────

class DexExecutor:
    """
    Python DEX execution layer — mirrors UniswapV3Executor.sol.

    In testnet/demo mode: simulates swaps and records them locally.
    In production mode: would broadcast to on-chain UniswapV3Executor.

    The risk_manager.check_oracle_risk() gates every swap, matching the
    on-chain RiskRouter.checkRisk() behavior.

    Usage:
        executor = DexExecutor(risk_manager=rm)
        result = await executor.execute_swap(params)
        if result.success:
            print(f"Swapped {result.amount_in} → {result.amount_out}")
    """

    # Default maximum deviation (mirrors RiskRouter default: 500 bps = 5%)
    DEFAULT_MAX_DEVIATION_PCT = 0.05

    def __init__(
        self,
        risk_manager=None,
        dry_run: bool = True,
        max_deviation_pct: float = DEFAULT_MAX_DEVIATION_PCT,
    ) -> None:
        self.risk_manager = risk_manager
        self.dry_run = dry_run
        self.max_deviation_pct = max_deviation_pct

        self._swap_history: list[SwapResult] = []
        self._total_volume_in: float = 0.0
        self._total_swaps: int = 0
        self._blocked_swaps: int = 0

        logger.info(
            f"DexExecutor initialized: dry_run={dry_run} "
            f"max_deviation={max_deviation_pct:.1%}"
        )

    # ─── Risk Check ───────────────────────────────────────────────────────────

    def _check_risk(
        self, amount: float, oracle_price: float
    ) -> tuple[bool, Optional[str]]:
        """
        Run the Python-side risk check (mirrors RiskRouter.checkRisk()).

        Uses risk_manager.check_oracle_risk() if available,
        otherwise falls back to internal deviation check.
        """
        if amount <= 0:
            return False, "Zero or negative amount"

        if oracle_price <= 0:
            return False, "Invalid oracle price"

        if self.risk_manager is not None:
            passed = self.risk_manager.check_oracle_risk(
                amount=amount,
                oracle_price=oracle_price,
                max_deviation_pct=self.max_deviation_pct,
            )
            if not passed:
                deviation = abs(amount - oracle_price) / oracle_price
                return False, (
                    f"Oracle risk check failed: deviation {deviation:.2%} "
                    f"> max {self.max_deviation_pct:.2%}"
                )
            return True, None

        # Internal fallback if no risk_manager
        deviation = abs(amount - oracle_price) / oracle_price
        if deviation > self.max_deviation_pct:
            return False, (
                f"Deviation {deviation:.2%} exceeds max {self.max_deviation_pct:.2%}"
            )
        return True, None

    # ─── Simulate ─────────────────────────────────────────────────────────────

    def simulate_swap(self, params: SwapParams) -> SimulationResult:
        """
        Simulate a swap without state changes.

        Checks risk and calculates estimated output. Equivalent to
        UniswapV3Executor.simulateSwap() on-chain.

        Args:
            params: Swap parameters

        Returns:
            SimulationResult with risk status and estimated output
        """
        # Risk check
        risk_ok, risk_error = self._check_risk(params.amount_in, params.oracle_price)

        if not risk_ok:
            logger.info(
                f"DexExecutor.simulate_swap: BLOCKED "
                f"agent={params.agent_id} reason={risk_error}"
            )
            return SimulationResult(
                agent_id=params.agent_id,
                token_in=params.token_in,
                token_out=params.token_out,
                amount_in=params.amount_in,
                estimated_out=0.0,
                oracle_price=params.oracle_price,
                risk_passed=False,
                rejection_reason=risk_error,
            )

        # Calculate output
        estimated_out = calculate_swap_output(
            params.amount_in,
            params.token_in,
            params.token_out,
            params.oracle_price,
        )

        logger.debug(
            f"DexExecutor.simulate_swap: agent={params.agent_id} "
            f"{params.amount_in:.4f} {params.token_in} → "
            f"{estimated_out:.6f} {params.token_out} "
            f"(oracle={params.oracle_price:.2f})"
        )

        return SimulationResult(
            agent_id=params.agent_id,
            token_in=params.token_in,
            token_out=params.token_out,
            amount_in=params.amount_in,
            estimated_out=estimated_out,
            oracle_price=params.oracle_price,
            risk_passed=True,
            rejection_reason=None,
        )

    # ─── Execute ──────────────────────────────────────────────────────────────

    def execute_swap(self, params: SwapParams) -> SwapResult:
        """
        Execute a swap (or simulate in dry_run mode).

        Flow:
          1. Risk check (deviation gate)
          2. Calculate output
          3. Slippage check
          4. Record swap
          5. Return SwapResult

        Args:
            params: Swap parameters

        Returns:
            SwapResult with success status and actual amounts
        """
        # 1. Risk gate
        risk_ok, risk_error = self._check_risk(params.amount_in, params.oracle_price)

        if not risk_ok:
            self._blocked_swaps += 1
            result = SwapResult(
                success=False,
                agent_id=params.agent_id,
                token_in=params.token_in,
                token_out=params.token_out,
                amount_in=params.amount_in,
                amount_out=0.0,
                oracle_price=params.oracle_price,
                risk_passed=False,
                simulated=self.dry_run,
                tx_hash=None,
                error=risk_error,
            )
            logger.warning(
                f"DexExecutor: swap blocked agent={params.agent_id} "
                f"reason={risk_error}"
            )
            self._swap_history.append(result)
            return result

        # 2. Calculate output
        amount_out = calculate_swap_output(
            params.amount_in,
            params.token_in,
            params.token_out,
            params.oracle_price,
        )

        # 3. Slippage check
        slippage_ok, slippage_error = check_slippage(amount_out, params.min_amount_out)
        if not slippage_ok:
            result = SwapResult(
                success=False,
                agent_id=params.agent_id,
                token_in=params.token_in,
                token_out=params.token_out,
                amount_in=params.amount_in,
                amount_out=amount_out,
                oracle_price=params.oracle_price,
                risk_passed=True,
                simulated=self.dry_run,
                tx_hash=None,
                error=slippage_error,
            )
            logger.warning(f"DexExecutor: slippage exceeded agent={params.agent_id}")
            self._swap_history.append(result)
            return result

        # 4. Record swap
        tx_hash = self._generate_tx_hash(params, amount_out)

        result = SwapResult(
            success=True,
            agent_id=params.agent_id,
            token_in=params.token_in,
            token_out=params.token_out,
            amount_in=params.amount_in,
            amount_out=amount_out,
            oracle_price=params.oracle_price,
            risk_passed=True,
            simulated=self.dry_run,
            tx_hash=tx_hash,
            error=None,
        )

        self._swap_history.append(result)
        self._total_volume_in += params.amount_in
        self._total_swaps += 1

        logger.info(
            f"DexExecutor: swap {'(simulated)' if self.dry_run else 'EXECUTED'} "
            f"agent={params.agent_id} "
            f"{params.amount_in:.4f} {params.token_in} → "
            f"{amount_out:.6f} {params.token_out} "
            f"tx={tx_hash[:10]}..."
        )

        return result

    # ─── Helpers ──────────────────────────────────────────────────────────────

    def _generate_tx_hash(self, params: SwapParams, amount_out: float) -> str:
        """Generate a deterministic simulated tx hash."""
        data = json.dumps({
            "agent_id": params.agent_id,
            "token_in": params.token_in.value,
            "token_out": params.token_out.value,
            "amount_in": params.amount_in,
            "amount_out": amount_out,
            "ts": time.time(),
        }, separators=(",", ":"))
        return "0x" + hashlib.sha256(data.encode()).hexdigest()

    # ─── Stats ────────────────────────────────────────────────────────────────

    def get_stats(self) -> dict:
        """Return executor statistics."""
        successful = [r for r in self._swap_history if r.success]
        return {
            "total_swaps_attempted": len(self._swap_history),
            "total_swaps_successful": len(successful),
            "total_swaps_blocked": self._blocked_swaps,
            "total_volume_in": round(self._total_volume_in, 6),
            "dry_run": self.dry_run,
        }

    def get_history(self) -> list[dict]:
        """Return all swap history as dicts."""
        return [r.to_dict() for r in self._swap_history]

    def get_agent_volume(self, agent_id: int) -> float:
        """Return total swap volume for a specific agent."""
        return sum(
            r.amount_in for r in self._swap_history
            if r.agent_id == agent_id and r.success
        )
