"""
x402_signal_server.py — ERC-8004 x402 Paid Signal API.

Exposes the agent's live trading signals as a paid API using x402
micropayments ($0.01 USDC per request). Implements the x402 payment
flow manually without external dependencies:

  1. Client requests /signals/latest without payment
  2. Server returns 402 with X-Payment-Required header
  3. Client retries with X-Payment header containing payment proof
  4. Server validates (or simulates validation in DEV_MODE) and returns data

Routes:
    GET /signals/latest          — last 10 trade signals (x402 gated)
    GET /signals/backtest/{sym}  — backtest summary for a symbol (x402 gated)
    GET /health                  — server status (free)

Port: 8083

Usage (standalone):
    python3 x402_signal_server.py

Usage (programmatic):
    from x402_signal_server import create_app
    app = create_app()
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import random
import time
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, Header, HTTPException, Request, Response
from fastapi.responses import JSONResponse

# ─── Constants ────────────────────────────────────────────────────────────────

PORT = 8083

# Wallet address to receive payments (Polygon mainnet)
PAYMENT_ADDRESS = "0x7483a9F237cf8043704D6b17DA31c12BfFF860DD"

# Price per request in USDC
SIGNAL_PRICE_USDC = 0.01
BACKTEST_PRICE_USDC = 0.01

# DEV_MODE: True = simulate payment validation (always accept X-Payment header)
DEV_MODE = os.environ.get("X402_DEV_MODE", "true").lower() != "false"

# Supported assets for demo signals
ASSETS = ["BTC/USD", "ETH/USD", "SOL/USD", "MATIC/USD", "AVAX/USD"]


# ─── Payment data classes ─────────────────────────────────────────────────────


@dataclass
class PaymentRequirement:
    """Describes what payment is required for a resource."""
    amount_usdc: float
    currency: str
    network: str
    address: str
    resource: str

    def to_header_value(self) -> str:
        return json.dumps({
            "amount": str(self.amount_usdc),
            "currency": self.currency,
            "network": self.network,
            "address": self.address,
            "resource": self.resource,
        })

    def to_dict(self) -> Dict[str, Any]:
        return {
            "amount": self.amount_usdc,
            "currency": self.currency,
            "network": self.network,
            "address": self.address,
            "resource": self.resource,
        }


@dataclass
class PaymentProof:
    """Payment proof submitted by the client via X-Payment header."""
    tx_hash: str
    amount: str
    currency: str
    raw: str = ""

    @classmethod
    def from_header(cls, header_value: str) -> "PaymentProof":
        try:
            data = json.loads(header_value)
            return cls(
                tx_hash=data.get("txHash", data.get("tx_hash", "")),
                amount=str(data.get("amount", "0")),
                currency=data.get("currency", "USDC"),
                raw=header_value,
            )
        except (json.JSONDecodeError, KeyError) as exc:
            raise ValueError(f"Invalid X-Payment header: {exc}") from exc


# ─── Signal generator ─────────────────────────────────────────────────────────


def _gbm_price(base: float, n_steps: int = 1, seed: Optional[int] = None) -> float:
    """Simple GBM price step."""
    rng = random.Random(seed)
    mu, sigma, dt = 0.05, 0.20, 1 / 8760
    p = base
    for _ in range(n_steps):
        u = max(rng.random(), 1e-10)
        v = rng.random()
        z = math.sqrt(-2.0 * math.log(u)) * math.cos(2 * math.pi * v)
        p *= math.exp((mu - 0.5 * sigma**2) * dt + sigma * math.sqrt(dt) * z)
    return p


class SignalStore:
    """
    In-memory store of demo trading signals.
    Generates synthetic signals on demand for the signal API.
    """

    def __init__(self, agent_id: str = "erc8004-v1") -> None:
        self.agent_id = agent_id
        self._signals: List[Dict[str, Any]] = []
        self._base_prices = {
            "BTC/USD": 65000.0,
            "ETH/USD": 3500.0,
            "SOL/USD": 180.0,
            "MATIC/USD": 0.85,
            "AVAX/USD": 38.0,
        }
        # Pre-populate 10 signals
        for i in range(10):
            self._generate_next(seed=i)

    def _generate_next(self, seed: Optional[int] = None) -> Dict[str, Any]:
        rng = random.Random(seed if seed is not None else int(time.time() * 1000) % 2**32)
        symbol = rng.choice(ASSETS)
        base = self._base_prices[symbol]
        price = _gbm_price(base, n_steps=rng.randint(1, 5), seed=seed)
        side = rng.choice(["BUY", "SELL", "HOLD"])
        confidence = round(rng.uniform(0.55, 0.95), 4)
        signal = {
            "signal_id": str(uuid.uuid4()),
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "symbol": symbol,
            "side": side,
            "price": round(price, 4),
            "confidence": confidence,
            "agent_id": self.agent_id,
            "source": "erc8004-momentum",
        }
        self._signals.append(signal)
        if len(self._signals) > 100:
            self._signals = self._signals[-100:]
        return signal

    def get_latest(self, n: int = 10) -> List[Dict[str, Any]]:
        """Return the last n signals, generating new ones if needed."""
        # Refresh with a new signal each call
        self._generate_next()
        return list(reversed(self._signals[-n:]))

    def get_backtest(self, symbol: str) -> Dict[str, Any]:
        """Return a synthetic backtest summary for a symbol."""
        rng = random.Random(hash(symbol) % 2**32)
        n_trades = rng.randint(80, 200)
        win_rate = round(rng.uniform(0.48, 0.62), 4)
        avg_pnl_bps = round(rng.uniform(-10, 40), 2)
        max_dd = round(rng.uniform(50, 300), 2)
        return {
            "symbol": symbol,
            "period": "2024-01-01 / 2024-12-31",
            "n_trades": n_trades,
            "win_rate": win_rate,
            "avg_pnl_bps": avg_pnl_bps,
            "max_drawdown_bps": max_dd,
            "sharpe_ratio": round(rng.uniform(0.5, 2.5), 3),
            "total_return_pct": round(avg_pnl_bps * n_trades / 100, 2),
            "agent_id": "erc8004-v1",
            "strategy": "momentum-v1",
        }


# ─── Payment validation ───────────────────────────────────────────────────────


class PaymentValidator:
    """
    Validates x402 payment proofs.

    In DEV_MODE: always accepts any X-Payment header that is structurally valid.
    In production: would verify txHash on-chain via web3.
    """

    def __init__(self, dev_mode: bool = DEV_MODE) -> None:
        self.dev_mode = dev_mode
        self._validated_hashes: set = set()
        self.total_requests_paid: int = 0
        self.total_earned_usdc: float = 0.0

    def validate(self, proof: PaymentProof, expected_amount: float) -> bool:
        """
        Validate a payment proof.

        Returns True if payment is accepted. Raises ValueError on invalid proof.
        """
        if not proof.tx_hash:
            raise ValueError("Missing txHash in payment proof")

        # Prevent replay: each tx_hash can only be used once
        if proof.tx_hash in self._validated_hashes:
            raise ValueError(f"Payment proof already used: {proof.tx_hash}")

        if self.dev_mode:
            # DEV_MODE: accept any structurally valid proof
            self._validated_hashes.add(proof.tx_hash)
            self.total_requests_paid += 1
            try:
                amount = float(proof.amount)
                self.total_earned_usdc += amount
            except (ValueError, TypeError):
                self.total_earned_usdc += expected_amount
            return True

        # Production: verify on-chain (placeholder — requires web3 + funded wallet)
        raise NotImplementedError(
            "Live payment verification requires web3 and testnet configuration"
        )

    def make_requirement(
        self,
        amount_usdc: float,
        resource: str,
        currency: str = "USDC",
        network: str = "polygon",
    ) -> PaymentRequirement:
        return PaymentRequirement(
            amount_usdc=amount_usdc,
            currency=currency,
            network=network,
            address=PAYMENT_ADDRESS,
            resource=resource,
        )


# ─── x402 middleware helpers ──────────────────────────────────────────────────


def _payment_required_response(requirement: PaymentRequirement) -> Response:
    """Return HTTP 402 with X-Payment-Required header."""
    return Response(
        content=json.dumps({
            "error": "Payment Required",
            "x402": requirement.to_dict(),
        }),
        status_code=402,
        media_type="application/json",
        headers={
            "X-Payment-Required": requirement.to_header_value(),
            "X-Payment-Address": PAYMENT_ADDRESS,
            "X-Payment-Currency": "USDC",
            "X-Payment-Amount": str(requirement.amount_usdc),
            "X-Payment-Network": requirement.network,
        },
    )


def _check_payment(
    x_payment: Optional[str],
    validator: PaymentValidator,
    amount: float,
    resource: str,
) -> Optional[Response]:
    """
    Inspect X-Payment header and validate.

    Returns None if payment is valid (proceed), or a Response to return.
    """
    if not x_payment:
        req = validator.make_requirement(amount, resource)
        return _payment_required_response(req)
    try:
        proof = PaymentProof.from_header(x_payment)
        validator.validate(proof, expected_amount=amount)
        return None  # Payment accepted — proceed
    except ValueError as exc:
        return Response(
            content=json.dumps({"error": str(exc)}),
            status_code=402,
            media_type="application/json",
        )


# ─── App Factory ─────────────────────────────────────────────────────────────


def create_app(
    agent_id: str = "erc8004-v1",
    dev_mode: bool = DEV_MODE,
) -> FastAPI:
    """
    Create and return the FastAPI application.

    Parameters
    ----------
    agent_id : Agent identity string embedded in signals.
    dev_mode : If True, payment validation always passes for valid headers.
    """
    app = FastAPI(
        title="ERC-8004 x402 Signal API",
        description="Live trading signals gated by x402 micropayments",
        version="1.0.0",
    )

    store = SignalStore(agent_id=agent_id)
    validator = PaymentValidator(dev_mode=dev_mode)
    _start_time = time.time()
    _total_free_requests: Dict[str, int] = {"count": 0}

    # ── Routes ────────────────────────────────────────────────────────────

    @app.get("/health")
    async def health() -> Dict[str, Any]:
        """Server health — no payment required."""
        _total_free_requests["count"] += 1
        return {
            "status": "ok",
            "agent_id": agent_id,
            "uptime_seconds": round(time.time() - _start_time, 1),
            "total_requests_paid": validator.total_requests_paid,
            "total_earned_usdc": round(validator.total_earned_usdc, 6),
            "payment_address": PAYMENT_ADDRESS,
            "dev_mode": dev_mode,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

    @app.get("/signals/latest")
    async def signals_latest(
        x_payment: Optional[str] = Header(None, alias="X-Payment"),
    ) -> Response:
        """Return last 10 trade signals. Requires x402 payment ($0.01 USDC)."""
        check = _check_payment(x_payment, validator, SIGNAL_PRICE_USDC, "/signals/latest")
        if check is not None:
            return check

        signals = store.get_latest(n=10)
        return JSONResponse(
            content={
                "signals": signals,
                "count": len(signals),
                "agent_id": agent_id,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            },
            headers={"X-Payment-Accepted": "true"},
        )

    @app.get("/signals/backtest/{symbol:path}")
    async def signals_backtest(
        symbol: str,
        x_payment: Optional[str] = Header(None, alias="X-Payment"),
    ) -> Response:
        """Return backtest summary for a symbol. Requires x402 payment ($0.01 USDC)."""
        check = _check_payment(
            x_payment, validator, BACKTEST_PRICE_USDC, f"/signals/backtest/{symbol}"
        )
        if check is not None:
            return check

        # Normalise symbol (e.g., "btc-usd" → "BTC/USD")
        norm_symbol = symbol.upper().replace("-", "/")
        data = store.get_backtest(norm_symbol)
        return JSONResponse(
            content=data,
            headers={"X-Payment-Accepted": "true"},
        )

    return app


# ─── Entrypoint ───────────────────────────────────────────────────────────────


if __name__ == "__main__":
    import uvicorn

    app = create_app()
    uvicorn.run(app, host="0.0.0.0", port=PORT)
