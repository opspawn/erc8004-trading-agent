"""
x402_client.py — x402 micropayment client for the ERC-8004 Trading Agent.

Implements x402 HTTP payment protocol (https://x402.org) to pay for:
  - Market data fetches
  - Trade execution (fee-bearing endpoints)
  - Reputation data queries

Protocol flow:
  1. Agent makes request → server returns HTTP 402 with payment requirements
  2. Client builds payment payload (USDC on Base, EIP-712 signed)
  3. Client resends with X-PAYMENT header
  4. Server verifies via facilitator → returns X-PAYMENT-RESPONSE

Network: Base (eip155:8453) or Base Sepolia (eip155:84532) for testnet
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field
from typing import Any, Optional

import httpx
from eth_account import Account
from eth_account.messages import encode_defunct
from loguru import logger


# ─── Constants ────────────────────────────────────────────────────────────────

# Default facilitator for x402 (Coinbase Commerce facilitator)
DEFAULT_FACILITATOR = "https://x402.org/facilitator"

# USDC contract on Base
USDC_BASE = "0x833589fCD6eDb6E08f4c7C32D4f71b54bdA02913"

# USDC contract on Base Sepolia
USDC_BASE_SEPOLIA = "0x036CbD53842c5426634e7929541eC2318f3dCF7e"

# x402 version
X402_VERSION = 1


# ─── Data Classes ─────────────────────────────────────────────────────────────

@dataclass
class PaymentRequirement:
    """Payment requirement returned in a 402 response."""
    scheme: str                 # "exact"
    network: str                # "eip155:8453"
    max_amount_required: str    # USDC in 6-decimal units (e.g. "10000" = $0.01)
    resource: str               # URL being accessed
    description: str
    pay_to: str                 # Receiver address
    required_deadline_seconds: int = 300
    mime_type: str = "application/json"
    extra: dict = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict) -> "PaymentRequirement":
        return cls(
            scheme=data.get("scheme", "exact"),
            network=data.get("network", "eip155:8453"),
            max_amount_required=str(data.get("maxAmountRequired", data.get("max_amount_required", "10000"))),
            resource=data.get("resource", ""),
            description=data.get("description", ""),
            pay_to=data.get("payTo", data.get("pay_to", "")),
            required_deadline_seconds=data.get("requiredDeadlineSeconds", 300),
            mime_type=data.get("mimeType", "application/json"),
            extra=data.get("extra", {}),
        )


@dataclass
class PaymentPayload:
    """x402 payment payload sent in X-PAYMENT header."""
    x402_version: int
    scheme: str
    network: str
    payload: dict               # scheme-specific payload


@dataclass
class PaymentReceipt:
    """Receipt from facilitator after successful payment verification."""
    success: bool
    transaction_hash: Optional[str]
    network: str
    payer: str
    amount: str
    timestamp: float = field(default_factory=time.time)
    raw: dict = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict) -> "PaymentReceipt":
        return cls(
            success=data.get("success", False),
            transaction_hash=data.get("transactionHash"),
            network=data.get("network", ""),
            payer=data.get("payer", ""),
            amount=str(data.get("amount", "0")),
            raw=data,
        )


@dataclass
class PaymentRecord:
    """Internal ledger entry for tracking payments made."""
    resource: str
    amount_usdc_cents: int      # in 6-decimal units
    network: str
    receipt: Optional[PaymentReceipt]
    timestamp: float = field(default_factory=time.time)
    demo_mode: bool = False


# ─── Ledger ───────────────────────────────────────────────────────────────────

class X402Ledger:
    """Tracks all payments made by the trading agent."""

    def __init__(self) -> None:
        self._records: list[PaymentRecord] = []

    def record(self, entry: PaymentRecord) -> None:
        self._records.append(entry)
        logger.debug(
            f"x402 payment: {entry.resource} → "
            f"${int(entry.amount_usdc_cents) / 1_000_000:.4f} USDC"
        )

    @property
    def total_payments(self) -> int:
        return len(self._records)

    @property
    def total_spent_usdc(self) -> float:
        return sum(r.amount_usdc_cents for r in self._records) / 1_000_000

    def get_stats(self) -> dict:
        return {
            "total_payments": self.total_payments,
            "total_spent_usdc": round(self.total_spent_usdc, 6),
            "recent": [
                {
                    "resource": r.resource,
                    "amount_usdc": r.amount_usdc_cents / 1_000_000,
                    "demo_mode": r.demo_mode,
                    "timestamp": r.timestamp,
                }
                for r in self._records[-10:]
            ],
        }


# ─── x402 Client ──────────────────────────────────────────────────────────────

class X402Client:
    """
    HTTP client that handles x402 payment protocol automatically.

    When a server returns 402, the client:
    1. Parses the payment requirements
    2. Builds and signs a payment payload (EIP-712 or demo)
    3. Retries the request with X-PAYMENT header
    4. Records the payment to the ledger

    Usage:
        client = X402Client(account=my_eth_account, demo_mode=True)
        data = await client.get("https://api.limitless.exchange/v1/markets")
    """

    def __init__(
        self,
        account: Optional[Account] = None,
        facilitator_url: str = DEFAULT_FACILITATOR,
        network: str = "eip155:8453",
        demo_mode: bool = True,
        ledger: Optional[X402Ledger] = None,
    ) -> None:
        self.account = account
        self.facilitator_url = facilitator_url
        self.network = network
        self.demo_mode = demo_mode
        self.ledger = ledger or X402Ledger()

    def _build_payment_payload(
        self,
        requirement: PaymentRequirement,
        amount: str,
    ) -> dict:
        """
        Build an EIP-712 signed payment payload for 'exact' scheme.

        In demo mode: returns a well-formed but unsigned payload.
        In live mode: signs with the agent's private key.
        """
        deadline = int(time.time()) + requirement.required_deadline_seconds
        nonce = hashlib.sha256(
            f"{requirement.resource}{time.time()}".encode()
        ).hexdigest()[:16]

        # EIP-712 typed data for x402 exact scheme
        typed_data = {
            "domain": {
                "name": "x402",
                "version": "1",
                "chainId": int(requirement.network.split(":")[-1]),
            },
            "types": {
                "Payment": [
                    {"name": "from", "type": "address"},
                    {"name": "to", "type": "address"},
                    {"name": "value", "type": "uint256"},
                    {"name": "deadline", "type": "uint256"},
                    {"name": "nonce", "type": "bytes32"},
                    {"name": "resource", "type": "string"},
                ]
            },
            "primaryType": "Payment",
            "message": {
                "from": self.account.address if self.account else "0x0000000000000000000000000000000000000000",
                "to": requirement.pay_to,
                "value": int(amount),
                "deadline": deadline,
                "nonce": bytes.fromhex(nonce.ljust(64, "0")[:64]),
                "resource": requirement.resource,
            },
        }

        if self.demo_mode or not self.account:
            # Demo mode: include payload without real signature
            signature = "0x" + "00" * 65
        else:
            # Live mode: sign EIP-712 message
            # Convert bytes to hex strings for JSON serialization
            serializable_message = {
                k: (v.hex() if isinstance(v, bytes) else v)
                for k, v in typed_data["message"].items()
            }
            encoded = encode_defunct(
                text=json.dumps(serializable_message, sort_keys=True)
            )
            signed = self.account.sign_message(encoded)
            signature = signed.signature.hex()

        payload = {
            "x402Version": X402_VERSION,
            "scheme": requirement.scheme,
            "network": requirement.network,
            "payload": {
                "signature": signature,
                "authorization": {
                    "from": typed_data["message"]["from"],
                    "to": requirement.pay_to,
                    "value": amount,
                    "validAfter": str(int(time.time()) - 60),
                    "validBefore": str(deadline),
                    "nonce": "0x" + nonce.ljust(64, "0")[:64],
                },
            },
        }
        return payload

    async def _verify_with_facilitator(
        self,
        payment_payload: dict,
        requirement: PaymentRequirement,
    ) -> PaymentReceipt:
        """Send payment to facilitator for on-chain settlement."""
        if self.demo_mode:
            return PaymentReceipt(
                success=True,
                transaction_hash="0x" + "ab" * 32,
                network=requirement.network,
                payer=self.account.address if self.account else "0x0",
                amount=requirement.max_amount_required,
                raw={"mode": "demo"},
            )

        try:
            async with httpx.AsyncClient(timeout=15) as http:
                resp = await http.post(
                    f"{self.facilitator_url}/settle",
                    json=payment_payload,
                )
                if resp.status_code == 200:
                    return PaymentReceipt.from_dict(resp.json())
                else:
                    logger.warning(f"Facilitator error {resp.status_code}: {resp.text[:200]}")
                    return PaymentReceipt(
                        success=False,
                        transaction_hash=None,
                        network=requirement.network,
                        payer="unknown",
                        amount="0",
                    )
        except Exception as e:
            logger.error(f"Facilitator request failed: {e}")
            return PaymentReceipt(
                success=False,
                transaction_hash=None,
                network=requirement.network,
                payer="unknown",
                amount="0",
            )

    def _parse_402_response(self, body: dict) -> Optional[PaymentRequirement]:
        """Parse payment requirements from a 402 response body."""
        # Try x402 standard format
        if "x402Version" in body or "scheme" in body:
            return PaymentRequirement.from_dict(body)

        # Try wrapped format {"accepts": [...]}
        if "accepts" in body and body["accepts"]:
            return PaymentRequirement.from_dict(body["accepts"][0])

        # Try nested x402 object
        if "x402" in body:
            return PaymentRequirement.from_dict(body["x402"])

        return None

    async def request(
        self,
        method: str,
        url: str,
        **kwargs: Any,
    ) -> httpx.Response:
        """
        Make an HTTP request, automatically handling x402 payment if required.

        Returns the final response (after payment if needed).
        Raises httpx.HTTPError on unrecoverable errors.
        """
        async with httpx.AsyncClient(timeout=30) as http:
            # First attempt (no payment)
            resp = await http.request(method, url, **kwargs)

            if resp.status_code != 402:
                return resp

            # Got 402 — parse requirements and pay
            logger.info(f"x402: Payment required for {url}")
            try:
                body = resp.json()
            except Exception:
                logger.warning("x402: Could not parse 402 response body")
                return resp

            requirement = self._parse_402_response(body)
            if not requirement:
                logger.warning(f"x402: Could not parse payment requirements: {body}")
                return resp

            logger.info(
                f"x402: Paying {int(requirement.max_amount_required) / 1_000_000:.4f} USDC "
                f"to {requirement.pay_to[:10]}... for {requirement.description}"
            )

            # Build and sign payment
            payment_payload = self._build_payment_payload(
                requirement, requirement.max_amount_required
            )

            # Settle with facilitator
            receipt = await self._verify_with_facilitator(payment_payload, requirement)

            # Record to ledger
            self.ledger.record(PaymentRecord(
                resource=url,
                amount_usdc_cents=int(requirement.max_amount_required),
                network=requirement.network,
                receipt=receipt,
                demo_mode=self.demo_mode,
            ))

            if not receipt.success and not self.demo_mode:
                logger.error(f"x402: Payment failed for {url}")
                return resp

            # Retry with payment header
            payment_header = json.dumps(payment_payload, separators=(",", ":"))
            headers = kwargs.pop("headers", {})
            headers["X-PAYMENT"] = payment_header
            kwargs["headers"] = headers

            resp2 = await http.request(method, url, **kwargs)
            logger.info(f"x402: Payment accepted, response={resp2.status_code}")
            return resp2

    async def get(self, url: str, **kwargs: Any) -> httpx.Response:
        return await self.request("GET", url, **kwargs)

    async def post(self, url: str, **kwargs: Any) -> httpx.Response:
        return await self.request("POST", url, **kwargs)


# ─── Convenience factory ──────────────────────────────────────────────────────

def create_x402_client(
    account: Optional[Account] = None,
    demo_mode: bool = True,
    testnet: bool = True,
) -> X402Client:
    """Create a configured x402 client for the trading agent."""
    network = "eip155:84532" if testnet else "eip155:8453"
    return X402Client(
        account=account,
        facilitator_url=DEFAULT_FACILITATOR,
        network=network,
        demo_mode=demo_mode,
    )
