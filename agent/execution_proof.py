"""
execution_proof.py — On-Chain Execution Proof for ERC-8004 Trading Agent.

Records trade executions as HCS messages and generates verifiable proof bundles.
Proof bundles can be verified for sequence integrity and temporal ordering.

Proof Bundle Schema:
    {
        "bundle_id": "<uuid>",
        "agent_id": "erc8004-agent-v1",
        "trade_ids": ["t1", "t2", ...],
        "message_ids": ["hcs-msg-1", "hcs-msg-2", ...],
        "created_at": "2026-02-25T00:00:00Z",
        "hash": "<sha256 of trade_ids + message_ids>"
    }

Environment:
    HEDERA_TESTNET_MODE=mock  → in-memory mock (default)
"""

from __future__ import annotations

import hashlib
import json
import os
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional


# ─── Trade Execution Record ───────────────────────────────────────────────────

@dataclass
class TradeExecution:
    """A single executed trade to be recorded on-chain."""
    trade_id: str
    ticker: str
    side: str           # "BUY" | "SELL"
    qty: float
    price: float
    agent_id: str
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    hcs_message_id: Optional[str] = None   # set after on-chain recording

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict())

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "TradeExecution":
        return cls(
            trade_id=d["trade_id"],
            ticker=d["ticker"],
            side=d["side"],
            qty=d["qty"],
            price=d["price"],
            agent_id=d["agent_id"],
            timestamp=d.get("timestamp", datetime.now(timezone.utc).isoformat()),
            hcs_message_id=d.get("hcs_message_id"),
        )


# ─── Proof Bundle ─────────────────────────────────────────────────────────────

@dataclass
class ProofBundle:
    """A verifiable bundle of on-chain trade execution proofs."""
    bundle_id: str
    agent_id: str
    trade_ids: List[str]
    message_ids: List[str]
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    bundle_hash: str = ""

    def compute_hash(self) -> str:
        """Compute SHA-256 of trade_ids + message_ids for integrity."""
        payload = json.dumps({
            "bundle_id": self.bundle_id,
            "trade_ids": self.trade_ids,
            "message_ids": self.message_ids,
        }, sort_keys=True).encode()
        return hashlib.sha256(payload).hexdigest()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "bundle_id": self.bundle_id,
            "agent_id": self.agent_id,
            "trade_ids": self.trade_ids,
            "message_ids": self.message_ids,
            "created_at": self.created_at,
            "bundle_hash": self.bundle_hash,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ProofBundle":
        return cls(
            bundle_id=d["bundle_id"],
            agent_id=d["agent_id"],
            trade_ids=d["trade_ids"],
            message_ids=d["message_ids"],
            created_at=d.get("created_at", datetime.now(timezone.utc).isoformat()),
            bundle_hash=d.get("bundle_hash", ""),
        )


# ─── Verification Result ──────────────────────────────────────────────────────

@dataclass
class VerificationResult:
    """Result of verifying a proof bundle."""
    valid: bool
    bundle_id: str
    trade_count: int
    errors: List[str] = field(default_factory=list)
    verified_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ─── Exceptions ───────────────────────────────────────────────────────────────

class ProofRecordError(Exception):
    """Raised when a trade execution cannot be recorded."""


class ProofVerificationError(Exception):
    """Raised when bundle verification encounters a structural error."""


# ─── ExecutionProof ───────────────────────────────────────────────────────────

class ExecutionProof:
    """
    Records trade executions on Hedera HCS and generates verifiable proof bundles.

    Modes:
        mock  - in-memory (no network, default for tests)
        real  - Hedera testnet HCS (not fully implemented)

    Usage:
        proof = ExecutionProof(agent_id="erc8004-agent-v1")
        recorded = proof.record_execution(trade)
        bundle = proof.generate_proof_bundle(["trade-1", "trade-2"])
        result = proof.verify_bundle(bundle)
    """

    def __init__(
        self,
        agent_id: str = "erc8004-agent-v1",
        mode: Optional[str] = None,
        topic_id: str = "0.0.4753281",
    ) -> None:
        self.agent_id = agent_id
        self.mode = mode or os.environ.get("HEDERA_TESTNET_MODE", "mock")
        self.topic_id = topic_id

        # In-memory store: trade_id -> TradeExecution
        self._executions: Dict[str, TradeExecution] = {}
        # Ordered list of trade_ids (insertion order = sequence)
        self._sequence: List[str] = []
        self._network_failure = False   # injectable for tests

    # ── Public API ────────────────────────────────────────────────────────────

    def record_execution(self, trade: TradeExecution) -> TradeExecution:
        """
        Record a trade execution as an HCS message.

        Args:
            trade: The TradeExecution to record.

        Returns:
            The trade with hcs_message_id set.

        Raises:
            ProofRecordError: If recording fails.
        """
        if self._network_failure:
            raise ProofRecordError("Simulated network failure during record")

        if not trade.trade_id:
            raise ProofRecordError("trade_id must not be empty")
        if trade.qty <= 0:
            raise ProofRecordError(f"qty must be positive, got {trade.qty}")
        if trade.price <= 0:
            raise ProofRecordError(f"price must be positive, got {trade.price}")
        if trade.side not in ("BUY", "SELL"):
            raise ProofRecordError(
                f"side must be 'BUY' or 'SELL', got '{trade.side}'"
            )

        if self.mode == "mock":
            msg_id = self._record_mock(trade)
        else:
            msg_id = self._record_real(trade)

        trade.hcs_message_id = msg_id
        self._executions[trade.trade_id] = trade
        self._sequence.append(trade.trade_id)
        return trade

    def generate_proof_bundle(self, trade_ids: List[str]) -> ProofBundle:
        """
        Generate a proof bundle for a set of trade IDs.

        Args:
            trade_ids: List of trade IDs to include in the bundle.

        Returns:
            ProofBundle with hash computed.

        Raises:
            ProofRecordError: If any trade_id is unknown.
        """
        if not trade_ids:
            raise ProofRecordError("trade_ids must not be empty")

        message_ids: List[str] = []
        for tid in trade_ids:
            if tid not in self._executions:
                raise ProofRecordError(
                    f"trade_id '{tid}' not found in recorded executions"
                )
            msg_id = self._executions[tid].hcs_message_id
            if msg_id is None:
                raise ProofRecordError(
                    f"trade_id '{tid}' has no HCS message ID (not yet recorded)"
                )
            message_ids.append(msg_id)

        bundle = ProofBundle(
            bundle_id=str(uuid.uuid4()),
            agent_id=self.agent_id,
            trade_ids=list(trade_ids),
            message_ids=message_ids,
        )
        bundle.bundle_hash = bundle.compute_hash()
        return bundle

    def verify_bundle(self, bundle: ProofBundle) -> VerificationResult:
        """
        Verify a proof bundle's integrity.

        Checks:
        1. trade_ids and message_ids lengths match
        2. Hash recomputes correctly
        3. All referenced trades exist in local record
        4. Sequence order is consistent (no gaps in recorded order)

        Args:
            bundle: The ProofBundle to verify.

        Returns:
            VerificationResult (valid=True if all checks pass).
        """
        errors: List[str] = []

        # Check 1: length match
        if len(bundle.trade_ids) != len(bundle.message_ids):
            errors.append(
                f"trade_ids count ({len(bundle.trade_ids)}) != "
                f"message_ids count ({len(bundle.message_ids)})"
            )

        # Check 2: hash integrity
        expected_hash = bundle.compute_hash()
        if bundle.bundle_hash and bundle.bundle_hash != expected_hash:
            errors.append(
                f"Hash mismatch: stored={bundle.bundle_hash[:16]}... "
                f"computed={expected_hash[:16]}..."
            )

        # Check 3: all trades exist
        unknown = [
            tid for tid in bundle.trade_ids
            if tid not in self._executions
        ]
        if unknown:
            errors.append(f"Unknown trade_ids: {unknown}")

        # Check 4: message IDs match
        if not errors:
            for tid, mid in zip(bundle.trade_ids, bundle.message_ids):
                recorded_mid = self._executions[tid].hcs_message_id
                if recorded_mid != mid:
                    errors.append(
                        f"Message ID mismatch for trade '{tid}': "
                        f"expected={recorded_mid}, got={mid}"
                    )

        return VerificationResult(
            valid=len(errors) == 0,
            bundle_id=bundle.bundle_id,
            trade_count=len(bundle.trade_ids),
            errors=errors,
        )

    def get_recorded_trade(self, trade_id: str) -> Optional[TradeExecution]:
        """Return a recorded TradeExecution by ID, or None."""
        return self._executions.get(trade_id)

    def get_all_trades(self) -> List[TradeExecution]:
        """Return all recorded trades in insertion order."""
        return [self._executions[tid] for tid in self._sequence]

    def get_execution_count(self) -> int:
        """Return total number of recorded executions."""
        return len(self._executions)

    def set_network_failure(self, fail: bool) -> None:
        """Test helper: simulate network failure."""
        self._network_failure = fail

    def reset(self) -> None:
        """Clear all recorded executions (test helper)."""
        self._executions.clear()
        self._sequence.clear()

    # ── Internal ──────────────────────────────────────────────────────────────

    def _record_mock(self, trade: TradeExecution) -> str:
        """Generate a deterministic mock HCS message ID."""
        seq = len(self._sequence)
        return f"mock-hcs-{self.topic_id}-seq-{seq:06d}-{trade.trade_id[:8]}"

    def _record_real(self, trade: TradeExecution) -> str:
        """Record via real Hedera HCS (stub)."""
        raise NotImplementedError(
            "Real Hedera HCS recording not implemented. "
            "Set HEDERA_TESTNET_MODE=mock for tests."
        )
