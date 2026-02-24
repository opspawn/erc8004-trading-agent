"""
validator.py — Creates and manages validation artifacts for trades.

Generates IPFS-compatible validation packages and interfaces with
the ValidationRegistry contract to submit trade outcomes for verification.
"""

import hashlib
import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional

from loguru import logger

from trader import TradeResult


@dataclass
class ValidationArtifact:
    """Package submitted to ValidationRegistry for a trade."""
    trade_id: str
    market_id: str
    side: str
    size_usdc: float
    executed_at: str
    outcome: Optional[str]
    confidence_score: int       # 0-100, self-assessed
    agent_version: str
    checksum: str               # SHA-256 of canonical JSON


class TradeValidator:
    """
    Builds validation artifacts for trade outcomes.

    Flow:
    1. Agent executes trade → TradeResult
    2. Validator.create_artifact(result) → ValidationArtifact
    3. Artifact serialized to JSON
    4. Registry.request_validation(agent_id, data_uri, data_hash)
    5. After settlement: Registry.submit_validation(request_id, score)
    """

    AGENT_VERSION = "0.1.0"

    def __init__(self, registry=None):
        """
        Args:
            registry: ERC8004Registry instance (optional for offline use)
        """
        self.registry = registry
        self.pending_validations: dict[int, TradeResult] = {}  # requestId → TradeResult

    def create_artifact(self, result: TradeResult) -> ValidationArtifact:
        """Build a validation artifact from a trade result."""
        trade_id = f"{result.market_id}-{result.executed_at}"

        # Self-assessed confidence score based on trade logic
        if result.outcome == "WIN":
            confidence = 90
        elif result.outcome == "LOSS":
            confidence = 10
        else:
            confidence = 50  # Pending

        # Canonical JSON for hashing (sorted keys, no whitespace)
        canonical = json.dumps({
            "trade_id": trade_id,
            "market_id": result.market_id,
            "side": result.side.value,
            "size_usdc": result.size_usdc,
            "executed_at": result.executed_at,
            "outcome": result.outcome,
            "agent_version": self.AGENT_VERSION,
        }, sort_keys=True, separators=(",", ":"))

        checksum = hashlib.sha256(canonical.encode()).hexdigest()

        return ValidationArtifact(
            trade_id=trade_id,
            market_id=result.market_id,
            side=result.side.value,
            size_usdc=result.size_usdc,
            executed_at=result.executed_at,
            outcome=result.outcome,
            confidence_score=confidence,
            agent_version=self.AGENT_VERSION,
            checksum=checksum,
        )

    def artifact_to_json(self, artifact: ValidationArtifact) -> str:
        """Serialize artifact to JSON string."""
        return json.dumps({
            "trade_id": artifact.trade_id,
            "market_id": artifact.market_id,
            "side": artifact.side,
            "size_usdc": artifact.size_usdc,
            "executed_at": artifact.executed_at,
            "outcome": artifact.outcome,
            "confidence_score": artifact.confidence_score,
            "agent_version": artifact.agent_version,
            "checksum": artifact.checksum,
        }, indent=2)

    def compute_data_hash(self, artifact: ValidationArtifact) -> bytes:
        """Compute the data hash for the ValidationRegistry contract."""
        json_str = self.artifact_to_json(artifact)
        return hashlib.sha256(json_str.encode()).digest()

    def get_data_uri(self, artifact: ValidationArtifact) -> str:
        """
        Get the data URI for the artifact.
        In production: upload to IPFS and return ipfs://Qm...
        For now: use inline data URI.
        """
        json_str = self.artifact_to_json(artifact)
        # TODO: In production, upload to IPFS using pinata/web3.storage
        return f"data:application/json;charset=utf-8,{json_str}"

    def submit_to_chain(
        self,
        agent_id: int,
        result: TradeResult,
    ) -> Optional[tuple[str, int]]:
        """
        Submit a validation request for a trade.

        Args:
            agent_id:  The agent's on-chain ID
            result:    Trade result to validate

        Returns:
            (tx_hash, request_id) or None if dry run / error
        """
        if not self.registry:
            logger.warning("No registry configured — skipping on-chain submission")
            return None

        try:
            artifact = self.create_artifact(result)
            data_uri = self.get_data_uri(artifact)
            data_hash = self.compute_data_hash(artifact)

            logger.info(
                f"Submitting validation request: "
                f"market={result.market_id}, checksum={artifact.checksum[:16]}..."
            )

            tx_hash, request_id = self.registry.request_validation(
                agent_id, data_uri, data_hash
            )

            self.pending_validations[request_id] = result
            logger.success(
                f"Validation requested: requestId={request_id}, tx={tx_hash}"
            )
            return tx_hash, request_id

        except Exception as e:
            logger.error(f"Failed to submit validation: {e}")
            return None

    def self_validate_trade(
        self,
        request_id: int,
        outcome: str,  # "WIN", "LOSS", "PENDING"
    ) -> Optional[str]:
        """
        Submit the agent's own assessment of a trade outcome.
        Score: WIN=85, LOSS=15, PENDING=50
        """
        if not self.registry:
            return None

        score_map = {"WIN": 85, "LOSS": 15, "PENDING": 50}
        score = score_map.get(outcome, 50)

        return self.registry.submit_validation(
            request_id,
            score,
            comment_uri="",
        )

    def get_pending_count(self) -> int:
        return len(self.pending_validations)
