"""
reputation.py — On-chain reputation logging for the ERC-8004 Trading Agent.

After each trade, the agent logs its performance to the ERC-8004
ReputationRegistry contract on Base Sepolia. This creates an immutable,
publicly verifiable track record of the agent's trading accuracy.

ERC-8004 ReputationRegistry deployed addresses:
  - Base Sepolia: 0x8004B663056A597Dffe9eCcC1965A193B7388713
  - Ethereum Sepolia: 0x8004B663056A597Dffe9eCcC1965A193B7388713

Contract interface (from ERC-8004 spec):
  giveFeedback(agentId, score, decimals, tag1, tag2, endpointUri, fileHash)
  getAggregateScore(agentId) → (score, count)
  feedbackCount(agentId) → uint256
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

from loguru import logger

from trader import TradeResult


# ─── ERC-8004 Deployed Addresses ─────────────────────────────────────────────

REPUTATION_REGISTRY_BASE_SEPOLIA = "0x8004B663056A597Dffe9eCcC1965A193B7388713"
IDENTITY_REGISTRY_BASE_SEPOLIA = "0x8004A818BFB912233c491871b3d84c89A494BD9e"

# Tags used in reputation feedback
TAG_ACCURACY = "accuracy"
TAG_TRADING = "trading"


# ─── Data Classes ─────────────────────────────────────────────────────────────

@dataclass
class ReputationEntry:
    """A single reputation log entry."""
    agent_id: int
    trade_id: str
    market_id: str
    outcome: Optional[str]      # "WIN", "LOSS", "PENDING"
    score: int                  # 0-1000 (x100 for 2 decimal places)
    tag1: str
    tag2: str
    tx_hash: Optional[str]
    timestamp: float = field(default_factory=time.time)
    on_chain: bool = False      # True if actually submitted to chain
    demo_mode: bool = True

    def to_dict(self) -> dict:
        return {
            "agent_id": self.agent_id,
            "trade_id": self.trade_id,
            "market_id": self.market_id,
            "outcome": self.outcome,
            "score": self.score,
            "score_normalized": self.score / 100.0,
            "tag1": self.tag1,
            "tag2": self.tag2,
            "tx_hash": self.tx_hash,
            "timestamp": self.timestamp,
            "on_chain": self.on_chain,
            "demo_mode": self.demo_mode,
        }


@dataclass
class ReputationStats:
    """Aggregated reputation stats for the agent."""
    agent_id: int
    total_feedback: int
    aggregate_score: float      # 0.0 - 10.0
    on_chain_count: int
    demo_count: int
    win_count: int
    loss_count: int
    pending_count: int

    @property
    def win_rate(self) -> float:
        settled = self.win_count + self.loss_count
        return self.win_count / settled if settled > 0 else 0.0

    def to_dict(self) -> dict:
        return {
            "agent_id": self.agent_id,
            "total_feedback": self.total_feedback,
            "aggregate_score": round(self.aggregate_score, 2),
            "on_chain_count": self.on_chain_count,
            "demo_count": self.demo_count,
            "win_count": self.win_count,
            "loss_count": self.loss_count,
            "pending_count": self.pending_count,
            "win_rate": round(self.win_rate, 4),
        }


# ─── Score Calculation ────────────────────────────────────────────────────────

def calculate_trade_score(result: TradeResult) -> int:
    """
    Calculate reputation score for a trade outcome.

    Score range: 0-1000 (2-decimal fixed-point per ERC-8004 spec).
    - WIN: 850 (= 8.50 / 10)
    - LOSS: 150 (= 1.50 / 10)
    - PENDING: 500 (= 5.00 / 10, neutral)

    Score reflects both accuracy and execution quality.
    In future: adjust based on confidence, size, timing.
    """
    if result.outcome == "WIN":
        # Bonus for high-confidence wins
        base_score = 850
        if result.pnl_usdc > 0:
            # Slightly higher for profitable trades
            base_score = min(950, base_score + int(result.pnl_usdc * 10))
        return base_score

    elif result.outcome == "LOSS":
        # Penalize losses but give partial credit for small losses
        base_score = 150
        if abs(result.pnl_usdc) < result.size_usdc * 0.1:
            # Small loss (< 10%) gets less penalty
            base_score = 300
        return base_score

    else:  # PENDING
        return 500


def build_file_hash(result: TradeResult) -> bytes:
    """
    Build a bytes32 file hash for the reputation record.
    Hash of canonical trade JSON — links reputation to the actual trade data.
    """
    canonical = json.dumps({
        "market_id": result.market_id,
        "side": result.side.value,
        "size_usdc": result.size_usdc,
        "executed_at": result.executed_at,
        "outcome": result.outcome,
        "pnl_usdc": result.pnl_usdc,
    }, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).digest()


# ─── Reputation Logger ────────────────────────────────────────────────────────

class ReputationLogger:
    """
    Logs trade outcomes to the ERC-8004 ReputationRegistry on-chain.

    In dry_run mode: records locally without submitting to chain.
    In live mode: submits giveFeedback() transactions to Base Sepolia.

    The ERC-8004 spec uses ReputationRegistry.giveFeedback() which allows
    third parties to rate agents. The agent rates itself after each trade.
    """

    DECIMALS = 2  # Score uses 2 decimal places (score 850 = 8.50)

    def __init__(
        self,
        registry=None,          # ERC8004Registry instance
        agent_id: int = 0,
        dry_run: bool = True,
    ) -> None:
        self.registry = registry
        self.agent_id = agent_id
        self.dry_run = dry_run
        self._log: list[ReputationEntry] = []

    def set_agent_id(self, agent_id: int) -> None:
        """Set the on-chain agent ID after registration."""
        self.agent_id = agent_id
        logger.info(f"ReputationLogger: agent_id set to {agent_id}")

    async def log_trade(self, result: TradeResult) -> Optional[ReputationEntry]:
        """
        Log a trade outcome to the ERC-8004 ReputationRegistry.

        This is the main entry point — called after each trade execution.

        Args:
            result: The completed trade result

        Returns:
            ReputationEntry with tx_hash if on-chain, or local record if dry_run.
        """
        trade_id = f"{result.market_id}-{result.executed_at}"
        score = calculate_trade_score(result)
        file_hash = build_file_hash(result)

        # Determine tags based on market category / type
        tag1 = TAG_ACCURACY
        tag2 = TAG_TRADING

        logger.info(
            f"ReputationLogger: logging trade {trade_id} "
            f"outcome={result.outcome} score={score}/1000"
        )

        if self.dry_run or not self.registry:
            # Dry run: record locally
            entry = ReputationEntry(
                agent_id=self.agent_id,
                trade_id=trade_id,
                market_id=result.market_id,
                outcome=result.outcome,
                score=score,
                tag1=tag1,
                tag2=tag2,
                tx_hash=None,
                on_chain=False,
                demo_mode=True,
            )
            self._log.append(entry)
            logger.info(
                f"[DRY RUN] Reputation recorded locally: "
                f"agent={self.agent_id} score={score}/1000"
            )
            return entry

        # Live: submit to ReputationRegistry
        try:
            tx_hash = self.registry.give_feedback(
                agent_id=self.agent_id,
                score=score,
                decimals=self.DECIMALS,
                tag1=tag1,
                tag2=tag2,
                endpoint_uri=result.data_uri if result.data_uri.startswith("ipfs://") else "",
                file_hash=file_hash,
            )
            entry = ReputationEntry(
                agent_id=self.agent_id,
                trade_id=trade_id,
                market_id=result.market_id,
                outcome=result.outcome,
                score=score,
                tag1=tag1,
                tag2=tag2,
                tx_hash=tx_hash,
                on_chain=True,
                demo_mode=False,
            )
            self._log.append(entry)
            logger.success(
                f"Reputation logged on-chain: "
                f"tx={tx_hash} agent={self.agent_id} score={score}/1000"
            )
            return entry

        except Exception as e:
            logger.error(f"ReputationLogger: on-chain submission failed: {e}")
            # Fall back to local recording
            entry = ReputationEntry(
                agent_id=self.agent_id,
                trade_id=trade_id,
                market_id=result.market_id,
                outcome=result.outcome,
                score=score,
                tag1=tag1,
                tag2=tag2,
                tx_hash=None,
                on_chain=False,
                demo_mode=False,
            )
            self._log.append(entry)
            return entry

    def get_stats(self) -> ReputationStats:
        """Aggregate all logged trades into reputation statistics."""
        on_chain = sum(1 for e in self._log if e.on_chain)
        demo = sum(1 for e in self._log if e.demo_mode)
        wins = sum(1 for e in self._log if e.outcome == "WIN")
        losses = sum(1 for e in self._log if e.outcome == "LOSS")
        pending = sum(1 for e in self._log if e.outcome == "PENDING" or e.outcome is None)

        total_score = sum(e.score for e in self._log)
        avg_score = (total_score / len(self._log) / 100.0) if self._log else 0.0

        return ReputationStats(
            agent_id=self.agent_id,
            total_feedback=len(self._log),
            aggregate_score=avg_score,
            on_chain_count=on_chain,
            demo_count=demo,
            win_count=wins,
            loss_count=losses,
            pending_count=pending,
        )

    def get_on_chain_score(self) -> Optional[tuple[int, int]]:
        """
        Query the current on-chain aggregate score from ReputationRegistry.
        Returns (score, count) or None if registry unavailable.
        """
        if not self.registry or self.agent_id == 0:
            return None
        try:
            return self.registry.get_aggregate_score(self.agent_id)
        except Exception as e:
            logger.error(f"ReputationLogger: get_aggregate_score failed: {e}")
            return None

    def get_log(self) -> list[dict]:
        """Return all reputation entries as dicts."""
        return [e.to_dict() for e in self._log]
