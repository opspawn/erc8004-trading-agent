"""
hedera_signals.py — Hedera HCS-10 Signal Bus for ERC-8004 Trading Agent.

Publishes and subscribes to trading signals via Hedera Consensus Service (HCS).
Supports both real testnet mode and mock/dry-run mode for testing.

HCS-10 Format:
    {
        "agent_id": "erc8004-agent-v1",
        "signal_type": "BUY" | "SELL" | "HOLD",
        "ticker": "BTC",
        "confidence": 0.75,
        "timestamp": "2026-02-25T00:00:00Z",
        "sequence_number": 1
    }

Environment:
    HEDERA_TESTNET_MODE=mock  → uses in-memory mock (default for tests)
    HEDERA_TESTNET_MODE=real  → uses Hedera REST API
"""

from __future__ import annotations

import json
import os
import time
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional
from collections import defaultdict

from loguru import logger


# ─── Signal Model ─────────────────────────────────────────────────────────────

VALID_SIGNAL_TYPES = {"BUY", "SELL", "HOLD"}


@dataclass
class HCSSignal:
    """A single HCS-10 trading signal message."""
    agent_id: str
    signal_type: str        # "BUY" | "SELL" | "HOLD"
    ticker: str
    confidence: float       # 0.0 – 1.0
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    message_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    sequence_number: int = 0
    topic_id: str = ""

    def validate(self) -> None:
        if self.signal_type not in VALID_SIGNAL_TYPES:
            raise ValueError(
                f"Invalid signal_type '{self.signal_type}'. "
                f"Must be one of {VALID_SIGNAL_TYPES}"
            )
        if not (0.0 <= self.confidence <= 1.0):
            raise ValueError(
                f"confidence must be between 0.0 and 1.0, got {self.confidence}"
            )
        if not self.ticker:
            raise ValueError("ticker must not be empty")
        if not self.agent_id:
            raise ValueError("agent_id must not be empty")

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict())

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "HCSSignal":
        return cls(
            agent_id=d["agent_id"],
            signal_type=d["signal_type"],
            ticker=d["ticker"],
            confidence=d["confidence"],
            timestamp=d.get("timestamp", datetime.now(timezone.utc).isoformat()),
            message_id=d.get("message_id", str(uuid.uuid4())),
            sequence_number=d.get("sequence_number", 0),
            topic_id=d.get("topic_id", ""),
        )

    @classmethod
    def from_json(cls, raw: str) -> "HCSSignal":
        return cls.from_dict(json.loads(raw))


# ─── Mock Storage ─────────────────────────────────────────────────────────────

class _MockTopicStore:
    """In-memory store simulating HCS topics for tests."""

    def __init__(self) -> None:
        # topic_id -> list of (sequence_number, signal)
        self._messages: Dict[str, List[HCSSignal]] = defaultdict(list)
        self._seq_counters: Dict[str, int] = defaultdict(int)

    def publish(self, topic_id: str, signal: HCSSignal) -> HCSSignal:
        seq = self._seq_counters[topic_id]
        self._seq_counters[topic_id] += 1
        signal.sequence_number = seq
        signal.topic_id = topic_id
        self._messages[topic_id].append(signal)
        return signal

    def get_messages(self, topic_id: str, limit: int = 100) -> List[HCSSignal]:
        msgs = self._messages[topic_id]
        return msgs[-limit:]

    def clear(self, topic_id: Optional[str] = None) -> None:
        if topic_id:
            self._messages[topic_id] = []
            self._seq_counters[topic_id] = 0
        else:
            self._messages.clear()
            self._seq_counters.clear()

    def topic_count(self, topic_id: str) -> int:
        return len(self._messages[topic_id])


# Module-level shared mock store (all HederaSignalBus instances in mock mode share it)
_MOCK_STORE = _MockTopicStore()


# ─── HederaSignalBus ──────────────────────────────────────────────────────────

class HCSNetworkError(Exception):
    """Raised when HCS network operations fail."""


class HCSValidationError(Exception):
    """Raised when a signal fails validation."""


class HederaSignalBus:
    """
    HCS-10 Signal Bus for publishing/subscribing to trading signals.

    Modes:
        mock  - in-memory, no network calls (default, for tests)
        real  - Hedera testnet REST API

    Usage:
        bus = HederaSignalBus(mode="mock")
        signal = bus.publish_signal(signal)
        history = bus.get_signal_history("0.0.12345")
        bus.subscribe_signals("0.0.12345", callback=my_fn)
    """

    DEFAULT_TOPIC_ID = "0.0.4753280"   # Hedera testnet demo topic

    def __init__(
        self,
        agent_id: str = "erc8004-agent-v1",
        mode: Optional[str] = None,
        topic_id: Optional[str] = None,
    ) -> None:
        self.agent_id = agent_id
        self.mode = mode or os.environ.get("HEDERA_TESTNET_MODE", "mock")
        self.topic_id = topic_id or self.DEFAULT_TOPIC_ID
        self._subscribers: List[Callable[[HCSSignal], None]] = []
        self._published_count = 0
        self._network_failure = False   # injectable for testing

        logger.debug(
            "HederaSignalBus init: agent_id={}, mode={}, topic={}",
            self.agent_id, self.mode, self.topic_id,
        )

    # ── Public API ────────────────────────────────────────────────────────────

    def publish_signal(self, signal: HCSSignal) -> HCSSignal:
        """
        Publish a trading signal to HCS.

        Args:
            signal: The HCSSignal to publish.

        Returns:
            The published signal (with sequence_number and topic_id set).

        Raises:
            HCSValidationError: If signal is malformed.
            HCSNetworkError: If the network call fails (real mode).
        """
        # Validate
        try:
            signal.validate()
        except ValueError as exc:
            raise HCSValidationError(str(exc)) from exc

        if self._network_failure:
            raise HCSNetworkError("Simulated network failure")

        if self.mode == "mock":
            result = _MOCK_STORE.publish(self.topic_id, signal)
        else:
            result = self._publish_real(signal)

        self._published_count += 1

        # Notify in-process subscribers
        for cb in self._subscribers:
            try:
                cb(result)
            except Exception as exc:
                logger.warning("Subscriber callback raised: {}", exc)

        logger.debug(
            "Published signal: type={} ticker={} seq={}",
            result.signal_type, result.ticker, result.sequence_number,
        )
        return result

    def subscribe_signals(
        self,
        topic_id: str,
        callback: Callable[[HCSSignal], None],
    ) -> None:
        """
        Register a callback for new signals on `topic_id`.

        In mock mode, callbacks are called synchronously on publish.
        In real mode, a polling thread would be used (not implemented here).

        Args:
            topic_id: The HCS topic ID to subscribe to.
            callback: Function called with each new HCSSignal.
        """
        self._subscribers.append(callback)
        logger.debug("Subscribed callback to topic={}", topic_id)

    def unsubscribe_all(self) -> None:
        """Remove all registered callbacks."""
        self._subscribers.clear()

    def get_signal_history(
        self,
        topic_id: Optional[str] = None,
        limit: int = 50,
    ) -> List[HCSSignal]:
        """
        Retrieve recent signals from an HCS topic.

        Args:
            topic_id: Topic to query. Defaults to self.topic_id.
            limit: Max number of messages to return.

        Returns:
            List of HCSSignal objects, oldest first.

        Raises:
            HCSNetworkError: On network failure.
        """
        if self._network_failure:
            raise HCSNetworkError("Simulated network failure")

        tid = topic_id or self.topic_id

        if self.mode == "mock":
            return _MOCK_STORE.get_messages(tid, limit=limit)
        else:
            return self._fetch_real(tid, limit=limit)

    def get_published_count(self) -> int:
        """Return total number of signals published by this bus instance."""
        return self._published_count

    def set_network_failure(self, fail: bool) -> None:
        """Test helper: simulate network failure."""
        self._network_failure = fail

    def reset_mock_store(self) -> None:
        """Test helper: clear all mock topic data."""
        _MOCK_STORE.clear()

    # ── Signal Factory ────────────────────────────────────────────────────────

    def make_signal(
        self,
        signal_type: str,
        ticker: str,
        confidence: float,
    ) -> HCSSignal:
        """Create an HCSSignal attributed to this bus's agent_id."""
        return HCSSignal(
            agent_id=self.agent_id,
            signal_type=signal_type,
            ticker=ticker,
            confidence=confidence,
        )

    # ── Internal (real mode) ──────────────────────────────────────────────────

    def _publish_real(self, signal: HCSSignal) -> HCSSignal:
        """Publish via Hedera REST API (stub for real mode)."""
        # Real implementation would use Hedera SDK or REST API.
        # For now, raises to prevent accidental testnet calls in tests.
        raise NotImplementedError(
            "Real Hedera testnet publishing not implemented. "
            "Set HEDERA_TESTNET_MODE=mock for tests."
        )

    def _fetch_real(self, topic_id: str, limit: int) -> List[HCSSignal]:
        """Fetch from Hedera mirror node REST API (stub for real mode)."""
        raise NotImplementedError(
            "Real Hedera mirror node fetch not implemented. "
            "Set HEDERA_TESTNET_MODE=mock for tests."
        )
