"""
sentiment_signal.py — Sentiment signal layer for the ERC-8004 Trading Agent.

Aggregates sentiment signals from multiple sources and applies a modifier to
the Kelly-computed position size:
  - Positive sentiment (score > 0.3)  → +10% size boost
  - Negative sentiment (score < -0.3) → -20% size reduction
  - Extreme negative (score < -0.7)   → trade blocked entirely

SentimentSignal dataclass:
    ticker     — asset symbol (e.g. "BTC", "ETH")
    score      — float in [-1.0, +1.0]
    source     — e.g. "twitter", "reddit", "news", "onchain"
    timestamp  — when the signal was collected
    confidence — 0.0–1.0 (how reliable this source is)

SentimentAggregator:
    Combines signals from multiple sources via confidence-weighted average.
    Signals older than max_age_seconds are discarded as stale.
"""

from __future__ import annotations

import time
import hashlib
from dataclasses import dataclass, field
from typing import Optional

from loguru import logger


# ─── Constants ────────────────────────────────────────────────────────────────

# Sentiment thresholds
EXTREME_NEGATIVE_THRESHOLD = -0.70   # block trade
NEGATIVE_THRESHOLD         = -0.30   # reduce size
POSITIVE_THRESHOLD         =  0.30   # boost size

# Size modifiers
POSITIVE_SIZE_BOOST        =  0.10   # +10%
NEGATIVE_SIZE_REDUCTION    = -0.20   # -20%

# Default signal staleness limit (seconds)
DEFAULT_MAX_AGE_SECONDS = 3600.0     # 1 hour

# Supported tickers for mock data
SUPPORTED_TICKERS = {"BTC", "ETH", "SOL", "LINK", "AAVE", "UNI", "MATIC", "ARB"}


# ─── Data Classes ─────────────────────────────────────────────────────────────

@dataclass
class SentimentSignal:
    """A single sentiment data point from one source."""
    ticker: str
    score: float                   # -1.0 (very negative) to +1.0 (very positive)
    source: str                    # "twitter", "reddit", "news", "onchain", "mock"
    timestamp: float = field(default_factory=time.time)
    confidence: float = 0.5        # 0.0–1.0

    def __post_init__(self) -> None:
        if not -1.0 <= self.score <= 1.0:
            raise ValueError(
                f"SentimentSignal score must be in [-1.0, 1.0], got {self.score}"
            )
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(
                f"SentimentSignal confidence must be in [0.0, 1.0], got {self.confidence}"
            )

    @property
    def age_seconds(self) -> float:
        return time.time() - self.timestamp

    def is_stale(self, max_age_seconds: float = DEFAULT_MAX_AGE_SECONDS) -> bool:
        return self.age_seconds > max_age_seconds

    @property
    def label(self) -> str:
        """Human-readable sentiment label."""
        if self.score >= 0.6:
            return "very_positive"
        elif self.score >= POSITIVE_THRESHOLD:
            return "positive"
        elif self.score > NEGATIVE_THRESHOLD:
            return "neutral"
        elif self.score > EXTREME_NEGATIVE_THRESHOLD:
            return "negative"
        else:
            return "extreme_negative"


@dataclass
class AggregatedSentiment:
    """Aggregated sentiment from multiple sources for one ticker."""
    ticker: str
    score: float                   # confidence-weighted average of all signals
    signal_count: int
    sources: list[str]
    timestamp: float = field(default_factory=time.time)

    @property
    def is_extreme_negative(self) -> bool:
        return self.score < EXTREME_NEGATIVE_THRESHOLD

    @property
    def label(self) -> str:
        if self.score >= 0.6:
            return "very_positive"
        elif self.score >= POSITIVE_THRESHOLD:
            return "positive"
        elif self.score > NEGATIVE_THRESHOLD:
            return "neutral"
        elif self.score > EXTREME_NEGATIVE_THRESHOLD:
            return "negative"
        else:
            return "extreme_negative"


# ─── Aggregator ───────────────────────────────────────────────────────────────

class SentimentAggregator:
    """
    Aggregates sentiment signals from multiple sources.

    Signals are stored per-ticker. When computing the aggregate, stale
    signals (older than max_age_seconds) are excluded. The aggregate
    score is a confidence-weighted average.

    Usage:
        agg = SentimentAggregator()
        agg.add(SentimentSignal("BTC", score=0.4, source="twitter", confidence=0.7))
        agg.add(SentimentSignal("BTC", score=0.1, source="reddit", confidence=0.5))
        agg_sent = agg.get("BTC")
        size = agg.apply_modifier(size=10.0, ticker="BTC")
    """

    def __init__(self, max_age_seconds: float = DEFAULT_MAX_AGE_SECONDS) -> None:
        self.max_age_seconds = max_age_seconds
        self._signals: dict[str, list[SentimentSignal]] = {}
        logger.debug(f"SentimentAggregator initialized (max_age={max_age_seconds}s)")

    def add(self, signal: SentimentSignal) -> None:
        """Add a sentiment signal to the aggregator."""
        ticker = signal.ticker.upper()
        if ticker not in self._signals:
            self._signals[ticker] = []
        self._signals[ticker].append(signal)
        logger.debug(
            f"SentimentAggregator: added {ticker} score={signal.score:.3f} "
            f"source={signal.source!r} conf={signal.confidence:.2f}"
        )

    def add_batch(self, signals: list[SentimentSignal]) -> None:
        """Add multiple signals at once."""
        for s in signals:
            self.add(s)

    def _fresh_signals(self, ticker: str) -> list[SentimentSignal]:
        """Return non-stale signals for a ticker."""
        ticker = ticker.upper()
        if ticker not in self._signals:
            return []
        return [
            s for s in self._signals[ticker]
            if not s.is_stale(self.max_age_seconds)
        ]

    def get(self, ticker: str) -> Optional[AggregatedSentiment]:
        """
        Compute aggregate sentiment for a ticker.

        Returns None if no fresh signals are available.
        """
        signals = self._fresh_signals(ticker)
        if not signals:
            return None

        total_weight = sum(s.confidence for s in signals)
        if total_weight == 0:
            return None

        weighted_score = sum(s.score * s.confidence for s in signals) / total_weight
        sources = list({s.source for s in signals})

        return AggregatedSentiment(
            ticker=ticker.upper(),
            score=weighted_score,
            signal_count=len(signals),
            sources=sources,
        )

    def get_score(self, ticker: str, default: float = 0.0) -> float:
        """Return aggregate sentiment score, or default if none available."""
        agg = self.get(ticker)
        return agg.score if agg is not None else default

    def apply_modifier(
        self,
        size: float,
        ticker: str,
        default_score: float = 0.0,
    ) -> tuple[float, str]:
        """
        Apply sentiment modifier to a proposed trade size.

        Returns:
            (modified_size, reason)
            modified_size = 0.0 if trade should be blocked.
        """
        agg = self.get(ticker)
        score = agg.score if agg is not None else default_score

        if score < EXTREME_NEGATIVE_THRESHOLD:
            logger.warning(
                f"SentimentAggregator: BLOCKING trade on {ticker} — "
                f"extreme negative sentiment score={score:.3f}"
            )
            return 0.0, f"Blocked: extreme negative sentiment ({score:.3f})"

        if score < NEGATIVE_THRESHOLD:
            modified = size * (1.0 + NEGATIVE_SIZE_REDUCTION)  # reduce by 20%
            reason = f"Negative sentiment ({score:.3f}): size reduced by 20%"
            logger.debug(f"SentimentAggregator: {ticker} {reason}")
            return max(0.0, modified), reason

        if score > POSITIVE_THRESHOLD:
            modified = size * (1.0 + POSITIVE_SIZE_BOOST)  # boost by 10%
            reason = f"Positive sentiment ({score:.3f}): size boosted by 10%"
            logger.debug(f"SentimentAggregator: {ticker} {reason}")
            return modified, reason

        # Neutral range — no modification
        return size, f"Neutral sentiment ({score:.3f}): no modification"

    def clear_stale(self) -> int:
        """Remove stale signals. Returns count removed."""
        removed = 0
        for ticker in list(self._signals.keys()):
            before = len(self._signals[ticker])
            self._signals[ticker] = [
                s for s in self._signals[ticker]
                if not s.is_stale(self.max_age_seconds)
            ]
            removed += before - len(self._signals[ticker])
        return removed

    def clear(self, ticker: Optional[str] = None) -> None:
        """Clear all signals, or signals for a specific ticker."""
        if ticker is not None:
            self._signals.pop(ticker.upper(), None)
        else:
            self._signals.clear()

    def get_summary(self) -> dict:
        """Return summary of all tickers with fresh signals."""
        result: dict[str, dict] = {}
        for ticker in list(self._signals.keys()):
            agg = self.get(ticker)
            if agg is not None:
                result[ticker] = {
                    "score": round(agg.score, 4),
                    "label": agg.label,
                    "signal_count": agg.signal_count,
                    "sources": agg.sources,
                }
        return result


# ─── Mock Data Provider ───────────────────────────────────────────────────────

# Deterministic sentiment scores based on ticker hash
# Ensures tests produce reproducible results
_MOCK_BASE_SCORES: dict[str, float] = {
    "BTC":   0.45,   # mildly positive
    "ETH":   0.38,   # mildly positive
    "SOL":   0.20,   # slightly positive
    "LINK":  0.15,   # slightly positive
    "AAVE":  0.10,   # neutral
    "UNI":   0.05,   # neutral
    "MATIC": -0.10,  # slightly negative
    "ARB":   -0.05,  # neutral-negative
}

_MOCK_SOURCES = ["twitter", "reddit", "news", "onchain"]
_MOCK_CONFIDENCES: dict[str, float] = {
    "twitter": 0.50,
    "reddit":  0.55,
    "news":    0.75,
    "onchain": 0.90,
}


def generate_mock_signals(
    ticker: str,
    num_signals: int = 4,
    base_score: Optional[float] = None,
    noise_scale: float = 0.05,
) -> list[SentimentSignal]:
    """
    Generate realistic mock sentiment signals for testing.

    The signals are deterministic: given the same ticker and num_signals,
    the same sequence is always produced.

    Args:
        ticker:       Asset ticker ("BTC", "ETH", etc.)
        num_signals:  Number of signals to generate
        base_score:   Override the base score (default: from _MOCK_BASE_SCORES)
        noise_scale:  Standard deviation of per-signal noise

    Returns:
        List of SentimentSignal objects
    """
    ticker = ticker.upper()
    base = base_score
    if base is None:
        # Deterministic base from known map or hash fallback
        if ticker in _MOCK_BASE_SCORES:
            base = _MOCK_BASE_SCORES[ticker]
        else:
            # Hash-based score for unknown tickers
            digest = int(hashlib.md5(ticker.encode()).hexdigest()[:8], 16)
            base = ((digest % 100) - 50) / 100.0   # -0.50 to +0.49

    signals: list[SentimentSignal] = []
    sources = _MOCK_SOURCES[:min(num_signals, len(_MOCK_SOURCES))]

    for i, source in enumerate(sources):
        # Deterministic noise from index
        noise_val = ((i * 7 + 3) % 11 - 5) / 100.0 * noise_scale * 20
        raw_score = base + noise_val
        score = max(-1.0, min(1.0, raw_score))
        confidence = _MOCK_CONFIDENCES.get(source, 0.5)

        signals.append(
            SentimentSignal(
                ticker=ticker,
                score=score,
                source=source,
                confidence=confidence,
            )
        )

    return signals


def create_mock_aggregator(tickers: Optional[list[str]] = None) -> SentimentAggregator:
    """
    Create a SentimentAggregator pre-populated with mock data for testing.

    Args:
        tickers: List of tickers to populate (default: all supported tickers)

    Returns:
        SentimentAggregator with mock signals loaded
    """
    tickers = tickers or list(SUPPORTED_TICKERS)
    agg = SentimentAggregator()
    for ticker in tickers:
        signals = generate_mock_signals(ticker)
        agg.add_batch(signals)
    return agg
