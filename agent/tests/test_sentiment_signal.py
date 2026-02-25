"""
test_sentiment_signal.py — Tests for the sentiment signal layer.

Coverage:
  - SentimentSignal creation and validation
  - Score boundary validation (must be -1.0 to +1.0)
  - Confidence boundary validation (must be 0.0 to 1.0)
  - Signal staleness detection
  - Signal age_seconds property
  - Label property (very_positive, positive, neutral, negative, extreme_negative)
  - SentimentAggregator add/add_batch
  - SentimentAggregator multi-source confidence-weighted aggregation
  - get() returns None for unknown tickers
  - get_score() with default
  - apply_modifier: positive boost (+10%)
  - apply_modifier: negative reduction (-20%)
  - apply_modifier: extreme negative blocks trade
  - apply_modifier: neutral — no change
  - clear_stale() removes old signals
  - clear() by ticker
  - clear() all
  - get_summary() structure
  - Mock data provider: generate_mock_signals
  - create_mock_aggregator
  - Integration: sentiment modifier with various scores
"""

from __future__ import annotations

import time
import pytest

from sentiment_signal import (
    AggregatedSentiment,
    SentimentAggregator,
    SentimentSignal,
    create_mock_aggregator,
    generate_mock_signals,
    EXTREME_NEGATIVE_THRESHOLD,
    NEGATIVE_THRESHOLD,
    POSITIVE_THRESHOLD,
    POSITIVE_SIZE_BOOST,
    NEGATIVE_SIZE_REDUCTION,
    SUPPORTED_TICKERS,
)


# ─── SentimentSignal ──────────────────────────────────────────────────────────

class TestSentimentSignalCreation:
    def test_basic_creation(self):
        s = SentimentSignal(ticker="BTC", score=0.5, source="twitter")
        assert s.ticker == "BTC"
        assert s.score == 0.5
        assert s.source == "twitter"

    def test_default_confidence(self):
        s = SentimentSignal(ticker="ETH", score=0.0, source="reddit")
        assert s.confidence == 0.5

    def test_custom_confidence(self):
        s = SentimentSignal(ticker="ETH", score=0.0, source="news", confidence=0.8)
        assert s.confidence == 0.8

    def test_default_timestamp_set(self):
        before = time.time()
        s = SentimentSignal(ticker="BTC", score=0.1, source="mock")
        assert s.timestamp >= before

    def test_score_minus_one_valid(self):
        s = SentimentSignal(ticker="BTC", score=-1.0, source="mock")
        assert s.score == -1.0

    def test_score_plus_one_valid(self):
        s = SentimentSignal(ticker="BTC", score=1.0, source="mock")
        assert s.score == 1.0

    def test_score_zero_valid(self):
        s = SentimentSignal(ticker="BTC", score=0.0, source="mock")
        assert s.score == 0.0

    def test_score_above_one_raises(self):
        with pytest.raises(ValueError, match="score"):
            SentimentSignal(ticker="BTC", score=1.1, source="mock")

    def test_score_below_minus_one_raises(self):
        with pytest.raises(ValueError, match="score"):
            SentimentSignal(ticker="BTC", score=-1.1, source="mock")

    def test_confidence_above_one_raises(self):
        with pytest.raises(ValueError, match="confidence"):
            SentimentSignal(ticker="BTC", score=0.0, source="mock", confidence=1.1)

    def test_confidence_below_zero_raises(self):
        with pytest.raises(ValueError, match="confidence"):
            SentimentSignal(ticker="BTC", score=0.0, source="mock", confidence=-0.1)

    def test_confidence_zero_valid(self):
        s = SentimentSignal(ticker="BTC", score=0.0, source="mock", confidence=0.0)
        assert s.confidence == 0.0

    def test_confidence_one_valid(self):
        s = SentimentSignal(ticker="BTC", score=0.0, source="mock", confidence=1.0)
        assert s.confidence == 1.0


class TestSentimentSignalAge:
    def test_fresh_signal_not_stale(self):
        s = SentimentSignal(ticker="BTC", score=0.1, source="mock")
        assert not s.is_stale(max_age_seconds=3600)

    def test_old_signal_is_stale(self):
        s = SentimentSignal(ticker="BTC", score=0.1, source="mock",
                             timestamp=time.time() - 7200)
        assert s.is_stale(max_age_seconds=3600)

    def test_age_seconds_approximately_zero(self):
        s = SentimentSignal(ticker="BTC", score=0.1, source="mock")
        assert s.age_seconds < 1.0


class TestSentimentSignalLabel:
    def test_very_positive(self):
        s = SentimentSignal(ticker="BTC", score=0.8, source="mock")
        assert s.label == "very_positive"

    def test_positive(self):
        s = SentimentSignal(ticker="BTC", score=0.4, source="mock")
        assert s.label == "positive"

    def test_neutral(self):
        s = SentimentSignal(ticker="BTC", score=0.0, source="mock")
        assert s.label == "neutral"

    def test_negative(self):
        s = SentimentSignal(ticker="BTC", score=-0.5, source="mock")
        assert s.label == "negative"

    def test_extreme_negative(self):
        s = SentimentSignal(ticker="BTC", score=-0.8, source="mock")
        assert s.label == "extreme_negative"

    def test_boundary_positive_threshold(self):
        s = SentimentSignal(ticker="BTC", score=POSITIVE_THRESHOLD + 0.01, source="mock")
        assert s.label in ("positive", "very_positive")

    def test_boundary_negative_threshold(self):
        s = SentimentSignal(ticker="BTC", score=NEGATIVE_THRESHOLD - 0.01, source="mock")
        assert s.label == "negative"


# ─── SentimentAggregator ─────────────────────────────────────────────────────

class TestAggregatorBasics:
    def test_empty_aggregator_returns_none(self):
        agg = SentimentAggregator()
        assert agg.get("BTC") is None

    def test_add_signal_makes_get_work(self):
        agg = SentimentAggregator()
        agg.add(SentimentSignal("BTC", 0.5, "mock"))
        result = agg.get("BTC")
        assert result is not None

    def test_get_returns_aggregated_sentiment(self):
        agg = SentimentAggregator()
        agg.add(SentimentSignal("BTC", 0.5, "mock"))
        result = agg.get("BTC")
        assert isinstance(result, AggregatedSentiment)

    def test_ticker_case_insensitive_add(self):
        agg = SentimentAggregator()
        agg.add(SentimentSignal("btc", 0.5, "mock"))
        assert agg.get("BTC") is not None

    def test_get_score_default_on_empty(self):
        agg = SentimentAggregator()
        assert agg.get_score("BTC", default=0.123) == 0.123

    def test_get_score_returns_score(self):
        agg = SentimentAggregator()
        agg.add(SentimentSignal("BTC", 0.5, "mock", confidence=1.0))
        score = agg.get_score("BTC")
        assert abs(score - 0.5) < 0.01


class TestAggregatorWeighting:
    def test_single_signal_score_matches(self):
        agg = SentimentAggregator()
        agg.add(SentimentSignal("ETH", 0.6, "news", confidence=0.8))
        result = agg.get("ETH")
        assert abs(result.score - 0.6) < 0.01

    def test_two_equal_confidence_signals_average(self):
        agg = SentimentAggregator()
        agg.add(SentimentSignal("ETH", 0.4, "twitter", confidence=0.5))
        agg.add(SentimentSignal("ETH", 0.8, "reddit", confidence=0.5))
        result = agg.get("ETH")
        assert abs(result.score - 0.6) < 0.01

    def test_higher_confidence_dominates(self):
        agg = SentimentAggregator()
        agg.add(SentimentSignal("ETH", 0.9, "onchain", confidence=0.9))
        agg.add(SentimentSignal("ETH", 0.1, "twitter", confidence=0.1))
        result = agg.get("ETH")
        # Weighted: (0.9*0.9 + 0.1*0.1) / (0.9+0.1) = (0.81+0.01)/1.0 = 0.82
        assert result.score > 0.7

    def test_signal_count_correct(self):
        agg = SentimentAggregator()
        agg.add(SentimentSignal("SOL", 0.2, "twitter"))
        agg.add(SentimentSignal("SOL", 0.3, "reddit"))
        agg.add(SentimentSignal("SOL", 0.4, "news"))
        result = agg.get("SOL")
        assert result.signal_count == 3

    def test_sources_list_unique(self):
        agg = SentimentAggregator()
        agg.add(SentimentSignal("SOL", 0.2, "twitter"))
        agg.add(SentimentSignal("SOL", 0.3, "twitter"))
        result = agg.get("SOL")
        assert result.sources.count("twitter") == 1

    def test_add_batch(self):
        agg = SentimentAggregator()
        signals = [
            SentimentSignal("LINK", 0.3, "twitter"),
            SentimentSignal("LINK", 0.5, "news"),
        ]
        agg.add_batch(signals)
        assert agg.get("LINK") is not None
        assert agg.get("LINK").signal_count == 2


# ─── Apply Modifier ───────────────────────────────────────────────────────────

class TestApplyModifier:
    def test_neutral_no_change(self):
        agg = SentimentAggregator()
        agg.add(SentimentSignal("BTC", 0.0, "mock", confidence=1.0))
        size, reason = agg.apply_modifier(10.0, "BTC")
        assert size == 10.0

    def test_positive_boost_applied(self):
        agg = SentimentAggregator()
        agg.add(SentimentSignal("BTC", 0.5, "mock", confidence=1.0))
        size, reason = agg.apply_modifier(10.0, "BTC")
        assert abs(size - 11.0) < 0.01

    def test_negative_reduction_applied(self):
        agg = SentimentAggregator()
        agg.add(SentimentSignal("BTC", -0.5, "mock", confidence=1.0))
        size, reason = agg.apply_modifier(10.0, "BTC")
        assert abs(size - 8.0) < 0.01

    def test_extreme_negative_blocks(self):
        agg = SentimentAggregator()
        agg.add(SentimentSignal("BTC", -0.8, "mock", confidence=1.0))
        size, reason = agg.apply_modifier(10.0, "BTC")
        assert size == 0.0

    def test_extreme_negative_reason_contains_blocked(self):
        agg = SentimentAggregator()
        agg.add(SentimentSignal("BTC", -0.9, "mock", confidence=1.0))
        size, reason = agg.apply_modifier(10.0, "BTC")
        assert "block" in reason.lower() or "extreme" in reason.lower()

    def test_positive_reason_contains_boost(self):
        agg = SentimentAggregator()
        agg.add(SentimentSignal("BTC", 0.5, "mock", confidence=1.0))
        _, reason = agg.apply_modifier(10.0, "BTC")
        assert "boost" in reason.lower() or "positive" in reason.lower()

    def test_negative_reason_contains_reduced(self):
        agg = SentimentAggregator()
        agg.add(SentimentSignal("BTC", -0.5, "mock", confidence=1.0))
        _, reason = agg.apply_modifier(10.0, "BTC")
        assert "reduc" in reason.lower() or "negative" in reason.lower()

    def test_unknown_ticker_uses_default_neutral(self):
        agg = SentimentAggregator()
        size, reason = agg.apply_modifier(10.0, "UNKNOWN")
        # default_score=0.0 → neutral → no modification
        assert size == 10.0

    def test_at_extreme_threshold_boundary_not_blocked(self):
        agg = SentimentAggregator()
        # exactly at threshold: condition is `score < threshold`, so -0.70 is NOT blocked
        agg.add(SentimentSignal("BTC", EXTREME_NEGATIVE_THRESHOLD, "mock", confidence=1.0))
        size, _ = agg.apply_modifier(10.0, "BTC")
        # -0.70 equals threshold but is not strictly less than, so not blocked
        assert size > 0.0

    def test_below_extreme_threshold_is_blocked(self):
        agg = SentimentAggregator()
        agg.add(SentimentSignal("BTC", EXTREME_NEGATIVE_THRESHOLD - 0.05, "mock", confidence=1.0))
        size, _ = agg.apply_modifier(10.0, "BTC")
        assert size == 0.0

    def test_just_above_extreme_threshold_not_blocked(self):
        agg = SentimentAggregator()
        agg.add(SentimentSignal("BTC", EXTREME_NEGATIVE_THRESHOLD + 0.05, "mock", confidence=1.0))
        size, _ = agg.apply_modifier(10.0, "BTC")
        # should be in negative range but not blocked
        assert 0.0 < size < 10.0

    def test_positive_modifier_constant(self):
        assert POSITIVE_SIZE_BOOST == 0.10

    def test_negative_modifier_constant(self):
        assert NEGATIVE_SIZE_REDUCTION == -0.20


# ─── Stale Signals ────────────────────────────────────────────────────────────

class TestStaleSignals:
    def test_stale_signal_excluded_from_aggregate(self):
        agg = SentimentAggregator(max_age_seconds=60)
        old_ts = time.time() - 7200
        agg.add(SentimentSignal("BTC", 0.9, "mock", timestamp=old_ts))
        assert agg.get("BTC") is None

    def test_fresh_signal_included(self):
        agg = SentimentAggregator(max_age_seconds=3600)
        agg.add(SentimentSignal("BTC", 0.5, "mock"))
        assert agg.get("BTC") is not None

    def test_clear_stale_removes_old(self):
        agg = SentimentAggregator(max_age_seconds=60)
        old_ts = time.time() - 7200
        agg.add(SentimentSignal("BTC", 0.9, "mock", timestamp=old_ts))
        removed = agg.clear_stale()
        assert removed >= 1

    def test_clear_stale_keeps_fresh(self):
        agg = SentimentAggregator(max_age_seconds=3600)
        agg.add(SentimentSignal("BTC", 0.5, "mock"))
        removed = agg.clear_stale()
        assert removed == 0
        assert agg.get("BTC") is not None


# ─── Clear ────────────────────────────────────────────────────────────────────

class TestClear:
    def test_clear_all(self):
        agg = SentimentAggregator()
        agg.add(SentimentSignal("BTC", 0.5, "mock"))
        agg.add(SentimentSignal("ETH", 0.3, "mock"))
        agg.clear()
        assert agg.get("BTC") is None
        assert agg.get("ETH") is None

    def test_clear_specific_ticker(self):
        agg = SentimentAggregator()
        agg.add(SentimentSignal("BTC", 0.5, "mock"))
        agg.add(SentimentSignal("ETH", 0.3, "mock"))
        agg.clear("BTC")
        assert agg.get("BTC") is None
        assert agg.get("ETH") is not None

    def test_get_summary_empty(self):
        agg = SentimentAggregator()
        assert agg.get_summary() == {}

    def test_get_summary_populated(self):
        agg = SentimentAggregator()
        agg.add(SentimentSignal("BTC", 0.5, "mock"))
        summary = agg.get_summary()
        assert "BTC" in summary

    def test_get_summary_has_score_key(self):
        agg = SentimentAggregator()
        agg.add(SentimentSignal("BTC", 0.5, "mock"))
        summary = agg.get_summary()
        assert "score" in summary["BTC"]

    def test_get_summary_has_label_key(self):
        agg = SentimentAggregator()
        agg.add(SentimentSignal("BTC", 0.5, "mock"))
        summary = agg.get_summary()
        assert "label" in summary["BTC"]


# ─── Mock Data Provider ───────────────────────────────────────────────────────

class TestMockDataProvider:
    def test_generate_mock_signals_count(self):
        signals = generate_mock_signals("BTC", num_signals=4)
        assert len(signals) == 4

    def test_generate_mock_signals_ticker(self):
        signals = generate_mock_signals("ETH")
        for s in signals:
            assert s.ticker == "ETH"

    def test_generate_mock_signals_scores_in_range(self):
        signals = generate_mock_signals("BTC")
        for s in signals:
            assert -1.0 <= s.score <= 1.0

    def test_generate_mock_signals_deterministic(self):
        signals1 = generate_mock_signals("BTC", num_signals=4)
        signals2 = generate_mock_signals("BTC", num_signals=4)
        for s1, s2 in zip(signals1, signals2):
            assert s1.score == s2.score

    def test_generate_mock_signals_different_sources(self):
        signals = generate_mock_signals("BTC", num_signals=4)
        sources = {s.source for s in signals}
        assert len(sources) > 1

    def test_generate_mock_unknown_ticker(self):
        signals = generate_mock_signals("UNKNOWN_TOKEN")
        assert len(signals) > 0
        for s in signals:
            assert -1.0 <= s.score <= 1.0

    def test_create_mock_aggregator_default(self):
        agg = create_mock_aggregator()
        assert agg is not None

    def test_create_mock_aggregator_has_btc(self):
        agg = create_mock_aggregator(["BTC"])
        assert agg.get("BTC") is not None

    def test_create_mock_aggregator_custom_tickers(self):
        agg = create_mock_aggregator(["BTC", "ETH"])
        assert agg.get("BTC") is not None
        assert agg.get("ETH") is not None
        assert agg.get("SOL") is None
