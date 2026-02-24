"""
Tests for the reputation logging module.
"""

import hashlib
import json
import time
import pytest
from unittest.mock import MagicMock, AsyncMock, patch
from datetime import datetime, timezone

from reputation import (
    ReputationEntry,
    ReputationLogger,
    ReputationStats,
    calculate_trade_score,
    build_file_hash,
    REPUTATION_REGISTRY_BASE_SEPOLIA,
    TAG_ACCURACY,
    TAG_TRADING,
)
from trader import MarketSide, TradeResult


# ─── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def pending_trade():
    return TradeResult(
        market_id="market-001",
        question="Will BTC exceed $100k?",
        side=MarketSide.YES,
        size_usdc=5.0,
        executed_at=datetime.now(timezone.utc).isoformat(),
        outcome="PENDING",
        pnl_usdc=0.0,
        tx_hash=None,
        data_uri="data:application/json,{}",
        data_hash=b"\x00" * 32,
    )


@pytest.fixture
def win_trade():
    return TradeResult(
        market_id="market-002",
        question="Will ETH exceed $4k?",
        side=MarketSide.YES,
        size_usdc=10.0,
        executed_at=datetime.now(timezone.utc).isoformat(),
        outcome="WIN",
        pnl_usdc=2.50,
        tx_hash="0x" + "aa" * 32,
        data_uri="ipfs://QmTest",
        data_hash=b"\x11" * 32,
    )


@pytest.fixture
def loss_trade():
    return TradeResult(
        market_id="market-003",
        question="Will BNB stay above $400?",
        side=MarketSide.NO,
        size_usdc=8.0,
        executed_at=datetime.now(timezone.utc).isoformat(),
        outcome="LOSS",
        pnl_usdc=-3.0,
        tx_hash=None,
        data_uri="data:application/json,{}",
        data_hash=b"\x22" * 32,
    )


@pytest.fixture
def dry_run_logger():
    return ReputationLogger(registry=None, agent_id=1, dry_run=True)


# ─── Score calculation tests ──────────────────────────────────────────────────

class TestCalculateTradeScore:
    def test_pending_score(self, pending_trade):
        score = calculate_trade_score(pending_trade)
        assert score == 500

    def test_win_score_base(self, win_trade):
        win_trade.pnl_usdc = 0.0
        score = calculate_trade_score(win_trade)
        assert score == 850

    def test_win_score_with_profit(self, win_trade):
        win_trade.pnl_usdc = 5.0
        score = calculate_trade_score(win_trade)
        assert score >= 850

    def test_win_score_capped(self, win_trade):
        win_trade.pnl_usdc = 1000.0  # Huge profit
        score = calculate_trade_score(win_trade)
        assert score <= 950

    def test_loss_score_base(self, loss_trade):
        loss_trade.pnl_usdc = -5.0  # Big loss
        score = calculate_trade_score(loss_trade)
        assert score == 150

    def test_loss_score_small_loss(self, loss_trade):
        loss_trade.pnl_usdc = -0.5  # Small loss (< 10% of 8.0)
        score = calculate_trade_score(loss_trade)
        assert score == 300

    def test_score_in_valid_range(self, pending_trade, win_trade, loss_trade):
        for trade in [pending_trade, win_trade, loss_trade]:
            score = calculate_trade_score(trade)
            assert 0 <= score <= 1000

    def test_null_outcome_returns_500(self, pending_trade):
        pending_trade.outcome = None
        score = calculate_trade_score(pending_trade)
        assert score == 500


# ─── build_file_hash tests ────────────────────────────────────────────────────

class TestBuildFileHash:
    def test_returns_bytes(self, pending_trade):
        file_hash = build_file_hash(pending_trade)
        assert isinstance(file_hash, bytes)
        assert len(file_hash) == 32

    def test_deterministic(self, pending_trade):
        hash1 = build_file_hash(pending_trade)
        hash2 = build_file_hash(pending_trade)
        assert hash1 == hash2

    def test_different_trades_different_hashes(self, pending_trade, win_trade):
        hash1 = build_file_hash(pending_trade)
        hash2 = build_file_hash(win_trade)
        assert hash1 != hash2

    def test_hash_is_sha256(self, pending_trade):
        file_hash = build_file_hash(pending_trade)
        # Verify it's a valid sha256 output (32 bytes)
        assert len(file_hash) == 32

    def test_outcome_affects_hash(self, pending_trade):
        hash_pending = build_file_hash(pending_trade)
        pending_trade.outcome = "WIN"
        hash_win = build_file_hash(pending_trade)
        assert hash_pending != hash_win


# ─── ReputationEntry tests ────────────────────────────────────────────────────

class TestReputationEntry:
    def test_to_dict_structure(self):
        entry = ReputationEntry(
            agent_id=1,
            trade_id="trade-001",
            market_id="market-001",
            outcome="WIN",
            score=850,
            tag1="accuracy",
            tag2="trading",
            tx_hash="0xabc",
            on_chain=True,
            demo_mode=False,
        )
        d = entry.to_dict()
        assert d["agent_id"] == 1
        assert d["trade_id"] == "trade-001"
        assert d["outcome"] == "WIN"
        assert d["score"] == 850
        assert d["score_normalized"] == 8.50
        assert d["on_chain"] is True
        assert "timestamp" in d

    def test_score_normalized_calculation(self):
        entry = ReputationEntry(
            agent_id=1, trade_id="t", market_id="m",
            outcome="PENDING", score=500,
            tag1="a", tag2="b", tx_hash=None,
        )
        assert entry.to_dict()["score_normalized"] == 5.0


# ─── ReputationStats tests ────────────────────────────────────────────────────

class TestReputationStats:
    def test_win_rate_no_trades(self):
        stats = ReputationStats(
            agent_id=1, total_feedback=0, aggregate_score=0.0,
            on_chain_count=0, demo_count=0,
            win_count=0, loss_count=0, pending_count=0,
        )
        assert stats.win_rate == 0.0

    def test_win_rate_calculation(self):
        stats = ReputationStats(
            agent_id=1, total_feedback=10, aggregate_score=7.5,
            on_chain_count=5, demo_count=5,
            win_count=7, loss_count=3, pending_count=0,
        )
        assert abs(stats.win_rate - 0.7) < 0.001

    def test_win_rate_all_wins(self):
        stats = ReputationStats(
            agent_id=1, total_feedback=5, aggregate_score=9.0,
            on_chain_count=5, demo_count=0,
            win_count=5, loss_count=0, pending_count=0,
        )
        assert stats.win_rate == 1.0

    def test_to_dict_structure(self):
        stats = ReputationStats(
            agent_id=2, total_feedback=3, aggregate_score=7.5,
            on_chain_count=1, demo_count=2,
            win_count=2, loss_count=1, pending_count=0,
        )
        d = stats.to_dict()
        assert d["agent_id"] == 2
        assert d["total_feedback"] == 3
        assert "aggregate_score" in d
        assert "win_rate" in d


# ─── ReputationLogger tests ───────────────────────────────────────────────────

class TestReputationLogger:
    @pytest.mark.asyncio
    async def test_log_pending_trade_dry_run(self, dry_run_logger, pending_trade):
        entry = await dry_run_logger.log_trade(pending_trade)
        assert entry is not None
        assert entry.outcome == "PENDING"
        assert entry.score == 500
        assert entry.on_chain is False
        assert entry.demo_mode is True

    @pytest.mark.asyncio
    async def test_log_win_trade_dry_run(self, dry_run_logger, win_trade):
        entry = await dry_run_logger.log_trade(win_trade)
        assert entry is not None
        assert entry.outcome == "WIN"
        assert entry.score >= 850

    @pytest.mark.asyncio
    async def test_log_loss_trade_dry_run(self, dry_run_logger, loss_trade):
        entry = await dry_run_logger.log_trade(loss_trade)
        assert entry is not None
        assert entry.outcome == "LOSS"
        assert entry.score <= 300

    @pytest.mark.asyncio
    async def test_multiple_trades_accumulate(self, dry_run_logger, pending_trade, win_trade):
        await dry_run_logger.log_trade(pending_trade)
        await dry_run_logger.log_trade(win_trade)
        stats = dry_run_logger.get_stats()
        assert stats.total_feedback == 2

    @pytest.mark.asyncio
    async def test_stats_empty(self, dry_run_logger):
        stats = dry_run_logger.get_stats()
        assert stats.total_feedback == 0
        assert stats.aggregate_score == 0.0
        assert stats.win_rate == 0.0

    @pytest.mark.asyncio
    async def test_stats_after_trades(self, dry_run_logger, win_trade, loss_trade, pending_trade):
        await dry_run_logger.log_trade(win_trade)
        await dry_run_logger.log_trade(loss_trade)
        await dry_run_logger.log_trade(pending_trade)
        stats = dry_run_logger.get_stats()
        assert stats.total_feedback == 3
        assert stats.win_count == 1
        assert stats.loss_count == 1
        assert stats.pending_count == 1
        assert 0.0 <= stats.aggregate_score <= 10.0

    def test_set_agent_id(self, dry_run_logger):
        dry_run_logger.set_agent_id(42)
        assert dry_run_logger.agent_id == 42

    def test_get_log_returns_dicts(self):
        logger_obj = ReputationLogger(agent_id=1, dry_run=True)
        log = logger_obj.get_log()
        assert isinstance(log, list)

    @pytest.mark.asyncio
    async def test_get_log_after_trades(self, dry_run_logger, win_trade, pending_trade):
        await dry_run_logger.log_trade(win_trade)
        await dry_run_logger.log_trade(pending_trade)
        log = dry_run_logger.get_log()
        assert len(log) == 2
        assert all(isinstance(e, dict) for e in log)

    def test_get_on_chain_score_no_registry(self, dry_run_logger):
        result = dry_run_logger.get_on_chain_score()
        assert result is None

    def test_get_on_chain_score_agent_id_zero(self):
        mock_registry = MagicMock()
        logger_obj = ReputationLogger(registry=mock_registry, agent_id=0, dry_run=False)
        result = logger_obj.get_on_chain_score()
        assert result is None

    @pytest.mark.asyncio
    async def test_live_mode_submits_to_registry(self, win_trade):
        mock_registry = MagicMock()
        mock_registry.give_feedback.return_value = "0x" + "ff" * 32
        logger_obj = ReputationLogger(registry=mock_registry, agent_id=5, dry_run=False)

        entry = await logger_obj.log_trade(win_trade)
        assert entry is not None
        assert entry.on_chain is True
        assert entry.tx_hash == "0x" + "ff" * 32
        mock_registry.give_feedback.assert_called_once()

    @pytest.mark.asyncio
    async def test_live_mode_handles_registry_error(self, win_trade):
        mock_registry = MagicMock()
        mock_registry.give_feedback.side_effect = Exception("Connection error")
        logger_obj = ReputationLogger(registry=mock_registry, agent_id=5, dry_run=False)

        entry = await logger_obj.log_trade(win_trade)
        assert entry is not None
        assert entry.on_chain is False
        assert entry.tx_hash is None

    @pytest.mark.asyncio
    async def test_trade_id_format(self, dry_run_logger, pending_trade):
        entry = await dry_run_logger.log_trade(pending_trade)
        assert pending_trade.market_id in entry.trade_id
        assert pending_trade.executed_at in entry.trade_id

    @pytest.mark.asyncio
    async def test_tags_are_set(self, dry_run_logger, pending_trade):
        entry = await dry_run_logger.log_trade(pending_trade)
        assert entry.tag1 == TAG_ACCURACY
        assert entry.tag2 == TAG_TRADING


# ─── Constants tests ──────────────────────────────────────────────────────────

class TestConstants:
    def test_registry_address_format(self):
        assert REPUTATION_REGISTRY_BASE_SEPOLIA.startswith("0x")
        assert len(REPUTATION_REGISTRY_BASE_SEPOLIA) == 42

    def test_tags(self):
        assert TAG_ACCURACY == "accuracy"
        assert TAG_TRADING == "trading"
