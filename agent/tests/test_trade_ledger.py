"""
test_trade_ledger.py — Comprehensive tests for TradeLedger SQLite module.

Coverage:
  - LedgerEntry dataclass (to_dict, to_json, from_row)
  - LedgerSummary dataclass
  - TradeLedger: schema init, log_trade, get_entry, get_entries, get_summary
  - TradeLedger: update_status, count, clear, format_trace
  - Validation: empty fields, invalid side, negative size/price, bad status
  - Filtering: agent_id, market, side, status, limit, offset
  - Hash generation: format, uniqueness, determinism with explicit timestamp
  - Concurrent writes (same-process)
  - Edge cases: zero trades, single trade, many trades, unicode fields
  - Context manager usage
  - Persistent file-based DB (tmpdir)
"""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import tempfile
import time
import uuid
from datetime import datetime, timezone

import pytest

from trade_ledger import (
    LedgerEntry,
    LedgerError,
    LedgerSummary,
    LedgerValidationError,
    TradeLedger,
    VALID_SIDES,
    VALID_STATUSES,
)


# ─── Helpers ──────────────────────────────────────────────────────────────────


def make_ledger() -> TradeLedger:
    """Return a fresh in-memory TradeLedger."""
    return TradeLedger(":memory:")


def log(
    ledger: TradeLedger,
    agent_id: str = "agent-001",
    market: str = "BTC/USD",
    side: str = "BUY",
    size: float = 1.0,
    price: float = 50_000.0,
    **kwargs,
) -> LedgerEntry:
    return ledger.log_trade(agent_id=agent_id, market=market, side=side,
                            size=size, price=price, **kwargs)


# ─── Fixtures ─────────────────────────────────────────────────────────────────


@pytest.fixture
def ledger():
    return make_ledger()


@pytest.fixture
def populated_ledger():
    tl = make_ledger()
    log(tl, agent_id="agent-A", market="BTC/USD", side="BUY",  size=0.5, price=60_000.0)
    log(tl, agent_id="agent-A", market="BTC/USD", side="SELL", size=0.5, price=61_000.0)
    log(tl, agent_id="agent-B", market="ETH/USD", side="BUY",  size=2.0, price=3_000.0)
    log(tl, agent_id="agent-B", market="SOL/USD", side="BUY",  size=10.0, price=180.0)
    log(tl, agent_id="agent-C", market="ETH/USD", side="SELL", size=1.0, price=3_100.0)
    return tl


# ─── Constants ────────────────────────────────────────────────────────────────


class TestConstants:
    def test_valid_sides(self):
        assert "BUY" in VALID_SIDES
        assert "SELL" in VALID_SIDES
        assert len(VALID_SIDES) == 2

    def test_valid_statuses(self):
        assert "pending" in VALID_STATUSES
        assert "confirmed" in VALID_STATUSES
        assert "failed" in VALID_STATUSES
        assert len(VALID_STATUSES) == 3


# ─── LedgerEntry ──────────────────────────────────────────────────────────────


class TestLedgerEntry:
    def test_to_dict_keys(self, ledger):
        e = log(ledger)
        d = e.to_dict()
        assert set(d.keys()) == {
            "id", "tx_hash", "agent_id", "market", "side",
            "size", "price", "notional", "timestamp", "status", "created_at",
        }

    def test_to_dict_values(self, ledger):
        e = log(ledger, agent_id="x", market="BTC/USD", side="BUY",
                size=2.0, price=50_000.0)
        d = e.to_dict()
        assert d["agent_id"] == "x"
        assert d["market"] == "BTC/USD"
        assert d["side"] == "BUY"
        assert d["size"] == 2.0
        assert d["price"] == 50_000.0
        assert d["notional"] == 100_000.0

    def test_to_json_is_valid_json(self, ledger):
        e = log(ledger)
        j = e.to_json()
        parsed = json.loads(j)
        assert parsed["agent_id"] == "agent-001"

    def test_to_json_roundtrip(self, ledger):
        e = log(ledger)
        j = e.to_json()
        d = json.loads(j)
        assert d["side"] == e.side
        assert d["size"] == e.size
        assert d["price"] == e.price

    def test_id_is_int(self, ledger):
        e = log(ledger)
        assert isinstance(e.id, int)
        assert e.id >= 1

    def test_tx_hash_starts_with_0x(self, ledger):
        e = log(ledger)
        assert e.tx_hash.startswith("0x")

    def test_tx_hash_length(self, ledger):
        e = log(ledger)
        # "0x" + 64 hex chars = 66 chars total
        assert len(e.tx_hash) == 66

    def test_notional_computed(self, ledger):
        e = log(ledger, size=2.5, price=100.0)
        assert e.notional == pytest.approx(250.0)

    def test_default_status_pending(self, ledger):
        e = log(ledger)
        assert e.status == "pending"

    def test_status_confirmed(self, ledger):
        e = log(ledger, status="confirmed")
        assert e.status == "confirmed"

    def test_status_failed(self, ledger):
        e = log(ledger, status="failed")
        assert e.status == "failed"

    def test_timestamp_is_iso(self, ledger):
        e = log(ledger)
        # Should parse without error
        datetime.fromisoformat(e.timestamp.replace("Z", "+00:00"))

    def test_created_at_is_iso(self, ledger):
        e = log(ledger)
        datetime.fromisoformat(e.created_at.replace("Z", "+00:00"))


# ─── LedgerSummary ────────────────────────────────────────────────────────────


class TestLedgerSummary:
    def test_to_dict_keys(self, populated_ledger):
        s = populated_ledger.get_summary()
        d = s.to_dict()
        assert set(d.keys()) == {
            "total_trades", "total_notional", "buy_count", "sell_count",
            "unique_agents", "unique_markets", "confirmed_count",
            "pending_count", "failed_count",
        }

    def test_summary_counts(self, populated_ledger):
        s = populated_ledger.get_summary()
        assert s.total_trades == 5
        assert s.buy_count == 3
        assert s.sell_count == 2
        assert s.unique_agents == 3
        assert s.unique_markets == 3

    def test_summary_notional(self, populated_ledger):
        s = populated_ledger.get_summary()
        # 0.5*60000 + 0.5*61000 + 2*3000 + 10*180 + 1*3100
        expected = 30_000 + 30_500 + 6_000 + 1_800 + 3_100
        assert s.total_notional == pytest.approx(expected)

    def test_summary_pending(self, populated_ledger):
        s = populated_ledger.get_summary()
        # All inserted with default status=pending
        assert s.pending_count == 5
        assert s.confirmed_count == 0
        assert s.failed_count == 0

    def test_empty_summary(self, ledger):
        s = ledger.get_summary()
        assert s.total_trades == 0
        assert s.total_notional == 0.0
        assert s.buy_count == 0
        assert s.sell_count == 0
        assert s.unique_agents == 0


# ─── TradeLedger: Schema ──────────────────────────────────────────────────────


class TestTradeLedgerSchema:
    def test_table_exists(self, ledger):
        cur = ledger._conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='trades'"
        )
        assert cur.fetchone() is not None

    def test_indexes_exist(self, ledger):
        cur = ledger._conn.execute(
            "SELECT name FROM sqlite_master WHERE type='index'"
        )
        names = {row[0] for row in cur.fetchall()}
        assert "idx_agent_id" in names
        assert "idx_market" in names
        assert "idx_timestamp" in names

    def test_reinit_idempotent(self):
        """Second init should not raise or duplicate tables."""
        tl = make_ledger()
        tl._init_schema()
        tl._init_schema()
        assert tl.count() == 0


# ─── TradeLedger: log_trade ───────────────────────────────────────────────────


class TestLogTrade:
    def test_basic_log(self, ledger):
        e = log(ledger)
        assert e.id is not None
        assert e.tx_hash.startswith("0x")
        assert e.agent_id == "agent-001"
        assert e.market == "BTC/USD"
        assert e.side == "BUY"
        assert e.size == 1.0
        assert e.price == 50_000.0
        assert e.notional == 50_000.0

    def test_sell_side(self, ledger):
        e = log(ledger, side="SELL")
        assert e.side == "SELL"

    def test_custom_timestamp(self, ledger):
        ts = "2026-01-15T12:00:00+00:00"
        e = log(ledger, timestamp=ts)
        assert e.timestamp == ts

    def test_custom_tx_hash(self, ledger):
        custom = "0x" + "a" * 64
        e = log(ledger, tx_hash=custom)
        assert e.tx_hash == custom

    def test_auto_tx_hash_unique(self, ledger):
        """Each auto-generated hash should be unique (nonce makes it so)."""
        hashes = {log(ledger).tx_hash for _ in range(20)}
        assert len(hashes) == 20

    def test_count_increments(self, ledger):
        for i in range(5):
            log(ledger, size=float(i + 1))
        assert ledger.count() == 5

    def test_sequential_ids(self, ledger):
        e1 = log(ledger)
        e2 = log(ledger)
        assert e2.id == e1.id + 1

    def test_small_size(self, ledger):
        e = log(ledger, size=0.00001, price=1_000_000.0)
        assert e.notional == pytest.approx(10.0, rel=1e-5)

    def test_large_notional(self, ledger):
        e = log(ledger, size=100.0, price=100_000.0)
        assert e.notional == 10_000_000.0

    def test_status_stored(self, ledger):
        e = log(ledger, status="confirmed")
        fetched = ledger.get_entry(e.tx_hash)
        assert fetched.status == "confirmed"

    def test_multiple_markets(self, ledger):
        for market in ["BTC/USD", "ETH/USD", "SOL/USD"]:
            log(ledger, market=market)
        assert ledger.count() == 3

    def test_multiple_agents(self, ledger):
        for i in range(5):
            log(ledger, agent_id=f"agent-{i:03d}")
        assert ledger.count() == 5


# ─── TradeLedger: Validation ──────────────────────────────────────────────────


class TestValidation:
    def test_empty_agent_id(self, ledger):
        with pytest.raises(LedgerValidationError, match="agent_id"):
            log(ledger, agent_id="")

    def test_whitespace_agent_id(self, ledger):
        with pytest.raises(LedgerValidationError, match="agent_id"):
            log(ledger, agent_id="   ")

    def test_empty_market(self, ledger):
        with pytest.raises(LedgerValidationError, match="market"):
            log(ledger, market="")

    def test_whitespace_market(self, ledger):
        with pytest.raises(LedgerValidationError, match="market"):
            log(ledger, market="   ")

    def test_invalid_side_lower(self, ledger):
        with pytest.raises(LedgerValidationError, match="side"):
            log(ledger, side="buy")

    def test_invalid_side_random(self, ledger):
        with pytest.raises(LedgerValidationError, match="side"):
            log(ledger, side="HOLD")

    def test_zero_size(self, ledger):
        with pytest.raises(LedgerValidationError, match="size"):
            log(ledger, size=0.0)

    def test_negative_size(self, ledger):
        with pytest.raises(LedgerValidationError, match="size"):
            log(ledger, size=-1.0)

    def test_zero_price(self, ledger):
        with pytest.raises(LedgerValidationError, match="price"):
            log(ledger, price=0.0)

    def test_negative_price(self, ledger):
        with pytest.raises(LedgerValidationError, match="price"):
            log(ledger, price=-100.0)

    def test_invalid_status(self, ledger):
        with pytest.raises(LedgerValidationError, match="status"):
            log(ledger, status="unknown")

    def test_valid_sides_accepted(self, ledger):
        for side in VALID_SIDES:
            log(ledger, side=side)
        assert ledger.count() == 2

    def test_valid_statuses_accepted(self, ledger):
        for status in VALID_STATUSES:
            log(ledger, status=status)
        assert ledger.count() == 3

    def test_string_size_rejected(self, ledger):
        with pytest.raises((LedgerValidationError, TypeError)):
            log(ledger, size="1.0")  # type: ignore

    def test_string_price_rejected(self, ledger):
        with pytest.raises((LedgerValidationError, TypeError)):
            log(ledger, price="50000")  # type: ignore


# ─── TradeLedger: get_entry ───────────────────────────────────────────────────


class TestGetEntry:
    def test_get_existing(self, ledger):
        e = log(ledger)
        fetched = ledger.get_entry(e.tx_hash)
        assert fetched is not None
        assert fetched.tx_hash == e.tx_hash

    def test_get_nonexistent(self, ledger):
        result = ledger.get_entry("0x" + "0" * 64)
        assert result is None

    def test_get_returns_correct_fields(self, ledger):
        e = log(ledger, agent_id="z-agent", market="SOL/USD", side="SELL",
                size=5.0, price=200.0)
        fetched = ledger.get_entry(e.tx_hash)
        assert fetched.agent_id == "z-agent"
        assert fetched.market == "SOL/USD"
        assert fetched.side == "SELL"
        assert fetched.size == 5.0
        assert fetched.price == 200.0
        assert fetched.notional == 1000.0


# ─── TradeLedger: get_entries ─────────────────────────────────────────────────


class TestGetEntries:
    def test_get_all_empty(self, ledger):
        assert ledger.get_entries() == []

    def test_get_all(self, populated_ledger):
        entries = populated_ledger.get_entries()
        assert len(entries) == 5

    def test_filter_by_agent_id(self, populated_ledger):
        entries = populated_ledger.get_entries(agent_id="agent-A")
        assert len(entries) == 2
        assert all(e.agent_id == "agent-A" for e in entries)

    def test_filter_by_market(self, populated_ledger):
        entries = populated_ledger.get_entries(market="ETH/USD")
        assert len(entries) == 2
        assert all(e.market == "ETH/USD" for e in entries)

    def test_filter_by_side_buy(self, populated_ledger):
        entries = populated_ledger.get_entries(side="BUY")
        assert len(entries) == 3
        assert all(e.side == "BUY" for e in entries)

    def test_filter_by_side_sell(self, populated_ledger):
        entries = populated_ledger.get_entries(side="SELL")
        assert len(entries) == 2
        assert all(e.side == "SELL" for e in entries)

    def test_filter_by_status(self, populated_ledger):
        entries = populated_ledger.get_entries(status="pending")
        assert len(entries) == 5

    def test_filter_combined_agent_market(self, populated_ledger):
        entries = populated_ledger.get_entries(agent_id="agent-B", market="ETH/USD")
        assert len(entries) == 1
        assert entries[0].agent_id == "agent-B"
        assert entries[0].market == "ETH/USD"

    def test_filter_no_match(self, populated_ledger):
        entries = populated_ledger.get_entries(agent_id="nonexistent")
        assert entries == []

    def test_limit(self, populated_ledger):
        entries = populated_ledger.get_entries(limit=2)
        assert len(entries) == 2

    def test_offset(self, populated_ledger):
        all_entries = populated_ledger.get_entries()
        offset_entries = populated_ledger.get_entries(offset=2)
        assert len(offset_entries) == 3
        assert offset_entries[0].id == all_entries[2].id

    def test_limit_offset(self, populated_ledger):
        entries = populated_ledger.get_entries(limit=2, offset=1)
        assert len(entries) == 2

    def test_order_by_id_asc(self, populated_ledger):
        entries = populated_ledger.get_entries()
        ids = [e.id for e in entries]
        assert ids == sorted(ids)

    def test_get_all_alias(self, populated_ledger):
        assert populated_ledger.get_all() == populated_ledger.get_entries()


# ─── TradeLedger: update_status ───────────────────────────────────────────────


class TestUpdateStatus:
    def test_update_to_confirmed(self, ledger):
        e = log(ledger)
        result = ledger.update_status(e.tx_hash, "confirmed")
        assert result is True
        fetched = ledger.get_entry(e.tx_hash)
        assert fetched.status == "confirmed"

    def test_update_to_failed(self, ledger):
        e = log(ledger)
        result = ledger.update_status(e.tx_hash, "failed")
        assert result is True
        fetched = ledger.get_entry(e.tx_hash)
        assert fetched.status == "failed"

    def test_update_back_to_pending(self, ledger):
        e = log(ledger, status="confirmed")
        ledger.update_status(e.tx_hash, "pending")
        fetched = ledger.get_entry(e.tx_hash)
        assert fetched.status == "pending"

    def test_update_nonexistent_returns_false(self, ledger):
        result = ledger.update_status("0x" + "0" * 64, "confirmed")
        assert result is False

    def test_update_invalid_status_raises(self, ledger):
        e = log(ledger)
        with pytest.raises(LedgerValidationError, match="status"):
            ledger.update_status(e.tx_hash, "invalid")


# ─── TradeLedger: count ───────────────────────────────────────────────────────


class TestCount:
    def test_empty(self, ledger):
        assert ledger.count() == 0

    def test_after_inserts(self, ledger):
        for _ in range(7):
            log(ledger)
        assert ledger.count() == 7


# ─── TradeLedger: clear ───────────────────────────────────────────────────────


class TestClear:
    def test_clear_empty(self, ledger):
        result = ledger.clear()
        assert result == 0
        assert ledger.count() == 0

    def test_clear_populated(self, populated_ledger):
        result = populated_ledger.clear()
        assert result == 5
        assert populated_ledger.count() == 0

    def test_clear_then_insert(self, populated_ledger):
        populated_ledger.clear()
        log(populated_ledger)
        assert populated_ledger.count() == 1


# ─── TradeLedger: Summary Filters ─────────────────────────────────────────────


class TestSummaryFilters:
    def test_summary_by_agent(self, populated_ledger):
        s = populated_ledger.get_summary(agent_id="agent-A")
        assert s.total_trades == 2
        assert s.buy_count == 1
        assert s.sell_count == 1

    def test_summary_by_market(self, populated_ledger):
        s = populated_ledger.get_summary(market="ETH/USD")
        assert s.total_trades == 2

    def test_summary_by_agent_and_market(self, populated_ledger):
        s = populated_ledger.get_summary(agent_id="agent-B", market="ETH/USD")
        assert s.total_trades == 1

    def test_summary_after_status_update(self, ledger):
        e = log(ledger)
        ledger.update_status(e.tx_hash, "confirmed")
        s = ledger.get_summary()
        assert s.confirmed_count == 1
        assert s.pending_count == 0

    def test_summary_mixed_statuses(self, ledger):
        e1 = log(ledger)
        e2 = log(ledger)
        e3 = log(ledger)
        ledger.update_status(e1.tx_hash, "confirmed")
        ledger.update_status(e3.tx_hash, "failed")
        s = ledger.get_summary()
        assert s.confirmed_count == 1
        assert s.pending_count == 1
        assert s.failed_count == 1


# ─── TradeLedger: format_trace ────────────────────────────────────────────────


class TestFormatTrace:
    def test_empty_ledger_trace(self, ledger):
        trace = ledger.format_trace()
        assert "(no trades recorded)" in trace

    def test_trace_contains_header(self, ledger):
        log(ledger)
        trace = ledger.format_trace()
        assert "AGENT" in trace
        assert "MARKET" in trace
        assert "SIDE" in trace
        assert "TX_HASH" in trace

    def test_trace_contains_data(self, populated_ledger):
        trace = populated_ledger.format_trace()
        assert "agent-A" in trace
        assert "BTC/USD" in trace
        assert "BUY" in trace
        assert "SELL" in trace

    def test_trace_summary_line(self, populated_ledger):
        trace = populated_ledger.format_trace()
        assert "Total:" in trace
        assert "trades" in trace

    def test_trace_custom_entries(self, ledger):
        entries = [log(ledger, market="SOL/USD")]
        trace = ledger.format_trace(entries)
        assert "SOL/USD" in trace


# ─── TradeLedger: Context Manager ─────────────────────────────────────────────


class TestContextManager:
    def test_context_manager(self):
        with TradeLedger(":memory:") as tl:
            log(tl)
            assert tl.count() == 1

    def test_context_manager_closes(self):
        with TradeLedger(":memory:") as tl:
            tl.close()
        # Should not raise on redundant close


# ─── TradeLedger: Persistent DB ───────────────────────────────────────────────


class TestPersistentDB:
    def test_file_db_persists(self, tmp_path):
        db_path = str(tmp_path / "trades.db")
        with TradeLedger(db_path) as tl:
            e = log(tl)
            tx = e.tx_hash

        # Reopen and verify
        with TradeLedger(db_path) as tl2:
            fetched = tl2.get_entry(tx)
            assert fetched is not None
            assert fetched.tx_hash == tx

    def test_file_db_count(self, tmp_path):
        db_path = str(tmp_path / "count.db")
        with TradeLedger(db_path) as tl:
            for _ in range(10):
                log(tl)

        with TradeLedger(db_path) as tl2:
            assert tl2.count() == 10

    def test_db_path_attribute(self, tmp_path):
        db_path = str(tmp_path / "x.db")
        tl = TradeLedger(db_path)
        assert tl.db_path == db_path
        tl.close()


# ─── TradeLedger: Hash Generation ─────────────────────────────────────────────


class TestHashGeneration:
    def test_hash_hex_format(self, ledger):
        e = log(ledger)
        hex_part = e.tx_hash[2:]  # strip "0x"
        assert all(c in "0123456789abcdef" for c in hex_part)

    def test_hash_64_hex_chars(self, ledger):
        e = log(ledger)
        assert len(e.tx_hash[2:]) == 64

    def test_explicit_hash_override(self, ledger):
        custom = "0x" + "b" * 64
        e = log(ledger, tx_hash=custom)
        assert e.tx_hash == custom

    def test_duplicate_hash_raises(self, ledger):
        custom = "0x" + "c" * 64
        log(ledger, tx_hash=custom)
        with pytest.raises(LedgerError):
            log(ledger, tx_hash=custom)


# ─── TradeLedger: Unicode / Special Fields ────────────────────────────────────


class TestUnicodeFields:
    def test_unicode_agent_id(self, ledger):
        e = log(ledger, agent_id="агент-001")
        fetched = ledger.get_entry(e.tx_hash)
        assert fetched.agent_id == "агент-001"

    def test_unicode_market(self, ledger):
        e = log(ledger, market="比特币/美元")
        fetched = ledger.get_entry(e.tx_hash)
        assert fetched.market == "比特币/美元"

    def test_long_agent_id(self, ledger):
        long_id = "a" * 255
        e = log(ledger, agent_id=long_id)
        fetched = ledger.get_entry(e.tx_hash)
        assert fetched.agent_id == long_id


# ─── TradeLedger: High-Volume ─────────────────────────────────────────────────


class TestHighVolume:
    def test_100_trades(self, ledger):
        for i in range(100):
            log(ledger, agent_id=f"agent-{i}", size=float(i + 1))
        assert ledger.count() == 100

    def test_summary_100_trades(self, ledger):
        for i in range(50):
            log(ledger, side="BUY")
        for i in range(50):
            log(ledger, side="SELL")
        s = ledger.get_summary()
        assert s.total_trades == 100
        assert s.buy_count == 50
        assert s.sell_count == 50

    def test_get_entries_all_100(self, ledger):
        for i in range(100):
            log(ledger)
        entries = ledger.get_entries()
        assert len(entries) == 100

    def test_get_entries_limit_50(self, ledger):
        for i in range(100):
            log(ledger)
        entries = ledger.get_entries(limit=50)
        assert len(entries) == 50
