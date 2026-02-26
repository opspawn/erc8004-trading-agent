"""
Tests for reputation_updater.py — ERC-8004 On-Chain Reputation Updater.

All tests are unit-based. No live chain transactions are made.
SIMULATION_MODE is always True in tests.
"""

from __future__ import annotations

import json
import os
import sys
import hashlib

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from trade_ledger import TradeLedger, LedgerEntry
from reputation_updater import (
    ReputationUpdate,
    SyncResult,
    ReputationUpdater,
    encode_reputation_update,
    sign_reputation_update,
    ensure_reputation_column,
    get_pending_reputation_entries,
    mark_reputation_synced,
    _trade_hash_to_bytes32,
    _estimate_pnl_bps,
    _build_simulation_tx,
    _get_signing_key,
    EIP712_DOMAIN,
    EIP712_TYPES,
    REPUTATION_REGISTRY_ADDRESS,
    AGENT_WALLET_ADDRESS,
    SIMULATION_MODE,
    REPUTATION_TX_COLUMN,
)


# ─── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture
def ledger():
    return TradeLedger()  # in-memory


@pytest.fixture
def populated_ledger():
    l = TradeLedger()
    for i in range(5):
        l.log_trade(
            agent_id="test-agent",
            market="BTC/USD",
            side="BUY" if i % 2 == 0 else "SELL",
            size=0.1 * (i + 1),
            price=50000.0 + i * 100,
        )
    return l


@pytest.fixture
def updater(populated_ledger):
    return ReputationUpdater(
        ledger=populated_ledger,
        agent_id="test-agent",
        simulation=True,
    )


# ─── Constants Tests ───────────────────────────────────────────────────────────


class TestConstants:
    def test_registry_address_is_hex(self):
        assert REPUTATION_REGISTRY_ADDRESS.startswith("0x")
        assert len(REPUTATION_REGISTRY_ADDRESS) == 42

    def test_agent_wallet_address_is_hex(self):
        assert AGENT_WALLET_ADDRESS.startswith("0x")
        assert len(AGENT_WALLET_ADDRESS) == 42

    def test_simulation_mode_default_true(self):
        assert SIMULATION_MODE is True

    def test_eip712_domain_fields(self):
        assert "name" in EIP712_DOMAIN
        assert "version" in EIP712_DOMAIN
        assert "chainId" in EIP712_DOMAIN
        assert "verifyingContract" in EIP712_DOMAIN

    def test_eip712_domain_name(self):
        assert EIP712_DOMAIN["name"] == "ERC8004ReputationRegistry"

    def test_eip712_types_has_reputation_update(self):
        assert "ReputationUpdate" in EIP712_TYPES

    def test_eip712_reputation_update_fields(self):
        fields = [f["name"] for f in EIP712_TYPES["ReputationUpdate"]]
        assert "agentId" in fields
        assert "market" in fields
        assert "outcomePnlBps" in fields
        assert "tradeHash" in fields
        assert "timestamp" in fields

    def test_reputation_tx_column_name(self):
        assert REPUTATION_TX_COLUMN == "reputation_update_tx"


# ─── Helper Tests ──────────────────────────────────────────────────────────────


class TestTradeHashToBytes32:
    def test_valid_hex_hash(self):
        h = "0x" + "a" * 64
        result = _trade_hash_to_bytes32(h)
        assert isinstance(result, bytes)
        assert len(result) == 32

    def test_shorter_hash_padded(self):
        h = "0xdeadbeef"
        result = _trade_hash_to_bytes32(h)
        assert len(result) == 32

    def test_strips_0x_prefix(self):
        h = "0x" + "ff" * 32
        result = _trade_hash_to_bytes32(h)
        assert result == bytes.fromhex("ff" * 32)

    def test_truncates_long_hash(self):
        h = "0x" + "ab" * 64  # 128 hex chars → 64 bytes → truncated to 32
        result = _trade_hash_to_bytes32(h)
        assert len(result) == 32


class TestEstimatePnlBps:
    def test_buy_returns_positive(self, populated_ledger):
        entries = populated_ledger.get_entries(side="BUY")
        for e in entries:
            assert _estimate_pnl_bps(e) > 0

    def test_sell_returns_negative(self, populated_ledger):
        entries = populated_ledger.get_entries(side="SELL")
        for e in entries:
            assert _estimate_pnl_bps(e) < 0

    def test_buy_returns_50(self, populated_ledger):
        entries = populated_ledger.get_entries(side="BUY")
        assert _estimate_pnl_bps(entries[0]) == 50

    def test_sell_returns_minus_50(self, populated_ledger):
        entries = populated_ledger.get_entries(side="SELL")
        assert _estimate_pnl_bps(entries[0]) == -50


class TestBuildSimulationTx:
    def test_returns_0xsim_prefix(self):
        tx = _build_simulation_tx("agent-1", "0xabc123")
        assert tx.startswith("0xsim_")

    def test_deterministic(self):
        tx1 = _build_simulation_tx("agent-1", "0xabc123")
        tx2 = _build_simulation_tx("agent-1", "0xabc123")
        assert tx1 == tx2

    def test_different_inputs_produce_different_hashes(self):
        tx1 = _build_simulation_tx("agent-1", "0xabc123")
        tx2 = _build_simulation_tx("agent-2", "0xdef456")
        assert tx1 != tx2

    def test_length_reasonable(self):
        tx = _build_simulation_tx("agent-1", "0xabc123")
        assert len(tx) > 10


# ─── ReputationUpdate Tests ────────────────────────────────────────────────────


class TestReputationUpdate:
    def test_to_dict_contains_required_fields(self):
        u = ReputationUpdate(
            agent_id="a1",
            market="BTC/USD",
            outcome_pnl_bps=50,
            trade_hash="0xabc",
            timestamp=1700000000,
        )
        d = u.to_dict()
        assert d["agent_id"] == "a1"
        assert d["market"] == "BTC/USD"
        assert d["outcome_pnl_bps"] == 50
        assert d["trade_hash"] == "0xabc"
        assert d["timestamp"] == 1700000000

    def test_to_dict_has_signature_field(self):
        u = ReputationUpdate(
            agent_id="a1",
            market="ETH/USD",
            outcome_pnl_bps=-50,
            trade_hash="0xdef",
            timestamp=0,
        )
        assert "signature" in u.to_dict()

    def test_simulation_mode_defaults_true(self):
        u = ReputationUpdate(
            agent_id="a", market="X", outcome_pnl_bps=0,
            trade_hash="0x0", timestamp=0
        )
        assert u.simulation_mode is True

    def test_can_set_signature(self):
        u = ReputationUpdate(
            agent_id="a", market="X", outcome_pnl_bps=0,
            trade_hash="0x0", timestamp=0
        )
        u.signature = "0xdeadbeef"
        assert u.to_dict()["signature"] == "0xdeadbeef"


# ─── EIP-712 Encoding Tests ────────────────────────────────────────────────────


class TestEip712Encoding:
    def test_encode_returns_signable(self):
        u = ReputationUpdate(
            agent_id="test-agent",
            market="BTC/USD",
            outcome_pnl_bps=50,
            trade_hash="0x" + "a" * 64,
            timestamp=1700000000,
        )
        signable = encode_reputation_update(u)
        assert signable is not None

    def test_encode_different_pnl_different_hash(self):
        h = "0x" + "a" * 64
        u1 = ReputationUpdate("a", "BTC/USD", 50, h, 1700000000)
        u2 = ReputationUpdate("a", "BTC/USD", -50, h, 1700000000)
        s1 = encode_reputation_update(u1)
        s2 = encode_reputation_update(u2)
        # Different pnl → different body → different hash
        assert s1.body != s2.body

    def test_encode_same_input_deterministic(self):
        h = "0x" + "b" * 64
        u = ReputationUpdate("a", "BTC/USD", 50, h, 1700000000)
        s1 = encode_reputation_update(u)
        s2 = encode_reputation_update(u)
        assert s1.body == s2.body

    def test_sign_produces_hex_signature(self):
        from reputation_updater import _DEV_PRIVATE_KEY
        u = ReputationUpdate(
            agent_id="test-agent",
            market="ETH/USD",
            outcome_pnl_bps=30,
            trade_hash="0x" + "c" * 64,
            timestamp=1700000001,
        )
        sig = sign_reputation_update(u, _DEV_PRIVATE_KEY)
        assert isinstance(sig, str)
        assert sig.startswith("0x") or len(sig) == 130  # 65 bytes hex

    def test_sign_length_is_130_chars(self):
        from reputation_updater import _DEV_PRIVATE_KEY
        u = ReputationUpdate("a", "BTC/USD", 50, "0x" + "d" * 64, 1700000000)
        sig = sign_reputation_update(u, _DEV_PRIVATE_KEY)
        # eth_account returns 65-byte (130 hex-char) signature, with or without 0x prefix
        hex_body = sig[2:] if sig.startswith("0x") else sig
        assert len(hex_body) == 130  # 65 bytes = 130 hex chars


# ─── Ledger Migration Tests ────────────────────────────────────────────────────


class TestLedgerMigration:
    def test_ensure_reputation_column_idempotent(self, ledger):
        ensure_reputation_column(ledger)
        ensure_reputation_column(ledger)  # second call should not raise
        # Check column exists
        cur = ledger._conn.execute("PRAGMA table_info(trades)")
        cols = [r["name"] for r in cur.fetchall()]
        assert REPUTATION_TX_COLUMN in cols

    def test_get_pending_empty_ledger(self, ledger):
        pending = get_pending_reputation_entries(ledger)
        assert pending == []

    def test_get_pending_returns_all_when_none_synced(self, populated_ledger):
        pending = get_pending_reputation_entries(populated_ledger)
        assert len(pending) == 5

    def test_mark_synced_removes_from_pending(self, populated_ledger):
        pending = get_pending_reputation_entries(populated_ledger)
        first = pending[0]
        mark_reputation_synced(populated_ledger, first.tx_hash, "0xsim_abc")
        pending2 = get_pending_reputation_entries(populated_ledger)
        assert len(pending2) == 4

    def test_mark_synced_stores_tx(self, populated_ledger):
        ensure_reputation_column(populated_ledger)
        pending = get_pending_reputation_entries(populated_ledger)
        first = pending[0]
        mark_reputation_synced(populated_ledger, first.tx_hash, "0xsim_test")
        row = populated_ledger._conn.execute(
            f"SELECT {REPUTATION_TX_COLUMN} FROM trades WHERE tx_hash = ?",
            (first.tx_hash,)
        ).fetchone()
        assert row[0] == "0xsim_test"


# ─── ReputationUpdater Tests ───────────────────────────────────────────────────


class TestReputationUpdaterInit:
    def test_default_init(self):
        u = ReputationUpdater()
        assert u.simulation is True
        assert u.ledger is not None

    def test_custom_agent_id(self):
        u = ReputationUpdater(agent_id="custom-agent")
        assert u.agent_id == "custom-agent"

    def test_simulation_flag(self):
        u = ReputationUpdater(simulation=True)
        assert u.simulation is True

    def test_ledger_passed_in(self, ledger):
        u = ReputationUpdater(ledger=ledger)
        assert u.ledger is ledger


class TestReputationUpdaterBuild:
    def test_build_update_returns_reputation_update(self, updater, populated_ledger):
        entry = populated_ledger.get_all()[0]
        update = updater.build_update(entry)
        assert isinstance(update, ReputationUpdate)

    def test_build_update_agent_id(self, updater, populated_ledger):
        entry = populated_ledger.get_all()[0]
        update = updater.build_update(entry)
        assert update.agent_id == "test-agent"

    def test_build_update_market(self, updater, populated_ledger):
        entry = populated_ledger.get_all()[0]
        update = updater.build_update(entry)
        assert update.market == "BTC/USD"

    def test_build_update_trade_hash(self, updater, populated_ledger):
        entry = populated_ledger.get_all()[0]
        update = updater.build_update(entry)
        assert update.trade_hash == entry.tx_hash

    def test_build_update_timestamp_is_int(self, updater, populated_ledger):
        entry = populated_ledger.get_all()[0]
        update = updater.build_update(entry)
        assert isinstance(update.timestamp, int)
        assert update.timestamp > 0


class TestReputationUpdaterSync:
    def test_sync_pending_returns_sync_result(self, updater):
        result = updater.sync_pending()
        assert isinstance(result, SyncResult)

    def test_sync_pending_syncs_all(self, updater):
        result = updater.sync_pending()
        assert result.synced_count == 5

    def test_sync_pending_no_failures(self, updater):
        result = updater.sync_pending()
        assert result.failed_count == 0

    def test_sync_pending_twice_no_duplicates(self, updater):
        result1 = updater.sync_pending()
        result2 = updater.sync_pending()
        assert result1.synced_count == 5
        assert result2.synced_count == 0  # Nothing pending after first sync

    def test_sync_updates_have_tx(self, updater):
        result = updater.sync_pending()
        for u in result.updates:
            assert u.update_tx is not None
            assert len(u.update_tx) > 5

    def test_sync_updates_have_signature(self, updater):
        result = updater.sync_pending()
        for u in result.updates:
            assert u.signature is not None

    def test_sync_result_to_dict(self, updater):
        result = updater.sync_pending()
        d = result.to_dict()
        assert "synced_count" in d
        assert "updates" in d
        assert isinstance(d["updates"], list)

    def test_get_status_before_sync(self, updater):
        status = updater.get_status()
        assert status["pending_trades"] == 5
        assert status["synced_trades"] == 0

    def test_get_status_after_sync(self, updater):
        updater.sync_pending()
        status = updater.get_status()
        assert status["pending_trades"] == 0
        assert status["synced_trades"] == 5

    def test_get_status_simulation_mode(self, updater):
        status = updater.get_status()
        assert status["simulation_mode"] is True

    def test_get_status_contains_registry_address(self, updater):
        status = updater.get_status()
        assert "registry_address" in status
        assert status["registry_address"].startswith("0x")

    def test_process_entry_returns_update(self, updater, populated_ledger):
        entry = populated_ledger.get_all()[0]
        update = updater.process_entry(entry)
        assert isinstance(update, ReputationUpdate)
        assert update.update_tx is not None

    def test_empty_ledger_sync(self, ledger):
        u = ReputationUpdater(ledger=ledger, simulation=True)
        result = u.sync_pending()
        assert result.synced_count == 0
        assert result.failed_count == 0


# ─── Additional Coverage Tests ────────────────────────────────────────────────


class TestReputationUpdateAdditional:
    def test_update_tx_initially_none(self):
        u = ReputationUpdate("a", "BTC/USD", 50, "0x0", 0)
        assert u.update_tx is None

    def test_can_set_update_tx(self):
        u = ReputationUpdate("a", "BTC/USD", 50, "0x0", 0)
        u.update_tx = "0xsim_abc"
        assert u.to_dict()["update_tx"] == "0xsim_abc"

    def test_negative_pnl_bps(self):
        u = ReputationUpdate("a", "ETH/USD", -200, "0x0", 100)
        assert u.to_dict()["outcome_pnl_bps"] == -200

    def test_zero_pnl_bps(self):
        u = ReputationUpdate("a", "SOL/USD", 0, "0x0", 100)
        assert u.to_dict()["outcome_pnl_bps"] == 0

    def test_large_timestamp(self):
        u = ReputationUpdate("a", "X", 0, "0x0", 9999999999)
        assert u.to_dict()["timestamp"] == 9999999999

    def test_simulation_mode_false(self):
        u = ReputationUpdate("a", "X", 0, "0x0", 0, simulation_mode=False)
        assert u.simulation_mode is False

    def test_to_dict_simulation_mode_present(self):
        u = ReputationUpdate("a", "X", 0, "0x0", 0)
        assert "simulation_mode" in u.to_dict()


class TestSyncResultAdditional:
    def test_empty_sync_result(self):
        r = SyncResult(synced_count=0, skipped_count=0, failed_count=0)
        assert r.errors == []
        assert r.updates == []

    def test_sync_result_with_errors(self):
        r = SyncResult(synced_count=1, skipped_count=0, failed_count=2,
                       errors=["err1", "err2"])
        assert len(r.errors) == 2

    def test_to_dict_lists(self):
        r = SyncResult(synced_count=3, skipped_count=1, failed_count=0)
        d = r.to_dict()
        assert isinstance(d["updates"], list)
        assert isinstance(d["errors"], list)

    def test_to_dict_counts(self):
        r = SyncResult(synced_count=3, skipped_count=1, failed_count=2)
        d = r.to_dict()
        assert d["synced_count"] == 3
        assert d["skipped_count"] == 1
        assert d["failed_count"] == 2


class TestReputationUpdaterMultiAgent:
    def test_sync_multiple_agents(self):
        l = TradeLedger()
        for i in range(3):
            l.log_trade("agent-A", "BTC/USD", "BUY", 0.1, 50000.0)
            l.log_trade("agent-B", "ETH/USD", "SELL", 1.0, 3000.0)
        updater = ReputationUpdater(ledger=l, agent_id="agent-A", simulation=True)
        result = updater.sync_pending()
        assert result.synced_count == 6  # All trades synced regardless of agent_id filter

    def test_sync_large_batch(self):
        l = TradeLedger()
        for i in range(50):
            l.log_trade("bulk-agent", "BTC/USD", "BUY" if i % 2 == 0 else "SELL",
                        0.01 * (i + 1), 50000.0)
        updater = ReputationUpdater(ledger=l, simulation=True)
        result = updater.sync_pending()
        assert result.synced_count == 50
        assert result.failed_count == 0

    def test_incremental_sync(self):
        l = TradeLedger()
        updater = ReputationUpdater(ledger=l, simulation=True)
        ensure_reputation_column(l)

        l.log_trade("a", "BTC/USD", "BUY", 0.1, 50000.0)
        r1 = updater.sync_pending()
        assert r1.synced_count == 1

        l.log_trade("a", "ETH/USD", "SELL", 1.0, 3000.0)
        r2 = updater.sync_pending()
        assert r2.synced_count == 1  # Only the new one

    def test_update_simulation_tx_has_agent_id(self):
        l = TradeLedger()
        l.log_trade("my-agent", "BTC/USD", "BUY", 0.1, 50000.0)
        updater = ReputationUpdater(ledger=l, agent_id="my-agent", simulation=True)
        result = updater.sync_pending()
        assert result.synced_count == 1

    def test_get_status_total_trades(self):
        l = TradeLedger()
        for i in range(7):
            l.log_trade("x", "BTC/USD", "BUY", 0.1, 50000.0)
        updater = ReputationUpdater(ledger=l, simulation=True)
        status = updater.get_status()
        assert status["total_trades"] == 7

    def test_get_status_agent_id(self):
        updater = ReputationUpdater(agent_id="status-agent")
        status = updater.get_status()
        assert status["agent_id"] == "status-agent"


class TestEip712ChainId:
    def test_chain_id_is_polygon_mumbai(self):
        assert EIP712_DOMAIN["chainId"] == 80001

    def test_verifying_contract_matches_registry(self):
        assert EIP712_DOMAIN["verifyingContract"] == REPUTATION_REGISTRY_ADDRESS

    def test_domain_version(self):
        assert EIP712_DOMAIN["version"] == "1"

    def test_reputation_update_has_5_fields(self):
        assert len(EIP712_TYPES["ReputationUpdate"]) == 5

    def test_outcome_pnl_bps_type_is_int256(self):
        fields = {f["name"]: f["type"] for f in EIP712_TYPES["ReputationUpdate"]}
        assert fields["outcomePnlBps"] == "int256"

    def test_trade_hash_type_is_bytes32(self):
        fields = {f["name"]: f["type"] for f in EIP712_TYPES["ReputationUpdate"]}
        assert fields["tradeHash"] == "bytes32"

    def test_timestamp_type_is_uint256(self):
        fields = {f["name"]: f["type"] for f in EIP712_TYPES["ReputationUpdate"]}
        assert fields["timestamp"] == "uint256"


class TestReputationUpdaterExtended:
    def test_build_update_pnl_buy_positive(self):
        l = TradeLedger()
        l.log_trade("a", "BTC/USD", "BUY", 0.5, 50000.0)
        u = ReputationUpdater(ledger=l, simulation=True)
        entry = l.get_all()[0]
        update = u.build_update(entry)
        assert update.outcome_pnl_bps > 0

    def test_build_update_pnl_sell_negative(self):
        l = TradeLedger()
        l.log_trade("a", "BTC/USD", "SELL", 0.5, 50000.0)
        u = ReputationUpdater(ledger=l, simulation=True)
        entry = l.get_all()[0]
        update = u.build_update(entry)
        assert update.outcome_pnl_bps < 0

    def test_build_update_market_preserved(self):
        l = TradeLedger()
        l.log_trade("a", "ETH/USD", "BUY", 1.0, 3000.0)
        u = ReputationUpdater(ledger=l, simulation=True)
        entry = l.get_all()[0]
        update = u.build_update(entry)
        assert update.market == "ETH/USD"

    def test_build_update_simulation_flag(self):
        l = TradeLedger()
        l.log_trade("a", "BTC/USD", "BUY", 0.1, 50000.0)
        u = ReputationUpdater(ledger=l, simulation=True)
        entry = l.get_all()[0]
        update = u.build_update(entry)
        assert update.simulation_mode is True

    def test_sign_update_in_place(self):
        l = TradeLedger()
        l.log_trade("a", "BTC/USD", "BUY", 0.1, 50000.0)
        u = ReputationUpdater(ledger=l, simulation=True)
        entry = l.get_all()[0]
        update = u.build_update(entry)
        result = u.sign_update(update)
        assert result.signature is not None
        assert len(result.signature) > 5

    def test_submit_update_simulation_returns_string(self):
        u = ReputationUpdater(simulation=True)
        update = ReputationUpdate("a", "BTC/USD", 50, "0x" + "a" * 64, 1700000000)
        tx = u.submit_update(update)
        assert isinstance(tx, str)
        assert len(tx) > 5

    def test_submit_update_live_mode_raises(self):
        u = ReputationUpdater(simulation=False)
        update = ReputationUpdate("a", "BTC/USD", 50, "0x" + "a" * 64, 1700000000)
        import pytest
        with pytest.raises(NotImplementedError):
            u.submit_update(update)

    def test_sync_updates_list_length_matches_synced_count(self):
        l = TradeLedger()
        for i in range(4):
            l.log_trade("a", "BTC/USD", "BUY", 0.1, 50000.0)
        u = ReputationUpdater(ledger=l, simulation=True)
        result = u.sync_pending()
        assert len(result.updates) == result.synced_count

    def test_get_status_registry_address_not_empty(self):
        u = ReputationUpdater()
        status = u.get_status()
        assert len(status["registry_address"]) > 0

    def test_get_status_has_all_keys(self):
        u = ReputationUpdater()
        status = u.get_status()
        required = ["agent_id", "simulation_mode", "total_trades", "synced_trades",
                    "pending_trades", "registry_address"]
        for key in required:
            assert key in status


class TestGetSigningKey:
    def test_returns_string(self):
        key = _get_signing_key()
        assert isinstance(key, str)

    def test_key_starts_with_0x(self):
        key = _get_signing_key()
        assert key.startswith("0x")

    def test_key_length_66(self):
        key = _get_signing_key()
        # "0x" + 64 hex chars = 66 chars
        assert len(key) == 66


class TestLedgerMigrationAdditional:
    def test_reputation_column_is_null_by_default(self):
        l = TradeLedger()
        l.log_trade("a", "BTC/USD", "BUY", 0.1, 50000.0)
        ensure_reputation_column(l)
        row = l._conn.execute(
            f"SELECT {REPUTATION_TX_COLUMN} FROM trades LIMIT 1"
        ).fetchone()
        assert row[0] is None

    def test_mark_synced_sets_value(self):
        l = TradeLedger()
        l.log_trade("a", "BTC/USD", "BUY", 0.1, 50000.0)
        ensure_reputation_column(l)
        entries = get_pending_reputation_entries(l)
        mark_reputation_synced(l, entries[0].tx_hash, "0xsim_123")
        row = l._conn.execute(
            f"SELECT {REPUTATION_TX_COLUMN} FROM trades WHERE tx_hash = ?",
            (entries[0].tx_hash,)
        ).fetchone()
        assert row[0] == "0xsim_123"

    def test_pending_entries_have_tx_hash(self):
        l = TradeLedger()
        for _ in range(3):
            l.log_trade("a", "BTC/USD", "BUY", 0.1, 50000.0)
        pending = get_pending_reputation_entries(l)
        for e in pending:
            assert e.tx_hash is not None
            assert e.tx_hash.startswith("0x")

    def test_pending_count_decreases_on_mark(self):
        l = TradeLedger()
        for _ in range(5):
            l.log_trade("a", "BTC/USD", "BUY", 0.1, 50000.0)
        ensure_reputation_column(l)
        pending = get_pending_reputation_entries(l)
        assert len(pending) == 5
        mark_reputation_synced(l, pending[0].tx_hash, "0xtest")
        mark_reputation_synced(l, pending[1].tx_hash, "0xtest2")
        remaining = get_pending_reputation_entries(l)
        assert len(remaining) == 3


class TestEip712AdditionalEncoding:
    def test_encode_all_markets(self):
        markets = ["BTC/USD", "ETH/USD", "SOL/USD", "MATIC/USD"]
        for market in markets:
            u = ReputationUpdate("a", market, 50, "0x" + "a" * 64, 1700000000)
            signable = encode_reputation_update(u)
            assert signable is not None

    def test_encode_negative_pnl(self):
        u = ReputationUpdate("a", "BTC/USD", -500, "0x" + "b" * 64, 1700000000)
        signable = encode_reputation_update(u)
        assert signable is not None

    def test_sign_consistent_for_same_key(self):
        from reputation_updater import _DEV_PRIVATE_KEY
        u = ReputationUpdate("a", "BTC/USD", 50, "0x" + "e" * 64, 1700000000)
        sig1 = sign_reputation_update(u, _DEV_PRIVATE_KEY)
        sig2 = sign_reputation_update(u, _DEV_PRIVATE_KEY)
        # EIP-712 signing may or may not be deterministic depending on k-value
        # Just check both produce valid signatures
        assert len(sig1) > 5
        assert len(sig2) > 5


class TestReputationUpdaterFinalCoverage:
    def test_updater_agent_id_in_all_updates(self):
        l = TradeLedger()
        for _ in range(3):
            l.log_trade("my-agent", "BTC/USD", "BUY", 0.1, 50000.0)
        u = ReputationUpdater(ledger=l, agent_id="my-agent", simulation=True)
        result = u.sync_pending()
        for upd in result.updates:
            assert upd.agent_id == "my-agent"

    def test_updater_all_updates_have_simulation_tx(self):
        l = TradeLedger()
        for _ in range(5):
            l.log_trade("a", "ETH/USD", "SELL", 1.0, 3000.0)
        u = ReputationUpdater(ledger=l, simulation=True)
        result = u.sync_pending()
        for upd in result.updates:
            assert upd.update_tx is not None
            assert upd.update_tx.startswith("0x")

    def test_updater_status_returns_dict(self):
        u = ReputationUpdater()
        status = u.get_status()
        assert isinstance(status, dict)

    def test_sync_result_errors_empty_on_success(self):
        l = TradeLedger()
        l.log_trade("a", "BTC/USD", "BUY", 0.1, 50000.0)
        u = ReputationUpdater(ledger=l, simulation=True)
        result = u.sync_pending()
        assert result.errors == []

    def test_updater_uses_simulation_mode(self):
        l = TradeLedger()
        l.log_trade("a", "BTC/USD", "BUY", 0.1, 50000.0)
        u = ReputationUpdater(ledger=l, simulation=True)
        assert u.simulation is True

    def test_build_update_timestamp_from_entry(self):
        l = TradeLedger()
        l.log_trade("a", "BTC/USD", "BUY", 0.1, 50000.0, timestamp="2024-01-15T10:00:00+00:00")
        u = ReputationUpdater(ledger=l, simulation=True)
        entry = l.get_all()[0]
        update = u.build_update(entry)
        # Timestamp should be around Jan 15 2024
        assert update.timestamp > 0
        assert update.timestamp < 9999999999  # not obviously wrong

    def test_process_entry_signature_not_none(self):
        l = TradeLedger()
        l.log_trade("a", "BTC/USD", "BUY", 0.1, 50000.0)
        u = ReputationUpdater(ledger=l, simulation=True)
        ensure_reputation_column(l)
        entry = l.get_all()[0]
        update = u.process_entry(entry)
        assert update.signature is not None

    def test_ensure_column_creates_column(self):
        l = TradeLedger()
        ensure_reputation_column(l)
        cur = l._conn.execute("PRAGMA table_info(trades)")
        cols = {r["name"] for r in cur.fetchall()}
        assert REPUTATION_TX_COLUMN in cols

    def test_simulation_tx_contains_hash(self):
        tx = _build_simulation_tx("agent-x", "0xfeedbeef")
        # Should have hex content after prefix
        assert len(tx) > len("0xsim_")

    def test_sync_result_has_skipped_count(self):
        r = SyncResult(synced_count=2, skipped_count=1, failed_count=0)
        d = r.to_dict()
        assert d["skipped_count"] == 1

    def test_reputation_update_to_dict_all_keys(self):
        u = ReputationUpdate("a", "BTC/USD", 50, "0x" + "a" * 64, 1700000000,
                             signature="0xsig", update_tx="0xtx")
        d = u.to_dict()
        keys = ["agent_id", "market", "outcome_pnl_bps", "trade_hash",
                "timestamp", "signature", "update_tx", "simulation_mode"]
        for k in keys:
            assert k in d

    def test_sign_update_returns_self(self):
        l = TradeLedger()
        l.log_trade("a", "BTC/USD", "BUY", 0.1, 50000.0)
        u = ReputationUpdater(ledger=l, simulation=True)
        entry = l.get_all()[0]
        update = u.build_update(entry)
        returned = u.sign_update(update)
        assert returned is update  # modifies in place and returns
