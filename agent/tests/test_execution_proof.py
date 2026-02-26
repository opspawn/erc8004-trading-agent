"""
test_execution_proof.py — Unit tests for ExecutionProof on-chain trade recording.

All tests use mock mode (HEDERA_TESTNET_MODE=mock) — no network calls.
Target: ~80 tests covering record, bundle generation, verification, and error cases.
"""

from __future__ import annotations

import pytest
import uuid

from execution_proof import (
    ExecutionProof,
    TradeExecution,
    ProofBundle,
    VerificationResult,
    ProofRecordError,
    ProofVerificationError,
)


# ─── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture
def proof():
    ep = ExecutionProof(agent_id="test-agent", mode="mock")
    return ep


def make_trade(
    trade_id: str = None,
    ticker: str = "BTC",
    side: str = "BUY",
    qty: float = 0.1,
    price: float = 50000.0,
    agent_id: str = "test-agent",
) -> TradeExecution:
    return TradeExecution(
        trade_id=trade_id or str(uuid.uuid4()),
        ticker=ticker,
        side=side,
        qty=qty,
        price=price,
        agent_id=agent_id,
    )


# ─── TradeExecution Model ─────────────────────────────────────────────────────

class TestTradeExecution:
    def test_to_dict(self):
        t = make_trade(trade_id="t1")
        d = t.to_dict()
        assert d["trade_id"] == "t1"
        assert d["ticker"] == "BTC"
        assert d["side"] == "BUY"

    def test_to_json_roundtrip(self):
        t = make_trade(trade_id="t2")
        j = t.to_json()
        restored = TradeExecution.from_dict(__import__("json").loads(j))
        assert restored.trade_id == "t2"
        assert restored.ticker == "BTC"

    def test_from_dict(self):
        d = {
            "trade_id": "x1",
            "ticker": "ETH",
            "side": "SELL",
            "qty": 1.0,
            "price": 3000.0,
            "agent_id": "agent-a",
        }
        t = TradeExecution.from_dict(d)
        assert t.trade_id == "x1"
        assert t.side == "SELL"

    def test_has_timestamp(self):
        t = make_trade()
        assert t.timestamp
        assert "T" in t.timestamp

    def test_hcs_message_id_initially_none(self):
        t = make_trade()
        assert t.hcs_message_id is None


# ─── ExecutionProof Init ──────────────────────────────────────────────────────

class TestExecutionProofInit:
    def test_default_mode_mock(self):
        ep = ExecutionProof()
        assert ep.mode == "mock"

    def test_custom_agent_id(self):
        ep = ExecutionProof(agent_id="my-agent")
        assert ep.agent_id == "my-agent"

    def test_initial_count(self, proof):
        assert proof.get_execution_count() == 0

    def test_initial_all_trades_empty(self, proof):
        assert proof.get_all_trades() == []


# ─── Record Execution ─────────────────────────────────────────────────────────

class TestRecordExecution:
    def test_record_basic(self, proof):
        t = make_trade(trade_id="t1")
        result = proof.record_execution(t)
        assert result.hcs_message_id is not None

    def test_record_sets_message_id(self, proof):
        t = make_trade(trade_id="t1")
        result = proof.record_execution(t)
        assert "mock-hcs" in result.hcs_message_id

    def test_record_increments_count(self, proof):
        proof.record_execution(make_trade(trade_id="t1"))
        assert proof.get_execution_count() == 1

    def test_record_multiple(self, proof):
        for i in range(5):
            proof.record_execution(make_trade(trade_id=f"t{i}"))
        assert proof.get_execution_count() == 5

    def test_record_empty_trade_id_raises(self, proof):
        t = TradeExecution(trade_id="", ticker="BTC", side="BUY", qty=0.1, price=50000.0, agent_id="a")
        with pytest.raises(ProofRecordError, match="trade_id"):
            proof.record_execution(t)

    def test_record_zero_qty_raises(self, proof):
        t = make_trade(trade_id="t1", qty=0.0)
        with pytest.raises(ProofRecordError, match="qty"):
            proof.record_execution(t)

    def test_record_negative_qty_raises(self, proof):
        t = make_trade(trade_id="t1", qty=-1.0)
        with pytest.raises(ProofRecordError, match="qty"):
            proof.record_execution(t)

    def test_record_zero_price_raises(self, proof):
        t = make_trade(trade_id="t1", price=0.0)
        with pytest.raises(ProofRecordError, match="price"):
            proof.record_execution(t)

    def test_record_negative_price_raises(self, proof):
        t = make_trade(trade_id="t1", price=-100.0)
        with pytest.raises(ProofRecordError, match="price"):
            proof.record_execution(t)

    def test_record_invalid_side_raises(self, proof):
        t = make_trade(trade_id="t1", side="LONG")
        with pytest.raises(ProofRecordError, match="side"):
            proof.record_execution(t)

    def test_record_sell_valid(self, proof):
        t = make_trade(trade_id="t1", side="SELL")
        result = proof.record_execution(t)
        assert result.hcs_message_id is not None

    def test_record_network_failure(self, proof):
        proof.set_network_failure(True)
        t = make_trade(trade_id="t1")
        with pytest.raises(ProofRecordError, match="network"):
            proof.record_execution(t)

    def test_record_stores_trade(self, proof):
        t = make_trade(trade_id="t99")
        proof.record_execution(t)
        stored = proof.get_recorded_trade("t99")
        assert stored is not None
        assert stored.trade_id == "t99"

    def test_get_unknown_trade_returns_none(self, proof):
        result = proof.get_recorded_trade("nonexistent")
        assert result is None

    def test_get_all_trades_order(self, proof):
        for i in range(3):
            proof.record_execution(make_trade(trade_id=f"trade-{i}"))
        all_trades = proof.get_all_trades()
        ids = [t.trade_id for t in all_trades]
        assert ids == ["trade-0", "trade-1", "trade-2"]

    def test_message_ids_unique_per_trade(self, proof):
        t1 = make_trade(trade_id="t1")
        t2 = make_trade(trade_id="t2")
        proof.record_execution(t1)
        proof.record_execution(t2)
        assert t1.hcs_message_id != t2.hcs_message_id

    def test_message_id_contains_sequence(self, proof):
        t1 = make_trade(trade_id="t1")
        t2 = make_trade(trade_id="t2")
        proof.record_execution(t1)
        proof.record_execution(t2)
        # seq-0 for first, seq-1 for second
        assert "seq-000000" in t1.hcs_message_id
        assert "seq-000001" in t2.hcs_message_id


# ─── Generate Proof Bundle ────────────────────────────────────────────────────

class TestGenerateProofBundle:
    def test_generate_single_trade(self, proof):
        t = make_trade(trade_id="t1")
        proof.record_execution(t)
        bundle = proof.generate_proof_bundle(["t1"])
        assert bundle.bundle_id
        assert "t1" in bundle.trade_ids

    def test_generate_multiple_trades(self, proof):
        for i in range(3):
            proof.record_execution(make_trade(trade_id=f"t{i}"))
        bundle = proof.generate_proof_bundle(["t0", "t1", "t2"])
        assert len(bundle.trade_ids) == 3
        assert len(bundle.message_ids) == 3

    def test_generate_sets_hash(self, proof):
        proof.record_execution(make_trade(trade_id="t1"))
        bundle = proof.generate_proof_bundle(["t1"])
        assert bundle.bundle_hash
        assert len(bundle.bundle_hash) == 64  # SHA-256 hex

    def test_generate_empty_trade_ids_raises(self, proof):
        with pytest.raises(ProofRecordError, match="empty"):
            proof.generate_proof_bundle([])

    def test_generate_unknown_trade_id_raises(self, proof):
        with pytest.raises(ProofRecordError, match="not found"):
            proof.generate_proof_bundle(["unknown-id"])

    def test_generate_bundle_agent_id(self, proof):
        proof.record_execution(make_trade(trade_id="t1"))
        bundle = proof.generate_proof_bundle(["t1"])
        assert bundle.agent_id == proof.agent_id

    def test_generate_bundle_preserves_order(self, proof):
        for i in range(4):
            proof.record_execution(make_trade(trade_id=f"t{i}"))
        bundle = proof.generate_proof_bundle(["t3", "t1", "t0"])
        assert bundle.trade_ids == ["t3", "t1", "t0"]

    def test_generate_bundle_has_created_at(self, proof):
        proof.record_execution(make_trade(trade_id="t1"))
        bundle = proof.generate_proof_bundle(["t1"])
        assert bundle.created_at
        assert "T" in bundle.created_at

    def test_bundle_hash_deterministic(self, proof):
        proof.record_execution(make_trade(trade_id="t1"))
        bundle = proof.generate_proof_bundle(["t1"])
        h1 = bundle.compute_hash()
        h2 = bundle.compute_hash()
        assert h1 == h2

    def test_bundle_different_trade_sets_different_hash(self, proof):
        proof.record_execution(make_trade(trade_id="t1"))
        proof.record_execution(make_trade(trade_id="t2"))
        b1 = proof.generate_proof_bundle(["t1"])
        b2 = proof.generate_proof_bundle(["t2"])
        assert b1.bundle_hash != b2.bundle_hash


# ─── Verify Bundle ────────────────────────────────────────────────────────────

class TestVerifyBundle:
    def test_verify_valid_bundle(self, proof):
        proof.record_execution(make_trade(trade_id="t1"))
        bundle = proof.generate_proof_bundle(["t1"])
        result = proof.verify_bundle(bundle)
        assert result.valid
        assert result.errors == []

    def test_verify_multi_trade_bundle(self, proof):
        for i in range(3):
            proof.record_execution(make_trade(trade_id=f"t{i}"))
        bundle = proof.generate_proof_bundle(["t0", "t1", "t2"])
        result = proof.verify_bundle(bundle)
        assert result.valid

    def test_verify_hash_mismatch(self, proof):
        proof.record_execution(make_trade(trade_id="t1"))
        bundle = proof.generate_proof_bundle(["t1"])
        bundle.bundle_hash = "deadbeef" * 8  # tamper with hash
        result = proof.verify_bundle(bundle)
        assert not result.valid
        assert any("Hash mismatch" in e for e in result.errors)

    def test_verify_unknown_trade_ids(self, proof):
        # Bundle with unknown trade IDs (not in proof's records)
        bundle = ProofBundle(
            bundle_id=str(uuid.uuid4()),
            agent_id="test-agent",
            trade_ids=["unknown-1"],
            message_ids=["mock-msg-1"],
        )
        bundle.bundle_hash = bundle.compute_hash()
        result = proof.verify_bundle(bundle)
        assert not result.valid
        assert any("Unknown" in e for e in result.errors)

    def test_verify_length_mismatch(self, proof):
        proof.record_execution(make_trade(trade_id="t1"))
        bundle = proof.generate_proof_bundle(["t1"])
        bundle.message_ids.append("extra-msg")  # extra message ID
        result = proof.verify_bundle(bundle)
        assert not result.valid
        assert any("count" in e.lower() for e in result.errors)

    def test_verify_message_id_mismatch(self, proof):
        proof.record_execution(make_trade(trade_id="t1"))
        bundle = proof.generate_proof_bundle(["t1"])
        original_msg = bundle.message_ids[0]
        bundle.message_ids[0] = "wrong-message-id"
        # Recompute hash with tampered data so hash check passes
        bundle.bundle_hash = bundle.compute_hash()
        result = proof.verify_bundle(bundle)
        assert not result.valid
        assert any("mismatch" in e.lower() for e in result.errors)

    def test_verify_returns_trade_count(self, proof):
        for i in range(3):
            proof.record_execution(make_trade(trade_id=f"t{i}"))
        bundle = proof.generate_proof_bundle(["t0", "t1", "t2"])
        result = proof.verify_bundle(bundle)
        assert result.trade_count == 3

    def test_verify_returns_bundle_id(self, proof):
        proof.record_execution(make_trade(trade_id="t1"))
        bundle = proof.generate_proof_bundle(["t1"])
        result = proof.verify_bundle(bundle)
        assert result.bundle_id == bundle.bundle_id

    def test_verify_result_has_timestamp(self, proof):
        proof.record_execution(make_trade(trade_id="t1"))
        bundle = proof.generate_proof_bundle(["t1"])
        result = proof.verify_bundle(bundle)
        assert result.verified_at

    def test_verify_valid_result_to_dict(self, proof):
        proof.record_execution(make_trade(trade_id="t1"))
        bundle = proof.generate_proof_bundle(["t1"])
        result = proof.verify_bundle(bundle)
        d = result.to_dict()
        assert d["valid"] is True
        assert d["trade_count"] == 1


# ─── Reset ────────────────────────────────────────────────────────────────────

class TestReset:
    def test_reset_clears_executions(self, proof):
        proof.record_execution(make_trade(trade_id="t1"))
        proof.reset()
        assert proof.get_execution_count() == 0

    def test_reset_clears_all_trades(self, proof):
        proof.record_execution(make_trade(trade_id="t1"))
        proof.reset()
        assert proof.get_all_trades() == []

    def test_reset_allows_reuse(self, proof):
        proof.record_execution(make_trade(trade_id="t1"))
        proof.reset()
        proof.record_execution(make_trade(trade_id="t2"))
        assert proof.get_execution_count() == 1

    def test_reset_sequence_restarts(self, proof):
        proof.record_execution(make_trade(trade_id="t1"))
        proof.reset()
        t2 = make_trade(trade_id="t2")
        proof.record_execution(t2)
        assert "seq-000000" in t2.hcs_message_id


# ─── Real Mode Guard ──────────────────────────────────────────────────────────

class TestRealModeGuard:
    def test_real_mode_record_raises(self):
        ep = ExecutionProof(mode="real")
        t = make_trade(trade_id="t1")
        with pytest.raises(NotImplementedError):
            ep.record_execution(t)


# ─── ProofBundle Model ────────────────────────────────────────────────────────

class TestProofBundleModel:
    def test_to_dict(self):
        b = ProofBundle(
            bundle_id="b1",
            agent_id="agent",
            trade_ids=["t1"],
            message_ids=["m1"],
        )
        d = b.to_dict()
        assert d["bundle_id"] == "b1"
        assert d["trade_ids"] == ["t1"]

    def test_from_dict(self):
        d = {
            "bundle_id": "b2",
            "agent_id": "agent",
            "trade_ids": ["t1", "t2"],
            "message_ids": ["m1", "m2"],
            "bundle_hash": "",
        }
        b = ProofBundle.from_dict(d)
        assert b.bundle_id == "b2"
        assert len(b.trade_ids) == 2

    def test_compute_hash_changes_with_data(self):
        b1 = ProofBundle(
            bundle_id="same", agent_id="a",
            trade_ids=["t1"], message_ids=["m1"]
        )
        b2 = ProofBundle(
            bundle_id="same", agent_id="a",
            trade_ids=["t2"], message_ids=["m2"]
        )
        assert b1.compute_hash() != b2.compute_hash()
