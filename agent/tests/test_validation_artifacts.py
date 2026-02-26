"""
Tests for validation_artifacts.py — ERC-8004 Validation Artifact Generator.
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
import uuid

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from trade_ledger import TradeLedger
from validation_artifacts import (
    ArtifactGenerator,
    ValidationArtifact,
    keccak256_config,
    canonical_artifact_hash,
    _compute_metrics,
    _sign_artifact_hash,
    _get_signing_key,
    DEFAULT_AGENT_ID,
    DEFAULT_STRATEGY_CONFIG,
    ARTIFACTS_DIR,
)


# ─── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture
def ledger():
    l = TradeLedger()
    for i in range(10):
        l.log_trade(
            agent_id="test-agent",
            market="BTC/USD",
            side="BUY" if i % 2 == 0 else "SELL",
            size=0.1,
            price=50000.0 + i * 50,
        )
    return l


@pytest.fixture
def empty_ledger():
    return TradeLedger()


@pytest.fixture
def tmp_dir():
    with tempfile.TemporaryDirectory() as d:
        yield d


@pytest.fixture
def generator(ledger, tmp_dir):
    return ArtifactGenerator(
        ledger=ledger,
        agent_id="test-agent",
        artifacts_dir=tmp_dir,
    )


# ─── keccak256_config Tests ────────────────────────────────────────────────────


class TestKeccak256Config:
    def test_returns_0x_string(self):
        h = keccak256_config({"key": "value"})
        assert h.startswith("0x")

    def test_length_is_66(self):
        h = keccak256_config({"key": "value"})
        assert len(h) == 66  # "0x" + 64 hex chars

    def test_deterministic(self):
        cfg = {"type": "momentum", "version": "1.0"}
        h1 = keccak256_config(cfg)
        h2 = keccak256_config(cfg)
        assert h1 == h2

    def test_different_config_different_hash(self):
        h1 = keccak256_config({"type": "momentum"})
        h2 = keccak256_config({"type": "mean_reversion"})
        assert h1 != h2

    def test_empty_config(self):
        h = keccak256_config({})
        assert h.startswith("0x")
        assert len(h) == 66


# ─── canonical_artifact_hash Tests ────────────────────────────────────────────


class TestCanonicalArtifactHash:
    def test_returns_0x_string(self):
        d = {
            "agent_id": "a",
            "session_id": "s",
            "strategy_hash": "0x" + "a" * 64,
            "trades_count": 5,
            "win_rate": 0.6,
            "avg_pnl_bps": 10.0,
            "max_drawdown_bps": 50.0,
            "risk_violations": 0,
            "validation_timestamp": "2024-01-01T00:00:00+00:00",
            "artifact_hash": "0x" + "b" * 64,
            "validator_signature": "0x" + "c" * 130,
        }
        h = canonical_artifact_hash(d)
        assert h.startswith("0x")

    def test_excludes_signature_from_hash(self):
        base = {
            "agent_id": "a", "session_id": "s",
            "strategy_hash": "0x" + "a" * 64,
            "trades_count": 5, "win_rate": 0.6,
            "avg_pnl_bps": 10.0, "max_drawdown_bps": 50.0,
            "risk_violations": 0,
            "validation_timestamp": "2024-01-01T00:00:00+00:00",
            "artifact_hash": "0x1234",
        }
        d1 = {**base, "validator_signature": "0xaaa"}
        d2 = {**base, "validator_signature": "0xbbb"}
        assert canonical_artifact_hash(d1) == canonical_artifact_hash(d2)


# ─── _compute_metrics Tests ────────────────────────────────────────────────────


class TestComputeMetrics:
    def test_empty_returns_zeros(self):
        m = _compute_metrics([])
        assert m["trades_count"] == 0
        assert m["win_rate"] == 0.0
        assert m["avg_pnl_bps"] == 0.0

    def test_all_buys_win_rate_1(self, ledger):
        entries = ledger.get_entries(side="BUY")
        m = _compute_metrics(entries)
        assert m["win_rate"] == 1.0

    def test_all_sells_win_rate_0(self, ledger):
        entries = ledger.get_entries(side="SELL")
        m = _compute_metrics(entries)
        assert m["win_rate"] == 0.0

    def test_mixed_win_rate(self, ledger):
        entries = ledger.get_all()
        m = _compute_metrics(entries)
        assert 0.0 < m["win_rate"] < 1.0

    def test_trades_count_correct(self, ledger):
        entries = ledger.get_all()
        m = _compute_metrics(entries)
        assert m["trades_count"] == len(entries)

    def test_max_drawdown_non_negative(self, ledger):
        entries = ledger.get_all()
        m = _compute_metrics(entries)
        assert m["max_drawdown_bps"] >= 0.0

    def test_risk_violations_non_negative(self, ledger):
        entries = ledger.get_all()
        m = _compute_metrics(entries)
        assert m["risk_violations"] >= 0


# ─── ValidationArtifact Tests ──────────────────────────────────────────────────


class TestValidationArtifact:
    def test_to_dict_has_all_fields(self):
        a = ValidationArtifact(
            agent_id="a", session_id="s", strategy_hash="0x" + "a" * 64,
            trades_count=5, win_rate=0.6, avg_pnl_bps=10.0,
            max_drawdown_bps=50.0, risk_violations=0,
            validation_timestamp="2024-01-01T00:00:00+00:00",
            artifact_hash="0x" + "b" * 64,
            validator_signature="0x" + "c" * 130,
        )
        d = a.to_dict()
        required = [
            "agent_id", "session_id", "strategy_hash", "trades_count",
            "win_rate", "avg_pnl_bps", "max_drawdown_bps", "risk_violations",
            "validation_timestamp", "artifact_hash", "validator_signature",
        ]
        for field in required:
            assert field in d, f"Missing field: {field}"

    def test_to_json_is_valid_json(self):
        a = ValidationArtifact(
            agent_id="a", session_id="s", strategy_hash="0x" + "a" * 64,
            trades_count=5, win_rate=0.6, avg_pnl_bps=10.0,
            max_drawdown_bps=50.0, risk_violations=0,
            validation_timestamp="2024-01-01",
            artifact_hash="0x1234",
            validator_signature="0x5678",
        )
        parsed = json.loads(a.to_json())
        assert parsed["agent_id"] == "a"

    def test_from_dict_roundtrip(self):
        a = ValidationArtifact(
            agent_id="a", session_id="s", strategy_hash="0x" + "a" * 64,
            trades_count=5, win_rate=0.6, avg_pnl_bps=10.0,
            max_drawdown_bps=50.0, risk_violations=0,
            validation_timestamp="2024-01-01",
            artifact_hash="0x1234",
            validator_signature="0x5678",
        )
        d = a.to_dict()
        a2 = ValidationArtifact.from_dict(d)
        assert a2.agent_id == a.agent_id
        assert a2.session_id == a.session_id
        assert a2.win_rate == a.win_rate


# ─── ArtifactGenerator Tests ───────────────────────────────────────────────────


class TestArtifactGeneratorInit:
    def test_default_init(self):
        gen = ArtifactGenerator()
        assert gen.agent_id == DEFAULT_AGENT_ID
        assert gen.strategy_config is not None

    def test_custom_agent_id(self, tmp_dir):
        gen = ArtifactGenerator(agent_id="custom", artifacts_dir=tmp_dir)
        assert gen.agent_id == "custom"

    def test_custom_strategy_config(self, tmp_dir):
        cfg = {"type": "custom"}
        gen = ArtifactGenerator(strategy_config=cfg, artifacts_dir=tmp_dir)
        assert gen.strategy_config == cfg


class TestArtifactGeneratorGenerate:
    def test_generate_returns_artifact(self, generator):
        a = generator.generate()
        assert isinstance(a, ValidationArtifact)

    def test_generate_with_session_id(self, generator):
        sid = "test-session-123"
        a = generator.generate(session_id=sid)
        assert a.session_id == sid

    def test_generate_auto_session_id(self, generator):
        a = generator.generate()
        assert a.session_id is not None
        assert len(a.session_id) > 0

    def test_generate_agent_id(self, generator):
        a = generator.generate()
        assert a.agent_id == "test-agent"

    def test_generate_strategy_hash_0x(self, generator):
        a = generator.generate()
        assert a.strategy_hash.startswith("0x")

    def test_generate_trades_count(self, generator, ledger):
        a = generator.generate()
        assert a.trades_count == ledger.count()

    def test_generate_win_rate_in_range(self, generator):
        a = generator.generate()
        assert 0.0 <= a.win_rate <= 1.0

    def test_generate_has_signature(self, generator):
        a = generator.generate()
        assert a.validator_signature is not None
        assert len(a.validator_signature) > 5

    def test_generate_has_artifact_hash(self, generator):
        a = generator.generate()
        assert a.artifact_hash.startswith("0x")

    def test_generate_has_timestamp(self, generator):
        a = generator.generate()
        assert a.validation_timestamp is not None
        assert len(a.validation_timestamp) > 5

    def test_generate_saved_to_disk(self, generator, tmp_dir):
        a = generator.generate(session_id="disk-test")
        import os
        assert os.path.exists(a.artifact_path)

    def test_generate_artifact_json_valid(self, generator, tmp_dir):
        a = generator.generate(session_id="json-test")
        with open(a.artifact_path) as f:
            data = json.load(f)
        assert data["session_id"] == "json-test"

    def test_generate_empty_ledger(self, empty_ledger, tmp_dir):
        gen = ArtifactGenerator(ledger=empty_ledger, artifacts_dir=tmp_dir)
        a = gen.generate(session_id="empty-test")
        assert a.trades_count == 0
        assert a.win_rate == 0.0

    def test_generate_max_drawdown_non_negative(self, generator):
        a = generator.generate()
        assert a.max_drawdown_bps >= 0.0

    def test_generate_risk_violations_non_negative(self, generator):
        a = generator.generate()
        assert a.risk_violations >= 0


class TestArtifactGeneratorLoadList:
    def test_list_empty_dir(self, empty_ledger, tmp_dir):
        gen = ArtifactGenerator(ledger=empty_ledger, artifacts_dir=tmp_dir)
        assert gen.list_artifacts() == []

    def test_list_after_generate(self, generator, tmp_dir):
        generator.generate(session_id="sess-1")
        generator.generate(session_id="sess-2")
        artifacts = generator.list_artifacts()
        assert "sess-1" in artifacts
        assert "sess-2" in artifacts

    def test_load_generated_artifact(self, generator, tmp_dir):
        sid = "load-test-session"
        a1 = generator.generate(session_id=sid)
        a2 = generator.load(sid)
        assert a2.session_id == sid
        assert a2.agent_id == a1.agent_id


class TestArtifactGeneratorVerify:
    def test_verify_valid_artifact(self, generator):
        a = generator.generate(session_id="verify-test")
        result = generator.verify(a)
        assert result is True

    def test_verify_tampered_artifact(self, generator):
        a = generator.generate(session_id="tamper-test")
        # Tamper with trades_count
        original = a.trades_count
        a.trades_count = original + 999
        # Re-compute hash to simulate verification check
        result = generator.verify(a)
        # Hash should no longer match (tampered)
        assert result is False


# ─── Additional Coverage Tests ────────────────────────────────────────────────


class TestKeccak256ConfigAdditional:
    def test_nested_config(self):
        cfg = {"outer": {"inner": "value"}, "list": [1, 2, 3]}
        h = keccak256_config(cfg)
        assert h.startswith("0x")
        assert len(h) == 66

    def test_numeric_values(self):
        h = keccak256_config({"num": 3.14, "int": 42})
        assert h.startswith("0x")

    def test_same_keys_different_values(self):
        h1 = keccak256_config({"key": "value1"})
        h2 = keccak256_config({"key": "value2"})
        assert h1 != h2

    def test_different_key_order_same_hash(self):
        # sort_keys=True ensures order independence
        h1 = keccak256_config({"a": 1, "b": 2})
        h2 = keccak256_config({"b": 2, "a": 1})
        assert h1 == h2


class TestComputeMetricsAdditional:
    def test_single_buy_trade(self):
        l = TradeLedger()
        l.log_trade("a", "BTC/USD", "BUY", 0.1, 50000.0)
        entries = l.get_all()
        m = _compute_metrics(entries)
        assert m["trades_count"] == 1
        assert m["win_rate"] == 1.0
        assert m["avg_pnl_bps"] == 50.0

    def test_single_sell_trade(self):
        l = TradeLedger()
        l.log_trade("a", "BTC/USD", "SELL", 0.1, 50000.0)
        entries = l.get_all()
        m = _compute_metrics(entries)
        assert m["trades_count"] == 1
        assert m["win_rate"] == 0.0
        assert m["avg_pnl_bps"] == -50.0

    def test_drawdown_all_losses(self):
        l = TradeLedger()
        for _ in range(5):
            l.log_trade("a", "BTC/USD", "SELL", 0.1, 50000.0)
        entries = l.get_all()
        m = _compute_metrics(entries)
        assert m["max_drawdown_bps"] > 0

    def test_drawdown_all_wins(self):
        l = TradeLedger()
        for _ in range(5):
            l.log_trade("a", "BTC/USD", "BUY", 0.1, 50000.0)
        entries = l.get_all()
        m = _compute_metrics(entries)
        # No drawdown when all trades are winners
        assert m["max_drawdown_bps"] == 0.0

    def test_risk_violations_normally_zero(self):
        l = TradeLedger()
        l.log_trade("a", "BTC/USD", "BUY", 0.1, 50000.0)
        m = _compute_metrics(l.get_all())
        # 50 bps is below 200 bps violation threshold
        assert m["risk_violations"] == 0


class TestValidationArtifactAdditional:
    def test_win_rate_precision(self):
        a = ValidationArtifact(
            agent_id="a", session_id="s", strategy_hash="0x" + "a" * 64,
            trades_count=3, win_rate=0.6666666666, avg_pnl_bps=10.0,
            max_drawdown_bps=50.0, risk_violations=0,
            validation_timestamp="2024-01-01", artifact_hash="0x1234",
            validator_signature="0x5678",
        )
        d = a.to_dict()
        # win_rate should be rounded to 6 decimal places
        assert d["win_rate"] == round(0.6666666666, 6)

    def test_avg_pnl_bps_precision(self):
        a = ValidationArtifact(
            agent_id="a", session_id="s", strategy_hash="0x" + "a" * 64,
            trades_count=1, win_rate=1.0, avg_pnl_bps=33.333333333,
            max_drawdown_bps=0.0, risk_violations=0,
            validation_timestamp="2024-01-01", artifact_hash="0x1234",
            validator_signature="0x5678",
        )
        d = a.to_dict()
        assert d["avg_pnl_bps"] == round(33.333333333, 4)

    def test_max_drawdown_precision(self):
        a = ValidationArtifact(
            agent_id="a", session_id="s", strategy_hash="0x" + "a" * 64,
            trades_count=1, win_rate=0.5, avg_pnl_bps=0.0,
            max_drawdown_bps=12.345678, risk_violations=0,
            validation_timestamp="2024-01-01", artifact_hash="0x1234",
            validator_signature="0x5678",
        )
        d = a.to_dict()
        assert d["max_drawdown_bps"] == round(12.345678, 4)

    def test_from_dict_preserves_risk_violations(self):
        d = {
            "agent_id": "a", "session_id": "s",
            "strategy_hash": "0x" + "a" * 64,
            "trades_count": 10, "win_rate": 0.5, "avg_pnl_bps": 5.0,
            "max_drawdown_bps": 20.0, "risk_violations": 3,
            "validation_timestamp": "2024-01-01",
            "artifact_hash": "0x1234", "validator_signature": "0x5678",
        }
        a = ValidationArtifact.from_dict(d)
        assert a.risk_violations == 3


class TestArtifactGeneratorAdditional:
    def test_generate_multiple_sessions(self, generator, tmp_dir):
        for i in range(3):
            generator.generate(session_id=f"multi-{i}")
        artifacts = generator.list_artifacts()
        assert len(artifacts) == 3

    def test_generate_same_session_overwrites(self, generator, tmp_dir):
        generator.generate(session_id="overwrite-test")
        generator.generate(session_id="overwrite-test")
        artifacts = generator.list_artifacts()
        assert artifacts.count("overwrite-test") == 1

    def test_strategy_hash_consistent(self, generator):
        a1 = generator.generate(session_id="hash-test-1")
        a2 = generator.generate(session_id="hash-test-2")
        assert a1.strategy_hash == a2.strategy_hash

    def test_different_strategy_different_hash(self, ledger, tmp_dir):
        gen1 = ArtifactGenerator(ledger=ledger, strategy_config={"type": "A"},
                                 artifacts_dir=tmp_dir)
        gen2 = ArtifactGenerator(ledger=ledger, strategy_config={"type": "B"},
                                 artifacts_dir=tmp_dir)
        a1 = gen1.generate(session_id="strat-a")
        a2 = gen2.generate(session_id="strat-b")
        assert a1.strategy_hash != a2.strategy_hash

    def test_artifact_hash_changes_on_different_sessions(self, generator):
        # Different session_ids → different session_id in content → different hash
        a1 = generator.generate(session_id="unique-111")
        a2 = generator.generate(session_id="unique-222")
        assert a1.artifact_hash != a2.artifact_hash

    def test_list_artifacts_sorted(self, generator, tmp_dir):
        generator.generate(session_id="zzz-session")
        generator.generate(session_id="aaa-session")
        artifacts = generator.list_artifacts()
        # Should be sorted alphabetically (pathlib glob sorts by name)
        assert artifacts.index("aaa-session") < artifacts.index("zzz-session")

    def test_generate_with_custom_config(self, ledger, tmp_dir):
        cfg = {"type": "custom_strat", "version": "2.0"}
        gen = ArtifactGenerator(ledger=ledger, strategy_config=cfg,
                                artifacts_dir=tmp_dir)
        a = gen.generate(session_id="custom-cfg-test")
        assert a.strategy_hash.startswith("0x")

    def test_verify_freshly_generated(self, generator):
        a = generator.generate(session_id="fresh-verify")
        assert generator.verify(a) is True

    def test_to_json_has_validator_signature(self, generator):
        a = generator.generate(session_id="sig-json-test")
        parsed = json.loads(a.to_json())
        assert "validator_signature" in parsed
        assert len(parsed["validator_signature"]) > 5


class TestArtifactHashIntegrity:
    def test_canonical_hash_excludes_validator_signature(self):
        base = {
            "agent_id": "a", "session_id": "s",
            "strategy_hash": "0x" + "a" * 64,
            "trades_count": 5, "win_rate": 0.5, "avg_pnl_bps": 0.0,
            "max_drawdown_bps": 0.0, "risk_violations": 0,
            "validation_timestamp": "2024-01-01",
        }
        d1 = {**base, "validator_signature": "0xaaa"}
        d2 = {**base, "validator_signature": "0xbbb"}
        assert canonical_artifact_hash(d1) == canonical_artifact_hash(d2)

    def test_canonical_hash_excludes_artifact_hash(self):
        base = {
            "agent_id": "a", "session_id": "s",
            "strategy_hash": "0x" + "a" * 64,
            "trades_count": 5, "win_rate": 0.5, "avg_pnl_bps": 0.0,
            "max_drawdown_bps": 0.0, "risk_violations": 0,
            "validation_timestamp": "2024-01-01",
        }
        d1 = {**base, "artifact_hash": "0xold_hash"}
        d2 = {**base, "artifact_hash": "0xnew_hash"}
        assert canonical_artifact_hash(d1) == canonical_artifact_hash(d2)

    def test_canonical_hash_sensitive_to_agent_id(self):
        base = {
            "session_id": "s", "strategy_hash": "0x" + "a" * 64,
            "trades_count": 5, "win_rate": 0.5, "avg_pnl_bps": 0.0,
            "max_drawdown_bps": 0.0, "risk_violations": 0,
            "validation_timestamp": "2024-01-01",
        }
        h1 = canonical_artifact_hash({**base, "agent_id": "agent-1"})
        h2 = canonical_artifact_hash({**base, "agent_id": "agent-2"})
        assert h1 != h2

    def test_canonical_hash_sensitive_to_trades_count(self):
        base = {
            "agent_id": "a", "session_id": "s",
            "strategy_hash": "0x" + "a" * 64, "win_rate": 0.5,
            "avg_pnl_bps": 0.0, "max_drawdown_bps": 0.0,
            "risk_violations": 0, "validation_timestamp": "2024-01-01",
        }
        h1 = canonical_artifact_hash({**base, "trades_count": 10})
        h2 = canonical_artifact_hash({**base, "trades_count": 20})
        assert h1 != h2

    def test_canonical_hash_length(self):
        d = {"agent_id": "a", "trades_count": 1}
        h = canonical_artifact_hash(d)
        assert len(h) == 66  # "0x" + 64 hex chars


class TestStrategyHashAdditional:
    def test_default_strategy_config_has_type(self):
        assert "type" in DEFAULT_STRATEGY_CONFIG

    def test_default_strategy_config_has_version(self):
        assert "version" in DEFAULT_STRATEGY_CONFIG

    def test_default_strategy_config_has_risk_params(self):
        assert "max_position_pct" in DEFAULT_STRATEGY_CONFIG
        assert "stop_loss_pct" in DEFAULT_STRATEGY_CONFIG

    def test_strategy_hash_of_default_config(self):
        h = keccak256_config(DEFAULT_STRATEGY_CONFIG)
        assert h.startswith("0x")
        assert len(h) == 66

    def test_strategy_hash_changes_with_version(self):
        cfg1 = {**DEFAULT_STRATEGY_CONFIG, "version": "1.0.0"}
        cfg2 = {**DEFAULT_STRATEGY_CONFIG, "version": "2.0.0"}
        assert keccak256_config(cfg1) != keccak256_config(cfg2)


class TestArtifactDirAdditional:
    def test_artifacts_dir_is_path(self):
        from pathlib import Path
        assert isinstance(ARTIFACTS_DIR, Path)

    def test_generator_creates_dir(self, ledger):
        import tempfile, os
        with tempfile.TemporaryDirectory() as d:
            new_dir = os.path.join(d, "subdir", "artifacts")
            gen = ArtifactGenerator(ledger=ledger, artifacts_dir=new_dir)
            gen.generate(session_id="mkdir-test")
            assert os.path.isdir(new_dir)

    def test_artifact_filename_pattern(self, generator):
        a = generator.generate(session_id="file-pattern-test")
        assert a.artifact_path.endswith(".artifact.json")
        assert "file-pattern-test" in a.artifact_path

    def test_load_missing_session_raises(self, generator):
        import pytest
        with pytest.raises(FileNotFoundError):
            generator.load("nonexistent-session-xyz")


class TestSignArtifactHashAdditional:
    def test_sign_returns_nonempty_string(self):
        from validation_artifacts import _DEV_PRIVATE_KEY
        sig = _sign_artifact_hash("0x" + "a" * 64, _DEV_PRIVATE_KEY)
        assert len(sig) > 0

    def test_sign_different_hashes_different_sigs(self):
        from validation_artifacts import _DEV_PRIVATE_KEY
        sig1 = _sign_artifact_hash("0x" + "a" * 64, _DEV_PRIVATE_KEY)
        sig2 = _sign_artifact_hash("0x" + "b" * 64, _DEV_PRIVATE_KEY)
        assert sig1 != sig2

    def test_sign_deterministic(self):
        from validation_artifacts import _DEV_PRIVATE_KEY
        h = "0x" + "c" * 64
        sig1 = _sign_artifact_hash(h, _DEV_PRIVATE_KEY)
        sig2 = _sign_artifact_hash(h, _DEV_PRIVATE_KEY)
        assert sig1 == sig2


class TestFinalArtifactCoverage:
    def test_generate_win_rate_0_when_all_sells(self):
        l = TradeLedger()
        for _ in range(5):
            l.log_trade("a", "BTC/USD", "SELL", 0.1, 50000.0)
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            gen = ArtifactGenerator(ledger=l, artifacts_dir=d)
            a = gen.generate(session_id="all-sells")
            assert a.win_rate == 0.0

    def test_generate_win_rate_1_when_all_buys(self):
        l = TradeLedger()
        for _ in range(5):
            l.log_trade("a", "BTC/USD", "BUY", 0.1, 50000.0)
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            gen = ArtifactGenerator(ledger=l, artifacts_dir=d)
            a = gen.generate(session_id="all-buys")
            assert a.win_rate == 1.0

    def test_artifact_hash_is_66_chars(self, generator):
        a = generator.generate(session_id="hashlen-test")
        assert len(a.artifact_hash) == 66

    def test_strategy_hash_is_66_chars(self, generator):
        a = generator.generate(session_id="strat-len-test")
        assert len(a.strategy_hash) == 66

    def test_artifact_session_id_preserved_on_disk(self, generator):
        sid = "disk-preserve-test"
        a = generator.generate(session_id=sid)
        loaded = generator.load(sid)
        assert loaded.session_id == sid

    def test_artifact_agent_id_preserved_on_disk(self, generator):
        sid = "agent-disk-test"
        a = generator.generate(session_id=sid)
        loaded = generator.load(sid)
        assert loaded.agent_id == "test-agent"

    def test_generate_returns_correct_trades_count(self):
        l = TradeLedger()
        for _ in range(7):
            l.log_trade("a", "BTC/USD", "BUY", 0.1, 50000.0)
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            gen = ArtifactGenerator(ledger=l, artifacts_dir=d)
            a = gen.generate(session_id="count7")
            assert a.trades_count == 7

    def test_default_agent_id_constant(self):
        assert DEFAULT_AGENT_ID == "erc8004-trading-agent-v1"

    def test_canonical_hash_returns_0x_hex_string(self):
        d = {"agent_id": "test", "trades_count": 5}
        h = canonical_artifact_hash(d)
        assert h.startswith("0x")
        # Should be valid hex after prefix
        int(h[2:], 16)  # raises ValueError if not valid hex
