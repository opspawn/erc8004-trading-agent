"""
test_s30_compliance_validation.py — Sprint 30: Compliance Guardrails,
                                     Trustless Validation, and Circuit Breaker tests.

130 tests covering:
  - _check_compliance_rules: unit tests for each rule
  - get_compliance_status: compliance status report
  - validate_trade: trade validation
  - get_compliance_audit: audit log retrieval
  - get_validation_proof: deterministic proof generation
  - _deterministic_hash: hash utility
  - get_validation_consensus: weighted majority consensus
  - get_circuit_breaker_status: circuit breaker status
  - trigger_circuit_breaker_test: circuit breaker simulation
  - reset_circuit_breaker: circuit breaker reset
  - HTTP GET  /demo/compliance/status
  - HTTP POST /demo/compliance/validate
  - HTTP GET  /demo/compliance/audit
  - HTTP GET  /demo/validation/proof
  - HTTP GET  /demo/validation/consensus
  - HTTP GET  /demo/circuit-breaker/status
  - HTTP POST /demo/circuit-breaker/test
  - HTTP POST /demo/circuit-breaker/reset
  - Edge cases and error handling
"""

from __future__ import annotations

import json
import threading
import time
from urllib.request import urlopen, Request
from urllib.error import HTTPError

import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from demo_server import (
    # Compliance
    _check_compliance_rules,
    get_compliance_status,
    validate_trade,
    get_compliance_audit,
    _COMPLIANCE_AUDIT_LOG,
    _COMPLIANCE_AUDIT_LOCK,
    # Validation / proof
    _deterministic_hash,
    get_validation_proof,
    get_validation_consensus,
    # Circuit breaker
    _CB_STATE,
    _CB_LOCK,
    get_circuit_breaker_status,
    trigger_circuit_breaker_test,
    reset_circuit_breaker,
    # Server
    DemoServer,
)

# ── Test Server Fixture ────────────────────────────────────────────────────────

_SERVER_PORT = 18430
_BASE_URL = f"http://localhost:{_SERVER_PORT}"
_server: DemoServer = None


def setup_module(module):
    global _server
    _server = DemoServer(port=_SERVER_PORT)
    _server.start()
    time.sleep(0.3)


def teardown_module(module):
    global _server
    if _server:
        _server.stop()


def _get(path: str, timeout: int = 5) -> dict:
    with urlopen(f"{_BASE_URL}{path}", timeout=timeout) as r:
        return json.loads(r.read())


def _post(path: str, body: dict = None, timeout: int = 5) -> dict:
    data = json.dumps(body or {}).encode()
    req = Request(f"{_BASE_URL}{path}", data=data,
                  headers={"Content-Type": "application/json"})
    with urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def _post_expect_error(path: str, body: dict = None, timeout: int = 5) -> tuple:
    """POST and return (status_code, response_dict) including error responses."""
    data = json.dumps(body or {}).encode()
    req = Request(f"{_BASE_URL}{path}", data=data,
                  headers={"Content-Type": "application/json"})
    try:
        with urlopen(req, timeout=timeout) as r:
            return r.status, json.loads(r.read())
    except HTTPError as e:
        return e.code, json.loads(e.read())


# ── _check_compliance_rules Unit Tests ────────────────────────────────────────

class TestCheckComplianceRules:
    def test_valid_small_trade(self):
        trade = {"symbol": "ETH/USDC", "side": "buy", "amount": 1, "price": 100}
        violations, warnings, risk_score = _check_compliance_rules(trade)
        assert violations == []
        assert risk_score < 1.0

    def test_position_size_violation_over_5pct(self):
        # 10_000 portfolio, need trade_value > 500 to exceed 5%
        trade = {"symbol": "ETH/USDC", "side": "buy", "amount": 1, "price": 600}
        violations, warnings, risk_score = _check_compliance_rules(trade)
        rule_names = [v["rule"] for v in violations]
        assert "risk_limits_enforced" in rule_names

    def test_position_size_warning_2_to_5pct(self):
        # 2.5% position: amount=250, price=1.0 → trade_value=250
        trade = {"symbol": "ETH/USDC", "side": "buy", "amount": 250, "price": 1.0}
        violations, warnings, risk_score = _check_compliance_rules(trade)
        assert violations == []
        warn_rules = [w["rule"] for w in warnings]
        assert "position_size" in warn_rules

    def test_invalid_symbol_no_slash(self):
        trade = {"symbol": "ETHUSD", "side": "buy", "amount": 1, "price": 100}
        violations, _, _ = _check_compliance_rules(trade)
        rule_names = [v["rule"] for v in violations]
        assert "agent_identity_registered" in rule_names

    def test_invalid_symbol_empty(self):
        trade = {"symbol": "", "side": "buy", "amount": 1, "price": 100}
        violations, _, _ = _check_compliance_rules(trade)
        rule_names = [v["rule"] for v in violations]
        assert "agent_identity_registered" in rule_names

    def test_invalid_side(self):
        trade = {"symbol": "ETH/USDC", "side": "short", "amount": 1, "price": 100}
        violations, _, _ = _check_compliance_rules(trade)
        rule_names = [v["rule"] for v in violations]
        assert "audit_trail_complete" in rule_names

    def test_side_sell_valid(self):
        trade = {"symbol": "BTC/USDC", "side": "sell", "amount": 0.01, "price": 30000}
        violations, _, _ = _check_compliance_rules(trade)
        # May warn about position size but no side violation
        rule_names = [v["rule"] for v in violations]
        assert "audit_trail_complete" not in rule_names

    def test_zero_price(self):
        trade = {"symbol": "ETH/USDC", "side": "buy", "amount": 1, "price": 0}
        violations, _, _ = _check_compliance_rules(trade)
        rule_names = [v["rule"] for v in violations]
        assert "slippage_controlled" in rule_names

    def test_negative_price(self):
        trade = {"symbol": "ETH/USDC", "side": "buy", "amount": 1, "price": -100}
        violations, _, _ = _check_compliance_rules(trade)
        rule_names = [v["rule"] for v in violations]
        assert "slippage_controlled" in rule_names

    def test_zero_amount(self):
        trade = {"symbol": "ETH/USDC", "side": "buy", "amount": 0, "price": 100}
        violations, _, _ = _check_compliance_rules(trade)
        rule_names = [v["rule"] for v in violations]
        assert "risk_limits_enforced" in rule_names

    def test_negative_amount(self):
        trade = {"symbol": "ETH/USDC", "side": "buy", "amount": -5, "price": 100}
        violations, _, _ = _check_compliance_rules(trade)
        rule_names = [v["rule"] for v in violations]
        assert "risk_limits_enforced" in rule_names

    def test_risk_score_at_limit(self):
        # Exactly 5% position: trade_value = 500, portfolio = 10_000
        trade = {"symbol": "ETH/USDC", "side": "buy", "amount": 5, "price": 100}
        violations, _, risk_score = _check_compliance_rules(trade)
        assert abs(risk_score - 1.0) < 1e-6

    def test_risk_score_half_limit(self):
        # 2.5% position: trade_value = 250, portfolio = 10_000
        trade = {"symbol": "ETH/USDC", "side": "buy", "amount": 2.5, "price": 100}
        violations, _, risk_score = _check_compliance_rules(trade)
        assert abs(risk_score - 0.5) < 1e-4

    def test_risk_score_capped_at_one(self):
        # Trade exceeds portfolio value
        trade = {"symbol": "ETH/USDC", "side": "buy", "amount": 1000, "price": 100}
        violations, _, risk_score = _check_compliance_rules(trade)
        assert risk_score == 1.0

    def test_multiple_violations(self):
        trade = {"symbol": "BAD", "side": "invalid", "amount": -1, "price": -1}
        violations, _, _ = _check_compliance_rules(trade)
        assert len(violations) >= 3


# ── get_compliance_status Unit Tests ──────────────────────────────────────────

class TestGetComplianceStatus:
    def setup_method(self):
        # Ensure circuit breaker is off
        reset_circuit_breaker()

    def test_returns_dict(self):
        status = get_compliance_status()
        assert isinstance(status, dict)

    def test_erc8004_compliant_true(self):
        status = get_compliance_status()
        assert status["erc8004_compliant"] is True

    def test_version_field(self):
        status = get_compliance_status()
        assert "version" in status
        assert status["version"] == "0.4.0"

    def test_checks_is_list(self):
        status = get_compliance_status()
        assert isinstance(status["checks"], list)

    def test_checks_count(self):
        status = get_compliance_status()
        assert len(status["checks"]) == 6

    def test_checks_have_required_fields(self):
        status = get_compliance_status()
        for check in status["checks"]:
            assert "rule" in check
            assert "status" in check
            assert "detail" in check

    def test_all_checks_pass_when_cb_off(self):
        status = get_compliance_status()
        statuses = {c["rule"]: c["status"] for c in status["checks"]}
        assert statuses["circuit_breaker_active"] == "PASS"

    def test_cb_check_warns_when_triggered(self):
        trigger_circuit_breaker_test()
        try:
            status = get_compliance_status()
            statuses = {c["rule"]: c["status"] for c in status["checks"]}
            assert statuses["circuit_breaker_active"] == "WARN"
        finally:
            reset_circuit_breaker()

    def test_compliance_score_100_when_normal(self):
        status = get_compliance_status()
        assert status["compliance_score"] == 100

    def test_compliance_score_lower_when_cb_active(self):
        trigger_circuit_breaker_test()
        try:
            status = get_compliance_status()
            assert status["compliance_score"] < 100
        finally:
            reset_circuit_breaker()

    def test_last_violation_none(self):
        status = get_compliance_status()
        assert status["last_violation"] is None

    def test_portfolio_drawdown_field(self):
        status = get_compliance_status()
        assert "portfolio_drawdown_1h" in status
        assert isinstance(status["portfolio_drawdown_1h"], float)

    def test_known_rule_names_present(self):
        status = get_compliance_status()
        rules = {c["rule"] for c in status["checks"]}
        assert "agent_identity_registered" in rules
        assert "risk_limits_enforced" in rules
        assert "no_wash_trading" in rules
        assert "slippage_controlled" in rules


# ── validate_trade Unit Tests ──────────────────────────────────────────────────

class TestValidateTrade:
    def test_valid_trade_approved(self):
        result = validate_trade({"symbol": "ETH/USDC", "side": "buy", "amount": 1, "price": 100})
        assert result["approved"] is True
        assert result["valid"] is True
        assert result["violations"] == []

    def test_invalid_trade_rejected(self):
        result = validate_trade({"symbol": "BAD", "side": "buy", "amount": 1, "price": 100})
        assert result["approved"] is False
        assert result["valid"] is False
        assert len(result["violations"]) > 0

    def test_risk_score_present(self):
        result = validate_trade({"symbol": "ETH/USDC", "side": "buy", "amount": 1, "price": 100})
        assert "risk_score" in result
        assert 0.0 <= result["risk_score"] <= 1.0

    def test_warnings_present(self):
        result = validate_trade({"symbol": "ETH/USDC", "side": "buy", "amount": 1, "price": 100})
        assert "warnings" in result
        assert isinstance(result["warnings"], list)

    def test_audit_log_grows(self):
        with _COMPLIANCE_AUDIT_LOCK:
            before = len(_COMPLIANCE_AUDIT_LOG)
        validate_trade({"symbol": "ETH/USDC", "side": "sell", "amount": 0.5, "price": 2000})
        with _COMPLIANCE_AUDIT_LOCK:
            after = len(_COMPLIANCE_AUDIT_LOG)
        assert after == before + 1

    def test_audit_entry_has_outcome_approved(self):
        with _COMPLIANCE_AUDIT_LOCK:
            before = len(_COMPLIANCE_AUDIT_LOG)
        validate_trade({"symbol": "ETH/USDC", "side": "buy", "amount": 1, "price": 100})
        with _COMPLIANCE_AUDIT_LOCK:
            entry = _COMPLIANCE_AUDIT_LOG[before]
        assert entry["outcome"] == "APPROVED"

    def test_audit_entry_has_outcome_rejected(self):
        with _COMPLIANCE_AUDIT_LOCK:
            before = len(_COMPLIANCE_AUDIT_LOG)
        validate_trade({"symbol": "BAD", "side": "buy", "amount": 1, "price": 100})
        with _COMPLIANCE_AUDIT_LOCK:
            entry = _COMPLIANCE_AUDIT_LOG[before]
        assert entry["outcome"] == "REJECTED"

    def test_large_trade_violation(self):
        # 10% of $10,000 portfolio
        result = validate_trade({"symbol": "ETH/USDC", "side": "buy", "amount": 1000, "price": 1.0})
        assert result["approved"] is False


# ── get_compliance_audit Unit Tests ───────────────────────────────────────────

class TestGetComplianceAudit:
    def test_returns_dict(self):
        result = get_compliance_audit()
        assert isinstance(result, dict)

    def test_has_required_keys(self):
        result = get_compliance_audit()
        assert "total_trades_audited" in result
        assert "violations" in result
        assert "violation_rate" in result
        assert "entries" in result

    def test_entries_is_list(self):
        result = get_compliance_audit()
        assert isinstance(result["entries"], list)

    def test_default_limit_50(self):
        result = get_compliance_audit(limit=50)
        assert len(result["entries"]) <= 50

    def test_custom_limit(self):
        result = get_compliance_audit(limit=10)
        assert len(result["entries"]) <= 10

    def test_limit_clamped_min(self):
        result = get_compliance_audit(limit=0)
        assert len(result["entries"]) >= 1

    def test_total_audited_positive(self):
        result = get_compliance_audit()
        assert result["total_trades_audited"] > 0

    def test_violation_rate_between_0_and_1(self):
        result = get_compliance_audit()
        assert 0.0 <= result["violation_rate"] <= 1.0

    def test_entries_have_required_fields(self):
        result = get_compliance_audit()
        for entry in result["entries"]:
            assert "entry_id" in entry
            assert "timestamp" in entry
            assert "outcome" in entry


# ── _deterministic_hash Unit Tests ────────────────────────────────────────────

class TestDeterministicHash:
    def test_returns_string(self):
        h = _deterministic_hash("test")
        assert isinstance(h, str)

    def test_starts_with_sha256(self):
        h = _deterministic_hash("test")
        assert h.startswith("sha256:")

    def test_deterministic_same_input(self):
        h1 = _deterministic_hash("same_input")
        h2 = _deterministic_hash("same_input")
        assert h1 == h2

    def test_different_inputs_different_hash(self):
        h1 = _deterministic_hash("input_a")
        h2 = _deterministic_hash("input_b")
        assert h1 != h2

    def test_empty_string(self):
        h = _deterministic_hash("")
        assert h.startswith("sha256:")
        assert len(h) > 10

    def test_hash_length(self):
        h = _deterministic_hash("test")
        # sha256: + 64 hex chars
        assert len(h) == 7 + 64


# ── get_validation_proof Unit Tests ───────────────────────────────────────────

class TestGetValidationProof:
    def test_returns_dict(self):
        proof = get_validation_proof()
        assert isinstance(proof, dict)

    def test_proof_type(self):
        proof = get_validation_proof()
        assert proof["proof_type"] == "ERC-8004-DecisionProof"

    def test_agent_id_present(self):
        proof = get_validation_proof()
        assert "opspawn" in proof["agent_id"].lower()

    def test_timestamp_present(self):
        proof = get_validation_proof()
        assert "timestamp" in proof
        assert "2026" in proof["timestamp"]

    def test_inputs_hash_format(self):
        proof = get_validation_proof()
        assert proof["decision_inputs_hash"].startswith("sha256:")

    def test_attestation_format(self):
        proof = get_validation_proof()
        assert proof["attestation"].startswith("sha256:")

    def test_reasoning_trace_is_list(self):
        proof = get_validation_proof()
        assert isinstance(proof["reasoning_trace"], list)
        assert len(proof["reasoning_trace"]) >= 3

    def test_model_version(self):
        proof = get_validation_proof()
        assert "claude" in proof["model_version"].lower()

    def test_strategy_in_proof(self):
        proof = get_validation_proof(strategy="arbitrage")
        assert proof["strategy"] == "arbitrage"

    def test_deterministic_same_strategy(self):
        p1 = get_validation_proof(strategy="momentum")
        p2 = get_validation_proof(strategy="momentum")
        assert p1["decision_inputs_hash"] == p2["decision_inputs_hash"]
        assert p1["attestation"] == p2["attestation"]

    def test_different_strategies_different_hash(self):
        p1 = get_validation_proof(strategy="momentum")
        p2 = get_validation_proof(strategy="mean_reversion")
        assert p1["decision_inputs_hash"] != p2["decision_inputs_hash"]


# ── get_validation_consensus Unit Tests ───────────────────────────────────────

class TestGetValidationConsensus:
    def test_returns_dict(self):
        result = get_validation_consensus(seed=42)
        assert isinstance(result, dict)

    def test_has_required_keys(self):
        result = get_validation_consensus(seed=42)
        for key in ("consensus_round", "participants", "votes", "consensus", "consensus_strength", "method"):
            assert key in result

    def test_votes_is_list(self):
        result = get_validation_consensus(seed=42)
        assert isinstance(result["votes"], list)

    def test_vote_count(self):
        result = get_validation_consensus(seed=42)
        assert result["participants"] == len(result["votes"])
        assert result["participants"] == 4

    def test_vote_fields(self):
        result = get_validation_consensus(seed=42)
        for vote in result["votes"]:
            assert "agent" in vote
            assert "vote" in vote
            assert "confidence" in vote

    def test_confidence_between_0_and_1(self):
        result = get_validation_consensus(seed=42)
        for vote in result["votes"]:
            assert 0.0 <= vote["confidence"] <= 1.0

    def test_consensus_is_valid_action(self):
        result = get_validation_consensus(seed=42)
        assert result["consensus"] in ("BUY", "SELL", "HOLD")

    def test_consensus_strength_between_0_and_1(self):
        result = get_validation_consensus(seed=42)
        assert 0.0 <= result["consensus_strength"] <= 1.0

    def test_method_field(self):
        result = get_validation_consensus(seed=42)
        assert result["method"] == "weighted_majority"

    def test_deterministic_with_same_seed(self):
        r1 = get_validation_consensus(seed=99)
        r2 = get_validation_consensus(seed=99)
        # Confidence values should match for same seed
        for v1, v2 in zip(r1["votes"], r2["votes"]):
            assert v1["confidence"] == v2["confidence"]

    def test_round_increments(self):
        r1 = get_validation_consensus(seed=1)
        r2 = get_validation_consensus(seed=2)
        assert r2["consensus_round"] == r1["consensus_round"] + 1

    def test_buy_majority_with_default_seed(self):
        result = get_validation_consensus(seed=42)
        # With default agents (3 BUY, 1 HOLD), BUY should win
        assert result["consensus"] == "BUY"


# ── get_circuit_breaker_status Unit Tests ─────────────────────────────────────

class TestGetCircuitBreakerStatus:
    def setup_method(self):
        reset_circuit_breaker()

    def test_returns_dict(self):
        status = get_circuit_breaker_status()
        assert isinstance(status, dict)

    def test_active_false_by_default(self):
        status = get_circuit_breaker_status()
        assert status["active"] is False

    def test_triggers_is_list(self):
        status = get_circuit_breaker_status()
        assert isinstance(status["triggers"], list)
        assert len(status["triggers"]) == 3

    def test_trigger_fields(self):
        status = get_circuit_breaker_status()
        for t in status["triggers"]:
            assert "condition" in t
            assert "current_value" in t
            assert "triggered" in t

    def test_last_triggered_none_by_default(self):
        status = get_circuit_breaker_status()
        assert status["last_triggered"] is None

    def test_auto_resume_field(self):
        status = get_circuit_breaker_status()
        assert "auto_resume" in status
        assert "minutes" in status["auto_resume"]

    def test_active_true_after_trigger(self):
        trigger_circuit_breaker_test()
        status = get_circuit_breaker_status()
        assert status["active"] is True

    def test_drawdown_trigger_reflected(self):
        trigger_circuit_breaker_test()
        status = get_circuit_breaker_status()
        # After test trigger, drawdown should be > 5%
        drawdown_trigger = next(t for t in status["triggers"] if "drawdown" in t["condition"])
        assert drawdown_trigger["triggered"] is True


# ── trigger_circuit_breaker_test Unit Tests ───────────────────────────────────

class TestTriggerCircuitBreaker:
    def setup_method(self):
        reset_circuit_breaker()

    def test_returns_triggered_true(self):
        result = trigger_circuit_breaker_test()
        assert result["triggered"] is True

    def test_halt_sequence_is_list(self):
        result = trigger_circuit_breaker_test()
        assert isinstance(result["halt_sequence"], list)
        assert len(result["halt_sequence"]) >= 4

    def test_halt_sequence_steps(self):
        result = trigger_circuit_breaker_test()
        steps = [s["step"] for s in result["halt_sequence"]]
        assert steps == sorted(steps)  # steps are ordered

    def test_sets_cb_active(self):
        trigger_circuit_breaker_test()
        with _CB_LOCK:
            assert _CB_STATE["active"] is True

    def test_auto_resume_at_future(self):
        result = trigger_circuit_breaker_test()
        assert result["auto_resume_at"] > result["trigger_ts"]

    def test_reason_is_test_trigger(self):
        result = trigger_circuit_breaker_test()
        assert result["reason"] == "test_trigger"

    def test_note_field_present(self):
        result = trigger_circuit_breaker_test()
        assert "note" in result


# ── reset_circuit_breaker Unit Tests ──────────────────────────────────────────

class TestResetCircuitBreaker:
    def test_reset_returns_dict(self):
        result = reset_circuit_breaker()
        assert isinstance(result, dict)

    def test_reset_true(self):
        result = reset_circuit_breaker()
        assert result["reset"] is True

    def test_was_active_false_when_not_triggered(self):
        reset_circuit_breaker()  # ensure clean
        result = reset_circuit_breaker()
        assert result["was_active"] is False

    def test_was_active_true_when_triggered(self):
        trigger_circuit_breaker_test()
        result = reset_circuit_breaker()
        assert result["was_active"] is True

    def test_active_false_after_reset(self):
        trigger_circuit_breaker_test()
        reset_circuit_breaker()
        with _CB_LOCK:
            assert _CB_STATE["active"] is False

    def test_drawdown_reset_to_normal(self):
        trigger_circuit_breaker_test()
        reset_circuit_breaker()
        with _CB_LOCK:
            assert _CB_STATE["portfolio_drawdown_1h"] < 0.05


# ── HTTP Endpoint Tests ────────────────────────────────────────────────────────

class TestHTTPComplianceStatus:
    def setup_method(self):
        reset_circuit_breaker()

    def test_get_compliance_status_200(self):
        data = _get("/demo/compliance/status")
        assert "erc8004_compliant" in data

    def test_compliance_status_checks(self):
        data = _get("/demo/compliance/status")
        assert isinstance(data["checks"], list)
        assert len(data["checks"]) == 6

    def test_compliance_score_100(self):
        data = _get("/demo/compliance/status")
        assert data["compliance_score"] == 100

    def test_compliance_version(self):
        data = _get("/demo/compliance/status")
        assert data["version"] == "0.4.0"


class TestHTTPComplianceValidate:
    def test_valid_trade_approved(self):
        data = _post("/demo/compliance/validate", {
            "trade": {"symbol": "ETH/USDC", "side": "buy", "amount": 1, "price": 100}
        })
        assert data["approved"] is True

    def test_invalid_trade_rejected(self):
        data = _post("/demo/compliance/validate", {
            "trade": {"symbol": "BAD", "side": "buy", "amount": 1, "price": 100}
        })
        assert data["approved"] is False

    def test_missing_trade_field_400(self):
        code, data = _post_expect_error("/demo/compliance/validate", {})
        assert code == 400
        assert "error" in data

    def test_invalid_json_400(self):
        req = Request(
            f"{_BASE_URL}/demo/compliance/validate",
            data=b"not json",
            headers={"Content-Type": "application/json"},
        )
        try:
            with urlopen(req, timeout=5) as r:
                code = r.status
        except HTTPError as e:
            code = e.code
        assert code == 400

    def test_trade_not_dict_400(self):
        code, data = _post_expect_error("/demo/compliance/validate", {"trade": "ETH"})
        assert code == 400

    def test_response_has_risk_score(self):
        data = _post("/demo/compliance/validate", {
            "trade": {"symbol": "ETH/USDC", "side": "sell", "amount": 0.5, "price": 2400}
        })
        assert "risk_score" in data
        assert 0.0 <= data["risk_score"] <= 1.0

    def test_response_has_warnings(self):
        data = _post("/demo/compliance/validate", {
            "trade": {"symbol": "ETH/USDC", "side": "buy", "amount": 1, "price": 100}
        })
        assert "warnings" in data


class TestHTTPComplianceAudit:
    def test_get_audit_200(self):
        data = _get("/demo/compliance/audit")
        assert "entries" in data

    def test_audit_total_positive(self):
        data = _get("/demo/compliance/audit")
        assert data["total_trades_audited"] > 0

    def test_audit_violation_rate(self):
        data = _get("/demo/compliance/audit")
        assert 0.0 <= data["violation_rate"] <= 1.0

    def test_audit_limit_param(self):
        data = _get("/demo/compliance/audit?limit=5")
        assert len(data["entries"]) <= 5

    def test_audit_entries_have_timestamps(self):
        data = _get("/demo/compliance/audit")
        for entry in data["entries"]:
            assert "timestamp" in entry


class TestHTTPValidationProof:
    def test_get_proof_200(self):
        data = _get("/demo/validation/proof")
        assert data["proof_type"] == "ERC-8004-DecisionProof"

    def test_proof_attestation_format(self):
        data = _get("/demo/validation/proof")
        assert data["attestation"].startswith("sha256:")

    def test_proof_strategy_param(self):
        data = _get("/demo/validation/proof?strategy=arbitrage")
        assert data["strategy"] == "arbitrage"

    def test_proof_reasoning_trace(self):
        data = _get("/demo/validation/proof")
        assert isinstance(data["reasoning_trace"], list)
        assert len(data["reasoning_trace"]) >= 3


class TestHTTPValidationConsensus:
    def test_get_consensus_200(self):
        data = _get("/demo/validation/consensus")
        assert "consensus" in data
        assert "votes" in data

    def test_consensus_has_votes(self):
        data = _get("/demo/validation/consensus")
        assert len(data["votes"]) == 4

    def test_consensus_strength(self):
        data = _get("/demo/validation/consensus")
        assert 0.0 <= data["consensus_strength"] <= 1.0

    def test_consensus_seed_param(self):
        d1 = _get("/demo/validation/consensus?seed=42")
        d2 = _get("/demo/validation/consensus?seed=42")
        # Same seed → same confidence values
        for v1, v2 in zip(d1["votes"], d2["votes"]):
            assert v1["confidence"] == v2["confidence"]


class TestHTTPCircuitBreaker:
    def setup_method(self):
        reset_circuit_breaker()

    def test_get_status_200(self):
        data = _get("/demo/circuit-breaker/status")
        assert "active" in data
        assert data["active"] is False

    def test_status_has_triggers(self):
        data = _get("/demo/circuit-breaker/status")
        assert len(data["triggers"]) == 3

    def test_post_test_triggers(self):
        data = _post("/demo/circuit-breaker/test")
        assert data["triggered"] is True
        assert "halt_sequence" in data

    def test_active_after_test(self):
        _post("/demo/circuit-breaker/test")
        data = _get("/demo/circuit-breaker/status")
        assert data["active"] is True

    def test_reset_restores_normal(self):
        _post("/demo/circuit-breaker/test")
        _post("/demo/circuit-breaker/reset")
        data = _get("/demo/circuit-breaker/status")
        assert data["active"] is False

    def test_reset_response(self):
        data = _post("/demo/circuit-breaker/reset")
        assert data["reset"] is True
