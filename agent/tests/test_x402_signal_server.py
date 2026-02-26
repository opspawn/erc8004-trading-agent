"""
Tests for x402_signal_server.py — ERC-8004 x402 Paid Signal API.

All tests are unit/mock-based. No actual HTTP server is started except
via FastAPI TestClient (in-process).
"""

from __future__ import annotations

import json
import os
import sys

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from x402_signal_server import (
    create_app,
    PaymentRequirement,
    PaymentProof,
    PaymentValidator,
    SignalStore,
    _payment_required_response,
    _check_payment,
    PAYMENT_ADDRESS,
    SIGNAL_PRICE_USDC,
    BACKTEST_PRICE_USDC,
    PORT,
)


# ─── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture
def app():
    """FastAPI app in dev_mode (always accepts payments)."""
    return create_app(agent_id="test-agent", dev_mode=True)


@pytest.fixture
def client(app):
    return TestClient(app, raise_server_exceptions=True)


@pytest.fixture
def validator():
    return PaymentValidator(dev_mode=True)


@pytest.fixture
def store():
    return SignalStore(agent_id="test-agent")


# ─── Constants Tests ───────────────────────────────────────────────────────────


class TestConstants:
    def test_payment_address_is_hex(self):
        assert PAYMENT_ADDRESS.startswith("0x")
        assert len(PAYMENT_ADDRESS) == 42

    def test_signal_price_is_001(self):
        assert SIGNAL_PRICE_USDC == 0.01

    def test_backtest_price_is_001(self):
        assert BACKTEST_PRICE_USDC == 0.01

    def test_port_is_8083(self):
        assert PORT == 8083


# ─── PaymentRequirement Tests ─────────────────────────────────────────────────


class TestPaymentRequirement:
    def test_to_header_value_is_json(self):
        req = PaymentRequirement(
            amount_usdc=0.01,
            currency="USDC",
            network="polygon",
            address=PAYMENT_ADDRESS,
            resource="/signals/latest",
        )
        parsed = json.loads(req.to_header_value())
        assert parsed["address"] == PAYMENT_ADDRESS
        assert parsed["currency"] == "USDC"

    def test_to_dict_has_all_fields(self):
        req = PaymentRequirement(
            amount_usdc=0.01, currency="USDC",
            network="polygon", address=PAYMENT_ADDRESS,
            resource="/signals/latest",
        )
        d = req.to_dict()
        assert "amount" in d
        assert "currency" in d
        assert "network" in d
        assert "address" in d
        assert "resource" in d

    def test_amount_in_header(self):
        req = PaymentRequirement(
            amount_usdc=0.01, currency="USDC",
            network="polygon", address=PAYMENT_ADDRESS,
            resource="/test",
        )
        parsed = json.loads(req.to_header_value())
        assert float(parsed["amount"]) == 0.01


# ─── PaymentProof Tests ────────────────────────────────────────────────────────


class TestPaymentProof:
    def test_from_header_valid_json(self):
        header = json.dumps({"txHash": "0xabc", "amount": "0.01", "currency": "USDC"})
        proof = PaymentProof.from_header(header)
        assert proof.tx_hash == "0xabc"
        assert proof.amount == "0.01"
        assert proof.currency == "USDC"

    def test_from_header_tx_hash_alias(self):
        header = json.dumps({"tx_hash": "0xdef", "amount": "0.01", "currency": "USDC"})
        proof = PaymentProof.from_header(header)
        assert proof.tx_hash == "0xdef"

    def test_from_header_invalid_json_raises(self):
        with pytest.raises(ValueError):
            PaymentProof.from_header("not-json")

    def test_from_header_stores_raw(self):
        header = json.dumps({"txHash": "0x123", "amount": "0.01", "currency": "USDC"})
        proof = PaymentProof.from_header(header)
        assert proof.raw == header


# ─── PaymentValidator Tests ────────────────────────────────────────────────────


class TestPaymentValidator:
    def test_dev_mode_accepts_valid_proof(self, validator):
        proof = PaymentProof(tx_hash="0xvalid123", amount="0.01", currency="USDC")
        result = validator.validate(proof, 0.01)
        assert result is True

    def test_empty_tx_hash_raises(self, validator):
        proof = PaymentProof(tx_hash="", amount="0.01", currency="USDC")
        with pytest.raises(ValueError, match="txHash"):
            validator.validate(proof, 0.01)

    def test_duplicate_tx_hash_raises(self, validator):
        proof = PaymentProof(tx_hash="0xdup123", amount="0.01", currency="USDC")
        validator.validate(proof, 0.01)
        proof2 = PaymentProof(tx_hash="0xdup123", amount="0.01", currency="USDC")
        with pytest.raises(ValueError, match="already used"):
            validator.validate(proof2, 0.01)

    def test_total_earned_increments(self, validator):
        proof = PaymentProof(tx_hash="0xearned1", amount="0.01", currency="USDC")
        validator.validate(proof, 0.01)
        assert validator.total_earned_usdc > 0

    def test_total_requests_increments(self, validator):
        proof = PaymentProof(tx_hash="0xreq1", amount="0.01", currency="USDC")
        validator.validate(proof, 0.01)
        assert validator.total_requests_paid == 1

    def test_make_requirement_returns_object(self, validator):
        req = validator.make_requirement(0.01, "/signals/latest")
        assert isinstance(req, PaymentRequirement)
        assert req.address == PAYMENT_ADDRESS
        assert req.amount_usdc == 0.01


# ─── SignalStore Tests ─────────────────────────────────────────────────────────


class TestSignalStore:
    def test_get_latest_returns_list(self, store):
        signals = store.get_latest(10)
        assert isinstance(signals, list)

    def test_get_latest_max_count(self, store):
        signals = store.get_latest(10)
        assert len(signals) <= 10

    def test_signal_has_required_fields(self, store):
        signals = store.get_latest(1)
        s = signals[0]
        assert "signal_id" in s
        assert "timestamp" in s
        assert "symbol" in s
        assert "side" in s
        assert "price" in s
        assert "confidence" in s
        assert "agent_id" in s

    def test_signal_side_valid(self, store):
        signals = store.get_latest(10)
        valid_sides = {"BUY", "SELL", "HOLD"}
        for s in signals:
            assert s["side"] in valid_sides

    def test_signal_price_positive(self, store):
        signals = store.get_latest(10)
        for s in signals:
            assert s["price"] > 0

    def test_signal_confidence_range(self, store):
        signals = store.get_latest(10)
        for s in signals:
            assert 0.0 <= s["confidence"] <= 1.0

    def test_get_backtest_returns_dict(self, store):
        bt = store.get_backtest("BTC/USD")
        assert isinstance(bt, dict)

    def test_get_backtest_fields(self, store):
        bt = store.get_backtest("ETH/USD")
        assert "symbol" in bt
        assert "n_trades" in bt
        assert "win_rate" in bt
        assert "avg_pnl_bps" in bt
        assert "max_drawdown_bps" in bt
        assert "sharpe_ratio" in bt

    def test_get_backtest_win_rate_range(self, store):
        bt = store.get_backtest("SOL/USD")
        assert 0.0 <= bt["win_rate"] <= 1.0

    def test_get_backtest_n_trades_positive(self, store):
        bt = store.get_backtest("BTC/USD")
        assert bt["n_trades"] > 0


# ─── HTTP Endpoint Tests ───────────────────────────────────────────────────────


class TestHealthEndpoint:
    def test_health_returns_200(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200

    def test_health_returns_ok(self, client):
        data = client.get("/health").json()
        assert data["status"] == "ok"

    def test_health_has_payment_address(self, client):
        data = client.get("/health").json()
        assert data["payment_address"] == PAYMENT_ADDRESS

    def test_health_has_total_requests(self, client):
        data = client.get("/health").json()
        assert "total_requests_paid" in data

    def test_health_has_total_earned(self, client):
        data = client.get("/health").json()
        assert "total_earned_usdc" in data

    def test_health_has_uptime(self, client):
        data = client.get("/health").json()
        assert "uptime_seconds" in data
        assert data["uptime_seconds"] >= 0


class TestSignalsLatendoint:
    def test_without_payment_returns_402(self, client):
        resp = client.get("/signals/latest")
        assert resp.status_code == 402

    def test_402_has_x_payment_required_header(self, client):
        resp = client.get("/signals/latest")
        assert "x-payment-required" in resp.headers or "X-Payment-Required" in resp.headers

    def test_402_has_payment_address_header(self, client):
        resp = client.get("/signals/latest")
        headers_lower = {k.lower(): v for k, v in resp.headers.items()}
        assert "x-payment-address" in headers_lower

    def test_with_payment_returns_200(self, client):
        payment = json.dumps({"txHash": "0xpay001", "amount": "0.01", "currency": "USDC"})
        resp = client.get("/signals/latest", headers={"X-Payment": payment})
        assert resp.status_code == 200

    def test_with_payment_returns_signals(self, client):
        payment = json.dumps({"txHash": "0xpay002", "amount": "0.01", "currency": "USDC"})
        data = client.get("/signals/latest", headers={"X-Payment": payment}).json()
        assert "signals" in data
        assert isinstance(data["signals"], list)

    def test_signals_count_up_to_10(self, client):
        payment = json.dumps({"txHash": "0xpay003", "amount": "0.01", "currency": "USDC"})
        data = client.get("/signals/latest", headers={"X-Payment": payment}).json()
        assert len(data["signals"]) <= 10

    def test_payment_accepted_header(self, client):
        payment = json.dumps({"txHash": "0xpay004", "amount": "0.01", "currency": "USDC"})
        resp = client.get("/signals/latest", headers={"X-Payment": payment})
        headers_lower = {k.lower(): v for k, v in resp.headers.items()}
        assert headers_lower.get("x-payment-accepted") == "true"

    def test_replay_attack_returns_402(self, client):
        payment = json.dumps({"txHash": "0xreplay001", "amount": "0.01", "currency": "USDC"})
        client.get("/signals/latest", headers={"X-Payment": payment})
        resp = client.get("/signals/latest", headers={"X-Payment": payment})
        assert resp.status_code == 402

    def test_invalid_payment_json_returns_402(self, client):
        resp = client.get("/signals/latest", headers={"X-Payment": "not-json"})
        assert resp.status_code == 402


class TestSignalsBacktestEndpoint:
    def test_without_payment_returns_402(self, client):
        resp = client.get("/signals/backtest/BTC-USD")
        assert resp.status_code == 402

    def test_with_payment_returns_200(self, client):
        payment = json.dumps({"txHash": "0xbt001", "amount": "0.01", "currency": "USDC"})
        resp = client.get("/signals/backtest/BTC-USD", headers={"X-Payment": payment})
        assert resp.status_code == 200

    def test_backtest_has_symbol(self, client):
        payment = json.dumps({"txHash": "0xbt002", "amount": "0.01", "currency": "USDC"})
        data = client.get("/signals/backtest/ETH-USD", headers={"X-Payment": payment}).json()
        assert "symbol" in data

    def test_backtest_has_n_trades(self, client):
        payment = json.dumps({"txHash": "0xbt003", "amount": "0.01", "currency": "USDC"})
        data = client.get("/signals/backtest/SOL-USD", headers={"X-Payment": payment}).json()
        assert "n_trades" in data
        assert data["n_trades"] > 0

    def test_backtest_has_win_rate(self, client):
        payment = json.dumps({"txHash": "0xbt004", "amount": "0.01", "currency": "USDC"})
        data = client.get("/signals/backtest/BTC-USD", headers={"X-Payment": payment}).json()
        assert "win_rate" in data

    def test_backtest_payment_accepted_header(self, client):
        payment = json.dumps({"txHash": "0xbt005", "amount": "0.01", "currency": "USDC"})
        resp = client.get("/signals/backtest/AVAX-USD", headers={"X-Payment": payment})
        headers_lower = {k.lower(): v for k, v in resp.headers.items()}
        assert headers_lower.get("x-payment-accepted") == "true"


class TestX402Flow:
    def test_full_x402_flow(self, client):
        """Test the complete x402 flow: 402 → pay → 200."""
        # Step 1: No payment → 402
        resp1 = client.get("/signals/latest")
        assert resp1.status_code == 402

        # Step 2: Parse payment requirement from response
        headers_lower = {k.lower(): v for k, v in resp1.headers.items()}
        assert "x-payment-required" in headers_lower

        req_header = headers_lower["x-payment-required"]
        req_data = json.loads(req_header)
        assert float(req_data["amount"]) == SIGNAL_PRICE_USDC
        assert req_data["address"] == PAYMENT_ADDRESS

        # Step 3: Submit payment → 200
        payment = json.dumps({
            "txHash": "0xfull_flow_001",
            "amount": req_data["amount"],
            "currency": req_data["currency"],
        })
        resp2 = client.get("/signals/latest", headers={"X-Payment": payment})
        assert resp2.status_code == 200

        # Step 4: Verify response contains signals
        data = resp2.json()
        assert "signals" in data
        assert len(data["signals"]) > 0


# ─── Additional x402 Tests ────────────────────────────────────────────────────


class TestPaymentRequirementAdditional:
    def test_resource_in_header(self):
        req = PaymentRequirement(0.01, "USDC", "polygon", PAYMENT_ADDRESS, "/test")
        parsed = json.loads(req.to_header_value())
        assert parsed["resource"] == "/test"

    def test_network_in_header(self):
        req = PaymentRequirement(0.01, "USDC", "polygon", PAYMENT_ADDRESS, "/test")
        parsed = json.loads(req.to_header_value())
        assert parsed["network"] == "polygon"

    def test_to_dict_amount_type(self):
        req = PaymentRequirement(0.01, "USDC", "polygon", PAYMENT_ADDRESS, "/test")
        d = req.to_dict()
        assert d["amount"] == 0.01

    def test_different_resources(self):
        req1 = PaymentRequirement(0.01, "USDC", "polygon", PAYMENT_ADDRESS, "/signals/latest")
        req2 = PaymentRequirement(0.01, "USDC", "polygon", PAYMENT_ADDRESS, "/signals/backtest/BTC")
        assert req1.to_dict()["resource"] != req2.to_dict()["resource"]


class TestPaymentProofAdditional:
    def test_missing_tx_hash_empty_string(self):
        header = json.dumps({"amount": "0.01", "currency": "USDC"})
        proof = PaymentProof.from_header(header)
        assert proof.tx_hash == ""

    def test_from_header_defaults_currency(self):
        header = json.dumps({"txHash": "0xabc"})
        proof = PaymentProof.from_header(header)
        assert proof.currency == "USDC"

    def test_amount_zero_string(self):
        header = json.dumps({"txHash": "0xabc", "amount": "0"})
        proof = PaymentProof.from_header(header)
        assert proof.amount == "0"


class TestPaymentValidatorAdditional:
    def test_multiple_unique_proofs_accepted(self):
        v = PaymentValidator(dev_mode=True)
        for i in range(10):
            proof = PaymentProof(tx_hash=f"0xunique_{i:03d}", amount="0.01", currency="USDC")
            assert v.validate(proof, 0.01) is True

    def test_total_earned_accumulates(self):
        v = PaymentValidator(dev_mode=True)
        for i in range(5):
            proof = PaymentProof(tx_hash=f"0xacc_{i}", amount="0.01", currency="USDC")
            v.validate(proof, 0.01)
        assert v.total_earned_usdc >= 0.05 - 1e-9

    def test_total_requests_counts_correctly(self):
        v = PaymentValidator(dev_mode=True)
        for i in range(7):
            proof = PaymentProof(tx_hash=f"0xcount_{i}", amount="0.01", currency="USDC")
            v.validate(proof, 0.01)
        assert v.total_requests_paid == 7

    def test_make_requirement_usdc_network(self):
        v = PaymentValidator(dev_mode=True)
        req = v.make_requirement(0.01, "/test", currency="USDC", network="polygon")
        assert req.currency == "USDC"
        assert req.network == "polygon"


class TestSignalStoreAdditional:
    def test_agent_id_in_signals(self, store):
        signals = store.get_latest(5)
        for s in signals:
            assert s["agent_id"] == "test-agent"

    def test_signal_source_field(self, store):
        signals = store.get_latest(3)
        for s in signals:
            assert "source" in s

    def test_signal_id_unique(self, store):
        signals = store.get_latest(10)
        ids = [s["signal_id"] for s in signals]
        assert len(set(ids)) == len(ids)

    def test_get_backtest_deterministic(self, store):
        bt1 = store.get_backtest("BTC/USD")
        bt2 = store.get_backtest("BTC/USD")
        assert bt1["n_trades"] == bt2["n_trades"]
        assert bt1["win_rate"] == bt2["win_rate"]

    def test_get_backtest_different_symbols(self, store):
        bt1 = store.get_backtest("BTC/USD")
        bt2 = store.get_backtest("ETH/USD")
        # Different symbols should generally produce different results
        assert bt1["symbol"] != bt2["symbol"]

    def test_backtest_has_total_return(self, store):
        bt = store.get_backtest("BTC/USD")
        assert "total_return_pct" in bt

    def test_backtest_has_period(self, store):
        bt = store.get_backtest("BTC/USD")
        assert "period" in bt

    def test_backtest_has_strategy(self, store):
        bt = store.get_backtest("BTC/USD")
        assert "strategy" in bt

    def test_pre_populated_10_signals(self):
        s = SignalStore(agent_id="pre-pop")
        assert len(s._signals) >= 10


class TestHTTPEndpointsAdditional:
    def test_health_dev_mode_flag(self, client):
        data = client.get("/health").json()
        assert "dev_mode" in data

    def test_health_agent_id(self, client):
        data = client.get("/health").json()
        assert data["agent_id"] == "test-agent"

    def test_health_timestamp_present(self, client):
        data = client.get("/health").json()
        assert "timestamp" in data

    def test_signals_has_count_field(self, client):
        payment = json.dumps({"txHash": "0xcount_field", "amount": "0.01", "currency": "USDC"})
        data = client.get("/signals/latest", headers={"X-Payment": payment}).json()
        assert "count" in data
        assert data["count"] == len(data["signals"])

    def test_signals_has_timestamp(self, client):
        payment = json.dumps({"txHash": "0xts_field", "amount": "0.01", "currency": "USDC"})
        data = client.get("/signals/latest", headers={"X-Payment": payment}).json()
        assert "timestamp" in data

    def test_backtest_sharpe_ratio(self, client):
        payment = json.dumps({"txHash": "0xsharpe01", "amount": "0.01", "currency": "USDC"})
        data = client.get("/signals/backtest/BTC-USD", headers={"X-Payment": payment}).json()
        assert "sharpe_ratio" in data

    def test_backtest_agent_id(self, client):
        payment = json.dumps({"txHash": "0xagentid01", "amount": "0.01", "currency": "USDC"})
        data = client.get("/signals/backtest/ETH-USD", headers={"X-Payment": payment}).json()
        assert "agent_id" in data

    def test_create_app_custom_agent_id(self):
        from fastapi.testclient import TestClient as TC
        custom_app = create_app(agent_id="custom-id", dev_mode=True)
        c = TC(custom_app)
        data = c.get("/health").json()
        assert data["agent_id"] == "custom-id"

    def test_402_body_has_x402_key(self, client):
        resp = client.get("/signals/latest")
        body = resp.json()
        assert "x402" in body

    def test_402_body_x402_has_address(self, client):
        resp = client.get("/signals/latest")
        body = resp.json()
        assert body["x402"]["address"] == PAYMENT_ADDRESS

    def test_signals_latest_request_tracking(self):
        # Each paid request should increment total_requests_paid on health
        from fastapi.testclient import TestClient as TC
        app2 = create_app(agent_id="tracker", dev_mode=True)
        c = TC(app2)
        h1 = c.get("/health").json()
        initial = h1["total_requests_paid"]

        payment = json.dumps({"txHash": "0xtrack001", "amount": "0.01", "currency": "USDC"})
        c.get("/signals/latest", headers={"X-Payment": payment})
        h2 = c.get("/health").json()
        assert h2["total_requests_paid"] == initial + 1


class TestSignalStoreExtended:
    def test_multiple_get_latest_calls_different(self, store):
        # Each call generates a new signal, so results may differ
        s1 = store.get_latest(5)
        s2 = store.get_latest(5)
        # At least the first call should have had results
        assert len(s1) >= 1
        assert len(s2) >= 1

    def test_backtest_all_assets(self, store):
        from x402_signal_server import ASSETS
        for sym in ASSETS:
            bt = store.get_backtest(sym)
            assert bt["n_trades"] > 0

    def test_get_latest_n1(self, store):
        signals = store.get_latest(1)
        assert len(signals) == 1

    def test_get_latest_default_n(self, store):
        signals = store.get_latest(n=10)
        assert 1 <= len(signals) <= 10

    def test_backtest_max_drawdown_non_negative(self, store):
        bt = store.get_backtest("BTC/USD")
        assert bt["max_drawdown_bps"] >= 0

    def test_signal_symbol_in_assets(self, store):
        from x402_signal_server import ASSETS
        signals = store.get_latest(10)
        for s in signals:
            assert s["symbol"] in ASSETS


class TestCheckPaymentHelper:
    def test_no_payment_header_returns_response(self):
        v = PaymentValidator(dev_mode=True)
        result = _check_payment(None, v, 0.01, "/test")
        assert result is not None

    def test_valid_payment_returns_none(self):
        v = PaymentValidator(dev_mode=True)
        payment = json.dumps({"txHash": "0xhelper001", "amount": "0.01", "currency": "USDC"})
        result = _check_payment(payment, v, 0.01, "/test")
        assert result is None

    def test_empty_tx_hash_returns_response(self):
        v = PaymentValidator(dev_mode=True)
        payment = json.dumps({"txHash": "", "amount": "0.01", "currency": "USDC"})
        result = _check_payment(payment, v, 0.01, "/test")
        assert result is not None

    def test_replay_returns_response(self):
        v = PaymentValidator(dev_mode=True)
        payment = json.dumps({"txHash": "0xreplay_h", "amount": "0.01", "currency": "USDC"})
        _check_payment(payment, v, 0.01, "/test")  # first use
        result = _check_payment(payment, v, 0.01, "/test")  # replay
        assert result is not None


class TestPaymentRequiredResponse:
    def test_status_code_is_402(self):
        req = PaymentRequirement(0.01, "USDC", "polygon", PAYMENT_ADDRESS, "/test")
        resp = _payment_required_response(req)
        assert resp.status_code == 402

    def test_has_x_payment_required_header(self):
        req = PaymentRequirement(0.01, "USDC", "polygon", PAYMENT_ADDRESS, "/test")
        resp = _payment_required_response(req)
        headers_lower = {k.lower(): v for k, v in resp.headers.items()}
        assert "x-payment-required" in headers_lower

    def test_has_x_payment_amount_header(self):
        req = PaymentRequirement(0.01, "USDC", "polygon", PAYMENT_ADDRESS, "/test")
        resp = _payment_required_response(req)
        headers_lower = {k.lower(): v for k, v in resp.headers.items()}
        assert "x-payment-amount" in headers_lower

    def test_amount_in_header(self):
        req = PaymentRequirement(0.05, "USDC", "polygon", PAYMENT_ADDRESS, "/test")
        resp = _payment_required_response(req)
        headers_lower = {k.lower(): v for k, v in resp.headers.items()}
        assert float(headers_lower["x-payment-amount"]) == 0.05


class TestAppCreation:
    def test_create_app_returns_fastapi(self):
        from fastapi import FastAPI
        app = create_app()
        assert isinstance(app, FastAPI)

    def test_create_app_has_routes(self):
        app = create_app()
        paths = [r.path for r in app.routes]
        assert "/health" in paths

    def test_app_title(self):
        app = create_app()
        assert "x402" in app.title.lower() or "ERC" in app.title

    def test_create_app_default_dev_mode(self):
        import os
        os.environ["X402_DEV_MODE"] = "true"
        app = create_app()
        assert app is not None


class TestX402HeaderParsing:
    def test_payment_header_with_extra_fields(self):
        v = PaymentValidator(dev_mode=True)
        payment = json.dumps({
            "txHash": "0xextra001", "amount": "0.01", "currency": "USDC",
            "extra_field": "should be ignored"
        })
        result = _check_payment(payment, v, 0.01, "/test")
        assert result is None

    def test_payment_currency_usdc_accepted(self):
        v = PaymentValidator(dev_mode=True)
        payment = json.dumps({"txHash": "0xcurr001", "amount": "0.01", "currency": "USDC"})
        assert _check_payment(payment, v, 0.01, "/test") is None

    def test_payment_proof_raw_preserved(self):
        raw = json.dumps({"txHash": "0xraw001", "amount": "0.01", "currency": "USDC"})
        proof = PaymentProof.from_header(raw)
        assert proof.raw == raw

    def test_multiple_sequential_payments_accepted(self, client):
        for i in range(5):
            payment = json.dumps({"txHash": f"0xseq_{i:04d}", "amount": "0.01", "currency": "USDC"})
            resp = client.get("/signals/latest", headers={"X-Payment": payment})
            assert resp.status_code == 200

    def test_backtest_symbol_normalisation(self, client):
        # "btc-usd" should work after normalisation to "BTC/USD"
        payment = json.dumps({"txHash": "0xnorm001", "amount": "0.01", "currency": "USDC"})
        resp = client.get("/signals/backtest/btc-usd", headers={"X-Payment": payment})
        assert resp.status_code == 200

    def test_health_no_payment_needed(self, client):
        # Health should work without any payment 5 times
        for _ in range(5):
            resp = client.get("/health")
            assert resp.status_code == 200


class TestX402EdgeCases:
    def test_402_body_error_key(self, client):
        resp = client.get("/signals/latest")
        body = resp.json()
        assert "error" in body

    def test_signals_agent_id_in_response(self, client):
        payment = json.dumps({"txHash": "0xagent_resp", "amount": "0.01", "currency": "USDC"})
        data = client.get("/signals/latest", headers={"X-Payment": payment}).json()
        assert "agent_id" in data

    def test_backtest_max_drawdown_in_response(self, client):
        payment = json.dumps({"txHash": "0xmax_dd", "amount": "0.01", "currency": "USDC"})
        data = client.get("/signals/backtest/BTC-USD", headers={"X-Payment": payment}).json()
        assert "max_drawdown_bps" in data
        assert data["max_drawdown_bps"] >= 0

    def test_validator_no_dev_mode_production_raises(self):
        v = PaymentValidator(dev_mode=False)
        proof = PaymentProof(tx_hash="0xlive", amount="0.01", currency="USDC")
        import pytest
        with pytest.raises(NotImplementedError):
            v.validate(proof, 0.01)
