"""
Tests for the x402 payment client.
"""

import json
import time
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
from eth_account import Account

from x402_client import (
    DEFAULT_FACILITATOR,
    X402_VERSION,
    X402Client,
    X402Ledger,
    PaymentRecord,
    PaymentReceipt,
    PaymentRequirement,
    PaymentPayload,
    create_x402_client,
)


# ─── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def test_account():
    """Generate a test Ethereum account."""
    return Account.from_key("0x" + "aa" * 32)


@pytest.fixture
def payment_requirement():
    return PaymentRequirement(
        scheme="exact",
        network="eip155:8453",
        max_amount_required="10000",
        resource="https://api.example.com/data",
        description="Market data access",
        pay_to="0x" + "bb" * 20,
    )


@pytest.fixture
def demo_client(test_account):
    return X402Client(account=test_account, demo_mode=True)


# ─── PaymentRequirement tests ─────────────────────────────────────────────────

class TestPaymentRequirement:
    def test_from_dict_standard_format(self):
        data = {
            "scheme": "exact",
            "network": "eip155:8453",
            "maxAmountRequired": "10000",
            "resource": "https://api.test.com",
            "description": "Test",
            "payTo": "0xabc",
        }
        req = PaymentRequirement.from_dict(data)
        assert req.scheme == "exact"
        assert req.network == "eip155:8453"
        assert req.max_amount_required == "10000"
        assert req.pay_to == "0xabc"

    def test_from_dict_snake_case_fallback(self):
        data = {
            "scheme": "exact",
            "network": "eip155:84532",
            "max_amount_required": "5000",
            "resource": "https://api.test.com",
            "description": "Test",
            "pay_to": "0xdef",
        }
        req = PaymentRequirement.from_dict(data)
        assert req.max_amount_required == "5000"
        assert req.pay_to == "0xdef"
        assert req.network == "eip155:84532"

    def test_from_dict_defaults(self):
        req = PaymentRequirement.from_dict({})
        assert req.scheme == "exact"
        assert req.max_amount_required == "10000"
        assert req.required_deadline_seconds == 300

    def test_from_dict_custom_deadline(self):
        req = PaymentRequirement.from_dict({"requiredDeadlineSeconds": 600})
        assert req.required_deadline_seconds == 600


# ─── X402Ledger tests ─────────────────────────────────────────────────────────

class TestX402Ledger:
    def test_initially_empty(self):
        ledger = X402Ledger()
        assert ledger.total_payments == 0
        assert ledger.total_spent_usdc == 0.0

    def test_record_payment(self):
        ledger = X402Ledger()
        entry = PaymentRecord(
            resource="https://example.com",
            amount_usdc_cents=10000,
            network="eip155:8453",
            receipt=None,
            demo_mode=True,
        )
        ledger.record(entry)
        assert ledger.total_payments == 1
        assert abs(ledger.total_spent_usdc - 0.01) < 1e-9

    def test_record_multiple_payments(self):
        ledger = X402Ledger()
        for i in range(5):
            ledger.record(PaymentRecord(
                resource=f"https://example.com/{i}",
                amount_usdc_cents=10000,
                network="eip155:8453",
                receipt=None,
                demo_mode=True,
            ))
        assert ledger.total_payments == 5
        assert abs(ledger.total_spent_usdc - 0.05) < 1e-9

    def test_get_stats_structure(self):
        ledger = X402Ledger()
        ledger.record(PaymentRecord(
            resource="https://test.com",
            amount_usdc_cents=5000,
            network="eip155:8453",
            receipt=None,
            demo_mode=True,
        ))
        stats = ledger.get_stats()
        assert "total_payments" in stats
        assert "total_spent_usdc" in stats
        assert "recent" in stats
        assert stats["total_payments"] == 1

    def test_get_stats_recent_limit(self):
        ledger = X402Ledger()
        for i in range(15):
            ledger.record(PaymentRecord(
                resource=f"r{i}", amount_usdc_cents=100,
                network="eip155:8453", receipt=None, demo_mode=True,
            ))
        stats = ledger.get_stats()
        assert len(stats["recent"]) <= 10


# ─── PaymentReceipt tests ─────────────────────────────────────────────────────

class TestPaymentReceipt:
    def test_from_dict_success(self):
        data = {
            "success": True,
            "transactionHash": "0x" + "aa" * 32,
            "network": "eip155:8453",
            "payer": "0x" + "bb" * 20,
            "amount": "10000",
        }
        receipt = PaymentReceipt.from_dict(data)
        assert receipt.success is True
        assert receipt.transaction_hash == "0x" + "aa" * 32
        assert receipt.amount == "10000"

    def test_from_dict_failure(self):
        receipt = PaymentReceipt.from_dict({"success": False})
        assert receipt.success is False
        assert receipt.transaction_hash is None

    def test_timestamp_set(self):
        before = time.time()
        receipt = PaymentReceipt.from_dict({"success": True, "payer": "0x0"})
        after = time.time()
        assert before <= receipt.timestamp <= after


# ─── X402Client tests ─────────────────────────────────────────────────────────

class TestX402Client:
    def test_init_defaults(self, test_account):
        client = X402Client(account=test_account)
        assert client.demo_mode is True
        assert client.facilitator_url == DEFAULT_FACILITATOR
        assert client.ledger is not None

    def test_init_custom_network(self, test_account):
        client = X402Client(account=test_account, network="eip155:84532")
        assert client.network == "eip155:84532"

    def test_init_no_account(self):
        client = X402Client()
        assert client.account is None
        assert client.demo_mode is True

    def test_build_payment_payload_demo_mode(self, demo_client, payment_requirement):
        payload = demo_client._build_payment_payload(payment_requirement, "10000")
        assert payload["x402Version"] == X402_VERSION
        assert payload["scheme"] == "exact"
        assert payload["network"] == payment_requirement.network
        assert "payload" in payload
        assert "signature" in payload["payload"]
        # Demo mode signature is zeroed
        assert payload["payload"]["signature"] == "0x" + "00" * 65

    def test_build_payment_payload_authorization_fields(self, demo_client, payment_requirement):
        payload = demo_client._build_payment_payload(payment_requirement, "10000")
        auth = payload["payload"]["authorization"]
        assert "from" in auth
        assert "to" in auth
        assert auth["value"] == "10000"
        assert "validBefore" in auth

    def test_build_payment_payload_live_mode(self, test_account, payment_requirement):
        client = X402Client(account=test_account, demo_mode=False)
        payload = client._build_payment_payload(payment_requirement, "10000")
        # Live mode should have a real signature (not all zeros)
        sig = payload["payload"]["signature"]
        # Signature may or may not have 0x prefix depending on eth_account version
        sig_hex = sig[2:] if sig.startswith("0x") else sig
        # A real ECDSA signature is 65 bytes = 130 hex chars
        assert len(sig_hex) == 130
        # Should not be all zeros (demo mode)
        assert sig_hex != "00" * 65

    def test_parse_402_response_standard(self, demo_client):
        body = {
            "scheme": "exact",
            "network": "eip155:8453",
            "maxAmountRequired": "10000",
            "resource": "https://api.test.com",
            "description": "Test",
            "payTo": "0xabc",
        }
        req = demo_client._parse_402_response(body)
        assert req is not None
        assert req.scheme == "exact"

    def test_parse_402_response_accepts_format(self, demo_client):
        body = {
            "accepts": [{
                "scheme": "exact",
                "network": "eip155:8453",
                "maxAmountRequired": "5000",
                "resource": "https://test.com",
                "description": "Test",
                "payTo": "0xdef",
            }]
        }
        req = demo_client._parse_402_response(body)
        assert req is not None
        assert req.max_amount_required == "5000"

    def test_parse_402_response_x402_wrapped(self, demo_client):
        body = {
            "x402": {
                "scheme": "exact",
                "network": "eip155:8453",
                "maxAmountRequired": "20000",
                "resource": "https://test.com",
                "description": "Test",
                "payTo": "0x123",
            }
        }
        req = demo_client._parse_402_response(body)
        assert req is not None
        assert req.max_amount_required == "20000"

    def test_parse_402_response_unrecognized_returns_none(self, demo_client):
        body = {"error": "some error", "message": "not a payment"}
        req = demo_client._parse_402_response(body)
        assert req is None

    @pytest.mark.asyncio
    async def test_verify_with_facilitator_demo_mode(self, demo_client, payment_requirement):
        payload = {"x402Version": 1}
        receipt = await demo_client._verify_with_facilitator(payload, payment_requirement)
        assert receipt.success is True
        assert receipt.raw.get("mode") == "demo"

    @pytest.mark.asyncio
    async def test_request_non_402_passthrough(self, demo_client):
        """Non-402 responses should pass through without payment."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"data": "test"}

        with patch("httpx.AsyncClient") as mock_http:
            mock_http.return_value.__aenter__ = AsyncMock(return_value=mock_http.return_value)
            mock_http.return_value.__aexit__ = AsyncMock(return_value=False)
            mock_http.return_value.request = AsyncMock(return_value=mock_response)

            resp = await demo_client.request("GET", "https://example.com")
            assert resp.status_code == 200
            assert demo_client.ledger.total_payments == 0

    @pytest.mark.asyncio
    async def test_request_handles_402_and_retries(self, demo_client, payment_requirement):
        """On 402, client should pay and retry."""
        response_402 = MagicMock()
        response_402.status_code = 402
        response_402.json.return_value = {
            "scheme": "exact",
            "network": "eip155:8453",
            "maxAmountRequired": "10000",
            "resource": "https://api.test.com/data",
            "description": "Pay for data",
            "payTo": "0x" + "bb" * 20,
        }

        response_200 = MagicMock()
        response_200.status_code = 200
        response_200.json.return_value = {"markets": []}

        with patch("httpx.AsyncClient") as mock_http:
            mock_http.return_value.__aenter__ = AsyncMock(return_value=mock_http.return_value)
            mock_http.return_value.__aexit__ = AsyncMock(return_value=False)
            mock_http.return_value.request = AsyncMock(
                side_effect=[response_402, response_200]
            )

            resp = await demo_client.request("GET", "https://api.test.com/data")
            assert resp.status_code == 200
            assert demo_client.ledger.total_payments == 1

    @pytest.mark.asyncio
    async def test_get_convenience_method(self, demo_client):
        mock_response = MagicMock()
        mock_response.status_code = 200

        with patch.object(demo_client, "request", AsyncMock(return_value=mock_response)) as mock_req:
            await demo_client.get("https://example.com", params={"key": "val"})
            mock_req.assert_called_once_with("GET", "https://example.com", params={"key": "val"})

    @pytest.mark.asyncio
    async def test_post_convenience_method(self, demo_client):
        mock_response = MagicMock()
        mock_response.status_code = 201

        with patch.object(demo_client, "request", AsyncMock(return_value=mock_response)) as mock_req:
            await demo_client.post("https://example.com", json={"key": "val"})
            mock_req.assert_called_once_with("POST", "https://example.com", json={"key": "val"})


# ─── Factory function tests ───────────────────────────────────────────────────

class TestCreateX402Client:
    def test_create_default(self):
        client = create_x402_client()
        assert client.demo_mode is True
        assert "84532" in client.network  # testnet by default

    def test_create_mainnet(self):
        client = create_x402_client(testnet=False)
        assert client.network == "eip155:8453"

    def test_create_with_account(self, test_account):
        client = create_x402_client(account=test_account, demo_mode=False)
        assert client.account == test_account
        assert client.demo_mode is False

    def test_create_testnet_network(self):
        client = create_x402_client(testnet=True)
        assert client.network == "eip155:84532"
