"""
test_oracle_client.py — Tests for oracle_client.py (12 tests).

All HTTP calls are mocked — no real network access.
"""

from __future__ import annotations

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

from oracle_client import (
    OracleClient,
    _parse_price,
    FALLBACK_ETH_PRICE,
    FALLBACK_BTC_PRICE,
    REDSTONE_GATEWAY_URL,
)


# ─── Helper: build a fake RedStone response ───────────────────────────────────

def _make_response(eth: float = 3200.0, btc: float = 65000.0) -> dict:
    """Build a minimal RedStone data-packages response dict."""
    def _pkg(symbol: str, value: float) -> dict:
        return {
            "dataPoints": [{"dataFeedId": symbol, "value": value}],
            "timestampMilliseconds": 1_700_000_000_000,
        }

    return {
        "ETH": [_pkg("ETH", eth)],
        "BTC": [_pkg("BTC", btc)],
    }


# ─── _parse_price ─────────────────────────────────────────────────────────────


class TestParsePrice:
    def test_extracts_eth_price(self):
        data = _make_response(eth=3100.0)
        assert _parse_price(data, "ETH") == pytest.approx(3100.0)

    def test_extracts_btc_price(self):
        data = _make_response(btc=62000.0)
        assert _parse_price(data, "BTC") == pytest.approx(62000.0)

    def test_returns_none_for_missing_symbol(self):
        assert _parse_price({}, "ETH") is None

    def test_returns_none_for_empty_packages(self):
        data = {"ETH": []}
        assert _parse_price(data, "ETH") is None

    def test_returns_none_for_empty_datapoints(self):
        data = {"ETH": [{"dataPoints": []}]}
        assert _parse_price(data, "ETH") is None

    def test_returns_none_for_null_value(self):
        data = {"ETH": [{"dataPoints": [{"dataFeedId": "ETH", "value": None}]}]}
        assert _parse_price(data, "ETH") is None

    def test_returns_none_for_non_positive_price(self):
        data = {"ETH": [{"dataPoints": [{"dataFeedId": "ETH", "value": -1.0}]}]}
        assert _parse_price(data, "ETH") is None


# ─── OracleClient ─────────────────────────────────────────────────────────────


class TestOracleClientInit:
    def test_default_url(self):
        c = OracleClient()
        assert c.gateway_url == REDSTONE_GATEWAY_URL

    def test_custom_url(self):
        c = OracleClient(gateway_url="http://mock-oracle/")
        assert c.gateway_url == "http://mock-oracle/"

    def test_fallback_enabled_by_default(self):
        c = OracleClient()
        assert c.use_fallback is True

    def test_cache_initially_empty(self):
        c = OracleClient()
        cached = c.get_cached_prices()
        assert cached["ETH"] is None
        assert cached["BTC"] is None


class TestFetchEthPrice:
    @pytest.mark.asyncio
    async def test_returns_eth_price_on_success(self):
        c = OracleClient()
        c._fetch_raw = AsyncMock(return_value=_make_response(eth=3500.0))
        price = await c.fetch_eth_price()
        assert price == pytest.approx(3500.0)

    @pytest.mark.asyncio
    async def test_updates_cache_on_success(self):
        c = OracleClient()
        c._fetch_raw = AsyncMock(return_value=_make_response(eth=3200.0))
        await c.fetch_eth_price()
        assert c.get_cached_prices()["ETH"] == pytest.approx(3200.0)

    @pytest.mark.asyncio
    async def test_falls_back_to_constant_on_http_error(self):
        c = OracleClient(use_fallback=True)
        c._fetch_raw = AsyncMock(side_effect=Exception("timeout"))
        price = await c.fetch_eth_price()
        assert price == pytest.approx(FALLBACK_ETH_PRICE)

    @pytest.mark.asyncio
    async def test_falls_back_to_cached_value_if_available(self):
        c = OracleClient(use_fallback=True)
        c._last_eth = 3800.0
        c._fetch_raw = AsyncMock(side_effect=Exception("timeout"))
        price = await c.fetch_eth_price()
        assert price == pytest.approx(3800.0)


class TestFetchBtcPrice:
    @pytest.mark.asyncio
    async def test_returns_btc_price_on_success(self):
        c = OracleClient()
        c._fetch_raw = AsyncMock(return_value=_make_response(btc=61000.0))
        price = await c.fetch_btc_price()
        assert price == pytest.approx(61000.0)

    @pytest.mark.asyncio
    async def test_falls_back_to_constant_on_error(self):
        c = OracleClient(use_fallback=True)
        c._fetch_raw = AsyncMock(side_effect=RuntimeError("network"))
        price = await c.fetch_btc_price()
        assert price == pytest.approx(FALLBACK_BTC_PRICE)
