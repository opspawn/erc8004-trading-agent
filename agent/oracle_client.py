"""
oracle_client.py — RedStone oracle price client for the ERC-8004 Trading Agent.

Fetches live ETH/USD and BTC/USD prices from the RedStone oracle REST gateway.
Endpoint: https://oracle-gateway-1.b.redstone.finance/data-packages/latest/redstone-main-demo

Price data structure (abbreviated):
  {
    "ETH": [
      {
        "dataPoints": [{"dataFeedId": "ETH", "value": 3200.5}],
        "timestampMilliseconds": 1700000000000,
        ...
      }
    ],
    "BTC": [...]
  }

Usage:
    client = OracleClient()
    eth_price = await client.fetch_eth_price()
    btc_price = await client.fetch_btc_price()
"""

from __future__ import annotations

import asyncio
from typing import Optional

import httpx
from loguru import logger


# ─── Constants ────────────────────────────────────────────────────────────────

REDSTONE_GATEWAY_URL = (
    "https://oracle-gateway-1.b.redstone.finance"
    "/data-packages/latest/redstone-main-demo"
)

DEFAULT_TIMEOUT_SECONDS = 5.0

# Fallback prices used when the gateway is unreachable (prevents hard failures)
FALLBACK_ETH_PRICE: float = 3000.0
FALLBACK_BTC_PRICE: float = 60000.0


# ─── Price Parsing ────────────────────────────────────────────────────────────


def _parse_price(data: dict, symbol: str) -> Optional[float]:
    """
    Extract the median price for a symbol from a RedStone data-packages response.

    RedStone returns a list of signed data packages for each symbol.
    We take the first package's first data point value (they are pre-validated
    and deduplicated by the gateway).

    Args:
        data:   Parsed JSON response from the gateway
        symbol: Feed ID (e.g. "ETH", "BTC")

    Returns:
        Price as float, or None if parsing fails.
    """
    try:
        packages = data.get(symbol, [])
        if not packages:
            logger.warning(f"OracleClient: no packages for {symbol}")
            return None

        # First package, first data point
        data_points = packages[0].get("dataPoints", [])
        if not data_points:
            logger.warning(f"OracleClient: no dataPoints for {symbol}")
            return None

        value = data_points[0].get("value")
        if value is None:
            logger.warning(f"OracleClient: null value for {symbol}")
            return None

        price = float(value)
        if price <= 0:
            logger.warning(f"OracleClient: non-positive price {price} for {symbol}")
            return None

        return price

    except (KeyError, IndexError, TypeError, ValueError) as exc:
        logger.warning(f"OracleClient: parse error for {symbol}: {exc}")
        return None


# ─── Oracle Client ────────────────────────────────────────────────────────────


class OracleClient:
    """
    Async HTTP client for the RedStone oracle gateway.

    Fetches ETH and BTC prices with automatic fallback to last-known values
    if the request times out or the gateway returns unexpected data.
    """

    def __init__(
        self,
        gateway_url: str = REDSTONE_GATEWAY_URL,
        timeout: float = DEFAULT_TIMEOUT_SECONDS,
        use_fallback: bool = True,
    ) -> None:
        self.gateway_url = gateway_url
        self.timeout = timeout
        self.use_fallback = use_fallback

        # In-memory cache of last successful prices
        self._last_eth: Optional[float] = None
        self._last_btc: Optional[float] = None

    async def _fetch_raw(self) -> dict:
        """Fetch raw JSON from the RedStone gateway."""
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.get(self.gateway_url)
            response.raise_for_status()
            return response.json()

    async def fetch_eth_price(self) -> float:
        """
        Fetch the current ETH/USD price from RedStone.

        Returns cached or fallback price on failure.
        """
        try:
            data = await self._fetch_raw()
            price = _parse_price(data, "ETH")
            if price is not None:
                self._last_eth = price
                logger.debug(f"OracleClient: ETH/USD = {price}")
                return price
        except Exception as exc:
            logger.warning(f"OracleClient: ETH fetch failed: {exc}")

        # Fallback chain: last-known → constant default
        if self.use_fallback:
            fallback = self._last_eth or FALLBACK_ETH_PRICE
            logger.info(f"OracleClient: using ETH fallback price {fallback}")
            return fallback

        raise RuntimeError("OracleClient: ETH price unavailable and fallback disabled")

    async def fetch_btc_price(self) -> float:
        """
        Fetch the current BTC/USD price from RedStone.

        Returns cached or fallback price on failure.
        """
        try:
            data = await self._fetch_raw()
            price = _parse_price(data, "BTC")
            if price is not None:
                self._last_btc = price
                logger.debug(f"OracleClient: BTC/USD = {price}")
                return price
        except Exception as exc:
            logger.warning(f"OracleClient: BTC fetch failed: {exc}")

        if self.use_fallback:
            fallback = self._last_btc or FALLBACK_BTC_PRICE
            logger.info(f"OracleClient: using BTC fallback price {fallback}")
            return fallback

        raise RuntimeError("OracleClient: BTC price unavailable and fallback disabled")

    def get_cached_prices(self) -> dict:
        """Return currently cached prices (None if never successfully fetched)."""
        return {
            "ETH": self._last_eth,
            "BTC": self._last_btc,
        }
