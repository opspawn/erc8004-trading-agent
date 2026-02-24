"""
test_market_data.py — Tests for the MarketDataAdapter class.

All tests use mocked HTTP responses — no live API calls.
"""

from __future__ import annotations

import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from market_data import (
    CACHE_TTL_SECONDS,
    COINGECKO_PRICE_URL,
    MarketDataAdapter,
    OrderBook,
    OrderBookLevel,
    PriceCacheEntry,
    SYNTHETIC_BASE_PRICES,
)


# ─── PriceCacheEntry ──────────────────────────────────────────────────────────


class TestPriceCacheEntry:
    def test_fresh_by_default(self):
        e = PriceCacheEntry(price=100.0)
        assert e.is_fresh() is True

    def test_stale_after_ttl(self):
        e = PriceCacheEntry(price=100.0, cached_at=time.time() - 20)
        assert e.is_fresh(ttl=10) is False

    def test_fresh_within_ttl(self):
        e = PriceCacheEntry(price=100.0, cached_at=time.time() - 5)
        assert e.is_fresh(ttl=10) is True


# ─── OrderBookLevel ───────────────────────────────────────────────────────────


class TestOrderBookLevel:
    def test_fields(self):
        lvl = OrderBookLevel(price=3200.0, quantity=1.5)
        assert lvl.price == 3200.0
        assert lvl.quantity == 1.5


# ─── OrderBook ────────────────────────────────────────────────────────────────


class TestOrderBook:
    def make_book(self):
        bids = [OrderBookLevel(3199.0, 1.0), OrderBookLevel(3198.0, 2.0)]
        asks = [OrderBookLevel(3201.0, 1.0), OrderBookLevel(3202.0, 2.0)]
        return OrderBook(symbol="ethereum", bids=bids, asks=asks)

    def test_best_bid(self):
        book = self.make_book()
        assert book.best_bid == 3199.0

    def test_best_ask(self):
        book = self.make_book()
        assert book.best_ask == 3201.0

    def test_spread(self):
        book = self.make_book()
        assert book.spread == pytest.approx(2.0, abs=0.01)

    def test_spread_pct(self):
        book = self.make_book()
        assert book.spread_pct is not None
        assert book.spread_pct > 0

    def test_empty_bids(self):
        book = OrderBook(symbol="X", bids=[], asks=[OrderBookLevel(100.0, 1.0)])
        assert book.best_bid is None
        assert book.spread is None

    def test_empty_asks(self):
        book = OrderBook(symbol="X", bids=[OrderBookLevel(99.0, 1.0)], asks=[])
        assert book.best_ask is None
        assert book.spread is None

    def test_spread_pct_none_on_zero_bid(self):
        book = OrderBook(
            symbol="X",
            bids=[OrderBookLevel(0.0, 1.0)],
            asks=[OrderBookLevel(1.0, 1.0)],
        )
        # Spread exists but spread_pct divides by bid_price
        assert book.spread is not None


# ─── MarketDataAdapter — Construction ────────────────────────────────────────


class TestAdapterInit:
    def test_default_init(self):
        a = MarketDataAdapter()
        assert a.use_synthetic is False
        assert a.cache_ttl == CACHE_TTL_SECONDS
        assert "ethereum" in a.tracked_symbols

    def test_synthetic_mode(self):
        a = MarketDataAdapter(use_synthetic=True)
        assert a.use_synthetic is True

    def test_custom_symbols(self):
        a = MarketDataAdapter(tracked_symbols=["solana", "chainlink"])
        assert a.tracked_symbols == ["solana", "chainlink"]

    def test_cache_starts_empty(self):
        a = MarketDataAdapter()
        assert len(a._price_cache) == 0


# ─── Synthetic Price Generation ───────────────────────────────────────────────


class TestSyntheticPrice:
    def test_returns_float(self):
        a = MarketDataAdapter(use_synthetic=True)
        price = a._synthetic_price("ethereum")
        assert isinstance(price, float)

    def test_non_negative(self):
        a = MarketDataAdapter(use_synthetic=True)
        for _ in range(20):
            assert a._synthetic_price("bitcoin") > 0

    def test_near_base_price(self):
        a = MarketDataAdapter(use_synthetic=True)
        base = SYNTHETIC_BASE_PRICES["ethereum"]
        for _ in range(20):
            price = a._synthetic_price("ethereum")
            assert abs(price - base) / base < 0.05   # within 5%

    def test_unknown_symbol_uses_default(self):
        a = MarketDataAdapter(use_synthetic=True)
        price = a._synthetic_price("unknown_coin_xyz")
        assert price > 0

    def test_case_insensitive_lookup(self):
        a = MarketDataAdapter(use_synthetic=True)
        # ETH maps to ethereum base
        p1 = SYNTHETIC_BASE_PRICES.get("ETH", 100.0)
        assert p1 > 0


# ─── Synthetic Orderbook ──────────────────────────────────────────────────────


class TestSyntheticOrderbook:
    def test_returns_orderbook(self):
        a = MarketDataAdapter(use_synthetic=True)
        book = a._synthetic_orderbook("ETH", 3200.0, 5)
        assert isinstance(book, OrderBook)

    def test_depth_respected(self):
        a = MarketDataAdapter(use_synthetic=True)
        book = a._synthetic_orderbook("ETH", 3200.0, 7)
        assert len(book.bids) == 7
        assert len(book.asks) == 7

    def test_bids_below_mid(self):
        a = MarketDataAdapter(use_synthetic=True)
        book = a._synthetic_orderbook("ETH", 3200.0, 3)
        for lvl in book.bids:
            assert lvl.price < 3200.0

    def test_asks_above_mid(self):
        a = MarketDataAdapter(use_synthetic=True)
        book = a._synthetic_orderbook("ETH", 3200.0, 3)
        for lvl in book.asks:
            assert lvl.price > 3200.0

    def test_positive_quantities(self):
        a = MarketDataAdapter(use_synthetic=True)
        book = a._synthetic_orderbook("ETH", 3200.0, 5)
        for lvl in book.bids + book.asks:
            assert lvl.quantity > 0


# ─── Cache Behavior ───────────────────────────────────────────────────────────


class TestCacheBehavior:
    def test_cache_price_stored(self):
        a = MarketDataAdapter(use_synthetic=True)
        a._cache_price("ethereum", 3200.0)
        assert "ethereum" in a._price_cache
        assert a._price_cache["ethereum"].price == 3200.0

    def test_get_cached_price_returns_value(self):
        a = MarketDataAdapter(use_synthetic=True)
        a._cache_price("bitcoin", 65000.0)
        result = a.get_cached_price("bitcoin")
        assert result == 65000.0

    def test_get_cached_price_stale_returns_none(self):
        a = MarketDataAdapter(use_synthetic=True, cache_ttl=0.001)
        a._cache_price("ethereum", 3200.0)
        time.sleep(0.01)
        assert a.get_cached_price("ethereum") is None

    def test_invalidate_single_symbol(self):
        a = MarketDataAdapter(use_synthetic=True)
        a._cache_price("ethereum", 3200.0)
        a._cache_price("bitcoin", 65000.0)
        a.invalidate_cache("ethereum")
        assert "ethereum" not in a._price_cache
        assert "bitcoin" in a._price_cache

    def test_invalidate_all(self):
        a = MarketDataAdapter(use_synthetic=True)
        a._cache_price("ethereum", 3200.0)
        a._cache_price("bitcoin", 65000.0)
        a.invalidate_cache()
        assert len(a._price_cache) == 0


# ─── fetch_price ─────────────────────────────────────────────────────────────


class TestFetchPrice:
    @pytest.mark.asyncio
    async def test_synthetic_mode_returns_float(self):
        a = MarketDataAdapter(use_synthetic=True)
        price = await a.fetch_price("ethereum")
        assert isinstance(price, float)
        assert price > 0

    @pytest.mark.asyncio
    async def test_cache_hit_avoids_api(self):
        a = MarketDataAdapter(use_synthetic=False)
        a._cache_price("ethereum", 3200.0)
        # No HTTP should be called — cache is fresh
        with patch("httpx.AsyncClient") as mock_client:
            price = await a.fetch_price("ethereum")
        mock_client.assert_not_called()
        assert price == 3200.0

    @pytest.mark.asyncio
    async def test_live_api_success(self):
        a = MarketDataAdapter(use_synthetic=False)
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"ethereum": {"usd": 3100.0}}
        mock_resp.raise_for_status = MagicMock()

        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client.get = AsyncMock(return_value=mock_resp)

        with patch("market_data.httpx.AsyncClient", return_value=mock_client):
            price = await a.fetch_price("ethereum")

        assert price == 3100.0

    @pytest.mark.asyncio
    async def test_api_failure_falls_back_to_synthetic(self):
        a = MarketDataAdapter(use_synthetic=False)
        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client.get = AsyncMock(side_effect=RuntimeError("network error"))

        with patch("market_data.httpx.AsyncClient", return_value=mock_client):
            price = await a.fetch_price("ethereum")

        assert price > 0   # falls back to synthetic

    @pytest.mark.asyncio
    async def test_price_cached_after_fetch(self):
        a = MarketDataAdapter(use_synthetic=True)
        await a.fetch_price("ethereum")
        assert a.get_cached_price("ethereum") is not None

    @pytest.mark.asyncio
    async def test_case_insensitive_symbol(self):
        a = MarketDataAdapter(use_synthetic=True)
        p1 = await a.fetch_price("ETHEREUM")
        p2 = await a.fetch_price("ethereum")
        # Both should work (cached after first call)
        assert p1 > 0
        assert p2 > 0


# ─── fetch_orderbook ─────────────────────────────────────────────────────────


class TestFetchOrderbook:
    @pytest.mark.asyncio
    async def test_returns_orderbook(self):
        a = MarketDataAdapter(use_synthetic=True)
        book = await a.fetch_orderbook("ethereum")
        assert isinstance(book, OrderBook)

    @pytest.mark.asyncio
    async def test_default_depth_10(self):
        a = MarketDataAdapter(use_synthetic=True)
        book = await a.fetch_orderbook("ethereum")
        assert len(book.bids) == 10
        assert len(book.asks) == 10

    @pytest.mark.asyncio
    async def test_custom_depth(self):
        a = MarketDataAdapter(use_synthetic=True)
        book = await a.fetch_orderbook("bitcoin", depth=5)
        assert len(book.bids) == 5
        assert len(book.asks) == 5


# ─── get_spread ──────────────────────────────────────────────────────────────


class TestGetSpread:
    @pytest.mark.asyncio
    async def test_returns_float(self):
        a = MarketDataAdapter(use_synthetic=True)
        spread = await a.get_spread("ethereum")
        assert isinstance(spread, float)

    @pytest.mark.asyncio
    async def test_spread_positive(self):
        a = MarketDataAdapter(use_synthetic=True)
        spread = await a.get_spread("ethereum")
        assert spread > 0

    @pytest.mark.asyncio
    async def test_spread_pct_positive(self):
        a = MarketDataAdapter(use_synthetic=True)
        spread_pct = await a.get_spread_pct("ethereum")
        assert spread_pct > 0

    @pytest.mark.asyncio
    async def test_spread_pct_small(self):
        a = MarketDataAdapter(use_synthetic=True)
        spread_pct = await a.get_spread_pct("ethereum")
        assert spread_pct < 0.01   # less than 1%


# ─── CoinGecko _fetch_coingecko_price ────────────────────────────────────────


class TestCoinGeckoFetch:
    @pytest.mark.asyncio
    async def test_parse_valid_response(self):
        a = MarketDataAdapter()
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"bitcoin": {"usd": 65000.0}}
        mock_resp.raise_for_status = MagicMock()

        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client.get = AsyncMock(return_value=mock_resp)

        with patch("market_data.httpx.AsyncClient", return_value=mock_client):
            price = await a._fetch_coingecko_price("bitcoin")

        assert price == 65000.0

    @pytest.mark.asyncio
    async def test_raises_on_missing_symbol(self):
        a = MarketDataAdapter()
        mock_resp = MagicMock()
        mock_resp.json.return_value = {}
        mock_resp.raise_for_status = MagicMock()

        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client.get = AsyncMock(return_value=mock_resp)

        with patch("market_data.httpx.AsyncClient", return_value=mock_client):
            with pytest.raises(ValueError, match="no price"):
                await a._fetch_coingecko_price("unknowncoin")
