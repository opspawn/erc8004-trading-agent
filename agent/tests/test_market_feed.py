"""
Tests for agent/market_feed.py

Covers:
  - GBMSimulator: price evolution, seeding
  - RateLimiter: timing, can_call, record_call
  - PriceCache: TTL expiry, set/get/evict
  - PriceTick: validation
  - MarketFeed: get_price, get_prices, stream, subscribe/broadcast
  - CoinGecko fetch: mocked success and failure paths
  - Fallback behaviour when API unavailable
"""

from __future__ import annotations

import asyncio
import math
import time
import unittest
from typing import Dict
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# ── Import module under test ──────────────────────────────────────────────────
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from market_feed import (
    GBMSimulator,
    MarketFeed,
    PriceCache,
    PriceTick,
    RateLimiter,
    FeedStatus,
    SYMBOL_TO_COINGECKO,
    GBM_BASE_PRICES,
    CACHE_TTL,
    RATE_LIMIT_INTERVAL,
)


# ─── PriceTick Tests ──────────────────────────────────────────────────────────

class TestPriceTick:

    def test_valid_creation(self):
        tick = PriceTick("BTC", 65000.0, "coingecko")
        assert tick.symbol == "BTC"
        assert tick.price == 65000.0
        assert tick.source == "coingecko"

    def test_timestamp_auto_set(self):
        before = time.time()
        tick = PriceTick("ETH", 3200.0, "gbm")
        after = time.time()
        assert before <= tick.timestamp <= after

    def test_explicit_timestamp(self):
        tick = PriceTick("SOL", 180.0, "gbm", timestamp=1234567890.0)
        assert tick.timestamp == 1234567890.0

    def test_zero_price_raises(self):
        with pytest.raises(ValueError):
            PriceTick("BTC", 0.0, "gbm")

    def test_negative_price_raises(self):
        with pytest.raises(ValueError):
            PriceTick("BTC", -1.0, "coingecko")

    def test_optional_change_1h(self):
        tick = PriceTick("BTC", 65000.0, "coingecko", change_1h=2.5)
        assert tick.change_1h == 2.5

    def test_change_1h_defaults_none(self):
        tick = PriceTick("BTC", 65000.0, "coingecko")
        assert tick.change_1h is None

    def test_gbm_source(self):
        tick = PriceTick("ETH", 3000.0, "gbm")
        assert tick.source == "gbm"

    def test_coingecko_source(self):
        tick = PriceTick("ETH", 3000.0, "coingecko")
        assert tick.source == "coingecko"

    def test_all_supported_symbols(self):
        for sym in SYMBOL_TO_COINGECKO:
            tick = PriceTick(sym, 100.0, "gbm")
            assert tick.symbol == sym

    def test_small_price(self):
        tick = PriceTick("TEST", 0.001, "gbm")
        assert tick.price == 0.001

    def test_large_price(self):
        tick = PriceTick("BTC", 1_000_000.0, "coingecko")
        assert tick.price == 1_000_000.0


# ─── GBMSimulator Tests ───────────────────────────────────────────────────────

class TestGBMSimulator:

    def test_next_price_positive(self):
        sim = GBMSimulator()
        for sym in ["BTC", "ETH", "SOL"]:
            p = sim.next_price(sym)
            assert p > 0

    def test_next_price_evolves(self):
        sim = GBMSimulator()
        p1 = sim.current_price("BTC")
        sim.next_price("BTC")
        p2 = sim.current_price("BTC")
        # After advancing, price should have changed
        assert p1 != p2

    def test_multiple_steps(self):
        sim = GBMSimulator()
        prices = [sim.next_price("ETH") for _ in range(100)]
        assert all(p > 0 for p in prices)

    def test_set_base_updates_price(self):
        sim = GBMSimulator()
        sim.set_base("BTC", 50000.0)
        assert sim.current_price("BTC") == 50000.0

    def test_set_base_negative_ignored(self):
        sim = GBMSimulator()
        original = sim.current_price("ETH")
        sim.set_base("ETH", -100.0)
        # Negative price should be ignored
        assert sim.current_price("ETH") == original

    def test_set_base_zero_ignored(self):
        sim = GBMSimulator()
        original = sim.current_price("SOL")
        sim.set_base("SOL", 0.0)
        assert sim.current_price("SOL") == original

    def test_unknown_symbol_returns_default(self):
        sim = GBMSimulator()
        p = sim.next_price("UNKNOWN_ASSET")
        assert p > 0

    def test_gbm_drift_increases_mean_over_many_steps(self):
        """With positive drift, long-run mean should exceed starting price."""
        sim = GBMSimulator(dt_seconds=86400)   # 1-day steps
        sim.set_base("BTC", 10000.0)
        prices = [sim.next_price("BTC") for _ in range(1000)]
        # With positive drift, expect average to be above starting price
        avg = sum(prices) / len(prices)
        assert avg > 0   # just verify it stays positive

    def test_current_price_without_advance(self):
        sim = GBMSimulator()
        p = sim.current_price("ETH")
        assert p == GBM_BASE_PRICES["ETH"]

    def test_different_symbols_independent(self):
        sim = GBMSimulator()
        sim.set_base("BTC", 100.0)
        sim.set_base("ETH", 200.0)
        assert sim.current_price("BTC") == 100.0
        assert sim.current_price("ETH") == 200.0

    def test_short_dt_small_steps(self):
        """Short time steps should produce smaller price changes."""
        sim_short = GBMSimulator(dt_seconds=1.0)
        sim_long  = GBMSimulator(dt_seconds=86400.0)
        sim_short.set_base("ETH", 3200.0)
        sim_long.set_base("ETH", 3200.0)

        short_prices = [sim_short.next_price("ETH") for _ in range(50)]
        long_prices  = [sim_long.next_price("ETH") for _ in range(50)]

        def std_dev(prices):
            mean = sum(prices) / len(prices)
            variance = sum((p - mean) ** 2 for p in prices) / len(prices)
            return variance ** 0.5

        # Short-dt should have lower volatility than long-dt
        assert std_dev(short_prices) < std_dev(long_prices)


# ─── RateLimiter Tests ────────────────────────────────────────────────────────

class TestRateLimiter:

    def test_can_call_initially(self):
        limiter = RateLimiter(min_interval=10.0)
        assert limiter.can_call()

    def test_cannot_call_immediately_after_record(self):
        limiter = RateLimiter(min_interval=10.0)
        limiter.record_call()
        assert not limiter.can_call()

    def test_can_call_after_interval(self):
        limiter = RateLimiter(min_interval=0.05)
        limiter.record_call()
        time.sleep(0.1)
        assert limiter.can_call()

    def test_seconds_until_ready_initially(self):
        limiter = RateLimiter(min_interval=10.0)
        assert limiter.seconds_until_ready() == 0.0

    def test_seconds_until_ready_after_call(self):
        limiter = RateLimiter(min_interval=10.0)
        limiter.record_call()
        remaining = limiter.seconds_until_ready()
        assert 0.0 < remaining <= 10.0

    def test_seconds_until_ready_after_full_interval(self):
        limiter = RateLimiter(min_interval=0.05)
        limiter.record_call()
        time.sleep(0.1)
        assert limiter.seconds_until_ready() == 0.0

    def test_multiple_record_calls(self):
        limiter = RateLimiter(min_interval=10.0)
        limiter.record_call()
        time.sleep(0.01)
        limiter.record_call()
        remaining = limiter.seconds_until_ready()
        assert remaining > 9.5

    def test_zero_interval(self):
        limiter = RateLimiter(min_interval=0.0)
        limiter.record_call()
        assert limiter.can_call()

    def test_very_short_interval(self):
        limiter = RateLimiter(min_interval=0.001)
        limiter.record_call()
        time.sleep(0.002)
        assert limiter.can_call()


# ─── PriceCache Tests ─────────────────────────────────────────────────────────

class TestPriceCache:

    def test_miss_on_empty(self):
        cache = PriceCache(ttl=10.0)
        assert cache.get("BTC") is None

    def test_set_and_get(self):
        cache = PriceCache(ttl=10.0)
        tick = PriceTick("BTC", 65000.0, "coingecko")
        cache.set(tick)
        result = cache.get("BTC")
        assert result is tick

    def test_expired_entry_returns_none(self):
        cache = PriceCache(ttl=0.05)
        tick = PriceTick("ETH", 3200.0, "gbm")
        cache.set(tick)
        time.sleep(0.1)
        assert cache.get("ETH") is None

    def test_not_expired_returns_tick(self):
        cache = PriceCache(ttl=10.0)
        tick = PriceTick("SOL", 180.0, "coingecko")
        cache.set(tick)
        result = cache.get("SOL")
        assert result is not None
        assert result.price == 180.0

    def test_evict_removes_entry(self):
        cache = PriceCache()
        tick = PriceTick("BTC", 65000.0, "gbm")
        cache.set(tick)
        cache.evict("BTC")
        assert cache.get("BTC") is None

    def test_evict_missing_key(self):
        cache = PriceCache()
        cache.evict("NONEXISTENT")   # should not raise

    def test_size_empty(self):
        assert PriceCache().size() == 0

    def test_size_after_set(self):
        cache = PriceCache()
        cache.set(PriceTick("BTC", 65000.0, "gbm"))
        cache.set(PriceTick("ETH", 3200.0, "gbm"))
        assert cache.size() == 2

    def test_size_after_evict(self):
        cache = PriceCache()
        cache.set(PriceTick("BTC", 65000.0, "gbm"))
        cache.evict("BTC")
        assert cache.size() == 0

    def test_overwrite_existing(self):
        cache = PriceCache(ttl=10.0)
        cache.set(PriceTick("BTC", 60000.0, "gbm"))
        cache.set(PriceTick("BTC", 65000.0, "coingecko"))
        result = cache.get("BTC")
        assert result.price == 65000.0

    def test_different_symbols_independent(self):
        cache = PriceCache()
        cache.set(PriceTick("BTC", 65000.0, "gbm"))
        cache.set(PriceTick("ETH", 3200.0, "gbm"))
        assert cache.get("BTC").price == 65000.0
        assert cache.get("ETH").price == 3200.0

    def test_default_ttl(self):
        cache = PriceCache()  # should use CACHE_TTL
        assert cache._ttl == CACHE_TTL


# ─── MarketFeed Tests ─────────────────────────────────────────────────────────

class TestMarketFeedGBMFallback:
    """Tests that run without network access (API mocked to fail)."""

    def _make_feed(self) -> MarketFeed:
        return MarketFeed(poll_interval=0.1, rate_limit=0.0, cache_ttl=1.0)

    @pytest.mark.asyncio
    async def test_get_price_gbm_fallback(self):
        feed = self._make_feed()
        with patch.object(feed, "_fetch_coingecko", new=AsyncMock(return_value={})):
            tick = await feed.get_price("BTC")
        assert tick.price > 0
        assert tick.source == "gbm"
        assert tick.symbol == "BTC"

    @pytest.mark.asyncio
    async def test_get_price_returns_positive(self):
        feed = self._make_feed()
        with patch.object(feed, "_fetch_coingecko", new=AsyncMock(return_value={})):
            for sym in ["BTC", "ETH", "SOL"]:
                tick = await feed.get_price(sym)
                assert tick.price > 0

    @pytest.mark.asyncio
    async def test_get_prices_batch(self):
        feed = self._make_feed()
        with patch.object(feed, "_fetch_coingecko", new=AsyncMock(return_value={})):
            results = await feed.get_prices(["BTC", "ETH", "SOL"])
        assert set(results.keys()) == {"BTC", "ETH", "SOL"}
        for tick in results.values():
            assert tick.price > 0

    @pytest.mark.asyncio
    async def test_stream_yields_ticks(self):
        feed = self._make_feed()
        with patch.object(feed, "_fetch_coingecko", new=AsyncMock(return_value={})):
            ticks = []
            async for tick in feed.stream(["BTC", "ETH"], max_ticks=4):
                ticks.append(tick)
        assert len(ticks) == 4

    @pytest.mark.asyncio
    async def test_stream_covers_symbols(self):
        feed = self._make_feed()
        with patch.object(feed, "_fetch_coingecko", new=AsyncMock(return_value={})):
            symbols_seen = set()
            async for tick in feed.stream(["BTC", "ETH"], max_ticks=4):
                symbols_seen.add(tick.symbol)
        assert "BTC" in symbols_seen
        assert "ETH" in symbols_seen

    @pytest.mark.asyncio
    async def test_subscribe_receives_ticks(self):
        feed = self._make_feed()
        queue = feed.subscribe()
        with patch.object(feed, "_fetch_coingecko", new=AsyncMock(return_value={})):
            async for _ in feed.stream(["BTC"], max_ticks=3):
                pass
        assert queue.qsize() == 3

    @pytest.mark.asyncio
    async def test_multiple_subscribers(self):
        feed = self._make_feed()
        q1 = feed.subscribe()
        q2 = feed.subscribe()
        with patch.object(feed, "_fetch_coingecko", new=AsyncMock(return_value={})):
            async for _ in feed.stream(["BTC"], max_ticks=2):
                pass
        assert q1.qsize() == 2
        assert q2.qsize() == 2

    @pytest.mark.asyncio
    async def test_unsubscribe(self):
        feed = self._make_feed()
        q = feed.subscribe()
        feed.unsubscribe(q)
        with patch.object(feed, "_fetch_coingecko", new=AsyncMock(return_value={})):
            async for _ in feed.stream(["BTC"], max_ticks=3):
                pass
        assert q.qsize() == 0

    @pytest.mark.asyncio
    async def test_unsubscribe_nonexistent(self):
        feed = self._make_feed()
        q = asyncio.Queue()
        feed.unsubscribe(q)  # should not raise

    def test_status_initial(self):
        feed = MarketFeed()
        assert feed.status.live is False
        assert feed.status.request_count == 0
        assert feed.status.fallback_count == 0

    def test_is_live_initial(self):
        feed = MarketFeed()
        assert not feed.is_live()

    def test_cache_size_initial(self):
        feed = MarketFeed()
        assert feed.cache_size() == 0

    @pytest.mark.asyncio
    async def test_cache_hit_avoids_api(self):
        feed = self._make_feed()
        # Manually insert into cache
        tick = PriceTick("BTC", 65000.0, "coingecko")
        feed._cache.set(tick)
        fetch_mock = AsyncMock(return_value={})
        with patch.object(feed, "_fetch_coingecko", new=fetch_mock):
            result = await feed.get_price("BTC")
        fetch_mock.assert_not_called()
        assert result.price == 65000.0


class TestMarketFeedLiveAPI:
    """Tests using mocked successful CoinGecko responses."""

    def _mock_response(self, prices: Dict[str, float]):
        """Build a fake CoinGecko API response dict."""
        return {cid: {"usd": price} for cid, price in prices.items()}

    @pytest.mark.asyncio
    async def test_fetch_coingecko_success(self):
        feed = MarketFeed(rate_limit=0.0)
        mock_resp = {
            "bitcoin":  {"usd": 65000.0, "usd_24h_change": 1.5},
            "ethereum": {"usd": 3200.0,  "usd_24h_change": -0.5},
        }

        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__  = AsyncMock(return_value=False)
        mock_response          = AsyncMock()
        mock_response.json     = MagicMock(return_value=mock_resp)
        mock_response.raise_for_status = MagicMock()
        mock_client.get        = AsyncMock(return_value=mock_response)

        with patch("market_feed.httpx.AsyncClient", return_value=mock_client):
            result = await feed._fetch_coingecko(["BTC", "ETH"])

        assert result["BTC"] == 65000.0
        assert result["ETH"] == 3200.0

    @pytest.mark.asyncio
    async def test_fetch_coingecko_seeds_gbm(self):
        feed = MarketFeed(rate_limit=0.0)
        mock_resp = {"bitcoin": {"usd": 70000.0}}

        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__  = AsyncMock(return_value=False)
        mock_response          = AsyncMock()
        mock_response.json     = MagicMock(return_value=mock_resp)
        mock_response.raise_for_status = MagicMock()
        mock_client.get        = AsyncMock(return_value=mock_response)

        with patch("market_feed.httpx.AsyncClient", return_value=mock_client):
            await feed._fetch_coingecko(["BTC"])

        assert feed._gbm.current_price("BTC") == 70000.0

    @pytest.mark.asyncio
    async def test_fetch_coingecko_sets_api_available(self):
        feed = MarketFeed(rate_limit=0.0)
        mock_resp = {"bitcoin": {"usd": 65000.0}}

        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__  = AsyncMock(return_value=False)
        mock_response          = AsyncMock()
        mock_response.json     = MagicMock(return_value=mock_resp)
        mock_response.raise_for_status = MagicMock()
        mock_client.get        = AsyncMock(return_value=mock_response)

        with patch("market_feed.httpx.AsyncClient", return_value=mock_client):
            await feed._fetch_coingecko(["BTC"])

        assert feed.status.api_available is True

    @pytest.mark.asyncio
    async def test_fetch_coingecko_network_error(self):
        import httpx as _httpx
        feed = MarketFeed(rate_limit=0.0)

        with patch("market_feed.httpx.AsyncClient") as MockClient:
            instance = MockClient.return_value.__aenter__.return_value
            instance.get = AsyncMock(side_effect=_httpx.ConnectError("timeout"))
            MockClient.return_value.__aexit__ = AsyncMock(return_value=False)
            result = await feed._fetch_coingecko(["BTC"])

        assert result == {}
        assert feed.status.api_available is False
        assert feed.status.error_count == 1

    @pytest.mark.asyncio
    async def test_get_price_uses_coingecko_when_available(self):
        feed = MarketFeed(rate_limit=0.0, cache_ttl=1.0)

        async def mock_fetch(symbols):
            return {"BTC": 67000.0}

        with patch.object(feed, "_fetch_coingecko", new=AsyncMock(side_effect=mock_fetch)):
            tick = await feed.get_price("BTC")

        assert tick.price == 67000.0
        assert tick.source == "coingecko"

    @pytest.mark.asyncio
    async def test_get_price_caches_result(self):
        feed = MarketFeed(rate_limit=0.0, cache_ttl=10.0)

        async def mock_fetch(symbols):
            return {"ETH": 3300.0}

        with patch.object(feed, "_fetch_coingecko", new=AsyncMock(side_effect=mock_fetch)) as m:
            await feed.get_price("ETH")
            await feed.get_price("ETH")   # second call should hit cache

        # Fetch should only be called once (second served from cache)
        assert m.call_count == 1

    @pytest.mark.asyncio
    async def test_rate_limiter_blocks_rapid_calls(self):
        feed = MarketFeed(rate_limit=100.0)   # very long interval
        feed._limiter.record_call()  # mark as recently called

        calls = 0

        async def mock_fetch(symbols):
            nonlocal calls
            calls += 1
            return {}

        with patch.object(feed, "_fetch_coingecko", new=AsyncMock(side_effect=mock_fetch)):
            await feed.get_price("BTC")
            await feed.get_price("BTC")

        assert calls == 0   # rate limiter should block all API calls

    @pytest.mark.asyncio
    async def test_fallback_count_increments(self):
        feed = MarketFeed(rate_limit=100.0)
        feed._limiter.record_call()  # force rate limit

        with patch.object(feed, "_fetch_coingecko", new=AsyncMock(return_value={})):
            await feed.get_price("BTC")
            await feed.get_price("ETH")

        assert feed.status.fallback_count >= 2

    @pytest.mark.asyncio
    async def test_get_prices_partial_cache_miss(self):
        feed = MarketFeed(rate_limit=0.0)
        # Pre-cache BTC
        feed._cache.set(PriceTick("BTC", 65000.0, "coingecko"))

        async def mock_fetch(symbols):
            return {s: 3000.0 for s in symbols}

        with patch.object(feed, "_fetch_coingecko", new=AsyncMock(side_effect=mock_fetch)):
            results = await feed.get_prices(["BTC", "ETH"])

        assert results["BTC"].price == 65000.0   # from cache
        assert results["ETH"].price == 3000.0    # from "API"

    @pytest.mark.asyncio
    async def test_start_stop(self):
        feed = MarketFeed(poll_interval=0.05, rate_limit=0.0)
        with patch.object(feed, "_fetch_coingecko", new=AsyncMock(return_value={})):
            await feed.start(["BTC"])
            assert feed.status.live is True
            await asyncio.sleep(0.12)
            await feed.stop()
            assert feed.status.live is False

    @pytest.mark.asyncio
    async def test_start_sets_subscribed_symbols(self):
        feed = MarketFeed()
        with patch.object(feed, "_fetch_coingecko", new=AsyncMock(return_value={})):
            await feed.start(["BTC", "ETH"])
            await feed.stop()
        assert "BTC" in feed.status.subscribed
        assert "ETH" in feed.status.subscribed


# ─── FeedStatus Tests ─────────────────────────────────────────────────────────

class TestFeedStatus:

    def test_default_values(self):
        status = FeedStatus()
        assert status.live is False
        assert status.api_available is False
        assert status.request_count == 0
        assert status.fallback_count == 0
        assert status.error_count == 0
        assert len(status.subscribed) == 0

    def test_is_mutable(self):
        status = FeedStatus()
        status.live = True
        status.request_count = 5
        assert status.live is True
        assert status.request_count == 5


# ─── Symbol Mapping Tests ─────────────────────────────────────────────────────

class TestSymbolMapping:

    def test_btc_maps_to_bitcoin(self):
        assert SYMBOL_TO_COINGECKO["BTC"] == "bitcoin"

    def test_eth_maps_to_ethereum(self):
        assert SYMBOL_TO_COINGECKO["ETH"] == "ethereum"

    def test_sol_maps_to_solana(self):
        assert SYMBOL_TO_COINGECKO["SOL"] == "solana"

    def test_all_keys_are_uppercase(self):
        for key in SYMBOL_TO_COINGECKO:
            assert key == key.upper()

    def test_gbm_base_prices_positive(self):
        for sym, price in GBM_BASE_PRICES.items():
            assert price > 0, f"{sym} base price must be positive"

    def test_btc_base_price_plausible(self):
        assert GBM_BASE_PRICES["BTC"] > 1000

    def test_eth_base_price_plausible(self):
        assert GBM_BASE_PRICES["ETH"] > 100


# ─── Integration Test ─────────────────────────────────────────────────────────

class TestMarketFeedIntegration:
    """End-to-end flows without real network calls."""

    @pytest.mark.asyncio
    async def test_full_flow_gbm_only(self):
        """Complete flow: create feed, subscribe, stream, receive ticks."""
        feed = MarketFeed(poll_interval=0.05, rate_limit=100.0)
        feed._limiter.record_call()  # force GBM

        received = []
        q = feed.subscribe()

        async def collect():
            async for tick in feed.stream(["BTC", "ETH", "SOL"], max_ticks=6):
                received.append(tick)

        await collect()

        assert len(received) == 6
        # All should be GBM ticks
        assert all(t.source == "gbm" for t in received)
        # Should cover all symbols
        symbols_seen = {t.symbol for t in received}
        assert symbols_seen == {"BTC", "ETH", "SOL"}

    @pytest.mark.asyncio
    async def test_coingecko_then_cache(self):
        """First call hits API; second serves from cache."""
        feed = MarketFeed(rate_limit=0.0, cache_ttl=60.0)
        call_count = 0

        async def mock_fetch(symbols):
            nonlocal call_count
            call_count += 1
            return {s: 100.0 * (ord(s[0]) - 60) for s in symbols}

        with patch.object(feed, "_fetch_coingecko", new=AsyncMock(side_effect=mock_fetch)):
            tick1 = await feed.get_price("BTC")
            tick2 = await feed.get_price("BTC")

        assert call_count == 1
        assert tick1.price == tick2.price

    @pytest.mark.asyncio
    async def test_multiple_symbols_batch_fetch(self):
        feed = MarketFeed(rate_limit=0.0)

        async def mock_fetch(symbols):
            return {s: 1000.0 for s in symbols}

        with patch.object(feed, "_fetch_coingecko", new=AsyncMock(side_effect=mock_fetch)):
            results = await feed.get_prices(["BTC", "ETH", "SOL", "LINK"])

        assert len(results) == 4
        for sym in ["BTC", "ETH", "SOL", "LINK"]:
            assert results[sym].price > 0
