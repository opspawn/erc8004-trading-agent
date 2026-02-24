"""
market_data.py — Market data adapter for the ERC-8004 Trading Agent.

Provides price and orderbook data from CoinGecko (free, no API key) with:
- Price caching (10s TTL) to avoid rate limits
- Synthetic data fallback for testing / offline mode
- Spread calculation from orderbook data

Usage:
    adapter = MarketDataAdapter()
    price = await adapter.fetch_price("ethereum")
    book  = await adapter.fetch_orderbook("bitcoin", depth=5)
    spread = await adapter.get_spread("ethereum")
"""

from __future__ import annotations

import asyncio
import random
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import httpx
from loguru import logger


# ─── Constants ────────────────────────────────────────────────────────────────

COINGECKO_PRICE_URL = "https://api.coingecko.com/api/v3/simple/price"
COINGECKO_ORDERBOOK_URL = "https://api.coingecko.com/api/v3/coins/{id}/tickers"
CACHE_TTL_SECONDS = 10.0
DEFAULT_TIMEOUT = 5.0

# Base prices for synthetic fallback (approximate USD values)
SYNTHETIC_BASE_PRICES: Dict[str, float] = {
    "ethereum": 3200.0,
    "bitcoin": 65000.0,
    "solana": 180.0,
    "chainlink": 18.0,
    "uniswap": 12.0,
    "aave": 110.0,
    "ETH": 3200.0,
    "BTC": 65000.0,
    "SOL": 180.0,
    "LINK": 18.0,
}


# ─── Data Classes ─────────────────────────────────────────────────────────────


@dataclass
class OrderBookLevel:
    """Single price level in the order book."""
    price: float
    quantity: float


@dataclass
class OrderBook:
    """Snapshot of bids and asks."""
    symbol: str
    bids: List[OrderBookLevel]   # sorted descending (best bid first)
    asks: List[OrderBookLevel]   # sorted ascending (best ask first)
    fetched_at: float = field(default_factory=time.time)

    @property
    def best_bid(self) -> Optional[float]:
        return self.bids[0].price if self.bids else None

    @property
    def best_ask(self) -> Optional[float]:
        return self.asks[0].price if self.asks else None

    @property
    def spread(self) -> Optional[float]:
        if self.best_bid is not None and self.best_ask is not None:
            return self.best_ask - self.best_bid
        return None

    @property
    def spread_pct(self) -> Optional[float]:
        if self.spread is not None and self.best_bid and self.best_bid > 0:
            return self.spread / self.best_bid
        return None


@dataclass
class PriceCacheEntry:
    """Cached price with TTL."""
    price: float
    cached_at: float = field(default_factory=time.time)

    def is_fresh(self, ttl: float = CACHE_TTL_SECONDS) -> bool:
        return (time.time() - self.cached_at) < ttl


# ─── Market Data Adapter ──────────────────────────────────────────────────────


class MarketDataAdapter:
    """
    Fetch real-time price and orderbook data.

    Prioritizes CoinGecko API; falls back to synthetic data when offline
    or when use_synthetic=True (for testing).
    """

    def __init__(
        self,
        use_synthetic: bool = False,
        cache_ttl: float = CACHE_TTL_SECONDS,
        http_timeout: float = DEFAULT_TIMEOUT,
        tracked_symbols: Optional[List[str]] = None,
    ):
        self.use_synthetic = use_synthetic
        self.cache_ttl = cache_ttl
        self.http_timeout = http_timeout
        self.tracked_symbols = tracked_symbols or ["ethereum", "bitcoin"]
        self._price_cache: Dict[str, PriceCacheEntry] = {}
        self._client: Optional[httpx.AsyncClient] = None

    # ─── Public API ──────────────────────────────────────────────────────────

    async def fetch_price(self, symbol: str) -> float:
        """
        Fetch USD price for symbol.

        Uses cache if fresh; calls CoinGecko otherwise.
        Falls back to synthetic if API call fails.
        """
        symbol_lower = symbol.lower()

        # Check cache
        cached = self._price_cache.get(symbol_lower)
        if cached and cached.is_fresh(self.cache_ttl):
            logger.debug("Cache hit for {} → ${}", symbol, cached.price)
            return cached.price

        if self.use_synthetic:
            price = self._synthetic_price(symbol_lower)
            self._cache_price(symbol_lower, price)
            return price

        # Try live API
        try:
            price = await self._fetch_coingecko_price(symbol_lower)
            self._cache_price(symbol_lower, price)
            return price
        except Exception as exc:
            logger.warning("CoinGecko price fetch failed for {}: {}. Using synthetic.", symbol, exc)
            price = self._synthetic_price(symbol_lower)
            self._cache_price(symbol_lower, price)
            return price

    async def fetch_orderbook(self, symbol: str, depth: int = 10) -> OrderBook:
        """
        Fetch order book snapshot.

        CoinGecko free tier doesn't have real order books, so we synthesize
        a plausible book around the current price.
        """
        price = await self.fetch_price(symbol)
        return self._synthetic_orderbook(symbol, price, depth)

    async def get_spread(self, symbol: str) -> float:
        """
        Return bid-ask spread in USD for symbol.
        """
        book = await self.fetch_orderbook(symbol)
        return book.spread or 0.0

    async def get_spread_pct(self, symbol: str) -> float:
        """Return bid-ask spread as fraction of mid price."""
        book = await self.fetch_orderbook(symbol)
        return book.spread_pct or 0.0

    def get_cached_price(self, symbol: str) -> Optional[float]:
        """Return cached price without fetching (None if not cached or stale)."""
        symbol_lower = symbol.lower()
        cached = self._price_cache.get(symbol_lower)
        if cached and cached.is_fresh(self.cache_ttl):
            return cached.price
        return None

    def invalidate_cache(self, symbol: Optional[str] = None) -> None:
        """Invalidate cache entry for symbol (or all if None)."""
        if symbol:
            self._price_cache.pop(symbol.lower(), None)
        else:
            self._price_cache.clear()

    # ─── CoinGecko API ───────────────────────────────────────────────────────

    async def _fetch_coingecko_price(self, symbol: str) -> float:
        """Call CoinGecko simple price endpoint."""
        params = {"ids": symbol, "vs_currencies": "usd"}
        async with httpx.AsyncClient(timeout=self.http_timeout) as client:
            resp = await client.get(COINGECKO_PRICE_URL, params=params)
            resp.raise_for_status()
            data = resp.json()

        if symbol not in data or "usd" not in data[symbol]:
            raise ValueError(f"CoinGecko returned no price for '{symbol}': {data}")

        price = float(data[symbol]["usd"])
        logger.debug("CoinGecko {} → ${:.2f}", symbol, price)
        return price

    # ─── Synthetic Data ───────────────────────────────────────────────────────

    def _synthetic_price(self, symbol: str) -> float:
        """
        Generate a realistic synthetic price using Gaussian noise.

        ±0.5% noise around base price to simulate real market movement.
        """
        base = SYNTHETIC_BASE_PRICES.get(symbol, 100.0)
        noise_pct = random.gauss(0, 0.005)   # 0.5% std dev
        price = base * (1 + noise_pct)
        price = max(price, 0.001)            # never negative
        logger.debug("Synthetic price for {} → ${:.4f}", symbol, price)
        return round(price, 4)

    def _synthetic_orderbook(self, symbol: str, mid_price: float, depth: int) -> OrderBook:
        """
        Generate a plausible order book around mid_price.

        Bids below mid, asks above mid with realistic spread.
        """
        spread_pct = 0.001   # 0.1% spread
        tick = mid_price * spread_pct / 2

        bids = []
        asks = []
        for i in range(depth):
            bid_price = round(mid_price - tick * (i + 1), 6)
            ask_price = round(mid_price + tick * (i + 1), 6)
            qty = round(random.uniform(0.1, 5.0), 4)
            bids.append(OrderBookLevel(price=bid_price, quantity=qty))
            asks.append(OrderBookLevel(price=ask_price, quantity=qty))

        return OrderBook(symbol=symbol, bids=bids, asks=asks)

    # ─── Cache Helpers ────────────────────────────────────────────────────────

    def _cache_price(self, symbol: str, price: float) -> None:
        self._price_cache[symbol.lower()] = PriceCacheEntry(price=price)
