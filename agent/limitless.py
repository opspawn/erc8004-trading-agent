"""
limitless.py — Limitless Exchange (DeFi prediction market) integration.

Limitless is a CLOB-based prediction market on Base (no KYC).
API: https://api.limitless.exchange
Docs: https://docs.limitless.exchange

Supports:
  - Fetching active markets
  - Getting CLOB orderbooks
  - Placing limit orders (EIP-712 signed)
  - Querying portfolio positions

Authentication:
  - Public endpoints: none required
  - Trading endpoints: limitless_session cookie (from /auth/login)
    using EIP-712 wallet signature

For ERC-8004 demo: uses x402 client for data fetches that support
x402 payment protocol; falls back to direct HTTP otherwise.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any, Optional

import httpx
from eth_account import Account
from loguru import logger

from x402_client import X402Client, create_x402_client


# ─── Constants ────────────────────────────────────────────────────────────────

BASE_URL = "https://api.limitless.exchange"

# EIP-712 domain for Limitless order signing
EIP712_DOMAIN = {
    "name": "Limitless Exchange",
    "version": "1",
    "chainId": 8453,  # Base mainnet
}

# Order sides
SIDE_BUY = 0
SIDE_SELL = 1

# Order types
ORDER_TYPE_GTC = "GTC"   # Good till cancelled
ORDER_TYPE_FOK = "FOK"   # Fill or kill


# ─── Data Classes ─────────────────────────────────────────────────────────────

@dataclass
class LimitlessMarket:
    """A prediction market on Limitless Exchange."""
    id: int
    slug: str
    title: str
    prices: list[float]         # [yes_price, no_price] between 0 and 1
    volume: str                 # USDC volume as string
    liquidity: str
    trade_type: str             # "clob" or "amm"
    category_id: Optional[int] = None
    resolved: bool = False
    extra: dict = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict) -> "LimitlessMarket":
        return cls(
            id=data.get("id", 0),
            slug=data.get("slug", data.get("address", "")),
            title=data.get("title", data.get("question", "")),
            prices=data.get("prices", [0.5, 0.5]),
            volume=str(data.get("volume", "0")),
            liquidity=str(data.get("liquidity", "0")),
            trade_type=data.get("tradeType", "clob"),
            category_id=data.get("categoryId"),
            resolved=data.get("resolved", False),
            extra=data,
        )

    @property
    def yes_price(self) -> float:
        return self.prices[0] if self.prices else 0.5

    @property
    def no_price(self) -> float:
        return self.prices[1] if len(self.prices) > 1 else (1 - self.yes_price)

    @property
    def volume_usdc(self) -> float:
        try:
            return float(self.volume)
        except (ValueError, TypeError):
            return 0.0

    def to_trader_market(self):
        """Convert to the common Market dataclass used by TradingStrategy."""
        from trader import Market
        return Market(
            id=str(self.slug or self.id),
            question=self.title,
            end_date="",
            yes_price=self.yes_price,
            no_price=self.no_price,
            volume=self.volume_usdc,
            category="limitless",
        )


@dataclass
class OrderbookLevel:
    price: float
    size: float


@dataclass
class Orderbook:
    """CLOB orderbook for a Limitless market."""
    market_slug: str
    token_id: str
    bids: list[OrderbookLevel]
    asks: list[OrderbookLevel]
    adjusted_midpoint: float
    last_trade_price: float
    max_spread: float
    min_size: float

    @classmethod
    def from_dict(cls, slug: str, data: dict) -> "Orderbook":
        bids = [
            OrderbookLevel(price=float(b["price"]), size=float(b["size"]))
            for b in data.get("bids", [])
        ]
        asks = [
            OrderbookLevel(price=float(a["price"]), size=float(a["size"]))
            for a in data.get("asks", [])
        ]
        return cls(
            market_slug=slug,
            token_id=str(data.get("tokenId", "")),
            bids=bids,
            asks=asks,
            adjusted_midpoint=float(data.get("adjustedMidpoint", 0.5)),
            last_trade_price=float(data.get("lastTradePrice", 0.5)),
            max_spread=float(data.get("maxSpread", 0.1)),
            min_size=float(data.get("minSize", 1)),
        )

    @property
    def best_bid(self) -> Optional[float]:
        return max((b.price for b in self.bids), default=None)

    @property
    def best_ask(self) -> Optional[float]:
        return min((a.price for a in self.asks), default=None)

    @property
    def spread(self) -> Optional[float]:
        if self.best_bid is not None and self.best_ask is not None:
            return self.best_ask - self.best_bid
        return None


@dataclass
class PlacedOrder:
    """Result of placing an order on Limitless."""
    order_id: str
    market_slug: str
    side: int               # SIDE_BUY or SIDE_SELL
    size: float
    price: float
    order_type: str
    status: str             # "open", "filled", "cancelled"
    tx_hash: Optional[str] = None
    raw: dict = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict, market_slug: str = "") -> "PlacedOrder":
        order = data.get("order", data)
        return cls(
            order_id=str(order.get("id", order.get("salt", "unknown"))),
            market_slug=market_slug,
            side=int(order.get("side", 0)),
            size=float(order.get("makerAmount", 0)),
            price=float(order.get("price", 0)),
            order_type=data.get("orderType", "GTC"),
            status=order.get("status", "open"),
            raw=data,
        )


@dataclass
class Position:
    """A user's position in a Limitless market."""
    market_id: str
    title: str
    position_type: str      # "YES" or "NO"
    size: float
    entry_price: float
    current_price: float
    unrealized_pnl: float
    raw: dict = field(default_factory=dict)

    @classmethod
    def from_clob_dict(cls, data: dict) -> "Position":
        return cls(
            market_id=str(data.get("marketSlug", data.get("marketId", ""))),
            title=data.get("marketTitle", ""),
            position_type="YES" if data.get("outcome", 0) == 0 else "NO",
            size=float(data.get("tokenAmount", 0)),
            entry_price=float(data.get("averagePrice", data.get("fillPrice", 0))),
            current_price=float(data.get("currentPrice", 0)),
            unrealized_pnl=float(data.get("unrealizedPnl", 0)),
            raw=data,
        )


# ─── Limitless Client ─────────────────────────────────────────────────────────

class LimitlessClient:
    """
    Client for the Limitless Exchange API.

    Supports both authenticated (trading) and public (markets) endpoints.
    Integrates with x402Client for micropayment-gated data endpoints.

    Usage:
        client = LimitlessClient(demo_mode=True)

        # Public: fetch markets
        markets = await client.fetch_markets()

        # Public: get orderbook
        ob = await client.get_orderbook("bitcoin-above-100k")

        # Authenticated: place order (requires wallet + session)
        order = await client.place_order("bitcoin-above-100k", SIDE_BUY, 5.0, 0.40)

        # Authenticated: get positions
        positions = await client.get_positions()
    """

    def __init__(
        self,
        account: Optional[Account] = None,
        session_token: Optional[str] = None,
        x402_client: Optional[X402Client] = None,
        demo_mode: bool = True,
        base_url: str = BASE_URL,
    ) -> None:
        self.account = account
        self.session_token = session_token
        self.x402 = x402_client or create_x402_client(account=account, demo_mode=demo_mode)
        self.demo_mode = demo_mode
        self.base_url = base_url.rstrip("/")
        self._session_cookie: Optional[str] = None

    def _auth_headers(self) -> dict:
        """Build authentication headers."""
        headers = {"Content-Type": "application/json"}
        if self.session_token:
            headers["Authorization"] = f"Bearer {self.session_token}"
        return headers

    def _auth_cookies(self) -> dict:
        if self._session_cookie:
            return {"limitless_session": self._session_cookie}
        return {}

    # ─── Authentication ───────────────────────────────────────────────────────

    async def login(self) -> bool:
        """
        Authenticate with Limitless Exchange via EIP-712 wallet signature.

        Flow:
          1. GET /auth/signing-message → get nonce
          2. Sign nonce with wallet key
          3. POST /auth/login with signature → get session cookie
        """
        if not self.account:
            logger.warning("LimitlessClient: no account set, cannot login")
            return False

        if self.demo_mode:
            logger.info("LimitlessClient: demo mode, skipping real auth")
            self._session_cookie = "demo_session_token"
            return True

        try:
            async with httpx.AsyncClient(timeout=15) as http:
                # Step 1: get signing message
                resp = await http.get(
                    f"{self.base_url}/auth/signing-message",
                    params={"account": self.account.address},
                )
                if resp.status_code != 200:
                    logger.error(f"Failed to get signing message: {resp.status_code}")
                    return False

                data = resp.json()
                message = data.get("message", "")
                nonce = data.get("nonce", "")

                # Step 2: sign
                from eth_account.messages import encode_defunct
                encoded = encode_defunct(text=message)
                signed = self.account.sign_message(encoded)
                signature = signed.signature.hex()

                # Step 3: login
                login_resp = await http.post(
                    f"{self.base_url}/auth/login",
                    json={
                        "account": self.account.address,
                        "signature": signature,
                        "nonce": nonce,
                    },
                )
                if login_resp.status_code == 200:
                    # Extract session cookie
                    cookie = login_resp.cookies.get("limitless_session")
                    if cookie:
                        self._session_cookie = cookie
                        logger.info("LimitlessClient: authenticated successfully")
                        return True

                logger.error(f"Login failed: {login_resp.status_code}")
                return False

        except Exception as e:
            logger.error(f"LimitlessClient login error: {e}")
            return False

    # ─── Markets ──────────────────────────────────────────────────────────────

    async def fetch_markets(
        self,
        limit: int = 20,
        sort_by: str = "newest",
        trade_type: Optional[str] = "clob",
    ) -> list[LimitlessMarket]:
        """
        Fetch active prediction markets from Limitless.

        Uses x402 client — will automatically handle any 402 payment requirement.
        Falls back gracefully on error.

        Returns list of LimitlessMarket objects.
        """
        params: dict[str, Any] = {"limit": limit, "sortBy": sort_by}
        if trade_type:
            params["tradeType"] = trade_type

        url = f"{self.base_url}/markets/active"

        if self.demo_mode:
            return self._demo_markets()

        try:
            resp = await self.x402.get(url, params=params)
            if resp.status_code == 200:
                data = resp.json()
                markets_raw = data.get("data", data if isinstance(data, list) else [])
                markets = [LimitlessMarket.from_dict(m) for m in markets_raw]
                logger.info(f"LimitlessClient: fetched {len(markets)} markets")
                return markets
            else:
                logger.warning(f"fetch_markets: unexpected status {resp.status_code}")
                return self._demo_markets()
        except Exception as e:
            logger.error(f"LimitlessClient.fetch_markets error: {e}")
            return self._demo_markets()

    async def get_orderbook(self, market_slug: str) -> Optional[Orderbook]:
        """
        Get the current CLOB orderbook for a market.

        Args:
            market_slug: Market identifier (e.g. "bitcoin-above-100k-march")

        Returns:
            Orderbook or None if market not found / uses AMM.
        """
        url = f"{self.base_url}/markets/{market_slug}/orderbook"

        if self.demo_mode:
            return self._demo_orderbook(market_slug)

        try:
            resp = await self.x402.get(url)
            if resp.status_code == 200:
                data = resp.json()
                ob = Orderbook.from_dict(market_slug, data)
                logger.info(
                    f"LimitlessClient: orderbook for {market_slug} "
                    f"bid={ob.best_bid:.3f} ask={ob.best_ask:.3f}"
                    if ob.best_bid and ob.best_ask else
                    f"LimitlessClient: orderbook for {market_slug} empty"
                )
                return ob
            elif resp.status_code == 400:
                logger.info(f"Market {market_slug} uses AMM, no CLOB orderbook")
                return None
            elif resp.status_code == 404:
                logger.warning(f"Market {market_slug} not found")
                return None
            else:
                logger.warning(f"get_orderbook: unexpected status {resp.status_code}")
                return None
        except Exception as e:
            logger.error(f"LimitlessClient.get_orderbook error: {e}")
            return None

    # ─── Trading ──────────────────────────────────────────────────────────────

    async def place_order(
        self,
        market_slug: str,
        side: int,
        size: float,
        price: float,
        order_type: str = ORDER_TYPE_GTC,
        token_id: Optional[str] = None,
    ) -> Optional[PlacedOrder]:
        """
        Place a limit order on Limitless Exchange CLOB.

        Args:
            market_slug: Market identifier
            side: SIDE_BUY (0) or SIDE_SELL (1)
            size: Number of shares (USD value)
            price: Limit price between 0.01 and 0.99
            order_type: "GTC" or "FOK"
            token_id: CLOB token ID (fetched from orderbook if not provided)

        Returns:
            PlacedOrder on success, None on failure.
            In demo_mode: returns a simulated order without hitting the API.
        """
        if self.demo_mode:
            return self._demo_place_order(market_slug, side, size, price, order_type)

        if not self.account:
            logger.error("place_order: account required for live trading")
            return None

        if not self._session_cookie and not self.session_token:
            logger.warning("place_order: not authenticated, attempting login")
            if not await self.login():
                return None

        # Get token_id from orderbook if not provided
        if not token_id:
            ob = await self.get_orderbook(market_slug)
            if not ob:
                logger.error(f"place_order: cannot get orderbook for {market_slug}")
                return None
            token_id = ob.token_id

        # Build EIP-712 signed order
        order_data = self._build_order(
            token_id=token_id,
            side=side,
            size=size,
            price=price,
            order_type=order_type,
        )

        try:
            async with httpx.AsyncClient(timeout=30) as http:
                resp = await http.post(
                    f"{self.base_url}/orders",
                    json=order_data,
                    headers=self._auth_headers(),
                    cookies=self._auth_cookies(),
                )
                if resp.status_code == 201:
                    result = resp.json()
                    order = PlacedOrder.from_dict(result, market_slug)
                    logger.info(
                        f"Order placed: {market_slug} "
                        f"{'BUY' if side == SIDE_BUY else 'SELL'} "
                        f"{size}@{price} → id={order.order_id}"
                    )
                    return order
                else:
                    logger.error(f"place_order failed: {resp.status_code} {resp.text[:200]}")
                    return None
        except Exception as e:
            logger.error(f"LimitlessClient.place_order error: {e}")
            return None

    async def get_positions(self) -> list[Position]:
        """
        Get current portfolio positions.

        Returns list of open positions across all markets.
        Requires authentication.
        """
        if self.demo_mode:
            return self._demo_positions()

        if not self._session_cookie and not self.session_token:
            logger.warning("get_positions: not authenticated")
            return []

        try:
            async with httpx.AsyncClient(timeout=15) as http:
                resp = await http.get(
                    f"{self.base_url}/portfolio/positions",
                    headers=self._auth_headers(),
                    cookies=self._auth_cookies(),
                )
                if resp.status_code == 200:
                    data = resp.json()
                    positions = []
                    for p in data.get("clob", []):
                        positions.append(Position.from_clob_dict(p))
                    logger.info(f"LimitlessClient: {len(positions)} open positions")
                    return positions
                else:
                    logger.warning(f"get_positions: status {resp.status_code}")
                    return []
        except Exception as e:
            logger.error(f"LimitlessClient.get_positions error: {e}")
            return []

    # ─── Order Building ───────────────────────────────────────────────────────

    def _build_order(
        self,
        token_id: str,
        side: int,
        size: float,
        price: float,
        order_type: str = ORDER_TYPE_GTC,
    ) -> dict:
        """
        Build an EIP-712 signed order payload for Limitless Exchange.

        The order structure mirrors Polymarket's CLOB order format.
        """
        import secrets
        from eth_account.messages import encode_defunct

        salt = int(secrets.token_hex(8), 16)
        expiration = int(time.time()) + 86400  # 24h

        maker_amount = int(size * 1_000_000)     # USDC 6 decimals
        taker_amount = int(size / price * 1_000_000) if price > 0 else maker_amount

        order_message = {
            "salt": salt,
            "maker": self.account.address,
            "signer": self.account.address,
            "taker": "0x0000000000000000000000000000000000000000",
            "tokenId": token_id,
            "makerAmount": maker_amount,
            "takerAmount": taker_amount,
            "expiration": expiration,
            "nonce": 0,
            "feeRateBps": 0,
            "side": side,
            "signatureType": 0,
        }

        # EIP-712 sign
        canonical = json.dumps(order_message, sort_keys=True, separators=(",", ":"))
        encoded = encode_defunct(text=canonical)
        signed = self.account.sign_message(encoded)
        signature = signed.signature.hex()

        return {
            "order": {
                **order_message,
                "signature": signature,
                "price": price,
            },
            "ownerId": 0,
            "orderType": order_type,
            "marketSlug": "",   # set by caller context
        }

    # ─── Demo Data ────────────────────────────────────────────────────────────

    def _demo_markets(self) -> list[LimitlessMarket]:
        """Return realistic-looking demo markets for dry-run testing."""
        return [
            LimitlessMarket(
                id=1001,
                slug="bitcoin-above-100k-march-2026",
                title="Will Bitcoin exceed $100,000 by March 31, 2026?",
                prices=[0.38, 0.62],
                volume="1250000",
                liquidity="85000",
                trade_type="clob",
            ),
            LimitlessMarket(
                id=1002,
                slug="fed-rate-cut-march-2026",
                title="Will the US Fed cut rates in March 2026?",
                prices=[0.72, 0.28],
                volume="890000",
                liquidity="62000",
                trade_type="clob",
            ),
            LimitlessMarket(
                id=1003,
                slug="eth-above-4k-q1-2026",
                title="Will ETH exceed $4,000 by end of Q1 2026?",
                prices=[0.31, 0.69],
                volume="445000",
                liquidity="38000",
                trade_type="clob",
            ),
            LimitlessMarket(
                id=1004,
                slug="spacex-starship-launch-2026",
                title="Will SpaceX successfully orbit Starship in 2026?",
                prices=[0.68, 0.32],
                volume="320000",
                liquidity="28000",
                trade_type="amm",
            ),
        ]

    def _demo_orderbook(self, market_slug: str) -> Orderbook:
        """Return a synthetic orderbook for testing."""
        return Orderbook(
            market_slug=market_slug,
            token_id="0x" + "ab" * 32,
            bids=[
                OrderbookLevel(price=0.36, size=150.0),
                OrderbookLevel(price=0.35, size=320.0),
                OrderbookLevel(price=0.34, size=500.0),
            ],
            asks=[
                OrderbookLevel(price=0.40, size=120.0),
                OrderbookLevel(price=0.41, size=280.0),
                OrderbookLevel(price=0.42, size=450.0),
            ],
            adjusted_midpoint=0.38,
            last_trade_price=0.37,
            max_spread=0.10,
            min_size=1.0,
        )

    def _demo_place_order(
        self,
        market_slug: str,
        side: int,
        size: float,
        price: float,
        order_type: str,
    ) -> PlacedOrder:
        """Return a simulated order result for dry-run mode."""
        import secrets
        order_id = secrets.token_hex(8)
        logger.info(
            f"[DEMO] Order: {market_slug} "
            f"{'BUY' if side == SIDE_BUY else 'SELL'} {size}@{price} → {order_id}"
        )
        return PlacedOrder(
            order_id=order_id,
            market_slug=market_slug,
            side=side,
            size=size,
            price=price,
            order_type=order_type,
            status="open",
            tx_hash="0x" + "00" * 32,
        )

    def _demo_positions(self) -> list[Position]:
        """Return demo positions for testing."""
        return [
            Position(
                market_id="bitcoin-above-100k-march-2026",
                title="Will Bitcoin exceed $100,000 by March 31, 2026?",
                position_type="YES",
                size=10.0,
                entry_price=0.36,
                current_price=0.38,
                unrealized_pnl=0.20,
            )
        ]
