"""
Tests for the Limitless Exchange integration module.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from eth_account import Account

from limitless import (
    BASE_URL,
    SIDE_BUY,
    SIDE_SELL,
    ORDER_TYPE_GTC,
    ORDER_TYPE_FOK,
    LimitlessMarket,
    LimitlessClient,
    Orderbook,
    OrderbookLevel,
    PlacedOrder,
    Position,
)


# ─── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def test_account():
    return Account.from_key("0x" + "cc" * 32)


@pytest.fixture
def demo_client():
    return LimitlessClient(demo_mode=True)


@pytest.fixture
def sample_market_dict():
    return {
        "id": 1001,
        "slug": "btc-100k-march",
        "title": "Will Bitcoin reach $100k by March?",
        "prices": [0.38, 0.62],
        "volume": "1250000",
        "liquidity": "85000",
        "tradeType": "clob",
    }


@pytest.fixture
def sample_orderbook_dict():
    return {
        "tokenId": "0x" + "ab" * 32,
        "bids": [
            {"price": 0.36, "size": 150.0},
            {"price": 0.35, "size": 320.0},
        ],
        "asks": [
            {"price": 0.40, "size": 120.0},
            {"price": 0.41, "size": 280.0},
        ],
        "adjustedMidpoint": 0.38,
        "lastTradePrice": 0.37,
        "maxSpread": 0.10,
        "minSize": 1.0,
    }


# ─── LimitlessMarket tests ────────────────────────────────────────────────────

class TestLimitlessMarket:
    def test_from_dict_basic(self, sample_market_dict):
        market = LimitlessMarket.from_dict(sample_market_dict)
        assert market.id == 1001
        assert market.slug == "btc-100k-march"
        assert market.title == "Will Bitcoin reach $100k by March?"
        assert market.trade_type == "clob"

    def test_prices_parsed(self, sample_market_dict):
        market = LimitlessMarket.from_dict(sample_market_dict)
        assert abs(market.yes_price - 0.38) < 0.001
        assert abs(market.no_price - 0.62) < 0.001

    def test_volume_usdc_float(self, sample_market_dict):
        market = LimitlessMarket.from_dict(sample_market_dict)
        assert market.volume_usdc == 1250000.0

    def test_no_price_derived_from_yes(self):
        market = LimitlessMarket.from_dict({
            "id": 1, "slug": "x", "title": "Test",
            "prices": [0.7], "volume": "0", "liquidity": "0", "tradeType": "clob",
        })
        assert abs(market.no_price - 0.3) < 0.001

    def test_to_trader_market(self, sample_market_dict):
        market = LimitlessMarket.from_dict(sample_market_dict)
        trader_market = market.to_trader_market()
        assert trader_market.id == "btc-100k-march"
        assert trader_market.question == market.title
        assert trader_market.category == "limitless"
        assert trader_market.yes_price == market.yes_price

    def test_defaults_on_empty_dict(self):
        market = LimitlessMarket.from_dict({})
        assert market.id == 0
        assert market.prices == [0.5, 0.5]
        assert market.yes_price == 0.5
        assert market.trade_type == "clob"

    def test_resolved_default_false(self, sample_market_dict):
        market = LimitlessMarket.from_dict(sample_market_dict)
        assert market.resolved is False

    def test_resolved_true(self, sample_market_dict):
        sample_market_dict["resolved"] = True
        market = LimitlessMarket.from_dict(sample_market_dict)
        assert market.resolved is True


# ─── Orderbook tests ──────────────────────────────────────────────────────────

class TestOrderbook:
    def test_from_dict(self, sample_orderbook_dict):
        ob = Orderbook.from_dict("test-market", sample_orderbook_dict)
        assert ob.market_slug == "test-market"
        assert len(ob.bids) == 2
        assert len(ob.asks) == 2
        assert ob.adjusted_midpoint == 0.38

    def test_bid_levels(self, sample_orderbook_dict):
        ob = Orderbook.from_dict("test", sample_orderbook_dict)
        assert ob.bids[0].price == 0.36
        assert ob.bids[0].size == 150.0

    def test_ask_levels(self, sample_orderbook_dict):
        ob = Orderbook.from_dict("test", sample_orderbook_dict)
        assert ob.asks[0].price == 0.40
        assert ob.asks[0].size == 120.0

    def test_best_bid(self, sample_orderbook_dict):
        ob = Orderbook.from_dict("test", sample_orderbook_dict)
        assert ob.best_bid == 0.36

    def test_best_ask(self, sample_orderbook_dict):
        ob = Orderbook.from_dict("test", sample_orderbook_dict)
        assert ob.best_ask == 0.40

    def test_spread(self, sample_orderbook_dict):
        ob = Orderbook.from_dict("test", sample_orderbook_dict)
        assert abs(ob.spread - 0.04) < 0.001

    def test_empty_orderbook(self):
        ob = Orderbook.from_dict("test", {
            "tokenId": "0x0",
            "bids": [], "asks": [],
            "adjustedMidpoint": 0.5, "lastTradePrice": 0.5,
            "maxSpread": 0.1, "minSize": 1.0,
        })
        assert ob.best_bid is None
        assert ob.best_ask is None
        assert ob.spread is None

    def test_token_id_parsed(self, sample_orderbook_dict):
        ob = Orderbook.from_dict("test", sample_orderbook_dict)
        assert len(ob.token_id) > 0


# ─── PlacedOrder tests ────────────────────────────────────────────────────────

class TestPlacedOrder:
    def test_from_dict_with_order_wrapper(self):
        data = {
            "order": {
                "id": "order-123",
                "side": 0,
                "makerAmount": 1000000,
                "price": 0.40,
                "status": "open",
            },
            "orderType": "GTC",
        }
        order = PlacedOrder.from_dict(data, "btc-market")
        assert order.order_id == "order-123"
        assert order.side == SIDE_BUY
        assert order.price == 0.40
        assert order.order_type == "GTC"

    def test_from_dict_direct(self):
        data = {
            "salt": 12345,
            "side": 1,
            "price": 0.60,
            "makerAmount": 5000000,
            "status": "filled",
        }
        order = PlacedOrder.from_dict(data, "eth-market")
        assert order.side == SIDE_SELL
        assert order.status == "filled"

    def test_side_constants(self):
        assert SIDE_BUY == 0
        assert SIDE_SELL == 1


# ─── Position tests ───────────────────────────────────────────────────────────

class TestPosition:
    def test_from_clob_dict(self):
        data = {
            "marketSlug": "btc-100k",
            "marketTitle": "BTC to 100k?",
            "outcome": 0,
            "tokenAmount": 10.0,
            "averagePrice": 0.36,
            "currentPrice": 0.38,
            "unrealizedPnl": 0.20,
        }
        pos = Position.from_clob_dict(data)
        assert pos.market_id == "btc-100k"
        assert pos.position_type == "YES"
        assert pos.size == 10.0
        assert pos.entry_price == 0.36

    def test_no_position_type(self):
        data = {
            "marketSlug": "btc-100k",
            "outcome": 1,
            "tokenAmount": 5.0,
            "averagePrice": 0.62,
            "currentPrice": 0.60,
            "unrealizedPnl": -0.10,
        }
        pos = Position.from_clob_dict(data)
        assert pos.position_type == "NO"


# ─── LimitlessClient tests ────────────────────────────────────────────────────

class TestLimitlessClientDemo:
    @pytest.mark.asyncio
    async def test_fetch_markets_demo(self, demo_client):
        markets = await demo_client.fetch_markets()
        assert len(markets) > 0
        for m in markets:
            assert isinstance(m, LimitlessMarket)
            assert m.slug
            assert m.title

    @pytest.mark.asyncio
    async def test_fetch_markets_returns_clob_markets(self, demo_client):
        markets = await demo_client.fetch_markets(trade_type="clob")
        clob_markets = [m for m in markets if m.trade_type == "clob"]
        assert len(clob_markets) > 0

    @pytest.mark.asyncio
    async def test_get_orderbook_demo(self, demo_client):
        ob = await demo_client.get_orderbook("bitcoin-above-100k-march-2026")
        assert ob is not None
        assert isinstance(ob, Orderbook)
        assert ob.best_bid is not None
        assert ob.best_ask is not None

    @pytest.mark.asyncio
    async def test_get_orderbook_spread_positive(self, demo_client):
        ob = await demo_client.get_orderbook("test-market")
        assert ob.spread > 0

    @pytest.mark.asyncio
    async def test_place_order_demo(self, demo_client):
        order = await demo_client.place_order(
            market_slug="bitcoin-above-100k",
            side=SIDE_BUY,
            size=5.0,
            price=0.40,
        )
        assert order is not None
        assert isinstance(order, PlacedOrder)
        assert order.side == SIDE_BUY
        assert order.size == 5.0
        assert order.price == 0.40
        assert order.status == "open"

    @pytest.mark.asyncio
    async def test_place_sell_order_demo(self, demo_client):
        order = await demo_client.place_order(
            market_slug="eth-above-4k",
            side=SIDE_SELL,
            size=3.0,
            price=0.65,
        )
        assert order is not None
        assert order.side == SIDE_SELL

    @pytest.mark.asyncio
    async def test_get_positions_demo(self, demo_client):
        positions = await demo_client.get_positions()
        assert isinstance(positions, list)

    @pytest.mark.asyncio
    async def test_login_demo_mode_with_account(self, test_account):
        client = LimitlessClient(account=test_account, demo_mode=True)
        result = await client.login()
        assert result is True
        assert client._session_cookie is not None

    @pytest.mark.asyncio
    async def test_login_demo_mode_no_account(self, demo_client):
        result = await demo_client.login()
        assert result is False

    @pytest.mark.asyncio
    async def test_fetch_markets_to_trader_markets(self, demo_client):
        markets = await demo_client.fetch_markets()
        trader_markets = [m.to_trader_market() for m in markets]
        assert all(tm.category == "limitless" for tm in trader_markets)

    def test_auth_headers_with_token(self):
        client = LimitlessClient(session_token="my-jwt-token", demo_mode=True)
        headers = client._auth_headers()
        assert headers.get("Authorization") == "Bearer my-jwt-token"

    def test_auth_headers_without_token(self, demo_client):
        headers = demo_client._auth_headers()
        assert "Authorization" not in headers
        assert headers.get("Content-Type") == "application/json"

    def test_auth_cookies_with_session(self):
        client = LimitlessClient(demo_mode=True)
        client._session_cookie = "my-cookie"
        cookies = client._auth_cookies()
        assert cookies.get("limitless_session") == "my-cookie"

    def test_auth_cookies_without_session(self, demo_client):
        assert demo_client._auth_cookies() == {}

    @pytest.mark.asyncio
    async def test_place_order_fok(self, demo_client):
        order = await demo_client.place_order(
            market_slug="test-market",
            side=SIDE_BUY,
            size=10.0,
            price=0.50,
            order_type=ORDER_TYPE_FOK,
        )
        assert order.order_type == ORDER_TYPE_FOK


class TestLimitlessClientLive:
    """Tests for live API calls (mocked)."""

    @pytest.mark.asyncio
    async def test_fetch_markets_parses_response(self, test_account):
        client = LimitlessClient(account=test_account, demo_mode=False)
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "data": [
                {
                    "id": 42,
                    "slug": "test-market",
                    "title": "Test Market",
                    "prices": [0.6, 0.4],
                    "volume": "50000",
                    "liquidity": "10000",
                    "tradeType": "clob",
                }
            ],
            "totalMarketsCount": 1,
        }

        with patch.object(client.x402, "get", AsyncMock(return_value=mock_resp)):
            markets = await client.fetch_markets()
            assert len(markets) == 1
            assert markets[0].id == 42
            assert markets[0].slug == "test-market"

    @pytest.mark.asyncio
    async def test_fetch_markets_falls_back_on_error(self, test_account):
        client = LimitlessClient(account=test_account, demo_mode=False)
        with patch.object(client.x402, "get", AsyncMock(side_effect=Exception("timeout"))):
            markets = await client.fetch_markets()
            # Should fall back to demo markets
            assert len(markets) > 0

    @pytest.mark.asyncio
    async def test_get_orderbook_404_returns_none(self, test_account):
        client = LimitlessClient(account=test_account, demo_mode=False)
        mock_resp = MagicMock()
        mock_resp.status_code = 404

        with patch.object(client.x402, "get", AsyncMock(return_value=mock_resp)):
            result = await client.get_orderbook("nonexistent-market")
            assert result is None

    @pytest.mark.asyncio
    async def test_get_orderbook_400_amm_returns_none(self, test_account):
        client = LimitlessClient(account=test_account, demo_mode=False)
        mock_resp = MagicMock()
        mock_resp.status_code = 400

        with patch.object(client.x402, "get", AsyncMock(return_value=mock_resp)):
            result = await client.get_orderbook("amm-market")
            assert result is None

    @pytest.mark.asyncio
    async def test_get_orderbook_success(self, test_account, sample_orderbook_dict):
        client = LimitlessClient(account=test_account, demo_mode=False)
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = sample_orderbook_dict

        with patch.object(client.x402, "get", AsyncMock(return_value=mock_resp)):
            ob = await client.get_orderbook("test-market")
            assert ob is not None
            assert ob.best_bid == 0.36
            assert ob.best_ask == 0.40
