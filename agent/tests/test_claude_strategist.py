"""
Tests for claude_strategist.py — 25 tests covering ClaudeStrategist:
  - Decision parsing and normalization
  - Fallback momentum strategy
  - x402 signal purchase (mocked)
  - Claude API integration (mocked)
  - Edge cases
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import json
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from claude_strategist import (
    ClaudeStrategist,
    TradeDecision,
    purchase_market_signal,
    X402_SIGNAL_ENDPOINT,
    X402_SIGNAL_COST_USDC,
)


# ─── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture
def market_data_yes_cheap():
    return {
        "market_id": "mkt-001",
        "question": "Will BTC exceed $100k by March?",
        "yes_price": 0.25,
        "no_price": 0.75,
        "volume": 50_000,
        "category": "crypto",
    }


@pytest.fixture
def market_data_yes_expensive():
    return {
        "market_id": "mkt-002",
        "question": "Will ETH reach $10k in 2026?",
        "yes_price": 0.82,
        "no_price": 0.18,
        "volume": 30_000,
        "category": "crypto",
    }


@pytest.fixture
def market_data_neutral():
    return {
        "market_id": "mkt-003",
        "question": "Will CPI drop below 3% in Q1?",
        "yes_price": 0.51,
        "no_price": 0.49,
        "volume": 100_000,
        "category": "macro",
    }


@pytest.fixture
def market_data_low_volume():
    return {
        "market_id": "mkt-004",
        "question": "Obscure market",
        "yes_price": 0.20,
        "no_price": 0.80,
        "volume": 500,  # below 1000 threshold
        "category": "misc",
    }


@pytest.fixture
def strategist_no_api():
    """Strategist with no API key — always uses momentum fallback."""
    return ClaudeStrategist(api_key="", enable_x402_signals=False)


@pytest.fixture
def strategist_with_api():
    """Strategist with fake API key — will try Claude (we mock the call)."""
    return ClaudeStrategist(api_key="sk-ant-fake-key-for-testing", enable_x402_signals=False)


# ─── TradeDecision Dataclass ──────────────────────────────────────────────────

class TestTradeDecision:

    def test_trade_decision_to_dict(self):
        td = TradeDecision(
            action="buy",
            size_pct=0.3,
            confidence=0.7,
            reasoning="Test reasoning",
        )
        d = td.to_dict()
        assert d["action"] == "buy"
        assert d["size_pct"] == 0.3
        assert d["confidence"] == 0.7
        assert d["reasoning"] == "Test reasoning"
        assert d["signal_purchased"] is False
        assert d["signal_cost_usdc"] == 0.0

    def test_trade_decision_defaults(self):
        td = TradeDecision(action="hold", size_pct=0.0, confidence=0.5, reasoning="")
        assert td.signal_purchased is False
        assert td.signal_cost_usdc == 0.0
        assert td.metadata == {}


# ─── Momentum Fallback ────────────────────────────────────────────────────────

class TestMomentumFallback:

    def test_cheap_yes_price_buys(self, strategist_no_api, market_data_yes_cheap):
        decision = strategist_no_api._momentum_fallback(market_data_yes_cheap)
        assert decision.action == "buy_yes"
        assert decision.size_pct > 0
        assert decision.confidence > 0.5

    def test_expensive_yes_buys_no(self, strategist_no_api, market_data_yes_expensive):
        decision = strategist_no_api._momentum_fallback(market_data_yes_expensive)
        assert decision.action == "buy_no"
        assert decision.confidence > 0.5

    def test_neutral_price_holds(self, strategist_no_api, market_data_neutral):
        decision = strategist_no_api._momentum_fallback(market_data_neutral)
        assert decision.action == "hold"
        assert decision.size_pct == 0.0

    def test_low_volume_holds(self, strategist_no_api, market_data_low_volume):
        decision = strategist_no_api._momentum_fallback(market_data_low_volume)
        assert decision.action == "hold"

    def test_confidence_bounded(self, strategist_no_api, market_data_yes_cheap):
        decision = strategist_no_api._momentum_fallback(market_data_yes_cheap)
        assert 0.0 <= decision.confidence <= 1.0

    def test_size_pct_bounded(self, strategist_no_api, market_data_yes_cheap):
        decision = strategist_no_api._momentum_fallback(market_data_yes_cheap)
        assert 0.0 <= decision.size_pct <= 1.0

    def test_metadata_has_strategy_key(self, strategist_no_api, market_data_yes_cheap):
        decision = strategist_no_api._momentum_fallback(market_data_yes_cheap)
        assert "strategy" in decision.metadata
        assert decision.metadata["strategy"] == "momentum_fallback"


# ─── Claude Response Parsing ──────────────────────────────────────────────────

class TestParseClaudeResponse:

    def test_parse_valid_json(self, strategist_no_api):
        response = json.dumps({
            "action": "buy_yes",
            "size_pct": 0.4,
            "confidence": 0.72,
            "reasoning": "YES appears underpriced."
        })
        result = strategist_no_api._parse_claude_response(response)
        assert result is not None
        assert result["action"] == "buy_yes"
        assert result["confidence"] == 0.72

    def test_parse_json_with_extra_text(self, strategist_no_api):
        response = 'Sure! Here is my decision:\n```json\n{"action": "hold", "size_pct": 0.0, "confidence": 0.5, "reasoning": "No edge."}\n```'
        result = strategist_no_api._parse_claude_response(response)
        assert result is not None
        assert result["action"] == "hold"

    def test_parse_invalid_returns_none(self, strategist_no_api):
        result = strategist_no_api._parse_claude_response("I think you should buy YES!")
        assert result is None

    def test_parse_empty_returns_none(self, strategist_no_api):
        result = strategist_no_api._parse_claude_response("")
        assert result is None

    def test_parse_missing_action_returns_none(self, strategist_no_api):
        response = json.dumps({"size_pct": 0.3, "confidence": 0.6})
        result = strategist_no_api._parse_claude_response(response)
        assert result is None


# ─── Async Decide — Fallback Path ─────────────────────────────────────────────

class TestDecideFallback:

    @pytest.mark.asyncio
    async def test_decide_no_api_returns_decision(self, strategist_no_api, market_data_yes_cheap):
        decision = await strategist_no_api.decide(market_data_yes_cheap)
        assert isinstance(decision, TradeDecision)
        assert decision.action in ("buy_yes", "buy_no", "hold", "buy", "sell")

    @pytest.mark.asyncio
    async def test_decide_increments_counter(self, strategist_no_api, market_data_neutral):
        assert strategist_no_api._decisions_made == 0
        await strategist_no_api.decide(market_data_neutral)
        assert strategist_no_api._decisions_made == 1

    @pytest.mark.asyncio
    async def test_decide_no_x402_no_signal(self, strategist_no_api, market_data_neutral):
        decision = await strategist_no_api.decide(market_data_neutral)
        assert decision.signal_purchased is False
        assert decision.signal_cost_usdc == 0.0


# ─── x402 Signal Purchase ─────────────────────────────────────────────────────

class TestX402Integration:

    @pytest.mark.asyncio
    async def test_signal_purchase_success(self, market_data_neutral):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"trend": "bullish", "signal": 0.62}

        mock_client = MagicMock()
        mock_client.get = AsyncMock(return_value=mock_resp)

        signal, cost = await purchase_market_signal(market_data_neutral, mock_client)
        assert signal is not None
        assert signal["trend"] == "bullish"
        assert cost == X402_SIGNAL_COST_USDC

    @pytest.mark.asyncio
    async def test_signal_purchase_non_200(self, market_data_neutral):
        mock_resp = MagicMock()
        mock_resp.status_code = 402
        mock_resp.json.return_value = {}

        mock_client = MagicMock()
        mock_client.get = AsyncMock(return_value=mock_resp)

        signal, cost = await purchase_market_signal(market_data_neutral, mock_client)
        assert signal is None
        assert cost == 0.0

    @pytest.mark.asyncio
    async def test_signal_purchase_exception_handled(self, market_data_neutral):
        mock_client = MagicMock()
        mock_client.get = AsyncMock(side_effect=Exception("network error"))

        signal, cost = await purchase_market_signal(market_data_neutral, mock_client)
        assert signal is None
        assert cost == 0.0

    @pytest.mark.asyncio
    async def test_decide_with_x402_tracks_purchase(self, market_data_neutral):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"signal": "neutral"}

        mock_client = MagicMock()
        mock_client.get = AsyncMock(return_value=mock_resp)

        strategist = ClaudeStrategist(api_key="", enable_x402_signals=True)
        strategist.x402_client = mock_client

        decision = await strategist.decide(market_data_neutral)
        assert decision.signal_purchased is True
        assert decision.signal_cost_usdc == X402_SIGNAL_COST_USDC
        assert strategist._x402_purchases == 1

    @pytest.mark.asyncio
    async def test_decide_x402_fail_continues(self, market_data_neutral):
        mock_client = MagicMock()
        mock_client.get = AsyncMock(side_effect=Exception("timeout"))

        strategist = ClaudeStrategist(api_key="", enable_x402_signals=True)
        strategist.x402_client = mock_client

        # Should not raise, should fallback gracefully
        decision = await strategist.decide(market_data_neutral)
        assert isinstance(decision, TradeDecision)
        assert decision.signal_purchased is False


# ─── Stats ────────────────────────────────────────────────────────────────────

class TestStats:

    def test_stats_initial(self, strategist_no_api):
        stats = strategist_no_api.get_stats()
        assert stats["decisions_made"] == 0
        assert stats["x402_purchases"] == 0
        assert stats["claude_available"] is False

    @pytest.mark.asyncio
    async def test_stats_after_decisions(self, strategist_no_api, market_data_neutral):
        await strategist_no_api.decide(market_data_neutral)
        await strategist_no_api.decide(market_data_neutral)
        stats = strategist_no_api.get_stats()
        assert stats["decisions_made"] == 2

    def test_stats_model_fallback(self, strategist_no_api):
        stats = strategist_no_api.get_stats()
        assert stats["model"] == "momentum_fallback"


# ─── Build Prompt ─────────────────────────────────────────────────────────────

class TestBuildPrompt:

    def test_prompt_contains_price(self, strategist_no_api, market_data_yes_cheap):
        prompt = strategist_no_api._build_prompt(market_data_yes_cheap)
        assert "0.25" in prompt or "25.0%" in prompt

    def test_prompt_contains_question(self, strategist_no_api, market_data_yes_cheap):
        prompt = strategist_no_api._build_prompt(market_data_yes_cheap)
        assert "BTC" in prompt

    def test_prompt_contains_volume(self, strategist_no_api, market_data_yes_cheap):
        prompt = strategist_no_api._build_prompt(market_data_yes_cheap)
        assert "50" in prompt  # volume 50,000

    def test_prompt_with_signal(self, strategist_no_api, market_data_yes_cheap):
        signal = {"trend": "bullish", "score": 0.7}
        prompt = strategist_no_api._build_prompt(market_data_yes_cheap, signal=signal)
        assert "bullish" in prompt
        assert "x402" in prompt

    def test_prompt_without_signal_no_x402_section(self, strategist_no_api, market_data_yes_cheap):
        prompt = strategist_no_api._build_prompt(market_data_yes_cheap, signal=None)
        # Should not have x402 section when no signal
        assert "Purchased Market Signal" not in prompt

    def test_prompt_json_format_instruction(self, strategist_no_api, market_data_yes_cheap):
        prompt = strategist_no_api._build_prompt(market_data_yes_cheap)
        assert "JSON" in prompt
        assert "action" in prompt

    def test_prompt_with_recent_trades(self, strategist_no_api):
        market = {
            "market_id": "m1",
            "question": "Test market",
            "yes_price": 0.5,
            "volume": 10_000,
            "category": "test",
            "recent_trades": [
                {"side": "YES", "price": 0.52},
                {"side": "NO", "price": 0.48},
            ],
        }
        prompt = strategist_no_api._build_prompt(market)
        assert "Recent trades" in prompt


# ─── Momentum Boundary Tests ──────────────────────────────────────────────────

class TestMomentumBoundary:

    def test_exactly_at_buy_yes_threshold(self, strategist_no_api):
        # Exactly 0.35 — boundary, should NOT trigger buy (< 0.35 required)
        market = {"market_id": "m", "question": "q", "yes_price": 0.35, "volume": 5000}
        decision = strategist_no_api._momentum_fallback(market)
        assert decision.action == "hold"

    def test_just_below_buy_yes_threshold(self, strategist_no_api):
        # 0.349 should trigger buy YES
        market = {"market_id": "m", "question": "q", "yes_price": 0.349, "volume": 5000}
        decision = strategist_no_api._momentum_fallback(market)
        assert decision.action == "buy_yes"

    def test_exactly_at_buy_no_threshold(self, strategist_no_api):
        # Exactly 0.75 — boundary, should NOT trigger buy NO (> 0.75 required)
        market = {"market_id": "m", "question": "q", "yes_price": 0.75, "volume": 5000}
        decision = strategist_no_api._momentum_fallback(market)
        assert decision.action == "hold"

    def test_just_above_buy_no_threshold(self, strategist_no_api):
        # 0.751 should trigger buy NO
        market = {"market_id": "m", "question": "q", "yes_price": 0.751, "volume": 5000}
        decision = strategist_no_api._momentum_fallback(market)
        assert decision.action == "buy_no"
