"""
claude_strategist.py — Claude-powered trade decision engine for ERC-8004 Trading Agent.

Uses Claude (claude-haiku-4-5-20251001) to analyze market data and generate trade
decisions with confidence scores and reasoning.

KEY DIFFERENTIATOR: Before deciding, the agent autonomously purchases market
signals via x402 micropayment protocol (http://api.opspawn.com/signals).
No competitor combines x402 + ERC-8004 — this is our unique angle.

Usage:
    strategist = ClaudeStrategist()
    decision = await strategist.decide(market_data)
    if decision.action != "hold":
        # execute trade
        pass
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any, Optional

import httpx
from loguru import logger

# ─── Data Classes ────────────────────────────────────────────────────────────

@dataclass
class TradeDecision:
    """Decision returned by ClaudeStrategist."""
    action: str                      # "buy", "sell", "hold"
    size_pct: float                  # 0.0–1.0 fraction of max allowed position
    confidence: float                # 0.0–1.0
    reasoning: str
    signal_purchased: bool = False   # True if x402 signal was bought
    signal_cost_usdc: float = 0.0    # Amount paid for signal
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "action": self.action,
            "size_pct": round(self.size_pct, 4),
            "confidence": round(self.confidence, 4),
            "reasoning": self.reasoning,
            "signal_purchased": self.signal_purchased,
            "signal_cost_usdc": self.signal_cost_usdc,
            "metadata": self.metadata,
        }


# ─── x402 Signal Purchase ────────────────────────────────────────────────────

X402_SIGNAL_ENDPOINT = "http://api.opspawn.com/signals"
X402_SIGNAL_COST_USDC = 0.001       # $0.001 per signal


async def purchase_market_signal(
    market_data: dict,
    x402_client: Any = None,
) -> tuple[Optional[dict], float]:
    """
    Attempt to purchase a market signal from the x402 endpoint.

    Args:
        market_data: Current market state
        x402_client: Optional X402Client instance. If None, uses direct httpx.

    Returns:
        (signal_data, cost_usdc) — signal_data is None if purchase failed.
    """
    try:
        if x402_client is not None:
            # Use the x402 client which handles 402 → payment → retry
            resp = await x402_client.get(
                X402_SIGNAL_ENDPOINT,
                params={"market_id": market_data.get("market_id", "unknown")},
            )
        else:
            async with httpx.AsyncClient(timeout=5.0) as http:
                resp = await http.get(
                    X402_SIGNAL_ENDPOINT,
                    params={"market_id": market_data.get("market_id", "unknown")},
                )

        if resp.status_code == 200:
            signal = resp.json()
            logger.info(f"x402: Paid ${X402_SIGNAL_COST_USDC:.4f} for market signal")
            return signal, X402_SIGNAL_COST_USDC
        else:
            logger.debug(f"x402: Signal endpoint returned {resp.status_code}, using fallback")
            return None, 0.0

    except Exception as e:
        logger.debug(f"x402: Signal purchase failed ({e}), using fallback")
        return None, 0.0


# ─── Claude Strategist ───────────────────────────────────────────────────────

class ClaudeStrategist:
    """
    Claude-powered trade decision engine.

    Decision flow:
      1. Attempt x402 signal purchase (autonomous data buying)
      2. Build prompt with market data + signal (if purchased)
      3. Call Claude claude-haiku-4-5-20251001 (cheapest/fastest)
      4. Parse decision (action, size_pct, confidence, reasoning)
      5. Fallback to momentum strategy if Claude unavailable

    The x402 purchase is logged in TradeDecision.metadata — judges can
    see the agent autonomously spending to improve its decisions.
    """

    MODEL = "claude-haiku-4-5-20251001"
    MAX_TOKENS = 512

    def __init__(
        self,
        api_key: Optional[str] = None,
        x402_client: Any = None,
        enable_x402_signals: bool = True,
    ) -> None:
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY", "")
        self.x402_client = x402_client
        self.enable_x402_signals = enable_x402_signals
        self._decisions_made: int = 0
        self._x402_purchases: int = 0

        # Try to import Anthropic SDK
        self._anthropic_available = False
        if self.api_key:
            try:
                import anthropic  # noqa: F401
                self._anthropic_available = True
                logger.info(f"ClaudeStrategist: Anthropic SDK ready (model={self.MODEL})")
            except ImportError:
                logger.warning("ClaudeStrategist: anthropic package not installed, using fallback")
        else:
            logger.info("ClaudeStrategist: no API key, using momentum fallback")

    def _build_prompt(self, market_data: dict, signal: Optional[dict] = None) -> str:
        """Build the decision prompt for Claude."""
        price = market_data.get("yes_price", market_data.get("price", 0.5))
        volume = market_data.get("volume", 0)
        spread = market_data.get("spread", abs(price - (1 - price)))
        recent_trades = market_data.get("recent_trades", [])
        question = market_data.get("question", market_data.get("market_id", "unknown market"))
        category = market_data.get("category", "general")

        signal_section = ""
        if signal:
            signal_section = f"""
## Purchased Market Signal (via x402 micropayment)
```json
{json.dumps(signal, indent=2)}
```
"""

        recent_str = ""
        if recent_trades:
            recent_str = f"\nRecent trades: {json.dumps(recent_trades[-5:])}"

        return f"""You are an autonomous prediction market trading agent. Analyze the following market and decide whether to buy YES, buy NO, or hold.

## Market Data
- Question: {question}
- Category: {category}
- YES price: {price:.4f} (implied probability: {price*100:.1f}%)
- NO price: {(1-price):.4f}
- Volume: ${volume:,.0f}
- Spread: {spread:.4f}{recent_str}
{signal_section}
## Instructions
Based on the market data, provide a trading decision. Consider:
1. Is the current price fair or does it deviate from reasonable estimates?
2. Is there sufficient volume to support a position?
3. What is your confidence level?

Respond ONLY with valid JSON in this exact format:
{{
  "action": "buy_yes" | "buy_no" | "hold",
  "size_pct": <float 0.0-1.0>,
  "confidence": <float 0.0-1.0>,
  "reasoning": "<one sentence explanation>"
}}"""

    def _parse_claude_response(self, text: str) -> Optional[dict]:
        """Extract and parse JSON from Claude's response."""
        # Try to extract JSON from the response
        text = text.strip()

        # Find JSON block
        start = text.find("{")
        end = text.rfind("}") + 1
        if start == -1 or end == 0:
            return None

        try:
            data = json.loads(text[start:end])
            # Validate required fields
            if "action" not in data or "confidence" not in data:
                return None
            return data
        except json.JSONDecodeError:
            return None

    def _momentum_fallback(self, market_data: dict) -> TradeDecision:
        """
        Simple momentum strategy as fallback when Claude is unavailable.

        Logic:
        - If YES price < 0.35, moderate confidence buy YES (underpriced)
        - If YES price > 0.75, moderate confidence buy NO (overpriced)
        - Otherwise, hold
        """
        price = market_data.get("yes_price", market_data.get("price", 0.5))
        volume = market_data.get("volume", 0)

        # Need minimum volume
        if volume < 1000:
            return TradeDecision(
                action="hold",
                size_pct=0.0,
                confidence=0.5,
                reasoning="Insufficient volume for momentum strategy",
                metadata={"strategy": "momentum_fallback", "reason": "low_volume"},
            )

        if price < 0.35:
            confidence = 0.5 + (0.35 - price) * 2  # More confident the lower the price
            confidence = min(confidence, 0.75)
            return TradeDecision(
                action="buy_yes",
                size_pct=0.3,
                confidence=confidence,
                reasoning=f"YES appears underpriced at {price:.2f} (momentum: mean reversion expected)",
                metadata={"strategy": "momentum_fallback", "signal": "oversold"},
            )
        elif price > 0.75:
            confidence = 0.5 + (price - 0.75) * 2
            confidence = min(confidence, 0.75)
            return TradeDecision(
                action="buy_no",
                size_pct=0.3,
                confidence=confidence,
                reasoning=f"YES appears overpriced at {price:.2f} (momentum: mean reversion expected)",
                metadata={"strategy": "momentum_fallback", "signal": "overbought"},
            )
        else:
            return TradeDecision(
                action="hold",
                size_pct=0.0,
                confidence=0.5,
                reasoning=f"No clear edge at YES price {price:.2f}",
                metadata={"strategy": "momentum_fallback", "signal": "neutral"},
            )

    async def decide(self, market_data: dict) -> TradeDecision:
        """
        Generate a trade decision for the given market.

        Args:
            market_data: Dict with keys: market_id, question, yes_price,
                         no_price, volume, category, recent_trades (optional)

        Returns:
            TradeDecision with action, size_pct, confidence, reasoning
        """
        self._decisions_made += 1

        # Step 1: Purchase market signal via x402 (our differentiator)
        signal = None
        signal_cost = 0.0
        signal_purchased = False

        if self.enable_x402_signals:
            signal, signal_cost = await purchase_market_signal(
                market_data, self.x402_client
            )
            if signal is not None:
                signal_purchased = True
                self._x402_purchases += 1
                logger.info(
                    f"ClaudeStrategist: x402 signal purchased for "
                    f"{market_data.get('market_id', 'market')} "
                    f"(${signal_cost:.4f} USDC)"
                )

        # Step 2: Try Claude API
        if self._anthropic_available:
            try:
                decision = await self._call_claude(market_data, signal)
                decision.signal_purchased = signal_purchased
                decision.signal_cost_usdc = signal_cost
                decision.metadata["x402_purchases_session"] = self._x402_purchases
                return decision
            except Exception as e:
                logger.warning(f"ClaudeStrategist: Claude API failed ({e}), using fallback")

        # Step 3: Fallback to momentum
        decision = self._momentum_fallback(market_data)
        decision.signal_purchased = signal_purchased
        decision.signal_cost_usdc = signal_cost
        if signal:
            decision.metadata["signal_data"] = signal
        return decision

    async def _call_claude(
        self, market_data: dict, signal: Optional[dict]
    ) -> TradeDecision:
        """Call Claude API and parse the response."""
        import anthropic

        prompt = self._build_prompt(market_data, signal)

        client = anthropic.Anthropic(api_key=self.api_key)

        message = client.messages.create(
            model=self.MODEL,
            max_tokens=self.MAX_TOKENS,
            messages=[{"role": "user", "content": prompt}],
        )

        response_text = message.content[0].text
        logger.debug(f"ClaudeStrategist: raw response: {response_text[:200]}")

        parsed = self._parse_claude_response(response_text)
        if not parsed:
            logger.warning(
                f"ClaudeStrategist: could not parse Claude response, using fallback. "
                f"Response: {response_text[:100]}"
            )
            return self._momentum_fallback(market_data)

        # Normalize action
        action = parsed.get("action", "hold").lower()
        if action == "buy_yes":
            action = "buy"
        elif action == "buy_no":
            action = "sell"
        elif action not in ("buy", "sell", "hold"):
            action = "hold"

        size_pct = float(parsed.get("size_pct", 0.3))
        size_pct = max(0.0, min(1.0, size_pct))

        confidence = float(parsed.get("confidence", 0.5))
        confidence = max(0.0, min(1.0, confidence))

        reasoning = str(parsed.get("reasoning", "No reasoning provided"))

        return TradeDecision(
            action=action,
            size_pct=size_pct,
            confidence=confidence,
            reasoning=reasoning,
            metadata={
                "model": self.MODEL,
                "raw_action": parsed.get("action"),
                "source": "claude",
            },
        )

    def get_stats(self) -> dict:
        """Return strategist usage statistics."""
        return {
            "decisions_made": self._decisions_made,
            "x402_purchases": self._x402_purchases,
            "claude_available": self._anthropic_available,
            "model": self.MODEL if self._anthropic_available else "momentum_fallback",
        }
