"""
trader.py — Trading strategy implementation.

Adapts Polymarket patterns for ERC-8004 hackathon.
Focuses on binary prediction markets with clear outcomes.
"""

import asyncio
import hashlib
import json
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Optional

import httpx
from loguru import logger


class MarketSide(str, Enum):
    YES = "YES"
    NO = "NO"


@dataclass
class Market:
    id: str
    question: str
    end_date: str
    yes_price: float        # 0.0-1.0
    no_price: float
    volume: float
    category: str


@dataclass
class TradeDecision:
    market_id: str
    question: str
    side: MarketSide
    size_usdc: float
    confidence: float       # 0.0-1.0
    reasoning: str
    estimated_return: float


@dataclass
class TradeResult:
    market_id: str
    question: str
    side: MarketSide
    size_usdc: float
    executed_at: str
    outcome: Optional[str]       # "WIN", "LOSS", "PENDING"
    pnl_usdc: float
    tx_hash: Optional[str]
    data_uri: str               # IPFS URI of trade data
    data_hash: bytes            # keccak256 of trade data

    def to_dict(self) -> dict:
        return {
            "market_id": self.market_id,
            "question": self.question,
            "side": self.side.value,
            "size_usdc": self.size_usdc,
            "executed_at": self.executed_at,
            "outcome": self.outcome,
            "pnl_usdc": self.pnl_usdc,
            "tx_hash": self.tx_hash,
        }


class TradingStrategy:
    """
    Simple prediction market trading strategy.

    Strategy: Look for markets where the current price significantly
    deviates from historical base rates or where there's clear
    information advantage.
    """

    MIN_VOLUME = 1000           # $1,000 minimum market volume
    MAX_POSITION_USDC = 10      # Max $10 per trade
    MIN_EDGE = 0.05             # Min 5% edge required
    MIN_CONFIDENCE = 0.6        # Min 60% confidence

    def __init__(self, dry_run: bool = True):
        self.dry_run = dry_run
        self.trades: list[TradeResult] = []
        logger.info(f"Strategy initialized (dry_run={dry_run})")

    def evaluate_market(self, market: Market) -> Optional[TradeDecision]:
        """
        Evaluate a market and return a trade decision if there's edge.

        Simple heuristics:
        - If YES price < 0.3 for a market that historically resolves ~50%,
          there may be value in YES
        - If YES price > 0.8 but it's a speculative market, NO may have value
        """
        if market.volume < self.MIN_VOLUME:
            logger.debug(f"Skipping {market.id}: volume ${market.volume:.0f} < ${self.MIN_VOLUME}")
            return None

        # Simple edge calculation
        # In production: use ML model, sentiment analysis, base rates
        yes_price = market.yes_price
        estimated_prob = self._estimate_probability(market)

        edge_yes = estimated_prob - yes_price
        edge_no = (1 - estimated_prob) - market.no_price

        if abs(edge_yes) < self.MIN_EDGE and abs(edge_no) < self.MIN_EDGE:
            logger.debug(f"No edge in market {market.id}: edge_yes={edge_yes:.3f}")
            return None

        if edge_yes > edge_no and edge_yes >= self.MIN_EDGE:
            side = MarketSide.YES
            edge = edge_yes
            confidence = min(0.5 + edge * 2, 0.95)
            reasoning = f"YES underpriced: market={yes_price:.2f}, estimated={estimated_prob:.2f}, edge={edge:.3f}"
        elif edge_no >= self.MIN_EDGE:
            side = MarketSide.NO
            edge = edge_no
            confidence = min(0.5 + edge * 2, 0.95)
            reasoning = f"NO underpriced: market={market.no_price:.2f}, estimated={1-estimated_prob:.2f}, edge={edge:.3f}"
        else:
            return None

        if confidence < self.MIN_CONFIDENCE:
            return None

        # Kelly criterion sizing (conservative)
        kelly = edge / (1 - (1 - edge))
        size = min(kelly * 100, self.MAX_POSITION_USDC)  # Cap at max

        return TradeDecision(
            market_id=market.id,
            question=market.question,
            side=side,
            size_usdc=round(size, 2),
            confidence=confidence,
            reasoning=reasoning,
            estimated_return=round(edge * size, 3),
        )

    def _estimate_probability(self, market: Market) -> float:
        """
        Estimate true probability for a market.
        Placeholder — replace with ML model or signal aggregation.
        """
        # Simple: use current price with slight mean reversion
        base = 0.5  # Prior
        weight = 0.3  # How much weight to give prior vs market
        return weight * base + (1 - weight) * market.yes_price

    async def execute_trade(
        self, decision: TradeDecision
    ) -> TradeResult:
        """
        Execute a trade on Polymarket.
        In dry_run mode, simulates without actually trading.
        """
        now = datetime.now(timezone.utc).isoformat()

        if self.dry_run:
            logger.info(
                f"[DRY RUN] Would trade {decision.side} "
                f"${decision.size_usdc} on '{decision.question[:60]}...'"
            )
            trade_data = {
                "market_id": decision.market_id,
                "side": decision.side.value,
                "size_usdc": decision.size_usdc,
                "executed_at": now,
                "dry_run": True,
                "reasoning": decision.reasoning,
            }
            data_json = json.dumps(trade_data, separators=(",", ":"))
            data_hash = hashlib.sha256(data_json.encode()).digest()

            result = TradeResult(
                market_id=decision.market_id,
                question=decision.question,
                side=decision.side,
                size_usdc=decision.size_usdc,
                executed_at=now,
                outcome="PENDING",
                pnl_usdc=0.0,
                tx_hash=None,
                data_uri=f"data:application/json,{data_json}",
                data_hash=data_hash,
            )
        else:
            # Production: call Polymarket CLOB API
            result = await self._execute_polymarket(decision, now)

        self.trades.append(result)
        logger.info(
            f"Trade logged: {result.side} ${result.size_usdc} "
            f"on {result.market_id} at {result.executed_at}"
        )
        return result

    async def _execute_polymarket(
        self, decision: TradeDecision, executed_at: str
    ) -> TradeResult:
        """Execute trade via Polymarket CLOB API (production)."""
        # TODO: Implement Polymarket CLOB integration
        # Reference: https://docs.polymarket.com/
        raise NotImplementedError(
            "Polymarket CLOB integration pending. Set DRY_RUN=true for now."
        )

    def get_performance_summary(self) -> dict:
        """Return a summary of trading performance."""
        total_trades = len(self.trades)
        settled = [t for t in self.trades if t.outcome in ("WIN", "LOSS")]
        wins = sum(1 for t in settled if t.outcome == "WIN")
        total_pnl = sum(t.pnl_usdc for t in settled)

        return {
            "total_trades": total_trades,
            "settled": len(settled),
            "wins": wins,
            "losses": len(settled) - wins,
            "win_rate": wins / len(settled) if settled else 0,
            "total_pnl_usdc": total_pnl,
        }
