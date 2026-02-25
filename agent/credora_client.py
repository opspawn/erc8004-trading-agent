"""
credora_client.py — Credora protocol risk ratings for ERC-8004 Trading Agent.

Credora provides institutional-grade risk ratings for DeFi protocols and assets.
These ratings are integrated with our Kelly Criterion position sizing:
  AAA  → Kelly multiplier 1.0  (max trust)
  AA   → Kelly multiplier 0.9
  A    → Kelly multiplier 0.8
  BBB  → Kelly multiplier 0.65
  BB   → Kelly multiplier 0.50
  B    → Kelly multiplier 0.35
  CCC  → Kelly multiplier 0.20  (near-junk)
  NR   → Kelly multiplier 0.10  (not rated — treat as junk)

If the live Credora API is unavailable or returns errors, we fall back to a
deterministic mock provider that uses protocol-name heuristics to assign tiers.

Usage:
    client = CredoraClient()
    rating = client.get_rating("Aave")
    print(rating.tier, rating.kelly_multiplier)  # CredoraRatingTier.AA  0.9

    # Direct multiplier lookup (convenience):
    mult = client.kelly_multiplier_for("Uniswap")  # 0.8 (A-rated)
"""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

from loguru import logger


# ─── Rating Tiers ─────────────────────────────────────────────────────────────

class CredoraRatingTier(str, Enum):
    """Credora risk rating tiers (investment-grade → speculative → distressed)."""
    AAA = "AAA"  # Highest quality, negligible risk
    AA  = "AA"   # Very high quality, very low risk
    A   = "A"    # High quality, low risk
    BBB = "BBB"  # Upper medium grade, moderate risk
    BB  = "BB"   # Lower medium grade, speculative elements
    B   = "B"    # Speculative, significant risk
    CCC = "CCC"  # Substantial risk, near-distressed
    NR  = "NR"   # Not rated


# ─── Kelly Multipliers ────────────────────────────────────────────────────────

KELLY_MULTIPLIERS: dict[CredoraRatingTier, float] = {
    CredoraRatingTier.AAA: 1.00,
    CredoraRatingTier.AA:  0.90,
    CredoraRatingTier.A:   0.80,
    CredoraRatingTier.BBB: 0.65,
    CredoraRatingTier.BB:  0.50,
    CredoraRatingTier.B:   0.35,
    CredoraRatingTier.CCC: 0.20,
    CredoraRatingTier.NR:  0.10,
}

# Whether this tier is considered investment grade
INVESTMENT_GRADE: frozenset[CredoraRatingTier] = frozenset({
    CredoraRatingTier.AAA,
    CredoraRatingTier.AA,
    CredoraRatingTier.A,
    CredoraRatingTier.BBB,
})


# ─── Rating Dataclass ─────────────────────────────────────────────────────────

@dataclass
class CredoraRating:
    """A Credora risk rating for a specific protocol or asset."""
    protocol: str
    tier: CredoraRatingTier
    score: float                    # 0–100 numeric score backing the tier
    source: str = "mock"            # "live" | "mock" | "cache"
    timestamp: float = field(default_factory=time.time)

    @property
    def kelly_multiplier(self) -> float:
        """Return the Kelly Criterion fractional multiplier for this rating."""
        return KELLY_MULTIPLIERS[self.tier]

    @property
    def is_investment_grade(self) -> bool:
        """True if BBB or above (safe enough for standard Kelly sizing)."""
        return self.tier in INVESTMENT_GRADE

    @property
    def age_seconds(self) -> float:
        """How many seconds ago this rating was fetched."""
        return time.time() - self.timestamp

    def __str__(self) -> str:
        return (
            f"CredoraRating({self.protocol!r} {self.tier.value} "
            f"score={self.score:.1f} kelly={self.kelly_multiplier:.2f} "
            f"source={self.source!r})"
        )


# ─── Mock Rating Provider ─────────────────────────────────────────────────────

# Well-known DeFi protocol → tier mappings (deterministic)
_WELL_KNOWN: dict[str, CredoraRatingTier] = {
    # Top-tier blue chips
    "aave":       CredoraRatingTier.AA,
    "uniswap":    CredoraRatingTier.AA,
    "compound":   CredoraRatingTier.A,
    "maker":      CredoraRatingTier.AA,
    "curve":      CredoraRatingTier.A,
    "lido":       CredoraRatingTier.A,
    "chainlink":  CredoraRatingTier.AAA,
    "ethereum":   CredoraRatingTier.AAA,
    "eth":        CredoraRatingTier.AAA,
    "bitcoin":    CredoraRatingTier.AAA,
    "btc":        CredoraRatingTier.AAA,
    "wbtc":       CredoraRatingTier.AA,
    "usdc":       CredoraRatingTier.AAA,
    "usdt":       CredoraRatingTier.AA,
    "dai":        CredoraRatingTier.AA,
    # Mid-tier
    "balancer":   CredoraRatingTier.BBB,
    "yearn":      CredoraRatingTier.BBB,
    "convex":     CredoraRatingTier.BBB,
    "frax":       CredoraRatingTier.BBB,
    "gmx":        CredoraRatingTier.BB,
    "dydx":       CredoraRatingTier.BB,
    "synthetix":  CredoraRatingTier.BBB,
    "mantle":     CredoraRatingTier.BBB,
    # Speculative
    "sushiswap":  CredoraRatingTier.BB,
    "pancakeswap":CredoraRatingTier.BB,
    "osmosis":    CredoraRatingTier.B,
    # Unknown / unrated handled by hash-based assignment
}

# Tier score midpoints (for producing a numeric score from a tier)
_TIER_SCORE_MID: dict[CredoraRatingTier, float] = {
    CredoraRatingTier.AAA: 95.0,
    CredoraRatingTier.AA:  87.5,
    CredoraRatingTier.A:   80.0,
    CredoraRatingTier.BBB: 70.0,
    CredoraRatingTier.BB:  57.5,
    CredoraRatingTier.B:   45.0,
    CredoraRatingTier.CCC: 30.0,
    CredoraRatingTier.NR:  10.0,
}

_TIER_ORDER = [
    CredoraRatingTier.CCC,
    CredoraRatingTier.B,
    CredoraRatingTier.BB,
    CredoraRatingTier.BBB,
    CredoraRatingTier.A,
    CredoraRatingTier.AA,
    CredoraRatingTier.AAA,
]


class CredoraRatingProvider:
    """
    Deterministic mock Credora rating provider.

    For well-known protocols, returns the hard-coded tier above.
    For unknown protocols, uses a seeded hash to assign a reproducible tier
    (so tests are deterministic without network access).
    """

    def get_rating(self, protocol: str) -> CredoraRating:
        """
        Return a deterministic mock Credora rating for a protocol/asset name.

        The protocol name is case-insensitive. Unknown protocols get a hash-
        based assignment so ratings are stable across calls.
        """
        key = protocol.lower().strip()
        if key in _WELL_KNOWN:
            tier = _WELL_KNOWN[key]
        else:
            tier = self._hash_assign(key)

        score = self._tier_to_score(tier, key)
        return CredoraRating(
            protocol=protocol,
            tier=tier,
            score=score,
            source="mock",
        )

    def _hash_assign(self, key: str) -> CredoraRatingTier:
        """Assign tier deterministically from name hash (skews toward mid-range)."""
        digest = int(hashlib.sha256(key.encode()).hexdigest(), 16)
        # Map to indices 1–5 (BB through AA), avoiding extremes for unknown protocols
        idx = (digest % 5) + 1   # 1=B, 2=BB, 3=BBB, 4=A, 5=AA
        return _TIER_ORDER[idx]

    def _tier_to_score(self, tier: CredoraRatingTier, key: str) -> float:
        """Compute a deterministic numeric score within a tier's range (+/- 4 pts)."""
        mid = _TIER_SCORE_MID[tier]
        digest = int(hashlib.md5(key.encode()).hexdigest(), 16)
        jitter = ((digest % 80) - 40) / 10.0   # -4.0 to +4.0
        return round(mid + jitter, 1)


# ─── Live API Client (with mock fallback) ─────────────────────────────────────

class CredoraClient:
    """
    Credora API client with automatic fallback to mock provider.

    Attempts to contact https://api.credora.io/v1/ratings for live data.
    If the request fails (network error, auth required, timeout), falls back
    transparently to CredoraRatingProvider with a warning log.

    Ratings are cached for `cache_ttl_seconds` to avoid repeated API calls.

    Args:
        api_key: Optional Credora API key for authenticated endpoints.
        cache_ttl_seconds: How long to cache ratings (default 300s = 5 min).
        timeout_seconds: HTTP request timeout.
        use_mock: Force-use mock provider without attempting live API.
    """

    BASE_URL = "https://api.credora.io/v1"

    def __init__(
        self,
        api_key: Optional[str] = None,
        cache_ttl_seconds: float = 300.0,
        timeout_seconds: float = 5.0,
        use_mock: bool = False,
    ) -> None:
        self._api_key = api_key
        self._cache_ttl = cache_ttl_seconds
        self._timeout = timeout_seconds
        self._use_mock = use_mock
        self._mock_provider = CredoraRatingProvider()
        self._cache: dict[str, CredoraRating] = {}
        logger.info(
            f"CredoraClient initialized: use_mock={use_mock} "
            f"cache_ttl={cache_ttl_seconds}s"
        )

    def get_rating(self, protocol: str) -> CredoraRating:
        """
        Fetch Credora risk rating for a protocol/asset.

        Returns a cached result if still fresh.  Falls back to mock on error.
        """
        key = protocol.lower().strip()

        # Cache hit
        if key in self._cache:
            cached = self._cache[key]
            if cached.age_seconds < self._cache_ttl:
                logger.debug(f"CredoraClient: cache hit for {protocol!r}")
                return cached

        # Force mock
        if self._use_mock:
            rating = self._mock_provider.get_rating(protocol)
            self._cache[key] = rating
            return rating

        # Attempt live API
        rating = self._try_live_api(protocol)
        if rating is None:
            logger.warning(
                f"CredoraClient: live API unavailable for {protocol!r}, "
                f"using mock fallback"
            )
            rating = self._mock_provider.get_rating(protocol)

        self._cache[key] = rating
        return rating

    def _try_live_api(self, protocol: str) -> Optional[CredoraRating]:
        """Attempt to fetch a rating from the live Credora API."""
        try:
            import urllib.request, urllib.error, json as _json
            url = f"{self.BASE_URL}/ratings/{urllib.parse.quote(protocol)}"
            req = urllib.request.Request(url)
            if self._api_key:
                req.add_header("Authorization", f"Bearer {self._api_key}")
            req.add_header("Accept", "application/json")
            with urllib.request.urlopen(req, timeout=self._timeout) as resp:
                data = _json.loads(resp.read())
                tier = CredoraRatingTier(data.get("rating", "NR"))
                score = float(data.get("score", _TIER_SCORE_MID[tier]))
                return CredoraRating(
                    protocol=protocol,
                    tier=tier,
                    score=score,
                    source="live",
                )
        except Exception as exc:
            logger.debug(f"CredoraClient: live API error for {protocol!r}: {exc}")
            return None

    def kelly_multiplier_for(self, protocol: str) -> float:
        """
        Convenience: return just the Kelly multiplier for a protocol.

        Example:
            mult = client.kelly_multiplier_for("Aave")  # 0.9
        """
        return self.get_rating(protocol).kelly_multiplier

    def get_bulk_ratings(self, protocols: list[str]) -> dict[str, CredoraRating]:
        """Fetch ratings for multiple protocols. Returns mapping name→rating."""
        return {p: self.get_rating(p) for p in protocols}

    def clear_cache(self) -> None:
        """Evict all cached ratings (useful for testing)."""
        self._cache.clear()
        logger.debug("CredoraClient: cache cleared")

    def cached_count(self) -> int:
        """Return number of entries currently in cache."""
        return len(self._cache)


# ─── Agent Credit History Tracking ───────────────────────────────────────────

from dataclasses import dataclass as _dc, field as _field


@_dc
class TradeRecord:
    """Single trade entry in an agent's credit history."""
    market_id: str
    protocol: str
    size_usdc: float
    outcome: str                    # "WIN" | "LOSS" | "PENDING"
    pnl_usdc: float
    credora_tier: CredoraRatingTier
    timestamp: float = _field(default_factory=time.time)

    @property
    def is_win(self) -> bool:
        return self.outcome == "WIN"

    @property
    def is_loss(self) -> bool:
        return self.outcome == "LOSS"


class AgentCreditHistory:
    """
    Tracks an agent's trading history and derives a credit score.

    The credit score (0–100) is computed from:
      - Win rate (40% weight)
      - Average PnL per trade (30% weight)
      - Protocol diversification (15% weight)
      - Average Credora tier of traded protocols (15% weight)

    The score maps to a CredoraRatingTier for risk-adjusted sizing.
    """

    # Score thresholds → tier (descending order)
    _SCORE_TIERS: list[tuple[float, CredoraRatingTier]] = [
        (93.0, CredoraRatingTier.AAA),
        (83.0, CredoraRatingTier.AA),
        (73.0, CredoraRatingTier.A),
        (60.0, CredoraRatingTier.BBB),
        (47.0, CredoraRatingTier.BB),
        (33.0, CredoraRatingTier.B),
        (20.0, CredoraRatingTier.CCC),
        (0.0,  CredoraRatingTier.NR),
    ]

    def __init__(self, agent_id: str) -> None:
        self.agent_id = agent_id
        self._trades: list[TradeRecord] = []
        logger.info(f"AgentCreditHistory created for agent {agent_id!r}")

    # ── Recording ──────────────────────────────────────────────────────────

    def record_trade(
        self,
        market_id: str,
        protocol: str,
        size_usdc: float,
        outcome: str,
        pnl_usdc: float,
        credora_tier: CredoraRatingTier = CredoraRatingTier.NR,
    ) -> TradeRecord:
        """Record a completed or pending trade in the agent's credit history."""
        if outcome not in ("WIN", "LOSS", "PENDING"):
            raise ValueError(f"Invalid outcome {outcome!r}: must be WIN, LOSS, or PENDING")
        rec = TradeRecord(
            market_id=market_id,
            protocol=protocol,
            size_usdc=abs(size_usdc),
            outcome=outcome,
            pnl_usdc=pnl_usdc,
            credora_tier=credora_tier,
        )
        self._trades.append(rec)
        logger.debug(
            f"AgentCreditHistory[{self.agent_id}]: recorded {outcome} "
            f"{market_id!r} pnl=${pnl_usdc:.2f} tier={credora_tier.value}"
        )
        return rec

    def update_outcome(self, market_id: str, outcome: str, pnl_usdc: float) -> bool:
        """Update the outcome of a pending trade. Returns True if found."""
        for rec in reversed(self._trades):
            if rec.market_id == market_id and rec.outcome == "PENDING":
                rec.outcome = outcome
                rec.pnl_usdc = pnl_usdc
                return True
        return False

    # ── Metrics ────────────────────────────────────────────────────────────

    @property
    def total_trades(self) -> int:
        return len(self._trades)

    @property
    def settled_trades(self) -> list[TradeRecord]:
        return [t for t in self._trades if t.outcome != "PENDING"]

    @property
    def win_rate(self) -> float:
        """Win rate across settled trades (0.0–1.0). Returns 0.5 if no settled trades."""
        settled = self.settled_trades
        if not settled:
            return 0.5
        wins = sum(1 for t in settled if t.is_win)
        return wins / len(settled)

    @property
    def total_pnl(self) -> float:
        return sum(t.pnl_usdc for t in self._trades)

    @property
    def average_pnl_per_trade(self) -> float:
        if not self._trades:
            return 0.0
        return self.total_pnl / len(self._trades)

    @property
    def protocol_diversity(self) -> int:
        """Number of unique protocols traded."""
        return len({t.protocol.lower() for t in self._trades})

    @property
    def average_credora_tier_score(self) -> float:
        """Average numeric score of Credora tiers traded (0–100)."""
        if not self._trades:
            return _TIER_SCORE_MID[CredoraRatingTier.BBB]
        scores = [_TIER_SCORE_MID[t.credora_tier] for t in self._trades]
        return sum(scores) / len(scores)

    # ── Credit Score ───────────────────────────────────────────────────────

    def compute_credit_score(self) -> float:
        """
        Compute a composite credit score (0–100).

        Components:
          - Win rate:            0–40 points  (win_rate * 40)
          - Avg PnL per trade:   0–30 points  (clamped: $-5 → 0, $+5 → 30)
          - Protocol diversity:  0–15 points  (1 → 5 pts, 5+ → 15 pts)
          - Avg Credora tier:    0–15 points  (scaled from avg_tier_score / 100 * 15)
        """
        # Component 1: win rate
        win_score = self.win_rate * 40.0

        # Component 2: average PnL (clamped to [-5, +5])
        avg_pnl = max(-5.0, min(5.0, self.average_pnl_per_trade))
        pnl_score = ((avg_pnl + 5.0) / 10.0) * 30.0

        # Component 3: protocol diversity (1=5pts, 2=8, 3=11, 4=13, 5+=15)
        div = min(self.protocol_diversity, 5)
        div_score = [0, 5, 8, 11, 13, 15][div]

        # Component 4: average Credora tier quality
        tier_score = (self.average_credora_tier_score / 100.0) * 15.0

        total = win_score + pnl_score + div_score + tier_score
        return round(min(100.0, max(0.0, total)), 2)

    def get_credit_tier(self) -> CredoraRatingTier:
        """Return the CredoraRatingTier corresponding to the agent's credit score."""
        score = self.compute_credit_score()
        for threshold, tier in self._SCORE_TIERS:
            if score >= threshold:
                return tier
        return CredoraRatingTier.NR

    def get_kelly_multiplier(self) -> float:
        """Return Kelly multiplier based on agent's own credit history."""
        return KELLY_MULTIPLIERS[self.get_credit_tier()]

    def as_credora_rating(self) -> CredoraRating:
        """Return a CredoraRating built from this agent's credit history."""
        tier = self.get_credit_tier()
        score = self.compute_credit_score()
        return CredoraRating(
            protocol=f"agent:{self.agent_id}",
            tier=tier,
            score=score,
            source="credit_history",
        )

    def get_summary(self) -> dict:
        """Return a dict summary of credit history for dashboards/logging."""
        return {
            "agent_id": self.agent_id,
            "total_trades": self.total_trades,
            "settled_trades": len(self.settled_trades),
            "win_rate": round(self.win_rate, 4),
            "total_pnl_usdc": round(self.total_pnl, 4),
            "average_pnl_per_trade": round(self.average_pnl_per_trade, 4),
            "protocol_diversity": self.protocol_diversity,
            "average_credora_tier_score": round(self.average_credora_tier_score, 2),
            "credit_score": self.compute_credit_score(),
            "credit_tier": self.get_credit_tier().value,
            "kelly_multiplier": self.get_kelly_multiplier(),
        }

    def recent_trades(self, n: int = 10) -> list[TradeRecord]:
        """Return the n most recent trades."""
        return self._trades[-n:]

    def trades_by_protocol(self, protocol: str) -> list[TradeRecord]:
        """Return all trades for a given protocol (case-insensitive)."""
        return [t for t in self._trades if t.protocol.lower() == protocol.lower()]

    def clear(self) -> None:
        """Clear all trade history."""
        self._trades.clear()
        logger.debug(f"AgentCreditHistory[{self.agent_id}]: history cleared")


# ─── Credora API Response Shape (realistic mock) ────────────────────────────

def build_credora_api_response(protocol: str, tier: CredoraRatingTier) -> dict:
    """
    Build a realistic Credora API response shape for a given protocol and tier.

    Mirrors the actual Credora API v1 response format with all expected fields.
    Used for mock testing and documentation of the API contract.
    """
    score = _TIER_SCORE_MID[tier]
    outlook_map = {
        CredoraRatingTier.AAA: "Stable",
        CredoraRatingTier.AA:  "Stable",
        CredoraRatingTier.A:   "Stable",
        CredoraRatingTier.BBB: "Stable",
        CredoraRatingTier.BB:  "Negative",
        CredoraRatingTier.B:   "Negative",
        CredoraRatingTier.CCC: "Negative Watch",
        CredoraRatingTier.NR:  "Not Rated",
    }
    sub_scores = {
        CredoraRatingTier.AAA: {"liquidity": 97, "smart_contract": 96, "governance": 95, "market": 94},
        CredoraRatingTier.AA:  {"liquidity": 90, "smart_contract": 88, "governance": 86, "market": 87},
        CredoraRatingTier.A:   {"liquidity": 82, "smart_contract": 80, "governance": 78, "market": 79},
        CredoraRatingTier.BBB: {"liquidity": 71, "smart_contract": 70, "governance": 68, "market": 70},
        CredoraRatingTier.BB:  {"liquidity": 59, "smart_contract": 56, "governance": 55, "market": 58},
        CredoraRatingTier.B:   {"liquidity": 46, "smart_contract": 44, "governance": 42, "market": 46},
        CredoraRatingTier.CCC: {"liquidity": 31, "smart_contract": 29, "governance": 27, "market": 30},
        CredoraRatingTier.NR:  {"liquidity": 0,  "smart_contract": 0,  "governance": 0,  "market": 0},
    }
    return {
        "protocol": protocol,
        "rating": tier.value,
        "score": score,
        "outlook": outlook_map[tier],
        "is_investment_grade": tier in INVESTMENT_GRADE,
        "kelly_multiplier": KELLY_MULTIPLIERS[tier],
        "sub_scores": sub_scores[tier],
        "methodology": "Credora v2.1 — Quantitative Credit Assessment",
        "valid_until": int(time.time()) + 86400,  # 24h validity
        "data_sources": ["on-chain", "oracle", "governance"],
    }


# ─── Import fix for live API path ────────────────────────────────────────────

import urllib.parse  # noqa: E402  (used inside _try_live_api)
