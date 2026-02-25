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


# ─── Import fix for live API path ────────────────────────────────────────────

import urllib.parse  # noqa: E402  (used inside _try_live_api)
