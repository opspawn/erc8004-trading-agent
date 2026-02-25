"""
agent_coordinator.py — Multi-Agent Coordinator for ERC-8004 Trading System.

Manages a pool of Claude strategist instances with different risk profiles,
implements ensemble voting for trade signals, and tracks per-agent performance.

Classes:
    RiskProfile        — enum for conservative/moderate/aggressive
    AgentConfig        — configuration per agent instance
    AgentSignal        — signal from one agent instance
    EnsembleSignal     — aggregated weighted vote across all agents
    AgentPerformance   — P&L and accuracy tracking per agent
    AgentPerformanceTracker — manages performance records, updates weights
    AgentPool          — manages multiple strategist instances
    MultiAgentCoordinator  — top-level orchestrator, produces consensus decisions
"""

from __future__ import annotations

import asyncio
import statistics
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Optional
from unittest.mock import AsyncMock


# ─── Risk Profiles ────────────────────────────────────────────────────────────


class RiskProfile(str, Enum):
    CONSERVATIVE = "conservative"
    MODERATE = "moderate"
    AGGRESSIVE = "aggressive"


# ─── Data Classes ─────────────────────────────────────────────────────────────


@dataclass
class AgentConfig:
    """Configuration for one agent instance."""
    agent_id: str
    risk_profile: RiskProfile
    initial_weight: float = 1.0
    max_position_pct: float = 0.10      # max position as fraction of portfolio
    min_confidence: float = 0.60        # minimum confidence to signal buy/sell

    def __post_init__(self):
        if self.initial_weight <= 0:
            raise ValueError("initial_weight must be positive")
        if not 0 < self.max_position_pct <= 1:
            raise ValueError("max_position_pct must be in (0, 1]")
        if not 0 <= self.min_confidence <= 1:
            raise ValueError("min_confidence must be in [0, 1]")

    @classmethod
    def conservative(cls, agent_id: str) -> "AgentConfig":
        return cls(
            agent_id=agent_id,
            risk_profile=RiskProfile.CONSERVATIVE,
            initial_weight=1.0,
            max_position_pct=0.05,
            min_confidence=0.75,
        )

    @classmethod
    def moderate(cls, agent_id: str) -> "AgentConfig":
        return cls(
            agent_id=agent_id,
            risk_profile=RiskProfile.MODERATE,
            initial_weight=1.0,
            max_position_pct=0.10,
            min_confidence=0.65,
        )

    @classmethod
    def aggressive(cls, agent_id: str) -> "AgentConfig":
        return cls(
            agent_id=agent_id,
            risk_profile=RiskProfile.AGGRESSIVE,
            initial_weight=1.0,
            max_position_pct=0.20,
            min_confidence=0.55,
        )


@dataclass
class AgentSignal:
    """Signal produced by one agent instance."""
    agent_id: str
    action: str          # "buy" | "sell" | "hold"
    confidence: float    # 0.0–1.0
    size_pct: float      # suggested position size (fraction of portfolio)
    reasoning: str = ""
    timestamp: float = field(default_factory=time.time)

    def __post_init__(self):
        if self.action not in ("buy", "sell", "hold"):
            raise ValueError(f"Invalid action: {self.action}")
        if not 0 <= self.confidence <= 1:
            raise ValueError("confidence must be in [0, 1]")
        if not 0 <= self.size_pct <= 1:
            raise ValueError("size_pct must be in [0, 1]")

    @property
    def is_directional(self) -> bool:
        return self.action in ("buy", "sell")


@dataclass
class EnsembleSignal:
    """
    Aggregated signal from all agents via weighted voting.

    Consensus is reached when enough weighted votes agree on direction
    (above the consensus_threshold).
    """
    action: str               # final consensus action
    consensus_weight: float   # total weight of agreeing agents / total weight
    mean_confidence: float    # weighted mean confidence
    mean_size_pct: float      # weighted mean position size
    agent_votes: dict[str, str]  # agent_id → action voted
    dissenting_agents: list[str]  # agents that voted differently
    has_consensus: bool       # True if consensus_weight >= threshold
    reasoning: str = ""

    @property
    def is_actionable(self) -> bool:
        """True if we should act (consensus reached and not hold)."""
        return self.has_consensus and self.action != "hold"


@dataclass
class AgentPerformance:
    """Per-agent P&L and accuracy tracking."""
    agent_id: str
    weight: float = 1.0
    total_trades: int = 0
    wins: int = 0
    losses: int = 0
    total_pnl: float = 0.0
    recent_pnl: list[float] = field(default_factory=list)
    _max_recent: int = 20  # rolling window size

    @property
    def win_rate(self) -> float:
        if self.total_trades == 0:
            return 0.0
        return self.wins / self.total_trades

    @property
    def avg_pnl(self) -> float:
        if self.total_trades == 0:
            return 0.0
        return self.total_pnl / self.total_trades

    @property
    def recent_avg_pnl(self) -> float:
        if not self.recent_pnl:
            return 0.0
        return statistics.mean(self.recent_pnl)

    @property
    def sharpe_proxy(self) -> float:
        """Simple Sharpe-like metric using recent PnL."""
        if len(self.recent_pnl) < 2:
            return 0.0
        mean = statistics.mean(self.recent_pnl)
        stdev = statistics.stdev(self.recent_pnl)
        if stdev == 0:
            return 0.0
        return mean / stdev

    def record_trade(self, pnl: float) -> None:
        self.total_trades += 1
        self.total_pnl += pnl
        if pnl > 0:
            self.wins += 1
        elif pnl < 0:
            self.losses += 1
        self.recent_pnl.append(pnl)
        if len(self.recent_pnl) > self._max_recent:
            self.recent_pnl.pop(0)

    def to_dict(self) -> dict:
        return {
            "agent_id": self.agent_id,
            "weight": round(self.weight, 4),
            "total_trades": self.total_trades,
            "wins": self.wins,
            "losses": self.losses,
            "total_pnl": round(self.total_pnl, 4),
            "win_rate": round(self.win_rate, 4),
            "avg_pnl": round(self.avg_pnl, 4),
            "recent_avg_pnl": round(self.recent_avg_pnl, 4),
            "sharpe_proxy": round(self.sharpe_proxy, 4),
        }


# ─── Performance Tracker ──────────────────────────────────────────────────────


class AgentPerformanceTracker:
    """
    Tracks per-agent performance and dynamically adjusts voting weights.

    Weight update rule:
        new_weight = base_weight * (1 + win_rate_bonus + sharpe_bonus)

    Weights are clipped to [min_weight, max_weight].
    """

    def __init__(
        self,
        min_weight: float = 0.1,
        max_weight: float = 5.0,
        update_interval: int = 10,  # update weights every N trades
    ):
        self.min_weight = min_weight
        self.max_weight = max_weight
        self.update_interval = update_interval
        self._records: dict[str, AgentPerformance] = {}
        self._trade_count: int = 0

    def register_agent(self, agent_id: str, initial_weight: float = 1.0) -> None:
        if agent_id not in self._records:
            perf = AgentPerformance(agent_id=agent_id, weight=initial_weight)
            self._records[agent_id] = perf

    def record_outcome(self, agent_id: str, pnl: float) -> None:
        """Record trade outcome for an agent, update weights if due."""
        if agent_id not in self._records:
            self.register_agent(agent_id)
        self._records[agent_id].record_trade(pnl)
        self._trade_count += 1
        if self._trade_count % self.update_interval == 0:
            self._rebalance_weights()

    def get_weight(self, agent_id: str) -> float:
        if agent_id not in self._records:
            return 1.0
        return self._records[agent_id].weight

    def get_all_weights(self) -> dict[str, float]:
        return {aid: rec.weight for aid, rec in self._records.items()}

    def get_performance(self, agent_id: str) -> Optional[AgentPerformance]:
        return self._records.get(agent_id)

    def get_all_performance(self) -> list[dict]:
        return [rec.to_dict() for rec in self._records.values()]

    def _rebalance_weights(self) -> None:
        """Recompute weights based on recent performance."""
        for perf in self._records.values():
            if perf.total_trades < 3:
                continue  # not enough data
            win_bonus = (perf.win_rate - 0.5) * 0.5   # ±0.25 bonus
            sharpe_bonus = max(-0.25, min(0.25, perf.sharpe_proxy * 0.1))
            new_weight = 1.0 + win_bonus + sharpe_bonus
            new_weight = max(self.min_weight, min(self.max_weight, new_weight))
            perf.weight = new_weight

    def top_performers(self, n: int = 3) -> list[AgentPerformance]:
        """Return top N agents by win rate."""
        sorted_agents = sorted(
            self._records.values(),
            key=lambda p: (p.win_rate, p.total_pnl),
            reverse=True,
        )
        return sorted_agents[:n]

    def worst_performer(self) -> Optional[AgentPerformance]:
        """Return the worst-performing agent (by avg PnL)."""
        if not self._records:
            return None
        return min(self._records.values(), key=lambda p: p.avg_pnl)


# ─── Agent Pool ───────────────────────────────────────────────────────────────


class AgentPool:
    """
    Manages multiple strategist instances with different risk profiles.

    Each agent in the pool is associated with a strategist object (duck-typed:
    must have an async `decide(market_snapshot)` method).
    """

    def __init__(self, performance_tracker: Optional[AgentPerformanceTracker] = None):
        self._agents: dict[str, tuple[AgentConfig, Any]] = {}  # id → (config, strategist)
        self.tracker = performance_tracker or AgentPerformanceTracker()

    def add_agent(self, config: AgentConfig, strategist: Any) -> None:
        """Register a strategist in the pool."""
        self._agents[config.agent_id] = (config, strategist)
        self.tracker.register_agent(config.agent_id, config.initial_weight)

    def remove_agent(self, agent_id: str) -> bool:
        """Remove an agent from the pool. Returns True if found."""
        if agent_id in self._agents:
            del self._agents[agent_id]
            return True
        return False

    @property
    def agent_ids(self) -> list[str]:
        return list(self._agents.keys())

    @property
    def size(self) -> int:
        return len(self._agents)

    def get_config(self, agent_id: str) -> Optional[AgentConfig]:
        entry = self._agents.get(agent_id)
        return entry[0] if entry else None

    async def collect_signals(self, market_snapshot: dict) -> list[AgentSignal]:
        """
        Query all agents concurrently and collect their signals.

        Each agent's decide() is called with the market snapshot.
        Failures are caught and skipped (non-failing agents still vote).
        """
        tasks = []
        for agent_id, (config, strategist) in self._agents.items():
            tasks.append(self._query_agent(agent_id, config, strategist, market_snapshot))
        results = await asyncio.gather(*tasks, return_exceptions=True)
        signals = []
        for r in results:
            if isinstance(r, AgentSignal):
                signals.append(r)
        return signals

    async def _query_agent(
        self,
        agent_id: str,
        config: AgentConfig,
        strategist: Any,
        market_snapshot: dict,
    ) -> AgentSignal:
        """Query a single agent and convert its decision to AgentSignal."""
        decision = await strategist.decide(market_snapshot)
        action = getattr(decision, "action", "hold")
        confidence = getattr(decision, "confidence", 0.0)
        size_pct = getattr(decision, "size_pct", 0.0)
        reasoning = getattr(decision, "reasoning", "")

        # Apply risk profile constraints
        if confidence < config.min_confidence:
            action = "hold"
        size_pct = min(size_pct, config.max_position_pct)

        return AgentSignal(
            agent_id=agent_id,
            action=action,
            confidence=confidence,
            size_pct=size_pct,
            reasoning=reasoning,
        )


# ─── Multi-Agent Coordinator ──────────────────────────────────────────────────


class MultiAgentCoordinator:
    """
    Orchestrates the AgentPool to produce consensus EnsembleSignals.

    Voting algorithm:
    1. Collect signals from all agents.
    2. For each possible action, sum the weighted votes (weight from tracker).
    3. Winning action is the one with the highest weighted vote.
    4. Consensus is reached if winning_weight / total_weight >= consensus_threshold.
    5. If no consensus, emit "hold".
    """

    def __init__(
        self,
        pool: AgentPool,
        consensus_threshold: float = 0.60,
        min_agents_required: int = 2,
        on_consensus: Optional[Callable[[EnsembleSignal], None]] = None,
    ):
        self.pool = pool
        self.consensus_threshold = consensus_threshold
        self.min_agents_required = min_agents_required
        self.on_consensus = on_consensus
        self._decision_history: list[EnsembleSignal] = []

    async def get_ensemble_signal(self, market_snapshot: dict) -> EnsembleSignal:
        """
        Main entry point: collect signals, compute ensemble, return decision.
        """
        signals = await self.pool.collect_signals(market_snapshot)

        if len(signals) < self.min_agents_required:
            ensemble = self._no_quorum_signal(signals)
        else:
            ensemble = self._compute_ensemble(signals)

        self._decision_history.append(ensemble)
        if self.on_consensus and ensemble.has_consensus:
            self.on_consensus(ensemble)
        return ensemble

    def _compute_ensemble(self, signals: list[AgentSignal]) -> EnsembleSignal:
        """Weighted vote aggregation."""
        vote_weights: dict[str, float] = {"buy": 0.0, "sell": 0.0, "hold": 0.0}
        vote_confidence: dict[str, list[float]] = {"buy": [], "sell": [], "hold": []}
        vote_size: dict[str, list[float]] = {"buy": [], "sell": [], "hold": []}
        agent_votes: dict[str, str] = {}
        total_weight = 0.0

        for sig in signals:
            w = self.pool.tracker.get_weight(sig.agent_id)
            vote_weights[sig.action] += w
            vote_confidence[sig.action].append(sig.confidence)
            vote_size[sig.action].append(sig.size_pct)
            agent_votes[sig.agent_id] = sig.action
            total_weight += w

        if total_weight == 0:
            return self._no_quorum_signal(signals)

        # Find winning action
        winning_action = max(vote_weights, key=lambda k: vote_weights[k])
        winning_weight = vote_weights[winning_action]
        consensus_weight = winning_weight / total_weight
        has_consensus = consensus_weight >= self.consensus_threshold

        # If no consensus, default to hold
        if not has_consensus:
            winning_action = "hold"

        mean_conf = (
            statistics.mean(vote_confidence[winning_action])
            if vote_confidence[winning_action]
            else 0.0
        )
        mean_size = (
            statistics.mean(vote_size[winning_action])
            if vote_size[winning_action]
            else 0.0
        )

        dissenting = [
            aid for aid, action in agent_votes.items()
            if action != winning_action
        ]

        return EnsembleSignal(
            action=winning_action,
            consensus_weight=round(consensus_weight, 4),
            mean_confidence=round(mean_conf, 4),
            mean_size_pct=round(mean_size, 4),
            agent_votes=agent_votes,
            dissenting_agents=dissenting,
            has_consensus=has_consensus,
            reasoning=f"Weighted vote: {winning_action}@{consensus_weight:.0%} of {total_weight:.1f}w",
        )

    def _no_quorum_signal(self, signals: list[AgentSignal]) -> EnsembleSignal:
        agent_votes = {s.agent_id: s.action for s in signals}
        return EnsembleSignal(
            action="hold",
            consensus_weight=0.0,
            mean_confidence=0.0,
            mean_size_pct=0.0,
            agent_votes=agent_votes,
            dissenting_agents=[],
            has_consensus=False,
            reasoning="No quorum — insufficient agents responded",
        )

    def record_trade_outcome(self, agent_id: str, pnl: float) -> None:
        """Update performance tracker after a trade settles."""
        self.pool.tracker.record_outcome(agent_id, pnl)

    def record_outcome_for_all_agreeing(
        self, ensemble: EnsembleSignal, pnl: float
    ) -> None:
        """Distribute PnL outcome to all agents that voted with the consensus."""
        for agent_id, action in ensemble.agent_votes.items():
            if action == ensemble.action:
                self.pool.tracker.record_outcome(agent_id, pnl)
            else:
                # Dissenting agents had it right if pnl < 0
                if pnl < 0:
                    self.pool.tracker.record_outcome(agent_id, -pnl * 0.5)

    @property
    def decision_count(self) -> int:
        return len(self._decision_history)

    @property
    def consensus_rate(self) -> float:
        """Fraction of decisions where consensus was reached."""
        if not self._decision_history:
            return 0.0
        with_consensus = sum(1 for d in self._decision_history if d.has_consensus)
        return with_consensus / len(self._decision_history)

    def performance_summary(self) -> dict:
        return {
            "pool_size": self.pool.size,
            "decision_count": self.decision_count,
            "consensus_rate": round(self.consensus_rate, 4),
            "consensus_threshold": self.consensus_threshold,
            "agent_performance": self.pool.tracker.get_all_performance(),
        }


# ─── Factory helpers ──────────────────────────────────────────────────────────


def build_default_pool(mock_strategist_factory: Optional[Callable] = None) -> AgentPool:
    """
    Build a default 3-agent pool (conservative, moderate, aggressive).

    If mock_strategist_factory is provided, it's called with (agent_id, config)
    to create each strategist. Otherwise, async mocks are used.
    """
    pool = AgentPool()
    configs = [
        AgentConfig.conservative("agent-con"),
        AgentConfig.moderate("agent-mod"),
        AgentConfig.aggressive("agent-agg"),
    ]
    for cfg in configs:
        if mock_strategist_factory:
            strategist = mock_strategist_factory(cfg.agent_id, cfg)
        else:
            strategist = AsyncMock()
        pool.add_agent(cfg, strategist)
    return pool


def build_coordinator(
    consensus_threshold: float = 0.60,
    mock_strategist_factory: Optional[Callable] = None,
) -> MultiAgentCoordinator:
    """Build a ready-to-use coordinator with default 3-agent pool."""
    pool = build_default_pool(mock_strategist_factory)
    return MultiAgentCoordinator(pool, consensus_threshold=consensus_threshold)
