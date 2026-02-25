"""
mesh_coordinator.py — Multi-agent mesh coordination for ERC-8004 Trading Agent.

Runs three specialist agents simultaneously, each with different risk profiles
aligned to their ERC-8004 reputation score. A 2/3 consensus is required before
any trade executes.

Agent profiles:
  ConservativeAgent  — credora_min_grade=A,   kelly_fraction=0.15, max_position=0.05
  BalancedAgent      — credora_min_grade=BBB,  kelly_fraction=0.25, max_position=0.10
  AggressiveAgent    — credora_min_grade=BB,   kelly_fraction=0.35, max_position=0.15

The coordinator weights votes by each agent's current ERC-8004 reputation score,
ensuring that agents with better historical performance have proportionally more
influence over execution decisions.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

from loguru import logger

from credora_client import CredoraRatingTier


# ─── Enums ────────────────────────────────────────────────────────────────────

class AgentProfile(str, Enum):
    CONSERVATIVE = "conservative"
    BALANCED     = "balanced"
    AGGRESSIVE   = "aggressive"


class VoteAction(str, Enum):
    BUY      = "BUY"
    SELL     = "SELL"
    HOLD     = "HOLD"
    REJECT   = "REJECT"


# ─── Data Classes ─────────────────────────────────────────────────────────────

@dataclass
class AgentConfig:
    """Configuration for a specialist mesh agent."""
    profile: AgentProfile
    credora_min_grade: CredoraRatingTier
    kelly_fraction: float          # base Kelly fraction before multiplier
    max_position_pct: float        # maximum single-trade size as fraction of portfolio
    initial_reputation: float = 5.0  # ERC-8004 reputation score 0–10


@dataclass
class AgentVote:
    """A single agent's vote on a proposed trade."""
    agent_id: str
    profile: AgentProfile
    action: VoteAction
    confidence: float              # 0.0–1.0
    adjusted_size: float           # proposed position size after Kelly
    reasoning: str
    reputation_score: float        # agent's current ERC-8004 reputation
    timestamp: float = field(default_factory=time.time)


@dataclass
class MeshConsensus:
    """Aggregated consensus result from the multi-agent mesh."""
    agents: list[str]              # agent IDs that participated
    proposal: dict                 # original trade proposal
    votes_for: list[AgentVote]     # agents voting BUY/SELL (active)
    votes_against: list[AgentVote] # agents voting HOLD/REJECT
    consensus_reached: bool
    final_action: VoteAction
    weighted_size: float           # reputation-weighted average position size
    total_reputation_weight: float
    timestamp: float = field(default_factory=time.time)

    @property
    def vote_count(self) -> int:
        return len(self.votes_for) + len(self.votes_against)

    @property
    def approval_ratio(self) -> float:
        if self.vote_count == 0:
            return 0.0
        return len(self.votes_for) / self.vote_count


# ─── Specialist Agents ────────────────────────────────────────────────────────

# Default configurations for the three specialist agents
AGENT_CONFIGS: dict[AgentProfile, AgentConfig] = {
    AgentProfile.CONSERVATIVE: AgentConfig(
        profile=AgentProfile.CONSERVATIVE,
        credora_min_grade=CredoraRatingTier.A,
        kelly_fraction=0.15,
        max_position_pct=0.05,
        initial_reputation=7.0,   # conservative agents start with higher reputation
    ),
    AgentProfile.BALANCED: AgentConfig(
        profile=AgentProfile.BALANCED,
        credora_min_grade=CredoraRatingTier.BBB,
        kelly_fraction=0.25,
        max_position_pct=0.10,
        initial_reputation=6.5,
    ),
    AgentProfile.AGGRESSIVE: AgentConfig(
        profile=AgentProfile.AGGRESSIVE,
        credora_min_grade=CredoraRatingTier.BB,
        kelly_fraction=0.35,
        max_position_pct=0.15,
        initial_reputation=5.5,
    ),
}

# Tier ordering for grade comparisons (highest first)
_TIER_ORDER: list[CredoraRatingTier] = [
    CredoraRatingTier.AAA,
    CredoraRatingTier.AA,
    CredoraRatingTier.A,
    CredoraRatingTier.BBB,
    CredoraRatingTier.BB,
    CredoraRatingTier.B,
    CredoraRatingTier.CCC,
    CredoraRatingTier.NR,
]


class SpecialistAgent:
    """
    A specialist agent in the multi-agent mesh.

    Each agent evaluates trades through its own risk lens and updates its
    ERC-8004 reputation based on trade outcomes.
    """

    def __init__(self, agent_id: str, config: AgentConfig) -> None:
        self.agent_id = agent_id
        self.config = config
        self._reputation_score: float = config.initial_reputation
        self._trade_count: int = 0
        self._win_count: int = 0
        logger.debug(
            f"SpecialistAgent {agent_id} ({config.profile.value}) initialized "
            f"rep={self._reputation_score:.1f} kelly={config.kelly_fraction:.2f}"
        )

    @property
    def reputation_score(self) -> float:
        return self._reputation_score

    @property
    def profile(self) -> AgentProfile:
        return self.config.profile

    def _meets_grade_requirement(self, protocol_grade: CredoraRatingTier) -> bool:
        """Check if the protocol's Credora grade meets this agent's minimum."""
        min_idx = _TIER_ORDER.index(self.config.credora_min_grade)
        try:
            grade_idx = _TIER_ORDER.index(protocol_grade)
        except ValueError:
            return False
        # lower index = higher grade
        return grade_idx <= min_idx

    def evaluate(
        self,
        side: str,
        size: float,
        price: float,
        portfolio_value: float,
        protocol_grade: CredoraRatingTier = CredoraRatingTier.BBB,
        edge: float = 0.05,        # estimated probability edge
    ) -> AgentVote:
        """
        Evaluate a proposed trade and return a vote.

        Args:
            side:            "BUY" or "SELL"
            size:            Proposed size in USDC
            price:           Market price (0–1)
            portfolio_value: Total portfolio value in USDC
            protocol_grade:  Credora grade of the underlying protocol
            edge:            Estimated edge (probability advantage)

        Returns:
            AgentVote with action BUY/SELL/HOLD/REJECT and adjusted size.
        """
        # 1. Grade check
        if not self._meets_grade_requirement(protocol_grade):
            return AgentVote(
                agent_id=self.agent_id,
                profile=self.config.profile,
                action=VoteAction.REJECT,
                confidence=0.9,
                adjusted_size=0.0,
                reasoning=(
                    f"Protocol grade {protocol_grade.value} below minimum "
                    f"{self.config.credora_min_grade.value}"
                ),
                reputation_score=self._reputation_score,
            )

        # 2. Position size limit
        max_size = portfolio_value * self.config.max_position_pct
        if size > max_size:
            # Clip to allowed size
            clipped_size = max_size
            reasoning = (
                f"Size clipped from ${size:.2f} to ${clipped_size:.2f} "
                f"(max {self.config.max_position_pct:.0%})"
            )
        else:
            clipped_size = size
            reasoning = "Size within limits"

        # 3. Kelly sizing
        kelly_size = portfolio_value * self.config.kelly_fraction * edge * 2
        adjusted_size = min(clipped_size, kelly_size)

        # 4. Edge threshold — HOLD if edge is too small
        min_edge = 0.02 if self.config.profile == AgentProfile.AGGRESSIVE else 0.03
        if edge < min_edge:
            return AgentVote(
                agent_id=self.agent_id,
                profile=self.config.profile,
                action=VoteAction.HOLD,
                confidence=0.6,
                adjusted_size=0.0,
                reasoning=f"Edge {edge:.3f} below threshold {min_edge:.3f}",
                reputation_score=self._reputation_score,
            )

        # 5. Confidence based on reputation and edge
        confidence = min(0.95, 0.5 + edge * 3.0 + self._reputation_score * 0.02)

        action = VoteAction.BUY if side.upper() in ("BUY", "YES") else VoteAction.SELL

        return AgentVote(
            agent_id=self.agent_id,
            profile=self.config.profile,
            action=action,
            confidence=confidence,
            adjusted_size=max(0.0, adjusted_size),
            reasoning=reasoning,
            reputation_score=self._reputation_score,
        )

    def update_reputation(self, trade_won: bool, pnl_pct: float = 0.0) -> None:
        """
        Update agent's ERC-8004 reputation score based on trade outcome.

        Reputation moves by ±0.1 per trade, bounded to [0, 10].
        A win on a large edge moves it more; a loss on a small edge moves it less.
        """
        self._trade_count += 1
        delta = 0.1 * (1.0 + abs(pnl_pct))
        if trade_won:
            self._win_count += 1
            self._reputation_score = min(10.0, self._reputation_score + delta)
        else:
            self._reputation_score = max(0.0, self._reputation_score - delta * 0.7)

        logger.debug(
            f"SpecialistAgent {self.agent_id}: reputation updated to "
            f"{self._reputation_score:.2f} (won={trade_won} pnl={pnl_pct:.3f})"
        )

    def get_stats(self) -> dict:
        return {
            "agent_id": self.agent_id,
            "profile": self.config.profile.value,
            "reputation_score": round(self._reputation_score, 3),
            "trade_count": self._trade_count,
            "win_count": self._win_count,
            "win_rate": (
                self._win_count / self._trade_count
                if self._trade_count > 0 else 0.0
            ),
            "kelly_fraction": self.config.kelly_fraction,
            "max_position_pct": self.config.max_position_pct,
            "credora_min_grade": self.config.credora_min_grade.value,
        }


# ─── Mesh Coordinator ─────────────────────────────────────────────────────────

class MeshCoordinator:
    """
    Coordinates multiple specialist agents and aggregates their signals.

    Consensus rule: at least 2 out of 3 agents must vote for the same active
    action (BUY or SELL) for a trade to proceed. Votes are weighted by each
    agent's current ERC-8004 reputation score.

    Usage:
        coord = MeshCoordinator()
        consensus = coord.evaluate(side="BUY", size=5.0, price=0.65,
                                   portfolio_value=100.0)
        if consensus.consensus_reached:
            execute(consensus.final_action, consensus.weighted_size)
    """

    def __init__(
        self,
        configs: Optional[dict[AgentProfile, AgentConfig]] = None,
        min_votes_for_consensus: int = 2,
    ) -> None:
        configs = configs or AGENT_CONFIGS
        self.min_votes_for_consensus = min_votes_for_consensus

        self._agents: dict[str, SpecialistAgent] = {
            f"{profile.value}_agent": SpecialistAgent(
                agent_id=f"{profile.value}_agent",
                config=cfg,
            )
            for profile, cfg in configs.items()
        }
        logger.info(
            f"MeshCoordinator initialized with {len(self._agents)} agents: "
            + ", ".join(self._agents.keys())
        )

    @property
    def agents(self) -> list[SpecialistAgent]:
        return list(self._agents.values())

    def get_agent(self, agent_id: str) -> Optional[SpecialistAgent]:
        return self._agents.get(agent_id)

    def evaluate(
        self,
        side: str,
        size: float,
        price: float,
        portfolio_value: float,
        protocol_grade: CredoraRatingTier = CredoraRatingTier.BBB,
        edge: float = 0.05,
    ) -> MeshConsensus:
        """
        Ask all agents to evaluate a trade and compute mesh consensus.

        Returns:
            MeshConsensus with consensus_reached=True if ≥2 agents agree.
        """
        proposal = {
            "side": side,
            "size": size,
            "price": price,
            "portfolio_value": portfolio_value,
            "protocol_grade": protocol_grade.value,
            "edge": edge,
        }

        # Collect votes
        votes: list[AgentVote] = []
        for agent in self._agents.values():
            vote = agent.evaluate(
                side=side,
                size=size,
                price=price,
                portfolio_value=portfolio_value,
                protocol_grade=protocol_grade,
                edge=edge,
            )
            votes.append(vote)
            logger.debug(
                f"Agent {vote.agent_id}: {vote.action.value} "
                f"size={vote.adjusted_size:.2f} rep={vote.reputation_score:.2f}"
            )

        # Separate active (BUY/SELL) from passive (HOLD/REJECT)
        active_actions = {VoteAction.BUY, VoteAction.SELL}
        votes_for = [v for v in votes if v.action in active_actions]
        votes_against = [v for v in votes if v.action not in active_actions]

        consensus_reached = len(votes_for) >= self.min_votes_for_consensus

        # Determine final action: majority of active votes
        if votes_for:
            buy_count = sum(1 for v in votes_for if v.action == VoteAction.BUY)
            sell_count = sum(1 for v in votes_for if v.action == VoteAction.SELL)
            final_action = VoteAction.BUY if buy_count >= sell_count else VoteAction.SELL
        else:
            final_action = VoteAction.HOLD

        # Reputation-weighted average size (only votes_for contribute)
        weighted_size = 0.0
        total_rep = 0.0
        if votes_for:
            for v in votes_for:
                rep = max(v.reputation_score, 0.1)  # floor to avoid zero-weight
                weighted_size += v.adjusted_size * rep
                total_rep += rep
            weighted_size = weighted_size / total_rep if total_rep > 0 else 0.0
        else:
            total_rep = sum(max(v.reputation_score, 0.1) for v in votes)

        consensus = MeshConsensus(
            agents=list(self._agents.keys()),
            proposal=proposal,
            votes_for=votes_for,
            votes_against=votes_against,
            consensus_reached=consensus_reached,
            final_action=final_action,
            weighted_size=weighted_size,
            total_reputation_weight=total_rep,
        )

        if consensus_reached:
            logger.info(
                f"MeshCoordinator: CONSENSUS {final_action.value} "
                f"size=${weighted_size:.2f} "
                f"({len(votes_for)}/{len(votes)} voted active)"
            )
        else:
            logger.info(
                f"MeshCoordinator: NO CONSENSUS — "
                f"only {len(votes_for)}/{len(votes)} active votes "
                f"(need {self.min_votes_for_consensus})"
            )

        return consensus

    def record_outcome(
        self,
        trade_won: bool,
        pnl_pct: float = 0.0,
        agent_ids: Optional[list[str]] = None,
    ) -> None:
        """
        Update reputation for agents that participated in a trade.

        Args:
            trade_won:  Whether the trade was profitable
            pnl_pct:    P&L as fraction of trade size
            agent_ids:  Which agents to update (defaults to all)
        """
        targets = agent_ids or list(self._agents.keys())
        for aid in targets:
            if aid in self._agents:
                self._agents[aid].update_reputation(trade_won, pnl_pct)

    def get_mesh_stats(self) -> dict:
        """Return status summary for all agents in the mesh."""
        return {
            "agent_count": len(self._agents),
            "min_votes_for_consensus": self.min_votes_for_consensus,
            "agents": [a.get_stats() for a in self._agents.values()],
        }
