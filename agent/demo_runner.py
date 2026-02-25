"""
demo_runner.py — End-to-End Demo Scenario Runner for ERC-8004 Trading Agent.

Orchestrates a complete trading demonstration:
  1. Start 3 agent instances (Conservative / Balanced / Aggressive)
  2. Feed 50 sequential price ticks (realistic GBM walk)
  3. Each tick: mesh consensus → paper trade → reputation update
  4. Track: P&L per agent, reputation changes, risk violations
  5. Output: JSON report + human-readable summary

Usage:
    runner = DemoRunner(seed=42)
    report = runner.run()
    print(report.summary())
    print(report.to_json())
"""

from __future__ import annotations

import json
import math
import random
import time
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional


# ─── Price Generator ──────────────────────────────────────────────────────────

class DemoGBM:
    """Minimal GBM price generator for demo purposes."""

    def __init__(
        self,
        initial_price: float = 100.0,
        mu: float = 0.05,
        sigma: float = 0.25,
        dt: float = 1 / 8760,
        seed: Optional[int] = None,
    ) -> None:
        self.price = initial_price
        self.mu = mu
        self.sigma = sigma
        self.dt = dt
        self._rng = random.Random(seed)

    def _normal(self) -> float:
        while True:
            u = self._rng.uniform(1e-9, 1.0)
            v = self._rng.uniform(0.0, 1.0)
            if u > 0.0:
                break
        return math.sqrt(-2.0 * math.log(u)) * math.cos(2.0 * math.pi * v)

    def next_price(self) -> float:
        z = self._normal()
        drift = (self.mu - 0.5 * self.sigma ** 2) * self.dt
        diffusion = self.sigma * math.sqrt(self.dt) * z
        self.price *= math.exp(drift + diffusion)
        return self.price

    def generate(self, n: int) -> List[float]:
        return [self.next_price() for _ in range(n)]


# ─── Agent State ──────────────────────────────────────────────────────────────

@dataclass
class DemoAgent:
    """State for one demo agent instance."""
    agent_id: str
    profile: str                    # conservative / balanced / aggressive
    capital: float
    reputation: float               # ERC-8004 reputation 0–10
    kelly_fraction: float
    max_position_pct: float
    risk_tolerance: float           # sigma multiplier before risk veto

    # Runtime state
    position: float = 0.0           # current position in units
    entry_price: float = 0.0
    pnl: float = 0.0
    trades: int = 0
    wins: int = 0
    risk_violations: int = 0
    rep_history: List[float] = field(default_factory=list)
    pnl_history: List[float] = field(default_factory=list)


# ─── Tick Result ──────────────────────────────────────────────────────────────

@dataclass
class TickResult:
    """Result of processing one price tick."""
    tick: int
    price: float
    consensus_action: str           # BUY / SELL / HOLD
    agents_voted_buy: List[str]
    agents_voted_sell: List[str]
    agents_voted_hold: List[str]
    trades_executed: List[Dict]
    risk_violations: List[str]
    timestamp: float = field(default_factory=time.time)


# ─── Demo Report ──────────────────────────────────────────────────────────────

@dataclass
class DemoReport:
    """Full end-to-end demo scenario report."""
    scenario: str
    ticks: int
    duration_ms: float
    agents: List[Dict]
    tick_results: List[Dict]
    summary_stats: Dict

    def summary(self) -> str:
        lines = [
            f"=== ERC-8004 Demo Report: {self.scenario} ===",
            f"Ticks: {self.ticks}  |  Duration: {self.duration_ms:.1f}ms",
            "",
            "Agent Performance:",
        ]
        for agent in self.agents:
            lines.append(
                f"  [{agent['profile'].upper():12s}]  "
                f"P&L: ${agent['pnl']:+8.2f}  "
                f"Trades: {agent['trades']:3d}  "
                f"Wins: {agent['wins']:3d}  "
                f"Rep: {agent['final_reputation']:.2f}  "
                f"Violations: {agent['risk_violations']}"
            )
        stats = self.summary_stats
        lines += [
            "",
            f"Consensus reached: {stats['consensus_reached']} / {self.ticks} ticks",
            f"Total trades executed: {stats['total_trades']}",
            f"Total risk violations: {stats['total_risk_violations']}",
        ]
        return "\n".join(lines)

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2, default=str)


# ─── Demo Runner ──────────────────────────────────────────────────────────────

class DemoRunner:
    """
    Orchestrates a complete ERC-8004 multi-agent trading demonstration.

    Parameters
    ----------
    n_ticks      : Number of price ticks to simulate (default 50)
    initial_price: Starting asset price
    seed         : RNG seed for reproducibility
    initial_capital: Starting capital per agent
    """

    AGENT_CONFIGS = [
        {
            "agent_id": "agent-conservative-001",
            "profile": "conservative",
            "kelly_fraction": 0.15,
            "max_position_pct": 0.05,
            "risk_tolerance": 1.5,
            "initial_reputation": 7.5,
        },
        {
            "agent_id": "agent-balanced-002",
            "profile": "balanced",
            "kelly_fraction": 0.25,
            "max_position_pct": 0.10,
            "risk_tolerance": 2.0,
            "initial_reputation": 6.0,
        },
        {
            "agent_id": "agent-aggressive-003",
            "profile": "aggressive",
            "kelly_fraction": 0.35,
            "max_position_pct": 0.15,
            "risk_tolerance": 3.0,
            "initial_reputation": 5.0,
        },
    ]

    def __init__(
        self,
        n_ticks: int = 50,
        initial_price: float = 100.0,
        seed: Optional[int] = 42,
        initial_capital: float = 10_000.0,
    ) -> None:
        self.n_ticks = n_ticks
        self.initial_price = initial_price
        self.seed = seed
        self.initial_capital = initial_capital
        self._rng = random.Random(seed)

    def _build_agents(self) -> List[DemoAgent]:
        agents = []
        for cfg in self.AGENT_CONFIGS:
            agents.append(DemoAgent(
                agent_id=cfg["agent_id"],
                profile=cfg["profile"],
                capital=self.initial_capital,
                reputation=cfg["initial_reputation"],
                kelly_fraction=cfg["kelly_fraction"],
                max_position_pct=cfg["max_position_pct"],
                risk_tolerance=cfg["risk_tolerance"],
            ))
        return agents

    def _agent_vote(
        self,
        agent: DemoAgent,
        price: float,
        prev_price: float,
        volatility: float,
    ) -> str:
        """Compute an agent's vote for this tick."""
        returns = (price - prev_price) / prev_price if prev_price > 0 else 0.0
        # Conservative: only buy on strong up-moves within tolerance
        # Aggressive: trades on smaller signals
        threshold = agent.risk_tolerance * volatility
        if abs(returns) > threshold * 2:
            agent.risk_violations += 1
            return "HOLD"  # veto extreme moves
        if returns > threshold:
            return "BUY"
        elif returns < -threshold:
            return "SELL"
        return "HOLD"

    def _mesh_consensus(
        self,
        votes: Dict[str, str],
        reputations: Dict[str, float],
    ) -> str:
        """Reputation-weighted mesh consensus."""
        weights = {"BUY": 0.0, "SELL": 0.0, "HOLD": 0.0}
        total = sum(reputations.values())
        if total <= 0:
            return "HOLD"
        for agent_id, vote in votes.items():
            w = reputations.get(agent_id, 1.0) / total
            weights[vote] += w
        # Require 2/3 weighted majority
        best = max(weights, key=weights.__getitem__)
        if weights[best] >= 2 / 3:
            return best
        return "HOLD"

    def _execute_trade(
        self,
        agent: DemoAgent,
        action: str,
        price: float,
    ) -> Optional[Dict]:
        """Execute a paper trade for one agent. Returns trade dict or None."""
        if action == "HOLD":
            return None

        # Compute position size using Kelly fraction
        size_usd = agent.capital * agent.kelly_fraction * agent.max_position_pct
        units = size_usd / price if price > 0 else 0.0

        if action == "BUY" and agent.position <= 0:
            # Close any short, open long
            if agent.position < 0:
                pnl = (agent.entry_price - price) * abs(agent.position)
                agent.pnl += pnl
                agent.trades += 1
                if pnl > 0:
                    agent.wins += 1
            agent.position = units
            agent.entry_price = price
            agent.capital -= size_usd
            agent.pnl_history.append(agent.pnl)
            return {"action": "BUY", "units": units, "price": price, "size_usd": size_usd}

        elif action == "SELL" and agent.position >= 0:
            # Close any long, open short
            if agent.position > 0:
                pnl = (price - agent.entry_price) * agent.position
                agent.pnl += pnl
                agent.trades += 1
                if pnl > 0:
                    agent.wins += 1
            agent.position = -units
            agent.entry_price = price
            agent.capital += size_usd
            agent.pnl_history.append(agent.pnl)
            return {"action": "SELL", "units": units, "price": price, "size_usd": size_usd}

        return None

    def _update_reputation(
        self,
        agent: DemoAgent,
        action: str,
        price: float,
        prev_price: float,
    ) -> None:
        """Update ERC-8004 reputation based on trade outcome."""
        if action == "HOLD":
            agent.rep_history.append(agent.reputation)
            return
        ret = (price - prev_price) / prev_price if prev_price > 0 else 0.0
        # Correct direction → +0.05 rep, wrong → -0.1 rep
        if (action == "BUY" and ret > 0) or (action == "SELL" and ret < 0):
            delta = 0.05
        else:
            delta = -0.10
        agent.reputation = max(0.0, min(10.0, agent.reputation + delta))
        agent.rep_history.append(agent.reputation)

    def run(self, scenario: str = "Standard E2E Demo") -> DemoReport:
        """Run the full demo scenario and return a DemoReport."""
        start_ms = time.time() * 1000

        agents = self._build_agents()
        gbm = DemoGBM(
            initial_price=self.initial_price,
            seed=self.seed,
        )

        prices = [self.initial_price] + gbm.generate(self.n_ticks)
        tick_results: List[TickResult] = []
        consensus_count = 0

        for tick_idx in range(self.n_ticks):
            price = prices[tick_idx + 1]
            prev_price = prices[tick_idx]
            volatility = abs(price - prev_price) / prev_price if prev_price > 0 else 0.01

            # Each agent votes
            votes: Dict[str, str] = {}
            reputations: Dict[str, float] = {}
            for agent in agents:
                vote = self._agent_vote(agent, price, prev_price, volatility)
                votes[agent.agent_id] = vote
                reputations[agent.agent_id] = agent.reputation

            # Mesh consensus
            consensus = self._mesh_consensus(votes, reputations)
            if consensus != "HOLD":
                consensus_count += 1

            # Execute and track
            trades_executed = []
            risk_violations = []
            voted_buy = [aid for aid, v in votes.items() if v == "BUY"]
            voted_sell = [aid for aid, v in votes.items() if v == "SELL"]
            voted_hold = [aid for aid, v in votes.items() if v == "HOLD"]

            for agent in agents:
                trade = self._execute_trade(agent, consensus, price)
                if trade:
                    trade["agent_id"] = agent.agent_id
                    trades_executed.append(trade)
                # Always update reputation
                self._update_reputation(agent, consensus, price, prev_price)
                if agent.risk_violations > 0:
                    risk_violations.append(f"{agent.agent_id}:violations={agent.risk_violations}")

            tick_results.append(TickResult(
                tick=tick_idx + 1,
                price=price,
                consensus_action=consensus,
                agents_voted_buy=voted_buy,
                agents_voted_sell=voted_sell,
                agents_voted_hold=voted_hold,
                trades_executed=trades_executed,
                risk_violations=risk_violations,
            ))

        end_ms = time.time() * 1000
        duration_ms = end_ms - start_ms

        # Build agent summaries
        agent_summaries = []
        for agent in agents:
            # Unrealized P&L on open position
            unrealized = 0.0
            if agent.position != 0:
                final_price = prices[-1]
                if agent.position > 0:
                    unrealized = (final_price - agent.entry_price) * agent.position
                else:
                    unrealized = (agent.entry_price - final_price) * abs(agent.position)

            agent_summaries.append({
                "agent_id": agent.agent_id,
                "profile": agent.profile,
                "initial_capital": self.initial_capital,
                "final_capital": agent.capital + unrealized,
                "pnl": agent.pnl + unrealized,
                "trades": agent.trades,
                "wins": agent.wins,
                "win_rate": agent.wins / agent.trades if agent.trades > 0 else 0.0,
                "initial_reputation": self.AGENT_CONFIGS[agents.index(agent)]["initial_reputation"],
                "final_reputation": agent.reputation,
                "rep_delta": agent.reputation - self.AGENT_CONFIGS[agents.index(agent)]["initial_reputation"],
                "risk_violations": agent.risk_violations,
            })

        total_trades = sum(a["trades"] for a in agent_summaries)
        total_violations = sum(a["risk_violations"] for a in agent_summaries)

        summary_stats = {
            "consensus_reached": consensus_count,
            "consensus_rate": consensus_count / self.n_ticks,
            "total_trades": total_trades,
            "total_risk_violations": total_violations,
            "price_start": prices[0],
            "price_end": prices[-1],
            "price_return_pct": (prices[-1] - prices[0]) / prices[0] * 100,
        }

        return DemoReport(
            scenario=scenario,
            ticks=self.n_ticks,
            duration_ms=duration_ms,
            agents=agent_summaries,
            tick_results=[asdict(t) for t in tick_results],
            summary_stats=summary_stats,
        )

    def run_edge_case(self, scenario: str) -> DemoReport:
        """
        Run a named edge-case scenario.

        Scenarios:
          'zero_liquidity'   — all prices identical (no signal)
          'extreme_volatility' — sigma = 5.0 (500% annualized vol)
        """
        if scenario == "zero_liquidity":
            # Flat prices → all HOLD
            original_generate = DemoGBM.generate

            class FlatGBM(DemoGBM):
                def next_price(self):
                    return self.price  # never moves

            gbm_cls = FlatGBM
        elif scenario == "extreme_volatility":
            class HighVolGBM(DemoGBM):
                def __init__(self, **kwargs):
                    super().__init__(**kwargs)
                    self.sigma = 5.0

            gbm_cls = HighVolGBM
        else:
            raise ValueError(f"Unknown scenario: {scenario!r}")

        # Monkey-patch for this run
        orig_runner = DemoRunner(
            n_ticks=self.n_ticks,
            initial_price=self.initial_price,
            seed=self.seed,
            initial_capital=self.initial_capital,
        )
        # Inject patched GBM via subclassed runner
        agents = orig_runner._build_agents()
        gbm = gbm_cls(initial_price=self.initial_price, seed=self.seed)
        prices = [self.initial_price] + gbm.generate(self.n_ticks)

        tick_results = []
        consensus_count = 0

        for tick_idx in range(self.n_ticks):
            price = prices[tick_idx + 1]
            prev_price = prices[tick_idx]
            volatility = abs(price - prev_price) / prev_price if prev_price > 0 else 0.001

            votes = {}
            reputations = {}
            for agent in agents:
                vote = orig_runner._agent_vote(agent, price, prev_price, volatility)
                votes[agent.agent_id] = vote
                reputations[agent.agent_id] = agent.reputation

            consensus = orig_runner._mesh_consensus(votes, reputations)
            if consensus != "HOLD":
                consensus_count += 1

            trades_executed = []
            risk_violations = []
            voted_buy = [aid for aid, v in votes.items() if v == "BUY"]
            voted_sell = [aid for aid, v in votes.items() if v == "SELL"]
            voted_hold = [aid for aid, v in votes.items() if v == "HOLD"]

            for agent in agents:
                trade = orig_runner._execute_trade(agent, consensus, price)
                if trade:
                    trade["agent_id"] = agent.agent_id
                    trades_executed.append(trade)
                orig_runner._update_reputation(agent, consensus, price, prev_price)

            tick_results.append(TickResult(
                tick=tick_idx + 1,
                price=price,
                consensus_action=consensus,
                agents_voted_buy=voted_buy,
                agents_voted_sell=voted_sell,
                agents_voted_hold=voted_hold,
                trades_executed=trades_executed,
                risk_violations=risk_violations,
            ))

        agent_summaries = []
        for i, agent in enumerate(agents):
            unrealized = 0.0
            if agent.position != 0:
                fp = prices[-1]
                if agent.position > 0:
                    unrealized = (fp - agent.entry_price) * agent.position
                else:
                    unrealized = (agent.entry_price - fp) * abs(agent.position)
            cfg = orig_runner.AGENT_CONFIGS[i]
            agent_summaries.append({
                "agent_id": agent.agent_id,
                "profile": agent.profile,
                "initial_capital": self.initial_capital,
                "final_capital": agent.capital + unrealized,
                "pnl": agent.pnl + unrealized,
                "trades": agent.trades,
                "wins": agent.wins,
                "win_rate": agent.wins / agent.trades if agent.trades > 0 else 0.0,
                "initial_reputation": cfg["initial_reputation"],
                "final_reputation": agent.reputation,
                "rep_delta": agent.reputation - cfg["initial_reputation"],
                "risk_violations": agent.risk_violations,
            })

        total_trades = sum(a["trades"] for a in agent_summaries)
        total_violations = sum(a["risk_violations"] for a in agent_summaries)

        return DemoReport(
            scenario=scenario,
            ticks=self.n_ticks,
            duration_ms=0.0,
            agents=agent_summaries,
            tick_results=[asdict(t) for t in tick_results],
            summary_stats={
                "consensus_reached": consensus_count,
                "consensus_rate": consensus_count / self.n_ticks,
                "total_trades": total_trades,
                "total_risk_violations": total_violations,
                "price_start": prices[0],
                "price_end": prices[-1],
                "price_return_pct": (prices[-1] - prices[0]) / prices[0] * 100,
            },
        )


if __name__ == "__main__":
    runner = DemoRunner(seed=42)
    report = runner.run()
    print(report.summary())
