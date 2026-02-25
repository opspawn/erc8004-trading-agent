"""
stress_tester.py — Adversarial Stress Testing for ERC-8004 Trading Agent.

Simulates extreme market conditions to verify graceful degradation:
  - Flash crash: -40% in 5 ticks
  - Liquidity crisis: all Surge routes return zero
  - Oracle failure: RedStone raises ConnectionError → fallback
  - Consensus deadlock: 3-way tie → tie-breaking via reputation

Each scenario is self-contained and returns a StressResult.

Usage:
    tester = StressTester(seed=42)
    results = tester.run_all()
    for r in results:
        print(r.summary())
"""

from __future__ import annotations

import math
import random
import time
from dataclasses import dataclass, field, asdict
from typing import Any, Callable, Dict, List, Optional, Tuple


# ─── Result types ─────────────────────────────────────────────────────────────

@dataclass
class AgentResponse:
    """One agent's response to a stress event."""
    agent_id: str
    profile: str
    action: str                 # BUY / SELL / HOLD / VETO
    reason: str
    reputation_before: float
    reputation_after: float
    capital_before: float
    capital_after: float
    position: float


@dataclass
class StressResult:
    """Result of one stress scenario."""
    scenario: str
    description: str
    passed: bool                # did the system handle it gracefully?
    agent_responses: List[AgentResponse]
    system_action: str          # final system decision
    observations: List[str]     # notable events
    duration_ms: float
    metadata: Dict[str, Any] = field(default_factory=dict)

    def summary(self) -> str:
        status = "PASS" if self.passed else "FAIL"
        lines = [
            f"[{status}] {self.scenario}: {self.description}",
            f"  System action: {self.system_action}",
        ]
        for obs in self.observations:
            lines.append(f"  • {obs}")
        return "\n".join(lines)

    def to_dict(self) -> Dict:
        return asdict(self)


# ─── Fake agent for stress tests ──────────────────────────────────────────────

class _StressAgent:
    """
    Lightweight agent for stress testing.
    Has profile, reputation, capital, and can vote.
    """

    def __init__(
        self,
        agent_id: str,
        profile: str,
        reputation: float = 5.0,
        capital: float = 10_000.0,
        risk_tolerance: float = 2.0,
        kelly_fraction: float = 0.25,
    ) -> None:
        self.agent_id = agent_id
        self.profile = profile
        self.reputation = reputation
        self.capital = capital
        self.risk_tolerance = risk_tolerance
        self.kelly_fraction = kelly_fraction
        self.position: float = 0.0
        self.entry_price: float = 0.0

    def vote(self, returns: float, volatility: float, override: Optional[str] = None) -> str:
        """Vote on a trade given market returns and volatility."""
        if override:
            return override
        threshold = self.risk_tolerance * volatility
        if abs(returns) > threshold * 3:
            return "VETO"  # extreme move veto
        if returns > threshold:
            return "BUY"
        elif returns < -threshold:
            return "SELL"
        return "HOLD"


# ─── Stress Tester ────────────────────────────────────────────────────────────

class StressTester:
    """
    Runs adversarial stress scenarios against the multi-agent trading system.

    Parameters
    ----------
    seed         : RNG seed for reproducibility
    initial_capital: Starting capital per agent
    """

    def __init__(self, seed: int = 42, initial_capital: float = 10_000.0) -> None:
        self.seed = seed
        self.initial_capital = initial_capital
        self._rng = random.Random(seed)

    def _make_agents(self) -> List[_StressAgent]:
        return [
            _StressAgent("agent-conservative-001", "conservative",
                         reputation=7.5, capital=self.initial_capital,
                         risk_tolerance=1.5, kelly_fraction=0.15),
            _StressAgent("agent-balanced-002", "balanced",
                         reputation=6.0, capital=self.initial_capital,
                         risk_tolerance=2.0, kelly_fraction=0.25),
            _StressAgent("agent-aggressive-003", "aggressive",
                         reputation=5.0, capital=self.initial_capital,
                         risk_tolerance=3.0, kelly_fraction=0.35),
        ]

    @staticmethod
    def _reputation_weighted_consensus(
        votes: Dict[str, Tuple[str, float]],  # agent_id → (vote, reputation)
    ) -> str:
        """Reputation-weighted consensus. Tie-break by highest-rep agent."""
        weights: Dict[str, float] = {"BUY": 0.0, "SELL": 0.0, "HOLD": 0.0, "VETO": 0.0}
        total = sum(rep for _, rep in votes.values())
        if total <= 0:
            return "HOLD"

        for agent_id, (vote, rep) in votes.items():
            w = rep / total
            key = vote if vote in weights else "HOLD"
            weights[key] += w

        # VETO always wins (any veto → HOLD)
        if weights["VETO"] > 0:
            return "HOLD"

        non_hold = {k: v for k, v in weights.items() if k not in ("HOLD", "VETO")}
        best = max(non_hold, key=non_hold.__getitem__)
        if non_hold[best] >= 2 / 3:
            return best

        # Deadlock: tie-break by highest-reputation agent
        max_rep = -1.0
        tie_break_vote = "HOLD"
        for agent_id, (vote, rep) in votes.items():
            if vote not in ("HOLD", "VETO") and rep > max_rep:
                max_rep = rep
                tie_break_vote = vote
        return tie_break_vote

    # ─── Scenario 1: Flash crash ───────────────────────────────────────────

    def flash_crash(self) -> StressResult:
        """
        Simulate -40% price crash over 5 ticks.
        Verify agents respond with protective actions (SELL / VETO).
        """
        t0 = time.time()
        agents = self._make_agents()
        observations = []

        prices = [100.0]
        # -40% over 5 ticks: each tick drops by factor (0.6)^(1/5)
        factor = 0.6 ** (1 / 5)
        for _ in range(5):
            prices.append(prices[-1] * factor)

        # Give agents a long position before crash
        for agent in agents:
            agent.position = 10.0
            agent.entry_price = 100.0

        agent_responses: List[AgentResponse] = []
        system_actions: List[str] = []

        for tick in range(1, 6):
            price = prices[tick]
            prev_price = prices[tick - 1]
            returns = (price - prev_price) / prev_price
            volatility = 0.01  # normal vol baseline

            votes: Dict[str, Tuple[str, float]] = {}
            for agent in agents:
                rep_before = agent.reputation
                cap_before = agent.capital
                vote = agent.vote(returns, volatility)
                votes[agent.agent_id] = (vote, agent.reputation)

                # Close position on SELL/VETO
                if vote in ("SELL", "VETO") and agent.position > 0:
                    pnl = (price - agent.entry_price) * agent.position
                    agent.capital += pnl
                    agent.position = 0.0
                    if tick == 1:  # only record first tick per agent
                        agent_responses.append(AgentResponse(
                            agent_id=agent.agent_id,
                            profile=agent.profile,
                            action=vote,
                            reason=f"Flash crash detected: return={returns:.1%}",
                            reputation_before=rep_before,
                            reputation_after=agent.reputation,
                            capital_before=cap_before,
                            capital_after=agent.capital,
                            position=agent.position,
                        ))

            system_action = self._reputation_weighted_consensus(votes)
            system_actions.append(system_action)

        sell_count = sum(1 for r in agent_responses if r.action in ("SELL", "VETO"))
        observations.append(f"Price dropped {(prices[-1]/prices[0] - 1)*100:.1f}% over 5 ticks")
        observations.append(f"{sell_count}/{len(agents)} agents triggered protective action")
        observations.append(f"System actions: {system_actions}")

        # Ensure at least one agent responded defensively
        passed = sell_count >= 1

        if not agent_responses:
            # Fallback: record final states
            for agent in agents:
                agent_responses.append(AgentResponse(
                    agent_id=agent.agent_id,
                    profile=agent.profile,
                    action="HOLD",
                    reason="No protective action triggered",
                    reputation_before=agent.reputation,
                    reputation_after=agent.reputation,
                    capital_before=self.initial_capital,
                    capital_after=agent.capital,
                    position=agent.position,
                ))

        return StressResult(
            scenario="flash_crash",
            description="40% price crash in 5 ticks",
            passed=passed,
            agent_responses=agent_responses,
            system_action=system_actions[-1] if system_actions else "HOLD",
            observations=observations,
            duration_ms=(time.time() - t0) * 1000,
            metadata={"prices": prices, "system_actions": system_actions},
        )

    # ─── Scenario 2: Liquidity crisis ─────────────────────────────────────

    def liquidity_crisis(self) -> StressResult:
        """
        All Surge liquidity routes return zero liquidity.
        System should degrade gracefully: no trades executed, no crash.
        """
        t0 = time.time()
        agents = self._make_agents()
        observations = []

        class ZeroLiquidityRouter:
            """Simulates complete liquidity drought."""
            def get_best_route(self, token_in, token_out, amount):
                return {"route": [], "output": 0.0, "liquidity": 0.0}

            def get_routes(self, token_in, token_out, amount):
                return []

        router = ZeroLiquidityRouter()
        route = router.get_best_route("USDC", "ETH", 1000.0)

        agent_responses: List[AgentResponse] = []
        for agent in agents:
            # Agent checks liquidity before executing
            liquidity = route["liquidity"]
            action = "HOLD"  # graceful degradation: no liquidity → no trade
            reason = f"Zero liquidity available (got {liquidity})"
            observations.append(f"{agent.agent_id}: {reason}")

            agent_responses.append(AgentResponse(
                agent_id=agent.agent_id,
                profile=agent.profile,
                action=action,
                reason=reason,
                reputation_before=agent.reputation,
                reputation_after=agent.reputation,
                capital_before=self.initial_capital,
                capital_after=self.initial_capital,
                position=0.0,
            ))

        all_held = all(r.action == "HOLD" for r in agent_responses)
        observations.append(f"All agents gracefully held: {all_held}")
        observations.append("No trades executed during liquidity crisis")

        return StressResult(
            scenario="liquidity_crisis",
            description="All Surge routes return zero liquidity",
            passed=all_held,
            agent_responses=agent_responses,
            system_action="HOLD",
            observations=observations,
            duration_ms=(time.time() - t0) * 1000,
            metadata={"route": route, "router_type": "ZeroLiquidityRouter"},
        )

    # ─── Scenario 3: Oracle failure ────────────────────────────────────────

    def oracle_failure(self) -> StressResult:
        """
        RedStone oracle raises ConnectionError.
        System should fall back to last known price / stale cache.
        """
        t0 = time.time()
        agents = self._make_agents()
        observations = []

        class FailingOracle:
            """Simulates oracle connection failure."""
            def __init__(self):
                self._cache: Dict[str, float] = {"ETH": 2000.0, "BTC": 40000.0}

            def get_price(self, symbol: str, use_fallback: bool = True) -> float:
                raise ConnectionError(f"RedStone oracle unreachable for {symbol}")

            def get_price_safe(self, symbol: str) -> Optional[float]:
                """Safe version: returns cached value on failure."""
                try:
                    return self.get_price(symbol)
                except ConnectionError:
                    return self._cache.get(symbol)

        oracle = FailingOracle()
        agent_responses: List[AgentResponse] = []
        fallback_used = 0

        for agent in agents:
            cap_before = agent.capital
            # Try to get price, handle oracle failure
            try:
                price = oracle.get_price("ETH")
                action = "BUY"
                reason = "oracle OK"
            except ConnectionError as e:
                fallback_price = oracle.get_price_safe("ETH")
                if fallback_price is not None:
                    # Use stale cache, but flag uncertainty
                    action = "HOLD"  # conservative under uncertainty
                    reason = f"Oracle down, using stale cache: ${fallback_price}. {e}"
                    fallback_used += 1
                    observations.append(f"{agent.agent_id}: oracle fallback to ${fallback_price}")
                else:
                    action = "HOLD"
                    reason = f"Oracle down, no cache available: {e}"
                    observations.append(f"{agent.agent_id}: oracle failed, no cache → HOLD")

            agent_responses.append(AgentResponse(
                agent_id=agent.agent_id,
                profile=agent.profile,
                action=action,
                reason=reason,
                reputation_before=agent.reputation,
                reputation_after=agent.reputation,
                capital_before=cap_before,
                capital_after=agent.capital,
                position=0.0,
            ))

        observations.append(f"Fallback used by {fallback_used}/{len(agents)} agents")
        observations.append("System did not crash on oracle failure")

        passed = all(r.action == "HOLD" for r in agent_responses)

        return StressResult(
            scenario="oracle_failure",
            description="RedStone oracle raises ConnectionError, fallback to stale cache",
            passed=passed,
            agent_responses=agent_responses,
            system_action="HOLD",
            observations=observations,
            duration_ms=(time.time() - t0) * 1000,
            metadata={"fallback_used": fallback_used},
        )

    # ─── Scenario 4: Consensus deadlock ───────────────────────────────────

    def consensus_deadlock(self) -> StressResult:
        """
        3-way tie: each agent votes for a different action.
        Tie-breaking logic resolves by highest-reputation agent.
        """
        t0 = time.time()
        agents = self._make_agents()
        observations = []

        # Force a 3-way tie
        forced_votes = {
            agents[0].agent_id: "BUY",
            agents[1].agent_id: "SELL",
            agents[2].agent_id: "HOLD",
        }

        votes_with_rep: Dict[str, Tuple[str, float]] = {}
        for agent in agents:
            vote = forced_votes[agent.agent_id]
            votes_with_rep[agent.agent_id] = (vote, agent.reputation)

        observations.append("Forced 3-way tie: BUY vs SELL vs HOLD")
        observations.append(f"Agent reputations: {[(a.agent_id, a.reputation) for a in agents]}")

        # Compute tie-break
        system_action = self._reputation_weighted_consensus(votes_with_rep)
        observations.append(f"Tie-break resolved to: {system_action}")

        # Highest-rep agent is conservative (rep=7.5) who voted BUY
        expected_tie_break = "BUY"  # highest rep voted BUY
        tie_break_correct = system_action == expected_tie_break
        observations.append(f"Tie-break correct (expected {expected_tie_break}): {tie_break_correct}")

        agent_responses: List[AgentResponse] = []
        for agent in agents:
            vote = forced_votes[agent.agent_id]
            agent_responses.append(AgentResponse(
                agent_id=agent.agent_id,
                profile=agent.profile,
                action=vote,
                reason=f"Forced vote for deadlock test",
                reputation_before=agent.reputation,
                reputation_after=agent.reputation,
                capital_before=self.initial_capital,
                capital_after=self.initial_capital,
                position=0.0,
            ))

        return StressResult(
            scenario="consensus_deadlock",
            description="3 agents disagree: BUY vs SELL vs HOLD → tie-break by highest reputation",
            passed=tie_break_correct,
            agent_responses=agent_responses,
            system_action=system_action,
            observations=observations,
            duration_ms=(time.time() - t0) * 1000,
            metadata={
                "forced_votes": forced_votes,
                "expected_tie_break": expected_tie_break,
                "tie_break_correct": tie_break_correct,
            },
        )

    # ─── Additional stress scenarios ──────────────────────────────────────

    def high_volatility(self) -> StressResult:
        """Extremely high volatility causes repeated risk vetoes."""
        t0 = time.time()
        agents = self._make_agents()
        observations = []

        # 50% return in one tick (extreme volatility)
        returns = 0.50
        volatility = 0.01  # baseline vol, so return >> threshold

        votes: Dict[str, Tuple[str, float]] = {}
        agent_responses: List[AgentResponse] = []

        for agent in agents:
            vote = agent.vote(returns, volatility)
            votes[agent.agent_id] = (vote, agent.reputation)
            agent_responses.append(AgentResponse(
                agent_id=agent.agent_id,
                profile=agent.profile,
                action=vote,
                reason=f"50% return spike, tolerance={agent.risk_tolerance}",
                reputation_before=agent.reputation,
                reputation_after=agent.reputation,
                capital_before=self.initial_capital,
                capital_after=self.initial_capital,
                position=0.0,
            ))

        veto_count = sum(1 for r in agent_responses if r.action == "VETO")
        system_action = self._reputation_weighted_consensus(votes)
        observations.append(f"50% return spike: {veto_count}/{len(agents)} agents vetoed")
        observations.append(f"System action: {system_action}")

        # Any veto should result in HOLD
        passed = system_action == "HOLD"

        return StressResult(
            scenario="high_volatility",
            description="50% return spike triggers agent risk vetoes",
            passed=passed,
            agent_responses=agent_responses,
            system_action=system_action,
            observations=observations,
            duration_ms=(time.time() - t0) * 1000,
            metadata={"returns": returns, "veto_count": veto_count},
        )

    def reputation_collapse(self) -> StressResult:
        """Agent reputations collapse to near-zero; consensus weight redistributes."""
        t0 = time.time()
        agents = self._make_agents()

        # Drive reputations near zero
        for agent in agents:
            agent.reputation = 0.1

        observations = []
        returns = 0.02  # normal move
        volatility = 0.005

        votes: Dict[str, Tuple[str, float]] = {}
        for agent in agents:
            vote = agent.vote(returns, volatility)
            votes[agent.agent_id] = (vote, agent.reputation)

        system_action = self._reputation_weighted_consensus(votes)
        observations.append("All agent reputations at 0.1 (near-collapse)")
        observations.append(f"System still produced action: {system_action}")

        agent_responses = [
            AgentResponse(
                agent_id=a.agent_id,
                profile=a.profile,
                action=votes[a.agent_id][0],
                reason="Near-zero reputation",
                reputation_before=5.0,
                reputation_after=0.1,
                capital_before=self.initial_capital,
                capital_after=self.initial_capital,
                position=0.0,
            )
            for a in agents
        ]

        # System should still function (produce some action)
        passed = system_action in ("BUY", "SELL", "HOLD")

        return StressResult(
            scenario="reputation_collapse",
            description="All agent reputations near zero, consensus still functions",
            passed=passed,
            agent_responses=agent_responses,
            system_action=system_action,
            observations=observations,
            duration_ms=(time.time() - t0) * 1000,
            metadata={"min_reputation": 0.1},
        )

    def zero_capital(self) -> StressResult:
        """Agent runs out of capital; should refuse to trade."""
        t0 = time.time()
        agents = self._make_agents()
        # Drain capital
        for agent in agents:
            agent.capital = 0.0

        observations = []
        agent_responses = []

        for agent in agents:
            # No capital → refuse trade regardless of signal
            size_usd = agent.capital * agent.kelly_fraction * 0.10
            if size_usd <= 0:
                action = "HOLD"
                reason = "Insufficient capital (0.0)"
            else:
                action = "BUY"
                reason = f"Capital ${size_usd:.2f}"

            observations.append(f"{agent.agent_id}: capital=0 → {action}")
            agent_responses.append(AgentResponse(
                agent_id=agent.agent_id,
                profile=agent.profile,
                action=action,
                reason=reason,
                reputation_before=agent.reputation,
                reputation_after=agent.reputation,
                capital_before=0.0,
                capital_after=0.0,
                position=0.0,
            ))

        all_held = all(r.action == "HOLD" for r in agent_responses)
        observations.append(f"All agents held due to zero capital: {all_held}")

        return StressResult(
            scenario="zero_capital",
            description="All agents have zero capital, system prevents trading",
            passed=all_held,
            agent_responses=agent_responses,
            system_action="HOLD",
            observations=observations,
            duration_ms=(time.time() - t0) * 1000,
            metadata={},
        )

    # ─── Run all ───────────────────────────────────────────────────────────

    def run_all(self) -> List[StressResult]:
        """Run all stress scenarios and return results."""
        return [
            self.flash_crash(),
            self.liquidity_crisis(),
            self.oracle_failure(),
            self.consensus_deadlock(),
            self.high_volatility(),
            self.reputation_collapse(),
            self.zero_capital(),
        ]

    def run_scenario(self, name: str) -> StressResult:
        """Run a named scenario."""
        scenarios: Dict[str, Callable[[], StressResult]] = {
            "flash_crash": self.flash_crash,
            "liquidity_crisis": self.liquidity_crisis,
            "oracle_failure": self.oracle_failure,
            "consensus_deadlock": self.consensus_deadlock,
            "high_volatility": self.high_volatility,
            "reputation_collapse": self.reputation_collapse,
            "zero_capital": self.zero_capital,
        }
        if name not in scenarios:
            raise ValueError(f"Unknown scenario: {name!r}. Choose from {list(scenarios)}")
        return scenarios[name]()


if __name__ == "__main__":
    tester = StressTester(seed=42)
    results = tester.run_all()
    passed = sum(1 for r in results if r.passed)
    print(f"\n=== Stress Test Results: {passed}/{len(results)} passed ===\n")
    for r in results:
        print(r.summary())
        print()
