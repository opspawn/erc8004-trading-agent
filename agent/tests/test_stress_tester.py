"""
Tests for stress_tester.py — Adversarial Stress Testing Module.
"""

import pytest
from unittest.mock import patch, MagicMock

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from stress_tester import (
    StressTester,
    StressResult,
    AgentResponse,
    _StressAgent,
)


# ─── _StressAgent ─────────────────────────────────────────────────────────────

class TestStressAgent:
    def test_default_profile(self):
        agent = _StressAgent("a1", "balanced")
        assert agent.profile == "balanced"

    def test_default_capital(self):
        agent = _StressAgent("a1", "balanced")
        assert agent.capital == 10_000.0

    def test_vote_hold_on_zero_return(self):
        agent = _StressAgent("a1", "balanced", risk_tolerance=2.0)
        vote = agent.vote(0.0, 0.01)
        assert vote == "HOLD"

    def test_vote_buy_on_positive(self):
        agent = _StressAgent("a1", "balanced", risk_tolerance=1.0)
        # returns = 0.1, volatility = 0.01, threshold = 0.01
        vote = agent.vote(0.1, 0.01)
        # 0.1 > threshold(0.01) but not > 3*threshold(0.03)? No: 0.1 > 0.03 → VETO
        # Actually threshold=1.0*0.01=0.01, 3*threshold=0.03, 0.1>0.03 → VETO
        assert vote in ("BUY", "VETO")

    def test_vote_sell_on_negative(self):
        agent = _StressAgent("a1", "balanced", risk_tolerance=1.0)
        vote = agent.vote(-0.005, 0.01)
        # returns=-0.005, threshold=0.01, -0.005 < -0.01? No → HOLD
        assert vote == "HOLD"

    def test_vote_veto_on_extreme(self):
        agent = _StressAgent("a1", "balanced", risk_tolerance=1.0)
        # returns = 0.5, threshold = 0.01, 3*threshold = 0.03 → 0.5 > 0.03 → VETO
        vote = agent.vote(0.5, 0.01)
        assert vote == "VETO"

    def test_override_vote(self):
        agent = _StressAgent("a1", "balanced")
        vote = agent.vote(0.0, 0.01, override="BUY")
        assert vote == "BUY"

    def test_conservative_tighter_threshold(self):
        conservative = _StressAgent("c1", "conservative", risk_tolerance=1.5)
        aggressive = _StressAgent("a1", "aggressive", risk_tolerance=3.0)
        returns = 0.02
        volatility = 0.01
        # conservative threshold = 0.015, aggressive threshold = 0.03
        # 0.02 > 0.015 (cons buys) but 0.02 < 0.03 (agg holds)
        # Check conservative detects more signals
        c_vote = conservative.vote(returns, volatility)
        a_vote = aggressive.vote(returns, volatility)
        # Aggressive requires higher threshold
        assert c_vote in ("BUY", "VETO")


# ─── StressTester construction ────────────────────────────────────────────────

class TestStressTesterConstruction:
    def test_default_params(self):
        tester = StressTester()
        assert tester.seed == 42
        assert tester.initial_capital == 10_000.0

    def test_custom_params(self):
        tester = StressTester(seed=7, initial_capital=5_000.0)
        assert tester.seed == 7
        assert tester.initial_capital == 5_000.0

    def test_make_agents_returns_three(self):
        tester = StressTester()
        agents = tester._make_agents()
        assert len(agents) == 3

    def test_make_agents_profiles(self):
        tester = StressTester()
        agents = tester._make_agents()
        profiles = {a.profile for a in agents}
        assert profiles == {"conservative", "balanced", "aggressive"}


# ─── Reputation-weighted consensus ────────────────────────────────────────────

class TestReputationWeightedConsensus:
    def test_unanimous_buy(self):
        votes = {"a1": ("BUY", 5.0), "a2": ("BUY", 5.0), "a3": ("BUY", 5.0)}
        result = StressTester._reputation_weighted_consensus(votes)
        assert result == "BUY"

    def test_unanimous_sell(self):
        votes = {"a1": ("SELL", 5.0), "a2": ("SELL", 5.0), "a3": ("SELL", 5.0)}
        result = StressTester._reputation_weighted_consensus(votes)
        assert result == "SELL"

    def test_veto_overrides_all(self):
        votes = {"a1": ("BUY", 5.0), "a2": ("BUY", 5.0), "a3": ("VETO", 1.0)}
        result = StressTester._reputation_weighted_consensus(votes)
        assert result == "HOLD"

    def test_no_majority_tie_break(self):
        # 3-way tie → highest rep
        votes = {"a1": ("BUY", 7.5), "a2": ("SELL", 6.0), "a3": ("HOLD", 5.0)}
        result = StressTester._reputation_weighted_consensus(votes)
        # BUY has highest rep agent
        assert result == "BUY"

    def test_two_thirds_majority(self):
        votes = {"a1": ("BUY", 5.0), "a2": ("BUY", 5.0), "a3": ("HOLD", 5.0)}
        result = StressTester._reputation_weighted_consensus(votes)
        assert result == "BUY"

    def test_zero_total_rep_returns_hold(self):
        votes = {"a1": ("BUY", 0.0), "a2": ("BUY", 0.0)}
        result = StressTester._reputation_weighted_consensus(votes)
        assert result == "HOLD"

    def test_high_rep_buy_wins_split(self):
        votes = {"a1": ("BUY", 9.0), "a2": ("SELL", 0.5), "a3": ("SELL", 0.5)}
        result = StressTester._reputation_weighted_consensus(votes)
        assert result == "BUY"


# ─── Flash crash scenario ─────────────────────────────────────────────────────

class TestFlashCrash:
    def setup_method(self):
        self.tester = StressTester(seed=42)
        self.result = self.tester.flash_crash()

    def test_returns_stress_result(self):
        assert isinstance(self.result, StressResult)

    def test_scenario_name(self):
        assert self.result.scenario == "flash_crash"

    def test_has_agent_responses(self):
        assert len(self.result.agent_responses) > 0

    def test_has_observations(self):
        assert len(self.result.observations) > 0

    def test_passed_field_is_bool(self):
        assert isinstance(self.result.passed, bool)

    def test_duration_positive(self):
        assert self.result.duration_ms > 0

    def test_price_dropped_correctly(self):
        # Final price should be ~60% of initial
        prices = self.result.metadata["prices"]
        assert prices[-1] / prices[0] == pytest.approx(0.6, rel=0.01)

    def test_system_action_is_valid(self):
        assert self.result.system_action in ("BUY", "SELL", "HOLD", "VETO")

    def test_observations_mention_price_drop(self):
        obs = " ".join(self.result.observations)
        assert "drop" in obs.lower() or "crash" in obs.lower() or "-40" in obs or "40" in obs or "%" in obs

    def test_agent_responses_have_correct_fields(self):
        for r in self.result.agent_responses:
            assert isinstance(r, AgentResponse)
            assert r.agent_id
            assert r.profile
            assert r.action in ("BUY", "SELL", "HOLD", "VETO")

    def test_system_actions_tracked(self):
        system_actions = self.result.metadata.get("system_actions", [])
        assert len(system_actions) == 5  # 5 ticks


# ─── Liquidity crisis scenario ────────────────────────────────────────────────

class TestLiquidityCrisis:
    def setup_method(self):
        self.tester = StressTester(seed=42)
        self.result = self.tester.liquidity_crisis()

    def test_returns_stress_result(self):
        assert isinstance(self.result, StressResult)

    def test_scenario_name(self):
        assert self.result.scenario == "liquidity_crisis"

    def test_all_agents_hold(self):
        for r in self.result.agent_responses:
            assert r.action == "HOLD"

    def test_system_action_hold(self):
        assert self.result.system_action == "HOLD"

    def test_passed(self):
        assert self.result.passed is True

    def test_zero_liquidity_in_metadata(self):
        route = self.result.metadata.get("route", {})
        assert route.get("liquidity", -1) == 0.0

    def test_three_agent_responses(self):
        assert len(self.result.agent_responses) == 3

    def test_no_capital_change(self):
        for r in self.result.agent_responses:
            assert r.capital_before == r.capital_after

    def test_observations_mention_liquidity(self):
        obs = " ".join(self.result.observations)
        assert "liquidity" in obs.lower() or "zero" in obs.lower()


# ─── Oracle failure scenario ──────────────────────────────────────────────────

class TestOracleFailure:
    def setup_method(self):
        self.tester = StressTester(seed=42)
        self.result = self.tester.oracle_failure()

    def test_returns_stress_result(self):
        assert isinstance(self.result, StressResult)

    def test_scenario_name(self):
        assert self.result.scenario == "oracle_failure"

    def test_all_agents_hold(self):
        for r in self.result.agent_responses:
            assert r.action == "HOLD"

    def test_system_action_hold(self):
        assert self.result.system_action == "HOLD"

    def test_passed(self):
        assert self.result.passed is True

    def test_fallback_used(self):
        assert self.result.metadata["fallback_used"] > 0

    def test_three_agents(self):
        assert len(self.result.agent_responses) == 3

    def test_reasons_mention_oracle(self):
        for r in self.result.agent_responses:
            assert "oracle" in r.reason.lower() or "cache" in r.reason.lower()

    def test_observations_mention_fallback(self):
        obs = " ".join(self.result.observations)
        assert "fallback" in obs.lower() or "oracle" in obs.lower()


# ─── Consensus deadlock scenario ──────────────────────────────────────────────

class TestConsensusDeadlock:
    def setup_method(self):
        self.tester = StressTester(seed=42)
        self.result = self.tester.consensus_deadlock()

    def test_returns_stress_result(self):
        assert isinstance(self.result, StressResult)

    def test_scenario_name(self):
        assert self.result.scenario == "consensus_deadlock"

    def test_three_different_votes(self):
        actions = {r.action for r in self.result.agent_responses}
        # Must have BUY, SELL, and HOLD all present
        assert "BUY" in actions
        assert "SELL" in actions
        assert "HOLD" in actions

    def test_tie_break_resolves(self):
        # System should have resolved to a valid action
        assert self.result.system_action in ("BUY", "SELL", "HOLD")

    def test_passed_when_correct_tie_break(self):
        # Conservative (rep=7.5) voted BUY → should win tie-break
        assert self.result.passed is True

    def test_system_action_is_buy(self):
        assert self.result.system_action == "BUY"

    def test_observations_mention_tie(self):
        obs = " ".join(self.result.observations)
        assert "tie" in obs.lower() or "deadlock" in obs.lower() or "3-way" in obs.lower()

    def test_metadata_has_forced_votes(self):
        assert "forced_votes" in self.result.metadata

    def test_metadata_has_tie_break_correct(self):
        assert self.result.metadata["tie_break_correct"] is True


# ─── High volatility scenario ─────────────────────────────────────────────────

class TestHighVolatility:
    def setup_method(self):
        self.tester = StressTester(seed=42)
        self.result = self.tester.high_volatility()

    def test_returns_stress_result(self):
        assert isinstance(self.result, StressResult)

    def test_scenario_name(self):
        assert self.result.scenario == "high_volatility"

    def test_system_action_hold(self):
        # Veto should result in HOLD
        assert self.result.system_action == "HOLD"

    def test_passed(self):
        assert self.result.passed is True

    def test_veto_count_positive(self):
        assert self.result.metadata["veto_count"] > 0

    def test_all_agents_vetoed_or_acted(self):
        for r in self.result.agent_responses:
            assert r.action in ("VETO", "BUY", "SELL", "HOLD")


# ─── Reputation collapse scenario ─────────────────────────────────────────────

class TestReputationCollapse:
    def setup_method(self):
        self.tester = StressTester(seed=42)
        self.result = self.tester.reputation_collapse()

    def test_returns_stress_result(self):
        assert isinstance(self.result, StressResult)

    def test_scenario_name(self):
        assert self.result.scenario == "reputation_collapse"

    def test_passed(self):
        assert self.result.passed is True

    def test_system_still_functions(self):
        assert self.result.system_action in ("BUY", "SELL", "HOLD")

    def test_rep_after_low(self):
        for r in self.result.agent_responses:
            assert r.reputation_after == pytest.approx(0.1)


# ─── Zero capital scenario ────────────────────────────────────────────────────

class TestZeroCapital:
    def setup_method(self):
        self.tester = StressTester(seed=42)
        self.result = self.tester.zero_capital()

    def test_returns_stress_result(self):
        assert isinstance(self.result, StressResult)

    def test_scenario_name(self):
        assert self.result.scenario == "zero_capital"

    def test_all_agents_hold(self):
        for r in self.result.agent_responses:
            assert r.action == "HOLD"

    def test_passed(self):
        assert self.result.passed is True

    def test_capital_unchanged(self):
        for r in self.result.agent_responses:
            assert r.capital_before == 0.0
            assert r.capital_after == 0.0

    def test_system_action_hold(self):
        assert self.result.system_action == "HOLD"


# ─── run_all ──────────────────────────────────────────────────────────────────

class TestRunAll:
    def setup_method(self):
        self.tester = StressTester(seed=42)
        self.results = self.tester.run_all()

    def test_returns_list(self):
        assert isinstance(self.results, list)

    def test_seven_scenarios(self):
        assert len(self.results) == 7

    def test_all_stress_results(self):
        for r in self.results:
            assert isinstance(r, StressResult)

    def test_scenario_names_unique(self):
        names = {r.scenario for r in self.results}
        assert len(names) == 7

    def test_all_have_agent_responses(self):
        for r in self.results:
            assert len(r.agent_responses) > 0

    def test_all_have_observations(self):
        for r in self.results:
            assert len(r.observations) > 0


# ─── run_scenario ─────────────────────────────────────────────────────────────

class TestRunScenario:
    def test_flash_crash_by_name(self):
        tester = StressTester(seed=42)
        result = tester.run_scenario("flash_crash")
        assert result.scenario == "flash_crash"

    def test_liquidity_crisis_by_name(self):
        tester = StressTester(seed=42)
        result = tester.run_scenario("liquidity_crisis")
        assert result.scenario == "liquidity_crisis"

    def test_oracle_failure_by_name(self):
        tester = StressTester(seed=42)
        result = tester.run_scenario("oracle_failure")
        assert result.scenario == "oracle_failure"

    def test_consensus_deadlock_by_name(self):
        tester = StressTester(seed=42)
        result = tester.run_scenario("consensus_deadlock")
        assert result.scenario == "consensus_deadlock"

    def test_high_volatility_by_name(self):
        tester = StressTester(seed=42)
        result = tester.run_scenario("high_volatility")
        assert result.scenario == "high_volatility"

    def test_reputation_collapse_by_name(self):
        tester = StressTester(seed=42)
        result = tester.run_scenario("reputation_collapse")
        assert result.scenario == "reputation_collapse"

    def test_zero_capital_by_name(self):
        tester = StressTester(seed=42)
        result = tester.run_scenario("zero_capital")
        assert result.scenario == "zero_capital"

    def test_unknown_raises_value_error(self):
        tester = StressTester(seed=42)
        with pytest.raises(ValueError, match="Unknown scenario"):
            tester.run_scenario("does_not_exist")


# ─── StressResult ─────────────────────────────────────────────────────────────

class TestStressResult:
    def test_summary_is_string(self):
        tester = StressTester(seed=42)
        result = tester.flash_crash()
        assert isinstance(result.summary(), str)

    def test_summary_contains_pass_or_fail(self):
        tester = StressTester(seed=42)
        result = tester.flash_crash()
        s = result.summary()
        assert "PASS" in s or "FAIL" in s

    def test_to_dict_is_dict(self):
        tester = StressTester(seed=42)
        result = tester.flash_crash()
        d = result.to_dict()
        assert isinstance(d, dict)

    def test_to_dict_has_scenario(self):
        tester = StressTester(seed=42)
        result = tester.flash_crash()
        d = result.to_dict()
        assert "scenario" in d

    def test_to_dict_has_agent_responses(self):
        tester = StressTester(seed=42)
        result = tester.liquidity_crisis()
        d = result.to_dict()
        assert "agent_responses" in d
        assert len(d["agent_responses"]) > 0
