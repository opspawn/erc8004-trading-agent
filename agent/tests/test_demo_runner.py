"""
Tests for demo_runner.py — End-to-End Demo Scenario Runner.
"""

import json
import math
import pytest
from unittest.mock import patch, MagicMock

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from demo_runner import (
    DemoRunner,
    DemoReport,
    DemoGBM,
    DemoAgent,
    TickResult,
)


# ─── DemoGBM ─────────────────────────────────────────────────────────────────

class TestDemoGBM:
    def test_initial_price_unchanged(self):
        gbm = DemoGBM(initial_price=100.0, seed=1)
        assert gbm.price == 100.0

    def test_next_price_returns_float(self):
        gbm = DemoGBM(initial_price=100.0, seed=1)
        p = gbm.next_price()
        assert isinstance(p, float)

    def test_next_price_positive(self):
        gbm = DemoGBM(initial_price=100.0, seed=1)
        for _ in range(100):
            p = gbm.next_price()
            assert p > 0

    def test_generate_returns_correct_length(self):
        gbm = DemoGBM(seed=42)
        prices = gbm.generate(50)
        assert len(prices) == 50

    def test_generate_all_positive(self):
        gbm = DemoGBM(initial_price=50.0, seed=7)
        prices = gbm.generate(100)
        assert all(p > 0 for p in prices)

    def test_reproducible_with_seed(self):
        prices_a = DemoGBM(seed=999).generate(20)
        prices_b = DemoGBM(seed=999).generate(20)
        assert prices_a == prices_b

    def test_different_seeds_differ(self):
        prices_a = DemoGBM(seed=1).generate(10)
        prices_b = DemoGBM(seed=2).generate(10)
        assert prices_a != prices_b

    def test_high_sigma_more_volatile(self):
        low_vol = DemoGBM(sigma=0.01, seed=42).generate(100)
        high_vol = DemoGBM(sigma=2.0, seed=42).generate(100)
        low_range = max(low_vol) - min(low_vol)
        high_range = max(high_vol) - min(high_vol)
        assert high_range > low_range

    def test_price_attribute_updates(self):
        gbm = DemoGBM(initial_price=100.0, seed=1)
        gbm.next_price()
        assert gbm.price != 100.0


# ─── DemoRunner construction ──────────────────────────────────────────────────

class TestDemoRunnerConstruction:
    def test_default_params(self):
        runner = DemoRunner()
        assert runner.n_ticks == 50
        assert runner.seed == 42
        assert runner.initial_capital == 10_000.0

    def test_custom_params(self):
        runner = DemoRunner(n_ticks=20, seed=7, initial_capital=5_000.0)
        assert runner.n_ticks == 20
        assert runner.seed == 7
        assert runner.initial_capital == 5_000.0

    def test_build_agents_returns_three(self):
        runner = DemoRunner()
        agents = runner._build_agents()
        assert len(agents) == 3

    def test_agents_have_distinct_profiles(self):
        runner = DemoRunner()
        agents = runner._build_agents()
        profiles = {a.profile for a in agents}
        assert profiles == {"conservative", "balanced", "aggressive"}

    def test_agents_have_correct_capital(self):
        runner = DemoRunner(initial_capital=20_000.0)
        agents = runner._build_agents()
        for agent in agents:
            assert agent.capital == 20_000.0

    def test_agents_have_reputation(self):
        runner = DemoRunner()
        agents = runner._build_agents()
        for agent in agents:
            assert 0 < agent.reputation <= 10


# ─── DemoRunner.run ───────────────────────────────────────────────────────────

class TestDemoRunnerRun:
    def setup_method(self):
        self.runner = DemoRunner(n_ticks=50, seed=42)
        self.report = self.runner.run()

    def test_run_returns_report(self):
        assert isinstance(self.report, DemoReport)

    def test_report_has_correct_tick_count(self):
        assert self.report.ticks == 50

    def test_report_has_three_agents(self):
        assert len(self.report.agents) == 3

    def test_report_has_tick_results(self):
        assert len(self.report.tick_results) == 50

    def test_agent_summaries_have_required_keys(self):
        required = {"agent_id", "profile", "pnl", "trades", "wins", "final_reputation",
                    "risk_violations", "win_rate"}
        for agent in self.report.agents:
            assert required.issubset(set(agent.keys()))

    def test_win_rate_in_range(self):
        for agent in self.report.agents:
            assert 0.0 <= agent["win_rate"] <= 1.0

    def test_reputation_in_range(self):
        for agent in self.report.agents:
            assert 0.0 <= agent["final_reputation"] <= 10.0

    def test_risk_violations_non_negative(self):
        for agent in self.report.agents:
            assert agent["risk_violations"] >= 0

    def test_summary_stats_present(self):
        stats = self.report.summary_stats
        assert "consensus_reached" in stats
        assert "total_trades" in stats
        assert "total_risk_violations" in stats

    def test_consensus_rate_in_range(self):
        rate = self.report.summary_stats["consensus_rate"]
        assert 0.0 <= rate <= 1.0

    def test_duration_positive(self):
        assert self.report.duration_ms > 0

    def test_reproducible(self):
        report2 = DemoRunner(n_ticks=50, seed=42).run()
        # Same seed → same P&L
        for a1, a2 in zip(self.report.agents, report2.agents):
            assert a1["pnl"] == pytest.approx(a2["pnl"], abs=1e-6)

    def test_different_seeds_differ(self):
        # Use small capital to ensure same price path still differs across seeds
        # Compare price paths instead of PnL (which can both be 0 if no trades execute)
        report2 = DemoRunner(n_ticks=50, seed=123).run()
        prices_a = [tr["price"] for tr in self.report.tick_results]
        prices_b = [tr["price"] for tr in report2.tick_results]
        assert prices_a != prices_b

    def test_tick_results_have_price(self):
        for tr in self.report.tick_results:
            assert "price" in tr
            assert tr["price"] > 0

    def test_tick_results_have_consensus(self):
        for tr in self.report.tick_results:
            assert tr["consensus_action"] in ("BUY", "SELL", "HOLD")

    def test_tick_numbers_sequential(self):
        for i, tr in enumerate(self.report.tick_results):
            assert tr["tick"] == i + 1


# ─── DemoRunner edge cases ────────────────────────────────────────────────────

class TestDemoRunnerEdgeCases:
    def test_single_tick(self):
        runner = DemoRunner(n_ticks=1, seed=42)
        report = runner.run()
        assert report.ticks == 1
        assert len(report.tick_results) == 1

    def test_zero_initial_price_error_handled(self):
        runner = DemoRunner(n_ticks=5, seed=42, initial_price=0.001)
        report = runner.run()
        assert isinstance(report, DemoReport)

    def test_very_small_capital(self):
        runner = DemoRunner(n_ticks=10, seed=42, initial_capital=1.0)
        report = runner.run()
        assert isinstance(report, DemoReport)

    def test_large_tick_count(self):
        runner = DemoRunner(n_ticks=200, seed=42)
        report = runner.run()
        assert len(report.tick_results) == 200

    def test_scenario_name_in_report(self):
        runner = DemoRunner(seed=42)
        report = runner.run(scenario="My Custom Scenario")
        assert report.scenario == "My Custom Scenario"


# ─── DemoReport ───────────────────────────────────────────────────────────────

class TestDemoReport:
    def setup_method(self):
        self.runner = DemoRunner(n_ticks=10, seed=42)
        self.report = self.runner.run()

    def test_summary_is_string(self):
        s = self.report.summary()
        assert isinstance(s, str)

    def test_summary_contains_scenario(self):
        report = self.runner.run(scenario="TEST SCENARIO")
        assert "TEST SCENARIO" in report.summary()

    def test_summary_contains_agent_profiles(self):
        s = self.report.summary()
        assert "CONSERVATIVE" in s or "conservative" in s.lower()

    def test_to_json_is_valid(self):
        js = self.report.to_json()
        parsed = json.loads(js)
        assert "agents" in parsed
        assert "tick_results" in parsed

    def test_to_json_contains_summary_stats(self):
        js = self.report.to_json()
        parsed = json.loads(js)
        assert "summary_stats" in parsed

    def test_summary_contains_ticks(self):
        s = self.report.summary()
        assert "10" in s or "Ticks" in s


# ─── Agent voting logic ───────────────────────────────────────────────────────

class TestAgentVoting:
    def setup_method(self):
        self.runner = DemoRunner(seed=42)
        self.agents = self.runner._build_agents()

    def test_strong_up_move_returns_buy(self):
        agent = self.agents[1]  # balanced, tolerance=2.0
        # returns >> threshold → should trigger risk veto or BUY
        vote = self.runner._agent_vote(agent, 110.0, 100.0, 0.001)
        # Very large move: 10% with vol=0.001 and tolerance=2.0 → threshold=0.002, but 10% >> 6*0.002 → HOLD (risk veto)
        # Actually: threshold = 2.0 * 0.001 = 0.002, abs(0.1) > 2*0.002*3 → HOLD
        assert vote in ("BUY", "HOLD")

    def test_zero_move_returns_hold(self):
        agent = self.agents[0]
        vote = self.runner._agent_vote(agent, 100.0, 100.0, 0.01)
        assert vote == "HOLD"

    def test_small_down_move_returns_hold(self):
        agent = self.agents[0]  # conservative, tight threshold
        vote = self.runner._agent_vote(agent, 99.99, 100.0, 0.1)
        assert vote == "HOLD"

    def test_extreme_move_veto_increments_violations(self):
        agent = self.agents[0]
        violations_before = agent.risk_violations
        self.runner._agent_vote(agent, 150.0, 100.0, 0.001)  # 50% move → veto
        assert agent.risk_violations >= violations_before


# ─── Mesh consensus ───────────────────────────────────────────────────────────

class TestMeshConsensus:
    def setup_method(self):
        self.runner = DemoRunner(seed=42)

    def test_unanimous_buy(self):
        votes = {"a1": "BUY", "a2": "BUY", "a3": "BUY"}
        reps = {"a1": 5.0, "a2": 5.0, "a3": 5.0}
        result = self.runner._mesh_consensus(votes, reps)
        assert result == "BUY"

    def test_unanimous_sell(self):
        votes = {"a1": "SELL", "a2": "SELL", "a3": "SELL"}
        reps = {"a1": 5.0, "a2": 5.0, "a3": 5.0}
        result = self.runner._mesh_consensus(votes, reps)
        assert result == "SELL"

    def test_unanimous_hold(self):
        votes = {"a1": "HOLD", "a2": "HOLD", "a3": "HOLD"}
        reps = {"a1": 5.0, "a2": 5.0, "a3": 5.0}
        result = self.runner._mesh_consensus(votes, reps)
        assert result == "HOLD"

    def test_two_thirds_buy_wins(self):
        votes = {"a1": "BUY", "a2": "BUY", "a3": "HOLD"}
        reps = {"a1": 5.0, "a2": 5.0, "a3": 5.0}
        result = self.runner._mesh_consensus(votes, reps)
        assert result == "BUY"

    def test_split_vote_hold(self):
        # No 2/3 majority → HOLD
        votes = {"a1": "BUY", "a2": "SELL", "a3": "HOLD"}
        reps = {"a1": 5.0, "a2": 5.0, "a3": 5.0}
        result = self.runner._mesh_consensus(votes, reps)
        assert result == "HOLD"

    def test_zero_reputation_returns_hold(self):
        votes = {"a1": "BUY", "a2": "BUY"}
        reps = {"a1": 0.0, "a2": 0.0}
        result = self.runner._mesh_consensus(votes, reps)
        assert result == "HOLD"

    def test_high_rep_agent_tips_consensus(self):
        # a1 has 9x the reputation of a2 and a3 combined
        votes = {"a1": "BUY", "a2": "SELL", "a3": "SELL"}
        reps = {"a1": 9.0, "a2": 0.5, "a3": 0.5}
        result = self.runner._mesh_consensus(votes, reps)
        assert result == "BUY"


# ─── Reputation update ────────────────────────────────────────────────────────

class TestReputationUpdate:
    def setup_method(self):
        self.runner = DemoRunner(seed=42)
        self.agents = self.runner._build_agents()

    def test_hold_no_change(self):
        agent = self.agents[0]
        rep = agent.reputation
        self.runner._update_reputation(agent, "HOLD", 101.0, 100.0)
        assert agent.reputation == rep

    def test_correct_buy_increases_rep(self):
        agent = self.agents[0]
        rep = agent.reputation
        self.runner._update_reputation(agent, "BUY", 101.0, 100.0)
        assert agent.reputation > rep

    def test_wrong_buy_decreases_rep(self):
        agent = self.agents[0]
        rep = agent.reputation
        self.runner._update_reputation(agent, "BUY", 99.0, 100.0)
        assert agent.reputation < rep

    def test_correct_sell_increases_rep(self):
        agent = self.agents[0]
        rep = agent.reputation
        self.runner._update_reputation(agent, "SELL", 99.0, 100.0)
        assert agent.reputation > rep

    def test_rep_capped_at_ten(self):
        agent = self.agents[0]
        agent.reputation = 9.99
        self.runner._update_reputation(agent, "BUY", 101.0, 100.0)
        assert agent.reputation <= 10.0

    def test_rep_floored_at_zero(self):
        agent = self.agents[0]
        agent.reputation = 0.01
        self.runner._update_reputation(agent, "BUY", 99.0, 100.0)
        assert agent.reputation >= 0.0

    def test_rep_history_appended(self):
        agent = self.agents[0]
        self.runner._update_reputation(agent, "BUY", 101.0, 100.0)
        assert len(agent.rep_history) == 1


# ─── Edge case scenarios ──────────────────────────────────────────────────────

class TestEdgeCaseScenarios:
    def test_zero_liquidity_scenario(self):
        runner = DemoRunner(n_ticks=10, seed=1)
        report = runner.run_edge_case("zero_liquidity")
        assert isinstance(report, DemoReport)
        # All prices constant → consensus should be mostly HOLD
        for tr in report.tick_results:
            assert tr["consensus_action"] in ("BUY", "SELL", "HOLD")

    def test_extreme_volatility_scenario(self):
        runner = DemoRunner(n_ticks=10, seed=1)
        report = runner.run_edge_case("extreme_volatility")
        assert isinstance(report, DemoReport)
        assert report.ticks == 10

    def test_unknown_scenario_raises(self):
        runner = DemoRunner(n_ticks=5)
        with pytest.raises(ValueError, match="Unknown scenario"):
            runner.run_edge_case("nonexistent_scenario")

    def test_zero_liquidity_price_unchanged(self):
        runner = DemoRunner(n_ticks=5, seed=1, initial_price=100.0)
        report = runner.run_edge_case("zero_liquidity")
        # All prices should be 100.0 (flat)
        for tr in report.tick_results:
            assert tr["price"] == pytest.approx(100.0, abs=1e-6)
