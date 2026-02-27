"""
S57 Tests: Agent Reasoning Narrative + Strategy Comparison Endpoints
Target: 80+ tests covering /api/v1/agent/narrative and /api/v1/strategies/compare
"""
import threading
import time
import pytest
import requests

from demo_server import DemoServer, get_s57_agent_narrative, get_s57_strategy_compare

# ── Fixtures ──────────────────────────────────────────────────────────────────

_S57_PORT = 8191


@pytest.fixture(scope="module")
def server():
    srv = DemoServer(port=_S57_PORT)
    srv.start()
    time.sleep(0.4)
    yield srv
    srv.stop()


def _get(path, server):
    return requests.get(f"http://localhost:{_S57_PORT}{path}", timeout=5)


# ═════════════════════════════════════════════════════════════════════════════
# 1. Unit tests: get_s57_agent_narrative()
# ═════════════════════════════════════════════════════════════════════════════

class TestS57NarrativeUnit:

    def test_narrative_returns_dict(self):
        assert isinstance(get_s57_agent_narrative(), dict)

    def test_narrative_has_timestamp(self):
        d = get_s57_agent_narrative()
        assert "timestamp" in d
        assert isinstance(d["timestamp"], str)
        assert len(d["timestamp"]) > 0

    def test_narrative_has_symbol(self):
        d = get_s57_agent_narrative()
        assert "symbol" in d
        assert isinstance(d["symbol"], str)
        assert len(d["symbol"]) > 0

    def test_narrative_has_final_action(self):
        d = get_s57_agent_narrative()
        assert "final_action" in d

    def test_narrative_final_action_valid(self):
        d = get_s57_agent_narrative()
        assert d["final_action"] in ("BUY", "SELL", "HOLD")

    def test_narrative_has_consensus_threshold(self):
        d = get_s57_agent_narrative()
        assert "consensus_threshold" in d
        assert isinstance(d["consensus_threshold"], float)

    def test_narrative_consensus_threshold_range(self):
        d = get_s57_agent_narrative()
        assert 0.0 < d["consensus_threshold"] < 1.0

    def test_narrative_has_decision_steps(self):
        d = get_s57_agent_narrative()
        assert "decision_steps" in d
        assert isinstance(d["decision_steps"], list)

    def test_narrative_decision_steps_count(self):
        d = get_s57_agent_narrative()
        assert len(d["decision_steps"]) == 3

    def test_narrative_has_weighted_vote(self):
        d = get_s57_agent_narrative()
        assert "weighted_vote" in d
        assert isinstance(d["weighted_vote"], float)

    def test_narrative_weighted_vote_range(self):
        d = get_s57_agent_narrative()
        assert 0.0 <= d["weighted_vote"] <= 1.0

    def test_narrative_has_consensus_reached(self):
        d = get_s57_agent_narrative()
        assert "consensus_reached" in d
        assert isinstance(d["consensus_reached"], bool)

    def test_narrative_has_narrative_summary(self):
        d = get_s57_agent_narrative()
        assert "narrative_summary" in d
        assert isinstance(d["narrative_summary"], str)
        assert len(d["narrative_summary"]) > 0

    def test_narrative_version_s57(self):
        d = get_s57_agent_narrative()
        assert d.get("version") == "S57"


# ─── Decision step field tests ────────────────────────────────────────────────

class TestS57NarrativeStepFields:

    @pytest.fixture(autouse=True)
    def setup(self):
        self.steps = get_s57_agent_narrative()["decision_steps"]

    def test_step1_has_step_number(self):
        assert "step" in self.steps[0]
        assert self.steps[0]["step"] == 1

    def test_step2_has_step_number(self):
        assert self.steps[1]["step"] == 2

    def test_step3_has_step_number(self):
        assert self.steps[2]["step"] == 3

    def test_all_steps_have_agent(self):
        for s in self.steps:
            assert "agent" in s
            assert isinstance(s["agent"], str)
            assert len(s["agent"]) > 0

    def test_all_steps_have_signal(self):
        for s in self.steps:
            assert "signal" in s

    def test_all_steps_signal_valid(self):
        for s in self.steps:
            assert s["signal"] in ("BUY", "SELL", "HOLD")

    def test_all_steps_have_confidence(self):
        for s in self.steps:
            assert "confidence" in s
            assert isinstance(s["confidence"], (int, float))

    def test_all_steps_confidence_range(self):
        for s in self.steps:
            assert 0 <= s["confidence"] <= 100

    def test_all_steps_have_reasoning(self):
        for s in self.steps:
            assert "reasoning" in s
            assert isinstance(s["reasoning"], str)
            assert len(s["reasoning"]) > 0

    def test_all_steps_have_reputation_weight(self):
        for s in self.steps:
            assert "reputation_weight" in s
            assert isinstance(s["reputation_weight"], float)

    def test_all_steps_reputation_weight_range(self):
        for s in self.steps:
            assert 0.0 < s["reputation_weight"] < 1.0

    def test_reputation_weights_sum_approx_one(self):
        total = sum(s["reputation_weight"] for s in self.steps)
        assert abs(total - 1.0) < 0.01

    def test_step1_agent_conservative(self):
        assert "Conservative" in self.steps[0]["agent"] or "RSI" in self.steps[0]["agent"]

    def test_step2_agent_balanced(self):
        assert "Balanced" in self.steps[1]["agent"] or "MACD" in self.steps[1]["agent"]

    def test_step3_agent_aggressive(self):
        assert "Aggressive" in self.steps[2]["agent"] or "Combined" in self.steps[2]["agent"]


# ═════════════════════════════════════════════════════════════════════════════
# 2. Unit tests: get_s57_strategy_compare()
# ═════════════════════════════════════════════════════════════════════════════

class TestS57StrategyCompareUnit:

    def test_compare_returns_dict(self):
        assert isinstance(get_s57_strategy_compare(), dict)

    def test_compare_has_period_days(self):
        d = get_s57_strategy_compare()
        assert "period_days" in d
        assert d["period_days"] == 30

    def test_compare_has_strategies(self):
        d = get_s57_strategy_compare()
        assert "strategies" in d
        assert isinstance(d["strategies"], list)

    def test_compare_strategies_count(self):
        d = get_s57_strategy_compare()
        assert len(d["strategies"]) == 3

    def test_compare_has_winner(self):
        d = get_s57_strategy_compare()
        assert "winner" in d
        assert isinstance(d["winner"], str)
        assert len(d["winner"]) > 0

    def test_compare_has_risk_adjusted_winner(self):
        d = get_s57_strategy_compare()
        assert "risk_adjusted_winner" in d
        assert isinstance(d["risk_adjusted_winner"], str)
        assert len(d["risk_adjusted_winner"]) > 0

    def test_compare_has_insight(self):
        d = get_s57_strategy_compare()
        assert "insight" in d
        assert isinstance(d["insight"], str)
        assert len(d["insight"]) > 0

    def test_compare_version_s57(self):
        d = get_s57_strategy_compare()
        assert d.get("version") == "S57"

    def test_compare_has_generated_at(self):
        d = get_s57_strategy_compare()
        assert "generated_at" in d


# ─── Strategy field tests ─────────────────────────────────────────────────────

class TestS57StrategyFields:

    @pytest.fixture(autouse=True)
    def setup(self):
        self.strategies = get_s57_strategy_compare()["strategies"]

    def test_all_strategies_have_name(self):
        for s in self.strategies:
            assert "name" in s
            assert isinstance(s["name"], str)
            assert len(s["name"]) > 0

    def test_all_strategies_have_agent_id(self):
        for s in self.strategies:
            assert "agent_id" in s
            assert isinstance(s["agent_id"], str)

    def test_all_strategies_have_erc8004_token_id(self):
        for s in self.strategies:
            assert "erc8004_token_id" in s
            assert isinstance(s["erc8004_token_id"], int)

    def test_all_strategies_have_initial_capital(self):
        for s in self.strategies:
            assert "initial_capital" in s
            assert s["initial_capital"] > 0

    def test_all_strategies_have_final_value(self):
        for s in self.strategies:
            assert "final_value" in s
            assert s["final_value"] > 0

    def test_all_strategies_have_total_return_pct(self):
        for s in self.strategies:
            assert "total_return_pct" in s
            assert isinstance(s["total_return_pct"], float)

    def test_all_strategies_have_max_drawdown_pct(self):
        for s in self.strategies:
            assert "max_drawdown_pct" in s
            assert isinstance(s["max_drawdown_pct"], float)

    def test_max_drawdown_is_negative(self):
        for s in self.strategies:
            assert s["max_drawdown_pct"] < 0

    def test_all_strategies_have_sharpe_ratio(self):
        for s in self.strategies:
            assert "sharpe_ratio" in s
            assert isinstance(s["sharpe_ratio"], float)

    def test_sharpe_ratio_positive(self):
        for s in self.strategies:
            assert s["sharpe_ratio"] > 0

    def test_all_strategies_have_trades(self):
        for s in self.strategies:
            assert "trades" in s
            assert isinstance(s["trades"], int)
            assert s["trades"] > 0

    def test_all_strategies_have_win_rate(self):
        for s in self.strategies:
            assert "win_rate" in s
            assert isinstance(s["win_rate"], float)

    def test_win_rate_range(self):
        for s in self.strategies:
            assert 0.0 <= s["win_rate"] <= 1.0

    def test_all_strategies_have_risk_level(self):
        for s in self.strategies:
            assert "risk_level" in s
            assert s["risk_level"] in ("LOW", "MEDIUM", "HIGH")

    def test_all_strategies_have_best_trade(self):
        for s in self.strategies:
            assert "best_trade" in s
            assert isinstance(s["best_trade"], dict)

    def test_all_strategies_have_worst_trade(self):
        for s in self.strategies:
            assert "worst_trade" in s
            assert isinstance(s["worst_trade"], dict)

    def test_best_trade_has_pnl(self):
        for s in self.strategies:
            assert "pnl" in s["best_trade"]
            assert s["best_trade"]["pnl"] > 0

    def test_worst_trade_has_pnl(self):
        for s in self.strategies:
            assert "pnl" in s["worst_trade"]
            assert s["worst_trade"]["pnl"] < 0

    def test_conservative_low_risk(self):
        conservative = next(s for s in self.strategies if "Conservative" in s["name"])
        assert conservative["risk_level"] == "LOW"

    def test_aggressive_high_risk(self):
        aggressive = next(s for s in self.strategies if "Aggressive" in s["name"])
        assert aggressive["risk_level"] == "HIGH"

    def test_aggressive_highest_return(self):
        returns = sorted(s["total_return_pct"] for s in self.strategies)
        aggressive = next(s for s in self.strategies if "Aggressive" in s["name"])
        assert aggressive["total_return_pct"] == returns[-1]

    def test_conservative_best_sharpe(self):
        sharpes = sorted(s["sharpe_ratio"] for s in self.strategies)
        conservative = next(s for s in self.strategies if "Conservative" in s["name"])
        assert conservative["sharpe_ratio"] == sharpes[-1]


# ═════════════════════════════════════════════════════════════════════════════
# 3. HTTP endpoint tests
# ═════════════════════════════════════════════════════════════════════════════

class TestS57NarrativeHTTP:

    def test_narrative_returns_200(self, server):
        r = _get("/api/v1/agent/narrative", server)
        assert r.status_code == 200

    def test_narrative_content_type_json(self, server):
        r = _get("/api/v1/agent/narrative", server)
        assert "application/json" in r.headers.get("Content-Type", "")

    def test_narrative_http_has_decision_steps(self, server):
        d = _get("/api/v1/agent/narrative", server).json()
        assert "decision_steps" in d
        assert isinstance(d["decision_steps"], list)

    def test_narrative_http_steps_count(self, server):
        d = _get("/api/v1/agent/narrative", server).json()
        assert len(d["decision_steps"]) == 3

    def test_narrative_http_has_consensus_reached(self, server):
        d = _get("/api/v1/agent/narrative", server).json()
        assert "consensus_reached" in d
        assert isinstance(d["consensus_reached"], bool)

    def test_narrative_http_has_weighted_vote(self, server):
        d = _get("/api/v1/agent/narrative", server).json()
        assert "weighted_vote" in d
        assert isinstance(d["weighted_vote"], float)

    def test_narrative_http_has_timestamp(self, server):
        d = _get("/api/v1/agent/narrative", server).json()
        assert "timestamp" in d
        assert len(d["timestamp"]) > 0

    def test_narrative_http_has_narrative_summary(self, server):
        d = _get("/api/v1/agent/narrative", server).json()
        assert "narrative_summary" in d
        assert len(d["narrative_summary"]) > 0

    def test_narrative_http_final_action_valid(self, server):
        d = _get("/api/v1/agent/narrative", server).json()
        assert d["final_action"] in ("BUY", "SELL", "HOLD")

    def test_narrative_http_x_erc8004_header(self, server):
        r = _get("/api/v1/agent/narrative", server)
        assert "X-ERC8004-Version" in r.headers


class TestS57StrategyCompareHTTP:

    def test_compare_returns_200(self, server):
        r = _get("/api/v1/strategies/compare", server)
        assert r.status_code == 200

    def test_compare_content_type_json(self, server):
        r = _get("/api/v1/strategies/compare", server)
        assert "application/json" in r.headers.get("Content-Type", "")

    def test_compare_http_has_strategies(self, server):
        d = _get("/api/v1/strategies/compare", server).json()
        assert "strategies" in d
        assert isinstance(d["strategies"], list)

    def test_compare_http_strategies_count(self, server):
        d = _get("/api/v1/strategies/compare", server).json()
        assert len(d["strategies"]) == 3

    def test_compare_http_has_winner(self, server):
        d = _get("/api/v1/strategies/compare", server).json()
        assert "winner" in d
        assert len(d["winner"]) > 0

    def test_compare_http_has_risk_adjusted_winner(self, server):
        d = _get("/api/v1/strategies/compare", server).json()
        assert "risk_adjusted_winner" in d
        assert len(d["risk_adjusted_winner"]) > 0

    def test_compare_http_has_insight(self, server):
        d = _get("/api/v1/strategies/compare", server).json()
        assert "insight" in d
        assert len(d["insight"]) > 0

    def test_compare_http_strategies_have_return_pct(self, server):
        d = _get("/api/v1/strategies/compare", server).json()
        for s in d["strategies"]:
            assert "total_return_pct" in s

    def test_compare_http_strategies_have_sharpe(self, server):
        d = _get("/api/v1/strategies/compare", server).json()
        for s in d["strategies"]:
            assert "sharpe_ratio" in s

    def test_compare_http_strategies_have_drawdown(self, server):
        d = _get("/api/v1/strategies/compare", server).json()
        for s in d["strategies"]:
            assert "max_drawdown_pct" in s

    def test_compare_http_x_erc8004_header(self, server):
        r = _get("/api/v1/strategies/compare", server)
        assert "X-ERC8004-Version" in r.headers

    def test_compare_http_period_days(self, server):
        d = _get("/api/v1/strategies/compare", server).json()
        assert d.get("period_days") == 30


# ═════════════════════════════════════════════════════════════════════════════
# 4. Additional edge-case tests
# ═════════════════════════════════════════════════════════════════════════════

class TestS57AdditionalCoverage:

    def test_narrative_consensus_threshold_is_667(self):
        d = get_s57_agent_narrative()
        assert abs(d["consensus_threshold"] - 0.667) < 0.01

    def test_narrative_decision_steps_are_ordered(self):
        steps = get_s57_agent_narrative()["decision_steps"]
        for i, s in enumerate(steps, 1):
            assert s["step"] == i

    def test_compare_winner_is_aggressive(self):
        d = get_s57_strategy_compare()
        assert "Aggressive" in d["winner"]

    def test_compare_risk_adjusted_winner_is_conservative(self):
        d = get_s57_strategy_compare()
        assert "Conservative" in d["risk_adjusted_winner"]

    def test_narrative_http_version_is_s57(self, server):
        d = _get("/api/v1/agent/narrative", server).json()
        assert d.get("version") == "S57"

    def test_compare_http_version_is_s57(self, server):
        d = _get("/api/v1/strategies/compare", server).json()
        assert d.get("version") == "S57"
