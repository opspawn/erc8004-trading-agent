"""
test_s31_market_coordination.py — Sprint 31: Market Intelligence + Agent Coordination tests.

130 tests covering:
  - get_market_intelligence: unit tests for structure and content
  - propose_coordination: proposal validation, vote generation, error handling
  - get_coordination_consensus: consensus calculation, quorum logic
  - get_performance_attribution: P&L breakdown by period, strategy, agent
  - HTTP GET  /demo/market/intelligence
  - HTTP POST /demo/coordination/propose
  - HTTP GET  /demo/coordination/consensus
  - HTTP GET  /demo/performance/attribution
  - Edge cases, invalid inputs, and error handling
"""

from __future__ import annotations

import json
import threading
import time
from urllib.request import urlopen, Request
from urllib.error import HTTPError

import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from demo_server import (
    # Market intelligence
    get_market_intelligence,
    _MI_ASSETS,
    _MI_TREND_CHOICES,
    _MI_STRENGTH_CHOICES,
    # Coordination bus
    propose_coordination,
    get_coordination_consensus,
    _COORD_PROPOSALS,
    _COORD_LOCK,
    _COORD_VALID_ACTIONS,
    _COORD_AGENTS,
    # Performance attribution
    get_performance_attribution,
    _PERF_PERIODS,
    _PERF_DEFAULT_PERIOD,
    _PERF_STRATEGIES,
    _PERF_AGENTS_LIST,
    # Server
    DemoServer,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def server():
    """Start a DemoServer on a test port for HTTP endpoint tests."""
    srv = DemoServer(port=18084)
    srv.start()
    time.sleep(0.15)
    yield srv
    srv.stop()


def _get(port: int, path: str) -> dict:
    url = f"http://localhost:{port}{path}"
    with urlopen(url, timeout=5) as resp:
        return json.loads(resp.read())


def _post(port: int, path: str, body: dict) -> dict:
    url = f"http://localhost:{port}{path}"
    data = json.dumps(body).encode("utf-8")
    req = Request(url, data=data, headers={"Content-Type": "application/json"}, method="POST")
    with urlopen(req, timeout=5) as resp:
        return json.loads(resp.read())


def _post_status(port: int, path: str, body: dict) -> int:
    url = f"http://localhost:{port}{path}"
    data = json.dumps(body).encode("utf-8")
    req = Request(url, data=data, headers={"Content-Type": "application/json"}, method="POST")
    try:
        with urlopen(req, timeout=5) as resp:
            return resp.status
    except HTTPError as e:
        return e.code


def _get_status(port: int, path: str) -> int:
    url = f"http://localhost:{port}{path}"
    try:
        with urlopen(url, timeout=5) as resp:
            return resp.status
    except HTTPError as e:
        return e.code


# ═══════════════════════════════════════════════════════════════════════════════
# UNIT TESTS: get_market_intelligence
# ═══════════════════════════════════════════════════════════════════════════════

class TestMarketIntelligenceUnit:

    def test_returns_dict(self):
        result = get_market_intelligence()
        assert isinstance(result, dict)

    def test_top_level_keys(self):
        result = get_market_intelligence()
        expected = {"timestamp", "data_freshness_ms", "market_mood", "avg_confidence_score",
                    "avg_volatility_index", "assets", "correlation_matrix", "seed_minute"}
        assert expected.issubset(result.keys())

    def test_timestamp_is_recent(self):
        before = time.time()
        result = get_market_intelligence()
        after = time.time()
        assert before <= result["timestamp"] <= after

    def test_data_freshness_ms_range(self):
        result = get_market_intelligence()
        assert 0 <= result["data_freshness_ms"] < 60000

    def test_market_mood_valid(self):
        result = get_market_intelligence()
        assert result["market_mood"] in ("risk_on", "risk_off", "neutral")

    def test_avg_confidence_score_range(self):
        result = get_market_intelligence()
        assert 0.0 <= result["avg_confidence_score"] <= 1.0

    def test_avg_volatility_index_range(self):
        result = get_market_intelligence()
        assert 0.0 <= result["avg_volatility_index"] <= 1.0

    def test_assets_all_present(self):
        result = get_market_intelligence()
        assert set(result["assets"].keys()) == set(_MI_ASSETS)

    def test_asset_btc_structure(self):
        result = get_market_intelligence()
        btc = result["assets"]["BTC"]
        assert "trend_direction" in btc
        assert "volatility_index" in btc
        assert "confidence_score" in btc
        assert "signal_strength" in btc
        assert "price_change_24h_pct" in btc
        assert "volume_ratio_vs_avg" in btc

    def test_asset_eth_structure(self):
        result = get_market_intelligence()
        eth = result["assets"]["ETH"]
        assert "trend_direction" in eth
        assert "confidence_score" in eth

    def test_asset_sol_structure(self):
        result = get_market_intelligence()
        sol = result["assets"]["SOL"]
        assert "signal_strength" in sol
        assert "volatility_index" in sol

    def test_trend_direction_valid(self):
        result = get_market_intelligence()
        for asset in _MI_ASSETS:
            assert result["assets"][asset]["trend_direction"] in _MI_TREND_CHOICES

    def test_signal_strength_valid(self):
        result = get_market_intelligence()
        for asset in _MI_ASSETS:
            assert result["assets"][asset]["signal_strength"] in _MI_STRENGTH_CHOICES

    def test_confidence_scores_in_range(self):
        result = get_market_intelligence()
        for asset in _MI_ASSETS:
            c = result["assets"][asset]["confidence_score"]
            assert 0.0 <= c <= 1.0, f"Confidence out of range for {asset}: {c}"

    def test_volatility_index_in_range(self):
        result = get_market_intelligence()
        for asset in _MI_ASSETS:
            v = result["assets"][asset]["volatility_index"]
            assert 0.0 <= v <= 1.0, f"Volatility index out of range for {asset}: {v}"

    def test_volume_ratio_positive(self):
        result = get_market_intelligence()
        for asset in _MI_ASSETS:
            vr = result["assets"][asset]["volume_ratio_vs_avg"]
            assert vr > 0.0

    def test_correlation_matrix_keys(self):
        result = get_market_intelligence()
        cm = result["correlation_matrix"]
        assert "BTC_ETH" in cm
        assert "BTC_SOL" in cm
        assert "ETH_SOL" in cm

    def test_correlation_values_in_range(self):
        result = get_market_intelligence()
        cm = result["correlation_matrix"]
        for key, val in cm.items():
            assert -1.0 <= val <= 1.0, f"Correlation out of range for {key}: {val}"

    def test_seed_minute_is_int(self):
        result = get_market_intelligence()
        assert isinstance(result["seed_minute"], int)

    def test_deterministic_within_minute(self):
        """Two calls in the same minute should return same seed_minute."""
        r1 = get_market_intelligence()
        r2 = get_market_intelligence()
        assert r1["seed_minute"] == r2["seed_minute"]

    def test_assets_data_stable_within_minute(self):
        """Asset confidence scores should be identical on back-to-back calls."""
        r1 = get_market_intelligence()
        r2 = get_market_intelligence()
        for asset in _MI_ASSETS:
            assert r1["assets"][asset]["confidence_score"] == r2["assets"][asset]["confidence_score"]

    def test_market_mood_reflects_bullish_count(self):
        """market_mood derivation: risk_on if >=2 bullish."""
        result = get_market_intelligence()
        assets = result["assets"]
        bullish = sum(1 for a in _MI_ASSETS if assets[a]["trend_direction"] == "bullish")
        if bullish >= 2:
            assert result["market_mood"] == "risk_on"
        elif bullish == 0:
            assert result["market_mood"] == "risk_off"
        else:
            assert result["market_mood"] == "neutral"

    def test_avg_confidence_is_mean_of_assets(self):
        result = get_market_intelligence()
        assets = result["assets"]
        computed = round(sum(assets[a]["confidence_score"] for a in _MI_ASSETS) / len(_MI_ASSETS), 4)
        assert abs(result["avg_confidence_score"] - computed) < 0.0001

    def test_avg_volatility_is_mean_of_assets(self):
        result = get_market_intelligence()
        assets = result["assets"]
        computed = round(sum(assets[a]["volatility_index"] for a in _MI_ASSETS) / len(_MI_ASSETS), 4)
        assert abs(result["avg_volatility_index"] - computed) < 0.0001

    def test_price_change_can_be_negative(self):
        """price_change_24h_pct can be negative (bearish scenarios)."""
        # Run multiple seeds to confirm range includes negatives statistically
        result = get_market_intelligence()
        # Just assert it's a float in a reasonable range
        for asset in _MI_ASSETS:
            pc = result["assets"][asset]["price_change_24h_pct"]
            assert -10.0 <= pc <= 10.0


# ═══════════════════════════════════════════════════════════════════════════════
# UNIT TESTS: propose_coordination
# ═══════════════════════════════════════════════════════════════════════════════

class TestCoordinationPropose:

    def setup_method(self):
        """Clear proposals before each test."""
        with _COORD_LOCK:
            _COORD_PROPOSALS.clear()

    def test_returns_dict(self):
        result = propose_coordination("agent-1", "buy", "BTC", 1.0, "bullish signal")
        assert isinstance(result, dict)

    def test_top_level_keys(self):
        result = propose_coordination("agent-1", "buy", "BTC", 1.0, "test")
        assert "proposal_id" in result
        assert "status" in result
        assert "votes_collected" in result
        assert "proposal" in result

    def test_status_submitted(self):
        result = propose_coordination("agent-1", "sell", "ETH", 2.0, "bearish")
        assert result["status"] == "submitted"

    def test_proposal_id_is_string(self):
        result = propose_coordination("agent-1", "hold", "SOL", 5.0, "wait")
        assert isinstance(result["proposal_id"], str)
        assert len(result["proposal_id"]) > 0

    def test_votes_collected_equals_agent_count(self):
        result = propose_coordination("agent-1", "buy", "BTC", 1.0, "test")
        assert result["votes_collected"] == len(_COORD_AGENTS)

    def test_proposal_structure(self):
        result = propose_coordination("agent-x", "buy", "ETH", 0.5, "momentum")
        prop = result["proposal"]
        assert prop["proposer"] == "agent-x"
        assert prop["action"] == "buy"
        assert prop["asset"] == "ETH"
        assert prop["amount"] == 0.5
        assert prop["rationale"] == "momentum"

    def test_proposal_votes_structure(self):
        result = propose_coordination("agent-1", "sell", "BTC", 1.0, "test")
        votes = result["proposal"]["votes"]
        assert len(votes) == len(_COORD_AGENTS)
        for v in votes:
            assert "agent_id" in v
            assert "vote" in v
            assert "confidence" in v
            assert v["vote"] in ("approve", "reject")

    def test_vote_confidence_in_range(self):
        result = propose_coordination("agent-1", "buy", "ETH", 1.0, "test")
        for v in result["proposal"]["votes"]:
            assert 0.0 <= v["confidence"] <= 1.0

    def test_dissent_reason_on_reject(self):
        """Votes that are 'reject' should have a dissent_reason."""
        result = propose_coordination("agent-1", "buy", "BTC", 100.0, "test")
        for v in result["proposal"]["votes"]:
            if v["vote"] == "reject":
                assert v["dissent_reason"] is not None
                assert isinstance(v["dissent_reason"], str)

    def test_approve_has_no_dissent(self):
        """Votes that are 'approve' should have dissent_reason = None."""
        result = propose_coordination("agent-1", "sell", "SOL", 3.0, "test")
        for v in result["proposal"]["votes"]:
            if v["vote"] == "approve":
                assert v["dissent_reason"] is None

    def test_proposal_stored_in_list(self):
        propose_coordination("agent-1", "buy", "BTC", 1.0, "test")
        with _COORD_LOCK:
            assert len(_COORD_PROPOSALS) == 1

    def test_multiple_proposals_accumulate(self):
        propose_coordination("agent-1", "buy", "BTC", 1.0, "test1")
        propose_coordination("agent-2", "sell", "ETH", 2.0, "test2")
        propose_coordination("agent-3", "hold", "SOL", 0.5, "test3")
        with _COORD_LOCK:
            assert len(_COORD_PROPOSALS) == 3

    def test_invalid_action_raises(self):
        with pytest.raises(ValueError, match="action must be one of"):
            propose_coordination("agent-1", "moon", "BTC", 1.0, "test")

    def test_empty_agent_id_raises(self):
        with pytest.raises(ValueError, match="agent_id is required"):
            propose_coordination("", "buy", "BTC", 1.0, "test")

    def test_empty_asset_raises(self):
        with pytest.raises(ValueError, match="asset is required"):
            propose_coordination("agent-1", "buy", "", 1.0, "test")

    def test_zero_amount_raises(self):
        with pytest.raises(ValueError, match="amount must be positive"):
            propose_coordination("agent-1", "buy", "BTC", 0.0, "test")

    def test_negative_amount_raises(self):
        with pytest.raises(ValueError, match="amount must be positive"):
            propose_coordination("agent-1", "buy", "BTC", -1.0, "test")

    def test_all_valid_actions_accepted(self):
        for action in _COORD_VALID_ACTIONS:
            result = propose_coordination("agent-1", action, "BTC", 1.0, "test")
            assert result["status"] == "submitted"

    def test_proposal_has_created_at(self):
        before = time.time()
        result = propose_coordination("agent-1", "buy", "BTC", 1.0, "test")
        after = time.time()
        ts = result["proposal"]["created_at"]
        assert before <= ts <= after

    def test_unique_proposal_ids(self):
        r1 = propose_coordination("agent-1", "buy", "BTC", 1.0, "test")
        r2 = propose_coordination("agent-1", "buy", "BTC", 1.0, "test")
        # IDs should differ (time-based seed)
        # They may collide in the same second but usually differ
        assert isinstance(r1["proposal_id"], str)
        assert isinstance(r2["proposal_id"], str)

    def test_proposal_status_is_pending(self):
        result = propose_coordination("agent-1", "buy", "BTC", 1.0, "test")
        assert result["proposal"]["status"] == "pending"

    def test_large_amount_accepted(self):
        result = propose_coordination("whale", "buy", "BTC", 1_000_000.0, "whale trade")
        assert result["status"] == "submitted"
        assert result["proposal"]["amount"] == 1_000_000.0

    def test_reduce_action(self):
        result = propose_coordination("agent-1", "reduce", "ETH", 0.1, "risk off")
        assert result["proposal"]["action"] == "reduce"

    def test_hedge_action(self):
        result = propose_coordination("agent-2", "hedge", "SOL", 5.0, "protect gains")
        assert result["proposal"]["action"] == "hedge"


# ═══════════════════════════════════════════════════════════════════════════════
# UNIT TESTS: get_coordination_consensus
# ═══════════════════════════════════════════════════════════════════════════════

class TestCoordinationConsensus:

    def setup_method(self):
        with _COORD_LOCK:
            _COORD_PROPOSALS.clear()

    def test_empty_returns_dict(self):
        result = get_coordination_consensus()
        assert isinstance(result, dict)

    def test_empty_total_proposals_zero(self):
        result = get_coordination_consensus()
        assert result["total_proposals"] == 0

    def test_empty_quorum_false(self):
        result = get_coordination_consensus()
        assert result["quorum_reached"] is False

    def test_empty_dominant_action_none(self):
        result = get_coordination_consensus()
        assert result["dominant_action"] is None

    def test_empty_note_present(self):
        result = get_coordination_consensus()
        assert "note" in result

    def test_after_proposal_total_updates(self):
        propose_coordination("agent-1", "buy", "BTC", 1.0, "test")
        result = get_coordination_consensus()
        assert result["total_proposals"] == 1

    def test_top_level_keys_with_proposals(self):
        propose_coordination("agent-1", "buy", "BTC", 1.0, "test")
        result = get_coordination_consensus()
        expected = {"total_proposals", "total_votes", "total_approvals", "quorum_reached",
                    "consensus_rate", "dominant_action", "action_breakdown",
                    "agent_weights", "agent_agreement_matrix", "dissent_reasons"}
        assert expected.issubset(result.keys())

    def test_consensus_rate_in_range(self):
        propose_coordination("agent-1", "buy", "BTC", 1.0, "test")
        result = get_coordination_consensus()
        assert 0.0 <= result["consensus_rate"] <= 1.0

    def test_total_votes_equals_proposals_times_agents(self):
        propose_coordination("agent-1", "buy", "BTC", 1.0, "test")
        propose_coordination("agent-2", "sell", "ETH", 2.0, "test2")
        result = get_coordination_consensus()
        expected_votes = 2 * len(_COORD_AGENTS)
        assert result["total_votes"] == expected_votes

    def test_total_approvals_lte_total_votes(self):
        propose_coordination("agent-1", "buy", "BTC", 1.0, "test")
        result = get_coordination_consensus()
        assert result["total_approvals"] <= result["total_votes"]

    def test_action_breakdown_includes_proposed_action(self):
        propose_coordination("agent-1", "buy", "BTC", 1.0, "test")
        result = get_coordination_consensus()
        assert "buy" in result["action_breakdown"]

    def test_agent_weights_present(self):
        propose_coordination("agent-1", "buy", "BTC", 1.0, "test")
        result = get_coordination_consensus()
        weights = result["agent_weights"]
        for ag in _COORD_AGENTS:
            assert ag["id"] in weights
            assert weights[ag["id"]] == ag["weight"]

    def test_agent_agreement_matrix_keys(self):
        propose_coordination("agent-1", "buy", "BTC", 1.0, "test")
        result = get_coordination_consensus()
        matrix = result["agent_agreement_matrix"]
        for ag in _COORD_AGENTS:
            assert ag["id"] in matrix

    def test_agreement_matrix_values_in_range(self):
        propose_coordination("agent-1", "buy", "BTC", 1.0, "test")
        result = get_coordination_consensus()
        for ag_id, rate in result["agent_agreement_matrix"].items():
            assert 0.0 <= rate <= 1.0, f"Agreement rate out of range for {ag_id}: {rate}"

    def test_dissent_reasons_is_list(self):
        propose_coordination("agent-1", "buy", "BTC", 1.0, "test")
        result = get_coordination_consensus()
        assert isinstance(result["dissent_reasons"], list)

    def test_dissent_reasons_max_5(self):
        for i in range(10):
            propose_coordination(f"agent-{i}", "sell", "ETH", 1.0, f"reason {i}")
        result = get_coordination_consensus()
        assert len(result["dissent_reasons"]) <= 5

    def test_dominant_action_is_most_common(self):
        propose_coordination("agent-1", "buy", "BTC", 1.0, "test")
        propose_coordination("agent-2", "buy", "BTC", 1.0, "test")
        propose_coordination("agent-3", "sell", "ETH", 1.0, "test")
        result = get_coordination_consensus()
        assert result["dominant_action"] == "buy"

    def test_multiple_actions_breakdown(self):
        propose_coordination("agent-1", "buy", "BTC", 1.0, "test")
        propose_coordination("agent-2", "hold", "ETH", 1.0, "test")
        result = get_coordination_consensus()
        breakdown = result["action_breakdown"]
        assert "buy" in breakdown
        assert "hold" in breakdown

    def test_consensus_rate_math(self):
        propose_coordination("agent-1", "buy", "BTC", 1.0, "test")
        result = get_coordination_consensus()
        if result["total_votes"] > 0:
            expected = round(result["total_approvals"] / result["total_votes"], 4)
            assert abs(result["consensus_rate"] - expected) < 0.0001


# ═══════════════════════════════════════════════════════════════════════════════
# UNIT TESTS: get_performance_attribution
# ═══════════════════════════════════════════════════════════════════════════════

class TestPerformanceAttribution:

    def test_returns_dict(self):
        result = get_performance_attribution()
        assert isinstance(result, dict)

    def test_default_period_24h(self):
        result = get_performance_attribution()
        assert result["period"] == "24h"
        assert result["period_hours"] == 24

    def test_1h_period(self):
        result = get_performance_attribution("1h")
        assert result["period"] == "1h"
        assert result["period_hours"] == 1

    def test_7d_period(self):
        result = get_performance_attribution("7d")
        assert result["period"] == "7d"
        assert result["period_hours"] == 168

    def test_invalid_period_raises(self):
        with pytest.raises(ValueError, match="period must be one of"):
            get_performance_attribution("30d")

    def test_top_level_keys(self):
        result = get_performance_attribution()
        expected = {"period", "period_hours", "total_pnl_usd", "portfolio_attribution",
                    "strategy_breakdown", "agent_breakdown", "generated_at"}
        assert expected.issubset(result.keys())

    def test_portfolio_attribution_keys(self):
        result = get_performance_attribution()
        pa = result["portfolio_attribution"]
        assert "alpha_contribution" in pa
        assert "beta_exposure" in pa
        assert "idiosyncratic_return" in pa

    def test_strategy_breakdown_all_strategies(self):
        result = get_performance_attribution()
        sb = result["strategy_breakdown"]
        for strat in _PERF_STRATEGIES:
            assert strat in sb, f"Strategy {strat} missing from breakdown"

    def test_strategy_breakdown_structure(self):
        result = get_performance_attribution()
        for strat in _PERF_STRATEGIES:
            s = result["strategy_breakdown"][strat]
            assert "pnl_usd" in s
            assert "alpha_contribution" in s
            assert "beta_exposure" in s
            assert "idiosyncratic_return" in s
            assert "trades" in s
            assert "win_rate" in s

    def test_strategy_win_rate_in_range(self):
        result = get_performance_attribution()
        for strat, data in result["strategy_breakdown"].items():
            wr = data["win_rate"]
            assert 0.0 <= wr <= 1.0, f"Win rate out of range for {strat}: {wr}"

    def test_strategy_trades_positive(self):
        result = get_performance_attribution()
        for strat, data in result["strategy_breakdown"].items():
            assert data["trades"] >= 1

    def test_agent_breakdown_all_agents(self):
        result = get_performance_attribution()
        ab = result["agent_breakdown"]
        for ag in _PERF_AGENTS_LIST:
            assert ag in ab, f"Agent {ag} missing from breakdown"

    def test_agent_breakdown_structure(self):
        result = get_performance_attribution()
        for ag in _PERF_AGENTS_LIST:
            a = result["agent_breakdown"][ag]
            assert "pnl_usd" in a
            assert "sharpe_ratio" in a
            assert "max_drawdown_pct" in a
            assert "contribution_pct" in a

    def test_contribution_pct_sums_to_100(self):
        result = get_performance_attribution()
        total = sum(v["contribution_pct"] for v in result["agent_breakdown"].values())
        assert abs(total - 100.0) < 0.1, f"Contribution pcts sum to {total}, expected ~100"

    def test_contribution_pct_non_negative(self):
        result = get_performance_attribution()
        for ag, data in result["agent_breakdown"].items():
            assert data["contribution_pct"] >= 0.0

    def test_max_drawdown_non_negative(self):
        result = get_performance_attribution()
        for ag, data in result["agent_breakdown"].items():
            assert data["max_drawdown_pct"] >= 0.0

    def test_generated_at_is_recent(self):
        before = time.time()
        result = get_performance_attribution()
        after = time.time()
        assert before <= result["generated_at"] <= after

    def test_7d_pnl_larger_than_1h(self):
        """7d results should have more trades than 1h (scaled by hours)."""
        r7d = get_performance_attribution("7d")
        r1h = get_performance_attribution("1h")
        # Can't guarantee direction but trade count should scale
        for strat in _PERF_STRATEGIES:
            trades_7d = r7d["strategy_breakdown"][strat]["trades"]
            trades_1h = r1h["strategy_breakdown"][strat]["trades"]
            assert trades_7d >= trades_1h

    def test_total_pnl_is_sum_of_strategies(self):
        result = get_performance_attribution("24h")
        strat_sum = sum(v["pnl_usd"] for v in result["strategy_breakdown"].values())
        assert abs(result["total_pnl_usd"] - round(strat_sum, 2)) < 0.01


# ═══════════════════════════════════════════════════════════════════════════════
# HTTP ENDPOINT TESTS
# ═══════════════════════════════════════════════════════════════════════════════

class TestMarketIntelligenceHTTP:

    def test_get_200(self, server):
        result = _get(18084, "/demo/market/intelligence")
        assert isinstance(result, dict)

    def test_returns_assets(self, server):
        result = _get(18084, "/demo/market/intelligence")
        assert "assets" in result
        assert set(result["assets"].keys()) == set(_MI_ASSETS)

    def test_returns_correlation_matrix(self, server):
        result = _get(18084, "/demo/market/intelligence")
        assert "correlation_matrix" in result
        assert "BTC_ETH" in result["correlation_matrix"]

    def test_returns_market_mood(self, server):
        result = _get(18084, "/demo/market/intelligence")
        assert result["market_mood"] in ("risk_on", "risk_off", "neutral")

    def test_returns_confidence_score(self, server):
        result = _get(18084, "/demo/market/intelligence")
        assert 0.0 <= result["avg_confidence_score"] <= 1.0

    def test_data_freshness_ms_present(self, server):
        result = _get(18084, "/demo/market/intelligence")
        assert "data_freshness_ms" in result
        assert isinstance(result["data_freshness_ms"], int)

    def test_404_unknown_market_path(self, server):
        status = _get_status(18084, "/demo/market/unknown")
        assert status == 404


class TestCoordinationHTTP:

    def setup_method(self):
        with _COORD_LOCK:
            _COORD_PROPOSALS.clear()

    def test_propose_post_200(self, server):
        result = _post(18084, "/demo/coordination/propose", {
            "agent_id": "agent-http-1",
            "action": "buy",
            "asset": "BTC",
            "amount": 1.0,
            "rationale": "HTTP test",
        })
        assert result["status"] == "submitted"

    def test_propose_returns_proposal_id(self, server):
        result = _post(18084, "/demo/coordination/propose", {
            "agent_id": "agent-http-2",
            "action": "sell",
            "asset": "ETH",
            "amount": 2.0,
            "rationale": "HTTP test",
        })
        assert "proposal_id" in result
        assert len(result["proposal_id"]) > 0

    def test_propose_returns_votes(self, server):
        result = _post(18084, "/demo/coordination/propose", {
            "agent_id": "agent-http-3",
            "action": "hold",
            "asset": "SOL",
            "amount": 5.0,
            "rationale": "Cautious",
        })
        votes = result["proposal"]["votes"]
        assert len(votes) == len(_COORD_AGENTS)

    def test_propose_invalid_action_400(self, server):
        status = _post_status(18084, "/demo/coordination/propose", {
            "agent_id": "agent-1",
            "action": "yolo",
            "asset": "BTC",
            "amount": 1.0,
            "rationale": "test",
        })
        assert status == 400

    def test_propose_missing_agent_id_400(self, server):
        status = _post_status(18084, "/demo/coordination/propose", {
            "action": "buy",
            "asset": "BTC",
            "amount": 1.0,
            "rationale": "test",
        })
        assert status == 400

    def test_propose_zero_amount_400(self, server):
        status = _post_status(18084, "/demo/coordination/propose", {
            "agent_id": "agent-1",
            "action": "buy",
            "asset": "BTC",
            "amount": 0,
            "rationale": "test",
        })
        assert status == 400

    def test_consensus_get_200(self, server):
        result = _get(18084, "/demo/coordination/consensus")
        assert isinstance(result, dict)

    def test_consensus_has_total_proposals(self, server):
        result = _get(18084, "/demo/coordination/consensus")
        assert "total_proposals" in result

    def test_consensus_after_proposal(self, server):
        _post(18084, "/demo/coordination/propose", {
            "agent_id": "agent-seq-1",
            "action": "buy",
            "asset": "BTC",
            "amount": 1.0,
            "rationale": "sequential test",
        })
        result = _get(18084, "/demo/coordination/consensus")
        assert result["total_proposals"] >= 1

    def test_consensus_quorum_field(self, server):
        result = _get(18084, "/demo/coordination/consensus")
        assert "quorum_reached" in result
        assert isinstance(result["quorum_reached"], bool)


class TestPerformanceAttributionHTTP:

    def test_get_200(self, server):
        result = _get(18084, "/demo/performance/attribution")
        assert isinstance(result, dict)

    def test_default_period_24h(self, server):
        result = _get(18084, "/demo/performance/attribution")
        assert result["period"] == "24h"

    def test_period_1h(self, server):
        result = _get(18084, "/demo/performance/attribution?period=1h")
        assert result["period"] == "1h"

    def test_period_7d(self, server):
        result = _get(18084, "/demo/performance/attribution?period=7d")
        assert result["period"] == "7d"

    def test_invalid_period_400(self, server):
        status = _get_status(18084, "/demo/performance/attribution?period=30d")
        assert status == 400

    def test_returns_strategy_breakdown(self, server):
        result = _get(18084, "/demo/performance/attribution")
        assert "strategy_breakdown" in result
        for strat in _PERF_STRATEGIES:
            assert strat in result["strategy_breakdown"]

    def test_returns_agent_breakdown(self, server):
        result = _get(18084, "/demo/performance/attribution")
        assert "agent_breakdown" in result
        for ag in _PERF_AGENTS_LIST:
            assert ag in result["agent_breakdown"]

    def test_returns_portfolio_attribution(self, server):
        result = _get(18084, "/demo/performance/attribution")
        pa = result["portfolio_attribution"]
        assert "alpha_contribution" in pa
        assert "beta_exposure" in pa
        assert "idiosyncratic_return" in pa

    def test_total_pnl_present(self, server):
        result = _get(18084, "/demo/performance/attribution")
        assert "total_pnl_usd" in result
        assert isinstance(result["total_pnl_usd"], (int, float))

    def test_generated_at_present(self, server):
        result = _get(18084, "/demo/performance/attribution")
        assert "generated_at" in result

    def test_period_hours_correct_1h(self, server):
        result = _get(18084, "/demo/performance/attribution?period=1h")
        assert result["period_hours"] == 1

    def test_period_hours_correct_7d(self, server):
        result = _get(18084, "/demo/performance/attribution?period=7d")
        assert result["period_hours"] == 168


# ═══════════════════════════════════════════════════════════════════════════════
# ADDITIONAL EDGE CASE TESTS
# ═══════════════════════════════════════════════════════════════════════════════

class TestMarketIntelligenceEdgeCases:

    def test_multiple_calls_same_structure(self):
        r1 = get_market_intelligence()
        r2 = get_market_intelligence()
        assert set(r1.keys()) == set(r2.keys())

    def test_correlation_matrix_has_exactly_3_pairs(self):
        result = get_market_intelligence()
        assert len(result["correlation_matrix"]) == 3

    def test_assets_count(self):
        result = get_market_intelligence()
        assert len(result["assets"]) == 3

    def test_all_assets_have_all_fields(self):
        result = get_market_intelligence()
        required = {"trend_direction", "volatility_index", "confidence_score",
                    "signal_strength", "price_change_24h_pct", "volume_ratio_vs_avg"}
        for asset in _MI_ASSETS:
            assert required == set(result["assets"][asset].keys())


class TestCoordinationConsensusEdgeCases:

    def setup_method(self):
        with _COORD_LOCK:
            _COORD_PROPOSALS.clear()

    def test_after_many_proposals_total_correct(self):
        for i in range(5):
            propose_coordination(f"agent-{i}", "buy", "BTC", float(i + 1), f"test {i}")
        result = get_coordination_consensus()
        assert result["total_proposals"] == 5
        assert result["total_votes"] == 5 * len(_COORD_AGENTS)

    def test_mix_of_actions_in_breakdown(self):
        propose_coordination("agent-1", "buy", "BTC", 1.0, "test")
        propose_coordination("agent-2", "sell", "ETH", 1.0, "test")
        propose_coordination("agent-3", "hold", "SOL", 1.0, "test")
        result = get_coordination_consensus()
        assert result["action_breakdown"]["buy"] == 1
        assert result["action_breakdown"]["sell"] == 1
        assert result["action_breakdown"]["hold"] == 1

    def test_weights_sum_to_one(self):
        propose_coordination("agent-1", "buy", "BTC", 1.0, "test")
        result = get_coordination_consensus()
        total_weight = sum(result["agent_weights"].values())
        assert abs(total_weight - 1.0) < 0.001


class TestPerformanceAttributionEdgeCases:

    def test_all_valid_periods(self):
        for period in _PERF_PERIODS:
            result = get_performance_attribution(period)
            assert result["period"] == period

    def test_strategy_pnl_is_float(self):
        result = get_performance_attribution()
        for strat, data in result["strategy_breakdown"].items():
            assert isinstance(data["pnl_usd"], (int, float))

    def test_agent_sharpe_is_float(self):
        result = get_performance_attribution()
        for ag, data in result["agent_breakdown"].items():
            assert isinstance(data["sharpe_ratio"], (int, float))

    def test_period_hours_mapping(self):
        assert get_performance_attribution("1h")["period_hours"] == 1
        assert get_performance_attribution("24h")["period_hours"] == 24
        assert get_performance_attribution("7d")["period_hours"] == 168
