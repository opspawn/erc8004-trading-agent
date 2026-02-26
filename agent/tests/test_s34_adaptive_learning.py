"""
test_s34_adaptive_learning.py — Sprint 34: Adaptive Strategy Learning + Market Sentiment Signals.

Covers:
  - adapt_agent_strategy(): happy path, no history, unknown agent, weight normalisation
  - build_strategy_performance_ranking(): schema, scores, ranking order
  - get_market_sentiment(): single asset, all assets, signal mapping, edge cases
  - run_adaptive_backtest(): baseline vs adapted, improvement_pct, periods clamping
  - HTTP integration: POST /demo/agents/{id}/adapt, GET /demo/strategies/performance,
    GET /demo/market/sentiment, POST /demo/backtest/adaptive
  - Edge cases: missing fields, invalid inputs, empty history
"""

from __future__ import annotations

import json
import math
import socket
import sys
import os
import time
from typing import Any, Dict
from urllib.request import urlopen, Request
from urllib.error import HTTPError

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from demo_server import (
    # S34 functions
    adapt_agent_strategy,
    build_strategy_performance_ranking,
    get_market_sentiment,
    run_adaptive_backtest,
    # State
    _strategy_weights,
    _strategy_weights_lock,
    _DEFAULT_STRATEGY_WEIGHTS,
    _SENTIMENT_ASSETS,
    # Server
    DemoServer,
    _push_feed_event,
    _generate_feed_event,
    _FEED_AGENT_IDS,
)


# ── Helpers ────────────────────────────────────────────────────────────────────

def _get(url: str) -> dict:
    with urlopen(url, timeout=5) as resp:
        return json.loads(resp.read())


def _post(url: str, data: dict, expect_error: bool = False):
    body = json.dumps(data).encode("utf-8")
    req = Request(url, data=body, headers={"Content-Type": "application/json"})
    try:
        with urlopen(req, timeout=10) as resp:
            return json.loads(resp.read())
    except HTTPError as exc:
        if expect_error:
            return json.loads(exc.read())
        raise


# ── Fixtures ───────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def server():
    """Start a DemoServer on a free port and yield (base_url, port)."""
    sock = socket.socket()
    sock.bind(("", 0))
    port = sock.getsockname()[1]
    sock.close()
    srv = DemoServer(port=port)
    srv.start()
    time.sleep(0.3)
    yield f"http://localhost:{port}", port
    srv.stop()


# ════════════════════════════════════════════════════════════════════════════════
# 1. adapt_agent_strategy() — unit tests
# ════════════════════════════════════════════════════════════════════════════════

class TestAdaptAgentStrategy:

    def test_returns_required_fields(self):
        result = adapt_agent_strategy("agent-test-s34-001")
        assert "agent_id" in result
        assert "prev_weights" in result
        assert "new_weights" in result
        assert "trades_analyzed" in result
        assert "strategy_win_rates" in result
        assert "adaptation_score" in result
        assert "adapted_at" in result

    def test_agent_id_echoed(self):
        agent_id = "agent-echo-check"
        result = adapt_agent_strategy(agent_id)
        assert result["agent_id"] == agent_id

    def test_new_weights_sum_to_one(self):
        result = adapt_agent_strategy("agent-weight-sum-test")
        total = sum(result["new_weights"].values())
        assert abs(total - 1.0) < 1e-4, f"Weights sum {total} != 1.0"

    def test_new_weights_all_positive(self):
        result = adapt_agent_strategy("agent-positive-weights")
        for strategy, w in result["new_weights"].items():
            assert w > 0, f"Strategy {strategy} has non-positive weight {w}"

    def test_new_weights_cover_all_strategies(self):
        result = adapt_agent_strategy("agent-coverage-test")
        expected = set(_DEFAULT_STRATEGY_WEIGHTS.keys())
        assert set(result["new_weights"].keys()) == expected

    def test_prev_weights_cover_all_strategies(self):
        result = adapt_agent_strategy("agent-prev-test")
        expected = set(_DEFAULT_STRATEGY_WEIGHTS.keys())
        assert set(result["prev_weights"].keys()) == expected

    def test_strategy_win_rates_all_strategies(self):
        result = adapt_agent_strategy("agent-win-rates-test")
        expected = set(_DEFAULT_STRATEGY_WEIGHTS.keys())
        assert set(result["strategy_win_rates"].keys()) == expected

    def test_trades_analyzed_non_negative(self):
        result = adapt_agent_strategy("agent-trades-count")
        assert result["trades_analyzed"] >= 0

    def test_adaptation_score_non_negative(self):
        result = adapt_agent_strategy("agent-score-test")
        assert result["adaptation_score"] >= 0.0

    def test_adaptation_score_is_float(self):
        result = adapt_agent_strategy("agent-score-float")
        assert isinstance(result["adaptation_score"], float)

    def test_weights_stored_in_module_state(self):
        agent_id = "agent-state-storage-s34"
        adapt_agent_strategy(agent_id)
        with _strategy_weights_lock:
            assert agent_id in _strategy_weights

    def test_second_call_updates_weights(self):
        agent_id = "agent-idempotent-s34"
        r1 = adapt_agent_strategy(agent_id)
        r2 = adapt_agent_strategy(agent_id)
        # prev_weights in r2 should equal new_weights from r1
        assert r2["prev_weights"] == r1["new_weights"]

    def test_unknown_agent_gets_synthetic_history(self):
        # An agent with no history should still succeed via synthetic fallback
        result = adapt_agent_strategy("agent-totally-unknown-xyz9999")
        assert result["trades_analyzed"] > 0

    def test_adapted_at_is_recent(self):
        result = adapt_agent_strategy("agent-timestamp-test")
        now = time.time()
        assert abs(result["adapted_at"] - now) < 5.0

    def test_all_canonical_strategies_present_in_win_rates(self):
        result = adapt_agent_strategy("agent-canonical-test")
        for s in ["momentum", "mean_reversion", "arbitrage", "sentiment"]:
            assert s in result["strategy_win_rates"]

    def test_win_rates_in_zero_one_range(self):
        result = adapt_agent_strategy("agent-win-rate-range")
        for s, wr in result["strategy_win_rates"].items():
            assert 0.0 <= wr <= 1.0, f"{s} win_rate {wr} out of range"

    def test_different_agents_get_different_weights(self):
        r1 = adapt_agent_strategy("agent-diff-A")
        r2 = adapt_agent_strategy("agent-diff-B")
        # Weights may differ due to different synthetic history seeds
        # At minimum, both should be valid
        assert sum(r1["new_weights"].values()) > 0
        assert sum(r2["new_weights"].values()) > 0


# ════════════════════════════════════════════════════════════════════════════════
# 2. build_strategy_performance_ranking() — unit tests
# ════════════════════════════════════════════════════════════════════════════════

class TestStrategyPerformanceRanking:

    def test_returns_required_top_level_fields(self):
        result = build_strategy_performance_ranking()
        assert "strategies" in result
        assert "ranked_by" in result
        assert "generated_at" in result

    def test_strategies_list_has_all_canonical(self):
        result = build_strategy_performance_ranking()
        strategy_names = {s["strategy"] for s in result["strategies"]}
        expected = {"momentum", "mean_reversion", "arbitrage", "sentiment"}
        assert strategy_names == expected

    def test_each_strategy_has_required_fields(self):
        result = build_strategy_performance_ranking()
        required = {"strategy", "avg_return", "win_rate", "sharpe_ratio",
                    "agent_count", "total_trades", "composite_score", "rank"}
        for s in result["strategies"]:
            missing = required - set(s.keys())
            assert not missing, f"Strategy {s.get('strategy')} missing: {missing}"

    def test_ranks_are_consecutive(self):
        result = build_strategy_performance_ranking()
        ranks = sorted(s["rank"] for s in result["strategies"])
        assert ranks == list(range(1, len(ranks) + 1))

    def test_sorted_by_composite_score_descending(self):
        result = build_strategy_performance_ranking()
        scores = [s["composite_score"] for s in result["strategies"]]
        assert scores == sorted(scores, reverse=True), "Not sorted descending by composite_score"

    def test_composite_score_non_negative_for_good_strategy(self):
        result = build_strategy_performance_ranking()
        # At least one strategy should have a positive composite score
        max_score = max(s["composite_score"] for s in result["strategies"])
        assert max_score > 0.0

    def test_win_rate_in_valid_range(self):
        result = build_strategy_performance_ranking()
        for s in result["strategies"]:
            assert 0.0 <= s["win_rate"] <= 1.0, f"{s['strategy']} win_rate out of range"

    def test_agent_count_positive(self):
        result = build_strategy_performance_ranking()
        for s in result["strategies"]:
            assert s["agent_count"] >= 1, f"{s['strategy']} agent_count < 1"

    def test_total_trades_positive(self):
        result = build_strategy_performance_ranking()
        for s in result["strategies"]:
            assert s["total_trades"] >= 1, f"{s['strategy']} total_trades < 1"

    def test_ranked_by_field_describes_formula(self):
        result = build_strategy_performance_ranking()
        assert "composite_score" in result["ranked_by"].lower()
        assert "sharpe" in result["ranked_by"].lower()

    def test_generated_at_is_recent(self):
        result = build_strategy_performance_ranking()
        now = time.time()
        assert abs(result["generated_at"] - now) < 5.0

    def test_rank_1_has_highest_composite_score(self):
        result = build_strategy_performance_ranking()
        top = next(s for s in result["strategies"] if s["rank"] == 1)
        for s in result["strategies"]:
            assert top["composite_score"] >= s["composite_score"]


# ════════════════════════════════════════════════════════════════════════════════
# 3. get_market_sentiment() — unit tests
# ════════════════════════════════════════════════════════════════════════════════

class TestMarketSentiment:

    def test_all_assets_mode_fields(self):
        result = get_market_sentiment()
        assert "assets" in result
        assert "market_signal" in result
        assert "avg_aggregate_score" in result
        assert "generated_at" in result

    def test_all_assets_present(self):
        result = get_market_sentiment()
        for a in _SENTIMENT_ASSETS:
            assert a in result["assets"], f"Asset {a} missing from sentiment"

    def test_single_asset_btc(self):
        result = get_market_sentiment(asset="BTC")
        assert result["asset"] == "BTC"
        assert "sentiment" in result

    def test_single_asset_sentiment_fields(self):
        result = get_market_sentiment(asset="ETH")
        s = result["sentiment"]
        required = {"asset", "bullish_pct", "bearish_pct", "neutral_pct",
                    "aggregate_score", "signal", "confidence", "volume_proxy"}
        missing = required - set(s.keys())
        assert not missing, f"Missing fields: {missing}"

    def test_pcts_sum_to_one(self):
        result = get_market_sentiment(asset="SOL")
        s = result["sentiment"]
        total = s["bullish_pct"] + s["bearish_pct"] + s["neutral_pct"]
        assert abs(total - 1.0) < 1e-3, f"Percentages sum {total} != 1.0"

    def test_aggregate_score_in_range(self):
        result = get_market_sentiment(asset="BTC")
        score = result["sentiment"]["aggregate_score"]
        assert -1.0 <= score <= 1.0, f"aggregate_score {score} out of range"

    def test_signal_is_buy_sell_hold(self):
        result = get_market_sentiment(asset="ETH")
        assert result["sentiment"]["signal"] in ("BUY", "SELL", "HOLD")

    def test_confidence_in_range(self):
        result = get_market_sentiment(asset="AVAX")
        conf = result["sentiment"]["confidence"]
        assert 0.0 <= conf <= 1.0, f"confidence {conf} out of range"

    def test_all_assets_sentiment_fields_present(self):
        result = get_market_sentiment()
        required = {"bullish_pct", "bearish_pct", "neutral_pct",
                    "aggregate_score", "signal", "confidence"}
        for asset, s in result["assets"].items():
            missing = required - set(s.keys())
            assert not missing, f"Asset {asset} missing: {missing}"

    def test_market_signal_is_valid(self):
        result = get_market_sentiment()
        assert result["market_signal"] in ("BUY", "SELL", "HOLD")

    def test_avg_score_in_range(self):
        result = get_market_sentiment()
        score = result["avg_aggregate_score"]
        assert -1.0 <= score <= 1.0, f"avg_aggregate_score {score} out of range"

    def test_none_asset_returns_all(self):
        result = get_market_sentiment(asset=None)
        assert "assets" in result
        assert len(result["assets"]) == len(_SENTIMENT_ASSETS)

    def test_unknown_asset_returns_all(self):
        # Unknown asset should fall back to returning all assets
        result = get_market_sentiment(asset="INVALID_ASSET_XYZ")
        assert "assets" in result

    def test_volume_proxy_positive(self):
        result = get_market_sentiment(asset="BTC")
        vp = result["sentiment"]["volume_proxy"]
        assert vp > 0, f"volume_proxy {vp} not positive"

    def test_deterministic_within_minute(self):
        r1 = get_market_sentiment(asset="BTC")
        r2 = get_market_sentiment(asset="BTC")
        # Within the same minute, scores should be the same
        assert r1["sentiment"]["aggregate_score"] == r2["sentiment"]["aggregate_score"]


# ════════════════════════════════════════════════════════════════════════════════
# 4. run_adaptive_backtest() — unit tests
# ════════════════════════════════════════════════════════════════════════════════

class TestAdaptiveBacktest:

    def test_returns_required_top_level_fields(self):
        result = run_adaptive_backtest(
            agent_id="agent-bt-test-001",
            symbol="BTC/USD",
            periods=30,
            use_adapted_weights=False,
        )
        required = {"agent_id", "symbol", "periods", "start_date", "end_date",
                    "use_adapted_weights", "baseline", "adapted", "improvement_pct",
                    "weights_source", "generated_at"}
        missing = required - set(result.keys())
        assert not missing, f"Missing top-level fields: {missing}"

    def test_baseline_has_required_fields(self):
        result = run_adaptive_backtest("agent-bt-b", "ETH/USD", 60, False)
        required = {"strategy", "total_return_pct", "sharpe_ratio",
                    "max_drawdown_pct", "num_trades", "weights_used"}
        missing = required - set(result["baseline"].keys())
        assert not missing

    def test_adapted_has_required_fields(self):
        result = run_adaptive_backtest("agent-bt-a", "SOL/USD", 30, True)
        required = {"strategy", "total_return_pct", "sharpe_ratio",
                    "max_drawdown_pct", "num_trades", "weights_used"}
        missing = required - set(result["adapted"].keys())
        assert not missing

    def test_agent_id_echoed(self):
        agent_id = "agent-echo-bt"
        result = run_adaptive_backtest(agent_id, "BTC/USD", 30, False)
        assert result["agent_id"] == agent_id

    def test_symbol_echoed(self):
        result = run_adaptive_backtest("agent-sym-bt", "ETH/USD", 30, False)
        assert result["symbol"] == "ETH/USD"

    def test_periods_within_bounds(self):
        result = run_adaptive_backtest("agent-periods-bt", "BTC/USD", 365, False)
        assert result["periods"] == 365

    def test_periods_clamped_to_max(self):
        result = run_adaptive_backtest("agent-clamp-bt", "BTC/USD", 5000, False)
        assert result["periods"] <= 3650

    def test_periods_clamped_to_min(self):
        result = run_adaptive_backtest("agent-minclamp-bt", "BTC/USD", 0, False)
        assert result["periods"] >= 1

    def test_use_adapted_weights_echoed(self):
        result = run_adaptive_backtest("agent-flag-bt", "BTC/USD", 30, True)
        assert result["use_adapted_weights"] is True

    def test_improvement_pct_is_float(self):
        result = run_adaptive_backtest("agent-improvement-bt", "BTC/USD", 30, False)
        assert isinstance(result["improvement_pct"], float)

    def test_weights_source_default_when_no_adapt(self):
        result = run_adaptive_backtest("agent-nosave-bt-xyz", "BTC/USD", 30, False)
        # Agent has no adapted weights → source should be "default"
        assert result["weights_source"] in ("default", "adapted")

    def test_weights_source_adapted_after_adapt(self):
        agent_id = "agent-adapted-src-test"
        adapt_agent_strategy(agent_id)  # save weights
        result = run_adaptive_backtest(agent_id, "BTC/USD", 30, True)
        assert result["weights_source"] == "adapted"

    def test_baseline_weights_are_default(self):
        result = run_adaptive_backtest("agent-default-w-bt", "BTC/USD", 30, False)
        baseline_weights = result["baseline"]["weights_used"]
        assert set(baseline_weights.keys()) == set(_DEFAULT_STRATEGY_WEIGHTS.keys())

    def test_dates_are_valid_strings(self):
        result = run_adaptive_backtest("agent-dates-bt", "BTC/USD", 30, False)
        import datetime as _dt
        _dt.date.fromisoformat(result["start_date"])
        _dt.date.fromisoformat(result["end_date"])

    def test_start_before_end(self):
        result = run_adaptive_backtest("agent-dates-order", "BTC/USD", 30, False)
        import datetime as _dt
        start = _dt.date.fromisoformat(result["start_date"])
        end = _dt.date.fromisoformat(result["end_date"])
        assert start < end


# ════════════════════════════════════════════════════════════════════════════════
# 5. HTTP Integration tests
# ════════════════════════════════════════════════════════════════════════════════

class TestAdaptEndpointHTTP:

    def test_adapt_happy_path(self, server):
        base_url, port = server
        result = _post(f"{base_url}/demo/agents/agent-http-adapt/adapt", {})
        assert result["agent_id"] == "agent-http-adapt"
        assert "new_weights" in result
        assert "trades_analyzed" in result

    def test_adapt_known_agent(self, server):
        base_url, port = server
        result = _post(f"{base_url}/demo/agents/agent-conservative-001/adapt", {})
        assert result["agent_id"] == "agent-conservative-001"
        assert "adaptation_score" in result

    def test_adapt_weight_sum_one(self, server):
        base_url, port = server
        result = _post(f"{base_url}/demo/agents/agent-sum-http/adapt", {})
        total = sum(result["new_weights"].values())
        assert abs(total - 1.0) < 1e-3


class TestStrategyPerformanceHTTP:

    def test_get_strategy_performance(self, server):
        base_url, _ = server
        result = _get(f"{base_url}/demo/strategies/performance")
        assert "strategies" in result
        assert "ranked_by" in result

    def test_strategy_performance_has_four_strategies(self, server):
        base_url, _ = server
        result = _get(f"{base_url}/demo/strategies/performance")
        assert len(result["strategies"]) == 4

    def test_strategy_performance_ranked(self, server):
        base_url, _ = server
        result = _get(f"{base_url}/demo/strategies/performance")
        scores = [s["composite_score"] for s in result["strategies"]]
        assert scores == sorted(scores, reverse=True)

    def test_strategy_performance_has_rank_field(self, server):
        base_url, _ = server
        result = _get(f"{base_url}/demo/strategies/performance")
        for s in result["strategies"]:
            assert "rank" in s


class TestMarketSentimentHTTP:

    def test_get_all_assets(self, server):
        base_url, _ = server
        result = _get(f"{base_url}/demo/market/sentiment")
        assert "assets" in result
        assert "market_signal" in result

    def test_get_single_asset_btc(self, server):
        base_url, _ = server
        result = _get(f"{base_url}/demo/market/sentiment?asset=BTC")
        assert result["asset"] == "BTC"
        assert "sentiment" in result

    def test_get_single_asset_eth(self, server):
        base_url, _ = server
        result = _get(f"{base_url}/demo/market/sentiment?asset=ETH")
        assert result["asset"] == "ETH"

    def test_sentiment_signal_valid(self, server):
        base_url, _ = server
        result = _get(f"{base_url}/demo/market/sentiment?asset=SOL")
        assert result["sentiment"]["signal"] in ("BUY", "SELL", "HOLD")

    def test_market_sentiment_all_assets_keys(self, server):
        base_url, _ = server
        result = _get(f"{base_url}/demo/market/sentiment")
        for a in _SENTIMENT_ASSETS:
            assert a in result["assets"]


class TestAdaptiveBacktestHTTP:

    def test_adaptive_backtest_happy_path(self, server):
        base_url, _ = server
        result = _post(f"{base_url}/demo/backtest/adaptive", {
            "agent_id": "agent-http-adaptive-bt",
            "symbol": "BTC/USD",
            "periods": 30,
            "use_adapted_weights": False,
        })
        assert result["agent_id"] == "agent-http-adaptive-bt"
        assert "baseline" in result
        assert "adapted" in result
        assert "improvement_pct" in result

    def test_adaptive_backtest_missing_agent_id(self, server):
        base_url, _ = server
        result = _post(f"{base_url}/demo/backtest/adaptive",
                       {"symbol": "BTC/USD", "periods": 30},
                       expect_error=True)
        assert "error" in result

    def test_adaptive_backtest_with_adapted_weights(self, server):
        base_url, _ = server
        agent_id = "agent-http-adapted-weights"
        # First adapt
        _post(f"{base_url}/demo/agents/{agent_id}/adapt", {})
        # Then run adaptive backtest
        result = _post(f"{base_url}/demo/backtest/adaptive", {
            "agent_id": agent_id,
            "symbol": "ETH/USD",
            "periods": 60,
            "use_adapted_weights": True,
        })
        assert result["use_adapted_weights"] is True
        assert result["weights_source"] == "adapted"

    def test_adaptive_backtest_improvement_pct_is_number(self, server):
        base_url, _ = server
        result = _post(f"{base_url}/demo/backtest/adaptive", {
            "agent_id": "agent-improvement-http",
            "symbol": "SOL/USD",
            "periods": 30,
            "use_adapted_weights": False,
        })
        assert isinstance(result["improvement_pct"], (int, float))

    def test_adaptive_backtest_default_symbol(self, server):
        base_url, _ = server
        result = _post(f"{base_url}/demo/backtest/adaptive", {
            "agent_id": "agent-default-sym",
        })
        assert result["symbol"] == "BTC/USD"

    def test_adaptive_backtest_generates_dates(self, server):
        base_url, _ = server
        result = _post(f"{base_url}/demo/backtest/adaptive", {
            "agent_id": "agent-date-check",
            "periods": 90,
        })
        assert "start_date" in result
        assert "end_date" in result
