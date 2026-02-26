"""
Tests for demo_server.py — ERC-8004 Live Demo HTTP Endpoint.

Covers:
  - X402Gate: dev_mode bypass, payment required response
  - run_demo_pipeline: output schema, field types, artifact generation
  - DemoServer: start/stop lifecycle, HTTP GET/POST routing
  - /demo/run: status 200 happy path, 402 gate, 400 bad params, 404 unknown
  - /demo/health: GET response structure
  - /demo/info: GET endpoint docs
  - Response field completeness and types
  - Edge cases: ticks clamped to [1, 100], unknown paths
"""

from __future__ import annotations

import json
import sys
import os
import time
import urllib.request
import urllib.error
from typing import Dict, Any

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from demo_server import (
    X402Gate,
    run_demo_pipeline,
    build_portfolio_summary,
    DemoServer,
    DEFAULT_PORT,
    DEFAULT_TICKS,
    SERVER_VERSION,
    X402_PRICE_USDC,
    X402_RECEIVER,
    _DEFAULT_PORTFOLIO,
)


# ─── Fixtures ────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def demo_server():
    """Start a DemoServer on a free test port for the entire module."""
    port = 18084
    server = DemoServer(port=port)
    server.start()
    time.sleep(0.3)
    yield port
    server.stop()


def _get(port: int, path: str) -> Dict[str, Any]:
    resp = urllib.request.urlopen(f"http://localhost:{port}{path}")
    return json.loads(resp.read())


def _post(port: int, path: str, extra_headers: Dict[str, str] | None = None) -> Dict[str, Any]:
    req = urllib.request.Request(
        f"http://localhost:{port}{path}", method="POST"
    )
    req.add_header("Content-Length", "0")
    if extra_headers:
        for k, v in extra_headers.items():
            req.add_header(k, v)
    resp = urllib.request.urlopen(req)
    return json.loads(resp.read())


def _post_expect_error(port: int, path: str, extra_headers: Dict[str, str] | None = None):
    req = urllib.request.Request(
        f"http://localhost:{port}{path}", method="POST"
    )
    req.add_header("Content-Length", "0")
    if extra_headers:
        for k, v in extra_headers.items():
            req.add_header(k, v)
    try:
        urllib.request.urlopen(req)
        return None  # no error
    except urllib.error.HTTPError as e:
        return e


# ─── X402Gate ────────────────────────────────────────────────────────────────

class TestX402Gate:
    def test_dev_mode_always_passes(self):
        gate = X402Gate(dev_mode=True)
        passed, err = gate.check({})
        assert passed is True
        assert err is None

    def test_dev_mode_passes_without_header(self):
        gate = X402Gate(dev_mode=True)
        passed, err = gate.check({"x-payment": ""})
        assert passed is True

    def test_live_mode_no_header_returns_402(self):
        gate = X402Gate(dev_mode=False)
        passed, err = gate.check({})
        assert passed is False
        assert err is not None
        assert "x402Version" in err

    def test_live_mode_with_payment_header_passes(self):
        gate = X402Gate(dev_mode=False)
        passed, err = gate.check({"x-payment": "some-payment-token"})
        assert passed is True
        assert err is None

    def test_payment_required_body_structure(self):
        gate = X402Gate(dev_mode=False, price_usdc="5000")
        passed, err = gate.check({})
        assert err["x402Version"] == 1
        assert err["error"] == "Payment required"
        accepts = err["accepts"]
        assert len(accepts) == 1
        a = accepts[0]
        assert a["scheme"] == "exact"
        assert a["network"] == "eip155:8453"
        assert a["maxAmountRequired"] == "5000"
        assert "/demo/run" in a["resource"]
        assert a["payTo"] == X402_RECEIVER

    def test_payment_required_uses_price_param(self):
        gate = X402Gate(dev_mode=False, price_usdc="99999")
        _, err = gate.check({})
        assert err["accepts"][0]["maxAmountRequired"] == "99999"

    def test_x_payment_uppercase_header(self):
        gate = X402Gate(dev_mode=False)
        passed, _ = gate.check({"X-PAYMENT": "token"})
        assert passed is True

    def test_default_dev_mode_is_true(self):
        gate = X402Gate()
        passed, _ = gate.check({})
        assert passed is True


# ─── run_demo_pipeline ────────────────────────────────────────────────────────

class TestRunDemoPipeline:
    def test_returns_dict(self):
        result = run_demo_pipeline(ticks=5, seed=1)
        assert isinstance(result, dict)

    def test_top_level_keys(self):
        result = run_demo_pipeline(ticks=5, seed=1)
        assert result["status"] == "ok"
        assert result["pipeline"] == "ERC-8004 Autonomous Trading Agent"
        assert result["version"] == SERVER_VERSION
        assert "demo" in result
        assert "agents" in result
        assert "validation_artifact" in result
        assert "x402" in result

    def test_demo_section_fields(self):
        result = run_demo_pipeline(ticks=5, seed=42)
        demo = result["demo"]
        assert demo["ticks_run"] == 5
        assert isinstance(demo["trades_executed"], int)
        assert isinstance(demo["consensus_reached"], int)
        assert 0.0 <= demo["consensus_rate"] <= 1.0
        assert isinstance(demo["total_pnl_usd"], float)
        assert isinstance(demo["avg_reputation_score"], float)
        assert demo["price_start"] > 0
        assert demo["price_end"] > 0
        assert isinstance(demo["duration_ms"], float)
        assert demo["duration_ms"] >= 0

    def test_demo_ticks_matches_input(self):
        for n in [1, 5, 20]:
            r = run_demo_pipeline(ticks=n, seed=7)
            assert r["demo"]["ticks_run"] == n

    def test_agents_list_length(self):
        result = run_demo_pipeline(ticks=5, seed=1)
        assert len(result["agents"]) == 3

    def test_agent_profiles(self):
        result = run_demo_pipeline(ticks=5, seed=1)
        profiles = [a["profile"] for a in result["agents"]]
        assert "conservative" in profiles
        assert "balanced" in profiles
        assert "aggressive" in profiles

    def test_agent_fields(self):
        result = run_demo_pipeline(ticks=10, seed=1)
        for agent in result["agents"]:
            assert "id" in agent
            assert "profile" in agent
            assert "trades" in agent
            assert 0.0 <= agent["win_rate"] <= 1.0
            assert isinstance(agent["pnl_usd"], float)
            assert agent["reputation_start"] > 0
            assert agent["reputation_end"] >= 0
            assert isinstance(agent["reputation_delta"], float)

    def test_validation_artifact_fields(self):
        result = run_demo_pipeline(ticks=5, seed=1)
        art = result["validation_artifact"]
        assert "session_id" in art
        assert "artifact_hash" in art
        assert "strategy_hash" in art
        assert art["artifact_hash"].startswith("0x")
        assert art["strategy_hash"].startswith("0x")
        assert isinstance(art["trades_count"], int)
        assert 0.0 <= art["win_rate"] <= 1.0
        assert isinstance(art["avg_pnl_bps"], float)
        assert isinstance(art["max_drawdown_bps"], float)
        assert isinstance(art["risk_violations"], int)
        assert "timestamp" in art
        assert "signature" in art

    def test_x402_section_dev_mode(self):
        result = run_demo_pipeline(ticks=5, seed=1)
        x = result["x402"]
        assert x["dev_mode"] is True
        assert x["payment_gated"] is False
        assert x["price_usdc"] == X402_PRICE_USDC
        assert x["receiver"] == X402_RECEIVER

    def test_reputation_scores_in_range(self):
        result = run_demo_pipeline(ticks=20, seed=99)
        for agent in result["agents"]:
            assert 0.0 <= agent["reputation_end"] <= 10.0

    def test_consensus_rate_bounded(self):
        result = run_demo_pipeline(ticks=10, seed=42)
        assert 0.0 <= result["demo"]["consensus_rate"] <= 1.0

    def test_deterministic_with_seed(self):
        r1 = run_demo_pipeline(ticks=10, seed=123)
        r2 = run_demo_pipeline(ticks=10, seed=123)
        assert r1["demo"]["price_end"] == r2["demo"]["price_end"]
        assert r1["demo"]["consensus_reached"] == r2["demo"]["consensus_reached"]

    def test_different_seeds_differ(self):
        r1 = run_demo_pipeline(ticks=20, seed=1)
        r2 = run_demo_pipeline(ticks=20, seed=2)
        # Different seeds should produce different prices
        assert r1["demo"]["price_end"] != r2["demo"]["price_end"]

    def test_symbol_in_response(self):
        result = run_demo_pipeline(ticks=5, seed=1, symbol="ETH/USD")
        assert result["demo"]["symbol"] == "ETH/USD"

    def test_btc_default_symbol(self):
        result = run_demo_pipeline(ticks=5, seed=1)
        assert result["demo"]["symbol"] == "BTC/USD"

    def test_artifact_hash_is_hex(self):
        result = run_demo_pipeline(ticks=5, seed=1)
        h = result["validation_artifact"]["artifact_hash"]
        # "0x" + 64 hex chars = SHA-256
        assert len(h) >= 10
        int(h, 16)  # must be parseable as hex

    def test_agent_ids_are_strings(self):
        result = run_demo_pipeline(ticks=5, seed=1)
        for agent in result["agents"]:
            assert isinstance(agent["id"], str)
            assert len(agent["id"]) > 0

    def test_duration_positive(self):
        result = run_demo_pipeline(ticks=5, seed=1)
        assert result["demo"]["duration_ms"] > 0

    def test_large_tick_count(self):
        result = run_demo_pipeline(ticks=100, seed=42)
        assert result["demo"]["ticks_run"] == 100

    def test_single_tick(self):
        result = run_demo_pipeline(ticks=1, seed=42)
        assert result["demo"]["ticks_run"] == 1

    def test_price_return_is_float(self):
        result = run_demo_pipeline(ticks=10, seed=42)
        assert isinstance(result["demo"]["price_return_pct"], float)


# ─── DemoServer HTTP ──────────────────────────────────────────────────────────

class TestDemoServerHealth:
    def test_health_200(self, demo_server):
        data = _get(demo_server, "/demo/health")
        assert data["status"] == "ok"

    def test_health_service_name(self, demo_server):
        data = _get(demo_server, "/demo/health")
        assert data["service"] == "ERC-8004 Demo Server"

    def test_health_version(self, demo_server):
        data = _get(demo_server, "/demo/health")
        assert data["version"] == SERVER_VERSION

    def test_health_port(self, demo_server):
        data = _get(demo_server, "/demo/health")
        assert data["port"] == DEFAULT_PORT

    def test_health_dev_mode(self, demo_server):
        data = _get(demo_server, "/demo/health")
        assert isinstance(data["dev_mode"], bool)

    def test_health_shortpath(self, demo_server):
        data = _get(demo_server, "/health")
        assert data["status"] == "ok"


class TestDemoServerInfo:
    def test_info_200(self, demo_server):
        data = _get(demo_server, "/demo/info")
        assert "endpoints" in data

    def test_info_service_name(self, demo_server):
        data = _get(demo_server, "/demo/info")
        assert "ERC-8004" in data["service"]

    def test_info_version(self, demo_server):
        data = _get(demo_server, "/demo/info")
        assert data["version"] == SERVER_VERSION

    def test_info_endpoints_doc(self, demo_server):
        data = _get(demo_server, "/demo/info")
        endpoints = data["endpoints"]
        assert any("/demo/run" in k for k in endpoints)
        assert any("/demo/health" in k for k in endpoints)

    def test_info_query_params_doc(self, demo_server):
        data = _get(demo_server, "/demo/info")
        assert "ticks" in data["query_params"]
        assert "seed" in data["query_params"]

    def test_info_example_curl(self, demo_server):
        data = _get(demo_server, "/demo/info")
        assert "curl" in data["example_curl"].lower()

    def test_info_x402_docs(self, demo_server):
        data = _get(demo_server, "/demo/info")
        assert "x402" in data

    def test_info_shortpath(self, demo_server):
        data = _get(demo_server, "/info")
        assert "service" in data


class TestDemoServerRun:
    def test_run_returns_200(self, demo_server):
        data = _post(demo_server, "/demo/run")
        assert data["status"] == "ok"

    def test_run_default_ticks(self, demo_server):
        data = _post(demo_server, "/demo/run")
        assert data["demo"]["ticks_run"] == DEFAULT_TICKS

    def test_run_custom_ticks(self, demo_server):
        data = _post(demo_server, "/demo/run?ticks=5")
        assert data["demo"]["ticks_run"] == 5

    def test_run_custom_seed(self, demo_server):
        data1 = _post(demo_server, "/demo/run?ticks=5&seed=77")
        data2 = _post(demo_server, "/demo/run?ticks=5&seed=77")
        assert data1["demo"]["price_end"] == data2["demo"]["price_end"]

    def test_run_different_seeds_differ(self, demo_server):
        data1 = _post(demo_server, "/demo/run?ticks=20&seed=1")
        data2 = _post(demo_server, "/demo/run?ticks=20&seed=2")
        assert data1["demo"]["price_end"] != data2["demo"]["price_end"]

    def test_run_custom_symbol(self, demo_server):
        data = _post(demo_server, "/demo/run?symbol=ETH/USD")
        assert data["demo"]["symbol"] == "ETH/USD"

    def test_run_ticks_clamped_max(self, demo_server):
        data = _post(demo_server, "/demo/run?ticks=9999")
        assert data["demo"]["ticks_run"] == 100

    def test_run_ticks_clamped_min(self, demo_server):
        data = _post(demo_server, "/demo/run?ticks=0")
        assert data["demo"]["ticks_run"] == 1

    def test_run_agents_count(self, demo_server):
        data = _post(demo_server, "/demo/run")
        assert len(data["agents"]) == 3

    def test_run_validation_artifact_present(self, demo_server):
        data = _post(demo_server, "/demo/run")
        assert "artifact_hash" in data["validation_artifact"]
        assert data["validation_artifact"]["artifact_hash"].startswith("0x")

    def test_run_x402_dev_mode(self, demo_server):
        data = _post(demo_server, "/demo/run")
        assert data["x402"]["dev_mode"] is True

    def test_run_pipeline_name(self, demo_server):
        data = _post(demo_server, "/demo/run")
        assert "ERC-8004" in data["pipeline"]

    def test_run_version(self, demo_server):
        data = _post(demo_server, "/demo/run")
        assert data["version"] == SERVER_VERSION


class TestDemoServerErrors:
    def test_unknown_get_returns_404(self, demo_server):
        err = None
        try:
            urllib.request.urlopen(f"http://localhost:{demo_server}/unknown")
        except urllib.error.HTTPError as e:
            err = e
        assert err is not None
        assert err.code == 404

    def test_unknown_post_returns_404(self, demo_server):
        err = _post_expect_error(demo_server, "/not-a-route")
        assert err is not None
        assert err.code == 404

    def test_404_body_is_json(self, demo_server):
        try:
            urllib.request.urlopen(f"http://localhost:{demo_server}/bad/path")
        except urllib.error.HTTPError as e:
            body = json.loads(e.read())
            assert "error" in body

    def test_invalid_ticks_param(self, demo_server):
        # non-numeric ticks: server should return 400
        err = _post_expect_error(demo_server, "/demo/run?ticks=abc")
        assert err is not None
        assert err.code == 400

    def test_bad_seed_param(self, demo_server):
        err = _post_expect_error(demo_server, "/demo/run?seed=notanumber")
        assert err is not None
        assert err.code == 400


class TestDemoServerLifecycle:
    def test_server_starts_and_stops(self):
        port = 18085
        server = DemoServer(port=port)
        server.start()
        time.sleep(0.2)
        data = _get(port, "/demo/health")
        assert data["status"] == "ok"
        server.stop()

    def test_double_stop_is_safe(self):
        port = 18086
        server = DemoServer(port=port)
        server.start()
        time.sleep(0.2)
        server.stop()
        server.stop()  # should not raise

    def test_server_handles_multiple_requests(self, demo_server):
        for _ in range(5):
            data = _get(demo_server, "/demo/health")
            assert data["status"] == "ok"

    def test_concurrent_run_requests(self, demo_server):
        import threading
        results = []
        errors = []

        def do_run():
            try:
                data = _post(demo_server, "/demo/run?ticks=5")
                results.append(data["status"])
            except Exception as e:
                errors.append(str(e))

        threads = [threading.Thread(target=do_run) for _ in range(3)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10)

        assert len(errors) == 0
        assert all(r == "ok" for r in results)


class TestDemoServerResponseHeaders:
    def test_cors_header_present(self, demo_server):
        resp = urllib.request.urlopen(f"http://localhost:{demo_server}/demo/health")
        assert resp.headers.get("Access-Control-Allow-Origin") == "*"

    def test_content_type_json(self, demo_server):
        resp = urllib.request.urlopen(f"http://localhost:{demo_server}/demo/health")
        assert "application/json" in resp.headers.get("Content-Type", "")

    def test_erc8004_version_header(self, demo_server):
        resp = urllib.request.urlopen(f"http://localhost:{demo_server}/demo/health")
        assert resp.headers.get("X-ERC8004-Version") == SERVER_VERSION


class TestDemoServerConstants:
    def test_default_port(self):
        assert DEFAULT_PORT == 8084

    def test_default_ticks(self):
        assert DEFAULT_TICKS == 10

    def test_server_version_string(self):
        assert isinstance(SERVER_VERSION, str)
        assert len(SERVER_VERSION) > 0

    def test_x402_price_usdc_is_string(self):
        assert isinstance(X402_PRICE_USDC, str)
        int(X402_PRICE_USDC)  # must be numeric

    def test_x402_receiver_is_address(self):
        assert X402_RECEIVER.startswith("0x")
        assert len(X402_RECEIVER) == 42


# ─── Portfolio Default Data ────────────────────────────────────────────────────

class TestDefaultPortfolio:
    def test_default_portfolio_source(self):
        assert _DEFAULT_PORTFOLIO["source"] == "default"

    def test_default_portfolio_has_note(self):
        assert "note" in _DEFAULT_PORTFOLIO
        assert len(_DEFAULT_PORTFOLIO["note"]) > 0

    def test_default_portfolio_agent_profiles_count(self):
        assert len(_DEFAULT_PORTFOLIO["agent_profiles"]) == 3

    def test_default_portfolio_agent_fields(self):
        for ap in _DEFAULT_PORTFOLIO["agent_profiles"]:
            assert "agent_id" in ap
            assert "strategy" in ap
            assert "trades_won" in ap
            assert "trades_lost" in ap
            assert "win_rate" in ap
            assert "total_pnl" in ap
            assert "reputation_score" in ap

    def test_default_portfolio_strategies(self):
        strategies = [ap["strategy"] for ap in _DEFAULT_PORTFOLIO["agent_profiles"]]
        assert "conservative" in strategies
        assert "balanced" in strategies
        assert "aggressive" in strategies

    def test_default_portfolio_win_rates_bounded(self):
        for ap in _DEFAULT_PORTFOLIO["agent_profiles"]:
            assert 0.0 <= ap["win_rate"] <= 1.0

    def test_default_portfolio_consensus_stats_keys(self):
        cs = _DEFAULT_PORTFOLIO["consensus_stats"]
        assert "avg_agreement_rate" in cs
        assert "supermajority_hits" in cs
        assert "veto_count" in cs

    def test_default_portfolio_risk_metrics_keys(self):
        rm = _DEFAULT_PORTFOLIO["risk_metrics"]
        assert "max_drawdown" in rm
        assert "sharpe_estimate" in rm
        assert "volatility" in rm

    def test_default_portfolio_consensus_rate_bounded(self):
        rate = _DEFAULT_PORTFOLIO["consensus_stats"]["avg_agreement_rate"]
        assert 0.0 <= rate <= 1.0

    def test_default_portfolio_drawdown_negative_or_zero(self):
        dd = _DEFAULT_PORTFOLIO["risk_metrics"]["max_drawdown"]
        assert dd <= 0.0


# ─── build_portfolio_summary ──────────────────────────────────────────────────

class TestBuildPortfolioSummary:
    @pytest.fixture
    def run_result(self):
        return run_demo_pipeline(ticks=10, seed=42)

    def test_returns_dict(self, run_result):
        summary = build_portfolio_summary(run_result)
        assert isinstance(summary, dict)

    def test_source_is_live(self, run_result):
        summary = build_portfolio_summary(run_result)
        assert summary["source"] == "live"

    def test_has_agent_profiles(self, run_result):
        summary = build_portfolio_summary(run_result)
        assert "agent_profiles" in summary
        assert len(summary["agent_profiles"]) == 3

    def test_agent_profile_fields(self, run_result):
        summary = build_portfolio_summary(run_result)
        for ap in summary["agent_profiles"]:
            assert "agent_id" in ap
            assert "strategy" in ap
            assert "trades_won" in ap
            assert "trades_lost" in ap
            assert "win_rate" in ap
            assert "total_pnl" in ap
            assert "reputation_score" in ap

    def test_agent_win_rate_bounded(self, run_result):
        summary = build_portfolio_summary(run_result)
        for ap in summary["agent_profiles"]:
            assert 0.0 <= ap["win_rate"] <= 1.0

    def test_agent_trades_nonnegative(self, run_result):
        summary = build_portfolio_summary(run_result)
        for ap in summary["agent_profiles"]:
            assert ap["trades_won"] >= 0
            assert ap["trades_lost"] >= 0

    def test_has_consensus_stats(self, run_result):
        summary = build_portfolio_summary(run_result)
        cs = summary["consensus_stats"]
        assert "avg_agreement_rate" in cs
        assert "supermajority_hits" in cs
        assert "veto_count" in cs

    def test_consensus_rate_bounded(self, run_result):
        summary = build_portfolio_summary(run_result)
        rate = summary["consensus_stats"]["avg_agreement_rate"]
        assert 0.0 <= rate <= 1.0

    def test_has_risk_metrics(self, run_result):
        summary = build_portfolio_summary(run_result)
        rm = summary["risk_metrics"]
        assert "max_drawdown" in rm
        assert "sharpe_estimate" in rm
        assert "volatility" in rm

    def test_risk_metrics_are_floats(self, run_result):
        summary = build_portfolio_summary(run_result)
        rm = summary["risk_metrics"]
        assert isinstance(rm["max_drawdown"], float)
        assert isinstance(rm["sharpe_estimate"], float)
        assert isinstance(rm["volatility"], float)

    def test_strategies_present(self, run_result):
        summary = build_portfolio_summary(run_result)
        strategies = [ap["strategy"] for ap in summary["agent_profiles"]]
        assert "conservative" in strategies
        assert "balanced" in strategies
        assert "aggressive" in strategies

    def test_deterministic_from_same_run(self):
        r = run_demo_pipeline(ticks=10, seed=99)
        s1 = build_portfolio_summary(r)
        s2 = build_portfolio_summary(r)
        assert s1["consensus_stats"]["avg_agreement_rate"] == s2["consensus_stats"]["avg_agreement_rate"]

    def test_different_seeds_produce_live_portfolios(self):
        r1 = run_demo_pipeline(ticks=20, seed=1)
        r2 = run_demo_pipeline(ticks=20, seed=2)
        s1 = build_portfolio_summary(r1)
        s2 = build_portfolio_summary(r2)
        # Both runs should produce live portfolio summaries
        assert s1["source"] == "live"
        assert s2["source"] == "live"
        # Both should have valid agent profiles
        assert len(s1["agent_profiles"]) == 3
        assert len(s2["agent_profiles"]) == 3


# ─── Portfolio HTTP Endpoint ───────────────────────────────────────────────────

class TestDemoServerPortfolio:
    @pytest.fixture(scope="class")
    def portfolio_server(self):
        """Server with a fresh state (no runs completed)."""
        import demo_server as ds
        # Reset last run so we get default values
        with ds._portfolio_lock:
            ds._last_run_result = None
        port = 18090
        server = DemoServer(port=port)
        server.start()
        time.sleep(0.3)
        yield port
        server.stop()
        # Cleanup
        with ds._portfolio_lock:
            ds._last_run_result = None

    def test_portfolio_returns_200(self, portfolio_server):
        data = _get(portfolio_server, "/demo/portfolio")
        assert isinstance(data, dict)

    def test_portfolio_before_run_is_default(self, portfolio_server):
        data = _get(portfolio_server, "/demo/portfolio")
        assert data["source"] == "default"

    def test_portfolio_default_has_agent_profiles(self, portfolio_server):
        data = _get(portfolio_server, "/demo/portfolio")
        assert "agent_profiles" in data
        assert len(data["agent_profiles"]) >= 1

    def test_portfolio_default_has_consensus_stats(self, portfolio_server):
        data = _get(portfolio_server, "/demo/portfolio")
        assert "consensus_stats" in data

    def test_portfolio_default_has_risk_metrics(self, portfolio_server):
        data = _get(portfolio_server, "/demo/portfolio")
        assert "risk_metrics" in data

    def test_portfolio_after_run_is_live(self, portfolio_server):
        # Trigger a run
        _post(portfolio_server, "/demo/run?ticks=5")
        data = _get(portfolio_server, "/demo/portfolio")
        assert data["source"] == "live"

    def test_portfolio_after_run_has_agent_profiles(self, portfolio_server):
        _post(portfolio_server, "/demo/run?ticks=5")
        data = _get(portfolio_server, "/demo/portfolio")
        assert len(data["agent_profiles"]) == 3

    def test_portfolio_after_run_agent_ids_are_strings(self, portfolio_server):
        _post(portfolio_server, "/demo/run?ticks=5")
        data = _get(portfolio_server, "/demo/portfolio")
        for ap in data["agent_profiles"]:
            assert isinstance(ap["agent_id"], str)
            assert len(ap["agent_id"]) > 0

    def test_portfolio_after_run_win_rates_bounded(self, portfolio_server):
        _post(portfolio_server, "/demo/run?ticks=5")
        data = _get(portfolio_server, "/demo/portfolio")
        for ap in data["agent_profiles"]:
            assert 0.0 <= ap["win_rate"] <= 1.0

    def test_portfolio_after_run_consensus_stats_present(self, portfolio_server):
        _post(portfolio_server, "/demo/run?ticks=5")
        data = _get(portfolio_server, "/demo/portfolio")
        cs = data["consensus_stats"]
        assert "avg_agreement_rate" in cs
        assert "supermajority_hits" in cs
        assert "veto_count" in cs

    def test_portfolio_after_run_risk_metrics_present(self, portfolio_server):
        _post(portfolio_server, "/demo/run?ticks=5")
        data = _get(portfolio_server, "/demo/portfolio")
        rm = data["risk_metrics"]
        assert "max_drawdown" in rm
        assert "sharpe_estimate" in rm
        assert "volatility" in rm

    def test_portfolio_response_has_cors_header(self, portfolio_server):
        resp = urllib.request.urlopen(f"http://localhost:{portfolio_server}/demo/portfolio")
        assert resp.headers.get("Access-Control-Allow-Origin") == "*"

    def test_portfolio_response_is_json_content_type(self, portfolio_server):
        resp = urllib.request.urlopen(f"http://localhost:{portfolio_server}/demo/portfolio")
        assert "application/json" in resp.headers.get("Content-Type", "")

    def test_portfolio_shortpath(self, portfolio_server):
        data = _get(portfolio_server, "/portfolio")
        assert "agent_profiles" in data

    def test_portfolio_updates_after_second_run(self, portfolio_server):
        # First run
        _post(portfolio_server, "/demo/run?ticks=5&seed=10")
        data1 = _get(portfolio_server, "/demo/portfolio")
        # Second run with different seed
        _post(portfolio_server, "/demo/run?ticks=5&seed=99")
        data2 = _get(portfolio_server, "/demo/portfolio")
        # Both should be live
        assert data1["source"] == "live"
        assert data2["source"] == "live"

    def test_portfolio_info_endpoint_lists_portfolio(self, portfolio_server):
        data = _get(portfolio_server, "/demo/info")
        endpoints = data["endpoints"]
        assert any("portfolio" in k.lower() for k in endpoints)
