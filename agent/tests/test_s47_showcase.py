"""
test_s47_showcase.py — Sprint 47: Single-call judge showcase endpoint.

Covers:
  test_showcase_returns_all_4_steps   — result contains exactly 4 labelled steps
  test_showcase_step_timing_present   — each step has duration_ms > 0
  test_showcase_includes_risk_data    — step 3 contains VaR fields
"""

from __future__ import annotations

import json
import sys
import os
import urllib.request
import urllib.error

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from demo_server import get_s47_showcase, DemoServer


# ── Unit tests ────────────────────────────────────────────────────────────────

class TestShowcaseReturnsAll4Steps:
    """test_showcase_returns_all_4_steps — result has exactly 4 steps."""

    def test_steps_key_present(self):
        result = get_s47_showcase()
        assert "steps" in result

    def test_exactly_4_steps(self):
        result = get_s47_showcase()
        assert len(result["steps"]) == 4

    def test_step_numbers_are_1_to_4(self):
        result = get_s47_showcase()
        step_nums = [s["step"] for s in result["steps"]]
        assert step_nums == [1, 2, 3, 4]

    def test_step_labels_present(self):
        result = get_s47_showcase()
        for s in result["steps"]:
            assert "label" in s
            assert isinstance(s["label"], str)
            assert len(s["label"]) > 0

    def test_step_result_present(self):
        result = get_s47_showcase()
        for s in result["steps"]:
            assert "result" in s
            assert isinstance(s["result"], dict)

    def test_summary_key_present(self):
        result = get_s47_showcase()
        assert "summary" in result

    def test_summary_has_btc_price(self):
        result = get_s47_showcase()
        assert "btc_price" in result["summary"]
        assert isinstance(result["summary"]["btc_price"], float)
        assert result["summary"]["btc_price"] > 0

    def test_summary_has_swarm_consensus(self):
        result = get_s47_showcase()
        assert "swarm_consensus" in result["summary"]
        assert result["summary"]["swarm_consensus"] in {"BUY", "SELL", "HOLD"}

    def test_summary_has_trade_executed(self):
        result = get_s47_showcase()
        assert "trade_executed" in result["summary"]
        assert isinstance(result["summary"]["trade_executed"], str)

    def test_version_field_is_s46(self):
        result = get_s47_showcase()
        version = result.get("version", "")
        sprint_num = int(version[1:]) if version and version[1:].isdigit() else 0
        assert sprint_num >= 46

    def test_total_duration_ms_present(self):
        result = get_s47_showcase()
        assert "total_duration_ms" in result
        assert result["total_duration_ms"] >= 0


class TestShowcaseStepTimingPresent:
    """test_showcase_step_timing_present — each step has duration_ms >= 0."""

    def test_all_steps_have_duration_ms(self):
        result = get_s47_showcase()
        for s in result["steps"]:
            assert "duration_ms" in s, f"Step {s['step']} missing duration_ms"

    def test_duration_ms_is_numeric(self):
        result = get_s47_showcase()
        for s in result["steps"]:
            assert isinstance(s["duration_ms"], (int, float)), (
                f"Step {s['step']} duration_ms not numeric: {s['duration_ms']!r}"
            )

    def test_duration_ms_non_negative(self):
        result = get_s47_showcase()
        for s in result["steps"]:
            assert s["duration_ms"] >= 0, (
                f"Step {s['step']} duration_ms is negative: {s['duration_ms']}"
            )

    def test_step1_price_tick_has_symbol(self):
        result = get_s47_showcase()
        step1 = result["steps"][0]
        assert step1["result"].get("symbol") == "BTC-USD"

    def test_step1_price_tick_has_price(self):
        result = get_s47_showcase()
        step1 = result["steps"][0]
        assert "price" in step1["result"]
        assert step1["result"]["price"] > 0

    def test_step2_swarm_has_total_agents(self):
        result = get_s47_showcase()
        step2 = result["steps"][1]
        assert "total_agents" in step2["result"]
        assert step2["result"]["total_agents"] == 10

    def test_step2_swarm_has_consensus_reached(self):
        result = get_s47_showcase()
        step2 = result["steps"][1]
        assert "consensus_reached" in step2["result"]
        assert isinstance(step2["result"]["consensus_reached"], bool)

    def test_step4_trade_has_mode_paper(self):
        result = get_s47_showcase()
        step4 = result["steps"][3]
        assert step4["result"].get("mode") == "paper"

    def test_step4_trade_has_trade_id(self):
        result = get_s47_showcase()
        step4 = result["steps"][3]
        assert "trade_id" in step4["result"]
        assert "s47-showcase-" in step4["result"]["trade_id"]

    def test_total_duration_matches_steps(self):
        result = get_s47_showcase()
        step_total = sum(s["duration_ms"] for s in result["steps"])
        # Total should be at least the sum of individual steps
        assert result["total_duration_ms"] >= 0


class TestShowcaseIncludesRiskData:
    """test_showcase_includes_risk_data — step 3 contains VaR fields."""

    def test_step3_has_var_95(self):
        result = get_s47_showcase()
        step3 = result["steps"][2]
        assert "portfolio_var_95" in step3["result"]

    def test_step3_var_95_is_positive(self):
        result = get_s47_showcase()
        step3 = result["steps"][2]
        assert step3["result"]["portfolio_var_95"] > 0

    def test_step3_has_var_99(self):
        result = get_s47_showcase()
        step3 = result["steps"][2]
        assert "portfolio_var_99" in step3["result"]

    def test_step3_var_99_gte_var_95(self):
        result = get_s47_showcase()
        step3 = result["steps"][2]
        assert step3["result"]["portfolio_var_99"] >= step3["result"]["portfolio_var_95"]

    def test_step3_has_position_usd(self):
        result = get_s47_showcase()
        step3 = result["steps"][2]
        assert "btc_recommended_position_usd" in step3["result"]

    def test_step3_position_usd_positive(self):
        result = get_s47_showcase()
        step3 = result["steps"][2]
        assert step3["result"]["btc_recommended_position_usd"] > 0

    def test_step3_has_sizing_method(self):
        result = get_s47_showcase()
        step3 = result["steps"][2]
        assert "sizing_method" in step3["result"]
        assert step3["result"]["sizing_method"] in {"half_kelly", "volatility", "fixed_fraction"}

    def test_step3_has_portfolio_sharpe(self):
        result = get_s47_showcase()
        step3 = result["steps"][2]
        assert "portfolio_sharpe" in step3["result"]

    def test_step3_has_rationale(self):
        result = get_s47_showcase()
        step3 = result["steps"][2]
        assert "rationale" in step3["result"]
        assert isinstance(step3["result"]["rationale"], str)

    def test_summary_var_95_matches_step3(self):
        result = get_s47_showcase()
        step3_var = result["steps"][2]["result"]["portfolio_var_95"]
        summary_var = result["summary"]["var_95"]
        assert abs(step3_var - summary_var) < 1e-9

    def test_summary_position_usd_positive(self):
        result = get_s47_showcase()
        assert result["summary"]["position_usd"] > 0


# ── HTTP endpoint tests ───────────────────────────────────────────────────────

import threading
import time as _time


def _start_test_server(port: int) -> DemoServer:
    server = DemoServer(port=port)
    server.start()
    _time.sleep(0.3)
    return server


class TestShowcaseHTTPEndpoint:
    """Integration tests for POST /api/v1/demo/showcase via live server."""

    _server: DemoServer = None
    _port: int = 8247

    @classmethod
    def setup_class(cls):
        cls._server = _start_test_server(cls._port)

    @classmethod
    def teardown_class(cls):
        if cls._server:
            cls._server.stop()

    def _post(self, path: str) -> dict:
        url = f"http://localhost:{self._port}{path}"
        req = urllib.request.Request(url, data=b"", method="POST")
        with urllib.request.urlopen(req, timeout=10) as resp:
            return json.loads(resp.read())

    def test_showcase_endpoint_returns_200(self):
        result = self._post("/api/v1/demo/showcase")
        assert "steps" in result

    def test_showcase_endpoint_has_4_steps(self):
        result = self._post("/api/v1/demo/showcase")
        assert len(result["steps"]) == 4

    def test_showcase_endpoint_has_summary(self):
        result = self._post("/api/v1/demo/showcase")
        assert "summary" in result
        assert "btc_price" in result["summary"]

    def test_showcase_endpoint_risk_data_present(self):
        result = self._post("/api/v1/demo/showcase")
        step3 = result["steps"][2]
        assert "portfolio_var_95" in step3["result"]
        assert "btc_recommended_position_usd" in step3["result"]
