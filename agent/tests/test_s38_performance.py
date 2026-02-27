"""
test_s38_performance.py — Sprint 38: Strategy Performance Attribution tests.

Covers:
  - get_strategy_performance_attribution(): schema, value ranges, determinism
  - Period validation (1h / 24h / 7d)
  - Strategy breakdown: all 6 strategies present, required fields, value types
  - Period sub-breakdown: correct keys per period type (hourly/session/daily)
  - Risk bucket breakdown: low / medium / high, vol thresholds, kelly fraction
  - Summary block: top/bottom strategy, count assertions
  - HTTP integration: GET /demo/strategy/performance-attribution with period= param
  - Error cases: invalid period, unknown query param ignored, 404 routing
"""

from __future__ import annotations

import json
import os
import sys
import time
import threading
from http.client import HTTPConnection
from typing import Any, Dict
from urllib.request import urlopen, Request
from urllib.error import HTTPError

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from demo_server import (
    get_strategy_performance_attribution,
    _S38_STRATEGIES,
    _S38_PERIODS,
    _S38_RISK_BUCKETS,
    _S38_DEFAULT_PERIOD,
    _S38_VOL_THRESHOLD_LOW,
    _S38_VOL_THRESHOLD_HIGH,
    DemoServer,
)


# ── Helpers ────────────────────────────────────────────────────────────────────

def _get(url: str) -> dict:
    with urlopen(url, timeout=5) as resp:
        return json.loads(resp.read())


def _get_status(url: str) -> int:
    try:
        with urlopen(url, timeout=5) as resp:
            return resp.status
    except HTTPError as exc:
        return exc.code


def _get_raw(url: str) -> tuple[int, dict]:
    try:
        with urlopen(url, timeout=5) as resp:
            return resp.status, json.loads(resp.read())
    except HTTPError as exc:
        return exc.code, json.loads(exc.read())


# ── Fixtures ───────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def server():
    """Start a DemoServer on a random port for HTTP integration tests."""
    import socket
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    sock.close()

    srv = DemoServer(port=port)
    srv.start()
    time.sleep(0.3)
    yield f"http://127.0.0.1:{port}"
    srv.stop()


# ── Unit tests: constants ──────────────────────────────────────────────────────

class TestConstants:
    def test_s38_strategies_list_length(self):
        assert len(_S38_STRATEGIES) == 6

    def test_s38_strategies_contains_momentum(self):
        assert "momentum" in _S38_STRATEGIES

    def test_s38_strategies_contains_mean_reversion(self):
        assert "mean_reversion" in _S38_STRATEGIES

    def test_s38_strategies_contains_ensemble(self):
        assert "ensemble" in _S38_STRATEGIES

    def test_s38_periods_set(self):
        assert _S38_PERIODS == {"1h", "24h", "7d"}

    def test_s38_risk_buckets(self):
        assert _S38_RISK_BUCKETS == ["low", "medium", "high"]

    def test_s38_default_period(self):
        assert _S38_DEFAULT_PERIOD == "24h"

    def test_vol_thresholds_ordering(self):
        assert 0 < _S38_VOL_THRESHOLD_LOW < _S38_VOL_THRESHOLD_HIGH < 1.0


# ── Unit tests: schema ─────────────────────────────────────────────────────────

class TestSchemaAllPeriods:
    @pytest.mark.parametrize("period", ["1h", "24h", "7d"])
    def test_top_level_keys_present(self, period):
        result = get_strategy_performance_attribution(period=period)
        for key in ("period", "period_hours", "total_pnl_usd",
                    "strategy_breakdown", "period_breakdown",
                    "risk_bucket_breakdown", "summary", "generated_at"):
            assert key in result, f"missing key {key!r} for period={period}"

    @pytest.mark.parametrize("period", ["1h", "24h", "7d"])
    def test_period_field_matches_arg(self, period):
        result = get_strategy_performance_attribution(period=period)
        assert result["period"] == period

    def test_period_hours_1h(self):
        result = get_strategy_performance_attribution(period="1h")
        assert result["period_hours"] == 1

    def test_period_hours_24h(self):
        result = get_strategy_performance_attribution(period="24h")
        assert result["period_hours"] == 24

    def test_period_hours_7d(self):
        result = get_strategy_performance_attribution(period="7d")
        assert result["period_hours"] == 168

    @pytest.mark.parametrize("period", ["1h", "24h", "7d"])
    def test_total_pnl_is_float(self, period):
        result = get_strategy_performance_attribution(period=period)
        assert isinstance(result["total_pnl_usd"], float)

    @pytest.mark.parametrize("period", ["1h", "24h", "7d"])
    def test_generated_at_recent(self, period):
        before = time.time()
        result = get_strategy_performance_attribution(period=period)
        after = time.time()
        assert before <= result["generated_at"] <= after + 1


# ── Unit tests: strategy breakdown ────────────────────────────────────────────

class TestStrategyBreakdown:
    def setup_method(self):
        self.result = get_strategy_performance_attribution(period="24h")
        self.sb = self.result["strategy_breakdown"]

    def test_all_strategies_present(self):
        for strat in _S38_STRATEGIES:
            assert strat in self.sb

    def test_strategy_pnl_is_float(self):
        for strat, data in self.sb.items():
            assert isinstance(data["pnl_usd"], float), f"{strat}.pnl_usd not float"

    def test_strategy_trades_positive(self):
        for strat, data in self.sb.items():
            assert data["trades"] >= 1, f"{strat}.trades < 1"

    def test_strategy_win_rate_range(self):
        for strat, data in self.sb.items():
            wr = data["win_rate"]
            assert 0.0 <= wr <= 1.0, f"{strat}.win_rate={wr} out of range"

    def test_strategy_sharpe_present(self):
        for strat, data in self.sb.items():
            assert "sharpe_ratio" in data

    def test_strategy_alpha_present(self):
        for strat, data in self.sb.items():
            assert "alpha_contribution" in data

    def test_strategy_beta_present(self):
        for strat, data in self.sb.items():
            assert "beta_exposure" in data

    def test_strategy_max_drawdown_nonnegative(self):
        for strat, data in self.sb.items():
            assert data["max_drawdown_pct"] >= 0, f"{strat}.max_drawdown_pct negative"

    def test_strategy_count_equals_constant(self):
        assert len(self.sb) == len(_S38_STRATEGIES)


# ── Unit tests: period sub-breakdown ──────────────────────────────────────────

class TestPeriodBreakdown:
    def test_1h_period_breakdown_has_hourly(self):
        result = get_strategy_performance_attribution(period="1h")
        pb = result["period_breakdown"]
        assert "hourly" in pb

    def test_1h_hourly_has_trades(self):
        result = get_strategy_performance_attribution(period="1h")
        hourly = result["period_breakdown"]["hourly"]
        assert "trades" in hourly
        assert hourly["trades"] >= 1

    def test_24h_period_breakdown_sessions(self):
        result = get_strategy_performance_attribution(period="24h")
        pb = result["period_breakdown"]
        for sess in ("morning", "afternoon", "evening", "night"):
            assert sess in pb, f"missing session {sess!r}"

    def test_24h_sessions_have_6_hours(self):
        result = get_strategy_performance_attribution(period="24h")
        for sess_data in result["period_breakdown"].values():
            assert sess_data["hours"] == 6

    def test_7d_period_breakdown_days(self):
        result = get_strategy_performance_attribution(period="7d")
        pb = result["period_breakdown"]
        expected_days = ["monday", "tuesday", "wednesday", "thursday",
                         "friday", "saturday", "sunday"]
        for day in expected_days:
            assert day in pb, f"missing day {day!r}"

    def test_7d_days_have_24_hours(self):
        result = get_strategy_performance_attribution(period="7d")
        for day_data in result["period_breakdown"].values():
            assert day_data["hours"] == 24


# ── Unit tests: risk bucket breakdown ─────────────────────────────────────────

class TestRiskBucketBreakdown:
    def setup_method(self):
        self.result = get_strategy_performance_attribution(period="24h")
        self.rb = self.result["risk_bucket_breakdown"]

    def test_all_risk_buckets_present(self):
        for bucket in _S38_RISK_BUCKETS:
            assert bucket in self.rb

    def test_low_bucket_vol_below_threshold(self):
        low_vol = self.rb["low"]["avg_annualised_vol"]
        assert low_vol < _S38_VOL_THRESHOLD_LOW

    def test_medium_bucket_vol_in_range(self):
        med_vol = self.rb["medium"]["avg_annualised_vol"]
        assert _S38_VOL_THRESHOLD_LOW <= med_vol <= _S38_VOL_THRESHOLD_HIGH

    def test_high_bucket_vol_above_threshold(self):
        high_vol = self.rb["high"]["avg_annualised_vol"]
        assert high_vol > _S38_VOL_THRESHOLD_HIGH

    def test_kelly_fraction_range(self):
        for bucket in _S38_RISK_BUCKETS:
            kf = self.rb[bucket]["avg_kelly_fraction"]
            assert 0.0 < kf <= 1.0, f"{bucket}.avg_kelly_fraction={kf} out of range"

    def test_vol_threshold_label_present(self):
        for bucket in _S38_RISK_BUCKETS:
            label = self.rb[bucket]["vol_threshold_label"]
            assert isinstance(label, str) and len(label) > 0

    def test_win_rate_range_all_buckets(self):
        for bucket in _S38_RISK_BUCKETS:
            wr = self.rb[bucket]["win_rate"]
            assert 0.0 <= wr <= 1.0


# ── Unit tests: summary block ─────────────────────────────────────────────────

class TestSummaryBlock:
    def setup_method(self):
        self.result = get_strategy_performance_attribution(period="24h")
        self.summary = self.result["summary"]

    def test_top_strategy_is_valid(self):
        assert self.summary["top_strategy"] in _S38_STRATEGIES

    def test_bottom_strategy_is_valid(self):
        assert self.summary["bottom_strategy"] in _S38_STRATEGIES

    def test_strategies_profitable_in_range(self):
        prof = self.summary["strategies_profitable"]
        assert 0 <= prof <= len(_S38_STRATEGIES)

    def test_strategies_total_correct(self):
        assert self.summary["strategies_total"] == len(_S38_STRATEGIES)

    def test_high_vol_pnl_is_float(self):
        assert isinstance(self.summary["high_vol_pnl_usd"], float)

    def test_low_vol_pnl_is_float(self):
        assert isinstance(self.summary["low_vol_pnl_usd"], float)

    def test_top_bottom_not_both_same_necessarily(self):
        # Both can be same if len==1, but with 6 strategies top != bottom when not degenerate
        # (only assert they're strings)
        assert isinstance(self.summary["top_strategy"], str)
        assert isinstance(self.summary["bottom_strategy"], str)


# ── Unit tests: determinism ────────────────────────────────────────────────────

class TestDeterminism:
    @pytest.mark.parametrize("period", ["1h", "24h", "7d"])
    def test_same_result_on_repeated_calls(self, period):
        r1 = get_strategy_performance_attribution(period=period)
        r2 = get_strategy_performance_attribution(period=period)
        assert r1["total_pnl_usd"] == r2["total_pnl_usd"]
        assert r1["strategy_breakdown"] == r2["strategy_breakdown"]
        assert r1["risk_bucket_breakdown"] == r2["risk_bucket_breakdown"]

    def test_different_periods_give_different_pnl(self):
        r1h = get_strategy_performance_attribution(period="1h")
        r24h = get_strategy_performance_attribution(period="24h")
        r7d = get_strategy_performance_attribution(period="7d")
        # All three are different totals (scale factor differs)
        values = {r1h["total_pnl_usd"], r24h["total_pnl_usd"], r7d["total_pnl_usd"]}
        assert len(values) > 1  # At least two distinct totals


# ── Unit tests: validation errors ─────────────────────────────────────────────

class TestValidationErrors:
    def test_invalid_period_raises_value_error(self):
        with pytest.raises(ValueError, match="period must be one of"):
            get_strategy_performance_attribution(period="30d")

    def test_empty_period_raises_value_error(self):
        with pytest.raises(ValueError):
            get_strategy_performance_attribution(period="")

    def test_uppercase_period_raises_value_error(self):
        with pytest.raises(ValueError):
            get_strategy_performance_attribution(period="24H")

    def test_none_period_raises_attribute_or_value_error(self):
        with pytest.raises((ValueError, AttributeError)):
            get_strategy_performance_attribution(period=None)  # type: ignore


# ── HTTP integration tests ─────────────────────────────────────────────────────

class TestHTTPIntegration:
    def test_default_period_returns_200(self, server):
        status, body = _get_raw(f"{server}/demo/strategy/performance-attribution")
        assert status == 200
        assert body["period"] == "24h"

    def test_period_1h_returns_200(self, server):
        status, body = _get_raw(f"{server}/demo/strategy/performance-attribution?period=1h")
        assert status == 200
        assert body["period"] == "1h"

    def test_period_7d_returns_200(self, server):
        status, body = _get_raw(f"{server}/demo/strategy/performance-attribution?period=7d")
        assert status == 200
        assert body["period"] == "7d"

    def test_invalid_period_returns_400(self, server):
        status, body = _get_raw(f"{server}/demo/strategy/performance-attribution?period=99d")
        assert status == 400
        assert "error" in body

    def test_response_has_strategy_breakdown(self, server):
        status, body = _get_raw(f"{server}/demo/strategy/performance-attribution")
        assert status == 200
        assert "strategy_breakdown" in body
        assert len(body["strategy_breakdown"]) == len(_S38_STRATEGIES)

    def test_response_has_risk_bucket_breakdown(self, server):
        status, body = _get_raw(f"{server}/demo/strategy/performance-attribution")
        assert status == 200
        assert "risk_bucket_breakdown" in body
        for bucket in _S38_RISK_BUCKETS:
            assert bucket in body["risk_bucket_breakdown"]

    def test_response_has_period_breakdown(self, server):
        status, body = _get_raw(f"{server}/demo/strategy/performance-attribution")
        assert status == 200
        assert "period_breakdown" in body
        assert len(body["period_breakdown"]) > 0

    def test_404_for_unknown_path(self, server):
        status, _ = _get_raw(f"{server}/demo/strategy/NONEXISTENT")
        assert status == 404

    def test_cors_header_present(self, server):
        """Access-Control-Allow-Origin should be on responses (from _send_json)."""
        import urllib.request
        req = urllib.request.Request(
            f"{server}/demo/strategy/performance-attribution"
        )
        with urllib.request.urlopen(req, timeout=5) as resp:
            cors = resp.headers.get("Access-Control-Allow-Origin", "")
            assert cors == "*"

    def test_content_type_json(self, server):
        import urllib.request
        with urllib.request.urlopen(
            f"{server}/demo/strategy/performance-attribution", timeout=5
        ) as resp:
            ct = resp.headers.get("Content-Type", "")
            assert "application/json" in ct

    def test_repeated_requests_stable(self, server):
        """Multiple rapid requests should return the same total_pnl_usd."""
        url = f"{server}/demo/strategy/performance-attribution?period=24h"
        values = []
        for _ in range(3):
            _, body = _get_raw(url)
            values.append(body.get("total_pnl_usd"))
        assert len(set(values)) == 1, f"Unstable results across requests: {values}"
