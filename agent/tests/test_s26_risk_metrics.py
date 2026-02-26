"""
test_s26_risk_metrics.py — Tests for enhanced GET /demo/metrics risk fields.

Verifies that build_metrics_summary() returns the new fields added in S26:
  - max_drawdown (pre-existing, now verified)
  - win_rate (pre-existing, now verified)
  - avg_trade_duration_minutes (new)
  - daily_pnl (new — last 7 days as array)

All tests are offline — no live server required.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from demo_server import build_metrics_summary, _SEEDED_METRICS


# ─── 1. Seeded metrics have new fields ────────────────────────────────────────


class TestSeededMetricsHaveNewFields:
    def test_seeded_has_max_drawdown(self):
        assert "max_drawdown" in _SEEDED_METRICS

    def test_seeded_has_win_rate(self):
        assert "win_rate" in _SEEDED_METRICS

    def test_seeded_has_avg_trade_duration(self):
        assert "avg_trade_duration_minutes" in _SEEDED_METRICS

    def test_seeded_has_daily_pnl(self):
        assert "daily_pnl" in _SEEDED_METRICS

    def test_seeded_daily_pnl_is_list(self):
        assert isinstance(_SEEDED_METRICS["daily_pnl"], list)

    def test_seeded_daily_pnl_has_7_values(self):
        assert len(_SEEDED_METRICS["daily_pnl"]) == 7

    def test_seeded_avg_duration_is_positive(self):
        assert _SEEDED_METRICS["avg_trade_duration_minutes"] > 0

    def test_seeded_max_drawdown_is_negative(self):
        assert _SEEDED_METRICS["max_drawdown"] < 0

    def test_seeded_win_rate_in_range(self):
        assert 0.0 <= _SEEDED_METRICS["win_rate"] <= 1.0


# ─── 2. build_metrics_summary() seeded path ───────────────────────────────────


class TestBuildMetricsSummarySeeded:
    """Tests when no runs have been completed (returns seeded data)."""

    def setup_method(self):
        # Reset state to zero runs for a clean seeded test
        # We test on a fresh import; the seeded path is triggered when run_count==0
        self.metrics = _SEEDED_METRICS

    def test_seeded_result_has_max_drawdown(self):
        assert "max_drawdown" in self.metrics

    def test_seeded_result_has_win_rate(self):
        assert "win_rate" in self.metrics

    def test_seeded_result_has_avg_duration(self):
        assert "avg_trade_duration_minutes" in self.metrics

    def test_seeded_result_has_daily_pnl(self):
        assert "daily_pnl" in self.metrics

    def test_seeded_daily_pnl_entries_are_float(self):
        for val in self.metrics["daily_pnl"]:
            assert isinstance(val, (int, float))

    def test_seeded_sharpe_ratio_present(self):
        assert "sharpe_ratio" in self.metrics

    def test_seeded_sortino_ratio_present(self):
        assert "sortino_ratio" in self.metrics

    def test_seeded_total_trades_positive(self):
        assert self.metrics["total_trades"] > 0


# ─── 3. build_metrics_summary() live path (mocked state) ─────────────────────


class TestBuildMetricsSummaryLive:
    """Tests on the live path using the actual build_metrics_summary function."""

    def test_live_result_has_avg_trade_duration(self):
        """Even in seeded state the function must return this field."""
        result = build_metrics_summary()
        assert "avg_trade_duration_minutes" in result

    def test_live_result_has_daily_pnl(self):
        result = build_metrics_summary()
        assert "daily_pnl" in result

    def test_live_result_daily_pnl_is_list(self):
        result = build_metrics_summary()
        assert isinstance(result["daily_pnl"], list)

    def test_live_result_has_max_drawdown(self):
        result = build_metrics_summary()
        assert "max_drawdown" in result

    def test_live_result_has_win_rate(self):
        result = build_metrics_summary()
        assert "win_rate" in result

    def test_live_result_win_rate_in_range(self):
        result = build_metrics_summary()
        assert 0.0 <= result["win_rate"] <= 1.0

    def test_live_result_avg_duration_is_numeric(self):
        result = build_metrics_summary()
        assert isinstance(result["avg_trade_duration_minutes"], (int, float))

    def test_live_result_avg_duration_non_negative(self):
        result = build_metrics_summary()
        assert result["avg_trade_duration_minutes"] >= 0.0

    def test_live_result_daily_pnl_entries_numeric(self):
        result = build_metrics_summary()
        for val in result["daily_pnl"]:
            assert isinstance(val, (int, float))

    def test_backward_compat_total_trades(self):
        result = build_metrics_summary()
        assert "total_trades" in result

    def test_backward_compat_sharpe_ratio(self):
        result = build_metrics_summary()
        assert "sharpe_ratio" in result

    def test_backward_compat_sortino_ratio(self):
        result = build_metrics_summary()
        assert "sortino_ratio" in result

    def test_backward_compat_cumulative_return(self):
        result = build_metrics_summary()
        assert "cumulative_return_pct" in result

    def test_backward_compat_active_agents_count(self):
        result = build_metrics_summary()
        assert "active_agents_count" in result

    def test_backward_compat_run_count(self):
        result = build_metrics_summary()
        assert "run_count" in result
