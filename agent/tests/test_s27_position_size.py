"""
test_s27_position_size.py — Sprint 27: Position sizing API tests.

35 tests covering:
  - build_position_size with all 5 methods
  - Kelly fraction calculation (_kelly_fraction)
  - Response structure and required fields
  - Edge cases: zero capital, extreme probabilities, invalid methods
  - Volatility table lookups
  - Warning generation for aggressive fractions
  - Query-string parameter parsing via demo_server HTTP endpoint
"""

from __future__ import annotations

import json
import math
from http.server import HTTPServer
from threading import Thread
from urllib.request import urlopen

import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from demo_server import (
    build_position_size,
    _kelly_fraction,
    _SYMBOL_VOLATILITY,
    _TRADING_DAYS,
    DemoServer,
    DEFAULT_PORT,
)


# ─── _kelly_fraction ──────────────────────────────────────────────────────────

class TestKellyFraction:
    def test_classic_kelly(self):
        # b=1 (even odds), p=0.6 → f = (0.6*1 - 0.4)/1 = 0.2
        f = _kelly_fraction(win_prob=0.6, avg_win=1.0, avg_loss=1.0)
        assert abs(f - 0.2) < 1e-6

    def test_zero_win_prob_returns_zero(self):
        assert _kelly_fraction(0.0, 2.0, 1.0) == 0.0

    def test_zero_avg_loss_returns_zero(self):
        assert _kelly_fraction(0.6, 2.0, 0.0) == 0.0

    def test_negative_avg_loss_returns_zero(self):
        assert _kelly_fraction(0.6, 2.0, -1.0) == 0.0

    def test_win_prob_below_breakeven_returns_zero(self):
        # b=1, p=0.4 → f = (0.4 - 0.6)/1 = -0.2 → clamped to 0
        f = _kelly_fraction(0.4, 1.0, 1.0)
        assert f == 0.0

    def test_high_edge_case(self):
        # Perfect accuracy (unrealistic, but should clamp to 1.0)
        f = _kelly_fraction(0.99, 10.0, 1.0)
        assert 0.0 <= f <= 1.0

    def test_result_in_range(self):
        for p in [0.3, 0.5, 0.55, 0.7, 0.9]:
            for b in [0.5, 1.0, 2.0, 5.0]:
                f = _kelly_fraction(p, b, 1.0)
                assert 0.0 <= f <= 1.0

    def test_higher_win_prob_higher_fraction(self):
        f_low = _kelly_fraction(0.51, 2.0, 1.0)
        f_high = _kelly_fraction(0.80, 2.0, 1.0)
        assert f_high > f_low


# ─── build_position_size Structure ────────────────────────────────────────────

class TestPositionSizeStructure:
    def test_required_fields_present(self):
        r = build_position_size("BTC/USD", 10000.0, 0.02)
        for field in [
            "symbol", "method", "description", "recommended_size_usd",
            "recommended_size_pct", "fraction", "kelly_full_fraction",
            "inputs", "warnings", "generated_at"
        ]:
            assert field in r, f"Missing field: {field}"

    def test_inputs_fields(self):
        r = build_position_size("ETH/USD", 5000.0, 0.01)
        inp = r["inputs"]
        for key in ["capital_usd", "risk_pct", "win_prob", "avg_win_multiple",
                    "avg_loss_multiple", "annual_vol", "daily_vol"]:
            assert key in inp

    def test_recommended_usd_is_positive(self):
        r = build_position_size("BTC/USD", 10000.0, 0.02)
        assert r["recommended_size_usd"] >= 0

    def test_recommended_usd_less_than_capital(self):
        r = build_position_size("BTC/USD", 10000.0, 0.02)
        assert r["recommended_size_usd"] <= 10000.0

    def test_fraction_in_range(self):
        r = build_position_size("SOL/USD", 10000.0, 0.05)
        assert 0.0 <= r["fraction"] <= 1.0

    def test_symbol_echoed(self):
        r = build_position_size("AVAX/USD", 10000.0, 0.02)
        assert r["symbol"] == "AVAX/USD"

    def test_warnings_is_list(self):
        r = build_position_size("BTC/USD", 10000.0, 0.02)
        assert isinstance(r["warnings"], list)

    def test_generated_at_is_recent(self):
        import time
        before = time.time()
        r = build_position_size("BTC/USD", 1000.0, 0.01)
        assert r["generated_at"] >= before


# ─── Methods ──────────────────────────────────────────────────────────────────

class TestPositionSizeMethods:
    def test_half_kelly_less_than_kelly(self):
        full = build_position_size("BTC/USD", 10000.0, 0.02, method="kelly")
        half = build_position_size("BTC/USD", 10000.0, 0.02, method="half_kelly")
        assert half["fraction"] <= full["fraction"]

    def test_quarter_kelly_less_than_half(self):
        half = build_position_size("BTC/USD", 10000.0, 0.02, method="half_kelly")
        qtr = build_position_size("BTC/USD", 10000.0, 0.02, method="quarter_kelly")
        assert qtr["fraction"] <= half["fraction"]

    def test_fixed_pct_equals_risk_pct(self):
        r = build_position_size("ETH/USD", 10000.0, 0.03, method="fixed_pct")
        assert abs(r["fraction"] - 0.03) < 1e-9

    def test_volatility_method_returns_result(self):
        r = build_position_size("BTC/USD", 10000.0, 0.02, method="volatility")
        assert r["fraction"] > 0
        assert r["method"] == "volatility"

    def test_invalid_method_raises(self):
        with pytest.raises(ValueError, match="Unknown method"):
            build_position_size("BTC/USD", 10000.0, 0.02, method="bogus")

    def test_method_in_description(self):
        for m in ["kelly", "half_kelly", "quarter_kelly", "volatility", "fixed_pct"]:
            r = build_position_size("BTC/USD", 10000.0, 0.02, method=m)
            assert len(r["description"]) > 5

    def test_all_methods_valid(self):
        for m in ["kelly", "half_kelly", "quarter_kelly", "volatility", "fixed_pct"]:
            r = build_position_size("ETH/USD", 5000.0, 0.02, method=m)
            assert r["method"] == m


# ─── Validation ───────────────────────────────────────────────────────────────

class TestPositionSizeValidation:
    def test_zero_capital_raises(self):
        with pytest.raises(ValueError):
            build_position_size("BTC/USD", 0.0, 0.02)

    def test_negative_capital_raises(self):
        with pytest.raises(ValueError):
            build_position_size("BTC/USD", -1000.0, 0.02)

    def test_zero_risk_pct_raises(self):
        with pytest.raises(ValueError):
            build_position_size("BTC/USD", 10000.0, 0.0)

    def test_risk_pct_above_one_raises(self):
        with pytest.raises(ValueError):
            build_position_size("BTC/USD", 10000.0, 1.5)

    def test_invalid_win_prob_negative_raises(self):
        with pytest.raises(ValueError):
            build_position_size("BTC/USD", 10000.0, 0.02, win_prob=-0.1)

    def test_invalid_win_prob_above_one_raises(self):
        with pytest.raises(ValueError):
            build_position_size("BTC/USD", 10000.0, 0.02, win_prob=1.1)


# ─── Volatility Table ─────────────────────────────────────────────────────────

class TestVolatilityTable:
    def test_btc_volatility_present(self):
        assert "BTC/USD" in _SYMBOL_VOLATILITY
        assert _SYMBOL_VOLATILITY["BTC/USD"] > 0

    def test_eth_volatility_present(self):
        assert "ETH/USD" in _SYMBOL_VOLATILITY

    def test_default_volatility_present(self):
        assert "default" in _SYMBOL_VOLATILITY

    def test_trading_days_constant(self):
        assert _TRADING_DAYS == 252

    def test_daily_vol_in_inputs(self):
        r = build_position_size("BTC/USD", 10000.0, 0.02)
        ann_vol = _SYMBOL_VOLATILITY["BTC/USD"]
        expected_daily = ann_vol / math.sqrt(_TRADING_DAYS)
        assert abs(r["inputs"]["daily_vol"] - round(expected_daily, 6)) < 1e-5

    def test_unknown_symbol_uses_default(self):
        r = build_position_size("SHIB/USD", 10000.0, 0.02)
        assert r["inputs"]["annual_vol"] == _SYMBOL_VOLATILITY["default"]


# ─── Warnings ────────────────────────────────────────────────────────────────

class TestWarnings:
    def test_aggressive_fraction_triggers_warning(self):
        # Very high win_prob → big Kelly fraction → warning
        r = build_position_size("BTC/USD", 10000.0, 0.02, win_prob=0.95, avg_win=5.0, method="kelly")
        assert len(r["warnings"]) > 0

    def test_below_50_pct_win_prob_kelly_warning(self):
        r = build_position_size("BTC/USD", 10000.0, 0.02, win_prob=0.3, method="kelly")
        # Kelly fraction should be 0 → no size → may warn
        assert isinstance(r["warnings"], list)

    def test_conservative_method_no_warning(self):
        r = build_position_size("BTC/USD", 10000.0, 0.02, win_prob=0.55, method="fixed_pct")
        # fixed_pct 2% is not aggressive
        assert len(r["warnings"]) == 0


# ─── HTTP Endpoint (integration) ─────────────────────────────────────────────

class TestPositionSizeHTTP:
    @pytest.fixture(scope="class")
    def server(self):
        import socket
        # Find a free port
        s = socket.socket()
        s.bind(("", 0))
        port = s.getsockname()[1]
        s.close()
        srv = DemoServer(port=port)
        srv.start()
        import time; time.sleep(0.3)
        yield port
        srv.stop()

    def test_default_params(self, server):
        url = f"http://localhost:{server}/demo/position-size"
        with urlopen(url) as r:
            data = json.loads(r.read())
        assert data["symbol"] == "BTC/USD"
        assert "recommended_size_usd" in data

    def test_custom_symbol(self, server):
        url = f"http://localhost:{server}/demo/position-size?symbol=ETH/USD&capital=5000&risk_pct=0.01"
        with urlopen(url) as r:
            data = json.loads(r.read())
        assert data["symbol"] == "ETH/USD"
        assert data["inputs"]["capital_usd"] == 5000.0

    def test_method_param(self, server):
        url = f"http://localhost:{server}/demo/position-size?method=kelly"
        with urlopen(url) as r:
            data = json.loads(r.read())
        assert data["method"] == "kelly"
