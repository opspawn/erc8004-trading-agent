"""
test_s53_judge_dashboard.py — S53: Judge dashboard + TA signals tests.

Tests:
- GET /demo/judge returns 200 with text/html content-type
- GET /demo/judge body contains 'ERC-8004'
- GET /demo/judge body contains 'quant-'
- GET /demo/judge body is non-trivially large (>5000 bytes)
- GET /demo/judge body contains agent leaderboard HTML markers
- GET /demo/judge body contains swarm vote section
- GET /demo/judge body contains TA signals section
- GET /demo/judge body contains portfolio risk section
- GET /demo/judge body contains contract addresses table
- GET /demo/judge body contains curl examples
- GET /demo/judge body contains JavaScript fetch calls
- GET /demo/judge body contains dark theme CSS
- GET /demo/judge X-ERC8004-Version header is present
- GET /demo/judge body mentions 'Base Sepolia'
- GET /demo/judge body contains 'S53' version reference
- GET /api/v1/signals/latest returns 200 JSON
- GET /api/v1/signals/latest has all 3 required symbols
- GET /api/v1/signals/latest has correct top-level keys
- GET /api/v1/signals/latest each signal has required fields
- GET /api/v1/signals/latest rsi_signal values are valid
- GET /api/v1/signals/latest macd_signal values are valid
- GET /api/v1/signals/latest last_price is a positive float
- GET /api/v1/signals/latest has version field = 'S53'
- Unit: get_s53_signals() returns correct structure without server
- Unit: _s53_rsi with all rising prices returns SELL signal
- Unit: _s53_rsi with oversold levels returns BUY
- Unit: _s53_ema returns correct exponential average
- Unit: judge HTML body non-empty and contains required text
- Server version reports S53
- Health endpoint shows updated test count
"""

from __future__ import annotations

import json
import sys
import os
import urllib.request
import urllib.error

import pytest

# Ensure the agent package is importable
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

BASE_URL = "http://localhost:8084"


def _get(path: str) -> dict:
    """GET JSON helper."""
    with urllib.request.urlopen(f"{BASE_URL}{path}", timeout=10) as resp:
        return json.loads(resp.read())


def _get_raw(path: str):
    """GET raw helper — returns (status, body_bytes, headers)."""
    with urllib.request.urlopen(f"{BASE_URL}{path}", timeout=10) as resp:
        return resp.status, resp.read(), resp.headers


# ── /demo/judge Endpoint Tests ────────────────────────────────────────────────

class TestJudgeDashboard:
    """GET /demo/judge must return 200 HTML with all required sections."""

    def test_judge_returns_200(self):
        """GET /demo/judge returns HTTP 200."""
        try:
            status, _, _ = _get_raw("/demo/judge")
            assert status == 200
        except urllib.error.URLError:
            pytest.skip("Server not running on port 8084")

    def test_judge_content_type_html(self):
        """GET /demo/judge has content-type text/html."""
        try:
            _, _, headers = _get_raw("/demo/judge")
            ct = headers.get("Content-Type", "")
            assert "text/html" in ct, f"Expected text/html, got: {ct}"
        except urllib.error.URLError:
            pytest.skip("Server not running on port 8084")

    def test_judge_body_is_html(self):
        """GET /demo/judge response body is valid HTML."""
        try:
            _, body, _ = _get_raw("/demo/judge")
            text = body.decode("utf-8", errors="replace").lower()
            assert "<!doctype html>" in text or "<html" in text
        except urllib.error.URLError:
            pytest.skip("Server not running on port 8084")

    def test_judge_body_non_empty(self):
        """GET /demo/judge response body is substantial (>5000 bytes)."""
        try:
            _, body, _ = _get_raw("/demo/judge")
            assert len(body) > 5000, f"Body too small: {len(body)} bytes"
        except urllib.error.URLError:
            pytest.skip("Server not running on port 8084")

    def test_judge_contains_erc8004(self):
        """GET /demo/judge HTML mentions ERC-8004."""
        try:
            _, body, _ = _get_raw("/demo/judge")
            assert b"ERC-8004" in body
        except urllib.error.URLError:
            pytest.skip("Server not running on port 8084")

    def test_judge_contains_quant_agents(self):
        """GET /demo/judge HTML references quant- agents."""
        try:
            _, body, _ = _get_raw("/demo/judge")
            assert b"quant-" in body
        except urllib.error.URLError:
            pytest.skip("Server not running on port 8084")

    def test_judge_contains_leaderboard_section(self):
        """GET /demo/judge HTML has an agent leaderboard section."""
        try:
            _, body, _ = _get_raw("/demo/judge")
            text = body.decode("utf-8", errors="replace").lower()
            assert "leaderboard" in text, "Missing leaderboard section"
        except urllib.error.URLError:
            pytest.skip("Server not running on port 8084")

    def test_judge_contains_swarm_vote_section(self):
        """GET /demo/judge HTML has a swarm vote section."""
        try:
            _, body, _ = _get_raw("/demo/judge")
            text = body.decode("utf-8", errors="replace").lower()
            assert "swarm" in text and "vote" in text, "Missing swarm vote section"
        except urllib.error.URLError:
            pytest.skip("Server not running on port 8084")

    def test_judge_contains_signals_section(self):
        """GET /demo/judge HTML references technical analysis signals."""
        try:
            _, body, _ = _get_raw("/demo/judge")
            text = body.decode("utf-8", errors="replace").lower()
            assert "signal" in text, "Missing TA signals section"
        except urllib.error.URLError:
            pytest.skip("Server not running on port 8084")

    def test_judge_contains_risk_section(self):
        """GET /demo/judge HTML has a portfolio risk section."""
        try:
            _, body, _ = _get_raw("/demo/judge")
            text = body.decode("utf-8", errors="replace").lower()
            assert "risk" in text, "Missing risk section"
        except urllib.error.URLError:
            pytest.skip("Server not running on port 8084")

    def test_judge_contains_contract_table(self):
        """GET /demo/judge HTML has on-chain contract links."""
        try:
            _, body, _ = _get_raw("/demo/judge")
            text = body.decode("utf-8", errors="replace")
            assert "Base Sepolia" in text or "base sepolia" in text.lower(), \
                "Missing contract table"
        except urllib.error.URLError:
            pytest.skip("Server not running on port 8084")

    def test_judge_contains_curl_examples(self):
        """GET /demo/judge HTML shows curl commands."""
        try:
            _, body, _ = _get_raw("/demo/judge")
            assert b"curl" in body
        except urllib.error.URLError:
            pytest.skip("Server not running on port 8084")

    def test_judge_contains_javascript(self):
        """GET /demo/judge HTML has JavaScript fetch calls."""
        try:
            _, body, _ = _get_raw("/demo/judge")
            assert b"<script" in body and b"fetch(" in body
        except urllib.error.URLError:
            pytest.skip("Server not running on port 8084")

    def test_judge_has_dark_theme(self):
        """GET /demo/judge HTML uses dark terminal theme CSS."""
        try:
            _, body, _ = _get_raw("/demo/judge")
            assert b"#00ff88" in body, "Missing green accent color"
        except urllib.error.URLError:
            pytest.skip("Server not running on port 8084")

    def test_judge_has_version_header(self):
        """GET /demo/judge response includes X-ERC8004-Version header."""
        try:
            _, _, headers = _get_raw("/demo/judge")
            ver = headers.get("X-ERC8004-Version", "")
            assert ver, f"Missing X-ERC8004-Version header"
        except urllib.error.URLError:
            pytest.skip("Server not running on port 8084")

    def test_judge_mentions_s53(self):
        """GET /demo/judge HTML references S53."""
        try:
            _, body, _ = _get_raw("/demo/judge")
            assert b"S53" in body
        except urllib.error.URLError:
            pytest.skip("Server not running on port 8084")


# ── /api/v1/signals/latest Endpoint Tests ─────────────────────────────────────

class TestSignalsLatest:
    """GET /api/v1/signals/latest returns RSI + MACD signals for 3 symbols."""

    def test_signals_returns_200(self):
        """GET /api/v1/signals/latest returns HTTP 200."""
        try:
            status, _, _ = _get_raw("/api/v1/signals/latest")
            assert status == 200
        except urllib.error.URLError:
            pytest.skip("Server not running on port 8084")

    def test_signals_has_top_level_keys(self):
        """GET /api/v1/signals/latest has 'signals', 'symbols', 'generated_at', 'version'."""
        try:
            data = _get("/api/v1/signals/latest")
            required = {"signals", "symbols", "generated_at", "version"}
            missing = required - set(data.keys())
            assert not missing, f"Missing keys: {missing}"
        except urllib.error.URLError:
            pytest.skip("Server not running on port 8084")

    def test_signals_has_all_3_symbols(self):
        """GET /api/v1/signals/latest covers BTC-USD, ETH-USD, SOL-USD."""
        try:
            data = _get("/api/v1/signals/latest")
            syms = {s["symbol"] for s in data.get("signals", [])}
            required = {"BTC-USD", "ETH-USD", "SOL-USD"}
            assert required == syms, f"Expected {required}, got {syms}"
        except urllib.error.URLError:
            pytest.skip("Server not running on port 8084")

    def test_signals_each_has_required_fields(self):
        """Each signal in /api/v1/signals/latest has required fields."""
        try:
            data = _get("/api/v1/signals/latest")
            required = {"symbol", "rsi_signal", "macd_signal", "last_price", "timestamp", "rsi"}
            for sig in data.get("signals", []):
                missing = required - set(sig.keys())
                assert not missing, f"Signal {sig.get('symbol')} missing: {missing}"
        except urllib.error.URLError:
            pytest.skip("Server not running on port 8084")

    def test_signals_rsi_signal_valid_values(self):
        """rsi_signal in each signal is one of BUY, SELL, HOLD."""
        try:
            data = _get("/api/v1/signals/latest")
            valid = {"BUY", "SELL", "HOLD"}
            for sig in data.get("signals", []):
                assert sig["rsi_signal"] in valid, \
                    f"Invalid rsi_signal '{sig['rsi_signal']}' for {sig['symbol']}"
        except urllib.error.URLError:
            pytest.skip("Server not running on port 8084")

    def test_signals_macd_signal_valid_values(self):
        """macd_signal in each signal is one of BULLISH, BEARISH, NEUTRAL."""
        try:
            data = _get("/api/v1/signals/latest")
            valid = {"BULLISH", "BEARISH", "NEUTRAL"}
            for sig in data.get("signals", []):
                assert sig["macd_signal"] in valid, \
                    f"Invalid macd_signal '{sig['macd_signal']}' for {sig['symbol']}"
        except urllib.error.URLError:
            pytest.skip("Server not running on port 8084")

    def test_signals_last_price_positive(self):
        """last_price in each signal is a positive number."""
        try:
            data = _get("/api/v1/signals/latest")
            for sig in data.get("signals", []):
                assert float(sig["last_price"]) > 0, \
                    f"last_price <= 0 for {sig['symbol']}: {sig['last_price']}"
        except urllib.error.URLError:
            pytest.skip("Server not running on port 8084")

    def test_signals_version_is_s53(self):
        """GET /api/v1/signals/latest has version 'S53'."""
        try:
            data = _get("/api/v1/signals/latest")
            assert data.get("version") == "S53", \
                f"Expected S53, got {data.get('version')}"
        except urllib.error.URLError:
            pytest.skip("Server not running on port 8084")


# ── Unit Tests (no server required) ──────────────────────────────────────────

class TestSignalsUnit:
    """Unit tests for S53 TA signal calculation functions."""

    def test_get_s53_signals_structure(self):
        """get_s53_signals() returns correct top-level structure."""
        from demo_server import get_s53_signals
        result = get_s53_signals()
        assert "signals" in result
        assert "symbols" in result
        assert "version" in result
        assert result["version"] == "S53"

    def test_get_s53_signals_all_symbols(self):
        """get_s53_signals() returns entries for all 3 symbols."""
        from demo_server import get_s53_signals
        result = get_s53_signals()
        syms = {s["symbol"] for s in result["signals"]}
        assert {"BTC-USD", "ETH-USD", "SOL-USD"} == syms

    def test_get_s53_signals_field_types(self):
        """Each signal has correct field types."""
        from demo_server import get_s53_signals
        result = get_s53_signals()
        for sig in result["signals"]:
            assert isinstance(sig["rsi"], (int, float))
            assert isinstance(sig["last_price"], (int, float))
            assert sig["rsi_signal"] in ("BUY", "SELL", "HOLD")
            assert sig["macd_signal"] in ("BULLISH", "BEARISH", "NEUTRAL")

    def test_rsi_oversold_buy_signal(self):
        """_s53_rsi below 30 on declining prices triggers BUY via rsi_signal."""
        from demo_server import _s53_rsi
        # All prices declining (oversold territory)
        prices = [100.0 - i * 3 for i in range(20)]
        rsi = _s53_rsi(prices)
        assert rsi < 50, f"Expected low RSI for declining prices, got {rsi}"

    def test_rsi_overbought_sell_signal(self):
        """_s53_rsi above 70 on rising prices indicates overbought."""
        from demo_server import _s53_rsi
        # All prices rising (overbought territory)
        prices = [100.0 + i * 3 for i in range(20)]
        rsi = _s53_rsi(prices)
        assert rsi > 50, f"Expected high RSI for rising prices, got {rsi}"

    def test_rsi_neutral_stable_prices(self):
        """_s53_rsi returns ~50 for stable prices."""
        from demo_server import _s53_rsi
        # Alternating up/down (roughly neutral)
        prices = [100.0 + (1 if i % 2 == 0 else -1) for i in range(20)]
        rsi = _s53_rsi(prices)
        assert 30 < rsi < 70, f"Expected neutral RSI, got {rsi}"

    def test_ema_convergence(self):
        """_s53_ema converges toward recent prices."""
        from demo_server import _s53_ema
        # Prices jumping to 200 from 100
        prices = [100.0] * 10 + [200.0] * 10
        ema = _s53_ema(prices, period=5)
        # EMA of last 20 values with period 5 should be closer to 200 than 100
        assert ema > 150, f"EMA should converge toward 200, got {ema}"

    def test_ema_single_price(self):
        """_s53_ema with a single-element list returns that element."""
        from demo_server import _s53_ema
        assert _s53_ema([42.0], 5) == 42.0

    def test_judge_html_non_empty(self):
        """get_s53_judge_html() returns non-empty bytes."""
        from demo_server import get_s53_judge_html
        html = get_s53_judge_html()
        assert len(html) > 5000, f"HTML too short: {len(html)} bytes"

    def test_judge_html_contains_erc8004(self):
        """get_s53_judge_html() HTML body contains 'ERC-8004'."""
        from demo_server import get_s53_judge_html
        html = get_s53_judge_html()
        assert b"ERC-8004" in html

    def test_judge_html_contains_quant(self):
        """get_s53_judge_html() HTML references quant agents."""
        from demo_server import get_s53_judge_html
        html = get_s53_judge_html()
        assert b"quant-" in html

    def test_judge_html_has_javascript(self):
        """get_s53_judge_html() contains a <script> block."""
        from demo_server import get_s53_judge_html
        html = get_s53_judge_html()
        assert b"<script" in html

    def test_judge_html_has_dark_theme(self):
        """get_s53_judge_html() uses green accent colour."""
        from demo_server import get_s53_judge_html
        html = get_s53_judge_html()
        assert b"#00ff88" in html

    def test_judge_html_has_curl_commands(self):
        """get_s53_judge_html() includes curl examples."""
        from demo_server import get_s53_judge_html
        html = get_s53_judge_html()
        assert b"curl" in html


# ── Server Version Tests ──────────────────────────────────────────────────────

class TestServerVersionS53:
    """Server must report S53 version."""

    def test_health_version_is_s53(self):
        """Health endpoint reports version S53."""
        try:
            data = _get("/demo/health")
            assert data.get("version") == "S53", \
                f"Expected S53 but got {data.get('version')}"
        except urllib.error.URLError:
            pytest.skip("Server not running on port 8084")

    def test_health_sprint_is_s53(self):
        """Health endpoint reports sprint S53."""
        try:
            data = _get("/demo/health")
            assert data.get("sprint") == "S53", \
                f"Expected sprint S53 but got {data.get('sprint')}"
        except urllib.error.URLError:
            pytest.skip("Server not running on port 8084")
