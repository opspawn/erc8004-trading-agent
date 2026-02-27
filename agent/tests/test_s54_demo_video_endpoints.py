"""
S54 tests: Demo video endpoints validation.
Covers all demo HTML/JSON endpoints, judge dashboard content,
TA signals structure, and live-data sections.
"""
import json
import pytest
import urllib.request
import urllib.error

BASE_URL = "http://localhost:8084"


# ─────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────

def get(path, timeout=10):
    url = BASE_URL + path
    with urllib.request.urlopen(url, timeout=timeout) as resp:
        return resp.status, resp.read().decode("utf-8")


def get_json(path, timeout=10):
    status, body = get(path, timeout)
    return status, json.loads(body)


# ─────────────────────────────────────────────────────────────────
# Section 1: All demo endpoints return valid HTTP 200 (15 tests)
# ─────────────────────────────────────────────────────────────────

class TestDemoEndpointsReturnOK:
    def test_demo_health_returns_200(self):
        status, _ = get("/demo/health")
        assert status == 200

    def test_demo_judge_returns_200(self):
        status, _ = get("/demo/judge")
        assert status == 200

    def test_demo_ui_returns_200(self):
        status, _ = get("/demo/ui")
        assert status == 200

    def test_demo_live_data_returns_200(self):
        status, _ = get("/demo/live-data")
        assert status == 200

    def test_api_signals_latest_returns_200(self):
        status, _ = get("/api/v1/signals/latest")
        assert status == 200

    def test_api_swarm_performance_returns_200(self):
        status, _ = get("/api/v1/swarm/performance")
        assert status == 200

    def test_api_risk_portfolio_returns_200(self):
        status, _ = get("/api/v1/risk/portfolio")
        assert status == 200

    def test_demo_judge_returns_html(self):
        status, body = get("/demo/judge")
        assert status == 200
        assert "<!DOCTYPE html>" in body or "<html" in body.lower()

    def test_demo_ui_returns_html(self):
        status, body = get("/demo/ui")
        assert status == 200
        assert "<!DOCTYPE html>" in body or "<html" in body.lower()

    def test_demo_health_returns_json_content(self):
        status, body = get("/demo/health")
        assert status == 200
        data = json.loads(body)
        assert isinstance(data, dict)

    def test_demo_live_data_returns_json(self):
        status, body = get("/demo/live-data")
        assert status == 200
        data = json.loads(body)
        assert isinstance(data, dict)

    def test_api_signals_returns_json(self):
        status, body = get("/api/v1/signals/latest")
        assert status == 200
        data = json.loads(body)
        assert isinstance(data, dict)

    def test_swarm_performance_returns_json(self):
        status, body = get("/api/v1/swarm/performance")
        assert status == 200
        data = json.loads(body)
        assert isinstance(data, dict)

    def test_risk_portfolio_returns_json(self):
        status, body = get("/api/v1/risk/portfolio")
        assert status == 200
        data = json.loads(body)
        assert isinstance(data, dict)

    def test_demo_health_status_is_ok(self):
        _, data = get_json("/demo/health")
        assert data.get("status") == "ok"


# ─────────────────────────────────────────────────────────────────
# Section 2: /demo/judge HTML contains expected sections (10 tests)
# ─────────────────────────────────────────────────────────────────

class TestJudgeDashboardContent:
    @pytest.fixture(scope="class", autouse=True)
    def fetch_judge_html(self, request):
        _, body = get("/demo/judge")
        request.cls.html = body

    def test_judge_html_contains_quant_agent(self):
        assert "quant-1" in self.html

    def test_judge_html_contains_rsi(self):
        assert "RSI" in self.html

    def test_judge_html_contains_macd(self):
        assert "MACD" in self.html

    def test_judge_html_contains_var(self):
        assert "VaR" in self.html

    def test_judge_html_contains_btc_usd(self):
        assert "BTC-USD" in self.html

    def test_judge_html_contains_erc8004_reference(self):
        assert "ERC-8004" in self.html

    def test_judge_html_contains_swarm_section(self):
        # Swarm or agent vote section
        assert "swarm" in self.html.lower() or "vote" in self.html.lower() or "agent" in self.html.lower()

    def test_judge_html_contains_performance_data(self):
        # Should contain some performance metrics
        assert any(kw in self.html for kw in ["PnL", "pnl", "performance", "Performance", "sharpe", "Sharpe"])

    def test_judge_html_contains_signals_section(self):
        # TA signals section
        assert "signal" in self.html.lower() or "Signal" in self.html

    def test_judge_html_is_substantial(self):
        # A real dashboard should be at least 2KB of HTML
        assert len(self.html) > 2000


# ─────────────────────────────────────────────────────────────────
# Section 3: /api/v1/signals/latest structure (10 tests)
# ─────────────────────────────────────────────────────────────────

class TestSignalsEndpoint:
    @pytest.fixture(scope="class", autouse=True)
    def fetch_signals(self, request):
        _, data = get_json("/api/v1/signals/latest")
        request.cls.data = data

    def test_signals_has_signals_key(self):
        assert "signals" in self.data

    def test_signals_list_has_3_symbols(self):
        assert len(self.data["signals"]) == 3

    def test_signals_btc_present(self):
        symbols = [s["symbol"] for s in self.data["signals"]]
        assert "BTC-USD" in symbols

    def test_signals_eth_present(self):
        symbols = [s["symbol"] for s in self.data["signals"]]
        assert "ETH-USD" in symbols

    def test_signals_sol_present(self):
        symbols = [s["symbol"] for s in self.data["signals"]]
        assert "SOL-USD" in symbols

    def test_each_signal_has_rsi_signal(self):
        for sig in self.data["signals"]:
            assert "rsi_signal" in sig, f"Missing rsi_signal for {sig.get('symbol')}"

    def test_each_signal_has_macd_signal(self):
        for sig in self.data["signals"]:
            assert "macd_signal" in sig, f"Missing macd_signal for {sig.get('symbol')}"

    def test_each_signal_has_last_price(self):
        for sig in self.data["signals"]:
            assert "last_price" in sig
            assert sig["last_price"] > 0

    def test_rsi_signal_values_are_valid(self):
        valid = {"BUY", "SELL", "NEUTRAL"}
        for sig in self.data["signals"]:
            assert sig["rsi_signal"] in valid, f"Invalid rsi_signal: {sig['rsi_signal']}"

    def test_macd_signal_values_are_valid(self):
        # MACD signal can be: BUY, SELL, NEUTRAL, BULLISH, BEARISH
        valid = {"BUY", "SELL", "NEUTRAL", "BULLISH", "BEARISH"}
        for sig in self.data["signals"]:
            assert sig["macd_signal"] in valid, f"Invalid macd_signal: {sig['macd_signal']}"


# ─────────────────────────────────────────────────────────────────
# Section 4: /demo/ui HTML structure (5 tests)
# ─────────────────────────────────────────────────────────────────

class TestDemoUIHTML:
    @pytest.fixture(scope="class", autouse=True)
    def fetch_ui(self, request):
        _, body = get("/demo/ui")
        request.cls.html = body

    def test_demo_ui_is_html(self):
        assert "<html" in self.html.lower() or "<!DOCTYPE html>" in self.html

    def test_demo_ui_has_title(self):
        assert "<title>" in self.html

    def test_demo_ui_references_erc8004(self):
        assert "ERC-8004" in self.html or "erc8004" in self.html.lower()

    def test_demo_ui_has_script_or_form(self):
        # Interactive UI should have JS or form elements
        assert "<script" in self.html or "<form" in self.html or "<button" in self.html

    def test_demo_ui_is_substantial(self):
        assert len(self.html) > 3000


# ─────────────────────────────────────────────────────────────────
# Section 5: /demo/live-data all 5 data sections (10 tests)
# ─────────────────────────────────────────────────────────────────

class TestLiveDataEndpoint:
    @pytest.fixture(scope="class", autouse=True)
    def fetch_live_data(self, request):
        _, data = get_json("/demo/live-data")
        request.cls.data = data

    def test_live_data_has_health(self):
        assert "health" in self.data

    def test_live_data_has_swarm_vote(self):
        assert "swarm_vote" in self.data

    def test_live_data_has_risk_portfolio(self):
        assert "risk_portfolio" in self.data

    def test_live_data_has_performance(self):
        assert "performance" in self.data

    def test_live_data_has_showcase(self):
        assert "showcase" in self.data

    def test_live_data_health_is_ok(self):
        assert self.data["health"].get("status") == "ok"

    def test_live_data_swarm_vote_not_empty(self):
        sv = self.data["swarm_vote"]
        assert sv is not None
        assert isinstance(sv, dict)

    def test_live_data_risk_portfolio_not_empty(self):
        rp = self.data["risk_portfolio"]
        assert rp is not None
        assert isinstance(rp, dict)

    def test_live_data_has_version(self):
        assert "version" in self.data

    def test_live_data_version_is_string(self):
        assert isinstance(self.data["version"], str)
        assert len(self.data["version"]) > 0


# ─────────────────────────────────────────────────────────────────
# Section 6: Video asset existence tests (5 tests)
# ─────────────────────────────────────────────────────────────────

class TestDemoVideoAssets:
    import os
    BASE_PATH = "/home/agent/projects/erc8004-trading-agent"

    def test_demo_video_s54_exists(self):
        import os
        path = f"{self.BASE_PATH}/docs/demo-video-s54.mp4"
        assert os.path.exists(path), f"Video file missing: {path}"

    def test_demo_video_s54_is_not_empty(self):
        import os
        path = f"{self.BASE_PATH}/docs/demo-video-s54.mp4"
        size = os.path.getsize(path)
        assert size > 10000, f"Video file too small ({size} bytes)"

    def test_screenshots_dir_has_s54_files(self):
        import os
        d = f"{self.BASE_PATH}/docs/demo-screenshots"
        s54_files = [f for f in os.listdir(d) if f.startswith("s54-")]
        assert len(s54_files) >= 6, f"Expected >=6 screenshots, got {len(s54_files)}"

    def test_screenshot_01_judge_exists(self):
        import os
        path = f"{self.BASE_PATH}/docs/demo-screenshots/s54-01-judge-dashboard.png"
        assert os.path.exists(path)

    def test_screenshot_04_demo_ui_exists(self):
        import os
        path = f"{self.BASE_PATH}/docs/demo-screenshots/s54-04-demo-ui.png"
        assert os.path.exists(path)


# ─────────────────────────────────────────────────────────────────
# Section 7: Health endpoint S54 version checks (5 tests)
# ─────────────────────────────────────────────────────────────────

class TestHealthEndpointS54:
    @pytest.fixture(scope="class", autouse=True)
    def fetch_health(self, request):
        _, data = get_json("/demo/health")
        request.cls.data = data

    def test_health_version_is_s54(self):
        assert self.data.get("version") == "S54"

    def test_health_sprint_is_s54(self):
        assert self.data.get("sprint") == "S54"

    def test_health_test_count_above_6300(self):
        count = self.data.get("tests", 0) or self.data.get("test_count", 0)
        assert count >= 6300, f"Test count {count} below target 6300"

    def test_health_has_highlights(self):
        assert "highlights" in self.data
        assert len(self.data["highlights"]) >= 3

    def test_health_port_is_8084(self):
        assert self.data.get("port") == 8084
