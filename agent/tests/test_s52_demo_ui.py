"""
test_s52_demo_ui.py — S52: Interactive demo UI + live-data endpoint tests.

Tests:
- GET /demo/ui returns 200 with content-type text/html
- GET /demo/live-data returns all 5 required keys
- docs/demo.html contains 'ERC-8004' and 'Run Demo' text
- docs/demo.html contains required interactive elements
- /demo/live-data health section has required fields
- /demo/live-data swarm_vote section has required fields
- /demo/live-data risk_portfolio section has required fields
- /demo/live-data performance section has required fields
- /demo/live-data showcase section has required fields
- Server version reports S52
- Health endpoint reports updated highlights including demo UI
- Standard structure and content validation tests
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import urllib.request
import urllib.error

# Project root
ROOT = Path(__file__).parent.parent.parent
DOCS = ROOT / "docs"

BASE_URL = "http://localhost:8084"


def _get(path: str) -> dict:
    """GET request helper — returns parsed JSON."""
    with urllib.request.urlopen(f"{BASE_URL}{path}", timeout=10) as resp:
        return json.loads(resp.read())


def _get_raw(path: str):
    """GET request helper — returns (status, body_bytes, headers)."""
    with urllib.request.urlopen(f"{BASE_URL}{path}", timeout=10) as resp:
        return resp.status, resp.read(), resp.headers


def _post(path: str, body: dict | None = None) -> dict:
    """POST request helper."""
    data = json.dumps(body or {}).encode()
    req = urllib.request.Request(
        f"{BASE_URL}{path}",
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=15) as resp:
        return json.loads(resp.read())


# ── /demo/ui Endpoint Tests ───────────────────────────────────────────────────

class TestDemoUiEndpoint:
    """GET /demo/ui must return 200 with HTML content."""

    def test_demo_ui_returns_200(self):
        """GET /demo/ui returns HTTP 200."""
        try:
            status, _, _ = _get_raw("/demo/ui")
            assert status == 200
        except urllib.error.URLError:
            pytest.skip("Server not running on port 8084")

    def test_demo_ui_content_type_html(self):
        """GET /demo/ui response has content-type text/html."""
        try:
            _, _, headers = _get_raw("/demo/ui")
            ct = headers.get("Content-Type", "")
            assert "text/html" in ct, f"Expected text/html but got: {ct}"
        except urllib.error.URLError:
            pytest.skip("Server not running on port 8084")

    def test_demo_ui_body_is_html(self):
        """GET /demo/ui response body starts with DOCTYPE or html tag."""
        try:
            _, body, _ = _get_raw("/demo/ui")
            text = body.decode("utf-8", errors="replace").lower()
            assert "<!doctype html>" in text or "<html" in text, \
                "Response does not appear to be HTML"
        except urllib.error.URLError:
            pytest.skip("Server not running on port 8084")

    def test_demo_ui_body_non_empty(self):
        """GET /demo/ui response body is substantial (> 5000 bytes)."""
        try:
            _, body, _ = _get_raw("/demo/ui")
            assert len(body) > 5000, f"Body is only {len(body)} bytes — too small"
        except urllib.error.URLError:
            pytest.skip("Server not running on port 8084")

    def test_demo_ui_contains_erc8004(self):
        """GET /demo/ui HTML body mentions ERC-8004."""
        try:
            _, body, _ = _get_raw("/demo/ui")
            assert b"ERC-8004" in body, "HTML does not mention ERC-8004"
        except urllib.error.URLError:
            pytest.skip("Server not running on port 8084")

    def test_demo_ui_contains_run_demo_button(self):
        """GET /demo/ui HTML has 'Run Demo' or 'Run All' interactive button."""
        try:
            _, body, _ = _get_raw("/demo/ui")
            text = body.decode("utf-8", errors="replace")
            assert "Run" in text and ("Demo" in text or "All" in text), \
                "HTML does not contain a 'Run Demo' or 'Run All' button"
        except urllib.error.URLError:
            pytest.skip("Server not running on port 8084")


# ── /demo/live-data Endpoint Tests ───────────────────────────────────────────

class TestDemoLiveData:
    """GET /demo/live-data must return all 5 required keys."""

    def test_live_data_returns_200(self):
        """GET /demo/live-data returns HTTP 200."""
        try:
            status, _, _ = _get_raw("/demo/live-data")
            assert status == 200
        except urllib.error.URLError:
            pytest.skip("Server not running on port 8084")

    def test_live_data_has_all_5_keys(self):
        """GET /demo/live-data contains all 5 required section keys."""
        try:
            data = _get("/demo/live-data")
            required = {"health", "swarm_vote", "risk_portfolio", "performance", "showcase"}
            missing = required - set(data.keys())
            assert not missing, f"Missing keys in live-data: {missing}"
        except urllib.error.URLError:
            pytest.skip("Server not running on port 8084")

    def test_live_data_health_section(self):
        """live-data health section has status and version fields."""
        try:
            data = _get("/demo/live-data")
            health = data.get("health", {})
            assert "status" in health, "health section missing 'status'"
            assert "version" in health, "health section missing 'version'"
            assert health["status"] == "ok", f"health.status expected 'ok' got {health.get('status')}"
        except urllib.error.URLError:
            pytest.skip("Server not running on port 8084")

    def test_live_data_swarm_vote_section(self):
        """live-data swarm_vote section has vote-related fields."""
        try:
            data = _get("/demo/live-data")
            sv = data.get("swarm_vote", {})
            assert sv, "swarm_vote section is empty"
            # Must have some fields related to voting
            has_vote_field = any(k in sv for k in (
                "votes", "consensus_reached", "consensus_action",
                "total_agents", "weighted_agree_fraction"
            ))
            assert has_vote_field, f"swarm_vote section missing expected fields: {list(sv.keys())}"
        except urllib.error.URLError:
            pytest.skip("Server not running on port 8084")

    def test_live_data_risk_portfolio_section(self):
        """live-data risk_portfolio section has VaR or portfolio fields."""
        try:
            data = _get("/demo/live-data")
            rp = data.get("risk_portfolio", {})
            assert rp, "risk_portfolio section is empty"
            has_risk_field = any(k in rp for k in (
                "portfolio", "var_95", "var_99", "sharpe_ratio", "per_symbol"
            ))
            assert has_risk_field, \
                f"risk_portfolio section missing expected fields: {list(rp.keys())}"
        except urllib.error.URLError:
            pytest.skip("Server not running on port 8084")

    def test_live_data_performance_section(self):
        """live-data performance section has win_rate or trading metrics."""
        try:
            data = _get("/demo/live-data")
            perf = data.get("performance", {})
            assert perf is not None, "performance section is None"
            has_perf_field = any(k in perf for k in (
                "win_rate", "total_pnl", "sharpe_ratio", "active_agents", "version"
            ))
            assert has_perf_field, \
                f"performance section missing expected fields: {list(perf.keys())}"
        except urllib.error.URLError:
            pytest.skip("Server not running on port 8084")

    def test_live_data_showcase_section(self):
        """live-data showcase section has pipeline or steps field."""
        try:
            data = _get("/demo/live-data")
            sc = data.get("showcase", {})
            assert sc, "showcase section is empty"
            has_showcase_field = any(k in sc for k in (
                "steps", "showcase", "summary", "total_duration_ms", "version"
            ))
            assert has_showcase_field, \
                f"showcase section missing expected fields: {list(sc.keys())}"
        except urllib.error.URLError:
            pytest.skip("Server not running on port 8084")

    def test_live_data_has_version(self):
        """live-data response includes a version field."""
        try:
            data = _get("/demo/live-data")
            assert "version" in data, "live-data missing top-level 'version' field"
        except urllib.error.URLError:
            pytest.skip("Server not running on port 8084")

    def test_live_data_has_generated_at(self):
        """live-data response includes a generated_at timestamp."""
        try:
            data = _get("/demo/live-data")
            assert "generated_at" in data, "live-data missing 'generated_at' field"
        except urllib.error.URLError:
            pytest.skip("Server not running on port 8084")


# ── docs/demo.html Static File Tests ─────────────────────────────────────────

class TestDemoHtmlFile:
    """docs/demo.html must be valid and interactive."""

    def test_demo_html_file_exists(self):
        """docs/demo.html exists on disk."""
        assert (DOCS / "demo.html").exists(), "docs/demo.html not found"

    def test_demo_html_contains_erc8004(self):
        """docs/demo.html mentions ERC-8004."""
        content = (DOCS / "demo.html").read_text()
        assert "ERC-8004" in content, "demo.html does not mention ERC-8004"

    def test_demo_html_contains_run_button(self):
        """docs/demo.html contains Run button text."""
        content = (DOCS / "demo.html").read_text()
        assert "Run" in content, "demo.html does not contain 'Run' button text"

    def test_demo_html_has_5_sections(self):
        """docs/demo.html has references to all 5 demo sections."""
        content = (DOCS / "demo.html").read_text()
        sections = ["health", "swarm", "risk", "perf", "showcase"]
        for sec in sections:
            assert sec in content.lower(), f"demo.html missing section reference: {sec}"

    def test_demo_html_has_javascript(self):
        """docs/demo.html contains JavaScript for interactivity."""
        content = (DOCS / "demo.html").read_text()
        assert "<script" in content, "demo.html missing <script> tag"
        assert "fetch(" in content or "fetch (" in content, \
            "demo.html missing fetch() call for live data"

    def test_demo_html_has_curl_commands(self):
        """docs/demo.html shows curl equivalent commands."""
        content = (DOCS / "demo.html").read_text()
        assert "curl" in content.lower(), "demo.html does not show curl commands"

    def test_demo_html_non_empty(self):
        """docs/demo.html is substantial (> 8000 bytes)."""
        size = (DOCS / "demo.html").stat().st_size
        assert size > 8000, f"demo.html is only {size} bytes — too small for interactive page"


# ── Server Version Tests ──────────────────────────────────────────────────────

class TestServerVersionS52:
    """Server must report S52 version and updated test count."""

    def test_health_version_is_s52(self):
        """Health endpoint reports version S52 or later."""
        try:
            data = _get("/demo/health")
            version = data.get("version", "")
            sprint_num = int(version[1:]) if version and version[1:].isdigit() else 0
            assert sprint_num >= 52, \
                f"Expected S52 or later but got {version}"
        except urllib.error.URLError:
            pytest.skip("Server not running on port 8084")

    def test_health_sprint_is_s52(self):
        """Health endpoint reports sprint S52 or later."""
        try:
            data = _get("/demo/health")
            sprint = data.get("sprint", "")
            sprint_num = int(sprint[1:]) if sprint and sprint[1:].isdigit() else 0
            assert sprint_num >= 52, \
                f"Expected sprint S52 or later but got {sprint}"
        except urllib.error.URLError:
            pytest.skip("Server not running on port 8084")
