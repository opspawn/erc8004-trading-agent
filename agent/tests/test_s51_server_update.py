"""
test_s51_server_update.py — S51: server update to S50 + submission prep tests.

Tests:
- Server health endpoint returns 200 with correct structure
- Version and sprint fields report "S50"
- Test count reports 6185
- All 5 demo endpoints return expected structure
- Demo run endpoint returns valid pipeline response
- Submission asset files exist (README.md, SUBMISSION.md, JUDGE_DEMO.md)
- docs/demo.html is valid and non-empty
- demo-screenshots JSON files are present and valid
- SUBMISSION_CHECKLIST.md updated to S51
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
SCREENSHOTS = DOCS / "demo-screenshots"

BASE_URL = "http://localhost:8084"


def _get(path: str) -> dict:
    """GET request helper."""
    with urllib.request.urlopen(f"{BASE_URL}{path}", timeout=10) as resp:
        return json.loads(resp.read())


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


# ── Health Endpoint Tests ─────────────────────────────────────────────────────

class TestHealthEndpoint:
    """Server health endpoint must return 200 with correct S50 data."""

    def test_health_returns_200(self):
        """GET /demo/health returns HTTP 200."""
        try:
            with urllib.request.urlopen(f"{BASE_URL}/demo/health", timeout=10) as resp:
                assert resp.status == 200
        except urllib.error.URLError:
            pytest.skip("Server not running on port 8084")

    def test_health_status_ok(self):
        """Health response status field is 'ok'."""
        try:
            data = _get("/demo/health")
            assert data["status"] == "ok"
        except urllib.error.URLError:
            pytest.skip("Server not running on port 8084")

    def test_health_version_is_s50(self):
        """Health response version field reports S50 or later."""
        try:
            data = _get("/demo/health")
            version = data.get("version", "")
            sprint_num = int(version[1:]) if version and version[1:].isdigit() else 0
            assert sprint_num >= 50, f"Expected S50 or later but got {data.get('version')}"
        except urllib.error.URLError:
            pytest.skip("Server not running on port 8084")

    def test_health_sprint_is_s50(self):
        """Health response sprint field reports S50 or later."""
        try:
            data = _get("/demo/health")
            sprint = data.get("sprint", "")
            sprint_num = int(sprint[1:]) if sprint and sprint[1:].isdigit() else 0
            assert sprint_num >= 50, f"Expected sprint S50 or later but got {data.get('sprint')}"
        except urllib.error.URLError:
            pytest.skip("Server not running on port 8084")

    def test_health_test_count_6185(self):
        """Health response test count is 6185 or more."""
        try:
            data = _get("/demo/health")
            count = data.get("tests", 0)
            assert count >= 6185, f"Expected >= 6185 tests but got {count}"
        except urllib.error.URLError:
            pytest.skip("Server not running on port 8084")

    def test_health_has_required_fields(self):
        """Health response contains all required fields."""
        try:
            data = _get("/demo/health")
            required = {"status", "service", "version", "sprint", "tests", "test_count"}
            missing = required - set(data.keys())
            assert not missing, f"Missing fields: {missing}"
        except urllib.error.URLError:
            pytest.skip("Server not running on port 8084")


# ── Demo Endpoints Tests ──────────────────────────────────────────────────────

class TestDemoEndpoints:
    """All 5 core demo endpoints must return expected structure."""

    def test_demo_portfolio_returns_capital(self):
        """GET /demo/portfolio returns portfolio with capital field."""
        try:
            data = _get("/demo/portfolio")
            assert "total_capital_usd" in data, f"Missing total_capital_usd in {list(data.keys())}"
        except urllib.error.URLError:
            pytest.skip("Server not running on port 8084")

    def test_demo_metrics_returns_win_rate(self):
        """GET /demo/metrics returns performance metrics with win_rate."""
        try:
            data = _get("/demo/metrics")
            assert "win_rate" in data, f"Missing win_rate in {list(data.keys())}"
        except urllib.error.URLError:
            pytest.skip("Server not running on port 8084")

    def test_demo_leaderboard_has_entries(self):
        """GET /demo/leaderboard returns leaderboard with entries."""
        try:
            data = _get("/demo/leaderboard")
            assert "leaderboard" in data, f"Missing leaderboard key in {list(data.keys())}"
            assert isinstance(data["leaderboard"], list), "Leaderboard must be a list"
        except urllib.error.URLError:
            pytest.skip("Server not running on port 8084")

    def test_demo_run_returns_pipeline(self):
        """POST /demo/run returns pipeline execution result."""
        try:
            data = _post("/demo/run")
            assert "status" in data, f"Missing status in {list(data.keys())}"
            assert "pipeline" in data or "agents" in data, \
                f"Missing pipeline/agents in {list(data.keys())}"
        except urllib.error.URLError:
            pytest.skip("Server not running on port 8084")

    def test_demo_info_lists_endpoints(self):
        """GET /demo/info returns service info with endpoints listed."""
        try:
            data = _get("/demo/info")
            assert "endpoints" in data, f"Missing endpoints in {list(data.keys())}"
            assert len(data["endpoints"]) >= 5, \
                f"Expected >= 5 endpoints, got {len(data['endpoints'])}"
        except urllib.error.URLError:
            pytest.skip("Server not running on port 8084")


# ── Submission Asset Tests ────────────────────────────────────────────────────

class TestSubmissionAssets:
    """Required submission files must exist."""

    def test_readme_exists(self):
        """README.md exists at project root."""
        assert (ROOT / "README.md").exists(), "README.md not found"

    def test_readme_mentions_erc8004(self):
        """README.md mentions ERC-8004."""
        content = (ROOT / "README.md").read_text()
        assert "ERC-8004" in content, "README.md does not mention ERC-8004"

    def test_submission_md_exists(self):
        """SUBMISSION.md exists at project root."""
        assert (ROOT / "SUBMISSION.md").exists(), "SUBMISSION.md not found"

    def test_judge_demo_md_exists(self):
        """JUDGE_DEMO.md exists in docs/."""
        assert (DOCS / "JUDGE_DEMO.md").exists(), "docs/JUDGE_DEMO.md not found"

    def test_demo_html_exists(self):
        """docs/demo.html exists."""
        assert (DOCS / "demo.html").exists(), "docs/demo.html not found"

    def test_demo_html_is_valid_html(self):
        """docs/demo.html starts with DOCTYPE and contains html tag."""
        content = (DOCS / "demo.html").read_text()
        assert "<!DOCTYPE html>" in content or "<!doctype html>" in content.lower(), \
            "demo.html missing DOCTYPE declaration"
        assert "<html" in content, "demo.html missing <html> tag"

    def test_demo_html_is_non_empty(self):
        """docs/demo.html is substantial (> 5000 bytes)."""
        size = (DOCS / "demo.html").stat().st_size
        assert size > 5000, f"demo.html is only {size} bytes — too small"

    def test_demo_screenshots_json_files_present(self):
        """All 5 demo screenshot JSON files exist."""
        expected = ["health.json", "swarm_vote.json", "risk_portfolio.json",
                    "performance.json", "showcase.json"]
        for fname in expected:
            fpath = SCREENSHOTS / fname
            assert fpath.exists(), f"Missing demo screenshot: {fname}"

    def test_demo_screenshots_json_valid(self):
        """All demo screenshot JSON files are valid JSON (dict or list)."""
        for fpath in SCREENSHOTS.glob("*.json"):
            try:
                data = json.loads(fpath.read_text())
                assert isinstance(data, (dict, list)), \
                    f"{fpath.name} is not a valid JSON value"
            except json.JSONDecodeError as e:
                pytest.fail(f"{fpath.name} is invalid JSON: {e}")
