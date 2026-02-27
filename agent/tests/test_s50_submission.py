"""
test_s50_submission.py — S50: demo HTML page and submission polish tests.

Tests:
- SUBMISSION_CHECKLIST.md has correct test count (>= 6170)
- README.md mentions ERC-8004
- docs/demo.html exists
- docs/demo.html has step data and is substantial (> 5000 bytes)
- render.yaml has correct health check path
- All 5 demo screenshot JSON files are present
- build_demo_html.py script is runnable
"""

import json
import re
from pathlib import Path

import pytest

# Project root: agent/tests/ -> agent/ -> root
ROOT = Path(__file__).parent.parent.parent
DOCS = ROOT / "docs"
SCRIPTS = ROOT / "scripts"
SCREENSHOTS = DOCS / "demo-screenshots"


class TestSubmissionChecklist:
    """SUBMISSION_CHECKLIST.md must be up to date."""

    def test_submission_checklist_has_correct_test_count(self):
        """Test count in checklist must be >= 6170."""
        checklist = ROOT / "SUBMISSION_CHECKLIST.md"
        assert checklist.exists(), "SUBMISSION_CHECKLIST.md not found"
        content = checklist.read_text()
        # Find numeric test counts in the file
        matches = re.findall(r"(\d[\d,]+)\s*(?:passing\s*tests|tests)", content)
        counts = [int(m.replace(",", "")) for m in matches]
        assert counts, "No test count found in SUBMISSION_CHECKLIST.md"
        assert max(counts) >= 6170, (
            f"Test count in SUBMISSION_CHECKLIST.md is {max(counts)}, expected >= 6170"
        )

    def test_submission_checklist_mentions_s50(self):
        """Checklist must reference sprint S50."""
        checklist = ROOT / "SUBMISSION_CHECKLIST.md"
        content = checklist.read_text()
        assert "S50" in content, "SUBMISSION_CHECKLIST.md does not mention S50"


class TestReadme:
    """README.md must be judge-friendly and mention ERC-8004."""

    def test_readme_mentions_erc8004(self):
        """README must mention ERC-8004."""
        readme = ROOT / "README.md"
        assert readme.exists(), "README.md not found"
        content = readme.read_text()
        assert "ERC-8004" in content, "README.md does not mention ERC-8004"

    def test_readme_mentions_test_count(self):
        """README must mention at least 6000 tests."""
        readme = ROOT / "README.md"
        content = readme.read_text()
        matches = re.findall(r"(\d[\d,]+)\s*tests", content)
        counts = [int(m.replace(",", "")) for m in matches]
        assert counts, "No test count found in README.md"
        assert max(counts) >= 6000, f"README test count is {max(counts)}, expected >= 6000"

    def test_readme_has_demo_section(self):
        """README must have a demo section."""
        readme = ROOT / "README.md"
        content = readme.read_text()
        assert "demo" in content.lower() or "Demo" in content, "README has no demo section"


class TestDemoHtml:
    """docs/demo.html must exist and contain all step data."""

    def test_demo_html_exists(self):
        """docs/demo.html must exist."""
        demo_html = DOCS / "demo.html"
        assert demo_html.exists(), "docs/demo.html not found — run: python3 scripts/build_demo_html.py"

    def test_demo_html_has_step_data(self):
        """docs/demo.html must be > 5000 bytes and contain 'health'."""
        demo_html = DOCS / "demo.html"
        assert demo_html.exists(), "docs/demo.html not found"
        content = demo_html.read_text(encoding="utf-8")
        size = len(content.encode("utf-8"))
        assert size > 5000, f"docs/demo.html is only {size} bytes, expected > 5000"
        assert "health" in content.lower(), "docs/demo.html missing health step data"

    def test_demo_html_has_all_five_steps(self):
        """docs/demo.html must contain data for all 5 demo steps."""
        demo_html = DOCS / "demo.html"
        assert demo_html.exists(), "docs/demo.html not found"
        content = demo_html.read_text(encoding="utf-8")
        required_keywords = ["health", "swarm", "var_95", "performance", "showcase"]
        for kw in required_keywords:
            assert kw in content.lower() or kw in content, (
                f"docs/demo.html missing content for step: {kw}"
            )

    def test_demo_html_is_valid_html(self):
        """docs/demo.html must have DOCTYPE and html tags."""
        demo_html = DOCS / "demo.html"
        assert demo_html.exists(), "docs/demo.html not found"
        content = demo_html.read_text(encoding="utf-8")
        assert "<!DOCTYPE html>" in content, "docs/demo.html missing DOCTYPE"
        assert "<html" in content, "docs/demo.html missing <html> tag"
        assert "</html>" in content, "docs/demo.html missing </html> tag"


class TestRenderYaml:
    """render.yaml must have the correct health check path."""

    def test_render_yaml_has_health_check_path(self):
        """render.yaml must specify healthCheckPath: /demo/health."""
        render_yaml = ROOT / "render.yaml"
        assert render_yaml.exists(), "render.yaml not found"
        content = render_yaml.read_text()
        assert "healthCheckPath" in content, "render.yaml missing healthCheckPath"
        assert "/demo/health" in content, (
            "render.yaml healthCheckPath must be /demo/health"
        )


class TestDemoScreenshots:
    """All 5 demo screenshot JSON files must be present and valid."""

    REQUIRED_FILES = [
        "health.json",
        "swarm_vote.json",
        "risk_portfolio.json",
        "performance.json",
        "showcase.json",
    ]

    def test_demo_screenshots_all_five_present(self):
        """All 5 screenshot JSON files must exist."""
        assert SCREENSHOTS.exists(), f"demo-screenshots dir not found at {SCREENSHOTS}"
        for fname in self.REQUIRED_FILES:
            path = SCREENSHOTS / fname
            assert path.exists(), f"Missing demo screenshot: {fname}"

    def test_demo_screenshots_are_valid_json(self):
        """All screenshot files must be valid JSON."""
        for fname in self.REQUIRED_FILES:
            path = SCREENSHOTS / fname
            if not path.exists():
                pytest.skip(f"{fname} not found")
            with open(path) as f:
                data = json.load(f)
            assert isinstance(data, dict), f"{fname} root must be a JSON object"
            assert len(data) > 0, f"{fname} must not be empty"

    def test_health_json_has_status_ok(self):
        """health.json must have status=ok."""
        health_path = SCREENSHOTS / "health.json"
        if not health_path.exists():
            pytest.skip("health.json not found")
        with open(health_path) as f:
            data = json.load(f)
        assert data.get("status") == "ok", f"health.json status is {data.get('status')!r}"


class TestBuildScript:
    """scripts/build_demo_html.py must exist and be runnable."""

    def test_build_demo_html_script_exists(self):
        """scripts/build_demo_html.py must exist."""
        script = SCRIPTS / "build_demo_html.py"
        assert script.exists(), "scripts/build_demo_html.py not found"

    def test_build_demo_html_script_has_correct_output_path(self):
        """build_demo_html.py must reference docs/demo.html as output."""
        script = SCRIPTS / "build_demo_html.py"
        assert script.exists(), "scripts/build_demo_html.py not found"
        content = script.read_text()
        assert "demo.html" in content, "build_demo_html.py does not reference demo.html"
