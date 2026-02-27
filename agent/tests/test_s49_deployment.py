"""
test_s49_deployment.py — S49: deployment configuration tests.

Tests:
- demo_server.py reads PORT environment variable
- render.yaml exists with correct structure
- Procfile exists
- docs/DEPLOYMENT.md exists
- scripts/record_demo.py exists
"""

import os
import sys
from pathlib import Path
import pytest

# Project root (two levels up from this test file: agent/tests/ -> agent/ -> root)
ROOT = Path(__file__).parent.parent.parent


class TestPortEnvVar:
    """demo_server.py must respect the PORT environment variable."""

    def test_demo_server_respects_port_env_var(self):
        """Verify PORT env var is read in demo_server.py."""
        demo_server = ROOT / "agent" / "demo_server.py"
        assert demo_server.exists(), "demo_server.py not found"
        content = demo_server.read_text()
        # Must use os.environ.get('PORT', ...) or os.environ.get("PORT", ...)
        assert 'os.environ.get' in content and 'PORT' in content, (
            "demo_server.py must read PORT from environment via os.environ.get"
        )

    def test_demo_server_port_default_is_8084(self):
        """The default port in demo_server.py must be 8084."""
        demo_server = ROOT / "agent" / "demo_server.py"
        content = demo_server.read_text()
        assert '8084' in content, "Default port 8084 not found in demo_server.py"

    def test_demo_server_imports_os(self):
        """demo_server.py must import os (needed for os.environ)."""
        demo_server = ROOT / "agent" / "demo_server.py"
        content = demo_server.read_text()
        assert 'import os' in content, "demo_server.py must import os"


class TestRenderYaml:
    """render.yaml must exist and contain required fields."""

    def test_render_yaml_exists(self):
        """render.yaml must exist in project root."""
        render_yaml = ROOT / "render.yaml"
        assert render_yaml.exists(), "render.yaml not found in project root"

    def test_render_yaml_has_service(self):
        """render.yaml must define a web service."""
        render_yaml = ROOT / "render.yaml"
        content = render_yaml.read_text()
        assert 'type: web' in content, "render.yaml must define type: web"

    def test_render_yaml_has_start_command(self):
        """render.yaml must have a startCommand."""
        render_yaml = ROOT / "render.yaml"
        content = render_yaml.read_text()
        assert 'startCommand' in content, "render.yaml must have startCommand"

    def test_render_yaml_has_health_check(self):
        """render.yaml must have a healthCheckPath."""
        render_yaml = ROOT / "render.yaml"
        content = render_yaml.read_text()
        assert 'healthCheckPath' in content, "render.yaml must have healthCheckPath"

    def test_render_yaml_health_check_path(self):
        """render.yaml healthCheckPath must be /demo/health."""
        render_yaml = ROOT / "render.yaml"
        content = render_yaml.read_text()
        assert '/demo/health' in content, "healthCheckPath must be /demo/health"

    def test_render_yaml_has_port_env(self):
        """render.yaml must define PORT environment variable."""
        render_yaml = ROOT / "render.yaml"
        content = render_yaml.read_text()
        assert 'PORT' in content, "render.yaml must set PORT env var"

    def test_render_yaml_has_dev_mode_env(self):
        """render.yaml must define DEV_MODE environment variable."""
        render_yaml = ROOT / "render.yaml"
        content = render_yaml.read_text()
        assert 'DEV_MODE' in content, "render.yaml must set DEV_MODE env var"


class TestProcfile:
    """Procfile must exist with correct web process definition."""

    def test_procfile_exists(self):
        """Procfile must exist in project root."""
        procfile = ROOT / "Procfile"
        assert procfile.exists(), "Procfile not found in project root"

    def test_procfile_has_web_process(self):
        """Procfile must define a web process."""
        procfile = ROOT / "Procfile"
        content = procfile.read_text()
        assert content.startswith('web:'), "Procfile must start with 'web:'"

    def test_procfile_runs_demo_server(self):
        """Procfile web process must run demo_server.py."""
        procfile = ROOT / "Procfile"
        content = procfile.read_text()
        assert 'demo_server.py' in content, "Procfile must reference demo_server.py"


class TestDeploymentDocs:
    """Deployment documentation must exist and be complete."""

    def test_deployment_docs_exist(self):
        """docs/DEPLOYMENT.md must exist."""
        deployment_md = ROOT / "docs" / "DEPLOYMENT.md"
        assert deployment_md.exists(), "docs/DEPLOYMENT.md not found"

    def test_deployment_docs_has_render_section(self):
        """DEPLOYMENT.md must document Render.com deployment."""
        deployment_md = ROOT / "docs" / "DEPLOYMENT.md"
        content = deployment_md.read_text()
        assert 'Render' in content, "DEPLOYMENT.md must document Render.com deployment"

    def test_deployment_docs_has_env_vars(self):
        """DEPLOYMENT.md must document environment variables."""
        deployment_md = ROOT / "docs" / "DEPLOYMENT.md"
        content = deployment_md.read_text()
        assert 'PORT' in content, "DEPLOYMENT.md must document PORT env var"
        assert 'DEV_MODE' in content, "DEPLOYMENT.md must document DEV_MODE env var"

    def test_deployment_docs_has_key_endpoints(self):
        """DEPLOYMENT.md must list key API endpoints."""
        deployment_md = ROOT / "docs" / "DEPLOYMENT.md"
        content = deployment_md.read_text()
        assert '/demo/health' in content, "DEPLOYMENT.md must list /demo/health endpoint"
        assert '/api/v1/swarm/vote' in content, "DEPLOYMENT.md must list swarm/vote endpoint"


class TestRecordDemoScript:
    """scripts/record_demo.py must exist and be valid."""

    def test_record_demo_script_exists(self):
        """scripts/record_demo.py must exist."""
        script = ROOT / "scripts" / "record_demo.py"
        assert script.exists(), "scripts/record_demo.py not found"

    def test_record_demo_script_has_demo_steps(self):
        """record_demo.py must define DEMO_STEPS."""
        script = ROOT / "scripts" / "record_demo.py"
        content = script.read_text()
        assert 'DEMO_STEPS' in content, "record_demo.py must define DEMO_STEPS"

    def test_record_demo_script_covers_all_5_steps(self):
        """record_demo.py must have 5 demo steps."""
        script = ROOT / "scripts" / "record_demo.py"
        content = script.read_text()
        assert '/demo/health' in content, "Missing health check step"
        assert '/api/v1/swarm/vote' in content, "Missing swarm vote step"
        assert '/api/v1/risk/portfolio' in content, "Missing portfolio risk step"
        assert '/api/v1/performance/summary' in content, "Missing performance summary step"
        assert '/api/v1/demo/showcase' in content, "Missing showcase step"

    def test_record_demo_script_has_output_dir(self):
        """record_demo.py must write output to demo-screenshots dir."""
        script = ROOT / "scripts" / "record_demo.py"
        content = script.read_text()
        assert 'demo-screenshots' in content, "record_demo.py must write to demo-screenshots/"

    def test_record_demo_outputs_exist(self):
        """docs/demo-screenshots/ must contain output from a demo run."""
        screenshots_dir = ROOT / "docs" / "demo-screenshots"
        assert screenshots_dir.exists(), "docs/demo-screenshots/ not found"
        results_file = screenshots_dir / "demo-results.json"
        assert results_file.exists(), (
            "docs/demo-screenshots/demo-results.json not found — run scripts/record_demo.py first"
        )

    def test_record_demo_results_all_passed(self):
        """All 5 demo steps must have passed."""
        import json
        results_file = ROOT / "docs" / "demo-screenshots" / "demo-results.json"
        if not results_file.exists():
            pytest.skip("demo-results.json not found — run scripts/record_demo.py first")
        results = json.loads(results_file.read_text())
        failed = [r for r in results if r.get('status') != 'ok']
        assert not failed, f"Demo steps failed: {[r['step'] for r in failed]}"

    def test_record_demo_step_files_exist(self):
        """Individual step output files must exist in demo-screenshots/."""
        screenshots_dir = ROOT / "docs" / "demo-screenshots"
        expected = ["health.json", "swarm_vote.json", "risk_portfolio.json",
                    "performance.json", "showcase.json"]
        missing = [f for f in expected if not (screenshots_dir / f).exists()]
        assert not missing, f"Missing demo output files: {missing}"
