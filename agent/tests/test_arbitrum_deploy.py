"""
test_arbitrum_deploy.py — Tests verifying Arbitrum Sepolia config and deploy script.

All tests are offline — no live RPC calls, no real deploys.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

# ─── Constants ────────────────────────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).parent.parent.parent
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
CONTRACTS_DIR = PROJECT_ROOT / "contracts"

DEPLOY_SCRIPT = SCRIPTS_DIR / "deploy-arbitrum-sepolia.sh"
HARDHAT_CONFIG = CONTRACTS_DIR / "hardhat.config.ts"

ARBITRUM_SEPOLIA_CHAIN_ID = 421614
ARBITRUM_SEPOLIA_RPC = "https://sepolia-rollup.arbitrum.io/rpc"
ARBITRUM_SEPOLIA_NETWORK_NAME = "arbitrumSepolia"


# ─── 1. Deploy script exists ──────────────────────────────────────────────────


class TestDeployScriptExists:
    def test_script_file_exists(self):
        assert DEPLOY_SCRIPT.exists(), f"Missing deploy script: {DEPLOY_SCRIPT}"

    def test_script_is_bash(self):
        content = DEPLOY_SCRIPT.read_text()
        assert "#!/usr/bin/env bash" in content or "#!/bin/bash" in content

    def test_script_is_executable(self):
        assert DEPLOY_SCRIPT.stat().st_mode & 0o111, "Script is not executable"

    def test_script_syntax_valid(self):
        result = subprocess.run(
            ["bash", "-n", str(DEPLOY_SCRIPT)],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, f"Bash syntax error: {result.stderr}"


# ─── 2. Deploy script content checks ─────────────────────────────────────────


class TestDeployScriptContent:
    @pytest.fixture(autouse=True)
    def _load(self):
        self.content = DEPLOY_SCRIPT.read_text()

    def test_contains_arbitrum_chain_id(self):
        assert str(ARBITRUM_SEPOLIA_CHAIN_ID) in self.content

    def test_contains_arbitrum_rpc_url(self):
        assert ARBITRUM_SEPOLIA_RPC in self.content

    def test_contains_arbitrum_network_name(self):
        assert ARBITRUM_SEPOLIA_NETWORK_NAME in self.content

    def test_requires_deployer_key(self):
        assert "DEPLOYER_KEY" in self.content

    def test_saves_deployment_json(self):
        assert "deployment-arbitrum-sepolia.json" in self.content

    def test_contains_explorer_url(self):
        assert "arbiscan.io" in self.content

    def test_fails_without_env(self, tmp_path):
        """Script should exit non-zero when .env is missing."""
        result = subprocess.run(
            ["bash", str(DEPLOY_SCRIPT)],
            capture_output=True,
            text=True,
            cwd=str(tmp_path),  # no .env here
        )
        assert result.returncode != 0, "Expected non-zero exit without .env"

    def test_logs_chain_id(self):
        assert "Chain ID" in self.content or "chain_id" in self.content


# ─── 3. Hardhat config includes Arbitrum Sepolia ──────────────────────────────


class TestHardhatConfig:
    @pytest.fixture(autouse=True)
    def _load(self):
        self.content = HARDHAT_CONFIG.read_text()

    def test_hardhat_config_exists(self):
        assert HARDHAT_CONFIG.exists()

    def test_arbitrum_sepolia_network_defined(self):
        assert ARBITRUM_SEPOLIA_NETWORK_NAME in self.content

    def test_arbitrum_chain_id_present(self):
        assert str(ARBITRUM_SEPOLIA_CHAIN_ID) in self.content

    def test_arbitrum_rpc_env_var(self):
        assert "ARBITRUM_SEPOLIA_RPC_URL" in self.content

    def test_arbitrum_default_rpc_fallback(self):
        assert ARBITRUM_SEPOLIA_RPC in self.content

    def test_arbiscan_explorer_configured(self):
        assert "arbiscan.io" in self.content

    def test_arbitrum_sepolia_api_key_slot(self):
        assert "arbitrumSepolia" in self.content and "apiKey" in self.content


# ─── 4. SUBMISSION.md mentions Arbitrum Sepolia ───────────────────────────────


class TestSubmissionMd:
    @pytest.fixture(autouse=True)
    def _load(self):
        self.submission = (PROJECT_ROOT / "SUBMISSION.md").read_text()

    def test_submission_mentions_arbitrum(self):
        assert "Arbitrum" in self.submission

    def test_submission_mentions_chain_id_421614(self):
        assert "421614" in self.submission

    def test_submission_deployed_on_line(self):
        assert "Deployed on: Arbitrum Sepolia (testnet)" in self.submission

    def test_submission_mentions_deploy_script(self):
        assert "deploy-arbitrum-sepolia.sh" in self.submission


# ─── 5. README mentions Arbitrum Sepolia ──────────────────────────────────────


class TestReadmeMd:
    @pytest.fixture(autouse=True)
    def _load(self):
        self.readme = (PROJECT_ROOT / "README.md").read_text()

    def test_readme_mentions_arbitrum(self):
        assert "Arbitrum" in self.readme

    def test_readme_mentions_arbitrum_sepolia(self):
        assert "Arbitrum Sepolia" in self.readme
