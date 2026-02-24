"""
test_deploy_scripts.py — Tests for deploy_sepolia.sh and verify_deploy.py.

All tests mock the deployment flow — no live deploys.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch, PropertyMock

import pytest

# Add scripts dir to path for verify_deploy imports
SCRIPTS_DIR = Path(__file__).parent.parent.parent / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

from verify_deploy import load_deployment, verify_contract, print_status, main


# ─── Fixtures ─────────────────────────────────────────────────────────────────


VALID_DEPLOYMENT = {
    "contract_address": "0xAbCdEf1234567890AbCdEf1234567890AbCdEf12",
    "network": "sepolia",
    "deployed_at": "2026-02-24T21:00:00Z",
    "rpc_url": "https://sepolia.infura.io/v3/testkey",
}


@pytest.fixture
def deployment_file(tmp_path):
    f = tmp_path / "deployment.json"
    f.write_text(json.dumps(VALID_DEPLOYMENT))
    return str(f)


# ─── load_deployment ──────────────────────────────────────────────────────────


class TestLoadDeployment:
    def test_valid_file(self, deployment_file):
        data = load_deployment(deployment_file)
        assert data["contract_address"] == VALID_DEPLOYMENT["contract_address"]
        assert data["network"] == "sepolia"

    def test_file_not_found(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_deployment(str(tmp_path / "nonexistent.json"))

    def test_missing_contract_address(self, tmp_path):
        bad = {"network": "sepolia", "deployed_at": "2026-01-01T00:00:00Z"}
        f = tmp_path / "bad.json"
        f.write_text(json.dumps(bad))
        with pytest.raises(ValueError, match="contract_address"):
            load_deployment(str(f))

    def test_missing_network(self, tmp_path):
        bad = {
            "contract_address": "0xAbCdEf1234567890AbCdEf1234567890AbCdEf12",
            "deployed_at": "2026-01-01T00:00:00Z",
        }
        f = tmp_path / "bad.json"
        f.write_text(json.dumps(bad))
        with pytest.raises(ValueError, match="network"):
            load_deployment(str(f))

    def test_invalid_address_format(self, tmp_path):
        bad = {
            "contract_address": "not_an_address",
            "network": "sepolia",
            "deployed_at": "2026-01-01T00:00:00Z",
        }
        f = tmp_path / "bad.json"
        f.write_text(json.dumps(bad))
        with pytest.raises(ValueError, match="[Ii]nvalid"):
            load_deployment(str(f))

    def test_address_too_short(self, tmp_path):
        bad = {
            "contract_address": "0x1234",
            "network": "sepolia",
            "deployed_at": "2026-01-01T00:00:00Z",
        }
        f = tmp_path / "bad.json"
        f.write_text(json.dumps(bad))
        with pytest.raises(ValueError):
            load_deployment(str(f))

    def test_valid_address_42_chars(self, tmp_path):
        good = {
            "contract_address": "0x" + "a" * 40,
            "network": "sepolia",
            "deployed_at": "2026-01-01T00:00:00Z",
        }
        f = tmp_path / "good.json"
        f.write_text(json.dumps(good))
        data = load_deployment(str(f))
        assert data["contract_address"].startswith("0x")


# ─── verify_contract ─────────────────────────────────────────────────────────


class TestVerifyContract:
    def test_web3_not_installed(self):
        with patch.dict("sys.modules", {"web3": None}):
            result = verify_contract(
                VALID_DEPLOYMENT["contract_address"],
                VALID_DEPLOYMENT["rpc_url"],
                verbose=False,
            )
        assert result["passed"] is False
        assert any("web3" in e.lower() for e in result["errors"])

    def test_rpc_connection_failure(self):
        mock_web3 = MagicMock()
        mock_web3.is_connected.return_value = False
        mock_web3_class = MagicMock(return_value=mock_web3)

        with patch.dict("sys.modules", {"web3": MagicMock(Web3=mock_web3_class)}):
            result = verify_contract(
                VALID_DEPLOYMENT["contract_address"],
                VALID_DEPLOYMENT["rpc_url"],
                verbose=False,
            )
        assert result["passed"] is False
        assert any("connect" in e.lower() for e in result["errors"])

    def test_no_bytecode(self):
        mock_w3 = MagicMock()
        mock_w3.is_connected.return_value = True
        mock_w3.eth.get_code.return_value = b""

        mock_web3_mod = MagicMock()
        mock_web3_mod.Web3.return_value = mock_w3
        mock_web3_mod.Web3.HTTPProvider = MagicMock()
        mock_web3_mod.Web3.to_checksum_address = MagicMock(
            return_value=VALID_DEPLOYMENT["contract_address"]
        )

        with patch.dict("sys.modules", {"web3": mock_web3_mod}):
            result = verify_contract(
                VALID_DEPLOYMENT["contract_address"],
                VALID_DEPLOYMENT["rpc_url"],
                verbose=False,
            )
        assert result["passed"] is False
        assert result["checks"].get("has_bytecode") is False

    def test_successful_verification(self):
        mock_contract = MagicMock()
        mock_contract.functions.feedbackCount.return_value.call.return_value = 0
        mock_contract.functions.getAggregateScore.return_value.call.return_value = (0, 0)

        mock_w3 = MagicMock()
        mock_w3.is_connected.return_value = True
        mock_w3.eth.get_code.return_value = b"\x60\x80"  # non-empty bytecode
        mock_w3.eth.contract.return_value = mock_contract

        mock_web3_mod = MagicMock()
        mock_web3_mod.Web3.return_value = mock_w3
        mock_web3_mod.Web3.HTTPProvider = MagicMock()
        mock_web3_mod.Web3.to_checksum_address = MagicMock(
            return_value=VALID_DEPLOYMENT["contract_address"]
        )

        with patch.dict("sys.modules", {"web3": mock_web3_mod}):
            result = verify_contract(
                VALID_DEPLOYMENT["contract_address"],
                VALID_DEPLOYMENT["rpc_url"],
                verbose=False,
            )
        assert result["passed"] is True
        assert result["checks"]["rpc_connected"] is True
        assert result["checks"]["has_bytecode"] is True

    def test_result_dict_structure(self):
        with patch.dict("sys.modules", {"web3": None}):
            result = verify_contract("0x" + "a" * 40, "http://rpc", verbose=False)
        assert "contract_address" in result
        assert "rpc_url" in result
        assert "checks" in result
        assert "errors" in result
        assert "passed" in result


# ─── print_status ─────────────────────────────────────────────────────────────


class TestPrintStatus:
    def test_print_passed(self, capsys):
        result = {
            "contract_address": "0x" + "a" * 40,
            "rpc_url": "https://rpc",
            "passed": True,
            "checks": {"rpc_connected": True},
            "errors": [],
        }
        print_status(result)
        out = capsys.readouterr().out
        assert "PASSED" in out

    def test_print_failed(self, capsys):
        result = {
            "contract_address": "0x" + "a" * 40,
            "rpc_url": "https://rpc",
            "passed": False,
            "checks": {},
            "errors": ["Cannot connect"],
        }
        print_status(result)
        out = capsys.readouterr().out
        assert "FAILED" in out
        assert "Cannot connect" in out


# ─── main() CLI ───────────────────────────────────────────────────────────────


class TestMainCLI:
    def test_missing_deployment_file(self, tmp_path):
        exit_code = main(["--deployment", str(tmp_path / "nonexistent.json")])
        assert exit_code == 1

    def test_valid_deployment_web3_missing(self, deployment_file):
        with patch.dict("sys.modules", {"web3": None}):
            exit_code = main(["--deployment", deployment_file, "--quiet"])
        assert exit_code == 1   # web3 not installed → fails

    def test_no_rpc_url_in_deployment(self, tmp_path):
        no_rpc = {
            "contract_address": "0x" + "a" * 40,
            "network": "sepolia",
            "deployed_at": "2026-01-01T00:00:00Z",
        }
        f = tmp_path / "no_rpc.json"
        f.write_text(json.dumps(no_rpc))
        exit_code = main(["--deployment", str(f)])
        assert exit_code == 1


# ─── deploy_sepolia.sh syntax check ──────────────────────────────────────────


class TestDeployScript:
    SCRIPT_PATH = str(SCRIPTS_DIR / "deploy_sepolia.sh")

    def test_script_exists(self):
        assert Path(self.SCRIPT_PATH).exists()

    def test_script_executable(self):
        assert Path(self.SCRIPT_PATH).stat().st_mode & 0o111

    def test_script_is_bash(self):
        content = Path(self.SCRIPT_PATH).read_text()
        assert "#!/usr/bin/env bash" in content or "#!/bin/bash" in content

    def test_script_checks_env_file(self):
        content = Path(self.SCRIPT_PATH).read_text()
        assert ".env" in content

    def test_script_checks_sepolia_rpc_url(self):
        content = Path(self.SCRIPT_PATH).read_text()
        assert "SEPOLIA_RPC_URL" in content

    def test_script_checks_deployer_key(self):
        content = Path(self.SCRIPT_PATH).read_text()
        assert "DEPLOYER_KEY" in content

    def test_script_saves_deployment_json(self):
        content = Path(self.SCRIPT_PATH).read_text()
        assert "deployment.json" in content

    def test_script_syntax_check(self):
        result = subprocess.run(
            ["bash", "-n", self.SCRIPT_PATH],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, f"Syntax error: {result.stderr}"

    def test_script_fails_without_env(self, tmp_path):
        """Script should exit non-zero if .env missing."""
        result = subprocess.run(
            ["bash", self.SCRIPT_PATH],
            capture_output=True,
            text=True,
            cwd=str(tmp_path),   # no .env here
        )
        assert result.returncode != 0

    def test_verify_deploy_script_exists(self):
        assert (SCRIPTS_DIR / "verify_deploy.py").exists()
