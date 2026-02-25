"""
test_submission_validation.py — Tests validating that the ERC-8004 project
meets submission requirements: SUBMISSION.md, deployment URL, test coverage,
contract integration, and Credora/Surge feature completeness.

These tests serve as a pre-submission checklist ensuring all required
components are present and functional.
"""

from __future__ import annotations

import os
import re
import pytest
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ─── Project Root Detection ───────────────────────────────────────────────────

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
AGENT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# ─── Submission File Tests ────────────────────────────────────────────────────

class TestSubmissionFile:
    SUBMISSION_PATH = os.path.join(PROJECT_ROOT, "SUBMISSION.md")

    def test_submission_file_exists(self):
        assert os.path.exists(self.SUBMISSION_PATH), "SUBMISSION.md must exist"

    def test_submission_has_title(self):
        with open(self.SUBMISSION_PATH) as f:
            content = f.read()
        assert "ERC-8004" in content

    def test_submission_has_team_name(self):
        with open(self.SUBMISSION_PATH) as f:
            content = f.read()
        assert "OpSpawn" in content

    def test_submission_has_demo_url(self):
        with open(self.SUBMISSION_PATH) as f:
            content = f.read()
        assert "vercel.app" in content or "https://" in content

    def test_submission_has_description(self):
        with open(self.SUBMISSION_PATH) as f:
            content = f.read()
        # Should have substantial text (at least 200 chars beyond headers)
        non_header = [l for l in content.split("\n") if not l.startswith("#")]
        text = " ".join(non_header)
        assert len(text.strip()) >= 200

    def test_submission_references_erc8004_standard(self):
        with open(self.SUBMISSION_PATH) as f:
            content = f.read()
        assert "ERC-8004" in content

    def test_submission_mentions_credora(self):
        with open(self.SUBMISSION_PATH) as f:
            content = f.read()
        assert "Credora" in content or "credora" in content.lower()

    def test_submission_mentions_surge(self):
        with open(self.SUBMISSION_PATH) as f:
            content = f.read()
        assert "Surge" in content or "surge" in content.lower()

    def test_submission_has_tech_stack(self):
        with open(self.SUBMISSION_PATH) as f:
            content = f.read()
        # Should mention core tech
        assert "Python" in content or "Solidity" in content

    def test_submission_has_deadline_awareness(self):
        with open(self.SUBMISSION_PATH) as f:
            content = f.read()
        assert "2026" in content or "lablab" in content.lower()


# ─── Core Module Tests ────────────────────────────────────────────────────────

class TestCoreModulesExist:
    @pytest.mark.parametrize("module", [
        "risk_manager.py",
        "credora_client.py",
        "trader.py",
        "portfolio_manager.py",
        "strategy_runner.py",
        "reputation.py",
        "validator.py",
        "oracle_client.py",
        "surge_router.py",
        "backtester.py",
    ])
    def test_module_exists(self, module):
        path = os.path.join(AGENT_DIR, module)
        assert os.path.exists(path), f"{module} must exist"


# ─── ERC-8004 Feature Completeness ───────────────────────────────────────────

class TestERC8004Features:
    def test_credora_client_importable(self):
        from credora_client import CredoraClient, CredoraRatingTier, KELLY_MULTIPLIERS
        assert CredoraClient is not None

    def test_all_credit_tiers_defined(self):
        from credora_client import CredoraRatingTier
        required_tiers = ["AAA", "AA", "A", "BBB", "BB", "B", "CCC", "NR"]
        for tier in required_tiers:
            assert hasattr(CredoraRatingTier, tier)

    def test_kelly_multipliers_complete(self):
        from credora_client import CredoraRatingTier, KELLY_MULTIPLIERS
        for tier in CredoraRatingTier:
            assert tier in KELLY_MULTIPLIERS

    def test_agent_credit_history_importable(self):
        from credora_client import AgentCreditHistory
        h = AgentCreditHistory("test-agent")
        assert h is not None

    def test_risk_manager_credora_integration(self):
        from risk_manager import RiskManager
        from credora_client import CredoraClient
        client = CredoraClient(use_mock=True)
        rm = RiskManager(credora_client=client)
        ok, reason, adj = rm.validate_trade_with_credora(
            "YES", 5.0, 0.60, 100.0, "ethereum"
        )
        assert ok is True

    def test_surge_router_importable(self):
        from surge_router import SurgeRouterBase, MockSurgeRouter
        assert SurgeRouterBase is not None

    def test_oracle_client_importable(self):
        from oracle_client import OracleClient
        assert OracleClient is not None

    def test_reputation_importable(self):
        from reputation import ReputationLogger
        assert ReputationLogger is not None

    def test_validator_importable(self):
        from validator import TradeValidator
        assert TradeValidator is not None

    def test_backtester_importable(self):
        from backtester import Backtester
        assert Backtester is not None


# ─── Kelly Criterion Accuracy ────────────────────────────────────────────────

class TestKellyAccuracy:
    """Verify Kelly Criterion integration is mathematically correct."""

    def test_kelly_fraction_within_0_1(self):
        from credora_client import KELLY_MULTIPLIERS, CredoraRatingTier
        for tier, mult in KELLY_MULTIPLIERS.items():
            assert 0.0 < mult <= 1.0, f"Kelly mult for {tier} out of range: {mult}"

    def test_lower_tier_lower_fraction(self):
        from credora_client import KELLY_MULTIPLIERS, CredoraRatingTier
        assert KELLY_MULTIPLIERS[CredoraRatingTier.AAA] > KELLY_MULTIPLIERS[CredoraRatingTier.BBB]
        assert KELLY_MULTIPLIERS[CredoraRatingTier.BBB] > KELLY_MULTIPLIERS[CredoraRatingTier.B]

    def test_position_size_reduced_for_speculative(self):
        from risk_manager import RiskManager
        from credora_client import CredoraClient
        client = CredoraClient(use_mock=True)
        rm = RiskManager(credora_client=client)

        ok_aaa, _, size_aaa = rm.validate_trade_with_credora("YES", 10.0, 0.5, 200.0, "ethereum")
        ok_bb, _, size_bb = rm.validate_trade_with_credora("YES", 10.0, 0.5, 200.0, "gmx")

        assert ok_aaa and ok_bb
        assert size_aaa > size_bb

    def test_aaa_protocol_max_position(self):
        from risk_manager import RiskManager
        from credora_client import CredoraClient
        client = CredoraClient(use_mock=True)
        rm = RiskManager(credora_client=client)
        # ETH/AAA should not reduce size
        ok, _, adj = rm.validate_trade_with_credora("YES", 5.0, 0.5, 100.0, "ethereum")
        assert ok and abs(adj - 5.0) < 0.01


# ─── Dashboard & Deployment ───────────────────────────────────────────────────

class TestDashboardFiles:
    DASHBOARD_DIR = os.path.join(PROJECT_ROOT, "dashboard")

    def test_dashboard_dir_exists(self):
        assert os.path.isdir(self.DASHBOARD_DIR)

    def test_package_json_exists(self):
        path = os.path.join(PROJECT_ROOT, "package.json")
        assert os.path.exists(path)

    def test_vercel_json_exists(self):
        path = os.path.join(PROJECT_ROOT, "vercel.json")
        assert os.path.exists(path)

    def test_readme_exists(self):
        path = os.path.join(PROJECT_ROOT, "README.md")
        assert os.path.exists(path)
