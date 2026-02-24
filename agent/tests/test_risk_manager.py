"""
Tests for risk_manager.py — 35 tests covering RiskManager validation,
drawdown tracking, position management, and edge cases.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import pytest
from unittest.mock import patch
from datetime import date

from risk_manager import (
    RiskManager,
    RiskConfig,
    OpenPosition,
)


# ─── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture
def rm():
    """Default RiskManager with standard settings."""
    return RiskManager(
        max_position_pct=0.10,
        max_exposure_pct=0.30,
        max_open_positions=5,
        stop_loss_pct=0.50,
        max_daily_drawdown_pct=0.05,
        max_leverage=3.0,
        min_portfolio_value=10.0,
    )


@pytest.fixture
def rm_tight():
    """RiskManager with tight limits for boundary testing."""
    return RiskManager(
        max_position_pct=0.05,
        max_exposure_pct=0.15,
        max_open_positions=2,
        stop_loss_pct=0.20,
        max_daily_drawdown_pct=0.02,
        max_leverage=2.0,
        min_portfolio_value=50.0,
    )


# ─── Basic Validation ─────────────────────────────────────────────────────────

class TestValidateTrade:

    def test_valid_trade_approved(self, rm):
        ok, reason = rm.validate_trade("YES", 5.0, 0.50, 100.0)
        assert ok is True
        assert reason == "OK"

    def test_valid_buy_no(self, rm):
        ok, reason = rm.validate_trade("NO", 3.0, 0.30, 100.0)
        assert ok is True

    def test_exceeds_max_position_pct(self, rm):
        # 10% of 100 = $10 max, size=15 is too big
        ok, reason = rm.validate_trade("YES", 15.0, 0.50, 100.0)
        assert ok is False
        assert "max position" in reason.lower()

    def test_exactly_at_max_position(self, rm):
        # Exactly at limit (10% of 100 = 10), should be allowed
        ok, reason = rm.validate_trade("YES", 10.0, 0.50, 100.0)
        assert ok is True

    def test_zero_size_rejected(self, rm):
        ok, reason = rm.validate_trade("YES", 0.0, 0.50, 100.0)
        assert ok is False
        assert "size" in reason.lower()

    def test_negative_size_rejected(self, rm):
        ok, reason = rm.validate_trade("YES", -5.0, 0.50, 100.0)
        assert ok is False

    def test_price_zero_rejected(self, rm):
        ok, reason = rm.validate_trade("YES", 5.0, 0.0, 100.0)
        assert ok is False
        assert "price" in reason.lower()

    def test_price_one_rejected(self, rm):
        ok, reason = rm.validate_trade("YES", 5.0, 1.0, 100.0)
        assert ok is False

    def test_price_negative_rejected(self, rm):
        ok, reason = rm.validate_trade("YES", 5.0, -0.1, 100.0)
        assert ok is False

    def test_portfolio_below_minimum(self, rm):
        # min_portfolio_value = 10.0, portfolio = 5.0
        ok, reason = rm.validate_trade("YES", 0.5, 0.50, 5.0)
        assert ok is False
        assert "minimum" in reason.lower() or "portfolio" in reason.lower()

    def test_exactly_at_min_portfolio(self, rm):
        ok, reason = rm.validate_trade("YES", 0.5, 0.50, 10.0)
        assert ok is True

    def test_high_leverage_rejected(self, rm):
        # size=8, price=0.01, portfolio=100 → leverage = 8/(0.01*100) = 8x > 3x limit
        ok, reason = rm.validate_trade("YES", 8.0, 0.01, 100.0)
        assert ok is False
        assert "leverage" in reason.lower()

    def test_moderate_leverage_allowed(self, rm):
        # size=5, price=0.50, portfolio=100 → leverage = 5/(0.50*100) = 0.1x, fine
        ok, reason = rm.validate_trade("YES", 5.0, 0.50, 100.0)
        assert ok is True


# ─── Open Position Limits ─────────────────────────────────────────────────────

class TestPositionLimits:

    def test_max_positions_reached(self, rm_tight):
        # rm_tight: max_open_positions=2
        rm_tight.open_position("m1", "YES", 2.0, 0.50)
        rm_tight.open_position("m2", "YES", 2.0, 0.50)

        ok, reason = rm_tight.validate_trade("YES", 2.0, 0.50, 100.0)
        assert ok is False
        assert "position" in reason.lower()

    def test_positions_cleared_allows_new(self, rm_tight):
        rm_tight.open_position("m1", "YES", 2.0, 0.50)
        rm_tight.open_position("m2", "YES", 2.0, 0.50)
        rm_tight.close_position("m1", pnl_usdc=0.5)

        ok, reason = rm_tight.validate_trade("YES", 2.0, 0.50, 100.0)
        assert ok is True

    def test_exposure_limit_blocks_trade(self, rm):
        # max_exposure_pct=0.30 of 100 = $30
        rm.open_position("m1", "YES", 10.0, 0.50)
        rm.open_position("m2", "YES", 10.0, 0.50)
        rm.open_position("m3", "YES", 10.0, 0.50)
        # Exposure = $30, new trade of $1 would put it at $31

        ok, reason = rm.validate_trade("YES", 1.0, 0.50, 100.0)
        assert ok is False
        assert "exposure" in reason.lower()

    def test_exposure_within_limit_allowed(self, rm):
        # max_exposure = $30
        rm.open_position("m1", "YES", 10.0, 0.50)  # exposure = $10
        ok, reason = rm.validate_trade("YES", 9.0, 0.50, 100.0)
        assert ok is True  # total would be $19 < $30

    def test_open_position_tracking(self, rm):
        rm.open_position("m1", "YES", 5.0, 0.65)
        summary = rm.get_risk_summary()
        assert summary["open_positions"] == 1
        assert summary["total_exposure_usdc"] == 5.0

    def test_close_position_removes_from_tracking(self, rm):
        rm.open_position("m1", "YES", 5.0, 0.65)
        rm.close_position("m1", pnl_usdc=1.0)
        summary = rm.get_risk_summary()
        assert summary["open_positions"] == 0
        assert summary["total_exposure_usdc"] == 0.0

    def test_close_nonexistent_position_returns_false(self, rm):
        result = rm.close_position("nonexistent-market")
        assert result is False


# ─── Drawdown Tracking ────────────────────────────────────────────────────────

class TestDrawdown:

    def test_no_drawdown_continues(self, rm):
        # No losses recorded, should be fine
        result = rm.check_drawdown(100.0)
        assert result is True
        assert rm._trading_halted is False

    def test_drawdown_triggers_halt(self, rm):
        # Record a $6 loss on $100 portfolio → 6% > 5% limit
        rm.record_trade_pnl(-6.0)
        result = rm.check_drawdown(100.0)
        assert result is False
        assert rm._trading_halted is True

    def test_halted_blocks_validate_trade(self, rm):
        rm.record_trade_pnl(-6.0)
        rm.check_drawdown(100.0)
        ok, reason = rm.validate_trade("YES", 5.0, 0.50, 100.0)
        assert ok is False
        assert "halted" in reason.lower()

    def test_drawdown_exactly_at_limit(self, rm):
        # Exactly 5% loss on $100 → triggers halt (>= limit)
        rm.record_trade_pnl(-5.0)
        result = rm.check_drawdown(100.0)
        assert result is False

    def test_drawdown_below_limit_ok(self, rm):
        # 4.9% loss → below 5% limit, fine
        rm.record_trade_pnl(-4.9)
        result = rm.check_drawdown(100.0)
        assert result is True

    def test_positive_pnl_no_drawdown(self, rm):
        # Gains don't trigger drawdown
        rm.record_trade_pnl(10.0)
        result = rm.check_drawdown(100.0)
        assert result is True

    def test_reset_halt_clears_state(self, rm):
        rm.record_trade_pnl(-6.0)
        rm.check_drawdown(100.0)
        assert rm._trading_halted is True
        rm.reset_halt()
        assert rm._trading_halted is False

    def test_validate_after_halt_reset(self, rm):
        rm.record_trade_pnl(-6.0)
        rm.check_drawdown(100.0)
        rm.reset_halt()
        ok, _ = rm.validate_trade("YES", 5.0, 0.50, 100.0)
        assert ok is True

    def test_zero_portfolio_value_skips_drawdown(self, rm):
        rm.record_trade_pnl(-6.0)
        result = rm.check_drawdown(0.0)
        # Should not crash or halt (no denominator)
        assert result is True


# ─── Stop-Loss ────────────────────────────────────────────────────────────────

class TestStopLoss:

    def test_stop_loss_triggered(self, rm):
        pos = OpenPosition(
            market_id="m1", side="YES", size_usdc=10.0, entry_price=0.60,
            current_pnl=-5.5  # 55% loss > 50% stop
        )
        assert rm.check_stop_loss(pos) is True

    def test_stop_loss_not_triggered(self, rm):
        pos = OpenPosition(
            market_id="m1", side="YES", size_usdc=10.0, entry_price=0.60,
            current_pnl=-4.0  # 40% loss < 50% stop
        )
        assert rm.check_stop_loss(pos) is False

    def test_stop_loss_exactly_at_limit(self, rm):
        # 50% loss on $10 = -$5
        pos = OpenPosition(
            market_id="m1", side="YES", size_usdc=10.0, entry_price=0.60,
            current_pnl=-5.0
        )
        assert rm.check_stop_loss(pos) is True  # >= limit

    def test_profitable_position_no_stop(self, rm):
        pos = OpenPosition(
            market_id="m1", side="YES", size_usdc=10.0, entry_price=0.60,
            current_pnl=3.0  # profit
        )
        assert rm.check_stop_loss(pos) is False

    def test_zero_size_no_stop(self, rm):
        pos = OpenPosition(
            market_id="m1", side="YES", size_usdc=0.0, entry_price=0.60,
            current_pnl=-1.0
        )
        assert rm.check_stop_loss(pos) is False


# ─── Risk Summary ─────────────────────────────────────────────────────────────

class TestRiskSummary:

    def test_summary_structure(self, rm):
        summary = rm.get_risk_summary()
        required_keys = [
            "trading_halted", "halt_reason", "daily_pnl_usdc",
            "open_positions", "total_exposure_usdc", "positions", "limits"
        ]
        for key in required_keys:
            assert key in summary, f"Missing key: {key}"

    def test_summary_limits_populated(self, rm):
        summary = rm.get_risk_summary()
        limits = summary["limits"]
        assert limits["max_position_pct"] == 0.10
        assert limits["max_daily_drawdown_pct"] == 0.05
        assert limits["max_open_positions"] == 5

    def test_summary_reflects_state(self, rm):
        rm.open_position("m1", "YES", 8.0, 0.55)
        rm.record_trade_pnl(-2.0)
        summary = rm.get_risk_summary()
        assert summary["open_positions"] == 1
        assert summary["total_exposure_usdc"] == 8.0
        assert summary["daily_pnl_usdc"] == -2.0

    def test_config_class_direct_init(self):
        config = RiskConfig(
            max_position_pct=0.15,
            max_open_positions=3,
        )
        rm = RiskManager(config=config)
        summary = rm.get_risk_summary()
        assert summary["limits"]["max_position_pct"] == 0.15
        assert summary["limits"]["max_open_positions"] == 3


# ─── Edge Cases and Integration ───────────────────────────────────────────────

class TestEdgeCases:

    def test_multiple_positions_exposure_sum(self, rm):
        rm.open_position("m1", "YES", 5.0, 0.60)
        rm.open_position("m2", "NO", 7.0, 0.40)
        summary = rm.get_risk_summary()
        assert summary["total_exposure_usdc"] == pytest.approx(12.0)

    def test_close_updates_daily_pnl(self, rm):
        rm.open_position("m1", "YES", 10.0, 0.55)
        rm.close_position("m1", pnl_usdc=2.5)
        summary = rm.get_risk_summary()
        assert summary["daily_pnl_usdc"] == pytest.approx(2.5)

    def test_close_with_negative_pnl(self, rm):
        rm.open_position("m1", "YES", 10.0, 0.55)
        rm.close_position("m1", pnl_usdc=-3.0)
        summary = rm.get_risk_summary()
        assert summary["daily_pnl_usdc"] == pytest.approx(-3.0)

    def test_position_details_in_summary(self, rm):
        rm.open_position("mkt-abc", "YES", 4.0, 0.45)
        summary = rm.get_risk_summary()
        pos = summary["positions"][0]
        assert pos["market_id"] == "mkt-abc"
        assert pos["side"] == "YES"
        assert pos["size_usdc"] == 4.0
        assert pos["entry_price"] == 0.45

    def test_reset_halt_allows_trading(self, rm):
        rm._trading_halted = True
        rm._halt_reason = "manual test"
        rm.reset_halt()
        ok, _ = rm.validate_trade("YES", 5.0, 0.50, 100.0)
        assert ok is True

    def test_validate_with_price_near_zero(self, rm):
        # Very low price may trigger leverage limit — should not crash
        ok, reason = rm.validate_trade("YES", 1.0, 0.001, 100.0)
        assert isinstance(ok, bool)
        assert isinstance(reason, str)

    def test_validate_with_price_near_one(self, rm):
        # Price near 1.0 (but < 1.0)
        ok, reason = rm.validate_trade("YES", 5.0, 0.999, 100.0)
        assert isinstance(ok, bool)

    def test_large_portfolio_allows_large_position(self, rm):
        # 10% of $10,000 = $1,000
        ok, reason = rm.validate_trade("YES", 500.0, 0.50, 10_000.0)
        assert ok is True

    def test_multiple_daily_pnl_records(self, rm):
        rm.record_trade_pnl(2.0)
        rm.record_trade_pnl(-1.0)
        rm.record_trade_pnl(0.5)
        summary = rm.get_risk_summary()
        assert summary["daily_pnl_usdc"] == pytest.approx(1.5)

    def test_halt_reason_in_summary(self, rm):
        rm._trading_halted = True
        rm._halt_reason = "test halt reason"
        summary = rm.get_risk_summary()
        assert summary["halt_reason"] == "test halt reason"
        assert summary["trading_halted"] is True

    def test_open_position_notional_value(self):
        pos = OpenPosition(market_id="m1", side="YES", size_usdc=10.0, entry_price=0.5)
        # notional = 10 / 0.5 = 20 shares
        assert pos.notional_value == pytest.approx(20.0)

    def test_open_position_low_entry_price(self):
        # Should not divide by zero
        pos = OpenPosition(market_id="m1", side="YES", size_usdc=5.0, entry_price=0.0)
        assert pos.notional_value > 0  # uses max(price, 0.01)

    def test_validate_with_five_positions_at_limit(self, rm):
        # Fill up to max, last one should fail
        for i in range(5):
            rm.open_position(f"m{i}", "YES", 2.0, 0.50)
        ok, reason = rm.validate_trade("YES", 2.0, 0.50, 100.0)
        assert ok is False
        assert "position" in reason.lower()

    def test_validate_buy_no_side(self, rm):
        ok, reason = rm.validate_trade("NO", 8.0, 0.70, 100.0)
        assert ok is True

    def test_drawdown_check_without_losses(self, rm):
        # No P&L recorded yet — should pass
        assert rm.check_drawdown(100.0) is True

    def test_risk_config_defaults(self):
        config = RiskConfig()
        assert 0 < config.max_position_pct <= 1.0
        assert 0 < config.max_exposure_pct <= 1.0
        assert config.max_open_positions > 0
        assert 0 < config.stop_loss_pct <= 1.0
        assert 0 < config.max_daily_drawdown_pct <= 1.0
        assert config.max_leverage > 0
