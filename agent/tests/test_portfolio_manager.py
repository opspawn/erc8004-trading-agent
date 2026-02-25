"""
Tests for portfolio_manager.py — 55 tests covering:
  - kelly_fraction() and fractional_kelly() math
  - Position data class
  - TradeSignal data class
  - PortfolioStats data class
  - PortfolioManager.get_signal()
  - PortfolioManager position management (open/close)
  - Drawdown halt logic
  - Kelly sizing
  - Stats reporting
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import pytest
from unittest.mock import AsyncMock, MagicMock

from portfolio_manager import (
    PortfolioManager,
    Position,
    TradeSignal,
    PortfolioStats,
    kelly_fraction,
    fractional_kelly,
)


# ─── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture
def pm_basic():
    return PortfolioManager(capital_usdc=100.0)

@pytest.fixture
def pm_small():
    return PortfolioManager(capital_usdc=10.0)

@pytest.fixture
def market_data_bullish():
    return {
        "market_id": "mkt-eth-100k",
        "question": "Will ETH reach $5k?",
        "yes_price": 0.25,
        "volume": 50_000,
        "category": "crypto",
    }

@pytest.fixture
def market_data_neutral():
    return {
        "market_id": "mkt-neutral",
        "question": "Will CPI drop?",
        "yes_price": 0.50,
        "volume": 100_000,
        "category": "macro",
    }

@pytest.fixture
def mock_strategist_buy():
    s = MagicMock()
    decision = MagicMock()
    decision.action = "buy"
    decision.confidence = 0.75
    decision.reasoning = "Strong buy signal"
    s.decide = AsyncMock(return_value=decision)
    return s

@pytest.fixture
def mock_strategist_hold():
    s = MagicMock()
    decision = MagicMock()
    decision.action = "hold"
    decision.confidence = 0.50
    decision.reasoning = "No clear signal"
    s.decide = AsyncMock(return_value=decision)
    return s

@pytest.fixture
def mock_risk_manager_pass():
    rm = MagicMock()
    rm.check_oracle_risk.return_value = True
    return rm

@pytest.fixture
def mock_risk_manager_fail():
    rm = MagicMock()
    rm.check_oracle_risk.return_value = False
    return rm


# ─── kelly_fraction ───────────────────────────────────────────────────────────

class TestKellyFraction:

    def test_zero_win_prob_returns_zero(self):
        assert kelly_fraction(0.0) == 0.0

    def test_one_win_prob_returns_zero(self):
        # Edge case: certainty — Kelly formula isn't defined
        assert kelly_fraction(1.0) == 0.0

    def test_fifty_percent_even_odds_returns_zero(self):
        # f* = (0.5*1 - 0.5) / 1 = 0 — no edge
        result = kelly_fraction(0.5, win_return=1.0, loss_return=1.0)
        assert result == pytest.approx(0.0)

    def test_sixty_percent_even_odds(self):
        # f* = (0.6*1 - 0.4) / 1 = 0.2
        result = kelly_fraction(0.6)
        assert result == pytest.approx(0.2, abs=1e-6)

    def test_negative_edge_returns_zero(self):
        # 40% win prob — negative edge, should return 0
        result = kelly_fraction(0.4)
        assert result == 0.0

    def test_clamped_to_max_one(self):
        # Very high win prob shouldn't exceed 1.0
        result = kelly_fraction(0.99)
        assert 0.0 <= result <= 1.0

    def test_custom_odds(self):
        # 60% win prob, 2:1 odds: f* = (0.6*2 - 0.4) / 2 = 0.4
        result = kelly_fraction(0.6, win_return=2.0, loss_return=1.0)
        assert result == pytest.approx(0.4, abs=1e-6)

    def test_zero_loss_return_returns_zero(self):
        result = kelly_fraction(0.6, loss_return=0.0)
        assert result == 0.0


# ─── fractional_kelly ─────────────────────────────────────────────────────────

class TestFractionalKelly:

    def test_quarter_kelly_is_one_quarter_of_full(self):
        full = kelly_fraction(0.6)
        quarter = fractional_kelly(0.6, fraction=0.25)
        assert quarter == pytest.approx(full * 0.25, abs=1e-9)

    def test_half_kelly(self):
        full = kelly_fraction(0.65)
        half = fractional_kelly(0.65, fraction=0.5)
        assert half == pytest.approx(full * 0.5, abs=1e-9)

    def test_zero_probability_returns_zero(self):
        assert fractional_kelly(0.0) == 0.0

    def test_result_non_negative(self):
        # fractional kelly should never return negative values
        result = fractional_kelly(0.99, fraction=0.5)
        assert result >= 0.0


# ─── Position Dataclass ───────────────────────────────────────────────────────

class TestPosition:

    def test_position_creation(self):
        p = Position(
            position_id="pos-0001", market_id="mkt-1", side="BUY",
            size_usdc=10.0, entry_price=3000.0,
        )
        assert p.is_open is True
        assert p.pnl_usdc == 0.0
        assert p.closed_at is None

    def test_close_updates_fields(self):
        p = Position(
            position_id="pos-0001", market_id="mkt-1", side="BUY",
            size_usdc=10.0, entry_price=3000.0,
        )
        p.close(3300.0, 1.0)
        assert p.is_open is False
        assert p.exit_price == 3300.0
        assert p.pnl_usdc == 1.0
        assert p.closed_at is not None

    def test_position_to_dict(self):
        p = Position(
            position_id="pos-0001", market_id="mkt-1", side="BUY",
            size_usdc=10.0, entry_price=3000.0,
        )
        d = p.to_dict()
        assert d["position_id"] == "pos-0001"
        assert d["is_open"] is True
        assert d["pnl_usdc"] == 0.0


# ─── TradeSignal ──────────────────────────────────────────────────────────────

class TestTradeSignal:

    def test_to_dict(self):
        sig = TradeSignal(
            action="BUY", confidence=0.7, edge=0.2,
            kelly_fraction=0.05, recommended_size=5.0,
            reasoning="Good signal", oracle_price=3000.0, risk_score=0.1,
        )
        d = sig.to_dict()
        assert d["action"] == "BUY"
        assert d["confidence"] == 0.7
        assert d["blocked"] is False

    def test_blocked_signal(self):
        sig = TradeSignal(
            action="HOLD", confidence=0.0, edge=0.0,
            kelly_fraction=0.0, recommended_size=0.0,
            reasoning="Halted", oracle_price=3000.0, risk_score=0.0,
            blocked=True, block_reason="drawdown_halt",
        )
        d = sig.to_dict()
        assert d["blocked"] is True
        assert d["block_reason"] == "drawdown_halt"


# ─── PortfolioStats ───────────────────────────────────────────────────────────

class TestPortfolioStats:

    def test_win_rate_zero_when_no_closed_trades(self):
        stats = PortfolioStats(
            capital_usdc=100.0, peak_capital=100.0,
            current_drawdown_pct=0.0, open_positions=1, total_trades=1,
            winning_trades=0, total_pnl_usdc=0.0, drawdown_halted=False,
        )
        assert stats.win_rate == 0.0

    def test_win_rate_calculation(self):
        stats = PortfolioStats(
            capital_usdc=110.0, peak_capital=110.0,
            current_drawdown_pct=0.0, open_positions=0, total_trades=4,
            winning_trades=3, total_pnl_usdc=10.0, drawdown_halted=False,
        )
        assert stats.win_rate == pytest.approx(0.75)

    def test_to_dict(self):
        stats = PortfolioStats(
            capital_usdc=100.0, peak_capital=105.0,
            current_drawdown_pct=0.05, open_positions=2, total_trades=5,
            winning_trades=3, total_pnl_usdc=5.0, drawdown_halted=False,
        )
        d = stats.to_dict()
        assert d["capital_usdc"] == 100.0
        assert d["win_rate"] == pytest.approx(3 / 3)


# ─── PortfolioManager.get_signal ─────────────────────────────────────────────

class TestGetSignal:

    @pytest.mark.asyncio
    async def test_returns_hold_without_strategist_neutral_market(
        self, pm_basic, market_data_neutral
    ):
        signal = await pm_basic.get_signal(market_data_neutral, oracle_price=3000.0)
        assert signal.action == "HOLD"

    @pytest.mark.asyncio
    async def test_returns_buy_for_cheap_market_without_strategist(
        self, pm_basic, market_data_bullish
    ):
        signal = await pm_basic.get_signal(market_data_bullish, oracle_price=3000.0)
        assert signal.action == "BUY"

    @pytest.mark.asyncio
    async def test_with_strategist_buy_signal(
        self, pm_basic, mock_strategist_buy, market_data_neutral
    ):
        pm_basic.strategist = mock_strategist_buy
        signal = await pm_basic.get_signal(market_data_neutral, oracle_price=3000.0)
        assert signal.action == "BUY"
        assert signal.confidence == 0.75

    @pytest.mark.asyncio
    async def test_with_strategist_hold_signal(
        self, pm_basic, mock_strategist_hold, market_data_neutral
    ):
        pm_basic.strategist = mock_strategist_hold
        signal = await pm_basic.get_signal(market_data_neutral, oracle_price=3000.0)
        assert signal.action == "HOLD"

    @pytest.mark.asyncio
    async def test_oracle_risk_fail_blocks_signal(
        self, pm_basic, mock_strategist_buy, mock_risk_manager_fail, market_data_neutral
    ):
        pm_basic.strategist = mock_strategist_buy
        pm_basic.risk_manager = mock_risk_manager_fail
        signal = await pm_basic.get_signal(market_data_neutral, oracle_price=3000.0)
        assert signal.blocked is True
        assert signal.block_reason == "oracle_risk_check_failed"

    @pytest.mark.asyncio
    async def test_oracle_risk_pass_allows_signal(
        self, pm_basic, mock_strategist_buy, mock_risk_manager_pass, market_data_neutral
    ):
        pm_basic.strategist = mock_strategist_buy
        pm_basic.risk_manager = mock_risk_manager_pass
        signal = await pm_basic.get_signal(market_data_neutral, oracle_price=3000.0)
        assert not signal.blocked

    @pytest.mark.asyncio
    async def test_drawdown_halt_blocks_signal(self, pm_basic, market_data_bullish):
        pm_basic._drawdown_halted = True
        signal = await pm_basic.get_signal(market_data_bullish, oracle_price=3000.0)
        assert signal.blocked is True
        assert signal.block_reason == "drawdown_halt"

    @pytest.mark.asyncio
    async def test_recommended_size_positive_on_buy(
        self, pm_basic, market_data_bullish
    ):
        signal = await pm_basic.get_signal(market_data_bullish, oracle_price=3000.0)
        if signal.action == "BUY":
            assert signal.recommended_size > 0

    @pytest.mark.asyncio
    async def test_oracle_price_included_in_signal(
        self, pm_basic, market_data_bullish
    ):
        signal = await pm_basic.get_signal(market_data_bullish, oracle_price=5000.0)
        assert signal.oracle_price == 5000.0


# ─── Position Management ──────────────────────────────────────────────────────

class TestPositionManagement:

    def test_open_position_returns_position(self, pm_basic):
        pos = pm_basic.open_position("mkt-1", "BUY", 10.0, 3000.0)
        assert pos is not None
        assert pos.market_id == "mkt-1"
        assert pos.side == "BUY"

    def test_open_position_registers_in_tracking(self, pm_basic):
        pm_basic.open_position("mkt-1", "BUY", 10.0, 3000.0)
        stats = pm_basic.get_stats()
        assert stats.open_positions == 1

    def test_close_position_records_pnl(self, pm_basic):
        pm_basic.open_position("mkt-1", "BUY", 10.0, 3000.0)
        pm_basic.close_position("mkt-1", 3300.0, pnl_usdc=1.5)
        stats = pm_basic.get_stats()
        assert stats.total_pnl_usdc == pytest.approx(1.5)

    def test_close_position_updates_capital(self, pm_basic):
        pm_basic.open_position("mkt-1", "BUY", 10.0, 3000.0)
        pm_basic.close_position("mkt-1", 3300.0, pnl_usdc=2.0)
        assert pm_basic.capital_usdc == pytest.approx(102.0)

    def test_close_nonexistent_position_returns_none(self, pm_basic):
        result = pm_basic.close_position("nonexistent", 3000.0, 0.0)
        assert result is None

    def test_duplicate_open_returns_none(self, pm_basic):
        pm_basic.open_position("mkt-1", "BUY", 10.0, 3000.0)
        result = pm_basic.open_position("mkt-1", "BUY", 5.0, 3000.0)
        assert result is None

    def test_zero_size_blocked(self, pm_basic):
        result = pm_basic.open_position("mkt-1", "BUY", 0.0, 3000.0)
        assert result is None

    def test_size_clipped_to_max_pct(self, pm_small):
        # 20% max of $10 = $2
        pos = pm_small.open_position("mkt-1", "BUY", 100.0, 3000.0)
        assert pos.size_usdc == pytest.approx(2.0)  # clipped to 20% * $10

    def test_win_counted_on_profit(self, pm_basic):
        pm_basic.open_position("mkt-1", "BUY", 10.0, 3000.0)
        pm_basic.close_position("mkt-1", 3300.0, pnl_usdc=1.0)
        stats = pm_basic.get_stats()
        assert stats.winning_trades == 1

    def test_loss_not_counted_as_win(self, pm_basic):
        pm_basic.open_position("mkt-1", "BUY", 10.0, 3000.0)
        pm_basic.close_position("mkt-1", 2700.0, pnl_usdc=-2.0)
        stats = pm_basic.get_stats()
        assert stats.winning_trades == 0


# ─── Drawdown Halt ────────────────────────────────────────────────────────────

class TestDrawdownHalt:

    def test_drawdown_halt_triggered_at_10pct(self, pm_basic):
        # Initial capital: $100, lose $10+ = 10% drawdown
        pm_basic.open_position("mkt-1", "BUY", 20.0, 3000.0)
        pm_basic.close_position("mkt-1", 2700.0, pnl_usdc=-10.0)
        assert pm_basic._drawdown_halted is True

    def test_drawdown_halt_blocks_new_position(self, pm_basic):
        pm_basic.open_position("mkt-1", "BUY", 20.0, 3000.0)
        pm_basic.close_position("mkt-1", 2700.0, pnl_usdc=-10.0)
        # Now halted
        result = pm_basic.open_position("mkt-2", "BUY", 5.0, 3000.0)
        assert result is None

    def test_lift_drawdown_halt(self, pm_basic):
        pm_basic._drawdown_halted = True
        pm_basic.lift_drawdown_halt()
        assert pm_basic._drawdown_halted is False

    def test_small_loss_does_not_halt(self, pm_basic):
        pm_basic.open_position("mkt-1", "BUY", 5.0, 3000.0)
        pm_basic.close_position("mkt-1", 2900.0, pnl_usdc=-5.0)
        # $5 loss on $100 capital = 5% drawdown < 10% max
        assert pm_basic._drawdown_halted is False

    def test_profit_updates_peak_capital(self, pm_basic):
        pm_basic.open_position("mkt-1", "BUY", 10.0, 3000.0)
        pm_basic.close_position("mkt-1", 3300.0, pnl_usdc=5.0)
        assert pm_basic._peak_capital == pytest.approx(105.0)


# ─── Kelly Sizing ─────────────────────────────────────────────────────────────

class TestKellySizing:

    def test_compute_kelly_size_basic(self, pm_basic):
        size = pm_basic.compute_kelly_size(confidence=0.6)
        assert size >= 0.0
        assert size <= pm_basic.capital_usdc * pm_basic.MAX_POSITION_PCT

    def test_low_confidence_gives_small_size(self, pm_basic):
        size_low = pm_basic.compute_kelly_size(confidence=0.51)
        size_high = pm_basic.compute_kelly_size(confidence=0.75)
        assert size_low < size_high

    def test_size_never_exceeds_max_position_pct(self, pm_basic):
        size = pm_basic.compute_kelly_size(confidence=0.99)
        max_size = pm_basic.capital_usdc * pm_basic.MAX_POSITION_PCT
        assert size <= max_size

    def test_zero_confidence_returns_zero(self, pm_basic):
        size = pm_basic.compute_kelly_size(confidence=0.0)
        assert size == 0.0


# ─── Stats Reporting ──────────────────────────────────────────────────────────

class TestStatsReporting:

    def test_initial_stats(self, pm_basic):
        stats = pm_basic.get_stats()
        assert stats.capital_usdc == pytest.approx(100.0)
        assert stats.open_positions == 0
        assert stats.total_trades == 0
        assert stats.total_pnl_usdc == 0.0

    def test_get_open_positions_empty(self, pm_basic):
        assert pm_basic.get_open_positions() == []

    def test_get_closed_positions_empty(self, pm_basic):
        assert pm_basic.get_closed_positions() == []

    def test_get_open_positions_after_open(self, pm_basic):
        pm_basic.open_position("mkt-1", "BUY", 5.0, 3000.0)
        positions = pm_basic.get_open_positions()
        assert len(positions) == 1
        assert positions[0]["market_id"] == "mkt-1"

    def test_get_closed_positions_after_close(self, pm_basic):
        pm_basic.open_position("mkt-1", "BUY", 5.0, 3000.0)
        pm_basic.close_position("mkt-1", 3300.0, 1.0)
        closed = pm_basic.get_closed_positions()
        assert len(closed) == 1
        assert closed[0]["is_open"] is False
