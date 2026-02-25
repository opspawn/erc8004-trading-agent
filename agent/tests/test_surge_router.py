"""
Tests for surge_router.py — 45+ tests covering:
  - TradeResult data class and effective_rate
  - VaultBalance data class and get_balance()
  - Position data class and pnl_pct / value_usdc
  - MockSurgeRouter initialization and validation
  - execute_trade() success and failure paths
  - get_vault_balance()
  - get_position() and list_positions()
  - close_position()
  - get_token_price()
  - get_supported_tokens()
  - Slippage guard enforcement
  - Insufficient balance rejection
  - Price override helper
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import asyncio
import pytest
from datetime import datetime, timezone

from surge_router import (
    MockSurgeRouter,
    Position,
    TradeResult,
    VaultBalance,
)


# ─── Helpers ──────────────────────────────────────────────────────────────────

def run(coro):
    return asyncio.get_event_loop().run_until_complete(coro)


# ─── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture
def router():
    return MockSurgeRouter(initial_usdc=10_000.0, slippage_pct=0.001, fee_pct=0.003)

@pytest.fixture
def router_tiny():
    return MockSurgeRouter(initial_usdc=100.0)


# ─── TradeResult ──────────────────────────────────────────────────────────────

class TestTradeResult:
    def test_effective_rate_normal(self):
        r = TradeResult(True, "t1", "USDC", "ETH", 2000.0, 1.0, 2000.0, 0.001, 6.0)
        assert r.effective_rate == pytest.approx(0.0005, rel=1e-3)

    def test_effective_rate_zero_in(self):
        r = TradeResult(True, "t1", "USDC", "ETH", 0.0, 0.0, 0.0, 0.0, 0.0)
        assert r.effective_rate == 0.0

    def test_success_false_has_error(self):
        r = TradeResult(False, "t1", "USDC", "ETH", 100, 0, 0, 0, 0, error="fail")
        assert r.error == "fail"
        assert not r.success

    def test_tx_hash_none_on_failure(self):
        r = TradeResult(False, "t1", "USDC", "ETH", 100, 0, 0, 0, 0)
        assert r.tx_hash is None


# ─── VaultBalance ─────────────────────────────────────────────────────────────

class TestVaultBalance:
    def test_get_balance_usdc(self):
        v = VaultBalance("0x1", 1000.0, 0.5, 0.01, 2025.0)
        assert v.get_balance("USDC") == 1000.0

    def test_get_balance_eth(self):
        v = VaultBalance("0x1", 1000.0, 0.5, 0.01, 2025.0)
        assert v.get_balance("ETH") == 0.5

    def test_get_balance_btc(self):
        v = VaultBalance("0x1", 1000.0, 0.5, 0.01, 2025.0)
        assert v.get_balance("BTC") == 0.01

    def test_get_balance_unknown(self):
        v = VaultBalance("0x1", 1000.0, 0.5, 0.01, 2025.0)
        assert v.get_balance("SOL") == 0.0

    def test_get_balance_case_insensitive(self):
        v = VaultBalance("0x1", 500.0, 1.0, 0.0, 2500.0)
        assert v.get_balance("eth") == 1.0


# ─── Position ─────────────────────────────────────────────────────────────────

class TestPosition:
    def test_pnl_pct_profit(self):
        pos = Position("p1", "ETH", 1.0, 2000.0, 2200.0)
        assert pos.pnl_pct == pytest.approx(10.0, rel=1e-6)

    def test_pnl_pct_loss(self):
        pos = Position("p1", "ETH", 1.0, 2000.0, 1800.0)
        assert pos.pnl_pct == pytest.approx(-10.0, rel=1e-6)

    def test_pnl_pct_zero_entry(self):
        pos = Position("p1", "ETH", 1.0, 0.0, 100.0)
        assert pos.pnl_pct == 0.0

    def test_value_usdc(self):
        pos = Position("p1", "ETH", 2.0, 2000.0, 2100.0)
        assert pos.value_usdc == pytest.approx(4200.0)


# ─── MockSurgeRouter init ─────────────────────────────────────────────────────

class TestMockSurgeRouterInit:
    def test_default_balance(self):
        r = MockSurgeRouter(initial_usdc=5000.0)
        bal = run(r.get_vault_balance())
        assert bal.usdc_balance == pytest.approx(5000.0)

    def test_negative_usdc_raises(self):
        with pytest.raises(ValueError):
            MockSurgeRouter(initial_usdc=-100.0)

    def test_bad_slippage_raises(self):
        with pytest.raises(ValueError):
            MockSurgeRouter(slippage_pct=1.5)

    def test_bad_fee_raises(self):
        with pytest.raises(ValueError):
            MockSurgeRouter(fee_pct=0.5)

    def test_supported_tokens(self, router):
        tokens = run(router.get_supported_tokens())
        assert "ETH" in tokens
        assert "USDC" in tokens
        assert "BTC" in tokens


# ─── execute_trade() ──────────────────────────────────────────────────────────

class TestExecuteTrade:
    def test_buy_eth_success(self, router):
        result = run(router.execute_trade("USDC", "ETH", 2000.0))
        assert result.success is True
        assert result.amount_out > 0
        assert result.tx_hash is not None

    def test_usdc_deducted(self, router):
        run(router.execute_trade("USDC", "ETH", 1000.0))
        bal = run(router.get_vault_balance())
        assert bal.usdc_balance == pytest.approx(9000.0, abs=1.0)

    def test_eth_credited(self, router):
        run(router.execute_trade("USDC", "ETH", 2000.0))
        bal = run(router.get_vault_balance())
        assert bal.eth_balance > 0

    def test_zero_amount_fails(self, router):
        result = run(router.execute_trade("USDC", "ETH", 0.0))
        assert result.success is False
        assert "positive" in result.error.lower()

    def test_insufficient_balance_fails(self, router_tiny):
        result = run(router_tiny.execute_trade("USDC", "ETH", 999_999.0))
        assert result.success is False
        assert "Insufficient" in result.error

    def test_slippage_guard_fails(self, router):
        result = run(router.execute_trade("USDC", "ETH", 100.0, min_out=999.0))
        assert result.success is False
        assert "Slippage" in result.error

    def test_slippage_guard_passes_when_ok(self, router):
        result = run(router.execute_trade("USDC", "ETH", 2000.0, min_out=0.9))
        assert result.success is True

    def test_fee_charged(self, router):
        result = run(router.execute_trade("USDC", "ETH", 1000.0))
        assert result.fee_usdc > 0

    def test_trade_ids_unique(self, router):
        r1 = run(router.execute_trade("USDC", "ETH", 100.0))
        r2 = run(router.execute_trade("USDC", "ETH", 100.0))
        assert r1.trade_id != r2.trade_id

    def test_reverse_trade_eth_to_usdc(self, router):
        router.open_position("ETH", 1.0, 2000.0)
        result = run(router.execute_trade("ETH", "USDC", 0.5))
        assert result.success is True
        assert result.token_out == "USDC"

    def test_unsupported_token_raises(self, router):
        with pytest.raises(ValueError, match="Unsupported token"):
            run(router.execute_trade("USDC", "DOGE", 100.0))


# ─── get_vault_balance() ──────────────────────────────────────────────────────

class TestGetVaultBalance:
    def test_returns_vault_balance(self, router):
        bal = run(router.get_vault_balance())
        assert isinstance(bal, VaultBalance)

    def test_default_vault_address(self, router):
        bal = run(router.get_vault_balance())
        assert bal.vault_address == MockSurgeRouter.DEFAULT_VAULT

    def test_custom_vault_address(self, router):
        bal = run(router.get_vault_balance("0xCustom"))
        assert bal.vault_address == "0xCustom"

    def test_total_value_positive(self, router):
        bal = run(router.get_vault_balance())
        assert bal.total_value_usdc > 0


# ─── get_position() / list_positions() ────────────────────────────────────────

class TestPositions:
    def test_no_position_returns_none(self, router):
        pos = run(router.get_position("ETH"))
        assert pos is None

    def test_open_position_visible(self, router):
        router.open_position("ETH", 1.0, 2000.0)
        pos = run(router.get_position("ETH"))
        assert pos is not None
        assert pos.token == "ETH"
        assert pos.size == 1.0

    def test_list_positions_empty(self, router):
        positions = run(router.list_positions())
        assert positions == []

    def test_list_positions_after_open(self, router):
        router.open_position("ETH", 0.5, 2000.0)
        router.open_position("BTC", 0.01, 45000.0)
        positions = run(router.list_positions())
        assert len(positions) == 2

    def test_pnl_updated_on_price_change(self, router):
        router.open_position("ETH", 1.0, 2000.0)
        router.set_price("ETH", 2200.0)
        pos = run(router.get_position("ETH"))
        assert pos.pnl_usdc == pytest.approx(200.0, abs=1.0)


# ─── close_position() ─────────────────────────────────────────────────────────

class TestClosePosition:
    def test_close_open_position(self, router):
        pos = router.open_position("ETH", 1.0, 2000.0)
        result = run(router.close_position(pos.position_id))
        assert result.success is True

    def test_close_nonexistent_position(self, router):
        result = run(router.close_position("fake-id-9999"))
        assert result.success is False
        assert "not found" in result.error

    def test_position_closed_after_close(self, router):
        pos = router.open_position("ETH", 1.0, 2000.0)
        run(router.close_position(pos.position_id))
        assert pos.is_open is False


# ─── get_token_price() ────────────────────────────────────────────────────────

class TestGetTokenPrice:
    def test_eth_price(self, router):
        price = run(router.get_token_price("ETH"))
        assert price == pytest.approx(2000.0)

    def test_btc_price(self, router):
        price = run(router.get_token_price("BTC"))
        assert price == pytest.approx(45000.0)

    def test_usdc_price(self, router):
        price = run(router.get_token_price("USDC"))
        assert price == pytest.approx(1.0)

    def test_set_price_override(self, router):
        router.set_price("ETH", 3000.0)
        price = run(router.get_token_price("ETH"))
        assert price == pytest.approx(3000.0)

    def test_unknown_token_raises(self, router):
        with pytest.raises(ValueError):
            run(router.get_token_price("DOGE"))
