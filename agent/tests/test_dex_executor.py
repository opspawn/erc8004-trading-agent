"""
Tests for dex_executor.py — 50 tests covering:
  - SwapParams and SwapResult data classes
  - calculate_swap_output() math
  - check_slippage() validation
  - DexExecutor.simulate_swap()
  - DexExecutor.execute_swap()
  - Risk gating behavior
  - Stats tracking
  - Edge cases
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import pytest
from unittest.mock import MagicMock

from dex_executor import (
    DexExecutor,
    SwapParams,
    SwapResult,
    SimulationResult,
    TokenSymbol,
    calculate_swap_output,
    check_slippage,
    TOKEN_ADDRESSES,
)


# ─── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture
def eth_oracle_price():
    return 3000.0  # $3000 per ETH

@pytest.fixture
def btc_oracle_price():
    return 60000.0  # $60,000 per BTC

@pytest.fixture
def eth_to_usdc_params(eth_oracle_price):
    return SwapParams(
        agent_id=1,
        token_in=TokenSymbol.ETH,
        token_out=TokenSymbol.USDC,
        amount_in=eth_oracle_price,  # amount_in = oracle price (0% deviation)
        min_amount_out=0.0,
        oracle_price=eth_oracle_price,
    )

@pytest.fixture
def usdc_to_eth_params(eth_oracle_price):
    return SwapParams(
        agent_id=2,
        token_in=TokenSymbol.USDC,
        token_out=TokenSymbol.ETH,
        amount_in=eth_oracle_price,  # buy 1 ETH worth of USDC
        min_amount_out=0.0,
        oracle_price=eth_oracle_price,
    )

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

@pytest.fixture
def executor_dry_run():
    return DexExecutor(dry_run=True)

@pytest.fixture
def executor_with_risk(mock_risk_manager_pass):
    return DexExecutor(risk_manager=mock_risk_manager_pass, dry_run=True)

@pytest.fixture
def executor_risk_fail(mock_risk_manager_fail):
    return DexExecutor(risk_manager=mock_risk_manager_fail, dry_run=True)


# ─── TokenSymbol ──────────────────────────────────────────────────────────────

class TestTokenSymbol:

    def test_eth_symbol(self):
        assert TokenSymbol.ETH.value == "ETH"

    def test_btc_symbol(self):
        assert TokenSymbol.BTC.value == "BTC"

    def test_usdc_symbol(self):
        assert TokenSymbol.USDC.value == "USDC"

    def test_weth_symbol(self):
        assert TokenSymbol.WETH.value == "WETH"

    def test_token_addresses_populated(self):
        assert len(TOKEN_ADDRESSES) >= 4
        assert TokenSymbol.ETH in TOKEN_ADDRESSES
        assert TokenSymbol.USDC in TOKEN_ADDRESSES


# ─── SwapParams / SwapResult / SimulationResult ───────────────────────────────

class TestDataClasses:

    def test_swap_params_creation(self):
        p = SwapParams(
            agent_id=1, token_in=TokenSymbol.ETH, token_out=TokenSymbol.USDC,
            amount_in=3000.0, min_amount_out=2900.0, oracle_price=3000.0,
        )
        assert p.agent_id == 1
        assert p.token_in == TokenSymbol.ETH
        assert p.slippage_bps == 50  # default

    def test_swap_result_to_dict(self):
        r = SwapResult(
            success=True, agent_id=1, token_in=TokenSymbol.ETH, token_out=TokenSymbol.USDC,
            amount_in=1.0, amount_out=3000.0, oracle_price=3000.0, risk_passed=True,
            simulated=True, tx_hash="0xabc", error=None,
        )
        d = r.to_dict()
        assert d["success"] is True
        assert d["token_in"] == "ETH"
        assert d["simulated"] is True
        assert d["tx_hash"] == "0xabc"

    def test_simulation_result_is_executable_true(self):
        sr = SimulationResult(
            agent_id=1, token_in=TokenSymbol.ETH, token_out=TokenSymbol.USDC,
            amount_in=3000.0, estimated_out=3000000.0, oracle_price=3000.0,
            risk_passed=True, rejection_reason=None,
        )
        assert sr.is_executable is True

    def test_simulation_result_is_executable_false_risk(self):
        sr = SimulationResult(
            agent_id=1, token_in=TokenSymbol.ETH, token_out=TokenSymbol.USDC,
            amount_in=3000.0, estimated_out=0.0, oracle_price=3000.0,
            risk_passed=False, rejection_reason="deviation too high",
        )
        assert sr.is_executable is False

    def test_simulation_result_to_dict(self):
        sr = SimulationResult(
            agent_id=1, token_in=TokenSymbol.ETH, token_out=TokenSymbol.USDC,
            amount_in=1.0, estimated_out=3000.0, oracle_price=3000.0,
            risk_passed=True, rejection_reason=None,
        )
        d = sr.to_dict()
        assert d["risk_passed"] is True
        assert d["is_executable"] is True


# ─── calculate_swap_output ────────────────────────────────────────────────────

class TestCalculateSwapOutput:

    def test_eth_to_usdc(self):
        # 1 ETH at $3000 → $3000 USDC
        out = calculate_swap_output(1.0, TokenSymbol.ETH, TokenSymbol.USDC, 3000.0)
        assert out == pytest.approx(3000.0)

    def test_usdc_to_eth(self):
        # $3000 USDC → 1 ETH at $3000
        out = calculate_swap_output(3000.0, TokenSymbol.USDC, TokenSymbol.ETH, 3000.0)
        assert out == pytest.approx(1.0)

    def test_weth_to_usdc(self):
        # WETH behaves same as ETH
        out = calculate_swap_output(1.0, TokenSymbol.WETH, TokenSymbol.USDC, 3000.0)
        assert out == pytest.approx(3000.0)

    def test_usdc_to_weth(self):
        out = calculate_swap_output(3000.0, TokenSymbol.USDC, TokenSymbol.WETH, 3000.0)
        assert out == pytest.approx(1.0)

    def test_btc_to_usdc(self):
        # 1 BTC at $60k → $60k USDC
        out = calculate_swap_output(1.0, TokenSymbol.BTC, TokenSymbol.USDC, 60000.0)
        assert out == pytest.approx(60000.0)

    def test_usdc_to_btc(self):
        out = calculate_swap_output(60000.0, TokenSymbol.USDC, TokenSymbol.BTC, 60000.0)
        assert out == pytest.approx(1.0)

    def test_zero_oracle_price_returns_zero(self):
        out = calculate_swap_output(1.0, TokenSymbol.ETH, TokenSymbol.USDC, 0.0)
        assert out == 0.0

    def test_zero_amount_in_returns_zero(self):
        out = calculate_swap_output(0.0, TokenSymbol.ETH, TokenSymbol.USDC, 3000.0)
        assert out == 0.0

    def test_same_asset_fallback(self):
        # ETH → WETH (same, fallback 1:1)
        out = calculate_swap_output(1.5, TokenSymbol.ETH, TokenSymbol.WETH, 3000.0)
        assert out == pytest.approx(1.5)

    def test_output_scales_with_price(self):
        out1 = calculate_swap_output(1.0, TokenSymbol.ETH, TokenSymbol.USDC, 3000.0)
        out2 = calculate_swap_output(1.0, TokenSymbol.ETH, TokenSymbol.USDC, 6000.0)
        assert out2 == pytest.approx(out1 * 2)


# ─── check_slippage ───────────────────────────────────────────────────────────

class TestCheckSlippage:

    def test_passes_when_no_minimum(self):
        ok, err = check_slippage(100.0, 0.0)
        assert ok is True
        assert err is None

    def test_passes_when_above_minimum(self):
        ok, err = check_slippage(100.0, 95.0)
        assert ok is True
        assert err is None

    def test_passes_exactly_at_minimum(self):
        ok, err = check_slippage(100.0, 100.0)
        assert ok is True

    def test_fails_when_below_minimum(self):
        ok, err = check_slippage(90.0, 95.0)
        assert ok is False
        assert err is not None
        assert "Slippage" in err

    def test_negative_minimum_passes(self):
        ok, err = check_slippage(50.0, -1.0)
        assert ok is True


# ─── DexExecutor — simulate_swap ─────────────────────────────────────────────

class TestSimulateSwap:

    def test_returns_simulation_result(self, executor_dry_run, eth_to_usdc_params):
        result = executor_dry_run.simulate_swap(eth_to_usdc_params)
        assert isinstance(result, SimulationResult)

    def test_risk_passes_when_amount_equals_oracle(self, executor_dry_run, eth_to_usdc_params):
        result = executor_dry_run.simulate_swap(eth_to_usdc_params)
        assert result.risk_passed is True

    def test_risk_fails_when_amount_far_from_oracle(self, executor_dry_run):
        params = SwapParams(
            agent_id=1, token_in=TokenSymbol.ETH, token_out=TokenSymbol.USDC,
            amount_in=10000.0,  # 233% deviation from $3000
            min_amount_out=0.0, oracle_price=3000.0,
        )
        result = executor_dry_run.simulate_swap(params)
        assert result.risk_passed is False
        assert result.estimated_out == 0.0

    def test_estimated_out_calculated_correctly(self, executor_dry_run, eth_to_usdc_params):
        result = executor_dry_run.simulate_swap(eth_to_usdc_params)
        # 1 ETH (amount_in = oracle_price = 3000) → 3000 * 3000 = 9,000,000 USDC
        # Actually amount_in=3000 ETH → 3000 * 3000 = 9,000,000 USDC
        expected = calculate_swap_output(
            eth_to_usdc_params.amount_in,
            eth_to_usdc_params.token_in,
            eth_to_usdc_params.token_out,
            eth_to_usdc_params.oracle_price,
        )
        assert result.estimated_out == pytest.approx(expected)

    def test_with_risk_manager_that_passes(self, executor_with_risk, eth_to_usdc_params):
        result = executor_with_risk.simulate_swap(eth_to_usdc_params)
        assert result.risk_passed is True

    def test_with_risk_manager_that_fails(self, executor_risk_fail, eth_to_usdc_params):
        result = executor_risk_fail.simulate_swap(eth_to_usdc_params)
        assert result.risk_passed is False
        assert result.rejection_reason is not None

    def test_zero_amount_blocked(self, executor_dry_run):
        params = SwapParams(
            agent_id=1, token_in=TokenSymbol.ETH, token_out=TokenSymbol.USDC,
            amount_in=0.0, min_amount_out=0.0, oracle_price=3000.0,
        )
        result = executor_dry_run.simulate_swap(params)
        assert result.risk_passed is False


# ─── DexExecutor — execute_swap ──────────────────────────────────────────────

class TestExecuteSwap:

    def test_returns_swap_result(self, executor_dry_run, eth_to_usdc_params):
        result = executor_dry_run.execute_swap(eth_to_usdc_params)
        assert isinstance(result, SwapResult)

    def test_successful_swap_is_marked_success(self, executor_dry_run, eth_to_usdc_params):
        result = executor_dry_run.execute_swap(eth_to_usdc_params)
        assert result.success is True
        assert result.risk_passed is True

    def test_simulated_flag_set_in_dry_run(self, executor_dry_run, eth_to_usdc_params):
        result = executor_dry_run.execute_swap(eth_to_usdc_params)
        assert result.simulated is True

    def test_tx_hash_generated_on_success(self, executor_dry_run, eth_to_usdc_params):
        result = executor_dry_run.execute_swap(eth_to_usdc_params)
        assert result.tx_hash is not None
        assert result.tx_hash.startswith("0x")

    def test_amount_out_positive_on_success(self, executor_dry_run, eth_to_usdc_params):
        result = executor_dry_run.execute_swap(eth_to_usdc_params)
        assert result.amount_out > 0

    def test_failed_swap_has_no_tx_hash(self, executor_dry_run):
        params = SwapParams(
            agent_id=1, token_in=TokenSymbol.ETH, token_out=TokenSymbol.USDC,
            amount_in=100000.0,  # far from oracle
            min_amount_out=0.0, oracle_price=3000.0,
        )
        result = executor_dry_run.execute_swap(params)
        assert result.success is False
        assert result.tx_hash is None

    def test_slippage_too_high_fails(self, executor_dry_run, eth_to_usdc_params):
        # Set min_amount_out to impossibly high value
        eth_to_usdc_params.min_amount_out = 999_999_999.0
        result = executor_dry_run.execute_swap(eth_to_usdc_params)
        assert result.success is False
        assert "Slippage" in (result.error or "")

    def test_risk_manager_fail_blocks_swap(self, executor_risk_fail, eth_to_usdc_params):
        result = executor_risk_fail.execute_swap(eth_to_usdc_params)
        assert result.success is False
        assert result.risk_passed is False

    def test_risk_manager_called_with_correct_args(
        self, mock_risk_manager_pass, eth_to_usdc_params
    ):
        executor = DexExecutor(risk_manager=mock_risk_manager_pass, dry_run=True)
        executor.execute_swap(eth_to_usdc_params)
        mock_risk_manager_pass.check_oracle_risk.assert_called_once_with(
            amount=eth_to_usdc_params.amount_in,
            oracle_price=eth_to_usdc_params.oracle_price,
            max_deviation_pct=executor.max_deviation_pct,
        )


# ─── DexExecutor — Stats ──────────────────────────────────────────────────────

class TestStats:

    def test_initial_stats_zero(self, executor_dry_run):
        stats = executor_dry_run.get_stats()
        assert stats["total_swaps_attempted"] == 0
        assert stats["total_swaps_successful"] == 0
        assert stats["total_swaps_blocked"] == 0
        assert stats["total_volume_in"] == 0.0

    def test_successful_swap_updates_stats(self, executor_dry_run, eth_to_usdc_params):
        executor_dry_run.execute_swap(eth_to_usdc_params)
        stats = executor_dry_run.get_stats()
        assert stats["total_swaps_attempted"] == 1
        assert stats["total_swaps_successful"] == 1

    def test_failed_swap_updates_attempted_not_successful(self, executor_dry_run):
        params = SwapParams(
            agent_id=1, token_in=TokenSymbol.ETH, token_out=TokenSymbol.USDC,
            amount_in=999999.0, min_amount_out=0.0, oracle_price=3000.0,
        )
        executor_dry_run.execute_swap(params)
        stats = executor_dry_run.get_stats()
        assert stats["total_swaps_attempted"] == 1
        assert stats["total_swaps_successful"] == 0
        assert stats["total_swaps_blocked"] == 1

    def test_volume_accumulates(self, executor_dry_run, eth_to_usdc_params):
        executor_dry_run.execute_swap(eth_to_usdc_params)
        executor_dry_run.execute_swap(eth_to_usdc_params)
        stats = executor_dry_run.get_stats()
        assert stats["total_volume_in"] == pytest.approx(
            eth_to_usdc_params.amount_in * 2
        )

    def test_get_history_empty_initially(self, executor_dry_run):
        assert executor_dry_run.get_history() == []

    def test_get_history_after_swaps(self, executor_dry_run, eth_to_usdc_params):
        executor_dry_run.execute_swap(eth_to_usdc_params)
        history = executor_dry_run.get_history()
        assert len(history) == 1
        assert history[0]["token_in"] == "ETH"

    def test_get_agent_volume(self, executor_dry_run, eth_to_usdc_params):
        executor_dry_run.execute_swap(eth_to_usdc_params)
        volume = executor_dry_run.get_agent_volume(eth_to_usdc_params.agent_id)
        assert volume == pytest.approx(eth_to_usdc_params.amount_in)

    def test_get_agent_volume_excludes_failed(self, executor_dry_run):
        params = SwapParams(
            agent_id=42, token_in=TokenSymbol.ETH, token_out=TokenSymbol.USDC,
            amount_in=999999.0, min_amount_out=0.0, oracle_price=3000.0,
        )
        executor_dry_run.execute_swap(params)
        volume = executor_dry_run.get_agent_volume(42)
        assert volume == 0.0

    def test_stats_dry_run_flag(self, executor_dry_run):
        stats = executor_dry_run.get_stats()
        assert stats["dry_run"] is True
