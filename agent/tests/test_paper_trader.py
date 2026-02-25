"""
Tests for paper_trader.py — Paper Trading Simulation Mode.

All tests are unit-based using the PaperTrader, GBMPriceGenerator,
PaperTrade, and SimulationReport classes directly.
"""

from __future__ import annotations

import math
import sys
import os
import time
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from paper_trader import (
    GBMPriceGenerator,
    PaperTrade,
    SimulationReport,
    PaperTrader,
    PROTOCOLS,
)


# ─── GBM Price Generator Tests ─────────────────────────────────────────────────

class TestGBMPriceGenerator:
    def test_initial_price_stored(self):
        gen = GBMPriceGenerator(initial_price=100.0)
        assert gen.price == pytest.approx(100.0)

    def test_next_price_changes_price(self):
        gen = GBMPriceGenerator(initial_price=100.0, seed=42)
        p = gen.next_price()
        assert p != pytest.approx(100.0, abs=0.001)

    def test_next_price_positive(self):
        gen = GBMPriceGenerator(initial_price=1.0, sigma=0.5, seed=1)
        for _ in range(100):
            p = gen.next_price()
            assert p > 0.0

    def test_steps_increments(self):
        gen = GBMPriceGenerator()
        assert gen.steps == 0
        gen.next_price()
        assert gen.steps == 1
        gen.next_price()
        assert gen.steps == 2

    def test_generate_returns_list(self):
        gen = GBMPriceGenerator(seed=0)
        prices = gen.generate(10)
        assert isinstance(prices, list)
        assert len(prices) == 10

    def test_generate_all_positive(self):
        gen = GBMPriceGenerator(initial_price=50.0, seed=7)
        prices = gen.generate(50)
        assert all(p > 0 for p in prices)

    def test_generate_steps_updated(self):
        gen = GBMPriceGenerator()
        gen.generate(20)
        assert gen.steps == 20

    def test_deterministic_with_seed(self):
        gen1 = GBMPriceGenerator(initial_price=100.0, seed=42)
        gen2 = GBMPriceGenerator(initial_price=100.0, seed=42)
        prices1 = gen1.generate(10)
        prices2 = gen2.generate(10)
        assert prices1 == prices2

    def test_different_seeds_differ(self):
        gen1 = GBMPriceGenerator(initial_price=100.0, seed=1)
        gen2 = GBMPriceGenerator(initial_price=100.0, seed=2)
        p1 = gen1.generate(5)
        p2 = gen2.generate(5)
        assert p1 != p2

    def test_high_drift_increases_price_on_average(self):
        """With high positive drift, mean price should exceed initial."""
        prices = []
        for _ in range(20):
            gen = GBMPriceGenerator(initial_price=100.0, mu=5.0, sigma=0.01,
                                    dt=1/100, seed=None)
            prices.append(gen.generate(50)[-1])
        assert sum(prices) / len(prices) > 100.0

    def test_normal_variate_via_generate(self):
        """Check that generated prices follow rough GBM distribution."""
        gen = GBMPriceGenerator(initial_price=1.0, mu=0.0, sigma=0.1,
                                 dt=1/252, seed=99)
        prices = gen.generate(252)
        # Log returns should be roughly normal
        log_returns = [math.log(prices[i] / prices[i-1]) for i in range(1, len(prices))]
        mean = sum(log_returns) / len(log_returns)
        assert -0.5 <= mean <= 0.5  # loose bound

    def test_different_initial_prices(self):
        for ip in [0.5, 1.0, 100.0, 10_000.0]:
            gen = GBMPriceGenerator(initial_price=ip, seed=0)
            p = gen.next_price()
            assert p > 0


# ─── PaperTrade Tests ──────────────────────────────────────────────────────────

class TestPaperTrade:
    def _make_trade(self, **kwargs) -> PaperTrade:
        defaults = dict(
            trade_id=1,
            token="ETH",
            side="BUY",
            entry_price=2500.0,
            exit_price=2600.0,
            size=0.1,
            size_usdc=250.0,
            pnl=10.0,
            pnl_pct=0.04,
            entry_ts=time.time() - 300,
            exit_ts=time.time(),
            mesh_consensus=True,
            agent_id="balanced_agent",
            protocol="ETH",
            confidence=0.70,
        )
        defaults.update(kwargs)
        return PaperTrade(**defaults)

    def test_is_winner_positive_pnl(self):
        trade = self._make_trade(pnl=10.0)
        assert trade.is_winner is True

    def test_is_winner_negative_pnl(self):
        trade = self._make_trade(pnl=-5.0)
        assert trade.is_winner is False

    def test_is_winner_zero_pnl(self):
        trade = self._make_trade(pnl=0.0)
        assert trade.is_winner is False

    def test_to_dict_returns_dict(self):
        trade = self._make_trade()
        result = trade.to_dict()
        assert isinstance(result, dict)

    def test_to_dict_required_fields(self):
        trade = self._make_trade()
        result = trade.to_dict()
        required = ["trade_id", "token", "side", "entry_price", "exit_price",
                    "pnl", "pnl_pct", "mesh_consensus", "agent_id",
                    "protocol", "confidence", "is_winner"]
        for f in required:
            assert f in result, f"Missing: {f}"

    def test_to_dict_pnl_rounded(self):
        trade = self._make_trade(pnl=10.123456789)
        result = trade.to_dict()
        assert abs(result["pnl"] - round(10.123456789, 4)) < 1e-8

    def test_to_dict_is_winner_matches(self):
        trade = self._make_trade(pnl=5.0)
        result = trade.to_dict()
        assert result["is_winner"] is True

    def test_to_dict_side_preserved(self):
        for side in ["BUY", "SELL"]:
            trade = self._make_trade(side=side)
            result = trade.to_dict()
            assert result["side"] == side

    def test_to_dict_mesh_consensus_preserved(self):
        for mc in [True, False]:
            trade = self._make_trade(mesh_consensus=mc)
            result = trade.to_dict()
            assert result["mesh_consensus"] is mc

    def test_to_dict_confidence_rounded(self):
        trade = self._make_trade(confidence=0.7654321)
        result = trade.to_dict()
        assert len(str(result["confidence"]).split(".")[-1]) <= 4


# ─── SimulationReport Tests ────────────────────────────────────────────────────

class TestSimulationReport:
    def _make_report(self, **kwargs) -> SimulationReport:
        defaults = dict(
            initial_capital=10_000.0,
            final_portfolio_value=10_500.0,
            total_pnl=500.0,
            total_trades=50,
            winning_trades=32,
            losing_trades=18,
            win_rate=0.64,
            sharpe_ratio=1.25,
            max_drawdown_pct=0.07,
            ticks_simulated=288,
            simulation_time_secs=0.85,
        )
        defaults.update(kwargs)
        return SimulationReport(**defaults)

    def test_total_return_pct_positive(self):
        rep = self._make_report(initial_capital=10_000, final_portfolio_value=11_000)
        assert rep.total_return_pct == pytest.approx(0.10, abs=0.001)

    def test_total_return_pct_negative(self):
        rep = self._make_report(initial_capital=10_000, final_portfolio_value=9_000)
        assert rep.total_return_pct == pytest.approx(-0.10, abs=0.001)

    def test_total_return_pct_zero_capital(self):
        rep = self._make_report(initial_capital=0, final_portfolio_value=100)
        assert rep.total_return_pct == 0.0

    def test_summary_returns_string(self):
        rep = self._make_report()
        assert isinstance(rep.summary(), str)

    def test_summary_contains_capital(self):
        rep = self._make_report()
        assert "10,000" in rep.summary() or "10000" in rep.summary()

    def test_summary_contains_pnl(self):
        rep = self._make_report(total_pnl=500.0)
        assert "500" in rep.summary()

    def test_summary_contains_sharpe(self):
        rep = self._make_report(sharpe_ratio=1.25)
        assert "1.25" in rep.summary() or "Sharpe" in rep.summary()

    def test_summary_contains_trades(self):
        rep = self._make_report(total_trades=50)
        assert "50" in rep.summary()

    def test_to_dict_returns_dict(self):
        rep = self._make_report()
        d = rep.to_dict()
        assert isinstance(d, dict)

    def test_to_dict_required_fields(self):
        rep = self._make_report()
        d = rep.to_dict()
        required = ["initial_capital", "final_portfolio_value", "total_pnl",
                    "total_trades", "win_rate", "sharpe_ratio",
                    "max_drawdown_pct", "ticks_simulated"]
        for f in required:
            assert f in d

    def test_to_dict_total_return_pct(self):
        rep = self._make_report(initial_capital=10_000, final_portfolio_value=11_000)
        d = rep.to_dict()
        assert d["total_return_pct"] == pytest.approx(0.10, abs=0.001)

    def test_to_dict_reputation_updates_count(self):
        rep = self._make_report()
        rep.agent_reputation_updates = [{"a": 1}, {"b": 2}]
        d = rep.to_dict()
        assert d["reputation_updates"] == 2

    def test_trades_defaults_empty(self):
        rep = self._make_report()
        assert rep.trades == []

    def test_reputation_updates_defaults_empty(self):
        rep = self._make_report()
        assert rep.agent_reputation_updates == []


# ─── PaperTrader Tests ─────────────────────────────────────────────────────────

class TestPaperTrader:
    def test_creates_with_defaults(self):
        trader = PaperTrader()
        assert trader.initial_capital == 10_000.0

    def test_portfolio_starts_at_initial_capital(self):
        trader = PaperTrader(initial_capital=5_000.0)
        assert trader.portfolio_value == 5_000.0

    def test_position_size_default(self):
        trader = PaperTrader()
        assert trader.position_size_pct == 0.05

    def test_min_confidence_default(self):
        trader = PaperTrader()
        assert trader.min_confidence > 0.0

    def test_seed_deterministic(self):
        t1 = PaperTrader(seed=42)
        t2 = PaperTrader(seed=42)
        r1 = t1.run(ticks=50)
        r2 = t2.run(ticks=50)
        assert r1.total_trades == r2.total_trades
        assert r1.total_pnl == pytest.approx(r2.total_pnl, abs=0.001)

    def test_run_returns_simulation_report(self):
        trader = PaperTrader(seed=0)
        report = trader.run(ticks=20)
        assert isinstance(report, SimulationReport)

    def test_run_ticks_simulated_matches(self):
        trader = PaperTrader(seed=1)
        report = trader.run(ticks=30)
        assert report.ticks_simulated == 30

    def test_run_total_trades_non_negative(self):
        trader = PaperTrader(seed=2)
        report = trader.run(ticks=50)
        assert report.total_trades >= 0

    def test_run_wins_plus_losses_equals_total(self):
        trader = PaperTrader(seed=3)
        report = trader.run(ticks=100)
        assert report.winning_trades + report.losing_trades == report.total_trades

    def test_run_win_rate_in_range(self):
        trader = PaperTrader(seed=4)
        report = trader.run(ticks=100)
        assert 0.0 <= report.win_rate <= 1.0

    def test_run_max_drawdown_non_negative(self):
        trader = PaperTrader(seed=5)
        report = trader.run(ticks=100)
        assert report.max_drawdown_pct >= 0.0

    def test_run_portfolio_value_positive(self):
        trader = PaperTrader(seed=6)
        report = trader.run(ticks=50)
        assert report.final_portfolio_value > 0

    def test_run_simulation_time_positive(self):
        trader = PaperTrader(seed=7)
        report = trader.run(ticks=50)
        assert report.simulation_time_secs > 0.0

    def test_run_fast_enough(self):
        """Full 288-tick simulation should complete in <5 seconds."""
        trader = PaperTrader(seed=42)
        start = time.perf_counter()
        report = trader.run()  # default 288 ticks
        elapsed = time.perf_counter() - start
        assert elapsed < 5.0

    def test_run_default_ticks_288(self):
        trader = PaperTrader(seed=0)
        report = trader.run()
        assert report.ticks_simulated == 288

    def test_run_trades_have_valid_sides(self):
        trader = PaperTrader(seed=8)
        report = trader.run(ticks=100)
        for trade in report.trades:
            assert trade.side in {"BUY", "SELL"}

    def test_run_trades_have_valid_protocols(self):
        trader = PaperTrader(seed=9)
        report = trader.run(ticks=100)
        for trade in report.trades:
            assert trade.token in PROTOCOLS

    def test_run_trades_all_mesh_consensus_true(self):
        trader = PaperTrader(seed=10)
        report = trader.run(ticks=100)
        for trade in report.trades:
            assert trade.mesh_consensus is True

    def test_run_trades_size_positive(self):
        trader = PaperTrader(seed=11)
        report = trader.run(ticks=100)
        for trade in report.trades:
            assert trade.size_usdc > 0.0

    def test_run_total_pnl_matches_sum(self):
        trader = PaperTrader(seed=12)
        report = trader.run(ticks=100)
        expected_pnl = sum(t.pnl for t in report.trades)
        assert report.total_pnl == pytest.approx(expected_pnl, abs=0.01)

    def test_run_reputation_updates_logged(self):
        trader = PaperTrader(seed=13)
        report = trader.run(ticks=100)
        assert len(report.agent_reputation_updates) == report.total_trades

    def test_kelly_size_high_confidence(self):
        trader = PaperTrader()
        size = trader._kelly_size(0.9)
        assert 0.0 < size <= trader.position_size_pct

    def test_kelly_size_low_confidence(self):
        trader = PaperTrader()
        size = trader._kelly_size(0.4)
        assert size == 0.0  # 2*0.4 - 1 = -0.2 < 0 → clamped to 0

    def test_kelly_size_boundary(self):
        trader = PaperTrader()
        size = trader._kelly_size(0.5)
        # 2*0.5-1 = 0, quarter = 0
        assert size == 0.0

    def test_compute_pnl_buy_profit(self):
        trader = PaperTrader()
        pnl = trader._compute_pnl("BUY", entry=100.0, exit_=110.0, size_usdc=1000.0)
        assert pnl > 0

    def test_compute_pnl_buy_loss(self):
        trader = PaperTrader()
        pnl = trader._compute_pnl("BUY", entry=100.0, exit_=90.0, size_usdc=1000.0)
        assert pnl < 0

    def test_compute_pnl_sell_profit(self):
        trader = PaperTrader()
        pnl = trader._compute_pnl("SELL", entry=100.0, exit_=90.0, size_usdc=1000.0)
        assert pnl > 0

    def test_compute_pnl_sell_loss(self):
        trader = PaperTrader()
        pnl = trader._compute_pnl("SELL", entry=100.0, exit_=110.0, size_usdc=1000.0)
        assert pnl < 0

    def test_compute_pnl_includes_fees(self):
        trader = PaperTrader()
        # If entry == exit, pure fees remain
        pnl = trader._compute_pnl("BUY", entry=100.0, exit_=100.0, size_usdc=1000.0)
        assert pnl < 0  # only fees

    def test_sharpe_ratio_empty(self):
        trader = PaperTrader()
        assert trader._sharpe_ratio([]) == 0.0

    def test_sharpe_ratio_single(self):
        trader = PaperTrader()
        assert trader._sharpe_ratio([1.0]) == 0.0

    def test_sharpe_ratio_positive_series(self):
        trader = PaperTrader()
        sr = trader._sharpe_ratio([10.0] * 100)
        assert sr == 0.0  # zero std → 0

    def test_sharpe_ratio_mixed_series(self):
        trader = PaperTrader()
        sr = trader._sharpe_ratio([5.0, -2.0, 3.0, -1.0, 4.0] * 20)
        # Should return a finite number
        assert math.isfinite(sr)

    def test_max_drawdown_empty(self):
        trader = PaperTrader()
        assert trader._max_drawdown([]) == 0.0

    def test_max_drawdown_no_loss(self):
        trader = PaperTrader()
        dd = trader._max_drawdown([100, 110, 120, 130])
        assert dd == 0.0

    def test_max_drawdown_with_loss(self):
        trader = PaperTrader()
        dd = trader._max_drawdown([100, 120, 80, 90])
        # Peak 120, trough 80: dd = (120-80)/120 = 0.333
        assert dd == pytest.approx(1/3, abs=0.01)

    def test_max_drawdown_in_range(self):
        trader = PaperTrader(seed=14)
        report = trader.run(ticks=200)
        assert 0.0 <= report.max_drawdown_pct <= 1.0

    def test_initial_price_eth(self):
        assert PaperTrader._initial_price("ETH") == 2500.0

    def test_initial_price_btc(self):
        assert PaperTrader._initial_price("BTC") == 45000.0

    def test_initial_price_unknown(self):
        assert PaperTrader._initial_price("UNKNOWN") == 1.0

    def test_volatility_eth(self):
        assert PaperTrader._volatility("ETH") > 0

    def test_volatility_unknown(self):
        assert PaperTrader._volatility("UNKNOWN") > 0

    def test_update_reputation_stores_entry(self):
        trader = PaperTrader()
        trader._trade_id = 1
        entry = trader._update_reputation("balanced_agent", True, 0.05)
        assert entry["won"] is True
        assert len(trader._reputation_updates) == 1

    def test_update_reputation_win_score(self):
        trader = PaperTrader()
        trader._trade_id = 1
        entry = trader._update_reputation("balanced_agent", True, 0.05)
        assert entry["score"] == 750

    def test_update_reputation_loss_score(self):
        trader = PaperTrader()
        trader._trade_id = 1
        entry = trader._update_reputation("balanced_agent", False, -0.03)
        assert entry["score"] == 350

    def test_run_with_custom_coordinator(self):
        from mesh_coordinator import MeshCoordinator
        coord = MeshCoordinator()
        trader = PaperTrader(coordinator=coord, seed=99)
        report = trader.run(ticks=50)
        assert isinstance(report, SimulationReport)

    def test_protocols_list_non_empty(self):
        assert len(PROTOCOLS) > 0

    def test_protocols_contains_eth(self):
        assert "ETH" in PROTOCOLS

    def test_report_to_dict_all_numeric_values(self):
        trader = PaperTrader(seed=15)
        report = trader.run(ticks=50)
        d = report.to_dict()
        for key in ["initial_capital", "final_portfolio_value", "total_pnl",
                    "win_rate", "sharpe_ratio", "max_drawdown_pct"]:
            assert isinstance(d[key], (int, float)), f"{key} not numeric"
