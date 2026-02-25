"""
test_demo_cli.py — Tests for DemoPipeline and demo CLI.

Coverage:
  - _gbm_prices: length, reproducibility, positive values
  - _box_muller: basic statistics
  - _compute_signal: BUY/SELL/HOLD logic, threshold behaviour
  - _risk_check: approved/rejected cases, drawdown limit
  - PipelineStep: dataclass construction
  - DemoResult: to_json, trace content
  - DemoPipeline: init defaults, run(), ledger integration, determinism
  - DemoPipeline: symbol lookup, custom agent_id, custom capital
  - DemoPipeline: multiple symbols (BTC, ETH, SOL)
  - DemoPipeline: zero ticks edge case
  - DemoPipeline: single tick
  - CLI: main() with various args, JSON output flag
"""

from __future__ import annotations

import json
import math
import uuid
from io import StringIO
from unittest.mock import patch

import pytest

from demo_cli import (
    DemoPipeline,
    DemoResult,
    PipelineStep,
    RiskCheck,
    _box_muller,
    _compute_signal,
    _gbm_prices,
    _parse_args,
    _risk_check,
    main,
)
from trade_ledger import TradeLedger


# ─── _gbm_prices ──────────────────────────────────────────────────────────────


class TestGbmPrices:
    def test_returns_n_plus_1_prices(self):
        prices = _gbm_prices(initial=100.0, n=10)
        assert len(prices) == 11  # initial + 10 ticks

    def test_first_price_is_initial(self):
        prices = _gbm_prices(initial=500.0, n=5)
        assert prices[0] == 500.0

    def test_all_positive(self):
        prices = _gbm_prices(initial=100.0, n=50, seed=1)
        assert all(p > 0 for p in prices)

    def test_reproducible_with_seed(self):
        a = _gbm_prices(initial=100.0, n=20, seed=99)
        b = _gbm_prices(initial=100.0, n=20, seed=99)
        assert a == b

    def test_different_seeds_differ(self):
        a = _gbm_prices(initial=100.0, n=20, seed=1)
        b = _gbm_prices(initial=100.0, n=20, seed=2)
        assert a != b

    def test_zero_ticks(self):
        prices = _gbm_prices(initial=100.0, n=0)
        assert prices == [100.0]

    def test_custom_sigma_increases_variance(self):
        # Higher sigma = more spread
        import statistics
        low = _gbm_prices(100.0, 100, sigma=0.01, seed=42)
        high = _gbm_prices(100.0, 100, sigma=2.0, seed=42)
        assert statistics.stdev(high) > statistics.stdev(low)


# ─── _box_muller ──────────────────────────────────────────────────────────────


class TestBoxMuller:
    def test_returns_float(self):
        import random
        rng = random.Random(42)
        z = _box_muller(rng)
        assert isinstance(z, float)

    def test_approximately_standard_normal(self):
        import random
        import statistics
        rng = random.Random(123)
        samples = [_box_muller(rng) for _ in range(1000)]
        mean = statistics.mean(samples)
        stdev = statistics.stdev(samples)
        assert abs(mean) < 0.15
        assert 0.8 < stdev < 1.2


# ─── _compute_signal ──────────────────────────────────────────────────────────


class TestComputeSignal:
    def test_hold_when_prev_zero(self):
        assert _compute_signal(100.0, 0.0, 0.01) == "HOLD"

    def test_hold_on_flat_price(self):
        assert _compute_signal(100.0, 100.0, 0.01) == "HOLD"

    def test_buy_on_strong_up_move(self):
        # 5% up move, threshold ≈ 0.0005
        result = _compute_signal(105.0, 100.0, 0.001)
        assert result == "BUY"

    def test_sell_on_strong_down_move(self):
        # 5% down move
        result = _compute_signal(95.0, 100.0, 0.001)
        assert result == "SELL"

    def test_hold_small_up(self):
        # 0.01% up, threshold = max(0.001*0.5, 0.0005) = 0.0005
        result = _compute_signal(100.01, 100.0, 0.001)
        assert result == "HOLD"

    def test_hold_small_down(self):
        result = _compute_signal(99.99, 100.0, 0.001)
        assert result == "HOLD"

    def test_buy_exactly_at_threshold(self):
        prev = 100.0
        # threshold = max(0.001 * 0.5, 0.0005) = 0.0005
        # need return > 0.0005 → price > 100.05
        price = prev * (1 + 0.001)  # 0.1% move
        result = _compute_signal(price, prev, 0.001)
        assert result == "BUY"

    def test_returns_string(self):
        r = _compute_signal(100.0, 100.0, 0.01)
        assert isinstance(r, str)
        assert r in ("BUY", "SELL", "HOLD")


# ─── _risk_check ──────────────────────────────────────────────────────────────


class TestRiskCheck:
    def test_hold_signal_rejected(self):
        r = _risk_check("HOLD", 100.0, 10_000.0, 0.0)
        assert r.approved is False
        assert "HOLD" in r.reason
        assert r.adjusted_size == 0.0

    def test_buy_approved(self):
        r = _risk_check("BUY", 100.0, 10_000.0, 0.0)
        assert r.approved is True
        assert r.adjusted_size > 0

    def test_sell_approved(self):
        r = _risk_check("SELL", 100.0, 10_000.0, 0.0)
        assert r.approved is True

    def test_size_respects_max_pct(self):
        r = _risk_check("BUY", 100.0, 10_000.0, 0.0, max_position_pct=0.05)
        # size = 10000 * 0.05 / 100 = 5.0
        assert r.adjusted_size == pytest.approx(5.0)

    def test_drawdown_limit_triggers(self):
        # Short position + BUY → unrealized loss
        # capital=1000, position=-10 units, price=100 → loss=1000=100%
        r = _risk_check("BUY", 100.0, 1000.0, -100.0, max_drawdown_pct=0.20)
        assert r.approved is False
        assert "Drawdown" in r.reason or "limit" in r.reason.lower()

    def test_zero_price_safe(self):
        """Ensure we don't divide by zero on zero price."""
        r = _risk_check("BUY", 0.0, 10_000.0, 0.0)
        # adjusted_size = max_notional / 0 → 0.0 — no crash
        assert isinstance(r.approved, bool)

    def test_returns_risk_check(self):
        r = _risk_check("BUY", 100.0, 10_000.0, 0.0)
        assert isinstance(r, RiskCheck)


# ─── PipelineStep ─────────────────────────────────────────────────────────────


class TestPipelineStep:
    def test_construct(self):
        step = PipelineStep(
            tick=1, symbol="BTC/USD", price=65000.0, prev_price=64800.0,
            volatility=0.003, signal="BUY", risk_approved=True,
            risk_reason="ok", size=0.1, notional=6500.0,
            ledger_entry={"tx_hash": "0x" + "a" * 64},
            elapsed_ms=1.2,
        )
        assert step.tick == 1
        assert step.signal == "BUY"


# ─── DemoResult ───────────────────────────────────────────────────────────────


class TestDemoResult:
    def _make_result(self) -> DemoResult:
        return DemoPipeline(ticks=5, seed=42).run()

    def test_to_json_is_valid(self):
        r = self._make_result()
        parsed = json.loads(r.to_json())
        assert "agent_id" in parsed
        assert "steps" in parsed
        assert "summary" in parsed

    def test_to_json_steps_count(self):
        r = self._make_result()
        parsed = json.loads(r.to_json())
        assert len(parsed["steps"]) == 5

    def test_trace_is_string(self):
        r = self._make_result()
        assert isinstance(r.trace, str)
        assert len(r.trace) > 50

    def test_trace_contains_header(self):
        r = self._make_result()
        assert "ERC-8004" in r.trace

    def test_trace_contains_symbol(self):
        r = self._make_result()
        assert "BTC/USD" in r.trace

    def test_trace_contains_summary(self):
        r = self._make_result()
        assert "Summary" in r.trace
        assert "Ticks processed" in r.trace

    def test_elapsed_positive(self):
        r = self._make_result()
        assert r.elapsed_total_ms >= 0


# ─── DemoPipeline ─────────────────────────────────────────────────────────────


class TestDemoPipeline:
    def test_default_ticks(self):
        dp = DemoPipeline()
        assert dp.ticks == 10

    def test_default_symbol(self):
        dp = DemoPipeline()
        assert dp.symbol == "BTC/USD"

    def test_custom_ticks(self):
        dp = DemoPipeline(ticks=25)
        assert dp.ticks == 25

    def test_custom_symbol(self):
        dp = DemoPipeline(symbol="ETH/USD")
        assert dp.symbol == "ETH/USD"

    def test_custom_agent_id(self):
        dp = DemoPipeline(agent_id="my-agent")
        assert dp.agent_id == "my-agent"

    def test_auto_agent_id(self):
        dp = DemoPipeline()
        assert dp.agent_id.startswith("demo-agent-")

    def test_btc_initial_price(self):
        dp = DemoPipeline(symbol="BTC/USD")
        assert dp.initial_price > 0
        assert dp.initial_price > 10_000.0

    def test_eth_initial_price(self):
        dp = DemoPipeline(symbol="ETH/USD")
        assert dp.initial_price > 0
        assert dp.initial_price < 10_000.0

    def test_run_returns_demo_result(self):
        r = DemoPipeline(ticks=5).run()
        assert isinstance(r, DemoResult)

    def test_run_steps_count(self):
        r = DemoPipeline(ticks=10, seed=42).run()
        assert len(r.steps) == 10

    def test_run_deterministic(self):
        r1 = DemoPipeline(ticks=10, seed=42, agent_id="x").run()
        r2 = DemoPipeline(ticks=10, seed=42, agent_id="x").run()
        assert r1.summary["price_start"] == r2.summary["price_start"]
        assert r1.summary["price_end"] == r2.summary["price_end"]

    def test_run_in_memory_ledger(self):
        dp = DemoPipeline(ticks=10, seed=42, db_path=":memory:")
        r = dp.run()
        assert r.summary["trades_executed"] >= 0

    def test_run_trades_logged_to_ledger(self):
        dp = DemoPipeline(ticks=10, seed=42, db_path=":memory:")
        r = dp.run()
        count = dp.ledger.count()
        assert count == r.summary["trades_executed"]

    def test_run_summary_keys(self):
        r = DemoPipeline(ticks=5, seed=1).run()
        for k in ["ticks", "trades_executed", "trades_rejected",
                  "total_notional", "buy_count", "sell_count",
                  "final_capital", "price_start", "price_end", "price_return_pct"]:
            assert k in r.summary

    def test_run_ticks_plus_rejected_equals_ticks(self):
        r = DemoPipeline(ticks=10, seed=42).run()
        s = r.summary
        assert s["trades_executed"] + s["trades_rejected"] == 10

    def test_sol_symbol(self):
        dp = DemoPipeline(symbol="SOL/USD", ticks=5, seed=7)
        r = dp.run()
        assert r.symbol == "SOL/USD"

    def test_btc_short_symbol(self):
        dp = DemoPipeline(symbol="BTC", ticks=5, seed=7)
        assert dp.initial_price > 10_000.0

    def test_custom_initial_price(self):
        dp = DemoPipeline(symbol="FAKE/USD", initial_price=999.0, ticks=3, seed=1)
        assert dp.initial_price == 999.0

    def test_large_tick_count(self):
        r = DemoPipeline(ticks=100, seed=42).run()
        assert len(r.steps) == 100

    def test_single_tick(self):
        r = DemoPipeline(ticks=1, seed=42).run()
        assert len(r.steps) == 1

    def test_steps_have_valid_signals(self):
        r = DemoPipeline(ticks=20, seed=5).run()
        for step in r.steps:
            assert step.signal in ("BUY", "SELL", "HOLD")

    def test_executed_steps_have_ledger_entry(self):
        r = DemoPipeline(ticks=20, seed=42).run()
        for step in r.steps:
            if step.risk_approved:
                assert step.ledger_entry is not None
                assert "tx_hash" in step.ledger_entry

    def test_rejected_steps_have_no_ledger_entry(self):
        r = DemoPipeline(ticks=20, seed=42).run()
        for step in r.steps:
            if not step.risk_approved:
                assert step.ledger_entry is None

    def test_ledger_property_creates_once(self):
        dp = DemoPipeline(ticks=3, seed=1)
        l1 = dp.ledger
        l2 = dp.ledger
        assert l1 is l2

    def test_to_json_parseable(self):
        r = DemoPipeline(ticks=5, seed=42).run()
        j = r.to_json()
        parsed = json.loads(j)
        assert parsed["ticks"] == 5

    def test_file_db_path(self, tmp_path):
        db_path = str(tmp_path / "demo.db")
        dp = DemoPipeline(ticks=10, seed=42, db_path=db_path)
        dp.run()
        # Verify trades were written to file
        tl = TradeLedger(db_path)
        assert tl.count() > 0
        tl.close()


# ─── CLI: _parse_args ─────────────────────────────────────────────────────────


class TestParseArgs:
    def test_defaults(self):
        args = _parse_args([])
        assert args.ticks == 10
        assert args.symbol == "BTC/USD"
        assert args.capital == 10_000.0
        assert args.seed == 42
        assert args.as_json is False
        assert args.db == ":memory:"

    def test_ticks_flag(self):
        args = _parse_args(["--ticks", "25"])
        assert args.ticks == 25

    def test_symbol_flag(self):
        args = _parse_args(["--symbol", "ETH/USD"])
        assert args.symbol == "ETH/USD"

    def test_capital_flag(self):
        args = _parse_args(["--capital", "50000"])
        assert args.capital == 50_000.0

    def test_seed_flag(self):
        args = _parse_args(["--seed", "123"])
        assert args.seed == 123

    def test_json_flag(self):
        args = _parse_args(["--json"])
        assert args.as_json is True

    def test_db_flag(self):
        args = _parse_args(["--db", "/tmp/x.db"])
        assert args.db == "/tmp/x.db"


# ─── CLI: main() ──────────────────────────────────────────────────────────────


class TestMain:
    def test_main_returns_demo_result(self):
        r = main(["--ticks", "3", "--seed", "42"])
        assert isinstance(r, DemoResult)

    def test_main_ticks_respected(self):
        r = main(["--ticks", "5", "--seed", "1"])
        assert r.ticks == 5

    def test_main_json_mode(self, capsys):
        main(["--ticks", "3", "--seed", "42", "--json"])
        out = capsys.readouterr().out
        parsed = json.loads(out)
        assert "steps" in parsed

    def test_main_text_mode(self, capsys):
        main(["--ticks", "3", "--seed", "42"])
        out = capsys.readouterr().out
        assert "ERC-8004" in out

    def test_main_eth_symbol(self):
        r = main(["--ticks", "3", "--symbol", "ETH/USD", "--seed", "5"])
        assert r.symbol == "ETH/USD"

    def test_main_sol_symbol(self):
        r = main(["--ticks", "3", "--symbol", "SOL/USD", "--seed", "5"])
        assert r.symbol == "SOL/USD"
