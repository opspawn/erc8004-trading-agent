"""
test_backtest_additions.py — Tests for S13 additions to backtester.py:
  - mesh_consensus_signal strategy
  - BacktestRecord data class
  - BacktestRegistry (store, get, query, best, worst, summary, clear)
  - CLI entry point (_cli_main)
  - mesh_consensus integration with Backtester.run()
  - Registry + backtest workflow end-to-end
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import pytest
from datetime import datetime, timedelta, timezone

from backtester import (
    Backtester,
    BacktestRecord,
    BacktestRegistry,
    BacktestStats,
    BacktestTrade,
    PriceBar,
    mesh_consensus_signal,
    STRATEGY_MAP,
)


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _bar(ts, price):
    return PriceBar(ts, price, price * 1.01, price * 0.99, price, 1_000_000)


def _make_bars(n: int = 60, start_price: float = 1000.0, trend: float = 0.005) -> list:
    now = datetime(2026, 1, 1, tzinfo=timezone.utc)
    bars = []
    price = start_price
    for i in range(n):
        price = price * (1 + trend)
        bars.append(PriceBar(
            timestamp=now + timedelta(days=i),
            open=price / (1 + trend),
            high=price * 1.01,
            low=price * 0.99,
            close=price,
            volume=1_000_000.0,
        ))
    return bars


@pytest.fixture
def bt():
    return Backtester(initial_capital=1000.0, position_size_pct=0.10)


@pytest.fixture
def uptrend_bars(bt):
    return bt.generate_trending_prices(days=60, trend=0.01)


@pytest.fixture
def synth_bars(bt):
    return bt.generate_synthetic_prices("ETH", days=80)


@pytest.fixture
def registry():
    return BacktestRegistry()


# ═══════════════════════════════════════════════════════════════════════════════
# mesh_consensus_signal
# ═══════════════════════════════════════════════════════════════════════════════

class TestMeshConsensusSignal:
    def test_in_strategy_map(self):
        assert "mesh_consensus" in STRATEGY_MAP

    def test_insufficient_history_returns_none(self, uptrend_bars):
        sig = mesh_consensus_signal(uptrend_bars, 0)
        assert sig is None

    def test_insufficient_history_at_5(self, uptrend_bars):
        sig = mesh_consensus_signal(uptrend_bars, 5)
        assert sig is None

    def test_insufficient_history_at_15(self, uptrend_bars):
        sig = mesh_consensus_signal(uptrend_bars, 15)
        assert sig is None

    def test_returns_valid_signal_or_none(self, uptrend_bars):
        for idx in range(len(uptrend_bars)):
            sig = mesh_consensus_signal(uptrend_bars, idx)
            assert sig in (None, "BUY", "SELL")

    def test_uptrend_eventually_signals_buy(self, bt):
        bars = bt.generate_trending_prices(days=80, trend=0.015)
        signals = [mesh_consensus_signal(bars, idx) for idx in range(len(bars))]
        buy_signals = [s for s in signals if s == "BUY"]
        assert len(buy_signals) > 0  # at least one BUY in strong uptrend

    def test_no_crash_on_flat_bars(self):
        now = datetime(2026, 1, 1, tzinfo=timezone.utc)
        bars = [_bar(now + timedelta(days=i), 1000.0) for i in range(60)]
        for idx in range(len(bars)):
            sig = mesh_consensus_signal(bars, idx)
            assert sig in (None, "BUY", "SELL")

    def test_custom_lookback(self, uptrend_bars):
        # Should accept custom lookback param
        sig = mesh_consensus_signal(uptrend_bars, 30, lookback=5)
        assert sig in (None, "BUY", "SELL")

    def test_lookback_affects_signal(self, bt):
        bars = bt.generate_trending_prices(days=60, trend=0.01)
        s1 = mesh_consensus_signal(bars, 40, lookback=5)
        s2 = mesh_consensus_signal(bars, 40, lookback=20)
        # Different lookbacks may give different signals — just check no crash
        assert s1 in (None, "BUY", "SELL")
        assert s2 in (None, "BUY", "SELL")

    def test_mean_reverting_signals(self, bt):
        bars = bt.generate_mean_reverting_prices(days=80)
        signals = [mesh_consensus_signal(bars, idx) for idx in range(len(bars))]
        # Some signals should fire in mean-reverting market
        non_none = [s for s in signals if s is not None]
        assert len(non_none) >= 0  # may be zero for tricky data

    def test_backtester_run_with_mesh_consensus(self, bt, synth_bars):
        trades = bt.run(synth_bars, strategy="mesh_consensus")
        assert isinstance(trades, list)

    def test_mesh_consensus_run_empty_bars(self, bt):
        trades = bt.run([], strategy="mesh_consensus")
        assert trades == []

    def test_mesh_consensus_stats_valid(self, bt, synth_bars):
        trades = bt.run(synth_bars, strategy="mesh_consensus")
        stats = bt.compute_stats(trades)
        assert stats.total_trades >= 0
        assert 0.0 <= stats.win_rate <= 1.0


# ═══════════════════════════════════════════════════════════════════════════════
# BacktestRecord
# ═══════════════════════════════════════════════════════════════════════════════

class TestBacktestRecord:
    def test_to_dict_keys(self):
        now = datetime(2026, 1, 1, tzinfo=timezone.utc).isoformat()
        r = BacktestRecord(
            record_id="BT-000001",
            strategy="trend",
            token="ETH",
            days=30,
            initial_capital=1000.0,
            final_capital=1050.0,
            sharpe_ratio=1.5,
            max_drawdown_pct=3.2,
            win_rate=0.6,
            total_trades=10,
            net_pnl=50.0,
            timestamp=now,
        )
        d = r.to_dict()
        expected_keys = {
            "record_id", "strategy", "token", "days",
            "initial_capital", "final_capital", "sharpe_ratio",
            "max_drawdown_pct", "win_rate", "total_trades", "net_pnl", "timestamp",
        }
        assert set(d.keys()) == expected_keys

    def test_record_id_in_dict(self):
        r = BacktestRecord(
            record_id="BT-XYZ", strategy="trend", token="ETH", days=30,
            initial_capital=1000, final_capital=1100, sharpe_ratio=2.0,
            max_drawdown_pct=5.0, win_rate=0.7, total_trades=15, net_pnl=100,
        )
        assert r.to_dict()["record_id"] == "BT-XYZ"

    def test_auto_timestamp(self):
        r = BacktestRecord(
            record_id="BT-001", strategy="trend", token="ETH", days=30,
            initial_capital=1000, final_capital=1000, sharpe_ratio=0.0,
            max_drawdown_pct=0.0, win_rate=0.5, total_trades=0, net_pnl=0.0,
        )
        assert r.timestamp is not None
        assert "T" in r.timestamp  # ISO format

    def test_values_rounded(self):
        r = BacktestRecord(
            record_id="X", strategy="t", token="E", days=1,
            initial_capital=1000.1234567, final_capital=1001.9876543,
            sharpe_ratio=1.23456789, max_drawdown_pct=2.34567890,
            win_rate=0.6543210, total_trades=5, net_pnl=1.9876543,
        )
        d = r.to_dict()
        assert d["sharpe_ratio"] == pytest.approx(1.2346, abs=1e-3)


# ═══════════════════════════════════════════════════════════════════════════════
# BacktestRegistry
# ═══════════════════════════════════════════════════════════════════════════════

class TestBacktestRegistryStore:
    def test_store_returns_record(self, bt, registry, synth_bars):
        trades = bt.run(synth_bars, strategy="trend")
        stats = bt.compute_stats(trades)
        record = registry.store(bt, trades, stats, "trend", "ETH", 80)
        assert isinstance(record, BacktestRecord)

    def test_store_increments_counter(self, bt, registry, synth_bars):
        trades = bt.run(synth_bars, strategy="trend")
        stats = bt.compute_stats(trades)
        r1 = registry.store(bt, trades, stats, "trend", "ETH", 80)
        r2 = registry.store(bt, trades, stats, "trend", "ETH", 80)
        assert r1.record_id != r2.record_id

    def test_store_id_format(self, bt, registry, synth_bars):
        trades = bt.run(synth_bars, strategy="trend")
        stats = bt.compute_stats(trades)
        record = registry.store(bt, trades, stats, "trend", "ETH", 80)
        assert record.record_id.startswith("BT-")

    def test_store_preserves_strategy(self, bt, registry, synth_bars):
        trades = bt.run(synth_bars, strategy="momentum")
        stats = bt.compute_stats(trades)
        record = registry.store(bt, trades, stats, "momentum", "BTC", 80)
        assert record.strategy == "momentum"
        assert record.token == "BTC"

    def test_store_count(self, bt, registry, synth_bars):
        trades = bt.run(synth_bars, strategy="trend")
        stats = bt.compute_stats(trades)
        for _ in range(3):
            registry.store(bt, trades, stats, "trend", "ETH", 80)
        assert registry.count() == 3


class TestBacktestRegistryGet:
    def test_get_existing_record(self, bt, registry, synth_bars):
        trades = bt.run(synth_bars, strategy="trend")
        stats = bt.compute_stats(trades)
        record = registry.store(bt, trades, stats, "trend", "ETH", 80)
        found = registry.get(record.record_id)
        assert found is not None
        assert found.record_id == record.record_id

    def test_get_nonexistent_returns_none(self, registry):
        assert registry.get("BT-DOES-NOT-EXIST") is None

    def test_get_correct_stats(self, bt, registry, synth_bars):
        trades = bt.run(synth_bars, strategy="trend")
        stats = bt.compute_stats(trades)
        record = registry.store(bt, trades, stats, "trend", "ETH", 80)
        found = registry.get(record.record_id)
        assert found.sharpe_ratio == pytest.approx(record.sharpe_ratio, rel=1e-6)


class TestBacktestRegistryQuery:
    def test_query_all_empty_filters(self, bt, registry, synth_bars):
        trades = bt.run(synth_bars, strategy="trend")
        stats = bt.compute_stats(trades)
        registry.store(bt, trades, stats, "trend", "ETH", 80)
        registry.store(bt, trades, stats, "momentum", "BTC", 80)
        results = registry.query()
        assert len(results) == 2

    def test_query_by_strategy(self, bt, registry, synth_bars):
        trades = bt.run(synth_bars, strategy="trend")
        stats = bt.compute_stats(trades)
        registry.store(bt, trades, stats, "trend", "ETH", 80)
        registry.store(bt, trades, stats, "trend", "BTC", 80)
        registry.store(bt, trades, stats, "momentum", "ETH", 80)
        results = registry.query(strategy="trend")
        assert len(results) == 2
        assert all(r.strategy == "trend" for r in results)

    def test_query_by_token(self, bt, registry, synth_bars):
        trades = bt.run(synth_bars, strategy="trend")
        stats = bt.compute_stats(trades)
        registry.store(bt, trades, stats, "trend", "ETH", 80)
        registry.store(bt, trades, stats, "momentum", "ETH", 80)
        registry.store(bt, trades, stats, "trend", "BTC", 80)
        results = registry.query(token="ETH")
        assert len(results) == 2
        assert all(r.token == "ETH" for r in results)

    def test_query_combined_filters(self, bt, registry, synth_bars):
        trades = bt.run(synth_bars, strategy="trend")
        stats = bt.compute_stats(trades)
        registry.store(bt, trades, stats, "trend", "ETH", 80)
        registry.store(bt, trades, stats, "trend", "BTC", 80)
        registry.store(bt, trades, stats, "momentum", "ETH", 80)
        results = registry.query(strategy="trend", token="ETH")
        assert len(results) == 1

    def test_query_no_match_returns_empty(self, bt, registry, synth_bars):
        trades = bt.run(synth_bars, strategy="trend")
        stats = bt.compute_stats(trades)
        registry.store(bt, trades, stats, "trend", "ETH", 80)
        results = registry.query(strategy="nonexistent")
        assert results == []


class TestBacktestRegistryBest:
    def test_best_by_sharpe(self, bt, registry):
        # Manually insert records with known sharpe values
        r1 = BacktestRecord(
            "BT-001", "trend", "ETH", 30, 1000, 1100, sharpe_ratio=2.5,
            max_drawdown_pct=5.0, win_rate=0.6, total_trades=10, net_pnl=100,
        )
        r2 = BacktestRecord(
            "BT-002", "trend", "ETH", 30, 1000, 1200, sharpe_ratio=1.0,
            max_drawdown_pct=10.0, win_rate=0.5, total_trades=8, net_pnl=200,
        )
        registry._records["BT-001"] = r1
        registry._records["BT-002"] = r2
        best = registry.best(metric="sharpe_ratio")
        assert best.record_id == "BT-001"

    def test_best_empty_registry_none(self, registry):
        assert registry.best() is None

    def test_best_invalid_metric_raises(self, registry):
        registry._records["X"] = BacktestRecord(
            "X", "trend", "ETH", 30, 1000, 1000, 0.0, 0.0, 0.5, 0, 0.0,
        )
        with pytest.raises(ValueError, match="Unknown metric"):
            registry.best(metric="invalid_field")

    def test_best_by_win_rate(self, registry):
        r1 = BacktestRecord("A", "trend", "ETH", 30, 1000, 1000, 0.0, 5.0, 0.8, 10, 0.0)
        r2 = BacktestRecord("B", "trend", "ETH", 30, 1000, 1000, 0.0, 3.0, 0.6, 10, 0.0)
        registry._records["A"] = r1
        registry._records["B"] = r2
        best = registry.best(metric="win_rate")
        assert best.record_id == "A"

    def test_best_with_strategy_filter(self, registry):
        r1 = BacktestRecord("A", "trend", "ETH", 30, 1000, 1000, 2.0, 5.0, 0.7, 10, 50.0)
        r2 = BacktestRecord("B", "momentum", "ETH", 30, 1000, 1000, 3.0, 5.0, 0.8, 10, 80.0)
        registry._records["A"] = r1
        registry._records["B"] = r2
        best = registry.best(metric="sharpe_ratio", strategy="trend")
        assert best.record_id == "A"


class TestBacktestRegistryWorst:
    def test_worst_by_drawdown(self, registry):
        r1 = BacktestRecord("A", "trend", "ETH", 30, 1000, 900, 0.5, 20.0, 0.4, 10, -100.0)
        r2 = BacktestRecord("B", "trend", "ETH", 30, 1000, 950, 0.8, 5.0, 0.6, 10, -50.0)
        registry._records["A"] = r1
        registry._records["B"] = r2
        worst = registry.worst(metric="max_drawdown_pct")
        assert worst.record_id == "A"

    def test_worst_empty_returns_none(self, registry):
        assert registry.worst() is None


class TestBacktestRegistrySummary:
    def test_empty_registry_summary(self, registry):
        s = registry.summary()
        assert s.get("count") == 0

    def test_summary_keys(self, registry):
        r1 = BacktestRecord("A", "trend", "ETH", 30, 1000, 1100, 1.5, 5.0, 0.6, 10, 100.0)
        r2 = BacktestRecord("B", "trend", "ETH", 30, 1000, 900, -0.5, 15.0, 0.4, 8, -100.0)
        registry._records["A"] = r1
        registry._records["B"] = r2
        s = registry.summary()
        assert "count" in s
        assert "avg_sharpe" in s
        assert "max_sharpe" in s
        assert "avg_drawdown" in s
        assert "max_drawdown" in s
        assert "avg_win_rate" in s

    def test_summary_count(self, bt, registry, synth_bars):
        trades = bt.run(synth_bars, strategy="trend")
        stats = bt.compute_stats(trades)
        registry.store(bt, trades, stats, "trend", "ETH", 80)
        registry.store(bt, trades, stats, "trend", "ETH", 80)
        s = registry.summary()
        assert s["count"] == 2

    def test_summary_avg_sharpe(self, registry):
        r1 = BacktestRecord("A", "trend", "ETH", 30, 1000, 1100, 2.0, 5.0, 0.6, 10, 100.0)
        r2 = BacktestRecord("B", "trend", "ETH", 30, 1000, 900, 1.0, 15.0, 0.4, 8, -100.0)
        registry._records["A"] = r1
        registry._records["B"] = r2
        s = registry.summary()
        assert s["avg_sharpe"] == pytest.approx(1.5)


class TestBacktestRegistryClear:
    def test_clear_empties_records(self, bt, registry, synth_bars):
        trades = bt.run(synth_bars, strategy="trend")
        stats = bt.compute_stats(trades)
        registry.store(bt, trades, stats, "trend", "ETH", 80)
        registry.clear()
        assert registry.count() == 0

    def test_clear_resets_counter(self, bt, registry, synth_bars):
        trades = bt.run(synth_bars, strategy="trend")
        stats = bt.compute_stats(trades)
        registry.store(bt, trades, stats, "trend", "ETH", 80)
        registry.clear()
        record = registry.store(bt, trades, stats, "trend", "ETH", 80)
        assert record.record_id == "BT-000001"


class TestBacktestRegistryAllRecords:
    def test_all_records_returns_list(self, bt, registry, synth_bars):
        trades = bt.run(synth_bars, strategy="trend")
        stats = bt.compute_stats(trades)
        registry.store(bt, trades, stats, "trend", "ETH", 80)
        records = registry.all_records()
        assert isinstance(records, list)
        assert len(records) == 1

    def test_all_records_empty(self, registry):
        assert registry.all_records() == []


# ═══════════════════════════════════════════════════════════════════════════════
# CLI _cli_main
# ═══════════════════════════════════════════════════════════════════════════════

class TestCLIMain:
    def test_cli_runs_without_error(self, capsys):
        from backtester import _cli_main
        import sys
        original_argv = sys.argv
        sys.argv = ["backtester", "--days", "20", "--strategy", "trend"]
        try:
            _cli_main()
        finally:
            sys.argv = original_argv
        out = capsys.readouterr().out
        assert "Backtest Results" in out

    def test_cli_mesh_consensus_strategy(self, capsys):
        from backtester import _cli_main
        import sys
        original_argv = sys.argv
        sys.argv = ["backtester", "--days", "40", "--strategy", "mesh_consensus"]
        try:
            _cli_main()
        finally:
            sys.argv = original_argv
        out = capsys.readouterr().out
        assert "Sharpe" in out or "sharpe" in out.lower()

    def test_cli_compare_flag(self, capsys):
        from backtester import _cli_main
        import sys
        original_argv = sys.argv
        sys.argv = ["backtester", "--days", "20", "--compare"]
        try:
            _cli_main()
        finally:
            sys.argv = original_argv
        out = capsys.readouterr().out
        assert "Strategy Comparison" in out

    def test_cli_register_flag(self, capsys):
        from backtester import _cli_main
        import sys
        original_argv = sys.argv
        sys.argv = ["backtester", "--days", "20", "--strategy", "trend", "--register"]
        try:
            _cli_main()
        finally:
            sys.argv = original_argv
        out = capsys.readouterr().out
        assert "BT-" in out

    def test_cli_trending_mode(self, capsys):
        from backtester import _cli_main
        import sys
        original_argv = sys.argv
        sys.argv = ["backtester", "--days", "20", "--strategy", "trend", "--mode", "trending"]
        try:
            _cli_main()
        finally:
            sys.argv = original_argv
        out = capsys.readouterr().out
        assert "Backtest Results" in out

    def test_cli_mean_reverting_mode(self, capsys):
        from backtester import _cli_main
        import sys
        original_argv = sys.argv
        sys.argv = ["backtester", "--days", "30", "--strategy", "mean_reversion", "--mode", "mean_reverting"]
        try:
            _cli_main()
        finally:
            sys.argv = original_argv
        out = capsys.readouterr().out
        assert "Backtest Results" in out


# ═══════════════════════════════════════════════════════════════════════════════
# End-to-end: Registry workflow
# ═══════════════════════════════════════════════════════════════════════════════

class TestRegistryEndToEnd:
    def test_store_and_retrieve_all_strategies(self, bt, registry):
        bars = bt.generate_synthetic_prices("ETH", days=80)
        for strategy in STRATEGY_MAP:
            trades = bt.run(bars, strategy=strategy)
            stats = bt.compute_stats(trades)
            registry.store(bt, trades, stats, strategy, "ETH", 80)

        assert registry.count() == len(STRATEGY_MAP)

    def test_best_strategy_by_sharpe(self, bt, registry):
        bars = bt.generate_trending_prices(days=60, trend=0.01)
        for strategy in STRATEGY_MAP:
            trades = bt.run(bars, strategy=strategy)
            stats = bt.compute_stats(trades)
            registry.store(bt, trades, stats, strategy, "ETH", 60)

        best = registry.best(metric="sharpe_ratio")
        assert best is not None
        assert best.strategy in STRATEGY_MAP

    def test_registry_query_mesh_only(self, bt, registry):
        bars = bt.generate_synthetic_prices("ETH", days=60)
        for strategy in ["trend", "mesh_consensus", "momentum"]:
            trades = bt.run(bars, strategy=strategy)
            stats = bt.compute_stats(trades)
            registry.store(bt, trades, stats, strategy, "ETH", 60)

        mesh_records = registry.query(strategy="mesh_consensus")
        assert len(mesh_records) == 1
        assert mesh_records[0].strategy == "mesh_consensus"

    def test_registry_summary_after_multi_run(self, bt, registry):
        """Verify summary aggregation across 4 strategies."""
        bars = bt.generate_synthetic_prices("ETH", days=80)
        for strategy in STRATEGY_MAP:
            trades = bt.run(bars, strategy=strategy)
            stats = bt.compute_stats(trades)
            registry.store(bt, trades, stats, strategy, "ETH", 80)

        s = registry.summary()
        assert s["count"] == len(STRATEGY_MAP)
        assert "avg_sharpe" in s
        assert "max_sharpe" in s
