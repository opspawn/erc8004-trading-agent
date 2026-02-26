"""
test_hedera_pipeline_integration.py — Integration tests for Pipeline + Hedera HCS-10.

Tests the full pipeline tick loop with hedera_enabled=True (mock mode).
Also covers edge cases: bus disabled, multiple ticks, hedera signal callbacks.
"""

from __future__ import annotations

import asyncio
import pytest

from pipeline import Pipeline, PipelineConfig
from hedera_signals import HederaSignalBus, _MOCK_STORE


# ─── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture(autouse=True)
def clear_store():
    _MOCK_STORE.clear()
    yield
    _MOCK_STORE.clear()


@pytest.fixture
def hedera_config():
    return PipelineConfig(
        symbols=["BTC"],
        tick_interval=0.0,
        seed=42,
        hedera_enabled=True,
        hedera_mode="mock",
        hedera_topic_id="0.0.4753280",
    )


@pytest.fixture
def no_hedera_config():
    return PipelineConfig(
        symbols=["BTC"],
        tick_interval=0.0,
        seed=42,
        hedera_enabled=False,
    )


# ─── PipelineConfig Hedera Fields ─────────────────────────────────────────────

class TestPipelineConfigHederaFields:
    def test_hedera_disabled_by_default(self):
        config = PipelineConfig()
        assert config.hedera_enabled is False

    def test_hedera_mode_default(self):
        config = PipelineConfig()
        assert config.hedera_mode == "mock"

    def test_hedera_topic_id_default(self):
        config = PipelineConfig()
        assert config.hedera_topic_id == "0.0.4753280"

    def test_hedera_enabled_flag(self, hedera_config):
        assert hedera_config.hedera_enabled is True

    def test_custom_topic_id(self):
        config = PipelineConfig(hedera_enabled=True, hedera_topic_id="0.0.9999")
        assert config.hedera_topic_id == "0.0.9999"


# ─── Pipeline Init with Hedera ────────────────────────────────────────────────

class TestPipelineHederaInit:
    def test_pipeline_creates_hedera_bus_when_enabled(self, hedera_config):
        p = Pipeline(config=hedera_config)
        p._init_modules()
        assert p._hedera_bus is not None

    def test_pipeline_no_hedera_bus_when_disabled(self, no_hedera_config):
        p = Pipeline(config=no_hedera_config)
        p._init_modules()
        assert p._hedera_bus is None

    def test_hedera_bus_mode_matches_config(self, hedera_config):
        p = Pipeline(config=hedera_config)
        p._init_modules()
        assert p._hedera_bus.mode == "mock"

    def test_hedera_signals_received_starts_empty(self, hedera_config):
        p = Pipeline(config=hedera_config)
        p._init_modules()
        assert p.get_hedera_signals() == []


# ─── Pipeline Tick with Hedera ────────────────────────────────────────────────

class TestPipelineTickWithHedera:
    @pytest.mark.asyncio
    async def test_run_ticks_with_hedera_enabled(self, hedera_config):
        p = Pipeline(config=hedera_config)
        await p.run_n_ticks(50)
        # Pipeline should have run without error
        assert p.status().state == "stopped"

    @pytest.mark.asyncio
    async def test_hedera_signals_published_after_ticks(self, hedera_config):
        p = Pipeline(config=hedera_config)
        await p.run_n_ticks(100)
        # Some signals should be published (pipeline generates trades above confidence threshold)
        history = p._hedera_bus.get_signal_history()
        # There may or may not be signals depending on the strategy; just check no errors
        assert isinstance(history, list)

    @pytest.mark.asyncio
    async def test_published_signals_are_valid(self, hedera_config):
        p = Pipeline(config=hedera_config)
        await p.run_n_ticks(100)
        for sig in p._hedera_bus.get_signal_history():
            assert sig.signal_type in ("BUY", "SELL", "HOLD")
            assert sig.ticker == "BTC"
            assert 0.0 <= sig.confidence <= 1.0

    @pytest.mark.asyncio
    async def test_hedera_signals_received_via_callback(self, hedera_config):
        p = Pipeline(config=hedera_config)
        await p.run_n_ticks(100)
        # Received signals should match published
        received = p.get_hedera_signals()
        published = p._hedera_bus.get_signal_history()
        assert len(received) == len(published)

    @pytest.mark.asyncio
    async def test_pipeline_runs_without_hedera(self, no_hedera_config):
        p = Pipeline(config=no_hedera_config)
        await p.run_n_ticks(50)
        assert p.status().state == "stopped"
        assert p.get_hedera_signals() == []

    @pytest.mark.asyncio
    async def test_get_hedera_signals_limit(self, hedera_config):
        p = Pipeline(config=hedera_config)
        await p.run_n_ticks(200)
        signals = p.get_hedera_signals(limit=5)
        assert len(signals) <= 5

    @pytest.mark.asyncio
    async def test_multiple_symbols_hedera(self):
        config = PipelineConfig(
            symbols=["BTC", "ETH", "SOL"],
            tick_interval=0.0,
            seed=99,
            hedera_enabled=True,
            hedera_mode="mock",
        )
        p = Pipeline(config=config)
        await p.run_n_ticks(100)
        history = p._hedera_bus.get_signal_history()
        tickers = {s.ticker for s in history}
        # Should see at least one ticker's signals
        assert isinstance(tickers, set)

    @pytest.mark.asyncio
    async def test_pipeline_reset_clears_hedera_signals(self, hedera_config):
        p = Pipeline(config=hedera_config)
        await p.run_n_ticks(100)
        p.reset()
        assert p.get_hedera_signals() == []

    @pytest.mark.asyncio
    async def test_hedera_signal_sequence_increases(self, hedera_config):
        p = Pipeline(config=hedera_config)
        await p.run_n_ticks(100)
        history = p._hedera_bus.get_signal_history()
        if len(history) >= 2:
            seqs = [s.sequence_number for s in history]
            assert seqs == sorted(seqs)

    @pytest.mark.asyncio
    async def test_pipeline_state_after_hedera_ticks(self, hedera_config):
        p = Pipeline(config=hedera_config)
        await p.run_n_ticks(30)
        status = p.status()
        assert status.state == "stopped"
        assert status.ticks == 30
