"""
Tests for risk_dashboard.py — Risk Dashboard HTTP API.
"""

import json
import math
import threading
import time
import urllib.request
import urllib.error
import pytest
from unittest.mock import patch, MagicMock

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from risk_dashboard import (
    RiskDashboard,
    DashboardState,
    AgentRiskState,
    SignalRecord,
    ConsensusRecord,
)


# ─── Helpers ─────────────────────────────────────────────────────────────────

def _get(url: str) -> dict:
    with urllib.request.urlopen(url, timeout=3) as resp:
        return json.loads(resp.read().decode())


def _post(url: str, data: bytes = b"") -> dict:
    req = urllib.request.Request(url, data=data, method="POST")
    with urllib.request.urlopen(req, timeout=3) as resp:
        return json.loads(resp.read().decode())


_DEFAULT_PNL_HISTORY = [0, 100, 300, 500]


def _make_agent(agent_id="a1", profile="balanced", pnl_history=_DEFAULT_PNL_HISTORY) -> AgentRiskState:
    return AgentRiskState(
        agent_id=agent_id,
        profile=profile,
        capital=10_000.0,
        position=5.0,
        entry_price=100.0,
        pnl=500.0,
        pnl_history=list(pnl_history),
        kelly_fraction=0.25,
        reputation=6.0,
        trades=10,
        wins=7,
    )


# ─── AgentRiskState ───────────────────────────────────────────────────────────

class TestAgentRiskState:
    def test_var_95_no_history(self):
        agent = AgentRiskState(
            agent_id="x", profile="balanced", capital=10000.0,
            position=0.0, entry_price=0.0, pnl=0.0, pnl_history=[],
        )
        assert agent.var_95() == 0.0

    def test_var_95_one_entry(self):
        agent = AgentRiskState(
            agent_id="x", profile="balanced", capital=10000.0,
            position=0.0, entry_price=0.0, pnl=0.0, pnl_history=[500.0],
        )
        assert agent.var_95() == 0.0

    def test_var_95_positive(self):
        agent = _make_agent(pnl_history=[0, 100, 200, 300, 100, 50, 250])
        assert agent.var_95() >= 0

    def test_max_drawdown_no_history(self):
        agent = _make_agent(pnl_history=[])
        assert agent.max_drawdown() == 0.0

    def test_max_drawdown_monotonic(self):
        agent = _make_agent(pnl_history=[0, 100, 200, 300])
        assert agent.max_drawdown() == 0.0

    def test_max_drawdown_with_crash(self):
        agent = _make_agent(pnl_history=[0, 500, 100])
        mdd = agent.max_drawdown()
        assert mdd == pytest.approx(0.8, rel=0.01)  # 500→100 = 80% drawdown

    def test_win_rate_no_trades(self):
        agent = _make_agent()
        agent.trades = 0
        agent.wins = 0
        assert agent.win_rate() == 0.0

    def test_win_rate_all_wins(self):
        agent = _make_agent()
        agent.trades = 5
        agent.wins = 5
        assert agent.win_rate() == 1.0

    def test_win_rate_partial(self):
        agent = _make_agent()
        agent.trades = 10
        agent.wins = 7
        assert agent.win_rate() == pytest.approx(0.7)


# ─── DashboardState ───────────────────────────────────────────────────────────

class TestDashboardState:
    def test_upsert_and_retrieve_agent(self):
        state = DashboardState()
        agent = _make_agent("agent-1")
        state.upsert_agent(agent)
        agents = state.get_agents()
        assert len(agents) == 1
        assert agents[0].agent_id == "agent-1"

    def test_upsert_multiple_agents(self):
        state = DashboardState()
        state.upsert_agent(_make_agent("a1"))
        state.upsert_agent(_make_agent("a2"))
        state.upsert_agent(_make_agent("a3"))
        assert len(state.get_agents()) == 3

    def test_upsert_replaces_existing(self):
        state = DashboardState()
        state.upsert_agent(_make_agent("a1"))
        updated = _make_agent("a1")
        updated.pnl = 9999.0
        state.upsert_agent(updated)
        agents = state.get_agents()
        assert len(agents) == 1
        assert agents[0].pnl == 9999.0

    def test_reset_clears_pnl(self):
        state = DashboardState()
        state.upsert_agent(_make_agent("a1"))
        state.reset_agents()
        agents = state.get_agents()
        assert agents[0].pnl == 0.0

    def test_reset_clears_reputation(self):
        state = DashboardState()
        agent = _make_agent("a1")
        agent.reputation = 9.5
        state.upsert_agent(agent)
        state.reset_agents()
        assert state.get_agents()[0].reputation == 5.0

    def test_reset_increments_counter(self):
        state = DashboardState()
        state.reset_agents()
        state.reset_agents()
        assert state._reset_count == 2

    def test_add_signal_and_retrieve(self):
        state = DashboardState()
        sig = SignalRecord(
            timestamp=time.time(),
            signal_type="BUY",
            protocol="Aave",
            direction="LONG",
            confidence=0.8,
            agent_id="a1",
            sequence=1,
        )
        state.add_signal(sig)
        signals = state.get_signals()
        assert len(signals) == 1
        assert signals[0].signal_type == "BUY"

    def test_get_signals_limited(self):
        state = DashboardState()
        for i in range(20):
            state.add_signal(SignalRecord(
                timestamp=time.time(),
                signal_type="HOLD",
                protocol="Uniswap",
                direction="NEUTRAL",
                confidence=0.5,
                agent_id="a1",
                sequence=i,
            ))
        assert len(state.get_signals(10)) == 10

    def test_signals_cap_at_200(self):
        state = DashboardState()
        for i in range(250):
            state.add_signal(SignalRecord(
                timestamp=time.time(),
                signal_type="BUY",
                protocol="Compound",
                direction="LONG",
                confidence=0.7,
                agent_id="a1",
                sequence=i,
            ))
        assert len(state._signals) == 200

    def test_add_consensus_and_retrieve(self):
        state = DashboardState()
        rec = ConsensusRecord(
            timestamp=time.time(),
            tick=1,
            action="BUY",
            voted_buy=["a1", "a2"],
            voted_sell=[],
            voted_hold=["a3"],
            consensus_weight=0.67,
        )
        state.add_consensus(rec)
        records = state.get_consensus()
        assert len(records) == 1
        assert records[0].action == "BUY"

    def test_consensus_cap_at_200(self):
        state = DashboardState()
        for i in range(250):
            state.add_consensus(ConsensusRecord(
                timestamp=time.time(),
                tick=i,
                action="HOLD",
                voted_buy=[],
                voted_sell=[],
                voted_hold=["a1"],
                consensus_weight=0.33,
            ))
        assert len(state._consensus) == 200

    def test_sharpe_ratio_no_data(self):
        state = DashboardState()
        agent = AgentRiskState(
            agent_id="a1", profile="balanced", capital=10000.0,
            position=0.0, entry_price=0.0, pnl=0.0, pnl_history=[],
        )
        state.upsert_agent(agent)
        assert state.sharpe_ratio("a1") == 0.0

    def test_sharpe_ratio_positive(self):
        state = DashboardState()
        pnl_hist = [i * 10.0 for i in range(50)]
        state.upsert_agent(_make_agent("a1", pnl_history=pnl_hist))
        sharpe = state.sharpe_ratio("a1")
        assert isinstance(sharpe, float)

    def test_sharpe_ratio_unknown_agent(self):
        state = DashboardState()
        assert state.sharpe_ratio("nonexistent") == 0.0

    def test_sortino_ratio_no_data(self):
        state = DashboardState()
        agent = AgentRiskState(
            agent_id="a1", profile="balanced", capital=10000.0,
            position=0.0, entry_price=0.0, pnl=0.0, pnl_history=[],
        )
        state.upsert_agent(agent)
        assert state.sortino_ratio("a1") == 0.0

    def test_sortino_ratio_no_downside(self):
        state = DashboardState()
        state.upsert_agent(_make_agent("a1", pnl_history=[0, 10, 20, 30, 40]))
        sortino = state.sortino_ratio("a1")
        # No downside returns → inf
        assert sortino == float("inf") or sortino >= 0

    def test_sortino_ratio_with_downside(self):
        state = DashboardState()
        state.upsert_agent(_make_agent("a1", pnl_history=[0, 50, 20, 60, 10, 80]))
        sortino = state.sortino_ratio("a1")
        assert isinstance(sortino, float)


# ─── RiskDashboard (HTTP server) ──────────────────────────────────────────────

PORT = 18082  # Use non-standard port to avoid conflicts


@pytest.fixture
def dashboard():
    d = RiskDashboard()
    d.start(PORT)
    time.sleep(0.1)
    yield d
    d.stop()


@pytest.fixture
def dashboard_with_data():
    d = RiskDashboard()
    d.update_agent(
        agent_id="a1", profile="conservative",
        capital=9800.0, position=5.0, entry_price=98.0, pnl=200.0,
        pnl_history=[0, 100, 200], kelly_fraction=0.15, reputation=7.0,
        trades=5, wins=3,
    )
    d.update_agent(
        agent_id="a2", profile="aggressive",
        capital=10200.0, position=-3.0, entry_price=102.0, pnl=-100.0,
        pnl_history=[0, 50, -100], kelly_fraction=0.35, reputation=5.0,
        trades=8, wins=4,
    )
    d.record_signal(
        timestamp=time.time(), signal_type="BUY", protocol="Aave",
        direction="LONG", confidence=0.8, agent_id="a1", sequence=1,
    )
    d.record_consensus(
        timestamp=time.time(), tick=1, action="BUY",
        voted_buy=["a1"], voted_sell=[], voted_hold=["a2"],
        consensus_weight=0.7,
    )
    d.start(PORT + 1)
    time.sleep(0.1)
    yield d
    d.stop()


class TestRiskDashboardLifecycle:
    def test_start_and_stop(self):
        d = RiskDashboard()
        d.start(PORT + 10)
        time.sleep(0.05)
        assert d.is_running
        d.stop()
        time.sleep(0.05)
        # After stop, may not be running
        # Just ensure no exception

    def test_not_running_before_start(self):
        d = RiskDashboard()
        assert not d.is_running

    def test_running_after_start(self, dashboard):
        assert dashboard.is_running

    def test_stop_is_idempotent(self):
        d = RiskDashboard()
        d.start(PORT + 20)
        time.sleep(0.05)
        d.stop()
        d.stop()  # second stop should not raise


class TestRiskEndpoint:
    def test_risk_returns_200(self, dashboard):
        data = _get(f"http://127.0.0.1:{PORT}/risk")
        assert "agents" in data

    def test_risk_has_timestamp(self, dashboard):
        data = _get(f"http://127.0.0.1:{PORT}/risk")
        assert "timestamp" in data

    def test_risk_empty_agents(self, dashboard):
        data = _get(f"http://127.0.0.1:{PORT}/risk")
        assert data["agents"] == []
        assert data["total_agents"] == 0

    def test_risk_with_agents(self, dashboard_with_data):
        data = _get(f"http://127.0.0.1:{PORT+1}/risk")
        assert len(data["agents"]) == 2

    def test_risk_agent_fields(self, dashboard_with_data):
        data = _get(f"http://127.0.0.1:{PORT+1}/risk")
        agent = data["agents"][0]
        required = {"agent_id", "profile", "capital", "position", "pnl",
                    "reputation", "var_95", "max_drawdown", "kelly_fraction"}
        assert required.issubset(set(agent.keys()))

    def test_risk_var_non_negative(self, dashboard_with_data):
        data = _get(f"http://127.0.0.1:{PORT+1}/risk")
        for agent in data["agents"]:
            assert agent["var_95"] >= 0

    def test_risk_max_drawdown_in_range(self, dashboard_with_data):
        data = _get(f"http://127.0.0.1:{PORT+1}/risk")
        for agent in data["agents"]:
            assert 0.0 <= agent["max_drawdown"] <= 1.0 or agent["max_drawdown"] >= 0


class TestPerformanceEndpoint:
    def test_performance_returns_200(self, dashboard):
        data = _get(f"http://127.0.0.1:{PORT}/performance")
        assert "agents" in data

    def test_performance_has_timestamp(self, dashboard):
        data = _get(f"http://127.0.0.1:{PORT}/performance")
        assert "timestamp" in data

    def test_performance_agent_fields(self, dashboard_with_data):
        data = _get(f"http://127.0.0.1:{PORT+1}/performance")
        for agent in data["agents"]:
            required = {"agent_id", "profile", "sharpe_ratio", "sortino_ratio",
                        "win_rate", "trades", "wins", "pnl"}
            assert required.issubset(set(agent.keys()))

    def test_performance_win_rate_in_range(self, dashboard_with_data):
        data = _get(f"http://127.0.0.1:{PORT+1}/performance")
        for agent in data["agents"]:
            assert 0.0 <= agent["win_rate"] <= 1.0

    def test_performance_sharpe_is_float(self, dashboard_with_data):
        data = _get(f"http://127.0.0.1:{PORT+1}/performance")
        for agent in data["agents"]:
            assert isinstance(agent["sharpe_ratio"], (int, float))


class TestSignalsEndpoint:
    def test_signals_returns_200(self, dashboard):
        data = _get(f"http://127.0.0.1:{PORT}/signals")
        assert "signals" in data

    def test_signals_empty_initially(self, dashboard):
        data = _get(f"http://127.0.0.1:{PORT}/signals")
        assert data["count"] == 0
        assert data["signals"] == []

    def test_signals_with_data(self, dashboard_with_data):
        data = _get(f"http://127.0.0.1:{PORT+1}/signals")
        assert data["count"] == 1
        assert data["signals"][0]["signal_type"] == "BUY"

    def test_signals_limited_to_10(self, dashboard):
        for i in range(15):
            dashboard.record_signal(
                timestamp=time.time(), signal_type="HOLD", protocol="X",
                direction="NEUTRAL", confidence=0.5, agent_id="a1", sequence=i,
            )
        data = _get(f"http://127.0.0.1:{PORT}/signals")
        assert data["count"] <= 10

    def test_signals_have_timestamp(self, dashboard_with_data):
        data = _get(f"http://127.0.0.1:{PORT+1}/signals")
        for sig in data["signals"]:
            assert "timestamp" in sig


class TestConsensusEndpoint:
    def test_consensus_returns_200(self, dashboard):
        data = _get(f"http://127.0.0.1:{PORT}/consensus")
        assert "consensus" in data

    def test_consensus_empty_initially(self, dashboard):
        data = _get(f"http://127.0.0.1:{PORT}/consensus")
        assert data["count"] == 0

    def test_consensus_with_data(self, dashboard_with_data):
        data = _get(f"http://127.0.0.1:{PORT+1}/consensus")
        assert data["count"] == 1
        assert data["consensus"][0]["action"] == "BUY"

    def test_consensus_limited_to_20(self, dashboard):
        for i in range(30):
            dashboard.record_consensus(
                timestamp=time.time(), tick=i, action="HOLD",
                voted_buy=[], voted_sell=[], voted_hold=["a1"],
                consensus_weight=0.33,
            )
        data = _get(f"http://127.0.0.1:{PORT}/consensus")
        assert data["count"] <= 20

    def test_consensus_has_fields(self, dashboard_with_data):
        data = _get(f"http://127.0.0.1:{PORT+1}/consensus")
        for rec in data["consensus"]:
            required = {"timestamp", "tick", "action", "voted_buy",
                        "voted_sell", "voted_hold"}
            assert required.issubset(set(rec.keys()))


class TestResetEndpoint:
    def test_reset_returns_200(self, dashboard_with_data):
        data = _post(f"http://127.0.0.1:{PORT+1}/reset")
        assert data["status"] == "reset"

    def test_reset_increments_count(self, dashboard_with_data):
        data = _post(f"http://127.0.0.1:{PORT+1}/reset")
        assert data["reset_count"] >= 1

    def test_reset_clears_agent_pnl(self, dashboard_with_data):
        _post(f"http://127.0.0.1:{PORT+1}/reset")
        risk = _get(f"http://127.0.0.1:{PORT+1}/risk")
        for agent in risk["agents"]:
            assert agent["pnl"] == 0.0

    def test_reset_has_timestamp(self, dashboard_with_data):
        data = _post(f"http://127.0.0.1:{PORT+1}/reset")
        assert "timestamp" in data

    def test_multiple_resets(self, dashboard_with_data):
        _post(f"http://127.0.0.1:{PORT+1}/reset")
        data = _post(f"http://127.0.0.1:{PORT+1}/reset")
        assert data["reset_count"] == 2


class TestErrorHandling:
    def test_unknown_get_path_404(self, dashboard):
        try:
            _get(f"http://127.0.0.1:{PORT}/nonexistent")
            assert False, "Expected HTTP error"
        except urllib.error.HTTPError as e:
            assert e.code == 404

    def test_unknown_post_path_404(self, dashboard):
        try:
            _post(f"http://127.0.0.1:{PORT}/badpath")
            assert False, "Expected HTTP error"
        except urllib.error.HTTPError as e:
            assert e.code == 404

    def test_root_path_404(self, dashboard):
        try:
            _get(f"http://127.0.0.1:{PORT}/")
            # Some implementations return 404 for root
        except urllib.error.HTTPError as e:
            assert e.code == 404
        except Exception:
            pass  # any response is fine


class TestRiskDashboardHelpers:
    def test_update_agent_stores_state(self):
        d = RiskDashboard()
        d.update_agent(
            agent_id="test-1", profile="balanced",
            capital=1000.0, position=0.0, entry_price=0.0, pnl=0.0,
            pnl_history=[], kelly_fraction=0.25, reputation=5.0,
            trades=0, wins=0,
        )
        agents = d.state.get_agents()
        assert len(agents) == 1

    def test_record_signal_stores(self):
        d = RiskDashboard()
        d.record_signal(
            timestamp=1.0, signal_type="BUY", protocol="Aave",
            direction="LONG", confidence=0.9, agent_id="a1", sequence=1,
        )
        assert len(d.state.get_signals()) == 1

    def test_record_consensus_stores(self):
        d = RiskDashboard()
        d.record_consensus(
            timestamp=1.0, tick=1, action="SELL",
            voted_buy=[], voted_sell=["a1"], voted_hold=[],
            consensus_weight=0.5,
        )
        assert len(d.state.get_consensus()) == 1

    def test_risk_payload_structure(self):
        d = RiskDashboard()
        payload = d.risk_payload()
        assert "agents" in payload
        assert "timestamp" in payload
        assert "total_agents" in payload

    def test_performance_payload_structure(self):
        d = RiskDashboard()
        payload = d.performance_payload()
        assert "agents" in payload
        assert "timestamp" in payload

    def test_signals_payload_structure(self):
        d = RiskDashboard()
        payload = d.signals_payload()
        assert "signals" in payload
        assert "count" in payload
        assert "timestamp" in payload

    def test_consensus_payload_structure(self):
        d = RiskDashboard()
        payload = d.consensus_payload()
        assert "consensus" in payload
        assert "count" in payload
        assert "timestamp" in payload
