"""
risk_dashboard.py — Risk Dashboard HTTP API for ERC-8004 Trading Agent.

Exposes risk and performance data via stdlib HTTP server on port 8082.

Endpoints:
  GET  /risk        — Current risk exposure per agent (VaR, max drawdown, Kelly fraction)
  GET  /performance — Sharpe ratio, Sortino ratio, win rate from paper trader
  GET  /signals     — Last 10 signals from signal server with timestamps
  GET  /consensus   — Mesh coordinator vote history (last 20 decisions)
  POST /reset       — Reset paper trader state + reputation (for demo)

No external dependencies — pure stdlib + project modules.

Usage:
    dashboard = RiskDashboard()
    dashboard.start(port=8082)
    # ... use it ...
    dashboard.stop()
"""

from __future__ import annotations

import json
import math
import threading
import time
from dataclasses import dataclass, field, asdict
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse


# ─── In-memory store ──────────────────────────────────────────────────────────

@dataclass
class AgentRiskState:
    """Risk state for one agent in the dashboard."""
    agent_id: str
    profile: str
    capital: float
    position: float
    entry_price: float
    pnl: float
    pnl_history: List[float] = field(default_factory=list)
    kelly_fraction: float = 0.25
    reputation: float = 5.0
    trades: int = 0
    wins: int = 0

    def var_95(self) -> float:
        """Parametric 95% 1-day VaR estimate using portfolio volatility."""
        if len(self.pnl_history) < 2:
            return 0.0
        n = len(self.pnl_history)
        mean = sum(self.pnl_history) / n
        variance = sum((x - mean) ** 2 for x in self.pnl_history) / max(n - 1, 1)
        std = math.sqrt(variance)
        # 95th percentile z = 1.645
        return abs(mean - 1.645 * std)

    def max_drawdown(self) -> float:
        """Maximum drawdown from peak equity."""
        if not self.pnl_history:
            return 0.0
        peak = self.pnl_history[0]
        mdd = 0.0
        for p in self.pnl_history:
            if p > peak:
                peak = p
            dd = (peak - p) / abs(peak) if peak != 0 else 0.0
            if dd > mdd:
                mdd = dd
        return mdd

    def win_rate(self) -> float:
        return self.wins / self.trades if self.trades > 0 else 0.0


@dataclass
class SignalRecord:
    """One signal record stored in the dashboard."""
    timestamp: float
    signal_type: str
    protocol: str
    direction: str
    confidence: float
    agent_id: str
    sequence: int


@dataclass
class ConsensusRecord:
    """One consensus decision stored in the dashboard."""
    timestamp: float
    tick: int
    action: str
    voted_buy: List[str]
    voted_sell: List[str]
    voted_hold: List[str]
    consensus_weight: float


# ─── Dashboard State ──────────────────────────────────────────────────────────

class DashboardState:
    """Thread-safe in-memory state for the risk dashboard."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._agents: Dict[str, AgentRiskState] = {}
        self._signals: List[SignalRecord] = []
        self._consensus: List[ConsensusRecord] = []
        self._started_at = time.time()
        self._reset_count = 0

    # ── Agents ────────────────────────────────────────────────────────────────

    def upsert_agent(self, state: AgentRiskState) -> None:
        with self._lock:
            self._agents[state.agent_id] = state

    def get_agents(self) -> List[AgentRiskState]:
        with self._lock:
            return list(self._agents.values())

    def reset_agents(self) -> None:
        """Reset all agent states (for demo)."""
        with self._lock:
            for agent in self._agents.values():
                agent.capital = 10_000.0
                agent.position = 0.0
                agent.entry_price = 0.0
                agent.pnl = 0.0
                agent.pnl_history = []
                agent.reputation = 5.0
                agent.trades = 0
                agent.wins = 0
            self._reset_count += 1

    # ── Signals ───────────────────────────────────────────────────────────────

    def add_signal(self, sig: SignalRecord) -> None:
        with self._lock:
            self._signals.append(sig)
            if len(self._signals) > 200:
                self._signals = self._signals[-200:]

    def get_signals(self, n: int = 10) -> List[SignalRecord]:
        with self._lock:
            return self._signals[-n:]

    # ── Consensus ─────────────────────────────────────────────────────────────

    def add_consensus(self, rec: ConsensusRecord) -> None:
        with self._lock:
            self._consensus.append(rec)
            if len(self._consensus) > 200:
                self._consensus = self._consensus[-200:]

    def get_consensus(self, n: int = 20) -> List[ConsensusRecord]:
        with self._lock:
            return self._consensus[-n:]

    # ── Stats ─────────────────────────────────────────────────────────────────

    def sharpe_ratio(self, agent_id: str) -> float:
        with self._lock:
            agent = self._agents.get(agent_id)
        if not agent or len(agent.pnl_history) < 2:
            return 0.0
        returns = []
        for i in range(1, len(agent.pnl_history)):
            returns.append(agent.pnl_history[i] - agent.pnl_history[i - 1])
        if not returns:
            return 0.0
        n = len(returns)
        mean = sum(returns) / n
        variance = sum((r - mean) ** 2 for r in returns) / max(n - 1, 1)
        std = math.sqrt(variance)
        if std == 0:
            return 0.0
        return mean / std * math.sqrt(252)  # annualised

    def sortino_ratio(self, agent_id: str) -> float:
        with self._lock:
            agent = self._agents.get(agent_id)
        if not agent or len(agent.pnl_history) < 2:
            return 0.0
        returns = []
        for i in range(1, len(agent.pnl_history)):
            returns.append(agent.pnl_history[i] - agent.pnl_history[i - 1])
        if not returns:
            return 0.0
        n = len(returns)
        mean = sum(returns) / n
        downside = [r for r in returns if r < 0]
        if not downside:
            return float("inf")
        downside_var = sum(r ** 2 for r in downside) / len(downside)
        downside_std = math.sqrt(downside_var)
        if downside_std == 0:
            return 0.0
        return mean / downside_std * math.sqrt(252)


# ─── HTTP Handler ─────────────────────────────────────────────────────────────

class _DashboardHandler(BaseHTTPRequestHandler):
    _dashboard: "RiskDashboard" = None  # type: ignore

    def log_message(self, format, *args):
        pass  # suppress access log

    def _send_json(self, code: int, data: Any) -> None:
        body = json.dumps(data, indent=2, default=str).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        path = parsed.path.rstrip("/") or "/"

        if path == "/risk":
            self._send_json(200, self._dashboard.risk_payload())
        elif path == "/performance":
            self._send_json(200, self._dashboard.performance_payload())
        elif path == "/signals":
            self._send_json(200, self._dashboard.signals_payload())
        elif path == "/consensus":
            self._send_json(200, self._dashboard.consensus_payload())
        else:
            self._send_json(404, {"error": "Not found", "path": path})

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        path = parsed.path.rstrip("/") or "/"

        if path == "/reset":
            self._dashboard.state.reset_agents()
            self._send_json(200, {
                "status": "reset",
                "reset_count": self._dashboard.state._reset_count,
                "timestamp": time.time(),
            })
        else:
            self._send_json(404, {"error": "Not found", "path": path})

    def do_OPTIONS(self) -> None:
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()


# ─── Risk Dashboard ───────────────────────────────────────────────────────────

class RiskDashboard:
    """
    HTTP Risk Dashboard server.

    Parameters
    ----------
    state : DashboardState (optional — creates one if not provided)
    """

    def __init__(self, state: Optional[DashboardState] = None) -> None:
        self.state = state or DashboardState()
        self._server: Optional[HTTPServer] = None
        self._thread: Optional[threading.Thread] = None
        self._port: int = 8082

    # ── Server lifecycle ──────────────────────────────────────────────────────

    def start(self, port: int = 8082) -> None:
        """Start HTTP server in background thread."""
        self._port = port
        handler = type(
            "_BoundHandler",
            (_DashboardHandler,),
            {"_dashboard": self},
        )
        self._server = HTTPServer(("127.0.0.1", port), handler)
        self._thread = threading.Thread(
            target=self._server.serve_forever,
            daemon=True,
            name="risk-dashboard",
        )
        self._thread.start()

    def stop(self) -> None:
        """Stop the HTTP server."""
        if self._server:
            self._server.shutdown()
            self._server = None
        if self._thread:
            self._thread.join(timeout=2)
            self._thread = None

    @property
    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    # ── Payload builders ──────────────────────────────────────────────────────

    def risk_payload(self) -> Dict:
        agents = self.state.get_agents()
        result = []
        for agent in agents:
            result.append({
                "agent_id": agent.agent_id,
                "profile": agent.profile,
                "capital": agent.capital,
                "position": agent.position,
                "pnl": agent.pnl,
                "reputation": agent.reputation,
                "var_95": agent.var_95(),
                "max_drawdown": agent.max_drawdown(),
                "kelly_fraction": agent.kelly_fraction,
            })
        return {
            "timestamp": time.time(),
            "agents": result,
            "total_agents": len(result),
        }

    def performance_payload(self) -> Dict:
        agents = self.state.get_agents()
        result = []
        for agent in agents:
            result.append({
                "agent_id": agent.agent_id,
                "profile": agent.profile,
                "sharpe_ratio": self.state.sharpe_ratio(agent.agent_id),
                "sortino_ratio": self.state.sortino_ratio(agent.agent_id),
                "win_rate": agent.win_rate(),
                "trades": agent.trades,
                "wins": agent.wins,
                "pnl": agent.pnl,
            })
        return {
            "timestamp": time.time(),
            "agents": result,
        }

    def signals_payload(self) -> Dict:
        signals = self.state.get_signals(10)
        return {
            "timestamp": time.time(),
            "count": len(signals),
            "signals": [asdict(s) for s in signals],
        }

    def consensus_payload(self) -> Dict:
        records = self.state.get_consensus(20)
        return {
            "timestamp": time.time(),
            "count": len(records),
            "consensus": [asdict(r) for r in records],
        }

    # ── Data ingestion helpers ────────────────────────────────────────────────

    def record_signal(self, **kwargs) -> None:
        """Add a signal record to the state."""
        self.state.add_signal(SignalRecord(**kwargs))

    def record_consensus(self, **kwargs) -> None:
        """Add a consensus record to the state."""
        self.state.add_consensus(ConsensusRecord(**kwargs))

    def update_agent(self, **kwargs) -> None:
        """Upsert an agent risk state."""
        self.state.upsert_agent(AgentRiskState(**kwargs))


if __name__ == "__main__":
    import urllib.request

    dashboard = RiskDashboard()
    dashboard.start(8082)
    print("Risk Dashboard running on http://127.0.0.1:8082")

    # Populate demo data
    dashboard.update_agent(
        agent_id="agent-demo-001",
        profile="conservative",
        capital=9_800.0,
        position=5.0,
        entry_price=98.0,
        pnl=200.0,
        pnl_history=[0, 50, 120, 200],
        kelly_fraction=0.15,
        reputation=7.5,
        trades=10,
        wins=6,
    )

    print(urllib.request.urlopen("http://127.0.0.1:8082/risk").read().decode())
    dashboard.stop()
