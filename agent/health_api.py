"""
health_api.py — Agent Health & Metrics HTTP API for ERC-8004 Trading Agent.

Exposes three REST endpoints via Python stdlib http.server:
  GET /health  → system status, uptime, per-agent ERC-8004 identity
  GET /metrics → portfolio & trading statistics (Sharpe, drawdown, win rate)
  GET /agents  → full agent roster with ERC-8004 credit tier and reputation

No frameworks, no dependencies beyond stdlib + project modules.

Usage:
    server = HealthAPIServer(coordinator=coord, backtester=bt)
    server.start(port=8080)   # starts in a background thread
    server.stop()
"""

from __future__ import annotations

import json
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Optional
from urllib.parse import urlparse


# ─── Request Handler ──────────────────────────────────────────────────────────

class _HealthHandler(BaseHTTPRequestHandler):
    """HTTP request handler for the health API."""

    # Injected by HealthAPIServer before use
    _api: "HealthAPIServer" = None  # type: ignore

    def log_message(self, format, *args):
        # Suppress default access log to keep output clean
        pass

    def _send_json(self, code: int, data: dict) -> None:
        body = json.dumps(data, indent=2).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def _send_error_json(self, code: int, message: str) -> None:
        self._send_json(code, {"error": message, "code": code})

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        path = parsed.path.rstrip("/") or "/"
        if path == "/health":
            self._send_json(200, self._api.health())
        elif path == "/metrics":
            self._send_json(200, self._api.metrics())
        elif path == "/agents":
            self._send_json(200, self._api.agents_list())
        else:
            self._send_error_json(404, f"Unknown endpoint: {path}")

    def do_HEAD(self) -> None:
        parsed = urlparse(self.path)
        path = parsed.path.rstrip("/") or "/"
        known = {"/health", "/metrics", "/agents"}
        code = 200 if path in known else 404
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.end_headers()


# ─── Health API Server ────────────────────────────────────────────────────────

class HealthAPIServer:
    """
    Lightweight HTTP health/metrics server for the ERC-8004 trading agent.

    Aggregates live data from the mesh coordinator, backtester, and
    credora client to expose structured JSON endpoints.
    """

    def __init__(
        self,
        coordinator=None,
        backtester=None,
        credora_client=None,
        start_time: Optional[float] = None,
    ) -> None:
        self._coordinator = coordinator
        self._backtester = backtester
        self._credora_client = credora_client
        self._start_time = start_time or time.time()
        self._server: Optional[HTTPServer] = None
        self._thread: Optional[threading.Thread] = None
        self._port: int = 0

        # Trade history injected externally (list of trade dicts)
        self.trade_history: list[dict] = []

    # ─── Endpoint Logic ───────────────────────────────────────────────────────

    def health(self) -> dict:
        """Return /health response payload."""
        uptime = time.time() - self._start_time
        agents_data = []
        if self._coordinator is not None:
            for ag in self._coordinator.agents:
                stats = ag.get_stats() if hasattr(ag, "get_stats") else {}
                agents_data.append({
                    "id": ag.agent_id,
                    "credit_tier": stats.get("credora_min_grade", "BBB"),
                    "reputation": round(stats.get("reputation_score", 5.0), 4),
                    "last_trade": stats.get("last_trade_ts", None),
                    "trades_today": stats.get("trades_today", 0),
                    "wins": stats.get("wins", 0),
                    "losses": stats.get("losses", 0),
                })
        else:
            # Default agent list when no coordinator is injected
            agents_data = [
                {
                    "id": "conservative_agent",
                    "credit_tier": "A",
                    "reputation": 5.0,
                    "last_trade": None,
                    "trades_today": 0,
                    "wins": 0,
                    "losses": 0,
                },
                {
                    "id": "balanced_agent",
                    "credit_tier": "BBB",
                    "reputation": 5.0,
                    "last_trade": None,
                    "trades_today": 0,
                    "wins": 0,
                    "losses": 0,
                },
                {
                    "id": "aggressive_agent",
                    "credit_tier": "BB",
                    "reputation": 5.0,
                    "last_trade": None,
                    "trades_today": 0,
                    "wins": 0,
                    "losses": 0,
                },
            ]

        return {
            "status": "ok",
            "uptime_seconds": round(uptime, 2),
            "timestamp": time.time(),
            "version": "1.0.0",
            "agents": agents_data,
        }

    def metrics(self) -> dict:
        """Return /metrics response payload."""
        if self._backtester and self.trade_history:
            try:
                stats = self._backtester.compute_stats(self.trade_history)
                return {
                    "total_trades": len(self.trade_history),
                    "win_rate": round(stats.win_rate, 4),
                    "sharpe_ratio": round(stats.sharpe_ratio, 4),
                    "max_drawdown": round(stats.max_drawdown_pct, 4),
                    "portfolio_value": round(
                        getattr(stats, "final_portfolio_value", 10_000.0), 2
                    ),
                    "total_pnl": round(getattr(stats, "total_pnl", 0.0), 2),
                    "profit_factor": round(getattr(stats, "profit_factor", 1.0), 4),
                }
            except Exception:
                pass

        # Synthetic metrics when no real data
        wins = sum(1 for t in self.trade_history if t.get("pnl", 0) > 0)
        losses = len(self.trade_history) - wins
        win_rate = wins / len(self.trade_history) if self.trade_history else 0.0
        total_pnl = sum(t.get("pnl", 0.0) for t in self.trade_history)

        return {
            "total_trades": len(self.trade_history),
            "win_rate": round(win_rate, 4),
            "sharpe_ratio": 0.0,
            "max_drawdown": 0.0,
            "portfolio_value": round(10_000.0 + total_pnl, 2),
            "total_pnl": round(total_pnl, 2),
            "profit_factor": 1.0,
        }

    def agents_list(self) -> dict:
        """Return /agents response payload."""
        agents = []
        if self._coordinator is not None:
            for ag in self._coordinator.agents:
                stats = ag.get_stats() if hasattr(ag, "get_stats") else {}
                # ERC-8004 identity
                erc8004_id = {
                    "agent_id": ag.agent_id,
                    "reputation_registry": "0x8004B663056A597Dffe9eCcC1965A193B7388713",
                    "identity_registry": "0x8004A818BFB912233c491871b3d84c89A494BD9e",
                    "chain": "Base Sepolia",
                    "score": round(stats.get("reputation_score", 5.0) * 100, 0),
                    "score_normalized": round(stats.get("reputation_score", 5.0) / 10.0, 3),
                }
                agents.append({
                    "erc8004": erc8004_id,
                    "status": {
                        "active": True,
                        "profile": stats.get("profile", ag.agent_id.split("_")[0]),
                        "credit_tier": stats.get("credora_min_grade", "BBB"),
                        "reputation": round(stats.get("reputation_score", 5.0), 4),
                        "trades_today": stats.get("trades_today", 0),
                        "wins": stats.get("wins", 0),
                        "losses": stats.get("losses", 0),
                        "last_trade": stats.get("last_trade_ts", None),
                    },
                })
        return {"agents": agents, "count": len(agents)}

    # ─── Server Lifecycle ─────────────────────────────────────────────────────

    def start(self, port: int = 8080, host: str = "localhost") -> None:
        """Start the HTTP server in a background daemon thread."""
        # Inject reference so handler can call back
        handler_cls = type(
            "_BoundHandler",
            (_HealthHandler,),
            {"_api": self},
        )
        self._server = HTTPServer((host, port), handler_cls)
        self._port = self._server.server_address[1]
        self._thread = threading.Thread(
            target=self._server.serve_forever,
            daemon=True,
            name="HealthAPIThread",
        )
        self._thread.start()

    def stop(self) -> None:
        """Shutdown the HTTP server."""
        if self._server:
            self._server.shutdown()
            self._server = None
        if self._thread:
            self._thread.join(timeout=2.0)
            self._thread = None

    @property
    def port(self) -> int:
        return self._port

    @property
    def running(self) -> bool:
        return self._server is not None

    def get_base_url(self, host: str = "localhost") -> str:
        return f"http://{host}:{self._port}"


# ─── Entrypoint ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from mesh_coordinator import MeshCoordinator
    coord = MeshCoordinator()
    api = HealthAPIServer(coordinator=coord)
    api.start(port=8080)
    print(f"Health API running at http://localhost:{api.port}")
    print("  GET /health   → system status")
    print("  GET /metrics  → trading statistics")
    print("  GET /agents   → agent roster")
    try:
        import time as _t
        while True:
            _t.sleep(1)
    except KeyboardInterrupt:
        api.stop()
