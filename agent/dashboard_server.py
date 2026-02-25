"""
dashboard_server.py — FastAPI dashboard backend for ERC-8004 Trading Agent.

Serves live agent metrics at port 8001 with WebSocket real-time updates.

HTTP Endpoints:
  GET /           — dashboard HTML
  GET /health     — health check
  GET /status     — agent health, PnL, last trade, risk summary
  GET /trades     — recent trade history from ledger
  GET /reputation — current reputation score from reputation.py
  POST /state/update — push state updates from agent loop

WebSocket Endpoints:
  WS /ws/trading  — real-time event stream

WebSocket Events:
  price_update    — new market price data
  trade_executed  — a trade was placed
  portfolio_update — portfolio value/positions changed
  agent_signal    — strategist made a signal
  risk_alert      — risk manager raised an alert
  ensemble_signal — multi-agent coordinator decision

Run standalone:
    cd agent/
    uvicorn dashboard_server:app --host 0.0.0.0 --port 8001 --reload

Or programmatically:
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
"""

from __future__ import annotations

import asyncio
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from loguru import logger


# ─── App ──────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="ERC-8004 Trading Agent Dashboard",
    description="Real-time dashboard for the autonomous trading agent",
    version="0.4.0",
)

# ─── Global State (injected by main agent loop) ───────────────────────────────

_state: dict[str, Any] = {
    "tests_passing": 0,
    "active_positions": 0,
    "total_pnl": 0.0,
    "last_trade": None,
    "risk_summary": {},
    "trades": [],
    "reputation": {
        "agent_id": 0,
        "aggregate_score": 0.0,
        "total_feedback": 0,
        "on_chain_count": 0,
        "win_count": 0,
        "loss_count": 0,
    },
    "started_at": datetime.now(timezone.utc).isoformat(),
    "strategist_stats": {},
    # Multi-agent coordinator fields (S7)
    "agent_pool": {
        "pool_size": 0,
        "agents": [],
        "consensus_rate": 0.0,
        "decision_count": 0,
    },
    "last_ensemble": None,
    "price_feed": {},
}

# ─── State Helpers ────────────────────────────────────────────────────────────


def update_state(key: str, value: Any) -> None:
    """Thread-safe state update (single-process, so no lock needed)."""
    _state[key] = value


def get_state() -> dict:
    """Return a copy of the current state."""
    return dict(_state)


# ─── WebSocket Connection Manager ────────────────────────────────────────────


class ConnectionManager:
    """Manages active WebSocket connections and event broadcasting."""

    def __init__(self):
        self._connections: list[WebSocket] = []
        self._event_log: list[dict] = []
        self._max_log: int = 200

    @property
    def connection_count(self) -> int:
        return len(self._connections)

    async def connect(self, websocket: WebSocket) -> None:
        await websocket.accept()
        self._connections.append(websocket)
        logger.info(f"WS connect — {self.connection_count} clients")

    def disconnect(self, websocket: WebSocket) -> None:
        if websocket in self._connections:
            self._connections.remove(websocket)
        logger.info(f"WS disconnect — {self.connection_count} clients")

    async def send_event(self, websocket: WebSocket, event: dict) -> None:
        """Send an event to a single connection."""
        try:
            await websocket.send_json(event)
        except Exception:
            self.disconnect(websocket)

    async def broadcast(self, event: dict) -> None:
        """Broadcast an event to all connected clients."""
        self._log_event(event)
        dead: list[WebSocket] = []
        for ws in list(self._connections):
            try:
                await ws.send_json(event)
            except Exception:
                dead.append(ws)
        for ws in dead:
            self.disconnect(ws)

    def _log_event(self, event: dict) -> None:
        self._event_log.append(event)
        if len(self._event_log) > self._max_log:
            self._event_log.pop(0)

    def recent_events(self, n: int = 50) -> list[dict]:
        return self._event_log[-n:]

    async def broadcast_price_update(self, symbol: str, price: float, change_pct: float = 0.0) -> None:
        event = {
            "type": "price_update",
            "symbol": symbol,
            "price": price,
            "change_pct": round(change_pct, 4),
            "timestamp": _now(),
        }
        _state["price_feed"][symbol] = {"price": price, "change_pct": change_pct}
        await self.broadcast(event)

    async def broadcast_trade_executed(
        self,
        market_id: str,
        side: str,
        size_usdc: float,
        tx_hash: str,
        pnl: Optional[float] = None,
    ) -> None:
        event = {
            "type": "trade_executed",
            "market_id": market_id,
            "side": side,
            "size_usdc": size_usdc,
            "tx_hash": tx_hash,
            "pnl": pnl,
            "timestamp": _now(),
        }
        _state["last_trade"] = event
        await self.broadcast(event)

    async def broadcast_portfolio_update(
        self,
        total_value: float,
        pnl: float,
        active_positions: int,
    ) -> None:
        event = {
            "type": "portfolio_update",
            "total_value": round(total_value, 4),
            "pnl": round(pnl, 4),
            "active_positions": active_positions,
            "timestamp": _now(),
        }
        _state["total_pnl"] = pnl
        _state["active_positions"] = active_positions
        await self.broadcast(event)

    async def broadcast_agent_signal(
        self,
        agent_id: str,
        action: str,
        confidence: float,
        symbol: str,
    ) -> None:
        event = {
            "type": "agent_signal",
            "agent_id": agent_id,
            "action": action,
            "confidence": round(confidence, 4),
            "symbol": symbol,
            "timestamp": _now(),
        }
        await self.broadcast(event)

    async def broadcast_ensemble_signal(
        self,
        action: str,
        consensus_weight: float,
        agent_votes: dict,
        has_consensus: bool,
    ) -> None:
        event = {
            "type": "ensemble_signal",
            "action": action,
            "consensus_weight": round(consensus_weight, 4),
            "agent_votes": agent_votes,
            "has_consensus": has_consensus,
            "timestamp": _now(),
        }
        _state["last_ensemble"] = event
        await self.broadcast(event)

    async def broadcast_risk_alert(
        self,
        alert_type: str,
        message: str,
        severity: str = "warning",
    ) -> None:
        event = {
            "type": "risk_alert",
            "alert_type": alert_type,
            "message": message,
            "severity": severity,
            "timestamp": _now(),
        }
        await self.broadcast(event)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


# Singleton manager
manager = ConnectionManager()


# ─── Ledger File Reader ───────────────────────────────────────────────────────

LEDGER_PATH = Path(__file__).parent / "logs" / "trades.json"


def _load_trades_from_ledger(limit: int = 50) -> list[dict]:
    """
    Load trades from the JSON ledger file if it exists,
    otherwise return the in-memory state.
    """
    if LEDGER_PATH.exists():
        try:
            with open(LEDGER_PATH) as f:
                trades = json.load(f)
            if isinstance(trades, list):
                return trades[-limit:]
        except Exception as e:
            logger.warning(f"Could not read ledger: {e}")
    return _state.get("trades", [])[-limit:]


# ─── Static Files ─────────────────────────────────────────────────────────────

STATIC_DIR = Path(__file__).parent / "static"
STATIC_DIR.mkdir(exist_ok=True)

# Mount static files if directory exists
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


# ─── HTTP Routes ──────────────────────────────────────────────────────────────


@app.get("/", response_class=HTMLResponse)
async def root() -> HTMLResponse:
    """Serve the dashboard HTML."""
    index_path = STATIC_DIR / "index.html"
    if index_path.exists():
        return HTMLResponse(content=index_path.read_text())
    return HTMLResponse(content=_minimal_html())


@app.get("/health")
async def health() -> dict:
    """Health check endpoint."""
    return {
        "status": "ok",
        "timestamp": _now(),
        "uptime_since": _state.get("started_at"),
        "ws_connections": manager.connection_count,
    }


@app.get("/status")
async def status() -> dict:
    """
    Return agent status summary.

    Response schema:
        tests_passing: int
        active_positions: int
        total_pnl: float
        last_trade: dict | null
        risk_summary: dict
        started_at: str (ISO datetime)
        strategist_stats: dict
        agent_pool: dict
        last_ensemble: dict | null
        ws_connections: int
    """
    risk = _state.get("risk_summary", {})
    halted = risk.get("trading_halted", False)

    return {
        "tests_passing": _state.get("tests_passing", 0),
        "active_positions": _state.get("active_positions", 0),
        "total_pnl": round(_state.get("total_pnl", 0.0), 4),
        "last_trade": _state.get("last_trade"),
        "risk_summary": risk,
        "trading_halted": halted,
        "started_at": _state.get("started_at"),
        "strategist_stats": _state.get("strategist_stats", {}),
        "agent_pool": _state.get("agent_pool", {}),
        "last_ensemble": _state.get("last_ensemble"),
        "ws_connections": manager.connection_count,
        "timestamp": _now(),
    }


@app.get("/trades")
async def trades(limit: int = 20) -> list[dict]:
    """
    Return recent trade history.

    Query params:
        limit: max number of trades to return (default 20, max 100)
    """
    if limit > 100:
        limit = 100
    return _load_trades_from_ledger(limit=limit)


@app.get("/reputation")
async def reputation() -> dict:
    """
    Return current agent reputation score.
    """
    rep = _state.get("reputation", {})
    win_count = rep.get("win_count", 0)
    loss_count = rep.get("loss_count", 0)
    settled = win_count + loss_count
    win_rate = win_count / settled if settled > 0 else 0.0

    return {
        "agent_id": rep.get("agent_id", 0),
        "aggregate_score": round(rep.get("aggregate_score", 0.0), 2),
        "total_feedback": rep.get("total_feedback", 0),
        "on_chain_count": rep.get("on_chain_count", 0),
        "win_count": win_count,
        "loss_count": loss_count,
        "win_rate": round(win_rate, 4),
        "timestamp": _now(),
    }


@app.get("/agents")
async def agents() -> dict:
    """Return multi-agent pool status."""
    return {
        "agent_pool": _state.get("agent_pool", {}),
        "last_ensemble": _state.get("last_ensemble"),
        "timestamp": _now(),
    }


@app.get("/events")
async def events(limit: int = 50) -> list[dict]:
    """Return recent WebSocket event log."""
    return manager.recent_events(min(limit, 200))


@app.post("/state/update")
async def update_agent_state(payload: dict) -> dict:
    """
    Internal endpoint for the agent to push state updates.
    Called by main.py after each trading cycle.

    Accepts partial updates — only provided keys are updated.
    """
    for key, value in payload.items():
        if key in _state:
            _state[key] = value
    return {"ok": True, "updated": list(payload.keys())}


@app.post("/events/price")
async def push_price_event(payload: dict) -> dict:
    """Push a price update event to all WebSocket clients."""
    symbol = payload.get("symbol", "ETH")
    price = float(payload.get("price", 0))
    change_pct = float(payload.get("change_pct", 0.0))
    await manager.broadcast_price_update(symbol, price, change_pct)
    return {"ok": True, "event": "price_update"}


@app.post("/events/trade")
async def push_trade_event(payload: dict) -> dict:
    """Push a trade executed event to all WebSocket clients."""
    await manager.broadcast_trade_executed(
        market_id=payload.get("market_id", ""),
        side=payload.get("side", "buy"),
        size_usdc=float(payload.get("size_usdc", 0)),
        tx_hash=payload.get("tx_hash", ""),
        pnl=payload.get("pnl"),
    )
    return {"ok": True, "event": "trade_executed"}


@app.post("/events/risk-alert")
async def push_risk_alert(payload: dict) -> dict:
    """Push a risk alert to all WebSocket clients."""
    await manager.broadcast_risk_alert(
        alert_type=payload.get("alert_type", "generic"),
        message=payload.get("message", ""),
        severity=payload.get("severity", "warning"),
    )
    return {"ok": True, "event": "risk_alert"}


# ─── WebSocket Route ──────────────────────────────────────────────────────────


@app.websocket("/ws/trading")
async def websocket_trading(websocket: WebSocket):
    """
    WebSocket endpoint for real-time trading updates.

    On connect: sends current dashboard snapshot.
    On message: echos heartbeat or processes client commands.
    On disconnect: cleans up connection.

    Events pushed from server:
        price_update, trade_executed, portfolio_update,
        agent_signal, ensemble_signal, risk_alert
    """
    await manager.connect(websocket)
    try:
        # Send initial state snapshot
        snapshot = {
            "type": "snapshot",
            "state": {
                "total_pnl": _state.get("total_pnl", 0.0),
                "active_positions": _state.get("active_positions", 0),
                "last_trade": _state.get("last_trade"),
                "agent_pool": _state.get("agent_pool", {}),
                "last_ensemble": _state.get("last_ensemble"),
                "price_feed": _state.get("price_feed", {}),
                "reputation": _state.get("reputation", {}),
            },
            "timestamp": _now(),
        }
        await websocket.send_json(snapshot)

        # Listen for client messages (heartbeat, commands)
        while True:
            try:
                data = await asyncio.wait_for(websocket.receive_json(), timeout=30.0)
                msg_type = data.get("type", "")
                if msg_type == "ping":
                    await websocket.send_json({"type": "pong", "timestamp": _now()})
                elif msg_type == "request_snapshot":
                    await websocket.send_json(snapshot)
            except asyncio.TimeoutError:
                # Send keepalive
                await websocket.send_json({"type": "keepalive", "timestamp": _now()})
    except WebSocketDisconnect:
        manager.disconnect(websocket)
    except Exception as e:
        logger.warning(f"WS error: {e}")
        manager.disconnect(websocket)


# ─── Minimal fallback HTML ────────────────────────────────────────────────────


def _minimal_html() -> str:
    """Minimal dashboard served if static/index.html is missing."""
    return """<!DOCTYPE html>
<html>
<head><title>ERC-8004 Agent Dashboard</title></head>
<body>
<h1>ERC-8004 Trading Agent</h1>
<p>Dashboard loading... Fetching <a href="/status">/status</a></p>
<script>
  fetch('/status').then(r=>r.json()).then(d=>document.body.innerHTML+=
    '<pre>'+JSON.stringify(d,null,2)+'</pre>');
</script>
</body>
</html>"""


# ─── Standalone entry point ───────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("DASHBOARD_PORT", "8001"))
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")
