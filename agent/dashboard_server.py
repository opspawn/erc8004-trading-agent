"""
dashboard_server.py — FastAPI dashboard backend for ERC-8004 Trading Agent.

Serves live agent metrics at port 8001.

Endpoints:
  GET /status      — agent health, PnL, last trade, risk summary
  GET /trades      — recent trade history from ledger
  GET /reputation  — current reputation score from reputation.py

Run standalone:
    cd agent/
    uvicorn dashboard_server:app --host 0.0.0.0 --port 8001 --reload

Or programmatically:
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from loguru import logger

# ─── App ──────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="ERC-8004 Trading Agent Dashboard",
    description="Real-time dashboard for the autonomous trading agent",
    version="0.3.0",
)

# ─── Global State (injected by main agent loop) ───────────────────────────────

# These are set by the main agent when it starts the dashboard server
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
}

# ─── State Helpers ────────────────────────────────────────────────────────────

def update_state(key: str, value: Any) -> None:
    """Thread-safe state update (single-process, so no lock needed)."""
    _state[key] = value


def get_state() -> dict:
    """Return a copy of the current state."""
    return dict(_state)


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


# ─── Routes ───────────────────────────────────────────────────────────────────

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
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "uptime_since": _state.get("started_at"),
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
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


@app.get("/trades")
async def trades(limit: int = 20) -> list[dict]:
    """
    Return recent trade history.

    Query params:
        limit: max number of trades to return (default 20, max 100)

    Returns list of trade dicts with fields:
        market_id, question, side, size_usdc, executed_at,
        outcome, pnl_usdc, tx_hash
    """
    if limit > 100:
        limit = 100
    return _load_trades_from_ledger(limit=limit)


@app.get("/reputation")
async def reputation() -> dict:
    """
    Return current agent reputation score.

    Response schema:
        agent_id: int
        aggregate_score: float (0.0–10.0)
        total_feedback: int
        on_chain_count: int
        win_count: int
        loss_count: int
        win_rate: float
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
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


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
