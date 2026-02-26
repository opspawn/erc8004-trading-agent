"""
ws_server.py — ERC-8004 WebSocket Streaming Server (port 8085).

Provides real-time tick streaming to judges via WebSocket protocol.

Endpoint:
    ws://localhost:8085/demo/ws    → live tick + signal stream
    ws://localhost:8085/demo/ws?symbol=ETH/USD  → filter by symbol

Client message protocol (JSON):
    {"type": "ping"}               → server replies {"type": "pong", ...}
    {"type": "subscribe", "symbol": "BTC/USD"}  → subscribe to a symbol
    {"type": "unsubscribe"}        → revert to all symbols

Server → client messages:
    {"type": "connected", "ts": ..., "message": ..., "session_id": ...}
    {"type": "tick", "ts": ..., "symbol": ..., "price": ..., "volume": ..., "change_pct": ...}
    {"type": "signal", "ts": ..., "agent_id": ..., "action": ..., "confidence": ..., "symbol": ...}
    {"type": "trade", "ts": ..., "agent_id": ..., "side": ..., "qty": ..., "price": ..., "pnl": ...}
    {"type": "risk_alert", "ts": ..., "level": ..., "message": ...}
    {"type": "pong", "ts": ..., "echo": ...}
    {"type": "error", "ts": ..., "message": ...}
"""

from __future__ import annotations

import asyncio
import json
import math
import os
import random
import time
import threading
import uuid
from typing import Any, Dict, List, Optional, Set

# ── Constants ─────────────────────────────────────────────────────────────────

WS_PORT = 8085
TICK_INTERVAL = 1.0        # seconds between tick broadcasts
SYMBOLS = ["BTC/USD", "ETH/USD", "SOL/USD", "AVAX/USD"]
AGENT_IDS = ["momentum_alpha", "mean_revert_beta", "ensemble_gamma", "arb_delta"]

# Seed prices for deterministic-ish simulation
_SEED_PRICES: Dict[str, float] = {
    "BTC/USD": 43500.0,
    "ETH/USD": 2280.0,
    "SOL/USD": 98.5,
    "AVAX/USD": 35.2,
}


# ── Shared State ──────────────────────────────────────────────────────────────

class _StreamState:
    """Mutable market state for the WebSocket stream."""

    def __init__(self) -> None:
        self.prices: Dict[str, float] = dict(_SEED_PRICES)
        self.volumes: Dict[str, float] = {s: 1000.0 for s in SYMBOLS}
        self.tick_count: int = 0
        self.connected_clients: int = 0
        self._rng = random.Random(42)

    def next_tick(self, symbol: str) -> Dict[str, Any]:
        """Generate the next price tick for a symbol."""
        old_price = self.prices[symbol]
        # Geometric Brownian motion: drift + volatility
        drift = 0.0001
        vol = 0.002
        move = math.exp(
            (drift - 0.5 * vol ** 2) * TICK_INTERVAL
            + vol * self._rng.gauss(0, 1) * math.sqrt(TICK_INTERVAL)
        )
        new_price = round(old_price * move, 6 if old_price < 10 else 2)
        change_pct = round((new_price - old_price) / old_price * 100, 4)
        volume = round(self._rng.uniform(500, 5000), 2)

        self.prices[symbol] = new_price
        self.volumes[symbol] = volume
        self.tick_count += 1

        return {
            "type": "tick",
            "ts": time.time(),
            "symbol": symbol,
            "price": new_price,
            "volume": volume,
            "change_pct": change_pct,
            "tick_num": self.tick_count,
        }

    def random_signal(self, symbol: Optional[str] = None) -> Dict[str, Any]:
        """Generate a random agent signal."""
        sym = symbol or self._rng.choice(SYMBOLS)
        return {
            "type": "signal",
            "ts": time.time(),
            "agent_id": self._rng.choice(AGENT_IDS),
            "action": self._rng.choice(["BUY", "SELL", "HOLD"]),
            "confidence": round(self._rng.uniform(0.5, 0.99), 4),
            "symbol": sym,
            "price": self.prices[sym],
        }

    def random_trade(self, symbol: Optional[str] = None) -> Dict[str, Any]:
        """Generate a random trade execution."""
        sym = symbol or self._rng.choice(SYMBOLS)
        side = self._rng.choice(["BUY", "SELL"])
        qty = round(self._rng.uniform(0.01, 2.0), 4)
        price = self.prices[sym]
        pnl = round(self._rng.uniform(-50, 200), 2)
        return {
            "type": "trade",
            "ts": time.time(),
            "agent_id": self._rng.choice(AGENT_IDS),
            "side": side,
            "qty": qty,
            "price": price,
            "symbol": sym,
            "pnl": pnl,
        }


# Module-level state (shared across all sessions)
_state = _StreamState()
_connected_sessions: Set[str] = set()
_sessions_lock = threading.Lock()


# ── Message Builders ──────────────────────────────────────────────────────────

def build_welcome_message(session_id: str) -> str:
    """Build the initial connection welcome message."""
    msg = {
        "type": "connected",
        "ts": time.time(),
        "session_id": session_id,
        "message": "ERC-8004 WebSocket stream active",
        "symbols": SYMBOLS,
        "agents": AGENT_IDS,
        "tick_interval_s": TICK_INTERVAL,
    }
    return json.dumps(msg)


def build_pong_message(echo: Any = None) -> str:
    """Build a pong response."""
    msg: Dict[str, Any] = {"type": "pong", "ts": time.time()}
    if echo is not None:
        msg["echo"] = echo
    return json.dumps(msg)


def build_error_message(text: str) -> str:
    return json.dumps({"type": "error", "ts": time.time(), "message": text})


def build_risk_alert(level: str = "medium", message: str = "Drawdown threshold exceeded") -> str:
    return json.dumps({
        "type": "risk_alert",
        "ts": time.time(),
        "level": level,
        "message": message,
        "agent_id": random.choice(AGENT_IDS),
    })


# ── Session Handler ───────────────────────────────────────────────────────────

async def _handle_session(websocket: Any, path: str) -> None:
    """
    Handle a single WebSocket session.

    Streams ticks, signals, and trades to the client.
    Handles incoming ping/subscribe/unsubscribe messages.
    """
    # Parse optional ?symbol= query param
    symbol_filter: Optional[str] = None
    if "?" in path:
        qs = path.split("?", 1)[1]
        for part in qs.split("&"):
            if part.startswith("symbol="):
                raw = part.split("=", 1)[1].replace("%2F", "/")
                if raw in SYMBOLS:
                    symbol_filter = raw
                break

    if path.split("?")[0] not in ("/demo/ws", "/ws"):
        await websocket.close(1008, "Unknown path")
        return

    session_id = str(uuid.uuid4())[:8]
    with _sessions_lock:
        _connected_sessions.add(session_id)
        _state.connected_clients = len(_connected_sessions)

    try:
        # Send welcome
        await websocket.send(build_welcome_message(session_id))

        tick_counter = 0

        async def _stream() -> None:
            nonlocal tick_counter
            sym_list = [symbol_filter] if symbol_filter else SYMBOLS
            while True:
                await asyncio.sleep(TICK_INTERVAL)
                # Broadcast one tick per symbol in rotation
                sym = sym_list[tick_counter % len(sym_list)]
                tick = _state.next_tick(sym)
                await websocket.send(json.dumps(tick))
                tick_counter += 1
                # Occasionally emit a signal
                if tick_counter % 3 == 0:
                    await websocket.send(json.dumps(_state.random_signal(sym)))
                # Occasionally emit a trade
                if tick_counter % 5 == 0:
                    await websocket.send(json.dumps(_state.random_trade(sym)))
                # Very occasionally emit a risk alert
                if tick_counter % 20 == 0:
                    await websocket.send(build_risk_alert())

        async def _recv() -> None:
            nonlocal symbol_filter
            async for raw in websocket:
                try:
                    msg = json.loads(raw)
                except (json.JSONDecodeError, TypeError):
                    await websocket.send(build_error_message("Invalid JSON"))
                    continue
                mtype = msg.get("type", "")
                if mtype == "ping":
                    await websocket.send(build_pong_message(msg.get("id")))
                elif mtype == "subscribe":
                    sym = msg.get("symbol")
                    if sym in SYMBOLS:
                        symbol_filter = sym
                        await websocket.send(json.dumps({
                            "type": "subscribed",
                            "ts": time.time(),
                            "symbol": sym,
                        }))
                    else:
                        await websocket.send(build_error_message(
                            f"Unknown symbol: {sym}. Valid: {SYMBOLS}"
                        ))
                elif mtype == "unsubscribe":
                    symbol_filter = None
                    await websocket.send(json.dumps({
                        "type": "unsubscribed",
                        "ts": time.time(),
                    }))
                else:
                    await websocket.send(build_error_message(f"Unknown message type: {mtype}"))

        # Run both coroutines concurrently; stop when client disconnects
        stream_task = asyncio.ensure_future(_stream())
        recv_task = asyncio.ensure_future(_recv())
        done, pending = await asyncio.wait(
            [stream_task, recv_task],
            return_when=asyncio.FIRST_COMPLETED,
        )
        for task in pending:
            task.cancel()

    except Exception:
        pass
    finally:
        with _sessions_lock:
            _connected_sessions.discard(session_id)
            _state.connected_clients = len(_connected_sessions)


# ── Server Lifecycle ─────────────────────────────────────────────────────────

class WSServer:
    """
    WebSocket server for ERC-8004 live tick streaming.

    Usage:
        server = WSServer(port=8085)
        server.start()         # starts background thread with its own event loop
        server.stop()          # graceful shutdown
        server.serve_forever() # block calling thread (standalone)
    """

    def __init__(self, port: int = WS_PORT) -> None:
        self.port = port
        self._thread: Optional[threading.Thread] = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._server_obj: Any = None

    def _run_loop(self) -> None:
        """Entry point for the background thread."""
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        self._loop.run_until_complete(self._serve())

    async def _serve(self) -> None:
        try:
            import websockets as _ws  # local import avoids top-level dep in tests
            self._server_obj = await _ws.serve(
                _handle_session,
                "0.0.0.0",
                self.port,
            )
            await self._server_obj.wait_closed()
        except Exception as exc:
            pass  # server stopped externally

    def start(self) -> None:
        """Start the WebSocket server in a background daemon thread."""
        self._thread = threading.Thread(
            target=self._run_loop,
            daemon=True,
            name="ws-server",
        )
        self._thread.start()
        # Brief pause to let the server bind
        time.sleep(0.2)

    def stop(self) -> None:
        """Gracefully stop the WebSocket server."""
        if self._server_obj and self._loop:
            self._loop.call_soon_threadsafe(self._server_obj.close)
        if self._thread:
            self._thread.join(timeout=3)
            self._thread = None

    def serve_forever(self) -> None:
        """Block calling thread (for standalone mode)."""
        import asyncio as _asyncio

        async def _main() -> None:
            import websockets as _ws
            async with _ws.serve(_handle_session, "0.0.0.0", self.port) as srv:
                print(f"[ERC-8004 WS Server] Listening on ws://0.0.0.0:{self.port}/demo/ws")
                await asyncio.Future()  # block forever

        try:
            _asyncio.run(_main())
        except KeyboardInterrupt:
            pass


# ── Utility: get connected client count ──────────────────────────────────────

def get_connected_count() -> int:
    with _sessions_lock:
        return len(_connected_sessions)


def get_stream_state() -> Dict[str, Any]:
    return {
        "tick_count": _state.tick_count,
        "connected_clients": _state.connected_clients,
        "prices": dict(_state.prices),
        "symbols": SYMBOLS,
        "agents": AGENT_IDS,
    }


# ── Entry Point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="ERC-8004 WebSocket Server")
    parser.add_argument("--port", type=int, default=WS_PORT)
    args = parser.parse_args()

    print(f"[ERC-8004 WS Server] Starting on port {args.port}")
    print(f"  ws://localhost:{args.port}/demo/ws")
    print(f"  ws://localhost:{args.port}/demo/ws?symbol=BTC/USD")

    server = WSServer(port=args.port)
    server.serve_forever()
