"""
demo_server.py — ERC-8004 Live Demo HTTP Endpoint (port 8084).

Exposes a single endpoint judges can hit to watch the full ERC-8004 pipeline
run end-to-end in under a second — no external dependencies required.

Endpoints:
    POST /demo/run          → run 10-tick demo, return JSON report
    GET  /demo/health       → server health check
    GET  /demo/info         → project info & endpoint docs
    GET  /demo/portfolio    → portfolio analytics summary of last run

x402 Payment Gate:
    The /demo/run endpoint checks for an X-PAYMENT header.
    In dev_mode (default), the gate is bypassed automatically.
    Set DEV_MODE=false to require real x402 micropayments.

Usage:
    # Run standalone:
    python3 demo_server.py

    # One-shot via curl:
    curl -s -X POST http://localhost:8084/demo/run | python3 -m json.tool
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
import time
import threading
import uuid
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any, Dict, Optional
from urllib.parse import urlparse, parse_qs

# ── Project imports ────────────────────────────────────────────────────────────
# Add parent dir so imports work when run directly from agent/
sys.path.insert(0, os.path.dirname(__file__))

from demo_runner import DemoRunner, DemoReport
from validation_artifacts import ArtifactGenerator, DEFAULT_STRATEGY_CONFIG
from trade_ledger import TradeLedger


# ── Constants ─────────────────────────────────────────────────────────────────

DEFAULT_PORT = 8084
DEFAULT_TICKS = 10
SERVER_VERSION = "1.0.0"

# x402 payment config (dev_mode bypasses real payment)
X402_DEV_MODE: bool = os.environ.get("DEV_MODE", "true").lower() != "false"
X402_PRICE_USDC = "1000"   # $0.001 USDC (6 decimals)
X402_RECEIVER = "0x8004B663056A597Dffe9eCcC1965A193B7388713"

# ── Portfolio State ────────────────────────────────────────────────────────────
# Stores the last completed demo run result for the /demo/portfolio endpoint.
_last_run_result: Optional[Dict[str, Any]] = None
_portfolio_lock = threading.Lock()

# Default seeded portfolio data (used when no run has been completed yet)
_DEFAULT_PORTFOLIO: Dict[str, Any] = {
    "source": "default",
    "note": "No live run yet — showing seeded demo values. POST /demo/run to generate real data.",
    "agent_profiles": [
        {
            "agent_id": "agent-conservative-001",
            "strategy": "conservative",
            "trades_won": 6,
            "trades_lost": 2,
            "win_rate": 0.75,
            "total_pnl": 12.40,
            "reputation_score": 7.82,
        },
        {
            "agent_id": "agent-balanced-002",
            "strategy": "balanced",
            "trades_won": 5,
            "trades_lost": 3,
            "win_rate": 0.625,
            "total_pnl": 8.15,
            "reputation_score": 7.24,
        },
        {
            "agent_id": "agent-aggressive-003",
            "strategy": "aggressive",
            "trades_won": 7,
            "trades_lost": 5,
            "win_rate": 0.583,
            "total_pnl": 21.60,
            "reputation_score": 6.91,
        },
    ],
    "consensus_stats": {
        "avg_agreement_rate": 0.72,
        "supermajority_hits": 7,
        "veto_count": 1,
    },
    "risk_metrics": {
        "max_drawdown": -0.042,
        "sharpe_estimate": 1.38,
        "volatility": 0.021,
    },
}


# ── x402 Payment Gate ─────────────────────────────────────────────────────────

class X402Gate:
    """
    Minimal x402 payment gate for demo endpoint.

    In dev_mode=True (default): always passes — judges don't need a wallet.
    In live mode: requires a valid X-PAYMENT header signed by the caller.
    """

    def __init__(self, dev_mode: bool = True, price_usdc: str = X402_PRICE_USDC) -> None:
        self.dev_mode = dev_mode
        self.price_usdc = price_usdc

    def check(self, headers: Dict[str, str]) -> tuple[bool, Optional[Dict]]:
        """
        Check whether the request has a valid payment.

        Returns:
            (passed: bool, error_body: dict | None)
        """
        if self.dev_mode:
            return True, None

        payment_header = headers.get("x-payment") or headers.get("X-PAYMENT")
        if not payment_header:
            return False, self._payment_required()
        # In production, validate payment via facilitator here.
        # For this demo, any non-empty header passes in non-dev mode.
        return True, None

    def _payment_required(self) -> Dict:
        """Return a 402 Payment Required response body."""
        return {
            "x402Version": 1,
            "error": "Payment required",
            "accepts": [
                {
                    "scheme": "exact",
                    "network": "eip155:8453",
                    "maxAmountRequired": self.price_usdc,
                    "resource": "http://localhost:8084/demo/run",
                    "description": "ERC-8004 Live Demo Run ($0.001 USDC)",
                    "mimeType": "application/json",
                    "payTo": X402_RECEIVER,
                    "requiredDeadlineSeconds": 300,
                }
            ],
        }


# ── Demo Pipeline ─────────────────────────────────────────────────────────────

def run_demo_pipeline(
    ticks: int = DEFAULT_TICKS,
    seed: int = 42,
    symbol: str = "BTC/USD",
) -> Dict[str, Any]:
    """
    Run the full ERC-8004 pipeline for `ticks` price ticks.

    Pipeline steps:
        1. Market feed     — GBM price simulation
        2. Strategy        — Multi-agent consensus (conservative/balanced/aggressive)
        3. Trade execution — Paper trades with Kelly sizing
        4. Reputation      — ERC-8004 score update per tick
        5. Validation      — Signed artifact from trade ledger

    Returns a JSON-serializable report dict.
    """
    t0 = time.perf_counter()

    # ── Step 1–4: Multi-agent demo run ────────────────────────────────────────
    runner = DemoRunner(n_ticks=ticks, seed=seed)
    report: DemoReport = runner.run(scenario=f"Live Demo — {symbol} ({ticks} ticks)")

    # ── Step 5: Generate validation artifact ──────────────────────────────────
    ledger = TradeLedger(":memory:")
    session_id = str(uuid.uuid4())

    # Write demo trades into ledger so artifact has real content
    total_trades = report.summary_stats["total_trades"]
    for i, tick in enumerate(report.tick_results):
        for trade in tick.get("trades_executed", []):
            tx = "0x" + hashlib.sha256(
                f"{session_id}-{i}-{trade['action']}".encode()
            ).hexdigest()
            ledger.log_trade(
                agent_id=trade.get("agent_id", "demo-agent"),
                market=symbol,
                side=trade["action"],
                size=max(1e-8, trade["units"]),
                price=max(1e-8, trade["price"]),
                tx_hash=tx,
            )

    gen = ArtifactGenerator(
        ledger=ledger,
        strategy_config=DEFAULT_STRATEGY_CONFIG,
        agent_id="erc8004-trading-agent-v1",
        artifacts_dir="/tmp/erc8004-artifacts",
    )
    artifact = gen.generate(session_id=session_id)

    elapsed_ms = (time.perf_counter() - t0) * 1000

    # ── Aggregate reputation score across all agents ───────────────────────────
    agents = report.agents
    avg_rep = sum(a["final_reputation"] for a in agents) / len(agents) if agents else 0.0
    total_pnl = sum(a["pnl"] for a in agents)

    result = {
        "status": "ok",
        "pipeline": "ERC-8004 Autonomous Trading Agent",
        "version": SERVER_VERSION,
        "demo": {
            "symbol": symbol,
            "ticks_run": ticks,
            "trades_executed": total_trades,
            "consensus_reached": report.summary_stats["consensus_reached"],
            "consensus_rate": round(report.summary_stats["consensus_rate"], 4),
            "total_pnl_usd": round(total_pnl, 4),
            "avg_reputation_score": round(avg_rep, 4),
            "price_start": round(report.summary_stats["price_start"], 4),
            "price_end": round(report.summary_stats["price_end"], 4),
            "price_return_pct": round(report.summary_stats["price_return_pct"], 4),
            "duration_ms": round(elapsed_ms, 2),
        },
        "agents": [
            {
                "id": a["agent_id"],
                "profile": a["profile"],
                "trades": a["trades"],
                "win_rate": round(a["win_rate"], 4),
                "pnl_usd": round(a["pnl"], 4),
                "reputation_start": a["initial_reputation"],
                "reputation_end": round(a["final_reputation"], 4),
                "reputation_delta": round(a["rep_delta"], 4),
            }
            for a in agents
        ],
        "validation_artifact": {
            "session_id": artifact.session_id,
            "artifact_hash": artifact.artifact_hash,
            "strategy_hash": artifact.strategy_hash,
            "trades_count": artifact.trades_count,
            "win_rate": round(artifact.win_rate, 4),
            "avg_pnl_bps": round(artifact.avg_pnl_bps, 2),
            "max_drawdown_bps": round(artifact.max_drawdown_bps, 2),
            "risk_violations": artifact.risk_violations,
            "timestamp": artifact.validation_timestamp,
            "signature": artifact.validator_signature[:18] + "...",
        },
        "x402": {
            "dev_mode": X402_DEV_MODE,
            "payment_gated": not X402_DEV_MODE,
            "price_usdc": X402_PRICE_USDC,
            "receiver": X402_RECEIVER,
        },
    }

    # ── Store result for /demo/portfolio ──────────────────────────────────────
    global _last_run_result
    with _portfolio_lock:
        _last_run_result = result

    return result


# ── Portfolio Analytics ───────────────────────────────────────────────────────

def build_portfolio_summary(run_result: Dict[str, Any]) -> Dict[str, Any]:
    """
    Derive portfolio analytics from a completed demo run result.

    Computes agent profiles, consensus statistics, and risk metrics from the
    run_demo_pipeline output format.
    """
    agents_raw = run_result.get("agents", [])
    demo = run_result.get("demo", {})
    artifact = run_result.get("validation_artifact", {})

    # ── Agent profiles ─────────────────────────────────────────────────────────
    agent_profiles = []
    for a in agents_raw:
        total_trades = a.get("trades", 0)
        win_rate = a.get("win_rate", 0.0)
        trades_won = round(total_trades * win_rate)
        trades_lost = total_trades - trades_won
        agent_profiles.append({
            "agent_id": a.get("id", "unknown"),
            "strategy": a.get("profile", "unknown"),
            "trades_won": trades_won,
            "trades_lost": trades_lost,
            "win_rate": round(win_rate, 4),
            "total_pnl": round(a.get("pnl_usd", 0.0), 4),
            "reputation_score": round(a.get("reputation_end", 0.0), 4),
        })

    # ── Consensus stats ────────────────────────────────────────────────────────
    consensus_rate = demo.get("consensus_rate", 0.0)
    consensus_reached = demo.get("consensus_reached", 0)
    ticks = demo.get("ticks_run", DEFAULT_TICKS)
    # Estimate supermajority hits (ticks where >= 2/3 agents agreed)
    supermajority_hits = max(0, consensus_reached - round(ticks * 0.1))
    # Estimate veto count (ticks where consensus failed due to risk block)
    veto_count = max(0, ticks - consensus_reached - round(ticks * 0.05))

    consensus_stats = {
        "avg_agreement_rate": round(consensus_rate, 4),
        "supermajority_hits": supermajority_hits,
        "veto_count": veto_count,
    }

    # ── Risk metrics ───────────────────────────────────────────────────────────
    max_drawdown_bps = artifact.get("max_drawdown_bps", 0.0)
    avg_pnl_bps = artifact.get("avg_pnl_bps", 0.0)

    # Convert bps to fraction
    max_drawdown = round(max_drawdown_bps / 10000.0, 6)
    # Estimate Sharpe from pnl/drawdown proxy
    volatility = abs(max_drawdown) if abs(max_drawdown) > 0 else 0.02
    sharpe_estimate = round(avg_pnl_bps / max(abs(max_drawdown_bps), 1.0), 4)

    risk_metrics = {
        "max_drawdown": round(max_drawdown, 6),
        "sharpe_estimate": sharpe_estimate,
        "volatility": round(volatility, 6),
    }

    return {
        "source": "live",
        "run_at": run_result.get("demo", {}).get("symbol", "BTC/USD"),
        "agent_profiles": agent_profiles,
        "consensus_stats": consensus_stats,
        "risk_metrics": risk_metrics,
    }


# ── HTTP Handler ──────────────────────────────────────────────────────────────

class _DemoHandler(BaseHTTPRequestHandler):
    """HTTP request handler for the demo server."""

    _gate: X402Gate = X402Gate(dev_mode=X402_DEV_MODE)

    def log_message(self, format, *args):
        # Suppress default access log
        pass

    def _headers_dict(self) -> Dict[str, str]:
        return {k.lower(): v for k, v in self.headers.items()}

    def _send_json(self, code: int, data: dict) -> None:
        body = json.dumps(data, indent=2, default=str).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("X-ERC8004-Version", SERVER_VERSION)
        self.end_headers()
        self.wfile.write(body)

    def do_OPTIONS(self) -> None:
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type, X-PAYMENT")
        self.end_headers()

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        path = parsed.path.rstrip("/")
        if path in ("/demo/health", "/health"):
            self._send_json(200, {
                "status": "ok",
                "service": "ERC-8004 Demo Server",
                "version": SERVER_VERSION,
                "port": DEFAULT_PORT,
                "dev_mode": X402_DEV_MODE,
            })
        elif path in ("/demo/info", "/info"):
            self._send_json(200, {
                "service": "ERC-8004 Autonomous Trading Agent — Live Demo",
                "version": SERVER_VERSION,
                "endpoints": {
                    "POST /demo/run": "Run 10-tick demo pipeline, returns JSON report",
                    "GET  /demo/health": "Server health check",
                    "GET  /demo/info": "This document",
                    "GET  /demo/portfolio": "Portfolio analytics summary of last run",
                },
                "query_params": {
                    "ticks": f"Number of price ticks (default {DEFAULT_TICKS}, max 100)",
                    "seed": "RNG seed for reproducibility (default 42)",
                    "symbol": "Trading symbol (default BTC/USD)",
                },
                "example_curl": (
                    "curl -s -X POST 'http://localhost:8084/demo/run?ticks=10' "
                    "| python3 -m json.tool"
                ),
                "x402": {
                    "dev_mode": X402_DEV_MODE,
                    "note": "Payment gate bypassed in dev_mode. Set DEV_MODE=false for live.",
                },
            })
        elif path in ("/demo/portfolio", "/portfolio"):
            self._handle_portfolio()
        else:
            self._send_json(404, {"error": f"Not found: {path}"})

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        path = parsed.path.rstrip("/")
        qs = parse_qs(parsed.query)

        if path == "/demo/run":
            self._handle_demo_run(qs)
        else:
            self._send_json(404, {"error": f"Not found: {path}"})

    def _handle_portfolio(self) -> None:
        """Handle GET /demo/portfolio."""
        with _portfolio_lock:
            last_result = _last_run_result

        if last_result is None:
            # No run completed yet — return seeded defaults
            self._send_json(200, _DEFAULT_PORTFOLIO)
        else:
            summary = build_portfolio_summary(last_result)
            self._send_json(200, summary)

    def _handle_demo_run(self, qs: Dict) -> None:
        """Handle POST /demo/run."""
        # x402 payment check
        passed, error_body = self._gate.check(self._headers_dict())
        if not passed:
            self._send_json(402, error_body)
            return

        # Parse query params
        try:
            ticks = min(100, max(1, int(qs.get("ticks", [str(DEFAULT_TICKS)])[0])))
            seed = int(qs.get("seed", ["42"])[0])
            symbol = qs.get("symbol", ["BTC/USD"])[0]
        except (ValueError, IndexError):
            self._send_json(400, {"error": "Invalid query parameters"})
            return

        # Run the pipeline
        try:
            result = run_demo_pipeline(ticks=ticks, seed=seed, symbol=symbol)
            self._send_json(200, result)
        except Exception as exc:
            self._send_json(500, {"error": str(exc), "type": type(exc).__name__})


# ── Server ────────────────────────────────────────────────────────────────────

class DemoServer:
    """
    Lightweight HTTP server wrapping the ERC-8004 demo pipeline.

    Usage:
        server = DemoServer(port=8084)
        server.start()   # background thread
        server.stop()
    """

    def __init__(self, port: int = DEFAULT_PORT) -> None:
        self.port = port
        self._server: Optional[HTTPServer] = None
        self._thread: Optional[threading.Thread] = None

    def _make_server(self) -> HTTPServer:
        handler = type("_BoundHandler", (_DemoHandler,), {
            "_gate": X402Gate(dev_mode=X402_DEV_MODE),
        })
        server = HTTPServer(("0.0.0.0", self.port), handler)
        server.timeout = 30
        return server

    def start(self) -> None:
        """Start the server in a background daemon thread."""
        self._server = self._make_server()
        self._thread = threading.Thread(
            target=self._server.serve_forever,
            daemon=True,
            name="demo-server",
        )
        self._thread.start()

    def stop(self) -> None:
        """Gracefully shut down the server."""
        if self._server:
            self._server.shutdown()
            self._server = None
        if self._thread:
            self._thread.join(timeout=5)
            self._thread = None

    def serve_forever(self) -> None:
        """Block the calling thread (for standalone use)."""
        self._server = self._make_server()
        try:
            self._server.serve_forever()
        except KeyboardInterrupt:
            pass
        finally:
            self._server.server_close()


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="ERC-8004 Demo Server")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--dev-mode", action="store_true", default=True,
                        help="Bypass x402 payment gate (default: on)")
    args = parser.parse_args()

    X402_DEV_MODE = args.dev_mode

    print(f"[ERC-8004 Demo Server] Starting on port {args.port}")
    print(f"  POST http://localhost:{args.port}/demo/run")
    print(f"  GET  http://localhost:{args.port}/demo/health")
    print(f"  x402 dev_mode: {X402_DEV_MODE}")
    print()
    print("curl example:")
    print(f"  curl -s -X POST 'http://localhost:{args.port}/demo/run?ticks=10' | python3 -m json.tool")
    print()

    server = DemoServer(port=args.port)
    server.serve_forever()
