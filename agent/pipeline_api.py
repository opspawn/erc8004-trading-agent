"""
pipeline_api.py — FastAPI server wrapping the integration Pipeline.

Endpoints:
    POST /pipeline/start    — start the pipeline
    POST /pipeline/stop     — stop the pipeline
    GET  /pipeline/status   — current state (running, trades, P&L)
    GET  /pipeline/trades   — last 50 trades

Usage:
    uvicorn pipeline_api:app --port 8090
"""

from __future__ import annotations

from typing import Any, Dict, List

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse

from pipeline import Pipeline, PipelineConfig, PipelineStatus, TradeRecord


# ─── App & Global Pipeline ────────────────────────────────────────────────────

app = FastAPI(
    title="ERC-8004 Pipeline API",
    description="Integration orchestrator for the ERC-8004 trading agent",
    version="1.0.0",
)

# Singleton pipeline instance shared across requests
_pipeline: Pipeline = Pipeline(PipelineConfig())


def get_pipeline() -> Pipeline:
    """Return the global pipeline instance."""
    return _pipeline


def set_pipeline(p: Pipeline) -> None:
    """Override the global pipeline (used in tests)."""
    global _pipeline
    _pipeline = p


# ─── Routes ───────────────────────────────────────────────────────────────────


@app.post("/pipeline/start", response_model=Dict[str, Any])
async def start_pipeline() -> Dict[str, Any]:
    """Start the trading pipeline."""
    pipeline = get_pipeline()
    if pipeline.is_running():
        raise HTTPException(status_code=409, detail="Pipeline already running")

    started = await pipeline.start()
    if not started:
        raise HTTPException(status_code=500, detail="Failed to start pipeline")

    return {
        "ok": True,
        "message": "Pipeline started",
        "status": pipeline.status().to_dict(),
    }


@app.post("/pipeline/stop", response_model=Dict[str, Any])
async def stop_pipeline() -> Dict[str, Any]:
    """Stop the trading pipeline."""
    pipeline = get_pipeline()
    if not pipeline.is_running():
        raise HTTPException(status_code=409, detail="Pipeline is not running")

    stopped = await pipeline.stop()
    if not stopped:
        raise HTTPException(status_code=500, detail="Failed to stop pipeline")

    return {
        "ok": True,
        "message": "Pipeline stopped",
        "status": pipeline.status().to_dict(),
    }


@app.get("/pipeline/status", response_model=Dict[str, Any])
async def pipeline_status() -> Dict[str, Any]:
    """Return current pipeline status."""
    pipeline = get_pipeline()
    return pipeline.status().to_dict()


@app.get("/pipeline/trades", response_model=List[Dict[str, Any]])
async def pipeline_trades(limit: int = 50) -> List[Dict[str, Any]]:
    """Return last N trades (default 50)."""
    if limit < 1 or limit > 500:
        raise HTTPException(status_code=422, detail="limit must be 1–500")
    pipeline = get_pipeline()
    trades = pipeline.get_trades(limit=limit)
    return [t.to_dict() for t in trades]


@app.get("/pipeline/health")
async def health() -> Dict[str, str]:
    """Simple health check."""
    return {"status": "ok"}


# ─── Error Handlers ───────────────────────────────────────────────────────────


@app.exception_handler(Exception)
async def generic_error_handler(request: Any, exc: Exception) -> JSONResponse:
    return JSONResponse(
        status_code=500,
        content={"detail": f"Internal error: {exc}"},
    )
