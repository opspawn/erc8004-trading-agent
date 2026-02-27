#!/usr/bin/env python3
"""Automated demo recorder using requests + screenshots.
Generates a sequence of terminal command outputs for demo video."""
import json
import sys
from pathlib import Path

try:
    import requests
except ImportError:
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "requests", "-q"])
    import requests

BASE_URL = "http://localhost:8084"

DEMO_STEPS = [
    {"name": "health", "method": "GET", "path": "/demo/health", "desc": "Step 1: Health Check"},
    {"name": "swarm_vote", "method": "POST", "path": "/api/v1/swarm/vote", "body": {"signal": "LONG", "asset": "BTC-USD"}, "desc": "Step 2: 10-Agent Swarm Vote"},
    {"name": "risk_portfolio", "method": "GET", "path": "/api/v1/risk/portfolio", "desc": "Step 3: Portfolio Risk"},
    {"name": "performance", "method": "GET", "path": "/api/v1/performance/summary", "desc": "Step 4: Performance Summary"},
    {"name": "showcase", "method": "POST", "path": "/api/v1/demo/showcase", "desc": "Step 5: Live Showcase"},
]


def run_step(step):
    url = BASE_URL + step["path"]
    if step["method"] == "GET":
        r = requests.get(url, timeout=10)
    else:
        r = requests.post(url, json=step.get("body", {}), timeout=10)
    r.raise_for_status()
    return r.json()


def main():
    output_dir = Path(__file__).parent.parent / "docs" / "demo-screenshots"
    output_dir.mkdir(exist_ok=True)

    results = []
    for step in DEMO_STEPS:
        print(f"\n{'='*60}")
        print(f"  {step['desc']}")
        print('='*60)
        try:
            result = run_step(step)
            print(json.dumps(result, indent=2))
            results.append({"step": step["name"], "status": "ok", "response": result})
            # Save to file
            (output_dir / f"{step['name']}.json").write_text(json.dumps(result, indent=2))
        except Exception as e:
            print(f"ERROR: {e}")
            results.append({"step": step["name"], "status": "error", "error": str(e)})

    # Summary
    ok_count = sum(1 for r in results if r['status'] == 'ok')
    print(f"\n\nDemo complete: {ok_count}/{len(results)} steps passed")
    (output_dir / "demo-results.json").write_text(json.dumps(results, indent=2))
    return 0 if all(r["status"] == "ok" for r in results) else 1


if __name__ == "__main__":
    sys.exit(main())
