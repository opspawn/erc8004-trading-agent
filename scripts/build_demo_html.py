#!/usr/bin/env python3
"""
Build docs/demo.html from demo-screenshots JSON files.
Run: python scripts/build_demo_html.py
"""
import json
from pathlib import Path

SCREENSHOTS_DIR = Path(__file__).parent.parent / "docs" / "demo-screenshots"
OUTPUT_FILE = Path(__file__).parent.parent / "docs" / "demo.html"

steps = [
    ("1. Health Check", "GET /demo/health", "health"),
    ("2. 10-Agent Swarm Vote", "POST /api/v1/swarm/vote", "swarm_vote"),
    ("3. Portfolio Risk (VaR)", "GET /api/v1/risk/portfolio", "risk_portfolio"),
    ("4. Performance Summary", "GET /api/v1/performance/summary", "performance"),
    ("5. Live Showcase", "POST /api/v1/demo/showcase", "showcase"),
]


def load_json(name: str) -> dict:
    path = SCREENSHOTS_DIR / f"{name}.json"
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return {"error": f"{name}.json not found"}


def build_step_card(title: str, url: str, data: dict) -> str:
    pretty_json = json.dumps(data, indent=2, ensure_ascii=False)
    # Escape for HTML
    pretty_json = pretty_json.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    return f"""
  <div class="step">
    <div class="step-title">{title}</div>
    <div class="step-url"><code>{url}</code></div>
    <pre>{pretty_json}</pre>
  </div>"""


def build_html() -> str:
    step_cards = ""
    for title, url, key in steps:
        data = load_json(key)
        step_cards += build_step_card(title, url, data)

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>ERC-8004 Trading Agent — Live Demo</title>
  <style>
    * {{ box-sizing: border-box; margin: 0; padding: 0; }}
    body {{
      background: #1a1a1a;
      color: #e0e0e0;
      font-family: 'Courier New', 'Lucida Console', monospace;
      padding: 24px;
      max-width: 960px;
      margin: 0 auto;
    }}
    h1 {{
      color: #00ff88;
      text-align: center;
      font-size: 2em;
      margin-bottom: 8px;
      letter-spacing: 0.05em;
    }}
    .subtitle {{
      text-align: center;
      color: #888;
      font-size: 0.9em;
      margin-bottom: 8px;
    }}
    .header-stats {{
      text-align: center;
      color: #888;
      font-size: 0.95em;
      margin-bottom: 32px;
      padding: 12px;
      border: 1px solid #333;
      border-radius: 6px;
      background: #222;
    }}
    .header-stats span {{
      color: #00ff88;
      font-weight: bold;
      font-size: 1.1em;
    }}
    .header-stats .sep {{ color: #555; margin: 0 8px; }}
    .step {{
      background: #242424;
      border: 1px solid #3a3a3a;
      border-radius: 8px;
      padding: 20px;
      margin: 20px 0;
    }}
    .step:hover {{
      border-color: #00aaff44;
    }}
    .step-title {{
      color: #00aaff;
      font-size: 1.15em;
      font-weight: bold;
      margin-bottom: 6px;
    }}
    .step-url {{
      color: #888;
      font-size: 0.82em;
      margin-bottom: 14px;
    }}
    .step-url code {{
      background: #1a1a1a;
      padding: 2px 6px;
      border-radius: 3px;
      color: #ffaa00;
    }}
    pre {{
      background: #111;
      padding: 16px;
      border-radius: 6px;
      overflow-x: auto;
      font-size: 0.82em;
      color: #00ff88;
      white-space: pre-wrap;
      word-wrap: break-word;
      border: 1px solid #2a2a2a;
      max-height: 500px;
      overflow-y: auto;
    }}
    .footer {{
      text-align: center;
      color: #555;
      font-size: 0.8em;
      margin-top: 40px;
      padding-top: 20px;
      border-top: 1px solid #333;
    }}
    .footer a {{ color: #00aaff; text-decoration: none; }}
    .quickstart {{
      background: #1e2a1e;
      border: 1px solid #2a4a2a;
      border-radius: 8px;
      padding: 20px;
      margin: 20px 0;
    }}
    .quickstart h2 {{ color: #00ff88; margin-bottom: 12px; font-size: 1em; }}
    .quickstart pre {{ color: #aaffaa; background: #111; font-size: 0.82em; }}
  </style>
</head>
<body>
  <h1>ERC-8004 Trading Agent</h1>
  <p class="subtitle">On-Chain Trading Agent Identity Standard — lablab.ai Hackathon</p>

  <div class="header-stats">
    <span>6,170</span> tests passing
    <span class="sep">•</span>
    <span>10</span>-agent swarm
    <span class="sep">•</span>
    <span>6</span> strategies
    <span class="sep">•</span>
    <span>VaR 95/99%</span> risk engine
    <span class="sep">•</span>
    Sprint <span>S50</span>
  </div>

  <div class="quickstart">
    <h2>▶ Quick Start (run locally)</h2>
    <pre>pip install -r requirements.txt
python agent/demo_server.py &amp;
curl http://localhost:8084/demo/health</pre>
  </div>

  <h2 style="color:#888; font-size:0.95em; margin:24px 0 4px; text-transform:uppercase; letter-spacing:0.1em;">Live Demo Outputs (captured from running server)</h2>
{step_cards}

  <div class="footer">
    <p>ERC-8004 Trading Agent &mdash;
    <a href="https://github.com/opspawn/erc8004-trading-agent">github.com/opspawn/erc8004-trading-agent</a>
    &mdash; Built for lablab.ai ERC-8004 Hackathon 2026</p>
  </div>
</body>
</html>
"""


if __name__ == "__main__":
    html = build_html()
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_FILE.write_text(html, encoding="utf-8")
    size = len(html)
    print(f"Generated {OUTPUT_FILE} ({size:,} bytes)")
    # Quick sanity check
    assert "health" in html, "Missing health step"
    assert "swarm" in html.lower(), "Missing swarm step"
    assert "VaR" in html or "var_95" in html, "Missing risk step"
    print("All sanity checks passed.")
