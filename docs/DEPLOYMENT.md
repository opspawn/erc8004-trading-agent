# ERC-8004 Demo Server Deployment

## Local Development

```bash
pip install -r agent/requirements.txt
python agent/demo_server.py
# Server runs at http://localhost:8084
curl http://localhost:8084/demo/health
```

## Public Deployment (Render.com)

1. Fork/clone repository
2. Create new Web Service on Render.com
3. Connect GitHub repo
4. Build command: `pip install -r agent/requirements.txt`
5. Start command: `python agent/demo_server.py`
6. Health check: `/demo/health`

The `render.yaml` in the repo root pre-configures all of this. Render will
detect it automatically.

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `PORT`   | `8084`  | Server listen port |
| `DEV_MODE` | `true` | Bypass x402 payment gate |

## Key Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET`  | `/demo/health` | Health check |
| `POST` | `/api/v1/swarm/vote` | 10-agent stake-weighted swarm consensus |
| `GET`  | `/api/v1/risk/portfolio` | Portfolio risk (VaR 95/99%, Sharpe, Sortino) |
| `GET`  | `/api/v1/performance/summary` | Paper trading performance metrics |
| `POST` | `/api/v1/demo/showcase` | Full end-to-end pipeline demo |

## Demo Recorder

Run the automated demo recorder to capture all endpoint outputs:

```bash
python scripts/record_demo.py
# Output saved to docs/demo-screenshots/
```

## Current Demo URL

`https://erc8004-demo.onrender.com` (TBD — manual deploy step required)
