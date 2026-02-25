#!/usr/bin/env bash
# deploy-demo.sh — Deploy ERC-8004 Trading Agent Dashboard to Vercel
# Usage: ./scripts/deploy-demo.sh [--preview]
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

echo "=== ERC-8004 Trading Agent — Demo Deployment ==="
echo ""

# Prerequisites check
if ! command -v vercel &>/dev/null; then
  echo "[install] Installing Vercel CLI..."
  npm install -g vercel
fi

echo "[check] Node: $(node --version)"
echo "[check] Vercel: $(vercel --version)"
echo ""

# Install dashboard deps
echo "[install] Installing dashboard dependencies..."
cd "$ROOT/dashboard"
npm install --silent
echo "[install] Done."
echo ""

# Build locally first to catch errors
echo "[build] Building Next.js dashboard..."
npm run build
echo "[build] Build succeeded."
echo ""

cd "$ROOT"

# Deploy
if [[ "${1:-}" == "--preview" ]]; then
  echo "[deploy] Deploying preview..."
  vercel deploy
else
  echo "[deploy] Deploying to production..."
  vercel deploy --prod
fi

echo ""
echo "=== Deployment complete ==="
echo "Dashboard URL will appear above."
echo "Features:"
echo "  • Agent identity card (ERC-8004 address + reputation)"
echo "  • Live P&L simulation with chart"
echo "  • Strategy status (active/paused)"
echo "  • Credora risk ratings display"
echo "  • Recent trades feed"
