#!/usr/bin/env bash
# deploy-arbitrum-sepolia.sh — Deploy ERC-8004 trading contracts to Arbitrum Sepolia testnet.
#
# Chain: Arbitrum Sepolia (chain ID 421614)
# RPC:   https://sepolia-rollup.arbitrum.io/rpc
# Explorer: https://sepolia.arbiscan.io
#
# Prerequisites:
#   - ARBITRUM_SEPOLIA_RPC_URL in .env (defaults to public RPC)
#   - DEPLOYER_KEY              in .env (private key with Arbitrum Sepolia ETH)
#   - Get free testnet ETH from https://www.alchemy.com/faucets/arbitrum-sepolia
#
# Usage:
#   bash scripts/deploy-arbitrum-sepolia.sh
#
# Outputs:
#   deployment-arbitrum-sepolia.json  — contract addresses + metadata

set -euo pipefail

# ─── Constants ────────────────────────────────────────────────────────────────

ARBITRUM_SEPOLIA_CHAIN_ID=421614
ARBITRUM_SEPOLIA_DEFAULT_RPC="https://sepolia-rollup.arbitrum.io/rpc"
ARBITRUM_SEPOLIA_EXPLORER="https://sepolia.arbiscan.io"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
ENV_FILE="${PROJECT_ROOT}/.env"
DEPLOYMENT_JSON="${PROJECT_ROOT}/deployment-arbitrum-sepolia.json"
DEPLOY_SCRIPT="${PROJECT_ROOT}/scripts/deploy.js"
TIMESTAMP=$(date -u +"%Y-%m-%dT%H:%M:%SZ")

log() { echo "[deploy-arbitrum-sepolia] $*"; }
err() { echo "[deploy-arbitrum-sepolia] ERROR: $*" >&2; exit 1; }

# ─── 1. Validate environment ─────────────────────────────────────────────────

log "Checking environment…"

if [[ ! -f "${ENV_FILE}" ]]; then
    err ".env file not found at ${ENV_FILE}. Copy .env.example and fill in values."
fi

# shellcheck disable=SC1090
source "${ENV_FILE}"

# Use Arbitrum Sepolia RPC or fall back to default public endpoint
ARBI_RPC="${ARBITRUM_SEPOLIA_RPC_URL:-${ARBITRUM_SEPOLIA_DEFAULT_RPC}}"
log "Using RPC: ${ARBI_RPC}"

if [[ -z "${DEPLOYER_KEY:-}" ]]; then
    err "DEPLOYER_KEY is not set in .env"
fi

# Sanity-check key format (64 hex chars, optional 0x prefix)
KEY_STRIPPED="${DEPLOYER_KEY#0x}"
if [[ ${#KEY_STRIPPED} -ne 64 ]] || ! [[ "${KEY_STRIPPED}" =~ ^[0-9a-fA-F]{64}$ ]]; then
    err "DEPLOYER_KEY does not look like a valid private key (expected 64 hex chars)"
fi

log "Environment OK. Chain ID: ${ARBITRUM_SEPOLIA_CHAIN_ID}"

# ─── 2. Check dependencies ───────────────────────────────────────────────────

log "Checking dependencies…"

if ! command -v node &>/dev/null; then
    err "node is not installed"
fi

if ! command -v npx &>/dev/null; then
    err "npx is not installed"
fi

if [[ ! -d "${PROJECT_ROOT}/node_modules" ]]; then
    log "node_modules not found — running npm install…"
    npm install --prefix "${PROJECT_ROOT}"
fi

log "Dependencies OK."

# ─── 3. Run Hardhat deploy on Arbitrum Sepolia ───────────────────────────────

log "Deploying to Arbitrum Sepolia (chain ${ARBITRUM_SEPOLIA_CHAIN_ID}) via Hardhat…"
cd "${PROJECT_ROOT}"

# Export ARBITRUM_SEPOLIA_RPC_URL so hardhat.config.ts picks it up
export ARBITRUM_SEPOLIA_RPC_URL="${ARBI_RPC}"
export DEPLOYER_PRIVATE_KEY="${DEPLOYER_KEY}"

DEPLOY_OUTPUT=$(npx hardhat run "${DEPLOY_SCRIPT}" --network arbitrumSepolia 2>&1)
EXIT_CODE=$?

echo "${DEPLOY_OUTPUT}"

if [[ ${EXIT_CODE} -ne 0 ]]; then
    err "Hardhat deploy failed (exit code ${EXIT_CODE})"
fi

# ─── 4. Parse contract address ───────────────────────────────────────────────

# Deploy script should print: "Contract deployed to: 0x..."
CONTRACT_ADDRESS=$(echo "${DEPLOY_OUTPUT}" | grep -oE '0x[0-9a-fA-F]{40}' | tail -1)

if [[ -z "${CONTRACT_ADDRESS}" ]]; then
    err "Could not parse contract address from deploy output"
fi

log "Contract deployed at: ${CONTRACT_ADDRESS}"
log "Explorer: ${ARBITRUM_SEPOLIA_EXPLORER}/address/${CONTRACT_ADDRESS}"

# ─── 5. Save deployment-arbitrum-sepolia.json ────────────────────────────────

cat > "${DEPLOYMENT_JSON}" <<EOF
{
  "network": "arbitrum-sepolia",
  "chain_id": ${ARBITRUM_SEPOLIA_CHAIN_ID},
  "contract_address": "${CONTRACT_ADDRESS}",
  "deployed_at": "${TIMESTAMP}",
  "deployer": "$(echo "${DEPLOYER_KEY}" | head -c 6)…[redacted]",
  "rpc_url": "${ARBI_RPC}",
  "explorer_url": "${ARBITRUM_SEPOLIA_EXPLORER}/address/${CONTRACT_ADDRESS}"
}
EOF

log "Saved deployment info to ${DEPLOYMENT_JSON}"
log ""
log "=== Arbitrum Sepolia Deployment Complete ==="
log "Contract:  ${CONTRACT_ADDRESS}"
log "Chain ID:  ${ARBITRUM_SEPOLIA_CHAIN_ID}"
log "Explorer:  ${ARBITRUM_SEPOLIA_EXPLORER}/address/${CONTRACT_ADDRESS}"
log "Deploy complete!"
