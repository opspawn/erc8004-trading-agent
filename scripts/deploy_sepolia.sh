#!/usr/bin/env bash
# deploy_sepolia.sh — Deploy ERC-8004 trading contracts to Sepolia testnet.
#
# Prerequisites:
#   - SEPOLIA_RPC_URL  in .env (e.g. from Alchemy/Infura)
#   - DEPLOYER_KEY     in .env (private key with Sepolia ETH)
#
# Usage:
#   bash scripts/deploy_sepolia.sh
#
# Outputs:
#   deployment.json  — contract addresses + metadata

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
ENV_FILE="${PROJECT_ROOT}/.env"
DEPLOYMENT_JSON="${PROJECT_ROOT}/deployment.json"
DEPLOY_SCRIPT="${PROJECT_ROOT}/scripts/deploy.js"
TIMESTAMP=$(date -u +"%Y-%m-%dT%H:%M:%SZ")

log() { echo "[deploy_sepolia] $*"; }
err() { echo "[deploy_sepolia] ERROR: $*" >&2; exit 1; }

# ─── 1. Validate environment ─────────────────────────────────────────────────

log "Checking environment…"

if [[ ! -f "${ENV_FILE}" ]]; then
    err ".env file not found at ${ENV_FILE}. Copy .env.example and fill in values."
fi

# shellcheck disable=SC1090
source "${ENV_FILE}"

if [[ -z "${SEPOLIA_RPC_URL:-}" ]]; then
    err "SEPOLIA_RPC_URL is not set in .env"
fi

if [[ -z "${DEPLOYER_KEY:-}" ]]; then
    err "DEPLOYER_KEY is not set in .env"
fi

# Sanity-check key format (64 hex chars, optional 0x prefix)
KEY_STRIPPED="${DEPLOYER_KEY#0x}"
if [[ ${#KEY_STRIPPED} -ne 64 ]] || ! [[ "${KEY_STRIPPED}" =~ ^[0-9a-fA-F]{64}$ ]]; then
    err "DEPLOYER_KEY does not look like a valid private key (expected 64 hex chars)"
fi

log "Environment OK."

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

# ─── 3. Run Hardhat deploy ───────────────────────────────────────────────────

log "Deploying to Sepolia via Hardhat…"
cd "${PROJECT_ROOT}"

DEPLOY_OUTPUT=$(npx hardhat run "${DEPLOY_SCRIPT}" --network sepolia 2>&1)
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

# ─── 5. Save deployment.json ─────────────────────────────────────────────────

cat > "${DEPLOYMENT_JSON}" <<EOF
{
  "network": "sepolia",
  "contract_address": "${CONTRACT_ADDRESS}",
  "deployed_at": "${TIMESTAMP}",
  "deployer": "$(echo "${DEPLOYER_KEY}" | head -c 6)…[redacted]",
  "rpc_url": "${SEPOLIA_RPC_URL}"
}
EOF

log "Saved deployment info to ${DEPLOYMENT_JSON}"
log "Deploy complete!"
