"""
verify_deploy.py — Verify an ERC-8004 contract deployment on Sepolia.

Loads deployment.json, calls read-only contract functions to confirm
the contract is live and responding correctly.

Usage:
    python3 scripts/verify_deploy.py [--deployment deployment.json]

Exit codes:
    0 — verification passed
    1 — verification failed
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Optional


# ─── Minimal ABI for verification calls ──────────────────────────────────────

ERC8004_VERIFY_ABI = [
    {
        "name": "feedbackCount",
        "type": "function",
        "stateMutability": "view",
        "inputs": [{"name": "agentId", "type": "address"}],
        "outputs": [{"name": "", "type": "uint256"}],
    },
    {
        "name": "getAggregateScore",
        "type": "function",
        "stateMutability": "view",
        "inputs": [{"name": "agentId", "type": "address"}],
        "outputs": [
            {"name": "score", "type": "uint256"},
            {"name": "count", "type": "uint256"},
        ],
    },
]

# Dummy address used for read-only verification calls
ZERO_ADDRESS = "0x0000000000000000000000000000000000000000"


# ─── Verification Logic ───────────────────────────────────────────────────────


def load_deployment(path: str) -> Dict[str, Any]:
    """Load and validate deployment.json."""
    deployment_path = Path(path)
    if not deployment_path.exists():
        raise FileNotFoundError(f"deployment.json not found at {path}")

    data = json.loads(deployment_path.read_text())

    required_fields = ["contract_address", "network", "deployed_at"]
    for field in required_fields:
        if field not in data:
            raise ValueError(f"deployment.json missing required field: {field}")

    address = data["contract_address"]
    if not address.startswith("0x") or len(address) != 42:
        raise ValueError(f"Invalid contract address in deployment.json: {address}")

    return data


def verify_contract(
    contract_address: str,
    rpc_url: str,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Call read-only contract functions to verify deployment.

    Returns dict with verification results.
    """
    result: Dict[str, Any] = {
        "contract_address": contract_address,
        "rpc_url": rpc_url,
        "checks": {},
        "passed": False,
        "errors": [],
    }

    try:
        from web3 import Web3

        w3 = Web3(Web3.HTTPProvider(rpc_url))

        # Check connection
        if not w3.is_connected():
            result["errors"].append("Cannot connect to RPC endpoint")
            return result

        result["checks"]["rpc_connected"] = True

        # Check contract has bytecode (is deployed)
        code = w3.eth.get_code(contract_address)
        if code == b"" or code == b"0x":
            result["errors"].append("No bytecode at contract address — not deployed")
            result["checks"]["has_bytecode"] = False
            return result

        result["checks"]["has_bytecode"] = True
        result["checks"]["bytecode_length"] = len(code)

        # Create contract instance
        contract = w3.eth.contract(
            address=Web3.to_checksum_address(contract_address),
            abi=ERC8004_VERIFY_ABI,
        )

        # Call feedbackCount (should return 0 for fresh deployment)
        feedback_count = contract.functions.feedbackCount(ZERO_ADDRESS).call()
        result["checks"]["feedbackCount_call"] = True
        result["checks"]["feedbackCount_value"] = feedback_count

        # Call getAggregateScore
        score, count = contract.functions.getAggregateScore(ZERO_ADDRESS).call()
        result["checks"]["getAggregateScore_call"] = True
        result["checks"]["aggregate_score"] = score
        result["checks"]["aggregate_count"] = count

        result["passed"] = True

        if verbose:
            print(f"[verify_deploy] Contract: {contract_address}")
            print(f"[verify_deploy] Network:  {rpc_url}")
            print(f"[verify_deploy] Bytecode: {len(code)} bytes")
            print(f"[verify_deploy] feedbackCount(0x0): {feedback_count}")
            print(f"[verify_deploy] getAggregateScore(0x0): score={score}, count={count}")
            print("[verify_deploy] ✓ Verification PASSED")

    except ImportError:
        result["errors"].append("web3 not installed — run: pip install web3")
    except Exception as exc:
        result["errors"].append(str(exc))

    return result


def print_status(result: Dict[str, Any]) -> None:
    """Print human-readable verification status."""
    print("\n=== Deployment Verification ===")
    print(f"Contract:  {result['contract_address']}")
    print(f"RPC:       {result['rpc_url']}")
    print(f"Status:    {'PASSED ✓' if result['passed'] else 'FAILED ✗'}")

    if result["checks"]:
        print("\nChecks:")
        for k, v in result["checks"].items():
            print(f"  {k}: {v}")

    if result["errors"]:
        print("\nErrors:")
        for e in result["errors"]:
            print(f"  ! {e}")
    print()


# ─── CLI Entry Point ──────────────────────────────────────────────────────────


def main(argv: Optional[list] = None) -> int:
    parser = argparse.ArgumentParser(description="Verify ERC-8004 Sepolia deployment")
    parser.add_argument(
        "--deployment",
        default="deployment.json",
        help="Path to deployment.json (default: deployment.json)",
    )
    parser.add_argument("--quiet", action="store_true", help="Suppress output")
    args = parser.parse_args(argv)

    try:
        deployment = load_deployment(args.deployment)
    except (FileNotFoundError, ValueError) as exc:
        print(f"[verify_deploy] ERROR: {exc}", file=sys.stderr)
        return 1

    contract_address = deployment["contract_address"]
    rpc_url = deployment.get("rpc_url", "")

    if not rpc_url:
        print("[verify_deploy] ERROR: rpc_url not in deployment.json", file=sys.stderr)
        return 1

    result = verify_contract(contract_address, rpc_url, verbose=not args.quiet)

    if not args.quiet:
        print_status(result)

    return 0 if result["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
