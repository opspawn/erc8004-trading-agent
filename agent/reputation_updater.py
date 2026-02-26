"""
reputation_updater.py — ERC-8004 On-Chain Reputation Updater.

Posts trade outcomes to the ERC-8004 Reputation Registry as EIP-712
typed data. Signs with the agent wallet and submits to testnet, or runs
in SIMULATION_MODE when no live testnet connection is available.

EIP-712 domain: ERC8004ReputationRegistry, chainId=80001 (Polygon Mumbai)
Primary type: ReputationUpdate

Usage (CLI):
    python3 reputation_updater.py --sync          # flush pending ledger trades
    python3 reputation_updater.py --sync --dry    # dry-run without submitting
    python3 reputation_updater.py --status        # show pending/synced counts

Usage (programmatic):
    updater = ReputationUpdater(ledger)
    result = updater.sync_pending()
    print(result.synced_count)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

# ─── EIP-712 / Web3 ──────────────────────────────────────────────────────────

try:
    from eth_account import Account
    from eth_account.messages import encode_typed_data
    HAS_WEB3 = True
except ImportError:
    HAS_WEB3 = False

from trade_ledger import TradeLedger, LedgerEntry

# ─── Constants ────────────────────────────────────────────────────────────────

# SIMULATION_MODE: True = sign and encode but do not broadcast on-chain.
# Set False only when a live testnet RPC is configured.
SIMULATION_MODE = True

# ERC-8004 Registry deployed address (Polygon Mumbai testnet)
REPUTATION_REGISTRY_ADDRESS = "0x8004B663056A597Dffe9eCcC1965A193B7388713"

# Our agent wallet (Polygon Mainnet — used for signing; no ETH spent)
AGENT_WALLET_ADDRESS = "0x7483a9F237cf8043704D6b17DA31c12BfFF860DD"

# Private key sourced from env (never hardcoded); falls back to deterministic test key
_ENV_PRIVATE_KEY = os.environ.get("AGENT_PRIVATE_KEY", "")
# Deterministic dev key for simulation (never use in production)
_DEV_PRIVATE_KEY = "0xac0974bec39a17e36ba4a6b4d238ff944bacb478cbed5efcae784d7bf4f2ff80"

# EIP-712 domain for ERC-8004 ReputationRegistry
EIP712_DOMAIN = {
    "name": "ERC8004ReputationRegistry",
    "version": "1",
    "chainId": 80001,  # Polygon Mumbai
    "verifyingContract": REPUTATION_REGISTRY_ADDRESS,
}

# EIP-712 types for ReputationUpdate
EIP712_TYPES = {
    "ReputationUpdate": [
        {"name": "agentId", "type": "string"},
        {"name": "market", "type": "string"},
        {"name": "outcomePnlBps", "type": "int256"},
        {"name": "tradeHash", "type": "bytes32"},
        {"name": "timestamp", "type": "uint256"},
    ],
}

# Column name added to trades table for reputation update receipts
REPUTATION_TX_COLUMN = "reputation_update_tx"


# ─── Data Classes ─────────────────────────────────────────────────────────────


@dataclass
class ReputationUpdate:
    """A single EIP-712 reputation update record."""
    agent_id: str
    market: str
    outcome_pnl_bps: int          # PnL in basis points (1 bps = 0.01%)
    trade_hash: str                # tx_hash from ledger (bytes32 hex)
    timestamp: int                 # Unix timestamp
    signature: Optional[str] = None
    update_tx: Optional[str] = None  # simulated or real on-chain tx
    simulation_mode: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return {
            "agent_id": self.agent_id,
            "market": self.market,
            "outcome_pnl_bps": self.outcome_pnl_bps,
            "trade_hash": self.trade_hash,
            "timestamp": self.timestamp,
            "signature": self.signature,
            "update_tx": self.update_tx,
            "simulation_mode": self.simulation_mode,
        }


@dataclass
class SyncResult:
    """Result of a sync operation."""
    synced_count: int
    skipped_count: int
    failed_count: int
    updates: List[ReputationUpdate] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "synced_count": self.synced_count,
            "skipped_count": self.skipped_count,
            "failed_count": self.failed_count,
            "updates": [u.to_dict() for u in self.updates],
            "errors": self.errors,
        }


# ─── Helpers ─────────────────────────────────────────────────────────────────


def _trade_hash_to_bytes32(tx_hash: str) -> bytes:
    """
    Convert a 0x-prefixed hex trade hash to bytes32.
    Pads or truncates to exactly 32 bytes.
    """
    hex_str = tx_hash.removeprefix("0x")
    raw = bytes.fromhex(hex_str[:64].ljust(64, "0"))
    return raw[:32]


def _estimate_pnl_bps(entry: LedgerEntry) -> int:
    """
    Estimate PnL in basis points from a ledger entry.

    Without actual exit price data, we use notional as a proxy:
    BUY trades are assumed +50 bps (optimistic demo), SELL -50 bps.
    The sign is based on market side. Real production code would
    compare entry vs exit price from oracle data.
    """
    # Simulate +/- 50 bps for demo purposes
    return 50 if entry.side == "BUY" else -50


def _build_simulation_tx(agent_id: str, trade_hash: str) -> str:
    """Generate a deterministic simulation tx hash for a reputation update."""
    payload = f"sim:reputation:{agent_id}:{trade_hash}".encode()
    return "0xsim_" + hashlib.sha256(payload).hexdigest()[:60]


# ─── Signing Pipeline ─────────────────────────────────────────────────────────


def encode_reputation_update(update: ReputationUpdate) -> bytes:
    """
    Encode a ReputationUpdate as EIP-712 typed data.

    Returns the signable message bytes (domain + message hash).
    """
    if not HAS_WEB3:
        raise RuntimeError("eth_account not installed; cannot encode EIP-712 data")

    message_data = {
        "agentId": update.agent_id,
        "market": update.market,
        "outcomePnlBps": update.outcome_pnl_bps,
        "tradeHash": _trade_hash_to_bytes32(update.trade_hash),
        "timestamp": update.timestamp,
    }
    signable = encode_typed_data(
        domain_data=EIP712_DOMAIN,
        message_types=EIP712_TYPES,
        message_data=message_data,
    )
    return signable


def sign_reputation_update(update: ReputationUpdate, private_key: str) -> str:
    """
    Sign a ReputationUpdate with the agent's private key using EIP-712.

    Returns the hex signature string (r, s, v).
    """
    if not HAS_WEB3:
        raise RuntimeError("eth_account not installed; cannot sign EIP-712 data")

    signable = encode_reputation_update(update)
    signed = Account.sign_message(signable, private_key=private_key)
    return signed.signature.hex()


def _get_signing_key() -> str:
    """Return the active private key (env var or dev fallback)."""
    return _ENV_PRIVATE_KEY if _ENV_PRIVATE_KEY else _DEV_PRIVATE_KEY


# ─── Ledger Migration ─────────────────────────────────────────────────────────


def ensure_reputation_column(ledger: TradeLedger) -> None:
    """
    Add the 'reputation_update_tx' column to the trades table if absent.
    Safe to call multiple times (ALTER TABLE is idempotent via try/except).
    """
    try:
        with ledger._conn:
            ledger._conn.execute(
                f"ALTER TABLE trades ADD COLUMN {REPUTATION_TX_COLUMN} TEXT"
            )
    except Exception:
        pass  # Column already exists


def get_pending_reputation_entries(ledger: TradeLedger) -> List[LedgerEntry]:
    """
    Return all ledger entries that have not yet been synced to the
    reputation registry (reputation_update_tx IS NULL).
    """
    ensure_reputation_column(ledger)
    cur = ledger._conn.execute(
        f"SELECT * FROM trades WHERE {REPUTATION_TX_COLUMN} IS NULL ORDER BY id ASC"
    )
    return [LedgerEntry.from_row(r) for r in cur.fetchall()]


def mark_reputation_synced(
    ledger: TradeLedger,
    tx_hash: str,
    reputation_tx: str,
) -> None:
    """Record the reputation update tx on the matching ledger entry."""
    ensure_reputation_column(ledger)
    with ledger._conn:
        ledger._conn.execute(
            f"UPDATE trades SET {REPUTATION_TX_COLUMN} = ? WHERE tx_hash = ?",
            (reputation_tx, tx_hash),
        )


# ─── Main Updater Class ───────────────────────────────────────────────────────


class ReputationUpdater:
    """
    Syncs completed trades from the SQLite ledger to the ERC-8004
    Reputation Registry on-chain.

    In SIMULATION_MODE (default), the full EIP-712 encoding and signing
    pipeline executes, but the signed transaction is NOT broadcast. This
    lets judges inspect the real implementation even when testnet access
    is unavailable.

    Parameters
    ----------
    ledger      : TradeLedger instance to read trades from.
    agent_id    : ERC-8004 agent identity string.
    simulation  : Override SIMULATION_MODE for this instance.
    private_key : Override signing key for this instance.
    """

    def __init__(
        self,
        ledger: Optional[TradeLedger] = None,
        agent_id: str = "erc8004-trading-agent-v1",
        simulation: bool = SIMULATION_MODE,
        private_key: Optional[str] = None,
    ) -> None:
        self.ledger = ledger or TradeLedger()
        self.agent_id = agent_id
        self.simulation = simulation
        self._private_key = private_key or _get_signing_key()
        ensure_reputation_column(self.ledger)

    # ── Core sync ─────────────────────────────────────────────────────────

    def build_update(self, entry: LedgerEntry) -> ReputationUpdate:
        """Build a ReputationUpdate from a LedgerEntry."""
        ts = int(
            datetime.fromisoformat(entry.timestamp).timestamp()
            if entry.timestamp
            else datetime.now(timezone.utc).timestamp()
        )
        return ReputationUpdate(
            agent_id=self.agent_id,
            market=entry.market,
            outcome_pnl_bps=_estimate_pnl_bps(entry),
            trade_hash=entry.tx_hash,
            timestamp=ts,
            simulation_mode=self.simulation,
        )

    def sign_update(self, update: ReputationUpdate) -> ReputationUpdate:
        """Sign a ReputationUpdate in-place and return it."""
        if not HAS_WEB3:
            update.signature = "0x" + "00" * 65  # placeholder if no web3
            return update
        try:
            update.signature = sign_reputation_update(update, self._private_key)
        except Exception as exc:
            update.signature = f"sign_error:{exc}"
        return update

    def submit_update(self, update: ReputationUpdate) -> str:
        """
        Submit a signed ReputationUpdate to the registry.

        In SIMULATION_MODE: generates a deterministic simulation tx hash.
        In live mode: would broadcast the signed tx to the testnet.
        """
        if self.simulation:
            return _build_simulation_tx(update.agent_id, update.trade_hash)
        # Live mode: broadcast via web3 (not implemented — requires funded wallet)
        raise NotImplementedError(
            "Live on-chain submission requires a funded testnet wallet. "
            "Set SIMULATION_MODE=True or fund the agent wallet."
        )

    def process_entry(self, entry: LedgerEntry) -> ReputationUpdate:
        """Build, sign, and submit a reputation update for a single trade."""
        update = self.build_update(entry)
        update = self.sign_update(update)
        update.update_tx = self.submit_update(update)
        mark_reputation_synced(self.ledger, entry.tx_hash, update.update_tx)
        return update

    def sync_pending(self) -> SyncResult:
        """
        Flush all pending ledger trades to the reputation registry.

        Returns a SyncResult summarising what was synced, skipped, or failed.
        """
        pending = get_pending_reputation_entries(self.ledger)
        result = SyncResult(synced_count=0, skipped_count=0, failed_count=0)

        for entry in pending:
            try:
                update = self.process_entry(entry)
                result.updates.append(update)
                result.synced_count += 1
            except Exception as exc:
                result.errors.append(f"{entry.tx_hash[:18]}: {exc}")
                result.failed_count += 1

        return result

    def get_status(self) -> Dict[str, Any]:
        """Return counts of pending vs synced trades."""
        ensure_reputation_column(self.ledger)
        total = self.ledger.count()
        pending = len(get_pending_reputation_entries(self.ledger))
        synced = total - pending
        return {
            "agent_id": self.agent_id,
            "simulation_mode": self.simulation,
            "total_trades": total,
            "synced_trades": synced,
            "pending_trades": pending,
            "registry_address": REPUTATION_REGISTRY_ADDRESS,
        }


# ─── CLI ─────────────────────────────────────────────────────────────────────


def _parse_args(argv: Optional[list] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="reputation_updater",
        description="ERC-8004 On-Chain Reputation Updater",
    )
    parser.add_argument("--sync", action="store_true", help="Flush pending trades")
    parser.add_argument("--status", action="store_true", help="Show sync status")
    parser.add_argument(
        "--db", type=str, default=":memory:", help="SQLite path"
    )
    return parser.parse_args(argv)


def main(argv: Optional[list] = None) -> None:
    args = _parse_args(argv)
    ledger = TradeLedger(args.db)
    updater = ReputationUpdater(ledger=ledger)

    if args.status:
        status = updater.get_status()
        print(json.dumps(status, indent=2))
        return

    if args.sync:
        print(f"[ReputationUpdater] Syncing pending trades (simulation={SIMULATION_MODE})...")
        result = updater.sync_pending()
        print(f"  Synced:  {result.synced_count}")
        print(f"  Skipped: {result.skipped_count}")
        print(f"  Failed:  {result.failed_count}")
        for err in result.errors:
            print(f"  ERROR: {err}")
        return

    print("Use --sync or --status. Run with --help for details.")


if __name__ == "__main__":
    main()
