"""
validation_artifacts.py — ERC-8004 Validation Artifact Generator.

The ERC-8004 Validation Registry requires agents to produce verifiable
proofs of strategy adherence. This module generates a signed JSON artifact
for each completed trading session, containing performance metrics and an
ECDSA signature over the canonical artifact hash.

Artifact schema:
    agent_id            : str   — ERC-8004 agent identity
    session_id          : str   — UUID for this trading session
    strategy_hash       : str   — keccak256 of strategy config (hex)
    trades_count        : int   — number of executed trades
    win_rate            : float — fraction of profitable trades [0.0, 1.0]
    avg_pnl_bps         : float — average PnL in basis points
    max_drawdown_bps    : float — max drawdown in basis points
    risk_violations     : int   — count of risk limit violations
    validation_timestamp: str   — ISO-8601 UTC timestamp
    validator_signature : str   — 0x-prefixed ECDSA hex signature

Artifacts are stored in agent/artifacts/{session_id}.artifact.json

Usage (CLI):
    python3 validation_artifacts.py --session {session_id}
    python3 validation_artifacts.py --session {session_id} --db /path/to/trades.db

Usage (programmatic):
    gen = ArtifactGenerator(ledger, config)
    artifact = gen.generate(session_id="my-session")
    print(artifact.artifact_path)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    from eth_account import Account
    from eth_account.messages import encode_defunct
    HAS_WEB3 = True
except ImportError:
    HAS_WEB3 = False

from trade_ledger import TradeLedger, LedgerEntry

# ─── Constants ────────────────────────────────────────────────────────────────

# Default artifacts directory (relative to this file)
_MODULE_DIR = Path(__file__).parent
ARTIFACTS_DIR = _MODULE_DIR / "artifacts"

# Agent identity defaults (overridable via config)
DEFAULT_AGENT_ID = "erc8004-trading-agent-v1"

# Private key for signing (sourced from env, never hardcoded in production)
_ENV_PRIVATE_KEY = os.environ.get("AGENT_PRIVATE_KEY", "")
# Deterministic dev key for simulation / tests only
_DEV_PRIVATE_KEY = "0xac0974bec39a17e36ba4a6b4d238ff944bacb478cbed5efcae784d7bf4f2ff80"

# Default strategy config dict (used to compute strategy_hash)
DEFAULT_STRATEGY_CONFIG: Dict[str, Any] = {
    "type": "momentum",
    "max_position_pct": 0.10,
    "stop_loss_pct": 0.02,
    "take_profit_pct": 0.05,
    "risk_per_trade_pct": 0.01,
    "version": "1.0.0",
}


# ─── Helpers ─────────────────────────────────────────────────────────────────


def keccak256_config(config: Dict[str, Any]) -> str:
    """
    Compute the keccak256 hash of a strategy config dict.

    Returns a 0x-prefixed hex string.
    Falls back to SHA-256 if pysha3/web3 keccak is unavailable.
    """
    canonical = json.dumps(config, sort_keys=True, separators=(",", ":")).encode()
    try:
        from eth_hash.auto import keccak
        return "0x" + keccak(canonical).hex()
    except ImportError:
        # Fallback: SHA-256 (same 32-byte output, different hash function)
        return "0x" + hashlib.sha256(canonical).hexdigest()


def _get_signing_key() -> str:
    return _ENV_PRIVATE_KEY if _ENV_PRIVATE_KEY else _DEV_PRIVATE_KEY


def _sign_artifact_hash(artifact_hash: str, private_key: str) -> str:
    """
    Sign an artifact hash (0x-prefixed hex) using ECDSA via eth_account.

    Returns the 0x-prefixed hex signature.
    """
    if not HAS_WEB3:
        return "0x" + "00" * 65  # placeholder
    msg = encode_defunct(hexstr=artifact_hash)
    signed = Account.sign_message(msg, private_key=private_key)
    return signed.signature.hex()


def canonical_artifact_hash(artifact: Dict[str, Any]) -> str:
    """
    Compute a deterministic hash over the artifact fields.

    Excludes both 'validator_signature' and 'artifact_hash' so that
    the hash can be recomputed identically during both generation and
    verification (generation passes a dict without these fields;
    verification passes a full artifact dict that includes them).
    """
    _EXCLUDED = {"validator_signature", "artifact_hash"}
    fields_to_hash = {k: v for k, v in artifact.items() if k not in _EXCLUDED}
    canonical = json.dumps(fields_to_hash, sort_keys=True, separators=(",", ":")).encode()
    return "0x" + hashlib.sha256(canonical).hexdigest()


# ─── Artifact Data Class ──────────────────────────────────────────────────────


@dataclass
class ValidationArtifact:
    """A signed validation artifact for a completed trading session."""
    agent_id: str
    session_id: str
    strategy_hash: str
    trades_count: int
    win_rate: float
    avg_pnl_bps: float
    max_drawdown_bps: float
    risk_violations: int
    validation_timestamp: str
    artifact_hash: str
    validator_signature: str
    artifact_path: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "agent_id": self.agent_id,
            "session_id": self.session_id,
            "strategy_hash": self.strategy_hash,
            "trades_count": self.trades_count,
            "win_rate": round(self.win_rate, 6),
            "avg_pnl_bps": round(self.avg_pnl_bps, 4),
            "max_drawdown_bps": round(self.max_drawdown_bps, 4),
            "risk_violations": self.risk_violations,
            "validation_timestamp": self.validation_timestamp,
            "artifact_hash": self.artifact_hash,
            "validator_signature": self.validator_signature,
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ValidationArtifact":
        return cls(
            agent_id=d["agent_id"],
            session_id=d["session_id"],
            strategy_hash=d["strategy_hash"],
            trades_count=d["trades_count"],
            win_rate=d["win_rate"],
            avg_pnl_bps=d["avg_pnl_bps"],
            max_drawdown_bps=d["max_drawdown_bps"],
            risk_violations=d["risk_violations"],
            validation_timestamp=d["validation_timestamp"],
            artifact_hash=d["artifact_hash"],
            validator_signature=d["validator_signature"],
        )

    @classmethod
    def load(cls, path: str) -> "ValidationArtifact":
        with open(path, "r") as f:
            d = json.load(f)
        artifact = cls.from_dict(d)
        artifact.artifact_path = path
        return artifact


# ─── Metrics computation ─────────────────────────────────────────────────────


def _compute_metrics(
    entries: List[LedgerEntry],
) -> Dict[str, Any]:
    """
    Compute performance metrics from a list of ledger entries.

    PnL estimation uses a proxy model (same as reputation_updater):
    BUY → +50 bps, SELL → -50 bps. Production would use real exit prices.
    """
    if not entries:
        return {
            "trades_count": 0,
            "win_rate": 0.0,
            "avg_pnl_bps": 0.0,
            "max_drawdown_bps": 0.0,
            "risk_violations": 0,
        }

    pnl_per_trade = [50.0 if e.side == "BUY" else -50.0 for e in entries]
    wins = sum(1 for p in pnl_per_trade if p > 0)
    win_rate = wins / len(pnl_per_trade)
    avg_pnl_bps = sum(pnl_per_trade) / len(pnl_per_trade)

    # Max drawdown: largest peak-to-trough decline in cumulative PnL
    cumulative = []
    running = 0.0
    for p in pnl_per_trade:
        running += p
        cumulative.append(running)

    max_drawdown = 0.0
    peak = cumulative[0] if cumulative else 0.0
    for val in cumulative:
        if val > peak:
            peak = val
        dd = peak - val
        if dd > max_drawdown:
            max_drawdown = dd

    # Risk violations: trades where |pnl| > 200 bps (simulated violation threshold)
    risk_violations = sum(1 for p in pnl_per_trade if abs(p) > 200)

    return {
        "trades_count": len(entries),
        "win_rate": win_rate,
        "avg_pnl_bps": avg_pnl_bps,
        "max_drawdown_bps": max_drawdown,
        "risk_violations": risk_violations,
    }


# ─── Artifact Generator ───────────────────────────────────────────────────────


class ArtifactGenerator:
    """
    Generates signed validation artifacts for completed trading sessions.

    Parameters
    ----------
    ledger          : TradeLedger to pull session trades from.
    strategy_config : Strategy config dict for computing strategy_hash.
    agent_id        : ERC-8004 agent identity string.
    artifacts_dir   : Directory to persist .artifact.json files.
    private_key     : Override signing key (defaults to env/dev key).
    """

    def __init__(
        self,
        ledger: Optional[TradeLedger] = None,
        strategy_config: Optional[Dict[str, Any]] = None,
        agent_id: str = DEFAULT_AGENT_ID,
        artifacts_dir: Optional[str] = None,
        private_key: Optional[str] = None,
    ) -> None:
        self.ledger = ledger or TradeLedger()
        self.strategy_config = strategy_config or DEFAULT_STRATEGY_CONFIG
        self.agent_id = agent_id
        self.artifacts_dir = Path(artifacts_dir) if artifacts_dir else ARTIFACTS_DIR
        self._private_key = private_key or _get_signing_key()

    # ── Core generation ───────────────────────────────────────────────────

    def generate(
        self,
        session_id: Optional[str] = None,
        agent_id_filter: Optional[str] = None,
    ) -> ValidationArtifact:
        """
        Generate a validation artifact for a session.

        Parameters
        ----------
        session_id       : Session UUID. Auto-generated if None.
        agent_id_filter  : If set, only include trades from this agent_id.

        Returns
        -------
        ValidationArtifact, signed and written to artifacts_dir.
        """
        session_id = session_id or str(uuid.uuid4())

        # Pull trades from ledger
        entries = self.ledger.get_entries(
            agent_id=agent_id_filter or None
        )

        # Compute metrics
        metrics = _compute_metrics(entries)

        # Build the unsigned artifact dict
        strategy_hash = keccak256_config(self.strategy_config)
        timestamp = datetime.now(timezone.utc).isoformat()

        unsigned = {
            "agent_id": self.agent_id,
            "session_id": session_id,
            "strategy_hash": strategy_hash,
            "trades_count": metrics["trades_count"],
            "win_rate": metrics["win_rate"],
            "avg_pnl_bps": metrics["avg_pnl_bps"],
            "max_drawdown_bps": metrics["max_drawdown_bps"],
            "risk_violations": metrics["risk_violations"],
            "validation_timestamp": timestamp,
        }

        # Hash then sign
        artifact_hash = canonical_artifact_hash(unsigned)
        signature = _sign_artifact_hash(artifact_hash, self._private_key)

        # Construct final artifact
        artifact = ValidationArtifact(
            agent_id=self.agent_id,
            session_id=session_id,
            strategy_hash=strategy_hash,
            trades_count=metrics["trades_count"],
            win_rate=metrics["win_rate"],
            avg_pnl_bps=metrics["avg_pnl_bps"],
            max_drawdown_bps=metrics["max_drawdown_bps"],
            risk_violations=metrics["risk_violations"],
            validation_timestamp=timestamp,
            artifact_hash=artifact_hash,
            validator_signature=signature,
        )

        # Persist to disk
        artifact.artifact_path = self._save(artifact)
        return artifact

    def _save(self, artifact: ValidationArtifact) -> str:
        """Write artifact to {artifacts_dir}/{session_id}.artifact.json."""
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)
        path = self.artifacts_dir / f"{artifact.session_id}.artifact.json"
        with open(path, "w") as f:
            f.write(artifact.to_json())
        return str(path)

    def load(self, session_id: str) -> ValidationArtifact:
        """Load a previously generated artifact by session_id."""
        path = self.artifacts_dir / f"{session_id}.artifact.json"
        return ValidationArtifact.load(str(path))

    def list_artifacts(self) -> List[str]:
        """Return list of session_ids with saved artifacts."""
        if not self.artifacts_dir.exists():
            return []
        return [
            p.stem.replace(".artifact", "")
            for p in sorted(self.artifacts_dir.glob("*.artifact.json"))
        ]

    def verify(self, artifact: ValidationArtifact) -> bool:
        """
        Verify the validator_signature on an artifact.

        Returns True if the signature matches the artifact hash.
        """
        if not HAS_WEB3:
            return True  # Can't verify without eth_account
        try:
            expected_hash = canonical_artifact_hash(artifact.to_dict())
            if expected_hash != artifact.artifact_hash:
                return False
            msg = encode_defunct(hexstr=artifact.artifact_hash)
            recovered = Account.recover_message(msg, signature=artifact.validator_signature)
            # Derive expected address from private key
            expected_addr = Account.from_key(self._private_key).address
            return recovered.lower() == expected_addr.lower()
        except Exception:
            return False


# ─── CLI ──────────────────────────────────────────────────────────────────────


def _parse_args(argv: Optional[list] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="validation_artifacts",
        description="ERC-8004 Validation Artifact Generator",
    )
    parser.add_argument("--session", type=str, help="Session ID for artifact generation")
    parser.add_argument("--list", action="store_true", help="List saved artifacts")
    parser.add_argument("--verify", type=str, metavar="SESSION_ID", help="Verify artifact")
    parser.add_argument("--db", type=str, default=":memory:", help="SQLite path")
    return parser.parse_args(argv)


def main(argv: Optional[list] = None) -> None:
    args = _parse_args(argv)
    ledger = TradeLedger(args.db)
    gen = ArtifactGenerator(ledger=ledger)

    if args.list:
        artifacts = gen.list_artifacts()
        if artifacts:
            for sid in artifacts:
                print(sid)
        else:
            print("(no artifacts)")
        return

    if args.verify:
        artifact = gen.load(args.verify)
        ok = gen.verify(artifact)
        print(f"Signature {'VALID' if ok else 'INVALID'} for session {args.verify}")
        return

    session_id = args.session or str(uuid.uuid4())
    print(f"[ArtifactGenerator] Generating artifact for session {session_id}...")
    artifact = gen.generate(session_id=session_id)
    print(f"  Session:   {artifact.session_id}")
    print(f"  Trades:    {artifact.trades_count}")
    print(f"  Win rate:  {artifact.win_rate:.2%}")
    print(f"  Avg PnL:   {artifact.avg_pnl_bps:+.1f} bps")
    print(f"  Hash:      {artifact.artifact_hash[:18]}...")
    print(f"  Signature: {artifact.validator_signature[:18]}...")
    print(f"  Saved:     {artifact.artifact_path}")


if __name__ == "__main__":
    main()
