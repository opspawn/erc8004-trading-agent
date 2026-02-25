"""
trade_ledger.py — SQLite Trade Execution Ledger for ERC-8004 Trading Agent.

Records every trade decision with a txHash placeholder for on-chain proof.
Supports in-memory (for tests) and file-based persistence.

Schema:
    CREATE TABLE trades (
        id          INTEGER PRIMARY KEY AUTOINCREMENT,
        tx_hash     TEXT NOT NULL,
        agent_id    TEXT NOT NULL,
        market      TEXT NOT NULL,
        side        TEXT NOT NULL CHECK(side IN ('BUY','SELL')),
        size        REAL NOT NULL CHECK(size > 0),
        price       REAL NOT NULL CHECK(price > 0),
        notional    REAL GENERATED ALWAYS AS (size * price) STORED,
        timestamp   TEXT NOT NULL,
        status      TEXT NOT NULL DEFAULT 'pending',
        created_at  TEXT NOT NULL
    );

Usage:
    ledger = TradeLedger()                       # in-memory (test/dry-run)
    ledger = TradeLedger("/data/trades.db")      # persistent

    entry = ledger.log_trade(
        agent_id="agent-001",
        market="BTC/USD",
        side="BUY",
        size=0.5,
        price=50000.0,
    )
    print(entry.tx_hash)   # "0x0000...{uuid}"
    entries = ledger.get_entries(agent_id="agent-001")
    summary = ledger.get_summary()
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, Generator, Iterator, List, Optional


# ─── Constants ────────────────────────────────────────────────────────────────

VALID_SIDES = frozenset({"BUY", "SELL"})
VALID_STATUSES = frozenset({"pending", "confirmed", "failed"})

_CREATE_TABLE = """
CREATE TABLE IF NOT EXISTS trades (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    tx_hash     TEXT NOT NULL UNIQUE,
    agent_id    TEXT NOT NULL,
    market      TEXT NOT NULL,
    side        TEXT NOT NULL,
    size        REAL NOT NULL,
    price       REAL NOT NULL,
    notional    REAL NOT NULL,
    timestamp   TEXT NOT NULL,
    status      TEXT NOT NULL DEFAULT 'pending',
    created_at  TEXT NOT NULL
)
"""

_CREATE_IDX_AGENT = "CREATE INDEX IF NOT EXISTS idx_agent_id ON trades(agent_id)"
_CREATE_IDX_MARKET = "CREATE INDEX IF NOT EXISTS idx_market ON trades(market)"
_CREATE_IDX_TS = "CREATE INDEX IF NOT EXISTS idx_timestamp ON trades(timestamp)"


# ─── Data Classes ─────────────────────────────────────────────────────────────


@dataclass
class LedgerEntry:
    """A single trade record persisted in the SQLite ledger."""
    id: Optional[int]
    tx_hash: str
    agent_id: str
    market: str
    side: str
    size: float
    price: float
    notional: float
    timestamp: str
    status: str
    created_at: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "tx_hash": self.tx_hash,
            "agent_id": self.agent_id,
            "market": self.market,
            "side": self.side,
            "size": self.size,
            "price": self.price,
            "notional": self.notional,
            "timestamp": self.timestamp,
            "status": self.status,
            "created_at": self.created_at,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict())

    @classmethod
    def from_row(cls, row: sqlite3.Row) -> "LedgerEntry":
        return cls(
            id=row["id"],
            tx_hash=row["tx_hash"],
            agent_id=row["agent_id"],
            market=row["market"],
            side=row["side"],
            size=row["size"],
            price=row["price"],
            notional=row["notional"],
            timestamp=row["timestamp"],
            status=row["status"],
            created_at=row["created_at"],
        )


@dataclass
class LedgerSummary:
    """Aggregate statistics across recorded trades."""
    total_trades: int
    total_notional: float
    buy_count: int
    sell_count: int
    unique_agents: int
    unique_markets: int
    confirmed_count: int
    pending_count: int
    failed_count: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_trades": self.total_trades,
            "total_notional": self.total_notional,
            "buy_count": self.buy_count,
            "sell_count": self.sell_count,
            "unique_agents": self.unique_agents,
            "unique_markets": self.unique_markets,
            "confirmed_count": self.confirmed_count,
            "pending_count": self.pending_count,
            "failed_count": self.failed_count,
        }


# ─── Exceptions ───────────────────────────────────────────────────────────────


class LedgerError(Exception):
    """Base exception for trade ledger errors."""


class LedgerValidationError(LedgerError):
    """Raised when a trade record fails validation."""


# ─── TradeLedger ──────────────────────────────────────────────────────────────


class TradeLedger:
    """
    SQLite-backed trade execution ledger.

    Every trade decision is written as a LedgerEntry with a deterministic
    tx_hash placeholder (SHA-256 of agent_id + market + side + size + price +
    timestamp) prefixed with "0x". This mirrors on-chain transaction hash
    semantics without requiring a live blockchain connection.

    Parameters
    ----------
    db_path : str
        Path to the SQLite database file. Use ":memory:" for in-memory
        (useful for tests and dry-runs). Default: ":memory:".
    """

    def __init__(self, db_path: str = ":memory:") -> None:
        self.db_path = db_path
        self._conn: sqlite3.Connection = sqlite3.connect(
            db_path,
            check_same_thread=False,
            detect_types=sqlite3.PARSE_DECLTYPES,
        )
        self._conn.row_factory = sqlite3.Row
        self._init_schema()

    # ── Lifecycle ──────────────────────────────────────────────────────────

    def _init_schema(self) -> None:
        """Create tables and indexes if they don't exist."""
        with self._conn:
            self._conn.execute(_CREATE_TABLE)
            self._conn.execute(_CREATE_IDX_AGENT)
            self._conn.execute(_CREATE_IDX_MARKET)
            self._conn.execute(_CREATE_IDX_TS)

    def close(self) -> None:
        """Close the database connection."""
        self._conn.close()

    def __enter__(self) -> "TradeLedger":
        return self

    def __exit__(self, *_) -> None:
        self.close()

    # ── Hash Generation ────────────────────────────────────────────────────

    @staticmethod
    def _generate_tx_hash(
        agent_id: str,
        market: str,
        side: str,
        size: float,
        price: float,
        timestamp: str,
    ) -> str:
        """
        Generate a deterministic tx_hash placeholder.

        Format: "0x" + first 64 chars of SHA-256(payload JSON).
        Mirrors Ethereum txhash format (0x-prefixed hex string).
        """
        payload = json.dumps(
            {
                "agent_id": agent_id,
                "market": market,
                "side": side,
                "size": size,
                "price": price,
                "timestamp": timestamp,
                "nonce": str(uuid.uuid4()),
            },
            sort_keys=True,
        ).encode()
        return "0x" + hashlib.sha256(payload).hexdigest()

    # ── Validation ─────────────────────────────────────────────────────────

    @staticmethod
    def _validate(
        agent_id: str,
        market: str,
        side: str,
        size: float,
        price: float,
    ) -> None:
        if not agent_id or not agent_id.strip():
            raise LedgerValidationError("agent_id must not be empty")
        if not market or not market.strip():
            raise LedgerValidationError("market must not be empty")
        if side not in VALID_SIDES:
            raise LedgerValidationError(
                f"side must be one of {sorted(VALID_SIDES)}, got {side!r}"
            )
        if not isinstance(size, (int, float)) or size <= 0:
            raise LedgerValidationError(f"size must be positive, got {size}")
        if not isinstance(price, (int, float)) or price <= 0:
            raise LedgerValidationError(f"price must be positive, got {price}")

    # ── Write ──────────────────────────────────────────────────────────────

    def log_trade(
        self,
        agent_id: str,
        market: str,
        side: str,
        size: float,
        price: float,
        timestamp: Optional[str] = None,
        tx_hash: Optional[str] = None,
        status: str = "pending",
    ) -> LedgerEntry:
        """
        Record a trade decision in the SQLite ledger.

        Parameters
        ----------
        agent_id  : Identifier of the trading agent.
        market    : Market symbol (e.g. "BTC/USD").
        side      : "BUY" or "SELL".
        size      : Trade size in base units (must be > 0).
        price     : Trade price in quote units (must be > 0).
        timestamp : ISO-8601 timestamp. Defaults to UTC now.
        tx_hash   : Override auto-generated hash (optional).
        status    : "pending" | "confirmed" | "failed". Default: "pending".

        Returns
        -------
        LedgerEntry with the auto-assigned row id and tx_hash.

        Raises
        ------
        LedgerValidationError : if any field is invalid.
        LedgerError           : on database write failure.
        """
        self._validate(agent_id, market, side, size, price)
        if status not in VALID_STATUSES:
            raise LedgerValidationError(
                f"status must be one of {sorted(VALID_STATUSES)}, got {status!r}"
            )

        now = datetime.now(timezone.utc).isoformat()
        ts = timestamp or now
        hash_ = tx_hash or self._generate_tx_hash(agent_id, market, side, size, price, ts)
        notional = round(size * price, 8)

        try:
            with self._conn:
                cur = self._conn.execute(
                    """
                    INSERT INTO trades
                        (tx_hash, agent_id, market, side, size, price,
                         notional, timestamp, status, created_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (hash_, agent_id, market, side, size, price,
                     notional, ts, status, now),
                )
                row_id = cur.lastrowid
        except sqlite3.IntegrityError as e:
            raise LedgerError(f"DB integrity error: {e}") from e
        except sqlite3.OperationalError as e:
            raise LedgerError(f"DB operational error: {e}") from e

        return LedgerEntry(
            id=row_id,
            tx_hash=hash_,
            agent_id=agent_id,
            market=market,
            side=side,
            size=size,
            price=price,
            notional=notional,
            timestamp=ts,
            status=status,
            created_at=now,
        )

    def update_status(self, tx_hash: str, new_status: str) -> bool:
        """
        Update the status of a trade by tx_hash.

        Returns True if a row was updated, False if tx_hash not found.
        """
        if new_status not in VALID_STATUSES:
            raise LedgerValidationError(
                f"status must be one of {sorted(VALID_STATUSES)}, got {new_status!r}"
            )
        with self._conn:
            cur = self._conn.execute(
                "UPDATE trades SET status = ? WHERE tx_hash = ?",
                (new_status, tx_hash),
            )
        return cur.rowcount > 0

    # ── Read ───────────────────────────────────────────────────────────────

    def get_entry(self, tx_hash: str) -> Optional[LedgerEntry]:
        """Fetch a single entry by tx_hash. Returns None if not found."""
        cur = self._conn.execute(
            "SELECT * FROM trades WHERE tx_hash = ?", (tx_hash,)
        )
        row = cur.fetchone()
        return LedgerEntry.from_row(row) if row else None

    def get_entries(
        self,
        agent_id: Optional[str] = None,
        market: Optional[str] = None,
        side: Optional[str] = None,
        status: Optional[str] = None,
        limit: Optional[int] = None,
        offset: int = 0,
    ) -> List[LedgerEntry]:
        """
        Query ledger entries with optional filters.

        Parameters are ANDed. Returns list ordered by id ASC.
        """
        clauses: List[str] = []
        params: List[Any] = []

        if agent_id is not None:
            clauses.append("agent_id = ?")
            params.append(agent_id)
        if market is not None:
            clauses.append("market = ?")
            params.append(market)
        if side is not None:
            clauses.append("side = ?")
            params.append(side)
        if status is not None:
            clauses.append("status = ?")
            params.append(status)

        where = ("WHERE " + " AND ".join(clauses)) if clauses else ""
        if limit is not None:
            limit_clause = f"LIMIT {int(limit)} OFFSET {int(offset)}"
        elif offset:
            # OFFSET without LIMIT: use a large sentinel
            limit_clause = f"LIMIT -1 OFFSET {int(offset)}"
        else:
            limit_clause = ""
        sql = f"SELECT * FROM trades {where} ORDER BY id ASC {limit_clause}"

        cur = self._conn.execute(sql, params)
        return [LedgerEntry.from_row(r) for r in cur.fetchall()]

    def get_all(self) -> List[LedgerEntry]:
        """Return all entries ordered by insertion order."""
        return self.get_entries()

    def get_summary(
        self,
        agent_id: Optional[str] = None,
        market: Optional[str] = None,
    ) -> LedgerSummary:
        """
        Return aggregate statistics for matching entries.

        Optionally filter by agent_id and/or market.
        """
        clauses: List[str] = []
        params: List[Any] = []
        if agent_id is not None:
            clauses.append("agent_id = ?")
            params.append(agent_id)
        if market is not None:
            clauses.append("market = ?")
            params.append(market)
        where = ("WHERE " + " AND ".join(clauses)) if clauses else ""

        row = self._conn.execute(
            f"""
            SELECT
                COUNT(*)                                              AS total_trades,
                COALESCE(SUM(notional), 0.0)                         AS total_notional,
                COALESCE(SUM(CASE WHEN side='BUY'  THEN 1 ELSE 0 END), 0) AS buy_count,
                COALESCE(SUM(CASE WHEN side='SELL' THEN 1 ELSE 0 END), 0) AS sell_count,
                COUNT(DISTINCT agent_id)                              AS unique_agents,
                COUNT(DISTINCT market)                                AS unique_markets,
                COALESCE(SUM(CASE WHEN status='confirmed' THEN 1 ELSE 0 END), 0) AS confirmed_count,
                COALESCE(SUM(CASE WHEN status='pending'   THEN 1 ELSE 0 END), 0) AS pending_count,
                COALESCE(SUM(CASE WHEN status='failed'    THEN 1 ELSE 0 END), 0) AS failed_count
            FROM trades {where}
            """,
            params,
        ).fetchone()

        return LedgerSummary(
            total_trades=row["total_trades"],
            total_notional=float(row["total_notional"]),
            buy_count=int(row["buy_count"]),
            sell_count=int(row["sell_count"]),
            unique_agents=row["unique_agents"],
            unique_markets=row["unique_markets"],
            confirmed_count=int(row["confirmed_count"]),
            pending_count=int(row["pending_count"]),
            failed_count=int(row["failed_count"]),
        )

    def count(self) -> int:
        """Return total number of entries in the ledger."""
        return self._conn.execute("SELECT COUNT(*) FROM trades").fetchone()[0]

    # ── Utilities ──────────────────────────────────────────────────────────

    def clear(self) -> int:
        """Delete all entries. Returns number of rows deleted."""
        with self._conn:
            cur = self._conn.execute("DELETE FROM trades")
        return cur.rowcount

    def format_trace(self, entries: Optional[List[LedgerEntry]] = None) -> str:
        """
        Format a list of ledger entries as a human-readable trace.

        Used by demo dry-run to print execution history.
        """
        if entries is None:
            entries = self.get_all()
        if not entries:
            return "(no trades recorded)"

        lines = [
            f"{'#':<4} {'AGENT':<28} {'MARKET':<10} {'SIDE':<5} "
            f"{'SIZE':>10} {'PRICE':>12} {'NOTIONAL':>12} {'STATUS':<10} TX_HASH",
            "-" * 110,
        ]
        for e in entries:
            lines.append(
                f"{e.id!s:<4} {e.agent_id:<28} {e.market:<10} {e.side:<5} "
                f"{e.size:>10.4f} {e.price:>12.2f} {e.notional:>12.2f} "
                f"{e.status:<10} {e.tx_hash[:18]}..."
            )
        summary = self.get_summary()
        lines += [
            "-" * 110,
            f"Total: {summary.total_trades} trades  |  "
            f"Notional: ${summary.total_notional:,.2f}  |  "
            f"BUY: {summary.buy_count}  SELL: {summary.sell_count}  |  "
            f"Pending: {summary.pending_count}  Confirmed: {summary.confirmed_count}",
        ]
        return "\n".join(lines)
