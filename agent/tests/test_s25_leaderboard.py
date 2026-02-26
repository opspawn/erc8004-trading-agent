"""
S25 Leaderboard Tests — sort params, edge cases, endpoint polish.

Tests the enhanced /demo/leaderboard endpoint that supports:
  - ?sort_by=sortino|sharpe|pnl|trades|win_rate|reputation
  - ?limit=N  (1–20)
  - Invalid param rejection with helpful error
  - 5-agent seeded leaderboard
  - Correct rank assignment after re-sort
"""
from __future__ import annotations

import io
import json
import sys
import threading
import unittest
from http.server import HTTPServer
from unittest.mock import patch

# ── Path setup ──────────────────────────────────────────────────────────────
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from demo_server import (
    LEADERBOARD_SORT_DEFAULT,
    LEADERBOARD_SORT_KEYS,
    _SEEDED_LEADERBOARD,
    _agent_cumulative,
    _leaderboard_lock,
    build_leaderboard,
)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _clear_leaderboard() -> None:
    """Reset live leaderboard state so tests get seeded data."""
    with _leaderboard_lock:
        _agent_cumulative.clear()


def _inject_agents(agents: list[dict]) -> None:
    """Inject synthetic cumulative agent data for live-data tests."""
    with _leaderboard_lock:
        for a in agents:
            _agent_cumulative[a["agent_id"]] = {
                "agent_id": a["agent_id"],
                "strategy": a.get("strategy", "balanced"),
                "total_trades": a.get("total_trades", 10),
                "total_wins": a.get("total_wins", 5),
                "total_pnl": a.get("total_pnl", 100.0),
                "pnl_history": a.get("pnl_history", [1.0, -0.5, 0.8]),
                "reputation_score": a.get("reputation_score", 7.0),
            }


# ─────────────────────────────────────────────────────────────────────────────
# 1. Seeded leaderboard shape & content
# ─────────────────────────────────────────────────────────────────────────────

class TestSeededLeaderboard(unittest.TestCase):
    """Verify the seeded leaderboard has correct shape before any live runs."""

    def setUp(self):
        _clear_leaderboard()

    def test_seeded_returns_five_agents(self):
        result = build_leaderboard()
        self.assertEqual(len(result), 5)

    def test_seeded_default_sort_sortino(self):
        result = build_leaderboard()
        sortinos = [e["sortino_ratio"] for e in result]
        self.assertEqual(sortinos, sorted(sortinos, reverse=True))

    def test_seeded_ranks_are_1_to_5(self):
        result = build_leaderboard()
        ranks = [e["rank"] for e in result]
        self.assertEqual(ranks, [1, 2, 3, 4, 5])

    def test_seeded_sort_by_pnl(self):
        result = build_leaderboard(sort_by="pnl")
        pnls = [e["total_return_pct"] for e in result]
        self.assertEqual(pnls, sorted(pnls, reverse=True))

    def test_seeded_sort_by_sharpe(self):
        result = build_leaderboard(sort_by="sharpe")
        sharpes = [e["sharpe_ratio"] for e in result]
        self.assertEqual(sharpes, sorted(sharpes, reverse=True))

    def test_seeded_sort_by_trades(self):
        result = build_leaderboard(sort_by="trades")
        trades = [e["trades_count"] for e in result]
        self.assertEqual(trades, sorted(trades, reverse=True))

    def test_seeded_sort_by_win_rate(self):
        result = build_leaderboard(sort_by="win_rate")
        rates = [e["win_rate"] for e in result]
        self.assertEqual(rates, sorted(rates, reverse=True))

    def test_seeded_sort_by_reputation(self):
        result = build_leaderboard(sort_by="reputation")
        reps = [e["reputation_score"] for e in result]
        self.assertEqual(reps, sorted(reps, reverse=True))

    def test_seeded_sort_by_annotation_present(self):
        result = build_leaderboard(sort_by="pnl")
        for entry in result:
            self.assertEqual(entry["sort_by"], "pnl")

    def test_seeded_required_fields(self):
        required = {
            "rank", "agent_id", "strategy", "total_return_pct",
            "sortino_ratio", "sharpe_ratio", "win_rate",
            "trades_count", "reputation_score",
        }
        result = build_leaderboard()
        for entry in result:
            self.assertTrue(required.issubset(entry.keys()), f"Missing fields in {entry}")

    def test_seeded_has_five_distinct_agent_ids(self):
        result = build_leaderboard()
        agent_ids = {e["agent_id"] for e in result}
        self.assertEqual(len(agent_ids), 5)

    def test_seeded_limit_two(self):
        result = build_leaderboard(limit=2)
        self.assertEqual(len(result), 2)

    def test_seeded_limit_one(self):
        result = build_leaderboard(limit=1)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["rank"], 1)

    def test_seeded_limit_capped_at_20(self):
        result = build_leaderboard(limit=100)
        # only 5 seeded agents so can't exceed 5
        self.assertLessEqual(len(result), 20)


# ─────────────────────────────────────────────────────────────────────────────
# 2. Live leaderboard (injected data)
# ─────────────────────────────────────────────────────────────────────────────

_LIVE_AGENTS = [
    {"agent_id": "live-A", "strategy": "aggressive", "total_trades": 20, "total_wins": 14,
     "total_pnl": 800.0, "pnl_history": [5.0, 3.0, -1.0, 4.0], "reputation_score": 8.5},
    {"agent_id": "live-B", "strategy": "conservative", "total_trades": 30, "total_wins": 22,
     "total_pnl": 350.0, "pnl_history": [2.0, 1.5, 0.5, 2.5], "reputation_score": 9.1},
    {"agent_id": "live-C", "strategy": "balanced", "total_trades": 15, "total_wins": 9,
     "total_pnl": 550.0, "pnl_history": [3.0, -2.0, 5.0, 1.0], "reputation_score": 7.8},
    {"agent_id": "live-D", "strategy": "momentum", "total_trades": 50, "total_wins": 25,
     "total_pnl": 200.0, "pnl_history": [1.0, 1.0, 1.0, -0.5], "reputation_score": 6.2},
    {"agent_id": "live-E", "strategy": "mean_reversion", "total_trades": 8, "total_wins": 6,
     "total_pnl": 650.0, "pnl_history": [4.0, 3.5, -0.5, 4.5], "reputation_score": 7.0},
    {"agent_id": "live-F", "strategy": "arbitrage", "total_trades": 60, "total_wins": 40,
     "total_pnl": 900.0, "pnl_history": [6.0, 5.0, 4.0, 5.5], "reputation_score": 9.8},
]


class TestLiveLeaderboard(unittest.TestCase):
    """Tests using injected live agent data."""

    def setUp(self):
        _clear_leaderboard()
        _inject_agents(_LIVE_AGENTS)

    def tearDown(self):
        _clear_leaderboard()

    def test_live_default_returns_top_5(self):
        result = build_leaderboard()
        self.assertEqual(len(result), 5)

    def test_live_sort_by_pnl_descending(self):
        result = build_leaderboard(sort_by="pnl", limit=6)
        pnls = [e["total_return_pct"] for e in result]
        self.assertEqual(pnls, sorted(pnls, reverse=True))

    def test_live_sort_by_trades_descending(self):
        result = build_leaderboard(sort_by="trades", limit=6)
        trades = [e["trades_count"] for e in result]
        self.assertEqual(trades, sorted(trades, reverse=True))

    def test_live_sort_by_reputation_descending(self):
        result = build_leaderboard(sort_by="reputation", limit=6)
        reps = [e["reputation_score"] for e in result]
        self.assertEqual(reps, sorted(reps, reverse=True))

    def test_live_sort_by_sharpe_descending(self):
        result = build_leaderboard(sort_by="sharpe", limit=6)
        sharpes = [e["sharpe_ratio"] for e in result]
        self.assertEqual(sharpes, sorted(sharpes, reverse=True))

    def test_live_limit_3(self):
        result = build_leaderboard(limit=3)
        self.assertEqual(len(result), 3)

    def test_live_ranks_sequential(self):
        result = build_leaderboard(limit=5)
        ranks = [e["rank"] for e in result]
        self.assertEqual(ranks, list(range(1, len(result) + 1)))

    def test_live_sort_by_annotation(self):
        for key in LEADERBOARD_SORT_KEYS:
            result = build_leaderboard(sort_by=key, limit=1)
            self.assertEqual(result[0]["sort_by"], key)


# ─────────────────────────────────────────────────────────────────────────────
# 3. LEADERBOARD_SORT_KEYS contract
# ─────────────────────────────────────────────────────────────────────────────

class TestSortKeysContract(unittest.TestCase):

    def test_required_sort_keys_present(self):
        for key in ("sortino", "sharpe", "pnl", "trades", "win_rate", "reputation"):
            self.assertIn(key, LEADERBOARD_SORT_KEYS)

    def test_default_sort_is_sortino(self):
        self.assertEqual(LEADERBOARD_SORT_DEFAULT, "sortino")

    def test_sort_keys_map_to_valid_fields(self):
        valid_fields = {
            "sortino_ratio", "sharpe_ratio", "total_return_pct",
            "trades_count", "win_rate", "reputation_score",
        }
        for k, v in LEADERBOARD_SORT_KEYS.items():
            self.assertIn(v, valid_fields, f"Key '{k}' maps to unknown field '{v}'")

    def test_unknown_sort_key_falls_back_to_default(self):
        """build_leaderboard with unknown sort_by falls back to default sort."""
        _clear_leaderboard()
        # Default seeded data — unknown key should use default (sortino)
        result_default = build_leaderboard(sort_by="sortino")
        result_unknown = build_leaderboard(sort_by="nonexistent_key")
        # Both should be sorted by sortino
        sortinos_default = [e["sortino_ratio"] for e in result_default]
        sortinos_unknown = [e["sortino_ratio"] for e in result_unknown]
        self.assertEqual(sortinos_default, sortinos_unknown)


# ─────────────────────────────────────────────────────────────────────────────
# 4. Edge cases
# ─────────────────────────────────────────────────────────────────────────────

class TestLeaderboardEdgeCases(unittest.TestCase):

    def setUp(self):
        _clear_leaderboard()

    def tearDown(self):
        _clear_leaderboard()

    def test_single_agent_returns_rank_1(self):
        _inject_agents([_LIVE_AGENTS[0]])
        result = build_leaderboard()
        self.assertEqual(result[0]["rank"], 1)

    def test_agent_with_zero_trades_win_rate_zero(self):
        _inject_agents([{
            "agent_id": "zero-trade-agent",
            "strategy": "balanced",
            "total_trades": 0,
            "total_wins": 0,
            "total_pnl": 0.0,
            "pnl_history": [],
            "reputation_score": 5.0,
        }])
        result = build_leaderboard()
        self.assertEqual(result[0]["win_rate"], 0.0)

    def test_agent_with_negative_pnl(self):
        _inject_agents([{
            "agent_id": "losing-agent",
            "strategy": "aggressive",
            "total_trades": 10,
            "total_wins": 3,
            "total_pnl": -500.0,
            "pnl_history": [-2.0, -1.5, -3.0],
            "reputation_score": 4.0,
        }])
        result = build_leaderboard()
        self.assertLess(result[0]["total_return_pct"], 0)

    def test_limit_zero_clamped_to_one(self):
        result = build_leaderboard(limit=0)
        self.assertGreaterEqual(len(result), 1)

    def test_limit_negative_clamped_to_one(self):
        result = build_leaderboard(limit=-5)
        self.assertGreaterEqual(len(result), 1)

    def test_seeded_conservative_agent_present(self):
        _clear_leaderboard()
        result = build_leaderboard()
        agent_ids = [e["agent_id"] for e in result]
        self.assertIn("agent-conservative-001", agent_ids)

    def test_seeded_momentum_agent_present(self):
        _clear_leaderboard()
        result = build_leaderboard()
        agent_ids = [e["agent_id"] for e in result]
        self.assertIn("agent-momentum-004", agent_ids)

    def test_seeded_meanrev_agent_present(self):
        _clear_leaderboard()
        result = build_leaderboard()
        agent_ids = [e["agent_id"] for e in result]
        self.assertIn("agent-meanrev-005", agent_ids)


if __name__ == "__main__":
    unittest.main()
