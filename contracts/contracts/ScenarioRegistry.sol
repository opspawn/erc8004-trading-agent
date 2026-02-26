// SPDX-License-Identifier: MIT
pragma solidity ^0.8.24;

/**
 * @title ScenarioRegistry
 * @notice ERC-8004 on-chain registry for demo scenario run outcomes.
 *         Records scenario simulations (bull_run, bear_crash, volatile_chop,
 *         stable_trend) as an immutable audit trail for judges.
 *
 * @dev Sprint 32: Demo Scenario Orchestrator
 *      - Scenarios are committed with their tick timeline hash
 *      - Circuit breaker activations are recorded per scenario
 *      - Leaderboard snapshots capture agent rankings after each scenario
 */
contract ScenarioRegistry {
    // -------------------------------------------------------------------------
    // Types
    // -------------------------------------------------------------------------

    enum ScenarioType { UNKNOWN, BULL_RUN, BEAR_CRASH, VOLATILE_CHOP, STABLE_TREND }

    struct ScenarioRun {
        bytes32 runId;
        ScenarioType scenarioType;
        uint256 tickCount;
        bytes32 timelineHash;   // keccak256 of the full tick timeline JSON
        int256 totalPnlCents;   // total PnL in USD cents (can be negative)
        uint32 consensusRate;   // 0-10000 basis points
        bool circuitBreakerFired;
        uint256 startedAt;
        uint256 completedAt;
    }

    struct LeaderboardSnapshot {
        bytes32 runId;
        uint256 rank;
        uint256 agentId;
        uint256 reputationScore;    // scaled by 1e4
        int256 pnlCents;
        uint32 winRate;             // 0-10000 basis points
        uint256 totalTrades;
        uint256 snapshotAt;
    }

    // -------------------------------------------------------------------------
    // State
    // -------------------------------------------------------------------------

    ScenarioRun[] private _runs;
    LeaderboardSnapshot[] private _snapshots;

    // runId → index + 1 in _runs array (0 = not found)
    mapping(bytes32 => uint256) private _runIndex;

    // Total scenarios by type
    mapping(ScenarioType => uint256) public scenarioTypeCount;

    // -------------------------------------------------------------------------
    // Events
    // -------------------------------------------------------------------------

    event ScenarioCommitted(
        bytes32 indexed runId,
        ScenarioType scenarioType,
        uint256 tickCount,
        int256 totalPnlCents,
        bool circuitBreakerFired
    );

    event LeaderboardSnapshotCommitted(
        bytes32 indexed runId,
        uint256 rank,
        uint256 indexed agentId,
        uint256 reputationScore
    );

    // -------------------------------------------------------------------------
    // Errors
    // -------------------------------------------------------------------------

    error DuplicateRunId(bytes32 runId);
    error RunNotFound(bytes32 runId);
    error InvalidTickCount(uint256 tickCount);
    error InvalidConsensusRate(uint32 rate);
    error InvalidRank(uint256 rank);
    error InvalidWinRate(uint32 winRate);

    // -------------------------------------------------------------------------
    // Scenario Submission
    // -------------------------------------------------------------------------

    /**
     * @notice Commit a completed scenario run to the registry.
     * @param runId              Unique run identifier (keccak256 of scenario + timestamp)
     * @param scenarioType       Type of scenario (BULL_RUN, BEAR_CRASH, etc.)
     * @param tickCount          Number of ticks simulated
     * @param timelineHash       keccak256 of the full tick timeline JSON
     * @param totalPnlCents      Total PnL in USD cents (signed)
     * @param consensusRate      Consensus agreement rate in basis points
     * @param circuitBreakerFired Whether the circuit breaker was triggered
     */
    function commitScenario(
        bytes32 runId,
        ScenarioType scenarioType,
        uint256 tickCount,
        bytes32 timelineHash,
        int256 totalPnlCents,
        uint32 consensusRate,
        bool circuitBreakerFired
    ) external {
        if (_runIndex[runId] != 0) {
            revert DuplicateRunId(runId);
        }
        if (tickCount == 0) {
            revert InvalidTickCount(tickCount);
        }
        if (consensusRate > 10000) {
            revert InvalidConsensusRate(consensusRate);
        }

        _runs.push(ScenarioRun({
            runId: runId,
            scenarioType: scenarioType,
            tickCount: tickCount,
            timelineHash: timelineHash,
            totalPnlCents: totalPnlCents,
            consensusRate: consensusRate,
            circuitBreakerFired: circuitBreakerFired,
            startedAt: block.timestamp,
            completedAt: block.timestamp
        }));

        _runIndex[runId] = _runs.length;  // store 1-based index
        scenarioTypeCount[scenarioType]++;

        emit ScenarioCommitted(runId, scenarioType, tickCount, totalPnlCents, circuitBreakerFired);
    }

    // -------------------------------------------------------------------------
    // Leaderboard Snapshots
    // -------------------------------------------------------------------------

    /**
     * @notice Commit an agent leaderboard snapshot for a scenario run.
     * @param runId             Run identifier (must exist)
     * @param rank              Agent rank (1-based)
     * @param agentId           ERC-8004 agent token ID
     * @param reputationScore   Score scaled by 1e4
     * @param pnlCents          Agent PnL in USD cents (signed)
     * @param winRate           Win rate in basis points
     * @param totalTrades       Total trades executed
     */
    function commitLeaderboardEntry(
        bytes32 runId,
        uint256 rank,
        uint256 agentId,
        uint256 reputationScore,
        int256 pnlCents,
        uint32 winRate,
        uint256 totalTrades
    ) external {
        if (_runIndex[runId] == 0) {
            revert RunNotFound(runId);
        }
        if (rank == 0) {
            revert InvalidRank(rank);
        }
        if (winRate > 10000) {
            revert InvalidWinRate(winRate);
        }

        _snapshots.push(LeaderboardSnapshot({
            runId: runId,
            rank: rank,
            agentId: agentId,
            reputationScore: reputationScore,
            pnlCents: pnlCents,
            winRate: winRate,
            totalTrades: totalTrades,
            snapshotAt: block.timestamp
        }));

        emit LeaderboardSnapshotCommitted(runId, rank, agentId, reputationScore);
    }

    // -------------------------------------------------------------------------
    // Getters
    // -------------------------------------------------------------------------

    function runCount() external view returns (uint256) {
        return _runs.length;
    }

    function getRunByIndex(uint256 index) external view returns (ScenarioRun memory) {
        return _runs[index];
    }

    function getRunById(bytes32 runId) external view returns (ScenarioRun memory) {
        uint256 idx = _runIndex[runId];
        if (idx == 0) revert RunNotFound(runId);
        return _runs[idx - 1];
    }

    function runExists(bytes32 runId) external view returns (bool) {
        return _runIndex[runId] != 0;
    }

    function snapshotCount() external view returns (uint256) {
        return _snapshots.length;
    }

    function getSnapshot(uint256 index) external view returns (LeaderboardSnapshot memory) {
        return _snapshots[index];
    }

    /**
     * @notice Get all leaderboard snapshots for a specific run.
     */
    function getSnapshotsForRun(bytes32 runId) external view returns (LeaderboardSnapshot[] memory) {
        uint256 count = 0;
        for (uint256 i = 0; i < _snapshots.length; i++) {
            if (_snapshots[i].runId == runId) count++;
        }
        LeaderboardSnapshot[] memory result = new LeaderboardSnapshot[](count);
        uint256 j = 0;
        for (uint256 i = 0; i < _snapshots.length; i++) {
            if (_snapshots[i].runId == runId) {
                result[j++] = _snapshots[i];
            }
        }
        return result;
    }

    /**
     * @notice Get scenarios filtered by type.
     */
    function getRunsByType(ScenarioType scenarioType) external view returns (ScenarioRun[] memory) {
        uint256 count = scenarioTypeCount[scenarioType];
        ScenarioRun[] memory result = new ScenarioRun[](count);
        uint256 j = 0;
        for (uint256 i = 0; i < _runs.length; i++) {
            if (_runs[i].scenarioType == scenarioType) {
                result[j++] = _runs[i];
            }
        }
        return result;
    }

    /**
     * @notice Count how many scenarios had circuit breaker fired.
     */
    function circuitBreakerFiredCount() external view returns (uint256) {
        uint256 count = 0;
        for (uint256 i = 0; i < _runs.length; i++) {
            if (_runs[i].circuitBreakerFired) count++;
        }
        return count;
    }
}
