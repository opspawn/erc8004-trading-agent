// SPDX-License-Identifier: MIT
pragma solidity ^0.8.24;

import "./IdentityRegistry.sol";

/**
 * @title TradingFeed
 * @notice ERC-8004 on-chain trading event feed.
 *         Records agent votes, consensus decisions, trade executions, and
 *         reputation updates as an immutable on-chain audit trail.
 *
 * @dev Sprint 32: Live WebSocket Feed + Demo Dashboard
 *      - Agents emit votes via castVote()
 *      - Coordinators record consensus via recordConsensus()
 *      - Trades are logged via recordTrade()
 *      - Reputation changes are logged via recordReputationUpdate()
 *      All events are queryable via paginated getter functions.
 */
contract TradingFeed {
    // -------------------------------------------------------------------------
    // Types
    // -------------------------------------------------------------------------

    enum TradeAction { NONE, BUY, SELL, HOLD }

    struct AgentVote {
        uint256 agentId;
        string symbol;
        TradeAction action;
        uint32 confidence;      // 0-10000 (basis points, i.e. 10000 = 100%)
        uint256 positionSize;   // in wei-equivalent units
        uint256 timestamp;
    }

    struct ConsensusRecord {
        string symbol;
        TradeAction decision;
        uint32 agreementRate;   // 0-10000 basis points
        uint256 participantCount;
        uint256 timestamp;
    }

    struct TradeRecord {
        uint256 agentId;
        string symbol;
        TradeAction action;
        uint256 quantity;       // token quantity (18 decimals)
        uint256 price;          // price in USD cents
        int256 pnlDelta;        // PnL change in USD cents (can be negative)
        uint256 timestamp;
    }

    struct ReputationUpdate {
        uint256 agentId;
        uint256 oldScore;       // scaled by 1e4 (e.g. 7820 = 7.82)
        uint256 newScore;       // scaled by 1e4
        uint256 timestamp;
    }

    // -------------------------------------------------------------------------
    // State
    // -------------------------------------------------------------------------

    IdentityRegistry public immutable identityRegistry;

    AgentVote[] private _votes;
    ConsensusRecord[] private _consensus;
    TradeRecord[] private _trades;
    ReputationUpdate[] private _reputationUpdates;

    // Circuit breaker
    bool public circuitBreakerTripped;
    uint256 public circuitBreakerTrippedAt;
    int256 public drawdownThreshold;    // negative value in USD cents

    // Feed sequence number (monotonically increasing across all event types)
    uint256 public feedSequence;

    // -------------------------------------------------------------------------
    // Events
    // -------------------------------------------------------------------------

    event AgentVoteCast(
        uint256 indexed agentId,
        string symbol,
        TradeAction action,
        uint32 confidence,
        uint256 positionSize,
        uint256 feedSeq
    );

    event ConsensusReached(
        string symbol,
        TradeAction decision,
        uint32 agreementRate,
        uint256 participantCount,
        uint256 feedSeq
    );

    event TradeExecuted(
        uint256 indexed agentId,
        string symbol,
        TradeAction action,
        uint256 quantity,
        uint256 price,
        int256 pnlDelta,
        uint256 feedSeq
    );

    event ReputationUpdated(
        uint256 indexed agentId,
        uint256 oldScore,
        uint256 newScore,
        uint256 feedSeq
    );

    event CircuitBreakerTripped(
        int256 drawdownAmount,
        uint256 triggeredAt
    );

    event CircuitBreakerReset(
        uint256 resetAt
    );

    // -------------------------------------------------------------------------
    // Errors
    // -------------------------------------------------------------------------

    error AgentNotRegistered(uint256 agentId);
    error InvalidConfidence(uint32 confidence);
    error InvalidAgreementRate(uint32 rate);
    error ZeroQuantity();
    error EmptySymbol();

    // -------------------------------------------------------------------------
    // Constructor
    // -------------------------------------------------------------------------

    constructor(address identityRegistryAddress, int256 initialDrawdownThreshold) {
        identityRegistry = IdentityRegistry(identityRegistryAddress);
        drawdownThreshold = initialDrawdownThreshold;
    }

    // -------------------------------------------------------------------------
    // Internal: Agent Validation
    // -------------------------------------------------------------------------

    function _requireRegisteredAgent(uint256 agentId) internal view {
        try identityRegistry.ownerOf(agentId) returns (address) {
            // valid
        } catch {
            revert AgentNotRegistered(agentId);
        }
    }

    // -------------------------------------------------------------------------
    // Agent Voting
    // -------------------------------------------------------------------------

    /**
     * @notice Record an agent's vote for a trading action.
     * @param agentId       ERC-8004 agent token ID
     * @param symbol        Trading pair symbol (e.g. "BTC/USD")
     * @param action        BUY, SELL, or HOLD
     * @param confidence    Confidence in basis points (0-10000)
     * @param positionSize  Desired position size in wei-equivalent units
     */
    function castVote(
        uint256 agentId,
        string calldata symbol,
        TradeAction action,
        uint32 confidence,
        uint256 positionSize
    ) external {
        _requireRegisteredAgent(agentId);
        if (confidence > 10000) {
            revert InvalidConfidence(confidence);
        }
        if (bytes(symbol).length == 0) {
            revert EmptySymbol();
        }

        uint256 seq = ++feedSequence;
        _votes.push(AgentVote({
            agentId: agentId,
            symbol: symbol,
            action: action,
            confidence: confidence,
            positionSize: positionSize,
            timestamp: block.timestamp
        }));

        emit AgentVoteCast(agentId, symbol, action, confidence, positionSize, seq);
    }

    // -------------------------------------------------------------------------
    // Consensus Recording
    // -------------------------------------------------------------------------

    /**
     * @notice Record a consensus decision reached by the agent pool.
     * @param symbol           Trading pair symbol
     * @param decision         The consensus action
     * @param agreementRate    Agreement rate in basis points (0-10000)
     * @param participantCount Number of agents that participated
     */
    function recordConsensus(
        string calldata symbol,
        TradeAction decision,
        uint32 agreementRate,
        uint256 participantCount
    ) external {
        if (agreementRate > 10000) {
            revert InvalidAgreementRate(agreementRate);
        }
        if (bytes(symbol).length == 0) {
            revert EmptySymbol();
        }

        uint256 seq = ++feedSequence;
        _consensus.push(ConsensusRecord({
            symbol: symbol,
            decision: decision,
            agreementRate: agreementRate,
            participantCount: participantCount,
            timestamp: block.timestamp
        }));

        emit ConsensusReached(symbol, decision, agreementRate, participantCount, seq);
    }

    // -------------------------------------------------------------------------
    // Trade Recording
    // -------------------------------------------------------------------------

    /**
     * @notice Record a trade execution.
     * @param agentId   ERC-8004 agent token ID
     * @param symbol    Trading pair symbol
     * @param action    BUY or SELL
     * @param quantity  Amount traded (18 decimal units)
     * @param price     Execution price in USD cents
     * @param pnlDelta  PnL change in USD cents (negative = loss)
     */
    function recordTrade(
        uint256 agentId,
        string calldata symbol,
        TradeAction action,
        uint256 quantity,
        uint256 price,
        int256 pnlDelta
    ) external {
        _requireRegisteredAgent(agentId);
        if (quantity == 0) {
            revert ZeroQuantity();
        }
        if (bytes(symbol).length == 0) {
            revert EmptySymbol();
        }

        uint256 seq = ++feedSequence;
        _trades.push(TradeRecord({
            agentId: agentId,
            symbol: symbol,
            action: action,
            quantity: quantity,
            price: price,
            pnlDelta: pnlDelta,
            timestamp: block.timestamp
        }));

        emit TradeExecuted(agentId, symbol, action, quantity, price, pnlDelta, seq);

        // Check circuit breaker
        if (pnlDelta < drawdownThreshold) {
            _tripCircuitBreaker(pnlDelta);
        }
    }

    // -------------------------------------------------------------------------
    // Reputation Updates
    // -------------------------------------------------------------------------

    /**
     * @notice Record a reputation score change for an agent.
     * @param agentId   ERC-8004 agent token ID
     * @param oldScore  Previous score (scaled by 1e4)
     * @param newScore  New score (scaled by 1e4)
     */
    function recordReputationUpdate(
        uint256 agentId,
        uint256 oldScore,
        uint256 newScore
    ) external {
        _requireRegisteredAgent(agentId);

        uint256 seq = ++feedSequence;
        _reputationUpdates.push(ReputationUpdate({
            agentId: agentId,
            oldScore: oldScore,
            newScore: newScore,
            timestamp: block.timestamp
        }));

        emit ReputationUpdated(agentId, oldScore, newScore, seq);
    }

    // -------------------------------------------------------------------------
    // Circuit Breaker
    // -------------------------------------------------------------------------

    function _tripCircuitBreaker(int256 drawdownAmount) internal {
        if (!circuitBreakerTripped) {
            circuitBreakerTripped = true;
            circuitBreakerTrippedAt = block.timestamp;
            emit CircuitBreakerTripped(drawdownAmount, block.timestamp);
        }
    }

    function resetCircuitBreaker() external {
        circuitBreakerTripped = false;
        circuitBreakerTrippedAt = 0;
        emit CircuitBreakerReset(block.timestamp);
    }

    function updateDrawdownThreshold(int256 newThreshold) external {
        drawdownThreshold = newThreshold;
    }

    // -------------------------------------------------------------------------
    // Getters — Votes
    // -------------------------------------------------------------------------

    function voteCount() external view returns (uint256) {
        return _votes.length;
    }

    function getVote(uint256 index) external view returns (AgentVote memory) {
        return _votes[index];
    }

    function getVotes(uint256 offset, uint256 limit) external view returns (AgentVote[] memory) {
        uint256 total = _votes.length;
        if (offset >= total) return new AgentVote[](0);
        uint256 end = offset + limit;
        if (end > total) end = total;
        AgentVote[] memory page = new AgentVote[](end - offset);
        for (uint256 i = offset; i < end; i++) {
            page[i - offset] = _votes[i];
        }
        return page;
    }

    // -------------------------------------------------------------------------
    // Getters — Consensus
    // -------------------------------------------------------------------------

    function consensusCount() external view returns (uint256) {
        return _consensus.length;
    }

    function getConsensus(uint256 index) external view returns (ConsensusRecord memory) {
        return _consensus[index];
    }

    // -------------------------------------------------------------------------
    // Getters — Trades
    // -------------------------------------------------------------------------

    function tradeCount() external view returns (uint256) {
        return _trades.length;
    }

    function getTrade(uint256 index) external view returns (TradeRecord memory) {
        return _trades[index];
    }

    function getTrades(uint256 offset, uint256 limit) external view returns (TradeRecord[] memory) {
        uint256 total = _trades.length;
        if (offset >= total) return new TradeRecord[](0);
        uint256 end = offset + limit;
        if (end > total) end = total;
        TradeRecord[] memory page = new TradeRecord[](end - offset);
        for (uint256 i = offset; i < end; i++) {
            page[i - offset] = _trades[i];
        }
        return page;
    }

    // -------------------------------------------------------------------------
    // Getters — Reputation Updates
    // -------------------------------------------------------------------------

    function reputationUpdateCount() external view returns (uint256) {
        return _reputationUpdates.length;
    }

    function getReputationUpdate(uint256 index) external view returns (ReputationUpdate memory) {
        return _reputationUpdates[index];
    }
}
