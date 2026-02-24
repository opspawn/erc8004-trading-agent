// SPDX-License-Identifier: MIT
pragma solidity ^0.8.24;

import "./IdentityRegistry.sol";

/**
 * @title ReputationRegistry
 * @notice ERC-8004 compliant on-chain reputation system.
 *         Clients submit feedback for agents. Feedback is aggregated
 *         into a weighted average score per agent.
 *
 * @dev Prevents self-feedback. Tracks per-client feedback per agent.
 *      Scores use int128 with configurable decimal precision.
 */
contract ReputationRegistry {
    // -------------------------------------------------------------------------
    // Types
    // -------------------------------------------------------------------------

    struct Feedback {
        address client;
        int128 value;       // raw score (divide by 10^decimals for actual)
        uint8 decimals;     // decimal places (e.g. 2 → divide by 100)
        bytes32 tag1;       // e.g. "accuracy", "speed", "trust"
        bytes32 tag2;       // secondary tag
        string endpointURI; // agent endpoint that was evaluated
        bytes32 fileHash;   // IPFS hash of supporting evidence
        uint256 timestamp;
    }

    // -------------------------------------------------------------------------
    // State
    // -------------------------------------------------------------------------

    IdentityRegistry public immutable identityRegistry;

    /// @dev agentId → list of feedback entries
    mapping(uint256 => Feedback[]) private _feedback;

    /// @dev agentId → client → feedback index + 1 (0 = none)
    mapping(uint256 => mapping(address => uint256)) private _clientFeedbackIndex;

    // -------------------------------------------------------------------------
    // Events
    // -------------------------------------------------------------------------

    event FeedbackGiven(
        uint256 indexed agentId,
        address indexed from,
        int128 value,
        uint8 decimals,
        bytes32 tag1,
        bytes32 tag2
    );

    event FeedbackUpdated(
        uint256 indexed agentId,
        address indexed from,
        int128 newValue
    );

    // -------------------------------------------------------------------------
    // Errors
    // -------------------------------------------------------------------------

    error SelfFeedbackNotAllowed(uint256 agentId, address owner);
    error AgentNotRegistered(uint256 agentId);
    error NoFeedbackFound(uint256 agentId, address client);
    error InvalidScore(int128 value, uint8 decimals);

    // -------------------------------------------------------------------------
    // Constructor
    // -------------------------------------------------------------------------

    constructor(address identityRegistryAddress) {
        identityRegistry = IdentityRegistry(identityRegistryAddress);
    }

    // -------------------------------------------------------------------------
    // Core Functions
    // -------------------------------------------------------------------------

    /**
     * @notice Submit feedback for an agent.
     * @param agentId      Target agent's ID
     * @param value        Score value (e.g. 850 with decimals=2 → 8.50)
     * @param decimals     Decimal places for value
     * @param tag1         Primary evaluation tag
     * @param tag2         Secondary evaluation tag
     * @param endpointURI  Agent endpoint that was tested
     * @param fileHash     IPFS hash of evidence file (bytes32(0) if none)
     */
    function giveFeedback(
        uint256 agentId,
        int128 value,
        uint8 decimals,
        bytes32 tag1,
        bytes32 tag2,
        string calldata endpointURI,
        bytes32 fileHash
    ) external {
        // Validate agent exists
        address agentOwner = _requireRegistered(agentId);

        // Prevent self-feedback
        if (msg.sender == agentOwner) {
            revert SelfFeedbackNotAllowed(agentId, agentOwner);
        }

        // Normalize: max score with given decimals is 10^decimals * 10
        // Just validate decimals <= 18 to prevent overflow
        if (decimals > 18) revert InvalidScore(value, decimals);

        Feedback memory fb = Feedback({
            client: msg.sender,
            value: value,
            decimals: decimals,
            tag1: tag1,
            tag2: tag2,
            endpointURI: endpointURI,
            fileHash: fileHash,
            timestamp: block.timestamp
        });

        uint256 existingIdx = _clientFeedbackIndex[agentId][msg.sender];
        if (existingIdx == 0) {
            // New feedback
            _feedback[agentId].push(fb);
            _clientFeedbackIndex[agentId][msg.sender] = _feedback[agentId].length;
        } else {
            // Update existing
            _feedback[agentId][existingIdx - 1] = fb;
            emit FeedbackUpdated(agentId, msg.sender, value);
            return;
        }

        emit FeedbackGiven(agentId, msg.sender, value, decimals, tag1, tag2);
    }

    /**
     * @notice Get feedback submitted by a specific client for an agent.
     * @param agentId        Target agent ID
     * @param clientAddress  Address of the client who gave feedback
     */
    function getFeedback(
        uint256 agentId,
        address clientAddress
    ) external view returns (Feedback memory) {
        uint256 idx = _clientFeedbackIndex[agentId][clientAddress];
        if (idx == 0) revert NoFeedbackFound(agentId, clientAddress);
        return _feedback[agentId][idx - 1];
    }

    /**
     * @notice Get all feedback entries for an agent.
     * @param agentId  Target agent ID
     */
    function getAllFeedback(uint256 agentId) external view returns (Feedback[] memory) {
        return _feedback[agentId];
    }

    /**
     * @notice Get the aggregate reputation score for an agent.
     *         Returns the average score normalized to 2 decimal places.
     * @param agentId  Target agent ID
     * @return score   Aggregate score (divide by 100 for actual value)
     * @return count   Number of feedback entries
     */
    function getAggregateScore(
        uint256 agentId
    ) external view returns (int256 score, uint256 count) {
        Feedback[] storage feedbacks = _feedback[agentId];
        count = feedbacks.length;
        if (count == 0) return (0, 0);

        int256 sum = 0;
        for (uint256 i = 0; i < count; i++) {
            Feedback storage fb = feedbacks[i];
            // Normalize all values to 2 decimal places
            int256 normalized;
            if (fb.decimals >= 2) {
                normalized = int256(fb.value) / int256(10 ** uint256(fb.decimals - 2));
            } else {
                normalized = int256(fb.value) * int256(10 ** uint256(2 - fb.decimals));
            }
            sum += normalized;
        }
        score = sum / int256(count);
    }

    /**
     * @notice Get feedback count for an agent.
     */
    function feedbackCount(uint256 agentId) external view returns (uint256) {
        return _feedback[agentId].length;
    }

    /**
     * @notice Check if a client has given feedback for an agent.
     */
    function hasFeedback(uint256 agentId, address client) external view returns (bool) {
        return _clientFeedbackIndex[agentId][client] != 0;
    }

    // -------------------------------------------------------------------------
    // Internal
    // -------------------------------------------------------------------------

    function _requireRegistered(uint256 agentId) internal view returns (address owner) {
        // agentId 0 is invalid; check it exists in registry
        try identityRegistry.ownerOf(agentId) returns (address o) {
            return o;
        } catch {
            revert AgentNotRegistered(agentId);
        }
    }
}
