// SPDX-License-Identifier: MIT
pragma solidity ^0.8.24;

import "./IdentityRegistry.sol";

/**
 * @title ValidationRegistry
 * @notice ERC-8004 compliant validation registry.
 *         Agents submit validation requests for their actions (e.g. trade outcomes).
 *         Validators (or the agent itself) respond with a 0-100 confidence score.
 *
 * @dev Request IDs are sequential. Any address can submit a validation response
 *      (in production, restrict to whitelisted validators).
 */
contract ValidationRegistry {
    // -------------------------------------------------------------------------
    // Types
    // -------------------------------------------------------------------------

    enum RequestStatus {
        Pending,
        Validated,
        Disputed,
        Expired
    }

    struct ValidationRequest {
        uint256 agentId;
        address requester;
        string dataURI;        // IPFS/HTTPS link to trade data
        bytes32 dataHash;      // keccak256 of the data at dataURI
        uint256 createdAt;
        RequestStatus status;
    }

    struct ValidationResult {
        address validator;
        uint8 response;         // 0-100 confidence score
        string commentURI;      // Optional IPFS link to validator's reasoning
        uint256 submittedAt;
    }

    // -------------------------------------------------------------------------
    // State
    // -------------------------------------------------------------------------

    IdentityRegistry public immutable identityRegistry;

    uint256 private _nextRequestId;

    /// @dev requestId → request
    mapping(uint256 => ValidationRequest) public requests;

    /// @dev requestId → result (exists only after validation)
    mapping(uint256 => ValidationResult) public results;

    /// @dev agentId → list of requestIds
    mapping(uint256 => uint256[]) private _agentRequests;

    // -------------------------------------------------------------------------
    // Events
    // -------------------------------------------------------------------------

    event ValidationRequested(
        uint256 indexed requestId,
        uint256 indexed agentId,
        address indexed requester,
        string dataURI,
        bytes32 dataHash
    );

    event ValidationSubmitted(
        uint256 indexed requestId,
        uint256 indexed agentId,
        address indexed validator,
        uint8 response
    );

    event RequestDisputed(uint256 indexed requestId, address indexed disputer);

    // -------------------------------------------------------------------------
    // Errors
    // -------------------------------------------------------------------------

    error AgentNotRegistered(uint256 agentId);
    error RequestNotFound(uint256 requestId);
    error AlreadyValidated(uint256 requestId);
    error InvalidResponse(uint8 response);
    error NotRequester(uint256 requestId, address caller);
    error InvalidDataHash();

    // -------------------------------------------------------------------------
    // Constants
    // -------------------------------------------------------------------------

    uint256 public constant REQUEST_EXPIRY = 30 days;

    // -------------------------------------------------------------------------
    // Constructor
    // -------------------------------------------------------------------------

    constructor(address identityRegistryAddress) {
        identityRegistry = IdentityRegistry(identityRegistryAddress);
        _nextRequestId = 1;
    }

    // -------------------------------------------------------------------------
    // Core Functions
    // -------------------------------------------------------------------------

    /**
     * @notice Submit a validation request for an agent action.
     * @param agentId   The requesting agent's ID
     * @param dataURI   URI pointing to the data to be validated
     * @param dataHash  keccak256 hash of the data at dataURI
     * @return requestId  The assigned request ID
     */
    function validationRequest(
        uint256 agentId,
        string calldata dataURI,
        bytes32 dataHash
    ) external returns (uint256 requestId) {
        if (dataHash == bytes32(0)) revert InvalidDataHash();
        _requireRegistered(agentId);

        requestId = _nextRequestId++;

        requests[requestId] = ValidationRequest({
            agentId: agentId,
            requester: msg.sender,
            dataURI: dataURI,
            dataHash: dataHash,
            createdAt: block.timestamp,
            status: RequestStatus.Pending
        });

        _agentRequests[agentId].push(requestId);

        emit ValidationRequested(requestId, agentId, msg.sender, dataURI, dataHash);
    }

    /**
     * @notice Submit a validation response for a request.
     * @param requestId   ID of the validation request
     * @param response    Confidence score 0-100
     * @param commentURI  Optional URI to validator reasoning (empty string if none)
     */
    function submitValidation(
        uint256 requestId,
        uint8 response,
        string calldata commentURI
    ) external {
        ValidationRequest storage req = _requireRequest(requestId);

        if (req.status != RequestStatus.Pending) {
            revert AlreadyValidated(requestId);
        }
        if (response > 100) revert InvalidResponse(response);

        req.status = RequestStatus.Validated;

        results[requestId] = ValidationResult({
            validator: msg.sender,
            response: response,
            commentURI: commentURI,
            submittedAt: block.timestamp
        });

        emit ValidationSubmitted(requestId, req.agentId, msg.sender, response);
    }

    /**
     * @notice Get the result of a completed validation.
     * @param requestId  The request ID to query
     * @return validator   Address that submitted the validation
     * @return response    Score 0-100
     * @return timestamp   When validation was submitted
     */
    function getValidationResult(
        uint256 requestId
    ) external view returns (address validator, uint8 response, uint256 timestamp) {
        _requireRequest(requestId);
        ValidationResult storage result = results[requestId];
        return (result.validator, result.response, result.submittedAt);
    }

    /**
     * @notice Dispute a validation result.
     * @param requestId  The request to dispute
     */
    function disputeValidation(uint256 requestId) external {
        ValidationRequest storage req = _requireRequest(requestId);
        if (req.requester != msg.sender) revert NotRequester(requestId, msg.sender);
        req.status = RequestStatus.Disputed;
        emit RequestDisputed(requestId, msg.sender);
    }

    /**
     * @notice Get all request IDs for an agent.
     */
    function getAgentRequests(uint256 agentId) external view returns (uint256[] memory) {
        return _agentRequests[agentId];
    }

    /**
     * @notice Total number of validation requests.
     */
    function totalRequests() external view returns (uint256) {
        return _nextRequestId - 1;
    }

    /**
     * @notice Check if a request has expired (pending for > 30 days).
     */
    function isExpired(uint256 requestId) external view returns (bool) {
        ValidationRequest storage req = requests[requestId];
        if (req.createdAt == 0) return false;
        return (req.status == RequestStatus.Pending &&
                block.timestamp > req.createdAt + REQUEST_EXPIRY);
    }

    // -------------------------------------------------------------------------
    // Internal
    // -------------------------------------------------------------------------

    function _requireRegistered(uint256 agentId) internal view {
        try identityRegistry.ownerOf(agentId) returns (address) {
            // ok
        } catch {
            revert AgentNotRegistered(agentId);
        }
    }

    function _requireRequest(
        uint256 requestId
    ) internal view returns (ValidationRequest storage) {
        ValidationRequest storage req = requests[requestId];
        if (req.createdAt == 0) revert RequestNotFound(requestId);
        return req;
    }
}
