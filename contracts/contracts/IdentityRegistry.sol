// SPDX-License-Identifier: MIT
pragma solidity ^0.8.24;

import "@openzeppelin/contracts/token/ERC721/extensions/ERC721URIStorage.sol";
import "@openzeppelin/contracts/access/Ownable.sol";

/**
 * @title IdentityRegistry
 * @notice ERC-8004 compliant agent identity registry.
 *         Each agent is represented as an ERC-721 token with a DID-style URI.
 *         Format: eip155:{chainId}:{contractAddress}:{agentId}
 *
 * @dev Extends ERC-721 with URI storage for agent metadata.
 *      Agent IDs are sequential, starting from 1.
 */
contract IdentityRegistry is ERC721URIStorage, Ownable {
    // -------------------------------------------------------------------------
    // State
    // -------------------------------------------------------------------------

    uint256 private _nextAgentId;

    /// @notice Maps agent address to their agentId (0 = not registered)
    mapping(address => uint256) public agentIdOf;

    /// @notice The chain ID used when constructing agent DIDs
    uint256 public immutable chainId;

    // -------------------------------------------------------------------------
    // Events
    // -------------------------------------------------------------------------

    /// @notice Emitted when a new agent identity is registered
    event AgentRegistered(uint256 indexed agentId, address indexed owner, string uri);

    /// @notice Emitted when an agent updates its metadata URI
    event AgentURIUpdated(uint256 indexed agentId, string newUri);

    // -------------------------------------------------------------------------
    // Errors
    // -------------------------------------------------------------------------

    error AlreadyRegistered(address agent, uint256 existingId);
    error NotTokenOwner(uint256 agentId, address caller);
    error InvalidAgentId(uint256 agentId);

    // -------------------------------------------------------------------------
    // Constructor
    // -------------------------------------------------------------------------

    constructor() ERC721("ERC8004 Agent Identity", "AGENT") Ownable(msg.sender) {
        _nextAgentId = 1;
        chainId = block.chainid;
    }

    // -------------------------------------------------------------------------
    // Core Functions
    // -------------------------------------------------------------------------

    /**
     * @notice Register a new agent identity.
     * @param metadataURI  IPFS/HTTPS URI pointing to agent metadata JSON
     * @return agentId     The newly assigned agent ID
     */
    function mint(string calldata metadataURI) external returns (uint256 agentId) {
        if (agentIdOf[msg.sender] != 0) {
            revert AlreadyRegistered(msg.sender, agentIdOf[msg.sender]);
        }

        agentId = _nextAgentId++;
        agentIdOf[msg.sender] = agentId;

        _safeMint(msg.sender, agentId);
        _setTokenURI(agentId, metadataURI);

        emit AgentRegistered(agentId, msg.sender, metadataURI);
    }

    /**
     * @notice Update the metadata URI for an agent.
     * @param agentId     The agent ID to update
     * @param newUri      New metadata URI
     */
    function setAgentURI(uint256 agentId, string calldata newUri) external {
        if (agentId == 0 || agentId >= _nextAgentId) revert InvalidAgentId(agentId);
        if (ownerOf(agentId) != msg.sender) revert NotTokenOwner(agentId, msg.sender);

        _setTokenURI(agentId, newUri);
        emit AgentURIUpdated(agentId, newUri);
    }

    /**
     * @notice Returns the DID-style identifier for an agent.
     * @param agentId  The agent ID
     * @return did     e.g. "eip155:11155111:0xABC...DEF:1"
     */
    function agentDID(uint256 agentId) external view returns (string memory did) {
        if (agentId == 0 || agentId >= _nextAgentId) revert InvalidAgentId(agentId);
        did = string(
            abi.encodePacked(
                "eip155:",
                _uint2str(chainId),
                ":",
                _addr2str(address(this)),
                ":",
                _uint2str(agentId)
            )
        );
    }

    /**
     * @notice Total number of registered agents.
     */
    function totalAgents() external view returns (uint256) {
        return _nextAgentId - 1;
    }

    /**
     * @notice Check if an address has a registered agent.
     */
    function isRegistered(address agent) external view returns (bool) {
        return agentIdOf[agent] != 0;
    }

    // -------------------------------------------------------------------------
    // Internal Helpers
    // -------------------------------------------------------------------------

    function _uint2str(uint256 v) internal pure returns (string memory) {
        if (v == 0) return "0";
        uint256 digits;
        uint256 tmp = v;
        while (tmp != 0) { digits++; tmp /= 10; }
        bytes memory buf = new bytes(digits);
        while (v != 0) { digits--; buf[digits] = bytes1(uint8(48 + v % 10)); v /= 10; }
        return string(buf);
    }

    function _addr2str(address addr) internal pure returns (string memory) {
        bytes20 b = bytes20(addr);
        bytes memory hex_ = new bytes(42);
        hex_[0] = "0"; hex_[1] = "x";
        bytes memory alphabet = "0123456789abcdef";
        for (uint256 i = 0; i < 20; i++) {
            hex_[2 + i * 2]     = alphabet[uint8(b[i]) >> 4];
            hex_[3 + i * 2]     = alphabet[uint8(b[i]) & 0x0f];
        }
        return string(hex_);
    }
}
