// SPDX-License-Identifier: MIT
pragma solidity ^0.8.24;

import "@openzeppelin/contracts/utils/cryptography/ECDSA.sol";
import "@openzeppelin/contracts/utils/cryptography/EIP712.sol";

/**
 * @title AgentWallet
 * @notice EIP-1271 compatible smart contract wallet for autonomous AI agents.
 *         The agent signs messages off-chain; the wallet verifies them on-chain.
 *
 * @dev Implements:
 *      - EIP-1271: isValidSignature for contract-based signature verification
 *      - EIP-712: Typed structured data signing
 *      - execute(): Generic call execution for on-chain interactions
 *
 * Usage:
 *   1. Agent holds the private key (owner EOA)
 *   2. AgentWallet is the on-chain identity for contract interactions
 *   3. Agent signs messages with its EOA key
 *   4. DeFi protocols verify via isValidSignature()
 */
contract AgentWallet is EIP712 {
    using ECDSA for bytes32;

    // -------------------------------------------------------------------------
    // Constants
    // -------------------------------------------------------------------------

    /// @notice EIP-1271 magic value returned on valid signature
    bytes4 public constant MAGIC_VALUE = 0x1626ba7e;

    /// @notice EIP-712 type hash for Execute operations
    bytes32 public constant EXECUTE_TYPEHASH = keccak256(
        "Execute(address target,uint256 value,bytes data,uint256 nonce)"
    );

    // -------------------------------------------------------------------------
    // State
    // -------------------------------------------------------------------------

    address public owner;
    uint256 public nonce;

    // -------------------------------------------------------------------------
    // Events
    // -------------------------------------------------------------------------

    event Executed(address indexed target, uint256 value, bytes data, bool success);
    event OwnershipTransferred(address indexed previousOwner, address indexed newOwner);
    event Received(address indexed sender, uint256 value);

    // -------------------------------------------------------------------------
    // Errors
    // -------------------------------------------------------------------------

    error Unauthorized(address caller);
    error ExecutionFailed(address target, bytes returnData);
    error InvalidSignature();
    error ZeroAddress();

    // -------------------------------------------------------------------------
    // Modifiers
    // -------------------------------------------------------------------------

    modifier onlyOwner() {
        if (msg.sender != owner) revert Unauthorized(msg.sender);
        _;
    }

    // -------------------------------------------------------------------------
    // Constructor
    // -------------------------------------------------------------------------

    constructor(address _owner) EIP712("AgentWallet", "1") {
        if (_owner == address(0)) revert ZeroAddress();
        owner = _owner;
        emit OwnershipTransferred(address(0), _owner);
    }

    // -------------------------------------------------------------------------
    // EIP-1271
    // -------------------------------------------------------------------------

    /**
     * @notice EIP-1271 signature verification.
     *         Returns magic value if the hash was signed by the owner.
     * @param hash       The message hash that was signed
     * @param signature  The signature bytes (65 bytes: r, s, v)
     * @return magicValue  0x1626ba7e if valid, 0xffffffff if invalid
     */
    function isValidSignature(
        bytes32 hash,
        bytes calldata signature
    ) external view returns (bytes4 magicValue) {
        address recovered = ECDSA.recover(hash, signature);
        if (recovered == owner) {
            return MAGIC_VALUE;
        }
        return 0xffffffff;
    }

    // -------------------------------------------------------------------------
    // Execution
    // -------------------------------------------------------------------------

    /**
     * @notice Execute an arbitrary call from this wallet.
     * @param target  Contract to call
     * @param value   ETH to send
     * @param data    Calldata
     */
    function execute(
        address target,
        uint256 value,
        bytes calldata data
    ) external onlyOwner returns (bytes memory returnData) {
        bool success;
        (success, returnData) = target.call{value: value}(data);
        emit Executed(target, value, data, success);
        if (!success) revert ExecutionFailed(target, returnData);
    }

    /**
     * @notice Execute via EIP-712 signed message (allows meta-transactions).
     *         The owner signs off-chain; anyone can relay the tx.
     * @param target     Contract to call
     * @param value      ETH to send
     * @param data       Calldata
     * @param signature  Owner's EIP-712 signature over (target, value, data, nonce)
     */
    function executeWithSignature(
        address target,
        uint256 value,
        bytes calldata data,
        bytes calldata signature
    ) external returns (bytes memory returnData) {
        bytes32 structHash = keccak256(
            abi.encode(EXECUTE_TYPEHASH, target, value, keccak256(data), nonce)
        );
        bytes32 digest = _hashTypedDataV4(structHash);
        address signer = ECDSA.recover(digest, signature);
        if (signer != owner) revert InvalidSignature();

        nonce++;

        bool success;
        (success, returnData) = target.call{value: value}(data);
        emit Executed(target, value, data, success);
        if (!success) revert ExecutionFailed(target, returnData);
    }

    /**
     * @notice Transfer ownership of this wallet.
     * @param newOwner  New owner address
     */
    function transferOwnership(address newOwner) external onlyOwner {
        if (newOwner == address(0)) revert ZeroAddress();
        emit OwnershipTransferred(owner, newOwner);
        owner = newOwner;
    }

    /**
     * @notice Get the EIP-712 domain separator.
     */
    function domainSeparator() external view returns (bytes32) {
        return _domainSeparatorV4();
    }

    /**
     * @notice Compute the EIP-712 digest for an execute operation.
     *         Used by the agent to sign off-chain.
     */
    function hashExecute(
        address target,
        uint256 value,
        bytes calldata data
    ) external view returns (bytes32) {
        bytes32 structHash = keccak256(
            abi.encode(EXECUTE_TYPEHASH, target, value, keccak256(data), nonce)
        );
        return _hashTypedDataV4(structHash);
    }

    // -------------------------------------------------------------------------
    // Receive
    // -------------------------------------------------------------------------

    receive() external payable {
        emit Received(msg.sender, msg.value);
    }
}
