// SPDX-License-Identifier: MIT
pragma solidity ^0.8.24;

import "./RiskRouter.sol";

/**
 * @title UniswapV3Executor
 * @notice DEX swap executor that gates all trades through the RiskRouter.
 *
 * Architecture:
 *   1. Agent calls executeSwap() with trade parameters
 *   2. UniswapV3Executor calls RiskRouter.checkRisk() — trade blocked if risk fails
 *   3. If risk passes, swap is recorded (simulated in test env / real on mainnet)
 *   4. TradeExecuted event is emitted for off-chain reputation tracking
 *
 * In production, step 3 calls the actual Uniswap V3 SwapRouter.
 * For testnet / hackathon: swap is simulated and events are emitted.
 *
 * Whitelist pattern: only whitelisted executors can call executeSwap()
 * to prevent unauthorized use of the risk gate.
 */
contract UniswapV3Executor {
    // ─── State ───────────────────────────────────────────────────────────────

    RiskRouter public riskRouter;
    address public owner;

    /// @notice Whitelisted agent addresses that can execute swaps
    mapping(address => bool) public whitelisted;

    /// @notice Simulated output amounts from swaps (agentId => cumulative)
    mapping(uint256 => uint256) public agentSwapVolume;

    /// @notice Tracks swap count per agent
    mapping(uint256 => uint256) public agentSwapCount;

    /// @notice Total swaps executed
    uint256 public totalSwaps;

    // ─── Events ──────────────────────────────────────────────────────────────

    event TradeExecuted(
        uint256 indexed agentId,
        address indexed tokenIn,
        address indexed tokenOut,
        uint256 amountIn,
        uint256 amountOut,
        uint256 oraclePrice,
        bool simulated
    );

    event TradeBlocked(
        uint256 indexed agentId,
        uint256 amountIn,
        string reason
    );

    event AgentWhitelisted(address indexed agent, bool status);
    event SwapSimulated(uint256 indexed agentId, uint256 amountIn, uint256 estimatedOut);

    // ─── Errors ──────────────────────────────────────────────────────────────

    error NotOwner(address caller);
    error NotWhitelisted(address caller);
    error ZeroRiskRouter();
    error ZeroAmount();
    error SlippageExceeded(uint256 expected, uint256 received);
    error RiskCheckFailed(uint256 agentId, string reason);

    // ─── Constructor ─────────────────────────────────────────────────────────

    constructor(address riskRouterAddress) {
        if (riskRouterAddress == address(0)) revert ZeroRiskRouter();
        riskRouter = RiskRouter(riskRouterAddress);
        owner = msg.sender;
        // Owner is whitelisted by default
        whitelisted[msg.sender] = true;
        emit AgentWhitelisted(msg.sender, true);
    }

    // ─── Modifiers ───────────────────────────────────────────────────────────

    modifier onlyOwner() {
        if (msg.sender != owner) revert NotOwner(msg.sender);
        _;
    }

    modifier onlyWhitelisted() {
        if (!whitelisted[msg.sender]) revert NotWhitelisted(msg.sender);
        _;
    }

    // ─── Whitelist Management ────────────────────────────────────────────────

    /// @notice Add or remove an address from the executor whitelist
    function setWhitelisted(address agent, bool status) external onlyOwner {
        whitelisted[agent] = status;
        emit AgentWhitelisted(agent, status);
    }

    // ─── Core Swap Functions ─────────────────────────────────────────────────

    /**
     * @notice Execute a swap after passing the risk gate.
     *
     * @param agentId      ERC-8004 agent ID (from IdentityRegistry)
     * @param tokenIn      Address of input token (address(0) = ETH)
     * @param tokenOut     Address of output token (address(0) = native BTC proxy)
     * @param amountIn     Amount of tokenIn to swap (in token's base unit)
     * @param minAmountOut Minimum acceptable output (slippage protection)
     *
     * @return amountOut   Actual output amount received
     * @return oraclePrice Oracle price used for risk check
     */
    function executeSwap(
        uint256 agentId,
        address tokenIn,
        address tokenOut,
        uint256 amountIn,
        uint256 minAmountOut
    ) external onlyWhitelisted returns (uint256 amountOut, uint256 oraclePrice) {
        if (amountIn == 0) revert ZeroAmount();

        // ── 1. Risk gate ─────────────────────────────────────────────────────
        (bool allowed, uint256 price) = riskRouter.checkRisk(
            agentId,
            amountIn,
            tokenIn,
            tokenOut
        );

        oraclePrice = price;

        if (!allowed) {
            emit TradeBlocked(agentId, amountIn, "risk check failed");
            revert RiskCheckFailed(agentId, "deviation exceeds threshold");
        }

        // ── 2. Simulate swap (testnet) / call Uniswap V3 router (mainnet) ───
        amountOut = _simulateSwap(agentId, tokenIn, tokenOut, amountIn, oraclePrice);

        // ── 3. Slippage check ────────────────────────────────────────────────
        if (amountOut < minAmountOut) {
            revert SlippageExceeded(minAmountOut, amountOut);
        }

        // ── 4. Record stats ──────────────────────────────────────────────────
        agentSwapVolume[agentId] += amountIn;
        agentSwapCount[agentId] += 1;
        totalSwaps += 1;

        emit TradeExecuted(
            agentId,
            tokenIn,
            tokenOut,
            amountIn,
            amountOut,
            oraclePrice,
            true // simulated = true for testnet
        );

        return (amountOut, oraclePrice);
    }

    /**
     * @notice Simulate a swap without executing it (dry run / quoting).
     *
     * @param agentId  ERC-8004 agent ID
     * @param tokenIn  Input token address
     * @param tokenOut Output token address
     * @param amountIn Proposed input amount
     *
     * @return estimatedOut Estimated output amount
     * @return riskPassed   Whether the trade would pass the risk check
     * @return oraclePrice  Current oracle price
     */
    function simulateSwap(
        uint256 agentId,
        address tokenIn,
        address tokenOut,
        uint256 amountIn
    ) external returns (uint256 estimatedOut, bool riskPassed, uint256 oraclePrice) {
        if (amountIn == 0) {
            return (0, false, 0);
        }

        (bool allowed, uint256 price) = riskRouter.checkRisk(
            agentId,
            amountIn,
            tokenIn,
            tokenOut
        );

        oraclePrice = price;
        riskPassed = allowed;

        if (allowed) {
            estimatedOut = _simulateSwap(agentId, tokenIn, tokenOut, amountIn, price);
            emit SwapSimulated(agentId, amountIn, estimatedOut);
        }

        return (estimatedOut, riskPassed, oraclePrice);
    }

    // ─── Internal ────────────────────────────────────────────────────────────

    /**
     * @dev Simulate swap output using oracle price.
     *      In production: calls ISwapRouter(UNISWAP_V3_ROUTER).exactInputSingle()
     *      For testnet: compute output deterministically from oracle price.
     *
     *      Formula: amountOut = amountIn * PRICE_DECIMALS_SCALE / oraclePrice
     *      (converts token units using oracle 8-decimal price)
     */
    function _simulateSwap(
        uint256, /* agentId */
        address, /* tokenIn */
        address, /* tokenOut */
        uint256 amountIn,
        uint256 oraclePrice
    ) internal pure returns (uint256) {
        if (oraclePrice == 0) return 0;
        // Scale: oracle uses 8 decimals. amountIn is in token base unit.
        // Simplified: output = amountIn * 1e8 / oraclePrice
        // (represents buying the other asset at oracle price)
        return (amountIn * 1e8) / oraclePrice;
    }

    // ─── View Functions ──────────────────────────────────────────────────────

    /// @notice Get swap statistics for an agent
    function getAgentStats(uint256 agentId)
        external
        view
        returns (uint256 swapCount, uint256 totalVolume)
    {
        return (agentSwapCount[agentId], agentSwapVolume[agentId]);
    }

    /// @notice Check if the risk router is configured correctly
    function getRiskRouter() external view returns (address) {
        return address(riskRouter);
    }
}
