// SPDX-License-Identifier: MIT
pragma solidity ^0.8.24;

import "./RedStoneRiskOracle.sol";

/**
 * @title RiskRouter
 * @notice Pre-trade risk gate: fetches RedStone oracle price and validates
 *         that the trade amount is within an acceptable deviation of the
 *         live market price (5% tolerance by default).
 *
 * Intended flow (agent side):
 *   1. Agent proposes a trade (amount, token pair)
 *   2. RiskRouter.checkRisk() fetches oracle price
 *   3. If price is within tolerance → emits RiskCheckPassed → trade proceeds
 *   4. If out of range → emits RiskCheckFailed → trade blocked
 *
 * No actual Uniswap calls — this contract owns only the risk-check interface.
 */
contract RiskRouter {
    // ─── State ───────────────────────────────────────────────────────────────

    RedStoneRiskOracle public oracle;
    address public owner;

    /// @notice Allowed deviation from oracle price, expressed in basis points.
    ///         Default 500 bps = 5%.
    uint256 public maxDeviationBps;

    uint256 private constant BPS_DENOMINATOR = 10_000;

    // ─── Events ──────────────────────────────────────────────────────────────

    event RiskCheckPassed(
        uint256 indexed agentId,
        uint256 amount,
        uint256 oraclePrice,
        uint256 deviationBps
    );

    event RiskCheckFailed(
        uint256 indexed agentId,
        uint256 amount,
        uint256 oraclePrice,
        string reason
    );

    event MaxDeviationUpdated(uint256 oldBps, uint256 newBps);

    // ─── Errors ──────────────────────────────────────────────────────────────

    error NotOwner(address caller);
    error ZeroOracle();
    error UnsupportedToken(address token);
    error InvalidDeviation(uint256 bps);

    // ─── Constructor ─────────────────────────────────────────────────────────

    constructor(address oracleAddress) {
        if (oracleAddress == address(0)) revert ZeroOracle();
        oracle = RedStoneRiskOracle(oracleAddress);
        owner = msg.sender;
        maxDeviationBps = 500; // 5% default
    }

    // ─── Modifiers ───────────────────────────────────────────────────────────

    modifier onlyOwner() {
        if (msg.sender != owner) revert NotOwner(msg.sender);
        _;
    }

    // ─── Configuration ───────────────────────────────────────────────────────

    /// @notice Update the maximum allowed deviation (in basis points)
    function setMaxDeviationBps(uint256 newBps) external onlyOwner {
        if (newBps == 0 || newBps > BPS_DENOMINATOR) revert InvalidDeviation(newBps);
        emit MaxDeviationUpdated(maxDeviationBps, newBps);
        maxDeviationBps = newBps;
    }

    // ─── Risk Check ──────────────────────────────────────────────────────────

    /**
     * @notice Validate a proposed trade amount against the oracle price.
     *
     * @param agentId     On-chain agent ID (from IdentityRegistry)
     * @param tradeAmount Trade size in the smallest token unit (e.g. wei for ETH)
     * @param tokenIn     Address of the token being sold (zero address = ETH)
     * @param tokenOut    Address of the token being bought (zero address = BTC proxy)
     *
     * @return allowed      True if the trade passes the risk check
     * @return oraclePrice  Current oracle price for the relevant asset
     */
    function checkRisk(
        uint256 agentId,
        uint256 tradeAmount,
        address tokenIn,
        address tokenOut
    ) external returns (bool allowed, uint256 oraclePrice) {
        // ── 1. Determine relevant oracle price ──────────────────────────────
        bool isEthPair = (tokenIn == address(0));
        if (isEthPair) {
            oraclePrice = oracle.getEthPrice();
        } else {
            oraclePrice = oracle.getBtcPrice();
        }

        // ── 2. Zero amount is never allowed ─────────────────────────────────
        if (tradeAmount == 0) {
            emit RiskCheckFailed(agentId, tradeAmount, oraclePrice, "zero amount");
            return (false, oraclePrice);
        }

        // ── 3. Compute deviation: |tradeAmount - oraclePrice| / oraclePrice ──
        //    Both sides are scaled the same way (8-decimal oracle units).
        //    Compare tradeAmount (in oracle units) against the oracle price.
        uint256 diff;
        if (tradeAmount >= oraclePrice) {
            diff = tradeAmount - oraclePrice;
        } else {
            diff = oraclePrice - tradeAmount;
        }

        uint256 deviationBps = (diff * BPS_DENOMINATOR) / oraclePrice;

        // ── 4. Pass or fail ─────────────────────────────────────────────────
        if (deviationBps <= maxDeviationBps) {
            emit RiskCheckPassed(agentId, tradeAmount, oraclePrice, deviationBps);
            return (true, oraclePrice);
        } else {
            emit RiskCheckFailed(
                agentId,
                tradeAmount,
                oraclePrice,
                "deviation exceeds threshold"
            );
            return (false, oraclePrice);
        }
    }
}
