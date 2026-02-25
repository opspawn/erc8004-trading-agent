// SPDX-License-Identifier: MIT
pragma solidity ^0.8.24;

/**
 * @title RedStoneRiskOracle
 * @notice Price oracle with risk threshold validation, modeled on RedStone data feeds.
 *
 * In production this contract inherits from RedStone's MainDemoConsumerBase and
 * reads prices by calling:
 *   getOracleNumericValueFromTxMsg(bytes32("ETH"))
 *
 * For local testing the owner updates prices directly (simulating RedStone push).
 * RedStone npm pkg: redstone-finance evm-connector (MainDemoConsumerBase)
 */
contract RedStoneRiskOracle {
    // ─── State ───────────────────────────────────────────────────────────────

    address public owner;

    /// @dev Price in USD with PRICE_DECIMALS decimal places (same as RedStone)
    uint256 private _ethPrice;
    uint256 private _btcPrice;

    /// @notice RedStone uses 8 decimals: 1 USD = 10^8
    uint256 public constant PRICE_DECIMALS = 8;

    // ─── Events ──────────────────────────────────────────────────────────────

    event PriceUpdated(bytes32 indexed feedId, uint256 price, uint256 timestamp);
    event ThresholdValidated(
        address indexed caller,
        uint256 price,
        uint256 minPrice,
        uint256 maxPrice,
        bool result
    );

    // ─── Errors ──────────────────────────────────────────────────────────────

    error NotOwner(address caller);
    error InvalidPriceRange(uint256 minPrice, uint256 maxPrice);
    error ZeroPrice();

    // ─── Constructor ─────────────────────────────────────────────────────────

    constructor() {
        owner = msg.sender;
        // Initialize with reasonable defaults (8-decimal fixed-point)
        _ethPrice = 3000 * 10 ** PRICE_DECIMALS;   // $3 000
        _btcPrice = 60000 * 10 ** PRICE_DECIMALS;  // $60 000
    }

    // ─── Modifiers ───────────────────────────────────────────────────────────

    modifier onlyOwner() {
        if (msg.sender != owner) revert NotOwner(msg.sender);
        _;
    }

    // ─── Price Updates (owner / RedStone relay) ───────────────────────────────

    /// @notice Update ETH/USD price (called by owner or RedStone keeper)
    function setEthPrice(uint256 price) external onlyOwner {
        if (price == 0) revert ZeroPrice();
        _ethPrice = price;
        emit PriceUpdated(bytes32("ETH"), price, block.timestamp);
    }

    /// @notice Update BTC/USD price
    function setBtcPrice(uint256 price) external onlyOwner {
        if (price == 0) revert ZeroPrice();
        _btcPrice = price;
        emit PriceUpdated(bytes32("BTC"), price, block.timestamp);
    }

    // ─── Price Reads ─────────────────────────────────────────────────────────

    /// @notice Get current ETH/USD price with PRICE_DECIMALS decimal places
    function getEthPrice() external view returns (uint256) {
        return _ethPrice;
    }

    /// @notice Get current BTC/USD price with PRICE_DECIMALS decimal places
    function getBtcPrice() external view returns (uint256) {
        return _btcPrice;
    }

    // ─── Risk Threshold Validation ───────────────────────────────────────────

    /**
     * @notice Check whether a price lies within an acceptable range.
     * @param price       Price to validate (same decimals as PRICE_DECIMALS)
     * @param minPrice    Inclusive lower bound
     * @param maxPrice    Inclusive upper bound
     * @return withinRange True if minPrice <= price <= maxPrice
     */
    function validatePriceThreshold(
        uint256 price,
        uint256 minPrice,
        uint256 maxPrice
    ) external returns (bool withinRange) {
        if (minPrice > maxPrice) revert InvalidPriceRange(minPrice, maxPrice);
        withinRange = price >= minPrice && price <= maxPrice;
        emit ThresholdValidated(msg.sender, price, minPrice, maxPrice, withinRange);
    }
}
