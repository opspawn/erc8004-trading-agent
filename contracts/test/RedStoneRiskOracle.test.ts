import { expect } from "chai";
import { ethers } from "hardhat";
import { RedStoneRiskOracle } from "../typechain-types";
import { SignerWithAddress } from "@nomicfoundation/hardhat-ethers/signers";

describe("RedStoneRiskOracle", function () {
  let oracle: RedStoneRiskOracle;
  let owner: SignerWithAddress;
  let nonOwner: SignerWithAddress;

  const PRICE_DECIMALS = 8n;
  const ONE_USD = 10n ** PRICE_DECIMALS;

  const DEFAULT_ETH = 3000n * ONE_USD;
  const DEFAULT_BTC = 60000n * ONE_USD;

  beforeEach(async function () {
    [owner, nonOwner] = await ethers.getSigners();
    const Oracle = await ethers.getContractFactory("RedStoneRiskOracle");
    oracle = await Oracle.deploy();
    await oracle.waitForDeployment();
  });

  // ─── Deployment ──────────────────────────────────────────────────────────────

  it("should set owner to deployer", async function () {
    expect(await oracle.owner()).to.equal(owner.address);
  });

  it("should initialize ETH price to $3000 (8 decimals)", async function () {
    expect(await oracle.getEthPrice()).to.equal(DEFAULT_ETH);
  });

  it("should initialize BTC price to $60000 (8 decimals)", async function () {
    expect(await oracle.getBtcPrice()).to.equal(DEFAULT_BTC);
  });

  // ─── Price Updates ───────────────────────────────────────────────────────────

  it("should allow owner to update ETH price and emit PriceUpdated", async function () {
    const newPrice = 3500n * ONE_USD;
    await expect(oracle.connect(owner).setEthPrice(newPrice))
      .to.emit(oracle, "PriceUpdated")
      .withArgs(ethers.encodeBytes32String("ETH").substring(0, 18) + "0".repeat(48), newPrice, await getTimestamp());

    // Verify price stored
    expect(await oracle.getEthPrice()).to.equal(newPrice);
  });

  it("should revert setEthPrice from non-owner", async function () {
    await expect(oracle.connect(nonOwner).setEthPrice(1000n * ONE_USD))
      .to.be.revertedWithCustomError(oracle, "NotOwner")
      .withArgs(nonOwner.address);
  });

  it("should revert setEthPrice with zero", async function () {
    await expect(oracle.connect(owner).setEthPrice(0n))
      .to.be.revertedWithCustomError(oracle, "ZeroPrice");
  });

  // ─── Threshold Validation ────────────────────────────────────────────────────

  it("should return true when price is within threshold", async function () {
    const price = 3000n * ONE_USD;
    const min = 2900n * ONE_USD;
    const max = 3100n * ONE_USD;
    await expect(oracle.validatePriceThreshold(price, min, max))
      .to.emit(oracle, "ThresholdValidated");
    // Call statically to read return value
    const result = await oracle.validatePriceThreshold.staticCall(price, min, max);
    expect(result).to.be.true;
  });

  it("should return false when price is outside threshold", async function () {
    const price = 4000n * ONE_USD;
    const min = 2900n * ONE_USD;
    const max = 3100n * ONE_USD;
    const result = await oracle.validatePriceThreshold.staticCall(price, min, max);
    expect(result).to.be.false;
  });

  it("should revert validatePriceThreshold when minPrice > maxPrice", async function () {
    await expect(
      oracle.validatePriceThreshold(3000n * ONE_USD, 4000n * ONE_USD, 2000n * ONE_USD)
    ).to.be.revertedWithCustomError(oracle, "InvalidPriceRange");
  });

  // ─── PRICE_DECIMALS constant ─────────────────────────────────────────────────

  it("should expose PRICE_DECIMALS = 8", async function () {
    expect(await oracle.PRICE_DECIMALS()).to.equal(8n);
  });
});

async function getTimestamp(): Promise<number> {
  const block = await ethers.provider.getBlock("latest");
  return block ? block.timestamp + 1 : 0;
}
