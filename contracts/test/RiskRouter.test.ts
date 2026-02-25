import { expect } from "chai";
import { ethers } from "hardhat";
import { RiskRouter, RedStoneRiskOracle } from "../typechain-types";
import { SignerWithAddress } from "@nomicfoundation/hardhat-ethers/signers";

describe("RiskRouter", function () {
  let router: RiskRouter;
  let oracle: RedStoneRiskOracle;
  let owner: SignerWithAddress;
  let agent: SignerWithAddress;
  let nonOwner: SignerWithAddress;

  const PRICE_DECIMALS = 8n;
  const ONE_USD = 10n ** PRICE_DECIMALS;

  // Matches oracle defaults: ETH=$3000, BTC=$60000
  const ETH_PRICE = 3000n * ONE_USD;

  // ETH_ADDRESS: zero address signals "ETH pair"
  const ETH_ADDRESS = ethers.ZeroAddress;
  const BTC_ADDRESS = "0x0000000000000000000000000000000000000001"; // dummy BTC proxy

  const AGENT_ID = 42n;

  beforeEach(async function () {
    [owner, agent, nonOwner] = await ethers.getSigners();

    const Oracle = await ethers.getContractFactory("RedStoneRiskOracle");
    oracle = await Oracle.deploy();
    await oracle.waitForDeployment();

    const Router = await ethers.getContractFactory("RiskRouter");
    router = await Router.deploy(await oracle.getAddress());
    await router.waitForDeployment();
  });

  // ─── Constructor / Config ────────────────────────────────────────────────────

  it("should store the oracle address", async function () {
    expect(await router.oracle()).to.equal(await oracle.getAddress());
  });

  it("should default maxDeviationBps to 500 (5%)", async function () {
    expect(await router.maxDeviationBps()).to.equal(500n);
  });

  it("should revert deployment with zero oracle address", async function () {
    const Router = await ethers.getContractFactory("RiskRouter");
    await expect(Router.deploy(ethers.ZeroAddress))
      .to.be.revertedWithCustomError(router, "ZeroOracle");
  });

  // ─── setMaxDeviationBps ──────────────────────────────────────────────────────

  it("should allow owner to update deviation and emit event", async function () {
    await expect(router.setMaxDeviationBps(300n))
      .to.emit(router, "MaxDeviationUpdated")
      .withArgs(500n, 300n);
    expect(await router.maxDeviationBps()).to.equal(300n);
  });

  it("should revert setMaxDeviationBps from non-owner", async function () {
    await expect(router.connect(nonOwner).setMaxDeviationBps(100n))
      .to.be.revertedWithCustomError(router, "NotOwner");
  });

  // ─── checkRisk ───────────────────────────────────────────────────────────────

  it("should return true for amount within 5% of ETH oracle price", async function () {
    // Within 5%: ETH_PRICE +/- 5% = [2850..3150] * ONE_USD
    const amount = 3050n * ONE_USD; // ~1.67% above
    const { allowed, oraclePrice } = await router.checkRisk.staticCall(
      AGENT_ID, amount, ETH_ADDRESS, BTC_ADDRESS
    );
    expect(allowed).to.be.true;
    expect(oraclePrice).to.equal(ETH_PRICE);
  });

  it("should return false for amount > 5% from oracle price", async function () {
    const amount = 4000n * ONE_USD; // 33% above — too much
    const { allowed } = await router.checkRisk.staticCall(
      AGENT_ID, amount, ETH_ADDRESS, BTC_ADDRESS
    );
    expect(allowed).to.be.false;
  });

  it("should return false for zero trade amount", async function () {
    const { allowed } = await router.checkRisk.staticCall(
      AGENT_ID, 0n, ETH_ADDRESS, BTC_ADDRESS
    );
    expect(allowed).to.be.false;
  });

  it("should emit RiskCheckPassed for valid trade", async function () {
    const amount = 3000n * ONE_USD; // exact oracle price — 0% deviation
    await expect(router.checkRisk(AGENT_ID, amount, ETH_ADDRESS, BTC_ADDRESS))
      .to.emit(router, "RiskCheckPassed")
      .withArgs(AGENT_ID, amount, ETH_PRICE, 0n);
  });

  it("should emit RiskCheckFailed for out-of-range trade", async function () {
    const amount = 5000n * ONE_USD;
    await expect(router.checkRisk(AGENT_ID, amount, ETH_ADDRESS, BTC_ADDRESS))
      .to.emit(router, "RiskCheckFailed");
  });

  it("should use BTC price for non-zero tokenIn", async function () {
    const BTC_PRICE = 60000n * ONE_USD;
    const amount = 60100n * ONE_USD; // ~0.17% above BTC price
    const { allowed, oraclePrice } = await router.checkRisk.staticCall(
      AGENT_ID, amount, BTC_ADDRESS, ETH_ADDRESS
    );
    expect(oraclePrice).to.equal(BTC_PRICE);
    expect(allowed).to.be.true;
  });

  it("should respect updated maxDeviationBps", async function () {
    // Tighten to 1%
    await router.setMaxDeviationBps(100n);
    const amount = 3050n * ONE_USD; // 1.67% deviation — would pass at 5% but fail at 1%
    const { allowed } = await router.checkRisk.staticCall(
      AGENT_ID, amount, ETH_ADDRESS, BTC_ADDRESS
    );
    expect(allowed).to.be.false;
  });

  it("should reflect updated oracle price in risk check", async function () {
    // Owner updates oracle to $3200
    await oracle.setEthPrice(3200n * ONE_USD);
    const amount = 3200n * ONE_USD; // now exact match
    const { allowed } = await router.checkRisk.staticCall(
      AGENT_ID, amount, ETH_ADDRESS, BTC_ADDRESS
    );
    expect(allowed).to.be.true;
  });
});
