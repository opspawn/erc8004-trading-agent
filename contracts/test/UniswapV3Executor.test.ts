import { expect } from "chai";
import { ethers } from "hardhat";
import {
  UniswapV3Executor,
  RiskRouter,
  RedStoneRiskOracle,
} from "../typechain-types";
import { SignerWithAddress } from "@nomicfoundation/hardhat-ethers/signers";

describe("UniswapV3Executor", function () {
  let executor: UniswapV3Executor;
  let router: RiskRouter;
  let oracle: RedStoneRiskOracle;
  let owner: SignerWithAddress;
  let agent: SignerWithAddress;
  let nonWhitelisted: SignerWithAddress;

  const PRICE_DECIMALS = 8n;
  const ONE_USD = 10n ** PRICE_DECIMALS;
  const ETH_PRICE = 3000n * ONE_USD; // $3000 with 8 decimals
  const AGENT_ID = 1n;

  // Token addresses
  const ETH_ADDRESS = ethers.ZeroAddress;
  const BTC_ADDRESS = "0x0000000000000000000000000000000000000001";

  beforeEach(async function () {
    [owner, agent, nonWhitelisted] = await ethers.getSigners();

    // Deploy oracle
    const OracleFactory = await ethers.getContractFactory("RedStoneRiskOracle");
    oracle = await OracleFactory.deploy();
    await oracle.waitForDeployment();

    // Deploy RiskRouter
    const RouterFactory = await ethers.getContractFactory("RiskRouter");
    router = await RouterFactory.deploy(await oracle.getAddress());
    await router.waitForDeployment();

    // Deploy UniswapV3Executor
    const ExecutorFactory = await ethers.getContractFactory("UniswapV3Executor");
    executor = await ExecutorFactory.deploy(await router.getAddress());
    await executor.waitForDeployment();
  });

  // ─── Constructor ──────────────────────────────────────────────────────────

  it("should set owner correctly", async function () {
    expect(await executor.owner()).to.equal(owner.address);
  });

  it("should set risk router correctly", async function () {
    expect(await executor.getRiskRouter()).to.equal(await router.getAddress());
  });

  it("should whitelist owner by default", async function () {
    expect(await executor.whitelisted(owner.address)).to.be.true;
  });

  it("should revert deployment with zero risk router address", async function () {
    const ExecutorFactory = await ethers.getContractFactory("UniswapV3Executor");
    await expect(ExecutorFactory.deploy(ethers.ZeroAddress))
      .to.be.revertedWithCustomError(executor, "ZeroRiskRouter");
  });

  it("should initialize totalSwaps to 0", async function () {
    expect(await executor.totalSwaps()).to.equal(0n);
  });

  // ─── Whitelist Management ─────────────────────────────────────────────────

  it("should allow owner to whitelist an agent", async function () {
    await executor.setWhitelisted(agent.address, true);
    expect(await executor.whitelisted(agent.address)).to.be.true;
  });

  it("should allow owner to remove from whitelist", async function () {
    await executor.setWhitelisted(agent.address, true);
    await executor.setWhitelisted(agent.address, false);
    expect(await executor.whitelisted(agent.address)).to.be.false;
  });

  it("should emit AgentWhitelisted event", async function () {
    await expect(executor.setWhitelisted(agent.address, true))
      .to.emit(executor, "AgentWhitelisted")
      .withArgs(agent.address, true);
  });

  it("should revert setWhitelisted from non-owner", async function () {
    await expect(
      executor.connect(nonWhitelisted).setWhitelisted(agent.address, true)
    ).to.be.revertedWithCustomError(executor, "NotOwner");
  });

  it("should revert executeSwap from non-whitelisted address", async function () {
    await expect(
      executor.connect(nonWhitelisted).executeSwap(
        AGENT_ID, ETH_ADDRESS, BTC_ADDRESS, 1000n, 0n
      )
    ).to.be.revertedWithCustomError(executor, "NotWhitelisted");
  });

  // ─── executeSwap ──────────────────────────────────────────────────────────

  it("should revert executeSwap with zero amount", async function () {
    await expect(
      executor.executeSwap(AGENT_ID, ETH_ADDRESS, BTC_ADDRESS, 0n, 0n)
    ).to.be.revertedWithCustomError(executor, "ZeroAmount");
  });

  it("should block trade when risk check fails (amount far from oracle price)", async function () {
    // Amount wildly different from ETH_PRICE ($3000) — should fail risk check
    const hugeDeviation = ETH_PRICE * 2n; // 100% above oracle price
    await expect(
      executor.executeSwap(AGENT_ID, ETH_ADDRESS, BTC_ADDRESS, hugeDeviation, 0n)
    ).to.be.revertedWithCustomError(executor, "RiskCheckFailed");
  });

  it("should execute swap when amount is within risk tolerance", async function () {
    // Amount within 5% of ETH_PRICE
    const amountIn = ETH_PRICE; // exactly at oracle price = 0% deviation
    const [amountOut, oraclePrice] = await executor.executeSwap.staticCall(
      AGENT_ID, ETH_ADDRESS, BTC_ADDRESS, amountIn, 0n
    );
    expect(amountOut).to.be.greaterThan(0n);
    expect(oraclePrice).to.equal(ETH_PRICE);
  });

  it("should emit TradeExecuted on successful swap", async function () {
    const amountIn = ETH_PRICE;
    await expect(
      executor.executeSwap(AGENT_ID, ETH_ADDRESS, BTC_ADDRESS, amountIn, 0n)
    )
      .to.emit(executor, "TradeExecuted")
      .withArgs(
        AGENT_ID,
        ETH_ADDRESS,
        BTC_ADDRESS,
        amountIn,
        (amountIn * BigInt(1e8)) / ETH_PRICE,
        ETH_PRICE,
        true
      );
  });

  it("should increment totalSwaps on successful swap", async function () {
    const amountIn = ETH_PRICE;
    await executor.executeSwap(AGENT_ID, ETH_ADDRESS, BTC_ADDRESS, amountIn, 0n);
    expect(await executor.totalSwaps()).to.equal(1n);
  });

  it("should update agentSwapCount on successful swap", async function () {
    const amountIn = ETH_PRICE;
    await executor.executeSwap(AGENT_ID, ETH_ADDRESS, BTC_ADDRESS, amountIn, 0n);
    const [count] = await executor.getAgentStats(AGENT_ID);
    expect(count).to.equal(1n);
  });

  it("should update agentSwapVolume on successful swap", async function () {
    const amountIn = ETH_PRICE;
    await executor.executeSwap(AGENT_ID, ETH_ADDRESS, BTC_ADDRESS, amountIn, 0n);
    const [, volume] = await executor.getAgentStats(AGENT_ID);
    expect(volume).to.equal(amountIn);
  });

  it("should revert if amountOut below minAmountOut (slippage protection)", async function () {
    const amountIn = ETH_PRICE;
    const expectedOut = (amountIn * BigInt(1e8)) / ETH_PRICE;
    const tooHighMin = expectedOut + 1n; // 1 unit above expected
    await expect(
      executor.executeSwap(AGENT_ID, ETH_ADDRESS, BTC_ADDRESS, amountIn, tooHighMin)
    ).to.be.revertedWithCustomError(executor, "SlippageExceeded");
  });

  it("should accumulate volume across multiple swaps", async function () {
    const amountIn = ETH_PRICE;
    await executor.executeSwap(AGENT_ID, ETH_ADDRESS, BTC_ADDRESS, amountIn, 0n);
    await executor.executeSwap(AGENT_ID, ETH_ADDRESS, BTC_ADDRESS, amountIn, 0n);
    const [count, volume] = await executor.getAgentStats(AGENT_ID);
    expect(count).to.equal(2n);
    expect(volume).to.equal(amountIn * 2n);
  });

  // ─── simulateSwap ─────────────────────────────────────────────────────────

  it("should return (0, false, 0) for zero amountIn in simulateSwap", async function () {
    const [estimatedOut, riskPassed, oraclePrice] =
      await executor.simulateSwap.staticCall(AGENT_ID, ETH_ADDRESS, BTC_ADDRESS, 0n);
    expect(estimatedOut).to.equal(0n);
    expect(riskPassed).to.be.false;
    expect(oraclePrice).to.equal(0n);
  });

  it("should return risk failed when amount deviates too much in simulateSwap", async function () {
    const badAmount = ETH_PRICE * 3n; // 200% deviation
    const [, riskPassed] = await executor.simulateSwap.staticCall(
      AGENT_ID, ETH_ADDRESS, BTC_ADDRESS, badAmount
    );
    expect(riskPassed).to.be.false;
  });

  it("should return estimated output when risk passes in simulateSwap", async function () {
    const amountIn = ETH_PRICE;
    const [estimatedOut, riskPassed, oraclePrice] =
      await executor.simulateSwap.staticCall(AGENT_ID, ETH_ADDRESS, BTC_ADDRESS, amountIn);
    expect(riskPassed).to.be.true;
    expect(estimatedOut).to.be.greaterThan(0n);
    expect(oraclePrice).to.equal(ETH_PRICE);
  });

  it("should emit SwapSimulated when risk passes", async function () {
    const amountIn = ETH_PRICE;
    await expect(
      executor.simulateSwap(AGENT_ID, ETH_ADDRESS, BTC_ADDRESS, amountIn)
    ).to.emit(executor, "SwapSimulated");
  });

  it("should NOT emit SwapSimulated when risk fails", async function () {
    const badAmount = ETH_PRICE * 3n;
    await expect(
      executor.simulateSwap(AGENT_ID, ETH_ADDRESS, BTC_ADDRESS, badAmount)
    ).to.not.emit(executor, "SwapSimulated");
  });

  // ─── getAgentStats ────────────────────────────────────────────────────────

  it("should return zero stats for new agent", async function () {
    const [count, volume] = await executor.getAgentStats(999n);
    expect(count).to.equal(0n);
    expect(volume).to.equal(0n);
  });

  it("should track stats separately per agent", async function () {
    const amountIn = ETH_PRICE;
    await executor.executeSwap(1n, ETH_ADDRESS, BTC_ADDRESS, amountIn, 0n);
    await executor.executeSwap(1n, ETH_ADDRESS, BTC_ADDRESS, amountIn, 0n);
    await executor.executeSwap(2n, ETH_ADDRESS, BTC_ADDRESS, amountIn, 0n);

    const [count1, volume1] = await executor.getAgentStats(1n);
    const [count2, volume2] = await executor.getAgentStats(2n);

    expect(count1).to.equal(2n);
    expect(volume1).to.equal(amountIn * 2n);
    expect(count2).to.equal(1n);
    expect(volume2).to.equal(amountIn);
  });
});
