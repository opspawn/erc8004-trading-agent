import { expect } from "chai";
import { ethers } from "hardhat";
import { IdentityRegistry, TradingFeed } from "../typechain-types";
import { SignerWithAddress } from "@nomicfoundation/hardhat-ethers/signers";

describe("TradingFeed", function () {
  let idRegistry: IdentityRegistry;
  let feed: TradingFeed;
  let deployer: SignerWithAddress;
  let agent1: SignerWithAddress;
  let agent2: SignerWithAddress;
  let stranger: SignerWithAddress;

  const AGENT_URI = "ipfs://QmTradingFeed/agent.json";
  const SYMBOL_BTC = "BTC/USD";
  const SYMBOL_ETH = "ETH/USD";
  const DEFAULT_THRESHOLD = -100000n; // -$1000.00 in cents

  // TradeAction enum values
  const NONE = 0n;
  const BUY = 1n;
  const SELL = 2n;
  const HOLD = 3n;

  beforeEach(async function () {
    [deployer, agent1, agent2, stranger] = await ethers.getSigners();

    const IdentityRegistry = await ethers.getContractFactory("IdentityRegistry");
    idRegistry = await IdentityRegistry.deploy();
    await idRegistry.waitForDeployment();

    const TradingFeed = await ethers.getContractFactory("TradingFeed");
    feed = await TradingFeed.deploy(
      await idRegistry.getAddress(),
      DEFAULT_THRESHOLD
    );
    await feed.waitForDeployment();

    // Register agents
    await idRegistry.connect(agent1).mint(AGENT_URI);
    await idRegistry.connect(agent2).mint(AGENT_URI);
  });

  // ─── Deployment ──────────────────────────────────────────────────────────────

  it("should deploy with correct identityRegistry address", async function () {
    expect(await feed.identityRegistry()).to.equal(await idRegistry.getAddress());
  });

  it("should deploy with correct drawdown threshold", async function () {
    expect(await feed.drawdownThreshold()).to.equal(DEFAULT_THRESHOLD);
  });

  it("should start with feedSequence = 0", async function () {
    expect(await feed.feedSequence()).to.equal(0n);
  });

  it("should start with circuit breaker not tripped", async function () {
    expect(await feed.circuitBreakerTripped()).to.be.false;
  });

  it("should start with zero votes", async function () {
    expect(await feed.voteCount()).to.equal(0n);
  });

  it("should start with zero consensus records", async function () {
    expect(await feed.consensusCount()).to.equal(0n);
  });

  it("should start with zero trade records", async function () {
    expect(await feed.tradeCount()).to.equal(0n);
  });

  it("should start with zero reputation updates", async function () {
    expect(await feed.reputationUpdateCount()).to.equal(0n);
  });

  // ─── castVote ─────────────────────────────────────────────────────────────

  it("should accept a valid BUY vote from registered agent", async function () {
    await expect(
      feed.connect(agent1).castVote(1n, SYMBOL_BTC, BUY, 7500n, 1000n)
    ).to.emit(feed, "AgentVoteCast");
  });

  it("should accept a valid SELL vote from registered agent", async function () {
    await expect(
      feed.connect(agent1).castVote(1n, SYMBOL_BTC, SELL, 6000n, 500n)
    ).to.emit(feed, "AgentVoteCast");
  });

  it("should accept a valid HOLD vote from registered agent", async function () {
    await expect(
      feed.connect(agent1).castVote(1n, SYMBOL_ETH, HOLD, 5000n, 0n)
    ).to.emit(feed, "AgentVoteCast");
  });

  it("should increment feedSequence on castVote", async function () {
    await feed.connect(agent1).castVote(1n, SYMBOL_BTC, BUY, 7500n, 1000n);
    expect(await feed.feedSequence()).to.equal(1n);
  });

  it("should increment voteCount on castVote", async function () {
    await feed.connect(agent1).castVote(1n, SYMBOL_BTC, BUY, 7500n, 1000n);
    expect(await feed.voteCount()).to.equal(1n);
  });

  it("should store vote data correctly", async function () {
    await feed.connect(agent1).castVote(1n, SYMBOL_BTC, BUY, 8000n, 2000n);
    const vote = await feed.getVote(0);
    expect(vote.agentId).to.equal(1n);
    expect(vote.symbol).to.equal(SYMBOL_BTC);
    expect(vote.action).to.equal(BUY);
    expect(vote.confidence).to.equal(8000n);
    expect(vote.positionSize).to.equal(2000n);
  });

  it("should revert castVote for unregistered agent", async function () {
    await expect(
      feed.connect(stranger).castVote(999n, SYMBOL_BTC, BUY, 7500n, 1000n)
    ).to.be.revertedWithCustomError(feed, "AgentNotRegistered");
  });

  it("should revert castVote with confidence > 10000", async function () {
    await expect(
      feed.connect(agent1).castVote(1n, SYMBOL_BTC, BUY, 10001n, 1000n)
    ).to.be.revertedWithCustomError(feed, "InvalidConfidence");
  });

  it("should accept confidence of exactly 10000", async function () {
    await expect(
      feed.connect(agent1).castVote(1n, SYMBOL_BTC, BUY, 10000n, 1000n)
    ).to.not.be.reverted;
  });

  it("should revert castVote with empty symbol", async function () {
    await expect(
      feed.connect(agent1).castVote(1n, "", BUY, 7500n, 1000n)
    ).to.be.revertedWithCustomError(feed, "EmptySymbol");
  });

  it("should allow second agent to vote independently", async function () {
    await feed.connect(agent1).castVote(1n, SYMBOL_BTC, BUY, 7500n, 1000n);
    await feed.connect(agent2).castVote(2n, SYMBOL_BTC, SELL, 6000n, 800n);
    expect(await feed.voteCount()).to.equal(2n);
    expect(await feed.feedSequence()).to.equal(2n);
  });

  // ─── recordConsensus ──────────────────────────────────────────────────────

  it("should record consensus and emit event", async function () {
    await expect(
      feed.connect(deployer).recordConsensus(SYMBOL_BTC, BUY, 7500n, 3n)
    ).to.emit(feed, "ConsensusReached");
  });

  it("should increment consensusCount after recording", async function () {
    await feed.connect(deployer).recordConsensus(SYMBOL_BTC, BUY, 7500n, 3n);
    expect(await feed.consensusCount()).to.equal(1n);
  });

  it("should store consensus data correctly", async function () {
    await feed.connect(deployer).recordConsensus(SYMBOL_ETH, SELL, 6667n, 2n);
    const rec = await feed.getConsensus(0);
    expect(rec.symbol).to.equal(SYMBOL_ETH);
    expect(rec.decision).to.equal(SELL);
    expect(rec.agreementRate).to.equal(6667n);
    expect(rec.participantCount).to.equal(2n);
  });

  it("should revert recordConsensus with agreementRate > 10000", async function () {
    await expect(
      feed.connect(deployer).recordConsensus(SYMBOL_BTC, BUY, 10001n, 3n)
    ).to.be.revertedWithCustomError(feed, "InvalidAgreementRate");
  });

  it("should revert recordConsensus with empty symbol", async function () {
    await expect(
      feed.connect(deployer).recordConsensus("", BUY, 7500n, 3n)
    ).to.be.revertedWithCustomError(feed, "EmptySymbol");
  });

  it("should accept agreementRate of exactly 10000", async function () {
    await expect(
      feed.connect(deployer).recordConsensus(SYMBOL_BTC, BUY, 10000n, 3n)
    ).to.not.be.reverted;
  });

  // ─── recordTrade ──────────────────────────────────────────────────────────

  it("should record a trade and emit TradeExecuted event", async function () {
    await expect(
      feed.connect(agent1).recordTrade(1n, SYMBOL_BTC, BUY, 1000000000000000000n, 4350000n, 1500n)
    ).to.emit(feed, "TradeExecuted");
  });

  it("should increment tradeCount after recording", async function () {
    await feed.connect(agent1).recordTrade(1n, SYMBOL_BTC, BUY, 1000000000000000000n, 4350000n, 1500n);
    expect(await feed.tradeCount()).to.equal(1n);
  });

  it("should store trade data correctly", async function () {
    await feed.connect(agent1).recordTrade(1n, SYMBOL_ETH, SELL, 500000000000000000n, 228000n, -200n);
    const trade = await feed.getTrade(0);
    expect(trade.agentId).to.equal(1n);
    expect(trade.symbol).to.equal(SYMBOL_ETH);
    expect(trade.action).to.equal(SELL);
    expect(trade.quantity).to.equal(500000000000000000n);
    expect(trade.price).to.equal(228000n);
    expect(trade.pnlDelta).to.equal(-200n);
  });

  it("should revert recordTrade for unregistered agent", async function () {
    await expect(
      feed.connect(stranger).recordTrade(999n, SYMBOL_BTC, BUY, 1000000n, 4350000n, 0n)
    ).to.be.revertedWithCustomError(feed, "AgentNotRegistered");
  });

  it("should revert recordTrade with zero quantity", async function () {
    await expect(
      feed.connect(agent1).recordTrade(1n, SYMBOL_BTC, BUY, 0n, 4350000n, 0n)
    ).to.be.revertedWithCustomError(feed, "ZeroQuantity");
  });

  it("should revert recordTrade with empty symbol", async function () {
    await expect(
      feed.connect(agent1).recordTrade(1n, "", BUY, 1000n, 4350000n, 0n)
    ).to.be.revertedWithCustomError(feed, "EmptySymbol");
  });

  // ─── Circuit Breaker ──────────────────────────────────────────────────────

  it("should trip circuit breaker on large loss exceeding threshold", async function () {
    const bigLoss = DEFAULT_THRESHOLD - 1n; // below threshold
    await feed.connect(agent1).recordTrade(1n, SYMBOL_BTC, SELL, 1000n, 4350000n, bigLoss);
    expect(await feed.circuitBreakerTripped()).to.be.true;
  });

  it("should emit CircuitBreakerTripped event on large loss", async function () {
    const bigLoss = DEFAULT_THRESHOLD - 1n;
    await expect(
      feed.connect(agent1).recordTrade(1n, SYMBOL_BTC, SELL, 1000n, 4350000n, bigLoss)
    ).to.emit(feed, "CircuitBreakerTripped");
  });

  it("should NOT trip circuit breaker on normal loss", async function () {
    await feed.connect(agent1).recordTrade(1n, SYMBOL_BTC, SELL, 1000n, 4350000n, -500n);
    expect(await feed.circuitBreakerTripped()).to.be.false;
  });

  it("should reset circuit breaker via resetCircuitBreaker()", async function () {
    const bigLoss = DEFAULT_THRESHOLD - 1n;
    await feed.connect(agent1).recordTrade(1n, SYMBOL_BTC, SELL, 1000n, 4350000n, bigLoss);
    await feed.resetCircuitBreaker();
    expect(await feed.circuitBreakerTripped()).to.be.false;
  });

  it("should emit CircuitBreakerReset event on reset", async function () {
    await expect(feed.resetCircuitBreaker()).to.emit(feed, "CircuitBreakerReset");
  });

  it("should update drawdown threshold via updateDrawdownThreshold()", async function () {
    await feed.updateDrawdownThreshold(-500000n);
    expect(await feed.drawdownThreshold()).to.equal(-500000n);
  });

  // ─── recordReputationUpdate ───────────────────────────────────────────────

  it("should record reputation update and emit event", async function () {
    await expect(
      feed.connect(deployer).recordReputationUpdate(1n, 7800n, 7900n)
    ).to.emit(feed, "ReputationUpdated");
  });

  it("should increment reputationUpdateCount after recording", async function () {
    await feed.connect(deployer).recordReputationUpdate(1n, 7800n, 7900n);
    expect(await feed.reputationUpdateCount()).to.equal(1n);
  });

  it("should store reputation update data correctly", async function () {
    await feed.connect(deployer).recordReputationUpdate(1n, 6910n, 7050n);
    const update = await feed.getReputationUpdate(0);
    expect(update.agentId).to.equal(1n);
    expect(update.oldScore).to.equal(6910n);
    expect(update.newScore).to.equal(7050n);
  });

  it("should revert recordReputationUpdate for unregistered agent", async function () {
    await expect(
      feed.connect(deployer).recordReputationUpdate(999n, 7800n, 7900n)
    ).to.be.revertedWithCustomError(feed, "AgentNotRegistered");
  });

  // ─── Paginated Getters ────────────────────────────────────────────────────

  it("should return empty array for getVotes with offset beyond length", async function () {
    const votes = await feed.getVotes(100n, 10n);
    expect(votes.length).to.equal(0);
  });

  it("should paginate votes correctly", async function () {
    // Cast 3 votes
    await feed.connect(agent1).castVote(1n, SYMBOL_BTC, BUY, 7500n, 1000n);
    await feed.connect(agent1).castVote(1n, SYMBOL_ETH, SELL, 6000n, 800n);
    await feed.connect(agent2).castVote(2n, SYMBOL_BTC, HOLD, 5000n, 0n);

    const page1 = await feed.getVotes(0n, 2n);
    expect(page1.length).to.equal(2);
    const page2 = await feed.getVotes(2n, 2n);
    expect(page2.length).to.equal(1);
  });

  it("should return empty array for getTrades with offset beyond length", async function () {
    const trades = await feed.getTrades(100n, 10n);
    expect(trades.length).to.equal(0);
  });

  it("should paginate trades correctly", async function () {
    await feed.connect(agent1).recordTrade(1n, SYMBOL_BTC, BUY, 1000n, 4350000n, 100n);
    await feed.connect(agent1).recordTrade(1n, SYMBOL_ETH, SELL, 500n, 228000n, -50n);
    await feed.connect(agent2).recordTrade(2n, SYMBOL_BTC, BUY, 2000n, 4360000n, 200n);

    const page = await feed.getTrades(1n, 2n);
    expect(page.length).to.equal(2);
    expect(page[0].agentId).to.equal(1n);
    expect(page[1].agentId).to.equal(2n);
  });

  // ─── feedSequence Monotonicity ────────────────────────────────────────────

  it("should increment feedSequence for each event type", async function () {
    await feed.connect(agent1).castVote(1n, SYMBOL_BTC, BUY, 7500n, 1000n);
    await feed.connect(deployer).recordConsensus(SYMBOL_BTC, BUY, 7500n, 3n);
    await feed.connect(agent1).recordTrade(1n, SYMBOL_BTC, BUY, 1000n, 4350000n, 100n);
    await feed.connect(deployer).recordReputationUpdate(1n, 7800n, 7900n);
    expect(await feed.feedSequence()).to.equal(4n);
  });
});
