import { expect } from "chai";
import { ethers } from "hardhat";
import { IdentityRegistry, ReputationRegistry } from "../typechain-types";
import { SignerWithAddress } from "@nomicfoundation/hardhat-ethers/signers";

describe("ReputationRegistry", function () {
  let idRegistry: IdentityRegistry;
  let repRegistry: ReputationRegistry;
  let deployer: SignerWithAddress;
  let agentOwner: SignerWithAddress;
  let client1: SignerWithAddress;
  let client2: SignerWithAddress;

  const AGENT_URI = "ipfs://QmAgentMetadata/agent.json";
  const TAG1 = ethers.encodeBytes32String("accuracy");
  const TAG2 = ethers.encodeBytes32String("speed");
  const ENDPOINT = "https://agent.opspawn.com/v1";
  const FILE_HASH = ethers.keccak256(ethers.toUtf8Bytes("trade_result_data"));

  beforeEach(async function () {
    [deployer, agentOwner, client1, client2] = await ethers.getSigners();

    const IdentityRegistry = await ethers.getContractFactory("IdentityRegistry");
    idRegistry = await IdentityRegistry.deploy();
    await idRegistry.waitForDeployment();

    const ReputationRegistry = await ethers.getContractFactory("ReputationRegistry");
    repRegistry = await ReputationRegistry.deploy(await idRegistry.getAddress());
    await repRegistry.waitForDeployment();

    // Register a test agent
    await idRegistry.connect(agentOwner).mint(AGENT_URI);
  });

  // ─── Self-Feedback Prevention ─────────────────────────────────────────────

  it("should prevent self-feedback by agent owner", async function () {
    await expect(
      repRegistry.connect(agentOwner).giveFeedback(
        1n, 850n, 2, TAG1, TAG2, ENDPOINT, FILE_HASH
      )
    ).to.be.revertedWithCustomError(repRegistry, "SelfFeedbackNotAllowed");
  });

  // ─── Feedback Submission ──────────────────────────────────────────────────

  it("should accept feedback from a client", async function () {
    await expect(
      repRegistry.connect(client1).giveFeedback(
        1n, 850n, 2, TAG1, TAG2, ENDPOINT, FILE_HASH
      )
    ).to.emit(repRegistry, "FeedbackGiven")
     .withArgs(1n, client1.address, 850n, 2, TAG1, TAG2);
  });

  it("should store feedback correctly", async function () {
    await repRegistry.connect(client1).giveFeedback(
      1n, 850n, 2, TAG1, TAG2, ENDPOINT, FILE_HASH
    );
    const fb = await repRegistry.getFeedback(1n, client1.address);
    expect(fb.client).to.equal(client1.address);
    expect(fb.value).to.equal(850n);
    expect(fb.decimals).to.equal(2);
    expect(fb.tag1).to.equal(TAG1);
    expect(fb.tag2).to.equal(TAG2);
    expect(fb.endpointURI).to.equal(ENDPOINT);
    expect(fb.fileHash).to.equal(FILE_HASH);
  });

  it("should allow multiple clients to give feedback", async function () {
    await repRegistry.connect(client1).giveFeedback(
      1n, 850n, 2, TAG1, TAG2, ENDPOINT, FILE_HASH
    );
    await repRegistry.connect(client2).giveFeedback(
      1n, 700n, 2, TAG1, TAG2, ENDPOINT, FILE_HASH
    );
    expect(await repRegistry.feedbackCount(1n)).to.equal(2n);
  });

  it("should update feedback when the same client submits again", async function () {
    await repRegistry.connect(client1).giveFeedback(
      1n, 850n, 2, TAG1, TAG2, ENDPOINT, FILE_HASH
    );
    await expect(
      repRegistry.connect(client1).giveFeedback(
        1n, 950n, 2, TAG1, TAG2, ENDPOINT, FILE_HASH
      )
    ).to.emit(repRegistry, "FeedbackUpdated")
     .withArgs(1n, client1.address, 950n);

    const fb = await repRegistry.getFeedback(1n, client1.address);
    expect(fb.value).to.equal(950n);
    // Count should stay at 1
    expect(await repRegistry.feedbackCount(1n)).to.equal(1n);
  });

  it("should revert getFeedback when no feedback exists", async function () {
    await expect(repRegistry.getFeedback(1n, client1.address))
      .to.be.revertedWithCustomError(repRegistry, "NoFeedbackFound");
  });

  it("should revert feedback for unregistered agent", async function () {
    await expect(
      repRegistry.connect(client1).giveFeedback(
        99n, 850n, 2, TAG1, TAG2, ENDPOINT, FILE_HASH
      )
    ).to.be.revertedWithCustomError(repRegistry, "AgentNotRegistered");
  });

  it("should revert on invalid decimals (> 18)", async function () {
    await expect(
      repRegistry.connect(client1).giveFeedback(
        1n, 850n, 19, TAG1, TAG2, ENDPOINT, FILE_HASH
      )
    ).to.be.revertedWithCustomError(repRegistry, "InvalidScore");
  });

  // ─── hasFeedback ─────────────────────────────────────────────────────────

  it("should report hasFeedback correctly", async function () {
    expect(await repRegistry.hasFeedback(1n, client1.address)).to.be.false;
    await repRegistry.connect(client1).giveFeedback(
      1n, 850n, 2, TAG1, TAG2, ENDPOINT, FILE_HASH
    );
    expect(await repRegistry.hasFeedback(1n, client1.address)).to.be.true;
  });

  // ─── Aggregate Score ─────────────────────────────────────────────────────

  it("should return zero aggregate score with no feedback", async function () {
    const [score, count] = await repRegistry.getAggregateScore(1n);
    expect(score).to.equal(0n);
    expect(count).to.equal(0n);
  });

  it("should compute correct aggregate score (single feedback)", async function () {
    // 850 with 2 decimals = 8.50; normalized to 2dp = 850
    await repRegistry.connect(client1).giveFeedback(
      1n, 850n, 2, TAG1, TAG2, ENDPOINT, FILE_HASH
    );
    const [score, count] = await repRegistry.getAggregateScore(1n);
    expect(score).to.equal(850n);
    expect(count).to.equal(1n);
  });

  it("should compute correct average with multiple feedbacks", async function () {
    // 800 + 600 = 1400, avg = 700 (all 2dp)
    await repRegistry.connect(client1).giveFeedback(
      1n, 800n, 2, TAG1, TAG2, ENDPOINT, FILE_HASH
    );
    await repRegistry.connect(client2).giveFeedback(
      1n, 600n, 2, TAG1, TAG2, ENDPOINT, FILE_HASH
    );
    const [score, count] = await repRegistry.getAggregateScore(1n);
    expect(score).to.equal(700n);
    expect(count).to.equal(2n);
  });

  // ─── getAllFeedback ───────────────────────────────────────────────────────

  it("should return all feedback entries", async function () {
    await repRegistry.connect(client1).giveFeedback(
      1n, 850n, 2, TAG1, TAG2, ENDPOINT, FILE_HASH
    );
    await repRegistry.connect(client2).giveFeedback(
      1n, 700n, 2, TAG1, TAG2, ENDPOINT, FILE_HASH
    );
    const all = await repRegistry.getAllFeedback(1n);
    expect(all.length).to.equal(2);
  });
});
