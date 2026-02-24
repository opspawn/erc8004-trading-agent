import { expect } from "chai";
import { ethers } from "hardhat";
import { IdentityRegistry, ValidationRegistry } from "../typechain-types";
import { SignerWithAddress } from "@nomicfoundation/hardhat-ethers/signers";

describe("ValidationRegistry", function () {
  let idRegistry: IdentityRegistry;
  let valRegistry: ValidationRegistry;
  let deployer: SignerWithAddress;
  let agentOwner: SignerWithAddress;
  let validator: SignerWithAddress;
  let otherUser: SignerWithAddress;

  const AGENT_URI = "ipfs://QmAgentMetadata/agent.json";
  const DATA_URI = "ipfs://QmTradeData/trade_2024_001.json";
  const DATA_HASH = ethers.keccak256(ethers.toUtf8Bytes("trade outcome data"));
  const COMMENT_URI = "ipfs://QmValidatorComment/comment.json";

  beforeEach(async function () {
    [deployer, agentOwner, validator, otherUser] = await ethers.getSigners();

    const IdentityRegistry = await ethers.getContractFactory("IdentityRegistry");
    idRegistry = await IdentityRegistry.deploy();
    await idRegistry.waitForDeployment();

    const ValidationRegistry = await ethers.getContractFactory("ValidationRegistry");
    valRegistry = await ValidationRegistry.deploy(await idRegistry.getAddress());
    await valRegistry.waitForDeployment();

    // Register agent
    await idRegistry.connect(agentOwner).mint(AGENT_URI);
  });

  // ─── Validation Request ──────────────────────────────────────────────────

  it("should create a validation request with requestId = 1", async function () {
    await expect(
      valRegistry.connect(agentOwner).validationRequest(1n, DATA_URI, DATA_HASH)
    ).to.emit(valRegistry, "ValidationRequested")
     .withArgs(1n, 1n, agentOwner.address, DATA_URI, DATA_HASH);
  });

  it("should assign sequential requestIds", async function () {
    await valRegistry.connect(agentOwner).validationRequest(1n, DATA_URI, DATA_HASH);
    await valRegistry.connect(agentOwner).validationRequest(1n, DATA_URI, DATA_HASH);
    expect(await valRegistry.totalRequests()).to.equal(2n);
  });

  it("should store request data correctly", async function () {
    await valRegistry.connect(agentOwner).validationRequest(1n, DATA_URI, DATA_HASH);
    const req = await valRegistry.requests(1n);
    expect(req.agentId).to.equal(1n);
    expect(req.requester).to.equal(agentOwner.address);
    expect(req.dataURI).to.equal(DATA_URI);
    expect(req.dataHash).to.equal(DATA_HASH);
    expect(req.status).to.equal(0); // Pending
  });

  it("should revert request for unregistered agent", async function () {
    await expect(
      valRegistry.connect(agentOwner).validationRequest(99n, DATA_URI, DATA_HASH)
    ).to.be.revertedWithCustomError(valRegistry, "AgentNotRegistered");
  });

  it("should revert request with zero data hash", async function () {
    await expect(
      valRegistry.connect(agentOwner).validationRequest(1n, DATA_URI, ethers.ZeroHash)
    ).to.be.revertedWithCustomError(valRegistry, "InvalidDataHash");
  });

  // ─── Validation Submission ───────────────────────────────────────────────

  it("should accept a validation response", async function () {
    await valRegistry.connect(agentOwner).validationRequest(1n, DATA_URI, DATA_HASH);
    await expect(
      valRegistry.connect(validator).submitValidation(1n, 85, COMMENT_URI)
    ).to.emit(valRegistry, "ValidationSubmitted")
     .withArgs(1n, 1n, validator.address, 85);
  });

  it("should store validation result correctly", async function () {
    await valRegistry.connect(agentOwner).validationRequest(1n, DATA_URI, DATA_HASH);
    await valRegistry.connect(validator).submitValidation(1n, 85, COMMENT_URI);

    const [validatorAddr, response, timestamp] = await valRegistry.getValidationResult(1n);
    expect(validatorAddr).to.equal(validator.address);
    expect(response).to.equal(85);
    expect(timestamp).to.be.greaterThan(0n);
  });

  it("should update request status to Validated", async function () {
    await valRegistry.connect(agentOwner).validationRequest(1n, DATA_URI, DATA_HASH);
    await valRegistry.connect(validator).submitValidation(1n, 85, COMMENT_URI);
    const req = await valRegistry.requests(1n);
    expect(req.status).to.equal(1); // Validated
  });

  it("should revert on double validation", async function () {
    await valRegistry.connect(agentOwner).validationRequest(1n, DATA_URI, DATA_HASH);
    await valRegistry.connect(validator).submitValidation(1n, 85, COMMENT_URI);
    await expect(
      valRegistry.connect(validator).submitValidation(1n, 90, COMMENT_URI)
    ).to.be.revertedWithCustomError(valRegistry, "AlreadyValidated");
  });

  it("should revert for invalid response score > 100", async function () {
    await valRegistry.connect(agentOwner).validationRequest(1n, DATA_URI, DATA_HASH);
    await expect(
      valRegistry.connect(validator).submitValidation(1n, 101, COMMENT_URI)
    ).to.be.revertedWithCustomError(valRegistry, "InvalidResponse");
  });

  it("should accept score of 0", async function () {
    await valRegistry.connect(agentOwner).validationRequest(1n, DATA_URI, DATA_HASH);
    await expect(
      valRegistry.connect(validator).submitValidation(1n, 0, "")
    ).to.not.be.reverted;
  });

  it("should accept score of 100", async function () {
    await valRegistry.connect(agentOwner).validationRequest(1n, DATA_URI, DATA_HASH);
    await expect(
      valRegistry.connect(validator).submitValidation(1n, 100, "")
    ).to.not.be.reverted;
  });

  it("should revert getValidationResult for non-existent request", async function () {
    await expect(
      valRegistry.getValidationResult(99n)
    ).to.be.revertedWithCustomError(valRegistry, "RequestNotFound");
  });

  // ─── Dispute ─────────────────────────────────────────────────────────────

  it("should allow requester to dispute a validation", async function () {
    await valRegistry.connect(agentOwner).validationRequest(1n, DATA_URI, DATA_HASH);
    await valRegistry.connect(validator).submitValidation(1n, 20, "");
    await expect(
      valRegistry.connect(agentOwner).disputeValidation(1n)
    ).to.emit(valRegistry, "RequestDisputed")
     .withArgs(1n, agentOwner.address);
    const req = await valRegistry.requests(1n);
    expect(req.status).to.equal(2); // Disputed
  });

  it("should revert dispute from non-requester", async function () {
    await valRegistry.connect(agentOwner).validationRequest(1n, DATA_URI, DATA_HASH);
    await valRegistry.connect(validator).submitValidation(1n, 20, "");
    await expect(
      valRegistry.connect(otherUser).disputeValidation(1n)
    ).to.be.revertedWithCustomError(valRegistry, "NotRequester");
  });

  // ─── Agent Requests ──────────────────────────────────────────────────────

  it("should track all requests per agent", async function () {
    await valRegistry.connect(agentOwner).validationRequest(1n, DATA_URI, DATA_HASH);
    await valRegistry.connect(agentOwner).validationRequest(1n, DATA_URI, DATA_HASH);
    const requests = await valRegistry.getAgentRequests(1n);
    expect(requests.length).to.equal(2);
    expect(requests[0]).to.equal(1n);
    expect(requests[1]).to.equal(2n);
  });
});
