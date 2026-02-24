import { expect } from "chai";
import { ethers } from "hardhat";
import { IdentityRegistry } from "../typechain-types";
import { SignerWithAddress } from "@nomicfoundation/hardhat-ethers/signers";

describe("IdentityRegistry", function () {
  let registry: IdentityRegistry;
  let owner: SignerWithAddress;
  let agent1: SignerWithAddress;
  let agent2: SignerWithAddress;

  const METADATA_URI = "ipfs://QmTestAgentMetadata/agent.json";
  const METADATA_URI_2 = "ipfs://QmTestAgentMetadata2/agent.json";

  beforeEach(async function () {
    [owner, agent1, agent2] = await ethers.getSigners();
    const IdentityRegistry = await ethers.getContractFactory("IdentityRegistry");
    registry = await IdentityRegistry.deploy();
    await registry.waitForDeployment();
  });

  // ─── ERC-721 Metadata ───────────────────────────────────────────────────────

  it("should have correct name and symbol", async function () {
    expect(await registry.name()).to.equal("ERC8004 Agent Identity");
    expect(await registry.symbol()).to.equal("AGENT");
  });

  // ─── Agent Registration ──────────────────────────────────────────────────────

  it("should mint an agent identity with agentId = 1", async function () {
    const tx = await registry.connect(agent1).mint(METADATA_URI);
    await tx.wait();

    const agentId = await registry.agentIdOf(agent1.address);
    expect(agentId).to.equal(1n);
  });

  it("should assign sequential IDs to multiple agents", async function () {
    await registry.connect(agent1).mint(METADATA_URI);
    await registry.connect(agent2).mint(METADATA_URI_2);

    expect(await registry.agentIdOf(agent1.address)).to.equal(1n);
    expect(await registry.agentIdOf(agent2.address)).to.equal(2n);
  });

  it("should emit AgentRegistered event on mint", async function () {
    await expect(registry.connect(agent1).mint(METADATA_URI))
      .to.emit(registry, "AgentRegistered")
      .withArgs(1n, agent1.address, METADATA_URI);
  });

  it("should set the token URI on mint", async function () {
    await registry.connect(agent1).mint(METADATA_URI);
    expect(await registry.tokenURI(1n)).to.equal(METADATA_URI);
  });

  it("should revert on double registration", async function () {
    await registry.connect(agent1).mint(METADATA_URI);
    await expect(registry.connect(agent1).mint(METADATA_URI))
      .to.be.revertedWithCustomError(registry, "AlreadyRegistered")
      .withArgs(agent1.address, 1n);
  });

  it("should correctly report isRegistered", async function () {
    expect(await registry.isRegistered(agent1.address)).to.be.false;
    await registry.connect(agent1).mint(METADATA_URI);
    expect(await registry.isRegistered(agent1.address)).to.be.true;
  });

  it("should report correct totalAgents", async function () {
    expect(await registry.totalAgents()).to.equal(0n);
    await registry.connect(agent1).mint(METADATA_URI);
    expect(await registry.totalAgents()).to.equal(1n);
    await registry.connect(agent2).mint(METADATA_URI_2);
    expect(await registry.totalAgents()).to.equal(2n);
  });

  // ─── Agent URI Update ────────────────────────────────────────────────────────

  it("should allow the owner to update agent URI", async function () {
    await registry.connect(agent1).mint(METADATA_URI);
    const newURI = "ipfs://QmNewMetadata/agent.json";
    await expect(registry.connect(agent1).setAgentURI(1n, newURI))
      .to.emit(registry, "AgentURIUpdated")
      .withArgs(1n, newURI);
    expect(await registry.tokenURI(1n)).to.equal(newURI);
  });

  it("should revert URI update from non-owner", async function () {
    await registry.connect(agent1).mint(METADATA_URI);
    await expect(
      registry.connect(agent2).setAgentURI(1n, "ipfs://Qm/evil.json")
    ).to.be.revertedWithCustomError(registry, "NotTokenOwner");
  });

  it("should revert URI update for invalid agentId", async function () {
    await expect(
      registry.connect(agent1).setAgentURI(99n, "ipfs://Qm/test.json")
    ).to.be.revertedWithCustomError(registry, "InvalidAgentId");
  });

  // ─── Agent DID ───────────────────────────────────────────────────────────────

  it("should return a valid DID string", async function () {
    await registry.connect(agent1).mint(METADATA_URI);
    const did = await registry.agentDID(1n);
    // Format: eip155:{chainId}:{contract}:{agentId}
    expect(did).to.match(/^eip155:\d+:0x[0-9a-f]{40}:1$/);
  });

  it("should revert agentDID for invalid agentId", async function () {
    await expect(registry.agentDID(99n))
      .to.be.revertedWithCustomError(registry, "InvalidAgentId");
  });

  // ─── ERC-721 Transfers ───────────────────────────────────────────────────────

  it("should allow transfer of agent token", async function () {
    await registry.connect(agent1).mint(METADATA_URI);
    await registry.connect(agent1).transferFrom(agent1.address, agent2.address, 1n);
    expect(await registry.ownerOf(1n)).to.equal(agent2.address);
  });
});
