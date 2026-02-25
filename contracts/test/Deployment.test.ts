/**
 * Deployment.test.ts
 * Verifies that the deploy.ts script logic produces the correct contract
 * topology when run on the local Hardhat network.
 *
 * We inline the deployment steps from scripts/deploy.ts so that tests
 * do not need to shell out to a child process.
 */

import { expect } from "chai";
import { ethers } from "hardhat";
import { SignerWithAddress } from "@nomicfoundation/hardhat-ethers/signers";
import {
  IdentityRegistry,
  ReputationRegistry,
  ValidationRegistry,
  AgentWallet,
} from "../typechain-types";

describe("Deployment", function () {
  let deployer: SignerWithAddress;
  let identityRegistry: IdentityRegistry;
  let reputationRegistry: ReputationRegistry;
  let validationRegistry: ValidationRegistry;
  let agentWallet: AgentWallet;

  before(async function () {
    [deployer] = await ethers.getSigners();

    // Mirror the steps in contracts/scripts/deploy.ts
    const IdentityRegistryFactory = await ethers.getContractFactory("IdentityRegistry");
    identityRegistry = await IdentityRegistryFactory.deploy();
    await identityRegistry.waitForDeployment();

    const ReputationRegistryFactory = await ethers.getContractFactory("ReputationRegistry");
    reputationRegistry = await ReputationRegistryFactory.deploy(await identityRegistry.getAddress());
    await reputationRegistry.waitForDeployment();

    const ValidationRegistryFactory = await ethers.getContractFactory("ValidationRegistry");
    validationRegistry = await ValidationRegistryFactory.deploy(await identityRegistry.getAddress());
    await validationRegistry.waitForDeployment();

    const AgentWalletFactory = await ethers.getContractFactory("AgentWallet");
    agentWallet = await AgentWalletFactory.deploy(deployer.address);
    await agentWallet.waitForDeployment();
  });

  // ─── Address Sanity ──────────────────────────────────────────────────────────

  it("should deploy IdentityRegistry to a valid address", async function () {
    const addr = await identityRegistry.getAddress();
    expect(addr).to.match(/^0x[0-9a-fA-F]{40}$/);
  });

  it("should deploy ReputationRegistry to a valid address", async function () {
    const addr = await reputationRegistry.getAddress();
    expect(addr).to.match(/^0x[0-9a-fA-F]{40}$/);
  });

  it("should deploy ValidationRegistry to a valid address", async function () {
    const addr = await validationRegistry.getAddress();
    expect(addr).to.match(/^0x[0-9a-fA-F]{40}$/);
  });

  it("should deploy AgentWallet to a valid address", async function () {
    const addr = await agentWallet.getAddress();
    expect(addr).to.match(/^0x[0-9a-fA-F]{40}$/);
  });

  // ─── Cross-Contract Wiring ───────────────────────────────────────────────────

  it("ReputationRegistry should reference the correct IdentityRegistry", async function () {
    const identityAddr = await identityRegistry.getAddress();
    const linkedAddr = await reputationRegistry.identityRegistry();
    expect(linkedAddr).to.equal(identityAddr);
  });

  it("ValidationRegistry should reference the correct IdentityRegistry", async function () {
    const identityAddr = await identityRegistry.getAddress();
    const linkedAddr = await validationRegistry.identityRegistry();
    expect(linkedAddr).to.equal(identityAddr);
  });
});
