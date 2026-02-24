import { expect } from "chai";
import { ethers } from "hardhat";
import { AgentWallet } from "../typechain-types";
import { SignerWithAddress } from "@nomicfoundation/hardhat-ethers/signers";

// A fixed test private key for EIP-712 signing (hardhat account #0)
const TEST_PRIVATE_KEY = "0xac0974bec39a17e36ba4a6b4d238ff944bacb478cbed5efcae784d7bf4f2ff80";

describe("AgentWallet", function () {
  let wallet: AgentWallet;
  let ownerWallet: ethers.Wallet;
  let owner: SignerWithAddress;
  let other: SignerWithAddress;
  let target: SignerWithAddress;

  const MAGIC_VALUE = "0x1626ba7e";

  beforeEach(async function () {
    [owner, other, target] = await ethers.getSigners();

    // ownerWallet is a local ethers.Wallet (same address as hardhat account #0)
    ownerWallet = new ethers.Wallet(TEST_PRIVATE_KEY, ethers.provider);

    const AgentWallet = await ethers.getContractFactory("AgentWallet");
    wallet = await AgentWallet.connect(ownerWallet).deploy(ownerWallet.address);
    await wallet.waitForDeployment();
  });

  // ─── Constructor ─────────────────────────────────────────────────────────

  it("should set owner correctly", async function () {
    expect(await wallet.owner()).to.equal(ownerWallet.address);
  });

  it("should revert deployment with zero address owner", async function () {
    const AgentWallet = await ethers.getContractFactory("AgentWallet");
    await expect(AgentWallet.deploy(ethers.ZeroAddress))
      .to.be.revertedWithCustomError(wallet, "ZeroAddress");
  });

  it("should start with nonce = 0", async function () {
    expect(await wallet.nonce()).to.equal(0n);
  });

  // ─── EIP-1271 isValidSignature ────────────────────────────────────────────

  it("should return MAGIC_VALUE for valid owner signature", async function () {
    const message = ethers.toUtf8Bytes("Hello, agent world!");
    const hash = ethers.hashMessage(message);
    const signature = await ownerWallet.signMessage(message);

    // isValidSignature expects the raw hash, not the prefixed one
    // Owner signed with personal_sign, so we need to verify with the prefixed hash
    const result = await wallet.isValidSignature(hash, signature);
    expect(result).to.equal(MAGIC_VALUE);
  });

  it("should return 0xffffffff for invalid signature", async function () {
    const hash = ethers.keccak256(ethers.toUtf8Bytes("test"));
    const fakeSignature = await other.signMessage(ethers.toUtf8Bytes("test"));
    const result = await wallet.isValidSignature(hash, fakeSignature);
    expect(result).to.equal("0xffffffff");
  });

  // ─── Execute ─────────────────────────────────────────────────────────────

  it("should allow owner to execute a call", async function () {
    // Fund the wallet
    await owner.sendTransaction({ to: await wallet.getAddress(), value: ethers.parseEther("1") });

    const balanceBefore = await ethers.provider.getBalance(target.address);
    await wallet.connect(ownerWallet).execute(target.address, ethers.parseEther("0.5"), "0x");
    const balanceAfter = await ethers.provider.getBalance(target.address);

    expect(balanceAfter - balanceBefore).to.equal(ethers.parseEther("0.5"));
  });

  it("should emit Executed event on successful call", async function () {
    await owner.sendTransaction({ to: await wallet.getAddress(), value: ethers.parseEther("1") });
    await expect(
      wallet.connect(ownerWallet).execute(target.address, ethers.parseEther("0.1"), "0x")
    ).to.emit(wallet, "Executed")
     .withArgs(target.address, ethers.parseEther("0.1"), "0x", true);
  });

  it("should revert execute from non-owner", async function () {
    await expect(
      wallet.connect(other).execute(target.address, 0n, "0x")
    ).to.be.revertedWithCustomError(wallet, "Unauthorized")
     .withArgs(other.address);
  });

  // ─── Execute With Signature (meta-tx) ─────────────────────────────────────

  it("should execute with valid EIP-712 signature", async function () {
    await owner.sendTransaction({ to: await wallet.getAddress(), value: ethers.parseEther("1") });

    const targetAddr = target.address;
    const value = ethers.parseEther("0.1");
    const data = "0x";

    // Get EIP-712 digest from contract (already includes \x19\x01 prefix)
    const digest = await wallet.hashExecute(targetAddr, value, data);

    // Sign raw digest using ownerWallet's signing key (no additional Ethereum prefix)
    const signingKey = new ethers.SigningKey(TEST_PRIVATE_KEY);
    const sig = signingKey.sign(digest);
    const signature = ethers.Signature.from(sig).serialized;

    const balanceBefore = await ethers.provider.getBalance(targetAddr);
    await wallet.connect(other).executeWithSignature(targetAddr, value, data, signature);
    const balanceAfter = await ethers.provider.getBalance(targetAddr);

    expect(balanceAfter - balanceBefore).to.equal(value);
    expect(await wallet.nonce()).to.equal(1n);
  });

  it("should increment nonce after meta-tx execution", async function () {
    await owner.sendTransaction({ to: await wallet.getAddress(), value: ethers.parseEther("1") });
    const digest = await wallet.hashExecute(target.address, 0n, "0x");

    // Sign raw digest using ownerWallet's signing key (no additional Ethereum prefix)
    const signingKey = new ethers.SigningKey(TEST_PRIVATE_KEY);
    const sig = signingKey.sign(digest);
    const signature = ethers.Signature.from(sig).serialized;

    expect(await wallet.nonce()).to.equal(0n);
    await wallet.connect(other).executeWithSignature(target.address, 0n, "0x", signature);
    expect(await wallet.nonce()).to.equal(1n);
  });

  it("should revert meta-tx with wrong signer", async function () {
    const digest = await wallet.hashExecute(target.address, 0n, "0x");
    const badSignature = await other.signMessage(ethers.getBytes(digest));

    await expect(
      wallet.connect(other).executeWithSignature(target.address, 0n, "0x", badSignature)
    ).to.be.revertedWithCustomError(wallet, "InvalidSignature");
  });

  // ─── Ownership Transfer ───────────────────────────────────────────────────

  it("should allow owner to transfer ownership", async function () {
    await expect(wallet.connect(ownerWallet).transferOwnership(other.address))
      .to.emit(wallet, "OwnershipTransferred")
      .withArgs(ownerWallet.address, other.address);
    expect(await wallet.owner()).to.equal(other.address);
  });

  it("should revert ownership transfer from non-owner", async function () {
    await expect(
      wallet.connect(other).transferOwnership(other.address)
    ).to.be.revertedWithCustomError(wallet, "Unauthorized");
  });

  it("should revert ownership transfer to zero address", async function () {
    await expect(
      wallet.connect(ownerWallet).transferOwnership(ethers.ZeroAddress)
    ).to.be.revertedWithCustomError(wallet, "ZeroAddress");
  });

  // ─── Receive ETH ─────────────────────────────────────────────────────────

  it("should receive ETH and emit Received event", async function () {
    await expect(
      owner.sendTransaction({ to: await wallet.getAddress(), value: ethers.parseEther("1") })
    ).to.emit(wallet, "Received")
     .withArgs(owner.address, ethers.parseEther("1"));
  });
});
