import { ethers, network } from "hardhat";
import * as fs from "fs";
import * as path from "path";

interface DeploymentRecord {
  network: string;
  chainId: number;
  deployedAt: string;
  contracts: {
    IdentityRegistry: string;
    ReputationRegistry: string;
    ValidationRegistry: string;
    AgentWallet: string;
  };
  deployer: string;
}

async function main() {
  const [deployer] = await ethers.getSigners();
  const chainId = (await ethers.provider.getNetwork()).chainId;
  const networkName = network.name;

  console.log(`\n=== ERC-8004 Deployment ===`);
  console.log(`Network:  ${networkName} (chainId: ${chainId})`);
  console.log(`Deployer: ${deployer.address}`);
  const balance = await ethers.provider.getBalance(deployer.address);
  console.log(`Balance:  ${ethers.formatEther(balance)} ETH`);
  console.log(`================================\n`);

  // 1. Deploy IdentityRegistry
  console.log("1. Deploying IdentityRegistry...");
  const IdentityRegistry = await ethers.getContractFactory("IdentityRegistry");
  const identityRegistry = await IdentityRegistry.deploy();
  await identityRegistry.waitForDeployment();
  const identityAddr = await identityRegistry.getAddress();
  console.log(`   IdentityRegistry: ${identityAddr}`);

  // 2. Deploy ReputationRegistry (depends on IdentityRegistry)
  console.log("2. Deploying ReputationRegistry...");
  const ReputationRegistry = await ethers.getContractFactory("ReputationRegistry");
  const reputationRegistry = await ReputationRegistry.deploy(identityAddr);
  await reputationRegistry.waitForDeployment();
  const reputationAddr = await reputationRegistry.getAddress();
  console.log(`   ReputationRegistry: ${reputationAddr}`);

  // 3. Deploy ValidationRegistry (depends on IdentityRegistry)
  console.log("3. Deploying ValidationRegistry...");
  const ValidationRegistry = await ethers.getContractFactory("ValidationRegistry");
  const validationRegistry = await ValidationRegistry.deploy(identityAddr);
  await validationRegistry.waitForDeployment();
  const validationAddr = await validationRegistry.getAddress();
  console.log(`   ValidationRegistry: ${validationAddr}`);

  // 4. Deploy AgentWallet (for the deployer / agent)
  console.log("4. Deploying AgentWallet...");
  const AgentWallet = await ethers.getContractFactory("AgentWallet");
  const agentWallet = await AgentWallet.deploy(deployer.address);
  await agentWallet.waitForDeployment();
  const walletAddr = await agentWallet.getAddress();
  console.log(`   AgentWallet: ${walletAddr}`);

  // Save deployment record
  const record: DeploymentRecord = {
    network: networkName,
    chainId: Number(chainId),
    deployedAt: new Date().toISOString(),
    contracts: {
      IdentityRegistry: identityAddr,
      ReputationRegistry: reputationAddr,
      ValidationRegistry: validationAddr,
      AgentWallet: walletAddr,
    },
    deployer: deployer.address,
  };

  const deploymentsDir = path.join(__dirname, "..", "deployments");
  if (!fs.existsSync(deploymentsDir)) {
    fs.mkdirSync(deploymentsDir, { recursive: true });
  }

  const outPath = path.join(deploymentsDir, `${networkName}.json`);
  fs.writeFileSync(outPath, JSON.stringify(record, null, 2));
  console.log(`\nDeployment record saved: ${outPath}`);

  // Print summary
  console.log("\n=== Deployment Complete ===");
  console.log(JSON.stringify(record.contracts, null, 2));

  // Verify hint
  if (networkName !== "hardhat" && networkName !== "localhost") {
    console.log("\n=== Verification Commands ===");
    console.log(`npx hardhat verify --network ${networkName} ${identityAddr}`);
    console.log(`npx hardhat verify --network ${networkName} ${reputationAddr} "${identityAddr}"`);
    console.log(`npx hardhat verify --network ${networkName} ${validationAddr} "${identityAddr}"`);
    console.log(`npx hardhat verify --network ${networkName} ${walletAddr} "${deployer.address}"`);
  }
}

main()
  .then(() => process.exit(0))
  .catch((error) => {
    console.error("Deployment failed:", error);
    process.exit(1);
  });
