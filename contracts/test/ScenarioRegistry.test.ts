import { expect } from "chai";
import { ethers } from "hardhat";
import { ScenarioRegistry } from "../typechain-types";
import { SignerWithAddress } from "@nomicfoundation/hardhat-ethers/signers";

describe("ScenarioRegistry", function () {
  let registry: ScenarioRegistry;
  let deployer: SignerWithAddress;
  let operator: SignerWithAddress;

  // ScenarioType enum values
  const UNKNOWN = 0n;
  const BULL_RUN = 1n;
  const BEAR_CRASH = 2n;
  const VOLATILE_CHOP = 3n;
  const STABLE_TREND = 4n;

  // Helper to generate a runId
  function makeRunId(seed: string): string {
    return ethers.keccak256(ethers.toUtf8Bytes(seed));
  }

  const TIMELINE_HASH = ethers.keccak256(ethers.toUtf8Bytes("tick_timeline_json_data"));

  beforeEach(async function () {
    [deployer, operator] = await ethers.getSigners();

    const ScenarioRegistry = await ethers.getContractFactory("ScenarioRegistry");
    registry = await ScenarioRegistry.deploy();
    await registry.waitForDeployment();
  });

  // ─── Deployment ──────────────────────────────────────────────────────────────

  it("should deploy with zero runs", async function () {
    expect(await registry.runCount()).to.equal(0n);
  });

  it("should deploy with zero snapshots", async function () {
    expect(await registry.snapshotCount()).to.equal(0n);
  });

  it("should start with zero scenario type counts", async function () {
    expect(await registry.scenarioTypeCount(BULL_RUN)).to.equal(0n);
    expect(await registry.scenarioTypeCount(BEAR_CRASH)).to.equal(0n);
  });

  // ─── commitScenario ───────────────────────────────────────────────────────

  it("should commit a BULL_RUN scenario and emit event", async function () {
    const runId = makeRunId("bull_run_001");
    await expect(
      registry.commitScenario(runId, BULL_RUN, 20n, TIMELINE_HASH, 15000n, 7500n, false)
    ).to.emit(registry, "ScenarioCommitted");
  });

  it("should commit a BEAR_CRASH scenario with circuit breaker fired", async function () {
    const runId = makeRunId("bear_crash_001");
    await expect(
      registry.commitScenario(runId, BEAR_CRASH, 20n, TIMELINE_HASH, -45000n, 4000n, true)
    ).to.emit(registry, "ScenarioCommitted");
  });

  it("should increment runCount after commit", async function () {
    const runId = makeRunId("volatile_001");
    await registry.commitScenario(runId, VOLATILE_CHOP, 20n, TIMELINE_HASH, 3000n, 5500n, false);
    expect(await registry.runCount()).to.equal(1n);
  });

  it("should increment scenarioTypeCount for correct type", async function () {
    const runId1 = makeRunId("bull_001");
    const runId2 = makeRunId("bull_002");
    await registry.commitScenario(runId1, BULL_RUN, 20n, TIMELINE_HASH, 12000n, 8000n, false);
    await registry.commitScenario(runId2, BULL_RUN, 20n, TIMELINE_HASH, 18000n, 7500n, false);
    expect(await registry.scenarioTypeCount(BULL_RUN)).to.equal(2n);
  });

  it("should store scenario data correctly", async function () {
    const runId = makeRunId("stable_001");
    await registry.commitScenario(runId, STABLE_TREND, 20n, TIMELINE_HASH, 8000n, 9000n, false);
    const run = await registry.getRunById(runId);
    expect(run.runId).to.equal(runId);
    expect(run.scenarioType).to.equal(STABLE_TREND);
    expect(run.tickCount).to.equal(20n);
    expect(run.timelineHash).to.equal(TIMELINE_HASH);
    expect(run.totalPnlCents).to.equal(8000n);
    expect(run.consensusRate).to.equal(9000n);
    expect(run.circuitBreakerFired).to.be.false;
  });

  it("should store bear_crash with circuit breaker fired = true", async function () {
    const runId = makeRunId("bear_002");
    await registry.commitScenario(runId, BEAR_CRASH, 20n, TIMELINE_HASH, -80000n, 3500n, true);
    const run = await registry.getRunById(runId);
    expect(run.circuitBreakerFired).to.be.true;
    expect(run.totalPnlCents).to.equal(-80000n);
  });

  it("should revert on duplicate runId", async function () {
    const runId = makeRunId("dup_test");
    await registry.commitScenario(runId, BULL_RUN, 20n, TIMELINE_HASH, 5000n, 7500n, false);
    await expect(
      registry.commitScenario(runId, BULL_RUN, 20n, TIMELINE_HASH, 5000n, 7500n, false)
    ).to.be.revertedWithCustomError(registry, "DuplicateRunId");
  });

  it("should revert on zero tickCount", async function () {
    const runId = makeRunId("zero_ticks");
    await expect(
      registry.commitScenario(runId, BULL_RUN, 0n, TIMELINE_HASH, 5000n, 7500n, false)
    ).to.be.revertedWithCustomError(registry, "InvalidTickCount");
  });

  it("should revert on consensusRate > 10000", async function () {
    const runId = makeRunId("bad_rate");
    await expect(
      registry.commitScenario(runId, BULL_RUN, 20n, TIMELINE_HASH, 5000n, 10001n, false)
    ).to.be.revertedWithCustomError(registry, "InvalidConsensusRate");
  });

  it("should accept consensusRate of exactly 10000", async function () {
    const runId = makeRunId("max_rate");
    await expect(
      registry.commitScenario(runId, BULL_RUN, 20n, TIMELINE_HASH, 5000n, 10000n, false)
    ).to.not.be.reverted;
  });

  // ─── runExists / getRunByIndex ────────────────────────────────────────────

  it("should return true for runExists after commit", async function () {
    const runId = makeRunId("exists_test");
    await registry.commitScenario(runId, BULL_RUN, 20n, TIMELINE_HASH, 5000n, 7500n, false);
    expect(await registry.runExists(runId)).to.be.true;
  });

  it("should return false for runExists for unknown runId", async function () {
    const runId = makeRunId("ghost_run");
    expect(await registry.runExists(runId)).to.be.false;
  });

  it("should retrieve run by index", async function () {
    const runId = makeRunId("index_test");
    await registry.commitScenario(runId, VOLATILE_CHOP, 15n, TIMELINE_HASH, 2500n, 6000n, false);
    const run = await registry.getRunByIndex(0n);
    expect(run.runId).to.equal(runId);
    expect(run.scenarioType).to.equal(VOLATILE_CHOP);
  });

  it("should revert getRunById for unknown runId", async function () {
    const runId = makeRunId("not_committed");
    await expect(registry.getRunById(runId)).to.be.revertedWithCustomError(registry, "RunNotFound");
  });

  // ─── commitLeaderboardEntry ───────────────────────────────────────────────

  it("should commit a leaderboard entry and emit event", async function () {
    const runId = makeRunId("lb_test_001");
    await registry.commitScenario(runId, BULL_RUN, 20n, TIMELINE_HASH, 12000n, 8000n, false);
    await expect(
      registry.commitLeaderboardEntry(runId, 1n, 1n, 7820n, 5000n, 7500n, 48n)
    ).to.emit(registry, "LeaderboardSnapshotCommitted");
  });

  it("should increment snapshotCount after commit", async function () {
    const runId = makeRunId("lb_test_002");
    await registry.commitScenario(runId, BULL_RUN, 20n, TIMELINE_HASH, 12000n, 8000n, false);
    await registry.commitLeaderboardEntry(runId, 1n, 1n, 7820n, 5000n, 7500n, 48n);
    expect(await registry.snapshotCount()).to.equal(1n);
  });

  it("should store leaderboard entry data correctly", async function () {
    const runId = makeRunId("lb_data_test");
    await registry.commitScenario(runId, BULL_RUN, 20n, TIMELINE_HASH, 12000n, 8000n, false);
    await registry.commitLeaderboardEntry(runId, 1n, 2n, 6910n, -2000n, 6250n, 71n);
    const snapshot = await registry.getSnapshot(0n);
    expect(snapshot.runId).to.equal(runId);
    expect(snapshot.rank).to.equal(1n);
    expect(snapshot.agentId).to.equal(2n);
    expect(snapshot.reputationScore).to.equal(6910n);
    expect(snapshot.pnlCents).to.equal(-2000n);
    expect(snapshot.winRate).to.equal(6250n);
    expect(snapshot.totalTrades).to.equal(71n);
  });

  it("should revert leaderboard entry for unknown runId", async function () {
    const runId = makeRunId("ghost_lb");
    await expect(
      registry.commitLeaderboardEntry(runId, 1n, 1n, 7820n, 5000n, 7500n, 48n)
    ).to.be.revertedWithCustomError(registry, "RunNotFound");
  });

  it("should revert leaderboard entry with rank = 0", async function () {
    const runId = makeRunId("rank_zero");
    await registry.commitScenario(runId, BULL_RUN, 20n, TIMELINE_HASH, 12000n, 8000n, false);
    await expect(
      registry.commitLeaderboardEntry(runId, 0n, 1n, 7820n, 5000n, 7500n, 48n)
    ).to.be.revertedWithCustomError(registry, "InvalidRank");
  });

  it("should revert leaderboard entry with winRate > 10000", async function () {
    const runId = makeRunId("bad_winrate");
    await registry.commitScenario(runId, BULL_RUN, 20n, TIMELINE_HASH, 12000n, 8000n, false);
    await expect(
      registry.commitLeaderboardEntry(runId, 1n, 1n, 7820n, 5000n, 10001n, 48n)
    ).to.be.revertedWithCustomError(registry, "InvalidWinRate");
  });

  // ─── getSnapshotsForRun ───────────────────────────────────────────────────

  it("should return all snapshots for a run", async function () {
    const runId = makeRunId("multi_lb");
    await registry.commitScenario(runId, BULL_RUN, 20n, TIMELINE_HASH, 12000n, 8000n, false);
    await registry.commitLeaderboardEntry(runId, 1n, 1n, 7820n, 5000n, 7500n, 48n);
    await registry.commitLeaderboardEntry(runId, 2n, 2n, 7240n, 3500n, 6250n, 52n);
    await registry.commitLeaderboardEntry(runId, 3n, 3n, 6910n, 8000n, 5830n, 57n);

    const snapshots = await registry.getSnapshotsForRun(runId);
    expect(snapshots.length).to.equal(3);
    expect(snapshots[0].rank).to.equal(1n);
    expect(snapshots[1].rank).to.equal(2n);
    expect(snapshots[2].rank).to.equal(3n);
  });

  it("should return empty array for getSnapshotsForRun with no entries", async function () {
    const runId = makeRunId("empty_lb");
    await registry.commitScenario(runId, BULL_RUN, 20n, TIMELINE_HASH, 12000n, 8000n, false);
    const snapshots = await registry.getSnapshotsForRun(runId);
    expect(snapshots.length).to.equal(0);
  });

  // ─── getRunsByType ────────────────────────────────────────────────────────

  it("should return correct runs by type", async function () {
    await registry.commitScenario(makeRunId("b1"), BULL_RUN, 20n, TIMELINE_HASH, 10000n, 8000n, false);
    await registry.commitScenario(makeRunId("c1"), BEAR_CRASH, 20n, TIMELINE_HASH, -50000n, 4000n, true);
    await registry.commitScenario(makeRunId("b2"), BULL_RUN, 20n, TIMELINE_HASH, 15000n, 7500n, false);

    const bulls = await registry.getRunsByType(BULL_RUN);
    expect(bulls.length).to.equal(2);
    const crashes = await registry.getRunsByType(BEAR_CRASH);
    expect(crashes.length).to.equal(1);
    expect(crashes[0].circuitBreakerFired).to.be.true;
  });

  it("should return empty for type with no runs", async function () {
    const stable = await registry.getRunsByType(STABLE_TREND);
    expect(stable.length).to.equal(0);
  });

  // ─── circuitBreakerFiredCount ─────────────────────────────────────────────

  it("should count only runs where circuit breaker fired", async function () {
    await registry.commitScenario(makeRunId("cb_1"), BEAR_CRASH, 20n, TIMELINE_HASH, -80000n, 3500n, true);
    await registry.commitScenario(makeRunId("cb_2"), BULL_RUN, 20n, TIMELINE_HASH, 10000n, 8000n, false);
    await registry.commitScenario(makeRunId("cb_3"), BEAR_CRASH, 20n, TIMELINE_HASH, -40000n, 4200n, true);

    expect(await registry.circuitBreakerFiredCount()).to.equal(2n);
  });

  it("should return zero circuitBreakerFiredCount when none fired", async function () {
    await registry.commitScenario(makeRunId("no_cb"), BULL_RUN, 20n, TIMELINE_HASH, 10000n, 8000n, false);
    expect(await registry.circuitBreakerFiredCount()).to.equal(0n);
  });

  // ─── Multiple scenario types ───────────────────────────────────────────────

  it("should handle all four scenario types", async function () {
    await registry.commitScenario(makeRunId("s1"), BULL_RUN, 20n, TIMELINE_HASH, 12000n, 8000n, false);
    await registry.commitScenario(makeRunId("s2"), BEAR_CRASH, 20n, TIMELINE_HASH, -40000n, 4000n, true);
    await registry.commitScenario(makeRunId("s3"), VOLATILE_CHOP, 20n, TIMELINE_HASH, 3000n, 5500n, false);
    await registry.commitScenario(makeRunId("s4"), STABLE_TREND, 20n, TIMELINE_HASH, 7000n, 9000n, false);

    expect(await registry.runCount()).to.equal(4n);
    expect(await registry.scenarioTypeCount(BULL_RUN)).to.equal(1n);
    expect(await registry.scenarioTypeCount(BEAR_CRASH)).to.equal(1n);
    expect(await registry.scenarioTypeCount(VOLATILE_CHOP)).to.equal(1n);
    expect(await registry.scenarioTypeCount(STABLE_TREND)).to.equal(1n);
  });
});
