"use client";

import { useState, useEffect } from "react";
import ReputationCard from "./ReputationCard";
import TradeHistory from "./TradeHistory";
import AgentIdentity from "./AgentIdentity";
import PnLChart from "./PnLChart";
import StrategyStatus from "./StrategyStatus";

interface AgentStats {
  agentId: number;
  did: string;
  reputationScore: number;
  feedbackCount: number;
  totalTrades: number;
  winRate: number;
  totalPnl: number;
  pendingValidations: number;
}

const MOCK_STATS: AgentStats = {
  agentId: 1,
  did: "eip155:11155111:0xabcdef...1234:1",
  reputationScore: 8.5,
  feedbackCount: 12,
  totalTrades: 47,
  winRate: 0.62,
  totalPnl: 34.5,
  pendingValidations: 3,
};

export default function AgentDashboard() {
  const [stats, setStats] = useState<AgentStats>(MOCK_STATS);
  const [loading, setLoading] = useState(false);

  // TODO: Replace with actual on-chain data fetching via wagmi hooks
  useEffect(() => {
    // Simulate loading
    setLoading(false);
  }, []);

  if (loading) {
    return (
      <div className="flex items-center justify-center h-64">
        <div className="text-gray-400 animate-pulse">Loading agent data...</div>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      {/* Identity */}
      <AgentIdentity agentId={stats.agentId} did={stats.did} />

      {/* Stats Grid */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        <StatCard label="Total Trades" value={stats.totalTrades.toString()} />
        <StatCard
          label="Win Rate"
          value={`${(stats.winRate * 100).toFixed(0)}%`}
          positive={stats.winRate > 0.5}
        />
        <StatCard
          label="Total PnL"
          value={`$${stats.totalPnl.toFixed(2)}`}
          positive={stats.totalPnl > 0}
        />
        <StatCard
          label="Pending Validations"
          value={stats.pendingValidations.toString()}
        />
      </div>

      {/* P&L Chart */}
      <PnLChart />

      {/* Strategy Status */}
      <StrategyStatus />

      {/* Reputation */}
      <ReputationCard
        score={stats.reputationScore}
        feedbackCount={stats.feedbackCount}
      />

      {/* Backtest Results */}
      <BacktestResults />

      {/* Portfolio Weights */}
      <PortfolioWeights />

      {/* Multi-Agent Consensus */}
      <MultiAgentConsensus />

      {/* Trade History */}
      <TradeHistory />
    </div>
  );
}

function StatCard({
  label,
  value,
  positive,
}: {
  label: string;
  value: string;
  positive?: boolean;
}) {
  return (
    <div className="bg-gray-900 rounded-lg p-4 border border-gray-800">
      <p className="text-gray-400 text-sm">{label}</p>
      <p
        className={`text-2xl font-bold mt-1 ${
          positive === undefined
            ? "text-white"
            : positive
            ? "text-green-400"
            : "text-red-400"
        }`}
      >
        {value}
      </p>
    </div>
  );
}

// ─── Backtest Results Panel ───────────────────────────────────────────────────

interface BacktestResult {
  strategy: string;
  sharpeRatio: number;
  maxDrawdownPct: number;
  winRate: number;
  netPnl: number;
  totalTrades: number;
  token: string;
}

const MOCK_BACKTEST_RESULTS: BacktestResult[] = [
  {
    strategy: "mesh_consensus",
    sharpeRatio: 2.14,
    maxDrawdownPct: 4.2,
    winRate: 0.65,
    netPnl: 87.3,
    totalTrades: 23,
    token: "ETH",
  },
  {
    strategy: "trend",
    sharpeRatio: 1.72,
    maxDrawdownPct: 6.8,
    winRate: 0.58,
    netPnl: 54.1,
    totalTrades: 31,
    token: "ETH",
  },
  {
    strategy: "mean_reversion",
    sharpeRatio: 1.41,
    maxDrawdownPct: 8.3,
    winRate: 0.52,
    netPnl: 32.7,
    totalTrades: 28,
    token: "ETH",
  },
  {
    strategy: "momentum",
    sharpeRatio: 1.18,
    maxDrawdownPct: 11.4,
    winRate: 0.49,
    netPnl: 12.6,
    totalTrades: 19,
    token: "ETH",
  },
];

function BacktestResults() {
  const best = MOCK_BACKTEST_RESULTS[0]; // mesh_consensus is best

  return (
    <div className="bg-gray-900 rounded-lg p-6 border border-gray-800">
      <div className="flex items-center justify-between mb-4">
        <h2 className="text-lg font-semibold text-white">Backtest Results</h2>
        <span className="text-xs text-gray-400 bg-gray-800 px-2 py-1 rounded">
          30-day · synthetic GBM · BacktestRegistry
        </span>
      </div>

      {/* Best Strategy Banner */}
      <div className="bg-blue-900/30 border border-blue-700 rounded-lg p-3 mb-4 flex items-center justify-between">
        <div>
          <p className="text-xs text-gray-400">Best Strategy</p>
          <p className="text-white font-bold capitalize">
            {best.strategy.replace("_", " ")}
          </p>
        </div>
        <div className="flex gap-6 text-right">
          <div>
            <p className="text-xs text-gray-400">Sharpe</p>
            <p className="text-blue-300 font-bold">{best.sharpeRatio.toFixed(2)}</p>
          </div>
          <div>
            <p className="text-xs text-gray-400">Max DD</p>
            <p className="text-orange-300 font-bold">{best.maxDrawdownPct.toFixed(1)}%</p>
          </div>
          <div>
            <p className="text-xs text-gray-400">Win Rate</p>
            <p className="text-green-300 font-bold">{(best.winRate * 100).toFixed(0)}%</p>
          </div>
        </div>
      </div>

      {/* Strategy Comparison Table */}
      <div className="overflow-x-auto">
        <table className="w-full text-sm">
          <thead>
            <tr className="text-gray-400 text-left border-b border-gray-800">
              <th className="pb-2 pr-4">Strategy</th>
              <th className="pb-2 pr-4 text-right">Sharpe</th>
              <th className="pb-2 pr-4 text-right">Max DD</th>
              <th className="pb-2 pr-4 text-right">Win Rate</th>
              <th className="pb-2 text-right">Net PnL</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-gray-800">
            {MOCK_BACKTEST_RESULTS.map((r, i) => (
              <tr
                key={r.strategy}
                className={i === 0 ? "text-blue-300" : "text-gray-300"}
              >
                <td className="py-2 pr-4 capitalize font-medium">
                  {r.strategy.replace("_", " ")}
                  {i === 0 && (
                    <span className="ml-2 text-xs text-blue-400 bg-blue-900/50 px-1.5 py-0.5 rounded">
                      best
                    </span>
                  )}
                </td>
                <td className="py-2 pr-4 text-right font-mono">
                  {r.sharpeRatio.toFixed(2)}
                </td>
                <td
                  className={`py-2 pr-4 text-right font-mono ${
                    r.maxDrawdownPct > 10 ? "text-red-400" : "text-orange-300"
                  }`}
                >
                  {r.maxDrawdownPct.toFixed(1)}%
                </td>
                <td className="py-2 pr-4 text-right font-mono">
                  {(r.winRate * 100).toFixed(0)}%
                </td>
                <td
                  className={`py-2 text-right font-mono ${
                    r.netPnl >= 0 ? "text-green-400" : "text-red-400"
                  }`}
                >
                  ${r.netPnl > 0 ? "+" : ""}
                  {r.netPnl.toFixed(1)}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      <p className="text-xs text-gray-500 mt-3">
        Registered in BacktestRegistry (on-chain analog). mesh_consensus aggregates
        trend + mean_reversion + momentum signals (2/3 vote required).
      </p>
    </div>
  );
}

// ─── Portfolio Weights Panel ──────────────────────────────────────────────────

interface ProtocolWeight {
  protocol: string;
  weight: number;
  credoraTier: string;
  targetWeight: number;
  drift: number;
  kellyMultiplier: number;
}

const MOCK_PORTFOLIO_WEIGHTS: ProtocolWeight[] = [
  {
    protocol: "Aave",
    weight: 0.32,
    credoraTier: "AAA",
    targetWeight: 0.30,
    drift: 0.02,
    kellyMultiplier: 1.0,
  },
  {
    protocol: "Uniswap",
    weight: 0.27,
    credoraTier: "A",
    targetWeight: 0.30,
    drift: -0.03,
    kellyMultiplier: 0.8,
  },
  {
    protocol: "Compound",
    weight: 0.22,
    credoraTier: "AA",
    targetWeight: 0.20,
    drift: 0.02,
    kellyMultiplier: 0.9,
  },
  {
    protocol: "MakerDAO",
    weight: 0.12,
    credoraTier: "BBB",
    targetWeight: 0.15,
    drift: -0.03,
    kellyMultiplier: 0.65,
  },
  {
    protocol: "Curve",
    weight: 0.07,
    credoraTier: "BB",
    targetWeight: 0.05,
    drift: 0.02,
    kellyMultiplier: 0.5,
  },
];

const TIER_COLORS: Record<string, string> = {
  AAA: "text-emerald-400",
  AA: "text-green-400",
  A: "text-lime-400",
  BBB: "text-yellow-400",
  BB: "text-orange-400",
  B: "text-red-400",
  CCC: "text-red-600",
  NR: "text-gray-500",
};

function PortfolioWeights() {
  const needsRebalance = MOCK_PORTFOLIO_WEIGHTS.some(
    (p) => Math.abs(p.drift) > 0.05
  );

  return (
    <div className="bg-gray-900 rounded-lg p-6 border border-gray-800">
      <div className="flex items-center justify-between mb-4">
        <h2 className="text-lg font-semibold text-white">
          Portfolio Weights
          <span className="ml-2 text-xs text-gray-400 font-normal">
            (mean-variance + Credora)
          </span>
        </h2>
        <div className="flex items-center gap-2">
          {needsRebalance ? (
            <span className="text-xs text-orange-400 bg-orange-900/30 border border-orange-700 px-2 py-1 rounded">
              Rebalance Pending
            </span>
          ) : (
            <span className="text-xs text-green-400 bg-green-900/30 border border-green-700 px-2 py-1 rounded">
              Balanced
            </span>
          )}
          <span className="text-xs text-gray-400 bg-gray-800 px-2 py-1 rounded">
            5% drift trigger
          </span>
        </div>
      </div>

      {/* Weight Bars */}
      <div className="space-y-3">
        {MOCK_PORTFOLIO_WEIGHTS.map((p) => {
          const isOverweight = p.drift > 0;
          const driftBig = Math.abs(p.drift) > 0.05;
          return (
            <div key={p.protocol}>
              <div className="flex items-center justify-between mb-1">
                <div className="flex items-center gap-2">
                  <span className="text-sm text-white font-medium">
                    {p.protocol}
                  </span>
                  <span
                    className={`text-xs font-bold ${TIER_COLORS[p.credoraTier] || "text-gray-400"}`}
                  >
                    {p.credoraTier}
                  </span>
                </div>
                <div className="flex items-center gap-3 text-xs">
                  <span className="text-gray-400">
                    Target: {(p.targetWeight * 100).toFixed(0)}%
                  </span>
                  <span
                    className={`font-mono ${driftBig ? (isOverweight ? "text-orange-400" : "text-blue-400") : "text-gray-400"}`}
                  >
                    {isOverweight ? "▲" : "▼"}
                    {(Math.abs(p.drift) * 100).toFixed(1)}%
                  </span>
                  <span className="text-white font-bold">
                    {(p.weight * 100).toFixed(0)}%
                  </span>
                </div>
              </div>
              <div className="relative w-full bg-gray-800 rounded-full h-2">
                {/* Target marker */}
                <div
                  className="absolute top-0 h-full w-0.5 bg-gray-500"
                  style={{ left: `${p.targetWeight * 100}%` }}
                />
                {/* Current weight bar */}
                <div
                  className={`h-2 rounded-full transition-all ${
                    driftBig ? "bg-orange-500" : "bg-blue-500"
                  }`}
                  style={{ width: `${p.weight * 100}%` }}
                />
              </div>
              <div className="flex justify-between text-xs text-gray-600 mt-0.5">
                <span>Kelly: {(p.kellyMultiplier * 100).toFixed(0)}%</span>
                <span>Cap: {p.credoraTier === "AAA" ? "40" : p.credoraTier === "AA" ? "35" : p.credoraTier === "A" ? "30" : p.credoraTier === "BBB" ? "25" : "20"}%</span>
              </div>
            </div>
          );
        })}
      </div>

      <p className="text-xs text-gray-500 mt-4">
        Weights optimized via PortfolioOptimizer (min-variance). Credora tier
        caps enforce max concentration. Drift {">"} 5% triggers rebalancing orders.
      </p>
    </div>
  );
}

// ─── Multi-Agent Consensus Panel ──────────────────────────────────────────────

interface MeshAgent {
  id: string;
  profile: "conservative" | "balanced" | "aggressive";
  reputationScore: number;
  minGrade: string;
  kellyFraction: number;
  lastVote: "BUY" | "SELL" | "HOLD" | "REJECT" | null;
}

const MESH_AGENTS: MeshAgent[] = [
  {
    id: "conservative_agent",
    profile: "conservative",
    reputationScore: 7.0,
    minGrade: "A",
    kellyFraction: 0.15,
    lastVote: "BUY",
  },
  {
    id: "balanced_agent",
    profile: "balanced",
    reputationScore: 6.5,
    minGrade: "BBB",
    kellyFraction: 0.25,
    lastVote: "BUY",
  },
  {
    id: "aggressive_agent",
    profile: "aggressive",
    reputationScore: 5.5,
    minGrade: "BB",
    kellyFraction: 0.35,
    lastVote: "HOLD",
  },
];

const MOCK_CONSENSUS = {
  reached: true,
  finalAction: "BUY" as const,
  votesFor: 2,
  votesAgainst: 1,
  weightedSize: 4.8,
};

const PROFILE_COLORS: Record<string, string> = {
  conservative: "border-blue-600",
  balanced: "border-yellow-500",
  aggressive: "border-red-500",
};

const VOTE_COLORS: Record<string, string> = {
  BUY: "text-green-400",
  SELL: "text-red-400",
  HOLD: "text-yellow-400",
  REJECT: "text-gray-500",
};

function MultiAgentConsensus() {
  return (
    <div className="bg-gray-900 rounded-lg p-6 border border-gray-800">
      <div className="flex items-center justify-between mb-4">
        <h2 className="text-lg font-semibold text-white">
          Multi-Agent Consensus Mesh
        </h2>
        <span className="text-xs text-gray-400 bg-gray-800 px-2 py-1 rounded">
          ERC-8004 Reputation Weighted
        </span>
      </div>

      {/* Consensus Result Banner */}
      <div
        className={`rounded-lg p-3 mb-4 flex items-center justify-between ${
          MOCK_CONSENSUS.reached
            ? "bg-green-900/30 border border-green-700"
            : "bg-gray-800 border border-gray-700"
        }`}
      >
        <div>
          <span className="text-sm font-medium text-gray-300">
            Last Consensus:{" "}
          </span>
          <span
            className={`font-bold ${
              MOCK_CONSENSUS.reached ? "text-green-400" : "text-gray-400"
            }`}
          >
            {MOCK_CONSENSUS.reached
              ? `${MOCK_CONSENSUS.finalAction} (${MOCK_CONSENSUS.votesFor}/${MOCK_CONSENSUS.votesFor + MOCK_CONSENSUS.votesAgainst} votes)`
              : "No Consensus"}
          </span>
        </div>
        {MOCK_CONSENSUS.reached && (
          <div className="text-right">
            <p className="text-xs text-gray-400">Weighted Size</p>
            <p className="text-white font-bold">
              ${MOCK_CONSENSUS.weightedSize.toFixed(2)}
            </p>
          </div>
        )}
      </div>

      {/* Agent Cards */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        {MESH_AGENTS.map((agent) => (
          <div
            key={agent.id}
            className={`bg-gray-800 rounded-lg p-4 border-l-4 ${PROFILE_COLORS[agent.profile]}`}
          >
            <div className="flex items-center justify-between mb-2">
              <h3 className="text-sm font-semibold text-white capitalize">
                {agent.profile}
              </h3>
              {agent.lastVote && (
                <span
                  className={`text-xs font-bold px-2 py-0.5 rounded bg-gray-700 ${VOTE_COLORS[agent.lastVote]}`}
                >
                  {agent.lastVote}
                </span>
              )}
            </div>

            <div className="space-y-1 text-xs text-gray-400">
              <div className="flex justify-between">
                <span>ERC-8004 Rep</span>
                <span className="text-white font-medium">
                  {agent.reputationScore.toFixed(1)} / 10
                </span>
              </div>
              <div className="flex justify-between">
                <span>Min Grade</span>
                <span className="text-yellow-300 font-medium">
                  {agent.minGrade}
                </span>
              </div>
              <div className="flex justify-between">
                <span>Kelly Fraction</span>
                <span className="text-white">
                  {(agent.kellyFraction * 100).toFixed(0)}%
                </span>
              </div>
            </div>

            {/* Reputation bar */}
            <div className="mt-3">
              <div className="w-full bg-gray-700 rounded-full h-1.5">
                <div
                  className="bg-blue-500 h-1.5 rounded-full"
                  style={{ width: `${(agent.reputationScore / 10) * 100}%` }}
                />
              </div>
            </div>
          </div>
        ))}
      </div>

      <p className="text-xs text-gray-500 mt-3">
        Requires 2/3 agent agreement to execute. Sizes weighted by ERC-8004
        reputation score.
      </p>
    </div>
  );
}
