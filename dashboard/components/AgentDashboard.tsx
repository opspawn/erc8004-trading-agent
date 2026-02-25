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
