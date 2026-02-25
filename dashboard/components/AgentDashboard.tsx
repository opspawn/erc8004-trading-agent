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
