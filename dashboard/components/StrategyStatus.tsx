"use client";

import { useState } from "react";

interface Strategy {
  name: string;
  status: "ACTIVE" | "PAUSED" | "STOPPED";
  type: string;
  winRate: number;
  trades: number;
  credoraRating: string;
}

const MOCK_STRATEGIES: Strategy[] = [
  {
    name: "Kelly Criterion — ETH/USD",
    status: "ACTIVE",
    type: "Trend-following",
    winRate: 0.62,
    trades: 23,
    credoraRating: "AA",
  },
  {
    name: "Mean Reversion — BTC Dominance",
    status: "ACTIVE",
    type: "Statistical arbitrage",
    winRate: 0.58,
    trades: 14,
    credoraRating: "A",
  },
  {
    name: "Macro Sentiment — Rate Markets",
    status: "PAUSED",
    type: "News-driven",
    winRate: 0.71,
    trades: 10,
    credoraRating: "BBB",
  },
];

const statusColors = {
  ACTIVE: "bg-green-900 text-green-300 border-green-700",
  PAUSED: "bg-yellow-900 text-yellow-300 border-yellow-700",
  STOPPED: "bg-red-900 text-red-300 border-red-700",
};

const ratingColors: Record<string, string> = {
  AAA: "text-emerald-400",
  AA: "text-green-400",
  A: "text-blue-400",
  BBB: "text-yellow-400",
  BB: "text-orange-400",
  B: "text-red-400",
  CCC: "text-red-600",
};

export default function StrategyStatus() {
  const [strategies] = useState<Strategy[]>(MOCK_STRATEGIES);
  const activeCount = strategies.filter((s) => s.status === "ACTIVE").length;

  return (
    <div className="bg-gray-900 rounded-lg border border-gray-800">
      <div className="px-6 py-4 border-b border-gray-800 flex items-center justify-between">
        <h2 className="text-lg font-semibold text-gray-200">Strategy Status</h2>
        <span className="text-xs text-gray-400">
          {activeCount}/{strategies.length} active
        </span>
      </div>
      <div className="divide-y divide-gray-800">
        {strategies.map((strategy) => (
          <div key={strategy.name} className="px-6 py-4">
            <div className="flex items-start justify-between gap-4">
              <div className="flex-1 min-w-0">
                <p className="text-white text-sm font-medium truncate">
                  {strategy.name}
                </p>
                <div className="flex items-center gap-2 mt-1 flex-wrap">
                  <span className="text-gray-500 text-xs">{strategy.type}</span>
                  <span className="text-gray-600 text-xs">•</span>
                  <span className="text-gray-400 text-xs">
                    {strategy.trades} trades
                  </span>
                  <span className="text-gray-600 text-xs">•</span>
                  <span className="text-gray-400 text-xs">
                    {(strategy.winRate * 100).toFixed(0)}% win
                  </span>
                  <span className="text-gray-600 text-xs">•</span>
                  <span className="text-xs text-gray-500">
                    Credora:{" "}
                    <span
                      className={`font-bold ${
                        ratingColors[strategy.credoraRating] ?? "text-gray-400"
                      }`}
                    >
                      {strategy.credoraRating}
                    </span>
                  </span>
                </div>
              </div>
              <span
                className={`text-xs px-2 py-1 rounded border font-medium ${
                  statusColors[strategy.status]
                }`}
              >
                {strategy.status}
              </span>
            </div>
          </div>
        ))}
      </div>
      <div className="px-6 py-3 bg-gray-900/50 border-t border-gray-800 rounded-b-lg">
        <p className="text-xs text-gray-500">
          Credora ratings sourced from{" "}
          <span className="text-blue-400 font-mono">credora.io</span> — used as
          Kelly fraction multiplier (AAA=1.0 → CCC=0.3)
        </p>
      </div>
    </div>
  );
}
