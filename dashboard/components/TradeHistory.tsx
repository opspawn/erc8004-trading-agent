"use client";

interface Trade {
  id: string;
  market: string;
  side: "YES" | "NO";
  size: number;
  outcome: "WIN" | "LOSS" | "PENDING";
  pnl: number;
  validationScore?: number;
  timestamp: string;
}

const MOCK_TRADES: Trade[] = [
  {
    id: "trade-001",
    market: "Will BTC exceed $100k by March 31?",
    side: "YES",
    size: 5.0,
    outcome: "PENDING",
    pnl: 0,
    timestamp: "2026-02-24T10:30:00Z",
  },
  {
    id: "trade-002",
    market: "Will Fed cut rates in March 2026?",
    side: "NO",
    size: 3.5,
    outcome: "WIN",
    pnl: 2.1,
    validationScore: 85,
    timestamp: "2026-02-23T15:45:00Z",
  },
  {
    id: "trade-003",
    market: "Will SpaceX Starship succeed in 2026?",
    side: "YES",
    size: 2.0,
    outcome: "PENDING",
    pnl: 0,
    timestamp: "2026-02-23T09:20:00Z",
  },
];

const outcomeColors = {
  WIN: "text-green-400",
  LOSS: "text-red-400",
  PENDING: "text-yellow-400",
};

export default function TradeHistory() {
  return (
    <div className="bg-gray-900 rounded-lg border border-gray-800 overflow-hidden">
      <div className="px-6 py-4 border-b border-gray-800">
        <h2 className="text-lg font-semibold text-gray-200">Recent Trades</h2>
      </div>
      <div className="divide-y divide-gray-800">
        {MOCK_TRADES.map((trade) => (
          <div key={trade.id} className="px-6 py-4">
            <div className="flex justify-between items-start">
              <div className="flex-1 min-w-0">
                <p className="text-white text-sm truncate">{trade.market}</p>
                <div className="flex items-center gap-3 mt-1">
                  <span
                    className={`text-xs px-2 py-0.5 rounded font-mono ${
                      trade.side === "YES"
                        ? "bg-green-900 text-green-300"
                        : "bg-red-900 text-red-300"
                    }`}
                  >
                    {trade.side}
                  </span>
                  <span className="text-gray-400 text-xs">${trade.size.toFixed(2)}</span>
                  {trade.validationScore !== undefined && (
                    <span className="text-gray-500 text-xs">
                      Validation: {trade.validationScore}/100
                    </span>
                  )}
                </div>
              </div>
              <div className="text-right ml-4">
                <p className={`font-semibold ${outcomeColors[trade.outcome]}`}>
                  {trade.outcome}
                </p>
                {trade.outcome !== "PENDING" && (
                  <p
                    className={`text-sm ${
                      trade.pnl >= 0 ? "text-green-400" : "text-red-400"
                    }`}
                  >
                    {trade.pnl >= 0 ? "+" : ""}${trade.pnl.toFixed(2)}
                  </p>
                )}
              </div>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}
