"use client";

interface ReputationCardProps {
  score: number;   // 0-10
  feedbackCount: number;
}

export default function ReputationCard({ score, feedbackCount }: ReputationCardProps) {
  const scorePercent = (score / 10) * 100;
  const color =
    score >= 8 ? "bg-green-400" : score >= 6 ? "bg-yellow-400" : "bg-red-400";

  return (
    <div className="bg-gray-900 rounded-lg p-6 border border-gray-800">
      <h2 className="text-lg font-semibold text-gray-200 mb-4">On-Chain Reputation</h2>
      <div className="flex items-end gap-4 mb-4">
        <div>
          <span className="text-5xl font-bold text-white">{score.toFixed(1)}</span>
          <span className="text-gray-400 text-lg">/10</span>
        </div>
        <div className="text-gray-400 text-sm mb-2">
          based on {feedbackCount} feedback entries
        </div>
      </div>
      <div className="w-full bg-gray-800 rounded-full h-3">
        <div
          className={`h-3 rounded-full transition-all ${color}`}
          style={{ width: `${scorePercent}%` }}
        />
      </div>
      <div className="mt-3 text-xs text-gray-500">
        Reputation stored on-chain via ERC-8004 ReputationRegistry
      </div>
    </div>
  );
}
