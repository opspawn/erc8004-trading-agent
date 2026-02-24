"use client";

interface AgentIdentityProps {
  agentId: number;
  did: string;
}

export default function AgentIdentity({ agentId, did }: AgentIdentityProps) {
  return (
    <div className="bg-gray-900 rounded-lg p-6 border border-blue-800">
      <div className="flex items-center gap-3 mb-3">
        <div className="w-3 h-3 bg-green-400 rounded-full animate-pulse" />
        <h2 className="text-lg font-semibold text-blue-300">Agent Identity</h2>
        <span className="ml-auto text-xs bg-blue-900 text-blue-300 px-2 py-1 rounded">
          ERC-8004
        </span>
      </div>
      <div className="space-y-2">
        <div className="flex">
          <span className="text-gray-400 w-24 text-sm">Agent ID:</span>
          <span className="text-white font-mono">#{agentId}</span>
        </div>
        <div className="flex">
          <span className="text-gray-400 w-24 text-sm">DID:</span>
          <span className="text-blue-300 font-mono text-sm break-all">{did}</span>
        </div>
        <div className="flex">
          <span className="text-gray-400 w-24 text-sm">Status:</span>
          <span className="text-green-400 text-sm">Active — trading prediction markets</span>
        </div>
      </div>
    </div>
  );
}
