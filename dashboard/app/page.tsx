import AgentDashboard from "@/components/AgentDashboard";

export default function Home() {
  return (
    <main className="min-h-screen bg-gray-950 text-white">
      <div className="container mx-auto px-4 py-8">
        <header className="mb-8">
          <h1 className="text-3xl font-bold text-blue-400">ERC-8004 Trading Agent</h1>
          <p className="text-gray-400 mt-1">
            Autonomous AI agent with on-chain identity, reputation, and validation
          </p>
        </header>
        <AgentDashboard />
      </div>
    </main>
  );
}
