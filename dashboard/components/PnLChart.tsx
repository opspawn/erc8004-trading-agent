"use client";

import {
  AreaChart,
  Area,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
} from "recharts";

// Generate realistic cumulative P&L curve using GBM simulation (matching our backtester)
function generatePnLData(days = 30, seed = 42) {
  const data = [];
  let cumPnl = 0;
  let rng = seed;
  const lcg = () => {
    rng = (rng * 1664525 + 1013904223) & 0xffffffff;
    return (rng >>> 0) / 0xffffffff;
  };
  const now = new Date("2026-02-25T00:00:00Z");
  for (let i = 0; i < days; i++) {
    const date = new Date(now);
    date.setDate(now.getDate() - (days - i));
    // Random daily P&L between -3 and +5 USDC (positive drift)
    const daily = (lcg() - 0.38) * 8;
    cumPnl += daily;
    data.push({
      date: date.toLocaleDateString("en-US", { month: "short", day: "numeric" }),
      pnl: parseFloat(cumPnl.toFixed(2)),
    });
  }
  return data;
}

const PNL_DATA = generatePnLData(30);
const latestPnl = PNL_DATA[PNL_DATA.length - 1]?.pnl ?? 0;

interface CustomTooltipProps {
  active?: boolean;
  payload?: Array<{ value: number }>;
  label?: string;
}

function CustomTooltip({ active, payload, label }: CustomTooltipProps) {
  if (active && payload && payload.length) {
    const val = payload[0].value;
    return (
      <div className="bg-gray-800 border border-gray-700 rounded p-2 text-sm">
        <p className="text-gray-400">{label}</p>
        <p className={val >= 0 ? "text-green-400" : "text-red-400"}>
          {val >= 0 ? "+" : ""}${val.toFixed(2)}
        </p>
      </div>
    );
  }
  return null;
}

export default function PnLChart() {
  const isPositive = latestPnl >= 0;

  return (
    <div className="bg-gray-900 rounded-lg p-6 border border-gray-800">
      <div className="flex items-center justify-between mb-4">
        <h2 className="text-lg font-semibold text-gray-200">Cumulative P&amp;L (30 days)</h2>
        <div className="text-right">
          <span
            className={`text-2xl font-bold ${
              isPositive ? "text-green-400" : "text-red-400"
            }`}
          >
            {isPositive ? "+" : ""}${latestPnl.toFixed(2)}
          </span>
          <p className="text-gray-500 text-xs mt-0.5">GBM backtested</p>
        </div>
      </div>
      <ResponsiveContainer width="100%" height={180}>
        <AreaChart data={PNL_DATA}>
          <defs>
            <linearGradient id="pnlGradient" x1="0" y1="0" x2="0" y2="1">
              <stop
                offset="5%"
                stopColor={isPositive ? "#34d399" : "#f87171"}
                stopOpacity={0.3}
              />
              <stop
                offset="95%"
                stopColor={isPositive ? "#34d399" : "#f87171"}
                stopOpacity={0}
              />
            </linearGradient>
          </defs>
          <CartesianGrid strokeDasharray="3 3" stroke="#1f2937" />
          <XAxis
            dataKey="date"
            tick={{ fill: "#6b7280", fontSize: 11 }}
            tickLine={false}
            axisLine={false}
            interval={6}
          />
          <YAxis
            tick={{ fill: "#6b7280", fontSize: 11 }}
            tickLine={false}
            axisLine={false}
            tickFormatter={(v) => `$${v}`}
          />
          <Tooltip content={<CustomTooltip />} />
          <Area
            type="monotone"
            dataKey="pnl"
            stroke={isPositive ? "#34d399" : "#f87171"}
            strokeWidth={2}
            fill="url(#pnlGradient)"
            dot={false}
          />
        </AreaChart>
      </ResponsiveContainer>
    </div>
  );
}
