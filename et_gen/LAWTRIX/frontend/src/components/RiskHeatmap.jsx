import { useState } from 'react'
import { motion } from 'framer-motion'
import { Treemap, ResponsiveContainer, Tooltip } from 'recharts'

const RISK_COLORS = {
  critical: '#ef4444',
  high: '#f97316',
  medium: '#eab308',
  low: '#22c55e',
  compliant: '#10b981',
}

const RISK_BG = {
  critical: 'bg-red-500',
  high: 'bg-orange-500',
  medium: 'bg-yellow-500',
  low: 'bg-green-500',
  compliant: 'bg-emerald-500',
}

function CustomTreemapContent({ x, y, width, height, name, risk_level, risk_score }) {
  if (width < 30 || height < 20) return null

  return (
    <g>
      <rect
        x={x}
        y={y}
        width={width}
        height={height}
        rx={4}
        fill={RISK_COLORS[risk_level] || '#6b7280'}
        fillOpacity={0.6}
        stroke={RISK_COLORS[risk_level] || '#6b7280'}
        strokeWidth={1.5}
        strokeOpacity={0.8}
        className="hover:fill-opacity-90 transition-all cursor-pointer"
      />
      {width > 60 && height > 35 && (
        <>
          <text
            x={x + width / 2}
            y={y + height / 2 - 6}
            textAnchor="middle"
            fill="white"
            fontSize={Math.min(11, width / 8)}
            fontWeight="600"
          >
            {name?.length > 12 ? name.slice(0, 12) + '...' : name}
          </text>
          <text
            x={x + width / 2}
            y={y + height / 2 + 10}
            textAnchor="middle"
            fill="rgba(255,255,255,0.7)"
            fontSize={Math.min(10, width / 10)}
            fontFamily="monospace"
          >
            {risk_score}/100
          </text>
        </>
      )}
      {width > 30 && width <= 60 && height > 20 && (
        <text
          x={x + width / 2}
          y={y + height / 2 + 4}
          textAnchor="middle"
          fill="white"
          fontSize={10}
          fontWeight="bold"
        >
          {risk_score}
        </text>
      )}
    </g>
  )
}

function CustomTooltip({ active, payload }) {
  if (!active || !payload?.length) return null
  const data = payload[0]?.payload

  return (
    <div className="bg-gray-900 border border-gray-700 rounded-xl p-3 shadow-xl text-sm">
      <div className="font-semibold text-white">{data?.name}</div>
      <div className="mt-1 space-y-1 text-xs">
        <div className="flex justify-between gap-4">
          <span className="text-gray-400">Risk Score</span>
          <span className="text-white font-mono">{data?.risk_score}/100</span>
        </div>
        <div className="flex justify-between gap-4">
          <span className="text-gray-400">Risk Level</span>
          <span style={{ color: RISK_COLORS[data?.risk_level] }} className="font-medium capitalize">
            {data?.risk_level}
          </span>
        </div>
        <div className="flex justify-between gap-4">
          <span className="text-gray-400">Overdue Amount</span>
          <span className="text-yellow-400 font-mono">
            INR {parseFloat(data?.overdue_amount || 0).toLocaleString('en-IN')}
          </span>
        </div>
        <div className="flex justify-between gap-4">
          <span className="text-gray-400">Interest Liability</span>
          <span className="text-red-400 font-mono">
            INR {parseFloat(data?.interest_liability || 0).toLocaleString('en-IN')}
          </span>
        </div>
      </div>
    </div>
  )
}

export default function RiskHeatmap({ vendors, visible, onVendorFilter }) {
  const [activeLevel, setActiveLevel] = useState(null)

  if (!visible || !vendors?.length) return null

  // Treemap data: size = overdue amount (minimum 1 for visibility), color = risk level
  const treemapData = vendors
    .filter(v => activeLevel ? v.risk_level === activeLevel : true)
    .map(v => ({
      name: v.vendor_name,
      size: Math.max(parseFloat(v.overdue_amount || 0), 1),
      risk_level: v.risk_level,
      risk_score: v.risk_score,
      overdue_amount: v.overdue_amount,
      interest_liability: v.interest_liability,
      vendor_id: v.vendor_id,
    }))

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      className="glass-card p-6"
    >
      <div className="flex items-center justify-between mb-4">
        <h2 className="text-lg font-semibold">Risk Heatmap</h2>
        <div className="text-xs text-gray-500">
          Size = overdue amount | Color = risk level
        </div>
      </div>

      {/* Treemap */}
      <div className="h-64 w-full">
        <ResponsiveContainer width="100%" height="100%">
          <Treemap
            data={treemapData}
            dataKey="size"
            aspectRatio={4 / 3}
            content={<CustomTreemapContent />}
          >
            <Tooltip content={<CustomTooltip />} />
          </Treemap>
        </ResponsiveContainer>
      </div>

      {/* Legend with filter */}
      <div className="flex items-center gap-4 mt-4 text-xs text-gray-400">
        <button
          onClick={() => setActiveLevel(null)}
          className={`flex items-center gap-1.5 px-2 py-1 rounded transition-colors ${
            !activeLevel ? 'bg-gray-700 text-white' : 'hover:bg-gray-800'
          }`}
        >
          All
        </button>
        {['critical', 'high', 'medium', 'low', 'compliant'].map(level => {
          const count = vendors.filter(v => v.risk_level === level).length
          if (count === 0) return null
          return (
            <button
              key={level}
              onClick={() => setActiveLevel(activeLevel === level ? null : level)}
              className={`flex items-center gap-1.5 px-2 py-1 rounded transition-colors ${
                activeLevel === level ? 'bg-gray-700 text-white' : 'hover:bg-gray-800'
              }`}
            >
              <div className={`w-3 h-3 rounded ${RISK_BG[level]}`} />
              <span className="capitalize">{level}</span>
              <span className="text-gray-600">({count})</span>
            </button>
          )
        })}
      </div>
    </motion.div>
  )
}
