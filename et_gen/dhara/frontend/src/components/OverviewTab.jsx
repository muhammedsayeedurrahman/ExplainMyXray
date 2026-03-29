import { motion } from 'framer-motion'
import CountUp from 'react-countup'
import { PieChart, Pie, Cell, ResponsiveContainer, BarChart, Bar, XAxis, YAxis, Tooltip } from 'recharts'

const formatINR = (val) => {
  const num = parseFloat(val) || 0
  if (num >= 10000000) return `${(num / 10000000).toFixed(1)}Cr`
  if (num >= 100000) return `${(num / 100000).toFixed(1)}L`
  if (num >= 1000) return `${(num / 1000).toFixed(1)}K`
  return num.toFixed(0)
}

const RISK_COLORS = {
  critical: '#ef4444',
  high: '#f97316',
  medium: '#eab308',
  low: '#22c55e',
  compliant: '#10b981',
}

function getScoreColor(score) {
  if (score >= 80) return { ring: '#10b981', text: 'text-emerald-400', bg: 'from-emerald-600/20 to-emerald-900/20', label: 'Excellent' }
  if (score >= 60) return { ring: '#22c55e', text: 'text-green-400', bg: 'from-green-600/20 to-green-900/20', label: 'Good' }
  if (score >= 40) return { ring: '#eab308', text: 'text-yellow-400', bg: 'from-yellow-600/20 to-yellow-900/20', label: 'Moderate Risk' }
  if (score >= 20) return { ring: '#f97316', text: 'text-orange-400', bg: 'from-orange-600/20 to-orange-900/20', label: 'High Risk' }
  return { ring: '#ef4444', text: 'text-red-400', bg: 'from-red-600/20 to-red-900/20', label: 'Critical' }
}

function KPICard({ icon, value, label, color, prefix, suffix, format, delay }) {
  const num = parseFloat(value) || 0
  const colorMap = {
    blue: 'from-blue-600/15 to-blue-900/15 border-blue-500/25',
    orange: 'from-orange-600/15 to-orange-900/15 border-orange-500/25',
    red: 'from-red-600/15 to-red-900/15 border-red-500/25',
    purple: 'from-purple-600/15 to-purple-900/15 border-purple-500/25',
    green: 'from-green-600/15 to-green-900/15 border-green-500/25',
    cyan: 'from-cyan-600/15 to-cyan-900/15 border-cyan-500/25',
  }

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ delay, duration: 0.4 }}
      className={`glass-card p-4 border bg-gradient-to-br ${colorMap[color]} hover:scale-[1.02] transition-transform cursor-default`}
    >
      <div className="flex items-center justify-between mb-2">
        <span className="text-xl">{icon}</span>
        <span className="text-[10px] text-gray-500 uppercase tracking-wider font-mono">{label}</span>
      </div>
      <div className="text-2xl font-bold font-mono">
        {prefix || ''}
        {format === 'inr' ? formatINR(value) : <CountUp end={num} duration={1.2} separator="," />}
        {suffix || ''}
      </div>
    </motion.div>
  )
}

function MiniScoreGauge({ score }) {
  const { ring, text, label } = getScoreColor(score)
  const circumference = 2 * Math.PI * 40
  const offset = circumference - (score / 100) * circumference

  return (
    <div className="flex flex-col items-center">
      <div className="relative w-24 h-24">
        <svg className="w-full h-full -rotate-90" viewBox="0 0 100 100">
          <circle cx="50" cy="50" r="40" fill="none" stroke="#1f2937" strokeWidth="6" />
          <motion.circle
            cx="50" cy="50" r="40" fill="none"
            stroke={ring}
            strokeWidth="6"
            strokeLinecap="round"
            strokeDasharray={circumference}
            initial={{ strokeDashoffset: circumference }}
            animate={{ strokeDashoffset: offset }}
            transition={{ duration: 1.5, ease: 'easeOut', delay: 0.3 }}
          />
        </svg>
        <div className="absolute inset-0 flex flex-col items-center justify-center">
          <span className={`text-xl font-bold font-mono ${text}`}>{score}</span>
        </div>
      </div>
      <span className={`text-xs font-semibold mt-1 ${text}`}>{label}</span>
    </div>
  )
}

function RiskDistribution({ vendors }) {
  if (!vendors?.length) return null

  const counts = {}
  for (const v of vendors) {
    const level = v.risk_level || 'unknown'
    counts[level] = (counts[level] || 0) + 1
  }

  const data = Object.entries(counts).map(([level, count]) => ({
    name: level.charAt(0).toUpperCase() + level.slice(1),
    value: count,
    color: RISK_COLORS[level] || '#6b7280',
  }))

  return (
    <div className="flex items-center gap-4">
      <div className="w-20 h-20">
        <ResponsiveContainer width="100%" height="100%">
          <PieChart>
            <Pie data={data} dataKey="value" cx="50%" cy="50%" innerRadius={22} outerRadius={36} paddingAngle={2} strokeWidth={0}>
              {data.map((entry, i) => (
                <Cell key={i} fill={entry.color} />
              ))}
            </Pie>
          </PieChart>
        </ResponsiveContainer>
      </div>
      <div className="space-y-1">
        {data.map(d => (
          <div key={d.name} className="flex items-center gap-2 text-xs">
            <div className="w-2 h-2 rounded-full" style={{ backgroundColor: d.color }} />
            <span className="text-gray-400">{d.name}</span>
            <span className="text-gray-200 font-mono font-bold">{d.value}</span>
          </div>
        ))}
      </div>
    </div>
  )
}

function TopVendorsBar({ vendors }) {
  if (!vendors?.length) return null

  const top5 = [...vendors]
    .sort((a, b) => b.risk_score - a.risk_score)
    .slice(0, 5)
    .map(v => ({
      name: v.vendor_name.length > 12 ? v.vendor_name.slice(0, 12) + '...' : v.vendor_name,
      risk: v.risk_score,
      overdue: parseFloat(v.overdue_amount) || 0,
      color: RISK_COLORS[v.risk_level] || '#6b7280',
    }))

  return (
    <div className="h-40">
      <ResponsiveContainer width="100%" height="100%">
        <BarChart data={top5} layout="vertical" margin={{ left: 0, right: 10, top: 0, bottom: 0 }}>
          <XAxis type="number" hide />
          <YAxis type="category" dataKey="name" width={90} tick={{ fill: '#9ca3af', fontSize: 11 }} />
          <Tooltip
            contentStyle={{ backgroundColor: '#111827', border: '1px solid #374151', borderRadius: '12px', fontSize: '12px' }}
            labelStyle={{ color: '#e5e7eb' }}
            formatter={(val) => [`Risk: ${val}/100`, '']}
          />
          <Bar dataKey="risk" radius={[0, 6, 6, 0]} barSize={16}>
            {top5.map((entry, i) => (
              <Cell key={i} fill={entry.color} fillOpacity={0.7} />
            ))}
          </Bar>
        </BarChart>
      </ResponsiveContainer>
    </div>
  )
}

export default function OverviewTab({ summary, vendors, actions, impact, onNavigate }) {
  if (!summary) return null

  const kpis = [
    { icon: '\u{1F3ED}', value: summary.msme_vendors, label: 'MSME Vendors', color: 'blue', format: 'num' },
    { icon: '\u26A0\uFE0F', value: summary.overdue_invoices, label: 'Overdue Invoices', color: 'orange', format: 'num' },
    { icon: '\u{1F4B0}', value: summary.total_overdue_amount, label: 'Overdue Amount', color: 'red', format: 'inr', prefix: 'INR ' },
    { icon: '\u{1F4C8}', value: summary.total_interest_liability, label: 'Interest Liability', color: 'purple', format: 'inr', prefix: 'INR ' },
    { icon: '\u{1F6E1}\uFE0F', value: summary.vendors_at_risk, label: 'At Risk', color: 'cyan', format: 'num' },
  ]

  const topActions = (actions || []).slice(0, 4)

  return (
    <div className="space-y-6">
      {/* KPI Cards */}
      <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-5 gap-3">
        {kpis.map((kpi, i) => (
          <KPICard key={kpi.label} {...kpi} delay={i * 0.08} />
        ))}
      </div>

      {/* Middle row: Score + Risk Distribution + Top Vendors */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        {/* Compliance Score */}
        <motion.div
          initial={{ opacity: 0, scale: 0.95 }}
          animate={{ opacity: 1, scale: 1 }}
          transition={{ delay: 0.3 }}
          className="glass-card p-5 flex flex-col items-center justify-center border border-gray-800/50"
        >
          <h3 className="text-xs text-gray-500 uppercase tracking-wider mb-3">Compliance Score</h3>
          <MiniScoreGauge score={summary.compliance_score} />
          {impact && (
            <div className="mt-3 flex items-center gap-2 text-xs">
              <span className="text-gray-500">After remediation:</span>
              <span className="text-green-400 font-bold font-mono">
                {Math.min(100, summary.compliance_score + 40)}
              </span>
            </div>
          )}
        </motion.div>

        {/* Risk Distribution */}
        <motion.div
          initial={{ opacity: 0, scale: 0.95 }}
          animate={{ opacity: 1, scale: 1 }}
          transition={{ delay: 0.4 }}
          className="glass-card p-5 border border-gray-800/50"
        >
          <h3 className="text-xs text-gray-500 uppercase tracking-wider mb-3">Risk Distribution</h3>
          <RiskDistribution vendors={vendors} />
          <button
            onClick={() => onNavigate('risk')}
            className="mt-3 text-xs text-blue-400 hover:text-blue-300 transition-colors"
          >
            View full heatmap &rarr;
          </button>
        </motion.div>

        {/* Top Risk Vendors */}
        <motion.div
          initial={{ opacity: 0, scale: 0.95 }}
          animate={{ opacity: 1, scale: 1 }}
          transition={{ delay: 0.5 }}
          className="glass-card p-5 border border-gray-800/50"
        >
          <h3 className="text-xs text-gray-500 uppercase tracking-wider mb-3">Highest Risk Vendors</h3>
          <TopVendorsBar vendors={vendors} />
          <button
            onClick={() => onNavigate('vendors')}
            className="mt-2 text-xs text-blue-400 hover:text-blue-300 transition-colors"
          >
            View all vendors &rarr;
          </button>
        </motion.div>
      </div>

      {/* Bottom row: Quick Actions + Financial Impact */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        {/* Quick Actions */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.6 }}
          className="glass-card p-5 border border-gray-800/50"
        >
          <div className="flex items-center justify-between mb-3">
            <h3 className="text-xs text-gray-500 uppercase tracking-wider">Priority Actions</h3>
            <button
              onClick={() => onNavigate('actions')}
              className="text-xs text-blue-400 hover:text-blue-300 transition-colors"
            >
              View all &rarr;
            </button>
          </div>
          <div className="space-y-2">
            {topActions.map((a, i) => (
              <motion.div
                key={a.action_id || i}
                initial={{ opacity: 0, x: -10 }}
                animate={{ opacity: 1, x: 0 }}
                transition={{ delay: 0.7 + i * 0.05 }}
                className="flex items-center gap-3 p-2.5 rounded-lg bg-gray-800/30 hover:bg-gray-800/50 transition-colors"
              >
                <span className={`text-[10px] font-mono font-bold px-1.5 py-0.5 rounded uppercase ${
                  a.priority === 'critical' ? 'bg-red-500/20 text-red-400' :
                  a.priority === 'high' ? 'bg-orange-500/20 text-orange-400' :
                  'bg-yellow-500/20 text-yellow-400'
                }`}>
                  {a.priority}
                </span>
                <span className="text-xs text-gray-300 flex-1 truncate">{a.description}</span>
                <span className="text-xs font-mono text-green-400 flex-shrink-0">
                  INR {formatINR(a.financial_impact)}
                </span>
              </motion.div>
            ))}
            {topActions.length === 0 && (
              <p className="text-xs text-gray-600 text-center py-4">No actions generated yet</p>
            )}
          </div>
        </motion.div>

        {/* Financial Impact Summary */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.7 }}
          className="glass-card p-5 border border-green-500/20 bg-gradient-to-br from-green-900/10 to-emerald-900/10"
        >
          <h3 className="text-xs text-gray-500 uppercase tracking-wider mb-4">Financial Impact</h3>
          {impact ? (
            <div className="space-y-3">
              <div className="text-center mb-4">
                <div className="text-xs text-gray-400">Total Savings</div>
                <div className="text-3xl font-bold text-green-400 font-mono">
                  INR <CountUp end={parseFloat(impact.total_financial_impact) || 0} duration={2} separator="," decimals={0} />
                </div>
              </div>
              <div className="grid grid-cols-2 gap-2">
                {[
                  { label: 'Penalties Avoided', value: impact.penalties_avoided, color: 'text-green-400' },
                  { label: 'Interest Saved', value: impact.interest_saved, color: 'text-blue-400' },
                  { label: 'Tax Deductions', value: impact.tax_deductions_preserved, color: 'text-purple-400' },
                  { label: 'Hours Saved', value: impact.hours_saved, color: 'text-cyan-400', suffix: 'hrs' },
                ].map(item => (
                  <div key={item.label} className="bg-gray-800/30 rounded-lg p-2.5">
                    <div className="text-[10px] text-gray-500">{item.label}</div>
                    <div className={`text-sm font-bold font-mono ${item.color}`}>
                      {item.suffix ? item.value + ' ' + item.suffix : 'INR ' + formatINR(item.value)}
                    </div>
                  </div>
                ))}
              </div>
            </div>
          ) : (
            <p className="text-xs text-gray-600 text-center py-4">Impact calculated after analysis</p>
          )}
        </motion.div>
      </div>
    </div>
  )
}
