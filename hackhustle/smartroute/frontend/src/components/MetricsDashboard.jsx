import { motion } from 'framer-motion'
import CountUp from 'react-countup'
import { PieChart, Pie, Cell, ResponsiveContainer, Tooltip } from 'recharts'

const container = { hidden: { opacity: 0 }, show: { opacity: 1, transition: { staggerChildren: 0.08 } } }
const item = { hidden: { y: 20, opacity: 0 }, show: { y: 0, opacity: 1, transition: { duration: 0.4 } } }

const PIE_COLORS = ['#34d399', '#fbbf24', '#f87171']

function MetricCard({ label, value, sub, color, icon, prefix = '', suffix = '', decimals = 0 }) {
  return (
    <motion.div
      variants={item}
      whileHover={{ y: -4, boxShadow: '0 8px 32px rgba(0,0,0,0.1), 0 2px 8px rgba(0,0,0,0.06)' }}
      className="bg-white rounded-2xl p-5 shadow-card border border-gray-100 cursor-default"
    >
      <div className="flex items-start justify-between mb-3">
        <span className="text-[11px] font-semibold text-gray-400 uppercase tracking-wider">{label}</span>
        <span className={`w-9 h-9 rounded-xl flex items-center justify-center shadow-sm ${color}`}>{icon}</span>
      </div>
      <p className="text-2xl font-bold text-gray-900">
        {prefix}
        {typeof value === 'number'
          ? <CountUp end={value} duration={1.2} separator="," decimals={decimals} />
          : value}
        {suffix}
      </p>
      {sub && <p className="text-[11px] text-gray-400 mt-1.5">{sub}</p>}
    </motion.div>
  )
}

function ProgressBar({ label, value, color, delay = 0 }) {
  return (
    <div>
      <div className="flex justify-between text-[11px] text-gray-500 mb-1.5">
        <span className="font-medium">{label}</span>
        <span className="font-semibold">{value}%</span>
      </div>
      <div className="h-2.5 bg-gray-100 rounded-full overflow-hidden">
        <motion.div
          initial={{ width: 0 }}
          animate={{ width: `${value}%` }}
          transition={{ duration: 1.2, ease: 'easeOut', delay }}
          className={`h-full bg-gradient-to-r ${color} rounded-full`}
        />
      </div>
    </div>
  )
}

function CustomTooltip({ active, payload }) {
  if (!active || !payload?.length) return null
  const d = payload[0]
  return (
    <div className="bg-white rounded-lg shadow-float border border-gray-100 px-3 py-2 text-xs">
      <span className="font-semibold text-gray-700">{d.name}: </span>
      <span className="text-gray-500">{d.value} orders</span>
    </div>
  )
}

export default function MetricsDashboard({ metrics, loading, optimized }) {
  if (loading) {
    return (
      <div className="grid grid-cols-2 lg:grid-cols-5 gap-4">
        {[...Array(5)].map((_, i) => (
          <div key={i} className="bg-white rounded-2xl p-5 shadow-card border border-gray-100 animate-pulse">
            <div className="h-3 bg-gray-100 rounded w-20 mb-4" />
            <div className="h-7 bg-gray-100 rounded w-16" />
          </div>
        ))}
      </div>
    )
  }

  if (!metrics) return null

  const { score_distribution: dist } = metrics

  const pieData = [
    { name: 'High (80-100)', value: dist.green },
    { name: 'Medium (50-79)', value: dist.yellow },
    { name: 'Low (0-49)', value: dist.red },
  ]

  return (
    <div className="space-y-6">
      {/* KPI Cards */}
      <motion.div variants={container} initial="hidden" animate="show" className="grid grid-cols-2 lg:grid-cols-5 gap-4">
        <MetricCard
          label="Total Orders"
          value={metrics.total_orders}
          sub={`Avg score: ${metrics.average_score}`}
          color="bg-brand-50 text-brand-600"
          icon={<svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}><path strokeLinecap="round" strokeLinejoin="round" d="M20 7l-8-4-8 4m16 0l-8 4m8-4v10l-8 4m0-10L4 7m8 4v10M4 7v10l8 4" /></svg>}
        />
        <MetricCard
          label="At Risk"
          value={metrics.deliveries_at_risk}
          sub={`${metrics.failure_rate_before}% failure rate`}
          color="bg-red-50 text-red-500"
          icon={<svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}><path strokeLinecap="round" strokeLinejoin="round" d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-2.5L13.732 4.5c-.77-.833-2.694-.833-3.464 0L3.34 16.5c-.77.833.192 2.5 1.732 2.5z" /></svg>}
        />
        <MetricCard
          label="Trips Saved"
          value={metrics.trips_saved}
          sub={`${metrics.trips_before} → ${metrics.trips_after} trips`}
          color="bg-emerald-50 text-emerald-600"
          icon={<svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}><path strokeLinecap="round" strokeLinejoin="round" d="M13 7h8m0 0v8m0-8l-8 8-4-4-6 6" /></svg>}
        />
        <MetricCard
          label="Cost Saved"
          value={metrics.cost_saved}
          prefix="₹"
          sub={`${metrics.clusters_formed} clusters formed`}
          color="bg-blue-50 text-blue-600"
          icon={<svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}><path strokeLinecap="round" strokeLinejoin="round" d="M12 8c-1.657 0-3 .895-3 2s1.343 2 3 2 3 .895 3 2-1.343 2-3 2m0-8c1.11 0 2.08.402 2.599 1M12 8V7m0 1v8m0 0v1m0-1c-1.11 0-2.08-.402-2.599-1M21 12a9 9 0 11-18 0 9 9 0 0118 0z" /></svg>}
        />
        <MetricCard
          label="After AI Fix"
          value={metrics.failure_rate_after}
          suffix="%"
          decimals={1}
          sub={`From ${metrics.failure_rate_before}% failure`}
          color="bg-purple-50 text-purple-600"
          icon={<svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}><path strokeLinecap="round" strokeLinejoin="round" d="M9 12l2 2 4-4m5.618-4.016A11.955 11.955 0 0112 2.944a11.955 11.955 0 01-8.618 3.04A12.02 12.02 0 003 9c0 5.591 3.824 10.29 9 11.622 5.176-1.332 9-6.03 9-11.622 0-1.042-.133-2.052-.382-3.016z" /></svg>}
        />
      </motion.div>

      {/* Chart + Progress Bars row */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Address Distribution Chart */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.5 }}
          className="bg-white rounded-2xl p-5 shadow-card border border-gray-100 hover:shadow-card-hover transition-shadow duration-300"
        >
          <p className="text-[11px] font-semibold text-gray-400 uppercase tracking-wider mb-4">Address Quality Distribution</p>
          <div className="flex items-center gap-6">
            <div className="w-36 h-36">
              <ResponsiveContainer width="100%" height="100%">
                <PieChart>
                  <Pie
                    data={pieData}
                    cx="50%"
                    cy="50%"
                    innerRadius={38}
                    outerRadius={60}
                    paddingAngle={3}
                    dataKey="value"
                    strokeWidth={0}
                  >
                    {pieData.map((_, i) => (
                      <Cell key={i} fill={PIE_COLORS[i]} />
                    ))}
                  </Pie>
                  <Tooltip content={<CustomTooltip />} />
                </PieChart>
              </ResponsiveContainer>
            </div>
            <div className="flex-1 space-y-3">
              {pieData.map((d, i) => (
                <div key={d.name} className="flex items-center justify-between">
                  <div className="flex items-center gap-2">
                    <span className="w-3 h-3 rounded-full shadow-sm" style={{ backgroundColor: PIE_COLORS[i] }} />
                    <span className="text-xs text-gray-600">{d.name}</span>
                  </div>
                  <span className="text-xs font-bold text-gray-800">{d.value}</span>
                </div>
              ))}
            </div>
          </div>
        </motion.div>

        {/* Progress Bars */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.6 }}
          className="bg-white rounded-2xl p-5 shadow-card border border-gray-100 hover:shadow-card-hover transition-shadow duration-300"
        >
          <p className="text-[11px] font-semibold text-gray-400 uppercase tracking-wider mb-4">Performance Metrics</p>
          <div className="space-y-4">
            <ProgressBar
              label="Trip Efficiency"
              value={optimized ? metrics.trip_efficiency : 0}
              color="from-brand-400 to-emerald-400"
              delay={optimized ? 0.2 : 0}
            />
            <ProgressBar
              label="Address Verification"
              value={metrics.address_verification}
              color="from-brand-400 to-brand-500"
              delay={0.3}
            />
            <ProgressBar
              label="Failure Reduction"
              value={optimized ? metrics.failure_reduction : 0}
              color="from-purple-400 to-purple-500"
              delay={optimized ? 0.4 : 0}
            />
          </div>
        </motion.div>
      </div>

      {/* Industry Benchmarks */}
      {metrics.industry_benchmarks && (
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.7 }}
          className="bg-white rounded-2xl p-5 shadow-card border border-gray-100 hover:shadow-card-hover transition-shadow duration-300"
        >
          <div className="flex items-center justify-between mb-3">
            <p className="text-[11px] font-semibold text-gray-400 uppercase tracking-wider">India Logistics Industry Context</p>
            <span className="text-[10px] text-gray-300 font-medium">Source: SVH Research 2026</span>
          </div>
          <div className="grid grid-cols-2 lg:grid-cols-4 gap-3">
            {[
              { value: `${metrics.industry_benchmarks.last_mile_cost_pct}%`, label: 'Last-mile as % of total shipping cost' },
              { value: `${metrics.industry_benchmarks.rto_rate_tier2_3}%`, label: 'Avg RTO rate in Tier 2/3 cities' },
              { value: metrics.industry_benchmarks.daily_ecommerce_deliveries, label: 'Daily e-commerce deliveries in India' },
              { value: `${metrics.industry_benchmarks.logistics_gdp_pct}%`, label: `India logistics cost (vs ${metrics.industry_benchmarks.global_logistics_gdp_pct}% global avg)`, highlight: true },
            ].map((b) => (
              <motion.div
                key={b.label}
                whileHover={{ y: -3, boxShadow: '0 8px 24px rgba(0,0,0,0.08)' }}
                className="bg-gray-50 rounded-xl p-3.5 cursor-default"
              >
                <p className={`text-lg font-bold ${b.highlight ? 'text-brand-600' : 'text-gray-900'}`}>{b.value}</p>
                <p className="text-[11px] text-gray-500 mt-0.5 leading-tight">{b.label}</p>
              </motion.div>
            ))}
          </div>
        </motion.div>
      )}
    </div>
  )
}
