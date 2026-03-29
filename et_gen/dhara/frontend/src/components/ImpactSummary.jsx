import { motion } from 'framer-motion'
import CountUp from 'react-countup'

export default function ImpactSummary({ impact, visible }) {
  if (!visible || !impact) return null

  const cards = [
    {
      label: 'Penalties Prevented',
      value: parseFloat(impact.penalties_avoided) || 0,
      prefix: 'INR ',
      color: 'from-green-600/20 to-green-900/20 border-green-500/30',
      textColor: 'text-green-400',
      icon: '🛡️',
    },
    {
      label: 'Interest Saved',
      value: parseFloat(impact.interest_saved) || 0,
      prefix: 'INR ',
      color: 'from-blue-600/20 to-blue-900/20 border-blue-500/30',
      textColor: 'text-blue-400',
      icon: '💰',
    },
    {
      label: 'Tax Deductions Preserved',
      value: parseFloat(impact.tax_deductions_preserved) || 0,
      prefix: 'INR ',
      color: 'from-purple-600/20 to-purple-900/20 border-purple-500/30',
      textColor: 'text-purple-400',
      icon: '📊',
    },
    {
      label: 'Hours Saved',
      value: impact.hours_saved || 0,
      suffix: ' hrs',
      color: 'from-cyan-600/20 to-cyan-900/20 border-cyan-500/30',
      textColor: 'text-cyan-400',
      icon: '⏱️',
    },
  ]

  return (
    <motion.div
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      className="space-y-4"
    >
      <h2 className="text-lg font-semibold">Financial Impact</h2>

      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        {cards.map((card, i) => (
          <motion.div
            key={card.label}
            initial={{ opacity: 0, y: 30 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: i * 0.15, duration: 0.6 }}
            className={`glass-card p-5 border bg-gradient-to-br ${card.color} glow-green`}
          >
            <div className="text-2xl mb-2">{card.icon}</div>
            <div className={`text-xl font-bold font-mono ${card.textColor}`}>
              {card.prefix || ''}
              <CountUp
                end={card.value}
                duration={2}
                separator=","
                decimals={card.value >= 1000 ? 0 : 1}
              />
              {card.suffix || ''}
            </div>
            <div className="text-xs text-gray-400 mt-1">{card.label}</div>
          </motion.div>
        ))}
      </div>

      {/* Big total */}
      <motion.div
        initial={{ opacity: 0, scale: 0.95 }}
        animate={{ opacity: 1, scale: 1 }}
        transition={{ delay: 0.8 }}
        className="glass-card p-6 text-center border border-green-500/30 bg-gradient-to-r from-green-900/20 to-emerald-900/20 glow-green"
      >
        <div className="text-sm text-gray-400 mb-1">Total Financial Impact</div>
        <div className="text-3xl font-bold text-green-400 font-mono">
          INR <CountUp end={parseFloat(impact.total_financial_impact) || 0} duration={2.5} separator="," decimals={0} />
        </div>
        <div className="text-sm text-gray-500 mt-2">
          Compliance: {impact.compliance_improvement} | Risk Reduction: {impact.risk_reduction} | Vendors: {impact.vendors_remediated}
        </div>
      </motion.div>
    </motion.div>
  )
}
