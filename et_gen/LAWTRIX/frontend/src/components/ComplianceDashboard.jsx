import { motion } from 'framer-motion'
import CountUp from 'react-countup'

const formatINR = (val) => {
  const num = parseFloat(val) || 0
  if (num >= 10000000) return `${(num / 10000000).toFixed(1)}Cr`
  if (num >= 100000) return `${(num / 100000).toFixed(1)}L`
  if (num >= 1000) return `${(num / 1000).toFixed(1)}K`
  return num.toFixed(0)
}

const CARDS = [
  { key: 'msme_vendors', label: 'MSME Vendors', icon: '🏭', color: 'blue', format: 'num' },
  { key: 'overdue_invoices', label: 'Overdue Invoices', icon: '⚠️', color: 'orange', format: 'num' },
  { key: 'total_overdue_amount', label: 'Overdue Amount', icon: '💰', color: 'red', format: 'inr', prefix: 'INR ' },
  { key: 'total_interest_liability', label: 'Interest Liability', icon: '📈', color: 'purple', format: 'inr', prefix: 'INR ' },
  { key: 'compliance_score', label: 'Compliance Score', icon: '🛡️', color: 'green', format: 'score', suffix: '/100' },
]

const colorMap = {
  blue: 'from-blue-600/20 to-blue-900/20 border-blue-500/30',
  orange: 'from-orange-600/20 to-orange-900/20 border-orange-500/30',
  red: 'from-red-600/20 to-red-900/20 border-red-500/30',
  purple: 'from-purple-600/20 to-purple-900/20 border-purple-500/30',
  green: 'from-green-600/20 to-green-900/20 border-green-500/30',
}

export default function ComplianceDashboard({ summary, visible }) {
  if (!visible || !summary) return null

  return (
    <div className="grid grid-cols-2 md:grid-cols-5 gap-4">
      {CARDS.map((card, i) => {
        const raw = summary[card.key]
        const num = parseFloat(raw) || 0

        return (
          <motion.div
            key={card.key}
            initial={{ opacity: 0, y: 30 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: i * 0.12, duration: 0.5 }}
            className={`glass-card p-5 border bg-gradient-to-br ${colorMap[card.color]}`}
          >
            <div className="text-2xl mb-2">{card.icon}</div>
            <div className="text-2xl font-bold font-mono">
              {card.prefix || ''}
              {card.format === 'inr' ? (
                formatINR(raw)
              ) : (
                <CountUp end={num} duration={1.5} separator="," />
              )}
              {card.suffix || ''}
            </div>
            <div className="text-xs text-gray-400 mt-1">{card.label}</div>
          </motion.div>
        )
      })}
    </div>
  )
}
