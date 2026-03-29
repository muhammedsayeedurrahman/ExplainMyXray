import { motion } from 'framer-motion'

export default function PaymentTimeline({ actions, visible }) {
  if (!visible || !actions?.length) return null

  const withDeadlines = actions.filter(a => a.deadline).slice(0, 8)
  if (!withDeadlines.length) return null

  const priorityColor = {
    critical: 'border-red-500 bg-red-500',
    high: 'border-orange-500 bg-orange-500',
    medium: 'border-yellow-500 bg-yellow-500',
    low: 'border-green-500 bg-green-500',
  }

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      className="glass-card p-6"
    >
      <h2 className="text-lg font-semibold mb-4">Payment Timeline</h2>

      <div className="relative">
        <div className="absolute left-4 top-0 bottom-0 w-0.5 bg-gray-800" />

        <div className="space-y-4">
          {withDeadlines.map((a, i) => (
            <motion.div
              key={a.action_id || i}
              initial={{ opacity: 0, x: -20 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ delay: i * 0.1 }}
              className="flex items-start gap-4 pl-2"
            >
              <div className={`w-3 h-3 rounded-full mt-1.5 shrink-0 ${priorityColor[a.priority]?.split(' ')[1] || 'bg-gray-500'}`} />
              <div className="flex-1 glass-card p-3 border border-gray-800/50">
                <div className="flex justify-between items-start">
                  <span className="text-sm font-medium">{a.vendor_name}</span>
                  <span className="text-xs font-mono text-gray-500">{a.deadline}</span>
                </div>
                <div className="text-xs text-gray-400 mt-1">{a.description}</div>
                <div className="text-xs text-green-400 font-mono mt-1">
                  INR {parseFloat(a.financial_impact).toLocaleString('en-IN', { maximumFractionDigits: 0 })}
                </div>
              </div>
            </motion.div>
          ))}
        </div>
      </div>
    </motion.div>
  )
}
