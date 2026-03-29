import { motion } from 'framer-motion'

const FEATURES = [
  { name: 'Penalty Prevention Guarantee', price: '1,999/mo', icon: '🛡️' },
  { name: 'Compliance Score API', price: 'B2B Pricing', icon: '🔌' },
  { name: 'Auto-Fix Mode', price: 'Pay-per-action', icon: '⚡' },
  { name: 'Scrutiny Defense Generator', price: '999/report', icon: '📄' },
  { name: 'ERP Plugin Marketplace', price: 'Coming Soon', icon: '🔗' },
  { name: 'Multi-Company Dashboard', price: 'CA Firms', icon: '🏢' },
]

export default function MonetizationTeaser({ visible }) {
  if (!visible) return null

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      className="glass-card p-6 border border-purple-500/20"
    >
      <div className="text-center mb-4">
        <h2 className="text-lg font-semibold gradient-text">Premium Features</h2>
        <p className="text-xs text-gray-500 mt-1">Unlock full autonomous compliance</p>
      </div>

      <div className="grid grid-cols-2 md:grid-cols-3 gap-3">
        {FEATURES.map((f, i) => (
          <motion.div
            key={f.name}
            initial={{ opacity: 0, scale: 0.9 }}
            animate={{ opacity: 1, scale: 1 }}
            transition={{ delay: i * 0.08 }}
            className="p-3 rounded-xl bg-gray-800/30 border border-gray-800 hover:border-purple-500/30 transition-colors cursor-pointer"
          >
            <div className="text-xl mb-1">{f.icon}</div>
            <div className="text-xs font-medium">{f.name}</div>
            <div className="text-xs text-purple-400 font-mono mt-1">INR {f.price}</div>
          </motion.div>
        ))}
      </div>
    </motion.div>
  )
}
