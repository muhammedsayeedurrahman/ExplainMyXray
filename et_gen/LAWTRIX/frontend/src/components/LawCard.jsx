import { motion } from 'framer-motion'

const LAWS = [
  {
    id: 'msmed',
    name: 'MSMED Act 2006',
    section: 'Sections 15-16',
    description: 'Mandates payment to MSME suppliers within 45 days. Interest at 3x RBI bank rate for delays.',
    color: 'from-blue-600/20 to-blue-900/20 border-blue-500/30',
    icon: '⚖️',
    rules: ['MSMED-001', 'MSMED-002', 'MSMED-003', 'MSMED-004'],
  },
  {
    id: 'it43bh',
    name: 'IT Act Section 43B(h)',
    section: 'Finance Act 2023',
    description: 'Unpaid MSME amounts beyond due date are disallowed as expense deduction in ITR.',
    color: 'from-purple-600/20 to-purple-900/20 border-purple-500/30',
    icon: '📊',
    rules: ['IT43BH-001', 'IT43BH-002'],
  },
  {
    id: 'msme1',
    name: 'MSME-1 Filing',
    section: 'Companies Act Sec 405',
    description: 'Half-yearly return of outstanding MSME payments required. Penalty for non-filing.',
    color: 'from-orange-600/20 to-orange-900/20 border-orange-500/30',
    icon: '📋',
    rules: ['MSME1-001', 'MSME1-002'],
  },
]

export default function LawCard({ visible, appliedRules }) {
  if (!visible) return null

  return (
    <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
      {LAWS.map((law, i) => {
        const matchedRules = law.rules.filter(r => appliedRules?.includes(r))
        return (
          <motion.div
            key={law.id}
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: i * 0.15 }}
            className={`glass-card p-5 border bg-gradient-to-br ${law.color}`}
          >
            <div className="flex items-center gap-2 mb-3">
              <span className="text-2xl">{law.icon}</span>
              <div>
                <h3 className="text-sm font-bold">{law.name}</h3>
                <p className="text-xs text-gray-500 font-mono">{law.section}</p>
              </div>
            </div>
            <p className="text-xs text-gray-400 mb-3">{law.description}</p>
            <div className="flex flex-wrap gap-1">
              {law.rules.map(r => (
                <span
                  key={r}
                  className={`text-xs px-2 py-0.5 rounded font-mono ${
                    matchedRules.includes(r)
                      ? 'bg-green-500/20 text-green-400 border border-green-500/30'
                      : 'bg-gray-800/50 text-gray-500'
                  }`}
                >
                  {r}
                </span>
              ))}
            </div>
          </motion.div>
        )
      })}
    </div>
  )
}
