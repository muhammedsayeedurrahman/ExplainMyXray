import { motion } from 'framer-motion'

const STEPS = [
  { num: '01', title: 'Upload', desc: 'Drop CSV/Excel with invoice data', icon: '📥' },
  { num: '02', title: 'Detect', desc: 'MSME vendors auto-identified', icon: '🔍' },
  { num: '03', title: 'Analyze', desc: 'Laws applied as executable code', icon: '⚖️' },
  { num: '04', title: 'Calculate', desc: 'Interest, tax impact, risk scores', icon: '📊' },
  { num: '05', title: 'Act', desc: 'Payment priorities & filings generated', icon: '🎯' },
  { num: '06', title: 'Prove', desc: 'Full audit trail of every decision', icon: '✅' },
]

export default function HowItWorksCard({ visible }) {
  if (!visible) return null

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      className="glass-card p-6"
    >
      <h2 className="text-lg font-semibold mb-4">How LAWTRIX Works</h2>

      <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-6 gap-3">
        {STEPS.map((step, i) => (
          <motion.div
            key={step.num}
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: i * 0.1 }}
            className="text-center p-3 rounded-xl bg-gray-800/30"
          >
            <div className="text-2xl mb-2">{step.icon}</div>
            <div className="text-xs text-blue-400 font-mono mb-1">{step.num}</div>
            <div className="text-sm font-semibold">{step.title}</div>
            <div className="text-xs text-gray-500 mt-1">{step.desc}</div>
          </motion.div>
        ))}
      </div>
    </motion.div>
  )
}
