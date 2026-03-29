import { motion } from 'framer-motion'
import CountUp from 'react-countup'

export default function SavingsTracker({ totalSavings, visible }) {
  if (!visible || !totalSavings) return null

  const amount = parseFloat(totalSavings) || 0

  return (
    <motion.div
      initial={{ opacity: 0, x: 20 }}
      animate={{ opacity: 1, x: 0 }}
      className="fixed bottom-6 right-6 z-40 glass-card p-4 border border-green-500/30 glow-green"
    >
      <div className="text-xs text-gray-400">Savings Identified</div>
      <div className="text-lg font-bold text-green-400 font-mono">
        INR <CountUp end={amount} duration={2} separator="," decimals={0} />
      </div>
    </motion.div>
  )
}
