import { useEffect, useState } from 'react'
import { motion } from 'framer-motion'

function getScoreColor(score) {
  if (score >= 80) return { ring: '#10b981', text: 'text-emerald-400', label: 'Excellent' }
  if (score >= 60) return { ring: '#22c55e', text: 'text-green-400', label: 'Good' }
  if (score >= 40) return { ring: '#eab308', text: 'text-yellow-400', label: 'Moderate Risk' }
  if (score >= 20) return { ring: '#f97316', text: 'text-orange-400', label: 'High Risk' }
  return { ring: '#ef4444', text: 'text-red-400', label: 'Critical' }
}

export default function ComplianceScore({ score, visible }) {
  const [animatedScore, setAnimatedScore] = useState(0)

  useEffect(() => {
    if (!visible || score == null) return
    let current = 0
    const target = score
    const step = Math.max(1, Math.floor(target / 40))
    const timer = setInterval(() => {
      current += step
      if (current >= target) {
        current = target
        clearInterval(timer)
      }
      setAnimatedScore(current)
    }, 30)
    return () => clearInterval(timer)
  }, [score, visible])

  if (!visible || score == null) return null

  const { ring, text, label } = getScoreColor(animatedScore)
  const circumference = 2 * Math.PI * 54
  const offset = circumference - (animatedScore / 100) * circumference

  return (
    <motion.div
      initial={{ opacity: 0, scale: 0.8 }}
      animate={{ opacity: 1, scale: 1 }}
      className="glass-card p-6 flex flex-col items-center"
    >
      <h2 className="text-lg font-semibold mb-4">Compliance Score</h2>

      <div className="relative w-36 h-36">
        <svg className="w-full h-full -rotate-90" viewBox="0 0 120 120">
          <circle cx="60" cy="60" r="54" fill="none" stroke="#1f2937" strokeWidth="8" />
          <motion.circle
            cx="60" cy="60" r="54" fill="none"
            stroke={ring}
            strokeWidth="8"
            strokeLinecap="round"
            strokeDasharray={circumference}
            strokeDashoffset={offset}
            initial={{ strokeDashoffset: circumference }}
            animate={{ strokeDashoffset: offset }}
            transition={{ duration: 1.5, ease: 'easeOut' }}
          />
        </svg>
        <div className="absolute inset-0 flex flex-col items-center justify-center">
          <span className={`text-3xl font-bold font-mono ${text}`}>{animatedScore}</span>
          <span className="text-xs text-gray-500">/100</span>
        </div>
      </div>

      <div className={`mt-3 text-sm font-semibold ${text}`}>{label}</div>
    </motion.div>
  )
}
