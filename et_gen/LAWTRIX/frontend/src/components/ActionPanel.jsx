import { useState, useEffect } from 'react'
import { motion, AnimatePresence } from 'framer-motion'

const PIPELINE_STEPS = [
  { icon: '📥', name: 'Data Ingestion', color: 'blue' },
  { icon: '🔍', name: 'MSME Detection', color: 'purple' },
  { icon: '⏱️', name: 'Delay Analysis', color: 'orange' },
  { icon: '💰', name: 'Interest Calculation', color: 'red' },
  { icon: '📊', name: 'Tax Impact (43B(h))', color: 'pink' },
  { icon: '🎯', name: 'Action Planning', color: 'green' },
]

const priorityColors = {
  critical: 'border-red-500/40 bg-red-500/10',
  high: 'border-orange-500/40 bg-orange-500/10',
  medium: 'border-yellow-500/40 bg-yellow-500/10',
  low: 'border-green-500/40 bg-green-500/10',
}

export default function ActionPanel({ actions, pipelineSteps, visible, scoreBefore, scoreAfter }) {
  const [activeStep, setActiveStep] = useState(0)

  useEffect(() => {
    if (!visible) return
    const timer = setInterval(() => {
      setActiveStep(s => (s < PIPELINE_STEPS.length - 1 ? s + 1 : s))
    }, 800)
    return () => clearInterval(timer)
  }, [visible])

  if (!visible) return null

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      className="glass-card p-6"
    >
      <h2 className="text-lg font-semibold mb-4">Autonomous Execution Engine</h2>

      {/* Pipeline Steps */}
      <div className="flex items-center gap-1 mb-6 overflow-x-auto pb-2">
        {PIPELINE_STEPS.map((step, i) => (
          <motion.div
            key={i}
            initial={{ opacity: 0, scale: 0.8 }}
            animate={{
              opacity: i <= activeStep ? 1 : 0.3,
              scale: i === activeStep ? 1.05 : 1,
            }}
            transition={{ delay: i * 0.15 }}
            className={`flex-1 min-w-[100px] p-3 rounded-xl text-center transition-all ${
              i <= activeStep ? 'bg-gray-800/60 border border-gray-700' : 'bg-gray-900/30'
            } ${i === activeStep ? 'glow-blue ring-1 ring-blue-500/30' : ''}`}
          >
            <div className="text-xl mb-1">{step.icon}</div>
            <div className="text-xs font-medium">{step.name}</div>
            {i <= activeStep && (
              <motion.div
                initial={{ width: 0 }}
                animate={{ width: '100%' }}
                className="h-0.5 bg-green-500 rounded-full mt-2"
              />
            )}
          </motion.div>
        ))}
      </div>

      {/* Score transformation */}
      {scoreBefore != null && scoreAfter != null && (
        <div className="flex items-center justify-center gap-4 mb-6 p-4 rounded-xl bg-gray-800/30">
          <div className="text-center">
            <div className="text-xs text-gray-500">Before</div>
            <div className="text-2xl font-bold text-red-400 font-mono">{scoreBefore}</div>
          </div>
          <motion.div
            animate={{ x: [0, 10, 0] }}
            transition={{ repeat: Infinity, duration: 1.5 }}
            className="text-2xl text-gray-500"
          >
            →
          </motion.div>
          <div className="text-center">
            <div className="text-xs text-gray-500">After</div>
            <div className="text-2xl font-bold text-green-400 font-mono">{scoreAfter}</div>
          </div>
        </div>
      )}

      {/* Action list */}
      {actions?.length > 0 && (
        <div className="space-y-2 max-h-64 overflow-y-auto">
          {actions.map((a, i) => (
            <motion.div
              key={a.action_id || i}
              initial={{ opacity: 0, x: -20 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ delay: i * 0.08 }}
              className={`p-3 rounded-xl border ${priorityColors[a.priority] || 'border-gray-700'} flex items-start gap-3`}
            >
              <span className="text-xs font-mono px-2 py-0.5 rounded bg-gray-800 uppercase shrink-0">
                {a.priority}
              </span>
              <div className="flex-1 min-w-0">
                <div className="text-sm">{a.description}</div>
                <div className="text-xs text-gray-500 mt-1 font-mono">{a.law_reference}</div>
              </div>
              <span className="text-xs font-mono text-green-400 shrink-0">
                INR {parseFloat(a.financial_impact).toLocaleString('en-IN', { maximumFractionDigits: 0 })}
              </span>
            </motion.div>
          ))}
        </div>
      )}
    </motion.div>
  )
}
