import { useState, useEffect, useRef, useCallback } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import CountUp from 'react-countup'

const ANIM_PHASES = [
  { key: 'analyzing', label: 'Analyzing 400 delivery addresses...', duration: 1200 },
  { key: 'scoring', label: 'Running AddressIQ scoring engine...', duration: 1000 },
  { key: 'clustering', label: 'PackBuddy clustering verified addresses...', duration: 1000 },
  { key: 'optimizing', label: 'Calculating route optimization...', duration: 800 },
]

export default function BeforeAfterPanel({ metrics, optimized, onOptimized, onReset }) {
  const [phase, setPhase] = useState('before') // before | animating | after
  const [animStep, setAnimStep] = useState(0)
  const timersRef = useRef([])

  // Sync phase with external optimized state
  useEffect(() => {
    if (optimized && phase === 'before') setPhase('after')
    if (!optimized && phase === 'after') setPhase('before')
  }, [optimized, phase])

  // Cleanup timers on unmount
  useEffect(() => () => timersRef.current.forEach(clearTimeout), [])

  const handleOptimize = useCallback(() => {
    if (phase !== 'before') return
    timersRef.current.forEach(clearTimeout)
    timersRef.current = []

    setPhase('animating')
    setAnimStep(0)

    let delay = 0
    ANIM_PHASES.forEach((p, i) => {
      delay += p.duration
      timersRef.current.push(setTimeout(() => setAnimStep(i + 1), delay))
    })

    timersRef.current.push(setTimeout(() => {
      setPhase('after')
      onOptimized?.()
    }, delay + 500))
  }, [phase, onOptimized])

  const handleReset = useCallback(() => {
    setPhase('before')
    setAnimStep(0)
    onReset?.()
  }, [onReset])

  if (!metrics) return null

  const before = {
    trips: metrics.trips_before,
    failureRate: metrics.failure_rate_before,
    cost: metrics.trips_before * 120,
  }
  const after = {
    trips: metrics.trips_after,
    failureRate: metrics.failure_rate_after,
    cost: metrics.trips_after * 120,
  }
  const current = phase === 'after' ? after : before
  const savings = {
    tripsReduced: before.trips - after.trips,
    failuresReduced: (before.failureRate - after.failureRate).toFixed(1),
    costSaved: before.cost - after.cost,
  }

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.5, delay: 0.3 }}
      className="bg-white rounded-2xl shadow-card border border-gray-100 overflow-hidden hover:shadow-card-hover transition-shadow duration-300"
    >
      {/* Header */}
      <div className="px-6 pt-5 pb-4 border-b border-gray-50 flex items-center justify-between">
        <div>
          <h2 className="text-sm font-bold text-gray-900 flex items-center gap-2">
            <svg className="w-4 h-4 text-brand-500" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
              <path strokeLinecap="round" strokeLinejoin="round" d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
            </svg>
            Optimization Impact
          </h2>
          <p className="text-xs text-gray-400 mt-0.5">Before vs After SmartRoute AI</p>
        </div>
        <AnimatePresence mode="wait">
          <motion.span
            key={phase}
            initial={{ opacity: 0, scale: 0.8 }}
            animate={{ opacity: 1, scale: 1 }}
            exit={{ opacity: 0, scale: 0.8 }}
            className={`text-[11px] font-semibold px-3 py-1 rounded-full ${
              phase === 'after' ? 'bg-emerald-100 text-emerald-700'
                : phase === 'animating' ? 'bg-blue-100 text-blue-700'
                : 'bg-red-100 text-red-700'
            }`}
          >
            {phase === 'after' ? 'Optimized' : phase === 'animating' ? 'Processing...' : 'Current State'}
          </motion.span>
        </AnimatePresence>
      </div>

      <div className="p-6">
        {/* Cinematic Animation */}
        <AnimatePresence>
          {phase === 'animating' && (
            <motion.div
              initial={{ opacity: 0, height: 0 }}
              animate={{ opacity: 1, height: 'auto' }}
              exit={{ opacity: 0, height: 0 }}
              className="mb-5 space-y-2"
            >
              {ANIM_PHASES.map((step, i) => (
                <motion.div
                  key={step.key}
                  initial={{ opacity: 0, x: -10 }}
                  animate={animStep > i ? { opacity: 1, x: 0 } : animStep === i ? { opacity: 1, x: 0 } : { opacity: 0.3, x: 0 }}
                  transition={{ duration: 0.3 }}
                  className={`flex items-center gap-2.5 px-3 py-2 rounded-lg text-xs transition-colors duration-300 ${
                    animStep > i ? 'bg-emerald-50 text-emerald-700'
                      : animStep === i ? 'bg-blue-50 text-blue-700'
                      : 'bg-gray-50 text-gray-400'
                  }`}
                >
                  {animStep > i ? (
                    <svg className="w-4 h-4 text-emerald-500" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}><path strokeLinecap="round" strokeLinejoin="round" d="M5 13l4 4L19 7" /></svg>
                  ) : animStep === i ? (
                    <svg className="w-4 h-4 animate-spin text-blue-500" viewBox="0 0 24 24" fill="none"><circle cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="3" className="opacity-25" /><path d="M4 12a8 8 0 018-8" stroke="currentColor" strokeWidth="3" strokeLinecap="round" className="opacity-75" /></svg>
                  ) : (
                    <span className="w-4 h-4 rounded-full border-2 border-gray-200" />
                  )}
                  <span className="font-medium">{step.label}</span>
                </motion.div>
              ))}
            </motion.div>
          )}
        </AnimatePresence>

        {/* Metrics Grid */}
        <AnimatePresence mode="wait">
          {phase !== 'animating' && (
            <motion.div
              key={phase}
              initial={{ opacity: 0, scale: 0.95 }}
              animate={{ opacity: 1, scale: 1 }}
              exit={{ opacity: 0, scale: 0.95 }}
              transition={{ duration: 0.4 }}
              className="grid grid-cols-3 gap-3 mb-5"
            >
              {[
                { label: 'Total Trips', value: current.trips, saved: savings.tripsReduced, savedLabel: 'reduced' },
                { label: 'Failure Rate', value: parseFloat(current.failureRate), saved: savings.failuresReduced, savedLabel: 'reduced', suffix: '%' },
                { label: 'Delivery Cost', value: current.cost, saved: savings.costSaved, savedLabel: 'saved', prefix: '₹' },
              ].map((m) => (
                <div
                  key={m.label}
                  className={`rounded-xl p-4 text-center transition-all duration-500 ${
                    phase === 'after' ? 'bg-emerald-50 border border-emerald-100' : 'bg-gray-50 border border-gray-100'
                  }`}
                >
                  <p className="text-2xl font-bold text-gray-900">
                    {m.prefix || ''}<CountUp end={m.value} duration={1} separator="," decimals={m.suffix === '%' ? 1 : 0} />{m.suffix || ''}
                  </p>
                  <p className="text-[11px] text-gray-500 mt-1">{m.label}</p>
                  {phase === 'after' && (
                    <motion.p
                      initial={{ opacity: 0, y: 5 }}
                      animate={{ opacity: 1, y: 0 }}
                      transition={{ delay: 0.3 }}
                      className="text-[11px] font-semibold text-emerald-600 mt-1"
                    >
                      {m.prefix === '₹' ? '₹' : '-'}{typeof m.saved === 'number' ? m.saved.toLocaleString('en-IN') : m.saved}{m.suffix || ''} {m.savedLabel}
                    </motion.p>
                  )}
                </div>
              ))}
            </motion.div>
          )}
        </AnimatePresence>

        {/* CTA Button */}
        {phase === 'before' && (
          <motion.button
            whileHover={{ scale: 1.02, boxShadow: '0 0 30px rgba(20,184,166,0.3)' }}
            whileTap={{ scale: 0.98 }}
            onClick={handleOptimize}
            className="w-full py-3.5 bg-gradient-to-r from-brand-500 via-brand-600 to-emerald-500 text-white text-sm font-semibold rounded-xl shadow-glow hover:shadow-glow-lg transition-all flex items-center justify-center gap-2"
          >
            <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
              <path strokeLinecap="round" strokeLinejoin="round" d="M13 10V3L4 14h7v7l9-11h-7z" />
            </svg>
            Run Smart Optimization
          </motion.button>
        )}

        {/* After state */}
        {phase === 'after' && (
          <motion.div initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }} className="space-y-2">
            <div className="bg-gradient-to-r from-emerald-50 to-brand-50 rounded-xl p-4 border border-emerald-100">
              <div className="flex items-center gap-2 mb-2">
                <svg className="w-4 h-4 text-emerald-500" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                  <path strokeLinecap="round" strokeLinejoin="round" d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" />
                </svg>
                <span className="text-xs font-bold text-emerald-700">Optimization Complete</span>
              </div>
              <p className="text-[11px] text-gray-600 leading-relaxed">
                AddressIQ verified <strong>{metrics.verified_addresses}</strong> addresses. PackBuddy formed <strong>{metrics.clusters_formed}</strong> delivery clusters.
                Trips reduced from <strong>{before.trips}</strong> to <strong>{after.trips}</strong> — saving <strong>₹{savings.costSaved.toLocaleString('en-IN')}</strong> per batch.
              </p>
            </div>
            <button
              onClick={handleReset}
              className="w-full py-2 text-xs font-medium text-gray-400 hover:text-gray-600 transition-colors"
            >
              Reset to Before State
            </button>
          </motion.div>
        )}
      </div>
    </motion.div>
  )
}
