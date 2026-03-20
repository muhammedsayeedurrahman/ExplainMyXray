import { useState, useEffect } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { API_BASE } from '../config'
import { useDemo } from '../demo/DemoContext'

const ICONS = {
  alert: <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}><path strokeLinecap="round" strokeLinejoin="round" d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-2.5L13.732 4.5c-.77-.833-2.694-.833-3.464 0L3.34 16.5c-.77.833.192 2.5 1.732 2.5z" /></svg>,
  warning: <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}><path strokeLinecap="round" strokeLinejoin="round" d="M20.618 5.984A11.955 11.955 0 0112 2.944a11.955 11.955 0 01-8.618 3.04A12.02 12.02 0 003 9c0 5.591 3.824 10.29 9 11.622 5.176-1.332 9-6.03 9-11.622 0-1.042-.133-2.052-.382-3.016z" /></svg>,
  map: <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}><path strokeLinecap="round" strokeLinejoin="round" d="M17.657 16.657L13.414 20.9a1.998 1.998 0 01-2.827 0l-4.244-4.243a8 8 0 1111.314 0z" /><path strokeLinecap="round" strokeLinejoin="round" d="M15 11a3 3 0 11-6 0 3 3 0 016 0z" /></svg>,
  sparkle: <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}><path strokeLinecap="round" strokeLinejoin="round" d="M13 10V3L4 14h7v7l9-11h-7z" /></svg>,
  chart: <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}><path strokeLinecap="round" strokeLinejoin="round" d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" /></svg>,
}

const TYPE_COLORS = {
  failure_pattern: { bg: 'bg-red-50', border: 'border-red-100', icon: 'text-red-500', dot: 'bg-red-400' },
  risk_concentration: { bg: 'bg-amber-50', border: 'border-amber-100', icon: 'text-amber-500', dot: 'bg-amber-400' },
  area_analysis: { bg: 'bg-blue-50', border: 'border-blue-100', icon: 'text-blue-500', dot: 'bg-blue-400' },
  optimization: { bg: 'bg-emerald-50', border: 'border-emerald-100', icon: 'text-emerald-500', dot: 'bg-emerald-400' },
  industry: { bg: 'bg-purple-50', border: 'border-purple-100', icon: 'text-purple-500', dot: 'bg-purple-400' },
}

const FALLBACK_COLORS = { bg: 'bg-gray-50', border: 'border-gray-100', icon: 'text-gray-500', dot: 'bg-gray-400' }

export default function AIInsightCard() {
  const [insights, setInsights] = useState([])
  const [activeIdx, setActiveIdx] = useState(0)
  const [loading, setLoading] = useState(true)

  const { isDemo, demoInsights } = useDemo()

  useEffect(() => {
    if (isDemo) {
      setInsights(demoInsights)
      setLoading(false)
      return
    }

    async function fetchInsights() {
      try {
        const res = await fetch(`${API_BASE}/insights`)
        if (!res.ok) throw new Error('Failed')
        const data = await res.json()
        setInsights(data.insights || [])
      } catch {
        setInsights([])
      } finally {
        setLoading(false)
      }
    }
    fetchInsights()
  }, [isDemo, demoInsights])

  useEffect(() => {
    if (insights.length <= 1) return
    const timer = setInterval(() => setActiveIdx((prev) => (prev + 1) % insights.length), 6000)
    return () => clearInterval(timer)
  }, [insights.length])

  if (loading) {
    return (
      <div className="bg-white rounded-2xl p-5 shadow-card border border-gray-100 animate-pulse h-full">
        <div className="h-3 bg-gray-100 rounded w-24 mb-4" />
        <div className="h-4 bg-gray-100 rounded w-full mb-2" />
        <div className="h-4 bg-gray-100 rounded w-3/4" />
      </div>
    )
  }

  if (!insights.length) {
    return (
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.5 }}
        className="bg-white rounded-2xl shadow-card border border-gray-100 p-5 h-full"
      >
        <h2 className="text-sm font-bold text-gray-900 flex items-center gap-2 mb-3">
          <span className="w-5 h-5 bg-gradient-to-br from-amber-400 to-orange-500 rounded-md flex items-center justify-center">
            <svg className="w-3 h-3 text-white" fill="currentColor" viewBox="0 0 20 20"><path d="M11.3 1.046A1 1 0 0112 2v5h4a1 1 0 01.82 1.573l-7 10A1 1 0 018 18v-5H4a1 1 0 01-.82-1.573l7-10a1 1 0 011.12-.38z" /></svg>
          </span>
          AI Insight
        </h2>
        <p className="text-xs text-gray-600 leading-relaxed">
          Most delivery failures occur in low-confidence address zones with incomplete or landmark-only descriptions.
        </p>
      </motion.div>
    )
  }

  const insight = insights[activeIdx]
  const colors = TYPE_COLORS[insight.type] ?? FALLBACK_COLORS

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.5, delay: 0.5 }}
      className="bg-white rounded-2xl shadow-card border border-gray-100 overflow-hidden hover:shadow-card-hover transition-shadow duration-300 h-full flex flex-col"
    >
      {/* Header */}
      <div className="px-5 pt-5 pb-3 border-b border-gray-50 flex items-center justify-between">
        <h2 className="text-sm font-bold text-gray-900 flex items-center gap-2">
          <span className="w-5 h-5 bg-gradient-to-br from-amber-400 to-orange-500 rounded-md flex items-center justify-center">
            <svg className="w-3 h-3 text-white" fill="currentColor" viewBox="0 0 20 20"><path d="M11.3 1.046A1 1 0 0112 2v5h4a1 1 0 01.82 1.573l-7 10A1 1 0 018 18v-5H4a1 1 0 01-.82-1.573l7-10a1 1 0 011.12-.38z" /></svg>
          </span>
          AI Insights
        </h2>
        {insights.length > 1 && (
          <div className="flex items-center gap-1">
            {insights.map((_, i) => (
              <button
                key={i}
                onClick={() => setActiveIdx(i)}
                aria-label={`View insight ${i + 1}`}
                className={`h-1.5 rounded-full transition-all duration-300 ${
                  i === activeIdx ? `${colors.dot} w-4` : 'bg-gray-200 w-1.5'
                }`}
              />
            ))}
          </div>
        )}
      </div>

      {/* Content */}
      <div className="p-5 flex-1 flex flex-col">
        <AnimatePresence mode="wait">
          <motion.div
            key={activeIdx}
            initial={{ opacity: 0, x: 20 }}
            animate={{ opacity: 1, x: 0 }}
            exit={{ opacity: 0, x: -20 }}
            transition={{ duration: 0.3 }}
            className={`rounded-xl p-4 border ${colors.bg} ${colors.border} flex-1`}
          >
            <div className="flex items-start gap-3">
              <span className={`mt-0.5 ${colors.icon}`}>{ICONS[insight.icon] || ICONS.chart}</span>
              <div className="flex-1 min-w-0">
                <p className="text-xs font-bold text-gray-800 mb-1">{insight.title}</p>
                <p className="text-[11px] text-gray-600 leading-relaxed">{insight.text}</p>
              </div>
            </div>
          </motion.div>
        </AnimatePresence>

        {/* Navigation */}
        {insights.length > 1 && (
          <div className="flex justify-between mt-3">
            <button onClick={() => setActiveIdx((prev) => (prev - 1 + insights.length) % insights.length)} className="text-[11px] text-gray-400 hover:text-gray-600 transition-colors flex items-center gap-1">
              <svg className="w-3 h-3" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}><path strokeLinecap="round" strokeLinejoin="round" d="M15 19l-7-7 7-7" /></svg>
              Previous
            </button>
            <span className="text-[11px] text-gray-300">{activeIdx + 1} of {insights.length}</span>
            <button onClick={() => setActiveIdx((prev) => (prev + 1) % insights.length)} className="text-[11px] text-gray-400 hover:text-gray-600 transition-colors flex items-center gap-1">
              Next
              <svg className="w-3 h-3" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}><path strokeLinecap="round" strokeLinejoin="round" d="M9 5l7 7-7 7" /></svg>
            </button>
          </div>
        )}
      </div>
    </motion.div>
  )
}
