import { motion } from 'framer-motion'
import { useDemo } from '../demo/DemoContext'

export default function Navbar() {
  const { isDemo, startDemo, stopDemo } = useDemo()

  return (
    <motion.nav
      initial={{ y: -20, opacity: 0 }}
      animate={{ y: 0, opacity: 1 }}
      transition={{ duration: 0.5 }}
      className="bg-white/80 backdrop-blur-lg border-b border-gray-100 shadow-sm sticky top-0 z-50"
    >
      <div className="max-w-7xl mx-auto px-6 py-3 flex items-center justify-between">
        <div className="flex items-center gap-3">
          <motion.div
            whileHover={{ scale: 1.05, rotate: 2 }}
            className="w-10 h-10 bg-gradient-to-br from-brand-400 via-brand-500 to-brand-700 rounded-xl flex items-center justify-center shadow-glow"
          >
            <svg className="w-5 h-5 text-white" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
              <path strokeLinecap="round" strokeLinejoin="round" d="M9 20l-5.447-2.724A1 1 0 013 16.382V5.618a1 1 0 011.447-.894L9 7m0 13l6-3m-6 3V7m6 10l4.553 2.276A1 1 0 0021 18.382V7.618a1 1 0 00-.553-.894L15 4m0 13V4m0 0L9 7" />
            </svg>
          </motion.div>
          <div>
            <h1 className="text-lg font-bold text-gray-900 leading-tight tracking-tight">SmartRoute 2.0</h1>
            <p className="text-[11px] text-brand-600 font-medium -mt-0.5">AddressIQ + PackBuddy — AI Logistics Intelligence</p>
          </div>
        </div>

        <div className="flex items-center gap-3">
          {isDemo ? (
            <span className="hidden sm:flex items-center gap-1.5 text-xs text-amber-700 bg-amber-50 px-3 py-1.5 rounded-full border border-amber-200">
              <span className="w-2 h-2 bg-amber-400 rounded-full animate-pulse-slow" />
              Demo Mode
            </span>
          ) : (
            <span className="hidden sm:flex items-center gap-1.5 text-xs text-gray-500 bg-gray-50/80 px-3 py-1.5 rounded-full border border-gray-100">
              <span className="w-2 h-2 bg-green-400 rounded-full animate-pulse-slow" />
              Live — Bengaluru
            </span>
          )}

          <motion.button
            whileHover={{ scale: 1.05 }}
            whileTap={{ scale: 0.95 }}
            onClick={isDemo ? stopDemo : startDemo}
            className={`text-xs font-semibold px-4 py-1.5 rounded-full border shadow-sm transition-all ${
              isDemo
                ? 'bg-amber-500 text-white border-amber-600 hover:bg-amber-600'
                : 'bg-gradient-to-r from-brand-500 to-emerald-500 text-white border-brand-600 hover:shadow-glow'
            }`}
          >
            {isDemo ? 'Exit Demo' : 'Demo'}
          </motion.button>

          <motion.span
            whileHover={{ scale: 1.05 }}
            className="text-xs font-semibold text-brand-700 bg-brand-50 px-3 py-1.5 rounded-full border border-brand-100 shadow-sm"
          >
            HackHustle 2026
          </motion.span>
        </div>
      </div>
    </motion.nav>
  )
}
