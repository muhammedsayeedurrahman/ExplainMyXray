import { motion } from 'framer-motion'

export default function Navbar({ onDemoClick, isRunning }) {
  return (
    <motion.nav
      initial={{ y: -80 }}
      animate={{ y: 0 }}
      className="fixed top-0 left-0 right-0 z-50 glass-card border-t-0 rounded-t-none"
    >
      <div className="max-w-7xl mx-auto px-6 py-4 flex items-center justify-between">
        <div className="flex items-center gap-3">
          <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-blue-500 to-purple-600 flex items-center justify-center font-bold text-lg">
            L
          </div>
          <div>
            <h1 className="text-xl font-bold tracking-tight">LAWTRIX</h1>
            <p className="text-xs text-gray-500 font-mono">Autonomous Compliance Engine</p>
          </div>
        </div>

        <div className="flex items-center gap-4">
          <span className="text-xs font-mono text-gray-500 hidden sm:block">
            MSMED Act 2006 | IT Act 43B(h) | RBI Rates
          </span>
          <button
            onClick={onDemoClick}
            disabled={isRunning}
            className="px-5 py-2.5 rounded-xl bg-gradient-to-r from-blue-600 to-purple-600 hover:from-blue-500 hover:to-purple-500 text-sm font-semibold transition-all disabled:opacity-50 disabled:cursor-not-allowed"
          >
            {isRunning ? 'Running...' : 'Launch Demo'}
          </button>
        </div>
      </div>
    </motion.nav>
  )
}
