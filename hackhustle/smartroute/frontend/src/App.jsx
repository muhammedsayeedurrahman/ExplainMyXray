import { useState, useEffect, useRef, Component, lazy, Suspense } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import Navbar from './components/Navbar'
import MetricsDashboard from './components/MetricsDashboard'
import AddressInput from './components/AddressInput'
import BeforeAfterPanel from './components/BeforeAfterPanel'
import AIInsightCard from './components/AIInsightCard'
import { API_BASE } from './config'
import { useDemo } from './demo/DemoContext'

const MapView = lazy(() => import('./components/MapView'))

/* -- Error Boundary -------------------------------------------------- */
class ErrorBoundary extends Component {
  constructor(props) {
    super(props)
    this.state = { hasError: false, error: null }
  }
  static getDerivedStateFromError(error) {
    return { hasError: true, error }
  }
  componentDidCatch(error, info) {
    console.error('React error boundary caught:', error, info)
  }
  render() {
    if (this.state.hasError) {
      return (
        <div className="min-h-screen bg-gray-50 flex items-center justify-center p-4">
          <div className="bg-white rounded-2xl shadow-depth border border-gray-100 p-8 max-w-md text-center">
            <div className="w-14 h-14 bg-red-50 rounded-2xl flex items-center justify-center mx-auto mb-4">
              <svg className="w-7 h-7 text-red-400" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                <path strokeLinecap="round" strokeLinejoin="round" d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-2.5L13.732 4.5c-.77-.833-2.694-.833-3.464 0L3.34 16.5c-.77.833.192 2.5 1.732 2.5z" />
              </svg>
            </div>
            <h2 className="text-lg font-bold text-gray-900 mb-2">Something went wrong</h2>
            <p className="text-sm text-gray-500 mb-5">{this.state.error?.message}</p>
            <button
              onClick={() => window.location.reload()}
              className="px-6 py-2.5 bg-brand-500 text-white text-sm font-semibold rounded-xl hover:bg-brand-600 shadow-glow"
            >
              Reload Page
            </button>
          </div>
        </div>
      )
    }
    return this.props.children
  }
}

/* -- Main App -------------------------------------------------------- */
export default function App() {
  const [orders, setOrders] = useState([])
  const [metrics, setMetrics] = useState(null)
  const [loadingOrders, setLoadingOrders] = useState(true)
  const [loadingMetrics, setLoadingMetrics] = useState(true)
  const [error, setError] = useState(null)

  // Lifted map view mode
  const [viewMode, setViewMode] = useState('cluster')
  const [optimized, setOptimized] = useState(false)

  // Demo mode
  const { isDemo, stopDemo, demoOrders, demoMetrics } = useDemo()

  // Cinematic demo states
  const [showIntro, setShowIntro] = useState(false)
  const [demoStatus, setDemoStatus] = useState('')
  const [demoComplete, setDemoComplete] = useState(false)
  const [mapZoom, setMapZoom] = useState(null)

  // Section refs for auto-scroll
  const dashboardRef = useRef(null)
  const intelRef = useRef(null)
  const optimizeRef = useRef(null)
  const demoTimersRef = useRef([])

  // Reusable fetch for live data
  async function loadLiveData() {
    setLoadingOrders(true)
    setLoadingMetrics(true)
    setError(null)
    try {
      const [ordersRes, metricsRes] = await Promise.all([
        fetch(`${API_BASE}/get_orders`),
        fetch(`${API_BASE}/metrics`),
      ])
      if (!ordersRes.ok || !metricsRes.ok) throw new Error('API unreachable')
      const ordersData = await ordersRes.json()
      const metricsData = await metricsRes.json()
      setOrders(ordersData.orders)
      setMetrics(metricsData)
    } catch {
      setError('Cannot connect to backend. Make sure the FastAPI server is running on port 8000.')
    } finally {
      setLoadingOrders(false)
      setLoadingMetrics(false)
    }
  }

  // Initial data load (only when not in demo)
  useEffect(() => {
    if (!isDemo) loadLiveData()
  }, []) // eslint-disable-line react-hooks/exhaustive-deps

  const handleOptimized = () => {
    setViewMode('cluster')
    setOptimized(true)
  }

  const handleReset = () => {
    setViewMode('raw')
    setOptimized(false)
  }

  // Demo walkthrough sequence with cinematic intro
  useEffect(() => {
    // Clear previous demo timers
    demoTimersRef.current.forEach(clearTimeout)
    demoTimersRef.current = []

    if (!isDemo) {
      // Exiting demo — reset to live state
      setOptimized(false)
      setViewMode('cluster')
      setShowIntro(false)
      setDemoStatus('')
      setDemoComplete(false)
      setMapZoom(null)
      loadLiveData()
      return
    }

    // Starting demo — reset state
    setError(null)
    setOrders([])
    setMetrics(null)
    setLoadingOrders(true)
    setLoadingMetrics(true)
    setOptimized(false)
    setViewMode('raw')
    setDemoStatus('')
    setDemoComplete(false)
    setMapZoom(null)

    // Show cinematic intro overlay
    setShowIntro(true)

    const schedule = (fn, ms) => {
      const id = setTimeout(fn, ms)
      demoTimersRef.current.push(id)
    }

    // 2500ms — dismiss intro overlay
    schedule(() => setShowIntro(false), 2500)

    // 2700ms — scroll to dashboard
    schedule(() => {
      dashboardRef.current?.scrollIntoView({ behavior: 'smooth', block: 'start' })
      setDemoStatus('Loading dashboard metrics...')
    }, 2700)

    // 3300ms — metrics appear
    schedule(() => {
      setMetrics(demoMetrics)
      setLoadingMetrics(false)
      setDemoStatus('Populating delivery data...')
    }, 3300)

    // 4500ms — scroll to delivery intelligence
    schedule(() => {
      intelRef.current?.scrollIntoView({ behavior: 'smooth', block: 'start' })
    }, 4500)

    // 4900ms — orders populate map (raw view)
    schedule(() => {
      setOrders(demoOrders)
      setLoadingOrders(false)
      setDemoStatus('Analyzing delivery patterns...')
    }, 4900)

    // 6500ms — switch to cluster view and zoom in to show colored dots
    schedule(() => {
      setViewMode('cluster')
      setDemoStatus('Detecting high-risk zones...')
      setMapZoom(13)
    }, 6500)

    // 9000ms — scroll to optimization
    schedule(() => {
      optimizeRef.current?.scrollIntoView({ behavior: 'smooth', block: 'start' })
      setDemoStatus('Optimizing delivery clusters...')
    }, 9000)

    // 9700ms — trigger optimization (BeforeAfterPanel runs its own 4.5s animation)
    schedule(() => {
      handleOptimized()
      setDemoStatus('Running route optimization...')
    }, 9700)

    // 14500ms — demo complete
    schedule(() => {
      setDemoStatus('')
      setDemoComplete(true)
    }, 14500)

    return () => {
      demoTimersRef.current.forEach(clearTimeout)
      demoTimersRef.current = []
    }
  }, [isDemo]) // eslint-disable-line react-hooks/exhaustive-deps

  if (error && !isDemo) {
    return (
      <div className="min-h-screen bg-gray-50">
        <Navbar />
        <div className="max-w-lg mx-auto mt-24 text-center px-4">
          <motion.div
            initial={{ scale: 0.9, opacity: 0 }}
            animate={{ scale: 1, opacity: 1 }}
            className="bg-white rounded-2xl shadow-depth border border-gray-100 p-8"
          >
            <div className="w-16 h-16 bg-red-50 rounded-2xl flex items-center justify-center mx-auto mb-4">
              <svg className="w-8 h-8 text-red-400" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                <path strokeLinecap="round" strokeLinejoin="round" d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-2.5L13.732 4.5c-.77-.833-2.694-.833-3.464 0L3.34 16.5c-.77.833.192 2.5 1.732 2.5z" />
              </svg>
            </div>
            <h2 className="text-lg font-bold text-gray-900 mb-2">Backend Not Connected</h2>
            <p className="text-sm text-gray-500 mb-4">{error}</p>
            <div className="bg-gray-50 rounded-xl p-4 text-left mb-4">
              <p className="text-xs font-medium text-gray-600 mb-2">Quick start:</p>
              <code className="text-xs text-brand-700 block leading-relaxed">
                cd smartroute/backend<br />
                pip install -r requirements.txt<br />
                python main.py
              </code>
            </div>
            <button
              onClick={() => window.location.reload()}
              className="px-6 py-2.5 bg-brand-500 text-white text-sm font-semibold rounded-xl hover:bg-brand-600 shadow-glow"
            >
              Retry Connection
            </button>
          </motion.div>
        </div>
      </div>
    )
  }

  return (
    <ErrorBoundary>
      <div className="min-h-screen bg-gradient-to-br from-gray-50 via-white to-gray-100/50">
        {/* Cinematic Intro Overlay */}
        <AnimatePresence>
          {showIntro && (
            <motion.div
              key="cinematic-intro"
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
              transition={{ duration: 0.5 }}
              className="fixed inset-0 z-50 flex flex-col items-center justify-center bg-gradient-to-br from-white via-blue-50 to-brand-50"
            >
              <motion.h1
                initial={{ scale: 0.9, opacity: 0 }}
                animate={{ scale: 1, opacity: 1 }}
                transition={{ duration: 0.6 }}
                className="text-5xl sm:text-6xl font-extrabold text-gray-900 tracking-tight"
              >
                SmartRoute <span className="text-brand-500">2.0</span>
              </motion.h1>
              <motion.p
                initial={{ opacity: 0, y: 10 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: 0.4, duration: 0.5 }}
                className="mt-4 text-lg sm:text-xl font-medium text-gray-500"
              >
                AI-Powered Logistics Intelligence
              </motion.p>
              <motion.p
                initial={{ opacity: 0, y: 10 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: 0.8, duration: 0.5 }}
                className="mt-2 text-sm text-gray-400"
              >
                Transforming Delivery Efficiency in Real-Time
              </motion.p>
              <motion.div
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                transition={{ delay: 1.2, duration: 0.5 }}
                className="mt-8"
              >
                <div className="w-8 h-8 border-2 border-brand-400 border-t-transparent rounded-full animate-spin" />
              </motion.div>
            </motion.div>
          )}
        </AnimatePresence>

        <Navbar />

        {/* Demo banner + status text */}
        {isDemo && (
          <motion.div
            initial={{ opacity: 0, y: -10 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -10 }}
          >
            <div className="bg-gradient-to-r from-amber-500 via-amber-400 to-yellow-400 text-center py-1.5">
              <p className="text-xs font-semibold text-amber-900 tracking-wide">
                DEMO MODE — Simulated Data
              </p>
            </div>
            <AnimatePresence mode="wait">
              {demoStatus && !demoComplete && (
                <motion.div
                  key="demo-status"
                  initial={{ opacity: 0, height: 0 }}
                  animate={{ opacity: 1, height: 'auto' }}
                  exit={{ opacity: 0, height: 0 }}
                  transition={{ duration: 0.3 }}
                  className="bg-gray-900 text-center py-1.5 overflow-hidden"
                >
                  <p className="text-xs font-mono text-gray-300 tracking-wide flex items-center justify-center gap-2">
                    <span className="inline-block w-1.5 h-1.5 rounded-full bg-brand-400 animate-pulse" />
                    {demoStatus}
                  </p>
                </motion.div>
              )}
              {demoComplete && (
                <motion.div
                  key="demo-complete"
                  initial={{ opacity: 0, scale: 0.95 }}
                  animate={{ opacity: 1, scale: 1 }}
                  transition={{ duration: 0.4, ease: 'easeOut' }}
                  className="bg-gradient-to-r from-emerald-500 via-emerald-400 to-green-400 text-center py-2"
                >
                  <p className="text-xs font-semibold text-emerald-950 tracking-wide flex items-center justify-center gap-2">
                    <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2.5}>
                      <path strokeLinecap="round" strokeLinejoin="round" d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" />
                    </svg>
                    Optimization Complete — 70% Efficiency Gain
                  </p>
                </motion.div>
              )}
            </AnimatePresence>
          </motion.div>
        )}

        <main className="max-w-7xl mx-auto px-4 sm:px-6 py-6 space-y-8">
          {/* Section 1: Dashboard Overview */}
          <section ref={dashboardRef}>
            <SectionHeader title="Dashboard Overview" delay={0.1} />
            <MetricsDashboard metrics={metrics} loading={loadingMetrics} optimized={optimized} />
          </section>

          {/* Section 2: Map + Address Validator */}
          <section ref={intelRef}>
            <SectionHeader title="Delivery Intelligence" delay={0.2} />
            <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
              <div className="lg:col-span-8">
                <Suspense fallback={
                <div className="bg-white rounded-2xl shadow-card border border-gray-100 w-full h-[520px] flex items-center justify-center">
                  <div className="flex flex-col items-center gap-2">
                    <svg className="w-8 h-8 animate-spin text-brand-500" viewBox="0 0 24 24" fill="none">
                      <circle cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="3" className="opacity-25" />
                      <path d="M4 12a8 8 0 018-8" stroke="currentColor" strokeWidth="3" strokeLinecap="round" className="opacity-75" />
                    </svg>
                    <span className="text-xs text-gray-500">Loading map...</span>
                  </div>
                </div>
              }>
                <MapView
                  orders={orders}
                  loading={loadingOrders}
                  viewMode={viewMode}
                  onViewModeChange={setViewMode}
                  zoomLevel={mapZoom}
                />
              </Suspense>
              </div>
              <div className="lg:col-span-4">
                <AddressInput />
              </div>
            </div>
          </section>

          {/* Section 3: Optimization + AI Insights */}
          <section ref={optimizeRef}>
            <SectionHeader title="Smart Optimization" delay={0.3} />
            <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
              <div className="lg:col-span-5">
                <BeforeAfterPanel
                  metrics={metrics}
                  optimized={optimized}
                  onOptimized={handleOptimized}
                  onReset={handleReset}
                />
              </div>
              <div className="lg:col-span-4">
                <HowItWorksCard />
              </div>
              <div className="lg:col-span-3">
                <AIInsightCard />
              </div>
            </div>
          </section>

          {/* Footer */}
          <motion.footer
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ delay: 1 }}
            className="text-center py-8 text-xs text-gray-400 border-t border-gray-200/50"
          >
            <p className="font-semibold">SmartRoute 2.0 — AddressIQ + PackBuddy</p>
            <p className="mt-0.5">Built for HackHustle 2026 — AI-Powered Logistics Intelligence Platform</p>
          </motion.footer>
        </main>
      </div>
    </ErrorBoundary>
  )
}

/* -- Section Header -------------------------------------------------- */
function SectionHeader({ title, delay = 0 }) {
  return (
    <motion.div
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      transition={{ delay }}
      className="flex items-center gap-2 mb-4"
    >
      <h2 className="text-xs font-bold text-gray-400 uppercase tracking-wider">{title}</h2>
      <div className="flex-1 h-px bg-gradient-to-r from-gray-200 to-transparent" />
    </motion.div>
  )
}

/* -- How It Works Card ----------------------------------------------- */
function HowItWorksCard() {
  const steps = [
    { step: '1', title: 'AddressIQ Scores Every Address', desc: 'Rule-based engine rates each address 0-100 based on completeness, PIN validity, and area recognition.', color: 'from-brand-400 to-brand-500' },
    { step: '2', title: 'Auto-Verify & Correct', desc: 'Low-confidence addresses get AI corrections — missing house numbers, streets, and PINs auto-fixed.', color: 'from-blue-400 to-blue-500' },
    { step: '3', title: 'PackBuddy Clusters Deliveries', desc: 'DBSCAN spatial clustering groups nearby verified addresses into optimized delivery zones.', color: 'from-purple-400 to-purple-500' },
    { step: '4', title: 'Route Optimization', desc: 'Clustered deliveries reduce trips, save fuel, and cut delivery failures by up to 70%.', color: 'from-emerald-400 to-emerald-500' },
  ]

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.5, delay: 0.4 }}
      className="bg-white rounded-2xl shadow-card border border-gray-100 overflow-hidden h-full hover:shadow-card-hover transition-shadow duration-300"
    >
      <div className="px-5 pt-5 pb-3 border-b border-gray-50">
        <h2 className="text-sm font-bold text-gray-900 flex items-center gap-2">
          <svg className="w-4 h-4 text-brand-500" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
            <path strokeLinecap="round" strokeLinejoin="round" d="M13 10V3L4 14h7v7l9-11h-7z" />
          </svg>
          How SmartRoute Works
        </h2>
        <p className="text-xs text-gray-400 mt-0.5">4-step AI pipeline</p>
      </div>
      <div className="p-5 space-y-3.5">
        {steps.map((s, i) => (
          <motion.div
            key={s.step}
            initial={{ opacity: 0, x: -10 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ delay: 0.6 + i * 0.1 }}
            className="flex items-start gap-3 group"
          >
            <div className={`w-7 h-7 rounded-lg bg-gradient-to-br ${s.color} flex items-center justify-center flex-shrink-0 shadow-sm group-hover:shadow-md group-hover:scale-105 transition-all duration-200`}>
              <span className="text-white text-xs font-bold">{s.step}</span>
            </div>
            <div>
              <p className="text-xs font-semibold text-gray-800">{s.title}</p>
              <p className="text-[11px] text-gray-500 leading-relaxed mt-0.5">{s.desc}</p>
            </div>
          </motion.div>
        ))}
      </div>
    </motion.div>
  )
}
