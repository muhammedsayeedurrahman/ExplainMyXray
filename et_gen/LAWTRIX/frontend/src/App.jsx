import { useState, useCallback } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { DemoProvider, useDemo } from './demo/DemoContext'
import { API_BASE } from './config'
import Navbar from './components/Navbar'
import FileUpload from './components/FileUpload'
import HowItWorksCard from './components/HowItWorksCard'
import Dashboard from './components/Dashboard'

function AppContent() {
  const { isRunning, currentStep, data, sessionId, startDemo, resetDemo, error } = useDemo()
  const [manualData, setManualData] = useState(null)
  const [manualSession, setManualSession] = useState(null)
  const [uploadLoading, setUploadLoading] = useState(false)

  const activeData = data || manualData
  const activeSession = sessionId || manualSession

  // Show dashboard when data is ready (demo complete or manual upload)
  const showDashboard = (activeData && (currentStep >= 6 || currentStep === -1)) || manualData

  const handleUploadComplete = useCallback(async (result) => {
    const sid = result.session_id
    setManualSession(sid)
    setUploadLoading(true)
    try {
      // Fetch full data from all endpoints to match demo response shape
      const [analysisRes, vendorsRes, actionsRes, impactRes] = await Promise.all([
        fetch(`${API_BASE}/analysis/${sid}`).then(r => r.json()).catch(() => null),
        fetch(`${API_BASE}/vendors/${sid}`).then(r => r.json()).catch(() => null),
        fetch(`${API_BASE}/actions/${sid}`).then(r => r.json()).catch(() => null),
        fetch(`${API_BASE}/impact/${sid}`).then(r => r.json()).catch(() => null),
      ])

      const metrics = analysisRes?.metrics || {}
      setManualData({
        session_id: sid,
        status: 'complete',
        summary: {
          total_invoices: metrics.total_invoices || result.total_invoices,
          total_vendors: metrics.total_vendors || result.total_vendors,
          msme_vendors: metrics.msme_vendors || 0,
          overdue_invoices: metrics.overdue_invoices || 0,
          total_overdue_amount: metrics.total_overdue_amount || '0',
          total_interest_liability: metrics.total_interest_liability || '0',
          total_tax_disallowance: metrics.total_tax_disallowance || '0',
          compliance_score: metrics.compliance_score || 0,
          vendors_at_risk: metrics.vendors_at_risk || 0,
        },
        msme_vendors: (vendorsRes?.vendors || []).filter(v => v.is_msme),
        actions: actionsRes?.actions || [],
        impact: impactRes || null,
        pipeline_steps: [],
        rules_evaluated: null,
      })
    } catch {
      // Fallback: use minimal data from upload response
      setManualData({
        session_id: sid,
        status: 'complete',
        summary: {
          total_invoices: result.total_invoices || 0,
          total_vendors: result.total_vendors || 0,
          msme_vendors: 0,
          overdue_invoices: 0,
          total_overdue_amount: '0',
          total_interest_liability: '0',
          compliance_score: 0,
          vendors_at_risk: 0,
        },
        msme_vendors: [],
        actions: [],
        impact: null,
        pipeline_steps: [],
        rules_evaluated: null,
      })
    } finally {
      setUploadLoading(false)
    }
  }, [])

  const handleReset = () => {
    resetDemo()
    setManualData(null)
    setManualSession(null)
  }

  const step = currentStep

  // Dashboard mode - full sidebar layout
  if (showDashboard) {
    return (
      <Dashboard
        data={activeData}
        sessionId={activeSession}
        onReset={handleReset}
      />
    )
  }

  // Landing / Demo mode
  return (
    <div className="min-h-screen bg-gray-950">
      <Navbar onDemoClick={isRunning ? resetDemo : startDemo} isRunning={isRunning && step < 6} />

      {/* Cinematic Intro */}
      <AnimatePresence>
        {step === 0 && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            className="fixed inset-0 z-40 flex flex-col items-center justify-center bg-gray-950"
          >
            <motion.h1
              initial={{ scale: 0.5, opacity: 0 }}
              animate={{ scale: 1, opacity: 1 }}
              transition={{ type: 'spring', duration: 0.8 }}
              className="text-7xl md:text-9xl font-black tracking-tighter gradient-text"
            >
              LAWTRIX
            </motion.h1>
            <motion.p
              initial={{ y: 20, opacity: 0 }}
              animate={{ y: 0, opacity: 1 }}
              transition={{ delay: 0.4 }}
              className="text-lg text-gray-400 mt-4 font-mono"
            >
              Autonomous Compliance Execution Engine
            </motion.p>
            <motion.div
              initial={{ scaleX: 0 }}
              animate={{ scaleX: 1 }}
              transition={{ delay: 0.8, duration: 1 }}
              className="w-48 h-0.5 bg-gradient-to-r from-blue-500 via-purple-500 to-cyan-500 mt-6"
            />
            <motion.p
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              transition={{ delay: 1.2 }}
              className="text-sm text-gray-600 mt-4"
            >
              MSMED Act 2006 | IT Act 43B(h) | RBI Interest Rules
            </motion.p>
          </motion.div>
        )}
      </AnimatePresence>

      {/* Main Content */}
      <div className="pt-24 pb-12 px-4 max-w-7xl mx-auto space-y-6">
        {/* Error state */}
        {error && (
          <div className="glass-card p-6 border border-red-500/30 text-red-400 text-center">
            {error}. Make sure the backend is running on port 8001.
          </div>
        )}

        {/* Pre-demo: show upload + how it works */}
        {!isRunning && !activeData && !uploadLoading && (
          <>
            <HowItWorksCard visible={true} />
            <FileUpload onUploadComplete={handleUploadComplete} />
          </>
        )}

        {/* Upload loading state */}
        {uploadLoading && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            className="glass-card p-8 text-center"
          >
            <div className="w-8 h-8 border-2 border-blue-500 border-t-transparent rounded-full animate-spin mx-auto mb-4" />
            <p className="text-lg font-semibold">Analyzing your data...</p>
            <p className="text-gray-400 text-sm mt-2">Running compliance pipeline</p>
          </motion.div>
        )}

        {/* Step 1: Data upload animation */}
        <AnimatePresence>
          {step === 1 && (
            <motion.div
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
              className="glass-card p-8 text-center"
            >
              <motion.div
                animate={{ rotate: 360 }}
                transition={{ repeat: Infinity, duration: 2, ease: 'linear' }}
                className="text-4xl inline-block mb-4"
              >
                ...
              </motion.div>
              <p className="text-lg font-semibold">Processing Invoice Data</p>
              <p className="text-gray-400 text-sm mt-2">
                Analyzing 847+ invoices from 52 vendors...
              </p>
            </motion.div>
          )}
        </AnimatePresence>

        {/* Steps 2-5: Progressive reveal during demo */}
        {isRunning && step >= 2 && step < 6 && (
          <div className="space-y-6">
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              className="glass-card p-6 text-center border border-blue-500/20"
            >
              <div className="flex items-center justify-center gap-3 mb-4">
                {[1, 2, 3, 4, 5, 6].map((s) => (
                  <div
                    key={s}
                    className={`w-10 h-10 rounded-full flex items-center justify-center text-sm font-bold transition-all ${
                      s <= step
                        ? 'bg-blue-600 text-white'
                        : s === step + 1
                        ? 'bg-blue-600/30 text-blue-300 animate-pulse'
                        : 'bg-gray-800 text-gray-600'
                    }`}
                  >
                    {s <= step ? '\u2713' : s}
                  </div>
                ))}
              </div>
              <h2 className="text-xl font-bold">
                {step === 2 && 'Detecting MSME Vendors...'}
                {step === 3 && 'Running Legal Engine...'}
                {step === 4 && 'Generating Actions...'}
                {step === 5 && 'Calculating Impact...'}
              </h2>
              <p className="text-sm text-gray-400 mt-2">
                {activeData?.pipeline_steps?.[step - 1]?.detail || 'Processing...'}
              </p>
              <div className="mt-4 w-full bg-gray-800 rounded-full h-1.5">
                <motion.div
                  className="bg-gradient-to-r from-blue-500 to-purple-500 h-1.5 rounded-full"
                  initial={{ width: '0%' }}
                  animate={{ width: `${((step - 1) / 5) * 100}%` }}
                  transition={{ duration: 0.5 }}
                />
              </div>
            </motion.div>
          </div>
        )}
      </div>
    </div>
  )
}

export default function App() {
  return (
    <DemoProvider>
      <AppContent />
    </DemoProvider>
  )
}
