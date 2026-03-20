import { useState, useEffect, useRef, useCallback } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { API_BASE } from '../config'
import { useDemo } from '../demo/DemoContext'

const RISK_FALLBACK = { bg: 'bg-gray-50', border: 'border-gray-200', text: 'text-gray-700', label: 'Unknown', dot: 'bg-gray-400' }
const riskConfig = {
  green:  { bg: 'bg-emerald-50', border: 'border-emerald-200', text: 'text-emerald-700', label: 'High Confidence (80-100)', dot: 'bg-emerald-400' },
  yellow: { bg: 'bg-amber-50',   border: 'border-amber-200',   text: 'text-amber-700',   label: 'Medium — Needs Review (50-79)', dot: 'bg-amber-400' },
  red:    { bg: 'bg-red-50',     border: 'border-red-200',     text: 'text-red-700',     label: 'Low — Verify Required (0-49)', dot: 'bg-red-400' },
}

function CircleScore({ score, risk }) {
  const radius = 36
  const circumference = 2 * Math.PI * radius
  const offset = circumference - (score / 100) * circumference
  const color = risk === 'green' ? '#10b981' : risk === 'yellow' ? '#f59e0b' : '#ef4444'

  return (
    <div className="relative w-24 h-24 mx-auto">
      <svg className="w-24 h-24 -rotate-90" viewBox="0 0 80 80">
        <circle cx="40" cy="40" r={radius} fill="none" stroke="#f3f4f6" strokeWidth="6" />
        <motion.circle
          cx="40" cy="40" r={radius} fill="none"
          stroke={color} strokeWidth="6" strokeLinecap="round"
          strokeDasharray={circumference}
          initial={{ strokeDashoffset: circumference }}
          animate={{ strokeDashoffset: offset }}
          transition={{ duration: 0.8, ease: 'easeOut' }}
        />
      </svg>
      <div className="absolute inset-0 flex items-center justify-center">
        <span className="text-xl font-bold text-gray-900">{score}</span>
      </div>
    </div>
  )
}

export default function AddressInput() {
  const [address, setAddress] = useState('')
  const [pinCode, setPinCode] = useState('')
  const [result, setResult] = useState(null)
  const [correction, setCorrection] = useState(null)
  const [loading, setLoading] = useState(false)
  const [verifying, setVerifying] = useState(false)
  const [verifyError, setVerifyError] = useState(null)
  const [submitCount, setSubmitCount] = useState(0)
  const [liveScore, setLiveScore] = useState(null)
  const [liveLoading, setLiveLoading] = useState(false)
  const debounceRef = useRef(null)
  const abortRef = useRef(null)
  const demoTimersRef = useRef([])

  const { isDemo, demoScoreResult, demoVerifyResult } = useDemo()

  const debouncedScore = useCallback((addr, pin) => {
    if (isDemo) return
    if (debounceRef.current) clearTimeout(debounceRef.current)
    abortRef.current?.abort()
    if (!addr.trim() || addr.trim().length < 5) {
      setLiveScore(null)
      setLiveLoading(false)
      return
    }
    setLiveLoading(true)
    debounceRef.current = setTimeout(async () => {
      abortRef.current = new AbortController()
      try {
        const res = await fetch(`${API_BASE}/score_address`, {
          method: 'POST', headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ address: addr.trim(), pin_code: pin.trim() }),
          signal: abortRef.current.signal,
        })
        if (!res.ok) throw new Error('Score failed')
        setLiveScore(await res.json())
      } catch (err) {
        if (err.name !== 'AbortError') setLiveScore(null)
      } finally {
        setLiveLoading(false)
      }
    }, 400)
  }, [isDemo])

  useEffect(() => {
    debouncedScore(address, pinCode)
    return () => { if (debounceRef.current) clearTimeout(debounceRef.current); abortRef.current?.abort() }
  }, [address, pinCode, debouncedScore])

  // Demo auto-type effect
  useEffect(() => {
    demoTimersRef.current.forEach(clearTimeout)
    demoTimersRef.current = []

    if (!isDemo) {
      // Reset when exiting demo
      setAddress('')
      setPinCode('')
      setResult(null)
      setCorrection(null)
      setLiveScore(null)
      return
    }

    const demoAddress = 'Near ISKCON Temple, Rajajinagar'
    let charIdx = 0

    const schedule = (fn, ms) => {
      const id = setTimeout(fn, ms)
      demoTimersRef.current.push(id)
    }

    // Start typing after map populates (offset from App.jsx sequence ~3s)
    schedule(() => {
      setAddress('')
      setPinCode('')
      setResult(null)
      setCorrection(null)

      const typeInterval = setInterval(() => {
        charIdx++
        if (charIdx <= demoAddress.length) {
          setAddress(demoAddress.slice(0, charIdx))
        } else {
          clearInterval(typeInterval)
        }
      }, 55)
      demoTimersRef.current.push(typeInterval)
    }, 3000)

    // Auto-score after typing finishes (~3000 + 30*55 + 600 = ~5250ms)
    const typingDuration = demoAddress.length * 55
    schedule(() => {
      setLoading(true)
      setSubmitCount((c) => c + 1)
    }, 3000 + typingDuration + 400)

    schedule(() => {
      setResult(demoScoreResult)
      setLoading(false)
    }, 3000 + typingDuration + 1100)

    // Auto-verify after score shows (+2200ms)
    schedule(() => {
      setVerifying(true)
    }, 3000 + typingDuration + 3300)

    schedule(() => {
      setCorrection(demoVerifyResult)
      setResult({ score: demoVerifyResult.score, risk: demoVerifyResult.risk, explanation: demoVerifyResult.explanation })
      setVerifying(false)
    }, 3000 + typingDuration + 4200)

    return () => {
      demoTimersRef.current.forEach(clearTimeout)
      demoTimersRef.current = []
    }
  }, [isDemo, demoScoreResult, demoVerifyResult])

  async function handleScore() {
    if (!address.trim()) return
    setLoading(true); setResult(null); setCorrection(null); setVerifyError(null)
    setSubmitCount((c) => c + 1)

    if (isDemo) {
      setTimeout(() => {
        setResult(demoScoreResult)
        setLoading(false)
      }, 700)
      return
    }

    try {
      const res = await fetch(`${API_BASE}/score_address`, {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ address: address.trim(), pin_code: pinCode.trim() }),
      })
      if (!res.ok) throw new Error(`Scoring failed (${res.status})`)
      setResult(await res.json())
    } catch {
      setResult({ score: 0, risk: 'red', explanation: 'Could not reach scoring API' })
    } finally { setLoading(false) }
  }

  async function handleVerify() {
    setVerifying(true); setVerifyError(null)

    if (isDemo) {
      setTimeout(() => {
        setCorrection(demoVerifyResult)
        setResult({ score: demoVerifyResult.score, risk: demoVerifyResult.risk, explanation: demoVerifyResult.explanation })
        setVerifying(false)
      }, 900)
      return
    }

    try {
      const res = await fetch(`${API_BASE}/verify_address`, {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ address: address.trim(), pin_code: pinCode.trim() }),
      })
      if (!res.ok) throw new Error(`Verification failed (${res.status})`)
      const data = await res.json()
      setCorrection(data)
      setResult({ score: data.score, risk: data.risk, explanation: data.explanation })
    } catch {
      setCorrection(null)
      setVerifyError('Verification failed. Please try again.')
    } finally { setVerifying(false) }
  }

  const risk = result ? (riskConfig[result.risk] ?? RISK_FALLBACK) : null

  return (
    <motion.div
      initial={{ x: 30, opacity: 0 }}
      animate={{ x: 0, opacity: 1 }}
      transition={{ duration: 0.5, delay: 0.2 }}
      className="bg-white rounded-2xl shadow-card border border-gray-100 overflow-hidden h-full flex flex-col hover:shadow-card-hover transition-shadow duration-300"
    >
      <div className="px-5 pt-5 pb-3 border-b border-gray-50">
        <h2 className="text-sm font-bold text-gray-900 flex items-center gap-2">
          <svg className="w-4 h-4 text-brand-500" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
            <path strokeLinecap="round" strokeLinejoin="round" d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z" />
          </svg>
          AddressIQ Validator
        </h2>
        <p className="text-xs text-gray-400 mt-0.5">Score any delivery address in real-time</p>
      </div>

      <div className="p-5 space-y-3 flex-1">
        <div>
          <label className="block text-xs font-medium text-gray-500 mb-1.5">Delivery Address</label>
          <textarea
            value={address} onChange={(e) => setAddress(e.target.value)}
            placeholder="e.g. #42, 5th Main Road, Koramangala, Bengaluru"
            rows={3}
            className="w-full px-3 py-2.5 text-sm border border-gray-200 rounded-xl focus:outline-none focus:ring-2 focus:ring-brand-400 focus:border-transparent resize-none bg-gray-50/80 placeholder:text-gray-300"
          />
        </div>

        <div className="flex gap-3">
          <div className="flex-1">
            <label className="block text-xs font-medium text-gray-500 mb-1.5">PIN Code</label>
            <input
              type="text" value={pinCode} onChange={(e) => setPinCode(e.target.value)}
              placeholder="560001" maxLength={6}
              className="w-full px-3 py-2.5 text-sm border border-gray-200 rounded-xl focus:outline-none focus:ring-2 focus:ring-brand-400 focus:border-transparent bg-gray-50/80 placeholder:text-gray-300"
            />
          </div>
          <AnimatePresence>
            {(liveScore || liveLoading) && !result && (
              <motion.div initial={{ opacity: 0, scale: 0.8 }} animate={{ opacity: 1, scale: 1 }} exit={{ opacity: 0, scale: 0.8 }} className="flex items-end pb-0.5">
                {liveLoading ? (
                  <div className="w-10 h-10 rounded-xl bg-gray-50 flex items-center justify-center">
                    <svg className="w-4 h-4 animate-spin text-gray-400" viewBox="0 0 24 24" fill="none"><circle cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="3" className="opacity-25" /><path d="M4 12a8 8 0 018-8" stroke="currentColor" strokeWidth="3" strokeLinecap="round" className="opacity-75" /></svg>
                  </div>
                ) : liveScore ? (
                  <div className={`w-10 h-10 rounded-xl flex items-center justify-center text-xs font-bold ${
                    liveScore.risk === 'green' ? 'bg-emerald-50 text-emerald-600' : liveScore.risk === 'yellow' ? 'bg-amber-50 text-amber-600' : 'bg-red-50 text-red-600'
                  }`}>{liveScore.score}</div>
                ) : null}
              </motion.div>
            )}
          </AnimatePresence>
        </div>

        <motion.button
          whileHover={{ scale: 1.02, boxShadow: '0 0 20px rgba(20,184,166,0.25)' }}
          whileTap={{ scale: 0.98 }}
          onClick={handleScore}
          disabled={loading || !address.trim()}
          className="w-full py-2.5 bg-gradient-to-r from-brand-500 to-brand-600 text-white text-sm font-semibold rounded-xl shadow-md hover:shadow-glow disabled:opacity-40 disabled:cursor-not-allowed flex items-center justify-center gap-2"
        >
          {loading ? (
            <><svg className="w-4 h-4 animate-spin" viewBox="0 0 24 24" fill="none"><circle cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="3" className="opacity-25" /><path d="M4 12a8 8 0 018-8" stroke="currentColor" strokeWidth="3" strokeLinecap="round" className="opacity-75" /></svg>Scoring...</>
          ) : 'Score Address'}
        </motion.button>

        {/* Result */}
        <AnimatePresence mode="wait">
          {result && (
            <motion.div key={submitCount} initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0, y: -10 }} className={`rounded-xl p-4 border ${risk.bg} ${risk.border}`}>
              <CircleScore score={result.score} risk={result.risk} />
              <div className="flex items-center justify-center mt-2 mb-3">
                <span className={`text-xs font-semibold ${risk.text} flex items-center gap-1.5`}>
                  <span className={`w-2 h-2 rounded-full ${risk.dot}`} />{risk.label}
                </span>
              </div>
              <div className="h-2 bg-white/60 rounded-full overflow-hidden mb-2">
                <motion.div initial={{ width: 0 }} animate={{ width: `${result.score}%` }} transition={{ duration: 0.6 }} className={`h-full rounded-full ${result.risk === 'green' ? 'bg-emerald-400' : result.risk === 'yellow' ? 'bg-amber-400' : 'bg-red-400'}`} />
              </div>
              <p className="text-xs text-gray-600 leading-relaxed">{result.explanation}</p>
              {result.risk !== 'green' && (
                <motion.button initial={{ opacity: 0 }} animate={{ opacity: 1 }} whileHover={{ scale: 1.02 }} whileTap={{ scale: 0.98 }} onClick={handleVerify} disabled={verifying}
                  className="mt-3 w-full py-2 text-xs font-semibold bg-white border border-gray-200 rounded-lg hover:bg-gray-50 flex items-center justify-center gap-1.5">
                  {verifying ? (<><svg className="w-3 h-3 animate-spin" viewBox="0 0 24 24" fill="none"><circle cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="3" className="opacity-25" /><path d="M4 12a8 8 0 018-8" stroke="currentColor" strokeWidth="3" strokeLinecap="round" className="opacity-75" /></svg>Verifying...</>) : (
                    <><svg className="w-3 h-3" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}><path strokeLinecap="round" strokeLinejoin="round" d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" /></svg>Auto-Verify & Correct</>
                  )}
                </motion.button>
              )}
              {verifyError && <p className="mt-2 text-xs text-red-500 text-center">{verifyError}</p>}
            </motion.div>
          )}
        </AnimatePresence>

        {/* Correction result */}
        <AnimatePresence>
          {correction && (
            <motion.div initial={{ opacity: 0, height: 0 }} animate={{ opacity: 1, height: 'auto' }} exit={{ opacity: 0, height: 0 }} className="rounded-xl p-4 bg-blue-50 border border-blue-200 space-y-2">
              <p className="text-xs font-semibold text-blue-700">AI Corrections Applied:</p>
              <ul className="space-y-1">
                {correction.corrections?.map((c, i) => (
                  <li key={i} className="text-xs text-blue-600 flex items-start gap-1.5">
                    <svg className="w-3 h-3 mt-0.5 flex-shrink-0 text-blue-400" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}><path strokeLinecap="round" strokeLinejoin="round" d="M5 13l4 4L19 7" /></svg>
                    {c}
                  </li>
                ))}
              </ul>
              <div className="pt-2 border-t border-blue-100">
                <p className="text-xs text-gray-500">Corrected address:</p>
                <p className="text-xs font-medium text-gray-700 mt-0.5">{correction.corrected_address}</p>
                <p className="text-xs text-gray-500 mt-1">PIN: {correction.corrected_pin}</p>
              </div>
            </motion.div>
          )}
        </AnimatePresence>

        {/* Example addresses */}
        {!result && (
          <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ delay: 0.5 }} className="pt-2">
            <p className="text-xs text-gray-400 mb-2">Try these examples:</p>
            <div className="space-y-1.5">
              {[
                { addr: '#42, 5th Main Road, Koramangala, Bengaluru', pin: '560034', label: 'Complete address' },
                { addr: 'Near ISKCON Temple, Rajajinagar', pin: '', label: 'Landmark only' },
                { addr: 'Bengaluru', pin: '110001', label: 'Vague + wrong PIN' },
              ].map((ex) => (
                <button key={ex.addr} onClick={() => { setAddress(ex.addr); setPinCode(ex.pin); setResult(null); setCorrection(null) }}
                  className="w-full text-left px-3 py-2 text-xs bg-gray-50/80 rounded-lg hover:bg-brand-50 hover:text-brand-700 border border-transparent hover:border-brand-200 hover:shadow-sm">
                  <span className="font-medium">{ex.label}</span>
                  <span className="block text-gray-400 truncate mt-0.5">{ex.addr}</span>
                </button>
              ))}
            </div>
          </motion.div>
        )}
      </div>
    </motion.div>
  )
}
