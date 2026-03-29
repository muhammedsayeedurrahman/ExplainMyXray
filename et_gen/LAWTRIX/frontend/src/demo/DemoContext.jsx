import { createContext, useContext, useState, useCallback } from 'react'
import { API_BASE } from '../config'

const DemoContext = createContext(null)

const DEMO_STEPS = [
  { id: 'intro', name: 'LAWTRIX', duration: 2500 },
  { id: 'upload', name: 'Data Ingestion', duration: 2500 },
  { id: 'detection', name: 'MSME Detection', duration: 3000 },
  { id: 'engine', name: 'Legal Engine', duration: 4000 },
  { id: 'actions', name: 'Autonomous Actions', duration: 6000 },
  { id: 'impact', name: 'Impact Reveal', duration: 4000 },
  { id: 'complete', name: 'Demo Complete', duration: 3000 },
]

export function DemoProvider({ children }) {
  const [demoState, setDemoState] = useState({
    isRunning: false,
    currentStep: -1,
    data: null,
    sessionId: null,
    error: null,
  })

  const startDemo = useCallback(async () => {
    setDemoState(s => ({ ...s, isRunning: true, currentStep: 0, error: null }))

    try {
      // Step 0: Intro
      await delay(DEMO_STEPS[0].duration)

      // Step 1: Upload - fetch demo data
      setDemoState(s => ({ ...s, currentStep: 1 }))
      const res = await fetch(`${API_BASE}/demo/run`)
      if (!res.ok) throw new Error('Demo API failed')
      const data = await res.json()
      setDemoState(s => ({ ...s, data, sessionId: data.session_id }))
      await delay(DEMO_STEPS[1].duration)

      // Step 2: Detection
      setDemoState(s => ({ ...s, currentStep: 2 }))
      await delay(DEMO_STEPS[2].duration)

      // Step 3: Legal Engine
      setDemoState(s => ({ ...s, currentStep: 3 }))
      await delay(DEMO_STEPS[3].duration)

      // Step 4: Actions
      setDemoState(s => ({ ...s, currentStep: 4 }))
      await delay(DEMO_STEPS[4].duration)

      // Step 5: Impact
      setDemoState(s => ({ ...s, currentStep: 5 }))
      await delay(DEMO_STEPS[5].duration)

      // Step 6: Complete
      setDemoState(s => ({ ...s, currentStep: 6 }))
    } catch (err) {
      setDemoState(s => ({ ...s, error: err.message, isRunning: false }))
    }
  }, [])

  const resetDemo = useCallback(() => {
    setDemoState({
      isRunning: false,
      currentStep: -1,
      data: null,
      sessionId: null,
      error: null,
    })
  }, [])

  return (
    <DemoContext.Provider value={{ ...demoState, steps: DEMO_STEPS, startDemo, resetDemo }}>
      {children}
    </DemoContext.Provider>
  )
}

export function useDemo() {
  const ctx = useContext(DemoContext)
  if (!ctx) throw new Error('useDemo must be used within DemoProvider')
  return ctx
}

function delay(ms) {
  return new Promise(resolve => setTimeout(resolve, ms))
}
