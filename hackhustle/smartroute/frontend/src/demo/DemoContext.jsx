import { createContext, useContext, useState, useCallback } from 'react'
import {
  DEMO_ORDERS, DEMO_METRICS, DEMO_INSIGHTS,
  DEMO_SCORE_RESULT, DEMO_VERIFY_RESULT,
} from './demoData'

const DemoContext = createContext(null)

export function DemoProvider({ children }) {
  const [isDemo, setIsDemo] = useState(false)

  const startDemo = useCallback(() => setIsDemo(true), [])
  const stopDemo = useCallback(() => setIsDemo(false), [])

  return (
    <DemoContext.Provider value={{
      isDemo,
      startDemo,
      stopDemo,
      demoOrders: DEMO_ORDERS,
      demoMetrics: DEMO_METRICS,
      demoInsights: DEMO_INSIGHTS,
      demoScoreResult: DEMO_SCORE_RESULT,
      demoVerifyResult: DEMO_VERIFY_RESULT,
    }}>
      {children}
    </DemoContext.Provider>
  )
}

export function useDemo() {
  const ctx = useContext(DemoContext)
  if (!ctx) throw new Error('useDemo must be used within DemoProvider')
  return ctx
}
