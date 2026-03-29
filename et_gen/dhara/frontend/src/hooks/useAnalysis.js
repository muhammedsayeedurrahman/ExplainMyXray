import { useState, useCallback } from 'react'
import { API_BASE } from '../config'

export function useAnalysis() {
  const [data, setData] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)

  const fetchAnalysis = useCallback(async (sessionId) => {
    setLoading(true)
    setError(null)
    try {
      const res = await fetch(`${API_BASE}/analysis/${sessionId}`)
      if (!res.ok) throw new Error('Failed to fetch analysis')
      const json = await res.json()
      setData(json)
    } catch (err) {
      setError(err.message)
    } finally {
      setLoading(false)
    }
  }, [])

  return { data, loading, error, fetchAnalysis }
}
