import { useState, useCallback } from 'react'
import { API_BASE } from '../config'

export function useAuditLog() {
  const [data, setData] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)

  const fetchAuditLog = useCallback(async (sessionId) => {
    setLoading(true)
    setError(null)
    try {
      const res = await fetch(`${API_BASE}/audit-log/${sessionId}`)
      if (!res.ok) throw new Error('Failed to fetch audit log')
      const json = await res.json()
      setData(json)
    } catch (err) {
      setError(err.message)
    } finally {
      setLoading(false)
    }
  }, [])

  return { data, loading, error, fetchAuditLog }
}
