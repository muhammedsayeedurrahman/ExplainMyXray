import { useEffect } from 'react'
import { motion } from 'framer-motion'
import { useAuditLog } from '../hooks/useAuditLog'

const stepIcons = {
  pipeline_start: '🚀',
  vendor_scan: '🔍',
  msme_detection: '🏭',
  due_date_calc: '📅',
  interest_calc: '💰',
  tax_impact_calc: '📊',
  risk_score_calc: '🎯',
  rule_evaluation: '⚖️',
  violation_detected: '⚠️',
  action_planned: '📋',
  pipeline_complete: '✅',
}

export default function AuditTrail({ sessionId, visible }) {
  const { data, fetchAuditLog } = useAuditLog()

  useEffect(() => {
    if (visible && sessionId) {
      fetchAuditLog(sessionId)
    }
  }, [visible, sessionId, fetchAuditLog])

  if (!visible || !data) return null

  // Show key events only (skip per-vendor detail for brevity)
  const keyEvents = data.events.filter(e =>
    ['pipeline_start', 'vendor_scan', 'action_planned', 'pipeline_complete'].includes(e.event_type) ||
    e.step !== data.events.find(ev => ev.step === e.step && ev !== e)?.step
  ).slice(0, 20)

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      className="glass-card p-6"
    >
      <h2 className="text-lg font-semibold mb-1">Audit Trail</h2>
      <p className="text-xs text-gray-500 mb-4">
        {data.total_events} events | {data.pipeline_steps_completed} pipeline steps
      </p>

      <div className="space-y-1 max-h-72 overflow-y-auto">
        {keyEvents.map((event, i) => (
          <motion.div
            key={event.event_id}
            initial={{ opacity: 0, x: -10 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ delay: i * 0.04 }}
            className="flex items-start gap-3 p-2 rounded-lg hover:bg-gray-800/30 transition-colors text-sm"
          >
            <span className="text-base shrink-0 mt-0.5">{stepIcons[event.event_type] || '📝'}</span>
            <div className="flex-1 min-w-0">
              <span className="text-gray-200">{event.description}</span>
              {event.law_reference && (
                <span className="text-xs text-blue-400 ml-2 font-mono">[{event.law_reference}]</span>
              )}
            </div>
            <span className="text-xs text-gray-600 font-mono shrink-0">
              Step {event.step}
            </span>
          </motion.div>
        ))}
      </div>
    </motion.div>
  )
}
