import { useState, useEffect } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { API_BASE } from '../config'

const NODE_COLORS = {
  inputNode: { bg: 'from-blue-600/30 to-blue-800/30', border: 'border-blue-500/40', text: 'text-blue-400' },
  switchNode: { bg: 'from-purple-600/30 to-purple-800/30', border: 'border-purple-500/40', text: 'text-purple-400' },
  decisionTableNode: { bg: 'from-cyan-600/30 to-cyan-800/30', border: 'border-cyan-500/40', text: 'text-cyan-400' },
  outputNode: { bg: 'from-green-600/30 to-green-800/30', border: 'border-green-500/40', text: 'text-green-400' },
}

const SEVERITY_COLORS = {
  critical: 'bg-red-500/20 text-red-400 border-red-500/30',
  high: 'bg-orange-500/20 text-orange-400 border-orange-500/30',
  medium: 'bg-yellow-500/20 text-yellow-400 border-yellow-500/30',
  low: 'bg-green-500/20 text-green-400 border-green-500/30',
}

function FlowNode({ node, index, triggered }) {
  const colors = NODE_COLORS[node.type] || NODE_COLORS.inputNode
  const isTriggered = triggered?.some(r => r.table === node.name)

  return (
    <motion.div
      initial={{ opacity: 0, scale: 0.8 }}
      animate={{ opacity: 1, scale: 1 }}
      transition={{ delay: index * 0.15, type: 'spring', stiffness: 200 }}
      className={`relative bg-gradient-to-br ${colors.bg} border ${colors.border} rounded-xl p-4 min-w-[180px] ${
        isTriggered ? 'ring-2 ring-green-400/50 shadow-lg shadow-green-500/10' : ''
      }`}
    >
      {/* Type badge */}
      <div className="flex items-center gap-2 mb-2">
        <div className={`w-2 h-2 rounded-full ${isTriggered ? 'bg-green-400 animate-pulse' : 'bg-gray-500'}`} />
        <span className="text-[10px] text-gray-500 uppercase tracking-wider font-mono">
          {node.type.replace('Node', '').replace('decisionTable', 'Decision Table')}
        </span>
      </div>
      <h4 className={`text-sm font-semibold ${colors.text}`}>{node.name}</h4>

      {/* Decision table preview */}
      {node.type === 'decisionTableNode' && node.content?.rules && (
        <div className="mt-3 space-y-1">
          {node.content.rules.slice(0, 4).map((rule, i) => (
            <div
              key={i}
              className={`text-[10px] px-2 py-1 rounded ${
                triggered?.some(t => t.rule_id === rule.o1)
                  ? 'bg-green-500/20 text-green-300 border border-green-500/30'
                  : 'bg-gray-800/50 text-gray-500'
              }`}
            >
              {rule._description || rule.o1}
            </div>
          ))}
        </div>
      )}
    </motion.div>
  )
}

function FlowArrow({ delay }) {
  return (
    <motion.div
      initial={{ opacity: 0, scaleX: 0 }}
      animate={{ opacity: 1, scaleX: 1 }}
      transition={{ delay, duration: 0.3 }}
      className="flex items-center mx-2"
    >
      <div className="w-8 h-0.5 bg-gradient-to-r from-gray-600 to-gray-400" />
      <div className="w-0 h-0 border-t-[5px] border-t-transparent border-b-[5px] border-b-transparent border-l-[8px] border-l-gray-400" />
    </motion.div>
  )
}

function DecisionTableGrid({ rules, triggered }) {
  if (!rules?.length) return null

  return (
    <div className="overflow-x-auto">
      <table className="w-full text-xs">
        <thead>
          <tr className="border-b border-gray-700">
            <th className="text-left py-2 px-2 text-gray-400">Rule</th>
            <th className="text-left py-2 px-2 text-gray-400">Severity</th>
            <th className="text-left py-2 px-2 text-gray-400">Action</th>
            <th className="text-left py-2 px-2 text-gray-400">Law Reference</th>
            <th className="text-center py-2 px-2 text-gray-400">Status</th>
          </tr>
        </thead>
        <tbody>
          {rules.map((rule, i) => {
            const isTriggered = triggered?.some(t => t.rule_id === rule.rule_id)
            return (
              <motion.tr
                key={i}
                initial={{ opacity: 0, x: -10 }}
                animate={{ opacity: 1, x: 0 }}
                transition={{ delay: i * 0.05 }}
                className={`border-b border-gray-800 ${isTriggered ? 'bg-green-500/10' : ''}`}
              >
                <td className="py-2 px-2 font-mono text-gray-300">
                  {rule.rule_id}
                  <div className="text-gray-500 font-sans">{rule.description}</div>
                </td>
                <td className="py-2 px-2">
                  <span className={`px-1.5 py-0.5 rounded text-[10px] font-semibold border ${
                    SEVERITY_COLORS[rule.severity] || ''
                  }`}>
                    {rule.severity?.toUpperCase()}
                  </span>
                </td>
                <td className="py-2 px-2 text-gray-400 max-w-[200px]">{rule.action}</td>
                <td className="py-2 px-2 text-gray-500 font-mono text-[10px]">{rule.law_reference}</td>
                <td className="py-2 px-2 text-center">
                  {isTriggered ? (
                    <span className="text-green-400 font-bold">TRIGGERED</span>
                  ) : (
                    <span className="text-gray-600">-</span>
                  )}
                </td>
              </motion.tr>
            )
          })}
        </tbody>
      </table>
    </div>
  )
}

export default function RuleVisualizer({ rulesEvaluated, visible }) {
  const [rulesSummary, setRulesSummary] = useState(null)
  const [engineInfo, setEngineInfo] = useState(null)
  const [jdm, setJdm] = useState(null)
  const [showGrid, setShowGrid] = useState(false)

  useEffect(() => {
    if (!visible) return
    // Fetch rules summary and engine info in parallel
    Promise.all([
      fetch(`${API_BASE}/rules/summary`).then(r => r.json()).catch(() => null),
      fetch(`${API_BASE}/rules/engine-info`).then(r => r.json()).catch(() => null),
      fetch(`${API_BASE}/rules/jdm`).then(r => r.json()).catch(() => null),
    ]).then(([summary, info, jdmData]) => {
      setRulesSummary(summary)
      setEngineInfo(info)
      setJdm(jdmData)
    })
  }, [visible])

  if (!visible) return null

  const triggeredRules = rulesEvaluated?.triggered_rules || []
  const flowNodes = jdm?.nodes?.filter(n => n.type !== 'outputNode' || n.id === 'output') || []

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      className="glass-card p-6 space-y-6"
    >
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-lg font-semibold">Rules Engine Visualization</h2>
          <p className="text-xs text-gray-500 mt-1">
            JSON Decision Model (JDM) - powered by zen-engine
          </p>
        </div>
        <div className="flex items-center gap-3">
          {rulesEvaluated?.timing_ms != null && (
            <motion.div
              initial={{ scale: 0 }}
              animate={{ scale: 1 }}
              className="px-3 py-1.5 rounded-full bg-green-500/10 border border-green-500/20"
            >
              <span className="text-xs text-green-400 font-mono font-bold">
                {rulesEvaluated.timing_ms.toFixed(3)}ms
              </span>
            </motion.div>
          )}
          <div className="px-3 py-1.5 rounded-full bg-purple-500/10 border border-purple-500/20">
            <span className="text-xs text-purple-400 font-mono">
              {engineInfo?.engine || rulesEvaluated?.engine || 'zen-engine'}
            </span>
          </div>
        </div>
      </div>

      {/* Engine badge */}
      <div className="bg-gradient-to-r from-gray-800/50 to-gray-900/50 rounded-xl p-4 flex items-center gap-4">
        <div className="flex-shrink-0 w-12 h-12 bg-gradient-to-br from-orange-500/20 to-red-500/20 border border-orange-500/30 rounded-xl flex items-center justify-center">
          <span className="text-lg font-black text-orange-400">Z</span>
        </div>
        <div className="flex-1">
          <div className="flex items-center gap-2">
            <span className="text-sm font-semibold text-white">gorules/zen</span>
            <span className="px-1.5 py-0.5 rounded text-[10px] bg-orange-500/20 text-orange-400 border border-orange-500/30 font-mono">
              Rust
            </span>
            <span className="px-1.5 py-0.5 rounded text-[10px] bg-blue-500/20 text-blue-400 border border-blue-500/30">
              Sub-ms Latency
            </span>
          </div>
          <p className="text-xs text-gray-400 mt-1">
            {engineInfo?.total_rules || 8} rules across {engineInfo?.decision_tables || 3} decision tables |{' '}
            {engineInfo?.laws_encoded?.length || 3} Indian statutes encoded
          </p>
        </div>
        <div className="text-right">
          <div className="text-xs text-gray-500">Triggered</div>
          <div className="text-2xl font-bold text-green-400">{triggeredRules.length}</div>
        </div>
      </div>

      {/* Flow diagram */}
      <div>
        <h3 className="text-sm font-semibold text-gray-300 mb-3">Decision Flow</h3>
        <div className="flex items-start overflow-x-auto pb-4 gap-0">
          {flowNodes.map((node, i) => (
            <div key={node.id} className="flex items-center flex-shrink-0">
              <FlowNode node={node} index={i} triggered={triggeredRules} />
              {i < flowNodes.length - 1 && <FlowArrow delay={i * 0.15 + 0.1} />}
            </div>
          ))}
        </div>
      </div>

      {/* Toggle grid view */}
      <button
        onClick={() => setShowGrid(!showGrid)}
        className="text-xs text-blue-400 hover:text-blue-300 transition-colors"
      >
        {showGrid ? 'Hide' : 'Show'} Decision Table Grid
      </button>

      {/* Decision table grid */}
      <AnimatePresence>
        {showGrid && rulesSummary?.rules && (
          <motion.div
            initial={{ opacity: 0, height: 0 }}
            animate={{ opacity: 1, height: 'auto' }}
            exit={{ opacity: 0, height: 0 }}
          >
            <DecisionTableGrid rules={rulesSummary.rules} triggered={triggeredRules} />
          </motion.div>
        )}
      </AnimatePresence>

      {/* Laws encoded */}
      {engineInfo?.laws_encoded && (
        <div className="flex flex-wrap gap-2">
          {engineInfo.laws_encoded.map((law, i) => (
            <span
              key={i}
              className="px-2 py-1 rounded-lg text-[10px] bg-gray-800/50 text-gray-400 border border-gray-700"
            >
              {law}
            </span>
          ))}
        </div>
      )}
    </motion.div>
  )
}
