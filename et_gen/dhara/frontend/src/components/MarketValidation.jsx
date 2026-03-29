import { useState, useEffect } from 'react'
import { motion } from 'framer-motion'
import { API_BASE } from '../config'

const CHECK = '\u2705'
const CROSS = '\u274C'
const PARTIAL = '\u{1F7E1}'

function FeatureCell({ value }) {
  if (value === true) return <span className="text-green-400 text-lg">{CHECK}</span>
  if (value === false) return <span className="text-red-400 text-lg">{CROSS}</span>
  if (value === 'Partial') return <span className="text-yellow-400 text-lg">{PARTIAL}</span>
  return <span className="text-gray-400 text-xs">{String(value)}</span>
}

export default function MarketValidation({ visible }) {
  const [data, setData] = useState(null)

  useEffect(() => {
    if (!visible) return
    fetch(`${API_BASE}/market/validation`)
      .then(r => r.json())
      .then(setData)
      .catch(() => null)
  }, [visible])

  if (!visible) return null

  const competitors = data?.competitors || []
  const lawtrix = competitors.find(c => c.tool === 'LAWTRIX')
  const others = competitors.filter(c => c.tool !== 'LAWTRIX')

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      className="glass-card p-6 space-y-6"
    >
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-lg font-semibold">Market Validation</h2>
          <p className="text-xs text-gray-500 mt-1">Competitor landscape analysis</p>
        </div>
        {data?.market_gap && (
          <a
            href={data.market_gap.issue_url}
            target="_blank"
            rel="noopener noreferrer"
            className="px-3 py-1.5 rounded-lg text-xs bg-purple-500/10 border border-purple-500/20 text-purple-400 hover:bg-purple-500/20 transition-colors"
          >
            india-compliance #3086
          </a>
        )}
      </div>

      {/* Feature gap callout */}
      {data?.market_gap && (
        <motion.div
          initial={{ opacity: 0, scale: 0.95 }}
          animate={{ opacity: 1, scale: 1 }}
          transition={{ delay: 0.2 }}
          className="bg-gradient-to-r from-purple-500/10 to-blue-500/10 border border-purple-500/20 rounded-xl p-4"
        >
          <p className="text-sm text-purple-300 font-medium">
            &quot;{data.market_gap.description}&quot;
          </p>
        </motion.div>
      )}

      {/* Comparison table */}
      <div className="overflow-x-auto">
        <table className="w-full text-sm">
          <thead>
            <tr className="border-b border-gray-700">
              <th className="text-left py-3 px-3 text-gray-400 font-medium">Tool</th>
              <th className="text-left py-3 px-3 text-gray-400 font-medium">Type</th>
              <th className="text-center py-3 px-3 text-gray-400 font-medium">MSME Compliance</th>
              <th className="text-center py-3 px-3 text-gray-400 font-medium">43B(h)</th>
              <th className="text-center py-3 px-3 text-gray-400 font-medium">MSME-1</th>
              <th className="text-center py-3 px-3 text-gray-400 font-medium">Autonomous</th>
            </tr>
          </thead>
          <tbody>
            {others.map((comp, i) => (
              <motion.tr
                key={comp.tool}
                initial={{ opacity: 0, x: -20 }}
                animate={{ opacity: 1, x: 0 }}
                transition={{ delay: 0.3 + i * 0.08 }}
                className="border-b border-gray-800 hover:bg-white/5 transition-colors"
              >
                <td className="py-3 px-3">
                  <div className="text-gray-300 font-medium">{comp.tool}</div>
                  {comp.stars !== 'N/A' && (
                    <div className="text-[10px] text-gray-500 mt-0.5">{comp.stars} stars</div>
                  )}
                </td>
                <td className="py-3 px-3 text-gray-400 text-xs">{comp.type}</td>
                <td className="py-3 px-3 text-center"><FeatureCell value={comp.msme_compliance} /></td>
                <td className="py-3 px-3 text-center"><FeatureCell value={comp.section_43bh} /></td>
                <td className="py-3 px-3 text-center"><FeatureCell value={comp.msme1_filing} /></td>
                <td className="py-3 px-3 text-center"><FeatureCell value={comp.autonomous} /></td>
              </motion.tr>
            ))}

            {/* LAWTRIX row - highlighted */}
            {lawtrix && (
              <motion.tr
                initial={{ opacity: 0, scale: 0.95 }}
                animate={{ opacity: 1, scale: 1 }}
                transition={{ delay: 0.8 }}
                className="bg-green-500/10 border-2 border-green-500/30"
              >
                <td className="py-3 px-3">
                  <div className="text-green-400 font-bold text-base">LAWTRIX</div>
                  <div className="text-[10px] text-green-500 mt-0.5">{lawtrix.stars}</div>
                </td>
                <td className="py-3 px-3 text-green-400 text-xs font-medium">{lawtrix.type}</td>
                <td className="py-3 px-3 text-center"><FeatureCell value={lawtrix.msme_compliance} /></td>
                <td className="py-3 px-3 text-center"><FeatureCell value={lawtrix.section_43bh} /></td>
                <td className="py-3 px-3 text-center"><FeatureCell value={lawtrix.msme1_filing} /></td>
                <td className="py-3 px-3 text-center"><FeatureCell value={lawtrix.autonomous} /></td>
              </motion.tr>
            )}
          </tbody>
        </table>
      </div>

      {/* Unique value */}
      {data?.unique_value && (
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ delay: 1 }}
          className="bg-green-500/10 border border-green-500/20 rounded-xl p-4 text-center"
        >
          <p className="text-sm text-green-300 font-medium">{data.unique_value}</p>
        </motion.div>
      )}
    </motion.div>
  )
}
