import { useState } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import VendorDetailModal from './VendorDetailModal'

const riskColors = {
  critical: 'bg-red-500/20 text-red-400 border-red-500/40',
  high: 'bg-orange-500/20 text-orange-400 border-orange-500/40',
  medium: 'bg-yellow-500/20 text-yellow-400 border-yellow-500/40',
  low: 'bg-green-500/20 text-green-400 border-green-500/40',
  compliant: 'bg-emerald-500/20 text-emerald-400 border-emerald-500/40',
}

const riskBarColors = {
  critical: 'bg-red-500',
  high: 'bg-orange-500',
  medium: 'bg-yellow-500',
  low: 'bg-green-500',
  compliant: 'bg-emerald-500',
}

export default function VendorTable({ vendors, visible }) {
  const [selected, setSelected] = useState(null)
  const [sortBy, setSortBy] = useState('risk_score')
  const [sortDir, setSortDir] = useState('desc')

  if (!visible || !vendors?.length) return null

  const sorted = [...vendors].sort((a, b) => {
    const av = a[sortBy], bv = b[sortBy]
    if (typeof av === 'number') return sortDir === 'desc' ? bv - av : av - bv
    return sortDir === 'desc' ? String(bv).localeCompare(String(av)) : String(av).localeCompare(String(bv))
  })

  const handleSort = (col) => {
    if (sortBy === col) {
      setSortDir(d => d === 'desc' ? 'asc' : 'desc')
    } else {
      setSortBy(col)
      setSortDir('desc')
    }
  }

  return (
    <>
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        className="glass-card overflow-hidden"
      >
        <div className="p-5 border-b border-gray-800">
          <h2 className="text-lg font-semibold">MSME Vendor Compliance</h2>
          <p className="text-xs text-gray-500 mt-1">Click any vendor for detailed breakdown</p>
        </div>
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead>
              <tr className="text-left text-xs text-gray-500 border-b border-gray-800/50">
                {[
                  { key: 'vendor_name', label: 'Vendor' },
                  { key: 'msme_category', label: 'Category' },
                  { key: 'overdue_invoices', label: 'Overdue' },
                  { key: 'overdue_amount', label: 'Overdue Amt' },
                  { key: 'interest_liability', label: 'Interest' },
                  { key: 'max_delay_days', label: 'Max Delay' },
                  { key: 'risk_score', label: 'Risk' },
                ].map(col => (
                  <th
                    key={col.key}
                    onClick={() => handleSort(col.key)}
                    className="px-4 py-3 cursor-pointer hover:text-gray-300 transition-colors"
                  >
                    {col.label} {sortBy === col.key && (sortDir === 'desc' ? '↓' : '↑')}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {sorted.map((v, i) => (
                <motion.tr
                  key={v.vendor_id}
                  initial={{ opacity: 0, x: -20 }}
                  animate={{ opacity: 1, x: 0 }}
                  transition={{ delay: i * 0.03 }}
                  onClick={() => setSelected(v)}
                  className="border-b border-gray-800/30 hover:bg-gray-800/30 cursor-pointer transition-colors"
                >
                  <td className="px-4 py-3 font-medium">{v.vendor_name}</td>
                  <td className="px-4 py-3">
                    <span className="px-2 py-0.5 rounded-full text-xs bg-gray-800 capitalize">
                      {v.msme_category}
                    </span>
                  </td>
                  <td className="px-4 py-3 font-mono">{v.overdue_invoices}</td>
                  <td className="px-4 py-3 font-mono text-red-400">
                    {parseFloat(v.overdue_amount).toLocaleString('en-IN', { maximumFractionDigits: 0 })}
                  </td>
                  <td className="px-4 py-3 font-mono text-orange-400">
                    {parseFloat(v.interest_liability).toLocaleString('en-IN', { maximumFractionDigits: 0 })}
                  </td>
                  <td className="px-4 py-3 font-mono">{v.max_delay_days}d</td>
                  <td className="px-4 py-3">
                    <div className="flex items-center gap-2">
                      <div className="w-16 h-2 bg-gray-800 rounded-full overflow-hidden">
                        <div
                          className={`h-full rounded-full ${riskBarColors[v.risk_level]}`}
                          style={{ width: `${v.risk_score}%` }}
                        />
                      </div>
                      <span className={`text-xs px-2 py-0.5 rounded-full border ${riskColors[v.risk_level]}`}>
                        {v.risk_score}
                      </span>
                    </div>
                  </td>
                </motion.tr>
              ))}
            </tbody>
          </table>
        </div>
      </motion.div>

      <AnimatePresence>
        {selected && (
          <VendorDetailModal vendor={selected} onClose={() => setSelected(null)} />
        )}
      </AnimatePresence>
    </>
  )
}
