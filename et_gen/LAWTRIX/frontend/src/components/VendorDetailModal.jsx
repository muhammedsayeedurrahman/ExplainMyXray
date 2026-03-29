import { motion } from 'framer-motion'

const riskColors = {
  critical: 'text-red-400',
  high: 'text-orange-400',
  medium: 'text-yellow-400',
  low: 'text-green-400',
  compliant: 'text-emerald-400',
}

export default function VendorDetailModal({ vendor, onClose }) {
  if (!vendor) return null

  const formatINR = (v) => parseFloat(v).toLocaleString('en-IN', { maximumFractionDigits: 0 })

  return (
    <motion.div
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      exit={{ opacity: 0 }}
      onClick={onClose}
      className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 backdrop-blur-sm p-4"
    >
      <motion.div
        initial={{ scale: 0.9, y: 30 }}
        animate={{ scale: 1, y: 0 }}
        exit={{ scale: 0.9, y: 30 }}
        onClick={e => e.stopPropagation()}
        className="glass-card p-6 max-w-lg w-full max-h-[80vh] overflow-y-auto"
      >
        <div className="flex justify-between items-start mb-6">
          <div>
            <h2 className="text-xl font-bold">{vendor.vendor_name}</h2>
            <p className="text-sm text-gray-400 capitalize">{vendor.msme_category} enterprise | {vendor.vendor_id}</p>
          </div>
          <button onClick={onClose} className="text-gray-500 hover:text-gray-300 text-2xl leading-none">&times;</button>
        </div>

        <div className="grid grid-cols-2 gap-4 mb-6">
          <Stat label="Risk Score" value={`${vendor.risk_score}/100`} className={riskColors[vendor.risk_level]} />
          <Stat label="Risk Level" value={vendor.risk_level.toUpperCase()} className={riskColors[vendor.risk_level]} />
          <Stat label="Overdue Invoices" value={vendor.overdue_invoices} />
          <Stat label="Max Delay" value={`${vendor.max_delay_days} days`} />
          <Stat label="Overdue Amount" value={`INR ${formatINR(vendor.overdue_amount)}`} className="text-red-400" />
          <Stat label="Interest Liability" value={`INR ${formatINR(vendor.interest_liability)}`} className="text-orange-400" />
          <Stat label="Tax Disallowance" value={`INR ${formatINR(vendor.tax_disallowance)}`} className="text-purple-400" />
          <Stat label="MSME-1 Filing" value={vendor.requires_msme1_filing ? 'REQUIRED' : 'Not needed'} className={vendor.requires_msme1_filing ? 'text-red-400' : 'text-green-400'} />
        </div>

        {vendor.violations?.length > 0 && (
          <div className="mb-4">
            <h3 className="text-sm font-semibold text-gray-400 mb-2">Violations</h3>
            <div className="flex flex-wrap gap-2">
              {vendor.violations.map(v => (
                <span key={v} className="px-2 py-1 rounded-lg bg-red-500/10 border border-red-500/30 text-red-400 text-xs font-mono">
                  {v}
                </span>
              ))}
            </div>
          </div>
        )}

        {vendor.recommended_actions?.length > 0 && (
          <div>
            <h3 className="text-sm font-semibold text-gray-400 mb-2">Recommended Actions</h3>
            <ul className="space-y-1">
              {vendor.recommended_actions.map((a, i) => (
                <li key={i} className="text-xs text-gray-300 pl-3 border-l-2 border-blue-500/50">{a}</li>
              ))}
            </ul>
          </div>
        )}
      </motion.div>
    </motion.div>
  )
}

function Stat({ label, value, className = '' }) {
  return (
    <div className="bg-gray-800/30 rounded-xl p-3">
      <div className="text-xs text-gray-500">{label}</div>
      <div className={`text-sm font-bold font-mono mt-1 ${className}`}>{value}</div>
    </div>
  )
}
