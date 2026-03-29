import { useState } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { API_BASE } from '../config'

const DOC_TYPES = [
  { id: 'msme1', label: 'MSME-1 Filing', icon: '\u{1F4CB}' },
  { id: 'defense', label: 'Scrutiny Defense', icon: '\u{1F6E1}\u{FE0F}' },
  { id: 'schedule', label: 'Payment Schedule', icon: '\u{1F4C5}' },
]

const priorityBadge = {
  critical: 'bg-red-500/20 text-red-400 border-red-500/30',
  high: 'bg-orange-500/20 text-orange-400 border-orange-500/30',
  medium: 'bg-yellow-500/20 text-yellow-400 border-yellow-500/30',
  low: 'bg-green-500/20 text-green-400 border-green-500/30',
}

function MSME1Document({ data }) {
  return (
    <div className="bg-white/5 rounded-xl p-6 space-y-6 font-serif">
      {/* Header */}
      <div className="text-center border-b border-gray-700 pb-4">
        <h3 className="text-xl font-bold text-white tracking-wide">FORM MSME-1</h3>
        <p className="text-sm text-gray-400 mt-1">Half-Yearly Return on Outstanding Payments to MSME Suppliers</p>
        <p className="text-xs text-gray-500 mt-1">
          [Under Section 405 of the Companies Act, 2013 read with MSME Notification dt. 02.11.2018]
        </p>
      </div>

      {/* Filing Period */}
      <div className="bg-blue-500/10 border border-blue-500/20 rounded-lg p-4">
        <div className="text-sm text-blue-400 font-semibold">Filing Period</div>
        <div className="text-white font-mono mt-1">{data.filing_period}</div>
      </div>

      {/* Company Details */}
      <div className="grid grid-cols-2 gap-4">
        <div className="bg-gray-800/50 rounded-lg p-3">
          <div className="text-xs text-gray-500 uppercase tracking-wider">Company CIN</div>
          <div className="text-sm text-gray-300 mt-1 font-mono">{data.company_details?.note || 'To be filled'}</div>
        </div>
        <div className="bg-gray-800/50 rounded-lg p-3">
          <div className="text-xs text-gray-500 uppercase tracking-wider">MSME Vendors with Overdue</div>
          <div className="text-lg text-white font-bold mt-1">{data.summary?.total_msme_vendors_with_overdue || 0}</div>
        </div>
      </div>

      {/* Supplier Table */}
      <div>
        <h4 className="text-sm font-semibold text-gray-300 mb-3 uppercase tracking-wider">
          Details of Outstanding Amounts to Micro/Small Enterprise Suppliers
        </h4>
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead>
              <tr className="border-b border-gray-700">
                <th className="text-left py-2 px-3 text-gray-400 font-medium">Supplier Name</th>
                <th className="text-left py-2 px-3 text-gray-400 font-medium">Udyam Reg.</th>
                <th className="text-right py-2 px-3 text-gray-400 font-medium">Outstanding (INR)</th>
                <th className="text-right py-2 px-3 text-gray-400 font-medium">Overdue Days</th>
              </tr>
            </thead>
            <tbody>
              {(data.vendor_details || []).map((v, i) => (
                <tr key={i} className="border-b border-gray-800 hover:bg-white/5 transition-colors">
                  <td className="py-2 px-3 text-gray-200">{v.vendor_name}</td>
                  <td className="py-2 px-3 text-gray-400 font-mono text-xs">{v.msme_registration}</td>
                  <td className="py-2 px-3 text-right text-yellow-400 font-mono">
                    {parseFloat(v.outstanding_amount).toLocaleString('en-IN')}
                  </td>
                  <td className="py-2 px-3 text-right">
                    <span className={`px-2 py-0.5 rounded text-xs font-medium ${
                      v.overdue_days > 45 ? 'bg-red-500/20 text-red-400' : 'bg-yellow-500/20 text-yellow-400'
                    }`}>
                      {v.overdue_days}d
                    </span>
                  </td>
                </tr>
              ))}
            </tbody>
            <tfoot>
              <tr className="border-t border-gray-600">
                <td colSpan={2} className="py-2 px-3 text-gray-300 font-semibold">Total Outstanding</td>
                <td className="py-2 px-3 text-right text-yellow-300 font-bold font-mono">
                  {parseFloat(data.summary?.total_overdue_amount || 0).toLocaleString('en-IN')}
                </td>
                <td />
              </tr>
            </tfoot>
          </table>
        </div>
      </div>

      {/* Declaration */}
      <div className="border-t border-gray-700 pt-4 space-y-3">
        <h4 className="text-sm font-semibold text-gray-300">Declaration</h4>
        <p className="text-xs text-gray-400 leading-relaxed">
          I hereby declare that the information furnished above is true and correct to the best of
          my knowledge and belief. The details of outstanding payments to micro and small
          enterprises have been verified and are in accordance with the records maintained by the company.
        </p>
        <div className="grid grid-cols-2 gap-8 mt-4">
          <div className="border-t border-dashed border-gray-600 pt-2 mt-6">
            <p className="text-xs text-gray-500">Authorized Signatory</p>
          </div>
          <div className="border-t border-dashed border-gray-600 pt-2 mt-6">
            <p className="text-xs text-gray-500">Date & Company Seal</p>
          </div>
        </div>
      </div>

      {/* Disclaimer */}
      <div className="bg-yellow-500/10 border border-yellow-500/20 rounded-lg p-3">
        <p className="text-xs text-yellow-400">{data.disclaimer}</p>
      </div>
    </div>
  )
}

function DefenseDocument({ data }) {
  return (
    <div className="bg-white/5 rounded-xl p-6 space-y-6 font-serif">
      {/* Letterhead */}
      <div className="border-b border-gray-700 pb-4">
        <div className="flex justify-between items-start">
          <div>
            <h3 className="text-lg font-bold text-white">{data.title}</h3>
            <p className="text-sm text-gray-400 mt-1">Prepared for: {data.prepared_for}</p>
          </div>
          <div className="text-right">
            <div className="text-xs text-gray-500">Session: {data.session_id}</div>
            <div className="text-xs text-gray-500">Date: {new Date().toLocaleDateString('en-IN')}</div>
          </div>
        </div>
      </div>

      {/* Addressee */}
      <div className="bg-gray-800/50 rounded-lg p-4">
        <p className="text-sm text-gray-300 font-medium">To: The Assessing Officer</p>
        <p className="text-xs text-gray-400 mt-1">Income Tax Department, Government of India</p>
      </div>

      {/* Subject */}
      <div>
        <p className="text-sm text-gray-300">
          <span className="font-semibold">Subject:</span> Compliance Statement regarding
          Section 43B(h) of the Income Tax Act, 1961 - Payments to Micro and Small Enterprise Suppliers
        </p>
      </div>

      {/* Sections */}
      <div className="space-y-4">
        {(data.sections || []).map((section, i) => (
          <motion.div
            key={i}
            initial={{ opacity: 0, x: -10 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ delay: i * 0.1 }}
            className="bg-gray-800/30 rounded-lg p-4"
          >
            <h4 className="text-sm font-semibold text-blue-400 mb-2">{section.heading}</h4>
            <p className="text-sm text-gray-300 leading-relaxed">{section.content}</p>
          </motion.div>
        ))}
      </div>

      {/* Legal References */}
      <div className="bg-purple-500/10 border border-purple-500/20 rounded-lg p-4">
        <h4 className="text-sm font-semibold text-purple-400 mb-2">Applicable Legal Provisions</h4>
        <ul className="text-xs text-gray-400 space-y-1 list-disc list-inside">
          <li>MSMED Act 2006 - Sections 15, 16 (Payment obligations and interest)</li>
          <li>Income Tax Act 1961 - Section 43B(h) (Deduction disallowance)</li>
          <li>Companies Act 2013 - Section 405 (MSME-1 filing obligation)</li>
          <li>RBI Circular - Bank Rate effective 08.02.2023 (6.50%)</li>
        </ul>
      </div>

      {/* Prayer */}
      <div className="border-t border-gray-700 pt-4">
        <h4 className="text-sm font-semibold text-gray-300 mb-2">Prayer</h4>
        <p className="text-xs text-gray-400 leading-relaxed">
          In view of the above submissions and the compliance measures undertaken, it is
          respectfully submitted that the payments to MSME suppliers have been identified,
          quantified, and action plans have been generated for timely remediation. The undersigned
          requests the Hon&apos;ble Assessing Officer to consider the compliance efforts demonstrated herein.
        </p>
      </div>

      {/* Signature */}
      <div className="grid grid-cols-2 gap-8 mt-4">
        <div className="border-t border-dashed border-gray-600 pt-2 mt-6">
          <p className="text-xs text-gray-500">Authorized Signatory</p>
        </div>
        <div className="border-t border-dashed border-gray-600 pt-2 mt-6">
          <p className="text-xs text-gray-500">Date & Place</p>
        </div>
      </div>

      <div className="bg-yellow-500/10 border border-yellow-500/20 rounded-lg p-3">
        <p className="text-xs text-yellow-400">{data.disclaimer}</p>
      </div>
    </div>
  )
}

function ScheduleDocument({ data }) {
  return (
    <div className="bg-white/5 rounded-xl p-6 space-y-6">
      {/* Header */}
      <div className="border-b border-gray-700 pb-4">
        <h3 className="text-lg font-bold text-white">{data.title}</h3>
        <p className="text-sm text-gray-400 mt-1">
          {data.total_payments} payment{data.total_payments !== 1 ? 's' : ''} scheduled for compliance remediation
        </p>
      </div>

      {/* Schedule Table */}
      <div className="overflow-x-auto">
        <table className="w-full text-sm">
          <thead>
            <tr className="border-b border-gray-700">
              <th className="text-left py-2 px-3 text-gray-400 font-medium">#</th>
              <th className="text-left py-2 px-3 text-gray-400 font-medium">Vendor</th>
              <th className="text-right py-2 px-3 text-gray-400 font-medium">Amount (INR)</th>
              <th className="text-center py-2 px-3 text-gray-400 font-medium">Priority</th>
              <th className="text-left py-2 px-3 text-gray-400 font-medium">Deadline</th>
              <th className="text-left py-2 px-3 text-gray-400 font-medium">Legal Basis</th>
            </tr>
          </thead>
          <tbody>
            {(data.schedule || []).map((item, i) => (
              <motion.tr
                key={i}
                initial={{ opacity: 0, y: 10 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: i * 0.05 }}
                className="border-b border-gray-800 hover:bg-white/5 transition-colors"
              >
                <td className="py-3 px-3 text-gray-500 font-mono text-xs">{i + 1}</td>
                <td className="py-3 px-3 text-gray-200 font-medium">{item.vendor}</td>
                <td className="py-3 px-3 text-right text-yellow-400 font-mono font-medium">
                  {parseFloat(item.amount).toLocaleString('en-IN')}
                </td>
                <td className="py-3 px-3 text-center">
                  <span className={`px-2 py-1 rounded-full text-xs font-semibold border ${
                    priorityBadge[item.priority] || priorityBadge.medium
                  }`}>
                    {item.priority?.toUpperCase()}
                  </span>
                </td>
                <td className="py-3 px-3 text-gray-300 font-mono text-xs">{item.deadline}</td>
                <td className="py-3 px-3 text-gray-400 text-xs">{item.reason}</td>
              </motion.tr>
            ))}
          </tbody>
        </table>
      </div>

      {data.schedule?.length === 0 && (
        <div className="text-center py-8 text-gray-500">
          No priority payments scheduled. All MSME vendors are compliant.
        </div>
      )}

      {/* Total */}
      {data.schedule?.length > 0 && (
        <div className="bg-gradient-to-r from-yellow-500/10 to-orange-500/10 border border-yellow-500/20 rounded-lg p-4 flex justify-between items-center">
          <span className="text-sm text-gray-300 font-semibold">Total Payment Required</span>
          <span className="text-lg text-yellow-400 font-bold font-mono">
            INR {data.schedule.reduce((sum, s) => sum + parseFloat(s.amount || 0), 0).toLocaleString('en-IN')}
          </span>
        </div>
      )}
    </div>
  )
}

const DOC_RENDERERS = {
  msme1: MSME1Document,
  defense: DefenseDocument,
  schedule: ScheduleDocument,
}

export default function DocumentViewer({ sessionId, visible }) {
  const [activeDoc, setActiveDoc] = useState(null)
  const [docData, setDocData] = useState(null)
  const [loading, setLoading] = useState(false)

  if (!visible || !sessionId) return null

  const fetchDoc = async (type) => {
    setLoading(true)
    setActiveDoc(type)
    try {
      const res = await fetch(`${API_BASE}/documents/${sessionId}/${type}`)
      const data = await res.json()
      setDocData(data)
    } catch {
      setDocData({ error: 'Failed to load document' })
    } finally {
      setLoading(false)
    }
  }

  const Renderer = activeDoc ? DOC_RENDERERS[activeDoc] : null

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      className="glass-card p-6"
    >
      <div className="flex items-center justify-between mb-4">
        <h2 className="text-lg font-semibold">Generated Legal Documents</h2>
        <button
          className="px-3 py-1.5 rounded-lg text-xs bg-gray-800 text-gray-400 hover:bg-gray-700 transition-colors cursor-not-allowed opacity-50"
          title="PDF export coming soon"
        >
          Download PDF
        </button>
      </div>

      {/* Tab navigation */}
      <div className="flex gap-2 mb-5">
        {DOC_TYPES.map(doc => (
          <button
            key={doc.id}
            onClick={() => fetchDoc(doc.id)}
            className={`px-4 py-2.5 rounded-xl text-sm font-medium transition-all ${
              activeDoc === doc.id
                ? 'bg-blue-600 text-white shadow-lg shadow-blue-500/20'
                : 'bg-gray-800 text-gray-400 hover:bg-gray-700 hover:text-gray-300'
            }`}
          >
            <span className="mr-1.5">{doc.icon}</span>
            {doc.label}
          </button>
        ))}
      </div>

      {/* Loading state */}
      {loading && (
        <div className="flex items-center justify-center py-12">
          <div className="w-6 h-6 border-2 border-blue-500 border-t-transparent rounded-full animate-spin mr-3" />
          <span className="text-sm text-gray-400">Generating document...</span>
        </div>
      )}

      {/* Document content */}
      <AnimatePresence mode="wait">
        {docData && !loading && !docData.error && Renderer && (
          <motion.div
            key={activeDoc}
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -10 }}
            className="max-h-[500px] overflow-y-auto scrollbar-thin"
          >
            <Renderer data={docData} />
          </motion.div>
        )}
      </AnimatePresence>

      {/* Error state */}
      {docData?.error && !loading && (
        <div className="text-center py-8 text-red-400 text-sm">{docData.error}</div>
      )}

      {/* Empty state */}
      {!activeDoc && !loading && (
        <div className="text-center py-8 text-gray-500 text-sm">
          Select a document type above to preview
        </div>
      )}
    </motion.div>
  )
}
