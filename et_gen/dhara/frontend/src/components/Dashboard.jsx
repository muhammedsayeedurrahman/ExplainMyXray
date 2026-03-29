import { useState } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import Sidebar from './Sidebar'
import OverviewTab from './OverviewTab'
import VendorTable from './VendorTable'
import RiskHeatmap from './RiskHeatmap'
import ComplianceScore from './ComplianceScore'
import ActionPanel from './ActionPanel'
import PaymentTimeline from './PaymentTimeline'
import DocumentViewer from './DocumentViewer'
import AuditTrail from './AuditTrail'
import RuleVisualizer from './RuleVisualizer'
import ArchitectureShowcase from './ArchitectureShowcase'
import MarketValidation from './MarketValidation'
import LawCard from './LawCard'
import ImpactSummary from './ImpactSummary'

const TAB_TITLES = {
  overview: 'Dashboard Overview',
  vendors: 'MSME Vendor Compliance',
  risk: 'Risk Analysis',
  actions: 'Actions & Pipeline',
  documents: 'Legal Documents',
  audit: 'Audit Trail',
  rules: 'Rules Engine & Architecture',
}

export default function Dashboard({ data, sessionId, onReset }) {
  const [activeTab, setActiveTab] = useState('overview')
  const [sidebarCollapsed, setSidebarCollapsed] = useState(false)

  const summary = data?.summary
  const vendors = data?.msme_vendors
  const actions = data?.actions
  const impact = data?.impact
  const rulesEvaluated = data?.rules_evaluated
  const pipelineSteps = data?.pipeline_steps

  const handleNavigate = (tab) => setActiveTab(tab)

  return (
    <div className="min-h-screen bg-gray-950">
      <Sidebar
        activeTab={activeTab}
        onTabChange={setActiveTab}
        collapsed={sidebarCollapsed}
        onToggle={() => setSidebarCollapsed(c => !c)}
      />

      {/* Main content area */}
      <div
        className="transition-all duration-200"
        style={{ marginLeft: sidebarCollapsed ? 64 : 240 }}
      >
        {/* Top bar */}
        <div className="sticky top-0 z-30 bg-gray-950/80 backdrop-blur-xl border-b border-gray-800/40 px-6 py-3">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-4">
              {/* Mobile menu toggle */}
              <button
                onClick={() => setSidebarCollapsed(c => !c)}
                className="lg:hidden p-2 rounded-lg bg-gray-800/50 text-gray-400 hover:text-gray-200"
              >
                <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.5}>
                  <path strokeLinecap="round" strokeLinejoin="round" d="M3.75 6.75h16.5M3.75 12h16.5m-16.5 5.25h16.5" />
                </svg>
              </button>
              <div>
                <h2 className="text-lg font-semibold">{TAB_TITLES[activeTab]}</h2>
                <p className="text-xs text-gray-500 font-mono">
                  Session: {sessionId} | {summary?.total_invoices || 0} invoices | {summary?.total_vendors || 0} vendors
                </p>
              </div>
            </div>
            <div className="flex items-center gap-3">
              <span className="hidden md:block text-[10px] font-mono text-gray-600 bg-gray-800/40 px-2 py-1 rounded-lg">
                MSMED Act 2006 | IT Act 43B(h) | RBI Rates
              </span>
              <button
                onClick={onReset}
                className="px-4 py-2 rounded-xl text-xs font-medium bg-gray-800 hover:bg-gray-700 text-gray-300 transition-colors"
              >
                New Analysis
              </button>
            </div>
          </div>
        </div>

        {/* Tab content */}
        <div className="p-6 max-w-[1400px] mx-auto">
          <AnimatePresence mode="wait">
            <motion.div
              key={activeTab}
              initial={{ opacity: 0, y: 8 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -8 }}
              transition={{ duration: 0.2 }}
            >
              {activeTab === 'overview' && (
                <OverviewTab
                  summary={summary}
                  vendors={vendors}
                  actions={actions}
                  impact={impact}
                  onNavigate={handleNavigate}
                />
              )}

              {activeTab === 'vendors' && (
                <div className="space-y-6">
                  <VendorTable vendors={vendors} visible={true} />
                </div>
              )}

              {activeTab === 'risk' && (
                <div className="space-y-6">
                  <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
                    <div className="lg:col-span-2">
                      <RiskHeatmap vendors={vendors} visible={true} />
                    </div>
                    <div>
                      <ComplianceScore score={summary?.compliance_score} visible={true} />
                    </div>
                  </div>
                  <LawCard
                    visible={true}
                    appliedRules={summary ? ['MSMED-001', 'MSMED-002', 'MSMED-003', 'MSMED-004', 'IT43BH-001', 'IT43BH-002'] : []}
                  />
                </div>
              )}

              {activeTab === 'actions' && (
                <div className="space-y-6">
                  <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
                    <div className="lg:col-span-2">
                      <ActionPanel
                        actions={actions}
                        pipelineSteps={pipelineSteps}
                        visible={true}
                        scoreBefore={summary?.compliance_score}
                        scoreAfter={summary?.compliance_score ? Math.min(100, summary.compliance_score + 40) : null}
                      />
                    </div>
                    <div>
                      <ComplianceScore score={summary?.compliance_score} visible={true} />
                    </div>
                  </div>
                  <PaymentTimeline actions={actions} visible={true} />
                  <ImpactSummary impact={impact} visible={true} />
                </div>
              )}

              {activeTab === 'documents' && (
                <DocumentViewer sessionId={sessionId} visible={true} />
              )}

              {activeTab === 'audit' && (
                <AuditTrail sessionId={sessionId} visible={true} />
              )}

              {activeTab === 'rules' && (
                <div className="space-y-6">
                  <RuleVisualizer rulesEvaluated={rulesEvaluated} visible={true} />
                  <ArchitectureShowcase visible={true} />
                  <MarketValidation visible={true} />
                </div>
              )}
            </motion.div>
          </AnimatePresence>
        </div>
      </div>
    </div>
  )
}
