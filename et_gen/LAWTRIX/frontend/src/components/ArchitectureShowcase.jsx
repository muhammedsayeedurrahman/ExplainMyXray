import { motion } from 'framer-motion'

const STACK_LAYERS = [
  {
    label: 'Document Parsing',
    tech: 'OpenNyAI-compatible',
    desc: 'CSV/Excel ingestion with entity extraction',
    color: 'from-blue-600/20 to-blue-800/20',
    border: 'border-blue-500/30',
    badge: 'text-blue-400',
    icon: '\u{1F4C4}',
  },
  {
    label: 'LLM Fact Extraction',
    tech: 'Hybrid Architecture',
    desc: 'LLM for fact extraction + deterministic rules for legal reasoning',
    color: 'from-purple-600/20 to-purple-800/20',
    border: 'border-purple-500/30',
    badge: 'text-purple-400',
    icon: '\u{1F9E0}',
  },
  {
    label: 'Rules Engine',
    tech: 'gorules/zen (Rust)',
    desc: 'JSON Decision Model, sub-ms latency, decision tables + graph flow',
    color: 'from-orange-600/20 to-orange-800/20',
    border: 'border-orange-500/30',
    badge: 'text-orange-400',
    icon: '\u2699\uFE0F',
    highlight: true,
  },
  {
    label: 'Period-Aware Rates',
    tech: 'OpenFisca-inspired',
    desc: 'Historical RBI rate lookup based on invoice date',
    color: 'from-cyan-600/20 to-cyan-800/20',
    border: 'border-cyan-500/30',
    badge: 'text-cyan-400',
    icon: '\u{1F4C8}',
  },
  {
    label: 'Compliance Pipeline',
    tech: 'FastAPI + Pydantic',
    desc: '6-step autonomous pipeline: Scan > Delay > Interest > Tax > Risk > Actions',
    color: 'from-green-600/20 to-green-800/20',
    border: 'border-green-500/30',
    badge: 'text-green-400',
    icon: '\u{1F680}',
  },
  {
    label: 'Audit Trail',
    tech: 'Event Sourcing',
    desc: 'Immutable audit log with legal references for every decision',
    color: 'from-yellow-600/20 to-yellow-800/20',
    border: 'border-yellow-500/30',
    badge: 'text-yellow-400',
    icon: '\u{1F4DC}',
  },
]

const TECH_BADGES = [
  { name: 'Python', color: 'bg-blue-500/20 text-blue-400 border-blue-500/30' },
  { name: 'FastAPI', color: 'bg-green-500/20 text-green-400 border-green-500/30' },
  { name: 'Rust (zen)', color: 'bg-orange-500/20 text-orange-400 border-orange-500/30' },
  { name: 'Pydantic', color: 'bg-purple-500/20 text-purple-400 border-purple-500/30' },
  { name: 'React 18', color: 'bg-cyan-500/20 text-cyan-400 border-cyan-500/30' },
  { name: 'Framer Motion', color: 'bg-pink-500/20 text-pink-400 border-pink-500/30' },
  { name: 'Recharts', color: 'bg-yellow-500/20 text-yellow-400 border-yellow-500/30' },
  { name: 'Tailwind CSS', color: 'bg-teal-500/20 text-teal-400 border-teal-500/30' },
]

export default function ArchitectureShowcase({ visible }) {
  if (!visible) return null

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      className="glass-card p-6 space-y-6"
    >
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-lg font-semibold">Architecture</h2>
          <p className="text-xs text-gray-500 mt-1">Open-source compliance engine stack</p>
        </div>
        <div className="px-3 py-1.5 rounded-full bg-green-500/10 border border-green-500/20">
          <span className="text-xs text-green-400 font-semibold">100% Open Source</span>
        </div>
      </div>

      {/* Architecture diagram - layered stack */}
      <div className="space-y-3">
        {STACK_LAYERS.map((layer, i) => (
          <motion.div
            key={layer.label}
            initial={{ opacity: 0, x: -30 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ delay: i * 0.1, type: 'spring', stiffness: 150 }}
            className={`bg-gradient-to-r ${layer.color} border ${layer.border} rounded-xl p-4 flex items-center gap-4 ${
              layer.highlight ? 'ring-1 ring-orange-500/30' : ''
            }`}
          >
            <div className="text-2xl flex-shrink-0 w-10 text-center">{layer.icon}</div>
            <div className="flex-1 min-w-0">
              <div className="flex items-center gap-2">
                <span className="text-sm font-semibold text-white">{layer.label}</span>
                <span className={`px-1.5 py-0.5 rounded text-[10px] font-mono ${layer.badge} bg-black/20 border border-current/20`}>
                  {layer.tech}
                </span>
              </div>
              <p className="text-xs text-gray-400 mt-1">{layer.desc}</p>
            </div>
            {/* Connection indicator */}
            {i < STACK_LAYERS.length - 1 && (
              <div className="absolute -bottom-2 left-1/2 w-px h-3 bg-gray-700" />
            )}
          </motion.div>
        ))}
      </div>

      {/* Tech badges */}
      <div>
        <h3 className="text-sm font-semibold text-gray-300 mb-3">Tech Stack</h3>
        <div className="flex flex-wrap gap-2">
          {TECH_BADGES.map((badge, i) => (
            <motion.span
              key={badge.name}
              initial={{ opacity: 0, scale: 0 }}
              animate={{ opacity: 1, scale: 1 }}
              transition={{ delay: 0.6 + i * 0.05 }}
              className={`px-3 py-1.5 rounded-full text-xs font-medium border ${badge.color}`}
            >
              {badge.name}
            </motion.span>
          ))}
        </div>
      </div>

      {/* Key differentiators */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        {[
          {
            title: 'Deterministic',
            desc: 'Law-as-Code with full audit trail. No hallucinations, no probabilistic outputs.',
            color: 'text-green-400',
          },
          {
            title: 'Sub-ms Latency',
            desc: 'Rust-powered zen-engine evaluates all 8 compliance rules in under 1 millisecond.',
            color: 'text-orange-400',
          },
          {
            title: 'Autonomous',
            desc: 'Upload data, get compliance report. No prompts, no chatbot, no manual steps.',
            color: 'text-blue-400',
          },
        ].map((diff, i) => (
          <motion.div
            key={diff.title}
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.8 + i * 0.1 }}
            className="bg-gray-800/30 rounded-xl p-4"
          >
            <h4 className={`text-sm font-bold ${diff.color}`}>{diff.title}</h4>
            <p className="text-xs text-gray-400 mt-1">{diff.desc}</p>
          </motion.div>
        ))}
      </div>
    </motion.div>
  )
}
