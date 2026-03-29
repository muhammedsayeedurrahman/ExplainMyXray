# LAWTRIX

**Autonomous Compliance Execution Engine for Indian MSME Payment Laws**

LAWTRIX automatically detects MSME vendor payment violations, calculates compound interest liabilities, assesses tax disallowance under Section 43B(h), and generates prioritized remediation actions — all without human prompts.

## Problem

Every Indian company buying from MSME suppliers must comply with three overlapping laws:
- **MSMED Act 2006** — Pay within 45 days or face compound interest at 3x RBI rate (19.5% p.a.)
- **IT Act Section 43B(h)** — Overdue payments are disallowed as tax deductions (effective AY 2024-25)
- **MSME-1 Filing** — Semi-annual reporting of overdue MSME payments to MCA

No existing open-source tool handles all three autonomously. The #1 Indian compliance plugin ([india-compliance, Issue #3086](https://github.com/resilient-tech/india-compliance/issues/3086)) has an open feature request for exactly this.

## Solution

Upload an AP ledger CSV → get full compliance analysis in <2 seconds:
- 23 MSME vendors detected from 52 total
- 313 overdue invoices flagged
- INR 39.2L overdue, INR 15L interest liability
- 89 prioritized remediation actions
- Compliance score: 46 → 86

## Quick Start

### Prerequisites
- Python 3.10+
- Node.js 18+

### Backend
```bash
cd dhara/backend
pip install -r requirements.txt
uvicorn main:app --host 0.0.0.0 --port 8001 --reload
```

### Frontend
```bash
cd dhara/frontend
npm install
npm run dev
```

Open http://localhost:5173

### Demo Mode
The app includes a built-in demo with 847 invoices from a manufacturing company's AP ledger. Click **"Run Demo"** or hit the `/demo/run` endpoint.

## Architecture

See [ARCHITECTURE.md](ARCHITECTURE.md) for the full architecture document.

**6-Step Autonomous Pipeline:**

1. **Scanner** — Detect MSME vendors from invoice data
2. **Clock** — Calculate statutory due dates and delay days
3. **Interest Engine** — Compound interest per MSMED Act Section 16
4. **Tax Analyzer** — Section 43B(h) disallowance calculation
5. **Dual Rules Engine** — gorules/zen (Rust) + Python evaluator
6. **Action Planner** — Prioritized remediation with deadlines

**Document Generators:** MSME-1 draft, scrutiny defense brief, payment schedule

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Backend | Python, FastAPI, Uvicorn |
| Rules Engine | gorules/zen (Rust, sub-ms) + Python fallback |
| Frontend | React, Vite, Tailwind CSS, Framer Motion, Recharts |
| Law Encoding | JSON Decision Model (JDM), custom JSON rulesets |
| Deployment | Vercel Serverless (backend + frontend) |

## API Endpoints

```bash
GET  /health                          # Health check
POST /demo/run                        # Run demo analysis
GET  /analysis/{session_id}           # Full compliance results
GET  /vendors/{session_id}            # Vendor breakdown
GET  /actions/{session_id}            # Prioritized action plan
GET  /compliance-score/{session_id}   # Compliance score details
GET  /audit-log/{session_id}          # Full audit trail
GET  /documents/{session_id}/msme1    # MSME-1 draft
GET  /documents/{session_id}/defense  # Scrutiny defense brief
GET  /documents/{session_id}/schedule # Payment schedule
GET  /impact/{session_id}             # Financial impact summary
```

## Impact

For a mid-size manufacturer with 50 MSME vendors:

| Metric | Annual Impact |
|--------|--------------|
| Cost avoided | INR 40-78 Lakhs |
| Time saved | 520+ hours/year |
| Tax deductions preserved | INR 10-15 Lakhs |
| Payback period | < 1 month |

See [IMPACT_MODEL.md](IMPACT_MODEL.md) for detailed calculations and assumptions.

## Submission Documents

- [ARCHITECTURE.md](ARCHITECTURE.md) — System architecture, agent roles, data flow
- [IMPACT_MODEL.md](IMPACT_MODEL.md) — Quantified business impact with assumptions
- [VIDEO_SCRIPT.md](VIDEO_SCRIPT.md) — 3-minute pitch video script

## Laws Encoded

| Ruleset | Statute | Key Sections |
|---------|---------|-------------|
| `msmed_act.json` | MSMED Act 2006 | Sections 2, 9, 15, 16 |
| `it_act_43bh.json` | Income Tax Act 1961 | Section 43B(h) |
| `msme1_filing.json` | Companies Act 2013 | Section 405 |

## License

MIT
