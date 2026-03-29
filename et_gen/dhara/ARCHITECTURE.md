# LAWTRIX - Architecture Document

## System Overview

LAWTRIX is an **Autonomous Compliance Execution Engine** that ingests accounts payable data, detects MSME vendor payment violations under Indian law, calculates financial liabilities, and generates prioritized remediation actions — all without human prompts.

```
┌─────────────────────────────────────────────────────────────────────┐
│                        LAWTRIX Architecture                        │
│                                                                    │
│  ┌──────────┐     ┌──────────────┐     ┌─────────────────────┐    │
│  │  React +  │────▶│  FastAPI      │────▶│  6-Step Pipeline    │    │
│  │  Tailwind │◀────│  REST API     │◀────│  Orchestrator       │    │
│  │  Frontend │     │  (Backend)    │     │  (Engine Core)      │    │
│  └──────────┘     └──────────────┘     └─────────────────────┘    │
│       ▲                  │                       │                 │
│       │                  ▼                       ▼                 │
│  Cinematic         Session Store          ┌─────────────┐         │
│  Demo Mode         + Audit Store          │ Dual Rules  │         │
│  (auto-play)       (In-Memory)            │ Engine      │         │
│                                           │ ┌─────────┐ │         │
│                                           │ │zen (Rust)│ │         │
│                                           │ │+ Python  │ │         │
│                                           │ └─────────┘ │         │
│                                           └─────────────┘         │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Agent Roles (Pipeline Steps)

LAWTRIX operates as a **6-step autonomous pipeline**. Each step is a specialized agent within the orchestrator:

| Step | Agent | Responsibility | Input | Output |
|------|-------|---------------|-------|--------|
| **1** | **Scanner** | Detect MSME vendors from invoice data | Raw invoice batch | Vendor registry with MSME classification (Micro/Small/Medium) |
| **2** | **Clock** | Calculate statutory due dates and delay days | Vendor invoices + agreed terms | Delay days, effective due dates, overdue flags |
| **3** | **Interest Calculator** | Compute compound interest per MSMED Act §16 | Outstanding amounts + delay days | Interest liability at 3× RBI bank rate (19.5% p.a.), compounded monthly |
| **4** | **Tax Impact Analyzer** | Calculate §43B(h) disallowance | Overdue amounts to MSME vendors | Tax deduction disallowed + additional tax liability at 25.17% effective rate |
| **5** | **Risk Scorer + Rules Engine** | Evaluate compliance rules, score vendor risk | Vendor data + all computed liabilities | Risk score (0-100), risk level, rule violations |
| **6** | **Action Planner** | Generate prioritized remediation plan | All vendor compliance results | Prioritized actions with deadlines, financial impact, projected score improvement |

### Document Generators (Post-Pipeline)

| Generator | Output | Legal Basis |
|-----------|--------|-------------|
| **MSME-1 Draft** | Half-yearly filing for MCA portal | Companies Act 2013, §405 |
| **Scrutiny Defense** | Pre-built defense brief for tax scrutiny | IT Act 1961, §43B(h) |
| **Payment Schedule** | Prioritized payment timeline with deadlines | MSMED Act 2006, §15 |

---

## Communication Flow

```
User uploads CSV/Excel
        │
        ▼
┌─────────────────┐
│  Parsers Layer   │  CSV Parser → Normalizer → InvoiceBatch
│  (csv, excel)    │  Handles date formats, column mapping,
│                  │  MSME flag detection
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  API Layer       │  FastAPI routes: /upload, /demo/run
│  (REST)          │  Creates session, triggers pipeline
└────────┬────────┘
         │
         ▼
┌─────────────────────────────────────────────┐
│  Orchestrator (engine/orchestrator.py)       │
│                                             │
│  Step 1: scanner.detect_msme_vendors()      │
│       │                                     │
│  Step 2: clock.calculate_delay_days()       │
│       │                                     │
│  Step 3: interest.calculate_interest()      │
│       │                                     │
│  Step 4: tax_impact.calculate_disallowance()│
│       │                                     │
│  Step 5: rules + risk_scorer (dual engine)  │
│       │   ├─ zen_evaluator (Rust, sub-ms)   │
│       │   └─ evaluator (Python, authority)  │
│       │                                     │
│  Step 6: action_planner.generate_plan()     │
│                                             │
│  Output: AnalysisSession + AuditEvents      │
└────────┬────────────────────────────────────┘
         │
         ▼
┌─────────────────┐     ┌──────────────────┐
│  Session Store   │     │  Audit Store     │
│  (in-memory)     │     │  (append-only)   │
│  Sessions by ID  │     │  Every decision  │
└────────┬────────┘     │  logged with law │
         │              │  references      │
         ▼              └──────────────────┘
┌─────────────────┐
│  API Response    │  /analysis, /vendors, /actions,
│  Endpoints       │  /compliance-score, /audit-log,
│                  │  /documents, /impact
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  React Frontend  │  Dashboard, Vendor Table, Risk Heatmap,
│  (Vite + TW)     │  Action Panel, Impact Summary, Audit Trail
└─────────────────┘
```

---

## Tool Integrations

| Tool | Purpose | Integration Point |
|------|---------|-------------------|
| **gorules/zen** (Rust) | Sub-millisecond rule evaluation via JSON Decision Model | `rules/zen_evaluator.py` — loads JDM, evaluates decision tables |
| **Python Rules Engine** | Authoritative rule evaluation with full condition DSL | `rules/evaluator.py` — 13 operators (equals, between, in, etc.) |
| **RBI Rate Config** | Period-aware interest rate lookup (OpenFisca pattern) | `rules/rbi_rates.json` — rates by effective date |
| **FastAPI** | REST API with auto-generated OpenAPI docs | `main.py` — 10 route modules |
| **React + Vite** | Cinematic dashboard with Framer Motion animations | `frontend/` — 20+ components |
| **Tailwind CSS** | Utility-first styling with responsive design | `tailwind.config.js` |

### Law Encodings (Rules as Code)

| JSON Ruleset | Statute | Rules Encoded |
|---|---|---|
| `msmed_act.json` | MSMED Act 2006 | Payment deadlines (§15), interest (§16), MSME categories (§2) |
| `it_act_43bh.json` | IT Act 1961 | Tax deduction disallowance for overdue MSME payments |
| `msme1_filing.json` | Companies Act 2013 | Half-yearly filing requirements (§405) |
| `msmed_compliance.json` (JDM) | All three acts | Unified decision table for zen-engine |

---

## Error Handling Logic

| Layer | Strategy | Implementation |
|-------|----------|---------------|
| **Parser** | Graceful degradation — skip unparseable rows, normalize dates across formats (DD-MM-YYYY, YYYY-MM-DD) | `parsers/normalizer.py` |
| **zen-engine** | Fallback pattern — if Rust engine fails or isn't installed, Python evaluator takes over seamlessly | `zen_evaluator.py:_evaluate_zen()` catches exceptions → `_evaluate_fallback()` |
| **API** | HTTP exceptions with session validation — 404 for missing sessions, structured error responses | Every API route validates `get_session(session_id)` |
| **Pipeline** | Per-vendor isolation — one vendor's error doesn't crash the pipeline for others | `orchestrator.py` processes vendors in a loop with per-vendor audit events |
| **Interest Calc** | Guard clauses — zero principal or zero delay returns `Decimal("0.00")` immediately | `interest.py:calculate_interest()` |
| **Audit Trail** | Append-only event store — every decision logged with event type, step number, law reference, and timestamp | `storage/audit_store.py` — immutable audit log |

---

## Data Models

```
InvoiceBatch ──┐
               ├──▶ Vendor (id, name, is_msme, category)
               │
               ├──▶ VendorCompliance (risk_score, risk_level,
               │        interest, tax_disallowance, violations)
               │
               ├──▶ ComplianceResult (metrics, violations, rules_applied)
               │
               ├──▶ ActionPlan (prioritized actions, savings, score delta)
               │
               └──▶ AuditEvent[] (append-only decision log)
                        │
                        ▼
               AnalysisSession (complete result envelope)
```

All models are **immutable dataclasses** — no mutation after creation. Sessions are stored by ID for retrieval by any API endpoint.

---

## Deployment

- **Backend**: FastAPI on Vercel Serverless (Python runtime) or `uvicorn` locally
- **Frontend**: Vite-built static bundle served from `dist/` or Vercel
- **No external database**: In-memory session/audit stores (stateless per deployment)
- **No API keys required**: All computation is local — no LLM calls, no external APIs
