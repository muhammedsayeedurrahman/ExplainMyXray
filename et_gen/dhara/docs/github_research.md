# LAWTRIX - GitHub Research: Relevant Repos & Prior Art

> Generated 2026-03-29 | Deep research across 3 parallel agents + web searches
> **Key Finding**: No standalone open-source MSME compliance engine exists anywhere on GitHub.

---

## Market Validation

- [resilient-tech/india-compliance Issue #3086](https://github.com/resilient-tech/india-compliance/issues/3086) — Open request for Section 43B(h) + MSME Form-1 (Feb 2025, still unimplemented)
- [frappe/erpnext Issue #42807](https://github.com/frappe/erpnext/issues/42807) — Open request for MSME supplier 45-day due date calculation
- SigzenMSME (Frappe Cloud) — Closest purpose-built MSME tool, but proprietary/commercial
- [MSME SAMADHAAN Portal](https://samadhaan.msme.gov.in/) — Government delayed payment monitoring system (no open API)

**Talking Point for Judges:**
> "The #1 Indian compliance tool (resilient-tech/india-compliance, 200+ stars, used by thousands of businesses) has an open feature request for exactly what LAWTRIX does. We built the autonomous version."

---

## TIER 1 — Directly Relevant to Indian MSME Compliance

| Repo | Stars | Lang | Why It Matters |
|------|-------|------|----------------|
| [resilient-tech/india-compliance](https://github.com/resilient-tech/india-compliance) | ~222 | Python | **Most relevant.** ERPNext-based Indian compliance. Has open Issue #3086 requesting exactly what LAWTRIX does — Section 43B(h) + MSME-1 filing. GST API integration, e-invoicing, GSTIN verification. |
| [frappe/erpnext](https://github.com/frappe/erpnext) | 32.5k | Python | India's #1 open-source ERP. GST-compliant, multi-language. The platform india-compliance extends. |
| [civictech-India/Indian-Law-Penal-Code-Json](https://github.com/civictech-India/Indian-Law-Penal-Code-Json) | 54 | JSON | Indian legal acts in machine-readable JSON + SQLite. IPC, CPC, Marriage Act, Evidence Act, etc. Pattern for encoding MSMED Act as JSON. |
| [SrujanPR/Simplify-Tax](https://github.com/SrujanPR/Simplify-Tax) | ~20 | Python | AI-powered Indian tax filing tool. FastAPI + CrewAI agents. Reads bank statements, classifies income, calculates tax. |

---

## TIER 2 — Law-as-Code Engines (Architecture Patterns)

| Repo | Stars | Lang | What to Learn |
|------|-------|------|---------------|
| [CatalaLang/catala](https://github.com/CatalaLang/catala) | ~1.9k | OCaml | **Gold standard** for law-as-code. Default logic as first-class feature. Compiles to Python. Used to encode French tax law + US IRC Section 121. Pattern for encoding MSMED Act faithfully. |
| [openfisca/openfisca-core](https://github.com/openfisca/openfisca-core) | ~185 | Python | Tax/benefit microsimulation framework. Country-package architecture (create `openfisca-india`). Period-aware rules, vectorial computation, JSON API. |
| [PolicyEngine/policyengine-core](https://github.com/PolicyEngine/policyengine-core) | ~100 | Python | Modernized OpenFisca fork. Production-grade US/UK tax models. Better DX, AI-assisted dev. |

---

## TIER 3 — Rules Engines (Pipeline Components)

| Repo | Stars | Lang | What to Use |
|------|-------|------|-------------|
| [gorules/zen](https://github.com/gorules/zen) | ~1.6k | Rust+Python | **Best fit for LAWTRIX.** JSON Decision Model (JDM), sub-ms latency, Python bindings via `pip install zen-engine`. Visual React editor for rules. Decision tables + graph-based flow. |
| [open-policy-agent/opa](https://github.com/open-policy-agent/opa) | 11.5k | Go | CNCF graduated. Rego policy language. Comprehensive audit trails. Pattern for decoupling policy from application. |
| [CacheControl/json-rules-engine](https://github.com/CacheControl/json-rules-engine) | ~2.9k | JS | JSON-serializable rules with conditions/facts/events. Direct pattern match for our JSON rules engine. |
| [venmo/business-rules](https://github.com/venmo/business-rules) | ~963 | Python | Variable-Operator-Action pattern. Clean Python DSL. Good pattern for rule definition. |
| [jruizgit/rules](https://github.com/jruizgit/rules) | ~1.1k | C+Python | Rete algorithm rules engine. Forward chaining, statecharts, flowcharts. Good for complex interdependent rules. |
| [MAIF/arta](https://github.com/MAIF/arta) | ~6 | Python | YAML-based rules engine built by French insurance co. Clean maintainable rule definitions. |
| [TencentBlueKing/bkflow-dmn](https://github.com/TencentBlueKing/bkflow-dmn) | ~50 | Python | DMN (Decision Model Notation) + FEEL expression language. Standard for decision tables. |
| [lwardzala/business_rules_reasoning](https://github.com/lwardzala/business_rules_reasoning) | ~20 | Python | **LLM + Rules hybrid.** Horn clause reasoning with LLM for fact extraction. Explainable, interruptible, serializable to JSON. |

---

## TIER 4 — Indian Legal NLP & AI

| Repo | Stars | Lang | What to Use |
|------|-------|------|-------------|
| [OpenNyAI/Opennyai](https://github.com/OpenNyAI/Opennyai) | ~75 | Python | NLP pipeline for Indian legal docs. Legal NER (statutes, provisions, parties), rhetorical role labeling, extractive summarization. |
| [Law-AI (IIT Kharagpur)](https://github.com/Law-AI) | Various | Python | InLegalBERT (HuggingFace), automatic statute identification, IL-TUR benchmark (9 Indian languages). |
| [LexPredict/lexpredict-lexnlp](https://github.com/LexPredict/lexpredict-lexnlp) | ~740 | Python | Extract dates, amounts, conditions, regulations from legal text. Multi-language. Pre-trained on SEC data. |
| [Legal-NLP-EkStep/legal_NER](https://github.com/Legal-NLP-EkStep/legal_NER) | ~50 | Python | OpenNyAI mission. Legal NER specifically for Indian court documents. |
| [NisaarAgharia/Indian-LawyerGPT](https://github.com/NisaarAgharia/Indian-LawyerGPT) | ~20 | Python | Fine-tuned Falcon-7B and LLAMA 2 with QLoRA for Indian legal context. |

---

## TIER 5 — Compliance Frameworks & Audit Patterns

| Repo | Stars | Lang | What to Learn |
|------|-------|------|---------------|
| [strongdm/comply](https://github.com/strongdm/comply) | ~1.5k | Go | SOC2 compliance automation. Markdown -> auditor-friendly docs pipeline. Template pattern for generating MSME-1 filings. |
| [cloud-custodian/cloud-custodian](https://github.com/cloud-custodian/cloud-custodian) | ~5.9k | Python | YAML policy DSL: resource type + filters + actions. Directly transferable to legal compliance rules. |
| [aliseylaneh/Python-Eventsourcing-CQRS](https://github.com/aliseylaneh/Python-Eventsourcing-CQRS) | ~50 | Python | Event Sourcing + CQRS with FastAPI + MongoDB. Pattern for immutable audit logs. |
| [pyeventsourcing](https://github.com/pyeventsourcing) | ~1.5k | Python | Python event sourcing library. Append-only event store with aggregate roots. |

---

## TIER 6 — Dashboard & UI Patterns

| Repo | Stars | Lang | What to Learn |
|------|-------|------|---------------|
| [AzzOu3108/Admin-Dashboard](https://github.com/AzzOu3108/Admin-Dashboard) | ~20 | React | React + Tailwind + Framer Motion + Recharts dashboard. Same tech stack as LAWTRIX frontend. |
| [motiondivision/motion](https://github.com/motiondivision/motion) | ~28k | JS | Framer Motion (now "Motion"). Animation library used in LAWTRIX. |
| [driaug/animated-counter](https://github.com/driaug/animated-counter) | ~50 | React | Animated counter component with Framer Motion. Pattern for CountUp animations. |
| [Sushreesatarupa/VISUALPE](https://github.com/Sushreesatarupa/VISUALPE) | ~10 | JS | **Hackathon winner** (Rank 2, INR 2L prize). FinTech analytics using PhonePe Pulse + Account Aggregator. |

---

## TIER 7 — MSME Domain (Not Compliance-Specific)

| Repo | Stars | Lang | Notes |
|------|-------|------|-------|
| [HeyfromNandini/MSME_Sol](https://github.com/HeyfromNandini/MSME_Sol) | ~5 | Kotlin | B2B MSME marketplace with blockchain verification (Aptos). Hackathon project. |
| [nathanyaqueby/m-sig-siemens](https://github.com/nathanyaqueby/m-sig-siemens) | ~5 | Python | MSME sustainability info grabber. NLP for regulation extraction. Siemens hackathon 2023. |
| [dividend-group/msme-registration-app-release](https://github.com/dividend-group/msme-registration-app-release) | ~5 | Android | Mobile app for MSME registration data collection. |
| [diptenduLF/MSME-LMS](https://github.com/diptenduLF/MSME-LMS) | ~5 | Java | MSME Loan Management System. |

---

## Recommended Architecture: How These Map to LAWTRIX Pipeline

| Pipeline Layer | Best Repos | Role in LAWTRIX |
|---|---|---|
| **Law Encoding** | CatalaLang/catala, OpenFisca, civictech-India/Indian-Law-Penal-Code-Json | Encode MSMED Act, IT Act as executable rules with legal references |
| **Document Ingestion** | OpenNyAI/Opennyai, LexPredict/lexnlp | Parse uploaded invoices/legal docs, extract entities, amounts, dates |
| **Rules Engine Core** | gorules/zen (Python bindings), json-rules-engine | Execute compliance rules as JSON decision models |
| **Decision Tables** | TencentBlueKing/bkflow-dmn, pyDMNrules | Handle tax slabs, eligibility thresholds |
| **Policy Orchestrator** | OPA (pattern), cloud-custodian (YAML DSL pattern) | Orchestrate multi-step rule evaluation |
| **LLM + Rules Hybrid** | lwardzala/business_rules_reasoning | LLM for fact extraction, deterministic engine for legal reasoning |
| **Indian Compliance** | resilient-tech/india-compliance | GST API integration, GSTIN verification patterns |
| **Legal AI Models** | Law-AI (IIT Kharagpur), InLegalBERT | Statute identification, legal NER |
| **Audit & Reporting** | strongdm/comply, pyeventsourcing | Auditor-ready docs, append-only event store |

---

## Immediate Upgrade Opportunities for LAWTRIX

### 1. Replace custom JSON rules engine with gorules/zen
- `pip install zen-engine` -- Rust-powered, sub-ms latency
- Visual React editor for rules (judges can see rule definitions live)
- JSON Decision Model format is more expressive than our custom schema

### 2. Adopt OpenFisca's period-aware architecture
- Laws change over time (RBI rate changes, 43B(h) effective from AY 2024-25)
- OpenFisca's Variable + Formula + Period pattern handles this natively

### 3. Use OpenNyAI for document parsing
- Extract statute references, amounts, dates from uploaded legal docs
- Legal NER trained on Indian legal corpus

### 4. Reference india-compliance Issue #3086 in demo
- Shows the exact feature gap in the market
- Validates LAWTRIX's unique value proposition for hackathon judges

### 5. LLM + Rules hybrid from business_rules_reasoning
- LLM extracts facts from uploaded invoices/documents
- Deterministic rules engine evaluates compliance (explainable, auditable)
- Best of both worlds for a hackathon demo

---

## Key Competitor Landscape

| Tool | Type | MSME Compliance? | 43B(h)? | MSME-1? | Autonomous? |
|------|------|-------------------|---------|---------|-------------|
| resilient-tech/india-compliance | ERPNext plugin | Requested (Issue #3086) | No | No | No |
| frappe/erpnext | Full ERP | Requested (Issue #42807) | No | No | No |
| SigzenMSME | Frappe Cloud app | Yes (basic) | Unknown | Unknown | No |
| Tally Prime | Commercial | Partial | Partial | No | No |
| Zoho Books | Commercial | Partial | Partial | No | No |
| ClearTax | Commercial SaaS | Partial | Yes | No | No |
| **LAWTRIX** | **Standalone engine** | **Yes (full)** | **Yes** | **Yes** | **Yes** |

**LAWTRIX is the only tool that autonomously does all four: MSME detection + 43B(h) tax impact + MSME-1 filing + action execution.**

---

## External References

- [MSME SAMADHAAN Portal](https://samadhaan.msme.gov.in/) -- Government delayed payment monitoring
- [ClearTax - Section 43B(h)](https://cleartax.in/s/section-43bh-of-income-tax-act) -- Legal reference
- [MSMED Act 2006 Full Text](https://samadhaan.msme.gov.in/WriteReadData/DocumentFile/MSMED2006act.pdf)
- [Zoho Books - MSME 45-day rule](https://www.zoho.com/in/books/academy/taxes-and-compliance/msme-45-days-payment-rule.html)
- [Nected - Top Open Source Rule Engines 2025](https://www.nected.ai/us/blog-us/open-source-rules-engine)
- [Nected - Python Rule Engines](https://www.nected.ai/blog/python-rule-engines-automate-and-enforce-with-python)
- [GoRules Python Docs](https://gorules.io/open-source/python-rules-engine)
- [OpenFisca Documentation](https://openfisca.org/doc/)
- [Catala Language](https://catala-lang.org/)
