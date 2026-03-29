# LAWTRIX - 3-Minute Pitch Video Script

## [0:00 - 0:25] THE PROBLEM

**[Screen: Dark background, text fades in]**

> *"Every company in India that buys from an MSME supplier is sitting on a legal time bomb."*

**[Screen: Three law references appear one by one]**

1. **MSMED Act 2006** — Pay MSME vendors within 45 days or face compound interest at 3x the RBI rate
2. **Section 43B(h)** — Miss the deadline? Your entire expense is disallowed as a tax deduction
3. **MSME-1 Filing** — Companies must report overdue MSME payments to the government every 6 months

**[Screen: Stats appear]**

The #1 Indian compliance tool — resilient-tech/india-compliance with 200+ stars — has an **open feature request** for exactly this. ERPNext has one too. Nobody has built it.

> *"63 million MSMEs. Three overlapping laws. Zero autonomous tools. Until now."*

---

## [0:25 - 0:45] THE SOLUTION

**[Screen: LAWTRIX logo animation — fullscreen gradient]**

> *"LAWTRIX — the Autonomous Compliance Execution Engine."*

**[Screen: Key differentiators appear]**

- **Not a chatbot.** No prompts. No LLM calls. Upload your AP ledger, get full compliance analysis in under 2 seconds.
- **Law as Code.** Indian statutes encoded as executable JSON rules with legal section references. Every decision is traceable.
- **Dual Rules Engine.** gorules/zen (Rust, sub-millisecond) for speed. Python evaluator for authority. Both agree or we flag it.
- **Full autonomy.** 6-step pipeline runs end-to-end: scan vendors → calculate delays → compute interest → assess tax impact → score risks → generate action plan.

---

## [0:45 - 1:45] THE DEMO

**[Screen: Upload animation]**

> *"Let's run it. 847 invoices from a manufacturing company's AP ledger. 52 vendors. Go."*

### Detection Dashboard [0:50 - 1:05]

**[Screen: KPI cards animate in with CountUp]**

- 23 MSME vendors detected
- 313 overdue invoices flagged
- INR 39.2 Lakhs in overdue payments
- INR 15 Lakhs in interest liability
- Compliance score: **46 out of 100**

> *"In 1.2 seconds, LAWTRIX found what would take a CA firm two weeks."*

### Legal Engine [1:05 - 1:20]

**[Screen: Risk heatmap + rule visualization]**

- Color-coded vendor risk grid — red (critical) to green (compliant)
- Each rule traces back to its exact legal section
- Section 16 interest calculated at 19.5% p.a., compounded monthly
- Section 43B(h) disallowance: INR 9.8L additional tax if unpaid before ITR deadline

> *"Every number has a law behind it. Every calculation is auditable."*

### Autonomous Actions [1:20 - 1:45]

**[Screen: Action panel with 89 prioritized actions]**

- **Critical:** Immediate payment to 5 high-risk vendors (INR 12L, saves INR 4.3L in penalties)
- **High:** Pay before ITR deadline to preserve INR 9.8L in tax deductions
- **Medium:** File MSME-1 returns for 17 vendors, provision interest liabilities

**[Screen: Compliance score transformation — 46 → 86]**

> *"89 actions. Prioritized by urgency. Each with a deadline, a financial impact, and a legal reference. Execute them, and your compliance score jumps from 46 to 86."*

---

## [1:45 - 2:15] THE ARCHITECTURE

**[Screen: Architecture diagram]**

> *"Under the hood: a 6-step pipeline."*

1. **Scanner** — detects MSME vendors from AP data (cross-references Udyam categories)
2. **Clock** — calculates statutory deadlines: 15 days without agreement, 45 days with
3. **Interest Engine** — compound interest per Section 16, period-aware RBI rate lookup
4. **Tax Analyzer** — Section 43B(h) disallowance at 25.17% effective corporate rate
5. **Dual Rules Engine** — Rust-powered zen-engine + Python evaluator, 3 decision tables, 12+ rules
6. **Action Planner** — generates prioritized remediation with deadlines and financial impact

**[Screen: Document previews]**

Plus three document generators: **MSME-1 draft** for MCA filing, **scrutiny defense brief** for tax audits, **payment schedule** with vendor deadlines.

> *"Everything logged. Append-only audit trail. Every decision traceable to a legal section."*

---

## [2:15 - 2:45] THE IMPACT

**[Screen: Impact numbers animate in]**

For a mid-size manufacturer with 50 MSME vendors:

| Metric | Annual Impact |
|--------|--------------|
| Interest liability avoided | INR 5-8 Lakhs |
| Tax deductions preserved | INR 10-15 Lakhs |
| Penalties avoided | INR 3.5-8 Lakhs |
| Time saved | 520+ hours/year |
| **Total financial impact** | **INR 40-78 Lakhs/year** |

> *"Payback period? Under one month. ROI? Over 1,500%."*

---

## [2:45 - 3:00] THE CLOSE

**[Screen: LAWTRIX logo + tagline]**

> *"The #1 Indian compliance platform has an open feature request for what we just demoed. Tally can't do it. Zoho can't do it. ClearTax does part of it, manually."*

> *"LAWTRIX is the only tool that autonomously does all four: MSME detection, 43B(h) tax impact, MSME-1 filing, and action execution."*

> *"63 million MSMEs. Three overlapping laws. One engine."*

**[Screen: LAWTRIX — Autonomous Compliance Execution Engine]**

> *"LAWTRIX."*

**[END]**
