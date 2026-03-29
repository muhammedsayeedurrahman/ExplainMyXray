# LAWTRIX Demo Script

## Quick Start

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

## Demo Flow (7 Steps, ~25 seconds)

### Step 1: Cinematic Intro (0-2.5s)
- "LAWTRIX" appears fullscreen with gradient animation
- Tagline: "Autonomous Compliance Execution Engine"
- Law references fade in: MSMED Act 2006, IT Act 43B(h), RBI Interest Rules

### Step 2: Data Upload (2.5-5s)
- Animated processing indicator
- "Processing 847+ invoices from 52 vendors..."

### Step 3: Detection Dashboard (5-8s)
- 5 KPI cards animate in with CountUp:
  - 23 MSME Vendors
  - 313 Overdue Invoices
  - INR 39.2L Overdue Amount
  - INR 15L Interest Liability
  - 46/100 Compliance Score
- Law reference cards expand showing which rules were applied
- Vendor table populates sorted by risk score

### Step 4: Legal Engine (8-12s)
- Risk heatmap shows color-coded vendor grid
- 6-step pipeline animation plays
- Compliance score gauge animates

### Step 5: Autonomous Actions (12-18s)
- Action panel shows 89 prioritized remediation actions
- Compliance score transformation: 46 → 86
- Payment timeline shows deadlines

### Step 6: Impact Reveal (18-22s)
- Financial impact cards with CountUp animation
- Total financial impact number
- Compliance improvement, risk reduction, hours saved

### Step 7: Demo Complete (22-25s)
- Green success banner
- Audit trail visible
- Document previews (MSME-1, Scrutiny Defense, Payment Schedule)
- Monetization teasers for premium features

## API Testing

```bash
# Health check
curl http://localhost:8001/health

# Run demo
curl http://localhost:8001/demo/run | python -m json.tool

# Test with session ID from demo response
curl http://localhost:8001/analysis/{session_id}
curl http://localhost:8001/vendors/{session_id}
curl http://localhost:8001/actions/{session_id}
curl http://localhost:8001/compliance-score/{session_id}
curl http://localhost:8001/audit-log/{session_id}
curl http://localhost:8001/documents/{session_id}/msme1
curl http://localhost:8001/documents/{session_id}/defense
curl http://localhost:8001/documents/{session_id}/schedule
curl http://localhost:8001/impact/{session_id}
```

## Key Talking Points

1. **Not a chatbot** - Fully autonomous. No prompts needed. Upload data → get results.
2. **Law as Code** - Indian statutes encoded as evaluable JSON rules with legal references.
3. **Full audit trail** - Every decision logged with legal reasoning. Defensible in scrutiny.
4. **Real calculations** - Interest at 3x RBI rate (19.5% p.a.) compounded monthly per MSMED Act.
5. **Tax impact** - Section 43B(h) disallowance calculated with actual tax rates.
6. **Actionable** - Not just detection. Generates payment priorities, filings, defense documents.
