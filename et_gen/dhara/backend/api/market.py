"""Market validation API - GET /market/*."""
from __future__ import annotations

from fastapi import APIRouter

router = APIRouter()

COMPETITORS = [
    {
        "tool": "resilient-tech/india-compliance",
        "type": "ERPNext plugin",
        "msme_compliance": "Requested (Issue #3086)",
        "section_43bh": False,
        "msme1_filing": False,
        "autonomous": False,
        "stars": "222+",
        "note": "Most popular Indian compliance tool. Open feature request validates LAWTRIX.",
    },
    {
        "tool": "frappe/erpnext",
        "type": "Full ERP",
        "msme_compliance": "Requested (Issue #42807)",
        "section_43bh": False,
        "msme1_filing": False,
        "autonomous": False,
        "stars": "32.5k",
        "note": "India's #1 open-source ERP. No MSME-specific compliance module.",
    },
    {
        "tool": "SigzenMSME",
        "type": "Frappe Cloud app",
        "msme_compliance": "Basic",
        "section_43bh": False,
        "msme1_filing": False,
        "autonomous": False,
        "stars": "N/A",
        "note": "Proprietary, limited compliance automation.",
    },
    {
        "tool": "Tally Prime",
        "type": "Commercial",
        "msme_compliance": "Partial",
        "section_43bh": "Partial",
        "msme1_filing": False,
        "autonomous": False,
        "stars": "N/A",
        "note": "Desktop-only, manual compliance checks.",
    },
    {
        "tool": "Zoho Books",
        "type": "Commercial",
        "msme_compliance": "Partial",
        "section_43bh": "Partial",
        "msme1_filing": False,
        "autonomous": False,
        "stars": "N/A",
        "note": "Cloud accounting, limited MSME-specific features.",
    },
    {
        "tool": "ClearTax",
        "type": "Commercial SaaS",
        "msme_compliance": "Partial",
        "section_43bh": True,
        "msme1_filing": False,
        "autonomous": False,
        "stars": "N/A",
        "note": "Tax filing focused. No autonomous compliance execution.",
    },
    {
        "tool": "LAWTRIX",
        "type": "Standalone engine",
        "msme_compliance": True,
        "section_43bh": True,
        "msme1_filing": True,
        "autonomous": True,
        "stars": "Open Source",
        "note": "Only tool that autonomously handles all four: MSME detection + 43B(h) + MSME-1 + action execution.",
    },
]


@router.get("/market/validation")
async def market_validation():
    """Return competitor comparison table data."""
    return {
        "competitors": COMPETITORS,
        "unique_value": (
            "LAWTRIX is the only tool that autonomously handles all four: "
            "MSME detection + 43B(h) tax impact + MSME-1 filing + action execution."
        ),
        "market_gap": {
            "issue_url": "https://github.com/resilient-tech/india-compliance/issues/3086",
            "description": (
                "The #1 Indian compliance tool (resilient-tech/india-compliance, "
                "222+ stars) has an open feature request for exactly what LAWTRIX does."
            ),
        },
    }


@router.get("/market/feature-gap")
async def feature_gap():
    """Return india-compliance Issue #3086 reference and analysis."""
    return {
        "source": "resilient-tech/india-compliance",
        "issue_number": 3086,
        "issue_url": "https://github.com/resilient-tech/india-compliance/issues/3086",
        "title": "Section 43B(h) + MSME Form-1 Support",
        "status": "Open (unimplemented as of March 2026)",
        "requested_features": [
            "Track MSME vendor payment timelines",
            "Calculate Section 43B(h) tax disallowance",
            "Generate MSME-1 half-yearly filing",
            "Flag overdue payments automatically",
        ],
        "lawtrix_coverage": {
            "msme_vendor_tracking": True,
            "section_43bh_calculation": True,
            "msme1_filing_generation": True,
            "overdue_payment_flagging": True,
            "autonomous_action_planning": True,
            "interest_calculation": True,
            "risk_scoring": True,
        },
        "talking_point": (
            "We built the autonomous version of what the #1 Indian compliance "
            "tool's community has been requesting since February 2025."
        ),
    }
