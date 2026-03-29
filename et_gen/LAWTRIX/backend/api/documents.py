"""Documents endpoint - GET /documents/{session_id}/{type}."""
from __future__ import annotations

from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse

from storage.session_store import get_session

router = APIRouter()

DOCUMENT_TYPES = {"msme1", "defense", "schedule"}


@router.get("/documents/{session_id}/{doc_type}")
async def get_document(session_id: str, doc_type: str):
    """Generate/retrieve compliance document."""
    session = get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail=f"Session {session_id} not found")

    if doc_type not in DOCUMENT_TYPES:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid document type. Choose from: {', '.join(DOCUMENT_TYPES)}",
        )

    result = session.compliance_result
    if not result:
        raise HTTPException(status_code=400, detail="Analysis not yet complete")

    metrics = result.metrics

    if doc_type == "msme1":
        return _generate_msme1_preview(session_id, session)
    elif doc_type == "defense":
        return _generate_defense_preview(session_id, session)
    elif doc_type == "schedule":
        return _generate_schedule_preview(session_id, session)


def _generate_msme1_preview(session_id, session):
    """Generate MSME-1 filing draft preview."""
    msme_vendors = [vc for vc in session.vendor_results if vc.vendor.is_msme and vc.overdue_invoices > 0]

    return {
        "document_type": "msme1",
        "title": "MSME-1 Half-Yearly Return (Draft)",
        "session_id": session_id,
        "filing_period": "October 2025 - March 2026",
        "company_details": {
            "note": "Fill in company CIN, name, and address",
        },
        "summary": {
            "total_msme_vendors_with_overdue": len(msme_vendors),
            "total_overdue_amount": str(sum(vc.overdue_amount for vc in msme_vendors)),
        },
        "vendor_details": [
            {
                "vendor_name": vc.vendor.vendor_name,
                "msme_registration": vc.vendor.msme_registration_number or "To be filled",
                "outstanding_amount": str(vc.overdue_amount),
                "overdue_days": vc.max_delay_days,
            }
            for vc in msme_vendors
        ],
        "disclaimer": "This is an auto-generated draft. Verify all details before filing.",
    }


def _generate_defense_preview(session_id, session):
    """Generate scrutiny defense document preview."""
    metrics = session.compliance_result.metrics

    return {
        "document_type": "defense",
        "title": "Scrutiny Defense - MSME Compliance",
        "session_id": session_id,
        "prepared_for": "Income Tax Assessment",
        "sections": [
            {
                "heading": "1. Compliance Overview",
                "content": f"Analysis of {metrics.total_invoices} invoices across {metrics.msme_vendors} MSME vendors. Compliance score: {metrics.compliance_score}/100.",
            },
            {
                "heading": "2. Section 43B(h) Disclosure",
                "content": f"Total potential disallowance: INR {metrics.total_tax_disallowance:,.2f}. Actions taken to remediate overdue payments.",
            },
            {
                "heading": "3. Interest Provisions",
                "content": f"Interest liability of INR {metrics.total_interest_liability:,.2f} has been identified and provisioned per MSMED Act Section 16.",
            },
            {
                "heading": "4. Remediation Actions",
                "content": f"Action plan generated with {len(session.action_plan.actions) if session.action_plan else 0} compliance actions.",
            },
        ],
        "disclaimer": "This is an auto-generated draft for review by legal counsel.",
    }


def _generate_schedule_preview(session_id, session):
    """Generate payment priority schedule preview."""
    if not session.action_plan:
        return {"error": "No action plan available"}

    payment_actions = [
        a for a in session.action_plan.actions
        if a.action_type.value in ("immediate_payment", "priority_payment")
    ]

    return {
        "document_type": "schedule",
        "title": "Payment Priority Schedule",
        "session_id": session_id,
        "total_payments": len(payment_actions),
        "schedule": [
            {
                "priority": a.priority.value,
                "vendor": a.vendor_name,
                "amount": str(a.financial_impact),
                "deadline": a.deadline.isoformat() if a.deadline else "ASAP",
                "reason": a.law_reference,
            }
            for a in payment_actions
        ],
    }
