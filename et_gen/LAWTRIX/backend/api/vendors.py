"""Vendors endpoint - GET /vendors/{session_id}."""
from __future__ import annotations

from fastapi import APIRouter, HTTPException

from storage.session_store import get_session

router = APIRouter()


@router.get("/vendors/{session_id}")
async def get_vendors(session_id: str):
    """Get vendor-level compliance breakdown."""
    session = get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail=f"Session {session_id} not found")

    vendors = []
    for vc in session.vendor_results:
        vendors.append({
            "vendor_id": vc.vendor.vendor_id,
            "vendor_name": vc.vendor.vendor_name,
            "is_msme": vc.vendor.is_msme,
            "msme_category": vc.vendor.msme_category.value,
            "total_invoices": vc.total_invoices,
            "overdue_invoices": vc.overdue_invoices,
            "total_outstanding": str(vc.total_outstanding),
            "overdue_amount": str(vc.overdue_amount),
            "interest_liability": str(vc.interest_liability),
            "tax_disallowance": str(vc.tax_disallowance),
            "max_delay_days": vc.max_delay_days,
            "avg_delay_days": round(vc.avg_delay_days, 1),
            "risk_level": vc.risk_level.value,
            "risk_score": vc.risk_score,
            "requires_msme1_filing": vc.requires_msme1_filing,
            "violations": list(vc.violations),
            "recommended_actions": list(vc.recommended_actions),
        })

    # Sort by risk score descending
    vendors.sort(key=lambda v: v["risk_score"], reverse=True)

    return {
        "session_id": session_id,
        "total_vendors": len(vendors),
        "msme_vendors": sum(1 for v in vendors if v["is_msme"]),
        "vendors": vendors,
    }
