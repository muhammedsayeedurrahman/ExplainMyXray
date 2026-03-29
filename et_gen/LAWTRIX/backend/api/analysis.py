"""Analysis endpoint - GET /analysis/{session_id}."""
from __future__ import annotations

from fastapi import APIRouter, HTTPException

from storage.session_store import get_session

router = APIRouter()


@router.get("/analysis/{session_id}")
async def get_analysis(session_id: str):
    """Get full compliance scan results for a session."""
    session = get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail=f"Session {session_id} not found")

    result = session.compliance_result
    if not result:
        raise HTTPException(status_code=400, detail="Analysis not yet complete")

    metrics = result.metrics
    return {
        "session_id": session_id,
        "status": session.status.value,
        "analyzed_at": result.analyzed_at.isoformat(),
        "processing_time_ms": round(result.processing_time_ms, 1),
        "metrics": {
            "total_vendors": metrics.total_vendors,
            "msme_vendors": metrics.msme_vendors,
            "non_msme_vendors": metrics.non_msme_vendors,
            "total_invoices": metrics.total_invoices,
            "overdue_invoices": metrics.overdue_invoices,
            "total_outstanding": str(metrics.total_outstanding),
            "total_overdue_amount": str(metrics.total_overdue_amount),
            "total_interest_liability": str(metrics.total_interest_liability),
            "total_tax_disallowance": str(metrics.total_tax_disallowance),
            "total_potential_penalty": str(metrics.total_potential_penalty),
            "compliance_score": metrics.compliance_score,
            "vendors_at_risk": metrics.vendors_at_risk,
            "msme1_filings_required": metrics.msme1_filings_required,
        },
        "violations_count": len(result.violations),
        "rules_applied": list(result.rules_applied),
    }
