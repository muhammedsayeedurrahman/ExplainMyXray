"""Compliance score endpoint - GET /compliance-score/{session_id}."""
from __future__ import annotations

from fastapi import APIRouter, HTTPException

from storage.session_store import get_session

router = APIRouter()


@router.get("/compliance-score/{session_id}")
async def get_compliance_score(session_id: str):
    """Get 0-100 compliance score with breakdown."""
    session = get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail=f"Session {session_id} not found")

    result = session.compliance_result
    plan = session.action_plan

    if not result:
        raise HTTPException(status_code=400, detail="Analysis not yet complete")

    metrics = result.metrics

    # Risk distribution
    risk_dist = {"critical": 0, "high": 0, "medium": 0, "low": 0, "compliant": 0}
    for vc in session.vendor_results:
        if vc.vendor.is_msme:
            risk_dist[vc.risk_level.value] += 1

    return {
        "session_id": session_id,
        "compliance_score": metrics.compliance_score,
        "score_after_actions": plan.compliance_score_after if plan else metrics.compliance_score,
        "breakdown": {
            "msme_vendors_analyzed": metrics.msme_vendors,
            "vendors_at_risk": metrics.vendors_at_risk,
            "overdue_invoices": metrics.overdue_invoices,
            "total_violations": len(result.violations),
            "rules_applied": len(result.rules_applied),
        },
        "risk_distribution": risk_dist,
        "interpretation": _interpret_score(metrics.compliance_score),
    }


def _interpret_score(score: int) -> str:
    if score >= 80:
        return "Excellent compliance posture. Minor issues only."
    if score >= 60:
        return "Good compliance with some areas needing attention."
    if score >= 40:
        return "Moderate risk. Multiple compliance gaps detected."
    if score >= 20:
        return "High risk. Significant compliance failures requiring immediate action."
    return "Critical risk. Severe non-compliance across multiple areas."
