"""Actions endpoints - GET/POST actions."""
from __future__ import annotations

from datetime import datetime

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from models.action import ActionStatus

from storage.session_store import get_session, save_session

router = APIRouter()


@router.get("/actions/{session_id}")
async def get_actions(session_id: str):
    """Get recommended compliance actions."""
    session = get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail=f"Session {session_id} not found")

    plan = session.action_plan
    if not plan:
        raise HTTPException(status_code=400, detail="No action plan available")

    actions = []
    for a in plan.actions:
        actions.append({
            "action_id": a.action_id,
            "action_type": a.action_type.value,
            "priority": a.priority.value,
            "vendor_id": a.vendor_id,
            "vendor_name": a.vendor_name,
            "description": a.description,
            "law_reference": a.law_reference,
            "financial_impact": str(a.financial_impact),
            "deadline": a.deadline.isoformat() if a.deadline else None,
            "status": a.status.value,
        })

    return {
        "session_id": session_id,
        "total_actions": len(actions),
        "total_savings": str(plan.total_savings_if_executed),
        "score_before": plan.compliance_score_before,
        "score_after": plan.compliance_score_after,
        "hours_saved": plan.estimated_hours_saved,
        "actions": actions,
    }


class ExecuteRequest(BaseModel):
    session_id: str
    action_id: str


@router.post("/actions/execute")
async def execute_action(req: ExecuteRequest):
    """Execute a specific compliance action (simulated for demo)."""
    session = get_session(req.session_id)
    if not session:
        raise HTTPException(status_code=404, detail=f"Session {req.session_id} not found")

    plan = session.action_plan
    if not plan:
        raise HTTPException(status_code=400, detail="No action plan available")

    # Find the action
    target = None
    for a in plan.actions:
        if a.action_id == req.action_id:
            target = a
            break

    if not target:
        raise HTTPException(status_code=404, detail=f"Action {req.action_id} not found")

    # Simulate execution by creating updated action
    return {
        "action_id": target.action_id,
        "status": "completed",
        "executed_at": datetime.now().isoformat(),
        "result": f"Action '{target.description}' executed successfully (demo mode)",
        "financial_impact": str(target.financial_impact),
    }
