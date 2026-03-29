"""Audit log endpoint - GET /audit-log/{session_id}."""
from __future__ import annotations

from fastapi import APIRouter, HTTPException

from storage.audit_store import get_decision_log

router = APIRouter()


@router.get("/audit-log/{session_id}")
async def get_audit_log(session_id: str):
    """Get decision trace log for a session."""
    log = get_decision_log(session_id)

    if log.total_events == 0:
        raise HTTPException(status_code=404, detail=f"No audit log for session {session_id}")

    events = []
    for e in log.events:
        events.append({
            "event_id": e.event_id,
            "event_type": e.event_type.value,
            "timestamp": e.timestamp.isoformat(),
            "step": e.step,
            "description": e.description,
            "details": e.details,
            "law_reference": e.law_reference,
            "decision": e.decision,
            "reasoning": e.reasoning,
        })

    return {
        "session_id": session_id,
        "total_events": log.total_events,
        "pipeline_steps_completed": log.pipeline_steps_completed,
        "events": events,
    }
