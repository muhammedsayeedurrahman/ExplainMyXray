"""Append-only audit log store."""
from __future__ import annotations

from models.audit import AuditEvent, DecisionLog

# In-memory: session_id -> list of AuditEvents
_audit_logs: dict[str, list[AuditEvent]] = {}


def append_events(session_id: str, events: tuple[AuditEvent, ...] | list[AuditEvent]) -> None:
    """Append audit events for a session."""
    if session_id not in _audit_logs:
        _audit_logs[session_id] = []
    _audit_logs[session_id].extend(events)


def get_decision_log(session_id: str) -> DecisionLog:
    """Get the full decision log for a session."""
    events = _audit_logs.get(session_id, [])
    steps = set()
    for e in events:
        if e.step is not None:
            steps.add(e.step)

    return DecisionLog(
        session_id=session_id,
        events=tuple(events),
        total_events=len(events),
        pipeline_steps_completed=len(steps),
    )


def clear_all() -> None:
    """Clear all audit logs."""
    _audit_logs.clear()
