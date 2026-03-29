"""In-memory session storage for demo."""
from __future__ import annotations

from models.session import AnalysisSession

# In-memory store: session_id -> AnalysisSession
_sessions: dict[str, AnalysisSession] = {}


def save_session(session: AnalysisSession) -> None:
    """Save or update a session."""
    _sessions[session.session_id] = session


def get_session(session_id: str) -> AnalysisSession | None:
    """Retrieve a session by ID."""
    return _sessions.get(session_id)


def list_sessions() -> list[AnalysisSession]:
    """List all sessions."""
    return list(_sessions.values())


def delete_session(session_id: str) -> bool:
    """Delete a session. Returns True if existed."""
    return _sessions.pop(session_id, None) is not None


def clear_all() -> None:
    """Clear all sessions."""
    _sessions.clear()
