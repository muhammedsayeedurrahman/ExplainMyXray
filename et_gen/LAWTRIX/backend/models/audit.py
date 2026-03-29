"""Audit trail models."""
from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field


class AuditEventType(str, Enum):
    DATA_UPLOAD = "data_upload"
    VENDOR_SCAN = "vendor_scan"
    MSME_DETECTION = "msme_detection"
    DUE_DATE_CALC = "due_date_calc"
    INTEREST_CALC = "interest_calc"
    TAX_IMPACT_CALC = "tax_impact_calc"
    RISK_SCORE_CALC = "risk_score_calc"
    RULE_EVALUATION = "rule_evaluation"
    VIOLATION_DETECTED = "violation_detected"
    ACTION_PLANNED = "action_planned"
    ACTION_EXECUTED = "action_executed"
    DOCUMENT_GENERATED = "document_generated"
    COMPLIANCE_SCORE = "compliance_score"
    PIPELINE_START = "pipeline_start"
    PIPELINE_COMPLETE = "pipeline_complete"


class AuditEvent(BaseModel):
    """Single audit trail event."""

    model_config = {"frozen": True}

    event_id: str
    session_id: str
    event_type: AuditEventType
    timestamp: datetime = Field(default_factory=datetime.now)
    step: Optional[int] = None
    description: str
    details: dict[str, Any] = Field(default_factory=dict)
    law_reference: Optional[str] = None
    decision: Optional[str] = None
    reasoning: Optional[str] = None


class DecisionLog(BaseModel):
    """Complete decision log for a session."""

    model_config = {"frozen": True}

    session_id: str
    events: tuple[AuditEvent, ...] = ()
    total_events: int = 0
    pipeline_steps_completed: int = 0
