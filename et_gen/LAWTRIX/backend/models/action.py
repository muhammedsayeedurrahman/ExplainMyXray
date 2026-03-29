"""Action plan models."""
from __future__ import annotations

from datetime import date, datetime
from decimal import Decimal
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class ActionType(str, Enum):
    IMMEDIATE_PAYMENT = "immediate_payment"
    PRIORITY_PAYMENT = "priority_payment"
    MSME1_FILING = "msme1_filing"
    TERM_RENEGOTIATION = "term_renegotiation"
    INTEREST_PROVISION = "interest_provision"
    TAX_ADJUSTMENT = "tax_adjustment"
    VENDOR_CLASSIFICATION = "vendor_classification"


class ActionStatus(str, Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"


class ActionPriority(str, Enum):
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class ComplianceAction(BaseModel):
    """Single recommended or executed action."""

    model_config = {"frozen": True}

    action_id: str
    action_type: ActionType
    priority: ActionPriority
    vendor_id: str
    vendor_name: str
    description: str
    law_reference: str
    financial_impact: Decimal = Decimal("0")
    deadline: Optional[date] = None
    status: ActionStatus = ActionStatus.PENDING
    executed_at: Optional[datetime] = None
    result: Optional[str] = None


class ActionPlan(BaseModel):
    """Full action plan with prioritized actions."""

    model_config = {"frozen": True}

    session_id: str
    generated_at: datetime = Field(default_factory=datetime.now)
    actions: tuple[ComplianceAction, ...] = ()
    total_savings_if_executed: Decimal = Decimal("0")
    compliance_score_before: int = 0
    compliance_score_after: int = 0
    estimated_hours_saved: float = 0.0
