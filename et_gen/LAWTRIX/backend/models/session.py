"""Analysis session models."""
from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field

from .action import ActionPlan
from .compliance import ComplianceResult
from .vendor import VendorCompliance


class SessionStatus(str, Enum):
    CREATED = "created"
    UPLOADING = "uploading"
    ANALYZING = "analyzing"
    COMPLETE = "complete"
    ERROR = "error"


class AnalysisSession(BaseModel):
    """Session tracking an end-to-end analysis run."""

    model_config = {"frozen": True}

    session_id: str
    created_at: datetime = Field(default_factory=datetime.now)
    status: SessionStatus = SessionStatus.CREATED
    source_filename: Optional[str] = None
    total_invoices: int = 0
    total_vendors: int = 0
    compliance_result: Optional[ComplianceResult] = None
    vendor_results: tuple[VendorCompliance, ...] = ()
    action_plan: Optional[ActionPlan] = None
    processing_time_ms: float = 0.0
    error_message: Optional[str] = None


class ImpactSummary(BaseModel):
    """Financial impact summary."""

    model_config = {"frozen": True}

    session_id: str
    penalties_avoided: str = "0"
    interest_saved: str = "0"
    tax_deductions_preserved: str = "0"
    total_financial_impact: str = "0"
    compliance_improvement: str = "0%"
    risk_reduction: str = "0%"
    hours_saved: float = 0.0
    vendors_remediated: int = 0
