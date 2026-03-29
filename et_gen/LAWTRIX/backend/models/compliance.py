"""Compliance result models."""
from __future__ import annotations

from datetime import datetime
from decimal import Decimal
from typing import Optional

from pydantic import BaseModel, Field


class RuleViolation(BaseModel):
    """Single rule violation instance."""

    model_config = {"frozen": True}

    rule_id: str
    rule_name: str
    law_reference: str
    vendor_id: str
    vendor_name: str
    invoice_id: Optional[str] = None
    severity: str  # critical, high, medium, low
    description: str
    financial_impact: Decimal = Decimal("0")
    recommended_action: str = ""


class ComplianceMetrics(BaseModel):
    """Aggregate compliance metrics."""

    model_config = {"frozen": True}

    total_vendors: int = 0
    msme_vendors: int = 0
    non_msme_vendors: int = 0
    total_invoices: int = 0
    overdue_invoices: int = 0
    total_outstanding: Decimal = Decimal("0")
    total_overdue_amount: Decimal = Decimal("0")
    total_interest_liability: Decimal = Decimal("0")
    total_tax_disallowance: Decimal = Decimal("0")
    total_potential_penalty: Decimal = Decimal("0")
    compliance_score: int = Field(default=0, ge=0, le=100)
    vendors_at_risk: int = 0
    msme1_filings_required: int = 0


class ComplianceResult(BaseModel):
    """Full compliance analysis output."""

    model_config = {"frozen": True}

    session_id: str
    analyzed_at: datetime = Field(default_factory=datetime.now)
    metrics: ComplianceMetrics
    violations: tuple[RuleViolation, ...] = ()
    rules_applied: tuple[str, ...] = ()
    processing_time_ms: float = 0.0
