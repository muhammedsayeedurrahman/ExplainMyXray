"""Vendor data models."""
from __future__ import annotations

from datetime import date
from decimal import Decimal
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class MSMECategory(str, Enum):
    MICRO = "micro"
    SMALL = "small"
    MEDIUM = "medium"
    NON_MSME = "non_msme"


class RiskLevel(str, Enum):
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    COMPLIANT = "compliant"


class Vendor(BaseModel):
    """Vendor master record."""

    model_config = {"frozen": True}

    vendor_id: str
    vendor_name: str
    is_msme: bool = False
    msme_category: MSMECategory = MSMECategory.NON_MSME
    msme_registration_number: Optional[str] = None
    agreed_payment_days: Optional[int] = None
    state: Optional[str] = None
    gstin: Optional[str] = None


class VendorCompliance(BaseModel):
    """Vendor-level compliance analysis result."""

    model_config = {"frozen": True}

    vendor: Vendor
    total_invoices: int = 0
    overdue_invoices: int = 0
    total_outstanding: Decimal = Decimal("0")
    overdue_amount: Decimal = Decimal("0")
    interest_liability: Decimal = Decimal("0")
    tax_disallowance: Decimal = Decimal("0")
    max_delay_days: int = 0
    avg_delay_days: float = 0.0
    risk_level: RiskLevel = RiskLevel.COMPLIANT
    risk_score: int = Field(default=0, ge=0, le=100)
    requires_msme1_filing: bool = False
    violations: tuple[str, ...] = ()
    recommended_actions: tuple[str, ...] = ()
    effective_due_days: int = 45  # max statutory limit
