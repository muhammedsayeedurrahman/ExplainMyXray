"""Invoice data models."""
from __future__ import annotations

from datetime import date, datetime
from decimal import Decimal
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class InvoiceStatus(str, Enum):
    PENDING = "pending"
    PAID = "paid"
    OVERDUE = "overdue"
    PARTIALLY_PAID = "partially_paid"


class Invoice(BaseModel):
    """Single invoice record."""

    model_config = {"frozen": True}

    invoice_id: str
    vendor_id: str
    vendor_name: str
    invoice_date: date
    due_date: date
    amount: Decimal = Field(ge=0)
    paid_amount: Decimal = Field(default=Decimal("0"), ge=0)
    payment_date: Optional[date] = None
    agreed_payment_days: Optional[int] = None
    status: InvoiceStatus = InvoiceStatus.PENDING
    is_msme_vendor: bool = False
    msme_category: Optional[str] = None  # micro, small, medium
    description: Optional[str] = None

    @property
    def outstanding(self) -> Decimal:
        return self.amount - self.paid_amount

    @property
    def is_overdue(self) -> bool:
        if self.status == InvoiceStatus.PAID:
            return False
        return date.today() > self.due_date

    @property
    def days_overdue(self) -> int:
        if not self.is_overdue:
            return 0
        return (date.today() - self.due_date).days


class InvoiceBatch(BaseModel):
    """Collection of invoices from a single upload."""

    model_config = {"frozen": True}

    batch_id: str
    invoices: tuple[Invoice, ...]
    uploaded_at: datetime = Field(default_factory=datetime.now)
    source_filename: str = ""
    total_records: int = 0
    parse_errors: tuple[str, ...] = ()
