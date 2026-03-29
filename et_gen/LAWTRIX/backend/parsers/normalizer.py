"""Column name and data normalization for invoice uploads."""
from __future__ import annotations

import re
from datetime import date, datetime
from decimal import Decimal, InvalidOperation
from typing import Any

# Common column name mappings
COLUMN_MAPPINGS: dict[str, list[str]] = {
    "invoice_id": ["invoice_id", "inv_id", "invoice_no", "invoice_number", "inv_no", "bill_no", "bill_number"],
    "vendor_id": ["vendor_id", "supplier_id", "vendor_code", "supplier_code", "party_code"],
    "vendor_name": ["vendor_name", "supplier_name", "party_name", "vendor", "supplier"],
    "invoice_date": ["invoice_date", "inv_date", "bill_date", "date", "invoice_dt"],
    "due_date": ["due_date", "payment_due_date", "due_dt", "payment_due"],
    "amount": ["amount", "invoice_amount", "total_amount", "bill_amount", "gross_amount", "inv_amount"],
    "paid_amount": ["paid_amount", "amount_paid", "payment_amount", "paid"],
    "payment_date": ["payment_date", "paid_date", "payment_dt", "date_paid"],
    "agreed_payment_days": ["agreed_payment_days", "payment_terms", "credit_days", "payment_days", "terms_days"],
    "is_msme_vendor": ["is_msme_vendor", "is_msme", "msme", "msme_vendor", "msme_flag"],
    "msme_category": ["msme_category", "msme_type", "enterprise_type", "msme_class"],
    "status": ["status", "invoice_status", "payment_status"],
    "description": ["description", "remarks", "narration", "notes"],
}


def normalize_column_name(raw_name: str) -> str | None:
    """Map a raw column name to a normalized field name."""
    cleaned = re.sub(r"[^a-z0-9]", "_", raw_name.strip().lower())
    cleaned = re.sub(r"_+", "_", cleaned).strip("_")

    for standard, variants in COLUMN_MAPPINGS.items():
        if cleaned in variants:
            return standard
    return None


def normalize_columns(raw_columns: list[str]) -> dict[str, str]:
    """Map raw column names to normalized names. Returns {raw: normalized}."""
    mapping = {}
    for raw in raw_columns:
        normalized = normalize_column_name(raw)
        if normalized:
            mapping[raw] = normalized
    return mapping


def parse_date(value: Any) -> date | None:
    """Parse various date formats to date object."""
    if value is None or (isinstance(value, str) and not value.strip()):
        return None
    if isinstance(value, date):
        return value
    if isinstance(value, datetime):
        return value.date()

    date_formats = [
        "%Y-%m-%d", "%d-%m-%Y", "%d/%m/%Y", "%m/%d/%Y",
        "%Y/%m/%d", "%d-%b-%Y", "%d %b %Y", "%d-%B-%Y",
        "%Y-%m-%d %H:%M:%S", "%d-%m-%Y %H:%M:%S",
    ]
    str_val = str(value).strip()
    for fmt in date_formats:
        try:
            return datetime.strptime(str_val, fmt).date()
        except ValueError:
            continue
    return None


def parse_amount(value: Any) -> Decimal:
    """Parse amount string to Decimal."""
    if value is None:
        return Decimal("0")
    try:
        cleaned = re.sub(r"[^\d.\-]", "", str(value))
        return Decimal(cleaned) if cleaned else Decimal("0")
    except InvalidOperation:
        return Decimal("0")


def parse_bool(value: Any) -> bool:
    """Parse boolean from various representations."""
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    s = str(value).strip().lower()
    return s in ("true", "yes", "1", "y", "t")


def parse_int(value: Any) -> int | None:
    """Parse integer from various representations."""
    if value is None:
        return None
    try:
        return int(float(str(value)))
    except (ValueError, TypeError):
        return None
