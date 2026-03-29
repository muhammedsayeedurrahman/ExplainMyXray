"""CSV file parser for invoice data."""
from __future__ import annotations

import csv
import io
import uuid
from datetime import date, timedelta

from models.invoice import Invoice, InvoiceBatch, InvoiceStatus

from .normalizer import (
    normalize_columns,
    parse_amount,
    parse_bool,
    parse_date,
    parse_int,
)


def parse_csv(content: str | bytes, filename: str = "upload.csv") -> InvoiceBatch:
    """Parse CSV content into an InvoiceBatch."""
    if isinstance(content, bytes):
        content = content.decode("utf-8-sig")

    reader = csv.DictReader(io.StringIO(content))
    if not reader.fieldnames:
        return InvoiceBatch(
            batch_id=str(uuid.uuid4())[:8],
            invoices=(),
            source_filename=filename,
            parse_errors=("No columns found in CSV",),
        )

    col_map = normalize_columns(list(reader.fieldnames))
    invoices: list[Invoice] = []
    errors: list[str] = []

    for row_num, row in enumerate(reader, start=2):
        try:
            inv = _parse_row(row, col_map, row_num)
            if inv:
                invoices.append(inv)
        except Exception as e:
            errors.append(f"Row {row_num}: {e}")

    batch_id = str(uuid.uuid4())[:8]
    return InvoiceBatch(
        batch_id=batch_id,
        invoices=tuple(invoices),
        source_filename=filename,
        total_records=len(invoices),
        parse_errors=tuple(errors),
    )


def _parse_row(
    row: dict, col_map: dict[str, str], row_num: int
) -> Invoice | None:
    """Parse a single CSV row into an Invoice."""
    def get(field: str):
        for raw, norm in col_map.items():
            if norm == field:
                return row.get(raw)
        return None

    invoice_id = get("invoice_id") or f"INV-{row_num:04d}"
    vendor_id = get("vendor_id") or f"V-{row_num:04d}"
    vendor_name = get("vendor_name") or f"Vendor {vendor_id}"

    invoice_date = parse_date(get("invoice_date"))
    if not invoice_date:
        invoice_date = date.today() - timedelta(days=60)

    amount = parse_amount(get("amount"))
    if amount <= 0:
        return None

    paid_amount = parse_amount(get("paid_amount"))
    payment_date = parse_date(get("payment_date"))
    agreed_days = parse_int(get("agreed_payment_days"))
    is_msme = parse_bool(get("is_msme_vendor"))
    msme_cat = get("msme_category")

    # Calculate due date
    due_date = parse_date(get("due_date"))
    if not due_date:
        if agreed_days:
            due_date = invoice_date + timedelta(days=agreed_days)
        else:
            due_date = invoice_date + timedelta(days=30)

    # Determine status
    if paid_amount >= amount:
        status = InvoiceStatus.PAID
    elif paid_amount > 0:
        status = InvoiceStatus.PARTIALLY_PAID
    elif date.today() > due_date:
        status = InvoiceStatus.OVERDUE
    else:
        status = InvoiceStatus.PENDING

    return Invoice(
        invoice_id=str(invoice_id).strip(),
        vendor_id=str(vendor_id).strip(),
        vendor_name=str(vendor_name).strip(),
        invoice_date=invoice_date,
        due_date=due_date,
        amount=amount,
        paid_amount=paid_amount,
        payment_date=payment_date,
        agreed_payment_days=agreed_days,
        status=status,
        is_msme_vendor=is_msme,
        msme_category=str(msme_cat).strip() if msme_cat else None,
        description=get("description"),
    )
