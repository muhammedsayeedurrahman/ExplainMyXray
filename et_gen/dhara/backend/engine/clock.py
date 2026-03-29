"""Payment due date and delay calculator."""
from __future__ import annotations

from datetime import date, timedelta

from models.invoice import Invoice

# MSMED Act statutory limits
MAX_AGREED_DAYS = 45
DEFAULT_DAYS_NO_AGREEMENT = 15


def calculate_effective_due_date(invoice: Invoice) -> date:
    """Calculate the effective due date per MSMED Act rules.

    Rules:
    - If agreement exists and <= 45 days: use agreed days
    - If agreement exists and > 45 days: cap at 45 days (illegal terms)
    - If no agreement: 15 days from invoice date
    """
    if not invoice.is_msme_vendor:
        return invoice.due_date

    if invoice.agreed_payment_days is not None:
        effective_days = min(invoice.agreed_payment_days, MAX_AGREED_DAYS)
    else:
        effective_days = DEFAULT_DAYS_NO_AGREEMENT

    return invoice.invoice_date + timedelta(days=effective_days)


def calculate_delay_days(invoice: Invoice, reference_date: date | None = None) -> int:
    """Calculate days of delay from effective due date.

    Returns 0 if not overdue or already paid on time.
    """
    ref = reference_date or date.today()

    if invoice.payment_date is not None:
        # Paid: delay = payment_date - effective_due_date (if late)
        effective_due = calculate_effective_due_date(invoice)
        if invoice.payment_date <= effective_due:
            return 0
        return (invoice.payment_date - effective_due).days

    # Unpaid: delay = today - effective_due_date
    effective_due = calculate_effective_due_date(invoice)
    if ref <= effective_due:
        return 0
    return (ref - effective_due).days


def is_terms_illegal(invoice: Invoice) -> bool:
    """Check if agreed payment terms exceed statutory 45-day limit."""
    if not invoice.is_msme_vendor:
        return False
    if invoice.agreed_payment_days is None:
        return False
    return invoice.agreed_payment_days > MAX_AGREED_DAYS


def has_agreement(invoice: Invoice) -> bool:
    """Check if payment terms agreement exists."""
    return invoice.agreed_payment_days is not None


def days_to_itr_deadline(reference_date: date | None = None) -> int:
    """Calculate days remaining to ITR filing deadline.

    ITR deadline for companies: 31st October of assessment year.
    For FY ending March 31, deadline is Oct 31 of same calendar year.
    """
    ref = reference_date or date.today()
    year = ref.year if ref.month >= 4 else ref.year
    deadline = date(year, 10, 31)
    if ref > deadline:
        deadline = date(year + 1, 10, 31)
    return (deadline - ref).days
