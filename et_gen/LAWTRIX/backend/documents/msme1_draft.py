"""MSME-1 filing draft generator."""
from __future__ import annotations

from models.session import AnalysisSession
from .generator import format_inr, get_filing_period


def generate_msme1_draft(session: AnalysisSession) -> dict:
    """Generate MSME-1 half-yearly return draft data."""
    msme_overdue = [
        vc for vc in session.vendor_results
        if vc.vendor.is_msme and vc.overdue_invoices > 0
    ]

    total_overdue = sum(vc.overdue_amount for vc in msme_overdue)
    period = get_filing_period()

    return {
        "form": "MSME-1",
        "title": f"Half-Yearly Return - {period}",
        "period": period,
        "total_msme_suppliers": len(msme_overdue),
        "total_outstanding": format_inr(total_overdue),
        "suppliers": [
            {
                "name": vc.vendor.vendor_name,
                "udyam_number": vc.vendor.msme_registration_number or "[To be filled]",
                "category": vc.vendor.msme_category.value,
                "outstanding_amount": format_inr(vc.overdue_amount),
                "delay_days": vc.max_delay_days,
                "interest_due": format_inr(vc.interest_liability),
                "reason_for_delay": "Payment processing delay",
            }
            for vc in msme_overdue
        ],
        "declaration": (
            "I hereby declare that the information given in this return "
            "is true and correct to the best of my knowledge and belief."
        ),
    }
