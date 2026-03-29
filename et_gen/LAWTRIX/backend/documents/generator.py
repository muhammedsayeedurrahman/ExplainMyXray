"""Base document generator utilities."""
from __future__ import annotations

from datetime import date


def format_inr(amount) -> str:
    """Format amount in Indian numbering system."""
    num = float(amount)
    if num >= 10000000:
        return f"INR {num / 10000000:.2f} Crore"
    if num >= 100000:
        return f"INR {num / 100000:.2f} Lakh"
    return f"INR {num:,.2f}"


def get_financial_year() -> str:
    """Get current financial year string."""
    today = date.today()
    if today.month >= 4:
        return f"FY {today.year}-{today.year + 1}"
    return f"FY {today.year - 1}-{today.year}"


def get_filing_period() -> str:
    """Get current MSME-1 filing period."""
    today = date.today()
    if today.month >= 4 and today.month <= 9:
        return f"April {today.year} - September {today.year}"
    if today.month >= 10:
        return f"October {today.year} - March {today.year + 1}"
    return f"October {today.year - 1} - March {today.year}"
