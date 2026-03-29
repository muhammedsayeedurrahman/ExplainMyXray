"""Compound interest calculation per MSMED Act."""
from __future__ import annotations

import json
from datetime import date, datetime
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path


def _load_rbi_rate() -> dict:
    """Load current RBI rate configuration."""
    path = Path(__file__).parent.parent / "rules" / "rbi_rates.json"
    with open(path, "r") as f:
        return json.load(f)


def get_rate_for_date(invoice_date: date | str) -> dict:
    """Find the applicable RBI bank rate for a given invoice date.

    Uses period-aware lookup (OpenFisca pattern): finds the rate entry
    whose effective_from date is the most recent one on or before the
    invoice date.

    Args:
        invoice_date: The date to look up the rate for.

    Returns:
        Dict with bank_rate, multiplier, effective_rate, effective_from.
    """
    if isinstance(invoice_date, str):
        invoice_date = datetime.strptime(invoice_date, "%Y-%m-%d").date()

    config = _load_rbi_rate()
    rates = config.get("rates", [])

    # Rates are sorted newest-first; find the first entry on or before invoice_date
    for entry in rates:
        entry_date = datetime.strptime(entry["effective_from"], "%Y-%m-%d").date()
        if entry_date <= invoice_date:
            return {
                "bank_rate": entry.get("bank_rate", entry.get("rate", config["current_rate"])),
                "multiplier": entry.get("multiplier", config["msmed_multiplier"]),
                "effective_rate": entry.get("effective_rate", entry.get("bank_rate", 6.5) * 3),
                "effective_from": entry["effective_from"],
            }

    # Fallback to oldest known rate
    if rates:
        oldest = rates[-1]
        return {
            "bank_rate": oldest.get("bank_rate", oldest.get("rate", config["current_rate"])),
            "multiplier": oldest.get("multiplier", config["msmed_multiplier"]),
            "effective_rate": oldest.get("effective_rate", oldest.get("bank_rate", 6.5) * 3),
            "effective_from": oldest["effective_from"],
        }

    return {
        "bank_rate": config["current_rate"],
        "multiplier": config["msmed_multiplier"],
        "effective_rate": config["effective_rate"],
        "effective_from": "2023-02-08",
    }


def calculate_interest(
    principal: Decimal,
    delay_days: int,
    annual_rate: float | None = None,
) -> Decimal:
    """Calculate compound interest per MSMED Act Section 16.

    Formula: P × (1 + r/12)^n - P
    Where:
        P = principal (outstanding amount)
        r = annual rate (3x RBI bank rate)
        n = number of months of delay

    Args:
        principal: Outstanding amount
        delay_days: Number of days overdue
        annual_rate: Override annual rate (default: 3x RBI rate)

    Returns:
        Interest amount (Decimal, 2 places)
    """
    if delay_days <= 0 or principal <= 0:
        return Decimal("0.00")

    if annual_rate is None:
        rbi_config = _load_rbi_rate()
        annual_rate = rbi_config["current_rate"] * rbi_config["msmed_multiplier"]

    monthly_rate = Decimal(str(annual_rate)) / Decimal("100") / Decimal("12")
    months = Decimal(str(delay_days)) / Decimal("30")

    # Compound interest: P × (1 + r/12)^n - P
    compound_factor = (Decimal("1") + monthly_rate) ** months
    interest = principal * compound_factor - principal

    return interest.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)


def calculate_total_interest(
    invoices_with_delays: list[tuple[Decimal, int]],
) -> Decimal:
    """Calculate total interest across multiple invoice-delay pairs.

    Args:
        invoices_with_delays: List of (outstanding_amount, delay_days)

    Returns:
        Total interest liability
    """
    total = Decimal("0.00")
    for amount, days in invoices_with_delays:
        total += calculate_interest(amount, days)
    return total


def get_current_rate_info() -> dict:
    """Get current interest rate configuration."""
    config = _load_rbi_rate()
    return {
        "rbi_bank_rate": config["current_rate"],
        "multiplier": config["msmed_multiplier"],
        "effective_annual_rate": config["effective_rate"],
        "compounding": config["compounding"],
    }
