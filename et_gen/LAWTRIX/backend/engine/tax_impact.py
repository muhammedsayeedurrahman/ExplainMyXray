"""Section 43B(h) tax impact calculator."""
from __future__ import annotations

from decimal import Decimal, ROUND_HALF_UP

# Indian corporate tax rates
DEFAULT_TAX_RATE = Decimal("0.252")   # 25.17% effective for turnover < 400Cr
HIGHER_TAX_RATE = Decimal("0.3492")   # 34.94% effective for others
SURCHARGE_RATE = Decimal("0.04")       # 4% health & education cess


def calculate_tax_disallowance(
    overdue_amount: Decimal,
    tax_rate: Decimal | None = None,
) -> dict:
    """Calculate tax impact of Section 43B(h) disallowance.

    When MSME payments are overdue beyond statutory limits, the entire
    expense is disallowed as a deduction, increasing taxable income.

    Args:
        overdue_amount: Total overdue amount to MSME vendor
        tax_rate: Applicable tax rate (default: 25.17%)

    Returns:
        Dict with disallowance breakdown
    """
    if overdue_amount <= 0:
        return {
            "overdue_amount": Decimal("0.00"),
            "disallowed_amount": Decimal("0.00"),
            "additional_tax": Decimal("0.00"),
            "effective_tax_rate": str(tax_rate or DEFAULT_TAX_RATE),
            "description": "No overdue amount",
        }

    rate = tax_rate or DEFAULT_TAX_RATE
    disallowed = overdue_amount
    additional_tax = (disallowed * rate).quantize(
        Decimal("0.01"), rounding=ROUND_HALF_UP
    )

    return {
        "overdue_amount": overdue_amount,
        "disallowed_amount": disallowed,
        "additional_tax": additional_tax,
        "effective_tax_rate": str(rate),
        "description": (
            f"INR {disallowed:,.2f} disallowed under Section 43B(h). "
            f"Additional tax liability: INR {additional_tax:,.2f} "
            f"at {float(rate) * 100:.1f}% effective rate."
        ),
    }


def calculate_total_tax_impact(
    overdue_amounts: list[Decimal],
    tax_rate: Decimal | None = None,
) -> dict:
    """Calculate aggregate tax impact across all overdue MSME invoices."""
    total_overdue = sum(overdue_amounts, Decimal("0"))
    return calculate_tax_disallowance(total_overdue, tax_rate)


def calculate_savings_if_paid(
    overdue_amount: Decimal,
    tax_rate: Decimal | None = None,
) -> Decimal:
    """Calculate tax savings if overdue amount is paid before ITR deadline."""
    rate = tax_rate or DEFAULT_TAX_RATE
    return (overdue_amount * rate).quantize(
        Decimal("0.01"), rounding=ROUND_HALF_UP
    )
