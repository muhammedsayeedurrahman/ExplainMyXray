"""Compliance risk scoring engine (0-100)."""
from __future__ import annotations

from decimal import Decimal

from models.vendor import RiskLevel, Vendor

# Risk score weights
WEIGHT_OVERDUE_RATIO = 25       # % of invoices overdue
WEIGHT_DELAY_SEVERITY = 25      # average delay days
WEIGHT_FINANCIAL_EXPOSURE = 25  # interest + tax impact relative to outstanding
WEIGHT_VIOLATION_COUNT = 25     # number of distinct rule violations


def calculate_vendor_risk_score(
    total_invoices: int,
    overdue_invoices: int,
    avg_delay_days: float,
    max_delay_days: int,
    total_outstanding: Decimal,
    interest_liability: Decimal,
    tax_disallowance: Decimal,
    violation_count: int,
) -> int:
    """Calculate 0-100 risk score for a vendor.

    Higher score = higher risk.

    Components (25 points each):
    1. Overdue ratio: % of invoices that are overdue
    2. Delay severity: based on average and max delay days
    3. Financial exposure: interest + tax as % of outstanding
    4. Violation count: number of distinct rule violations
    """
    # Component 1: Overdue ratio (0-25)
    if total_invoices == 0:
        overdue_score = 0
    else:
        ratio = overdue_invoices / total_invoices
        overdue_score = min(int(ratio * WEIGHT_OVERDUE_RATIO), WEIGHT_OVERDUE_RATIO)

    # Component 2: Delay severity (0-25)
    # 0-15 days: low, 15-45: medium, 45-90: high, 90+: critical
    if avg_delay_days <= 0:
        delay_score = 0
    elif avg_delay_days <= 15:
        delay_score = int((avg_delay_days / 15) * 8)
    elif avg_delay_days <= 45:
        delay_score = 8 + int(((avg_delay_days - 15) / 30) * 8)
    elif avg_delay_days <= 90:
        delay_score = 16 + int(((avg_delay_days - 45) / 45) * 5)
    else:
        delay_score = 21 + min(int((avg_delay_days - 90) / 30), 4)

    delay_score = min(delay_score, WEIGHT_DELAY_SEVERITY)

    # Component 3: Financial exposure (0-25)
    if total_outstanding > 0:
        exposure = float(interest_liability + tax_disallowance) / float(total_outstanding)
        financial_score = min(int(exposure * 100), WEIGHT_FINANCIAL_EXPOSURE)
    else:
        financial_score = 0

    # Component 4: Violation count (0-25)
    if violation_count == 0:
        violation_score = 0
    elif violation_count <= 2:
        violation_score = violation_count * 5
    elif violation_count <= 5:
        violation_score = 10 + (violation_count - 2) * 3
    else:
        violation_score = min(19 + violation_count - 5, WEIGHT_VIOLATION_COUNT)

    return min(overdue_score + delay_score + financial_score + violation_score, 100)


def score_to_risk_level(score: int) -> RiskLevel:
    """Convert numeric score to risk level."""
    if score >= 75:
        return RiskLevel.CRITICAL
    if score >= 50:
        return RiskLevel.HIGH
    if score >= 25:
        return RiskLevel.MEDIUM
    if score > 0:
        return RiskLevel.LOW
    return RiskLevel.COMPLIANT


def calculate_overall_compliance_score(
    vendor_risk_scores: list[int],
    total_vendors: int,
    msme_vendors: int,
) -> int:
    """Calculate overall 0-100 compliance score (higher = better).

    This is the inverse of risk - 100 means fully compliant.
    """
    if not vendor_risk_scores or total_vendors == 0:
        return 100

    avg_risk = sum(vendor_risk_scores) / len(vendor_risk_scores)

    # Weight MSME vendors more heavily
    msme_weight = msme_vendors / total_vendors if total_vendors > 0 else 0.5
    weighted_risk = avg_risk * (0.5 + msme_weight * 0.5)

    compliance = max(0, min(100, int(100 - weighted_risk)))
    return compliance
