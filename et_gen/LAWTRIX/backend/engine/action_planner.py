"""Compliance action planner - generates prioritized remediation actions."""
from __future__ import annotations

import uuid
from datetime import date, timedelta
from decimal import Decimal

from models.action import (
    ActionPlan,
    ActionPriority,
    ActionStatus,
    ActionType,
    ComplianceAction,
)
from models.vendor import RiskLevel, VendorCompliance

from .clock import days_to_itr_deadline


def generate_action_plan(
    session_id: str,
    vendor_results: list[VendorCompliance],
    compliance_score_before: int,
) -> ActionPlan:
    """Generate a prioritized action plan from vendor compliance results."""
    actions: list[ComplianceAction] = []

    for vc in vendor_results:
        if not vc.vendor.is_msme:
            continue
        actions.extend(_generate_vendor_actions(vc))

    # Sort by priority (critical first), then by financial impact (highest first)
    priority_order = {
        ActionPriority.CRITICAL: 0,
        ActionPriority.HIGH: 1,
        ActionPriority.MEDIUM: 2,
        ActionPriority.LOW: 3,
    }
    actions.sort(
        key=lambda a: (priority_order[a.priority], -float(a.financial_impact))
    )

    total_savings = sum(a.financial_impact for a in actions)
    score_after = _estimate_score_after(compliance_score_before, actions)

    return ActionPlan(
        session_id=session_id,
        actions=tuple(actions),
        total_savings_if_executed=total_savings,
        compliance_score_before=compliance_score_before,
        compliance_score_after=score_after,
        estimated_hours_saved=len(actions) * 2.5,
    )


def _generate_vendor_actions(vc: VendorCompliance) -> list[ComplianceAction]:
    """Generate actions for a single vendor."""
    actions: list[ComplianceAction] = []

    # Critical: Immediate payment for high-risk vendors
    if vc.risk_level in (RiskLevel.CRITICAL, RiskLevel.HIGH) and vc.overdue_amount > 0:
        actions.append(
            ComplianceAction(
                action_id=_uid(),
                action_type=ActionType.IMMEDIATE_PAYMENT,
                priority=ActionPriority.CRITICAL if vc.risk_level == RiskLevel.CRITICAL else ActionPriority.HIGH,
                vendor_id=vc.vendor.vendor_id,
                vendor_name=vc.vendor.vendor_name,
                description=f"Immediate payment of INR {vc.overdue_amount:,.2f} to {vc.vendor.vendor_name}",
                law_reference="MSMED Act 2006, Section 15",
                financial_impact=vc.interest_liability + vc.tax_disallowance,
                deadline=date.today() + timedelta(days=7),
            )
        )

    # High: Priority payment before ITR deadline
    if vc.tax_disallowance > 0 and days_to_itr_deadline() > 0:
        actions.append(
            ComplianceAction(
                action_id=_uid(),
                action_type=ActionType.TAX_ADJUSTMENT,
                priority=ActionPriority.HIGH,
                vendor_id=vc.vendor.vendor_id,
                vendor_name=vc.vendor.vendor_name,
                description=f"Pay before ITR deadline to preserve INR {vc.tax_disallowance:,.2f} deduction",
                law_reference="IT Act 1961, Section 43B(h)",
                financial_impact=vc.tax_disallowance,
                deadline=date.today() + timedelta(days=min(days_to_itr_deadline(), 30)),
            )
        )

    # Medium: MSME-1 filing
    if vc.requires_msme1_filing:
        actions.append(
            ComplianceAction(
                action_id=_uid(),
                action_type=ActionType.MSME1_FILING,
                priority=ActionPriority.MEDIUM,
                vendor_id=vc.vendor.vendor_id,
                vendor_name=vc.vendor.vendor_name,
                description=f"File MSME-1 return for overdue payments to {vc.vendor.vendor_name}",
                law_reference="MSMED Act 2006, Section 9; MSME Notification 02.11.2018",
                financial_impact=Decimal("25000"),  # Penalty for non-filing
            )
        )

    # Medium: Interest provision
    if vc.interest_liability > 0:
        actions.append(
            ComplianceAction(
                action_id=_uid(),
                action_type=ActionType.INTEREST_PROVISION,
                priority=ActionPriority.MEDIUM,
                vendor_id=vc.vendor.vendor_id,
                vendor_name=vc.vendor.vendor_name,
                description=f"Provision INR {vc.interest_liability:,.2f} for interest liability",
                law_reference="MSMED Act 2006, Section 16",
                financial_impact=vc.interest_liability,
            )
        )

    return actions


def _estimate_score_after(score_before: int, actions: list[ComplianceAction]) -> int:
    """Estimate compliance score after all actions are executed."""
    critical_actions = sum(1 for a in actions if a.priority == ActionPriority.CRITICAL)
    high_actions = sum(1 for a in actions if a.priority == ActionPriority.HIGH)
    medium_actions = sum(1 for a in actions if a.priority == ActionPriority.MEDIUM)

    improvement = critical_actions * 12 + high_actions * 8 + medium_actions * 4
    return min(100, score_before + improvement)


def _uid() -> str:
    return str(uuid.uuid4())[:8]
