"""Payment priority schedule generator."""
from __future__ import annotations

from models.session import AnalysisSession
from .generator import format_inr


def generate_payment_schedule(session: AnalysisSession) -> dict:
    """Generate a prioritized payment schedule."""
    if not session.action_plan:
        return {"error": "No action plan available"}

    payment_actions = [
        a for a in session.action_plan.actions
        if a.action_type.value in ("immediate_payment", "priority_payment", "tax_adjustment")
    ]

    return {
        "title": "MSME Payment Priority Schedule",
        "total_payments": len(payment_actions),
        "total_amount": format_inr(sum(a.financial_impact for a in payment_actions)),
        "schedule": [
            {
                "rank": i + 1,
                "priority": a.priority.value.upper(),
                "vendor": a.vendor_name,
                "action": a.description,
                "amount": format_inr(a.financial_impact),
                "deadline": a.deadline.isoformat() if a.deadline else "Immediate",
                "law": a.law_reference,
            }
            for i, a in enumerate(payment_actions)
        ],
    }
