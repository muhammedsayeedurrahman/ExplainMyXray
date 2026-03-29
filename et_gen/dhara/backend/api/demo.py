"""Demo endpoint - GET /demo/run."""
from __future__ import annotations

import uuid

from fastapi import APIRouter

from demo.demo_data import get_demo_csv
from parsers.csv_parser import parse_csv
from engine.orchestrator import run_pipeline, calculate_impact
from rules.zen_evaluator import evaluate as zen_evaluate, get_engine_info
from storage.session_store import save_session
from storage.audit_store import append_events

router = APIRouter()


@router.get("/demo/run")
async def run_demo():
    """Run full demo with sample data and return all results."""
    csv_data = get_demo_csv()
    batch = parse_csv(csv_data, "demo_sample_invoices.csv")

    session_id = f"demo-{str(uuid.uuid4())[:6]}"
    session, audit_events = run_pipeline(session_id, batch)

    save_session(session)
    append_events(session_id, audit_events)

    impact = calculate_impact(session)
    metrics = session.compliance_result.metrics

    # Build comprehensive demo response
    msme_vendors = [
        {
            "vendor_id": vc.vendor.vendor_id,
            "vendor_name": vc.vendor.vendor_name,
            "msme_category": vc.vendor.msme_category.value,
            "overdue_invoices": vc.overdue_invoices,
            "overdue_amount": str(vc.overdue_amount),
            "interest_liability": str(vc.interest_liability),
            "tax_disallowance": str(vc.tax_disallowance),
            "risk_score": vc.risk_score,
            "risk_level": vc.risk_level.value,
            "max_delay_days": vc.max_delay_days,
        }
        for vc in session.vendor_results
        if vc.vendor.is_msme
    ]
    msme_vendors.sort(key=lambda v: v["risk_score"], reverse=True)

    actions = []
    if session.action_plan:
        for a in session.action_plan.actions[:10]:
            actions.append({
                "action_id": a.action_id,
                "action_type": a.action_type.value,
                "priority": a.priority.value,
                "vendor_name": a.vendor_name,
                "description": a.description,
                "financial_impact": str(a.financial_impact),
                "law_reference": a.law_reference,
            })

    # Run zen-engine evaluation on aggregated context for demo showcase
    zen_demo_context = {
        "is_msme": True,
        "has_agreement": True,
        "agreed_payment_days": 30,
        "days_overdue": 60,
        "outstanding_amount": float(metrics.total_overdue_amount),
        "is_overdue": True,
        "days_to_itr_deadline": 180,
        "has_overdue_in_period": True,
        "filing_period": "Oct-Mar",
    }
    zen_result = zen_evaluate(zen_demo_context)
    engine_info = get_engine_info()

    return {
        "session_id": session_id,
        "status": "complete",
        "summary": {
            "total_invoices": session.total_invoices,
            "total_vendors": session.total_vendors,
            "msme_vendors": metrics.msme_vendors,
            "overdue_invoices": metrics.overdue_invoices,
            "total_overdue_amount": str(metrics.total_overdue_amount),
            "total_interest_liability": str(metrics.total_interest_liability),
            "total_tax_disallowance": str(metrics.total_tax_disallowance),
            "compliance_score": metrics.compliance_score,
            "vendors_at_risk": metrics.vendors_at_risk,
        },
        "msme_vendors": msme_vendors,
        "actions": actions,
        "impact": {
            "penalties_avoided": impact.penalties_avoided,
            "interest_saved": impact.interest_saved,
            "tax_deductions_preserved": impact.tax_deductions_preserved,
            "total_financial_impact": impact.total_financial_impact,
            "compliance_improvement": impact.compliance_improvement,
            "risk_reduction": impact.risk_reduction,
            "hours_saved": impact.hours_saved,
            "vendors_remediated": impact.vendors_remediated,
        },
        "rules_evaluated": {
            "triggered_rules": zen_result.get("triggered_rules", []),
            "timing_ms": zen_result.get("timing_ms", 0),
            "engine": zen_result.get("engine", "unknown"),
        },
        "engine_info": engine_info,
        "pipeline_steps": [
            {"step": 1, "name": "Data Ingestion", "status": "complete", "detail": f"Parsed {session.total_invoices} invoices"},
            {"step": 2, "name": "MSME Detection", "status": "complete", "detail": f"Identified {metrics.msme_vendors} MSME vendors"},
            {"step": 3, "name": "Due Date & Delay Analysis", "status": "complete", "detail": f"{metrics.overdue_invoices} overdue invoices found"},
            {"step": 4, "name": "Interest Calculation", "status": "complete", "detail": f"INR {metrics.total_interest_liability:,.2f} liability"},
            {"step": 5, "name": "Tax Impact (43B(h))", "status": "complete", "detail": f"INR {metrics.total_tax_disallowance:,.2f} at risk"},
            {"step": 6, "name": "Action Planning", "status": "complete", "detail": f"{len(actions)} actions generated"},
        ],
    }
