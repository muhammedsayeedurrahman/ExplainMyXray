"""Full 6-step compliance pipeline orchestrator."""
from __future__ import annotations

import time
import uuid
from datetime import datetime
from decimal import Decimal

from models.action import ActionPlan
from models.audit import AuditEvent, AuditEventType, DecisionLog
from models.compliance import ComplianceMetrics, ComplianceResult, RuleViolation
from models.invoice import InvoiceBatch
from models.session import AnalysisSession, ImpactSummary, SessionStatus
from models.vendor import RiskLevel, Vendor, VendorCompliance

from .action_planner import generate_action_plan
from .clock import calculate_delay_days, calculate_effective_due_date, has_agreement, is_terms_illegal
from .interest import calculate_interest
from .risk_scorer import calculate_overall_compliance_score, calculate_vendor_risk_score, score_to_risk_level
from .scanner import detect_msme_vendors, get_msme_vendors, get_vendor_invoices
from .tax_impact import calculate_tax_disallowance

from rules.evaluator import evaluate_ruleset, get_all_rulesets
from rules.zen_evaluator import evaluate as zen_evaluate, get_engine_info


def run_pipeline(session_id: str, batch: InvoiceBatch) -> AnalysisSession:
    """Execute the full 6-step compliance analysis pipeline.

    Steps:
    1. Scan & Detect MSME vendors
    2. Calculate due dates & delays
    3. Calculate interest liability
    4. Calculate tax impact (43B(h))
    5. Score risks & detect violations
    6. Generate action plan
    """
    start_time = time.time()
    audit_events: list[AuditEvent] = []

    # Pipeline start
    audit_events.append(_audit(
        session_id, AuditEventType.PIPELINE_START, 0,
        "Starting 6-step compliance analysis pipeline",
        details={"total_invoices": len(batch.invoices), "source": batch.source_filename},
    ))

    # ── Step 1: Scan & Detect ──
    all_vendors = detect_msme_vendors(batch)
    msme_vendors = get_msme_vendors(all_vendors)

    audit_events.append(_audit(
        session_id, AuditEventType.VENDOR_SCAN, 1,
        f"Scanned {len(all_vendors)} vendors, detected {len(msme_vendors)} MSME vendors",
        details={
            "total_vendors": len(all_vendors),
            "msme_vendors": len(msme_vendors),
            "msme_ids": list(msme_vendors.keys()),
        },
    ))

    # ── Step 2-5: Analyze each vendor ──
    vendor_results: list[VendorCompliance] = []
    all_violations: list[RuleViolation] = []
    rules_applied: set[str] = set()
    rulesets = get_all_rulesets()

    for vendor_id, vendor in all_vendors.items():
        invoices = get_vendor_invoices(vendor_id, batch)

        if not vendor.is_msme:
            vendor_results.append(VendorCompliance(
                vendor=vendor,
                total_invoices=len(invoices),
            ))
            continue

        # Step 2: Calculate delays
        delay_data = []
        overdue_count = 0
        total_outstanding = Decimal("0")
        overdue_amount = Decimal("0")
        max_delay = 0
        total_delay = 0

        for inv in invoices:
            delay = calculate_delay_days(inv)
            outstanding = inv.outstanding

            if delay > 0:
                overdue_count += 1
                overdue_amount += outstanding
                delay_data.append((outstanding, delay))
                max_delay = max(max_delay, delay)
                total_delay += delay

            total_outstanding += outstanding

        avg_delay = total_delay / overdue_count if overdue_count > 0 else 0.0

        audit_events.append(_audit(
            session_id, AuditEventType.DUE_DATE_CALC, 2,
            f"Vendor {vendor.vendor_name}: {overdue_count}/{len(invoices)} overdue, max {max_delay} days",
            details={"vendor_id": vendor_id, "overdue_count": overdue_count, "max_delay": max_delay},
        ))

        # Step 3: Interest calculation
        interest_total = Decimal("0")
        for amount, days in delay_data:
            interest_total += calculate_interest(amount, days)

        audit_events.append(_audit(
            session_id, AuditEventType.INTEREST_CALC, 3,
            f"Interest liability for {vendor.vendor_name}: INR {interest_total:,.2f}",
            law_reference="MSMED Act 2006, Section 16",
            details={"vendor_id": vendor_id, "interest": str(interest_total)},
        ))

        # Step 4: Tax impact
        tax_result = calculate_tax_disallowance(overdue_amount)
        tax_disallowance = tax_result["additional_tax"]

        audit_events.append(_audit(
            session_id, AuditEventType.TAX_IMPACT_CALC, 4,
            f"Tax disallowance for {vendor.vendor_name}: INR {tax_disallowance:,.2f}",
            law_reference="IT Act 1961, Section 43B(h)",
            details={"vendor_id": vendor_id, "tax_impact": str(tax_disallowance)},
        ))

        # Step 5: Rule evaluation & risk scoring (dual engine)
        # zen-engine evaluation (sub-ms, Rust-powered)
        zen_context = {
            "is_msme": vendor.is_msme,
            "has_agreement": has_agreement(invoices[0]) if invoices else False,
            "agreed_payment_days": invoices[0].agreed_payment_days if invoices else 0,
            "days_overdue": max_delay,
            "outstanding_amount": float(overdue_amount),
            "is_overdue": max_delay > 0,
            "days_to_itr_deadline": 180,
            "has_overdue_in_period": True,
            "filing_period": "Oct-Mar",
        }
        zen_result = zen_evaluate(zen_context)

        audit_events.append(_audit(
            session_id, AuditEventType.RISK_SCORE_CALC, 5,
            f"zen-engine evaluated {vendor.vendor_name} in {zen_result.get('timing_ms', 0):.3f}ms "
            f"({zen_result.get('engine', 'unknown')}), "
            f"{len(zen_result.get('triggered_rules', []))} rules triggered",
            details={
                "vendor_id": vendor_id,
                "zen_timing_ms": zen_result.get("timing_ms"),
                "zen_engine": zen_result.get("engine"),
                "zen_rules_triggered": len(zen_result.get("triggered_rules", [])),
            },
        ))

        # Existing Python rule evaluator (authoritative)
        violations = _evaluate_vendor_rules(vendor, invoices, rulesets, max_delay, overdue_amount)
        all_violations.extend(violations)
        for v in violations:
            rules_applied.add(v.rule_id)

        risk_score = calculate_vendor_risk_score(
            total_invoices=len(invoices),
            overdue_invoices=overdue_count,
            avg_delay_days=avg_delay,
            max_delay_days=max_delay,
            total_outstanding=total_outstanding,
            interest_liability=interest_total,
            tax_disallowance=tax_disallowance,
            violation_count=len(violations),
        )

        risk_level = score_to_risk_level(risk_score)
        requires_filing = max_delay > 45

        vendor_results.append(VendorCompliance(
            vendor=vendor,
            total_invoices=len(invoices),
            overdue_invoices=overdue_count,
            total_outstanding=total_outstanding,
            overdue_amount=overdue_amount,
            interest_liability=interest_total,
            tax_disallowance=tax_disallowance,
            max_delay_days=max_delay,
            avg_delay_days=avg_delay,
            risk_level=risk_level,
            risk_score=risk_score,
            requires_msme1_filing=requires_filing,
            violations=tuple(v.rule_id for v in violations),
            recommended_actions=tuple(v.recommended_action for v in violations if v.recommended_action),
        ))

        audit_events.append(_audit(
            session_id, AuditEventType.RISK_SCORE_CALC, 5,
            f"Risk score for {vendor.vendor_name}: {risk_score}/100 ({risk_level.value})",
            details={"vendor_id": vendor_id, "risk_score": risk_score, "risk_level": risk_level.value},
        ))

    # Aggregate metrics
    msme_results = [vr for vr in vendor_results if vr.vendor.is_msme]
    vendor_scores = [vr.risk_score for vr in msme_results]
    compliance_score = calculate_overall_compliance_score(
        vendor_scores, len(all_vendors), len(msme_vendors)
    )

    metrics = ComplianceMetrics(
        total_vendors=len(all_vendors),
        msme_vendors=len(msme_vendors),
        non_msme_vendors=len(all_vendors) - len(msme_vendors),
        total_invoices=len(batch.invoices),
        overdue_invoices=sum(vr.overdue_invoices for vr in vendor_results),
        total_outstanding=sum(vr.total_outstanding for vr in vendor_results),
        total_overdue_amount=sum(vr.overdue_amount for vr in vendor_results),
        total_interest_liability=sum(vr.interest_liability for vr in vendor_results),
        total_tax_disallowance=sum(vr.tax_disallowance for vr in vendor_results),
        total_potential_penalty=sum(vr.interest_liability + vr.tax_disallowance for vr in vendor_results),
        compliance_score=compliance_score,
        vendors_at_risk=sum(1 for vr in msme_results if vr.risk_level in (RiskLevel.CRITICAL, RiskLevel.HIGH)),
        msme1_filings_required=sum(1 for vr in msme_results if vr.requires_msme1_filing),
    )

    compliance_result = ComplianceResult(
        session_id=session_id,
        metrics=metrics,
        violations=tuple(all_violations),
        rules_applied=tuple(rules_applied),
        processing_time_ms=(time.time() - start_time) * 1000,
    )

    # ── Step 6: Action Plan ──
    action_plan = generate_action_plan(session_id, vendor_results, compliance_score)

    audit_events.append(_audit(
        session_id, AuditEventType.ACTION_PLANNED, 6,
        f"Generated {len(action_plan.actions)} actions, potential savings: INR {action_plan.total_savings_if_executed:,.2f}",
        details={
            "action_count": len(action_plan.actions),
            "savings": str(action_plan.total_savings_if_executed),
            "score_before": action_plan.compliance_score_before,
            "score_after": action_plan.compliance_score_after,
        },
    ))

    processing_time = (time.time() - start_time) * 1000

    audit_events.append(_audit(
        session_id, AuditEventType.PIPELINE_COMPLETE, 6,
        f"Pipeline complete in {processing_time:.0f}ms",
        details={"processing_time_ms": processing_time},
    ))

    return AnalysisSession(
        session_id=session_id,
        status=SessionStatus.COMPLETE,
        source_filename=batch.source_filename,
        total_invoices=len(batch.invoices),
        total_vendors=len(all_vendors),
        compliance_result=compliance_result,
        vendor_results=tuple(vendor_results),
        action_plan=action_plan,
        processing_time_ms=processing_time,
    ), tuple(audit_events)


def calculate_impact(session: AnalysisSession) -> ImpactSummary:
    """Calculate the financial impact summary for a completed session."""
    if not session.compliance_result or not session.action_plan:
        return ImpactSummary(session_id=session.session_id)

    metrics = session.compliance_result.metrics
    plan = session.action_plan

    total_penalty = metrics.total_interest_liability + metrics.total_tax_disallowance
    return ImpactSummary(
        session_id=session.session_id,
        penalties_avoided=f"{plan.total_savings_if_executed:,.2f}",
        interest_saved=f"{metrics.total_interest_liability:,.2f}",
        tax_deductions_preserved=f"{metrics.total_tax_disallowance:,.2f}",
        total_financial_impact=f"{total_penalty:,.2f}",
        compliance_improvement=f"{plan.compliance_score_before} → {plan.compliance_score_after}",
        risk_reduction=f"{max(0, plan.compliance_score_after - plan.compliance_score_before)}%",
        hours_saved=plan.estimated_hours_saved,
        vendors_remediated=metrics.vendors_at_risk,
    )


def _evaluate_vendor_rules(vendor, invoices, rulesets, max_delay, overdue_amount):
    """Evaluate all rules against a vendor's data."""
    violations = []

    for inv in invoices:
        context = {
            "is_msme": vendor.is_msme,
            "has_agreement": has_agreement(inv),
            "agreed_payment_days": inv.agreed_payment_days,
            "days_since_acceptance": calculate_delay_days(inv),
            "days_overdue": calculate_delay_days(inv),
            "is_overdue": calculate_delay_days(inv) > 0,
            "outstanding_amount": float(inv.outstanding),
            "days_to_itr_deadline": 180,  # Simplified
            "has_overdue_in_period": True,
            "filing_period": "Oct-Mar",
        }

        for rs_name, rs in rulesets.items():
            triggered = evaluate_ruleset(rs, context)
            for rule in triggered:
                violations.append(RuleViolation(
                    rule_id=rule.rule_id,
                    rule_name=rule.name,
                    law_reference=f"{rule.law}, {rule.section}",
                    vendor_id=vendor.vendor_id,
                    vendor_name=vendor.vendor_name,
                    invoice_id=inv.invoice_id,
                    severity=rule.severity.value,
                    description=rule.description,
                    financial_impact=inv.outstanding,
                    recommended_action=rule.actions[0].description if rule.actions else "",
                ))

    # Deduplicate by rule_id per vendor
    seen = set()
    unique = []
    for v in violations:
        key = (v.rule_id, v.vendor_id)
        if key not in seen:
            seen.add(key)
            unique.append(v)
    return unique


def _audit(
    session_id: str,
    event_type: AuditEventType,
    step: int,
    description: str,
    details: dict | None = None,
    law_reference: str | None = None,
) -> AuditEvent:
    return AuditEvent(
        event_id=str(uuid.uuid4())[:8],
        session_id=session_id,
        event_type=event_type,
        step=step,
        description=description,
        details=details or {},
        law_reference=law_reference,
    )
