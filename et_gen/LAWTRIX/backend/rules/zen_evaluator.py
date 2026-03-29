"""zen-engine evaluator - Rust-powered rules engine wrapper.

Loads JDM (JSON Decision Model) files and evaluates compliance rules
using gorules/zen engine with sub-millisecond latency.

Falls back gracefully to the existing Python evaluator if zen-engine
is not installed.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any


JDM_PATH = Path(__file__).parent / "jdm" / "msmed_compliance.json"

# Try to import zen-engine; graceful fallback if not installed
_zen_available = False
_engine = None

try:
    import zen as zen_engine

    _zen_available = True
except ImportError:
    try:
        import zen_engine

        _zen_available = True
    except ImportError:
        _zen_available = False


def _get_engine():
    """Lazy-initialize the zen engine singleton."""
    global _engine
    if _engine is None and _zen_available:
        _engine = zen_engine.ZenEngine()
    return _engine


def _load_jdm() -> dict:
    """Load the JDM decision model from disk."""
    with open(JDM_PATH, "r") as f:
        return json.load(f)


def is_zen_available() -> bool:
    """Check if zen-engine is installed and usable."""
    return _zen_available


def evaluate(context: dict[str, Any]) -> dict[str, Any]:
    """Evaluate compliance rules against the provided context.

    Uses zen-engine if available, otherwise falls back to a
    JDM-compatible Python evaluation.

    Args:
        context: Invoice/vendor data to evaluate.
            Expected fields: is_msme, has_agreement, agreed_payment_days,
            days_overdue, outstanding_amount, days_to_itr_deadline,
            has_overdue_in_period, filing_period

    Returns:
        Dict with triggered_rules, severity, actions, timing_ms, engine
    """
    start = time.perf_counter()

    if _zen_available:
        result = _evaluate_zen(context)
    else:
        result = _evaluate_fallback(context)

    elapsed_ms = (time.perf_counter() - start) * 1000
    result["timing_ms"] = round(elapsed_ms, 3)
    result["engine"] = "zen-engine (Rust)" if _zen_available else "python-fallback"

    return result


def _evaluate_zen(context: dict[str, Any]) -> dict[str, Any]:
    """Evaluate using the Rust-powered zen-engine."""
    engine = _get_engine()
    jdm = _load_jdm()
    jdm_content = json.dumps(jdm)

    try:
        decision = engine.create_decision(jdm_content)
        raw_result = decision.evaluate(context)

        return {
            "triggered_rules": _extract_rules_from_zen(raw_result),
            "raw_output": raw_result,
        }
    except Exception as e:
        # Fall back to Python evaluation on zen errors
        result = _evaluate_fallback(context)
        result["zen_error"] = str(e)
        return result


def _extract_rules_from_zen(raw_result: Any) -> list[dict]:
    """Extract triggered rules from zen-engine result."""
    rules = []
    if isinstance(raw_result, dict):
        for key, value in raw_result.items():
            if isinstance(value, list):
                for item in value:
                    if isinstance(item, dict) and "rule_id" in item:
                        rules.append(item)
            elif isinstance(value, dict) and "rule_id" in value:
                rules.append(value)
    return rules


def _evaluate_fallback(context: dict[str, Any]) -> dict[str, Any]:
    """Pure Python JDM evaluation fallback.

    Evaluates the JDM decision tables using simple condition matching.
    This mirrors what zen-engine does but in Python.
    """
    jdm = _load_jdm()
    triggered = []

    # Find decision table nodes
    for node in jdm.get("nodes", []):
        if node.get("type") != "decisionTableNode":
            continue

        content = node.get("content", {})
        inputs = content.get("inputs", [])
        outputs = content.get("outputs", [])
        rules = content.get("rules", [])

        for rule_def in rules:
            if _matches_rule(rule_def, inputs, context):
                rule_output = {}
                for out in outputs:
                    out_id = out["id"]
                    field = out.get("field", out.get("name", ""))
                    if out_id in rule_def:
                        rule_output[field] = rule_def[out_id]
                triggered.append({
                    "table": node["name"],
                    **rule_output,
                })

    return {"triggered_rules": triggered}


def _matches_rule(
    rule_def: dict,
    inputs: list[dict],
    context: dict[str, Any],
) -> bool:
    """Check if a context matches a JDM decision table rule row."""
    for inp in inputs:
        inp_id = inp["id"]
        field = inp.get("field", inp.get("name", ""))
        condition = rule_def.get(inp_id, "")

        if not condition or condition.startswith("_"):
            continue

        value = context.get(field)
        if value is None:
            return False

        if not _eval_condition(condition, value):
            return False

    return True


def _eval_condition(condition: str, value: Any) -> bool:
    """Evaluate a JDM condition string against a value."""
    condition = condition.strip()

    if condition == "true":
        return bool(value) is True
    if condition == "false":
        return bool(value) is False

    if condition.startswith("> "):
        return float(value) > float(condition[2:])
    if condition.startswith("< "):
        return float(value) < float(condition[2:])
    if condition.startswith(">= "):
        return float(value) >= float(condition[3:])
    if condition.startswith("<= "):
        return float(value) <= float(condition[3:])
    if condition.startswith("== "):
        return str(value) == condition[3:]

    # Direct string match
    return str(value) == condition


def get_jdm_content() -> dict:
    """Return the raw JDM file content for visualization."""
    return _load_jdm()


def get_rules_summary() -> list[dict]:
    """Return a human-readable summary of all rules with legal references."""
    jdm = _load_jdm()
    summary = []

    for node in jdm.get("nodes", []):
        if node.get("type") != "decisionTableNode":
            continue

        content = node.get("content", {})
        outputs = content.get("outputs", [])
        output_fields = {o["id"]: o.get("field", o.get("name", "")) for o in outputs}

        for rule_def in content.get("rules", []):
            entry = {
                "table": node["name"],
                "description": rule_def.get("_description", ""),
            }
            for out_id, field_name in output_fields.items():
                if out_id in rule_def:
                    entry[field_name] = rule_def[out_id]
            summary.append(entry)

    return summary


def get_engine_info() -> dict:
    """Return engine metadata and capabilities."""
    zen_version = "unknown"
    if _zen_available:
        try:
            import importlib.metadata
            zen_version = importlib.metadata.version("zen-engine")
        except Exception:
            zen_version = "installed"

    return {
        "engine": "gorules/zen" if _zen_available else "python-fallback",
        "zen_available": _zen_available,
        "zen_version": zen_version,
        "runtime": "Rust (WASM)" if _zen_available else "Python",
        "jdm_format": "JSON Decision Model v1",
        "total_rules": len(get_rules_summary()),
        "decision_tables": 3,
        "laws_encoded": [
            "MSMED Act 2006 (Sections 2, 9, 15, 16)",
            "Income Tax Act 1961 (Section 43B(h))",
            "MSME-1 Filing (Companies Act 2013, Section 405)",
        ],
        "capabilities": [
            "Sub-millisecond rule evaluation",
            "JSON Decision Model (JDM) format",
            "Decision table + graph-based flow",
            "Collect hit policy (all matching rules fire)",
        ],
    }
