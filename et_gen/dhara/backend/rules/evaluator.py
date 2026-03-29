"""Rule evaluation engine - evaluates declarative rules against data."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .schema import Condition, ConditionOperator, Rule, RuleSet


def load_ruleset(json_path: str) -> RuleSet:
    """Load a RuleSet from a JSON file."""
    path = Path(__file__).parent / json_path
    with open(path, "r") as f:
        data = json.load(f)
    return RuleSet(**data)


def evaluate_condition(condition: Condition, context: dict[str, Any]) -> bool:
    """Evaluate a single condition against a data context."""
    field_value = context.get(condition.field)

    match condition.operator:
        case ConditionOperator.EQUALS:
            return field_value == condition.value
        case ConditionOperator.NOT_EQUALS:
            return field_value != condition.value
        case ConditionOperator.GREATER_THAN:
            return field_value is not None and field_value > condition.value
        case ConditionOperator.LESS_THAN:
            return field_value is not None and field_value < condition.value
        case ConditionOperator.GREATER_EQUAL:
            return field_value is not None and field_value >= condition.value
        case ConditionOperator.LESS_EQUAL:
            return field_value is not None and field_value <= condition.value
        case ConditionOperator.IN:
            return field_value in (condition.value or [])
        case ConditionOperator.NOT_IN:
            return field_value not in (condition.value or [])
        case ConditionOperator.IS_TRUE:
            return bool(field_value) is True
        case ConditionOperator.IS_FALSE:
            return bool(field_value) is False
        case ConditionOperator.IS_NULL:
            return field_value is None
        case ConditionOperator.IS_NOT_NULL:
            return field_value is not None
        case ConditionOperator.BETWEEN:
            if field_value is None:
                return False
            return condition.value_min <= field_value <= condition.value_max
        case _:
            return False


def evaluate_rule(rule: Rule, context: dict[str, Any]) -> bool:
    """Evaluate all conditions of a rule (AND logic). Returns True if all match."""
    return all(evaluate_condition(c, context) for c in rule.conditions)


def evaluate_ruleset(
    ruleset: RuleSet, context: dict[str, Any]
) -> list[Rule]:
    """Evaluate all rules in a ruleset, return triggered rules."""
    return [rule for rule in ruleset.rules if evaluate_rule(rule, context)]


# Pre-load all rulesets at module level
_rulesets: dict[str, RuleSet] = {}


def get_all_rulesets() -> dict[str, RuleSet]:
    """Load and cache all rule JSON files."""
    global _rulesets
    if not _rulesets:
        _rulesets = {
            "msmed_act": load_ruleset("msmed_act.json"),
            "it_act_43bh": load_ruleset("it_act_43bh.json"),
            "msme1_filing": load_ruleset("msme1_filing.json"),
        }
    return _rulesets
