"""Rule engine schema - Law as Code."""
from __future__ import annotations

from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel


class ConditionOperator(str, Enum):
    EQUALS = "equals"
    NOT_EQUALS = "not_equals"
    GREATER_THAN = "greater_than"
    LESS_THAN = "less_than"
    GREATER_EQUAL = "greater_equal"
    LESS_EQUAL = "less_equal"
    IN = "in"
    NOT_IN = "not_in"
    IS_TRUE = "is_true"
    IS_FALSE = "is_false"
    IS_NULL = "is_null"
    IS_NOT_NULL = "is_not_null"
    BETWEEN = "between"


class Severity(str, Enum):
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"


class Condition(BaseModel):
    """Single evaluable condition."""

    model_config = {"frozen": True}

    field: str
    operator: ConditionOperator
    value: Any = None
    value_min: Any = None
    value_max: Any = None


class RuleAction(BaseModel):
    """Action to take when rule triggers."""

    model_config = {"frozen": True}

    action_type: str
    description: str
    parameters: dict[str, Any] = {}


class Rule(BaseModel):
    """Declarative compliance rule."""

    model_config = {"frozen": True}

    rule_id: str
    name: str
    law: str
    section: str
    description: str
    severity: Severity
    conditions: tuple[Condition, ...]
    actions: tuple[RuleAction, ...]
    legal_text: str = ""
    penalty_description: str = ""
    effective_date: Optional[str] = None


class RuleSet(BaseModel):
    """Collection of rules from a single law."""

    model_config = {"frozen": True}

    law_name: str
    version: str
    rules: tuple[Rule, ...]
