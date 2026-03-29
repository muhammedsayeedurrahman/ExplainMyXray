"""Rules visualization API - GET /rules/*."""
from __future__ import annotations

from fastapi import APIRouter

from rules.zen_evaluator import get_engine_info, get_jdm_content, get_rules_summary

router = APIRouter()


@router.get("/rules/jdm")
async def get_jdm():
    """Return the full JDM decision model for frontend visualization."""
    return get_jdm_content()


@router.get("/rules/summary")
async def rules_summary():
    """Return human-readable rules summary with legal references."""
    rules = get_rules_summary()
    return {
        "total_rules": len(rules),
        "rules": rules,
    }


@router.get("/rules/engine-info")
async def engine_info():
    """Return engine metadata, version, and latency stats."""
    return get_engine_info()
