"""LAWTRIX - Autonomous Compliance Execution Engine.

FastAPI application entry point with CORS, route mounting, and health check.
"""
from __future__ import annotations

import sys
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Add backend to sys.path for absolute imports
sys.path.insert(0, str(Path(__file__).parent))

from api import upload, analysis, vendors, actions, compliance_score, audit_log, documents, demo, rules_viz, market

app = FastAPI(
    title="LAWTRIX",
    description="Autonomous Compliance Execution Engine for Indian MSME Payment Laws",
    version="1.0.0",
)

# CORS - allow frontend dev server
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount all route modules
app.include_router(upload.router, tags=["Upload"])
app.include_router(analysis.router, tags=["Analysis"])
app.include_router(vendors.router, tags=["Vendors"])
app.include_router(actions.router, tags=["Actions"])
app.include_router(compliance_score.router, tags=["Compliance Score"])
app.include_router(audit_log.router, tags=["Audit Log"])
app.include_router(documents.router, tags=["Documents"])
app.include_router(demo.router, tags=["Demo"])
app.include_router(rules_viz.router, tags=["Rules Visualization"])
app.include_router(market.router, tags=["Market Validation"])


@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "LAWTRIX", "version": "1.0.0"}


@app.get("/impact/{session_id}")
async def get_impact(session_id: str):
    """Get financial impact summary."""
    from fastapi import HTTPException
    from storage.session_store import get_session
    from engine.orchestrator import calculate_impact

    session = get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail=f"Session {session_id} not found")

    impact = calculate_impact(session)
    return {
        "session_id": session_id,
        "penalties_avoided": impact.penalties_avoided,
        "interest_saved": impact.interest_saved,
        "tax_deductions_preserved": impact.tax_deductions_preserved,
        "total_financial_impact": impact.total_financial_impact,
        "compliance_improvement": impact.compliance_improvement,
        "risk_reduction": impact.risk_reduction,
        "hours_saved": impact.hours_saved,
        "vendors_remediated": impact.vendors_remediated,
    }
