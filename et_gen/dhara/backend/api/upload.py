"""Upload endpoint - POST /upload (CSV/Excel)."""
from __future__ import annotations

import uuid

from fastapi import APIRouter, File, UploadFile, HTTPException

from parsers.csv_parser import parse_csv
from parsers.excel_parser import parse_excel
from engine.orchestrator import run_pipeline
from storage.session_store import save_session
from storage.audit_store import append_events

router = APIRouter()


@router.post("/upload")
async def upload_invoices(file: UploadFile = File(...)):
    """Upload CSV or Excel file with invoice data and run compliance analysis."""
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file provided")

    content = await file.read()
    ext = file.filename.lower().rsplit(".", 1)[-1] if "." in file.filename else ""

    if ext == "csv":
        batch = parse_csv(content, file.filename)
    elif ext in ("xlsx", "xls"):
        batch = parse_excel(content, file.filename)
    else:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file format: .{ext}. Use CSV or Excel (.xlsx)",
        )

    if not batch.invoices:
        raise HTTPException(
            status_code=400,
            detail=f"No valid invoices found. Errors: {'; '.join(batch.parse_errors[:5])}",
        )

    session_id = str(uuid.uuid4())[:8]
    session, audit_events = run_pipeline(session_id, batch)

    save_session(session)
    append_events(session_id, audit_events)

    return {
        "session_id": session.session_id,
        "status": session.status.value,
        "total_invoices": session.total_invoices,
        "total_vendors": session.total_vendors,
        "parse_errors": list(batch.parse_errors[:5]),
        "message": f"Analyzed {session.total_invoices} invoices from {session.total_vendors} vendors",
    }
