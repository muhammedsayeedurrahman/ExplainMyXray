"""Excel file parser for invoice data."""
from __future__ import annotations

import io
import uuid

import pandas as pd

from models.invoice import InvoiceBatch

from .csv_parser import parse_csv


def parse_excel(content: bytes, filename: str = "upload.xlsx") -> InvoiceBatch:
    """Parse Excel content into an InvoiceBatch by converting to CSV internally."""
    try:
        df = pd.read_excel(io.BytesIO(content), engine="openpyxl")
        csv_content = df.to_csv(index=False)
        batch = parse_csv(csv_content, filename)
        return batch
    except Exception as e:
        return InvoiceBatch(
            batch_id=str(uuid.uuid4())[:8],
            invoices=(),
            source_filename=filename,
            parse_errors=(f"Excel parse error: {e}",),
        )
