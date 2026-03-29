"""MSME vendor detection engine."""
from __future__ import annotations

from models.invoice import Invoice, InvoiceBatch
from models.vendor import MSMECategory, Vendor


def detect_msme_vendors(batch: InvoiceBatch) -> dict[str, Vendor]:
    """Scan invoices and build vendor registry with MSME classification.

    In production, this would cross-reference with the Udyam portal.
    For demo, we rely on the is_msme_vendor flag in parsed invoice data.
    """
    vendors: dict[str, Vendor] = {}

    for inv in batch.invoices:
        if inv.vendor_id in vendors:
            continue

        category = _classify_msme(inv.msme_category) if inv.is_msme_vendor else MSMECategory.NON_MSME

        vendors[inv.vendor_id] = Vendor(
            vendor_id=inv.vendor_id,
            vendor_name=inv.vendor_name,
            is_msme=inv.is_msme_vendor,
            msme_category=category,
            agreed_payment_days=inv.agreed_payment_days,
        )

    return vendors


def _classify_msme(category_str: str | None) -> MSMECategory:
    """Map raw category string to MSMECategory enum."""
    if not category_str:
        return MSMECategory.NON_MSME
    normalized = category_str.strip().lower()
    mapping = {
        "micro": MSMECategory.MICRO,
        "small": MSMECategory.SMALL,
        "medium": MSMECategory.MEDIUM,
    }
    return mapping.get(normalized, MSMECategory.NON_MSME)


def get_msme_vendors(vendors: dict[str, Vendor]) -> dict[str, Vendor]:
    """Filter to MSME vendors only."""
    return {vid: v for vid, v in vendors.items() if v.is_msme}


def get_vendor_invoices(
    vendor_id: str, batch: InvoiceBatch
) -> tuple[Invoice, ...]:
    """Get all invoices for a specific vendor."""
    return tuple(inv for inv in batch.invoices if inv.vendor_id == vendor_id)
