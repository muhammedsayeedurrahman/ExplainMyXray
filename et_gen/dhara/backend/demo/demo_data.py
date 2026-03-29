"""Demo dataset: 52 vendors, 847 invoices with realistic Indian data."""
from __future__ import annotations

import csv
import io
import random
from datetime import date, timedelta

# Realistic Indian vendor names
MSME_VENDORS = [
    ("V001", "Sharma Precision Tools Pvt Ltd", "micro", 30),
    ("V002", "Gupta Electronics Components", "small", 45),
    ("V003", "Patel Fabrication Works", "micro", None),  # No agreement
    ("V004", "Reddy Industrial Supplies", "small", 60),  # Illegal terms > 45
    ("V005", "Mehta Polymers & Plastics", "medium", 30),
    ("V006", "Singh Auto Parts", "micro", None),
    ("V007", "Joshi Engineering Solutions", "small", 45),
    ("V008", "Agarwal Electrical Systems", "micro", 30),
    ("V009", "Iyer Packaging Industries", "small", 30),
    ("V010", "Kumar Metal Works", "micro", None),
    ("V011", "Deshmukh Rubber Products", "small", 45),
    ("V012", "Banerjee Textiles", "medium", 30),
    ("V013", "Nair Chemical Industries", "small", 30),
    ("V014", "Rao Hydraulics & Pneumatics", "micro", None),
    ("V015", "Verma Welding Electrodes", "micro", 30),
    ("V016", "Mishra Casting Works", "small", 45),
    ("V017", "Pillai Electronics Assembly", "micro", 30),
    ("V018", "Thakur Industrial Fasteners", "small", 90),  # Illegal terms
    ("V019", "Saxena Tooling Solutions", "micro", None),
    ("V020", "Bhatt CNC Components", "small", 45),
    ("V021", "Kulkarni Springs & Washers", "micro", 30),
    ("V022", "Choudhary Paint & Coatings", "small", 30),
    ("V023", "Tiwari Insulation Materials", "medium", 45),
]

NON_MSME_VENDORS = [
    ("V024", "Tata Steel Limited", None, 60),
    ("V025", "Reliance Industries Ltd", None, 90),
    ("V026", "Mahindra & Mahindra Ltd", None, 45),
    ("V027", "Larsen & Toubro Ltd", None, 60),
    ("V028", "Bharat Heavy Electricals", None, 45),
    ("V029", "Hindustan Unilever Ltd", None, 30),
    ("V030", "JSW Steel Limited", None, 60),
    ("V031", "Godrej Industries Ltd", None, 45),
    ("V032", "Adani Enterprises Ltd", None, 90),
    ("V033", "Wipro Technologies Ltd", None, 30),
    ("V034", "Siemens India Ltd", None, 60),
    ("V035", "ABB India Limited", None, 45),
    ("V036", "Bosch India Limited", None, 60),
    ("V037", "Havells India Ltd", None, 30),
    ("V038", "Blue Star Limited", None, 45),
    ("V039", "Voltas Limited", None, 60),
    ("V040", "Crompton Greaves", None, 45),
    ("V041", "Thermax Limited", None, 30),
    ("V042", "Kirloskar Brothers", None, 60),
    ("V043", "Escorts Limited", None, 45),
    ("V044", "Ashok Leyland Ltd", None, 30),
    ("V045", "Cummins India Ltd", None, 45),
    ("V046", "SKF India Limited", None, 60),
    ("V047", "Timken India Ltd", None, 45),
    ("V048", "Schaeffler India", None, 30),
    ("V049", "NSK Bearings India", None, 60),
    ("V050", "NTN Bearing India", None, 45),
    ("V051", "ITC Limited", None, 30),
    ("V052", "Bajaj Electricals", None, 45),
]


def generate_demo_csv() -> str:
    """Generate 847 invoices across 52 vendors as CSV string."""
    random.seed(42)  # Reproducible demo data
    today = date.today()
    rows = []

    all_vendors = [
        (vid, name, cat, terms, True) for vid, name, cat, terms in MSME_VENDORS
    ] + [
        (vid, name, cat, terms, False) for vid, name, cat, terms in NON_MSME_VENDORS
    ]

    invoice_counter = 1
    for vid, name, msme_cat, agreed_days, is_msme in all_vendors:
        # MSME vendors: 10-30 invoices, Non-MSME: 5-25 invoices
        num_invoices = random.randint(10, 30) if is_msme else random.randint(5, 25)

        for _ in range(num_invoices):
            inv_date = today - timedelta(days=random.randint(10, 180))
            amount = round(random.uniform(5000, 500000), 2)

            if agreed_days:
                due = inv_date + timedelta(days=agreed_days)
            else:
                due = inv_date + timedelta(days=random.choice([15, 30, 45]))

            # Varied payment scenarios
            scenario = random.random()
            if scenario < 0.35:  # 35% paid on time
                paid = amount
                pay_date = due - timedelta(days=random.randint(0, 5))
                status = "paid"
            elif scenario < 0.55:  # 20% paid late
                paid = amount
                pay_date = due + timedelta(days=random.randint(1, 60))
                status = "paid"
            elif scenario < 0.70:  # 15% partially paid
                paid = round(amount * random.uniform(0.3, 0.8), 2)
                pay_date = due + timedelta(days=random.randint(0, 30))
                status = "partially_paid"
            else:  # 30% unpaid
                paid = 0
                pay_date = None
                status = "overdue" if today > due else "pending"

            rows.append({
                "invoice_id": f"INV-{invoice_counter:04d}",
                "vendor_id": vid,
                "vendor_name": name,
                "invoice_date": inv_date.isoformat(),
                "due_date": due.isoformat(),
                "amount": f"{amount:.2f}",
                "paid_amount": f"{paid:.2f}",
                "payment_date": pay_date.isoformat() if pay_date else "",
                "agreed_payment_days": str(agreed_days) if agreed_days else "",
                "is_msme_vendor": "true" if is_msme else "false",
                "msme_category": msme_cat or "",
                "status": status,
                "description": f"Invoice from {name}",
            })
            invoice_counter += 1

    # Shuffle for realism
    random.shuffle(rows)

    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=list(rows[0].keys()))
    writer.writeheader()
    writer.writerows(rows)
    return output.getvalue()


def get_demo_csv() -> str:
    """Get demo CSV data (cached)."""
    return generate_demo_csv()
