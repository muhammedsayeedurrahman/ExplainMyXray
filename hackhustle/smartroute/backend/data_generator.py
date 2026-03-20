"""
SmartRoute 2.0 - Synthetic Delivery Data Generator
Generates 400 realistic delivery orders across Bengaluru with varying address quality.
"""

import random
import string
from datetime import datetime, timedelta, timezone

# Bengaluru center and bounds
CITY_CENTER = (12.9716, 77.5946)
LAT_RANGE = (12.85, 13.08)
LNG_RANGE = (77.48, 77.72)

AREAS = [
    "Koramangala", "Indiranagar", "HSR Layout", "Whitefield", "Electronic City",
    "Jayanagar", "JP Nagar", "Marathahalli", "BTM Layout", "Hebbal",
    "Yelahanka", "Rajajinagar", "Malleshwaram", "Banashankari", "Basavanagudi",
    "Sadashivanagar", "RT Nagar", "Banaswadi", "Frazer Town", "Cox Town",
    "Bellandur", "Sarjapur Road", "Domlur", "Ulsoor", "MG Road",
    "Brigade Road", "Church Street", "Lavelle Road", "Richmond Town", "Shivajinagar",
]

STREETS = [
    "1st Main Road", "2nd Cross", "3rd Main", "4th Cross Road", "5th Main Road",
    "6th Cross", "7th Main", "8th Cross Road", "80 Feet Road", "100 Feet Road",
    "CMH Road", "Old Airport Road", "Outer Ring Road", "Hosur Road", "Mysore Road",
    "Bellary Road", "MG Road", "Brigade Road", "Residency Road", "Infantry Road",
]

BUILDINGS = [
    "Prestige Shantiniketan", "Salarpuria Sattva", "Brigade Gateway", "Mantri Square",
    "Sobha Dream Acres", "Purva Fountain Square", "Embassy Golf Links", "Total Mall",
    "Phoenix Marketcity", "Orion Mall", "Gopalan Innovation Mall", "RMZ Ecoworld",
    "Bagmane Tech Park", "Manyata Tech Park", "ITPL", "Divyasree Chambers",
]

LANDMARKS = [
    "near ISKCON Temple", "opposite Jayadeva Hospital", "behind Forum Mall",
    "next to Lalbagh Gate", "near Cubbon Park", "beside Mantri Mall",
    "close to Majestic Bus Stand", "near Silk Board Junction", "opposite BDA Complex",
    "behind Big Bazaar", "near Meenakshi Temple", "next to Government School",
    "close to Metro Station", "near Petrol Pump", "beside Community Hall",
]

VALID_PINS = [
    "560001", "560002", "560003", "560004", "560005", "560008", "560009", "560010",
    "560011", "560017", "560018", "560020", "560025", "560027", "560029", "560030",
    "560034", "560037", "560038", "560040", "560041", "560043", "560047", "560048",
    "560050", "560052", "560054", "560055", "560058", "560060", "560062", "560064",
    "560066", "560068", "560069", "560070", "560071", "560073", "560076", "560078",
    "560080", "560085", "560087", "560093", "560095", "560097", "560100", "560102",
    "560103", "560108", "560110", "560114", "560117",
]

FIRST_NAMES = [
    "Aarav", "Vivaan", "Aditya", "Vihaan", "Arjun", "Sai", "Reyansh", "Ayaan",
    "Krishna", "Ishaan", "Ananya", "Diya", "Myra", "Sara", "Aanya", "Aadhya",
    "Priya", "Neha", "Rahul", "Amit", "Deepa", "Kavya", "Rohan", "Sneha",
    "Vikram", "Pooja", "Karthik", "Meera", "Suresh", "Lakshmi",
]

LAST_NAMES = [
    "Sharma", "Patel", "Reddy", "Kumar", "Singh", "Nair", "Rao", "Gupta",
    "Iyer", "Menon", "Joshi", "Verma", "Pillai", "Das", "Kulkarni", "Hegde",
    "Shetty", "Gowda", "Naidu", "Bhat",
]


def _random_coords_near(area_index: int, total_areas: int) -> tuple[float, float]:
    """Generate coordinates loosely mapped to area index for spatial realism."""
    base_lat = LAT_RANGE[0] + (area_index / total_areas) * (LAT_RANGE[1] - LAT_RANGE[0])
    base_lng = LNG_RANGE[0] + ((area_index * 7) % total_areas / total_areas) * (LNG_RANGE[1] - LNG_RANGE[0])
    lat = base_lat + random.uniform(-0.02, 0.02)
    lng = base_lng + random.uniform(-0.02, 0.02)
    return (
        round(max(LAT_RANGE[0], min(LAT_RANGE[1], lat)), 6),
        round(max(LNG_RANGE[0], min(LNG_RANGE[1], lng)), 6),
    )


def _good_address(area_idx: int) -> tuple[str, str]:
    """Complete address with house number, street, area, city, PIN."""
    house = f"#{random.randint(1, 999)}" if random.random() > 0.3 else f"Flat {random.randint(1, 12)}{random.choice('ABCD')}"
    street = random.choice(STREETS)
    area = AREAS[area_idx % len(AREAS)]
    building = f", {random.choice(BUILDINGS)}" if random.random() > 0.6 else ""
    pin = random.choice(VALID_PINS)
    address = f"{house}, {street}{building}, {area}, Bengaluru"
    return address, pin


def _incomplete_address(area_idx: int) -> tuple[str, str]:
    """Address missing key components."""
    area = AREAS[area_idx % len(AREAS)]
    variants = [
        (f"{area}, Bengaluru", random.choice(VALID_PINS)),
        (f"{random.choice(STREETS)}, Bengaluru", ""),
        (f"{area}", random.choice(VALID_PINS)),
        (f"Bengaluru {random.choice(VALID_PINS)}", random.choice(VALID_PINS)),
    ]
    return random.choice(variants)


def _landmark_address(area_idx: int) -> tuple[str, str]:
    """Address relying on landmarks instead of structured info."""
    area = AREAS[area_idx % len(AREAS)]
    landmark = random.choice(LANDMARKS)
    variants = [
        (f"{landmark}, {area}, Bengaluru", random.choice(VALID_PINS)),
        (f"{landmark}, {area}", ""),
        (f"{random.choice(BUILDINGS)}, {landmark}, Bengaluru", random.choice(VALID_PINS)),
        (f"{landmark}", ""),
    ]
    return random.choice(variants)


_BAD_PINS = ["110001", "400001", "500001", "600001", "700001", "999999", "123456", "000000"]


def _bad_pin_address(area_idx: int) -> tuple[str, str]:
    """Good address structure but with invalid PIN code."""
    house = f"#{random.randint(1, 999)}"
    street = random.choice(STREETS)
    area = AREAS[area_idx % len(AREAS)]
    address = f"{house}, {street}, {area}, Bengaluru"
    return address, random.choice(_BAD_PINS)


def generate_orders(count: int = 400, seed: int = 42) -> list[dict]:
    """Generate synthetic delivery orders with mixed address quality.

    Uses a fixed seed for deterministic, reproducible output across restarts.
    """
    random.seed(seed)
    orders = []
    base_date = datetime.now(tz=timezone.utc) - timedelta(days=7)

    # Distribution: 55% good, 15% incomplete, 15% landmark, 10% bad PIN, 5% overflow to good
    type_weights = [
        (_good_address, 0.55),
        (_incomplete_address, 0.15),
        (_landmark_address, 0.15),
        (_bad_pin_address, 0.10),
    ]

    for i in range(count):
        area_idx = random.randint(0, len(AREAS) - 1)

        # Pick address type
        r = random.random()
        cumulative = 0
        address_fn = _good_address
        for fn, weight in type_weights:
            cumulative += weight
            if r <= cumulative:
                address_fn = fn
                break

        address, pin = address_fn(area_idx)
        lat, lng = _random_coords_near(area_idx, len(AREAS))
        order_date = base_date + timedelta(hours=random.randint(0, 168))

        orders.append({
            "id": f"ORD-{1000 + i}",
            "customer_name": f"{random.choice(FIRST_NAMES)} {random.choice(LAST_NAMES)}",
            "address": address,
            "pin_code": pin,
            "lat": lat,
            "lng": lng,
            "order_date": order_date.isoformat(),
            "status": "pending",
        })

    return orders
