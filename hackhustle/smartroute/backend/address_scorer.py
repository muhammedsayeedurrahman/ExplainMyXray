"""
SmartRoute 2.0 - AddressIQ Scoring Engine
Rule-based heuristic model that scores delivery addresses 0-100.
"""

import re

BENGALURU_AREAS = [
    "koramangala", "indiranagar", "hsr layout", "whitefield", "electronic city",
    "jayanagar", "jp nagar", "marathahalli", "btm layout", "hebbal",
    "yelahanka", "rajajinagar", "malleshwaram", "banashankari", "basavanagudi",
    "sadashivanagar", "rt nagar", "banaswadi", "frazer town", "cox town",
    "bellandur", "sarjapur road", "sarjapur", "domlur", "ulsoor", "mg road",
    "brigade road", "church street", "lavelle road", "richmond town", "shivajinagar",
]

LANDMARK_KEYWORDS = [
    "near", "opposite", "behind", "beside", "next to", "close to",
    "temple", "church", "mosque", "school", "college", "hospital",
    "park", "ground", "lake", "bus stop", "metro", "signal",
    "petrol pump", "community hall",
]

PIN_RANGE = (560001, 560117)

HOUSE_NUMBER_PATTERN = re.compile(
    r'\b(no\.?\s*\d+|#\d+|flat\s*\d+[a-z]?|house\s*\d+|\d+[a-z]?\s*,)', re.IGNORECASE
)

STREET_PATTERN = re.compile(
    r'\b(\d+\w*\s*(st|nd|rd|th)\s*(main|cross)'
    r'|main\s*road|cross\s*road|street|road|lane|avenue|marg'
    r'|feet\s*road|ring\s*road'
    r'|\d+\s*feet)', re.IGNORECASE
)


def score_address(address: str, pin_code: str = "") -> dict:
    """
    Score an address from 0-100 based on completeness and quality.
    Returns score, risk level, and human-readable explanation.
    """
    score = 0
    reasons = []
    addr_lower = address.lower().strip()

    # 1. House/flat number (+20)
    if HOUSE_NUMBER_PATTERN.search(address):
        score += 20
    else:
        reasons.append("Missing house/flat number")

    # 2. Street/road name (+15)
    if STREET_PATTERN.search(address):
        score += 15
    else:
        reasons.append("No street or road name found")

    # 3. Recognized area/locality (+15)
    area_found = any(area in addr_lower for area in BENGALURU_AREAS)
    if area_found:
        score += 15
    else:
        reasons.append("Area/locality not recognized")

    # 4. City name (+10)
    if re.search(r'\b(bangalore|bengaluru)\b', addr_lower):
        score += 10
    else:
        reasons.append("City name missing")

    # 5. PIN code validation (+20 valid, -10 invalid)
    if pin_code and pin_code.strip():
        try:
            pin_int = int(pin_code.strip())
            if PIN_RANGE[0] <= pin_int <= PIN_RANGE[1]:
                score += 20
            else:
                score -= 10
                reasons.append(f"PIN {pin_code} is outside Bengaluru range")
        except ValueError:
            score -= 10
            reasons.append("PIN code is not a valid number")
    else:
        reasons.append("PIN code missing")

    # 6. Landmark dependency penalty
    landmarks_found = [kw for kw in LANDMARK_KEYWORDS if kw in addr_lower]
    if landmarks_found:
        penalty = min(len(landmarks_found) * 5, 15)
        score -= penalty
        reasons.append(f"Relies on landmarks: {', '.join(landmarks_found[:3])}")

    # 7. Address length check
    if len(address.strip()) < 15:
        score -= 10
        reasons.append("Address is too short/vague")
    elif len(address.strip()) > 50:
        score += 10

    # Clamp to 0-100
    score = max(0, min(100, score))

    # Risk classification
    if score >= 80:
        risk = "green"
    elif score >= 50:
        risk = "yellow"
    else:
        risk = "red"

    explanation = "; ".join(reasons) if reasons else "Address is complete and well-structured"

    return {
        "score": score,
        "risk": risk,
        "explanation": explanation,
    }


def simulate_correction(address: str, pin_code: str = "") -> dict:
    """Simulate address correction by adding missing components."""
    corrections = []
    addr_lower = address.lower()

    corrected_address = address.strip()

    if not HOUSE_NUMBER_PATTERN.search(address):
        corrected_address = f"#42, {corrected_address}"
        corrections.append("Added house number")

    if not STREET_PATTERN.search(address):
        corrected_address = corrected_address.rstrip(",. ") + ", 5th Main Road"
        corrections.append("Added street name")

    if not re.search(r'\b(bangalore|bengaluru)\b', addr_lower):
        corrected_address = corrected_address.rstrip(",. ") + ", Bengaluru"
        corrections.append("Added city name")

    corrected_pin = pin_code
    if not pin_code or not pin_code.strip():
        corrected_pin = "560001"
        corrections.append("Added default PIN code")
    else:
        try:
            pin_int = int(pin_code.strip())
            if not (PIN_RANGE[0] <= pin_int <= PIN_RANGE[1]):
                corrected_pin = "560001"
                corrections.append(f"Corrected PIN from {pin_code} to 560001")
        except ValueError:
            corrected_pin = "560001"
            corrections.append("Fixed invalid PIN format")

    new_score = score_address(corrected_address, corrected_pin)

    return {
        "original_address": address,
        "corrected_address": corrected_address,
        "corrected_pin": corrected_pin,
        "corrections": corrections,
        **new_score,
    }
