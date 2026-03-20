"""
SmartRoute 2.0 - FastAPI Backend
AI-Powered Logistics Intelligence Platform
"""

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from data_generator import generate_orders
from address_scorer import score_address, simulate_correction
from clustering import cluster_orders

# Business impact constants
COST_PER_TRIP_INR: int = 120
FAILURE_REDUCTION_FACTOR: float = 0.30  # 70% fewer failures with AI verification

# India logistics industry benchmarks
INDUSTRY_BENCHMARKS = {
    "logistics_gdp_pct": 13.5,
    "global_logistics_gdp_pct": 8.0,
    "last_mile_cost_pct": 53,
    "rto_rate_tier2_3": 22.5,
    "failed_delivery_cost_inr": 200,
    "daily_ecommerce_deliveries": "10M+",
    "d2c_brands_count": "50,000+",
    "warehouse_waste_pct": 35,
}

# Module-level state (single uvicorn worker)
_scored_orders: list[dict] = []
_cluster_cache: dict | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Generate and pre-score orders at startup, cache cluster result."""
    global _scored_orders, _cluster_cache
    orders = generate_orders(400)
    _scored_orders = [
        {**order, **score_address(order["address"], order["pin_code"])}
        for order in orders
    ]
    _cluster_cache = cluster_orders(_scored_orders)
    yield
    _scored_orders = []
    _cluster_cache = None


app = FastAPI(title="SmartRoute 2.0", version="2.0.0", lifespan=lifespan)

import os as _os
_ALLOWED_ORIGINS = [
    "http://localhost:5173",
    "http://localhost:5174",
    "http://localhost:5175",
    "http://localhost:3000",
]
# Allow deployed frontend origins via env var (comma-separated)
_extra = _os.environ.get("ALLOWED_ORIGINS", "")
if _extra:
    _ALLOWED_ORIGINS.extend(o.strip() for o in _extra.split(",") if o.strip())

app.add_middleware(
    CORSMiddleware,
    allow_origins=_ALLOWED_ORIGINS,
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)


class AddressRequest(BaseModel):
    address: str = Field(..., min_length=1, max_length=500)
    pin_code: str = Field(default="", max_length=6, pattern=r"^\d{0,6}$")


@app.get("/")
def root():
    return {"message": "SmartRoute 2.0 API", "version": "2.0.0"}


@app.get("/get_orders")
def get_orders():
    return {"orders": _scored_orders, "total": len(_scored_orders)}


@app.post("/score_address")
def score_single_address(req: AddressRequest):
    return score_address(req.address, req.pin_code)


@app.post("/verify_address")
def verify_address(req: AddressRequest):
    return simulate_correction(req.address, req.pin_code)


@app.get("/cluster_orders")
def get_clusters():
    return _cluster_cache if _cluster_cache is not None else cluster_orders(_scored_orders)


@app.get("/metrics")
def get_metrics():
    """Return business impact metrics with derived calculations."""
    total = len(_scored_orders)
    green = sum(1 for o in _scored_orders if o["risk"] == "green")
    yellow = sum(1 for o in _scored_orders if o["risk"] == "yellow")
    red = sum(1 for o in _scored_orders if o["risk"] == "red")

    cluster_data = _cluster_cache if _cluster_cache is not None else cluster_orders(_scored_orders)

    avg_score = sum(o["score"] for o in _scored_orders) / total if total else 0

    failure_rate_before = red / total if total else 0
    failure_rate_after = failure_rate_before * FAILURE_REDUCTION_FACTOR

    trips_before = cluster_data["trips_before"]
    trips_after = cluster_data["trips_after"]
    trips_saved = cluster_data["trips_saved"]

    # Derived metrics (percentage)
    trip_efficiency = round((trips_saved / trips_before * 100), 1) if trips_before else 0
    address_verification = round((green / total * 100), 1) if total else 0
    failure_reduction = round(((failure_rate_before - failure_rate_after) / failure_rate_before * 100), 1) if failure_rate_before else 0

    return {
        "total_orders": total,
        "score_distribution": {"green": green, "yellow": yellow, "red": red},
        "average_score": round(avg_score, 1),
        "clusters_formed": len(cluster_data["clusters"]),
        "trips_before": trips_before,
        "trips_after": trips_after,
        "trips_saved": trips_saved,
        "cost_saved": trips_saved * COST_PER_TRIP_INR,
        "failure_rate_before": round(failure_rate_before * 100, 1),
        "failure_rate_after": round(failure_rate_after * 100, 1),
        "deliveries_at_risk": red,
        "verified_addresses": green,
        # Derived percentages
        "trip_efficiency": trip_efficiency,
        "address_verification": address_verification,
        "failure_reduction": failure_reduction,
        "industry_benchmarks": INDUSTRY_BENCHMARKS,
    }


@app.get("/insights")
def get_insights():
    """Generate AI-driven insights from delivery data patterns."""
    total = len(_scored_orders)
    if not total:
        return {"insights": []}

    green = sum(1 for o in _scored_orders if o["risk"] == "green")
    red = sum(1 for o in _scored_orders if o["risk"] == "red")

    # Analyze common failure reasons
    reason_counts: dict[str, int] = {}
    area_scores: dict[str, list[int]] = {}
    for o in _scored_orders:
        explanation = o.get("explanation", "")
        for reason in explanation.split("; "):
            reason = reason.strip()
            if reason and reason != "Address is complete and well-structured":
                reason_counts[reason] = reason_counts.get(reason, 0) + 1
        addr_lower = o["address"].lower()
        for area_name in [
            "koramangala", "indiranagar", "hsr layout", "whitefield",
            "electronic city", "jayanagar", "jp nagar", "marathahalli",
            "btm layout", "hebbal", "yelahanka", "rajajinagar",
            "malleshwaram", "banashankari", "basavanagudi",
        ]:
            if area_name in addr_lower:
                area_scores.setdefault(area_name, []).append(o["score"])
                break

    insights = []

    # Top failure reason
    if reason_counts:
        top_reason = max(reason_counts, key=reason_counts.get)
        top_count = reason_counts[top_reason]
        insights.append({
            "type": "failure_pattern",
            "icon": "alert",
            "title": "Top Address Issue",
            "text": f'"{top_reason}" affects {top_count} of {total} deliveries ({round(top_count/total*100)}%). Fixing this alone could improve {round(top_count*0.7)} deliveries.',
        })

    # Risk concentration
    if red > 0:
        risk_pct = round(red / total * 100, 1)
        cost_at_risk = red * INDUSTRY_BENCHMARKS["failed_delivery_cost_inr"]
        insights.append({
            "type": "risk_concentration",
            "icon": "warning",
            "title": "Delivery Risk Alert",
            "text": f"{red} orders ({risk_pct}%) are flagged as high-risk. At \u20b9{INDUSTRY_BENCHMARKS['failed_delivery_cost_inr']}/failed attempt, this represents \u20b9{cost_at_risk:,} in potential losses per batch.",
        })

    # Area analysis
    if area_scores:
        best_area = max(area_scores, key=lambda a: sum(area_scores[a]) / len(area_scores[a]))
        best_avg = round(sum(area_scores[best_area]) / len(area_scores[best_area]), 1)
        worst_area = min(area_scores, key=lambda a: sum(area_scores[a]) / len(area_scores[a]))
        worst_avg = round(sum(area_scores[worst_area]) / len(area_scores[worst_area]), 1)
        insights.append({
            "type": "area_analysis",
            "icon": "map",
            "title": "Area Intelligence",
            "text": f"{best_area.title()} has the highest address quality (avg score: {best_avg}), while {worst_area.title()} scores lowest ({worst_avg}). Consider pre-verification for low-scoring areas.",
        })

    # Clustering effectiveness
    cluster_data = _cluster_cache if _cluster_cache is not None else cluster_orders(_scored_orders)
    if cluster_data["trips_saved"] > 0:
        efficiency = round(cluster_data["trips_saved"] / cluster_data["trips_before"] * 100, 1)
        insights.append({
            "type": "optimization",
            "icon": "sparkle",
            "title": "PackBuddy Efficiency",
            "text": f"DBSCAN clustering reduced {cluster_data['trips_before']} individual trips to {cluster_data['trips_after']} optimized routes ({efficiency}% reduction). {cluster_data['total_clustered']} of {cluster_data['total_eligible']} eligible orders are in clusters.",
        })

    # Industry context
    insights.append({
        "type": "industry",
        "icon": "chart",
        "title": "Market Opportunity",
        "text": f"India's logistics cost is {INDUSTRY_BENCHMARKS['logistics_gdp_pct']}% of GDP vs {INDUSTRY_BENCHMARKS['global_logistics_gdp_pct']}% globally. With {INDUSTRY_BENCHMARKS['daily_ecommerce_deliveries']} daily deliveries, even a 5% RTO reduction saves \u20b95-15 Crore/day industry-wide.",
    })

    return {"insights": insights, "total_issues": sum(reason_counts.values())}


if __name__ == "__main__":
    import uvicorn
    port = int(_os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
