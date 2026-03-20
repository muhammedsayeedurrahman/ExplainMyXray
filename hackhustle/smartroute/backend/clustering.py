"""
SmartRoute 2.0 - PackBuddy Clustering Engine
DBSCAN-based spatial clustering for high-confidence deliveries.
"""

import numpy as np
from sklearn.cluster import DBSCAN


def cluster_orders(
    orders: list[dict],
    min_score: int = 80,
    eps_km: float = 1.5,
    min_samples: int = 3,
) -> dict:
    """
    Cluster high-confidence orders using DBSCAN.

    Args:
        orders: List of scored order dicts (must have lat, lng, score).
        min_score: Minimum address score to include in clustering.
        eps_km: Cluster radius in kilometers.
        min_samples: Minimum orders to form a cluster.

    Returns:
        Dictionary with clusters, stats, and filtered order IDs.
    """
    # Filter to high-confidence orders only
    eligible = [o for o in orders if o.get("score", 0) >= min_score]

    if len(eligible) < min_samples:
        return {
            "clusters": [],
            "total_eligible": len(eligible),
            "total_clustered": 0,
            "noise_count": len(eligible),
            "trips_before": len(eligible),
            "trips_after": len(eligible),
            "trips_saved": 0,
        }

    # Extract coordinates
    coords = np.array([[o["lat"], o["lng"]] for o in eligible])

    # Use haversine metric for accurate geographic clustering
    coords_rad = np.radians(coords)
    eps_rad = eps_km / 6371.0  # Earth radius in km

    # Run DBSCAN with haversine distance (expects radians)
    db = DBSCAN(eps=eps_rad, min_samples=min_samples, metric="haversine")
    labels = db.fit_predict(coords_rad)

    # Build cluster info
    cluster_map: dict[int, list] = {}
    noise_orders = []

    for idx, label in enumerate(labels):
        if label == -1:
            noise_orders.append(eligible[idx]["id"])
        else:
            cluster_map.setdefault(int(label), []).append(idx)

    clusters = []
    for cluster_id, indices in sorted(cluster_map.items()):
        cluster_coords = coords[indices]
        center_lat = float(np.mean(cluster_coords[:, 0]))
        center_lng = float(np.mean(cluster_coords[:, 1]))
        order_ids = [eligible[i]["id"] for i in indices]

        clusters.append({
            "cluster_id": cluster_id,
            "center": {"lat": center_lat, "lng": center_lng},
            "size": len(indices),
            "order_ids": order_ids,
            "radius_km": float(np.max(np.sqrt(
                (cluster_coords[:, 0] - center_lat) ** 2
                + ((cluster_coords[:, 1] - center_lng) * np.cos(np.radians(center_lat))) ** 2
            )) * 111),
        })

    total_clustered = sum(c["size"] for c in clusters)

    # Trip estimation: without clustering = 1 trip per order
    # With clustering = 1 trip per cluster + 1 per noise order
    trips_before = len(eligible)
    trips_after = len(clusters) + len(noise_orders)

    return {
        "clusters": clusters,
        "total_eligible": len(eligible),
        "total_clustered": total_clustered,
        "noise_count": len(noise_orders),
        "trips_before": trips_before,
        "trips_after": trips_after,
        "trips_saved": trips_before - trips_after,
    }
