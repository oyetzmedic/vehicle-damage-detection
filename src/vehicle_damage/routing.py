from __future__ import annotations

import numpy as np


def route_scores(scores, threshold: float = 0.5) -> np.ndarray:
    """Route each detection to auto_accept or needs_review."""
    if not 0 <= threshold <= 1:
        raise ValueError("threshold must be between 0 and 1")
    values = np.asarray(scores, dtype=float)
    return np.where(values >= threshold, "auto_accept", "needs_review")


def review_rate(scores, threshold: float = 0.5) -> float:
    """Return the fraction of candidate detections routed to manual review."""
    values = np.asarray(scores, dtype=float)
    if values.size == 0:
        return 0.0
    routes = route_scores(values, threshold)
    return float(np.mean(routes == "needs_review"))


def threshold_sweep(scores, thresholds) -> list[dict[str, float | int]]:
    """Summarise operational review volume across confidence thresholds."""
    values = np.asarray(scores, dtype=float)
    rows = []
    for threshold in thresholds:
        routes = route_scores(values, float(threshold))
        needs_review = int(np.sum(routes == "needs_review"))
        rows.append(
            {
                "threshold": float(threshold),
                "candidate_detections": int(values.size),
                "needs_review": needs_review,
                "auto_accept": int(values.size - needs_review),
                "review_rate": review_rate(values, float(threshold)),
            }
        )
    return rows
