"""Small dependency-free statistics helpers used by experiment scripts."""

from __future__ import annotations

import math
import statistics


def percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    position = (len(ordered) - 1) * q
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(ordered[lower])
    weight = position - lower
    return float(ordered[lower] * (1 - weight) + ordered[upper] * weight)


def summarize(values: list[float]) -> dict[str, float]:
    if not values:
        return {
            'count': 0,
            'mean': 0.0,
            'median': 0.0,
            'p95': 0.0,
            'std': 0.0,
            'min': 0.0,
            'max': 0.0,
        }
    return {
        'count': len(values),
        'mean': float(statistics.fmean(values)),
        'median': float(statistics.median(values)),
        'p95': percentile(values, 0.95),
        'std': float(statistics.pstdev(values)),
        'min': float(min(values)),
        'max': float(max(values)),
    }
