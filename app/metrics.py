"""Shared probability evaluation metrics."""

from __future__ import annotations

def _expected_calibration_error(pairs: list[tuple[float, float]]) -> float | None:
    """Return ten-bin expected calibration error for forecast/actual pairs."""
    if not pairs:
        return None
    error = 0.0
    for bin_index in range(10):
        bucket = [
            (probability, actual)
            for probability, actual in pairs
            if min(9, int(probability * 10)) == bin_index
        ]
        if bucket:
            error += len(bucket) / len(pairs) * abs(
                sum(probability for probability, _ in bucket) / len(bucket)
                - sum(actual for _, actual in bucket) / len(bucket)
            )
    return error
