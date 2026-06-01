"""Shared timestamp-based duration rules for reconstructed I-VT events."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np

DEFAULT_SAMPLE_INTERVAL_MS = 1.0


def estimate_sample_interval_ms(
    timestamps_ms: Iterable[float],
    *,
    fallback_ms: float = DEFAULT_SAMPLE_INTERVAL_MS,
) -> float:
    """Return the robust sample interval used by event-duration calculations.

    The project convention is the median of finite, positive timestamp
    differences.  A fallback keeps single-sample inputs and degenerate timelines
    measurable; it preserves the historical 1 ms behavior by default.
    """
    timestamps = np.asarray(list(timestamps_ms), dtype=float)
    diffs = np.diff(timestamps)
    positive_diffs = diffs[np.isfinite(diffs) & (diffs > 0)]
    if len(positive_diffs) == 0:
        return float(fallback_ms)
    return float(np.median(positive_diffs))


def event_duration_ms(
    timestamps_ms: Iterable[float],
    *,
    sample_interval_ms: float | None = None,
) -> float:
    """Calculate an event duration using the inclusive-end-sample convention.

    Event samples represent intervals, not zero-width points.  Duration is
    therefore ``(last_timestamp - first_timestamp) + sample_interval_ms``.  Pass
    an interval estimated from the complete recording when measuring a slice so
    single-sample events use the same robust interval as neighboring events.
    """
    timestamps = np.asarray(list(timestamps_ms), dtype=float)
    if len(timestamps) == 0:
        return 0.0
    if sample_interval_ms is None:
        sample_interval_ms = estimate_sample_interval_ms(timestamps)
    return float(timestamps[-1] - timestamps[0] + sample_interval_ms)
