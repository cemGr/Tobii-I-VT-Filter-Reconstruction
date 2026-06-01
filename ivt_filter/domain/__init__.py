"""Shared domain rules and lightweight DataFrame stage contracts."""

from .durations import event_duration_ms, estimate_sample_interval_ms
from .events import iter_contiguous_events, rebuild_events_from_sample_labels
from .validity import parse_tobii_validity

__all__ = [
    "estimate_sample_interval_ms",
    "event_duration_ms",
    "iter_contiguous_events",
    "parse_tobii_validity",
    "rebuild_events_from_sample_labels",
]
