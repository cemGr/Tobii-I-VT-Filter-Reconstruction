"""Shared domain rules and lightweight DataFrame stage contracts."""

from .events import iter_contiguous_events, rebuild_events_from_sample_labels
from .validity import parse_tobii_validity

__all__ = [
    "iter_contiguous_events",
    "parse_tobii_validity",
    "rebuild_events_from_sample_labels",
]
