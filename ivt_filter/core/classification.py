"""Backward-compatible imports for the canonical classification implementation.

New code should import from :mod:`ivt_filter.processing.classification` directly.
"""

from ..processing.classification import (
    IVTClassifier,
    SampleValidator,
    VelocityValidator,
    apply_ivt_classifier,
    expand_gt_events_to_samples,
    rebuild_ivt_events_from_sample_types,
)

__all__ = [
    "IVTClassifier",
    "SampleValidator",
    "VelocityValidator",
    "apply_ivt_classifier",
    "expand_gt_events_to_samples",
    "rebuild_ivt_events_from_sample_types",
]
