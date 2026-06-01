"""Deprecated legacy compatibility facade for postprocessing imports.

Use :mod:`ivt_filter.postprocessing` for new imports.
"""

from .postprocessing import apply_fixation_postprocessing, merge_short_saccade_blocks

__all__ = [
    "merge_short_saccade_blocks",
    "apply_fixation_postprocessing",
]
