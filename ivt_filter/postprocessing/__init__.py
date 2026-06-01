"""Postprocessing stage: refinement of classified events."""

from .merge_saccades import merge_short_saccade_blocks
from .pipeline import apply_fixation_postprocessing
from .merge_fixations import merge_adjacent_fixations
from .discard_short_fixations import discard_short_fixations

__all__ = [
    "merge_short_saccade_blocks",
    "apply_fixation_postprocessing",
    "merge_adjacent_fixations",
    "discard_short_fixations",
]
