"""Postprocessing stage: refinement of classified events."""

from .merge_fixations import merge_adjacent_fixations
from .discard_short_fixations import discard_short_fixations

__all__ = [
    'merge_adjacent_fixations',
    'discard_short_fixations',
]
