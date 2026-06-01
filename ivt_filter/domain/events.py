"""Rules for reconstructing contiguous I-VT events from sample labels."""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass

import pandas as pd

from .schema import require_columns

INDEXED_EVENT_TYPES = frozenset({"Fixation", "Saccade"})


@dataclass(frozen=True)
class ContiguousEvent:
    """One contiguous run of a sample label, using inclusive positions."""

    label: str
    start: int
    end: int


def iter_contiguous_events(labels: Iterable[object]) -> Iterator[ContiguousEvent]:
    """Yield contiguous runs from sample labels, including unclassified runs."""
    iterator = iter(labels)
    try:
        first = next(iterator)
    except StopIteration:
        return

    label = str(first)
    start = 0
    end = 0
    for position, value in enumerate(iterator, start=1):
        end = position
        next_label = str(value)
        if next_label != label:
            yield ContiguousEvent(label, start, position - 1)
            label = next_label
            start = position
    yield ContiguousEvent(label, start, end)


def assign_event_indices(
    labels: Sequence[object],
) -> tuple[list[str], list[int | None]]:
    """Assign sequential indices to Fixation/Saccade runs.

    Non-event labels such as ``Unclassified`` and ``EyesNotFound`` retain their
    label, receive no index, and reset contiguity for the next indexed event.
    """
    event_types: list[str] = []
    event_indices: list[int | None] = []
    next_index = 0

    for event in iter_contiguous_events(labels):
        length = event.end - event.start + 1
        event_types.extend([event.label] * length)
        if event.label in INDEXED_EVENT_TYPES:
            next_index += 1
            event_indices.extend([next_index] * length)
        else:
            event_indices.extend([None] * length)

    return event_types, event_indices


def rebuild_events_from_sample_labels(
    df: pd.DataFrame,
    sample_col: str,
    event_type_col: str,
    event_index_col: str,
) -> pd.DataFrame:
    """Populate event type and index columns from contiguous sample labels."""
    require_columns(df, (sample_col,), stage="event reconstruction")
    event_types, event_indices = assign_event_indices(df[sample_col].tolist())
    df[event_type_col] = event_types
    df[event_index_col] = event_indices
    return df
