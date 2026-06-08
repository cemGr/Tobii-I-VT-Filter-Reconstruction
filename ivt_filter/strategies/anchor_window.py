# ivt_filter/strategies/anchor_window.py
"""
Anchor window strategies for the I-VT velocity calculator.

The velocity calculator requires three sample roles for each window:
  S_first -- gaze position start
  S_mid   -- eye position used for 3D angle calculation
  S_last  -- gaze position end
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class AnchorRoles:
    """Indices of the three sample roles within a velocity window."""

    first: int
    mid: int
    last: int


class AnchorWindowStrategy(ABC):
    """Abstract base class for anchor window strategies."""

    @abstractmethod
    def get_roles(self, anchor: int, N: int) -> AnchorRoles:
        """Return sample role indices for the given anchor and window size.

        Args:
            anchor: Centre sample index in the full data array.
            N: Total number of samples in the velocity window (>= 2).

        Returns:
            AnchorRoles with absolute sample indices.
        """
        ...


class SymmetricHalf(AnchorWindowStrategy):
    """Symmetric half-window strategy.

    Computes ``half = (N − 1) // 2`` and places first and last symmetrically
    around the anchor.  For even N (including N=2) this produces a degenerate
    window where ``first == last``.

    Roles::

        half  = (N - 1) // 2
        first = anchor - half
        mid   = anchor
        last  = anchor + half
    """

    def get_roles(self, anchor: int, N: int) -> AnchorRoles:
        half = (N - 1) // 2
        return AnchorRoles(first=anchor - half, mid=anchor, last=anchor + half)


class MidIndex(AnchorWindowStrategy):
    """Mid-index window strategy — the correct Tobii reconstruction.

    Computes ``mid_pos = N // 2``, offsets first by mid_pos before the anchor,
    and places last at first + N − 1.  For N=2 this yields a valid two-sample
    window; for N=3 it agrees with :class:`SymmetricHalf`.

    Roles::

        mid_pos = N // 2
        first   = anchor - mid_pos
        mid     = anchor
        last    = first + N - 1
    """

    def get_roles(self, anchor: int, N: int) -> AnchorRoles:
        mid_pos = N // 2
        first = anchor - mid_pos
        last = first + N - 1
        return AnchorRoles(first=first, mid=anchor, last=last)


def compute_window_samples(
    window_us: float,
    avg_dt_us: float,
    tolerance: float = 1.0,
) -> int:
    """Compute the velocity window size in samples.

    Formula::

        window_samples = max(1, int(window_us / avg_dt_us * tolerance)) + 1

    The result is always >= 2.

    Args:
        window_us: Velocity window duration in microseconds (> 0).
        avg_dt_us: Mean sample interval in microseconds (> 0).
        tolerance: Sampling irregularity factor; must be in [1.0, 1.05].
            Default is 1.0 (no tolerance adjustment).

    Returns:
        Integer window size >= 2.

    Raises:
        ValueError: If tolerance is outside [1.0, 1.05], avg_dt_us <= 0, or
            window_us <= 0.
    """
    if not (1.0 <= tolerance <= 1.05):
        raise ValueError(
            f"tolerance must be in [1.0, 1.05], got {tolerance!r}."
        )
    if avg_dt_us <= 0:
        raise ValueError(f"avg_dt_us must be > 0, got {avg_dt_us!r}.")
    if window_us <= 0:
        raise ValueError(f"window_us must be > 0, got {window_us!r}.")

    return max(1, int(window_us / avg_dt_us * tolerance)) + 1


def estimate_avg_dt_us(
    timestamps_us: np.ndarray,
    n_samples: int = 100,
) -> float:
    """Estimate mean sample interval from the first *n_samples* timestamps.

    Args:
        timestamps_us: 1-D array of timestamps in microseconds (monotonically
            increasing expected).
        n_samples: Maximum number of leading samples to use (default 100).

    Returns:
        Mean of consecutive differences in microseconds.

    Raises:
        ValueError: If fewer than 2 timestamps are provided.
    """
    ts = np.asarray(timestamps_us, dtype=float)
    n = min(n_samples, len(ts))
    if n < 2:
        raise ValueError(
            f"At least 2 timestamps are required to estimate avg_dt_us, "
            f"got {len(ts)}."
        )
    return float(np.mean(np.diff(ts[:n])))
