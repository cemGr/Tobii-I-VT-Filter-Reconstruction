"""Compatibility wrappers for the canonical sequential velocity implementation.

This module previously reserved an API for parallel velocity calculation, but no
parallel implementation was completed. Keep the public entry points for callers
that imported them while delegating all calculations to ``compute_olsen_velocity``.
"""
from __future__ import annotations

import warnings
from typing import Optional

import pandas as pd

from ..config import OlsenVelocityConfig
from .velocity import compute_olsen_velocity


def compute_olsen_velocity_parallel(
    df: pd.DataFrame,
    cfg: OlsenVelocityConfig,
    n_jobs: int = -1,
    chunk_size: Optional[int] = None,
) -> pd.DataFrame:
    """Deprecated compatibility wrapper around sequential ``compute_olsen_velocity``.

    No parallel velocity implementation is active. The ``n_jobs`` and
    ``chunk_size`` arguments are retained only for backwards compatibility and
    do not change execution. Use :func:`compute_olsen_velocity` directly for new
    code.

    Args:
        df: Input DataFrame with eye-tracking data.
        cfg: Velocity-computation configuration.
        n_jobs: Ignored compatibility argument.
        chunk_size: Ignored compatibility argument.

    Returns:
        The DataFrame returned by :func:`compute_olsen_velocity`.
    """
    warnings.warn(
        "compute_olsen_velocity_parallel() is deprecated; use "
        "compute_olsen_velocity(). Velocity computation is sequential.",
        DeprecationWarning,
        stacklevel=2,
    )
    return compute_olsen_velocity(df, cfg)


def compute_velocity_auto(
    df: pd.DataFrame,
    cfg: OlsenVelocityConfig,
    parallel: bool = True,
    n_jobs: int = -1,
) -> pd.DataFrame:
    """Compatibility wrapper around sequential ``compute_olsen_velocity``.

    This function does not select a parallel implementation, regardless of the
    DataFrame size. The ``parallel`` and ``n_jobs`` arguments are retained only
    for backwards compatibility and do not change execution. Use
    :func:`compute_olsen_velocity` directly for new code.

    Args:
        df: Input DataFrame with eye-tracking data.
        cfg: Velocity-computation configuration.
        parallel: Ignored compatibility argument.
        n_jobs: Ignored compatibility argument.

    Returns:
        The DataFrame returned by :func:`compute_olsen_velocity`.
    """
    return compute_olsen_velocity(df, cfg)
