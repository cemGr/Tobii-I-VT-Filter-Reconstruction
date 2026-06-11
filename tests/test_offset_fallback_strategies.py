"""WS-B(a): selectable Average-mode offset-fallback strategy.

The reconstruction's Average-mode handling of a mono interval is a documented
reconstruction assumption. ``last_offset`` (the default, unchanged) estimates the
missing eye from the last inter-eye offset; ``single_eye_direct`` uses the valid
eye directly. Making this a config enum keeps the assumption auditable and the
A/B comparison reproducible on the occluded-eye conditions.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ivt_filter.config import OlsenVelocityConfig
from ivt_filter.preprocessing.eye_selection import apply_tobii_eye_offset_interpolation

OFFSET = 10.0  # right eye sits +10 mm from the left while both are valid


def _mono_transition_frame(n: int = 10) -> pd.DataFrame:
    """Both eyes valid for the first half (offset +10), right eye missing after."""
    lx = np.full(n, 100.0)
    rx = np.full(n, 100.0 + OFFSET)
    validity_right = np.zeros(n, dtype=int)
    half = n // 2
    rx[half:] = np.nan          # right eye missing in the second half
    validity_right[half:] = 2   # invalid (> max_validity default 1)
    return pd.DataFrame(
        {
            "gaze_left_x_mm": lx,
            "gaze_left_y_mm": np.full(n, 200.0),
            "gaze_right_x_mm": rx,
            "gaze_right_y_mm": np.where(np.isnan(rx), np.nan, 200.0),
            "validity_left": np.zeros(n, dtype=int),
            "validity_right": validity_right,
        }
    )


def _cfg(strategy: str) -> OlsenVelocityConfig:
    return OlsenVelocityConfig(
        eye_mode="average",
        tobii_eye_offset_interpolation=True,
        offset_fallback_strategy=strategy,
    )


def test_last_offset_is_default() -> None:
    assert OlsenVelocityConfig().offset_fallback_strategy == "last_offset"


def test_last_offset_estimates_missing_eye_from_offset() -> None:
    df = _mono_transition_frame()
    out = apply_tobii_eye_offset_interpolation(df, _cfg("last_offset"))
    # right eye estimated as left + stored offset
    assert out.loc[5, "gaze_right_x_mm"] == pytest.approx(100.0 + OFFSET)


def test_single_eye_direct_uses_valid_eye() -> None:
    df = _mono_transition_frame()
    out = apply_tobii_eye_offset_interpolation(df, _cfg("single_eye_direct"))
    # right eye copied from the valid left eye (no parallax offset applied)
    assert out.loc[5, "gaze_right_x_mm"] == pytest.approx(100.0)


def test_strategy_invalid_value_raises() -> None:
    with pytest.raises(ValueError):
        OlsenVelocityConfig(offset_fallback_strategy="nonsense")
