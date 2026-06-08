from __future__ import annotations

import numpy as np
import pandas as pd

from ivt_filter.config import IVTClassifierConfig, OlsenVelocityConfig
from ivt_filter.io.pipeline import IVTPipeline


def _synthetic_120hz_fixation(n_samples: int = 240) -> pd.DataFrame:
    """Create repository-local input data for the default Tobii window policy."""
    dt_ms = 1000.0 / 120.0
    return pd.DataFrame(
        {
            "time_ms": np.arange(n_samples) * dt_ms,
            "gaze_left_x_mm": np.linspace(100.0, 104.0, n_samples),
            "gaze_left_y_mm": np.full(n_samples, 200.0),
            "gaze_right_x_mm": np.linspace(100.5, 104.5, n_samples),
            "gaze_right_y_mm": np.full(n_samples, 200.5),
            "validity_left": ["Valid"] * n_samples,
            "validity_right": ["Valid"] * n_samples,
            "eye_left_z_mm": np.full(n_samples, 600.0),
            "eye_right_z_mm": np.full(n_samples, 600.0),
        }
    )


def test_default_tobii_policy_handles_120hz_1ms_window(tmp_path) -> None:
    input_path = tmp_path / "synthetic_120hz.tsv"
    _synthetic_120hz_fixation().to_csv(input_path, sep="\t", index=False)

    velocity_config = OlsenVelocityConfig(
        window_length_ms=1.0,
        eye_mode="left",
        velocity_method="olsen2d",
        smoothing_mode="none",
    )
    classifier_config = IVTClassifierConfig(velocity_threshold_deg_per_sec=30.0)

    result = IVTPipeline(velocity_config, classifier_config).run(
        str(input_path),
        classify=True,
        evaluate=False,
        plot=False,
        with_events=False,
    )

    assert result["velocity_deg_per_sec"].notna().sum() >= 200
    assert (
        result["velocity_window_selector"]
        .value_counts(dropna=False)
        .get("TobiiGazeVelocityWindowSelector", 0)
        >= 200
    )
    assert result["window_width_samples"].value_counts(dropna=False).get(2, 0) >= 200
    assert result["ivt_sample_type"].value_counts(dropna=False).get("Unclassified", 0) < 10
