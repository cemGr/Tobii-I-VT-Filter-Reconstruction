from __future__ import annotations

from pathlib import Path

from ivt_filter.config import IVTClassifierConfig, OlsenVelocityConfig
from ivt_filter.io.pipeline import IVTPipeline


def test_default_tobii_policy_handles_left_v30_w1_1ms_window() -> None:
    path = (
        Path(__file__).parent.parent
        / "test_data"
        / "inputs"
        / "LeftV30W1_extracted.tsv"
    )
    velocity_config = OlsenVelocityConfig(
        window_length_ms=1.0,
        eye_mode="left",
        velocity_method="olsen2d",
        smoothing_mode="none",
    )
    classifier_config = IVTClassifierConfig(velocity_threshold_deg_per_sec=30.0)

    result = IVTPipeline(velocity_config, classifier_config).run(
        str(path),
        classify=True,
        evaluate=False,
        plot=False,
        with_events=False,
    )

    assert result["velocity_deg_per_sec"].notna().sum() > 16_000
    assert (
        result["velocity_window_selector"]
        .value_counts(dropna=False)
        .get("TobiiGazeVelocityWindowSelector", 0)
        > 16_000
    )
    assert result["window_width_samples"].value_counts(dropna=False).get(2, 0) > 16_000
    assert result["ivt_sample_type"].value_counts(dropna=False).get("Unclassified", 0) < 100
