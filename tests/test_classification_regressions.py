"""Regression tests for critical I-VT classification behavior."""

import math

import numpy as np
import pandas as pd
import pytest

from ivt_filter.config import IVTClassifierConfig
from ivt_filter.processing.classification import IVTClassifier, apply_ivt_classifier
from ivt_filter.processing.velocity import VelocitySampleComputer
from ivt_filter.strategies.velocity_calculation import (
    Ray3DGazeDir,
    TobiiGazeDirAngle,
    VelocityContext,
)


def _direction_context(dir1, dir2) -> VelocityContext:
    return VelocityContext(
        x1_mm=0.0,
        y1_mm=0.0,
        x2_mm=0.0,
        y2_mm=0.0,
        eye_x_mm=None,
        eye_y_mm=None,
        eye_z_mm=None,
        dir1=dir1,
        dir2=dir2,
    )


def test_ivt_classifier_uses_classifier_config_by_default() -> None:
    classifier = IVTClassifier()

    assert isinstance(classifier.cfg, IVTClassifierConfig)


def test_apply_ivt_classifier_uses_default_config_and_classifies_velocities() -> None:
    df = pd.DataFrame({"velocity_deg_per_sec": [1.0, 100.0, np.nan]})

    result = apply_ivt_classifier(df)

    assert result["ivt_sample_type"].tolist() == [
        "Fixation",
        "Saccade",
        "Unclassified",
    ]


@pytest.mark.parametrize("strategy", [Ray3DGazeDir(), TobiiGazeDirAngle()])
@pytest.mark.parametrize(
    ("dir1", "dir2"),
    [
        (None, (1.0, 0.0, 0.0)),
        ((1.0, 0.0, 0.0), None),
        ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0)),
        ((np.nan, 0.0, 0.0), (1.0, 0.0, 0.0)),
        ((np.inf, 0.0, 0.0), (1.0, 0.0, 0.0)),
    ],
    ids=["dir1-missing", "dir2-missing", "zero-vector", "nan-vector", "inf-vector"],
)
def test_direction_strategy_returns_nan_for_uncomputable_angle(
    strategy, dir1, dir2
) -> None:
    angle = strategy.calculate_visual_angle_ctx(_direction_context(dir1, dir2))

    assert math.isnan(angle)


def test_missing_direction_produces_nan_velocity_and_unclassified_sample() -> None:
    sample = VelocitySampleComputer.compute_sample(
        _direction_context(None, (1.0, 0.0, 0.0)),
        dt_ms=10.0,
        strategy=Ray3DGazeDir(),
    )

    result = apply_ivt_classifier(
        pd.DataFrame({"velocity_deg_per_sec": [sample.velocity_deg_per_sec]})
    )

    assert math.isnan(sample.velocity_deg_per_sec)
    assert result.at[0, "ivt_sample_type"] == "Unclassified"
