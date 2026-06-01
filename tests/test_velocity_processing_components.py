import math

import numpy as np
import pandas as pd

from ivt_filter.config import OlsenVelocityConfig
from ivt_filter.processing.velocity import (
    AverageNeighborImputer,
    FixedWindowEdgeFallbackContext,
    SamplingAnalyzer,
    VelocityInputArrays,
    VelocitySampleComputer,
    VelocityStrategyFactory,
    WindowSelectorFactory,
    apply_fixed_window_edge_fallback,
    find_single_eye_endpoints,
    normalize_timestamps,
    prepare_velocity_input,
)
from ivt_filter.strategies import (
    FixedSampleSymmetricWindowSelector,
    Olsen2DApproximation,
    VelocityContext,
)


def test_normalize_timestamps_converts_microseconds_and_sorts_copy():
    original = pd.DataFrame({"timestamp": [2000, 1000]})
    cfg = OlsenVelocityConfig(time_column="timestamp", time_unit="us")
    result = normalize_timestamps(original, cfg)
    assert result["time_ms"].tolist() == [1.0, 2.0]
    assert "time_ms" not in original


def test_sampling_analyzer_derives_odd_fixed_sample_window():
    cfg = OlsenVelocityConfig(window_length_ms=20, auto_fixed_window_from_ms=True)
    result = SamplingAnalyzer().analyze(np.array([0.0, 10.0, 20.0, 30.0]), cfg)
    assert result.dt_ms == 10.0
    assert result.hz_measured == 100.0
    assert result.config.fixed_window_samples == 3


def test_neighbor_imputer_uses_closest_valid_missing_eye_sample():
    arrays = VelocityInputArrays(
        times=np.array([0.0, 10.0, 20.0]),
        combined_x=np.array([0.0, 5.0, 10.0]),
        combined_y=np.zeros(3),
        left_valid=np.array([True, True, True]),
        right_valid=np.array([True, False, True]),
        valid=np.ones(3, dtype=bool),
        left_x=np.array([0.0, 10.0, 20.0]),
        left_y=np.zeros(3),
        right_x=np.array([2.0, np.nan, 22.0]),
        right_y=np.zeros(3),
    )
    assert AverageNeighborImputer(arrays).impute(1, 0, 2) == (6.0, 0.0)


def test_find_single_eye_endpoints_limits_search_to_window():
    valid = np.array([True, False, True, True, False, True])
    assert find_single_eye_endpoints(valid, 1, 4) == (2, 3)


def test_velocity_sample_computer_calculates_one_structured_result():
    context = VelocityContext(0, 0, 10, 0, 0, 0, 600)
    result = VelocitySampleComputer.compute_sample(
        context, 10.0, Olsen2DApproximation()
    )
    expected = math.degrees(math.atan(10 / 600)) / 0.01
    assert result.velocity_deg_per_sec == round(expected, 2)
    assert result.dt_ms == 10.0


def test_fixed_window_edge_fallback_copies_nearest_velocity():
    df = pd.DataFrame(
        {
            "velocity_deg_per_sec": [np.nan, np.nan, 12.5, np.nan],
            "gap_rule_triggered": [False] * 4,
        }
    )
    cfg = OlsenVelocityConfig(fixed_window_samples=3, fixed_window_edge_fallback=True)
    selector = WindowSelectorFactory.create(cfg)
    assert isinstance(selector, FixedSampleSymmetricWindowSelector)
    count = apply_fixed_window_edge_fallback(
        df,
        FixedWindowEdgeFallbackContext(
            cfg, selector, np.array([False, True, True, True])
        ),
    )
    assert count == 1
    assert df.at[1, "velocity_deg_per_sec"] == 12.5


def test_prepare_velocity_input_runs_preprocessing_steps_in_order(monkeypatch):
    calls = []

    def record(name):
        def apply(df, cfg):
            calls.append(name)
            return df.assign(**{name: True})

        return apply

    monkeypatch.setattr("ivt_filter.processing.velocity.gap_fill_gaze", record("gap"))
    monkeypatch.setattr(
        "ivt_filter.processing.velocity.prepare_combined_columns", record("combined")
    )
    monkeypatch.setattr(
        "ivt_filter.processing.velocity.smooth_combined_gaze", record("smooth")
    )
    result = prepare_velocity_input(
        pd.DataFrame(
            {
                "time_ms": [0.0],
                "gaze_left_x_mm": [0.0],
                "gaze_left_y_mm": [0.0],
                "gaze_right_x_mm": [0.0],
                "gaze_right_y_mm": [0.0],
                "validity_left": [0],
                "validity_right": [0],
                "combined_x_mm": [0.0],
                "combined_y_mm": [0.0],
                "combined_valid": [True],
                "smoothed_x_mm": [0.0],
                "smoothed_y_mm": [0.0],
            }
        ),
        OlsenVelocityConfig(),
    )
    assert calls == ["gap", "combined", "smooth"]
    assert result.at[0, "smooth"]


def test_velocity_strategy_factory_builds_requested_strategy():
    assert isinstance(VelocityStrategyFactory.create("olsen2d"), Olsen2DApproximation)
