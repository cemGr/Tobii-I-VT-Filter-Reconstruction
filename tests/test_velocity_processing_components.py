import math

import numpy as np
import pandas as pd
import pytest

from extractor import TobiiDataExtractor

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
from ivt_filter.utils.sampling import estimate_sampling_rate


def test_normalize_timestamps_converts_microseconds_and_sorts_copy():
    original = pd.DataFrame({"timestamp": [2000, 1000]})
    cfg = OlsenVelocityConfig(time_column="timestamp", time_unit="us")
    result = normalize_timestamps(original, cfg)
    assert result["time_ms"].tolist() == [1.0, 2.0]
    assert "time_ms" not in original


def test_normalize_timestamps_retains_source_row_provenance_after_sorting():
    result = normalize_timestamps(
        pd.DataFrame({"time_ms": [30, 10, 20], "value": ["third", "first", "second"]}),
        OlsenVelocityConfig(),
    )
    assert result["time_ms"].tolist() == [10.0, 20.0, 30.0]
    assert result["source_row_id"].tolist() == [1, 2, 0]
    assert result["value"].tolist() == ["first", "second", "third"]


def test_normalize_timestamps_preserves_existing_source_row_identifier():
    result = normalize_timestamps(
        pd.DataFrame({"time_ms": [20, 10], "source_row_id": ["b", "a"]}),
        OlsenVelocityConfig(),
    )
    assert result["source_row_id"].tolist() == ["a", "b"]


def test_normalize_timestamps_rejects_duplicate_timestamps():
    with pytest.raises(ValueError, match="duplicate timestamps.*rejected"):
        normalize_timestamps(pd.DataFrame({"time_ms": [10, 10]}), OlsenVelocityConfig())


@pytest.mark.parametrize("invalid_timestamp", ["not-a-number", np.nan, np.inf, -np.inf])
def test_normalize_timestamps_rejects_non_numeric_and_non_finite_values(
    invalid_timestamp,
):
    with pytest.raises(ValueError, match="numeric, finite values"):
        normalize_timestamps(
            pd.DataFrame({"time_ms": [0, invalid_timestamp]}), OlsenVelocityConfig()
        )


def test_extractor_sort_by_time_uses_same_duplicate_rejection_policy():
    with pytest.raises(ValueError, match="duplicate timestamps.*rejected"):
        TobiiDataExtractor()._sort_by_time(pd.DataFrame({"time_ms": [10, 10]}))


def test_sampling_analyzer_derives_odd_fixed_sample_window():
    cfg = OlsenVelocityConfig(window_length_ms=20, auto_fixed_window_from_ms=True)
    result = SamplingAnalyzer().analyze(np.array([0.0, 10.0, 20.0, 30.0]), cfg)
    assert result.dt_ms == 10.0
    assert result.hz_measured == 100.0
    assert result.config.fixed_window_samples == 3


def test_sampling_analyzer_uses_only_finite_strictly_positive_intervals():
    result = SamplingAnalyzer().analyze(
        np.array([0.0, 10.0, 10.0, 5.0, np.inf, 25.0]), OlsenVelocityConfig()
    )
    assert result.dt_ms == 10.0
    assert result.hz_measured == 100.0


def test_sampling_analyzer_rejects_input_without_valid_interval():
    with pytest.raises(ValueError, match="finite, strictly positive timestamp intervals"):
        SamplingAnalyzer().analyze(np.array([10.0, 10.0, 5.0, np.inf]), OlsenVelocityConfig())


def test_sampling_analyzer_rejects_input_without_any_interval():
    with pytest.raises(ValueError, match="finite, strictly positive timestamp intervals"):
        SamplingAnalyzer().analyze(np.array([10.0]), OlsenVelocityConfig())


def test_sampling_utility_uses_only_finite_strictly_positive_intervals():
    result = estimate_sampling_rate(pd.DataFrame({"time_ms": [0, 10, 10, 5, np.inf, 25]}))
    assert result["median_dt_ms"] == 10.0
    assert result["mean_dt_ms"] == 10.0


def test_sampling_utility_rejects_input_without_valid_interval():
    with pytest.raises(ValueError, match="finite, strictly positive timestamp intervals"):
        estimate_sampling_rate(pd.DataFrame({"time_ms": [10, 10, 5, np.inf]}))


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

    monkeypatch.setattr("ivt_filter.processing.velocity_input.gap_fill_gaze", record("gap"))
    monkeypatch.setattr(
        "ivt_filter.processing.velocity_input.prepare_combined_columns", record("combined")
    )
    monkeypatch.setattr(
        "ivt_filter.processing.velocity_input.smooth_combined_gaze", record("smooth")
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


def _minimal_prepared_velocity_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "time_ms": [0.0, 10.0, 20.0],
            "smoothed_x_mm": [0.0, 1.0, 2.0],
            "smoothed_y_mm": [0.0, 0.5, 1.0],
            "eye_x_mm": [0.0, 0.0, 0.0],
            "eye_y_mm": [0.0, 0.0, 0.0],
            "eye_z_mm": [600.0, 600.0, 600.0],
            "left_eye_valid": [True, True, False],
            "right_eye_valid": [True, False, True],
            "combined_valid": [True, True, True],
            "gaze_left_x_mm": [0.0, 1.0, np.nan],
            "gaze_left_y_mm": [0.0, 0.0, np.nan],
            "gaze_right_x_mm": [0.0, np.nan, 2.0],
            "gaze_right_y_mm": [0.0, np.nan, 1.0],
        }
    )


def test_initialize_velocity_columns_returns_copy_with_diagnostics():
    from ivt_filter.processing.velocity import _initialize_velocity_columns

    original = pd.DataFrame({"time_ms": [0.0, 10.0]})
    result = _initialize_velocity_columns(original)

    assert "velocity_deg_per_sec" not in original
    assert result["velocity_deg_per_sec"].isna().all()
    assert result["dt_ms"].isna().all()
    assert result["window_any_invalid"].tolist() == [False, False]
    for column in (
        "velocity_first_idx",
        "velocity_last_idx",
        "velocity_eye_used",
        "velocity_window_selector",
        "velocity_fallback_applied",
        "env_has_invalid_above",
        "env_has_invalid_below",
        "env_rule_triggered",
        "gap_rule_triggered",
        "gap_left_invalid_idx",
        "gap_right_invalid_idx",
    ):
        assert column in result.columns


def test_combine_direction_vectors_uses_valid_eye_fallbacks():
    from ivt_filter.processing.velocity import _combine_direction_vectors

    combined = _combine_direction_vectors(
        np.array([True, True, False, False]),
        np.array([True, False, True, False]),
        np.array([1.0, 2.0, 3.0, 4.0]),
        np.array([10.0, 20.0, 30.0, 40.0]),
        np.array([100.0, 200.0, 300.0, 400.0]),
        np.array([5.0, 6.0, 7.0, 8.0]),
        np.array([50.0, 60.0, 70.0, 80.0]),
        np.array([500.0, 600.0, 700.0, 800.0]),
    )

    np.testing.assert_allclose(combined[0][:3], [3.0, 2.0, 7.0])
    np.testing.assert_allclose(combined[1][:3], [30.0, 20.0, 70.0])
    np.testing.assert_allclose(combined[2][:3], [300.0, 200.0, 700.0])
    assert np.isnan(combined[0][3])
    assert np.isnan(combined[1][3])
    assert np.isnan(combined[2][3])


def test_prepare_velocity_arrays_selects_mode_validity_and_missing_directions():
    from ivt_filter.processing.velocity import _prepare_velocity_arrays

    df = _minimal_prepared_velocity_frame()
    arrays = _prepare_velocity_arrays(df, OlsenVelocityConfig(eye_mode="left"))

    assert arrays.eye_mode == "left"
    np.testing.assert_array_equal(arrays.valid, df["left_eye_valid"].to_numpy())
    assert np.isnan(arrays.directions.left_x).all()
    assert np.isnan(arrays.directions.combined_x).all()


def test_prepare_velocity_arrays_combines_present_direction_columns():
    from ivt_filter.processing.velocity import _prepare_velocity_arrays

    df = _minimal_prepared_velocity_frame().assign(
        gaze_dir_left_x=[1.0, 2.0, 3.0],
        gaze_dir_left_y=[10.0, 20.0, 30.0],
        gaze_dir_left_z=[100.0, 200.0, 300.0],
        gaze_dir_right_x=[5.0, 6.0, 7.0],
        gaze_dir_right_y=[50.0, 60.0, 70.0],
        gaze_dir_right_z=[500.0, 600.0, 700.0],
    )
    arrays = _prepare_velocity_arrays(df, OlsenVelocityConfig())

    np.testing.assert_allclose(arrays.directions.combined_x, [3.0, 2.0, 7.0])
    np.testing.assert_allclose(arrays.directions.combined_y, [30.0, 20.0, 70.0])
    np.testing.assert_allclose(arrays.directions.combined_z, [300.0, 200.0, 700.0])


def test_velocity_computation_context_builds_sampling_window_and_strategy():
    from ivt_filter.processing.velocity import (
        VelocityComputationContext,
        _prepare_velocity_arrays,
    )
    from ivt_filter.strategies import TimeSymmetricWindowSelector

    arrays = _prepare_velocity_arrays(_minimal_prepared_velocity_frame(), OlsenVelocityConfig())
    context = VelocityComputationContext.create(arrays, OlsenVelocityConfig())

    assert context.dt_med == 10.0
    assert context.hz_measured == 100.0
    assert context.half_window == 10.0
    assert isinstance(context.selector, TimeSymmetricWindowSelector)
    assert context.fallback_selector is None
    np.testing.assert_array_equal(context.prev_invalid_idx, [-1, -1, -1])
    np.testing.assert_array_equal(context.next_invalid_idx, [-1, -1, -1])
    assert context.neighbor_imputer.arrays is arrays.input
