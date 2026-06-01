"""Coverage increments for timestamp, classifier-boundary, and experiment behavior."""

from __future__ import annotations

from datetime import datetime

import numpy as np
import pandas as pd
import pytest

from ivt_filter.config import IVTClassifierConfig, OlsenVelocityConfig
from ivt_filter.evaluation.experiment import ExperimentConfig, ExperimentManager
from ivt_filter.processing.classification import apply_ivt_classifier
from ivt_filter.processing.velocity_input import normalize_timestamps


def test_normalize_timestamps_converts_nanoseconds_sorts_copy_and_preserves_source() -> None:
    source = pd.DataFrame({"clock_ns": [2_000_000.0, 1_000_000.0], "value": [2, 1]})

    result = normalize_timestamps(
        source, OlsenVelocityConfig(time_column="clock_ns", time_unit="ns")
    )

    assert result[["time_ms", "value"]].values.tolist() == [[1.0, 1.0], [2.0, 2.0]]
    assert "time_ms" not in source.columns


def test_normalize_timestamps_rejects_missing_configured_column() -> None:
    with pytest.raises(ValueError, match="clock_us"):
        normalize_timestamps(pd.DataFrame({"time_ms": [1.0]}), OlsenVelocityConfig(time_column="clock_us"))


def test_classifier_threshold_is_inclusive_and_invalid_window_spike_needs_support() -> None:
    cfg = IVTClassifierConfig(velocity_threshold_deg_per_sec=30.0)
    source = pd.DataFrame(
        {
            "velocity_deg_per_sec": [29.999, 30.0, 100.0, 30.0],
            "window_any_invalid": [False, False, True, True],
        }
    )

    result = apply_ivt_classifier(source, cfg)

    assert result["ivt_sample_type"].tolist() == ["Fixation", "Saccade", "Saccade", "Saccade"]
    assert result["velocity_neighbor_support"].tolist() == [False, False, True, True]


def test_classifier_invalid_window_rejects_isolated_velocity_spike() -> None:
    result = apply_ivt_classifier(
        pd.DataFrame(
            {
                "velocity_deg_per_sec": [1.0, 100.0, 1.0],
                "window_any_invalid": [False, True, False],
            }
        )
    )

    assert result["ivt_sample_type"].tolist() == ["Fixation", "Fixation", "Fixation"]


def _experiment(name: str, timestamp: datetime) -> ExperimentConfig:
    return ExperimentConfig(
        name=name,
        description=f"Experiment {name}",
        velocity_config=OlsenVelocityConfig(window_length_ms=20.0),
        classifier_config=IVTClassifierConfig(velocity_threshold_deg_per_sec=30.0),
        timestamp=timestamp,
        tags=["tracked"],
    )


def test_experiment_manager_round_trips_results_metrics_and_best_configuration(tmp_path) -> None:
    manager = ExperimentManager(str(tmp_path / "experiments"))
    older = _experiment("older", datetime(2025, 1, 1, 12, 0, 0))
    newer = _experiment("newer", datetime(2025, 1, 2, 12, 0, 0))
    manager.save_experiment(older, metrics={"score": np.float64(0.5)})
    manager.save_experiment(newer, pd.DataFrame({"velocity": [1.25]}), {"score": 0.75})

    loaded_config, loaded_results, loaded_metrics = manager.load_experiment("newer")

    assert loaded_config.to_dict() == newer.to_dict()
    assert loaded_results is not None
    assert loaded_results["velocity"].tolist() == [1.25]
    assert loaded_metrics == {"score": 0.75}
    assert [entry["name"] for entry in manager.list_experiments(tags=["tracked"])] == ["newer", "older"]
    best_name, best_score, best_config = manager.get_best_configuration(metric="score")
    assert (best_name, best_score, best_config.name) == ("newer", 0.75, "newer")


def test_experiment_manager_reports_missing_experiment_and_metric(tmp_path) -> None:
    manager = ExperimentManager(str(tmp_path / "experiments"))

    with pytest.raises(ValueError, match="not found"):
        manager.load_experiment("missing")
    with pytest.raises(ValueError, match="No experiments found"):
        manager.get_best_configuration(metric="missing")
