"""Coverage increments for timestamp, classifier-boundary, and experiment behavior."""

from __future__ import annotations

from datetime import datetime
import hashlib
import json

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
    cfg = IVTClassifierConfig(
        velocity_threshold_deg_per_sec=30.0,
        enable_invalid_window_neighbor_confirmation=True,
    )
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
        ),
        IVTClassifierConfig(enable_invalid_window_neighbor_confirmation=True),
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
    best_run_id, best_score, best_config = manager.get_best_configuration(metric="score")
    assert best_run_id == manager.list_experiments(names=["newer"])[0]["run_id"]
    assert (best_score, best_config.name) == (0.75, "newer")


def test_experiment_manager_reports_missing_experiment_and_metric(tmp_path) -> None:
    manager = ExperimentManager(str(tmp_path / "experiments"))

    with pytest.raises(ValueError, match="not found"):
        manager.load_experiment("missing")
    with pytest.raises(ValueError, match="No experiments found"):
        manager.get_best_configuration(metric="missing")


def test_experiment_manager_keeps_repeated_names_as_immutable_runs(tmp_path) -> None:
    manager = ExperimentManager(str(tmp_path / "experiments"))
    config = _experiment("repeat", datetime(2025, 1, 1, 12, 0, 0))

    first_path = manager.save_experiment(config, metrics={"score": 0.25})
    second_path = manager.save_experiment(config, metrics={"score": 0.75})

    runs = manager.list_experiments(names=["repeat"])
    assert len(runs) == 2
    assert runs[0]["run_id"] != runs[1]["run_id"]
    assert first_path != second_path
    assert first_path.parent == second_path.parent == tmp_path / "experiments" / "repeat"
    assert manager.load_experiment(runs[0]["run_id"])[2] == {"score": 0.75}
    assert manager.load_experiment(runs[1]["run_id"])[2] == {"score": 0.25}
    assert manager.compare_experiments(["repeat"], ["score"])["score"].tolist() == [0.25, 0.75]


def test_experiment_manager_persists_execution_provenance(tmp_path) -> None:
    input_path = tmp_path / "input.tsv"
    input_path.write_text("time\tvalue\n1\t2\n3\t4\n", encoding="utf-8")
    manager = ExperimentManager(str(tmp_path / "experiments"))

    run_path = manager.save_experiment(
        _experiment("provenance", datetime(2025, 1, 1, 12, 0, 0)),
        input_path=str(input_path),
        command=["ivt-filter", "--input", str(input_path)],
        api_parameters={"threshold": 30.0},
        reference_system_version="Tobii Pro Lab 1.0",
        reference_export_identifier="export-123",
    )

    config, _, _ = manager.load_experiment(run_path.name)
    provenance = config.metadata["provenance"]
    assert provenance["input_file_checksum_sha256"] == hashlib.sha256(input_path.read_bytes()).hexdigest()
    assert provenance["input_row_count"] == 2
    assert provenance["package_version"]
    assert "git_commit" in provenance
    assert provenance["python_version"]
    assert provenance["platform"]
    assert set(provenance["dependency_versions"]) == {"numpy", "pandas"}
    assert provenance["executed_command"] == ["ivt-filter", "--input", str(input_path)]
    assert provenance["api_parameters"] == {"threshold": 30.0}
    assert provenance["reference_system_version"] == "Tobii Pro Lab 1.0"
    assert provenance["reference_export_identifier"] == "export-123"


def test_experiment_manager_migrates_legacy_index_entries(tmp_path) -> None:
    experiments_dir = tmp_path / "experiments"
    legacy_dir = experiments_dir / "legacy"
    legacy_dir.mkdir(parents=True)
    config = _experiment("legacy", datetime(2025, 1, 1, 12, 0, 0))
    (legacy_dir / "config.json").write_text(json.dumps(config.to_dict()), encoding="utf-8")
    (legacy_dir / "metrics.json").write_text('{"score": 0.5}', encoding="utf-8")
    (experiments_dir / "experiments_index.json").write_text(
        json.dumps({"experiments": [{"name": "legacy", "timestamp": "2025-01-01T12:00:00", "path": "legacy"}]}),
        encoding="utf-8",
    )

    manager = ExperimentManager(str(experiments_dir))

    [entry] = manager.list_experiments(names=["legacy"])
    assert entry["run_id"].startswith("legacy-")
    assert manager.load_experiment(entry["run_id"])[2] == {"score": 0.5}
    migrated_index = json.loads((experiments_dir / "experiments_index.json").read_text(encoding="utf-8"))
    assert migrated_index["version"] == ExperimentManager.INDEX_VERSION
    assert migrated_index["experiments"][0]["run_id"] == entry["run_id"]


def test_experiment_manager_does_not_publish_run_if_atomic_index_write_is_interrupted(tmp_path, monkeypatch) -> None:
    experiments_dir = tmp_path / "experiments"
    manager = ExperimentManager(str(experiments_dir))
    original_atomic_write_json = manager._atomic_write_json

    def interrupt_index(path, value):
        if path == experiments_dir / "experiments_index.json":
            raise OSError("simulated interrupted index write")
        original_atomic_write_json(path, value)

    monkeypatch.setattr(manager, "_atomic_write_json", interrupt_index)
    with pytest.raises(OSError, match="simulated interrupted"):
        manager.save_experiment(_experiment("interrupted", datetime(2025, 1, 1, 12, 0, 0)), metrics={"score": 1.0})

    assert manager.list_experiments() == []
    assert not (experiments_dir / "interrupted").exists() or not any((experiments_dir / "interrupted").iterdir())
    assert not list(experiments_dir.rglob(".*.json.*"))
