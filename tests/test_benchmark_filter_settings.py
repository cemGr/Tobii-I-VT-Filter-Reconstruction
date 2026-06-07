from __future__ import annotations

import importlib.util
from pathlib import Path


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "benchmark_filter_settings.py"
SPEC = importlib.util.spec_from_file_location("benchmark_filter_settings", SCRIPT_PATH)
benchmark = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(benchmark)


def test_infers_left_threshold_window_and_median_smoothing() -> None:
    config, notes = benchmark.infer_pipeline_config(
        Path("LeftV30W20NoiseMedW9_input.tsv")
    )

    assert config.velocity.eye_mode == "left"
    assert config.velocity.window_length_ms == 20.0
    assert config.velocity.smoothing_mode == "median"
    assert config.velocity.smoothing_window_samples == 9
    assert config.classifier.velocity_threshold_deg_per_sec == 30.0
    assert notes["smoothing_mode"] == "median"


def test_infers_left20ms30_as_window_20_threshold_30() -> None:
    config, notes = benchmark.infer_pipeline_config(Path("left20ms30_input.tsv"))

    assert config.velocity.eye_mode == "left"
    assert config.velocity.window_length_ms == 20.0
    assert config.classifier.velocity_threshold_deg_per_sec == 30.0
    assert notes["threshold_deg_per_sec"] == 30.0


def test_infers_both_eye_gap_fill_and_no_noise() -> None:
    config, notes = benchmark.infer_pipeline_config(
        Path("IVT-Interp75-EyeBoth-NoNoise-W20-V30.tsv")
    )

    assert config.velocity.eye_mode == "average"
    assert config.velocity.window_length_ms == 20.0
    assert config.velocity.gap_fill_enabled is True
    assert config.velocity.gap_fill_max_gap_ms == 75.0
    assert config.velocity.smoothing_mode == "none"
    assert notes["gap_fill_enabled"] is True


def test_infers_time_us_when_time_ms_probe_is_empty(tmp_path) -> None:
    path = tmp_path / "IVT-Interp75-EyeBoth-NoNoise-W20-V30.tsv"
    path.write_text(
        "time_ms\ttime_us\tgt_event_type\tgt_event_index\n"
        "\t7066126260,0\tEyesNotFound\t1,0\n",
        encoding="utf-8",
    )

    config, notes = benchmark.infer_pipeline_config(path)

    assert config.velocity.time_column == "time_us"
    assert config.velocity.time_unit == "us"
    assert notes["time_column"] == "time_us"


def test_infers_merge_fixation_postprocessing_angle() -> None:
    config, notes = benchmark.infer_pipeline_config(
        Path("LeftV30W20MergeF75ms0-5angle_input.tsv")
    )

    assert config.fixation_post is not None
    assert config.fixation_post.merge_adjacent_fixations is True
    assert config.fixation_post.max_time_gap_ms == 75.0
    assert config.fixation_post.max_angle_deg == 0.5
    assert notes["merge_fix_max_angle_deg"] == 0.5


def test_build_aggregate_summary_uses_weighted_counts_and_errors() -> None:
    results = [
        {
            "file": "a.tsv",
            "status": "ok",
            "duration_seconds": 1.0,
            "metrics": {
                "n_samples_total": 10,
                "n_samples_gt_fix_or_sac": 8,
                "n_agree": 6,
                "n_agree_all": 7,
                "n_fix_in_gt": 5,
                "n_sac_in_gt": 3,
                "tp_fix": 4,
                "tp_sac": 2,
                "percentage_agreement": 75.0,
                "percentage_agreement_all": 70.0,
                "fixation_recall": 80.0,
                "saccade_recall": 66.6667,
                "cohen_kappa": 0.5,
                "labels": ["Fixation", "Saccade"],
                "confusion_matrix": [[4, 1], [1, 2]],
            },
            "event_iou": {
                "n_gt_events": 4,
                "n_pred_events": 5,
                "n_matched": 3,
                "n_fn": 1,
                "n_fp": 2,
                "n_correct_type": 2,
                "mean_iou": 0.75,
            },
        },
        {
            "file": "b.tsv",
            "status": "ok",
            "duration_seconds": 2.0,
            "metrics": {
                "n_samples_total": 20,
                "n_samples_gt_fix_or_sac": 10,
                "n_agree": 9,
                "n_agree_all": 18,
                "n_fix_in_gt": 6,
                "n_sac_in_gt": 4,
                "tp_fix": 6,
                "tp_sac": 3,
                "percentage_agreement": 90.0,
                "percentage_agreement_all": 90.0,
                "fixation_recall": 100.0,
                "saccade_recall": 75.0,
                "cohen_kappa": 0.75,
                "labels": ["Fixation", "Saccade"],
                "confusion_matrix": [[6, 0], [1, 3]],
            },
            "event_iou": {
                "n_gt_events": 6,
                "n_pred_events": 6,
                "n_matched": 6,
                "n_fn": 0,
                "n_fp": 0,
                "n_correct_type": 5,
                "mean_iou": 0.9,
            },
        },
        {
            "file": "bad.tsv",
            "status": "error",
            "duration_seconds": 0.5,
            "error_type": "ValueError",
            "error_message": "missing data",
        },
    ]

    summary = benchmark.build_aggregate_summary(results)

    assert summary["totals"]["files_ok"] == 2
    assert summary["totals"]["files_error"] == 1
    assert summary["totals"]["n_samples_total"] == 30
    assert summary["weighted"]["percentage_agreement"] == 15 / 18 * 100.0
    assert summary["weighted"]["percentage_agreement_all"] == 25 / 30 * 100.0
    assert summary["weighted"]["event_correct_type_rate"] == 7 / 10
    assert summary["errors"][0]["file"] == "bad.tsv"
