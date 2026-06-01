from __future__ import annotations

import logging
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from ivt_filter.config import FixationPostConfig, IVTClassifierConfig, OlsenVelocityConfig
from ivt_filter.io.pipeline import IVTPipeline
from ivt_filter.postprocessing.merge_fixations import merge_adjacent_fixations
from ivt_filter.processing.velocity import SamplingAnalyzer


class FailingObserver:
    def on_pipeline_start(self, config):
        raise RuntimeError("start failure")

    def on_pipeline_complete(self, config, results_df, metrics):
        raise RuntimeError("complete failure")

    def on_pipeline_error(self, config, error):
        raise RuntimeError("error failure")


@pytest.mark.parametrize(
    ("notify", "args", "expected"),
    [
        ("_notify_start", (SimpleNamespace(),), "failed on start: start failure"),
        (
            "_notify_complete",
            (SimpleNamespace(), pd.DataFrame(), None),
            "failed on complete: complete failure",
        ),
        (
            "_notify_error",
            (SimpleNamespace(), RuntimeError("pipeline failure")),
            "failed on error: error failure",
        ),
    ],
)
def test_pipeline_logs_observer_failures(caplog, notify, args, expected):
    pipeline = IVTPipeline(OlsenVelocityConfig(), IVTClassifierConfig())
    pipeline.register_observer(FailingObserver())

    with caplog.at_level(logging.WARNING, logger="ivt_filter.io.pipeline"):
        getattr(pipeline, notify)(*args)

    assert expected in caplog.text


def test_sampling_analyzer_logs_derived_window(caplog):
    cfg = OlsenVelocityConfig(window_length_ms=20, auto_fixed_window_from_ms=True)

    with caplog.at_level(logging.INFO, logger="ivt_filter.processing.velocity"):
        result = SamplingAnalyzer().analyze(np.array([0.0, 10.0, 20.0, 30.0]), cfg)

    assert result.config.fixed_window_samples == 3
    assert "[Sampling] mean dt = 10.000 ms -> measured ~100.0 Hz" in caplog.text
    assert "[Window] auto sample window: 3 samples total" in caplog.text


def test_merge_fixations_logs_angle_calculation_strategy(caplog):
    df = pd.DataFrame(
        {
            "ivt_sample_type": ["Fixation"],
            "time_ms": [0.0],
            "combined_x_mm": [0.0],
            "combined_y_mm": [0.0],
            "eye_z_mm": [600.0],
        }
    )

    with caplog.at_level(logging.INFO, logger="ivt_filter.postprocessing.merge_fixations"):
        merge_adjacent_fixations(
            df,
            FixationPostConfig(),
            sample_col="ivt_sample_type",
            time_col="time_ms",
            x_col="combined_x_mm",
            y_col="combined_y_mm",
            eye_z_col="eye_z_mm",
        )

    assert "[MergeFixations] Using Olsen 2D approximation" in caplog.text


def test_run_with_tracking_logs_metric_computation_failure(monkeypatch, caplog):
    pipeline = IVTPipeline(OlsenVelocityConfig(), IVTClassifierConfig())
    monkeypatch.setattr(pipeline, "run", lambda **kwargs: pd.DataFrame())
    monkeypatch.setattr(
        "ivt_filter.io.pipeline.compute_ivt_metrics",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("metrics failure")),
    )

    with caplog.at_level(logging.WARNING, logger="ivt_filter.io.pipeline"):
        result = pipeline.run_with_tracking("input.tsv", SimpleNamespace(), evaluate=True)

    assert result.empty
    assert "Could not compute metrics: metrics failure" in caplog.text
