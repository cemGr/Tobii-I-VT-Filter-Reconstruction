"""Smoke tests for supported public imports and optional plotting behavior."""
from __future__ import annotations

import importlib.util
from pathlib import Path
import subprocess
import sys

import pandas as pd
import pytest

from ivt_filter.config import OlsenVelocityConfig


REPOSITORY_ROOT = Path(__file__).parent.parent
EXAMPLE_MODULES = [
    "example_experiment_tracking.py",
    "example_sample_based_window.py",
    "example_window_sweep.py",
    "quick_window_test.py",
]


def test_import_package() -> None:
    import ivt_filter

    assert ivt_filter.__name__ == "ivt_filter"


def test_import_canonical_public_api() -> None:
    from ivt_filter.evaluation.experiment import ExperimentConfig, ExperimentManager
    from ivt_filter.io.observers import (
        ConsoleReporter,
        ExperimentTracker,
        MetricsLogger,
        PipelineObserver,
        ResultsPlotter,
    )
    from ivt_filter.io.pipeline import IVTPipeline
    from ivt_filter.utils.sampling import estimate_sampling_rate
    from ivt_filter.utils.window_utils import (
        create_adaptive_config,
        create_sample_based_config,
        create_time_based_config,
        detect_sampling_rate,
        milliseconds_to_samples,
        print_window_info,
        recommend_window_size,
        samples_to_milliseconds,
    )

    assert all(
        callable(public_object)
        for public_object in [
            IVTPipeline,
            ExperimentConfig,
            ExperimentManager,
            PipelineObserver,
            ConsoleReporter,
            MetricsLogger,
            ExperimentTracker,
            ResultsPlotter,
            estimate_sampling_rate,
            samples_to_milliseconds,
            milliseconds_to_samples,
            detect_sampling_rate,
            create_time_based_config,
            create_sample_based_config,
            create_adaptive_config,
            print_window_info,
            recommend_window_size,
        ]
    )


def test_import_convenience_subpackages() -> None:
    from ivt_filter.io import ExperimentTracker, IVTPipeline, PipelineObserver, ResultsPlotter
    from ivt_filter.utils import estimate_sampling_rate, samples_to_milliseconds

    assert all(
        callable(public_object)
        for public_object in [
            IVTPipeline,
            PipelineObserver,
            ExperimentTracker,
            ResultsPlotter,
            estimate_sampling_rate,
            samples_to_milliseconds,
        ]
    )


@pytest.mark.parametrize("example_file", EXAMPLE_MODULES)
def test_import_executable_example_without_running_it(
    example_file: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module_path = REPOSITORY_ROOT / example_file
    monkeypatch.chdir(tmp_path)
    spec = importlib.util.spec_from_file_location(module_path.stem, module_path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)

    before = set(tmp_path.iterdir())
    spec.loader.exec_module(module)

    assert set(tmp_path.iterdir()) == before


def test_core_imports_work_without_matplotlib() -> None:
    script = """
import importlib.abc
import sys

class MatplotlibBlocker(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "matplotlib" or fullname.startswith("matplotlib."):
            raise AssertionError(f"core import requested optional dependency: {fullname}")
        return None

sys.meta_path.insert(0, MatplotlibBlocker())
import ivt_filter
import ivt_filter.evaluation
from ivt_filter.io.pipeline import IVTPipeline
from ivt_filter.io.observers import ResultsPlotter
print("optional plotting dependency was not imported")
"""
    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=REPOSITORY_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    assert "optional plotting dependency was not imported" in completed.stdout


def test_plotting_error_explains_optional_extra_when_matplotlib_is_unavailable() -> None:
    script = """
import importlib.util
original_find_spec = importlib.util.find_spec
import ivt_filter._optional_dependencies as optional_dependencies
optional_dependencies.find_spec = lambda name: None if name == "matplotlib" else original_find_spec(name)
from ivt_filter.evaluation.plotting import plot_velocity_only
"""
    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=REPOSITORY_ROOT,
        capture_output=True,
        text=True,
    )

    assert completed.returncode != 0
    assert "pip install tobii-ivt-filter[plot]" in completed.stderr


def test_plotting_works_when_matplotlib_is_installed(monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.importorskip("matplotlib")
    from matplotlib import pyplot as plt
    from ivt_filter.evaluation.plotting import plot_velocity_only

    monkeypatch.setattr(plt, "show", lambda: None)
    df = pd.DataFrame({"time_ms": [0.0, 1.0], "velocity_deg_per_sec": [1.0, 2.0]})
    config = OlsenVelocityConfig(window_length_ms=20.0)

    plot_velocity_only(df, config)
    assert plt.gcf().axes
    plt.close("all")


def test_import_process_eye_tracking() -> None:
    from ivt_filter.simple_api import process_eye_tracking

    assert callable(process_eye_tracking)


def test_import_process_eye_tracking_adaptive() -> None:
    from ivt_filter.simple_api import process_eye_tracking_adaptive

    assert callable(process_eye_tracking_adaptive)


def test_import_get_statistics() -> None:
    from ivt_filter.simple_api import get_statistics

    assert callable(get_statistics)


def test_import_print_statistics() -> None:
    from ivt_filter.simple_api import print_statistics

    assert callable(print_statistics)
