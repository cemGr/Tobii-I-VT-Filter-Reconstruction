"""Smoke tests for supported public imports and optional plotting behavior."""
from __future__ import annotations

import ast
import importlib.util
import re
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
    "examples/velocity_comparison.py",
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


def test_import_processing_package_uses_canonical_velocity_api() -> None:
    import ivt_filter.processing as processing
    from ivt_filter.processing.velocity import SamplingAnalyzer

    assert processing.SamplingAnalyzer is SamplingAnalyzer
    assert not hasattr(processing, "VelocityComputer")


def test_legacy_core_imports_delegate_to_canonical_implementations() -> None:
    from ivt_filter.core.classification import IVTClassifier as legacy_classifier
    from ivt_filter.core.gaze import gap_fill_gaze as legacy_gap_fill
    from ivt_filter.core.velocity import SamplingAnalyzer as legacy_sampling_analyzer
    from ivt_filter.preprocessing import gap_fill_gaze
    from ivt_filter.processing.classification import IVTClassifier
    from ivt_filter.processing.velocity import SamplingAnalyzer

    assert legacy_classifier is IVTClassifier
    assert legacy_gap_fill is gap_fill_gaze
    assert legacy_sampling_analyzer is SamplingAnalyzer


def test_obsolete_velocity_computer_module_is_removed() -> None:
    assert importlib.util.find_spec("ivt_filter.processing.velocity_computer") is None


def test_legacy_postprocess_imports_delegate_to_canonical_implementations() -> None:
    from ivt_filter.postprocess import (
        apply_fixation_postprocessing as legacy_apply_fixation_postprocessing,
    )
    from ivt_filter.postprocess import (
        merge_short_saccade_blocks as legacy_merge_short_saccade_blocks,
    )
    from ivt_filter.postprocessing import (
        apply_fixation_postprocessing,
        merge_short_saccade_blocks,
    )

    assert legacy_apply_fixation_postprocessing is apply_fixation_postprocessing
    assert legacy_merge_short_saccade_blocks is merge_short_saccade_blocks


LEGACY_IMPORT_ROOTS = {"core", "postprocess"}
LEGACY_MIGRATION_TABLE_ROWS = {
    "docs/architecture.md": {
        "| `ivt_filter.core` | Canonical modules under `ivt_filter.preprocessing` and `ivt_filter.processing` |",
        "| `ivt_filter.postprocess` | `ivt_filter.postprocessing` |",
    },
}
LEGACY_TEXT_PATTERNS = [
    re.compile(r"ivt_filter\.(?:core|postprocess)(?![A-Za-z0-9_])"),
    re.compile(r"from\s+\.+(?:core|postprocess)(?![A-Za-z0-9_])"),
    re.compile(r"from\s+\.+\s+import[^\n]*(?<![A-Za-z0-9_])(?:core|postprocess)(?![A-Za-z0-9_])"),
    re.compile(r"from\s+ivt_filter\s+import[^\n]*(?<![A-Za-z0-9_])(?:core|postprocess)(?![A-Za-z0-9_])"),
]


def _is_legacy_module(module_name: str) -> bool:
    return any(
        module_name == f"ivt_filter.{legacy_root}"
        or module_name.startswith(f"ivt_filter.{legacy_root}.")
        for legacy_root in LEGACY_IMPORT_ROOTS
    )


def _legacy_imports(module_path: Path) -> list[str]:
    tree = ast.parse(module_path.read_text(encoding="utf-8"), filename=str(module_path))
    violations = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            violations.extend(alias.name for alias in node.names if _is_legacy_module(alias.name))
        elif isinstance(node, ast.ImportFrom):
            imported_names = {alias.name for alias in node.names}
            if node.level:
                if (
                    node.module
                    and any(
                        node.module == legacy_root or node.module.startswith(f"{legacy_root}.")
                        for legacy_root in LEGACY_IMPORT_ROOTS
                    )
                ) or (node.module is None and imported_names & LEGACY_IMPORT_ROOTS):
                    violations.append(ast.unparse(node))
            elif node.module and _is_legacy_module(node.module):
                violations.append(ast.unparse(node))
            elif node.module == "ivt_filter" and imported_names & LEGACY_IMPORT_ROOTS:
                violations.append(ast.unparse(node))
    return violations


def test_productive_modules_do_not_import_legacy_facades() -> None:
    package_root = REPOSITORY_ROOT / "ivt_filter"
    legacy_postprocess_facade = package_root / "postprocess.py"
    legacy_core_facade = package_root / "core"

    for module_path in package_root.rglob("*.py"):
        if module_path == legacy_postprocess_facade or legacy_core_facade in module_path.parents:
            continue
        legacy_imports = _legacy_imports(module_path)
        assert not legacy_imports, (
            f"{module_path.relative_to(REPOSITORY_ROOT)} must import canonical modules instead "
            f"of legacy facades: {legacy_imports}"
        )


def _documentation_and_example_files() -> list[Path]:
    paths = [REPOSITORY_ROOT / "README.md", REPOSITORY_ROOT / "quick_window_test.py"]
    paths.extend(REPOSITORY_ROOT.glob("example_*.py"))
    paths.extend((REPOSITORY_ROOT / "docs").rglob("*.md"))
    paths.extend((REPOSITORY_ROOT / "examples").rglob("*.py"))
    paths.extend((REPOSITORY_ROOT / "notebooks").rglob("*.ipynb"))
    return sorted(set(paths))


def test_documentation_and_examples_do_not_reference_legacy_facades() -> None:
    for content_path in _documentation_and_example_files():
        relative_path = content_path.relative_to(REPOSITORY_ROOT).as_posix()
        allowed_rows = LEGACY_MIGRATION_TABLE_ROWS.get(relative_path, set())
        for line_number, line in enumerate(content_path.read_text(encoding="utf-8").splitlines(), 1):
            if line in allowed_rows:
                continue
            assert not any(pattern.search(line) for pattern in LEGACY_TEXT_PATTERNS), (
                f"{relative_path}:{line_number} references a legacy facade outside the documented "
                "migration table"
            )
