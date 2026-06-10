"""Smoke tests for supported public imports and optional plotting behavior."""

from __future__ import annotations

import ast
import importlib
import importlib.util
import re
from pathlib import Path
import subprocess
import sys

import pandas as pd
import pytest

from ivt_filter.config import OlsenVelocityConfig

REPOSITORY_ROOT = Path(__file__).parent.parent
COMPATIBILITY_FACADES = {
    "ivt_filter.core": (
        "ivt_filter.postprocessing",
        "ivt_filter.processing.velocity",
        "ivt_filter.processing.classification",
        "ivt_filter.preprocessing",
    ),
    "ivt_filter.core.velocity": ("ivt_filter.processing.velocity",),
    "ivt_filter.core.classification": ("ivt_filter.processing.classification",),
    "ivt_filter.core.gaze": ("ivt_filter.preprocessing",),
}


def _compatibility_symbol_cases() -> list[object]:
    cases = []
    for facade_name, canonical_module_names in COMPATIBILITY_FACADES.items():
        facade = importlib.import_module(facade_name)
        for symbol_name in facade.__all__:
            matching_canonical_modules = [
                canonical_module_name
                for canonical_module_name in canonical_module_names
                if hasattr(importlib.import_module(canonical_module_name), symbol_name)
            ]
            assert (
                len(matching_canonical_modules) == 1
            ), f"{facade_name}.{symbol_name} must have exactly one canonical public source"
            cases.append(
                pytest.param(
                    facade_name,
                    matching_canonical_modules[0],
                    symbol_name,
                    id=f"{facade_name}-{symbol_name}",
                )
            )
    return cases


EXAMPLE_MODULES = [
    "example_experiment_tracking.py",
    "example_sample_based_window.py",
    "example_window_sweep.py",
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
    from ivt_filter.io import (
        ExperimentTracker,
        IVTPipeline,
        PipelineObserver,
        ResultsPlotter,
    )
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


def test_plotting_error_explains_optional_extra_when_matplotlib_is_unavailable() -> (
    None
):
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


def test_plotting_works_when_matplotlib_is_installed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
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


@pytest.mark.parametrize(
    ("facade_name", "canonical_module_name", "symbol_name"),
    _compatibility_symbol_cases(),
)
def test_compatibility_facade_symbols_delegate_to_canonical_implementations(
    facade_name: str, canonical_module_name: str, symbol_name: str
) -> None:
    facade = importlib.import_module(facade_name)
    canonical_module = importlib.import_module(canonical_module_name)

    assert getattr(facade, symbol_name) is getattr(canonical_module, symbol_name)


@pytest.mark.parametrize("facade_name", COMPATIBILITY_FACADES)
def test_compatibility_facades_contain_no_implementation_logic(
    facade_name: str,
) -> None:
    facade = importlib.import_module(facade_name)
    facade_path = Path(facade.__file__)
    module = ast.parse(
        facade_path.read_text(encoding="utf-8"), filename=str(facade_path)
    )

    for statement in module.body:
        if isinstance(statement, ast.Expr):
            assert isinstance(statement.value, ast.Constant)
            assert isinstance(statement.value.value, str)
        elif isinstance(statement, (ast.Import, ast.ImportFrom)):
            continue
        elif isinstance(statement, ast.Assign):
            assert all(
                isinstance(target, ast.Name) and target.id == "__all__"
                for target in statement.targets
            )
        else:
            pytest.fail(
                f"{facade_name} contains implementation logic: "
                f"{statement.__class__.__name__}"
            )


def test_obsolete_velocity_computer_module_is_removed() -> None:
    assert importlib.util.find_spec("ivt_filter.processing.velocity_computer") is None


def test_obsolete_postprocess_module_is_removed() -> None:
    assert importlib.util.find_spec("ivt_filter.postprocess") is None


LEGACY_MODULE_PREFIXES = ("ivt_filter.postprocess", "ivt_filter.core")
LEGACY_TEXT_PATTERN = re.compile(
    r"(?:\bivt_filter\.(?:postprocess|core)\b|"
    r"\bfrom\s+ivt_filter\s+import\s+(?:[^#\n]*,\s*)?(?:postprocess|core)\b|"
    r"\bfrom\s+\.+(?:postprocess|core)\b|"
    r"\bfrom\s+\.+\s+import\s+(?:[^#\n]*,\s*)?(?:postprocess|core)\b)"
)
MIGRATION_TABLE_LEGACY_REFERENCES = {
    "docs/architecture.md": {
        "| `ivt_filter.postprocess` | `ivt_filter.postprocessing` |",
        "| `ivt_filter.core.velocity` | `ivt_filter.processing.velocity` |",
        "| `ivt_filter.core.classification` | `ivt_filter.processing.classification` |",
        "| `ivt_filter.core.gaze` | `ivt_filter.preprocessing` |",
    }
}


def _is_legacy_module(module_name: str) -> bool:
    return any(
        module_name == prefix or module_name.startswith(f"{prefix}.")
        for prefix in LEGACY_MODULE_PREFIXES
    )


def _module_name(module_path: Path) -> str:
    relative_path = module_path.relative_to(REPOSITORY_ROOT).with_suffix("")
    parts = relative_path.parts
    if parts[-1] == "__init__":
        parts = parts[:-1]
    return ".".join(parts)


def _imported_module_names(
    statement: ast.Import | ast.ImportFrom, package: str
) -> set[str]:
    if isinstance(statement, ast.Import):
        return {alias.name for alias in statement.names}

    imported_from = statement.module or ""
    if statement.level:
        imported_from = importlib.util.resolve_name(
            f"{'.' * statement.level}{imported_from}", package
        )

    imported_modules = {imported_from} if imported_from else set()
    imported_modules.update(
        f"{imported_from}.{alias.name}" if imported_from else alias.name
        for alias in statement.names
        if alias.name != "*"
    )
    return imported_modules


def test_productive_modules_do_not_import_legacy_facades() -> None:
    package_root = REPOSITORY_ROOT / "ivt_filter"
    legacy_core = package_root / "core"

    for module_path in package_root.rglob("*.py"):
        if legacy_core in module_path.parents:
            continue

        module_name = _module_name(module_path)
        package = (
            module_name
            if module_path.name == "__init__.py"
            else module_name.rpartition(".")[0]
        )
        module = ast.parse(
            module_path.read_text(encoding="utf-8"), filename=str(module_path)
        )
        for statement in ast.walk(module):
            if not isinstance(statement, (ast.Import, ast.ImportFrom)):
                continue
            legacy_imports = sorted(
                imported_module
                for imported_module in _imported_module_names(statement, package)
                if _is_legacy_module(imported_module)
            )
            assert not legacy_imports, (
                f"{module_path.relative_to(REPOSITORY_ROOT)}:{statement.lineno} must not "
                f"import legacy facade(s) {legacy_imports}; use canonical modules instead"
            )


def _documentation_and_example_files() -> set[Path]:
    paths = {REPOSITORY_ROOT / "README.md"}
    paths.update(REPOSITORY_ROOT.glob("example_*.py"))
    paths.update((REPOSITORY_ROOT / "docs").rglob("*.md"))
    paths.update((REPOSITORY_ROOT / "examples").rglob("*.py"))
    paths.update((REPOSITORY_ROOT / "notebooks").rglob("*.ipynb"))
    return paths


def test_documentation_and_examples_do_not_recommend_legacy_imports() -> None:
    for file_path in sorted(_documentation_and_example_files()):
        relative_path = file_path.relative_to(REPOSITORY_ROOT).as_posix()
        allowed_lines = MIGRATION_TABLE_LEGACY_REFERENCES.get(relative_path, set())
        for line_number, line in enumerate(
            file_path.read_text(encoding="utf-8").splitlines(), 1
        ):
            if LEGACY_TEXT_PATTERN.search(line):
                assert line.strip() in allowed_lines, (
                    f"{relative_path}:{line_number} references a legacy import outside the "
                    "documented migration table"
                )
