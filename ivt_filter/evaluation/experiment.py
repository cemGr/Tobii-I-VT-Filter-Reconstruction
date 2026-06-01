# ivt_filter/experiment.py
"""
Experiment tracking and management for IVT Filter reverse engineering.

This module implements the Observer Pattern and Information Expert GRASP principle
to track, compare, and analyze different experimental configurations.

Example:
    >>> from experiment import ExperimentManager, ExperimentConfig
    >>> from config import OlsenVelocityConfig, IVTClassifierConfig
    >>> 
    >>> # Create experiment configuration
    >>> exp_config = ExperimentConfig(
    ...     name="olsen2d_median_20ms",
    ...     description="Testing Olsen 2D with median smoothing",
    ...     velocity_config=OlsenVelocityConfig(
    ...         window_length_ms=20.0,
    ...         velocity_method="olsen2d",
    ...         smoothing_mode="median"
    ...     ),
    ...     classifier_config=IVTClassifierConfig(
    ...         velocity_threshold_deg_per_sec=30.0
    ...     )
    ... )
    >>> 
    >>> # Run and track experiment
    >>> manager = ExperimentManager(experiments_dir="experiments")
    >>> manager.save_experiment(exp_config, results_df, metrics)
    >>> 
    >>> # Compare experiments
    >>> comparison = manager.compare_experiments(["exp1", "exp2"])
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from importlib import metadata as importlib_metadata
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple
from uuid import uuid4
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
import tempfile

import numpy as np
import pandas as pd

from ..config import (
    OlsenVelocityConfig,
    IVTClassifierConfig,
    SaccadeMergeConfig,
    FixationPostConfig,
)


@dataclass
class ExperimentConfig:
    """
    Complete configuration for a single experiment.
    
    Attributes:
        name: Unique identifier for the experiment (e.g., "olsen2d_median_20ms")
        description: Human-readable description of what is being tested
        velocity_config: Configuration for velocity computation
        classifier_config: Configuration for I-VT classification
        saccade_merge_config: Optional post-processing configuration for saccade merging
        fixation_post_config: Optional post-processing configuration for fixation filtering
        timestamp: When the experiment was created/run
        tags: Optional tags for categorization (e.g., ["baseline", "high-frequency"])
        metadata: Additional arbitrary metadata
    """
    
    name: str
    description: str
    velocity_config: OlsenVelocityConfig
    classifier_config: IVTClassifierConfig
    saccade_merge_config: Optional[SaccadeMergeConfig] = None
    fixation_post_config: Optional[FixationPostConfig] = None
    timestamp: Optional[datetime] = None
    tags: Optional[List[str]] = None
    metadata: Optional[Dict[str, Any]] = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()
        if self.tags is None:
            self.tags = []
        if self.metadata is None:
            self.metadata = {}
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "name": self.name,
            "description": self.description,
            "velocity_config": asdict(self.velocity_config),
            "classifier_config": asdict(self.classifier_config),
            "saccade_merge_config": asdict(self.saccade_merge_config) if self.saccade_merge_config else None,
            "fixation_post_config": asdict(self.fixation_post_config) if self.fixation_post_config else None,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
            "tags": self.tags,
            "metadata": self.metadata,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> ExperimentConfig:
        """Load from dictionary."""
        return cls(
            name=data["name"],
            description=data["description"],
            velocity_config=OlsenVelocityConfig(**data["velocity_config"]),
            classifier_config=IVTClassifierConfig(**data["classifier_config"]),
            saccade_merge_config=SaccadeMergeConfig(**data["saccade_merge_config"]) if data.get("saccade_merge_config") else None,
            fixation_post_config=FixationPostConfig(**data["fixation_post_config"]) if data.get("fixation_post_config") else None,
            timestamp=datetime.fromisoformat(data["timestamp"]) if data.get("timestamp") else None,
            tags=data.get("tags", []),
            metadata=data.get("metadata", {}),
        )


class ExperimentManager:
    """Persist immutable experiment runs and their provenance metadata."""

    INDEX_VERSION = 2

    def __init__(self, experiments_dir: str = "experiments"):
        self.experiments_dir = Path(experiments_dir)
        self.experiments_dir.mkdir(parents=True, exist_ok=True)
        self._index_file = self.experiments_dir / "experiments_index.json"
        self._load_index()

    def _load_index(self) -> None:
        """Load the index and migrate legacy name-only entries in place."""
        if self._index_file.exists():
            with open(self._index_file, "r", encoding="utf-8") as f:
                self._index = json.load(f)
        else:
            self._index = {"version": self.INDEX_VERSION, "experiments": []}
            return

        migrated = self._index.get("version") != self.INDEX_VERSION
        self._index.setdefault("experiments", [])
        for entry in self._index["experiments"]:
            if "run_id" not in entry:
                digest = hashlib.sha256(
                    json.dumps(entry, sort_keys=True).encode("utf-8")
                ).hexdigest()[:12]
                entry["run_id"] = f"legacy-{digest}"
                migrated = True
            entry.setdefault("experiment_name", entry.get("name"))
            entry["name"] = entry["experiment_name"]
        self._index["version"] = self.INDEX_VERSION
        if migrated:
            self._save_index()

    def _atomic_write_json(self, path: Path, value: Any) -> None:
        """Write JSON by replacement so readers never observe a partial file."""
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
        temporary_path = Path(temporary_name)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(value, f, indent=2, ensure_ascii=False)
                f.flush()
                os.fsync(f.fileno())
            os.replace(temporary_path, path)
        except BaseException:
            temporary_path.unlink(missing_ok=True)
            raise

    def _atomic_write_results(self, path: Path, results_df: pd.DataFrame) -> None:
        """Atomically write a TSV results file."""
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
        os.close(fd)
        temporary_path = Path(temporary_name)
        try:
            results_df.to_csv(temporary_path, sep="\t", index=False, decimal=",")
            os.replace(temporary_path, path)
        except BaseException:
            temporary_path.unlink(missing_ok=True)
            raise

    def _save_index(self) -> None:
        self._atomic_write_json(self._index_file, self._index)

    def _make_run_id(self, config: ExperimentConfig) -> str:
        config_digest = hashlib.sha256(
            json.dumps(config.to_dict(), sort_keys=True, default=str).encode("utf-8")
        ).hexdigest()[:12]
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
        return f"{timestamp}-{config_digest}-{uuid4().hex[:8]}"

    @staticmethod
    def _input_checksum(input_path: Optional[str]) -> Optional[str]:
        if not input_path:
            return None
        digest = hashlib.sha256()
        with open(input_path, "rb") as f:
            for block in iter(lambda: f.read(1024 * 1024), b""):
                digest.update(block)
        return digest.hexdigest()

    @staticmethod
    def _input_row_count(input_path: Optional[str]) -> Optional[int]:
        if not input_path:
            return None
        with open(input_path, "rb") as f:
            return max(sum(1 for _ in f) - 1, 0)

    @staticmethod
    def _git_commit() -> Optional[str]:
        try:
            return subprocess.run(
                ["git", "rev-parse", "HEAD"],
                check=True,
                capture_output=True,
                text=True,
                timeout=2,
            ).stdout.strip() or None
        except (OSError, subprocess.SubprocessError):
            return None

    @staticmethod
    def _package_version() -> str:
        try:
            return importlib_metadata.version("tobii-ivt-filter")
        except importlib_metadata.PackageNotFoundError:
            return "unknown"

    def _build_provenance(
        self,
        config: ExperimentConfig,
        *,
        input_path: Optional[str],
        input_row_count: Optional[int],
        command: Optional[List[str]],
        api_parameters: Optional[Dict[str, Any]],
        reference_system_version: Optional[str],
        reference_export_identifier: Optional[str],
        provenance: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        details = {
            "input_file": str(input_path) if input_path else None,
            "input_file_checksum_sha256": self._input_checksum(input_path),
            "input_row_count": input_row_count if input_row_count is not None else self._input_row_count(input_path),
            "package_version": self._package_version(),
            "git_commit": self._git_commit(),
            "python_version": platform.python_version(),
            "platform": platform.platform(),
            "dependency_versions": {"numpy": np.__version__, "pandas": pd.__version__},
            "executed_command": command if command is not None else sys.argv,
            "api_parameters": api_parameters if api_parameters is not None else config.to_dict(),
            "reference_system_version": reference_system_version,
            "reference_export_identifier": reference_export_identifier,
        }
        if provenance:
            details.update(provenance)
        return self._make_serializable(details)

    def save_experiment(
        self,
        config: ExperimentConfig,
        results_df: Optional[pd.DataFrame] = None,
        metrics: Optional[Dict[str, Any]] = None,
        output_path: Optional[str] = None,
        *,
        input_path: Optional[str] = None,
        input_row_count: Optional[int] = None,
        command: Optional[List[str]] = None,
        api_parameters: Optional[Dict[str, Any]] = None,
        reference_system_version: Optional[str] = None,
        reference_export_identifier: Optional[str] = None,
        provenance: Optional[Dict[str, Any]] = None,
    ) -> Path:
        """Save one immutable execution and append it to the experiment index."""
        tracking = (config.metadata or {}).get("tracking", {})
        input_path = input_path or tracking.get("input_path")
        input_row_count = input_row_count if input_row_count is not None else tracking.get("input_row_count")
        command = command if command is not None else tracking.get("command")
        api_parameters = api_parameters if api_parameters is not None else tracking.get("api_parameters")
        reference_system_version = reference_system_version or tracking.get("reference_system_version")
        reference_export_identifier = reference_export_identifier or tracking.get("reference_export_identifier")
        run_id = self._make_run_id(config)
        base_dir = Path(output_path) if output_path else self.experiments_dir / config.name
        exp_dir = base_dir / run_id
        exp_dir.mkdir(parents=True, exist_ok=False)

        config.metadata = dict(config.metadata or {})
        config.metadata["provenance"] = self._build_provenance(
            config,
            input_path=input_path,
            input_row_count=input_row_count,
            command=command,
            api_parameters=api_parameters,
            reference_system_version=reference_system_version,
            reference_export_identifier=reference_export_identifier,
            provenance=provenance,
        )
        try:
            self._atomic_write_json(exp_dir / "config.json", config.to_dict())
            if results_df is not None:
                self._atomic_write_results(exp_dir / "results.tsv", results_df)
            if metrics is not None:
                self._atomic_write_json(exp_dir / "metrics.json", self._make_serializable(metrics))

            try:
                stored_path = str(exp_dir.relative_to(self.experiments_dir))
            except ValueError:
                stored_path = str(exp_dir.resolve())
            entry = {
                "run_id": run_id,
                "name": config.name,
                "experiment_name": config.name,
                "description": config.description,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "config_timestamp": config.timestamp.isoformat() if config.timestamp else None,
                "tags": config.tags,
                "path": stored_path,
            }
            self._index["experiments"].append(entry)
            try:
                self._save_index()
            except BaseException:
                self._index["experiments"].pop()
                raise
        except BaseException:
            shutil.rmtree(exp_dir, ignore_errors=True)
            raise

        print(f"✓ Experiment '{config.name}' run '{run_id}' saved to {exp_dir}")
        return exp_dir

    def _matching_entries(self, identifier: str) -> List[Dict[str, Any]]:
        exact = [e for e in self._index["experiments"] if e["run_id"] == identifier]
        if exact:
            return exact
        return [e for e in self._index["experiments"] if e.get("experiment_name", e.get("name")) == identifier]

    def _find_entry(self, identifier: str) -> Dict[str, Any]:
        matches = self._matching_entries(identifier)
        if not matches:
            raise ValueError(f"Experiment run or name '{identifier}' not found")
        return sorted(matches, key=lambda x: x.get("timestamp", ""), reverse=True)[0]

    def load_experiment(self, run_id: str) -> Tuple[ExperimentConfig, Optional[pd.DataFrame], Optional[Dict]]:
        """Load a run ID; a human-readable name resolves to its latest run."""
        entry = self._find_entry(run_id)
        exp_dir = Path(entry["path"])
        if not exp_dir.is_absolute():
            exp_dir = self.experiments_dir / exp_dir
        with open(exp_dir / "config.json", "r", encoding="utf-8") as f:
            config = ExperimentConfig.from_dict(json.load(f))
        results_path = exp_dir / "results.tsv"
        results_df = pd.read_csv(results_path, sep="\t", decimal=",") if results_path.exists() else None
        metrics_path = exp_dir / "metrics.json"
        if metrics_path.exists():
            with open(metrics_path, "r", encoding="utf-8") as f:
                metrics = json.load(f)
        else:
            metrics = None
        return config, results_df, metrics

    def list_experiments(
        self,
        tags: Optional[List[str]] = None,
        names: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """List immutable runs, optionally filtered by tags and readable names."""
        experiments = self._index["experiments"]
        if tags:
            experiments = [e for e in experiments if any(tag in e.get("tags", []) for tag in tags)]
        if names:
            experiments = [e for e in experiments if e.get("experiment_name", e.get("name")) in names]
        return sorted(experiments, key=lambda x: x.get("timestamp", ""), reverse=True)

    def compare_experiments(
        self,
        experiment_ids: List[str],
        metrics_to_compare: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """Compare run IDs; readable names expand to all runs with that name."""
        metrics_to_compare = metrics_to_compare or [
            "percentage_agreement", "fixation_recall", "saccade_recall", "cohen_kappa"
        ]
        comparison_data = []
        for identifier in experiment_ids:
            matches = self._matching_entries(identifier)
            if not matches:
                print(f"Warning: Could not load experiment '{identifier}': not found")
            for entry in matches:
                try:
                    config, _, metrics = self.load_experiment(entry["run_id"])
                    row = {
                        "experiment": entry["run_id"],
                        "experiment_name": entry.get("experiment_name", entry.get("name")),
                        "window_ms": config.velocity_config.window_length_ms,
                        "velocity_method": config.velocity_config.velocity_method,
                        "eye_mode": config.velocity_config.eye_mode,
                        "smoothing": config.velocity_config.smoothing_mode,
                        "threshold": config.classifier_config.velocity_threshold_deg_per_sec,
                    }
                    if metrics:
                        for metric in metrics_to_compare:
                            row[metric] = metrics.get(metric)
                    comparison_data.append(row)
                except Exception as e:
                    print(f"Warning: Could not load experiment '{entry['run_id']}': {e}")
        return pd.DataFrame(comparison_data)

    def get_best_configuration(
        self,
        metric: str = "percentage_agreement",
        maximize: bool = True,
        tags: Optional[List[str]] = None,
        names: Optional[List[str]] = None,
    ) -> Tuple[str, float, ExperimentConfig]:
        """Return the best immutable run ID, metric value, and configuration."""
        best_run_id = None
        best_value = float("-inf") if maximize else float("inf")
        best_config = None
        for entry in self.list_experiments(tags=tags, names=names):
            try:
                config, _, metrics = self.load_experiment(entry["run_id"])
                if metrics and metric in metrics:
                    value = metrics[metric]
                    if (maximize and value > best_value) or (not maximize and value < best_value):
                        best_run_id, best_value, best_config = entry["run_id"], value, config
            except Exception as e:
                print(f"Warning: Could not load experiment '{entry['run_id']}': {e}")
        if best_run_id is None:
            raise ValueError(f"No experiments found with metric '{metric}'")
        return best_run_id, best_value, best_config  # type: ignore[return-value]

    def _make_serializable(self, obj: Any) -> Any:
        """Convert non-JSON-serializable objects to serializable format."""
        if isinstance(obj, dict):
            return {k: self._make_serializable(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [self._make_serializable(item) for item in obj]
        if isinstance(obj, (int, float, str, bool, type(None))):
            return obj
        if hasattr(obj, "item"):
            return self._make_serializable(obj.item())
        if hasattr(obj, "__dict__"):
            return self._make_serializable(obj.__dict__)
        return str(obj)
