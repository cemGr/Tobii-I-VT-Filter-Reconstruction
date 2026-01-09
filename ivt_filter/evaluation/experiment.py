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
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple
import json
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
    """
    Manages experiment tracking and comparison (GRASP: Information Expert).
    
    Responsibilities:
    - Save/load experiment configurations and results
    - Compare multiple experiments
    - Find best configurations based on metrics
    - Track experiment history
    
    Example:
        >>> manager = ExperimentManager(experiments_dir="experiments")
        >>> manager.save_experiment(config, results_df, metrics)
        >>> best = manager.get_best_configuration(metric="fixation_recall")
    """
    
    def __init__(self, experiments_dir: str = "experiments"):
        """
        Initialize experiment manager.
        
        Args:
            experiments_dir: Directory where experiments will be stored
        """
        self.experiments_dir = Path(experiments_dir)
        self.experiments_dir.mkdir(parents=True, exist_ok=True)
        self._index_file = self.experiments_dir / "experiments_index.json"
        self._load_index()
    
    def _load_index(self):
        """Load experiment index from disk."""
        if self._index_file.exists():
            with open(self._index_file, "r", encoding="utf-8") as f:
                self._index = json.load(f)
        else:
            self._index = {"experiments": []}
    
    def _save_index(self):
        """Save experiment index to disk."""
        with open(self._index_file, "w", encoding="utf-8") as f:
            json.dump(self._index, f, indent=2, ensure_ascii=False)
    
    def save_experiment(
        self,
        config: ExperimentConfig,
        results_df: Optional[pd.DataFrame] = None,
        metrics: Optional[Dict[str, Any]] = None,
        output_path: Optional[str] = None,
    ) -> Path:
        """
        Save experiment configuration, results, and metrics.
        
        Args:
            config: Experiment configuration
            results_df: Optional DataFrame with full results
            metrics: Optional dictionary with evaluation metrics
            output_path: Optional custom output path (default: experiments/{name}/)
        
        Returns:
            Path to the experiment directory
        """
        if output_path:
            exp_dir = Path(output_path)
        else:
            exp_dir = self.experiments_dir / config.name
        
        exp_dir.mkdir(parents=True, exist_ok=True)
        
        # Save configuration
        config_path = exp_dir / "config.json"
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(config.to_dict(), f, indent=2, ensure_ascii=False)
        
        # Save results DataFrame
        if results_df is not None:
            results_path = exp_dir / "results.tsv"
            results_df.to_csv(results_path, sep="\t", index=False, decimal=",")
        
        # Save metrics
        if metrics is not None:
            metrics_path = exp_dir / "metrics.json"
            # Convert non-serializable types
            serializable_metrics = self._make_serializable(metrics)
            with open(metrics_path, "w", encoding="utf-8") as f:
                json.dump(serializable_metrics, f, indent=2, ensure_ascii=False)
        
        # Update index
        exp_entry = {
            "name": config.name,
            "description": config.description,
            "timestamp": config.timestamp.isoformat() if config.timestamp else None,
            "tags": config.tags,
            "path": str(exp_dir.relative_to(self.experiments_dir)),
        }
        
        # Remove old entry if exists
        self._index["experiments"] = [
            e for e in self._index["experiments"] if e["name"] != config.name
        ]
        self._index["experiments"].append(exp_entry)
        self._save_index()
        
        print(f"✓ Experiment '{config.name}' saved to {exp_dir}")
        return exp_dir
    
    def load_experiment(self, name: str) -> Tuple[ExperimentConfig, Optional[pd.DataFrame], Optional[Dict]]:
        """
        Load experiment by name.
        
        Args:
            name: Experiment name
        
        Returns:
            Tuple of (config, results_df, metrics)
        """
        exp_entry = next((e for e in self._index["experiments"] if e["name"] == name), None)
        if not exp_entry:
            raise ValueError(f"Experiment '{name}' not found")
        
        exp_dir = self.experiments_dir / exp_entry["path"]
        
        # Load configuration
        config_path = exp_dir / "config.json"
        with open(config_path, "r", encoding="utf-8") as f:
            config_data = json.load(f)
        config = ExperimentConfig.from_dict(config_data)
        
        # Load results
        results_path = exp_dir / "results.tsv"
        results_df = pd.read_csv(results_path, sep="\t", decimal=",") if results_path.exists() else None
        
        # Load metrics
        metrics_path = exp_dir / "metrics.json"
        if metrics_path.exists():
            with open(metrics_path, "r", encoding="utf-8") as f:
                metrics = json.load(f)
        else:
            metrics = None
        
        return config, results_df, metrics
    
    def list_experiments(self, tags: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """
        List all experiments, optionally filtered by tags.
        
        Args:
            tags: Optional list of tags to filter by
        
        Returns:
            List of experiment entries
        """
        experiments = self._index["experiments"]
        
        if tags:
            experiments = [
                e for e in experiments
                if any(tag in e.get("tags", []) for tag in tags)
            ]
        
        return sorted(experiments, key=lambda x: x.get("timestamp", ""), reverse=True)
    
    def compare_experiments(
        self,
        experiment_names: List[str],
        metrics_to_compare: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """
        Compare multiple experiments side-by-side.
        
        Args:
            experiment_names: List of experiment names to compare
            metrics_to_compare: Optional list of specific metrics to compare
        
        Returns:
            DataFrame with comparison results
        """
        if metrics_to_compare is None:
            metrics_to_compare = [
                "percentage_agreement",
                "fixation_recall",
                "saccade_recall",
                "cohen_kappa",
            ]
        
        comparison_data = []
        
        for name in experiment_names:
            try:
                config, _, metrics = self.load_experiment(name)
                
                row = {
                    "experiment": name,
                    "window_ms": config.velocity_config.window_length_ms,
                    "velocity_method": config.velocity_config.velocity_method,
                    "eye_mode": config.velocity_config.eye_mode,
                    "smoothing": config.velocity_config.smoothing_mode,
                    "threshold": config.classifier_config.velocity_threshold_deg_per_sec,
                }
                
                if metrics:
                    for metric in metrics_to_compare:
                        row[metric] = metrics.get(metric, None)
                
                comparison_data.append(row)
            except Exception as e:
                print(f"Warning: Could not load experiment '{name}': {e}")
        
        return pd.DataFrame(comparison_data)
    
    def get_best_configuration(
        self,
        metric: str = "percentage_agreement",
        maximize: bool = True,
        tags: Optional[List[str]] = None,
    ) -> Tuple[str, float, ExperimentConfig]:
        """
        Find the best experiment configuration based on a metric.
        
        Args:
            metric: Metric name to optimize
            maximize: True to maximize metric, False to minimize
            tags: Optional tags to filter experiments
        
        Returns:
            Tuple of (experiment_name, metric_value, config)
        """
        experiments = self.list_experiments(tags=tags)
        
        best_name = None
        best_value = float("-inf") if maximize else float("inf")
        best_config = None
        
        for exp_entry in experiments:
            try:
                config, _, metrics = self.load_experiment(exp_entry["name"])
                
                if metrics and metric in metrics:
                    value = metrics[metric]
                    
                    if maximize and value > best_value:
                        best_name = exp_entry["name"]
                        best_value = value
                        best_config = config
                    elif not maximize and value < best_value:
                        best_name = exp_entry["name"]
                        best_value = value
                        best_config = config
            except Exception as e:
                print(f"Warning: Could not load experiment '{exp_entry['name']}': {e}")
        
        if best_name is None:
            raise ValueError(f"No experiments found with metric '{metric}'")
        
        return best_name, best_value, best_config  # type: ignore[return-value]
    
    def _make_serializable(self, obj: Any) -> Any:
        """Convert non-JSON-serializable objects to serializable format."""
        if isinstance(obj, dict):
            return {k: self._make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._make_serializable(item) for item in obj]
        elif isinstance(obj, (int, float, str, bool, type(None))):
            return obj
        elif hasattr(obj, "__dict__"):
            return self._make_serializable(obj.__dict__)
        else:
            return str(obj)
