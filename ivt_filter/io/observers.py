# ivt_filter/observers.py
"""
Observer Pattern implementation for automatic experiment tracking and logging.

This module implements the Observer Pattern to decouple pipeline execution
from logging, visualization, and metric tracking.

Example:
    >>> from observers import MetricsLogger, ResultsPlotter, ConsoleReporter
    >>> from pipeline import IVTPipeline
    >>> 
    >>> # Create pipeline
    >>> pipeline = IVTPipeline(velocity_config, classifier_config)
    >>> 
    >>> # Register observers
    >>> pipeline.register_observer(MetricsLogger("logs/metrics.csv"))
    >>> pipeline.register_observer(ResultsPlotter("plots/"))
    >>> pipeline.register_observer(ConsoleReporter())
    >>> 
    >>> # Run pipeline - observers are automatically notified
    >>> df = pipeline.run(input_df)
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime
import pandas as pd
import json

from ..evaluation.experiment import ExperimentConfig, ExperimentManager


class PipelineObserver(ABC):
    """
    Abstract base class for pipeline observers (Observer Pattern).
    
    Observers are notified when pipeline completes and can perform
    actions like logging, plotting, or saving results.
    """
    
    @abstractmethod
    def on_pipeline_start(self, config: ExperimentConfig):
        """
        Called when pipeline starts execution.
        
        Args:
            config: Experiment configuration
        """
        pass
    
    @abstractmethod
    def on_pipeline_complete(
        self,
        config: ExperimentConfig,
        results_df: pd.DataFrame,
        metrics: Optional[Dict[str, Any]] = None,
    ):
        """
        Called when pipeline completes execution.
        
        Args:
            config: Experiment configuration
            results_df: DataFrame with processing results
            metrics: Optional evaluation metrics dictionary
        """
        pass
    
    @abstractmethod
    def on_pipeline_error(self, config: ExperimentConfig, error: Exception):
        """
        Called when pipeline encounters an error.
        
        Args:
            config: Experiment configuration
            error: Exception that occurred
        """
        pass


class ConsoleReporter(PipelineObserver):
    """
    Reports pipeline progress and results to console.
    
    Example:
        >>> reporter = ConsoleReporter(verbose=True)
        >>> pipeline.register_observer(reporter)
    """
    
    def __init__(self, verbose: bool = True):
        """
        Initialize console reporter.
        
        Args:
            verbose: If True, prints detailed information
        """
        self.verbose = verbose
    
    def on_pipeline_start(self, config: ExperimentConfig):
        """Print pipeline start message."""
        print(f"\n{'='*70}")
        print(f"🚀 Starting experiment: {config.name}")
        print(f"   Description: {config.description}")
        if self.verbose:
            print(f"   Velocity method: {config.velocity_config.velocity_method}")
            print(f"   Window length: {config.velocity_config.window_length_ms} ms")
            print(f"   Eye mode: {config.velocity_config.eye_mode}")
            print(f"   Threshold: {config.classifier_config.velocity_threshold_deg_per_sec} deg/s")
        print(f"{'='*70}\n")
    
    def on_pipeline_complete(
        self,
        config: ExperimentConfig,
        results_df: pd.DataFrame,
        metrics: Optional[Dict[str, Any]] = None,
    ):
        """Print pipeline completion message with metrics."""
        print(f"\n{'='*70}")
        print(f"✅ Experiment '{config.name}' completed successfully")
        print(f"   Processed {len(results_df)} samples")
        
        if metrics and self.verbose:
            print(f"\n   Key Metrics:")
            if "percentage_agreement" in metrics:
                print(f"   - Agreement: {metrics['percentage_agreement']:.2f}%")
            if "fixation_recall" in metrics:
                print(f"   - Fixation Recall: {metrics['fixation_recall']:.2f}%")
            if "saccade_recall" in metrics:
                print(f"   - Saccade Recall: {metrics['saccade_recall']:.2f}%")
            if "cohen_kappa" in metrics:
                print(f"   - Cohen's Kappa: {metrics['cohen_kappa']:.4f}")
        
        print(f"{'='*70}\n")
    
    def on_pipeline_error(self, config: ExperimentConfig, error: Exception):
        """Print error message."""
        print(f"\n{'='*70}")
        print(f"❌ Experiment '{config.name}' failed")
        print(f"   Error: {error}")
        print(f"{'='*70}\n")


class MetricsLogger(PipelineObserver):
    """
    Logs metrics to CSV file for later analysis.
    
    Example:
        >>> logger = MetricsLogger("experiments/metrics_log.csv")
        >>> pipeline.register_observer(logger)
    """
    
    def __init__(self, log_file: str):
        """
        Initialize metrics logger.
        
        Args:
            log_file: Path to CSV log file
        """
        self.log_file = Path(log_file)
        self.log_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Create header if file doesn't exist
        if not self.log_file.exists():
            self._create_header()
    
    def _create_header(self):
        """Create CSV header."""
        header = [
            "timestamp",
            "experiment_name",
            "window_ms",
            "velocity_method",
            "eye_mode",
            "smoothing",
            "threshold",
            "n_samples",
            "agreement_pct",
            "fixation_recall",
            "saccade_recall",
            "cohen_kappa",
        ]
        with open(self.log_file, "w", encoding="utf-8") as f:
            f.write(",".join(header) + "\n")
    
    def on_pipeline_start(self, config: ExperimentConfig):
        """No action on start."""
        pass
    
    def on_pipeline_complete(
        self,
        config: ExperimentConfig,
        results_df: pd.DataFrame,
        metrics: Optional[Dict[str, Any]] = None,
    ):
        """Append metrics to log file."""
        timestamp = datetime.now().isoformat()
        
        row = [
            timestamp,
            config.name,
            str(config.velocity_config.window_length_ms),
            config.velocity_config.velocity_method,
            config.velocity_config.eye_mode,
            config.velocity_config.smoothing_mode or "none",
            str(config.classifier_config.velocity_threshold_deg_per_sec),
            str(len(results_df)),
        ]
        
        if metrics:
            row.extend([
                f"{metrics.get('percentage_agreement', 0):.2f}",
                f"{metrics.get('fixation_recall', 0):.2f}",
                f"{metrics.get('saccade_recall', 0):.2f}",
                f"{metrics.get('cohen_kappa', 0):.4f}",
            ])
        else:
            row.extend(["", "", "", ""])
        
        with open(self.log_file, "a", encoding="utf-8") as f:
            f.write(",".join(row) + "\n")
        
        print(f"📊 Metrics logged to {self.log_file}")
    
    def on_pipeline_error(self, config: ExperimentConfig, error: Exception):
        """Log error to file."""
        timestamp = datetime.now().isoformat()
        error_row = [
            timestamp,
            config.name,
            str(config.velocity_config.window_length_ms),
            config.velocity_config.velocity_method,
            config.velocity_config.eye_mode,
            config.velocity_config.smoothing_mode or "none",
            str(config.classifier_config.velocity_threshold_deg_per_sec),
            "ERROR",
            str(error),
            "", "", ""
        ]
        
        with open(self.log_file, "a", encoding="utf-8") as f:
            f.write(",".join(error_row) + "\n")


class ExperimentTracker(PipelineObserver):
    """
    Automatically tracks experiments using ExperimentManager.
    
    Example:
        >>> tracker = ExperimentTracker(experiments_dir="experiments")
        >>> pipeline.register_observer(tracker)
    """
    
    def __init__(self, experiments_dir: str = "experiments"):
        """
        Initialize experiment tracker.
        
        Args:
            experiments_dir: Directory for experiment storage
        """
        self.manager = ExperimentManager(experiments_dir=experiments_dir)
    
    def on_pipeline_start(self, config: ExperimentConfig):
        """No action on start."""
        pass
    
    def on_pipeline_complete(
        self,
        config: ExperimentConfig,
        results_df: pd.DataFrame,
        metrics: Optional[Dict[str, Any]] = None,
    ):
        """Save experiment to manager."""
        self.manager.save_experiment(config, results_df, metrics)
    
    def on_pipeline_error(self, config: ExperimentConfig, error: Exception):
        """Save error information."""
        error_metrics = {
            "error": str(error),
            "error_type": type(error).__name__,
        }
        self.manager.save_experiment(config, None, error_metrics)


class ResultsPlotter(PipelineObserver):
    """
    Automatically generates plots after pipeline completion.
    
    Example:
        >>> plotter = ResultsPlotter(
        ...     output_dir="plots",
        ...     plot_types=["velocity", "classification"]
        ... )
        >>> pipeline.register_observer(plotter)
    """
    
    def __init__(
        self,
        output_dir: str,
        plot_types: Optional[list] = None,
        auto_show: bool = False,
    ):
        """
        Initialize results plotter.
        
        Args:
            output_dir: Directory where plots will be saved
            plot_types: List of plot types to generate ("velocity", "classification", "both")
            auto_show: If True, displays plots interactively
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.plot_types = plot_types or ["velocity", "classification"]
        self.auto_show = auto_show
    
    def on_pipeline_start(self, config: ExperimentConfig):
        """No action on start."""
        pass
    
    def on_pipeline_complete(
        self,
        config: ExperimentConfig,
        results_df: pd.DataFrame,
        metrics: Optional[Dict[str, Any]] = None,
    ):
        """Generate and save plots."""
        try:
            from ..evaluation.plotting import plot_velocity_only, plot_velocity_and_classification
            import matplotlib.pyplot as plt
            
            for plot_type in self.plot_types:
                if plot_type == "velocity":
                    plot_velocity_only(results_df, config.velocity_config)
                    plot_path = self.output_dir / f"{config.name}_velocity.png"
                    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
                    print(f"📈 Velocity plot saved to {plot_path}")
                
                elif plot_type == "classification":
                    plot_velocity_and_classification(results_df, config.velocity_config)
                    plot_path = self.output_dir / f"{config.name}_classification.png"
                    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
                    print(f"📈 Classification plot saved to {plot_path}")
                
                if not self.auto_show:
                    plt.close()
            
            if self.auto_show:
                plt.show()
        
        except Exception as e:
            print(f"Warning: Could not generate plots: {e}")
    
    def on_pipeline_error(self, config: ExperimentConfig, error: Exception):
        """No action on error."""
        pass
