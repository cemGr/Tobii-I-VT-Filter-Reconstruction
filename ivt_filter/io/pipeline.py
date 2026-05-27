# ivt_filter/io/pipeline.py
"""IVT processing pipeline orchestration.

Separates pipeline orchestration from CLI and configuration (SRP).
Implements Observer Pattern for automatic experiment tracking.
"""
from __future__ import annotations

from typing import Optional, List, Dict, Any

import pandas as pd
from dataclasses import replace

from ..config import (
    OlsenVelocityConfig,
    IVTClassifierConfig,
    SaccadeMergeConfig,
    FixationPostConfig,
)
from .io import read_tsv, write_tsv
from ..processing.velocity import compute_olsen_velocity
from ..processing.classification import apply_ivt_classifier, expand_gt_events_to_samples
from ..postprocess import merge_short_saccade_blocks, apply_fixation_postprocessing
from ..evaluation.evaluation import evaluate_ivt_vs_ground_truth, compute_ivt_metrics
from ..evaluation.plotting import plot_velocity_only, plot_velocity_and_classification


class IVTPipeline:
    """Orchestrates the complete IVT processing pipeline with Observer Pattern support.
    
    Responsibilities:
        - Execute processing steps in correct order
        - Handle conditional execution based on options
        - Coordinate between modules
        - Report progress and statistics
        - Notify observers of pipeline events (Observer Pattern)
    
    Example:
        >>> from observers import ConsoleReporter, MetricsLogger, ExperimentTracker
        >>> from experiment import ExperimentConfig
        >>> 
        >>> pipeline = IVTPipeline(velocity_config, classifier_config)
        >>> pipeline.register_observer(ConsoleReporter())
        >>> pipeline.register_observer(MetricsLogger("logs/metrics.csv"))
        >>> pipeline.register_observer(ExperimentTracker())
        >>> 
        >>> # Run with automatic tracking
        >>> config = ExperimentConfig(name="exp1", ...)
        >>> df = pipeline.run_with_tracking(input_path, config)
    """

    def __init__(
        self,
        velocity_config: OlsenVelocityConfig,
        classifier_config: IVTClassifierConfig,
        saccade_merge_config: Optional[SaccadeMergeConfig] = None,
        fixation_post_config: Optional[FixationPostConfig] = None,
    ):
        self.velocity_config = velocity_config
        self.classifier_config = classifier_config
        self.saccade_merge_config = saccade_merge_config
        self.fixation_post_config = fixation_post_config
        self._observers: List[Any] = []  # List of PipelineObserver instances
    
    def register_observer(self, observer: Any) -> None:
        """
        Register an observer to receive pipeline notifications.
        
        Args:
            observer: PipelineObserver instance
        """
        if observer not in self._observers:
            self._observers.append(observer)
    
    def unregister_observer(self, observer: Any) -> None:
        """
        Unregister an observer.
        
        Args:
            observer: PipelineObserver instance to remove
        """
        if observer in self._observers:
            self._observers.remove(observer)
    
    def _notify_start(self, config: Any) -> None:
        """Notify all observers that pipeline is starting."""
        for observer in self._observers:
            try:
                observer.on_pipeline_start(config)
            except Exception as e:
                print(f"Warning: Observer {type(observer).__name__} failed on start: {e}")
    
    def _notify_complete(
        self,
        config: Any,
        results_df: pd.DataFrame,
        metrics: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Notify all observers that pipeline completed successfully."""
        for observer in self._observers:
            try:
                observer.on_pipeline_complete(config, results_df, metrics)
            except Exception as e:
                print(f"Warning: Observer {type(observer).__name__} failed on complete: {e}")
    
    def _notify_error(self, config: Any, error: Exception) -> None:
        """Notify all observers that pipeline encountered an error."""
        for observer in self._observers:
            try:
                observer.on_pipeline_error(config, error)
            except Exception as e:
                print(f"Warning: Observer {type(observer).__name__} failed on error: {e}")
    
    def run_with_tracking(
        self,
        input_path: str,
        config: Any,  # ExperimentConfig
        output_path: Optional[str] = None,
        evaluate: bool = True,
        evaluate_exclude_calibration: bool = False,
    ) -> pd.DataFrame:
        """
        Run pipeline with automatic experiment tracking via observers.
        
        This method wraps the standard run() method with observer notifications,
        making it ideal for systematic experiments.
        
        Args:
            input_path: Path to input TSV file
            config: ExperimentConfig with complete experiment setup
            output_path: Optional path to output TSV file
            evaluate: Whether to compute evaluation metrics
        
        Returns:
            Processed DataFrame
        
        Example:
            >>> from experiment import ExperimentConfig
            >>> config = ExperimentConfig(
            ...     name="olsen2d_20ms_30deg",
            ...     description="Baseline configuration",
            ...     velocity_config=velocity_config,
            ...     classifier_config=classifier_config
            ... )
            >>> df = pipeline.run_with_tracking(input_path, config)
        """
        self._notify_start(config)
        
        try:
            # Run standard pipeline
            df = self.run(
                input_path=input_path,
                output_path=output_path,
                classify=True,
                evaluate=False,  # We'll compute metrics separately
                plot=False,  # Observers handle plotting
            )
            
            # Compute metrics if requested
            metrics = None
            if evaluate:
                try:
                    metrics = compute_ivt_metrics(
                        df,
                        pred_col="ivt_sample_type",
                        exclude_calibration=evaluate_exclude_calibration,
                    )
                except Exception as e:
                    print(f"Warning: Could not compute metrics: {e}")
            
            # Notify observers of success
            self._notify_complete(config, df, metrics)
            
            return df
        
        except Exception as e:
            self._notify_error(config, e)
            raise

    def run(
        self,
        input_path: str,
        output_path: Optional[str] = None,
        classify: bool = False,
        evaluate: bool = False,
        post_smoothing_ms: Optional[float] = None,
        merge_close_fixations: bool = False,
        discard_short_fixations: bool = False,
        plot: bool = True,
        with_events: bool = False,
        evaluate_exclude_calibration: bool = False,
    ) -> pd.DataFrame:
        """Run the complete IVT pipeline.
        
        Args:
            input_path: Path to input TSV file
            output_path: Optional path to output TSV file
            classify: Whether to apply IVT classification
            evaluate: Whether to evaluate against ground truth
            post_smoothing_ms: Optional saccade smoothing duration
            merge_close_fixations: Whether to merge close fixations
            discard_short_fixations: Whether to discard short fixations
            plot: Whether to generate plots
            with_events: Whether to plot with GT events
            
        Returns:
            Processed DataFrame
        """
        # Step 1: Load data
        df = read_tsv(input_path)
        
        # Step 2: Compute velocity
        df = compute_olsen_velocity(df, self.velocity_config)

        # Optional: compute alternative velocity for confident switch
        if (
            getattr(self.classifier_config, "enable_confident_switch", False)
            and getattr(self.classifier_config, "confident_switch_method", None)
            and self.classifier_config.confident_switch_method != self.velocity_config.velocity_method
        ):
            alt_cfg = replace(self.velocity_config, velocity_method=self.classifier_config.confident_switch_method)
            df_alt = compute_olsen_velocity(df.copy(), alt_cfg)
            df["velocity_alt_deg_per_sec"] = df_alt["velocity_deg_per_sec"]
        
        pred_sample_col: Optional[str] = None
        pred_col_for_eval: Optional[str] = None
        
        # Step 3: Classification (if requested)
        if classify or evaluate or merge_close_fixations or discard_short_fixations:
            df = self._apply_classification(df)
            pred_sample_col = "ivt_sample_type"
            pred_col_for_eval = pred_sample_col
        
        # Step 4: Saccade smoothing (if configured)
        if post_smoothing_ms and post_smoothing_ms > 0 and pred_sample_col:
            df, pred_sample_col = self._apply_saccade_smoothing(
                df, pred_sample_col, post_smoothing_ms
            )
            pred_col_for_eval = pred_sample_col
        
        # Step 5: Fixation post-processing (if configured)
        if pred_sample_col and (merge_close_fixations or discard_short_fixations):
            df, pred_col_for_eval = self._apply_fixation_postprocessing(df, pred_sample_col)
        
        # Step 6: Write output (if requested)
        if output_path:
            write_tsv(df, output_path)
        
        # Step 7: Evaluation (if requested)
        if evaluate and pred_col_for_eval:
            evaluate_ivt_vs_ground_truth(
                df,
                pred_col=pred_col_for_eval,
                exclude_calibration=evaluate_exclude_calibration,
            )
        
        # Step 8: Plotting (if requested)
        if plot:
            self._generate_plots(df, with_events)
        
        return df

    def _apply_classification(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply IVT classification and expand GT events."""
        df = apply_ivt_classifier(df, self.classifier_config)  # type: ignore[arg-type]
        df = expand_gt_events_to_samples(df)
        
        # Add mismatch column: True if GT != Predicted (for valid Fixation/Saccade pairs)
        gt_col = "gt_event_type" if "gt_event_type" in df.columns else "Eye movement type"
        pred_col = "ivt_sample_type"
        
        if gt_col in df.columns and pred_col in df.columns:
            valid_labels = {"Fixation", "Saccade"}
            df["mismatch"] = (
                df[gt_col].isin(valid_labels) 
                & df[pred_col].isin(valid_labels) 
                & (df[gt_col].astype(str) != df[pred_col].astype(str))
            )
        else:
            df["mismatch"] = False
        
        return df

    def _apply_saccade_smoothing(
        self,
        df: pd.DataFrame,
        pred_sample_col: str,
        post_smoothing_ms: float,
    ) -> tuple[pd.DataFrame, str]:
        """Apply GT-guided saccade smoothing."""
        if not self.saccade_merge_config:
            return df, pred_sample_col
        
        df, merge_stats = merge_short_saccade_blocks(df, cfg=self.saccade_merge_config)
        
        # Determine which column to use
        if self.saccade_merge_config.use_sample_type_column:
            pred_sample_col = self.saccade_merge_config.use_sample_type_column + "_smoothed"
        else:
            pred_sample_col = "ivt_event_type_smoothed"
        
        print(
            f"[Post-Processing] merged short saccade blocks: "
            f"{merge_stats['n_blocks_merged']} / {merge_stats['n_blocks_total']} blocks, "
            f"{merge_stats['n_samples_merged']} samples."
        )
        
        return df, pred_sample_col

    def _apply_fixation_postprocessing(
        self,
        df: pd.DataFrame,
        pred_sample_col: str,
    ) -> tuple[pd.DataFrame, str]:
        """Apply Tobii-like fixation post-processing.
        
        Returns:
            (df, updated_pred_col): Updated DataFrame and the new prediction column name
        """
        if not self.fixation_post_config:
            return df, pred_sample_col
        
        use_ray3d = self.velocity_config.velocity_method == "ray3d"
        
        df, fix_stats = apply_fixation_postprocessing(
            df,
            cfg=self.fixation_post_config,
            sample_col=pred_sample_col,
            time_col="time_ms",
            x_col="smoothed_x_mm",
            y_col="smoothed_y_mm",
            eye_z_col="eye_z_mm",
            event_type_col="ivt_event_type_post",
            event_index_col="ivt_event_index_post",
            use_ray3d=use_ray3d,
        )
        
        print(
            f"[FixationPost] merged_pairs={fix_stats.get('merged_pairs', 0)}, "
            f"gap_samples_to_fixation={fix_stats.get('gap_samples_to_fixation', 0)}, "
            f"discarded_fixations={fix_stats.get('discarded_fixations', 0)}, "
            f"discarded_samples={fix_stats.get('discarded_samples', 0)}"
        )
        
        return df, "ivt_event_type_post"

    def _generate_plots(self, df: pd.DataFrame, with_events: bool) -> None:
        """Generate visualization plots."""
        if with_events:
            plot_velocity_and_classification(df, self.velocity_config)
        else:
            plot_velocity_only(df, self.velocity_config)
