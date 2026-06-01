# ivt_filter/io/pipeline.py
"""IVT processing pipeline orchestration.

Separates pipeline orchestration from CLI and configuration (SRP).
Implements Observer Pattern for automatic experiment tracking.
"""
from __future__ import annotations

import logging
from typing import Optional, List, Dict, Any, TYPE_CHECKING

import pandas as pd
from dataclasses import replace

from ..config import (
    OlsenVelocityConfig,
    IVTClassifierConfig,
    SaccadeMergeConfig,
    FixationPostConfig,
    PipelineConfig,
)
from .io import read_tsv, write_tsv
from ..processing.velocity import compute_olsen_velocity
from ..processing.classification import apply_ivt_classifier, expand_gt_events_to_samples
from ..postprocessing import merge_short_saccade_blocks, apply_fixation_postprocessing
from ..evaluation.evaluation import evaluate_ivt_vs_ground_truth, compute_ivt_metrics

if TYPE_CHECKING:
    from .observers import PipelineObserver


logger = logging.getLogger(__name__)


class IVTPipeline:
    """Orchestrates the complete IVT processing pipeline with Observer Pattern support.
    
    Responsibilities:
        - Execute processing steps in correct order
        - Handle conditional execution based on options
        - Coordinate between modules
        - Report progress and statistics
        - Notify observers of pipeline events (Observer Pattern)
    
    Example:
        >>> from ivt_filter.io.observers import ConsoleReporter, MetricsLogger, ExperimentTracker
        >>> from ivt_filter.evaluation.experiment import ExperimentConfig
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
        velocity_config: OlsenVelocityConfig | PipelineConfig,
        classifier_config: Optional[IVTClassifierConfig] = None,
        saccade_merge_config: Optional[SaccadeMergeConfig] = None,
        fixation_post_config: Optional[FixationPostConfig] = None,
    ):
        """Create a pipeline.

        Passing the individual stage configurations is a compatibility adapter.
        New code should pass one :class:`PipelineConfig` as ``velocity_config``.
        """
        if isinstance(velocity_config, PipelineConfig):
            if classifier_config or saccade_merge_config or fixation_post_config:
                raise ValueError("PipelineConfig cannot be combined with individual stage configs")
            self.config = velocity_config
        else:
            if classifier_config is None:
                raise ValueError("classifier_config is required with velocity_config")
            self.config = PipelineConfig(
                velocity=velocity_config,
                classifier=classifier_config,
                classify=True,
                saccade_merge=saccade_merge_config,
                fixation_post=fixation_post_config,
            )
        self._observers: List[PipelineObserver] = []

    @property
    def velocity_config(self) -> OlsenVelocityConfig:
        return self.config.velocity

    @property
    def classifier_config(self) -> IVTClassifierConfig:
        return self.config.classifier

    @property
    def saccade_merge_config(self) -> Optional[SaccadeMergeConfig]:
        return self.config.saccade_merge

    @property
    def fixation_post_config(self) -> Optional[FixationPostConfig]:
        return self.config.fixation_post
    
    def register_observer(self, observer: PipelineObserver) -> None:
        """
        Register an observer to receive pipeline notifications.
        
        Args:
            observer: PipelineObserver instance
        """
        if observer not in self._observers:
            self._observers.append(observer)
    
    def unregister_observer(self, observer: PipelineObserver) -> None:
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
                logger.warning("Observer %s failed on start: %s", type(observer).__name__, e)
    
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
                logger.warning("Observer %s failed on complete: %s", type(observer).__name__, e)
    
    def _notify_error(self, config: Any, error: Exception) -> None:
        """Notify all observers that pipeline encountered an error."""
        for observer in self._observers:
            try:
                observer.on_pipeline_error(config, error)
            except Exception as e:
                logger.warning("Observer %s failed on error: %s", type(observer).__name__, e)
    
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
            >>> from ivt_filter.evaluation.experiment import ExperimentConfig
            >>> config = ExperimentConfig(
            ...     name="olsen2d_20ms_30deg",
            ...     description="Baseline configuration",
            ...     velocity_config=velocity_config,
            ...     classifier_config=classifier_config
            ... )
            >>> df = pipeline.run_with_tracking(input_path, config)
        """
        if hasattr(config, "metadata"):
            config.metadata = dict(config.metadata or {})
            config.metadata["tracking"] = {
                "input_path": input_path,
                "api_parameters": {
                    "input_path": input_path,
                    "output_path": output_path,
                    "evaluate": evaluate,
                    "evaluate_exclude_calibration": evaluate_exclude_calibration,
                },
            }
        self._notify_start(config)
        
        try:
            # Run standard pipeline
            df = self.run(
                input_path=input_path,
                output_path=output_path,
                evaluate=False,  # We'll compute metrics separately
                plot=False,  # Observers handle plotting
            )
            
            # Compute metrics if requested
            metrics = None
            if evaluate:
                try:
                    metrics = compute_ivt_metrics(
                        df,
                        pred_col=self._prediction_column(self.config),
                        exclude_calibration=evaluate_exclude_calibration,
                    )
                except Exception as e:
                    logger.warning("Could not compute metrics: %s", e)
            
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
        classify: Optional[bool] = None,
        evaluate: bool = False,
        post_smoothing_ms: Optional[float] = None,
        merge_close_fixations: Optional[bool] = None,
        discard_short_fixations: Optional[bool] = None,
        plot: bool = True,
        with_events: bool = False,
        evaluate_exclude_calibration: bool = False,
    ) -> pd.DataFrame:
        """Run the file pipeline.

        ``classify`` and the three post-processing arguments are compatibility
        adapters for older callers. They are translated into a temporary
        :class:`PipelineConfig`; contradictory combinations are rejected.
        New callers should configure stages on ``self.config`` only.
        """
        config = self._config_with_legacy_run_arguments(
            classify=classify,
            post_smoothing_ms=post_smoothing_ms,
            merge_close_fixations=merge_close_fixations,
            discard_short_fixations=discard_short_fixations,
        )
        if evaluate and not config.classify:
            # Compatibility with the historical evaluate=True behavior.
            config = replace(config, classify=True)
        df = self.process_dataframe(read_tsv(input_path), config=config)

        if output_path:
            write_tsv(df, output_path)
        if evaluate:
            self._evaluate(df, config, evaluate_exclude_calibration)
        if plot:
            self._generate_plots(df, with_events)
        return df

    def process_dataframe(
        self,
        df: pd.DataFrame,
        *,
        config: Optional[PipelineConfig] = None,
    ) -> pd.DataFrame:
        """Execute configured core stages on a DataFrame without I/O or plotting."""
        config = config or self.config
        df = compute_olsen_velocity(df, config.velocity)
        if (
            config.classifier.enable_confident_switch
            and config.classifier.confident_switch_method
            and config.classifier.confident_switch_method != config.velocity.velocity_method
        ):
            alt_cfg = replace(config.velocity, velocity_method=config.classifier.confident_switch_method)
            df_alt = compute_olsen_velocity(df.copy(), alt_cfg)
            df["velocity_alt_deg_per_sec"] = df_alt["velocity_deg_per_sec"]

        if not config.classify:
            return df

        df = self._apply_classification(df, config.classifier)
        pred_sample_col = "ivt_sample_type"
        if config.saccade_merge:
            df, pred_sample_col = self._apply_saccade_smoothing(df, pred_sample_col, config.saccade_merge)
        if config.fixation_post:
            df, _ = self._apply_fixation_postprocessing(
                df,
                pred_sample_col,
                config.fixation_post,
                config.velocity,
            )
        return df

    def _config_with_legacy_run_arguments(
        self,
        *,
        classify: Optional[bool],
        post_smoothing_ms: Optional[float],
        merge_close_fixations: Optional[bool],
        discard_short_fixations: Optional[bool],
    ) -> PipelineConfig:
        """Translate deprecated run-time stage flags into one pipeline config."""
        config = self.config
        if classify is not None:
            if not classify and (config.saccade_merge or config.fixation_post):
                raise ValueError("classify=False contradicts configured post-processing")
            config = replace(config, classify=classify)

        if post_smoothing_ms is not None:
            if post_smoothing_ms <= 0:
                if config.saccade_merge:
                    raise ValueError("post_smoothing_ms disables configured saccade merge")
            elif config.saccade_merge:
                if config.saccade_merge.max_saccade_block_duration_ms != post_smoothing_ms:
                    raise ValueError("post_smoothing_ms contradicts configured saccade merge")
            else:
                config = replace(
                    config,
                    classify=True,
                    saccade_merge=SaccadeMergeConfig(max_saccade_block_duration_ms=post_smoothing_ms),
                )

        if merge_close_fixations is not None or discard_short_fixations is not None:
            fixation = config.fixation_post or FixationPostConfig()
            for argument, configured, name in (
                (merge_close_fixations, fixation.merge_adjacent_fixations, "merge_close_fixations"),
                (discard_short_fixations, fixation.discard_short_fixations, "discard_short_fixations"),
            ):
                if argument is False and configured:
                    raise ValueError(f"{name}=False contradicts configured fixation post-processing")
            fixation = replace(
                fixation,
                merge_adjacent_fixations=fixation.merge_adjacent_fixations if merge_close_fixations is None else merge_close_fixations,
                discard_short_fixations=fixation.discard_short_fixations if discard_short_fixations is None else discard_short_fixations,
            )
            if fixation.merge_adjacent_fixations or fixation.discard_short_fixations:
                config = replace(config, classify=True, fixation_post=fixation)
        return config

    def _evaluate(self, df: pd.DataFrame, config: PipelineConfig, exclude_calibration: bool) -> None:
        """Evaluate the final configured prediction column."""
        evaluate_ivt_vs_ground_truth(
            df,
            pred_col=self._prediction_column(config),
            exclude_calibration=exclude_calibration,
        )

    @staticmethod
    def _prediction_column(config: PipelineConfig) -> str:
        if config.fixation_post:
            return "ivt_event_type_post"
        if config.saccade_merge:
            sample_col = config.saccade_merge.use_sample_type_column
            return f"{sample_col}_smoothed" if sample_col else "ivt_event_type_smoothed"
        return "ivt_sample_type"

    def _apply_classification(self, df: pd.DataFrame, classifier_config: IVTClassifierConfig) -> pd.DataFrame:
        """Apply IVT classification and expand GT events."""
        df = apply_ivt_classifier(df, classifier_config)
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
        saccade_merge_config: SaccadeMergeConfig,
    ) -> tuple[pd.DataFrame, str]:
        """Apply configured saccade smoothing."""
        df, merge_stats = merge_short_saccade_blocks(df, cfg=saccade_merge_config)
        
        # Determine which column to use
        if saccade_merge_config.use_sample_type_column:
            pred_sample_col = saccade_merge_config.use_sample_type_column + "_smoothed"
        else:
            pred_sample_col = "ivt_event_type_smoothed"
        
        logger.info(
            "[Post-Processing] merged short saccade blocks: %s / %s blocks, %s samples.",
            merge_stats["n_blocks_merged"],
            merge_stats["n_blocks_total"],
            merge_stats["n_samples_merged"],
        )
        
        return df, pred_sample_col

    def _apply_fixation_postprocessing(
        self,
        df: pd.DataFrame,
        pred_sample_col: str,
        fixation_post_config: FixationPostConfig,
        velocity_config: OlsenVelocityConfig,
    ) -> tuple[pd.DataFrame, str]:
        """Apply Tobii-like fixation post-processing.
        
        Returns:
            (df, updated_pred_col): Updated DataFrame and the new prediction column name
        """
        use_ray3d = velocity_config.velocity_method == "ray3d"
        
        df, fix_stats = apply_fixation_postprocessing(
            df,
            cfg=fixation_post_config,
            sample_col=pred_sample_col,
            time_col="time_ms",
            x_col="smoothed_x_mm",
            y_col="smoothed_y_mm",
            eye_z_col="eye_z_mm",
            event_type_col="ivt_event_type_post",
            event_index_col="ivt_event_index_post",
            use_ray3d=use_ray3d,
        )
        
        logger.info(
            "[FixationPost] merged_pairs=%s, gap_samples_to_fixation=%s, "
            "discarded_fixations=%s, discarded_samples=%s",
            fix_stats.get("merged_pairs", 0),
            fix_stats.get("gap_samples_to_fixation", 0),
            fix_stats.get("discarded_fixations", 0),
            fix_stats.get("discarded_samples", 0),
        )
        
        return df, "ivt_event_type_post"

    def _generate_plots(self, df: pd.DataFrame, with_events: bool) -> None:
        """Generate visualization plots, loading matplotlib only when requested."""
        from ..evaluation.plotting import plot_velocity_only, plot_velocity_and_classification

        if with_events:
            plot_velocity_and_classification(df, self.velocity_config)
        else:
            plot_velocity_only(df, self.velocity_config)
