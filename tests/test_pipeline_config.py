from __future__ import annotations

from types import SimpleNamespace

import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from ivt_filter.config import (
    FixationPostConfig,
    IVTClassifierConfig,
    OlsenVelocityConfig,
    PipelineConfig,
    SaccadeMergeConfig,
)
from ivt_filter.io.pipeline import IVTPipeline


def _config(*, saccade: bool = False, merge_fixations: bool = False, discard: bool = False) -> PipelineConfig:
    return PipelineConfig(
        velocity=OlsenVelocityConfig(),
        classifier=IVTClassifierConfig(),
        saccade_merge=SaccadeMergeConfig(max_saccade_block_duration_ms=12.0) if saccade else None,
        fixation_post=FixationPostConfig(
            merge_adjacent_fixations=merge_fixations,
            discard_short_fixations=discard,
        ) if merge_fixations or discard else None,
    )


@pytest.fixture
def stage_spy(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    calls: list[str] = []

    def velocity(df: pd.DataFrame, config: OlsenVelocityConfig) -> pd.DataFrame:
        calls.append("velocity")
        return df.assign(velocity_deg_per_sec=0.0)

    def classify(df: pd.DataFrame, config: IVTClassifierConfig) -> pd.DataFrame:
        calls.append("classify")
        return df.assign(ivt_sample_type="Fixation", mismatch=False)

    def saccade(df: pd.DataFrame, pred_col: str, config: SaccadeMergeConfig) -> tuple[pd.DataFrame, str]:
        calls.append("saccade")
        return df.assign(ivt_sample_type_smoothed=df[pred_col]), "ivt_sample_type_smoothed"

    def fixation(
        df: pd.DataFrame,
        pred_col: str,
        config: FixationPostConfig,
        velocity_config: OlsenVelocityConfig,
    ) -> tuple[pd.DataFrame, str]:
        calls.append("fixation")
        return df.assign(ivt_event_type_post=df[pred_col]), "ivt_event_type_post"

    monkeypatch.setattr("ivt_filter.io.pipeline.compute_olsen_velocity", velocity)
    monkeypatch.setattr(IVTPipeline, "_apply_classification", staticmethod(classify))
    monkeypatch.setattr(IVTPipeline, "_apply_saccade_smoothing", staticmethod(saccade))
    monkeypatch.setattr(IVTPipeline, "_apply_fixation_postprocessing", staticmethod(fixation))
    return calls


@pytest.mark.parametrize(
    ("config", "expected"),
    [
        (_config(), ["velocity", "classify"]),
        (_config(saccade=True), ["velocity", "classify", "saccade"]),
        (_config(merge_fixations=True), ["velocity", "classify", "fixation"]),
        (_config(discard=True), ["velocity", "classify", "fixation"]),
        (_config(saccade=True, merge_fixations=True, discard=True), ["velocity", "classify", "saccade", "fixation"]),
    ],
)
def test_config_is_single_source_of_truth_for_processing_stages(
    config: PipelineConfig,
    expected: list[str],
    stage_spy: list[str],
) -> None:
    IVTPipeline(config).process_dataframe(pd.DataFrame({"time_ms": [0.0]}))
    assert stage_spy == expected


def test_run_and_run_with_tracking_use_equivalent_core_pipeline(
    monkeypatch: pytest.MonkeyPatch,
    stage_spy: list[str],
) -> None:
    config = _config(saccade=True, merge_fixations=True, discard=True)
    monkeypatch.setattr("ivt_filter.io.pipeline.read_tsv", lambda path: pd.DataFrame({"time_ms": [0.0]}))
    pipeline = IVTPipeline(config)

    standard = pipeline.run("input.tsv", plot=False)
    tracked = pipeline.run_with_tracking("input.tsv", SimpleNamespace(name="experiment"), evaluate=False)

    assert_frame_equal(standard, tracked)
    assert stage_spy == ["velocity", "classify", "saccade", "fixation"] * 2


def test_common_example_tracking_flow_returns_velocity_and_classification_columns(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("ivt_filter.io.pipeline.read_tsv", lambda path: pd.DataFrame({"time_ms": [0.0]}))
    monkeypatch.setattr(
        "ivt_filter.io.pipeline.compute_olsen_velocity",
        lambda df, config: df.assign(velocity_deg_per_sec=0.0),
    )
    monkeypatch.setattr(
        IVTPipeline,
        "_apply_classification",
        staticmethod(lambda df, config: df.assign(ivt_sample_type="Fixation")),
    )
    pipeline = IVTPipeline(OlsenVelocityConfig(), IVTClassifierConfig())

    result = pipeline.run_with_tracking("input.tsv", SimpleNamespace(name="experiment"), evaluate=False)

    assert {"velocity_deg_per_sec", "ivt_sample_type"} <= set(result.columns)


def test_explicit_pipeline_config_can_disable_classification(
    monkeypatch: pytest.MonkeyPatch,
    stage_spy: list[str],
) -> None:
    monkeypatch.setattr("ivt_filter.io.pipeline.read_tsv", lambda path: pd.DataFrame({"time_ms": [0.0]}))
    pipeline = IVTPipeline(
        PipelineConfig(
            velocity=OlsenVelocityConfig(),
            classifier=IVTClassifierConfig(),
            classify=False,
        )
    )

    result = pipeline.run_with_tracking("input.tsv", SimpleNamespace(name="experiment"), evaluate=False)

    assert pipeline.config.classify is False
    assert stage_spy == ["velocity"]
    assert "ivt_sample_type" not in result.columns


def test_legacy_run_flags_are_translated_into_pipeline_config(
    monkeypatch: pytest.MonkeyPatch,
    stage_spy: list[str],
) -> None:
    monkeypatch.setattr("ivt_filter.io.pipeline.read_tsv", lambda path: pd.DataFrame({"time_ms": [0.0]}))
    pipeline = IVTPipeline(OlsenVelocityConfig(), IVTClassifierConfig())

    pipeline.run(
        "input.tsv",
        post_smoothing_ms=12.0,
        merge_close_fixations=True,
        discard_short_fixations=True,
        plot=False,
    )

    assert stage_spy == ["velocity", "classify", "saccade", "fixation"]
    assert pipeline.config.saccade_merge is None
    assert pipeline.config.fixation_post is None


def test_legacy_run_flags_reject_contradictory_stage_configuration() -> None:
    pipeline = IVTPipeline(_config(saccade=True, merge_fixations=True))

    with pytest.raises(ValueError, match="post_smoothing_ms contradicts"):
        pipeline.run("unused.tsv", post_smoothing_ms=20.0, plot=False)
    with pytest.raises(ValueError, match="merge_close_fixations=False contradicts"):
        pipeline.run("unused.tsv", merge_close_fixations=False, plot=False)


def test_pipeline_config_rejects_inactive_optional_stage_config() -> None:
    with pytest.raises(ValueError, match="must enable at least one operation"):
        PipelineConfig(
            velocity=OlsenVelocityConfig(),
            classifier=IVTClassifierConfig(),
            fixation_post=FixationPostConfig(),
        )
