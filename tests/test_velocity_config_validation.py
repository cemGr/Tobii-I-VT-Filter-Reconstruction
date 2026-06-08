from __future__ import annotations

import pytest

from ivt_filter.config import (
    AsymmetricNeighborWindowPolicy,
    FixedSampleWindowPolicy,
    OlsenVelocityConfig,
    SampleSymmetricWindowPolicy,
    ShiftedValidWindowPolicy,
    TimeSymmetricWindowPolicy,
    TobiiWindowPolicy,
    WindowPolicy,
)


@pytest.mark.parametrize(
    "selector_mode",
    [
        {"asymmetric_neighbor_window": True},
        {"shifted_valid_window": True},
        {"fixed_window_samples": 5},
        {"auto_fixed_window_from_ms": True},
        {"symmetric_round_window": True},
        {"sample_symmetric_window": True},
    ],
)
def test_tobii_window_mode_rejects_non_tobii_selector_modes(
    selector_mode: dict[str, object],
) -> None:
    with pytest.raises(ValueError, match="tobii_window_mode takes precedence"):
        OlsenVelocityConfig(tobii_window_mode=True, **selector_mode)


@pytest.mark.parametrize(
    "selector_mode",
    [
        {"fixed_window_samples": 5},
        {"auto_fixed_window_from_ms": True},
        {"symmetric_round_window": True},
        {"shifted_valid_window": True},
        {"sample_symmetric_window": True},
    ],
)
def test_asymmetric_neighbor_window_rejects_shadowed_selector_modes(
    selector_mode: dict[str, object],
) -> None:
    with pytest.raises(
        ValueError, match="asymmetric_neighbor_window takes precedence"
    ):
        OlsenVelocityConfig(asymmetric_neighbor_window=True, **selector_mode)


@pytest.mark.parametrize(
    "selector_mode",
    [
        {"sample_symmetric_window": True},
        {"asymmetric_neighbor_window": True},
        {"tobii_window_mode": True},
    ],
)
def test_shifted_valid_window_rejects_incompatible_selector_flags(
    selector_mode: dict[str, object],
) -> None:
    with pytest.raises(ValueError, match="takes precedence"):
        OlsenVelocityConfig(shifted_valid_window=True, **selector_mode)


@pytest.mark.parametrize(
    "fixed_window_mode",
    [
        {"fixed_window_samples": 5},
        {"auto_fixed_window_from_ms": True},
        {"symmetric_round_window": True},
    ],
)
def test_fixed_window_modes_reject_sample_symmetric_window(
    fixed_window_mode: dict[str, object],
) -> None:
    with pytest.raises(
        ValueError, match="takes precedence over sample_symmetric_window"
    ):
        OlsenVelocityConfig(sample_symmetric_window=True, **fixed_window_mode)


@pytest.mark.parametrize(
    "fixed_window_mode",
    [
        {"fixed_window_samples": 5},
        {"auto_fixed_window_from_ms": True},
        {"symmetric_round_window": True},
    ],
)
def test_shifted_valid_window_accepts_compatible_fixed_window_modes(
    fixed_window_mode: dict[str, object],
) -> None:
    cfg = OlsenVelocityConfig(shifted_valid_window=True, **fixed_window_mode)
    assert cfg.shifted_valid_window is True


@pytest.mark.parametrize(
    ("config_kwargs", "selector_type"),
    [
        ({"sample_symmetric_window": True}, "SampleSymmetricWindowSelector"),
        ({"fixed_window_samples": 5}, "FixedSampleSymmetricWindowSelector"),
        ({"shifted_valid_window": True}, "TimeBasedShiftedValidWindowSelector"),
        (
            {"shifted_valid_window": True, "fixed_window_samples": 5},
            "ShiftedValidWindowSelector",
        ),
        ({"asymmetric_neighbor_window": True}, "AsymmetricNeighborWindowSelector"),
    ],
)
def test_compatible_legacy_window_modes_keep_existing_selector_behavior(
    config_kwargs: dict[str, object], selector_type: str
) -> None:
    from ivt_filter.processing.velocity import make_window_selector

    selector = make_window_selector(OlsenVelocityConfig(**config_kwargs))

    assert type(selector).__name__ == selector_type


def test_default_window_policy_is_unresolved_tobii_policy() -> None:
    cfg = OlsenVelocityConfig()

    assert cfg.window_policy == TobiiWindowPolicy(sample_interval_ms=None)


def test_unresolved_default_tobii_policy_requires_sampling_resolution() -> None:
    from ivt_filter.processing.velocity import make_window_selector

    with pytest.raises(ValueError, match="resolved sample_interval_ms"):
        make_window_selector(OlsenVelocityConfig())

@pytest.mark.parametrize(
    ("policy", "selector_type"),
    [
        (TimeSymmetricWindowPolicy(), "TimeSymmetricWindowSelector"),
        (SampleSymmetricWindowPolicy(), "SampleSymmetricWindowSelector"),
        (FixedSampleWindowPolicy(samples=5), "FixedSampleSymmetricWindowSelector"),
        (
            ShiftedValidWindowPolicy(samples=5, fallback="unclassified"),
            "ShiftedValidWindowSelector",
        ),
        (ShiftedValidWindowPolicy(), "TimeBasedShiftedValidWindowSelector"),
        (AsymmetricNeighborWindowPolicy(), "AsymmetricNeighborWindowSelector"),
        (TobiiWindowPolicy(sample_interval_ms=8.33), "TobiiGazeVelocityWindowSelector"),
    ],
)
def test_explicit_window_policy_variants_dispatch_directly(
    policy: WindowPolicy, selector_type: str
) -> None:
    from ivt_filter.processing.velocity import make_window_selector

    selector = make_window_selector(OlsenVelocityConfig(window_policy=policy))

    assert type(selector).__name__ == selector_type


def test_experiment_json_persists_and_restores_normalized_window_policy() -> None:
    from ivt_filter.config import IVTClassifierConfig
    from ivt_filter.evaluation.experiment import ExperimentConfig

    experiment = ExperimentConfig(
        name="fixed-window",
        description="policy persistence",
        velocity_config=OlsenVelocityConfig(
            window_policy=FixedSampleWindowPolicy(samples=5, symmetric_round=True)
        ),
        classifier_config=IVTClassifierConfig(),
    )

    serialized = experiment.to_dict()
    restored = ExperimentConfig.from_dict(serialized)

    assert serialized["velocity_config"]["window_policy"] == {
        "samples": 5,
        "derive_from_window_ms": False,
        "symmetric_round": True,
        "kind": "fixed_sample",
    }
    assert restored.velocity_config.window_policy == FixedSampleWindowPolicy(
        samples=5, symmetric_round=True
    )

@pytest.mark.parametrize(
    "cli_flags",
    [
        ["--sample-symmetric-window", "--fixed-window-samples", "5"],
        ["--sample-symmetric-window", "--shifted-valid-window"],
        ["--sample-symmetric-window", "--asymmetric-neighbor-window"],
        ["--fixed-window-samples", "5", "--asymmetric-neighbor-window"],
        ["--shifted-valid-window", "--asymmetric-neighbor-window"],
        ["--time-symmetric-window", "--asymmetric-neighbor-window"],
        ["--time-symmetric-window", "--fixed-window-samples", "5"],
    ],
)
def test_config_builder_rejects_contradictory_legacy_cli_flags(
    cli_flags: list[str],
) -> None:
    from ivt_filter.cli import build_arg_parser
    from ivt_filter.config import ConfigBuilder

    args = build_arg_parser().parse_args(["--input", "gaze.tsv", *cli_flags])

    with pytest.raises(ValueError, match="Contradictory legacy window flags"):
        ConfigBuilder.build_velocity_config(args)


def test_config_builder_translates_legacy_cli_flags_to_one_policy() -> None:
    from ivt_filter.cli import build_arg_parser
    from ivt_filter.config import ConfigBuilder

    args = build_arg_parser().parse_args(
        ["--input", "gaze.tsv", "--fixed-window-samples", "5"]
    )

    config = ConfigBuilder.build_velocity_config(args)

    assert config.window_policy == FixedSampleWindowPolicy(samples=5)
    assert config.fixed_window_samples is None


def test_config_builder_uses_default_tobii_policy_without_window_flags() -> None:
    from ivt_filter.cli import build_arg_parser
    from ivt_filter.config import ConfigBuilder

    args = build_arg_parser().parse_args(["--input", "gaze.tsv"])

    config = ConfigBuilder.build_velocity_config(args)

    assert config.window_policy == TobiiWindowPolicy(sample_interval_ms=None)


def test_config_builder_can_request_legacy_time_symmetric_policy() -> None:
    from ivt_filter.cli import build_arg_parser
    from ivt_filter.config import ConfigBuilder

    args = build_arg_parser().parse_args(
        ["--input", "gaze.tsv", "--time-symmetric-window"]
    )

    config = ConfigBuilder.build_velocity_config(args)

    assert config.window_policy == TimeSymmetricWindowPolicy()


@pytest.mark.parametrize(
    "config_kwargs",
    [
        {"window_length_ms": 0.0},
        {"min_dt_ms": 0.0},
        {"fixed_window_samples": 2},
        {"fixed_window_samples": 4},
        {"smoothing_window_samples": 0},
        {"smoothing_window_samples": 2},
        {"smoothing_min_samples": 0},
        {"smoothing_min_samples": 1.5},
        {"smoothing_expansion_radius": -1},
        {"smoothing_expansion_radius": 1.5},
        {"gap_fill_max_gap_ms": -1.0},
        {"tobii_window_mode": True, "tobii_sample_interval_ms": 0.0},
        {"window_policy": TobiiWindowPolicy(sample_interval_ms=-1.0)},
    ],
)
def test_velocity_config_rejects_numeric_boundary_values(
    config_kwargs: dict[str, object],
) -> None:
    with pytest.raises(ValueError):
        OlsenVelocityConfig(**config_kwargs)


@pytest.mark.parametrize(
    "config_kwargs",
    [
        {"time_unit": "seconds"},
        {"eye_mode": "both"},
        {"smoothing_mode": "average"},
        {"sampling_rate_method": "last_100"},
        {"dt_calculation_method": "minimum"},
        {"coordinate_rounding": "truncate"},
        {"velocity_method": "unknown"},
        {"shifted_valid_fallback": "nearest"},
    ],
)
def test_velocity_config_rejects_unknown_enum_like_values(
    config_kwargs: dict[str, object],
) -> None:
    with pytest.raises(ValueError):
        OlsenVelocityConfig(**config_kwargs)


def test_velocity_config_accepts_minimum_valid_values() -> None:
    config = OlsenVelocityConfig(
        window_length_ms=0.001,
        min_dt_ms=0.001,
        fixed_window_samples=3,
        smoothing_window_samples=1,
        smoothing_min_samples=1,
        smoothing_expansion_radius=0,
        gap_fill_max_gap_ms=0.0,
    )

    assert config.fixed_window_samples == 3
    assert config.smoothing_window_samples == 1


def test_velocity_config_accepts_even_fixed_window_when_asymmetry_is_enabled() -> None:
    config = OlsenVelocityConfig(fixed_window_samples=4, allow_asymmetric_window=True)

    assert config.fixed_window_samples == 4
