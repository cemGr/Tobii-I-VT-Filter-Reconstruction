from __future__ import annotations

import pytest

from ivt_filter.config import OlsenVelocityConfig


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
        ({}, "TimeSymmetricWindowSelector"),
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
