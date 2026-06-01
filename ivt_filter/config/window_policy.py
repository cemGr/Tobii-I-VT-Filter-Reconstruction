"""Explicit velocity-window selection policies and legacy flag translation."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Mapping, Optional, Union


@dataclass(frozen=True)
class TimeSymmetricWindowPolicy:
    """Select endpoints symmetrically inside the configured time span."""

    kind: Literal["time_symmetric"] = "time_symmetric"


@dataclass(frozen=True)
class SampleSymmetricWindowPolicy:
    """Select an equal number of samples on either side within the time span."""

    kind: Literal["sample_symmetric"] = "sample_symmetric"


@dataclass(frozen=True)
class FixedSampleWindowPolicy:
    """Select a symmetric sample window, optionally derived from the time span."""

    samples: Optional[int] = None
    derive_from_window_ms: bool = False
    symmetric_round: bool = False
    kind: Literal["fixed_sample"] = "fixed_sample"


@dataclass(frozen=True)
class ShiftedValidWindowPolicy:
    """Shift a time- or sample-sized window until all included samples are valid."""

    samples: Optional[int] = None
    derive_from_window_ms: bool = False
    symmetric_round: bool = False
    fallback: Literal["shrink", "unclassified"] = "shrink"
    kind: Literal["shifted_valid"] = "shifted_valid"


@dataclass(frozen=True)
class AsymmetricNeighborWindowPolicy:
    """Use the immediate backward neighbor with a forward-neighbor fallback."""

    kind: Literal["asymmetric_neighbor"] = "asymmetric_neighbor"


@dataclass(frozen=True)
class TobiiWindowPolicy:
    """Use Tobii's sample-count calculation for a nominal sample interval."""

    sample_interval_ms: Optional[float] = None
    kind: Literal["tobii"] = "tobii"


WindowPolicy = Union[
    TimeSymmetricWindowPolicy,
    SampleSymmetricWindowPolicy,
    FixedSampleWindowPolicy,
    ShiftedValidWindowPolicy,
    AsymmetricNeighborWindowPolicy,
    TobiiWindowPolicy,
]

_POLICY_TYPES = {
    "time_symmetric": TimeSymmetricWindowPolicy,
    "sample_symmetric": SampleSymmetricWindowPolicy,
    "fixed_sample": FixedSampleWindowPolicy,
    "shifted_valid": ShiftedValidWindowPolicy,
    "asymmetric_neighbor": AsymmetricNeighborWindowPolicy,
    "tobii": TobiiWindowPolicy,
}


def window_policy_from_dict(data: Mapping[str, Any]) -> WindowPolicy:
    """Deserialize a tagged window policy persisted in experiment JSON."""
    values = dict(data)
    kind = values.pop("kind", None)
    try:
        policy_type = _POLICY_TYPES[kind]
    except KeyError as exc:
        raise ValueError(f"Unknown window policy kind: {kind!r}") from exc
    return policy_type(**values)


def translate_legacy_window_flags(
    *,
    sample_symmetric_window: bool = False,
    fixed_window_samples: Optional[int] = None,
    auto_fixed_window_from_ms: bool = False,
    symmetric_round_window: bool = False,
    asymmetric_neighbor_window: bool = False,
    shifted_valid_window: bool = False,
    shifted_valid_fallback: Literal["shrink", "unclassified"] = "shrink",
    tobii_window_mode: bool = False,
    tobii_sample_interval_ms: Optional[float] = None,
) -> WindowPolicy:
    """Translate deprecated selector flags into exactly one explicit policy.

    Modifiers for fixed sample sizing remain compatible with each other. Selector
    modes that used to be silently hidden by precedence are rejected.
    """
    fixed_mode = (
        fixed_window_samples is not None
        or auto_fixed_window_from_ms
        or symmetric_round_window
    )
    selectors = []
    if tobii_window_mode:
        selectors.append("tobii_window_mode")
    if asymmetric_neighbor_window:
        selectors.append("asymmetric_neighbor_window")
    if shifted_valid_window:
        selectors.append("shifted_valid_window")
    elif fixed_mode:
        selectors.append("fixed_window_samples")
    if sample_symmetric_window:
        selectors.append("sample_symmetric_window")
    if len(selectors) > 1:
        raise ValueError(
            "Contradictory legacy window flags: "
            + selectors[0]
            + " takes precedence over "
            + ", ".join(selectors[1:])
            + "; configure exactly one window policy."
        )

    if tobii_window_mode:
        return TobiiWindowPolicy(sample_interval_ms=tobii_sample_interval_ms)
    if asymmetric_neighbor_window:
        return AsymmetricNeighborWindowPolicy()
    if shifted_valid_window:
        return ShiftedValidWindowPolicy(
            samples=fixed_window_samples,
            derive_from_window_ms=auto_fixed_window_from_ms,
            symmetric_round=symmetric_round_window,
            fallback=shifted_valid_fallback,
        )
    if fixed_mode:
        return FixedSampleWindowPolicy(
            samples=fixed_window_samples,
            derive_from_window_ms=auto_fixed_window_from_ms,
            symmetric_round=symmetric_round_window,
        )
    if sample_symmetric_window:
        return SampleSymmetricWindowPolicy()
    return TimeSymmetricWindowPolicy()
