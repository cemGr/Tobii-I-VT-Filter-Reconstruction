"""Tobii validity-code parsing rules."""

from __future__ import annotations


INVALID_VALIDITY = 999


def parse_tobii_validity(value: object) -> int:
    """Return a normalized Tobii validity code.

    ``Valid`` maps to ``0`` and ``Invalid`` or values that cannot be parsed map
    to ``999``. Numeric strings and numeric values use their integer value.
    """
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized == "valid":
            return 0
        if normalized == "invalid":
            return INVALID_VALIDITY

    try:
        return int(value)  # type: ignore[call-overload]
    except (TypeError, ValueError, OverflowError):
        return INVALID_VALIDITY
