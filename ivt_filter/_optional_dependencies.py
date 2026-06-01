"""Helpers for loading optional runtime dependencies."""
from __future__ import annotations

from importlib import import_module
from importlib.util import find_spec
from types import ModuleType


PLOT_INSTALL_HINT = "pip install tobii-ivt-filter[plot]"


def require_matplotlib_pyplot() -> ModuleType:
    """Load pyplot or explain how to install the optional plotting extra."""
    if find_spec("matplotlib") is None:
        raise ModuleNotFoundError(
            "Plotting requires matplotlib. Install it with: " + PLOT_INSTALL_HINT
        )
    return import_module("matplotlib.pyplot")
