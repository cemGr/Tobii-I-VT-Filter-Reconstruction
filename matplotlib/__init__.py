"""Lightweight matplotlib stub for offline test environments.

This module provides the minimal API surface required by the analysis
smoke tests. If the real matplotlib is available in the environment it
should shadow this stub on the import path.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

_backend: str | None = None


def use(backend: str) -> None:
    global _backend
    _backend = backend


class _Pyplot(SimpleNamespace):
    def figure(self) -> None:  # type: ignore[override]
        return None

    def plot(self, *args: Any, **kwargs: Any) -> None:  # type: ignore[override]
        return None

    def xlabel(self, *args: Any, **kwargs: Any) -> None:  # type: ignore[override]
        return None

    def ylabel(self, *args: Any, **kwargs: Any) -> None:  # type: ignore[override]
        return None

    def title(self, *args: Any, **kwargs: Any) -> None:  # type: ignore[override]
        return None

    def tight_layout(self, *args: Any, **kwargs: Any) -> None:  # type: ignore[override]
        return None

    def show(self, *args: Any, **kwargs: Any) -> None:  # type: ignore[override]
        return None


def __getattr__(name: str) -> Any:
    if name == "pyplot":
        return _Pyplot()
    raise AttributeError(name)
