"""Loader for the Lund2013 / Andersson et al. 2017 annotated dataset (WS-C).

The dataset (``EyeMovementDetectorEvaluation``, two human coders MN and RA) is
the external, hand-annotated benchmark for the third evaluation axis. Each
``.mat`` holds an ``ETdata`` struct whose ``pos`` matrix is N x 6:
``[timestamp_us, pupil_h, pupil_v, x_px, y_px, label]`` with the acquisition
geometry (screen size, resolution, viewing distance, sampling rate) stored
alongside. The loader maps one file into the ivt_filter slim schema.

I-VT has no smooth-pursuit or PSO class, so the Lund labels are mapped through a
protocol. The primary protocol P0 folds PSO into saccades and excludes pursuit
and undefined samples from scoring; sensitivity protocols S1 and S2 vary the PSO
and pursuit handling. Excluded labels map to "" and are dropped before scoring.

scipy is only needed here; it is an optional extra (``pip install
tobii-ivt-filter[external]``).
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

# Lund event-label integers.
LUND_UNDEFINED = 0
LUND_FIXATION = 1
LUND_SACCADE = 2
LUND_PSO = 3
LUND_PURSUIT = 4
LUND_BLINK = 5
LUND_OTHER = 6

_EXCLUDED = ""  # label that is dropped from scoring

# Label-mapping protocols: Lund integer -> ivt label ("" = excluded from scoring).
_P0 = {
    LUND_UNDEFINED: _EXCLUDED,
    LUND_FIXATION: "Fixation",
    LUND_SACCADE: "Saccade",
    LUND_PSO: "Saccade",        # PSO is part of the saccadic dynamics
    LUND_PURSUIT: _EXCLUDED,    # I-VT cannot represent pursuit
    LUND_BLINK: "EyesNotFound",
    LUND_OTHER: _EXCLUDED,
}
LUND_PROTOCOLS: dict[str, dict[int, str]] = {
    "P0": _P0,
    "S1": {**_P0, LUND_PSO: "Fixation"},      # PSO as fixation tail
    "S2": {**_P0, LUND_PURSUIT: "Fixation"},  # pursuit included as fixation
}

# Fallback acquisition geometry (used only when a field is missing from the file).
_DEFAULT_SCREEN_DIM_MM = (380.0, 300.0)
_DEFAULT_SCREEN_RES_PX = (1024, 768)
_DEFAULT_VIEW_DIST_MM = 670.0
_DEFAULT_SAMP_FREQ_HZ = 500.0


def map_lund_labels(labels: np.ndarray, protocol: str = "P0") -> np.ndarray:
    """Map raw Lund integer labels to ivt labels under a protocol."""
    if protocol not in LUND_PROTOCOLS:
        raise ValueError(f"unknown protocol {protocol!r}; known: {sorted(LUND_PROTOCOLS)}")
    mapping = LUND_PROTOCOLS[protocol]
    out = np.empty(len(labels), dtype=object)
    for i, raw in enumerate(labels):
        out[i] = mapping.get(int(raw), _EXCLUDED)
    return out


def _require_scipy():
    try:
        import scipy.io as sio
    except ImportError as exc:  # pragma: no cover - exercised only without scipy
        raise ImportError(
            "Reading Lund2013 .mat files needs scipy. Install the external extra: "
            "pip install tobii-ivt-filter[external]"
        ) from exc
    return sio


def _scalar(value, default: float) -> float:
    arr = np.asarray(value, dtype=float).ravel()
    return float(arr[0]) if arr.size else float(default)


def load_lund2013_mat(
    path: str | Path,
    *,
    protocol: str = "P0",
    screen_dim_mm: Optional[tuple[float, float]] = None,
    screen_res_px: Optional[tuple[float, float]] = None,
    view_dist_mm: Optional[float] = None,
) -> pd.DataFrame:
    """Map one Lund2013 ``.mat`` into the ivt_filter slim schema.

    Geometry is read from the file (``screenDim``, ``screenRes``, ``viewDist``)
    and may be overridden by keyword. Gaze pixels are converted to screen
    millimetres with the origin at the screen centre, duplicated to both eyes,
    and the eye origin is synthesised at ``(0, 0, view_dist_mm)`` (the dataset is
    a fixed-head, single-stream recording). Blink samples are marked invalid.
    """
    sio = _require_scipy()
    raw = sio.loadmat(str(path))
    record = raw["ETdata"][0, 0]
    pos = np.asarray(record["pos"], dtype=float)
    if pos.ndim != 2 or pos.shape[1] < 6:
        raise ValueError(f"unexpected ETdata.pos shape {pos.shape}; expected (N, 6)")

    res_x, res_y = screen_res_px or _field_pair(record, "screenRes", _DEFAULT_SCREEN_RES_PX)
    dim_x, dim_y = screen_dim_mm or _field_pair(record, "screenDim", _DEFAULT_SCREEN_DIM_MM)
    view_dist = view_dist_mm if view_dist_mm is not None else _scalar(
        record["viewDist"] if "viewDist" in record.dtype.names else None, _DEFAULT_VIEW_DIST_MM
    )

    # Build the time axis from the fixed sampling rate rather than the raw
    # timestamp column: these are continuous fixed-rate recordings, and some
    # files have degenerate (duplicate) raw timestamps that the velocity stage
    # rejects. A regular grid at sampFreq is unique, monotonic and reproducible.
    samp_freq = _scalar(
        record["sampFreq"] if "sampFreq" in record.dtype.names else None, _DEFAULT_SAMP_FREQ_HZ
    )
    if samp_freq <= 0:
        samp_freq = _DEFAULT_SAMP_FREQ_HZ
    n = len(pos)
    t_ms = np.arange(n, dtype=float) * (1000.0 / samp_freq)
    x_px = pos[:, 3]
    y_px = pos[:, 4]
    labels = pos[:, 5]

    mm_per_px_x = dim_x / res_x
    mm_per_px_y = dim_y / res_y
    x_mm = (x_px - res_x / 2.0) * mm_per_px_x
    y_mm = (y_px - res_y / 2.0) * mm_per_px_y

    finite = np.isfinite(x_px) & np.isfinite(y_px)
    is_blink = labels == LUND_BLINK
    valid = finite & ~is_blink
    validity = np.where(valid, 0, 1).astype(int)

    gt = map_lund_labels(labels, protocol)
    gt[is_blink] = "EyesNotFound"

    return pd.DataFrame(
        {
            "time_ms": t_ms,
            "gaze_left_x_mm": x_mm,
            "gaze_left_y_mm": y_mm,
            "gaze_right_x_mm": x_mm,
            "gaze_right_y_mm": y_mm,
            "gaze_left_x_px": x_px,
            "gaze_left_y_px": y_px,
            "gaze_right_x_px": x_px,
            "gaze_right_y_px": y_px,
            "eye_left_x_mm": np.zeros(n),
            "eye_left_y_mm": np.zeros(n),
            "eye_left_z_mm": np.full(n, view_dist),
            "eye_right_x_mm": np.zeros(n),
            "eye_right_y_mm": np.zeros(n),
            "eye_right_z_mm": np.full(n, view_dist),
            "validity_left": validity,
            "validity_right": validity,
            "lund_label": labels.astype(int),
            "gt_event_type": gt,
        }
    )


def _field_pair(record, field: str, default: tuple[float, float]) -> tuple[float, float]:
    if field not in record.dtype.names:
        return default
    arr = np.asarray(record[field], dtype=float).ravel()
    if arr.size < 2:
        return default
    return float(arr[0]), float(arr[1])
