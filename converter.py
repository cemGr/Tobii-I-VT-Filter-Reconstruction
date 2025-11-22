from __future__ import annotations

import os
from typing import Optional

import pandas as pd

from domain import EyeData, Recording, Sample


VALIDITY_OK = 0


def _is_valid(validity: int | str) -> bool:
    if isinstance(validity, str):
        return validity.strip().lower() == "valid"
    return validity == VALIDITY_OK


def _to_optional(value: object) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, float) and pd.isna(value):
        return None
    if isinstance(value, str):
        cleaned = value.replace(",", ".").strip()
        if cleaned == "":
            return None
        try:
            return float(cleaned)
        except ValueError:
            return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _validity_value(value: object) -> int | str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return 1
    if isinstance(value, str):
        return value
    try:
        return int(value)
    except (TypeError, ValueError):
        return 1


def load_tobii_tsv_to_recording(path: str, recording_id: Optional[str] = None) -> Recording:
    df = pd.read_csv(path, sep="\t")
    if "Sensor" in df.columns:
        df = df[df["Sensor"] == "Eye Tracker"]

    recording_identifier = recording_id or os.path.splitext(os.path.basename(path))[0]

    samples = []
    for _, row in df.iterrows():
        left = EyeData(
            gaze_x_px=_to_optional(row.get("Gaze point left X [DACS px]")),
            gaze_y_px=_to_optional(row.get("Gaze point left Y [DACS px]")),
            eye_pos_x_mm=_to_optional(row.get("Eye position left X [DACS mm]")),
            eye_pos_y_mm=_to_optional(row.get("Eye position left Y [DACS mm]")),
            eye_pos_z_mm=_to_optional(row.get("Eye position left Z [DACS mm]")),
        )
        right = EyeData(
            gaze_x_px=_to_optional(row.get("Gaze point right X [DACS px]")),
            gaze_y_px=_to_optional(row.get("Gaze point right Y [DACS px]")),
            eye_pos_x_mm=_to_optional(row.get("Eye position right X [DACS mm]")),
            eye_pos_y_mm=_to_optional(row.get("Eye position right Y [DACS mm]")),
            eye_pos_z_mm=_to_optional(row.get("Eye position right Z [DACS mm]")),
        )
        timestamp = _to_optional(row.get("Recording timestamp [ms]"))
        if timestamp is None:
            continue
        sample = Sample(
            time_ms=float(timestamp),
            left=left,
            right=right,
            validity_left=_validity_value(row.get("Validity left", 1)),
            validity_right=_validity_value(row.get("Validity right", 1)),
        )
        samples.append(sample)

    return Recording(id=recording_identifier, samples=samples)


def compute_combined_gaze_and_eye(sample: Sample) -> None:
    left_valid = _is_valid(sample.validity_left)
    right_valid = _is_valid(sample.validity_right)

    def average_optional(a: Optional[float], b: Optional[float]) -> Optional[float]:
        if a is None or b is None:
            if a is None and b is None:
                return None
            return a if b is None else b
        return (a + b) / 2.0

    if left_valid and right_valid:
        sample.combined_gaze_x_px = average_optional(sample.left.gaze_x_px, sample.right.gaze_x_px)
        sample.combined_gaze_y_px = average_optional(sample.left.gaze_y_px, sample.right.gaze_y_px)
        sample.eye_x_mm = average_optional(sample.left.eye_pos_x_mm, sample.right.eye_pos_x_mm)
        sample.eye_y_mm = average_optional(sample.left.eye_pos_y_mm, sample.right.eye_pos_y_mm)
        sample.eye_z_mm = average_optional(sample.left.eye_pos_z_mm, sample.right.eye_pos_z_mm)
    elif left_valid:
        sample.combined_gaze_x_px = sample.left.gaze_x_px
        sample.combined_gaze_y_px = sample.left.gaze_y_px
        sample.eye_x_mm = sample.left.eye_pos_x_mm
        sample.eye_y_mm = sample.left.eye_pos_y_mm
        sample.eye_z_mm = sample.left.eye_pos_z_mm
    elif right_valid:
        sample.combined_gaze_x_px = sample.right.gaze_x_px
        sample.combined_gaze_y_px = sample.right.gaze_y_px
        sample.eye_x_mm = sample.right.eye_pos_x_mm
        sample.eye_y_mm = sample.right.eye_pos_y_mm
        sample.eye_z_mm = sample.right.eye_pos_z_mm
    else:
        sample.combined_gaze_x_px = None
        sample.combined_gaze_y_px = None
        sample.eye_x_mm = None
        sample.eye_y_mm = None
        sample.eye_z_mm = None


def prepare_recording_for_velocity(recording: Recording) -> None:
    for sample in recording.samples:
        compute_combined_gaze_and_eye(sample)
