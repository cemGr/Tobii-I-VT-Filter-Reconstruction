import pandas as pd
from datetime import datetime, timedelta
from dataclasses import dataclass
from typing import List

import pytest

from ivt_filter.domain.dataset import EyeData, Recording, Sample


def _parse_recording_start(df: pd.DataFrame) -> datetime:
    """Return an absolute recording start time if the TSV provides one.

    Tobii exports include ``Recording date UTC`` and ``Recording start time UTC``
    columns. When present we combine them to anchor the relative
    ``Recording timestamp [ms]`` values to a realistic wall clock. If the
    columns are missing or unparsable we gracefully fall back to the Unix epoch
    so downstream computations continue to work.
    """

    date_col = "Recording date UTC"
    time_col = "Recording start time UTC"
    if date_col in df.columns and time_col in df.columns:
        date_value = df[date_col].iloc[0]
        time_value = df[time_col].iloc[0]
        if isinstance(date_value, str) and isinstance(time_value, str):
            try:
                combined = f"{date_value} {time_value}"
                dt = pd.to_datetime(combined, dayfirst=True, utc=True)
                return dt.tz_localize(None).to_pydatetime()
            except (TypeError, ValueError):
                pass
    return datetime.fromtimestamp(0)


@dataclass
class GroundTruthEvent:
    event_type: str
    index: int
    start_time: datetime
    end_time: datetime
    duration_ms: float
    fixation_x: float | None
    fixation_y: float | None


def load_recording_from_tsv(path: str) -> Recording:
    df = pd.read_csv(path, sep="\t")
    df = df[df["Sensor"] == "Eye Tracker"]
    start_time = _parse_recording_start(df)
    samples: List[Sample] = []
    for _, row in df.iterrows():
        ts = start_time + timedelta(milliseconds=float(row["Recording timestamp [ms]"]))
        left_eye = EyeData(
            gaze_x=row.get("Gaze point left X [DACS px]", float("nan")),
            gaze_y=row.get("Gaze point left Y [DACS px]", float("nan")),
            eye_pos_x_3d=row.get("Eye position left X [DACS mm]") if "Eye position left X [DACS mm]" in row else None,
            eye_pos_y_3d=row.get("Eye position left Y [DACS mm]") if "Eye position left Y [DACS mm]" in row else None,
            eye_pos_z_3d=row.get("Eye position left Z [DACS mm]") if "Eye position left Z [DACS mm]" in row else None,
        )
        right_eye = EyeData(
            gaze_x=row.get("Gaze point right X [DACS px]", float("nan")),
            gaze_y=row.get("Gaze point right Y [DACS px]", float("nan")),
            eye_pos_x_3d=row.get("Eye position right X [DACS mm]") if "Eye position right X [DACS mm]" in row else None,
            eye_pos_y_3d=row.get("Eye position right Y [DACS mm]") if "Eye position right Y [DACS mm]" in row else None,
            eye_pos_z_3d=row.get("Eye position right Z [DACS mm]") if "Eye position right Z [DACS mm]" in row else None,
        )
        def _parse_valid(value):
            try:
                return int(value)
            except (TypeError, ValueError):
                return 4

        sample = Sample(
            timestamp=ts,
            left_validity=_parse_valid(row.get("Validity left", 4)),
            right_validity=_parse_valid(row.get("Validity right", 4)),
            left_eye=left_eye,
            right_eye=right_eye,
        )
        samples.append(sample)
    start = samples[0].timestamp if samples else start_time
    end = samples[-1].timestamp if samples else start
    return Recording(id=path, start_time=start, end_time=end, samples=samples)


def load_ground_truth_events_from_tsv(path: str) -> List[GroundTruthEvent]:
    df = pd.read_csv(path, sep="\t")
    df = df[df["Sensor"] == "Eye Tracker"]
    start_time = _parse_recording_start(df)
    grouped = df.groupby(["Eye movement type", "Eye movement type index"])
    events: List[GroundTruthEvent] = []
    for (etype, idx), group in grouped:
        start_ms = group["Recording timestamp [ms]"].min()
        end_ms = group["Recording timestamp [ms]"].max()
        events.append(
            GroundTruthEvent(
                event_type=str(etype),
                index=int(idx),
                start_time=start_time + timedelta(milliseconds=float(start_ms)),
                end_time=start_time + timedelta(milliseconds=float(end_ms)),
                duration_ms=float(group["Eye movement event duration [ms]"].iloc[0]),
                fixation_x=float(group.get("Fixation point X [DACS px]", pd.Series([float("nan")])).iloc[0])
                if "Fixation point X [DACS px]" in group
                else None,
                fixation_y=float(group.get("Fixation point Y [DACS px]", pd.Series([float("nan")])).iloc[0])
                if "Fixation point Y [DACS px]" in group
                else None,
            )
        )
    return events


@pytest.fixture
def simple_recording() -> Recording:
    base = datetime.fromtimestamp(0)
    samples = []
    for i in range(10):
        ts = base + timedelta(milliseconds=i * 10)
        gaze_x = 100.0 if i < 5 else 200.0
        gaze_y = 100.0 if i < 5 else 200.0
        samples.append(
            Sample(
                timestamp=ts,
                left_validity=0,
                right_validity=0,
                left_eye=EyeData(gaze_x=gaze_x, gaze_y=gaze_y),
                right_eye=EyeData(gaze_x=gaze_x, gaze_y=gaze_y),
            )
        )
    return Recording(id="synthetic", start_time=samples[0].timestamp, end_time=samples[-1].timestamp, samples=samples)
