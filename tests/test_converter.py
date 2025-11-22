from __future__ import annotations

import os
from pathlib import Path

import pandas as pd

from converter import load_tobii_tsv_to_recording, prepare_recording_for_velocity


def test_load_tobii_tsv_to_recording_from_dataframe(tmp_path: Path) -> None:
    data = {
        "Sensor": ["Eye Tracker", "Eye Tracker"],
        "Recording timestamp [ms]": [0, 10],
        "Gaze point left X [DACS px]": [100.0, 101.0],
        "Gaze point left Y [DACS px]": [200.0, 201.0],
        "Eye position left X [DACS mm]": [10.0, 10.1],
        "Eye position left Y [DACS mm]": [20.0, 20.1],
        "Eye position left Z [DACS mm]": [600.0, 600.0],
        "Validity left": [0, 0],
        "Gaze point right X [DACS px]": [102.0, 103.0],
        "Gaze point right Y [DACS px]": [202.0, 203.0],
        "Eye position right X [DACS mm]": [10.2, 10.3],
        "Eye position right Y [DACS mm]": [20.2, 20.3],
        "Eye position right Z [DACS mm]": [600.0, 600.0],
        "Validity right": [0, 0],
    }
    df = pd.DataFrame(data)
    path = tmp_path / "test.tsv"
    df.to_csv(path, sep="\t", index=False)

    recording = load_tobii_tsv_to_recording(str(path), recording_id="rec1")

    assert recording.id == "rec1"
    assert len(recording.samples) == 2
    assert recording.samples[0].time_ms == 0
    assert recording.samples[1].left.gaze_x_px == 101.0
    assert recording.samples[1].right.eye_pos_y_mm == 20.3


def test_load_real_file() -> None:
    sample_file = "I-VT-normal Data export_short.tsv"
    assert os.path.exists(sample_file)
    recording = load_tobii_tsv_to_recording(sample_file)
    prepare_recording_for_velocity(recording)
    assert len(recording.samples) > 0
    # at least one sample should have combined gaze after preparation
    assert any(s.combined_gaze_x_px is not None for s in recording.samples)
