import pandas as pd

from ivt.extractor import RAW_COLUMNS, TobiiTSVExtractor


def test_extractor_filters_sensor_and_maps_columns(tmp_path):
    raw = pd.DataFrame(
        {
            "Sensor": ["Eye Tracker", "IMU"],
            RAW_COLUMNS["time_ms"]: [0, 1],
            RAW_COLUMNS["gaze_left_x_px"]: [1.0, 99.0],
            RAW_COLUMNS["gaze_left_y_px"]: [2.0, 99.0],
            RAW_COLUMNS["gaze_right_x_px"]: [3.0, 99.0],
            RAW_COLUMNS["gaze_right_y_px"]: [4.0, 99.0],
            RAW_COLUMNS["validity_left"]: [0, 9],
            RAW_COLUMNS["validity_right"]: [0, 9],
            RAW_COLUMNS["eye_left_z_mm"]: [600.0, 0.0],
            RAW_COLUMNS["eye_right_z_mm"]: [600.0, 0.0],
        }
    )

    raw_path = tmp_path / "raw.tsv"
    output_path = tmp_path / "slim.tsv"
    raw.to_csv(raw_path, sep="\t", index=False)

    extractor = TobiiTSVExtractor()
    extractor.convert(str(raw_path), str(output_path))

    slim = pd.read_csv(output_path, sep="\t")
    assert len(slim) == 1
    assert slim.loc[0, "time_ms"] == 0
    assert "gaze_left_x_px" in slim.columns
    assert "eye_right_z_mm" in slim.columns
