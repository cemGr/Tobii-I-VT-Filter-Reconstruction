import pandas as pd

from ivt.extractor import RAW_COLUMNS, TobiiTSVExtractor


def test_extractor_filters_sensor_and_maps_columns(tmp_path):
    raw = pd.DataFrame(
        {
            "Sensor": ["Eye Tracker", "Eye Tracker", "IMU"],
            RAW_COLUMNS["time_ms"]: [1, 0, 2],
            RAW_COLUMNS["gaze_left_x_mm"]: [1.0, 99.0, 0.0],
            RAW_COLUMNS["gaze_left_y_mm"]: [2.0, 99.0, 0.0],
            RAW_COLUMNS["gaze_right_x_mm"]: [3.0, 99.0, 0.0],
            RAW_COLUMNS["gaze_right_y_mm"]: [4.0, 99.0, 0.0],
            RAW_COLUMNS["gaze_left_x_px"]: [10.0, 990.0, 0.0],
            RAW_COLUMNS["gaze_left_y_px"]: [20.0, 990.0, 0.0],
            RAW_COLUMNS["gaze_right_x_px"]: [30.0, 990.0, 0.0],
            RAW_COLUMNS["gaze_right_y_px"]: [40.0, 990.0, 0.0],
            RAW_COLUMNS["validity_left"]: ["Valid", "Invalid", "Valid"],
            RAW_COLUMNS["validity_right"]: [0, 9, 0],
            RAW_COLUMNS["eye_left_x_mm"]: [100.0, 0.0, 0.0],
            RAW_COLUMNS["eye_left_y_mm"]: [200.0, 0.0, 0.0],
            RAW_COLUMNS["eye_left_z_mm"]: [600.0, 0.0, 0.0],
            RAW_COLUMNS["eye_right_x_mm"]: [150.0, 0.0, 0.0],
            RAW_COLUMNS["eye_right_y_mm"]: [250.0, 0.0, 0.0],
            RAW_COLUMNS["eye_right_z_mm"]: [650.0, 0.0, 0.0],
            RAW_COLUMNS["gt_event_type"]: ["Fixation", "Unknown", "Saccade"],
            RAW_COLUMNS["gt_event_index"]: [1, 2, 3],
        }
    )

    raw_path = tmp_path / "raw.tsv"
    output_path = tmp_path / "slim.tsv"
    raw.to_csv(raw_path, sep="\t", index=False, decimal=",")

    extractor = TobiiTSVExtractor()
    extractor.convert(str(raw_path), str(output_path))

    slim = pd.read_csv(output_path, sep="\t", decimal=",")
    assert list(slim.columns) == list(RAW_COLUMNS.keys())
    assert slim.loc[0, "time_ms"] == 0
    assert slim.loc[1, "gaze_left_x_mm"] == 1.0
    assert slim.loc[1, "gt_event_type"] == "Fixation"
