import pandas as pd
from pathlib import Path

RAW_COLUMNS = {
    "time_ms": "Recording timestamp [ms]",
    "gaze_left_x_px": "Gaze point left X [DACS px]",
    "gaze_left_y_px": "Gaze point left Y [DACS px]",
    "gaze_right_x_px": "Gaze point right X [DACS px]",
    "gaze_right_y_px": "Gaze point right Y [DACS px]",
    "validity_left": "Validity left",
    "validity_right": "Validity right",
    "eye_left_x_mm": "Eye position left X [DACS mm]",
    "eye_left_y_mm": "Eye position left Y [DACS mm]",
    "eye_left_z_mm": "Eye position left Z [DACS mm]",
    "eye_right_x_mm": "Eye position right X [DACS mm]",
    "eye_right_y_mm": "Eye position right Y [DACS mm]",
    "eye_right_z_mm": "Eye position right Z [DACS mm]",
    "gt_event_type": "Eye movement type",
    "gt_event_index": "Eye movement type index",
}

def convert_tobii_tsv_to_ivt_tsv(input_path: str, output_path: str) -> None:
    input_path = Path(input_path)
    df = pd.read_csv(input_path, sep="\t")

    # Nur Eye-Tracker-Zeilen
    if "Sensor" in df.columns:
        df = df[df["Sensor"] == "Eye Tracker"]

    slim = pd.DataFrame()
    for new_name, old_name in RAW_COLUMNS.items():
        if old_name in df.columns:
            slim[new_name] = df[old_name]
        else:
            slim[new_name] = pd.NA  # falls mal eine Spalte fehlt

    slim.to_csv(output_path, sep="\t", index=False)


if __name__ == "__main__":
    convert_tobii_tsv_to_ivt_tsv(
        "I-VT-frequency120Fixation export.tsv",
        "I-VT-frequency120Fixation export_input.tsv",
    )
