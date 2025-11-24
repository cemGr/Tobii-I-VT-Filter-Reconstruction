# Tobii-I-VT-Filter-Reconstruction

Utilities to reproduce the I-VT filter described in the exposé and Tobii paper.
Extraction, velocity computation, classification, and evaluation live in a
modular `ivt` package that follows SOLID principles and can be used via CLI or
Python API.

## Project layout
- `ivt/` – core library (extraction, velocity, classifier, evaluation, CLI).
- `app/` – lightweight smoke examples.
- `tests/` – TTD suite that guards extractor, velocity, and CLI behavior.
- `data/raw/` – untouched Tobii TSV exports (e.g. `ivt_frequency120_fixation_export.tsv`).
- `data/processed/` – pipeline outputs (slim TSVs, velocity TSVs, classified TSVs).
- `docs/` – exposé and Tobii paper; `docs/images/` for generated plots.

## Quickstart
1) Clone and activate an environment
```bash
git clone git@github.com:cemGr/Tobii-I-VT-Filter-Reconstruction.git
cd Tobii-I-VT-Filter-Reconstruction
python -m venv .venv
source .venv/bin/activate  # Linux/macOS
./.venv/Scripts/Activate.ps1  # Windows PowerShell
```

2) Install dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
```

3) Run the IVT pipeline (all steps are independent). Locale-specific decimal commas in gaze/eye columns are accepted during velocity computation.
```bash
# 3.1 Slim a Tobii export to the IVT-friendly columns
python -m ivt extract data/raw/ivt_frequency120_fixation_export.tsv data/processed/ivt_input.tsv

# 3.2 Compute Olsen velocity (e.g., 20 ms window, averaged eyes)
python -m ivt velocity data/processed/ivt_input.tsv data/processed/ivt_with_velocity.tsv --window 20 --eye average

# 3.3 Classify with I-VT (velocity threshold in deg/sec)
python -m ivt classify data/processed/ivt_with_velocity.tsv data/processed/ivt_with_classes.tsv --threshold 30

# 3.4 Plot velocity + gaze position (png, pdf, etc.)
# Control figure size with --figsize W H, pass --show to open the window interactively
python -m ivt analyze data/processed/ivt_with_classes.tsv docs/images/ivt_plot.png --threshold 30 --figsize 12 7 --show

# 3.5 Evaluate against ground truth labels if available
python -m ivt evaluate data/processed/ivt_with_classes.tsv --gt-col gt_event_type
```

4) Use the Python API when scripting
```python
from ivt import (
    TobiiTSVExtractor,
    VelocityCalculator,
    IVTClassifier,
    evaluate_ivt_vs_ground_truth,
)

slim_path = "data/processed/ivt_input.tsv"
TobiiTSVExtractor().convert("data/raw/ivt_frequency120_fixation_export.tsv", slim_path)

with_velocity = VelocityCalculator().compute_from_file(slim_path)
classified = IVTClassifier().classify(with_velocity)
stats = evaluate_ivt_vs_ground_truth(classified)
```

## Plotting quick recipe
You can visualize velocity and classification results directly from the TSVs using the Python API:
```python
import pandas as pd
import matplotlib.pyplot as plt

ivtdf = pd.read_csv("data/processed/ivt_with_classes.tsv", sep="\t")

fig, ax = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
ax[0].plot(ivtdf["time_ms"], ivtdf["velocity_deg_per_sec"], label="velocity")
ax[0].axhline(30, color="red", linestyle="--", label="threshold")
ax[0].set_ylabel("deg/sec")
ax[0].legend()

ax[1].plot(ivtdf["time_ms"], ivtdf["combined_x_px"], label="gaze x")
ax[1].plot(ivtdf["time_ms"], ivtdf["combined_y_px"], label="gaze y")
ax[1].set_ylabel("gaze (px)")
ax[1].set_xlabel("time (ms)")
ax[1].legend()
plt.tight_layout()
plt.show()
```

Or via the CLI helper. Adjust sizing with `--figsize`/`--dpi`, pass `--show` to open a window instead of only saving the file, and add `--show-events` if you also want the numbered event index as a third subplot when the column is present. Both gaze x/y traces are plotted by default:
```bash
python -m ivt analyze data/processed/ivt_with_classes.tsv docs/images/ivt_plot.png --threshold 30 --show-events
```

## Notes on data handling
- Extractor reads only the columns required for IVT and filters non-eye-tracker
  sensors, which suppresses the mixed-type `DtypeWarning` from large TSVs.
- Velocity and classifier CLI commands expect dot (`.`) decimals; both coerce
  non-numeric values to `NaN` so invalid rows are marked `Unclassified`.

## Testing (TTD)
```bash
pytest              # full suite
python -m unittest  # optional built-in runner
```

## Docker
```bash
docker build -t tobii-ivt:latest .
docker run --rm -it tobii-ivt:latest bash
```

## Release & publishing
```bash
# Bump version in setup.py, commit, then tag
git tag vX.Y.Z
git push origin --tags
```
