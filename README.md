# Tobii I-VT Filter Reconstruction

Reconstruction of the classic Tobii **Identification by Velocity Threshold (I-VT)** fixation filter based on Olsen's whitepaper. The project provides a staged Python 3 pipeline, pluggable noise-reduction strategies, and tests that validate behavior against synthetic fixtures and Tobii Pro Lab TSV exports.

## Features
- Seven-stage IVT pipeline mirroring Olsen §3.1 (gap fill-in, eye selection, noise reduction, velocity computation, classification, fixation merging, discard short fixations).
- Extensible noise filtering via the Strategy pattern (none, moving average, median).
- Typed domain and event model for eye-tracking recordings, samples, and gaze events.
- Pytest suite with synthetic scenarios and black-box comparisons against provided TSV fixtures.
- Docker image and GitHub Actions workflow to run the suite in CI.

## Repository layout
- `ivt_filter/` – Library source code.
  - `domain/` – Dataset, recording, sample, and gaze-event models.
  - `stages/` – Implementations of the seven IVT functions.
  - `noise/` – Noise filter strategies.
  - `docs/architecture.md` – Design overview with UML diagrams.
- `tests/` – Synthetic and TSV-based black-box tests plus helpers for loading fixtures.
- `Dockerfile` – Container to run pytest.
- `.github/workflows/ci.yml` – CI configuration for GitHub Actions.
- `*.tsv` – Shortened Tobii Pro Lab exports used as black-box fixtures.

## Quick start
1. Clone the repository:
   ```bash
   git clone git@github.com:cemGr/Tobii-I-VT-Filter-Reconstruction.git
   cd Tobii-I-VT-Filter-Reconstruction
   ```
2. (Optional) Create & activate a virtual environment:
   ```bash
   python -m venv .venv
   # Linux/macOS
   source .venv/bin/activate
   # Windows (PowerShell)
   .\.venv\Scripts\Activate.ps1
   ```
3. Install dependencies and the package in editable mode:
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   pip install -e .
   ```

## Running tests
Pytest is the primary test runner.

```bash
pytest                 # run all tests
pytest -q              # quiet output
pytest --maxfail=1     # stop at first failure
```

## Using the IVT filter
A minimal example using the public API:

```python
from ivt_filter.config import IVTFilterConfiguration, EyeSelectionMode
from ivt_filter.engine import IVTFilterEngine
from ivt_filter.noise import NoNoiseFilterStrategy
from tests.conftest import load_recording_from_tsv

recording = load_recording_from_tsv("I-VT-normal Data export_short.tsv")
config = IVTFilterConfiguration(
    max_gap_length_ms=75,
    eye_selection_mode=EyeSelectionMode.AVERAGE,
    velocity_threshold_deg_per_sec=30.0,
    window_length_ms=20,
    max_time_between_fixations_ms=75,
    max_angle_between_fixations_deg=0.5,
    minimum_fixation_duration_ms=60,
    noise_filter_strategy=NoNoiseFilterStrategy(),
)

engine = IVTFilterEngine()
result = engine.run(recording, config)
print([(e.event_type.name, e.start_time, e.end_time) for e in result.events])
```

Refer to `ivt_filter/docs/architecture.md` for detailed stage responsibilities and UML diagrams.

## Docker workflow
Build and run the containerized test environment:

```bash
docker build -t ivt-filter:latest .
docker run --rm ivt-filter:latest
```

## Releasing
1. Bump the version in `setup.py` and commit.
2. Tag and push:
   ```bash
   git tag vX.Y.Z
   git push origin --tags
   ```
3. (Optional) Publish to PyPI using `python -m build` and `twine upload dist/*`.
