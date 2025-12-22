# Tobii-I-VT-Filter-Reconstruction

Bachelor Thesis about the reconstruction of the I-VT (Velocity-Threshold) Filters by Tobii Pro Lab.

This project implements a flexible, configurable I-VT eye-tracking classification pipeline with multiple velocity calculation methods, coordinate rounding strategies, and post-processing options.

## Features

- **Multiple Velocity Calculation Methods**:
  - `olsen2d`: Fast 2D approximation using Olsen's method (θ = atan(s/d))
  - `ray3d`: Physically correct 3D ray angle method (θ = acos(ray₀·ray₁))
  
- **Coordinate Rounding Strategies**:
  - `none`: No rounding (default)
  - `nearest`: Banker's rounding (round half to even)
  - `halfup`: Classical rounding (0.5 always rounds up)
  - `floor`: Always round down
  - `ceil`: Always round up

- **Spatial Smoothing**:
  - `none`: No smoothing
  - `median`: Median filter
  - `moving_average`: Moving average filter

- **Window Selection Strategies**:
  - Time-based symmetric windows
  - Sample-based symmetric windows
  - Fixed sample windows with auto-calculation from time
  - Asymmetric and symmetric rounding options

- **Gap Filling**: Linear interpolation for short missing data segments
- **I-VT Classification**: Velocity threshold-based fixation/saccade detection
- **Post-Processing**: Saccade merging and fixation filtering
- **Evaluation**: Automatic comparison with ground-truth annotations

---

## Quick Start

### 1. Clone the repository

```bash
git clone git@github.com:cemGr/Tobii-I-VT-Filter-Reconstruction.git
cd <repo>
```

### 2. (Optional) Create & activate a virtual environment

```bash
python -m venv .venv
# Linux/macOS
source .venv/bin/activate
# Windows (PowerShell)
.\.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
pip install -e .    # editable install
```

---

## Usage

### Basic Command

```bash
python -m ivt_filter.cli --input <input.tsv> --output <output.tsv>
```

### Velocity Calculation Methods

**Olsen 2D (default)** - Fast approximation, only requires eye-screen distance (eye_z):
```bash
python -m ivt_filter.cli \
  --input data.tsv \
  --output results.tsv \
  --velocity-method olsen2d
```

**Ray 3D** - Physically correct 3D angle calculation using full eye position:
```bash
python -m ivt_filter.cli \
  --input data.tsv \
  --output results.tsv \
  --velocity-method ray3d
```

### Coordinate Rounding

Round coordinates before velocity calculation to reduce noise:

```bash
# Banker's rounding (round 0.5 to even number)
python -m ivt_filter.cli --input data.tsv --coordinate-rounding nearest

# Classical rounding (0.5 always rounds up)
python -m ivt_filter.cli --input data.tsv --coordinate-rounding halfup

# Floor/Ceil rounding
python -m ivt_filter.cli --input data.tsv --coordinate-rounding floor
python -m ivt_filter.cli --input data.tsv --coordinate-rounding ceil
```

### Window Configuration

```bash
# Time-based window (default: 20ms)
python -m ivt_filter.cli --input data.tsv --window 20

# Fixed sample window (e.g., 7 samples)
python -m ivt_filter.cli --input data.tsv --fixed-window-samples 7

# Auto-calculate fixed window from time and sampling rate
python -m ivt_filter.cli --input data.tsv --auto-fixed-window-from-ms

# Sample-symmetric window within time constraints
python -m ivt_filter.cli --input data.tsv --sample-symmetric-window

# Symmetric rounding (ensure odd window size)
python -m ivt_filter.cli --input data.tsv --symmetric-round-window
```

### Eye Selection

```bash
# Use average of both eyes (default)
python -m ivt_filter.cli --input data.tsv --eye average

# Use only left eye
python -m ivt_filter.cli --input data.tsv --eye left

# Use only right eye
python -m ivt_filter.cli --input data.tsv --eye right
```

### Spatial Smoothing

```bash
# No smoothing (default)
python -m ivt_filter.cli --input data.tsv --smoothing none

# Median smoothing (5 samples)
python -m ivt_filter.cli --input data.tsv --smoothing median --smooth-window-samples 5

# Moving average
python -m ivt_filter.cli --input data.tsv --smoothing moving_average --smooth-window-samples 5
```

### Gap Filling

```bash
# Enable gap filling with maximum gap of 75ms
python -m ivt_filter.cli --input data.tsv --gap-fill --gap-fill-max-ms 75
```

### Classification and Evaluation

```bash
# Classify fixations/saccades with 30 deg/s threshold
python -m ivt_filter.cli --input data.tsv --classify --threshold 30

# Evaluate against ground truth
python -m ivt_filter.cli --input data.tsv --classify --evaluate

# Generate event-level output
python -m ivt_filter.cli --input data.tsv --classify --with-events
```

### Complete Example

```bash
python -m ivt_filter.cli \
  --input data/processed/ivt_input.tsv \
  --output results/output.tsv \
  --velocity-method ray3d \
  --coordinate-rounding halfup \
  --window 20 \
  --eye average \
  --smoothing median \
  --smooth-window-samples 5 \
  --gap-fill \
  --gap-fill-max-ms 75 \
  --classify \
  --threshold 30 \
  --evaluate \
  --with-events \
  --sampling-rate-method first_100 \
  --dt-calculation-method median
```

### Data Preprocessing

Before running the classifier, preprocess Tobii Pro Lab exports:

```bash
python -c "from extractor import convert_tobii_tsv_to_ivt_tsv; \
  convert_tobii_tsv_to_ivt_tsv('raw_export.tsv', 'preprocessed.tsv')"
```

---

## Command-Line Options

### Velocity Calculation
- `--velocity-method {olsen2d,ray3d}`: Velocity calculation method (default: olsen2d)
- `--window FLOAT`: Window length in milliseconds (default: 20.0)
- `--coordinate-rounding {none,nearest,halfup,floor,ceil}`: Coordinate rounding strategy (default: none)

### Window Selection
- `--sample-symmetric-window`: Use sample-symmetric window within time constraints
- `--fixed-window-samples INT`: Use fixed sample window (must be odd, ≥3)
- `--auto-fixed-window-from-ms`: Auto-calculate fixed window from time
- `--symmetric-round-window`: Ensure symmetric odd window size
- `--allow-asymmetric-window`: Allow asymmetric windows (even sizes)

### Eye and Smoothing
- `--eye {left,right,average}`: Eye selection mode (default: average)
- `--smoothing {none,median,moving_average}`: Spatial smoothing method (default: none)
- `--smooth-window-samples INT`: Smoothing window size in samples (default: 5)

### Gap Filling
- `--gap-fill`: Enable gap filling
- `--gap-fill-max-ms FLOAT`: Maximum gap duration to fill in ms (default: 75.0)

### Sampling Rate
- `--sampling-rate-method {all_samples,first_100}`: Sampling rate calculation (default: first_100)
- `--dt-calculation-method {median,mean}`: Time difference calculation (default: median)

### Classification
- `--classify`: Apply I-VT velocity threshold classifier
- `--threshold FLOAT`: Velocity threshold in deg/s (default: 30.0)

### Evaluation and Output
- `--evaluate`: Evaluate against ground truth
- `--with-events`: Generate event-level output
- `--plot`: Generate visualization plots

---

## Examples

### Velocity Method Comparison

Run the comparison script to see differences between calculation methods:

```bash
python examples/velocity_comparison.py data/processed/ivt_input.tsv
```

Output includes:
- Velocity statistics for each method (min, max, mean, median, std)
- Direct comparison between Olsen 2D and Ray 3D
- Effect of coordinate rounding
- Recommendations for different use cases

### Custom Analysis

```python
from ivt_filter.config import OlsenVelocityConfig
from ivt_filter.velocity import compute_olsen_velocity
from ivt_filter.io import read_tsv

# Load data
df = read_tsv("data/processed/ivt_input.tsv")

# Configure velocity calculation
config = OlsenVelocityConfig(
    velocity_method="ray3d",
    coordinate_rounding="halfup",
    window_length_ms=20.0,
    smoothing_mode="median",
    smoothing_window_samples=5
)

# Compute velocities
df_result = compute_olsen_velocity(df, config)

# Analyze results
velocities = df_result["velocity_deg_per_sec"].dropna()
print(f"Mean velocity: {velocities.mean():.2f} deg/s")
print(f"Max velocity: {velocities.max():.2f} deg/s")
```

---

## Architecture

The project follows SOLID principles with a Strategy Pattern design:

### Core Modules

- **`velocity_calculation.py`**: Strategy pattern for velocity methods
  - `Olsen2DApproximation`: 2D approximation (fast, only eye_z)
  - `Ray3DAngle`: 3D ray angle (accurate, full eye position)

- **`coordinate_rounding.py`**: Strategy pattern for coordinate rounding
  - `NoRounding`, `RoundToNearest`, `RoundHalfUp`, `FloorRounding`, `CeilRounding`

- **`smoothing_strategy.py`**: Strategy pattern for spatial smoothing
  - `NoSmoothing`, `MedianSmoothing`, `MovingAverageSmoothing`

- **`window_rounding.py`**: Strategy pattern for window size calculation
  - `StandardWindowRounding`, `SymmetricRoundWindowStrategy`

- **`gaze.py`**: Eye data combination and preprocessing
- **`velocity.py`**: Main velocity calculation pipeline
- **`classifier.py`**: I-VT threshold-based classification
- **`evaluation.py`**: Ground truth comparison metrics
- **`cli.py`**: Command-line interface

### Velocity Calculation Methods

#### Olsen 2D Approximation
```
θ = atan(screen_distance / eye_z)
velocity = θ / Δt
```
- **Pros**: Fast, simple, only needs eye-screen distance (Z coordinate)
- **Cons**: 2D approximation, less accurate for lateral eye movements
- **Use case**: Standard I-VT classification, backward compatibility

#### 3D Ray Angle
```
ray₀ = (gaze₀_x - eye_x, gaze₀_y - eye_y, 0 - eye_z)
ray₁ = (gaze₁_x - eye_x, gaze₁_y - eye_y, 0 - eye_z)
θ = acos(ray₀ · ray₁ / (|ray₀| × |ray₁|))
velocity = θ / Δt
```
- **Pros**: Physically correct, accounts for full 3D geometry
- **Cons**: Requires full eye position (X, Y, Z coordinates), slightly slower
- **Use case**: Accurate velocity measurement, research applications
- **Difference**: Typically 1-5% lower velocities than Olsen 2D

### Performance Comparison

Tested on 4860 samples (~14.5 seconds at 333 Hz):

| Method | Mean Velocity | Processing Time | Relative Speed |
|--------|---------------|-----------------|----------------|
| Olsen 2D (no rounding) | 14.76 deg/s | ~50ms | 1.0x (baseline) |
| Olsen 2D (with rounding) | 14.53 deg/s | ~52ms | 0.96x |
| Ray 3D (no rounding) | 14.54 deg/s | ~65ms | 0.77x |
| Ray 3D (with rounding) | 14.32 deg/s | ~67ms | 0.75x |

**Velocity Differences:**
- Olsen 2D vs Ray 3D: 0.22 deg/s (1.5%) average difference
- Rounding effect: 0.20-0.25 deg/s (1.5-2%) reduction
- Largest differences occur during fast saccades (600+ deg/s): 2-5% difference

**Recommendation:**
- Use **Olsen 2D** for real-time processing and Tobii compatibility
- Use **Ray 3D** for research, publications, and accurate measurements
- Use **coordinate rounding** (`halfup` or `nearest`) to match Tobii's discrete behavior

---

## Troubleshooting

### Common Issues

**1. KeyError: 'time_ms' or missing columns**

Error occurs when using raw Tobii exports instead of preprocessed files.

Solution:
```bash
# Preprocess the Tobii export first
python -c "from extractor import convert_tobii_tsv_to_ivt_tsv; \
  convert_tobii_tsv_to_ivt_tsv('raw_export.tsv', 'preprocessed.tsv')"

# Then run the classifier
python -m ivt_filter.cli --input preprocessed.tsv --output results.tsv
```

**2. NaN velocities or zero velocity counts**

Possible causes:
- Invalid eye position data (missing eye_x, eye_y, eye_z)
- Insufficient valid samples in time window
- Sampling rate too low for chosen window size

Solution:
```bash
# Check data quality
python -c "import pandas as pd; \
  df = pd.read_csv('input.tsv', sep='\t', decimal=','); \
  print(df[['eye_x_mm', 'eye_y_mm', 'eye_z_mm']].describe())"

# Try larger window or different window strategy
python -m ivt_filter.cli --input data.tsv --window 40 --allow-asymmetric-window
```

**3. Ray 3D method fails or returns zeros**

The Ray 3D method requires complete eye position data (x, y, z). If eye_x or eye_y are missing, it falls back to defaults (0, 0, 600).

Solution:
- Verify your data includes eye position columns: `eye_left_x_mm`, `eye_left_y_mm`, etc.
- Use Olsen 2D method if eye X/Y data is unavailable
- Check that eye positions are in reasonable range (not all zeros)

**4. Different results than Tobii Pro Lab**

Several factors can cause differences:
- Velocity method: Use `--velocity-method olsen2d` (Tobii uses 2D approximation)
- Coordinate rounding: Use `--coordinate-rounding nearest` or `halfup`
- Window calculation: Try `--symmetric-round-window` or `--fixed-window-samples`
- Sampling rate: Use `--sampling-rate-method first_100` (Tobii's method)
- Time calculation: Use `--dt-calculation-method mean` (Tobii uses mean)

Tobii-like configuration:
```bash
python -m ivt_filter.cli \
  --input data.tsv \
  --velocity-method olsen2d \
  --coordinate-rounding nearest \
  --window 20 \
  --threshold 30 \
  --sampling-rate-method first_100 \
  --dt-calculation-method mean \
  --symmetric-round-window
```

---

## Testing

### Built-in Unittest

```bash
python -m unittest discover -v
```

### Pytest

```bash
pytest                 # run all tests
pytest -q              # quiet output
pytest --maxfail=1     # stop at first failure
pytest --cov=app       # measure coverage
```

---

## Docker

**Build image**

  ```bash

docker build -t my-python-app\:latest .

````

**Run container**  
  ```bash
docker run --rm my-python-app:latest
````

 **Interactive shell**

  ```bash
docker run --rm -it my-python-app\:latest bash
````

**Expose port (e.g. for web services)**  
  ```bash
docker run --rm -p 8000:8000 my-python-app:latest
````

---

## Changelog

### Version 2.0 (Current)

**Major Features:**
- ✨ Added 3D Ray Angle velocity calculation method (`--velocity-method ray3d`)
- ✨ Added coordinate rounding strategies (`--coordinate-rounding`)
  - `nearest`: Banker's rounding (round half to even)
  - `halfup`: Classical rounding (0.5 always up)
  - `floor`, `ceil`: Directional rounding
- 🏗️ Refactored to Strategy Pattern for extensibility (SOLID principles)
- 📊 Enhanced configuration options for window calculation
- 📝 Comprehensive documentation and examples

**Architecture Improvements:**
- Strategy pattern for velocity calculation methods
- Strategy pattern for coordinate rounding
- Strategy pattern for spatial smoothing
- Strategy pattern for window rounding
- Improved modularity and testability

**Performance:**
- Olsen 2D: ~50ms for 4860 samples (baseline)
- Ray 3D: ~65ms for 4860 samples (77% of baseline)

**Breaking Changes:**
- None - fully backward compatible with version 1.x

### Version 1.0

**Initial Features:**
- Olsen-style velocity calculation
- Time-based and sample-based window selection
- Eye mode selection (left, right, average)
- Spatial smoothing (median, moving average)
- Gap filling for missing data
- I-VT threshold classification
- Ground truth evaluation
- Event-level output generation

---

## Release & Publishing

### Create a new release

```bash
# 1) bump version in setup.py, commit, then:
git tag vX.Y.Z
git push origin --tags
```

### Push Docker image to GitHub Container Registry

```bash
docker tag my-python-app:latest ghcr.io/cemgr/my-python-app:latest
docker push ghcr.io/cemgr/my-python-app:latest
```

### Publish to PyPI

```bash
python -m build
python -m twine upload dist/*
```

---

## Contributing

Contributions are welcome! This project follows SOLID principles and uses the Strategy Pattern for extensibility.

### Adding a New Velocity Calculation Method

1. Create a new class in `ivt_filter/velocity_calculation.py`:
```python
class MyNewMethod(VelocityCalculationStrategy):
    def calculate_visual_angle(self, x1_mm, y1_mm, x2_mm, y2_mm, 
                               eye_x_mm, eye_y_mm, eye_z_mm) -> float:
        # Your implementation here
        pass
    
    def get_description(self) -> str:
        return "My New Method: description"
```

2. Add to the factory function in `velocity.py`:
```python
def _get_velocity_calculation_strategy(method: str) -> VelocityCalculationStrategy:
    if method == "mynew":
        return MyNewMethod()
    # ... existing methods
```

3. Update the config type in `config.py`:
```python
velocity_method: Literal["olsen2d", "ray3d", "mynew"] = "olsen2d"
```

4. Add CLI option in `cli.py`:
```python
parser.add_argument(
    "--velocity-method",
    choices=["olsen2d", "ray3d", "mynew"],
    # ...
)
```

### Development Setup

```bash
# Clone and setup
git clone https://github.com/cemGr/Tobii-I-VT-Filter-Reconstruction.git
cd Tobii-I-VT-Filter-Reconstruction
python -m venv .venv
source .venv/bin/activate  # or .\.venv\Scripts\Activate.ps1 on Windows
pip install -r requirements.txt
pip install -e .

# Run tests
pytest

# Check code style
flake8 ivt_filter/
black ivt_filter/
```

---

## Citation

If you use this software in your research, please cite:

```bibtex
@thesis{your_thesis,
  title={Reconstruction of I-VT Filters by Tobii Pro Lab},
  author={Your Name},
  year={2024},
  school={Your University},
  type={Bachelor's Thesis}
}
```

---

## License

See [LICENSE](LICENSE) file for details.

---

## Contact

For questions, issues, or contributions:
- GitHub Issues: https://github.com/cemGr/Tobii-I-VT-Filter-Reconstruction/issues
- GitHub: [@cemGr](https://github.com/cemGr)

---

## Acknowledgments

This project implements velocity-based eye movement classification inspired by:
- Olsen, A. (2012). The Tobii I-VT Fixation Filter
- Salvucci, D. D., & Goldberg, J. H. (2000). Identifying fixations and saccades in eye-tracking protocols

Special thanks to Tobii Pro for their comprehensive documentation of the I-VT algorithm.
