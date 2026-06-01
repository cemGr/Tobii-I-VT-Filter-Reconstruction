# Tobii I-VT Filter Reconstruction

A from-scratch Python implementation of Tobii's I-VT (Velocity-Threshold) filter. Classifies eye-tracking data into fixations, saccades, and unclassified samples based on gaze velocity.

## What Does This Do?

Eye-trackers record raw sample data at high frequency (typically 60–1200 Hz). To understand what someone is looking at, you need to:

1. **Recognize fixations**: periods where the eye is relatively stable 
2. **Recognize saccades**: rapid eye movements between fixations 
3. **Ignore unclassified noise**: invalid data, blinks, or ambiguous movements

This tool automates that classification using a velocity-based approach: if the eye moves faster than a threshold (e.g., 30°/s), it's a saccade; otherwise it's a fixation.

---

## How the Algorithm Works

The filter applies a processing pipeline to each eye-tracking dataset:

```
Raw eye-tracking data
    ↓
[1] Fill temporal gaps (interpolation)
[2] Tobii eye-offset interpolation (optional)
    ↓
[3] Select eye (left, right, or average)
    ↓
[4] Reduce noise (smoothing)
    ↓
[5] Calculate velocity (degrees/second)
    ↓
[6] Classify (Fixation if v < threshold, else Saccade)
    ↓
[7] Merge nearby fixations (post-processing)
    ↓
[8] Filter short fixations (optional)
    ↓
Output: classified eye-tracking events
```

### Key Concepts

**Velocity Calculation**: The algorithm computes how fast the eye is moving in 2D or 3D space. We support four methods:

- **Olsen 2D** (simple, Tobii-compatible): Uses only the Z distance from eye to screen. Fast but less accurate for lateral movements.
- **Ray 3D** (accurate): Full 3D ray-casting between eye position and gaze point. Most accurate but requires complete eye position data.
- **Ray 3D with Gaze Direction** (robust): Uses DACS NORM gaze direction vectors.
- **Tobii Gaze Dir** (Tobii-exact): Uses the same angular formula as Tobii Pro Lab for maximum compatibility.

**Windowing**: Velocity is calculated over a small window of samples (recommended 20 ms (3 samples for 120 Hz)). Larger windows = smoother but less responsive; smaller windows = more responsive but noisier. (Always depends on the frequency!)

**Smoothing** (optional): Apply median filter or moving average to reduce noise before classification. Helps with micro-fixations caused by measurement jitter.

**Post-Processing**: Merge saccades separated by brief periods (e.g., <75 ms) and discard very short fixations (<60 ms) that are likely noise.

### Strict I-VT Baseline and Optional Reconstruction Heuristics

The default classifier is a strict I-VT baseline. After validity handling, each finite velocity is classified with one inclusive threshold comparison:

- `velocity < threshold` → `Fixation`
- `velocity >= threshold` → `Saccade`
- invalid eye samples → `EyesNotFound`
- missing or non-finite velocity → `Unclassified`

Reconstruction heuristics are opt-in and are not part of the baseline algorithm:

| Option | Behavior when enabled |
|--------|-----------------------|
| `--enable-invalid-window-neighbor-confirmation` | Requires an adjacent above-threshold velocity before an invalid-window sample can become a saccade. |
| `--enable-hysteresis` | Retains the previous motion label while velocity remains in the band immediately below the saccade threshold. |
| `--hysteresis-width <deg/s>` | Sets the width of that optional hysteresis band (default: `2.0`). |
| `--enable-near-threshold-hybrid` | Allows alternative-velocity refinement near the threshold. |
| `--enable-eye-jump-rule` | Allows alternative-velocity correction for eye-position jumps. |
| `--confident-switch-enabled` | Allows confident alternative-velocity switching away from the threshold. |

Classifier output contains diagnostic columns including `classifier_refinement_rules_enabled`, `classifier_invalid_window_neighbor_confirmation_enabled`, `classifier_hysteresis_enabled`, and `classifier_hysteresis_width_deg_per_sec`, so exported runs state which optional rules were active.

---

## Installation & Setup

### 1. Clone and Install

```bash
git clone git@github.com:cemGr/Tobii-I-VT-Filter-Reconstruction.git
cd Tobii-I-VT-Filter-Reconstruction

python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
```

Plotting is optional. Install the plotting extra only if you want to generate figures:

```bash
pip install tobii-ivt-filter[plot]
```

### Supported Python Import Paths

Use the canonical module paths below in applications and examples. The package does not
maintain duplicate root-level compatibility modules for these APIs.

| API | Supported import path |
|-----|-----------------------|
| `IVTPipeline` | `ivt_filter.io.pipeline` |
| `PipelineObserver`, `ConsoleReporter`, `MetricsLogger`, `ExperimentTracker`, `ResultsPlotter` | `ivt_filter.io.observers` |
| `ExperimentConfig`, `ExperimentManager` | `ivt_filter.evaluation.experiment` |
| `estimate_sampling_rate` | `ivt_filter.utils.sampling` |
| Window sizing helpers | `ivt_filter.utils.window_utils` |

### 2. Prepare Your Data

The filter expects TSV (tab-separated values) with these columns:

```
time_us    gaze_left_x_mm  gaze_left_y_mm  eye_left_z_mm  validity_left  ...
0          123.45          456.78          450.0          0
1000       123.46          456.79          450.1          0
2000       123.47          456.80          450.0          0
```

**Required columns**:
- `time_us` or `time_ms`: Timestamp
- `gaze_left_x_mm`, `gaze_left_y_mm`: Left eye gaze position (mm)
- `gaze_right_x_mm`, `gaze_right_y_mm`: Right eye gaze position (mm)
- `eye_left_z_mm`, `eye_right_z_mm`: Eye-to-screen distance (mm)
- `validity_left`, `validity_right`: Validity code (0=valid, else=invalid)

**Optional**:
- `eye_left_x_mm`, `eye_left_y_mm`: Left eye position (needed for Ray 3D)
- `eye_right_x_mm`, `eye_right_y_mm`: Right eye position
- Ground truth column (e.g., `Eye movement type`) for evaluation

---

## Quick Workflow Example

Here's a typical workflow for processing a file:

### Step 1: Extract Data from Raw Format

If you export from Tobii Pro Lab, use the built-in extractor:

```bash
python extractor.py --input raw_tobii_export.tsv --output data.tsv --timestamp-unit auto
```

This converts the Tobii export to the slim format needed by the I-VT filter. It automatically:
- Detects timestamp units and populates millisecond and microsecond timestamps
- Maps Tobii column names to standard format
- Excludes calibration samples
- Removes rows without stimulus names

If your data is in another format (e.g., CSV), convert to the expected columns (see "Prepare Your Data").

### Step 2: Run the I-VT Filter

Apply the recommended settings:

```bash
python -m ivt_filter.cli \
  --input data.tsv \
  --output data_classified.tsv \
  --eye average \
  --velocity-method tobii_gaze_dir \
  --window 20 \
  --auto-fixed-window-from-ms \
  --threshold 30 \
  --smoothing median_strict \
  --smooth-window-samples 3 \
  --shifted-valid-window \
  --shifted-valid-fallback shrink \
  --classify \
  --merge-close-fixations \
  --merge-fix-max-gap-ms 75 \
  --merge-fix-max-angle-deg 0.5 \
  --discard-short-fixations \
  --min-fixation-duration-ms 60 \
  --time-column time_us \
  --time-unit us \
  --with-events
```

### What Each Parameter Does

| Parameter | Value | Purpose |
|-----------|-------|---------|
| `--input` | `data.tsv` | Input file path |
| `--output` | `data_classified.tsv` | Output file path |
| `--eye` | `average` | Use both eyes (left, right, or average) |
| `--velocity-method` | `tobii_gaze_dir` | How to calculate velocity (Tobii-exact, maximum compatibility) |
| `--window` | `20` | Time window in milliseconds |
| `--auto-fixed-window-from-ms` | - | Auto-convert ms window to sample window |
| `--threshold` | `30` | Saccade threshold: 30°/s |
| `--smoothing` | `median_strict` | Apply strict median smoothing (skip if window contains invalid samples) |
| `--smooth-window-samples` | `3` | Smooth over 3 samples |
| `--shifted-valid-window` | - | Handle windows that cross validity changes gracefully |
| `--shifted-valid-fallback` | `shrink` | Shrink window if needed (vs unclassified) |
| `--classify` | - | Classify samples as Fixation/Saccade |
| `--merge-close-fixations` | - | Merge fixations that are temporally/spatially close |
| `--merge-fix-max-gap-ms` | `75` | Max time gap between fixations to merge (ms) |
| `--merge-fix-max-angle-deg` | `0.5` | Max angular distance between fixation centers to merge (°) |
| `--discard-short-fixations` | - | Remove fixations <60 ms (noise) |
| `--time-column` | `time_us` | Name of your timestamp column |
| `--time-unit` | `us` | Unit of timestamps (us or ms) |
| `--with-events` | - | Output event-level grouping |
| `--evaluate` | - | If you have ground truth, compute metrics (optional) |

---

## Understanding the Parameters

### Velocity Method

**Tobii Gaze Dir** (`tobii_gaze_dir`, recommended for Tobii data):
- Uses the same angular formula as Tobii Pro Lab: θ = 2·asin(‖v₁−v₂‖/2)
- Numerically more stable than acos(dot product) for small angles
- Requires normalized gaze direction vectors (DACS norm)
- Use when you need results that match Tobii Pro Lab exactly

**Ray 3D with Gaze Direction** (`ray3d_gaze_dir`):
- Uses normalized gaze direction vectors (DACS norm), acos(dir₀·dir₁)
- Does not require screen or eye position data
- Robust for real-world data where eye positions are incomplete

**Ray 3D** (`ray3d`):
- Requires both X/Y/Z eye position (not all eye-trackers provide this)
- Pure 3D calculation, very accurate
- Falls back to Olsen 2D if eye position is missing

**Olsen 2D** (`olsen2d`):
- Uses only Z distance; ignores X/Y eye position
- Fast, but less accurate for side-to-side movements
- Use only if you need compatibility with the Olsen (2012) paper or eye position data is unavailable

### Window Selection

**Auto-Convert ms to Samples** (recommended):
```bash
--window 20 --auto-fixed-window-from-ms
```
- Automatically converts time window (e.g., 20 ms) to sample window
- Adapts to your actual sampling rate (e.g., 20 ms @ 120 Hz → 3 samples)
- Best of both worlds: specify intuitive ms values, get stable sample windows
- Uses `--dt-calculation-method` (mean or median) for conversion

**Fixed Sample Window** (explicit):
```bash
--fixed-window-samples 3
```
- Directly specify number of samples
- Most consistent across different sampling rates
- 3 samples gives good balance: not too noisy, not too sluggish
- At 120 Hz: ~25 ms window; at 60 Hz: ~50 ms window

**Time-Based Window** (not recommended):
```bash
--window 20
```
- Fixed 20 ms window without sample conversion
- Can be unstable if sampling rate varies
- Use `--auto-fixed-window-from-ms` instead for better results



### Smoothing

**Median Filter – Strict** (recommended):
```bash
--smoothing median_strict --smooth-window-samples 3
```
- Reduces jitter without blurring saccades
- Robust to outliers
- Strict variant: skips smoothing entirely if the window contains any invalid samples
- Good for noisy data

**Median Filter – Adaptive**:
```bash
--smoothing median_adaptive --smooth-window-samples 5
```
- Collects only valid samples within the window
- Can search beyond the nominal window if needed (set with `--smoothing-expansion-radius`)
- Use when invalid samples are frequent and you still want smoothing

**Moving Average**:
```bash
--smoothing moving_average_strict --smooth-window-samples 5
```
- Simple but aggressive
- Blurs saccades slightly
- Use if median isn't enough

**No Smoothing**:
```bash
--smoothing none
```
- Use when comparing to published Tobii results or for diagnostics
- Can be noisier

### Preprocessing: Tobii Eye-Offset Interpolation

```bash
--tobii-eye-offset-interpolation
```

When one eye drops out briefly (a common Tobii artifact), the missing eye position is reconstructed using the last known left-to-right eye offset. This prevents phantom velocities that would otherwise appear at the edges of data gaps when running in `average` eye mode.

Use whenever:
- You are processing Tobii data with `--eye average`
- Your data has short single-eye outages

### Shifted Valid Window

```bash
--shifted-valid-window --shifted-valid-fallback shrink
```
- Handles windows that span valid→invalid transitions
- `shrink`: Use shorter window if no constant-length valid block exists
- `unclassified`: Mark sample as unclassified instead
- Keeps you from getting NaN velocities due to one bad sample

### Post-Processing

**Merge Close Fixations**:
```bash
--merge-close-fixations --merge-fix-max-gap-ms 75 --merge-fix-max-angle-deg 0.5
```
- Two fixations separated by <75 ms and <0.5° → merge into one
- Reduces fragmentation from brief saccades
- Adjust to your use case (e.g., stricter for precise gaze analysis)

**Discard Short Fixations**:
```bash
--discard-short-fixations --min-fixation-duration-ms 60
```
- Removes fixations shorter than 60 ms
- Helps with noise-induced micro-fixations

---

## Velocity Calculation in Detail

### The Problem

Raw eye-tracking gives you position (X, Y) at each millisecond. You need *velocity* to classify:

```
Sample 1: (100, 200) at t=0 ms
Sample 2: (105, 205) at t=5 ms  ← only 5 mm movement
Sample 3: (130, 225) at t=10 ms ← large jump (25 mm)
```

Is this a fixation or saccade? Depends on how fast that jump is. But how do you convert mm to deg/s?

### The Solution: Ray Casting

Imagine a ray from your eye through your gaze point into the world. When you move your eye, that ray rotates. The rotation speed is the *angular velocity* in degrees/second.

**Ray 3D Method**:

```
Eye position:     E = (ex, ey, ez)
Gaze point:       G = (gx, gy, gz)
Ray from E to G:  R = G - E

Previous ray: R_old
Current ray:  R_new

Angle between them: θ = acos(R_old · R_new / (|R_old| × |R_new|))
Angular velocity:  v_deg_per_sec = θ / Δt × (180 / π)
```

**Tobii Gaze Dir / Ray 3D Gaze Dir** (using normalized direction vectors):

```
dir_old, dir_new: normalized gaze direction vectors

Tobii Gaze Dir:       θ = 2·asin(‖dir_new − dir_old‖ / 2)
Ray 3D Gaze Dir:      θ = acos(dir_old · dir_new)
```

**Olsen 2D Method** (simplified):

Uses only Z distance (eye-to-screen) and 2D gaze offsets:

```
Angle: θ = atan(screen_distance / eye_z)
```

Faster but ignores X/Y eye position changes.

### Example Calculation

Imagine:
- Eye at (0, 0, 600 mm) from screen
- Looking at screen point (50, 0) mm
- Moves to (60, 0) mm in 10 ms

Ray 3D:
```
R_old = (50, 0, -600)  → |R_old| = 610 mm
R_new = (60, 0, -600)  → |R_new| = 611 mm
dot product ≈ 600² + 50×60 = 366000
cos(θ) ≈ 0.999...
θ ≈ 0.47°
v ≈ 47°/s  ← Saccade (if threshold=30)
```

---

## Output Format

After running the filter, your TSV includes these new columns:

| Column | Example | Meaning |
|--------|---------|---------|
| `velocity_deg_per_sec` | 2.5 | Calculated gaze velocity |
| `ivt_sample_type` | Fixation | Sample classification |
| `ivt_event_type` | Fixation | Event-level classification |
| `ivt_event_index` | 5 | Which fixation/saccade this belongs to |

---

## Practical Tips

### 1. Use Shifted Valid Window

Always include these flags:
```bash
--shifted-valid-window --shifted-valid-fallback shrink
```

Prevents NaN velocities when a validity code changes in the middle of a window.

### 2. Use Auto Window Conversion

Instead of manually calculating samples, let the tool convert for you:
```bash
--window 20 --auto-fixed-window-from-ms  # Converts 20 ms to samples based on your data
```

Adapts to your actual sampling rate automatically.

### 3. Median Smoothing for Real Data

Real eye-trackers are noisy. Median filter helps:
```bash
--smoothing median_strict --smooth-window-samples 3
```

### 4. Post-Process to Clean Up

Always merge close fixations and discard short ones:
```bash
--merge-close-fixations --merge-fix-max-gap-ms 75 --merge-fix-max-angle-deg 0.5 \
--discard-short-fixations --min-fixation-duration-ms 60
```

This reduces fragmentation from noise.

### 5. Validate with Ground Truth

If you have labeled data, run with `--evaluate`:
```bash
python -m ivt_filter.cli --input data.tsv --output out.tsv --classify --evaluate
```

The evaluation outputs two complementary metrics:

**Sample-level** (Cohen's Kappa): Agreement per sample. A kappa >0.8 indicates good agreement. Fast to compute, but sensitive to fragmentation — a single misclassified sample can split one ground-truth event into three predicted events, inflating the error count.

**Event-level** (Maximum IoU, Startsev & Zemblys 2022): Each ground-truth event is matched 1-to-1 to the best-overlapping predicted event by Intersection-over-Union. Provides:
- Event confusion matrix (GT class → matched Pred class or FN)
- False positives (predicted events with no GT match)
- Timing quality per class: mean onset and offset deviation in ms

Use event-level metrics when fragmentation is a concern or when you care about how well the filter boundaries align with ground truth in time.

---

## Common Issues & Fixes

### Getting NaN Velocities?

**Cause**: Not enough valid samples in window or validity codes changing.

**Fix**:
```bash
--shifted-valid-window --shifted-valid-fallback shrink
```

### Phantom Velocities at Gap Edges?

**Cause**: When one eye drops out briefly and you use `--eye average`, the sudden switch between one-eye and two-eye averages creates artificial velocity spikes.

**Fix**:
```bash
--tobii-eye-offset-interpolation
```

Reconstructs the missing eye using the last known left-to-right offset, keeping the averaged position smooth across short outages.

### Results Don't Match Tobii Pro Lab?

Tobii Pro Lab uses `tobii_gaze_dir` with specific windowing. Match it:
```bash
--velocity-method tobii_gaze_dir --fixed-window-samples 3 --smoothing none
```

For the classic Olsen 2D match:
```bash
--velocity-method olsen2d --fixed-window-samples 3 --smoothing none
```

### Too Many Micro-Fixations?

Increase minimum fixation duration or smoothing:
```bash
--discard-short-fixations --min-fixation-duration-ms 100 \
--smoothing median_strict --smooth-window-samples 5
```



## Full Example Workflow

Here's a complete real-world example:

```bash
#!/bin/bash

INPUT="raw_eyetracking.tsv"
OUTPUT="eyetracking_classified.tsv"

# Classify with recommended settings
python -m ivt_filter.cli \
  --input "$INPUT" \
  --output "$OUTPUT" \
  --eye average \
  --velocity-method tobii_gaze_dir \
  --window 20 \
  --auto-fixed-window-from-ms \
  --threshold 30 \
  --smoothing median_strict \
  --smooth-window-samples 3 \
  --tobii-eye-offset-interpolation \
  --shifted-valid-window \
  --shifted-valid-fallback shrink \
  --classify \
  --merge-close-fixations \
  --merge-fix-max-gap-ms 75 \
  --merge-fix-max-angle-deg 0.5 \
  --discard-short-fixations \
  --min-fixation-duration-ms 60 \
  --time-column time_us \
  --time-unit us \
  --with-events

echo "Classification complete. Saved to $OUTPUT"

# Quick statistics
python3 << 'EOF'
import pandas as pd
df = pd.read_csv("$OUTPUT", sep='\t')
fixations = (df['ivt_event_type'] == 'Fixation').sum()
saccades = (df['ivt_event_type'] == 'Saccade').sum()
print(f"Fixations: {fixations}")
print(f"Saccades: {saccades}")
EOF
```

---

## Docker Usage

Build a minimal runtime image and run the CLI inside the container.

```bash
# Build locally
docker build -t ivt-filter:latest .

# Run with a mounted data folder
docker run --rm \
    -v "$(pwd)/data:/data" \
    ivt-filter:latest \
    --input /data/input.tsv \
    --output /data/output.tsv \
    --eye average \
    --velocity-method tobii_gaze_dir \
    --window 20 --auto-fixed-window-from-ms \
    --threshold 30 \
    --smoothing median_strict --smooth-window-samples 3 \
    --tobii-eye-offset-interpolation \
    --shifted-valid-window --shifted-valid-fallback shrink \
    --classify --with-events \
    --time-column time_us --time-unit us
```

Notes:
- The container entrypoint is `python -m ivt_filter.cli`; pass flags as shown.
- Mount input/output via `-v` to persist results on the host.
- For GHCR, images are published under `ghcr.io/<owner>/tobii-i-vt-filter-reconstruction`.

---

## Architecture

The code is organized by processing stage:

```
ivt_filter/
├── preprocessing/   # Gap fill, eye selection, smoothing
├── processing/      # Velocity calculation, classification
├── postprocessing/  # Merge fixations, discard short ones
├── evaluation/      # Metrics against ground truth (sample-level + event IoU)
├── config/          # Configuration management
├── strategies/      # Algorithm implementations
└── utils/           # Helper functions
```

Most users just need the CLI (`python -m ivt_filter.cli`). Developers can import modules directly:

```python
from ivt_filter import compute_olsen_velocity, apply_ivt_classifier
from ivt_filter.postprocessing.merge_fixations import merge_adjacent_fixations
from ivt_filter.evaluation.event_iou import compute_event_iou_metrics

# ... use functions directly
```

---

## References

- Olsen, A. (2012). *The Tobii I-VT Fixation Filter*
- Salvucci, D. D., & Goldberg, J. H. (2000). *Identifying fixations and saccades in eye-tracking protocols*
- Startsev, M., & Zemblys, R. (2022). *Evaluating eye movement event detection: A review of the state of the art*. Behavior Research Methods, 54, 1653–1714. https://doi.org/10.3758/s13428-021-01763-7

---

## License

MIT License. See [LICENSE](LICENSE) for details.
