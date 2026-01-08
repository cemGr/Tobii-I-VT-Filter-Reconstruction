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

The filter applies a 7-step processing pipeline to each eye-tracking dataset:

```
Raw eye-tracking data
    ↓
[1] Fill temporal gaps (interpolation)
    ↓
[2] Select eye (left, right, or average)
    ↓
[3] Reduce noise (smoothing)
    ↓
[4] Calculate velocity (degrees/second)
    ↓
[5] Classify (Fixation if v < threshold, else Saccade)
    ↓
[6] Merge nearby fixations (post-processing)
    ↓
[7] Filter short fixations (optional)
    ↓
Output: classified eye-tracking events
```

### Key Concepts

**Velocity Calculation**: The algorithm computes how fast the eye is moving in 2D or 3D space. We support three methods:

- **Olsen 2D** (simple, Tobii-compatible): Uses only the Z distance from eye to screen. Fast but less accurate for lateral movements.
- **Ray 3D** (accurate): Full 3D ray-casting between eye position and gaze point. Most accurate but requires complete eye position data.
- **Ray 3D with Gaze Direction** (robust): Uses DACS NORM Gaze direction.

**Windowing**: Velocity is calculated over a small window of samples (recommended 20 ms (3 samples for 120HZ)). Larger windows = smoother but less responsive; smaller windows = more responsive but noisier. (Always depends on the frequency!)

**Smoothing** (optional): Apply median filter or moving average to reduce noise before classification. Helps with micro-fixations caused by measurement jitter.

**Post-Processing**: Merge saccades separated by brief periods (e.g., <75 ms) and discard very short fixations (<60 ms) that are likely noise.

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
python extractor.py raw_tobii_export.tsv data.tsv
```

This converts the Tobii export to the slim format needed by the I-VT filter. It automatically:
- Detects timestamp units (ms/us)
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
  --velocity-method ray3d_gaze_dir \
  --window 20 \
  --auto-fixed-window-from-ms \
  --threshold 30 \
  --smoothing-mode median_strict \
  --smoothing-window-samples 3 \
  --shifted-valid-window \
  --shifted-valid-fallback shrink \
  --classify \
  --merge-adjacent-fixations \
  --max-time-gap-ms 75 \
  --max-angle-deg 0.5 \
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
| `--velocity-method` | `ray3d_gaze_dir` | How to calculate velocity (accurate, handles mixed validity) |
| `--window` | `20` | Time window in milliseconds |
| `--auto-fixed-window-from-ms` | - | Auto-convert ms window to sample window |
| `--threshold` | `30` | Saccade threshold: 30°/s |
| `--smoothing-mode` | `median` | Apply median smoothing (robust to outliers) |
| `--smoothing-window-samples` | `3` | Smooth over 3 samples |
| `--shifted-valid-window` | - | Handle windows that cross validity changes gracefully |
| `--shifted-valid-fallback` | `shrink` | Shrink window if needed (vs expand/ignore) |
| `--classify` | - | Classify samples as Fixation/Saccade |
| `--merge-adjacent-fixations` | - | Merge saccades <75 ms or 0.5° apart |
| `--discard-short-fixations` | - | Remove fixations <60 ms (noise) |
| `--time-column` | `time_us` | Name of your timestamp column |
| `--time-unit` | `us` | Unit of timestamps (us or ms) |
| `--with-events` | - | Output event-level grouping |
| `--evaluate` | - | If you have ground truth, compute metrics (optional) |

---

## Understanding the Parameters

### Velocity Method

**Ray 3D with Gaze Direction** (recommended):
- Uses full 3D geometry of eye position and gaze point
- Automatically picks the more reliable eye if one is invalid
- Most robust for real-world data
- Trade-off: Slightly slower than Olsen 2D (~20% more CPU)

**Ray 3D**:
- Requires both X/Y/Z eye position (not all eye-trackers provide this)
- Pure 3D calculation, very accurate
- Falls back to Olsen 2D if eye position is missing

**Olsen 2D**:
- Uses only Z distance; ignores X/Y eye position
- Fast, but less accurate for side-to-side movements
- Use only if you need exact Tobii Pro Lab compatibility or eye position data is unavailable

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

**Median Filter** (recommended):
```bash
--smoothing-mode median_strict --smoothing-window-samples 3
```
- Reduces jitter without blurring saccades
- Robust to outliers
- Good for noisy data

**Moving Average**:
```bash
--smoothing-mode moving_average_strict --smoothing-window-samples 5
```
- Simple but aggressive
- Blurs saccades slightly
- Use if median isn't enough

**No Smoothing**:
```bash
--smoothing-mode none
```
- Use when comparing to published Tobii results
- Can be noisier

### Shifted Valid Window

```bash
--shifted-valid-window --shifted-valid-fallback shrink
```
- Handles windows that span valid→invalid transitions
- `shrink`: Use shorter window if needed
- `expand`: Wait for valid data
- Keeps you from getting NaN velocities due to one bad sample

### Post-Processing

**Merge Adjacent Fixations**:
```bash
--merge-adjacent-fixations --max-time-gap-ms 75 --max-angle-deg 0.5
```
- Two fixations separated by <75 ms or <0.5° difference → merge into one
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
--smoothing-mode median_strict --smoothing-window-samples 3
```

### 4. Post-Process to Clean Up

Always merge adjacent fixations and discard short ones:
```bash
--merge-adjacent-fixations --max-time-gap-ms 75 --max-angle-deg 0.5 \
--discard-short-fixations --min-fixation-duration-ms 60
```

This reduces fragmentation from noise.

### 5. Validate with Ground Truth

If you have labeled data, run with `--evaluate`:
```bash
python -m ivt_filter.cli --input data.tsv --output out.tsv --classify --evaluate
```

Check Cohen's Kappa (should be >0.8 for good agreement).

---

## Common Issues & Fixes

### Getting NaN Velocities?

**Cause**: Not enough valid samples in window or validity codes changing.

**Fix**:
```bash
--shifted-valid-window --shifted-valid-fallback shrink
```

### Results Don't Match Tobii Pro Lab?

Tobii uses Olsen 2D with 3-sample window by default. Match it:
```bash
--velocity-method olsen2d --fixed-window-samples 3 --smoothing-mode none
```

### Too Many Micro-Fixations?

Increase minimum fixation duration or smoothing:
```bash
--discard-short-fixations --min-fixation-duration-ms 100 \
--smoothing-mode median_strict --smoothing-window-samples 5
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
  --velocity-method ray3d_gaze_dir \
  --window 20 \
  --auto-fixed-window-from-ms \
  --threshold 30 \
  --smoothing-mode median_strict \
  --smoothing-window-samples 3 \
  --shifted-valid-window \
  --shifted-valid-fallback shrink \
  --classify \
  --merge-adjacent-fixations \
  --max-time-gap-ms 75 \
  --max-angle-deg 0.5 \
  --discard-short-fixations \
  --min-fixation-duration-ms 60 \
  --time-column time_us \
  --time-unit us \
  --with-events

echo "✓ Classification complete. Saved to $OUTPUT"

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

## Architecture

The code is organized by processing stage:

```
ivt_filter/
├── preprocessing/   # Gap fill, eye selection, smoothing
├── processing/      # Velocity calculation, classification
├── postprocessing/  # Merge fixations, discard short ones
├── evaluation/      # Metrics against ground truth
├── config/          # Configuration management
├── strategies/      # Algorithm implementations
└── utils/           # Helper functions
```

Most users just need the CLI (`python -m ivt_filter.cli`). Developers can import modules directly:

```python
from ivt_filter.processing.velocity import compute_ray3d_velocity
from ivt_filter.processing.classification import classify_ivt
from ivt_filter.postprocessing.merge_fixations import merge_adjacent_fixations

# ... use functions directly
```

---

## References

- Olsen, A. (2012). *The Tobii I-VT Fixation Filter*
- Salvucci, D. D., & Goldberg, J. H. (2000). *Identifying fixations and saccades in eye-tracking protocols*

---

## License

MIT License. See [LICENSE](LICENSE) for details.
