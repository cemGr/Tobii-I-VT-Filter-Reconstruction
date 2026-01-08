# Velocity Classification Refinement

## Overview

Optional velocity-classification refinement consisting of two strategies:
1. **Stage 1: Near-threshold hybrid** - Uses alternative velocity for samples close to threshold
2. **Rule A: Eye-position jump correction** - Detects and corrects artificial velocity spikes from eye-position shifts

Both features are **disabled by default** and can be enabled independently via CLI flags.

## Stage 1: Near-Threshold Hybrid Strategy

### Purpose
Reduce noise-induced classification flips for samples near the velocity threshold without changing global behavior.

### Logic
For samples where the baseline velocity `v_base` is close to the threshold `T`:
```
|v_base - T| ≤ Band
```
the classification uses an **alternative velocity** `v_alt` instead of `v_base`.

### Alternative Velocity Computation
- Uses endpoint gaze coordinates with **per-sample eye positions** at each endpoint
- Instead of: `ray(middle_eye → gaze_start)` to `ray(middle_eye → gaze_end)`
- Computes: `ray(eye_start → gaze_start)` to `ray(eye_end → gaze_end)`
- Accounts for eye-position variation within the window

### CLI Flags
```bash
--enable-near-threshold-hybrid   # Enable the feature
--near-threshold-band 5.0        # Band around threshold (deg/s), default: 5.0
```

### Example
```bash
python -m ivt_filter.cli \
  --input data.tsv \
  --threshold 30 \
  --enable-near-threshold-hybrid \
  --near-threshold-band 5.0
```

With threshold=30 and band=5.0, refinement applies to samples with velocities in [25.0, 35.0].

## Rule A: Eye-Position Jump Correction

### Purpose
Suppress artificial velocity spikes caused by sudden eye-position shifts (e.g., head movement, tracker recalibration) when using a fixed eye-position reference.

### Logic
Independently checks if:
1. Eye position displacement between window endpoints exceeds threshold:
   ```
   |eyePos(last) - eyePos(first)| ≥ eye_jump_threshold
   ```
2. Baseline velocity indicates clear saccade:
   ```
   v_base ≥ v_strict
   ```

If **both conditions** are met, uses `v_alt` for classification.

### CLI Flags
```bash
--enable-eye-jump-rule              # Enable the feature
--eye-jump-threshold 10.0           # Eye displacement threshold (mm), default: 10.0
--eye-jump-velocity-threshold 50.0  # Velocity for "clear saccade" (deg/s), default: 50.0
```

### Example
```bash
python -m ivt_filter.cli \
  --input data.tsv \
  --threshold 30 \
  --enable-eye-jump-rule \
  --eye-jump-threshold 10.0 \
  --eye-jump-velocity-threshold 50.0
```

## Output Columns

When refinement is enabled, two diagnostic columns are added:

- **`velocity_refined`**: Alternative velocity value (only for refined samples)
- **`refinement_applied`**: Reason for refinement
  - `"near_threshold"` - Stage 1 applied
  - `"eye_jump"` - Rule A applied
  - `""` - No refinement

## Implementation Notes

### Priority
Stage 1 (near-threshold) has priority over Rule A. A sample is only checked for eye-jump if it's outside the near-threshold band.

### Fallback
If alternative velocity cannot be computed (e.g., insufficient valid samples), classification falls back to baseline velocity.

### Performance
- Alternative velocity pre-computed vectorized for all samples before classification
- Minimal overhead: ~1,500 samples refined out of 72,000 (typical dataset)

## Configuration

All parameters in `IVTClassifierConfig`:

```python
@dataclass
class IVTClassifierConfig:
    velocity_threshold_deg_per_sec: float = 30.0
    
    # Stage 1: Near-threshold hybrid
    enable_near_threshold_hybrid: bool = False
    near_threshold_band: float = 5.0
    
    # Rule A: Eye-position jump correction
    enable_eye_jump_rule: bool = False
    eye_jump_threshold_mm: float = 10.0
    eye_jump_velocity_threshold: float = 50.0
```

## Test Results

Tested on `left20ms30_input.tsv` (72,123 samples):
- **1,260 samples** refined via near-threshold hybrid (within 5 deg/s of threshold)
- **301 samples** refined via eye-jump rule
- **Overall accuracy**: 99.89% (unchanged from baseline)
- **Cohen's kappa**: 0.996

The refinement maintains accuracy while providing more stable classifications near the threshold.
