# I-VT Filter - Quick Reference

## Velocity Calculation Methods

### Olsen 2D (Default)
```bash
--velocity-method olsen2d
```
- Fast 2D approximation
- Only needs eye_z (distance)
- Formula: θ = atan(screen_distance / eye_z)
- Use for: Standard filtering, Tobii compatibility

### Ray 3D (Accurate)
```bash
--velocity-method ray3d
```
- Physically correct 3D calculation
- Needs eye_x, eye_y, eye_z
- Formula: θ = acos(ray₀ · ray₁ / (|ray₀| × |ray₁|))
- Use for: Research, accurate measurements
- ~1-5% lower velocities than Olsen 2D

## Coordinate Rounding

```bash
--coordinate-rounding {none|nearest|halfup|floor|ceil}
```

| Mode | Description | Use Case |
|------|-------------|----------|
| `none` | No rounding (default) | Maximum precision |
| `nearest` | Banker's rounding (0.5→even) | Python standard |
| `halfup` | Classical (0.5→up) | Tobii-like |
| `floor` | Always down | Lower velocities |
| `ceil` | Always up | Higher velocities |

Effect: ~1-3% velocity reduction with rounding

## Window Configuration

### Time-based (Default)
```bash
--window 20  # 20ms window
```

### Fixed Samples
```bash
--fixed-window-samples 7  # Always 7 samples
```

### Auto-calculate from Time
```bash
--auto-fixed-window-from-ms  # Calculate samples from time + rate
```

### Symmetric Options
```bash
--sample-symmetric-window      # Equal samples left/right
--symmetric-round-window       # Force odd window size
--allow-asymmetric-window      # Allow even sizes
```

## Common Configurations

### Tobii-like
```bash
python -m ivt_filter.cli \
  --velocity-method olsen2d \
  --coordinate-rounding nearest \
  --window 20 \
  --threshold 30 \
  --sampling-rate-method first_100 \
  --dt-calculation-method mean \
  --symmetric-round-window
```

### High Precision
```bash
python -m ivt_filter.cli \
  --velocity-method ray3d \
  --coordinate-rounding none \
  --window 20 \
  --smoothing median \
  --smooth-window-samples 5
```

### Fast Processing
```bash
python -m ivt_filter.cli \
  --velocity-method olsen2d \
  --coordinate-rounding none \
  --window 20 \
  --smoothing none
```

## Eye Selection

```bash
--eye {left|right|average}
```
- `left`: Left eye only
- `right`: Right eye only
- `average`: Both eyes averaged (default)

## Smoothing

```bash
--smoothing {none|median|moving_average}
--smooth-window-samples 5
```

## Gap Filling

```bash
--gap-fill --gap-fill-max-ms 75
```
Linear interpolation for gaps ≤75ms

## Classification

```bash
--classify --threshold 30  # 30 deg/s threshold
--evaluate                 # Compare with ground truth
--with-events             # Generate event output
```

## Performance Reference

4860 samples (~14.5s at 333 Hz):

| Configuration | Processing Time |
|---------------|-----------------|
| Olsen 2D, no rounding | ~50ms |
| Olsen 2D, with rounding | ~52ms |
| Ray 3D, no rounding | ~65ms |
| Ray 3D, with rounding | ~67ms |

## Troubleshooting

### KeyError: 'time_ms'
→ Preprocess data first:
```bash
python -c "from extractor import convert_tobii_tsv_to_ivt_tsv; \
  convert_tobii_tsv_to_ivt_tsv('raw.tsv', 'prep.tsv')"
```

### NaN velocities
→ Check data quality:
```bash
python -c "import pandas as pd; \
  df = pd.read_csv('input.tsv', sep='\t', decimal=','); \
  print(df[['eye_x_mm', 'eye_y_mm', 'eye_z_mm']].describe())"
```

### Ray 3D returns zeros
→ Requires eye X/Y data. Use Olsen 2D if unavailable.

## Example: Compare Methods

```python
from ivt_filter.config import OlsenVelocityConfig
from ivt_filter.velocity import compute_olsen_velocity
from ivt_filter.io import read_tsv

df = read_tsv("data.tsv")

# Olsen 2D
cfg_olsen = OlsenVelocityConfig(velocity_method="olsen2d")
df_olsen = compute_olsen_velocity(df.copy(), cfg_olsen)

# Ray 3D
cfg_ray = OlsenVelocityConfig(velocity_method="ray3d")
df_ray = compute_olsen_velocity(df.copy(), cfg_ray)

# Compare
v_olsen = df_olsen["velocity_deg_per_sec"].mean()
v_ray = df_ray["velocity_deg_per_sec"].mean()
print(f"Olsen 2D: {v_olsen:.2f} deg/s")
print(f"Ray 3D: {v_ray:.2f} deg/s")
print(f"Difference: {v_olsen - v_ray:.2f} deg/s ({(v_olsen-v_ray)/v_ray*100:.1f}%)")
```

## References

- Olsen, A. (2012). The Tobii I-VT Fixation Filter
- Salvucci & Goldberg (2000). Identifying fixations and saccades
