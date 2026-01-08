# User Guide - Simple Eye Tracking Analysis

**For users who just want to analyze eye tracking data without ground truth evaluation.**

## 🎯 Quick Start (2 Minutes)

### Installation

```bash
pip install -e .
```

### Basic Usage

```python
from ivt_filter.simple_api import process_eye_tracking

# Process your data - that's it!
df = process_eye_tracking("my_data.tsv")
```

**Done!** Your data now has:
- ✅ `velocity_deg_per_s` - Eye movement velocity
- ✅ `ivt_sample_type` - Classification (Fixation/Saccade)

---

## 📊 What You Get

### Input (Your Tobii Export)
```
time_ms  gaze_left_x_mm  gaze_left_y_mm  eye_left_z_mm  ...
0        120.5           85.3            650.0          ...
17       121.2           85.8            651.0          ...
33       119.8           84.9            649.5          ...
```

### Output (Processed Data)
```
time_ms  ...  velocity_deg_per_s  ivt_sample_type
0        ...  12.5                Fixation
17       ...  15.3                Fixation
33       ...  145.7               Saccade
```

---

## 🚀 Usage Examples

### 1. Minimal - Just Process

```python
from ivt_filter.simple_api import process_eye_tracking

df = process_eye_tracking("data.tsv")
```

### 2. Save Results

```python
df = process_eye_tracking(
    "data.tsv",
    output="results.tsv"  # Save processed data
)
```

### 3. Custom Threshold

```python
df = process_eye_tracking(
    "data.tsv",
    velocity_threshold=25.0  # More sensitive (default: 30.0)
)
```

### 4. Adaptive (Recommended!)

```python
from ivt_filter.simple_api import process_eye_tracking_adaptive

# Automatically adapts to your sampling rate
df = process_eye_tracking_adaptive(
    "data.tsv",
    n_samples=3  # Use 3 samples for velocity calculation
)
```

### 5. With Statistics

```python
from ivt_filter.simple_api import process_eye_tracking, print_statistics

df = process_eye_tracking("data.tsv")
print_statistics(df)
```

**Output:**
```
=== Eye Tracking Statistics ===
Total samples: 5000
Fixations: 3500 (70.0%)
Saccades: 1500 (30.0%)

Velocity:
  Average: 15.3 deg/s
  Median: 12.1 deg/s
  Maximum: 245.7 deg/s
```

---

## ⚙️ Configuration Options

### Velocity Threshold

Controls fixation vs saccade detection:

```python
# Sensitive - detects small saccades
df = process_eye_tracking("data.tsv", velocity_threshold=20.0)

# Standard (default) - balanced
df = process_eye_tracking("data.tsv", velocity_threshold=30.0)

# Conservative - only large saccades
df = process_eye_tracking("data.tsv", velocity_threshold=40.0)
```

**Recommendation:** Start with 30 deg/s, adjust if needed.

### Window Size

Time window for velocity calculation:

```python
# Short window (more responsive)
df = process_eye_tracking("data.tsv", window_ms=10.0)

# Standard (default)
df = process_eye_tracking("data.tsv", window_ms=20.0)

# Long window (more smoothing)
df = process_eye_tracking("data.tsv", window_ms=40.0)
```

### Eye Selection

```python
# Average both eyes (default, most robust)
df = process_eye_tracking("data.tsv", eye_selection="average")

# Left eye only
df = process_eye_tracking("data.tsv", eye_selection="left")

# Right eye only
df = process_eye_tracking("data.tsv", eye_selection="right")
```

### Velocity Method

```python
# Fast 2D approximation (default)
df = process_eye_tracking("data.tsv", velocity_method="olsen2d")

# Physically correct 3D method
df = process_eye_tracking("data.tsv", velocity_method="ray3d")
```

---

## 📋 Complete Example

```python
from ivt_filter.simple_api import (
    process_eye_tracking_adaptive,
    print_statistics
)

# 1. Process with optimal settings
df = process_eye_tracking_adaptive(
    input_path="my_recording.tsv",
    output_path="results.tsv",
    n_samples=3,
    velocity_threshold=30.0,
    eye_selection="average",
    velocity_method="olsen2d"
)

# 2. Show statistics
print_statistics(df)

# 3. Analyze fixations
fixations = df[df['ivt_sample_type'] == 'Fixation']
print(f"Total fixation time: {len(fixations) / 120:.1f} seconds")  # 120Hz

# 4. Export fixations only
fixations.to_csv("fixations_only.csv", index=False)
```

---

## 🎓 Understanding Results

### Velocity (`velocity_deg_per_s`)

- **Low (~5-25 deg/s)**: Smooth pursuit or fixation
- **Medium (~25-50 deg/s)**: Transition zone
- **High (>50 deg/s)**: Fast saccade

### Classification (`ivt_sample_type`)

- **Fixation**: Eye is relatively stable (velocity < threshold)
- **Saccade**: Eye is moving rapidly (velocity ≥ threshold)

---

## 🔧 Troubleshooting

### "No valid data found"
→ Check your TSV has required columns: `time_ms`, `gaze_*_x_mm`, `gaze_*_y_mm`, `eye_*_z_mm`

### "Too many/few fixations detected"
→ Adjust `velocity_threshold`:
- More fixations needed? **Lower** threshold (e.g., 20.0)
- Fewer fixations needed? **Raise** threshold (e.g., 40.0)

### "Results vary across recordings"
→ Use adaptive sample-based approach:
```python
process_eye_tracking_adaptive("data.tsv", n_samples=3)
```

---

## 📖 Input File Requirements

Your Tobii Pro Lab TSV export must contain:

### Required Columns
- `time_ms` - Timestamp in milliseconds
- `gaze_left_x_mm` - Left eye X gaze position
- `gaze_left_y_mm` - Left eye Y gaze position  
- `gaze_right_x_mm` - Right eye X gaze position
- `gaze_right_y_mm` - Right eye Y gaze position
- `eye_left_z_mm` - Left eye distance to screen
- `eye_right_z_mm` - Right eye distance to screen
- `validity_left` - Left eye validity (0 or 1)
- `validity_right` - Right eye validity (0 or 1)

### Optional Columns
- `gaze_left_x_px`, `gaze_left_y_px` - Pixel coordinates
- Any other columns from your export

---

## 🆚 Simple API vs Advanced Usage

### Use Simple API When:
- ✅ You just want to analyze your eye tracking data
- ✅ No ground truth available
- ✅ Need quick results
- ✅ Production use case

### Use Advanced API When:
- ✅ You have ground truth annotations
- ✅ Need to evaluate accuracy
- ✅ Reverse engineering Tobii's filter
- ✅ Complex parameter sweeps

---

## 💡 Best Practices

### 1. Start Simple
```python
# First, just try with defaults
df = process_eye_tracking("data.tsv")
print_statistics(df)
```

### 2. Use Adaptive for Multiple Recordings
```python
# If you have 60Hz, 120Hz, 300Hz recordings:
df = process_eye_tracking_adaptive("data.tsv", n_samples=3)
```

### 3. Adjust Threshold Based on Task
```python
# Reading task (small saccades)
df = process_eye_tracking("data.tsv", velocity_threshold=20.0)

# Search task (large saccades)
df = process_eye_tracking("data.tsv", velocity_threshold=40.0)
```

### 4. Validate Results
```python
df = process_eye_tracking("data.tsv")
print_statistics(df)

# Check if results make sense:
# - Fixations should be 60-90% of samples
# - Average velocity should be 10-30 deg/s
# - Max velocity should be < 500 deg/s
```

---

## 📚 Next Steps

- **Try it:** Run `python example_simple_usage.py`
- **Advanced:** See `README.md` for CLI and advanced features
- **Experiments:** Check `docs/experiment_tracking.md` for parameter testing
- **Theory:** Read `TOBII_ivt_filter.pdf` for algorithm details

---

## ❓ FAQ

**Q: Do I need ground truth data?**  
A: No! The simple API works without any ground truth.

**Q: What's the difference between time-based and sample-based?**  
A: 
- Time-based: Fixed milliseconds (e.g., 20ms window)
- Sample-based: Fixed samples (e.g., 3 samples, adapts to sampling rate)
- Use sample-based for multiple recordings with different rates.

**Q: Which velocity method should I use?**  
A: Start with `olsen2d` (fast). Use `ray3d` if you need highest accuracy.

**Q: My fixations seem wrong?**  
A: Adjust `velocity_threshold`. Lower = more fixations, higher = fewer fixations.

**Q: Can I use this in production?**  
A: Yes! The simple API is designed for production use. Just call `process_eye_tracking()` and you're done.

---

## 🎯 Summary

**Three lines to process eye tracking data:**

```python
from ivt_filter.simple_api import process_eye_tracking, print_statistics

df = process_eye_tracking("my_data.tsv", output="results.tsv")
print_statistics(df)
```

**That's it!** 🚀
