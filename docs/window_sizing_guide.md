# Window Size Configuration - Quick Guide

## 🎯 What is the difference?

### Time-Based
```python
window_ms = 20.0  # Always 20 ms
```
- ✅ Easy to understand
- ✅ Comparable with the literature
- ❌ Different number of samples at different sampling rates
- ❌ Not optimal for multi-rate datasets

**At 60 Hz**: 20 ms = 1.2 samples → **1 sample** 😐  
**At 120 Hz**: 20 ms = 2.4 samples → **2 samples** ✅  
**At 300 Hz**: 20 ms = 6.0 samples → **6 samples** ✅

### Sample-Based
```python
n_samples = 3  # Always 3 samples
```
- ✅ Consistent across different sampling rates
- ✅ Always the same "data quality"
- ✅ Better for multi-rate experiments
- ❌ Different time durations at different rates

**At 60 Hz**: 3 samples = 50.0 ms ✅  
**At 120 Hz**: 3 samples = 25.0 ms ✅  
**At 300 Hz**: 3 samples = 10.0 ms ✅

## 🚀 Quickstart

### 1. Time-Based (simplest method)
```python
from ivt_filter.utils.window_utils import create_time_based_config
from ivt_filter.io.pipeline import IVTPipeline

# Always use 20 ms
vel_cfg, clf_cfg = create_time_based_config(window_ms=20.0)

pipeline = IVTPipeline(vel_cfg, clf_cfg)
df = pipeline.run("data.tsv")
```

### 2. Sample-Based (recommended for multi-rate)
```python
from ivt_filter.utils.window_utils import create_sample_based_config

# Always use 3 samples
vel_cfg, clf_cfg = create_sample_based_config(
    n_samples=3,
    sampling_rate_hz=120.0  # Your rate
)

pipeline = IVTPipeline(vel_cfg, clf_cfg)
df = pipeline.run("data.tsv")
```

### 3. Adaptive (automatic)
```python
from ivt_filter.utils.window_utils import create_adaptive_config
from ivt_filter.io import read_tsv

# Automatically detect and adapt to the rate
df = read_tsv("data.tsv")
vel_cfg, clf_cfg = create_adaptive_config(df, n_samples=3)

pipeline = IVTPipeline(vel_cfg, clf_cfg)
df = pipeline.run("data.tsv")
```

## 📊 Experiment Examples

### Time-Based Window Sweep
```python
from ivt_filter.evaluation.experiment import ExperimentConfig
from ivt_filter.io.observers import ConsoleReporter, ExperimentTracker

# Test different time windows
for window_ms in [10.0, 20.0, 40.0, 60.0]:
    vel_cfg, clf_cfg = create_time_based_config(window_ms=window_ms)
    
    exp_config = ExperimentConfig(
        name=f"time_based_{int(window_ms)}ms",
        velocity_config=vel_cfg,
        classifier_config=clf_cfg,
        tags=["time_based"]
    )
    
    pipeline = IVTPipeline(vel_cfg, clf_cfg)
    pipeline.register_observer(ExperimentTracker())
    df = pipeline.run_with_tracking("data.tsv", exp_config)
```

### Sample-Based Window Sweep
```python
# Test different sample counts
sampling_rate_hz = 120.0  # Your rate

for n_samples in [2, 3, 4, 5, 7]:
    vel_cfg, clf_cfg = create_sample_based_config(
        n_samples=n_samples,
        sampling_rate_hz=sampling_rate_hz
    )
    
    exp_config = ExperimentConfig(
        name=f"sample_based_{n_samples}samples",
        velocity_config=vel_cfg,
        classifier_config=clf_cfg,
        tags=["sample_based"],
        metadata={"n_samples": n_samples}
    )
    
    pipeline = IVTPipeline(vel_cfg, clf_cfg)
    pipeline.register_observer(ExperimentTracker())
    df = pipeline.run_with_tracking("data.tsv", exp_config)
```

## 🔧 Utility Functions

### Conversion
```python
from ivt_filter.utils.window_utils import samples_to_milliseconds, milliseconds_to_samples

# Samples → milliseconds
ms = samples_to_milliseconds(3, 120.0)
print(ms)  # 25.0 ms

# Milliseconds → samples
n = milliseconds_to_samples(20.0, 120.0)
print(n)  # 2 samples
```

### Detect Sampling Rate
```python
from ivt_filter.utils.window_utils import detect_sampling_rate
from ivt_filter.io import read_tsv

df = read_tsv("data.tsv")
rate = detect_sampling_rate(df)
print(f"Detected: {rate} Hz")  # e.g. 120.0 Hz
```

### Display Window Info
```python
from ivt_filter.utils.window_utils import print_window_info

print_window_info(20.0, 120.0)
# Output:
# Window: 20.0 ms
# Sampling rate: 120.0 Hz
# Number of samples: 2 (2.4 actual)
# ⚠️  Warning: Window not aligned with sample rate!
#    Consider using 2 samples exactly: 16.67 ms
```

### Get Recommendations
```python
from ivt_filter.utils.window_utils import recommend_window_size

recommendations = recommend_window_size(120.0)
for n_samples, window_ms in recommendations:
    print(f"{n_samples} samples = {window_ms:.2f} ms")

# Output:
# 2 samples = 16.67 ms
# 3 samples = 25.00 ms
# 4 samples = 33.33 ms
# 5 samples = 41.67 ms
# 6 samples = 50.00 ms
# 7 samples = 58.33 ms
```

## 💡 Recommendations

### For single-rate datasets (e.g. only 120 Hz)
→ **Time-Based** is fine
```python
vel_cfg, clf_cfg = create_time_based_config(window_ms=20.0)
```

### For multi-rate datasets (60 Hz, 120 Hz, 300 Hz, ...)
→ **Sample-Based** is better
```python
vel_cfg, clf_cfg = create_sample_based_config(n_samples=3, sampling_rate_hz=rate)
```

### For reverse engineering / experiments
→ **Sample-Based** with experiment tracking
```python
# Test 2, 3, 4, 5 samples at your rate
# and find out which works best
```

### Minimum Number of Samples
- **Minimum**: 2 samples (otherwise velocity cannot be computed)
- **Recommended**: 3-5 samples (good compromise)
- **Maximum**: ~7 samples (more = too much smoothing)

## 📝 Complete Example

```python
from ivt_filter.io import read_tsv
from ivt_filter.utils.window_utils import (
    detect_sampling_rate,
    create_sample_based_config,
    recommend_window_size
)
from ivt_filter.evaluation.experiment import ExperimentConfig, ExperimentManager
from ivt_filter.io.observers import ConsoleReporter, ExperimentTracker
from ivt_filter.io.pipeline import IVTPipeline

# 1. Load data and detect the rate
df = read_tsv("data.tsv")
sampling_rate_hz = detect_sampling_rate(df)
print(f"Detected: {sampling_rate_hz} Hz")

# 2. Show recommendations
print("\nRecommended window sizes:")
for n, ms in recommend_window_size(sampling_rate_hz):
    print(f"  {n} samples = {ms:.2f} ms")

# 3. Test different configurations
for n_samples in [2, 3, 4, 5]:
    vel_cfg, clf_cfg = create_sample_based_config(
        n_samples=n_samples,
        sampling_rate_hz=sampling_rate_hz
    )
    
    exp_config = ExperimentConfig(
        name=f"test_{n_samples}samples",
        velocity_config=vel_cfg,
        classifier_config=clf_cfg,
        tags=["sample_test"],
        metadata={"n_samples": n_samples}
    )
    
    pipeline = IVTPipeline(vel_cfg, clf_cfg)
    pipeline.register_observer(ConsoleReporter(verbose=False))
    pipeline.register_observer(ExperimentTracker("experiments"))
    
    result_df = pipeline.run_with_tracking("data.tsv", exp_config, evaluate=True)
    print(f"✅ {n_samples} samples done")

# 4. Find the best configuration
manager = ExperimentManager("experiments")
best_name, best_value, best_config = manager.get_best_configuration(
    metric="percentage_agreement",
    tags=["sample_test"]
)

print(f"\n🏆 Best: {best_name} with {best_value:.2f}%")
print(f"   Optimal samples: {best_config.metadata['n_samples']}")
```

## 🎬 Ready-Made Scripts

1. **`example_window_sweep.py`** - Comprehensive time-based sweep
2. **`example_sample_based_window.py`** - Sample-based testing ⭐

```bash
# Sample-based (recommended!)
python example_sample_based_window.py
```
