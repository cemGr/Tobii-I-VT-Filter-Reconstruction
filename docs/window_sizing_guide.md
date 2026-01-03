# Window Size Configuration - Quick Guide

## 🎯 Was ist der Unterschied?

### Time-Based (Zeit-basiert)
```python
window_ms = 20.0  # Immer 20 ms
```
- ✅ Einfach zu verstehen
- ✅ Vergleichbar mit Literatur
- ❌ Unterschiedliche Anzahl Samples bei verschiedenen Sampling-Rates
- ❌ Nicht optimal für Multi-Rate-Datasets

**Bei 60 Hz**: 20 ms = 1.2 Samples → **1 Sample** 😐  
**Bei 120 Hz**: 20 ms = 2.4 Samples → **2 Samples** ✅  
**Bei 300 Hz**: 20 ms = 6.0 Samples → **6 Samples** ✅

### Sample-Based (Sample-basiert)
```python
n_samples = 3  # Immer 3 Samples
```
- ✅ Konsistent über verschiedene Sampling-Rates
- ✅ Immer gleiche "Datenqualität"
- ✅ Besser für Multi-Rate-Experimente
- ❌ Verschiedene Zeitdauer bei verschiedenen Rates

**Bei 60 Hz**: 3 Samples = 50.0 ms ✅  
**Bei 120 Hz**: 3 Samples = 25.0 ms ✅  
**Bei 300 Hz**: 3 Samples = 10.0 ms ✅

## 🚀 Schnellstart

### 1. Time-Based (einfachste Methode)
```python
from ivt_filter.window_utils import create_time_based_config
from ivt_filter.pipeline import IVTPipeline

# Immer 20 ms verwenden
vel_cfg, clf_cfg = create_time_based_config(window_ms=20.0)

pipeline = IVTPipeline(vel_cfg, clf_cfg)
df = pipeline.run("data.tsv")
```

### 2. Sample-Based (empfohlen für Multi-Rate)
```python
from ivt_filter.window_utils import create_sample_based_config

# Immer 3 Samples verwenden
vel_cfg, clf_cfg = create_sample_based_config(
    n_samples=3,
    sampling_rate_hz=120.0  # Deine Rate
)

pipeline = IVTPipeline(vel_cfg, clf_cfg)
df = pipeline.run("data.tsv")
```

### 3. Adaptive (automatisch)
```python
from ivt_filter.window_utils import create_adaptive_config
from ivt_filter.io import read_tsv

# Automatisch Rate erkennen und anpassen
df = read_tsv("data.tsv")
vel_cfg, clf_cfg = create_adaptive_config(df, n_samples=3)

pipeline = IVTPipeline(vel_cfg, clf_cfg)
df = pipeline.run("data.tsv")
```

## 📊 Experiment-Beispiele

### Time-Based Window Sweep
```python
from ivt_filter.experiment import ExperimentConfig
from ivt_filter.observers import ConsoleReporter, ExperimentTracker

# Verschiedene Zeit-Fenster testen
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
# Verschiedene Sample-Anzahlen testen
sampling_rate_hz = 120.0  # Deine Rate

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

## 🔧 Utility-Funktionen

### Umrechnung
```python
from ivt_filter.window_utils import samples_to_milliseconds, milliseconds_to_samples

# Samples → Millisekunden
ms = samples_to_milliseconds(3, 120.0)
print(ms)  # 25.0 ms

# Millisekunden → Samples
n = milliseconds_to_samples(20.0, 120.0)
print(n)  # 2 samples
```

### Sampling-Rate erkennen
```python
from ivt_filter.window_utils import detect_sampling_rate
from ivt_filter.io import read_tsv

df = read_tsv("data.tsv")
rate = detect_sampling_rate(df)
print(f"Detected: {rate} Hz")  # z.B. 120.0 Hz
```

### Window-Info anzeigen
```python
from ivt_filter.window_utils import print_window_info

print_window_info(20.0, 120.0)
# Output:
# Window: 20.0 ms
# Sampling rate: 120.0 Hz
# Number of samples: 2 (2.4 actual)
# ⚠️  Warning: Window not aligned with sample rate!
#    Consider using 2 samples exactly: 16.67 ms
```

### Empfehlungen erhalten
```python
from ivt_filter.window_utils import recommend_window_size

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

## 💡 Empfehlungen

### Für Single-Rate Datasets (z.B. nur 120 Hz)
→ **Time-Based** ist OK
```python
vel_cfg, clf_cfg = create_time_based_config(window_ms=20.0)
```

### Für Multi-Rate Datasets (60 Hz, 120 Hz, 300 Hz, ...)
→ **Sample-Based** ist besser
```python
vel_cfg, clf_cfg = create_sample_based_config(n_samples=3, sampling_rate_hz=rate)
```

### Für Reverse Engineering / Experimente
→ **Sample-Based** mit Experiment-Tracking
```python
# Teste 2, 3, 4, 5 Samples bei deiner Rate
# und finde heraus, was am besten funktioniert
```

### Minimum Sample-Anzahl
- **Minimum**: 2 Samples (sonst keine Geschwindigkeit berechenbar)
- **Empfohlen**: 3-5 Samples (guter Kompromiss)
- **Maximum**: ~7 Samples (mehr = zu viel Glättung)

## 📝 Vollständiges Beispiel

```python
from ivt_filter.io import read_tsv
from ivt_filter.window_utils import (
    detect_sampling_rate,
    create_sample_based_config,
    recommend_window_size
)
from ivt_filter.experiment import ExperimentConfig, ExperimentManager
from ivt_filter.observers import ConsoleReporter, ExperimentTracker
from ivt_filter.pipeline import IVTPipeline

# 1. Daten laden und Rate erkennen
df = read_tsv("data.tsv")
sampling_rate_hz = detect_sampling_rate(df)
print(f"Detected: {sampling_rate_hz} Hz")

# 2. Empfehlungen anzeigen
print("\nRecommended window sizes:")
for n, ms in recommend_window_size(sampling_rate_hz):
    print(f"  {n} samples = {ms:.2f} ms")

# 3. Verschiedene Konfigurationen testen
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

# 4. Beste Konfiguration finden
manager = ExperimentManager("experiments")
best_name, best_value, best_config = manager.get_best_configuration(
    metric="percentage_agreement",
    tags=["sample_test"]
)

print(f"\n🏆 Best: {best_name} with {best_value:.2f}%")
print(f"   Optimal samples: {best_config.metadata['n_samples']}")
```

## 🎬 Fertige Scripts

1. **`quick_window_test.py`** - Schneller Time-Based Test
2. **`example_window_sweep.py`** - Umfangreicher Time-Based Sweep
3. **`example_sample_based_window.py`** - Sample-Based Testing ⭐

```bash
# Time-based
python quick_window_test.py

# Sample-based (empfohlen!)
python example_sample_based_window.py
```
