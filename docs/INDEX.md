# Documentation Index

Welcome to the I-VT Filter Reconstruction documentation!

## Quick Links

- **[README.md](../README.md)** - Main project documentation with installation, usage, and examples
- **[QUICK_REFERENCE.md](QUICK_REFERENCE.md)** - Quick reference guide for common commands and configurations
- **[TECHNICAL_SPECIFICATION.md](TECHNICAL_SPECIFICATION.md)** - Detailed technical specification of velocity calculation methods

## Documentation Structure

### 1. Getting Started

Start here if you're new to the project:
1. Read the [README.md](../README.md) for project overview and installation
2. Check [QUICK_REFERENCE.md](QUICK_REFERENCE.md) for common commands
3. Run the example: `python examples/velocity_comparison.py`

### 2. User Guide

**Basic Usage:**
- [README.md - Usage Section](../README.md#usage): Command-line interface basics
- [README.md - Command-Line Options](../README.md#command-line-options): Complete CLI reference
- [QUICK_REFERENCE.md - Common Configurations](QUICK_REFERENCE.md#common-configurations): Ready-to-use configurations

**Features:**
- Velocity calculation methods (Olsen 2D vs Ray 3D)
- Coordinate rounding strategies
- Window selection methods
- Spatial smoothing options
- Gap filling
- I-VT classification

### 3. Advanced Topics

**Technical Details:**
- [TECHNICAL_SPECIFICATION.md](TECHNICAL_SPECIFICATION.md): Mathematical formulations and algorithms
- [README.md - Architecture](../README.md#architecture): System design and modules
- [README.md - Performance Comparison](../README.md#performance-comparison): Benchmarks and optimization

**Customization:**
- [README.md - Contributing](../README.md#contributing): How to add new methods
- [examples/velocity_comparison.py](../examples/velocity_comparison.py): Example code for custom analysis

### 4. Troubleshooting

- [README.md - Troubleshooting](../README.md#troubleshooting): Common issues and solutions
- [QUICK_REFERENCE.md - Troubleshooting](QUICK_REFERENCE.md#troubleshooting): Quick fixes

### 5. API Reference

**Core Modules:**

- **`ivt_filter.velocity_calculation`**
  - `VelocityCalculationStrategy` (abstract base)
  - `Olsen2DApproximation` (2D method)
  - `Ray3DAngle` (3D method)
  
- **`ivt_filter.coordinate_rounding`**
  - `CoordinateRoundingStrategy` (abstract base)
  - `NoRounding`, `RoundToNearest`, `RoundHalfUp`, `FloorRounding`, `CeilRounding`
  
- **`ivt_filter.smoothing_strategy`**
  - `SmoothingStrategy` (abstract base)
  - `NoSmoothing`, `MedianSmoothing`, `MovingAverageSmoothing`
  
- **`ivt_filter.window_rounding`**
  - `WindowRoundingStrategy` (abstract base)
  - `StandardWindowRounding`, `SymmetricRoundWindowStrategy`
  
- **`ivt_filter.config`**
  - `OlsenVelocityConfig` (velocity calculation settings)
  - `IVTClassifierConfig` (classification settings)
  - `SaccadeMergeConfig`, `FixationPostConfig` (post-processing settings)

## Quick Start Examples

### Example 1: Basic Velocity Calculation
```bash
python -m ivt_filter.cli \
  --input data/processed/ivt_input.tsv \
  --output results/output.tsv
```

### Example 2: High Precision Analysis
```bash
python -m ivt_filter.cli \
  --input data.tsv \
  --velocity-method ray3d \
  --coordinate-rounding none \
  --smoothing median
```

### Example 3: Tobii-Compatible Configuration
```bash
python -m ivt_filter.cli \
  --input data.tsv \
  --velocity-method olsen2d \
  --coordinate-rounding nearest \
  --sampling-rate-method first_100 \
  --dt-calculation-method mean \
  --symmetric-round-window
```

### Example 4: Compare Methods Programmatically
```python
from ivt_filter.config import OlsenVelocityConfig
from ivt_filter.velocity import compute_olsen_velocity
from ivt_filter.io import read_tsv

# Load data
df = read_tsv("data.tsv")

# Method 1: Olsen 2D
cfg1 = OlsenVelocityConfig(velocity_method="olsen2d")
df1 = compute_olsen_velocity(df.copy(), cfg1)
v1 = df1["velocity_deg_per_sec"].mean()

# Method 2: Ray 3D
cfg2 = OlsenVelocityConfig(velocity_method="ray3d")
df2 = compute_olsen_velocity(df.copy(), cfg2)
v2 = df2["velocity_deg_per_sec"].mean()

print(f"Olsen 2D: {v1:.2f} deg/s")
print(f"Ray 3D: {v2:.2f} deg/s")
print(f"Difference: {v1-v2:.2f} deg/s")
```

## Documentation by Topic

### Velocity Calculation
- [README - Velocity Calculation Methods](../README.md#velocity-calculation-methods)
- [TECHNICAL_SPECIFICATION - Method 1: Olsen 2D](TECHNICAL_SPECIFICATION.md#method-1-olsen-2d-approximation)
- [TECHNICAL_SPECIFICATION - Method 2: Ray 3D](TECHNICAL_SPECIFICATION.md#method-2-ray-3d-angle)
- [QUICK_REFERENCE - Velocity Methods](QUICK_REFERENCE.md#velocity-calculation-methods)

### Coordinate Rounding
- [README - Coordinate Rounding](../README.md#coordinate-rounding)
- [TECHNICAL_SPECIFICATION - Coordinate Rounding Effects](TECHNICAL_SPECIFICATION.md#coordinate-rounding-effects)
- [QUICK_REFERENCE - Coordinate Rounding](QUICK_REFERENCE.md#coordinate-rounding)

### Window Configuration
- [README - Window Configuration](../README.md#window-configuration)
- [QUICK_REFERENCE - Window Configuration](QUICK_REFERENCE.md#window-configuration)

### Performance
- [README - Performance Comparison](../README.md#performance-comparison)
- [TECHNICAL_SPECIFICATION - Comparison Table](TECHNICAL_SPECIFICATION.md#comparison-table)
- [QUICK_REFERENCE - Performance Reference](QUICK_REFERENCE.md#performance-reference)

## File Sizes

| File | Size | Purpose |
|------|------|---------|
| README.md | 18 KB | Main documentation |
| QUICK_REFERENCE.md | 4.2 KB | Quick command reference |
| TECHNICAL_SPECIFICATION.md | 7.1 KB | Technical details |
| examples/velocity_comparison.py | 4.9 KB | Example script |

Total: ~34 KB of documentation

## Version History

- **v2.0 (Current)**: Added Ray 3D method, coordinate rounding, comprehensive documentation
- **v1.0**: Initial release with Olsen 2D method

## Contributing to Documentation

Documentation improvements are welcome! Please:
1. Keep code examples tested and working
2. Update all relevant sections when adding features
3. Follow the existing structure and style
4. Include practical examples

## Support

- GitHub Issues: https://github.com/cemGr/Tobii-I-VT-Filter-Reconstruction/issues
- GitHub: [@cemGr](https://github.com/cemGr)

---

Last Updated: December 22, 2024
