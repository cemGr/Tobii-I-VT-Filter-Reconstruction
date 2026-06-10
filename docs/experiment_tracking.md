# Experiment Tracking & Observer Pattern

## Overview

This system implements **automatic experiment tracking** for systematic reverse engineering of the I-VT filter parameters. It is based on two design patterns:

1. **Observer Pattern**: Automatic notification on pipeline events
2. **Information Expert (GRASP)**: `ExperimentManager` manages experiment data

## Main Components

### 1. ExperimentConfig
Defines a complete experiment configuration:

```python
from ivt_filter.evaluation.experiment import ExperimentConfig
from ivt_filter.config import OlsenVelocityConfig, IVTClassifierConfig

config = ExperimentConfig(
    name="olsen2d_median_20ms",
    description="Testing Olsen 2D with median smoothing",
    velocity_config=OlsenVelocityConfig(
        window_length_ms=20.0,
        velocity_method="olsen2d",
        smoothing_mode="median"
    ),
    classifier_config=IVTClassifierConfig(
        velocity_threshold_deg_per_sec=30.0
    ),
    tags=["baseline", "olsen2d", "median"],
    metadata={"researcher": "Cem", "dataset": "120Hz"}
)
```

### 2. ExperimentManager
Manages stored experiments (Information Expert pattern):

```python
from ivt_filter.evaluation.experiment import ExperimentManager

manager = ExperimentManager("experiments/")

# Save experiment
manager.save_experiment(config, results_df, metrics)

# Load experiment
config, df, metrics = manager.load_experiment("olsen2d_median_20ms")

# Compare experiments
comparison = manager.compare_experiments([
    "exp1", "exp2", "exp3"
])
print(comparison)

# Find the best configuration
best_name, best_value, best_config = manager.get_best_configuration(
    metric="fixation_recall",
    maximize=True
)
```

### 3. Observer Pattern
Automatic notifications on pipeline events:

#### Available observers:

**ConsoleReporter** - Console output:
```python
from ivt_filter.io.observers import ConsoleReporter

reporter = ConsoleReporter(verbose=True)
pipeline.register_observer(reporter)
```

**MetricsLogger** - CSV logging:
```python
from ivt_filter.io.observers import MetricsLogger

logger = MetricsLogger("experiments/metrics_log.csv")
pipeline.register_observer(logger)
```

**ExperimentTracker** - Full tracking:
```python
from ivt_filter.io.observers import ExperimentTracker

tracker = ExperimentTracker("experiments/")
pipeline.register_observer(tracker)
```

**ResultsPlotter** - Automatic plots:
```python
from ivt_filter.io.observers import ResultsPlotter

plotter = ResultsPlotter(
    output_dir="plots/",
    plot_types=["velocity", "classification"],
    auto_show=False
)
pipeline.register_observer(plotter)
```

## Usage

### Simple Experiment

```python
from ivt_filter.io.pipeline import IVTPipeline
from ivt_filter.evaluation.experiment import ExperimentConfig
from ivt_filter.io.observers import ConsoleReporter, MetricsLogger, ExperimentTracker

# Create pipeline with observers
pipeline = IVTPipeline(velocity_config, classifier_config)
pipeline.register_observer(ConsoleReporter(verbose=True))
pipeline.register_observer(MetricsLogger("logs/metrics.csv"))
pipeline.register_observer(ExperimentTracker("experiments/"))

# Experiment configuration
exp_config = ExperimentConfig(
    name="baseline_test",
    description="Baseline configuration test",
    velocity_config=velocity_config,
    classifier_config=classifier_config,
    tags=["baseline"]
)

# Run with tracking
df = pipeline.run_with_tracking(
    input_path="data.tsv",
    config=exp_config,
    evaluate=True
)
```

### Parameter Sweep

```python
from ivt_filter.config import OlsenVelocityConfig, IVTClassifierConfig

# Test different thresholds
thresholds = [20.0, 30.0, 40.0, 50.0]

for threshold in thresholds:
    velocity_config = OlsenVelocityConfig(
        window_length_ms=20.0,
        velocity_method="olsen2d"
    )
    
    classifier_config = IVTClassifierConfig(
        velocity_threshold_deg_per_sec=threshold
    )
    
    exp_config = ExperimentConfig(
        name=f"threshold_{int(threshold)}deg",
        description=f"Testing {threshold} deg/s threshold",
        velocity_config=velocity_config,
        classifier_config=classifier_config,
        tags=["parameter_sweep", "threshold"]
    )
    
    pipeline = IVTPipeline(velocity_config, classifier_config)
    pipeline.register_observer(ConsoleReporter(verbose=False))
    pipeline.register_observer(ExperimentTracker("experiments/"))
    
    df = pipeline.run_with_tracking("data.tsv", exp_config)
```

### Compare Experiments

```python
from ivt_filter.evaluation.experiment import ExperimentManager

manager = ExperimentManager("experiments/")

# List all experiments tagged with "threshold"
threshold_experiments = manager.list_experiments(tags=["threshold"])

# Compare the top 5
exp_names = [e["name"] for e in threshold_experiments[:5]]
comparison = manager.compare_experiments(exp_names)

print("\nComparison Results:")
print(comparison[["experiment", "threshold", "percentage_agreement", 
                 "fixation_recall", "saccade_recall"]])

# Find the best configuration
best_name, best_value, best_config = manager.get_best_configuration(
    metric="percentage_agreement",
    tags=["threshold"]
)

print(f"\n🏆 Best configuration: {best_name}")
print(f"   Agreement: {best_value:.2f}%")
print(f"   Threshold: {best_config.classifier_config.velocity_threshold_deg_per_sec} deg/s")
```

## Experiment Structure on Disk

```
experiments/
├── experiments_index.json          # Index of all experiments
├── baseline_test/
│   ├── config.json                # Experiment configuration
│   ├── results.tsv               # Full results
│   └── metrics.json              # Evaluation metrics
├── threshold_20deg/
│   ├── config.json
│   ├── results.tsv
│   └── metrics.json
└── threshold_30deg/
    ├── config.json
    ├── results.tsv
    └── metrics.json
```

## Benefits for Reverse Engineering

✅ **Systematic testing**: All configurations logged automatically  
✅ **Reproducibility**: Full config + results stored  
✅ **Comparability**: Easy comparison of multiple experiments  
✅ **Best-config search**: Automatically find the best parameters  
✅ **Tag-based organization**: Categorize experiments (baseline, high-frequency, etc.)  
✅ **Automatic plots**: The observer generates plots automatically  
✅ **CSV export**: Metrics for Excel/R/Python analysis  

## Complete Example

See `example_experiment_tracking.py` for a complete example with:
- Individual experiments
- Parameter sweeps
- Experiment comparisons
- Best-configuration search
- Tag-based filtering

```bash
# Run the example
python example_experiment_tracking.py
```

## Design Patterns

### Observer Pattern
**Purpose**: Decouple pipeline execution from tracking

**Advantages**:
- The pipeline knows nothing about logging/plotting
- New observers are easy to add
- Multiple observers can run simultaneously

**Implementation**:
```python
class PipelineObserver(ABC):
    @abstractmethod
    def on_pipeline_start(self, config): pass
    
    @abstractmethod
    def on_pipeline_complete(self, config, df, metrics): pass
    
    @abstractmethod
    def on_pipeline_error(self, config, error): pass
```

### Information Expert (GRASP)
**Purpose**: ExperimentManager knows the experiment data best

**Responsibilities**:
- Saving/loading experiments
- Comparing experiments
- Finding the best configuration
- Index management

## Extensibility

### Creating Your Own Observer

```python
from ivt_filter.io.observers import PipelineObserver

class SlackNotifier(PipelineObserver):
    """Sends notifications to Slack."""
    
    def __init__(self, webhook_url):
        self.webhook_url = webhook_url
    
    def on_pipeline_start(self, config):
        # Send Slack message
        pass
    
    def on_pipeline_complete(self, config, df, metrics):
        # Send success message
        pass
    
    def on_pipeline_error(self, config, error):
        # Send error message
        pass

# Usage
pipeline.register_observer(SlackNotifier("https://hooks.slack.com/..."))
```

### Tracking Additional Metrics

```python
# Store arbitrary data in ExperimentConfig.metadata
exp_config = ExperimentConfig(
    name="custom_exp",
    ...,
    metadata={
        "researcher": "Cem",
        "dataset": "120Hz_highfreq",
        "hardware": "Tobii Pro Spectrum",
        "custom_metric_xyz": 123.45
    }
)
```

## Troubleshooting

**Problem**: Observer throws an error  
**Solution**: Observer errors are caught and the pipeline continues

**Problem**: Experiments not found  
**Solution**: Check or recreate `experiments_index.json`

**Problem**: Metrics missing  
**Solution**: Set `evaluate=True` in `run_with_tracking()`
