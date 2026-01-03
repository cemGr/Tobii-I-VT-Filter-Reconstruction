# Experiment Tracking & Observer Pattern

## Übersicht

Dieses System implementiert **automatisches Experiment-Tracking** für systematisches Reverse Engineering der I-VT Filter Parameter. Es basiert auf zwei Design Patterns:

1. **Observer Pattern**: Automatische Benachrichtigung bei Pipeline-Events
2. **Information Expert (GRASP)**: `ExperimentManager` verwaltet Experiment-Daten

## Hauptkomponenten

### 1. ExperimentConfig
Definiert eine komplette Experiment-Konfiguration:

```python
from ivt_filter.experiment import ExperimentConfig
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
Verwaltet gespeicherte Experimente (Information Expert Pattern):

```python
from ivt_filter.experiment import ExperimentManager

manager = ExperimentManager("experiments/")

# Experiment speichern
manager.save_experiment(config, results_df, metrics)

# Experiment laden
config, df, metrics = manager.load_experiment("olsen2d_median_20ms")

# Experimente vergleichen
comparison = manager.compare_experiments([
    "exp1", "exp2", "exp3"
])
print(comparison)

# Beste Konfiguration finden
best_name, best_value, best_config = manager.get_best_configuration(
    metric="fixation_recall",
    maximize=True
)
```

### 3. Observer Pattern
Automatische Benachrichtigungen bei Pipeline-Events:

#### Verfügbare Observer:

**ConsoleReporter** - Konsolen-Ausgabe:
```python
from ivt_filter.observers import ConsoleReporter

reporter = ConsoleReporter(verbose=True)
pipeline.register_observer(reporter)
```

**MetricsLogger** - CSV-Logging:
```python
from ivt_filter.observers import MetricsLogger

logger = MetricsLogger("experiments/metrics_log.csv")
pipeline.register_observer(logger)
```

**ExperimentTracker** - Vollständiges Tracking:
```python
from ivt_filter.observers import ExperimentTracker

tracker = ExperimentTracker("experiments/")
pipeline.register_observer(tracker)
```

**ResultsPlotter** - Automatische Plots:
```python
from ivt_filter.observers import ResultsPlotter

plotter = ResultsPlotter(
    output_dir="plots/",
    plot_types=["velocity", "classification"],
    auto_show=False
)
pipeline.register_observer(plotter)
```

## Verwendung

### Einfaches Experiment

```python
from ivt_filter.pipeline import IVTPipeline
from ivt_filter.experiment import ExperimentConfig
from ivt_filter.observers import ConsoleReporter, MetricsLogger, ExperimentTracker

# Pipeline mit Observers erstellen
pipeline = IVTPipeline(velocity_config, classifier_config)
pipeline.register_observer(ConsoleReporter(verbose=True))
pipeline.register_observer(MetricsLogger("logs/metrics.csv"))
pipeline.register_observer(ExperimentTracker("experiments/"))

# Experiment-Konfiguration
exp_config = ExperimentConfig(
    name="baseline_test",
    description="Baseline configuration test",
    velocity_config=velocity_config,
    classifier_config=classifier_config,
    tags=["baseline"]
)

# Mit Tracking ausführen
df = pipeline.run_with_tracking(
    input_path="data.tsv",
    config=exp_config,
    evaluate=True
)
```

### Parameter Sweep

```python
from ivt_filter.config import OlsenVelocityConfig, IVTClassifierConfig

# Verschiedene Thresholds testen
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

### Experimente vergleichen

```python
from ivt_filter.experiment import ExperimentManager

manager = ExperimentManager("experiments/")

# Alle Experimente mit "threshold" Tag auflisten
threshold_experiments = manager.list_experiments(tags=["threshold"])

# Top 5 vergleichen
exp_names = [e["name"] for e in threshold_experiments[:5]]
comparison = manager.compare_experiments(exp_names)

print("\nComparison Results:")
print(comparison[["experiment", "threshold", "percentage_agreement", 
                 "fixation_recall", "saccade_recall"]])

# Beste Konfiguration finden
best_name, best_value, best_config = manager.get_best_configuration(
    metric="percentage_agreement",
    tags=["threshold"]
)

print(f"\n🏆 Best configuration: {best_name}")
print(f"   Agreement: {best_value:.2f}%")
print(f"   Threshold: {best_config.classifier_config.velocity_threshold_deg_per_sec} deg/s")
```

## Experiment-Struktur auf der Festplatte

```
experiments/
├── experiments_index.json          # Index aller Experimente
├── baseline_test/
│   ├── config.json                # Experiment-Konfiguration
│   ├── results.tsv               # Vollständige Ergebnisse
│   └── metrics.json              # Evaluation-Metriken
├── threshold_20deg/
│   ├── config.json
│   ├── results.tsv
│   └── metrics.json
└── threshold_30deg/
    ├── config.json
    ├── results.tsv
    └── metrics.json
```

## Vorteile für Reverse Engineering

✅ **Systematisches Testen**: Alle Konfigurationen automatisch geloggt  
✅ **Reproduzierbarkeit**: Vollständige Config + Ergebnisse gespeichert  
✅ **Vergleichbarkeit**: Einfacher Vergleich mehrerer Experimente  
✅ **Best-Config-Suche**: Automatisch beste Parameter finden  
✅ **Tag-basierte Organisation**: Experimente kategorisieren (baseline, high-frequency, etc.)  
✅ **Automatische Plots**: Observer generiert Plots automatisch  
✅ **CSV-Export**: Metriken für Excel/R/Python-Analyse  

## Vollständiges Beispiel

Siehe `example_experiment_tracking.py` für ein vollständiges Beispiel mit:
- Einzelnen Experimenten
- Parameter Sweeps
- Experiment-Vergleichen
- Best-Configuration-Suche
- Tag-basierter Filterung

```bash
# Beispiel ausführen
python example_experiment_tracking.py
```

## Design Patterns

### Observer Pattern
**Zweck**: Entkopplung von Pipeline-Ausführung und Tracking

**Vorteile**:
- Pipeline weiß nichts über Logging/Plotting
- Neue Observer einfach hinzufügbar
- Mehrere Observer gleichzeitig möglich

**Implementierung**:
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
**Zweck**: ExperimentManager kennt die Experiment-Daten am besten

**Verantwortlichkeiten**:
- Speichern/Laden von Experimenten
- Vergleichen von Experimenten
- Finden der besten Konfiguration
- Index-Verwaltung

## Erweiterbarkeit

### Eigene Observer erstellen

```python
from ivt_filter.observers import PipelineObserver

class SlackNotifier(PipelineObserver):
    """Sendet Benachrichtigungen an Slack."""
    
    def __init__(self, webhook_url):
        self.webhook_url = webhook_url
    
    def on_pipeline_start(self, config):
        # Slack-Nachricht senden
        pass
    
    def on_pipeline_complete(self, config, df, metrics):
        # Erfolgs-Nachricht senden
        pass
    
    def on_pipeline_error(self, config, error):
        # Fehler-Nachricht senden
        pass

# Verwenden
pipeline.register_observer(SlackNotifier("https://hooks.slack.com/..."))
```

### Weitere Metriken tracken

```python
# In ExperimentConfig.metadata beliebige Daten speichern
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

**Problem**: Observer wirft Fehler  
**Lösung**: Observer-Fehler werden abgefangen, Pipeline läuft weiter

**Problem**: Experimente nicht gefunden  
**Lösung**: `experiments_index.json` prüfen oder neu erstellen

**Problem**: Metriken fehlen  
**Lösung**: `evaluate=True` in `run_with_tracking()` setzen
