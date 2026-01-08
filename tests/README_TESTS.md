# Testing Guide für das IVT Filter Projekt

## Übersicht

Das Projekt verwendet **pytest** für umfassende automatisierte Tests. Die Tests sind nach Modulen organisiert und folgen dem **Test-Driven Development (TDD)** Ansatz.

## Test-Struktur

```
tests/
├── conftest.py                      # Shared fixtures und test data
├── test_preprocessing.py            # Gap fill, eye selection, smoothing
├── test_processing.py               # Velocity calculation, classification
├── test_postprocessing.py           # Merging, filtering
├── test_evaluation.py               # Metrics, agreement, kappa
├── test_io_pipeline.py              # I/O und pipeline orchestration
├── test_utils.py                    # Window calculations, sampling rate
├── test_integration_archive.py      # Real archive data tests
├── test_velocity_strategies.py      # Existing velocity tests
├── test_calc.py                     # Legacy tests
└── test_main.py                     # Legacy tests
```

## Tests ausführen

### Alle Tests starten
```bash
pytest
```

### Spezifische Test-Datei
```bash
pytest tests/test_preprocessing.py
```

### Spezifische Test-Klasse
```bash
pytest tests/test_preprocessing.py::TestGapFill
```

### Spezifischer Test
```bash
pytest tests/test_preprocessing.py::TestGapFill::test_gap_fill_no_gaps
```

### Mit detaillierten Ausgaben
```bash
pytest -v
```

### Mit Coverage
```bash
pytest --cov=ivt_filter --cov-report=html
```

### Nur schnelle Tests (ohne archive data)
```bash
pytest -m "not archive"
```

### Nur Unit Tests
```bash
pytest -m unit
```

### Nur Integration Tests
```bash
pytest -m integration
```

## Test-Kategorien

### 1. **Preprocessing Tests** (`test_preprocessing.py`)
Tests für die Daten-Vorverarbeitung:

- ✅ **Gap Fill**: Interpolation von fehlenden Daten
- ✅ **Eye Selection**: Kombination von links/rechts Auge
- ✅ **Noise Reduction**: Smoothing (median, moving average)

**Beispiel ausführen:**
```bash
pytest tests/test_preprocessing.py -v
```

### 2. **Processing Tests** (`test_processing.py`)
Tests für die Kern-IVT-Algorithmen:

- ✅ **Olsen 2D**: 2D Velocity Approximation
- ✅ **Ray 3D**: 3D Ray Angle Calculation
- ✅ **Ray 3D Gaze Dir**: Ray 3D mit Gaze Direction
- ✅ **Velocity Computation**: End-to-End Pipeline

**Beispiel ausführen:**
```bash
pytest tests/test_processing.py::TestOlsen2DApproximation -v
```

### 3. **Postprocessing Tests** (`test_postprocessing.py`)
Tests für die Nachverarbeitung:

- ✅ **Merge Fixations**: Saccade Merging (Zeit + Winkel)
- ✅ **Discard Short Fixations**: Kurzfixationen entfernen
- ✅ **Integration**: Merge → Discard Pipeline

### 4. **Evaluation Tests** (`test_evaluation.py`)
Tests für Metriken und Validierung:

- ✅ **Agreement**: Sample-Level Übereinstimmung
- ✅ **Cohen's Kappa**: Statistische Übereinstimmung
- ✅ **Per-Class Metrics**: Precision, Recall, F1

### 5. **I/O & Pipeline Tests** (`test_io_pipeline.py`)
Tests für Input/Output und Pipeline-Orchestrierung:

- ✅ **TSV I/O**: Datei-Operationen
- ✅ **Pipeline Execution**: End-to-End Pipeline
- ✅ **Integration**: Vollständige Verarbeitung

### 6. **Utility Tests** (`test_utils.py`)
Tests für Hilfsfunktionen:

- ✅ **Window Calculations**: Fenster-Berechnung
- ✅ **Sampling Rate Detection**: Sampling-Rate erkennen
- ✅ **Time Conversions**: Zeit-Konversionen

### 7. **Archive Data Tests** (`test_integration_archive.py`)
Tests mit echten Daten aus dem Test-Archive:

- ✅ **Real Data Loading**: Archive-Daten laden
- ✅ **Pipeline on Real Data**: Pipeline mit echten Daten
- ✅ **Evaluation on GT**: Evaluierung mit Ground Truth
- ✅ **Performance**: Performance-Tests
- ✅ **Robustness**: NaN/Outlier Handling

**Beispiel:**
```bash
pytest tests/test_integration_archive.py -v
```

## Test-Daten (Fixtures)

Die Fixtures sind in `conftest.py` definiert und können in jedem Test verwendet werden:

### Synthetische Test-Daten

```python
def test_example(simple_eye_tracking_data):
    """Einfache 100-Sample Eye-Tracking Daten."""
    df = simple_eye_tracking_data
    # Verwende df in Test
```

### Verfügbare Fixtures

| Fixture | Beschreibung |
|---------|-------------|
| `simple_eye_tracking_data` | 100 Samples, stationär |
| `saccade_fixation_data` | 200 Samples mit Fixationen/Saccaden |
| `data_with_gaps` | Daten mit fehlenden Werten |
| `mixed_eye_validity_data` | Gemischte Auge-Validität |
| `real_data_sample` | Echte Daten aus Archive (falls vorhanden) |
| `archive_data_dir` | Pfad zu Archive-Daten |
| `test_data_dir` | Pfad zu Test-Daten |

## Test-Beispiele

### Test für Gap Filling
```python
def test_gap_fill_no_gaps(simple_eye_tracking_data):
    """Sollte unveränderte Daten zurückgeben."""
    df = simple_eye_tracking_data.copy()
    result = gap_fill_gaze(df, max_gap_ms=75)
    
    assert len(result) == len(df)
    assert result['gaze_left_x_mm'].notna().all()
```

### Test für Velocity Calculation
```python
def test_olsen_2d_perpendicular_movement():
    """Test bei senkrechter Bewegung."""
    calc = Olsen2DApproximation()
    angle = calc.calculate_visual_angle(
        x1_mm=0, y1_mm=0,
        x2_mm=10, y2_mm=0,
        eye_x_mm=0, eye_y_mm=0, eye_z_mm=600
    )
    
    expected = math.degrees(math.atan(10 / 600))
    assert math.isclose(angle, expected, rel_tol=1e-3)
```

### Test für Pipeline
```python
def test_pipeline_basic_execution(simple_eye_tracking_data):
    """Test Pipeline-Ausführung."""
    df = simple_eye_tracking_data.copy()
    
    velocity_cfg = OlsenVelocityConfig(velocity_method='olsen2d')
    classifier_cfg = IVTClassifierConfig(threshold=30)
    pipeline = IVTPipeline(velocity_cfg, classifier_cfg)
    
    result_df = pipeline.run(df)
    
    assert 'velocity_deg_per_sec' in result_df.columns
    assert 'ivt_sample_type' in result_df.columns
```

## Coverage

Coverage-Bericht generieren:

```bash
# HTML Coverage Report
pytest --cov=ivt_filter --cov-report=html tests/

# Terminal Coverage Report
pytest --cov=ivt_filter --cov-report=term tests/
```

Report befindet sich dann in `htmlcov/index.html`.

## Best Practices

### 1. **Fixtures verwenden**
```python
def test_with_fixture(simple_eye_tracking_data):
    """Fixtures sollten verwendet werden, nicht copy()."""
    df = simple_eye_tracking_data.copy()  # ← Wichtig: copy()!
```

### 2. **Assertions spezifisch machen**
```python
# ✅ Gut
assert agreement == 0.8
assert 'velocity_deg_per_sec' in result.columns

# ❌ Schlecht
assert result  # Zu vage
```

### 3. **Test-Namen aussagekräftig**
```python
# ✅ Gut
def test_gap_fill_respects_max_duration()
def test_olsen_2d_perpendicular_movement()

# ❌ Schlecht
def test_gap()
def test_velocity()
```

### 4. **Setup/Teardown nutzen**
```python
def test_with_temp_file():
    """Verwende tempfile für Datei-Tests."""
    with tempfile.TemporaryDirectory() as tmpdir:
        filepath = Path(tmpdir) / "test.tsv"
        # Test-Code
```

## Bekannte Einschränkungen

### Archive Data Tests
- Diese Tests können übersprungen werden, wenn Archive-Daten nicht vorhanden sind
- Verwende `pytest.skip()` statt `pytest.xfail()` für fehlende Daten

### Performance Tests
- Sind optional und können langsam sein
- Mit `-m "not slow"` ausschließen

## Troubleshooting

### Tests schlagen fehl mit ImportError
```bash
# Stelle sicher, dass das Package installiert ist
pip install -e .
```

### Fixtures nicht gefunden
```bash
# Stelle sicher, dass conftest.py im tests/ Verzeichnis ist
ls tests/conftest.py
```

### Fehlende Abhängigkeiten
```bash
# Installiere requirements
pip install -r requirements.txt
```

## Continuous Integration

Tests sollten im CI/CD Pipeline ausgeführt werden:

```yaml
# Beispiel GitHub Actions Workflow
- name: Run tests
  run: pytest --cov=ivt_filter tests/
```

## Weitere Ressourcen

- [Pytest Dokumentation](https://docs.pytest.org/)
- [Fixtures](https://docs.pytest.org/en/stable/how-to/fixtures.html)
- [Markers](https://docs.pytest.org/en/stable/how-to/mark.html)
- [Coverage.py](https://coverage.readthedocs.io/)

## Zusammenfassung

Das Test-System deckt ab:
- **7 Preprocessing/Processing/Postprocessing Module**
- **3 Velocity Calculation Methoden** (Olsen 2D, Ray 3D, Ray 3D Gaze Dir)
- **End-to-End Pipeline** mit echten Daten
- **Metriken & Evaluation**
- **I/O Operationen**
- **Utility Funktionen**
- **Archive Data** (optional)

Total: **60+ Tests** mit vollständiger Abdeckung der Kern-Funktionalität.
