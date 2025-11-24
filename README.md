# Tobii-I-VT-Filter-Reconstruction

Werkzeuge zur Rekonstruktion des I-VT-Filters auf Basis der Beschreibungen im
Exposé und im Tobii-Papier. Die Kernfunktionen (Extraktion, Geschwindigkeits-
berechnung, Klassifikation, Auswertung) sind nun in einem modularen
`ivt`-Package gekapselt und folgen einer klaren SOLID-Aufteilung.

## Projektstruktur

Die wichtigsten Artefakte sind nach Rollen gruppiert, sodass keine Dateien mehr
lose im Root-Verzeichnis liegen:

- `ivt/` – Kernbibliothek (Extraktion, Geschwindigkeitsberechnung,
  Klassifikation, Evaluation).
- `app/` – kleine Beispiel-/Smoke-Tests für Grundfunktionen.
- `tests/` – TTD-Suite für Extraktor, Velocity und CLI-Wrapper.
- `data/raw/` – unveränderte Tobii-TSV-Beispiele (z.B.
  `ivt_frequency120_fixation_export.tsv`).
- `data/processed/` – daraus erzeugte Slim-/Velocity-/Eval-Dateien (z.B.
  `ivt_normal_with_velocity.tsv`).
- `docs/` – Exposé und Tobii-Papier; unter `docs/images/` liegen die
  Auswertungsplots.

## Schneller Einstieg

### 1. Repository klonen
```bash
git clone git@github.com:cemGr/Tobii-I-VT-Filter-Reconstruction.git
cd Tobii-I-VT-Filter-Reconstruction
```

### 2. (Optional) Virtuelle Umgebung
```bash
python -m venv .venv
source .venv/bin/activate  # Linux/macOS
./.venv/Scripts/Activate.ps1  # Windows PowerShell
```

### 3. Abhängigkeiten installieren
```bash
pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
```

### 4. IVT-Pipeline nutzen
Das neue CLI bündelt die Schritte Extraktion → Geschwindigkeit → Klassifikation →
Evaluation. Jeder Schritt ist optional und kann einzeln ausgeführt werden.

```bash
# 4.1 Tobii-Export verschlanken
python -m ivt extract data/raw/ivt_frequency120_fixation_export.tsv data/processed/ivt_input.tsv

# 4.2 Olsen-Geschwindigkeit berechnen (z.B. 20 ms Fenster, Augen gemittelt)
python -m ivt velocity data/processed/ivt_input.tsv data/processed/ivt_with_velocity.tsv --window 20 --eye average

# 4.3 I-VT-Klassifikation anwenden
python -m ivt classify data/processed/ivt_with_velocity.tsv data/processed/ivt_with_classes.tsv --threshold 30

# 4.4 Gegen Ground-Truth auswerten
python -m ivt evaluate data/processed/ivt_with_classes.tsv --gt-col gt_event_type
```

Alle Schritte können auch als Python-API genutzt werden:
```python
from ivt import (
    TobiiTSVExtractor,
    VelocityCalculator,
    IVTClassifier,
    evaluate_ivt_vs_ground_truth,
)

slim_path = "data/processed/ivt_input.tsv"
TobiiTSVExtractor().convert("data/raw/ivt_frequency120_fixation_export.tsv", slim_path)

df_velocity = VelocityCalculator().compute_from_file(slim_path)
classified = IVTClassifier().classify(df_velocity)
stats = evaluate_ivt_vs_ground_truth(classified)
```

## Testen (TTD)
Die Kernfunktionen werden testgetrieben abgesichert, insbesondere die
Geschwindigkeitsberechnung. Neue Tests liegen unter `tests/`.

```bash
pytest              # alle Tests inkl. Velocity-/Extractor-Checks
python -m unittest  # falls du die eingebaute Test-Suite nutzen möchtest
```

## Docker
```bash
docker build -t tobii-ivt:latest .
docker run --rm -it tobii-ivt:latest bash
```

## Release & Publishing
```bash
# Version in setup.py anpassen, committen, dann taggen
git tag vX.Y.Z
git push origin --tags
```
