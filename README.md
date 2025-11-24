# Tobii-I-VT-Filter-Reconstruction

Werkzeuge zur Rekonstruktion des I-VT-Filters auf Basis der Beschreibungen im
Exposé und im Tobii-Papier. Die Kernfunktionen (Extraktion, Geschwindigkeits-
berechnung, Klassifikation, Auswertung) sind nun in einem modularen
`ivt`-Package gekapselt und folgen einer klaren SOLID-Aufteilung.

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
python -m ivt extract I-VT-frequency120Fixation\ export.tsv ivt_input.tsv

# 4.2 Olsen-Geschwindigkeit berechnen (z.B. 20 ms Fenster, Augen gemittelt)
python -m ivt velocity ivt_input.tsv ivt_with_velocity.tsv --window 20 --eye average

# 4.3 I-VT-Klassifikation anwenden
python -m ivt classify ivt_with_velocity.tsv ivt_with_classes.tsv --threshold 30

# 4.4 Gegen Ground-Truth auswerten
python -m ivt evaluate ivt_with_classes.tsv --gt-col gt_event_type
```

Alle Schritte können auch als Python-API genutzt werden:
```python
from ivt import (
    TobiiTSVExtractor,
    VelocityCalculator,
    IVTClassifier,
    evaluate_ivt_vs_ground_truth,
)

slim_path = "ivt_input.tsv"
TobiiTSVExtractor().convert("raw_tobii.tsv", slim_path)

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
