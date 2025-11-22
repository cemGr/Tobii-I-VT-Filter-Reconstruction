# Tobii-I-VT-Filter-Reconstruction
Bachelor Thesis about the reconstruction of the I-VT Filters by Tobii


## Quick Start

### 1. Clone the repository

```bash
git clone git@github.com:cemGr/Tobii-I-VT-Filter-Reconstruction.git
cd <repo>
```

### 2. (Optional) Create & activate a virtual environment

```bash
python -m venv .venv
# Linux/macOS
source .venv/bin/activate
# Windows (PowerShell)
.\.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
pip install -e .    # editable install
```

---

## Testing

### Built-in Unittest

```bash
python -m unittest discover -v
```

### Pytest

```bash
pytest                 # run all tests
pytest -q              # quiet output
pytest --maxfail=1     # stop at first failure
pytest --cov=app       # measure coverage
```

---

## Velocity analysis (IVT preview)

The repository includes a small analysis helper that loads a Tobii Pro Lab TSV export,
computes gaze velocities using the IVT-style angular window method, and shows a
velocity-over-time plot. Use one of the included `*_short.tsv` fixtures or your own
export:

```bash
python analysis.py --input "I-VT-normal Data export_short.tsv" --window 20
```

- `--input` points to the TSV file exported from Tobii Pro Lab.
- `--window` sets the velocity window length in milliseconds (default: `20`).

The command opens a matplotlib window with the computed velocity curve. If you run
this in a headless environment, set `MPLBACKEND=Agg` to suppress the UI while still
executing the computation.

---

## Docker

**Build image**

  ```bash

docker build -t my-python-app\:latest .

````

**Run container**  
  ```bash
docker run --rm my-python-app:latest
````

 **Interactive shell**

  ```bash
docker run --rm -it my-python-app\:latest bash
````

**Expose port (e.g. for web services)**  
  ```bash
docker run --rm -p 8000:8000 my-python-app:latest
````

---

## Release & Publishing

### Create a new release

```bash
# 1) bump version in setup.py, commit, then:
git tag vX.Y.Z
git push origin --tags
```

### Push Docker image to GitHub Container Registry

```bash
docker tag my-python-app:latest ghcr.io/cemgr/my-python-app:latest
docker push ghcr.io/cemgr/my-python-app:latest
```

### Publish to PyPI

```bash
python -m build
python -m twine upload dist/*
```

