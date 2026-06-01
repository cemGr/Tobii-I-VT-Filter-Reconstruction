# Reproducible dependency and Docker builds

Package metadata has one authoritative source: `pyproject.toml`. It declares the
supported Python versions, the minimal runtime dependencies, and optional extras.
The `plot` extra installs Matplotlib only when figures are needed, and the
`parallel` extra installs Joblib only when optional parallel processing is
needed. Neither optional dependency is installed in the minimal runtime image.

The dependency files under `requirements/` separate build, runtime, and development
concerns. The resolved runtime and development lock files are generated for Python
3.10, matching the Docker base image:

- `requirements/build.txt` pins the build tools installed only in the Docker
  builder stage. Keep these pins synchronized with `[build-system].requires` in
  `pyproject.toml`.
- `requirements/runtime.txt` locks only the default runtime dependency graph.
- `requirements/development.txt` locks the default dependency graph plus the
  `development` extra, including pytest for local test runs.

## Refreshing dependency locks

Install [uv](https://docs.astral.sh/uv/) and regenerate both lock files from the
repository root whenever dependency metadata changes or dependencies should be
updated:

```bash
uv pip compile pyproject.toml \
  --output-file requirements/runtime.txt \
  --python-version 3.10
uv pip compile pyproject.toml \
  --extra development \
  --output-file requirements/development.txt \
  --python-version 3.10
```

Review the resulting diff and run the test suite after every refresh. For local
development, install the locked development dependency set before installing the
editable package without a second dependency-resolution pass:

```bash
python -m pip install -r requirements/development.txt
python -m pip install --no-deps -e .
pytest
```

Optional features should be installed explicitly when needed. For example, use
`python -m pip install -e '.[plot]'` for plotting or
`python -m pip install -e '.[parallel]'` for parallel processing. These commands
resolve the requested optional dependency at install time; add a dedicated lock
file if a reproducibly pinned optional environment is required.

## Building the runtime image

The Dockerfile pins the Python base image by digest in both stages. The builder
stage installs the pinned tools from `requirements/build.txt` and creates a wheel
without build isolation or dependency resolution. The final stage installs
`requirements/runtime.txt`, then installs that wheel with dependency resolution
disabled. It intentionally does not upgrade pip during the build and does not
install pytest, Matplotlib, Joblib, setuptools, or wheel into the final image.

Use this exact build command from the repository root:

```bash
docker build --pull=false -t ivt-filter:latest .
```

When intentionally updating the base image, resolve the current digest for the
chosen Python tag, update the digest in `Dockerfile`, rebuild the image, and run
the test suite. Keeping `--pull=false` in the documented build command makes it
clear that routine builds consume the reviewed digest rather than refreshing a
mutable tag implicitly.
