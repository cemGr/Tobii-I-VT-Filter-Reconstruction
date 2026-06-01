FROM python:3.10-slim@sha256:70f65c721aaddfb22b20ed6ec12606c59d9592493c5fcb6639f3d0e8ba3fbc10 AS builder

WORKDIR /build

COPY requirements/build.txt ./requirements/build.txt
RUN pip install --no-cache-dir -r requirements/build.txt

COPY pyproject.toml setup.py README.md ./
COPY ivt_filter ./ivt_filter
RUN pip wheel --no-cache-dir --no-deps --no-build-isolation --wheel-dir /wheels .

FROM python:3.10-slim@sha256:70f65c721aaddfb22b20ed6ec12606c59d9592493c5fcb6639f3d0e8ba3fbc10

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

COPY requirements/runtime.txt ./requirements/runtime.txt
RUN pip install --no-cache-dir -r requirements/runtime.txt

COPY --from=builder /wheels/*.whl /tmp/wheels/
RUN pip install --no-cache-dir --no-deps /tmp/wheels/*.whl && rm -rf /tmp/wheels

# Default to the CLI; pass args at runtime
ENTRYPOINT ["python", "-m", "ivt_filter.cli"]
CMD ["--help"]
