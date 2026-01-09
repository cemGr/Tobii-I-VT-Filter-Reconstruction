FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

COPY requirements.txt ./
RUN pip install --upgrade pip && pip install -r requirements.txt

# Copy source and install package
COPY . .
RUN pip install .

# Default to the CLI; pass args at runtime
ENTRYPOINT ["python", "-m", "ivt_filter.cli"]
CMD ["--help"]