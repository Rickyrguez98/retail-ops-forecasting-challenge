# ── Stage 1: base ──────────────────────────────────────────────────────────────
FROM python:3.11-slim AS base

# System deps for pandas / matplotlib / pyarrow
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# ── Stage 2: deps ──────────────────────────────────────────────────────────────
FROM base AS deps

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ── Stage 3: app ───────────────────────────────────────────────────────────────
FROM deps AS app

# Copy source, configs, data, scripts, tests (not mlruns / models / interim)
COPY src/           src/
COPY scripts/       scripts/
COPY configs/       configs/
COPY data/raw/      data/raw/
COPY tests/         tests/
COPY Makefile       Makefile
COPY pyproject.toml pyproject.toml

# Create writable artifact dirs
RUN mkdir -p data/interim data/processed models reports/figures mlruns

# Install the package in editable mode
RUN pip install --no-cache-dir -e . --no-deps

# Expose MLflow UI port (optional; only used if you run mlflow ui inside container)
EXPOSE 5000

# Default: run the full pipeline
CMD ["make", "all"]
