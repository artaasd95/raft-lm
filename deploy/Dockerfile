# Local benchmark and demo only — NOT for production SaaS deployment.
FROM python:3.11-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONPATH=/app

WORKDIR /app

FROM base AS builder
RUN apt-get update && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/*
COPY pyproject.toml README.md requirements.txt requirements-benchmark.txt ./
COPY src ./src
RUN pip install --upgrade pip \
    && pip install --no-cache-dir -r requirements-benchmark.txt

FROM base AS runtime
RUN useradd --create-home --uid 10001 raft
COPY --from=builder /usr/local /usr/local
COPY --chown=raft:raft pyproject.toml README.md requirements.txt requirements-benchmark.txt ./
COPY --chown=raft:raft src ./src
COPY --chown=raft:raft configs ./configs
COPY --chown=raft:raft data/benchmark_corpus ./data/benchmark_corpus
COPY --chown=raft:raft docs/benchmarks/results/sample-comparison ./docs/benchmarks/results/sample-comparison
USER raft

ENV BENCHMARK_RESULTS_DIR=/app/docs/benchmarks/results

CMD ["python", "-m", "src.evals.benchmark_runner"]
