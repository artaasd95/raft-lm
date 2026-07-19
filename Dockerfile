# RAFT-LM training-only container (CPU smoke by default).
FROM python:3.11-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONPATH=/app

WORKDIR /app

FROM base AS builder
RUN apt-get update && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/*
COPY pyproject.toml README.md ./
COPY src ./src
COPY scripts ./scripts
COPY configs ./configs
RUN pip install --upgrade pip \
    && pip install torch --index-url https://download.pytorch.org/whl/cpu \
    && pip install -e ".[hf,qlora,dev]"

FROM base AS runtime
RUN useradd --create-home --uid 10001 raft
COPY --from=builder /usr/local /usr/local
COPY --chown=raft:raft . .
USER raft
WORKDIR /app

VOLUME ["/app/data", "/app/experiments", "/app/checkpoints"]

CMD ["python", "scripts/train.py", "--config", "configs/methods/grpo.yaml"]
