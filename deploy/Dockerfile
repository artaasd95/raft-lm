# Local benchmark and demo only — NOT for production SaaS deployment.
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt requirements-benchmark.txt ./
RUN pip install --no-cache-dir -r requirements-benchmark.txt

COPY . .

ENV BENCHMARK_RESULTS_DIR=/app/docs/benchmarks/results
ENV PYTHONPATH=/app

CMD ["python", "-m", "src.evals.benchmark_runner"]
