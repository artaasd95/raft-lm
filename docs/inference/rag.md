# RAG inference

Use RAG for **inference and usage** after training. Standard vs RAFT-LM pipelines live in `src/rag/`.

```bash
python scripts/run_benchmark.py   # benchmark harness
python scripts/infer.py --query "..." --rag-pipeline standard
```

See [benchmarks/BENCHMARK.md](../benchmarks/BENCHMARK.md) for the frozen contract.
