# Performance Protocol (Tracking & Optimization)

This guide defines **how to measure, track, and optimize performance** for training and inference.

---

## Purpose

Performance tracking helps you:
- Detect regressions (is training getting slower?)
- Optimize bottlenecks (where to focus effort)
- Plan resources (how much compute needed?)
- Compare methods fairly (computational cost matters)

---

## What to Measure

### 1. Training Metrics
- **Training time** (total, per epoch, per step)
- **Throughput** (samples/second, tokens/second)
- **Memory usage** (GPU VRAM, CPU RAM, peak usage)
- **GPU utilization** (% of GPU capacity used)
- **Convergence speed** (steps to reach target loss)

### 2. Inference Metrics
- **Latency** (time per prediction, p50/p95/p99)
- **Throughput** (predictions/second)
- **Memory usage** (model size, activation memory)
- **Batch efficiency** (speedup from batching)

### 3. Cost Metrics
- **Compute cost** (GPU-hours, estimated $ cost)
- **Storage cost** (checkpoint sizes, data sizes)
- **Time to experiment** (iteration speed)

---

## How to Measure

### Basic timing (Python):
```python
import time

start_time = time.time()
# ... training code ...
end_time = time.time()

elapsed = end_time - start_time
print(f"Training took {elapsed:.2f} seconds")
```

### GPU memory tracking:
```python
import torch

# Before training
torch.cuda.reset_peak_memory_stats()

# ... training code ...

# After training
peak_memory = torch.cuda.max_memory_allocated() / 1024**3  # GB
print(f"Peak GPU memory: {peak_memory:.2f} GB")
```

### Throughput calculation:
```python
num_samples = len(train_dataset)
training_time = 3600  # seconds
throughput = num_samples / training_time
print(f"Throughput: {throughput:.1f} samples/second")
```

### Using built-in profilers:
```python
from torch.profiler import profile, ProfilerActivity

with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
    # ... training code ...
    pass

print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
```

---

## Baseline Protocol (First Measurement)

When implementing a new component:

### 1. Choose a standard workload
- Example: Fine-tune GPT-2 on 10K samples for 3 epochs
- Document all hyperparameters

### 2. Run multiple times (at least 3)
- Discard first run (warmup)
- Record mean and std dev

### 3. Record in experiment artifact

Example `performance.json`:
```json
{
  "training": {
    "total_time_seconds": 1847,
    "time_per_epoch_seconds": 615,
    "throughput_samples_per_second": 13.2,
    "peak_gpu_memory_gb": 4.5
  },
  "hardware": {
    "gpu": "NVIDIA RTX 3060",
    "gpu_memory_gb": 12,
    "cpu": "Intel i7-12700",
    "ram_gb": 32
  },
  "software": {
    "python": "3.10.8",
    "pytorch": "2.0.1",
    "transformers": "4.35.0",
    "cuda": "11.8"
  },
  "measurement_date": "2026-01-05",
  "num_runs": 3,
  "mean_time": 1847,
  "std_time": 23
}
```

### 4. Set a budget
- Example: "This configuration should complete in < 2000 seconds"
- Document acceptable range

---

## Performance Tracking Over Time

### Maintain a performance log:

`docs/performance_log.csv`:
```csv
date,experiment,model,training_time_sec,peak_memory_gb,throughput,git_hash
2026-01-05,baseline,gpt2,1847,4.5,13.2,abc123
2026-01-08,cvar_loss,gpt2,1923,4.6,12.8,def456
2026-01-12,larger_batch,gpt2,1654,6.2,14.9,ghi789
```

### Visualize trends:
```python
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("docs/performance_log.csv")
df["date"] = pd.to_datetime(df["date"])

plt.figure(figsize=(10, 6))
plt.plot(df["date"], df["training_time_sec"], marker='o')
plt.xlabel("Date")
plt.ylabel("Training Time (seconds)")
plt.title("Training Performance Over Time")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("docs/performance_trend.png", dpi=300)
```

---

## Regression Detection

Before merging changes:

### 1. Run benchmark
- Use the same config as baseline
- Same hardware and environment

### 2. Compare to baseline
- **Acceptable**: within ±10% of baseline
- **Warning**: 10-20% slower
- **Regression**: >20% slower (investigate before merging)

### 3. Document if intentional
- Example: "15% slower but adds new feature X"
- Trade-offs should be conscious decisions

### Example check script:
```python
baseline_time = 1847  # from baseline run
current_time = 2100   # from new run

slowdown = (current_time - baseline_time) / baseline_time * 100

if slowdown < 10:
    print("✓ Performance acceptable")
elif slowdown < 20:
    print("⚠ Warning: 10-20% slowdown")
else:
    print("✗ Regression: >20% slowdown")
    print("Investigation required before merging")
```

---

## Common Bottlenecks & Fixes

### Bottleneck: Data Loading
**Symptoms**: GPU utilization < 80%, CPU maxed out

**Fixes**:
- Increase `num_workers` in DataLoader
- Use faster data formats (HDF5, Arrow instead of JSON)
- Pre-tokenize data
- Use data caching

```python
from torch.utils.data import DataLoader

# Before
dataloader = DataLoader(dataset, batch_size=8, num_workers=0)

# After
dataloader = DataLoader(dataset, batch_size=8, num_workers=4, 
                        pin_memory=True, persistent_workers=True)
```

### Bottleneck: Small Batch Size
**Symptoms**: Low GPU utilization, slow training

**Fixes**:
- Increase batch size (if memory allows)
- Use gradient accumulation
- Use mixed precision training (FP16)

```python
# Gradient accumulation
accumulation_steps = 4
for i, batch in enumerate(dataloader):
    loss = model(batch)
    loss = loss / accumulation_steps
    loss.backward()
    
    if (i + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()
```

### Bottleneck: Memory Constraints
**Symptoms**: Out of memory errors, can't increase batch size

**Fixes**:
- Gradient checkpointing
- Mixed precision (FP16/BF16)
- Smaller model or LoRA fine-tuning
- Reduce sequence length

```python
# Mixed precision training
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

for batch in dataloader:
    with autocast():
        loss = model(batch)
    
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
    optimizer.zero_grad()
```

### Bottleneck: Slow Loss Computation
**Symptoms**: High loss computation time

**Fixes**:
- Vectorize operations (avoid loops)
- Use in-place operations where safe
- Pre-compute constant terms
- Profile to find hotspots

---

## Optimization Workflow

When performance is below budget:

### 1. Profile
```bash
# Using PyTorch profiler
python -m torch.utils.bottleneck train.py --config config.json
```

Identify: data loading? forward pass? backward pass? optimizer step?

### 2. Optimize the Bottleneck
- Start with biggest bottleneck
- Make one change at a time
- Measure after each change

### 3. Verify Improvement
- Re-run benchmark
- Verify no accuracy regression
- Document optimization

### 4. Document
- What was changed
- Performance improvement achieved
- Any trade-offs or limitations

---

## Performance Budget Guidelines

### For training (per experiment):
- **Small models** (< 500M params): < 1 hour on consumer GPU
- **Medium models** (500M - 1B params): < 4 hours on consumer GPU
- **Large models** (> 1B params): Accept longer training or use distributed training

### For inference:
- **Batch inference**: > 100 samples/second
- **Real-time inference**: < 100ms per sample (p95)

### Memory:
- **Training**: Fit within single GPU (12GB for RTX 3060)
- **Inference**: Fit in <50% of GPU memory (leave room for batching)

**These are guidelines, not hard limits. Adjust based on your needs.**

---

## Experiment Performance Checklist

```
[ ] Training time measured and recorded
[ ] GPU memory usage tracked
[ ] Throughput computed (samples/sec)
[ ] Compared to baseline (if applicable)
[ ] No obvious bottlenecks (GPU utilization >80%)
[ ] Performance metrics saved to experiment folder
[ ] Any significant slowdowns investigated and explained
```

---

## When to Optimize

**Optimize when**:
- Performance is blocking research (experiments take too long)
- Clear bottleneck identified (GPU util <50%, memory issues)
- Preparing for production deployment
- Regression detected (>20% slowdown vs baseline)

**Don't optimize when**:
- Performance is "good enough" for research needs
- Would require major code refactoring
- Research questions are more important
- No clear bottleneck identified

**"Premature optimization is the root of all evil" - but tracking is always good.**

---

## Performance vs Accuracy Trade-offs

Sometimes optimizations affect accuracy:

### Mixed precision (FP16):
- **Speedup**: 1.5-2x faster
- **Accuracy impact**: Usually minimal (<1% degradation)
- **Recommendation**: Use it

### Smaller batch size:
- **Memory savings**: Significant
- **Accuracy impact**: May need more epochs to converge
- **Recommendation**: Use gradient accumulation to simulate larger batches

### Gradient checkpointing:
- **Memory savings**: ~40%
- **Speed impact**: ~20% slower
- **Recommendation**: Use if memory constrained

**Always measure accuracy after optimization.**

---

## Example: Benchmarking Training Configurations

```python
import time
import torch

configs = [
    {"batch_size": 8, "fp16": False, "name": "baseline"},
    {"batch_size": 16, "fp16": False, "name": "larger_batch"},
    {"batch_size": 8, "fp16": True, "name": "mixed_precision"},
    {"batch_size": 16, "fp16": True, "name": "optimized"},
]

results = []
for config in configs:
    torch.cuda.reset_peak_memory_stats()
    start = time.time()
    
    # Run training with this config
    # ... training code ...
    
    elapsed = time.time() - start
    peak_memory = torch.cuda.max_memory_allocated() / 1024**3
    
    results.append({
        "name": config["name"],
        "time": elapsed,
        "memory_gb": peak_memory
    })

# Print comparison
for r in results:
    print(f"{r['name']}: {r['time']:.0f}s, {r['memory_gb']:.1f}GB")
```

---

## Long-Term Performance Management

### Quarterly review:
- Review performance log
- Identify trends (getting slower?)
- Plan optimization work if needed

### Documentation:
- Keep performance budget documented
- Update as hardware/methods change
- Document all major optimizations

### Communication:
- Report performance in research notes
- Include compute cost in papers
- Help others reproduce efficiently

---

## Remember

- **Track everything**: Even if not optimizing now, data is valuable later
- **Compare fairly**: Same hardware, same config, same data
- **Document trade-offs**: Performance vs accuracy, time vs memory
- **Don't over-optimize**: Research progress > perfect efficiency
- **Measure, don't guess**: Profile before optimizing

Performance tracking is about making informed decisions, not achieving perfect efficiency.

