# Transformer Performance Engineering Project

**Measure. Profile. Optimize. Re-measure.**

A controlled, hands-on performance-engineering lab using a configurable decoder-only Transformer to investigate end-to-end training throughput, GPU memory use, attention kernels, optimizer implementations, input placement, and activation checkpointing on NVIDIA GPUs.

The goal is not to train a better model. It is to identify bottlenecks, change one variable at a time, measure the impact, and preserve the evidence.

## Results at a glance

**Primary workload:** NVIDIA L4 (23,034 MiB), sequence length 2,048, BF16 autocast, unless noted. The attention and optimizer results below use unprofiled benchmark runs. The activation-checkpointing sweep is a shorter preliminary experiment and is labelled separately.

| Experiment | Configuration | Median throughput | Peak allocated memory | Result |
|---|---|---:|---:|---|
| Eager attention | BF16, sequence 2,048 | 12,376.5 tokens/s | 6,873.6 MiB | Baseline |
| Forced Flash Attention | BF16, sequence 2,048 | 42,293.9 tokens/s | 3,398.5 MiB | 3.42× throughput; 50.6% less allocated memory |
| Fused AdamW | BF16 + Flash, sequence 2,048 | 49,675 tokens/s | 3,398.5 MiB | 7.9% higher throughput than standard AdamW in a separate controlled comparison |
| Host vs device inputs | BF16 + Flash + fused AdamW | 49,177 vs 49,270 tokens/s | 3,399 MiB | No material throughput difference in this test |
| Full checkpointing | Batch 2, BF16 + Flash + fused AdamW | 43,111.7 tokens/s | 2,458.0 MiB | 27.7% less memory, 13.1% lower throughput |

**Headline:** On this L4 workload, forced Flash Attention at sequence length 2,048 delivered 3.42× the eager-attention throughput and reduced peak allocated memory by 50.6%. Fused AdamW improved throughput in its separate controlled comparison. Activation checkpointing saved memory but did not improve throughput.

These results are observations for this model, GPU, software environment, and run protocol—not universal speedup guarantees.

## Project highlights

- Benchmark eager attention, PyTorch scaled dot-product attention (SDPA), and forced Flash Attention.
- Use BF16 autocast while retaining FP32 model parameters.
- Compare short-sequence and 2,048-token workloads.
- Compare standard and fused AdamW.
- Compare host-generated and device-generated synthetic inputs.
- Test activation checkpointing and validate loss/gradient agreement.
- Use PyTorch Profiler and Nsight Systems for diagnosis; do not use instrumented runs as throughput evidence.
- Preserve per-run JSON metrics, configuration details, and validation results.

## Detailed results

### 1. FP32 attention — short sequence

| Metric | Eager | SDPA |
|---|---:|---:|
| Median step time | 44.15 ms | 39.70 ms |
| Median throughput | 5,755.7 tokens/s | 6,379.6 tokens/s |

SDPA delivered 10.8% higher median throughput in four interleaved pairs.

### 2. BF16 attention — short sequence

| Metric | Eager | Forced Flash Attention |
|---|---:|---:|
| Median step time | 45.35 ms | 40.20 ms |
| Median throughput | 5,586.0 tokens/s | 6,299.8 tokens/s |
| Peak allocated memory | ~1,338 MiB | ~1,337 MiB |

Forced Flash Attention delivered 12.8% higher median throughput. Flash kernels were confirmed in profiler traces.

### 3. BF16 attention — sequence length 2,048

| Metric | Eager | Forced Flash Attention |
|---|---:|---:|
| Median step time | 330.8 ms | 96.7 ms |
| Median throughput | 12,376.5 tokens/s | 42,293.9 tokens/s |
| Peak allocated memory | 6,873.6 MiB | 3,398.5 MiB |
| Peak reserved memory | 7,504 MiB | 3,904 MiB |

Forced Flash Attention achieved:
- **3.42×** the median throughput.
- **70.8%** lower median step time.
- **50.6%** lower peak allocated memory.

In an instrumented profile, summed attention CUDA-event time fell from 284 ms to 17 ms (about 94%); attention's share of forward-pass event time fell from 45.7% to 4.5%. These summed event totals are not individual step latencies.

### 4. Standard vs fused AdamW — sequence length 2,048

BF16 + forced Flash Attention, four alternating pairs.

| Metric | Standard AdamW | Fused AdamW |
|---|---:|---:|
| Median step time | 88.45 ms | 82.15 ms |
| Median throughput | 46,056 tokens/s | 49,675 tokens/s |
| Peak allocated memory | 3,398.5 MiB | 3,398.5 MiB |

Fused AdamW delivered 7.9% higher median throughput without a meaningful change in peak allocated memory. This is a separate experiment from the attention comparison; compare results only within matching configurations.

### 5. Input placement — sequence length 2,048

BF16 + forced Flash Attention + fused AdamW, four interleaved pairs.

| Input mode | Median step time | Median throughput |
|---|---:|---:|
| Device-generated synthetic inputs | 82.9 ms | 49,270 tokens/s |
| Host inputs + non-blocking copy | 83.3 ms | 49,177 tokens/s |

The host-input run had about 0.5% higher step time. Input placement did not materially change throughput in this workload.

### 6. Activation checkpointing — sequence length 2,048

BF16 + forced Flash Attention + fused AdamW. This preliminary sweep used **2 warm-up steps and 5 measured steps per configuration**, rather than the longer protocol used in the attention and optimizer experiments. Correctness was checked separately.

| Checkpoint policy | Batch | Median step time | Throughput | Peak allocated memory |
|---|---:|---:|---:|---:|
| None | 2 | 82.3 ms | 49,631.7 tokens/s | 3,398.5 MiB |
| Every other block | 2 | 88.3 ms | 46,199.4 tokens/s | 2,930.6 MiB |
| Every block | 2 | 94.9 ms | 43,111.7 tokens/s | 2,458.0 MiB |
| Every block | 3 | 139.3 ms | 44,020.3 tokens/s | 3,257.1 MiB |
| Every block | 4 | 188.6 ms | 43,281.6 tokens/s | 4,056.1 MiB |

The correctness check matched baseline and full-checkpointing losses at 10.53256321, with maximum absolute gradient difference of 7.78e-05 across 45 checked gradients.

**Finding:** At batch size 2, partial checkpointing reduced allocated memory by 13.8% with 6.9% lower throughput; full checkpointing reduced memory by 27.7% with 13.1% lower throughput. Increasing full-checkpointing batch size to 3 brought memory close to baseline but throughput remained 11.3% lower. Batch size 4 exceeded baseline allocated memory and did not recover baseline throughput.

This is a negative throughput result for this workload, but a useful demonstration of the memory-versus-recomputation trade-off. See [the Phase 5 report](results/phase5/activation-checkpointing.md) and the raw JSON records in `results/raw/`.

## What the benchmark measures

A training step includes:
1. Synthetic input generation.
2. Forward pass and loss computation.
3. Backward pass.
4. Optimizer update.

Warm-up steps are excluded from timing. CUDA synchronization is used around the timed region. JSON results record hardware/software details, Git commit, configuration, seed, backend, precision, optimizer, timings, throughput, loss, and peak allocated/reserved GPU memory.

Throughput is a training-workload metric calculated from batch size, sequence length, and elapsed measured time. It is **not** autoregressive inference tokens per second. Synthetic inputs are used; the project makes no dataset-quality or convergence claims.

## Architecture and configuration

The model is a configurable pre-normalized decoder-only Transformer with RMSNorm, SwiGLU feed-forward layers, tied embeddings, and causal self-attention.

| Parameter | Baseline |
|---|---|
| Transformer layers | 6 |
| Hidden size | 768 |
| Attention heads | 12 |
| Intermediate size | 2,048 |
| Vocabulary size | 32,000 |
| Batch size | 2 |
| Sequence length | 128 (baseline); 2,048 (validation) |
| Attention backend | Eager (baseline) |
| Precision | FP32 baseline; BF16 autocast in validation |
| Random seed | 42 |

Long-context runs use `configs/validation_seq2048.yaml`; the baseline config is kept separate.

### Attention backends

- **Eager:** explicit attention operations.
- **SDPA:** `torch.nn.functional.scaled_dot_product_attention`, allowing PyTorch to select a supported backend.
- **Forced Flash Attention:** requires `SDPBackend.FLASH_ATTENTION`; the run fails if the requested backend is unavailable instead of silently falling back.

### Precision and optimizers

BF16 is applied through autocast during the forward pass; model parameters remain FP32. The benchmark supports standard AdamW (`fused=False`) and fused AdamW (`fused=True`).

## Profiling findings

PyTorch Profiler and Nsight Systems were used as separate diagnostic tools.

- The instrumented attention-event share of forward time fell from 45.7% (eager) to 4.5% (Flash).
- The profiler highlighted matrix multiplication (`aten::mm`), tensor copies (`aten::copy_`), optimizer work, kernel launches, and synchronization.
- Nsight Systems confirmed Flash forward and backward kernels in steady state. Library loading and long synchronizations were identified as startup effects.
- Host-to-device copies appeared in short captures but were not dominant in the unprofiled host-versus-device benchmark.
- PyTorch Profiler and Nsight Systems were not run together because of a CUPTI multi-subscriber conflict.
- Nsight Compute was unavailable because GPU performance counters were restricted (`ERR_NVGPUCTRPERM`).

Profiler timings include instrumentation overhead and overlapping scopes. Do not add event totals together or substitute them for unprofiled step times.

## Repository layout

```text
├── model.py
├── train.py
├── benchmark.py
├── profile_step.py
├── data.py
├── configs/
│   ├── baseline.yaml
│   └── validation_seq2048.yaml
├── tests/
└── results/
    ├── raw/          # benchmark_*.json
    ├── validation/
    ├── phase5/
    │   └── activation-checkpointing.md
    └── traces/       # local traces; not committed
```

Large profiler traces and archives are kept out of Git.

## Environment

| Component | Recorded environment |
|---|---|
| GPU | NVIDIA L4 |
| GPU memory | 23,034 MiB |
| NVIDIA driver | 580.126.20 |
| CUDA version reported by NVIDIA-SMI | 13.0 |
| PyTorch | 2.14.1+cu130 |
| Python | 3.12.3 |
| Operating system | Ubuntu-based container |

## Reproducing the experiments

### 1. Set up the environment

```bash
git clone https://github.com/devam1402/Profiling-end-to-end.git
cd Profiling-end-to-end
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
# Install project dependencies and a PyTorch CUDA build compatible with your environment.
```

Verify CUDA visibility:

```bash
python - <<'PY'
import torch
print("PyTorch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
PY
```

### 2. Baseline: eager attention, FP32

```bash
python benchmark.py \
  --config configs/baseline.yaml \
  --device cuda \
  --attention-backend eager \
  --precision fp32 \
  --warmup 5 \
  --steps 30
```

### 3. SDPA with forced Flash Attention

```bash
python benchmark.py \
  --config configs/baseline.yaml \
  --device cuda \
  --attention-backend sdpa \
  --precision bf16 \
  --sdpa-backend flash \
  --warmup 5 \
  --steps 30
```

### 4. Sequence length 2,048 with fused AdamW

```bash
python benchmark.py \
  --config configs/validation_seq2048.yaml \
  --device cuda \
  --attention-backend sdpa \
  --precision bf16 \
  --sdpa-backend flash \
  --optimizer fused_adamw \
  --warmup 5 \
  --steps 30
```

For eager attention at the same sequence length, use `--attention-backend eager` and remove `--sdpa-backend flash`.

### 5. Profile a training run (diagnostic only)

```bash
python profile_step.py \
  --config configs/validation_seq2048.yaml \
  --device cuda \
  --attention-backend sdpa \
  --precision bf16 \
  --sdpa-backend flash \
  --steps 2
```

**Do not use profiler-enabled runs for throughput comparisons.** Alternate configurations and repeat unprofiled runs; do not treat a single measurement as conclusive.

## Methodology

1. Fix the workload: model, batch, sequence length, precision, and seed.
2. Change one variable at a time.
3. Warm up, then measure with CUDA synchronization.
4. Repeat runs and report medians.
5. Profile only to explain a benchmark result.
6. Preserve raw JSON and the associated Git commit.

## Limitations

- Synthetic inputs only; no model-quality or convergence claims.
- Results cover one GPU type and one compact Transformer configuration.
- No multi-GPU scaling results are reported yet.
- Profiler timings include overhead and overlapping scopes.
- BF16 autocast can produce small numerical differences versus FP32.
- Activation-checkpointing results use a shorter five-step measurement window.
- All performance numbers are workload-specific observations, not universal guarantees.

## Future work

- Repeat the activation-checkpointing sweep with longer measurement windows.
- Investigate multi-GPU DDP, NCCL communication, and overlap using Nsight Systems.
- Explore `torch.compile` and Nsight Compute if GPU performance counters are available.

Each extension follows the same rule: one variable, unprofiled medians, and preserved evidence.

## Author

**Devam Thacker**

Master of Financial Technology candidate interested in AI infrastructure, GPU performance engineering, and ML systems.

GitHub: [@devam1402](https://github.com/devam1402)

---

*Controlled Transformer training workload. Measured kernel, memory, and data-movement effects on an NVIDIA L4. Evidence over claims.*
