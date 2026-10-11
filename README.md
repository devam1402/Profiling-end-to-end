# Transformer Performance Engineering Lab

**Measure. Profile. Optimize. Re-measure.**

A hands-on performance-engineering project for understanding and optimizing Transformer training workloads on NVIDIA GPUs. This repository implements a configurable decoder-only Transformer and a repeatable workflow for benchmarking attention implementations, profiling CUDA execution, measuring GPU memory, and evaluating optimizer implementations.

The goal is not just to make a Transformer run faster. It is to identify bottlenecks, change one variable at a time, measure the impact, and preserve the evidence.

## Project Highlights

- **Real GPU benchmarking:** Experiments performed on an NVIDIA L4 with 23 GB of GPU memory.
- **Attention optimization:** Compared eager attention with PyTorch scaled dot-product attention (SDPA) and forced Flash Attention.
- **Mixed precision:** Added BF16 autocast while retaining FP32 model parameters.
- **Long-context experiments:** Compared attention implementations at sequence lengths of 128 and 2,048.
- **CUDA profiling:** Used PyTorch Profiler and CUDA events to investigate attention, matrix multiplication, memory-copy operations, and optimizer execution.
- **Optimizer benchmarking:** Added an optional fused AdamW implementation and evaluated it against standard AdamW.
- **Reproducible results:** Saved per-run JSON metrics and validation logs.

## Results at a Glance

<sub>Hardware: NVIDIA L4 · 5 warm-up steps · 30 measured steps per benchmark run · four runs per configuration.</sub>

### 1. FP32 attention comparison — sequence length 128

| Metric | Eager attention | SDPA |
|---|---:|---:|
| Median step time | 44.15 ms | 39.70 ms |
| Median throughput | 5,755.7 tokens/s | 6,379.6 tokens/s |

SDPA delivered approximately **10.8% higher median throughput** in this repeated comparison.

### 2. BF16 attention comparison — sequence length 128

| Metric | Eager attention | Flash Attention |
|---|---:|---:|
| Median step time | 45.35 ms | 40.20 ms |
| Median throughput | 5,586.0 tokens/s | 6,299.8 tokens/s |
| Peak allocated GPU memory | ~1,338 MiB | ~1,337 MiB |

Flash Attention delivered approximately **12.8% higher median throughput** in this experiment.

### 3. BF16 attention comparison — sequence length 2,048

| Metric | Eager attention | Flash Attention |
|---|---:|---:|
| Median step time | 330.8 ms | 96.7 ms |
| Median throughput | 12,376.5 tokens/s | 42,293.9 tokens/s |
| Peak allocated GPU memory | 6,873.6 MiB | 3,398.5 MiB |
| Peak reserved GPU memory | 7,504 MiB | 3,904 MiB |

At sequence length 2,048, Flash Attention achieved approximately:

- **3.42× the throughput** of eager attention.
- **70.8% lower median step time.**
- **50.6% lower peak allocated GPU memory.**

This experiment demonstrates why attention implementation and memory efficiency become increasingly important as sequence length grows.

### 4. Standard vs. fused AdamW — sequence length 2,048

This separate controlled experiment used BF16, forced Flash Attention, five warm-up steps, and 30 measured steps per run.

| Metric | Standard AdamW | Fused AdamW |
|---|---:|---:|
| Median step time | 88.45 ms | 82.15 ms |
| Median throughput | 46,056 tokens/s | 49,675 tokens/s |
| Peak allocated GPU memory | 3,398.5 MiB | 3,398.5 MiB |
| Peak reserved GPU memory | 3,904 MiB | 3,904 MiB |

Fused AdamW delivered approximately **7.9% higher median throughput** across four alternating pairs, without a meaningful change in peak GPU memory usage.

The optimizer comparison is a separate experiment from the initial attention comparison. Its throughput figures should not be compared directly across those experiments as if they shared an identical optimizer configuration.

### How to interpret these results

These measurements apply to this model, GPU, configuration, and software environment. They are not universal speedup guarantees. Results may differ with model architecture, batch size, sequence length, GPU type, PyTorch version, and optimizer implementation.

---

## What This Project Measures

The benchmark captures the end-to-end training-step cost, including:

1. Synthetic input generation.
2. Forward pass and loss computation.
3. Backward pass.
4. Optimizer update.

Warm-up steps are excluded from the measured interval. CUDA synchronization is used around timed steps so asynchronous GPU execution does not invalidate wall-clock timing.

Each benchmark JSON records information such as:

- GPU and software versions.
- Git commit identifier.
- Configuration and random seed.
- Attention backend and precision.
- Optimizer implementation.
- Number of warm-up and measured steps.
- Per-step timings and aggregate throughput.
- Loss values.
- Peak allocated and reserved GPU memory.

The benchmark uses synthetic token inputs. The purpose is to measure execution behavior, not to train a production model or demonstrate model quality.

## Architecture and Experiment Design

The project uses a configurable decoder-only Transformer. The model has a pre-normalized Transformer structure with RMSNorm, SwiGLU feed-forward layers, tied embeddings, and causal self-attention.

The baseline experiment uses a small configuration to keep iteration practical while exposing meaningful compute and memory behavior.

| Parameter | Baseline setting |
|---|---|
| Transformer layers | 6 |
| Hidden size | 768 |
| Attention heads | 12 |
| Intermediate size | 2,048 |
| Vocabulary size | 32,000 |
| Batch size | 2 |
| Sequence length | 128 |
| Maximum sequence length | 1,024 |
| Attention backend | Eager |
| Compute precision | FP32 |
| Random seed | 42 |

The baseline configuration is maintained separately from validation configurations. Longer-context experiments use `configs/validation_seq2048.yaml`, leaving `configs/baseline.yaml` unchanged.

### Attention backends

**Eager attention**

Uses the model's explicit attention operations. This provides a useful comparison point for investigating intermediate tensors, memory traffic, and kernel-launch overhead.

**PyTorch SDPA**

Uses `torch.nn.functional.scaled_dot_product_attention`, allowing PyTorch to select a supported attention implementation.

**Forced Flash Attention**

Uses PyTorch's SDPA backend selection to require `SDPBackend.FLASH_ATTENTION`. The benchmark fails rather than silently substituting another SDPA backend if the requested implementation cannot run.

### Precision

BF16 is enabled through autocast around the forward pass. Model parameters remain FP32, and the optimizer is not converted to a BF16-parameter optimizer.

BF16 results are therefore comparisons of the tested autocast execution paths, not full-model BF16 parameter storage.

### Optimizers

The benchmark supports:

- `adamw`: standard AdamW execution with `fused=False`.
- `fused_adamw`: AdamW with `fused=True`.

The standard-versus-fused comparison uses the same model configuration, precision, attention backend, warm-up count, and measured step count.

---

## Profiling Findings

PyTorch Profiler and CUDA events were used to inspect two-step training runs at sequence length 2,048.

### Attention-compute timing

| Profile measurement | Eager BF16 | Flash BF16 |
|---|---:|---:|
| Attention CUDA-event time | 284.156 ms | 17.139 ms |
| Forward-pass CUDA-event time | 622.145 ms | 384.592 ms |
| Attention / forward-pass time | 45.7% | 4.5% |

The instrumented attention-compute region became approximately 94% faster with Flash Attention.

These event totals are summed across layers and profiled steps. They are not individual step latencies and should not be compared directly with ordinary benchmark timings.

The instrumented attention region excludes QKV projections, output projection, and backward computation.

### Other observed costs

The profiler also highlighted:

- Matrix multiplication operations (`aten::mm`).
- Tensor-copy operations (`aten::copy_`).
- AdamW optimizer execution.
- CUDA kernel launches and synchronization.

These findings helped guide the next experiment: evaluating fused AdamW after reducing the attention-compute cost.

Profiler output includes instrumentation overhead and overlapping timing scopes. Operator-table timings should not simply be added together.

---

## Repository Layout

The main files and experiment outputs are organized around benchmarking, profiling, configuration, and evidence.

```text
perf-transformer/
├── model.py
├── train.py
├── benchmark.py
├── profile_step.py
├── data.py
├── configs/
│   ├── baseline.yaml
│   └── validation_seq2048.yaml
├── tests/
├── results/
│   ├── raw/
│   │   └── benchmark_*.json
│   ├── validation/
│   │   └── *.log
│   └── traces/
│       └── profiler traces and metadata
└── README.md
```

The layout above highlights the core workflow; additional project directories may contain metrics, observability, experiment helpers, and supporting utilities.

Profiler traces and generated trace archives should remain out of Git unless there is a specific reason to publish them.

## Environment

The recorded experiment environment was:

| Component | Version or configuration |
|---|---|
| GPU | NVIDIA L4 |
| GPU memory | 23,034 MiB reported by NVIDIA-SMI |
| NVIDIA driver | 580.126.20 |
| CUDA version reported by NVIDIA-SMI | 13.0 |
| PyTorch | 2.14.1+cu130 |
| Python | 3.12.3 |
| Operating system | Ubuntu-based Linux environment |

The commands below assume a CUDA-enabled environment with compatible NVIDIA drivers and a working PyTorch CUDA installation.

## Reproducing the Experiments

### 1. Set up the environment

Clone the repository and create an isolated Python environment:

```bash
git clone https://github.com/devam1402/Profiling-end-to-end.git
cd Profiling-end-to-end

python3 -m venv .venv
source .venv/bin/activate

python -m pip install --upgrade pip
```

Install the project's dependencies using its dependency files, if present. Use a PyTorch build compatible with your GPU and CUDA environment.

Verify the environment:

```bash
python - <<'PY'
import torch

print("PyTorch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
PY
```

### 2. Run the baseline benchmark

```bash
python benchmark.py \
  --config configs/baseline.yaml \
  --device cuda \
  --attention-backend eager \
  --precision fp32 \
  --warmup 5 \
  --steps 30
```

### 3. Compare eager attention and SDPA

```bash
python benchmark.py \
  --config configs/baseline.yaml \
  --device cuda \
  --attention-backend sdpa \
  --precision fp32 \
  --warmup 5 \
  --steps 30
```

For a valid comparison, alternate the configurations and repeat the runs instead of relying on a single measurement.

### 4. Run BF16 Flash Attention

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

### 5. Run the sequence-2048 experiment

```bash
python benchmark.py \
  --config configs/validation_seq2048.yaml \
  --device cuda \
  --attention-backend sdpa \
  --precision bf16 \
  --sdpa-backend flash \
  --warmup 5 \
  --steps 30
```

To compare eager attention at the same sequence length, change the attention options to:

```bash
--attention-backend eager
```

Remove `--sdpa-backend flash` for the eager run.

### 6. Compare standard and fused AdamW

Standard AdamW:

```bash
python benchmark.py \
  --config configs/validation_seq2048.yaml \
  --device cuda \
  --attention-backend sdpa \
  --precision bf16 \
  --sdpa-backend flash \
  --optimizer adamw \
  --warmup 5 \
  --steps 30
```

Fused AdamW:

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

Fused optimizer support depends on the PyTorch build and device. If the fused implementation is unsupported, the benchmark may fail.

### 7. Profile a training run

Eager attention:

```bash
python profile_step.py \
  --config configs/validation_seq2048.yaml \
  --device cuda \
  --attention-backend eager \
  --precision bf16 \
  --steps 2
```

Forced Flash Attention:

```bash
python profile_step.py \
  --config configs/validation_seq2048.yaml \
  --device cuda \
  --attention-backend sdpa \
  --precision bf16 \
  --sdpa-backend flash \
  --steps 2
```

Profiler traces and metadata are written under `results/traces/`. They can be inspected with [Perfetto](https://ui.perfetto.dev/).

**Important:** Do not use profiler-enabled runs for throughput comparisons. Profiling changes execution behavior and adds overhead.

---

## Experimental Methodology

The project follows a controlled performance-engineering loop:

1. Establish a baseline.
2. Change one implementation choice.
3. Warm up the workload.
4. Measure repeated training steps.
5. Record latency, throughput, memory, and loss.
6. Profile when the benchmark reveals a potential bottleneck.
7. Repeat measurements and compare run-level statistics.
8. Preserve the configuration and raw evidence.

For throughput, the benchmark counts input tokens processed per training step using the configured batch size and sequence length. Throughput is a training-workload metric, not autoregressive inference tokens per second.

For fair comparisons:

- Use the same model and input configuration.
- Keep batch size, sequence length, precision, and warm-up settings constant.
- Change only the variable under investigation.
- Alternate benchmark configurations to reduce ordering effects.
- Repeat runs and report medians alongside individual measurements.
- Record the exact software and hardware environment.
- Keep the baseline configuration unchanged.

## Limitations

This is a performance-engineering lab, not a production training framework.

- Inputs are synthetic token sequences; no dataset-quality or convergence claims are made.
- Results represent one GPU type and one compact Transformer configuration.
- The experiments do not establish scaling behavior across multiple GPUs.
- Profiler timings include overhead and should be interpreted separately from unprofiled throughput.
- BF16 may introduce numerical differences relative to FP32.
- Short-run loss comparisons do not establish equivalent model quality or convergence.
- Performance improvements may vary across PyTorch versions, GPU architectures, model dimensions, and workload shapes.
- The reported numbers are experimental observations, not universal guarantees.

## Future Work

Potential follow-up experiments include:

- Automated result aggregation and comparison reports.
- Longer-context and batch-size sweeps.
- Profiling optimizer and tensor-copy overhead in more detail.
- Activation checkpointing and memory/compute trade-offs.
- Comparing throughput and memory under additional precision modes.
- Multi-GPU distributed training experiments.
- Profiling with NVIDIA Nsight Systems and Nsight Compute.
- Automated regression detection across benchmark runs.

Each extension should follow the same principle: measure the baseline, isolate the change, repeat the experiment, and retain the evidence.

## Author

**Devam Thacker**

Master of Financial Technology candidate with interests in AI infrastructure, GPU performance engineering, cloud-native platforms, and ML systems.

GitHub: [@devam1402](https://github.com/devam1402)

---

*Built as a practical study of Transformer performance engineering: from end-to-end measurement to CUDA profiling and evidence-driven optimization.*
