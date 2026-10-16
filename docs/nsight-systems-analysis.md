# Nsight Systems Analysis

**Status:** Initial CUDA timeline captured; attribution is still incomplete.  
**Date:** 2026-10-16  
**Hardware:** NVIDIA L4 (23,034 MiB reported by NVIDIA-SMI)  
**Software:** Ubuntu 24.04-based container, NVIDIA driver 580.126.20, PyTorch 2.14.1+cu130, Python 3.12.3, Nsight Systems CLI 2026.5.1.161.

## Objective

Use Nsight Systems to inspect end-to-end CUDA activity for the Transformer training workload, including CUDA API activity, GPU kernels, memory transfers, synchronization, and NVTX attribution. This is a diagnostic trace, not a throughput benchmark.

## Capture A: benchmark process without PyTorch Profiler

The first successful Nsight Systems capture ran `benchmark.py` directly, avoiding a nested PyTorch Profiler session:

```bash
nsys profile \
  --trace=cuda,nvtx,osrt \
  --sample=none \
  --force-overwrite=true \
  --output=results/traces/nsys_bf16_flash_seq2048 \
  .venv/bin/python benchmark.py \
    --config configs/validation_seq2048.yaml \
    --device cuda \
    --warmup 2 \
    --steps 5 \
    --precision bf16 \
    --attention-backend sdpa \
    --sdpa-backend flash
```

Configuration: sequence length 2,048, BF16 autocast, SDPA with forced Flash Attention, standard AdamW, two warm-up steps, and five measured steps.

The run reported an average step time of 88.4 ms, median step time of 88.5 ms, throughput of 46,303.5 tokens/s, final loss 10.5295, peak allocated memory 3,398.5 MiB, and peak reserved memory 3,904 MiB. Because this is a short, profiler-instrumented capture, these values are not used to replace the repeated unprofiled benchmark results in the README.

### CUDA API and GPU summary

Selected `cuda_api_gpu_sum` observations:

| Item | Observed value | Interpretation |
|---|---:|---|
| `cuLibraryLoadData` | 897.5 ms across 35 calls; 42.0% of report total | Large library-loading/startup component in this capture; not evidence of a steady-state step bottleneck by itself |
| `cudaDeviceSynchronize` | 254.0 ms across 11 calls | Synchronization warrants timeline inspection; aggregate API duration alone does not establish avoidable waiting |
| `cudaLaunchKernel` | 171.5 ms API time across 6,663 calls | High launch activity; API time is CPU-side launch duration, not GPU kernel execution time |
| `cudaStreamSynchronize` | 93.7 ms across 57 calls | Inspect placement and overlap in the timeline before proposing changes |
| Host-to-device memcpy | 45 operations; 22.2 ms aggregated in the summary | Transfers are present, but the short trace includes initialization and warm-up, so this does not yet prove an input-pipeline bottleneck |
| Device-to-device memcpy | 91 operations; 7.9 ms aggregated in the summary | Worth correlating with operation context if optimizing memory traffic |

The summary also showed BF16 matrix-multiplication kernels and Flash Attention forward/backward kernels. In the GPU trace, the Flash Attention forward kernel appeared 42 times with about 13.0 ms aggregate kernel time; the Flash Attention backward kernel appeared 42 times with about 28.4 ms aggregate kernel time. These are aggregate values from the captured process, not per-step latencies.

**Interpretation caveat:** the `cuda_api_gpu_sum` report combines different categories of activity. Its percentage column must not be read as a clean end-to-end wall-clock breakdown, and API time must not be equated with GPU execution time.

### NVTX attribution

The `nvtx_pushpop_sum` and `nvtx_gpu_proj_sum` reports showed mainly internal CCCL/CUB ranges:

- `CCCL:cub::DeviceScan::ExclusiveScan`
- `CCCL:cub::DeviceSelect::UniqueByKey`
- `CCCL:cub::DeviceRadixSort`

These do not provide useful top-level attribution for the model's forward, backward, and optimizer phases. The initial `benchmark.py` trace therefore identifies activity to investigate but does not yet isolate the full training-step critical path.

## Capture B: PyTorch Profiler nested inside Nsight Systems

A later attempt ran `profile_step.py` under `nsys profile` for three BF16/Flash steps. It printed:

```text
CUPTI_ERROR_MULTIPLE_SUBSCRIBERS_NOT_SUPPORTED
CUPTI initialization failed - CUDA profiler activities will be missing
```

PyTorch Profiler and Nsight Systems both attempted to subscribe to CUPTI. As a result, the generated Nsight Systems report may be missing CUDA activity and must **not** be used for GPU-kernel attribution. NVIDIA documents this error as a conflict between CUPTI clients when the active configuration does not support multiple subscribers ([CUPTI documentation](https://docs.nvidia.com/cupti/main/main.html)).

The PyTorch Profiler output from that run remains useful as a separate measurement:

| PyTorch Profiler measurement | Result |
|---|---:|
| CUDA-event attention-compute time, summed across layers and 3 steps | 30.984 ms |
| CUDA-event forward-pass time, summed across 3 steps | 492.715 ms |
| Attention / forward-pass event time | 6.3% |
| CPU-side `training_backward` self time | 259.955 ms |
| CPU-side `aten::mm` self time | 249.587 ms |
| CPU-side `aten::_foreach_sqrt` self time | 137.659 ms |

These CPU-side operator timings are not directly comparable with CUDA-event timings. The 6.3% attention ratio applies only to the instrumented forward-pass scope and the attention region measured; it is not attention's share of total training time. The optimizer-related operator observation is a lead for future investigation, not a confirmed bottleneck.

## Current conclusions

1. Flash Attention kernels are present in the initial Nsight Systems capture.
2. Library initialization is a substantial component of the short process-level trace; separate it from steady-state execution before optimizing.
3. Synchronization calls, launch activity, and memory transfers are visible, but no one of these is yet proven to be the dominant avoidable cost.
4. Existing NVTX output does not provide reliable top-level model-phase attribution.
5. Do not run PyTorch Profiler and Nsight Systems CUDA tracing together in this environment; collect each separately.

## Next steps

1. Capture a short Nsight Systems trace of `benchmark.py` without PyTorch Profiler and inspect the timeline around warmed-up measured steps.
2. If phase-level attribution is needed, add/enable compatible NVTX ranges in the application and verify they appear in the trace before interpreting projections.
3. Install and verify Nsight Compute (`ncu`) before collecting kernel-level metrics.
4. Use Nsight Compute on a representative kernel after choosing it from a valid Nsight Systems/PyTorch Profiler trace.
5. Keep profiling runs separate from unprofiled repeated throughput benchmarks.

## Artifact handling

The binary `.nsys-rep` file, exported SQLite database, text trace dumps, and PyTorch Profiler trace files are local analysis artifacts and should remain out of Git. This document records the conclusions without committing the raw traces.
