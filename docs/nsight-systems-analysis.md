# Nsight Systems Analysis

**Status:** Initial timeline captured; controlled input-placement comparison and clean post-warm-up slice are pending.  
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

## Controlled input-placement comparison

The existing benchmark already supports device-generated synthetic inputs. A host-input mode has now been added to `benchmark.py`:

- `--input-mode device`: create synthetic token IDs with `torch.randint(..., device="cuda")`.
- `--input-mode host`: create the same shape of synthetic token IDs on CPU, then call `.to(device, non_blocking=True)`.

Both modes include input generation in the measured step interval. The host mode uses ordinary CPU tensors rather than pinned-memory prefetching, so this experiment tests host generation plus transfer cost; it does not claim to represent an optimized production input pipeline.

Run the interleaved comparison on the L4 with the best measured model configuration (BF16 autocast, forced Flash Attention, fused AdamW, sequence length 2,048):

```bash
python compare_input_modes.py --pairs 4 --warmup 5 --steps 30
```

The script alternates run order across pairs, leaves `configs/baseline.yaml` unchanged, and writes individual benchmark JSONs plus an aggregate report under `results/validation/`. **The result must be run on the GPU host; this document does not yet claim that H2D traffic is or is not a limiter.**

The token batch itself is only about 32 KB for batch size 2, sequence length 2,048, and int64 token IDs. The previously observed ~6.3 MB H2D operations are much larger than that input batch, so their source must not be inferred from size alone. If needed, use a short clean Nsight Systems capture of each mode to correlate transfer activity with the warmed-up step timeline.

## Current conclusions

1. Flash Attention kernels were observed in the initial Nsight Systems capture; a clean post-warm-up slice is still needed to document their steady-state placement.
2. Library loading dominates part of the short process-level trace and should be treated as startup rather than steady-state step cost.
3. H2D transfers were observed, but their contribution to steady-state step time remains unproven until the controlled input-placement comparison is run.
4. Existing NVTX output did not provide reliable top-level forward/backward/optimizer attribution. Accept this limitation rather than adding ranges solely for this write-up.
5. The capture used for initial kernel observations ran `benchmark.py` without PyTorch Profiler and did not report the CUPTI multiple-subscriber error. The separate nested PyTorch Profiler + Nsight attempt did report that error and is excluded from Nsight kernel conclusions.
6. All published throughput comparisons remain from unprofiled, repeated/interleaved benchmark runs; profiled values are diagnostic only.

## Remaining close-out steps

1. Run `compare_input_modes.py` on the L4 and record median step time and throughput for both modes.
2. Capture a short `nsys profile` of the fused-AdamW device-input and host-input modes, then inspect only the post-warm-up measured region (after the initial ~1–1.2 seconds, if the trace timing supports that cutoff). Note whether kernels stay busy, where H2D/D2D copies occur, and whether there are multi-millisecond idle gaps at step boundaries.
3. If the input comparison shows no meaningful step-time difference and the corresponding H2D operations disappear in device mode, conclude that input transfers are present but not the limiter. If host mode is slower and transfers sit on the step critical path, test prefetching once and remeasure.
4. Do not start Nsight Compute unless the steady-state timeline identifies a specific unexplained hot kernel.

## Artifact handling

The binary `.nsys-rep` file, exported SQLite database, text trace dumps, and PyTorch Profiler trace files are local analysis artifacts and should remain out of Git. This document records the conclusions without committing the raw traces.
