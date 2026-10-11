# CPU Profiling Experiment

## Environment
- Device: CPU
- PyTorch: 2.14.1+cu130
- Model parameters: 67,839,744
- Batch size: 2
- Sequence length: 128
- Profiling steps: 2

## Results
- Profiler self CPU time total: 4.145 s
- Top operator: aten::mm
- aten::mm self CPU time: 35.74%
- AdamW inclusive CPU time: 39.57%
- Trace: results/traces/cpu_profile.json

## Interpretation
Matrix multiplication is the largest individual operator by self CPU
time in this CPU trace. AdamW has substantial inclusive time, including
its child operations.

These results are a CPU profiling exercise, not a GPU performance
baseline. Profiler overhead and the short run limit comparisons.

## Next
Repeat the baseline on a rented GPU, then compare eager attention with
PyTorch SDPA FlashAttention using the same model, batch and sequence.
