# Phase 5 — Activation Checkpointing

## Objective

Measure the memory/throughput trade-off of activation checkpointing
on an NVIDIA L4 using BF16, SDPA Flash Attention, fused AdamW,
and device-side synthetic inputs.

## Correctness

The activation-checkpointing correctness test passed.

- Configuration: configs/validation_seq2048.yaml
- Baseline loss: 10.53256321
- Full-checkpointing loss: 10.53256321
- Maximum absolute gradient difference: 7.7809207e-05
- Gradients checked: 45

## Benchmark Results

Hardware: NVIDIA L4
Measured steps: 5 per configuration
Warm-up steps: 2 per configuration

| Run | Policy | Batch | Median step (ms) | Throughput (tokens/s) | Peak allocated (MiB) | Peak reserved (MiB) |
|---|---|---:|---:|---:|---:|---:|
| A | none | 2 | 82.3 | 49631.7 | 3398.5 | 3904 |
| B | every_other | 2 | 88.3 | 46199.4 | 2930.6 | 3472 |
| C | every_block | 2 | 94.9 | 43111.7 | 2458.0 | 3092 |
| D | every_block | 4 | 188.6 | 43281.6 | 4056.1 | 4752 |
| E | every_block | 3 | 139.3 | 44020.3 | 3257.1 | 4434 |

## Findings

- Partial checkpointing reduced allocated memory by approximately
  13.8%, with throughput approximately 6.9% below baseline.
- Full checkpointing at batch size 2 reduced allocated memory by
  approximately 27.7%, with throughput approximately 13.1% below baseline.
- Full checkpointing at batch size 3 used approximately 4.2% less
  allocated memory than baseline, but throughput was approximately
  11.3% lower.
- Full checkpointing at batch size 4 exceeded baseline allocated
  memory and did not recover baseline throughput.

## Conclusion

For this workload, activation checkpointing primarily provides a
memory-saving trade-off rather than a throughput improvement.
Batch size 3 is the closest memory-matched full-checkpointing result.

These are preliminary measurements based on five measured steps
per configuration. Repeat runs are needed for stronger performance
claims. Different batch sizes also make final training losses
unsuitable for direct comparison.

## Raw Evidence

- benchmark_20261016T181055Z.json — A
- benchmark_20261016T181105Z.json — B
- benchmark_20261016T181114Z.json — C
- benchmark_20261016T181159Z.json — D
- benchmark_20261016T181330Z.json — E
