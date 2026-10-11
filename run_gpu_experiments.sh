#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

echo "=== 1. Check GPU ==="
python check_gpu.py

if ! python -c 'import torch; assert torch.cuda.is_available()' 2>/dev/null; then
    echo "STOP: CUDA is unavailable. No experiments were run."
    echo "Run this script on a compatible NVIDIA GPU environment."
    exit 2
fi

echo
echo "=== 2. Record environment ==="
nvidia-smi

echo
echo "=== 3. Run correctness tests ==="
python -m unittest discover -s tests -v

echo
echo "=== 4. Benchmark eager-attention baseline ==="
python benchmark.py --device cuda --warmup 5 --steps 20

echo
echo "=== 5. Capture separate profiler trace ==="
python profile_step.py --device cuda --steps 3

echo
echo "=== Experiment files ==="
find results/raw results/traces -maxdepth 1 -type f \
    \( -name 'benchmark_*.json' -o -name '*profile*' \) \
    -printf '%TY-%Tm-%Td %TH:%TM %p\n' | sort

echo
echo "Baseline experiment completed."
