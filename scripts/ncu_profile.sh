#!/usr/bin/env bash
# Nsight Compute profile of our kernel vs FlashInfer at one large and one tiny shape.
# Needs ncu on PATH (ships with the CUDA toolkit) and permission to read GPU counters.
set -euo pipefail
HERE="$(cd "$(dirname "$0")/.." && pwd)"
mkdir -p "$HERE/results/ncu"
for M in 1 4096; do
  ncu --set full --target-processes all -k "regex:rmsnorm|RMSNorm" -c 4 \
      -o "$HERE/results/ncu/rmsnorm_M${M}" -f \
      python "$HERE/bench/profile_once.py" --tokens "$M" --hidden 4096
done
echo "open with: ncu-ui results/ncu/rmsnorm_M*.ncu-rep"
echo "look at: DRAM Throughput, Memory Throughput %, Achieved Occupancy, L1/L2 sectors per request"
