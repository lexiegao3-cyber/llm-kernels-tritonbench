#!/usr/bin/env bash
# Run the two tritonbench operators we extend and save CSVs to results/.
set -euo pipefail
HERE="$(cd "$(dirname "$0")/.." && pwd)"
TB_DIR="${TB_DIR:-$HERE/../tritonbench}"
mkdir -p "$HERE/results"
cd "$TB_DIR"

# Plain RMSNorm, Llama-3 8B hidden size, sweep H via defaults or fix --H.
for M in 1 64 1024 4096 16384; do
  python run.py --op rms_norm --mode fwd --precision bf16 --M "$M" \
    --metrics latency,gbps,speedup,accuracy --csv \
    > "$HERE/results/tritonbench_rms_norm_M${M}.csv"
done

# Fused residual add + RMSNorm (batch*seq tokens).
python run.py --op inductor_residual_rmsnorm --precision bf16 --batch 4 --seq-len 2048 \
  --metrics latency,gbps,speedup --csv \
  > "$HERE/results/tritonbench_residual_rmsnorm.csv"

echo "results in $HERE/results/"
