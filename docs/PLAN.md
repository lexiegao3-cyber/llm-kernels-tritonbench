# Plan

| Phase | Dates | Deliverable |
| --- | --- | --- |
| 0. MVP | 2026-11-26 → 11-28 | RMSNorm kernels, FlashInfer tritonbench backend, GPU results + ncu analysis, PR to tritonbench |
| 1. Decode attention | 12-01 → 12-14 | Triton split-K flash-decoding with GQA; FlashInfer decode backend in `decoding_attention` |
| 2. GEMM + CUDA C++ | 12-15 → 12-31 | Triton GEMM ≥70% of cuBLAS; CUDA C++ RMSNorm (warp shuffle, 128-bit loads) |
| 3. End-to-end | January | Kernels inside a small Llama model; tokens/s; blog post |

## MVP checklist

- [ ] Rent a GPU; run `pytest tests/` (incl. FlashInfer comparison)
- [ ] `bench/bench_rmsnorm.py` for both modes, H ∈ {4096, 8192}, M from 1 to 16384
- [ ] `scripts/setup_tritonbench.sh` + `scripts/run_tritonbench.sh`
- [ ] `scripts/ncu_profile.sh`: DRAM throughput, achieved occupancy at M=1 and M=4096
- [ ] Fill in README Results + three findings
- [ ] Open PR to meta-pytorch/tritonbench with `0001` (sign Meta CLA first)
