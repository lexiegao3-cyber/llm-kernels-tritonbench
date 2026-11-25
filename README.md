# llm-kernels-tritonbench

Triton kernels for LLM inference, benchmarked side by side with **FlashInfer**, Liger-Kernel,
torch.compile and PyTorch eager inside [meta-pytorch/tritonbench](https://github.com/meta-pytorch/tritonbench).

> **Status (2026-11-25):** RMSNorm / fused residual-add + RMSNorm kernels are implemented and pass
> correctness tests in Triton's CPU interpreter. **GPU benchmarks have not been run yet** — every
> number in the Results section is a placeholder until measured.

## Why

Decode-time LLM inference is dominated by memory-bound kernels (norms, decode attention), so the
question is how close a hand-written Triton kernel gets to peak DRAM bandwidth, and how it compares
with production libraries.

tritonbench already wires in Liger-Kernel, FlashAttention, tilelang, Quack, AITer and others, but
(as of tritonbench `345c230`, 2026-11-24) it has **no FlashInfer backend**. This repo adds one
(`tritonbench_patches/0001-*`) and uses it as the reference point for my own kernels.

## What's here

| Path | What |
| --- | --- |
| `llm_kernels/rmsnorm.py` | Triton `rmsnorm` and in-place `fused_add_rmsnorm` with the same semantics as `flashinfer.rmsnorm` / `flashinfer.fused_add_rmsnorm`, plus PyTorch references |
| `tests/test_rmsnorm.py` | Correctness vs. PyTorch (fp32/fp16/bf16, Llama-3 shapes) and vs. FlashInfer on GPU |
| `bench/bench_rmsnorm.py` | Standalone benchmark: latency, GB/s, % of peak DRAM bandwidth → CSV |
| `bench/plot.py` | Bandwidth-vs-tokens plot from a CSV |
| `tritonbench_patches/0001-add-flashinfer-backends.patch` | Adds `flashinfer_rms` to `rms_norm` and `flashinfer_fused_add_rmsnorm` to `inductor_residual_rmsnorm` (upstreamable) |
| `tritonbench_patches/0002-add-llm-kernels-triton-backends.patch` | Adds this repo's Triton kernels as tritonbench backends |
| `scripts/setup_tritonbench.sh` | Clone tritonbench, apply patches, install FlashInfer / Liger / this package |
| `scripts/run_tritonbench.sh` | Run both operators and save CSVs to `results/` |
| `scripts/ncu_profile.sh` | Nsight Compute profile of ours vs FlashInfer (M=1 and M=4096) |
| `docs/PLAN.md` | Project phases and milestones |

## Kernel design (RMSNorm)

- One Triton program per row (token); the whole row lives in registers, so each byte of `x`
  (and `residual`) is read once and written once.
  Minimum traffic: `2·M·H` elements for RMSNorm, `4·M·H` for fused add + RMSNorm.
- Sum of squares accumulated in fp32; `BLOCK_SIZE = next_power_of_2(H)`; `num_warps` scales with H.
- Fused variant matches FlashInfer/vLLM rounding: add in fp32, store `residual` in the input dtype,
  normalise the rounded residual, write the output into `x` in place.

## How to run (needs an NVIDIA GPU)

Any Linux box with an NVIDIA GPU works: a cloud A100/H100/L4, or Colab (L4/A100 runtime).

```bash
git clone https://github.com/lexiegao3-cyber/llm-kernels-tritonbench.git
cd llm-kernels-tritonbench

# 1. Quick path: standalone benchmark, no tritonbench needed
pip install torch triton flashinfer-python liger-kernel matplotlib pytest
pip install -e .
pytest tests/                                   # correctness incl. vs FlashInfer
python bench/bench_rmsnorm.py --mode rmsnorm    # add --peak-gbps N if your GPU isn't auto-detected
python bench/bench_rmsnorm.py --mode fused_add
python bench/plot.py results/rmsnorm_bf16_*.csv

# 2. tritonbench path (installs PyTorch nightly; use a fresh venv/conda env)
bash scripts/setup_tritonbench.sh               # clones ../tritonbench and applies patches
bash scripts/run_tritonbench.sh                 # CSVs -> results/

# 3. Profiling
bash scripts/ncu_profile.sh
```

No GPU? You can still check correctness on CPU with Triton's interpreter:

```bash
pip install torch triton pytest && pip install -e .
TRITON_INTERPRET=1 pytest tests/
```

## Results

_GPU: TBD · driver/CUDA: TBD · torch/triton/flashinfer: TBD_

RMSNorm, bf16, achieved bandwidth (GB/s) and % of peak:

| hidden | tokens | torch eager | torch.compile | Liger | FlashInfer | **ours (Triton)** |
| --- | --- | --- | --- | --- | --- | --- |
| 4096 | 1 | TBD | TBD | TBD | TBD | TBD |
| 4096 | 4096 | TBD | TBD | TBD | TBD | TBD |
| 8192 | 1 | TBD | TBD | TBD | TBD | TBD |
| 8192 | 16384 | TBD | TBD | TBD | TBD | TBD |

Fused residual add + RMSNorm: TBD.

Findings (to fill in after profiling):

1. Large token counts: ours reaches TBD% of peak vs FlashInfer TBD%. Reason: TBD (ncu).
2. Small token counts (decode, M=1): every backend is far from peak because TBD
   (launch overhead / too few CTAs to fill the SMs).
3. torch.compile vs hand-written: TBD.

## Roadmap

- [x] Triton RMSNorm + fused add RMSNorm, tests, benchmark harness, FlashInfer backend patch
- [ ] GPU results + Nsight Compute analysis (target: ≥80% of peak DRAM BW at large M)
- [ ] Upstream `0001` as a PR to tritonbench
- [ ] Decode attention: Triton split-K flash-decoding with GQA, vs FlashInfer `BatchDecodeWithPagedKVCacheWrapper` and FA2/FA3
- [ ] Triton GEMM with autotuning + L2 grouped ordering vs cuBLAS; CUDA C++ RMSNorm
- [ ] End-to-end: swap kernels into a small Llama model and measure tokens/s

## API references (verified against FlashInfer 0.7.6 source)

- `flashinfer.rmsnorm(input, weight, eps=1e-6, out=None, enable_pdl=None) -> Tensor`
- `flashinfer.fused_add_rmsnorm(input, residual, weight, eps=1e-6, enable_pdl=None) -> None`
  (in place: `residual += input`, then `input = rmsnorm(residual) * weight`)
- `flashinfer.BatchDecodeWithPagedKVCacheWrapper(workspace_buffer, kv_layout="NHD", use_tensor_cores=False)`
  → `.plan(indptr, indices, last_page_len, num_qo_heads, num_kv_heads, head_dim, page_size, ...)` → `.run(q, paged_kv_cache)`
- `flashinfer.single_decode_with_kv_cache(q, k, v, kv_layout="NHD", ...)`
- pip package: `flashinfer-python`. Docs: https://docs.flashinfer.ai
