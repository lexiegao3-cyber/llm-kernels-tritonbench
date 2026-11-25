"""Standalone RMSNorm benchmark (no tritonbench needed).

Compares, for Llama-3 hidden sizes and a sweep of token counts:
  torch_ref        PyTorch eager reference (fp32 accumulate)
  torch_compile    torch.compile of the reference
  liger            Liger-Kernel (if installed)
  flashinfer       flashinfer.rmsnorm / flashinfer.fused_add_rmsnorm (if installed)
  ours_triton      llm_kernels.rmsnorm / fused_add_rmsnorm

Reports latency (median, us), achieved bandwidth (GB/s) and % of the GPU's
peak DRAM bandwidth. Writes a CSV to results/.

Usage:
  python bench/bench_rmsnorm.py --mode rmsnorm
  python bench/bench_rmsnorm.py --mode fused_add --peak-gbps 2039   # A100-80GB
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from datetime import datetime

import torch
import triton

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from llm_kernels import fused_add_rmsnorm, rmsnorm, rmsnorm_ref  # noqa: E402

try:
    import flashinfer
except ImportError:
    flashinfer = None

try:
    from liger_kernel.ops.rms_norm import LigerRMSNormFunction
except ImportError:
    LigerRMSNormFunction = None

# Theoretical DRAM bandwidth in GB/s. Pass --peak-gbps for other GPUs.
PEAK_GBPS = {
    "H100 80GB HBM3": 3350,
    "H100": 3350,
    "A100-SXM4-80GB": 2039,
    "A100-SXM4-40GB": 1555,
    "A100": 1555,
    "L4": 300,
    "A10G": 600,
    "T4": 320,
    "RTX 4090": 1008,
}


def guess_peak(name: str) -> float | None:
    for k, v in PEAK_GBPS.items():
        if k in name:
            return v
    return None


def bytes_moved(mode: str, M: int, H: int, elem: int) -> int:
    # Minimum DRAM traffic. Weight is H elements, read once (L2-resident).
    if mode == "rmsnorm":
        return (2 * M * H + H) * elem  # read x, write y
    return (4 * M * H + H) * elem  # read x, r; write r, y


def make_fns(mode: str, x, r, w, eps):
    fns = {}
    if mode == "rmsnorm":
        fns["torch_ref"] = lambda: rmsnorm_ref(x, w, eps)
        compiled = torch.compile(rmsnorm_ref, mode="max-autotune-no-cudagraphs")
        fns["torch_compile"] = lambda: compiled(x, w, eps)
        if LigerRMSNormFunction is not None:
            fns["liger"] = lambda: LigerRMSNormFunction.apply(x, w, eps)
        if flashinfer is not None:
            fns["flashinfer"] = lambda: flashinfer.rmsnorm(x, w, eps)
        fns["ours_triton"] = lambda: rmsnorm(x, w, eps)
    else:
        # In-place ops: reuse the same buffers every iteration. Values drift
        # but timing is unaffected; correctness is covered by tests/.
        def torch_fused():
            s = (x.float() + r.float()).to(x.dtype)
            r.copy_(s)
            x.copy_(rmsnorm_ref(s, w, eps))

        fns["torch_ref"] = torch_fused
        compiled = torch.compile(torch_fused, mode="max-autotune-no-cudagraphs")
        fns["torch_compile"] = compiled
        if flashinfer is not None:
            fns["flashinfer"] = lambda: flashinfer.fused_add_rmsnorm(x, r, w, eps)
        fns["ours_triton"] = lambda: fused_add_rmsnorm(x, r, w, eps)
    return fns


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["rmsnorm", "fused_add"], default="rmsnorm")
    p.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    p.add_argument("--hidden", type=int, nargs="+", default=[4096, 8192])  # Llama-3 8B / 70B
    p.add_argument("--tokens", type=int, nargs="+",
                   default=[1, 16, 64, 256, 1024, 4096, 16384])
    p.add_argument("--peak-gbps", type=float, default=None)
    p.add_argument("--out", default=None)
    args = p.parse_args()

    assert torch.cuda.is_available(), "This benchmark needs an NVIDIA GPU."
    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[args.dtype]
    gpu = torch.cuda.get_device_name()
    peak = args.peak_gbps or guess_peak(gpu)
    print(f"GPU: {gpu} | peak DRAM BW: {peak or 'unknown (pass --peak-gbps)'} GB/s")
    print(f"torch {torch.__version__} | triton {triton.__version__} | "
          f"flashinfer {getattr(flashinfer, '__version__', 'not installed')}")

    rows = []
    eps = 1e-6
    for H in args.hidden:
        for M in args.tokens:
            x = torch.randn(M, H, device="cuda", dtype=dtype)
            r = torch.randn(M, H, device="cuda", dtype=dtype)
            w = torch.randn(H, device="cuda", dtype=dtype)
            nbytes = bytes_moved(args.mode, M, H, x.element_size())
            for name, fn in make_fns(args.mode, x, r, w, eps).items():
                ms = triton.testing.do_bench(fn, warmup=25, rep=100, return_mode="median")
                gbps = nbytes / (ms * 1e-3) / 1e9
                row = dict(mode=args.mode, dtype=args.dtype, H=H, M=M, backend=name,
                           latency_us=round(ms * 1e3, 2), gbps=round(gbps, 1),
                           pct_peak=round(100 * gbps / peak, 1) if peak else "")
                rows.append(row)
                print(f"H={H:6d} M={M:6d} {name:14s} {row['latency_us']:9.2f} us "
                      f"{row['gbps']:8.1f} GB/s {row['pct_peak']}%")

    os.makedirs("results", exist_ok=True)
    out = args.out or f"results/{args.mode}_{args.dtype}_{gpu.replace(' ', '_')}_" \
                      f"{datetime.now():%Y%m%d_%H%M}.csv"
    with open(out, "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        wr.writeheader()
        wr.writerows(rows)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
