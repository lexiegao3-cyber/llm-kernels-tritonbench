"""RMSNorm and fused residual-add + RMSNorm in Triton.

Semantics intentionally mirror FlashInfer so the two can be compared 1:1:

* ``rmsnorm(x, w, eps)``            -> ``flashinfer.rmsnorm(x, w, eps)``
      out[i] = x[i] / RMS(x) * w[i]

* ``fused_add_rmsnorm(x, r, w, eps)`` -> ``flashinfer.fused_add_rmsnorm(x, r, w, eps)``
      r[i] += x[i]                       (in place)
      x[i]  = r[i] / RMS(r) * w[i]       (in place)

Design (memory-bound kernel, so the goal is to touch each byte exactly once):

* One Triton program per row (token). The whole row is held in registers,
  so ``x`` is read once, ``y`` is written once (plus ``residual`` read+write
  for the fused variant). Minimum DRAM traffic per row:
      rmsnorm:            2 * H * elem_size   (+ H weight, cached in L2)
      fused_add_rmsnorm:  4 * H * elem_size
* The sum of squares is accumulated in fp32 regardless of input dtype.
* ``BLOCK_SIZE = next_power_of_2(H)``; ``num_warps`` scales with H so each
  thread handles a reasonable number of elements (vectorised 128-bit loads).
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


def _num_warps(block_size: int) -> int:
    # Heuristic: ~ 8-16 elements per thread for bf16.
    if block_size >= 16384:
        return 16
    if block_size >= 4096:
        return 8
    if block_size >= 1024:
        return 4
    return 2


@triton.jit
def _rmsnorm_fwd_kernel(
    X,
    W,
    Y,
    stride_x,
    stride_y,
    H,
    eps,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < H

    x = tl.load(X + row * stride_x + cols, mask=mask, other=0.0).to(tl.float32)
    w = tl.load(W + cols, mask=mask, other=0.0).to(tl.float32)

    mean_sq = tl.sum(x * x, axis=0) / H
    rstd = 1.0 / tl.sqrt(mean_sq + eps)
    y = x * rstd * w

    tl.store(Y + row * stride_y + cols, y.to(Y.dtype.element_ty), mask=mask)


@triton.jit
def _fused_add_rmsnorm_fwd_kernel(
    X,  # in: hidden states, out: normalised output (in place)
    R,  # in: residual, out: residual + x (in place)
    W,
    stride_x,
    stride_r,
    H,
    eps,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < H

    x = tl.load(X + row * stride_x + cols, mask=mask, other=0.0)
    r = tl.load(R + row * stride_r + cols, mask=mask, other=0.0)

    # Same rounding as FlashInfer / vLLM: add in fp32, store residual in the
    # input dtype, and normalise the *rounded* residual.
    s = (x.to(tl.float32) + r.to(tl.float32)).to(R.dtype.element_ty)
    tl.store(R + row * stride_r + cols, s, mask=mask)

    s32 = s.to(tl.float32)
    w = tl.load(W + cols, mask=mask, other=0.0).to(tl.float32)
    mean_sq = tl.sum(s32 * s32, axis=0) / H
    rstd = 1.0 / tl.sqrt(mean_sq + eps)
    y = s32 * rstd * w

    tl.store(X + row * stride_x + cols, y.to(X.dtype.element_ty), mask=mask)


def _check(x: torch.Tensor, weight: torch.Tensor) -> tuple[torch.Tensor, int, int]:
    assert x.shape[-1] == weight.shape[0], "hidden size mismatch"
    assert weight.is_contiguous()
    H = x.shape[-1]
    x2d = x.reshape(-1, H)
    assert x2d.stride(-1) == 1, "last dim must be contiguous"
    return x2d, x2d.shape[0], H


def rmsnorm(
    x: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1e-6,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Out-of-place RMSNorm over the last dimension."""
    x2d, M, H = _check(x, weight)
    if out is None:
        out = torch.empty_like(x)
    y2d = out.view(-1, H)
    BLOCK_SIZE = triton.next_power_of_2(H)
    _rmsnorm_fwd_kernel[(M,)](
        x2d,
        weight,
        y2d,
        x2d.stride(0),
        y2d.stride(0),
        H,
        eps,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=_num_warps(BLOCK_SIZE),
    )
    return out


def fused_add_rmsnorm(
    x: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1e-6,
) -> None:
    """In place: ``residual += x``; ``x = rmsnorm(residual) * weight``.

    Matches the signature and in-place semantics of
    ``flashinfer.fused_add_rmsnorm``.
    """
    x2d, M, H = _check(x, weight)
    assert residual.shape == x.shape and residual.dtype == x.dtype
    r2d = residual.view(-1, H)
    BLOCK_SIZE = triton.next_power_of_2(H)
    _fused_add_rmsnorm_fwd_kernel[(M,)](
        x2d,
        r2d,
        weight,
        x2d.stride(0),
        r2d.stride(0),
        H,
        eps,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=_num_warps(BLOCK_SIZE),
    )


# ---------------------------------------------------------------------------
# PyTorch references (used by tests and as the benchmark baseline)
# ---------------------------------------------------------------------------


def rmsnorm_ref(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    x32 = x.float()
    rstd = torch.rsqrt(x32.pow(2).mean(-1, keepdim=True) + eps)
    return (x32 * rstd * weight.float()).to(x.dtype)


def fused_add_rmsnorm_ref(
    x: torch.Tensor, residual: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6
) -> tuple[torch.Tensor, torch.Tensor]:
    """Returns (normalised, new_residual) without modifying inputs."""
    new_res = (x.float() + residual.float()).to(x.dtype)
    return rmsnorm_ref(new_res, weight, eps), new_res
