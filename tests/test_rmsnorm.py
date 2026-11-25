"""Correctness tests.

On a GPU:      pytest tests/
Without a GPU: TRITON_INTERPRET=1 pytest tests/   (Triton's CPU interpreter, slow but exact)
"""

import os

import pytest
import torch

from llm_kernels import fused_add_rmsnorm, fused_add_rmsnorm_ref, rmsnorm, rmsnorm_ref

INTERPRET = os.environ.get("TRITON_INTERPRET") == "1"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

if DEVICE == "cpu" and not INTERPRET:
    pytest.skip("needs a CUDA GPU or TRITON_INTERPRET=1", allow_module_level=True)

# Keep the CPU-interpreter run fast; on GPU include Llama-3 shapes.
SHAPES = [(1, 128), (7, 1000), (16, 4096)] if INTERPRET else [
    (1, 4096), (128, 4096), (4096, 4096), (1, 8192), (2048, 8192), (33, 5120), (8, 16384)
]
DTYPES = [torch.float32] if INTERPRET else [torch.float16, torch.bfloat16, torch.float32]


def _tol(dtype):
    return {torch.float32: 1e-5, torch.float16: 2e-3, torch.bfloat16: 2e-2}[dtype]


@pytest.mark.parametrize("M,H", SHAPES)
@pytest.mark.parametrize("dtype", DTYPES)
def test_rmsnorm(M, H, dtype):
    torch.manual_seed(0)
    x = torch.randn(M, H, device=DEVICE, dtype=dtype)
    w = torch.randn(H, device=DEVICE, dtype=dtype)
    y = rmsnorm(x, w, 1e-6)
    torch.testing.assert_close(y, rmsnorm_ref(x, w, 1e-6), atol=_tol(dtype), rtol=_tol(dtype))


@pytest.mark.parametrize("M,H", SHAPES)
@pytest.mark.parametrize("dtype", DTYPES)
def test_fused_add_rmsnorm(M, H, dtype):
    torch.manual_seed(0)
    x = torch.randn(M, H, device=DEVICE, dtype=dtype)
    r = torch.randn(M, H, device=DEVICE, dtype=dtype)
    w = torch.randn(H, device=DEVICE, dtype=dtype)
    y_ref, r_ref = fused_add_rmsnorm_ref(x, r, w, 1e-6)
    fused_add_rmsnorm(x, r, w, 1e-6)  # in place
    torch.testing.assert_close(r, r_ref, atol=_tol(dtype), rtol=_tol(dtype))
    torch.testing.assert_close(x, y_ref, atol=_tol(dtype), rtol=_tol(dtype))


@pytest.mark.skipif(DEVICE != "cuda", reason="FlashInfer needs a GPU")
@pytest.mark.parametrize("M,H", [(128, 4096), (2048, 8192)])
def test_matches_flashinfer(M, H):
    flashinfer = pytest.importorskip("flashinfer")
    dtype = torch.bfloat16
    x = torch.randn(M, H, device=DEVICE, dtype=dtype)
    r = torch.randn(M, H, device=DEVICE, dtype=dtype)
    w = torch.randn(H, device=DEVICE, dtype=dtype)

    torch.testing.assert_close(rmsnorm(x, w), flashinfer.rmsnorm(x, w, 1e-6), atol=2e-2, rtol=2e-2)

    x1, r1, x2, r2 = x.clone(), r.clone(), x.clone(), r.clone()
    fused_add_rmsnorm(x1, r1, w)
    flashinfer.fused_add_rmsnorm(x2, r2, w, 1e-6)
    torch.testing.assert_close(r1, r2, atol=2e-2, rtol=2e-2)
    torch.testing.assert_close(x1, x2, atol=2e-2, rtol=2e-2)
