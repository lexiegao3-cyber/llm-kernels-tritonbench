"""Launch each RMSNorm backend a couple of times so ncu can capture them."""
import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from llm_kernels import rmsnorm  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument("--tokens", type=int, default=4096)
p.add_argument("--hidden", type=int, default=4096)
a = p.parse_args()

x = torch.randn(a.tokens, a.hidden, device="cuda", dtype=torch.bfloat16)
w = torch.randn(a.hidden, device="cuda", dtype=torch.bfloat16)
for _ in range(2):
    rmsnorm(x, w)
try:
    import flashinfer
    for _ in range(2):
        flashinfer.rmsnorm(x, w, 1e-6)
except ImportError:
    print("flashinfer not installed; profiling only our kernel")
torch.cuda.synchronize()
