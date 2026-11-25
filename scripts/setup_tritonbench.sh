#!/usr/bin/env bash
# Clone tritonbench next to this repo, apply our patches, install everything.
# Run on a Linux machine with an NVIDIA GPU and a recent CUDA driver.
set -euo pipefail
HERE="$(cd "$(dirname "$0")/.." && pwd)"
TB_DIR="${TB_DIR:-$HERE/../tritonbench}"

if [ ! -d "$TB_DIR" ]; then
  git clone https://github.com/meta-pytorch/tritonbench.git "$TB_DIR"
fi
cd "$TB_DIR"
git submodule update --init --recursive

# Installs PyTorch nightly + bundled Triton (see tritonbench README).
python install.py

# Our patches: 0001 = FlashInfer backends (upstreamable), 0002 = our Triton kernels.
for p in "$HERE"/tritonbench_patches/*.patch; do
  if git apply --check "$p" 2>/dev/null; then
    git apply "$p" && echo "applied $(basename "$p")"
  else
    echo "skip $(basename "$p") (already applied or conflicts)"
  fi
done

pip install flashinfer-python liger-kernel
pip install -e "$HERE"
pip install -e .
echo "done: tritonbench at $TB_DIR"
