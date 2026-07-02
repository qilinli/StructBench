#!/bin/bash
# Build the StructBench training environment on a DUG login node.
# Run ONCE from the repo root (needs internet on the login node for pip).
# Produces .venv, which deploy/dug/train_taylor.slurm activates on the GPU node.
#
# Mirrors the validated Windows env: torch 2.12.1 + CUDA 12.6 (Linux wheel),
# torch-geometric, numpy, h5py. The model needs only torch_geometric.nn.
# MessagePassing -- NO compiled torch-cluster/pyg-lib (native radius_graph,
# ADR-0020) -- so this install has no version-matching pain.
set -euo pipefail

# uv is the project's package manager; install it if the login node lacks it.
# The installer drops uv in ~/.local/bin, which is NOT on this shell's PATH
# yet -- export it, or every following uv call dies under `set -e`.
if ! command -v uv >/dev/null; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi

uv venv .venv --python 3.11
# Linux CUDA torch, same version validated locally (cu126 ships a Linux wheel):
uv pip install --python .venv "torch==2.12.1" --index-url https://download.pytorch.org/whl/cu126
# Everything else from pyproject (numpy, h5py, torch-geometric); torch already
# satisfied, so it is not reinstalled:
uv pip install --python .venv -e .

.venv/bin/python - <<'PY'
import torch
print("torch", torch.__version__, "| cuda build", torch.version.cuda,
      "| available", torch.cuda.is_available())
PY
echo
echo "Env ready. 'available False' HERE is expected -- login nodes usually have"
echo "no GPU. It will be True inside the SLURM GPU job (train_taylor.slurm)."
