#!/usr/bin/env bash
# Install S3F / TorchDrug stack. Official torchdrug wheels need Python <3.11;
# this script also supports 3.11/3.12 via --ignore-requires-python.
set -euo pipefail

python - <<'PY'
import sys
print(f"Python {sys.version}")
if sys.version_info >= (3, 11):
    print("Note: torchdrug has no official wheels for Python >=3.11; using source install.")
PY

pip install --ignore-requires-python --no-deps \
  "git+https://github.com/DeepGraphLearning/torchdrug.git"
pip install \
  'rdkit==2023.9.6' \
  'numpy<2' \
  lmdb \
  decorator \
  ninja \
  robust-laplacian \
  biopython \
  scikit-learn \
  torch-geometric

# Optional: only for on-the-fly surface generation from PDB (not needed for precomputed .pkl)
# pip install pykeops

python - <<'PY'
import torchdrug
print("torchdrug", torchdrug.__version__, "OK")
print("S3F inference deps ready (precomputed surfaces; no pykeops required).")
PY
