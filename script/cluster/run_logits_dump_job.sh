#!/usr/bin/env bash
# Cluster entrypoint. Args: [benchmark] <job-key>
#   run_logits_dump_job.sh pg carp-600k
#   run_logits_dump_job.sh vmh__protssn-k10-h512
set -euo pipefail

ROOT="${VRH_ROOT:-/inspire/hdd/global_user/USER/workspace/research/VenusREM-Harness}"
CONDA_ROOT="${CONDA_ROOT:-/inspire/hdd/global_user/USER/miniconda3}"

if [[ $# -eq 1 && "$1" == *"__"* ]]; then
  BENCHMARK="${1%%__*}"
  JOB="${1#*__}"
elif [[ $# -ge 2 ]]; then
  BENCHMARK="$1"
  JOB="$2"
else
  echo "usage: run_logits_dump_job.sh <pg|vmh|viro> <job-key>" >&2
  echo "   or: run_logits_dump_job.sh <benchmark>__<job-key>" >&2
  exit 1
fi

# shellcheck disable=SC1091
source "${CONDA_ROOT}/etc/profile.d/conda.sh"
conda activate vrh

export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_HOME="${HF_HOME:-/inspire/hdd/global_user/USER/.cache/huggingface}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME}"
export VRH="${VRH:-vrh}"
export BENCHMARK
export PY="${CONDA_ROOT}/envs/vrh/bin/python"

cd "$ROOT"
nvidia-smi || true
which vrh
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), 'gpu', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none')"
bash "$ROOT/script/baseline/dump_logits_cache.sh" "$JOB"
