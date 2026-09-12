#!/usr/bin/env bash
# ProteinGym: ESM-1b (wt + mask) × raw / full vrh.
#
#   bash script/baseline/run_esm1b.sh
#   RECIPE=raw STRATEGIES=wt bash script/baseline/run_esm1b.sh
#
# Needs GPU. This Jupyter node is CPU-only; run on a CUDA box.

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
BASE_DIR="${PROTEINGYM_DIR:-$ROOT/data/proteingym_v1}"
OUT_ROOT="${OUT_ROOT:-$ROOT/result/proteingym_esm1b}"
RECIPE="${RECIPE:-both}"
STRATEGIES="${STRATEGIES:-wt mask}"
VRH="${VRH:-vrh}"
MODEL="${MODEL:-esm1b}"

if [[ ! -d "$BASE_DIR" ]]; then
  echo "Missing ProteinGym dir: $BASE_DIR" >&2
  echo "Download first: vrh download ProteinGym" >&2
  exit 1
fi

# shellcheck disable=SC2206
STRATEGIES=(${STRATEGIES})

run_one() {
  local strategy="$1"
  local tag="$2"
  shift 2
  local dest="$OUT_ROOT/${MODEL}_${strategy}_${tag}"
  echo "=== ${MODEL} ${strategy} ${tag} → ${dest} ==="
  "$VRH" --model "$MODEL" --scoring_strategy "$strategy" --base_dir "$BASE_DIR" --out_scores_dir "$dest" "$@"
}

echo "ESM-1b ProteinGym"
echo "  base_dir   : ${BASE_DIR}"
echo "  out_root   : ${OUT_ROOT}"
echo "  recipe     : ${RECIPE}"
echo "  strategies : ${STRATEGIES[*]}"

for strategy in "${STRATEGIES[@]}"; do
  case "$RECIPE" in
    raw)
      run_one "$strategy" raw --alpha 0
      ;;
    vrh)
      run_one "$strategy" vrh
      ;;
    both)
      run_one "$strategy" raw --alpha 0
      run_one "$strategy" vrh
      ;;
    *)
      echo "RECIPE must be raw, vrh, or both (got ${RECIPE})" >&2
      exit 1
      ;;
  esac
done

echo "=== ESM-1b runs complete ==="
