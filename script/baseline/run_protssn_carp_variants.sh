#!/usr/bin/env bash
# ProteinGym: ProtSSN (k,h) singles + CARP small models + ESM-1b (raw and/or full vrh).
#
#   bash script/baseline/run_protssn_carp_variants.sh
#   RECIPE=raw MODELS="carp-600k carp-38m esm1b" bash script/baseline/run_protssn_carp_variants.sh
#   ESM-1b also has a dedicated runner: bash script/baseline/run_esm1b.sh
#
# Needs GPU. This Jupyter node is CPU-only; run on a CUDA box.

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
BASE_DIR="${PROTEINGYM_DIR:-$ROOT/data/proteingym_v1}"
OUT_ROOT="${OUT_ROOT:-$ROOT/result/proteingym_protssn_carp_variants}"
RECIPE="${RECIPE:-both}"
VRH="${VRH:-vrh}"

DEFAULT_MODELS=(
  protssn-k10-h512
  protssn-k10-h768
  protssn-k10-h1280
  protssn-k20-h512
  protssn-k20-h768
  protssn-k20-h1280
  protssn-k30-h512
  protssn-k30-h768
  protssn-k30-h1280
  carp-600k
  carp-38m
  carp-76m
  esm1b
)

if [[ -n "${MODELS:-}" ]]; then
  # shellcheck disable=SC2206
  MODELS=(${MODELS})
else
  MODELS=("${DEFAULT_MODELS[@]}")
fi

if [[ ! -d "$BASE_DIR" ]]; then
  echo "Missing ProteinGym dir: $BASE_DIR" >&2
  echo "Download first: vrh download ProteinGym" >&2
  exit 1
fi

run_one() {
  local model="$1"
  local tag="$2"
  shift 2
  local dest="$OUT_ROOT/${model}_${tag}"
  echo "=== ${model} ${tag} → ${dest} ==="
  "$VRH" --model "$model" --base_dir "$BASE_DIR" --out_scores_dir "$dest" "$@"
}

echo "ProtSSN (k,h) + CARP size variants + ESM-1b"
echo "  base_dir : ${BASE_DIR}"
echo "  out_root : ${OUT_ROOT}"
echo "  recipe   : ${RECIPE}"
echo "  models   : ${MODELS[*]}"

for model in "${MODELS[@]}"; do
  case "$RECIPE" in
    raw)
      run_one "$model" raw --alpha 0
      ;;
    vrh)
      run_one "$model" vrh
      ;;
    both)
      run_one "$model" raw --alpha 0
      run_one "$model" vrh
      ;;
    *)
      echo "RECIPE must be raw, vrh, or both (got ${RECIPE})" >&2
      exit 1
      ;;
  esac
  if [[ "$model" == "esm1b" ]]; then
    case "$RECIPE" in
      raw)
        run_one "$model" mask_raw --alpha 0 --scoring_strategy mask
        ;;
      vrh)
        run_one "$model" mask_vrh --scoring_strategy mask
        ;;
      both)
        run_one "$model" mask_raw --alpha 0 --scoring_strategy mask
        run_one "$model" mask_vrh --scoring_strategy mask
        ;;
    esac
  fi
done

echo "=== ProtSSN / CARP / ESM-1b variant runs complete ==="
