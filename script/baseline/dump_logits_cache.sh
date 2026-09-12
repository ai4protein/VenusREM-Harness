#!/usr/bin/env bash
# Dump raw backbone logits for ESM-1b / ProtSSN (k,h) / CARP sizes.
#
#   bash script/baseline/dump_logits_cache.sh protssn-k20-h512
#   BENCHMARK=vmh MODELS="carp-600k esm1b" bash script/baseline/dump_logits_cache.sh
#   BENCHMARK=viro bash script/baseline/dump_logits_cache.sh carp_600k
#
# BENCHMARK=pg|vmh|viro (default pg). ProteinGym / VenusMutHub write into
# experiments/full_recipe_wc0/extra_seq_gnn_variants/{pg,vmh}/cache/logits/{key}/
# (same layout as extra_structure_models). Viro90 writes into
# experiments/viro_clinvar/cache/logits/viro90/{key}/ via dump_logits.py.
#
# Cluster:
#   bash script/cluster/submit_logits_cache.sh

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
EXP="${VARIANT_EXP:-$ROOT/experiments/full_recipe_wc0/extra_seq_gnn_variants}"
VRH="${VRH:-vrh}"
PROTSSN_DIR="${PROTSSN_MODEL_DIR:-$ROOT/data/protssn_weights}"
BENCHMARK="${BENCHMARK:-pg}"

DEFAULT_JOBS=(
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
  esm1b-mask
)

if [[ $# -gt 0 ]]; then
  JOBS=("$@")
elif [[ -n "${MODELS:-}" ]]; then
  # shellcheck disable=SC2206
  JOBS=(${MODELS})
else
  JOBS=("${DEFAULT_JOBS[@]}")
fi

to_catalog_key() {
  local job="$1"
  case "$job" in
    esm1b-mask) echo "esm1b_mask" ;;
    esm1b) echo "esm1b_wt" ;;
    carp-600k|carp_600k) echo "carp_600k" ;;
    carp-38m|carp_38m|carp_38M) echo "carp_38m" ;;
    carp-76m|carp_76m|carp_76M) echo "carp_76m" ;;
    protssn-k*|protssn_k*) echo "${job//-/_}" ;;
    *) echo "${job//-/_}" ;;
  esac
}

dump_pg_or_vmh() {
  local job="$1"
  local key
  key="$(to_catalog_key "$job")"
  local model="$job"
  local extra=()
  case "$job" in
    esm1b-mask)
      model="esm1b"
      extra+=(--scoring_strategy mask)
      ;;
    esm1b)
      model="esm1b"
      extra+=(--scoring_strategy wt)
      ;;
    protssn-k*|protssn_k*)
      model="${job//_/-}"
      extra+=(--protssn_model_dir "$PROTSSN_DIR")
      ;;
    carp-*)
      model="$job"
      ;;
  esac

  local base_dir pdb_dir cache_root
  case "$BENCHMARK" in
    pg|proteingym)
      base_dir="${PROTEINGYM_DIR:-$ROOT/data/proteingym_v1}"
      pdb_dir="pdbs_af2_assay_resolved_full"
      cache_root="$EXP/pg/cache/logits"
      ;;
    vmh|muthub|venusmuthub)
      base_dir="${MUTHUB_DIR:-$ROOT/data/VenusMutHub}"
      pdb_dir="pdbs_af2"
      cache_root="$EXP/vmh/cache/logits"
      ;;
    *)
      echo "internal: dump_pg_or_vmh on $BENCHMARK" >&2
      exit 1
      ;;
  esac

  if [[ ! -d "$base_dir" ]]; then
    echo "Missing dataset dir: $base_dir" >&2
    exit 1
  fi

  local cache_dir="$cache_root/$key"
  local scratch="$EXP/$BENCHMARK/scratch/${key}_raw"
  mkdir -p "$cache_dir" "$scratch"
  echo "=== dump ${BENCHMARK} ${key} → ${cache_dir} ==="
  local limit=()
  if [[ -n "${MAX_PROTEINS:-}" && "${MAX_PROTEINS}" != "0" ]]; then
    limit+=(--max_proteins "${MAX_PROTEINS}")
  fi
  "$VRH" \
    --model "$model" \
    --base_dir "$base_dir" \
    --pdb_dir "$pdb_dir" \
    --alpha 0 \
    --scoring_mode log_odds \
    --logits_cache_dir "$cache_dir" \
    --write_logits_cache \
    --reuse_logits_cache \
    --logits_cache_tag "${key}_raw" \
    --logits_cache_stage raw \
    --skip_mutant_scoring \
    --no_print_compare_spearman \
    --out_scores_dir "$scratch" \
    "${extra[@]}" \
    "${limit[@]}"
}

dump_viro() {
  local job="$1"
  local key
  key="$(to_catalog_key "$job")"
  local py="${PY:-python}"
  echo "=== dump viro90 ${key} ==="
  "$py" -u "$ROOT/experiments/viro_clinvar/scripts/dump_logits.py" --dataset viro90 --model "$key"
}

echo "Logits dump"
echo "  benchmark : ${BENCHMARK}"
echo "  jobs      : ${JOBS[*]}"

for job in "${JOBS[@]}"; do
  case "$BENCHMARK" in
    pg|proteingym|vmh|muthub|venusmuthub)
      dump_pg_or_vmh "$job"
      ;;
    viro|virohub|venusvirohub|viro90)
      dump_viro "$job"
      ;;
    *)
      echo "BENCHMARK must be pg, vmh, or viro (got ${BENCHMARK})" >&2
      exit 1
      ;;
  esac
done

echo "=== logits dump complete ==="
