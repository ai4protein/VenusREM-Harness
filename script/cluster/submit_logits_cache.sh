#!/usr/bin/env bash
# Submit ProtSSN (k,h) / CARP sizes / ESM-1b logits dumps (PG + VMH + Viro).
#
#   bash script/cluster/submit_logits_cache.sh --dry-run
#   bash script/cluster/submit_logits_cache.sh
#
# Job keys are {pg|vmh|viro}__{model}. Set VRH_CLUSTER_CLI to the batch client.

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
ROOT="${VRH_ROOT:-$ROOT}"
TEMPLATE="${VRH_CLUSTER_BATCH_CONFIG:-$ROOT/script/cluster/logits_cache.json}"
CLI="${VRH_CLUSTER_CLI:-}"
export PATH="${HOME}/.local/bin:${PATH}"
export VRH_ROOT="$ROOT"

if [[ -z "$CLI" ]]; then
  echo "set VRH_CLUSTER_CLI to the cluster submission client" >&2
  exit 1
fi

if ! command -v "$CLI" >/dev/null 2>&1; then
  echo "cluster client not found on PATH: $CLI" >&2
  exit 1
fi

extra=()
if [[ "${1:-}" == "--dry-run" ]]; then
  extra+=(--dry-run)
fi

WORK="$(mktemp)"
trap 'rm -f "$WORK"' EXIT
python - "$TEMPLATE" "$ROOT" "$WORK" <<'PY'
from pathlib import Path
import sys

template, root, dest = sys.argv[1], sys.argv[2], sys.argv[3]
text = Path(template).read_text(encoding="utf-8")
Path(dest).write_text(text.replace("__VRH_ROOT__", root), encoding="utf-8")
PY

echo "Submitting logits-dump batch: $TEMPLATE (VRH_ROOT=$ROOT)"
"$CLI" batch "$WORK" --delay 3 --continue-on-error "${extra[@]}"
