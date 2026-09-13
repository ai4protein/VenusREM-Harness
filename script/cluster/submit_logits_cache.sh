#!/usr/bin/env bash
# Submit ProtSSN (k,h) / CARP sizes / ESM-1b logits dumps (PG + VMH + Viro).
#
#   bash script/cluster/submit_logits_cache.sh --dry-run
#   bash script/cluster/submit_logits_cache.sh
#
# Job keys are {pg|vmh|viro}__{model}. Requires qzcli login.

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
ROOT="${VRH_ROOT:-$ROOT}"
TEMPLATE="${QZCLI_BATCH_CONFIG:-$ROOT/script/cluster/qzcli_logits_cache.json}"
QZCLI="${QZCLI:-qzcli}"
export PATH="${HOME}/.local/bin:${PATH}"
export VRH_ROOT="$ROOT"

if ! command -v "$QZCLI" >/dev/null 2>&1; then
  echo "qzcli not found on PATH" >&2
  exit 1
fi

if ! "$QZCLI" ws >/dev/null 2>&1; then
  echo "qzcli cookie expired or missing. Run: qzcli login" >&2
  "$QZCLI" ws || true
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
"$QZCLI" batch "$WORK" --delay 3 --continue-on-error "${extra[@]}"
