#!/usr/bin/env bash
# Submit ProtSSN (k,h) / CARP sizes / ESM-1b logits dumps (PG + VMH + Viro).
#
#   bash script/cluster/submit_logits_cache.sh --dry-run
#   bash script/cluster/submit_logits_cache.sh
#
# Job keys are {pg|vmh|viro}__{model}. Requires qzcli login.

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
CONFIG="${QZCLI_BATCH_CONFIG:-$ROOT/script/cluster/qzcli_logits_cache.json}"
QZCLI="${QZCLI:-qzcli}"
export PATH="${HOME}/.local/bin:${PATH}"

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

echo "Submitting logits-dump batch: $CONFIG"
"$QZCLI" batch "$CONFIG" --delay 3 --continue-on-error "${extra[@]}"
