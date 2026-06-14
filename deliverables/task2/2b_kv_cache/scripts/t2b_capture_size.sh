#!/usr/bin/env bash
# Task 2b - capture the ACTUAL KV cache size line (src/llama-kv-cache.cpp:313).
# llama-cli mutes llama INFO logs by default; -lv 4 (DEBUG) raises the backend
# logger so the "size = ... K (...) MiB, V (...) MiB" line prints. We only need
# the init line, so generate just 1 token. Three cache configs.
set -uo pipefail

WORKDIR="$HOME/qvac"
REPO="$WORKDIR/llama.cpp"
GGUF="$WORKDIR/gguf"
LOG_DIR="$WORKDIR/logs"
BIN="$REPO/build/bin"
MODEL="$GGUF/llama-3.2-3b-q4_0.gguf"
mkdir -p "$LOG_DIR"
cd "$REPO"

cap() {
  local tag="$1"; shift
  "$BIN/llama-cli" -m "$MODEL" -p "hi" -n 1 -t 4 -ngl 0 -c 4096 \
    -st --simple-io -lv 4 "$@" < /dev/null > "$LOG_DIR/t2b_size_${tag}.log" 2>&1
  echo "=== [$tag] ==="
  grep -iE "size =.*K \(|KV self|kv_cache" "$LOG_DIR/t2b_size_${tag}.log" | head -3 \
    || echo "  (size line not found - paste full log)"
}

cap f16   --cache-type-k f16    --cache-type-v f16
cap q4_0  --cache-type-k q4_0   --cache-type-v q4_0   -fa on
cap q4hqq --cache-type-k q4_hqq --cache-type-v q4_hqq -fa on

echo "== Done. If size lines printed above, just paste this terminal output."
