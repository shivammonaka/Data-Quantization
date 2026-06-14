#!/usr/bin/env bash
# Task 2b - Q4_HQQ as KV cache type. Tests:
#   (1) K-cache = q4_hqq (no flash attn needed)
#   (2) K+V cache = q4_hqq (requires -fa; quantized V needs flash attention)
#   (3) baseline F16 cache for memory comparison
# Uses the WEIGHTS model in Q4_0 (any model works; we're testing the CACHE type,
# which is independent of the weight quant). Captures memory + coherence.
set -uo pipefail

WORKDIR="$HOME/qvac"
REPO="$WORKDIR/llama.cpp"
GGUF="$WORKDIR/gguf"
LOG_DIR="$WORKDIR/logs"
BIN="$REPO/build/bin"
PATCH="${1:-$HOME/Downloads/qvac_submission/deliverables/q4_hqq_full.patch}"
MODEL="$GGUF/llama-3.2-3b-q4_0.gguf"   # weights; cache type is what we vary
THREADS=4
mkdir -p "$LOG_DIR"
cd "$REPO"

echo "== reset + apply full patch"
git reset --hard ebc10770ac5a9331824c53ef0c6adad780904dc3
git apply --check "$PATCH" && git apply "$PATCH" || { echo "patch failed"; exit 1; }
cmake --build build --config Release -j 10 2>&1 | tee "$LOG_DIR/t2b_build.log"

run() {
  local tag="$1"; shift
  echo "== [$tag]  args: $*"
  "$BIN/llama-cli" -m "$MODEL" \
    -p "What is bitcoin?" -n 80 -t "$THREADS" -ngl 0 -c 4096 \
    --seed 42 -st --simple-io -lv 3 "$@" \
    < /dev/null 2>&1 | tee "$LOG_DIR/t2b_${tag}.log"
  echo "--- KV cache size line for [$tag]:"
  grep -iE "KV self|kv cache|cache_k|cache_v|cache type|q4_hqq" "$LOG_DIR/t2b_${tag}.log" | head -6
  echo; sleep 15
}

# 1. Baseline: default F16 cache
run f16cache

# 2. K cache = q4_hqq (no flash attn)
run k_q4hqq --cache-type-k q4_hqq

# 3. K+V cache = q4_hqq (requires flash attention)
run kv_q4hqq --cache-type-k q4_hqq --cache-type-v q4_hqq -fa on

echo "== Done. Send t2b_f16cache.log, t2b_k_q4hqq.log, t2b_kv_q4hqq.log"
