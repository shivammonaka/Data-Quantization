#!/usr/bin/env bash
# Task 3 (v2) - uses the new `hf` CLI (huggingface-cli is deprecated) and the
# verified ggml-org Qwen2-VL-2B filenames. Build is assumed already done by the
# previous run (apply q4_hqq_full.patch + build first); this downloads the
# VLM and exercises --mmproj-backend.
set -uo pipefail

WORKDIR="$HOME/qvac"
REPO="$WORKDIR/llama.cpp"
GGUF="$WORKDIR/gguf"
LOG_DIR="$WORKDIR/logs"
BIN="$REPO/build/bin"
VLM="$GGUF/vlm"
mkdir -p "$LOG_DIR" "$VLM"
cd "$REPO"

REPO_ID="ggml-org/Qwen2-VL-2B-Instruct-GGUF"
MODEL_FILE="Qwen2-VL-2B-Instruct-Q4_K_M.gguf"
MMPROJ_FILE="mmproj-Qwen2-VL-2B-Instruct-Q8_0.gguf"

echo "== downloading VLM with hf"
hf download "$REPO_ID" "$MODEL_FILE"  --local-dir "$VLM"
hf download "$REPO_ID" "$MMPROJ_FILE" --local-dir "$VLM"

MODEL="$VLM/$MODEL_FILE"
MMPROJ="$VLM/$MMPROJ_FILE"
IMG="$REPO/tools/mtmd/test-1.jpeg"
echo "model:  $MODEL  ($(ls -lh "$MODEL"  2>/dev/null | awk '{print $5}'))"
echo "mmproj: $MMPROJ ($(ls -lh "$MMPROJ" 2>/dev/null | awk '{print $5}'))"
[ -f "$MODEL" ] && [ -f "$MMPROJ" ] || { echo "ERROR: download failed"; exit 1; }

run() {
  local tag="$1"; shift
  echo "== [$tag]  extra args: $*"
  "$BIN/llama-mtmd-cli" -m "$MODEL" --mmproj "$MMPROJ" \
    --image "$IMG" -p "Describe this image briefly." \
    -ngl 99 -n 40 --seed 42 "$@" 2>&1 | tee "$LOG_DIR/t3_${tag}.log"
  echo "--- CLIP backend line for [$tag]:"
  grep -iE "clip_ctx|Failed to initialize|mmproj" "$LOG_DIR/t3_${tag}.log" | head -3
  echo; sleep 10
}

run default                              # projector auto (-> GPU like base)
run proj_cpu   --mmproj-backend CPU      # projector forced to CPU, base on GPU
run proj_metal --mmproj-backend Metal   # alias, not the device name: intentionally demonstrates the documented graceful fallback (see README)
# Note: --mmproj-backend Metal (alias, not the device name) intentionally
# demonstrates the graceful fallback documented in README.md.

echo "== Compare the clip_ctx backend lines across the three logs:"
grep -HiE "clip_ctx|Failed to initialize" "$LOG_DIR"/t3_default.log "$LOG_DIR"/t3_proj_cpu.log "$LOG_DIR"/t3_proj_metal.log || echo "(default/CPU runs print no warning; proj_metal shows the documented fallback)"
echo "== Done. Send t3_default.log, t3_proj_cpu.log, t3_proj_metal.log"
