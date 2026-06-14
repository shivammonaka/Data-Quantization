#!/usr/bin/env bash
# Task 1 - Step 2: quantize F16 GGUF -> Q4_0
set -euo pipefail

WORKDIR="$HOME/qvac"
GGUF="$WORKDIR/gguf"
LOG_DIR="$WORKDIR/logs"
cd "$WORKDIR/llama.cpp"

./build/bin/llama-quantize \
  "$GGUF/llama-3.2-3b-f16.gguf" \
  "$GGUF/llama-3.2-3b-q4_0.gguf" \
  Q4_0 \
  2>&1 | tee "$LOG_DIR/02_quantize.log"

echo "== Sizes (for the comparison table):"
ls -lh "$GGUF"/llama-3.2-3b-*.gguf
