#!/usr/bin/env bash
# Task 1 - Step 1: convert HF LLaMA 3.2 3B Instruct -> F16 GGUF
# Prereq: model downloaded to ~/models/llama-3.2-3b-instruct (hf download)
set -euo pipefail

WORKDIR="$HOME/qvac"
MODEL_DIR="$HOME/models/llama-3.2-3b-instruct"
OUT_DIR="$WORKDIR/gguf"
LOG_DIR="$WORKDIR/logs"
mkdir -p "$OUT_DIR" "$LOG_DIR"

source "$WORKDIR/venv/bin/activate"
cd "$WORKDIR/llama.cpp"

# Sanity: gated download sometimes leaves missing shards
ls "$MODEL_DIR"/*.safetensors >/dev/null || { echo "ERROR: no safetensors in $MODEL_DIR"; exit 1; }

python convert_hf_to_gguf.py "$MODEL_DIR" \
  --outfile "$OUT_DIR/llama-3.2-3b-f16.gguf" \
  --outtype f16 \
  2>&1 | tee "$LOG_DIR/01_convert.log"

echo "== Done:"
ls -lh "$OUT_DIR/llama-3.2-3b-f16.gguf"
