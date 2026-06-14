#!/usr/bin/env bash
# Task 2 bonus - perplexity, LIGHTWEIGHT re-run.
# The previous run skewed: Q4_0 went 10x slower than F16 because the fanless
# M4 heat-soaked across back-to-back 5-min passes (REPACK + throttling). Fix:
#   - far fewer chunks (perplexity converges fast; 12 is plenty for a delta)
#   - run ONE model per invocation so each starts from a cool chip
#   - longer cooldown is moot since you control spacing between manual runs
#
# Usage:
#   ./t2_ppl_one.sh f16     ~/qvac/gguf/llama-3.2-3b-f16.gguf
#   (wait a few minutes, let the laptop cool)
#   ./t2_ppl_one.sh q4_0    ~/qvac/gguf/llama-3.2-3b-q4_0.gguf
#   (wait)
#   ./t2_ppl_one.sh q4hqq   ~/qvac/gguf/llama-3.2-3b-q4hqq.gguf
set -uo pipefail

TAG="${1:?usage: $0 <tag> <model.gguf>}"
MODEL="${2:?usage: $0 <tag> <model.gguf>}"

WORKDIR="$HOME/qvac"
REPO="$WORKDIR/llama.cpp"
LOG_DIR="$WORKDIR/logs"
BIN="$REPO/build/bin"
THREADS=4
CHUNKS=12
mkdir -p "$LOG_DIR"
cd "$REPO"

if [ ! -f "wikitext-2-raw/wiki.test.raw" ]; then
  sh scripts/get-wikitext-2.sh
fi
DATA="wikitext-2-raw/wiki.test.raw"

echo "== perplexity: $TAG (chunks=$CHUNKS, CPU, cool start)"
"$BIN/llama-perplexity" \
  -m "$MODEL" -f "$DATA" \
  -t "$THREADS" -ngl 0 -c 512 --chunks "$CHUNKS" \
  2>&1 | tee "$LOG_DIR/t2_ppl_${TAG}.log"

echo ""
grep "Final estimate" "$LOG_DIR/t2_ppl_${TAG}.log" || true
echo "== Done with $TAG. Let the machine cool before the next model."
