#!/usr/bin/env bash
# Task 2c - Q4_HQQ Metal (GPU) kernel. Build, then run the Q4_HQQ model with
# -ngl 99 to offload to the M4 GPU. Compares CPU vs GPU output + speed.
# NOTE: the Metal shader can only be compiled/tested on-device (Apple GPU);
# expect possible build iterations. Paste any Metal compile errors verbatim.
set -uo pipefail

WORKDIR="$HOME/qvac"
REPO="$WORKDIR/llama.cpp"
GGUF="$WORKDIR/gguf"
LOG_DIR="$WORKDIR/logs"
BIN="$REPO/build/bin"
PATCH="${1:-$HOME/Downloads/qvac_submission/deliverables/q4_hqq_full.patch}"
MODEL="$GGUF/llama-3.2-3b-q4hqq.gguf"
mkdir -p "$LOG_DIR"
cd "$REPO"

echo "== reset + apply full patch"
git reset --hard ebc10770ac5a9331824c53ef0c6adad780904dc3
git apply --check "$PATCH" && git apply "$PATCH" || { echo "patch failed to apply"; exit 1; }

echo "== build (this recompiles the Metal shader library)"
cmake --build build --config Release -j 10 2>&1 | tee "$LOG_DIR/t2c_build.log"
BUILD_RC=${PIPESTATUS[0]}
echo "== build exit: $BUILD_RC"
[ "$BUILD_RC" -ne 0 ] && { echo "BUILD FAILED - paste tail of t2c_build.log"; tail -30 "$LOG_DIR/t2c_build.log"; exit 1; }

# GPU run: offload all layers to Metal
echo "== GPU (-ngl 99) correctness + speed:"
"$BIN/llama-cli" -m "$MODEL" \
  -p "What is bitcoin?" -n 80 -ngl 99 -c 4096 \
  --seed 42 -st --simple-io -lv 3 < /dev/null 2>&1 | tee "$LOG_DIR/t2c_gpu_smoke.log"

# GPU throughput vs CPU
echo "== llama-bench Q4_HQQ on Metal:"
"$BIN/llama-bench" -m "$MODEL" -ngl 99 -p 0 -n 128 2>&1 | tee "$LOG_DIR/t2c_bench_gpu.log"

echo "== Done. Send t2c_gpu_smoke.log and t2c_bench_gpu.log (and t2c_build.log if it failed)."
