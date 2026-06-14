#!/usr/bin/env bash
# Task 2 - NEON kernel: reset to pinned, apply combined patch, rebuild,
# re-run smoke test + llama-bench to measure the NEON speedup vs the scalar
# 13.5 t/s baseline. The quantized model is unchanged (format identical), so
# no re-quantize needed unless the file is missing.
set -uo pipefail

WORKDIR="$HOME/qvac"
REPO="$WORKDIR/llama.cpp"
GGUF="$WORKDIR/gguf"
LOG_DIR="$WORKDIR/logs"
PATCH="${1:-$HOME/Downloads/qvac_submission/deliverables/q4_hqq_full.patch}"
mkdir -p "$LOG_DIR"
cd "$REPO"

echo "== reset to pinned commit"
git reset --hard ebc10770ac5a9331824c53ef0c6adad780904dc3

if git apply --check "$PATCH"; then
  git apply "$PATCH"; echo "== patch applied"
else
  echo "ERROR: patch did not apply cleanly"; exit 1
fi

cmake --build build --config Release -j 10 2>&1 | tee "$LOG_DIR/t2_build_neon.log"
echo "== build exit: ${PIPESTATUS[0]}"

# Re-quantize only if model missing (format unchanged by NEON)
if [ ! -f "$GGUF/llama-3.2-3b-q4hqq.gguf" ]; then
  ./build/bin/llama-quantize "$GGUF/llama-3.2-3b-f16.gguf" \
    "$GGUF/llama-3.2-3b-q4hqq.gguf" Q4_HQQ 2>&1 | tee "$LOG_DIR/t2_quantize_hqq.log"
fi

echo "== correctness re-check (must still be coherent):"
./build/bin/llama-cli -m "$GGUF/llama-3.2-3b-q4hqq.gguf" \
  -p "What is bitcoin?" -n 60 -t 4 -ngl 0 -c 4096 \
  --seed 42 -st --simple-io -lv 3 < /dev/null 2>&1 | tee "$LOG_DIR/t2_smoke_neon.log"

echo "== speed: Q4_HQQ (NEON) vs Q4_0, CPU, tg128"
./build/bin/llama-bench -m "$GGUF/llama-3.2-3b-q4hqq.gguf" -t 4 -ngl 0 -p 0 -n 128 2>&1 | tee "$LOG_DIR/t2_bench_hqq_cpu.log"
./build/bin/llama-bench -m "$GGUF/llama-3.2-3b-q4_0.gguf"  -t 4 -ngl 0 -p 0 -n 128 2>&1 | tee "$LOG_DIR/t2_bench_q4_0_cpu.log"

echo "== Done. Send t2_smoke_neon.log, t2_bench_hqq_cpu.log, t2_bench_q4_0_cpu.log"
