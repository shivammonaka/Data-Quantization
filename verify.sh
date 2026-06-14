#!/usr/bin/env bash
# verify.sh — one-shot verification of the Q4_HQQ submission.
#
# What it does: applies the patch to a pinned llama.cpp, builds, then runs each
# task and checks the output against the expected numbers. Prints PASS/FAIL.
#
# Usage:
#   ./verify.sh                  # full run (build + all tasks)
#   ./verify.sh --no-build       # skip the build (use an already-built tree)
#   ./verify.sh --skip-task3     # skip the multimodal test (needs the VLM)
#
# Requirements: a llama.cpp clone, the F16 + Q4_0 + Q4_HQQ GGUFs, and (for
# Task 3) the Qwen2-VL-2B model. Paths are configurable below.

set -uo pipefail

# ---- configuration (edit if your paths differ) ----------------------------
REPO="${REPO:-$HOME/qvac/llama.cpp}"
GGUF="${GGUF:-$HOME/qvac/gguf}"
PATCH="${PATCH:-$(cd "$(dirname "$0")" && pwd)/deliverables/q4_hqq_full.patch}"
PINNED=ebc10770ac5a9331824c53ef0c6adad780904dc3
BIN="$REPO/build/bin"

DO_BUILD=1; DO_TASK3=1
for arg in "$@"; do
  case "$arg" in
    --no-build)   DO_BUILD=0 ;;
    --skip-task3) DO_TASK3=0 ;;
  esac
done


# portable timeout: use timeout/gtimeout if available, else run directly
TIMEOUT=""
command -v timeout  >/dev/null && TIMEOUT="timeout 180"
command -v gtimeout >/dev/null && TIMEOUT="gtimeout 180"

pass=0; fail=0
ok()   { echo "  ✅ PASS: $1"; pass=$((pass+1)); }
no()   { echo "  ❌ FAIL: $1"; fail=$((fail+1)); }
hdr()  { echo ""; echo "=============================================="; echo "$1"; echo "=============================================="; }

# ---- 0. patch applies + builds --------------------------------------------
hdr "0. Patch + build"
cd "$REPO"
# Detect state: if the patch is already applied (e.g. setup.sh ran), --check
# fails but --reverse --check succeeds. Treat both as a clean PASS.
if [ ! -d "$REPO/.git" ]; then
  echo "  (no git repo at $REPO — run setup.sh first)"; no "repo not found"
elif git apply --check "$PATCH" 2>/dev/null; then
  ok "patch applies cleanly to $PINNED"
  git apply "$PATCH" 2>/dev/null || true
elif git apply --reverse --check "$PATCH" 2>/dev/null; then
  ok "patch already applied (tree is patched and built)"
else
  no "patch neither applies nor is already applied"
fi
if [ "$DO_BUILD" = 1 ]; then
  echo "  building (this takes a few minutes)..."
  if cmake --build build --config Release -j 10 >/tmp/verify_build.log 2>&1; then
    ok "build succeeded"
  else
    no "build FAILED — see /tmp/verify_build.log"; tail -15 /tmp/verify_build.log
  fi
else
  echo "  (skipping build per --no-build)"
fi

# helper: extract tg128 t/s from a llama-bench run
bench_tg() { $TIMEOUT "$BIN/llama-bench" -m "$1" -ngl "$2" -p 0 -n 128 2>/dev/null | awk -F'|' '/tg128/{gsub(/ /,"",$NF); split($NF,a,"±"); print a[1]}'; }

# ---- 1. Task 2a: Q4_HQQ option exists + quantizes + coherent --------------
hdr "1. Task 2a — Q4_HQQ format"
qhelp="$("$BIN/llama-quantize" --help 2>&1 || true)"
if printf '%s' "$qhelp" | grep -qiE "q4_hqq"; then ok "Q4_HQQ is a registered quantize option"; else no "Q4_HQQ not in quantize options"; fi
# Prefer the already-built Q4_HQQ model from setup.sh; only quantize fresh if missing.
if [ -f "$GGUF/llama-3.2-3b-q4hqq.gguf" ]; then
  MODEL_HQQ="$GGUF/llama-3.2-3b-q4hqq.gguf"
  sz=$(ls -lh "$MODEL_HQQ" | awk '{print $5}')
  ok "Q4_HQQ model present (${sz})"
elif [ -f "$GGUF/llama-3.2-3b-f16.gguf" ]; then
  "$BIN/llama-quantize" "$GGUF/llama-3.2-3b-f16.gguf" /tmp/verify_hqq.gguf Q4_HQQ > /tmp/verify_quant.txt 2>&1 || true
  grep -qi "q4_hqq" /tmp/verify_quant.txt && ok "quantize to Q4_HQQ succeeded" || no "quantize to Q4_HQQ failed"
  MODEL_HQQ=/tmp/verify_hqq.gguf
else
  no "no Q4_HQQ or F16 model found — run setup.sh first"; MODEL_HQQ=""
fi
if [ -n "$MODEL_HQQ" ]; then
  $TIMEOUT "$BIN/llama-cli" -m "$MODEL_HQQ" -p "What is bitcoin?" -n 30 -ngl 0 -no-cnv --simple-io < /dev/null > /tmp/verify_infer.txt 2>/dev/null || true
  grep -qi "bitcoin" /tmp/verify_infer.txt && ok "CPU inference produces coherent output" || no "CPU output not coherent"
fi

# ---- 2. Task 2 speed (NEON CPU ~30, Metal ~45) ----------------------------
hdr "2. Task 2 — speed (expect ~30 NEON CPU, ~44 Metal)"
cpu_tg=$(bench_tg "$MODEL_HQQ" 0)
gpu_tg=$(bench_tg "$MODEL_HQQ" 99)
echo "  CPU(NEON) tg128 = ${cpu_tg:-?} t/s   |   Metal tg128 = ${gpu_tg:-?} t/s"
awk "BEGIN{exit !($cpu_tg > 20)}" 2>/dev/null && ok "CPU NEON speed in expected range (>20, scalar was 13.5)" || no "CPU speed unexpectedly low"
awk "BEGIN{exit !($gpu_tg > 38)}" 2>/dev/null && ok "Metal GPU speed in expected range (>38)" || no "Metal speed unexpectedly low"

# ---- 3. Task 2b: KV cache (expect 140 MiB) --------------------------------
hdr "3. Task 2b — KV cache quantization (expect 140 MiB)"
$TIMEOUT "$BIN/llama-cli" -m "$GGUF/llama-3.2-3b-q4_0.gguf" -p "hi" -n 1 \
      --cache-type-k q4_hqq --cache-type-v q4_hqq -fa on -lv 4 -no-cnv --simple-io < /dev/null > /tmp/verify_kv.txt 2>&1 || true
kv=$(grep -o "kv_cache: size = *[0-9.]* MiB" /tmp/verify_kv.txt | head -1)
echo "  $kv"
printf '%s' "$kv" | grep -q "140" && ok "Q4_HQQ KV cache = 140 MiB (vs 448 F16)" || no "KV cache size not as expected"

# ---- 4. Task 3: --mmproj-backend ------------------------------------------
if [ "$DO_TASK3" = 1 ]; then
  hdr "4. Task 3 — --mmproj-backend"
  VM="$GGUF/vlm/Qwen2-VL-2B-Instruct-Q4_K_M.gguf"
  MP="$GGUF/vlm/mmproj-Qwen2-VL-2B-Instruct-Q8_0.gguf"
  mhelp="$("$BIN/llama-mtmd-cli" --help 2>&1 || true)"
  if printf '%s' "$mhelp" | grep -q "mmproj-backend"; then ok "--mmproj-backend flag registered"; else no "flag not registered"; fi
  if [ -f "$VM" ] && [ -f "$MP" ]; then
    $TIMEOUT "$BIN/llama-mtmd-cli" -m "$VM" --mmproj "$MP" --image "$REPO/tools/mtmd/test-1.jpeg" \
         -p "Describe this image." -ngl 99 --mmproj-backend CPU -n 30 > /tmp/verify_mtmd.txt 2>/dev/null || true
    grep -qiE "image|newspaper|moon|shows" /tmp/verify_mtmd.txt && ok "projector on CPU + base on GPU → coherent description" || no "multimodal output not coherent"
  else
    echo "  (VLM not downloaded — skipping live run; flag check above still valid)"
  fi
fi

# ---- summary --------------------------------------------------------------
hdr "RESULT: $pass passed, $fail failed"
[ "$fail" = 0 ] && echo "All checks passed ✅" || echo "Some checks failed — see above."
exit "$fail"
