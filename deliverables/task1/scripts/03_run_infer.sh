#!/usr/bin/env bash
# Task 1 - Step 3: run both prompts on CPU and Metal, capture latency/speed/RSS.
# Notes: --simple-io is required (llama-cli writes to /dev/tty otherwise);
# -c 4096 fixes context for meaningful RSS; -lv 3 surfaces perf lines.
set -euo pipefail

WORKDIR="$HOME/qvac"
MODEL="$WORKDIR/gguf/llama-3.2-3b-q4_0.gguf"
LOG_DIR="$WORKDIR/logs"
BIN="$WORKDIR/llama.cpp/build/bin"
THREADS=4
NPREDICT=200
CTX=4096
COOLDOWN=60
mkdir -p "$LOG_DIR"

PROMPTS=(
  "What is bitcoin?"
  "Write a Python function to reverse a list."
)

run_with_peak_rss() {
  local logfile="$1"; shift
  "$@" < /dev/null > "$logfile" 2>&1 &
  local pid=$!
  local peak=0 rss=0
  while kill -0 "$pid" 2>/dev/null; do
    rss=$(ps -o rss= -p "$pid" 2>/dev/null | tr -d ' ')
    if [ -n "${rss:-}" ] && [ "$rss" -gt "$peak" ]; then
      peak="$rss"
    fi
    sleep 0.2
  done
  wait "$pid" && rc=0 || rc=$?
  echo "EXIT_CODE: $rc"        >> "$logfile"
  echo "PEAK_RSS_MB: $((peak / 1024))" >> "$logfile"
}

for backend in cpu metal; do
  [ "$backend" = "cpu" ] && NGL=0 || NGL=99
  for i in "${!PROMPTS[@]}"; do
    p="${PROMPTS[$i]}"
    names=("bitcoin" "reverse_list"); log="$LOG_DIR/infer_${backend}_${names[$i]}.log"
    echo "=== [$backend] prompt $((i+1)): $p"
    run_with_peak_rss "$log" \
      "$BIN/llama-cli" -m "$MODEL" \
        -p "$p" -n "$NPREDICT" -t "$THREADS" -ngl "$NGL" -c "$CTX" \
        --seed 42 -st --simple-io -lv 3
    echo "--- key lines from $log:"
    grep -E "prompt eval time|eval time|Prompt:|EXIT_CODE|PEAK_RSS_MB" "$log" || true
    # quick visibility check: did we actually capture the answer this time?
    echo "--- log size: $(wc -l < "$log") lines (should be >50, not 2)"
    echo "--- cooling down ${COOLDOWN}s..."
    sleep "$COOLDOWN"
  done
done

echo "== Done. Logs: infer_{cpu,metal}_{bitcoin,reverse_list}.log in $LOG_DIR"
