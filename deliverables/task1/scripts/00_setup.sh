#!/usr/bin/env bash
# Task 1 - Step 0: clone llama.cpp (pinned), build (Metal + CPU), set up venv
# Target: MacBook Air M4, 16 GB, macOS 26.x
set -euo pipefail

WORKDIR="$HOME/qvac"
PINNED_COMMIT="ebc10770ac5a9331824c53ef0c6adad780904dc3"  # master, 2026-06-12

mkdir -p "$WORKDIR" && cd "$WORKDIR"

# --- 1. Clone and pin ---------------------------------------------------
if [ ! -d llama.cpp ]; then
  git clone https://github.com/ggml-org/llama.cpp.git
fi
cd llama.cpp
git fetch --all --quiet
git checkout "$PINNED_COMMIT"
echo "== llama.cpp pinned at: $(git log -1 --format='%H %cd' --date=short)"

# --- 2. Build (Metal is enabled by default on Apple Silicon) ------------
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release -j 10
echo "== Build OK: $(./build/bin/llama-cli --version 2>&1 | head -1)"

# --- 3. Python venv for the conversion script ---------------------------
# Requires: brew install python@3.12   (system 3.9 is too old for torch 2.11)
PY=python3.12
command -v $PY >/dev/null || { echo "ERROR: python3.12 not found. Run: brew install python@3.12"; exit 1; }

$PY -m venv "$WORKDIR/venv"
source "$WORKDIR/venv/bin/activate"
pip install --upgrade pip
pip install -r requirements/requirements-convert_hf_to_gguf.txt

echo "== Setup complete. Environment info for the report:"
echo "---------------------------------------------------"
sw_vers
system_profiler SPHardwareDataType | grep -E "Chip|Total Number of Cores|Memory"
clang --version | head -1
cmake --version | head -1
python --version
git -C "$WORKDIR/llama.cpp" log -1 --format="llama.cpp commit: %H"
