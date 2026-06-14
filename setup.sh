#!/usr/bin/env bash
# setup.sh — prepare a fresh machine to run verify.sh.
# Clones llama.cpp at the pinned commit, applies the Q4_HQQ patch, builds,
# downloads the model, and produces the F16/Q4_0/Q4_HQQ GGUFs.
#
# After this completes, run:  ./verify.sh --no-build
#
# Usage:
#   ./setup.sh                 # full setup (build + models + Task 3 VLM)
#   ./setup.sh --skip-vlm      # skip the Task 3 vision model download
#
# Requirements: git, cmake, python3.12 (brew install python@3.12), hf
#   (pip install -U "huggingface_hub[cli]"). Apple Silicon for the Metal path.

set -uo pipefail

WORKDIR="$HOME/qvac"
REPO="$WORKDIR/llama.cpp"
GGUF="$WORKDIR/gguf"
MODEL_DIR="$HOME/models/llama-3.2-3b-instruct"
PINNED=ebc10770ac5a9331824c53ef0c6adad780904dc3
PATCH="$(cd "$(dirname "$0")" && pwd)/deliverables/q4_hqq_full.patch"

DO_VLM=1
for a in "$@"; do [ "$a" = "--skip-vlm" ] && DO_VLM=0; done

step() { echo ""; echo "==> $1"; }

# ---- 0. sanity: tools present ---------------------------------------------
step "Checking required tools"
for t in git cmake python3.12 hf; do
  command -v "$t" >/dev/null || { echo "MISSING: $t"; echo "  git/cmake: brew install git cmake"; echo "  python3.12: brew install python@3.12"; echo "  hf: pip install -U \"huggingface_hub[cli]\""; exit 1; }
done
[ -f "$PATCH" ] || { echo "Patch not found at $PATCH"; exit 1; }
echo "  all tools present, patch found"

# ---- 1. clone + pin + patch + build ---------------------------------------
step "Cloning llama.cpp and checking out pinned commit"
mkdir -p "$WORKDIR"
if [ ! -d "$REPO" ]; then
  git clone https://github.com/ggml-org/llama.cpp.git "$REPO"
fi
cd "$REPO"
git fetch --all --quiet
git checkout -q "$PINNED"
echo "  at commit: $(git rev-parse --short HEAD)"

step "Applying the Q4_HQQ patch"
if git apply --check "$PATCH" 2>/dev/null; then
  git apply "$PATCH"; echo "  patch applied cleanly"
else
  echo "  ERROR: patch does not apply to $PINNED"; exit 1
fi

step "Building (Metal + CPU) — this takes several minutes"
cmake -B build -DCMAKE_BUILD_TYPE=Release >/dev/null
if cmake --build build --config Release -j 10 >/tmp/setup_build.log 2>&1; then
  echo "  build OK"
else
  echo "  BUILD FAILED — tail of log:"; tail -20 /tmp/setup_build.log; exit 1
fi

# ---- 2. python venv for the converter -------------------------------------
step "Setting up Python venv for HF->GGUF conversion"
python3.12 -m venv "$WORKDIR/venv"
source "$WORKDIR/venv/bin/activate"
pip install -q --upgrade pip
pip install -q -r requirements/requirements-convert_hf_to_gguf.txt
echo "  venv ready"

# ---- 3. download + convert + quantize -------------------------------------
step "Downloading Llama-3.2-3B-Instruct (ungated mirror, ~6 GB)"
hf download unsloth/Llama-3.2-3B-Instruct --local-dir "$MODEL_DIR" >/dev/null
mkdir -p "$GGUF"

step "Converting to F16 GGUF"
python convert_hf_to_gguf.py "$MODEL_DIR" --outfile "$GGUF/llama-3.2-3b-f16.gguf" --outtype f16 >/dev/null
echo "  F16 ready: $(ls -lh "$GGUF/llama-3.2-3b-f16.gguf" | awk '{print $5}')"

step "Quantizing to Q4_0 and Q4_HQQ"
./build/bin/llama-quantize "$GGUF/llama-3.2-3b-f16.gguf" "$GGUF/llama-3.2-3b-q4_0.gguf"  Q4_0   >/dev/null 2>&1
./build/bin/llama-quantize "$GGUF/llama-3.2-3b-f16.gguf" "$GGUF/llama-3.2-3b-q4hqq.gguf" Q4_HQQ >/dev/null 2>&1
echo "  Q4_0:    $(ls -lh "$GGUF/llama-3.2-3b-q4_0.gguf"  | awk '{print $5}')"
echo "  Q4_HQQ:  $(ls -lh "$GGUF/llama-3.2-3b-q4hqq.gguf" | awk '{print $5}')"

# ---- 4. Task 3 vision model (optional) ------------------------------------
if [ "$DO_VLM" = 1 ]; then
  step "Downloading Qwen2-VL-2B for Task 3 (~1.7 GB; use --skip-vlm to skip)"
  mkdir -p "$GGUF/vlm"
  hf download ggml-org/Qwen2-VL-2B-Instruct-GGUF Qwen2-VL-2B-Instruct-Q4_K_M.gguf     --local-dir "$GGUF/vlm" >/dev/null
  hf download ggml-org/Qwen2-VL-2B-Instruct-GGUF mmproj-Qwen2-VL-2B-Instruct-Q8_0.gguf --local-dir "$GGUF/vlm" >/dev/null
  echo "  VLM ready"
fi

echo ""
echo "=============================================="
echo "Setup complete. Now run:  ./verify.sh --no-build"
echo "=============================================="
