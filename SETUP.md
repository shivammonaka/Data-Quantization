# Setup Instructions

## Environment

| | |
|---|---|
| Hardware | MacBook Air, Apple M4 (4 P-cores + 6 E-cores, 10-core GPU), 16 GB unified memory |
| OS | macOS 26.5.1 (build 25F80) |
| Compiler | Apple clang 21.0.0 |
| CMake | 4.3.3 |
| Python | 3.12 (venv; system 3.9 is too old for the converter's torch) |
| llama.cpp | commit `ebc10770ac5a9331824c53ef0c6adad780904dc3` (build b9614) |
| Backends | Metal (default on Apple Silicon) + CPU with NEON/DOTPROD |

## 0. Prerequisites

```bash
brew install cmake python@3.12
pip3 install -U "huggingface_hub[cli]"   # provides the `hf` command
```

## 1. Clone and pin llama.cpp

```bash
git clone https://github.com/ggml-org/llama.cpp.git
cd llama.cpp
git checkout ebc10770ac5a9331824c53ef0c6adad780904dc3
```

## 2. Apply the implementation patch (Task 2 + Task 3)

```bash
git apply /path/to/q4_hqq_full.patch
```

## 3. Build (Metal + CPU)

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release -j 10
```

## 4. Get the model (Task 1)

Llama-3.2-3B-Instruct via the ungated mirror (identical weights, no license gate):

```bash
hf download unsloth/Llama-3.2-3B-Instruct --local-dir ~/models/llama-3.2-3b-instruct
python convert_hf_to_gguf.py ~/models/llama-3.2-3b-instruct \
  --outfile llama-3.2-3b-f16.gguf --outtype f16
```

## 5. Reproduce each task

```bash
# Task 1: quantize to Q4_0 and run
./build/bin/llama-quantize llama-3.2-3b-f16.gguf llama-3.2-3b-q4_0.gguf Q4_0
./build/bin/llama-cli -m llama-3.2-3b-q4_0.gguf -p "What is bitcoin?" -n 100 --simple-io

# Task 2: quantize to Q4_HQQ and run (CPU)
./build/bin/llama-quantize llama-3.2-3b-f16.gguf llama-3.2-3b-q4hqq.gguf Q4_HQQ
./build/bin/llama-cli -m llama-3.2-3b-q4hqq.gguf -p "What is bitcoin?" -n 100 -ngl 0 --simple-io

# Task 2c: run on GPU
./build/bin/llama-cli -m llama-3.2-3b-q4hqq.gguf -p "What is bitcoin?" -n 100 -ngl 99 --simple-io

# Task 2 speed comparison
./build/bin/llama-bench -m llama-3.2-3b-q4hqq.gguf -ngl 0  -p 0 -n 128   # CPU (NEON)
./build/bin/llama-bench -m llama-3.2-3b-q4hqq.gguf -ngl 99 -p 0 -n 128   # Metal

# Task 2b: KV cache quantization
./build/bin/llama-cli -m llama-3.2-3b-q4_0.gguf -p "..." --cache-type-k q4_hqq
./build/bin/llama-cli -m llama-3.2-3b-q4_0.gguf -p "..." --cache-type-k q4_hqq --cache-type-v q4_hqq -fa on

# Task 2 perplexity (Wikitext-2)
sh scripts/get-wikitext-2.sh
./build/bin/llama-perplexity -m llama-3.2-3b-q4hqq.gguf -f wikitext-2-raw/wiki.test.raw -ngl 0 -c 512 --chunks 12

# Task 3: --mmproj-backend (needs a vision model)
hf download ggml-org/Qwen2-VL-2B-Instruct-GGUF Qwen2-VL-2B-Instruct-Q4_K_M.gguf --local-dir vlm
hf download ggml-org/Qwen2-VL-2B-Instruct-GGUF mmproj-Qwen2-VL-2B-Instruct-Q8_0.gguf --local-dir vlm
./build/bin/llama-mtmd-cli -m vlm/Qwen2-VL-2B-Instruct-Q4_K_M.gguf \
  --mmproj vlm/mmproj-Qwen2-VL-2B-Instruct-Q8_0.gguf \
  --image tools/mtmd/test-1.jpeg -p "Describe this image." -ngl 99 --mmproj-backend CPU
```

## Notes

- The fanless M4 throttles under sustained load; benchmark from a cool start and
  pin `-t 4` (P-cores). The included scripts handle this.
- Q4_HQQ valid KV-cache backend names are the ggml device names (`CPU`, `MTL0`),
  not aliases like `Metal`. Quantized V cache requires `-fa on`.
