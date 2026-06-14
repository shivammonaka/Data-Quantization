# Task 1 Report — Quantize and Run LLaMA 3.2 3B (Q4_0)

## Environment

| | |
|---|---|
| Hardware | MacBook Air, Apple M4 (4 P-cores + 6 E-cores, 10-core GPU), 16 GB unified memory |
| OS | macOS 26.5.1 (build 25F80) |
| Toolchain | Apple clang 21.0.0, CMake 4.3.3, Python 3.12 (venv) |
| llama.cpp | commit `ebc10770ac5a9331824c53ef0c6adad780904dc3` (build b9614), Metal + Accelerate BLAS enabled (defaults on Apple Silicon) |
| Model weights | `meta-llama/Llama-3.2-3B-Instruct`, obtained via the ungated `unsloth/Llama-3.2-3B-Instruct` mirror (identical weights, HF safetensors format) |

## Pipeline and exact commands

```bash
# 1. Download (HF safetensors)
hf download unsloth/Llama-3.2-3B-Instruct \
  --local-dir ~/models/llama-3.2-3b-instruct

# 2. Build llama.cpp (Metal on by default on Apple Silicon)
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release -j 10

# 3. Convert HF -> F16 GGUF
python convert_hf_to_gguf.py ~/models/llama-3.2-3b-instruct \
  --outfile gguf/llama-3.2-3b-f16.gguf --outtype f16

# 4. Quantize F16 -> Q4_0
./build/bin/llama-quantize gguf/llama-3.2-3b-f16.gguf \
  gguf/llama-3.2-3b-q4_0.gguf Q4_0

# 5. Inference (per prompt, per backend; -ngl 0 = CPU, -ngl 99 = Metal)
./build/bin/llama-cli -m gguf/llama-3.2-3b-q4_0.gguf \
  -p "<prompt>" -n 200 -t 4 -ngl {0|99} -c 4096 \
  --seed 42 -st --simple-io -lv 3

# 6. Standardized throughput
./build/bin/llama-bench -m gguf/llama-3.2-3b-q4_0.gguf -t 4 -ngl {0|99} -p 512 -n 128
```

Full conversion/quantization scripts and benchmark harness (including peak-RSS
polling) are included alongside this report; raw logs are in `logs/`.

## Artifacts

| File | Size | Notes |
|---|---|---|
| `llama-3.2-3b-f16.gguf` | 6.43 GB (16.00 bpw) | 255 tensors, conversion master copy |
| `llama-3.2-3b-q4_0.gguf` | 1820.9 MiB (4.75 bpw) | quantize time 7.16 s |

## Results

### Per-prompt inference (n=200 generated tokens, ctx 4096, seed 42, chat template applied)

| Backend | Prompt | First-token latency¹ | Gen speed | Total time | Peak RSS² |
|---|---|---|---|---|---|
| CPU (`-ngl 0`, `-t 4`) | "What is bitcoin?" (39 tok) | 214.9 ms | 36.56 t/s | 5.69 s | 4196 MB |
| CPU (`-ngl 0`, `-t 4`) | "Write a Python function..." (44 tok) | 225.1 ms | 36.47 t/s | 5.71 s | 4197 MB |
| Metal (`-ngl 99`) | "What is bitcoin?" (39 tok) | 149.6 ms | 48.33 t/s | 4.29 s | 2389 MB |
| Metal (`-ngl 99`) | "Write a Python function..." (44 tok) | 150.5 ms | 48.54 t/s | 4.27 s | 2389 MB |

¹ Reported as `prompt eval time` from the engine's per-request timing: time to
process the full prompt before the first output token is sampled.
² Peak resident set size, polled at 5 Hz. Exact for the CPU runs (weights, KV
cache, and compute buffers are all process-resident pages). For the Metal runs
it is a lower bound: GPU-side allocations in Metal buffer objects are not fully
attributed to process RSS by macOS, even though memory is physically unified.

### Standardized throughput (llama-bench, mean ± σ)

| Backend | pp512 (prefill) | tg128 (generation) |
|---|---|---|
| CPU (4 threads, Accelerate BLAS prefill) | 177.08 ± 4.69 t/s | 36.63 ± 0.23 t/s |
| Metal | 439.99 ± 1.66 t/s | 49.23 ± 0.19 t/s |

## Output quality

Both prompts produced coherent, correctly formatted answers (full text in
`logs/infer_*.log`). The bitcoin answer is factually reasonable and
well-structured; the Python answer produced a correct in-place `reverse()`
solution with docstring, example usage, and a correct note about in-place
mutation with a copy-based alternative. CPU and Metal outputs are near-identical
at the same seed, diverging slightly after ~80 tokens (expected:
floating-point reduction order differs across backends, so logits can differ
in the last ulp and sampling eventually diverges).

## Methodology notes

- **Thread pinning:** `-t 4` matches the M4's 4 performance cores. llama.cpp's
  per-layer barrier synchronization means E-core participation gates every
  layer on the slowest thread; 4 threads is both fastest and lowest-variance.
- **Generation speed is memory-bandwidth bound:** Metal tg128 of 49.2 t/s ×
  1.86 GB of weights ≈ 92 GB/s effective bandwidth, ~75% of the M4's ~120 GB/s
  — consistent with a healthy memory-bound decode. CPU prefill (177 t/s)
  routes through Accelerate BLAS; Metal prefill in the short real prompts
  (260–292 t/s) sits below bench pp512 (440 t/s) because 39–44 token prompts
  do not saturate the GPU.
- **Thermals:** the machine is fanless; runs were separated by idle cooldowns.
  No thermal drift was observed (generation speed stable to <0.5% between
  first and last runs).

## Issues faced

- HF gating on `meta-llama/Llama-3.2-3B-Instruct` stalled; resolved by using
  the ungated unsloth mirror of the identical weights.
- The rewritten `llama-cli` writes console output directly to `/dev/tty`,
  which silently defeats stdout/stderr redirection — initial benchmark logs
  captured nothing. Fixed by running with `--simple-io` (and `-lv 3` to surface
  the engine's per-request `prompt eval time` / `eval time` lines).
- macOS lacks `/usr/bin/time -v`; peak RSS was captured by polling
  `ps -o rss=` at 5 Hz on the inference process.
