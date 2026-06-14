# Task 2 Report — Q4_HQQ (Format, KV Cache, Metal, Bonuses)

This unifies Tasks 2a (format), 2b (KV cache), and 2c (Metal GPU), plus the NEON SIMD and perplexity bonuses.

---

## Task 2a Report — Q4_HQQ Quantization Format

## Summary

Implemented `Q4_HQQ`, a 4-bit asymmetric block-quantization format, end-to-end
in llama.cpp: block definition, quantize/dequantize/dot-product, registration
across both ggml trait tables and the llama load/quantize path, a CLI option,
and a hand-written ARM NEON kernel. The format quantizes, loads, and produces
coherent output on CPU, and was benchmarked against Q4_0 and the F16 reference
for size, speed, and perplexity.

All work is against pinned commit `ebc10770ac5a9331824c53ef0c6adad780904dc3`
(build b9614). Environment: MacBook Air M4, 16 GB, macOS 26.5.1.

## Format design

20 bytes per 32-weight block = 5.0 bits/weight:

```c
#define QK4_HQQ 32
typedef struct {
    ggml_half d;            // 1/scale = (max-min)/15
    ggml_half m;            // min  (== -zero/scale)
    uint8_t qs[QK4_HQQ/2];  // 32 4-bit weights, 2 per byte
} block_q4_hqq;
```

Spec formula (implemented verbatim in the quantizer):
`scale = 15/(max-min)`, `zero = -min*scale`,
`q = clamp(round(w*scale + zero), 0, 15)`, reconstruct `w = (q-zero)/scale`.

**Storage choice:** rather than store `scale` and `zero` directly, the block
stores the algebraically-equivalent `d = 1/scale` and `m = min`, so
reconstruction is `w = q*d + m`. This is byte-for-byte identical to Q4_1's
layout, which (a) lets the proven Q4_1 dot-product math be reused, and (b) was
verified numerically equivalent to the spec's `(q-zero)/scale` form (standalone
C harness + independent NumPy reference, max abs diff ~1e-3, attributable to
fp16 rounding).

## Integration points (13 files)

The assignment's file list reflected an older tree. The current code needed:
identity (`ggml.h` enum, `llama.h` ftype), block struct (`ggml-common.h`),
three functions (`ggml-quants.{h,c}`), the dot product
(`ggml-cpu/quants.{h,c}` + NEON in `ggml-cpu/arch/arm/quants.c`), registration
in **both** the core trait table (`ggml.c`) and the CPU trait table
(`ggml-cpu.c`), the quantize-chunk dispatch switch (`ggml.c`), row-data
validation and compute-op switches (`ggml-quants.c`, `ggml-cpu/ops.cpp`), the
load/quantize ftype maps (`llama-quant.cpp`, `llama-model-loader.cpp`), and the
CLI option (`quantize.cpp`).

## Key design decision: dot product pairs with Q8_1, not Q8_0

The spec names the dot `ggml_vec_dot_q4_hqq_q8_0`, but an asymmetric format
cannot correctly pair with Q8_0. Expanding the dot for `w = q*d + m`:

```
sum_i (q_i*d + m) * a_i  =  d*da * sum_i(q_i*qa_i)  +  m * (da * sum_i qa_i)
                         =  d*da*sumi  +  m*s
```

The offset term needs `s = da*sum(qa)`, the activation sum — which only
`block_q8_1` precomputes. `block_q8_0` has no such field, so the term cannot be
formed. The implementation keeps the spec's function name but pairs with Q8_1
internally and sets `vec_dot_type = GGML_TYPE_Q8_1`, exactly as Q4_1 does.

## Results

### Size (whole model, 3B)

| Format | Size | bpw |
|---|---|---|
| F16 | 6.43 GB | 16.0 |
| Q4_0 | 1820.9 MiB | 4.75 |
| Q4_HQQ | 1988.9 MiB | 5.19 |

Q4_HQQ is ~9% larger, consistent with 5.0 vs 4.5 bpw on quantized tensors
(1-D norm tensors stay F32 in both, raising both whole-model averages equally).

### Speed (CPU, llama-bench tg128, 4 threads)

| Format | Gen speed | Note |
|---|---|---|
| Q4_0 | 36.60 ± 0.07 t/s | tuned symmetric path |
| Q4_HQQ scalar | 13.5 t/s | generic C dot |
| Q4_HQQ NEON | 30.27 ± 1.29 t/s | hand-written kernel |

The NEON kernel gives a **2.24× speedup** over scalar (13.5 → 30.3 t/s),
reaching 83% of Q4_0's throughput. The residual gap is inherent: Q4_HQQ pays an
extra `m*s` offset term per block and pairs with the heavier Q8_1 activations,
whereas symmetric Q4_0 does neither.

### Quality (Wikitext-2 perplexity, 12 chunks, CPU, equal conditions)

| Model | Perplexity (lower better) | Δ vs F16 |
|---|---|---|
| F16 | 12.67 ± 0.65 | — |
| Q4_0 | 13.43 ± 0.69 | +0.76 |
| Q4_HQQ | 13.65 ± 0.71 | +0.99 |

**Q4_HQQ is marginally worse than Q4_0**, not better. This is the expected
result for the spec as written: it specifies asymmetric min-max round-to-nearest
(equivalent to Q4_1), while llama.cpp's Q4_0 path applies a scale-search plus
REPACK optimization. Min-max RTN is sensitive to per-block outliers (one
extreme weight widens the range and coarsens the other 31). The format omits
HQQ's defining feature — iterative zero-point optimization minimizing an Lp
reconstruction error — which is precisely what would let it beat Q4_0.

## Verification methodology

1. Algorithm correctness proven in isolation (C harness + NumPy) before any
   integration: dequant RMSE and dot-product error matched a float reference.
2. NEON kernel's lane-grouping logic validated against the scalar reference
   (exact match); intrinsic correctness inherited from the identical-layout
   Q4_1 production kernel.
3. End-to-end correctness confirmed on-device: coherent generation, valid
   perplexity.

## Issues faced

- Spec file list was stale; real integration surface is 13 files across two
  trait tables and several dispatch switches.
- First integration crashed with `GGML_ASSERT(result == nrows*row_size)`: the
  quantize function was registered in the trait tables but not added to the
  `ggml_quantize_chunk` switch, so it hit `default` and wrote zero bytes. Fixed
  by adding the case; then audited every quant-type switch on the build's code
  path to catch the class of bug rather than one instance.
- Perplexity timing on the fanless M4 is thermally sensitive — back-to-back
  passes throttle (a quantized run measured 10× slower than F16 purely from
  heat-soak). Models must be benchmarked from a cool start.

## Conclusion

Q4_HQQ is a fully functional, correctly integrated 4-bit format with a
vectorized CPU kernel. As specified it trades ~9% size and ~17% speed for no
quality gain over Q4_0, because the spec stops short of true HQQ optimization.
The highest-value improvement would be adding iterative zero-point refinement.


---

## Task 2b Report — Q4_HQQ as a KV-Cache Quantization Type

## Summary

Enabled `--cache-type-k q4_hqq` and `--cache-type-v q4_hqq`. Q4_HQQ is now a
selectable KV-cache type. All configurations produce coherent output, and the
cache memory footprint was measured directly and matches theory.

## Implementation

One line: adding `GGML_TYPE_Q4_HQQ` to the `kv_cache_types` list in
`common/arg.cpp`. No new compute code was required, for a structural reason:

- **Cache write** (storing new K/V) is an F32 → cache-type copy handled by
  `ggml_compute_forward_dup_to_q`, which dispatches through the trait table's
  `from_float` — already registered in Task 2a.
- **Cache read** (attention scoring against cached K/V) is a `mul_mat`, which
  dispatches through the trait table's `vec_dot` — also already registered.

Because Q4_HQQ was correctly wired into both trait tables for weight
quantization, the generic KV-cache paths picked it up automatically. The
parser resolves the `q4_hqq` string via `ggml_type_name`, which the core trait
registration already provides.

## Constraint: quantized V requires Flash Attention

`src/llama-context.cpp` enforces (generically, for any quantized type) that a
quantized V cache requires Flash Attention:

> quantized V cache was requested, but this requires Flash Attention

This is because the default non-flash path stores V transposed, which cannot be
written column-wise into a block-quantized tensor. It is not specific to
Q4_HQQ — Q8_0/Q4_0 V-cache quant carry the same requirement. Therefore:
`--cache-type-k q4_hqq` works alone; `--cache-type-v q4_hqq` requires `-fa on`.

## Results (Llama-3.2-3B, ctx=4096, CPU, measured)

| Cache type | Total (K+V) | Per side | vs F16 | bpw |
|---|---|---|---|---|
| F16 (default) | 448.00 MiB | 224.00 MiB | — | 16.0 |
| Q4_0 | 126.00 MiB | 63.00 MiB | −71.9% | 4.5 |
| **Q4_HQQ** | **140.00 MiB** | **70.00 MiB** | **−68.8%** | 5.0 |

Numbers are from llama.cpp's own `llama_kv_cache: size = ...` init line, and
match an independent hand calculation (28 layers × 4096 cells × 1024 KV-dim) to
the MiB. The 70.00 MiB per side confirms an exact 5.0 bpw cache rate.

Output quality: coherent and near-identical across F16, K-only Q4_HQQ, and
K+V Q4_HQQ (`-fa on`) cache configurations (logs: `t2b_*.log`).

## Interpretation

Q4_HQQ reduces KV-cache memory by ~69% versus the F16 default — the dominant
benefit of cache quantization, which matters most at long context where the
cache, not the weights, bounds memory. Q4_HQQ's cache is 14 MiB (11%) larger
than Q4_0's, the same 5.0-vs-4.5 bpw cost seen in the weights, in exchange for
asymmetric representation. As with the weight comparison, the spec's min-max
formulation gives no quality edge over Q4_0 here; iterative zero-point
optimization (true HQQ) would be required for that.

## Commands

```bash
## K cache quantized (no flash attn needed)
llama-cli -m model.gguf -p "..." --cache-type-k q4_hqq

## K and V cache quantized (requires flash attention)
llama-cli -m model.gguf -p "..." --cache-type-k q4_hqq --cache-type-v q4_hqq -fa on
```


---

## Task 2c Report — Q4_HQQ Metal (GPU) Kernel

## Summary

Implemented Metal GPU support for Q4_HQQ. The format now runs on the Apple M4
GPU via `-ngl 99`, producing coherent output at 44.85 t/s — 91% of Q4_0's GPU
throughput. This satisfies the verification requirement that the format run on
both CPU and GPU.

## Implementation

Because `block_q4_hqq` is byte-identical to `block_q4_1` and uses the same
reconstruction (`w = q*d + m`), the existing, proven Q4_1 Metal templates were
instantiated for the new type rather than writing new shader math:

In `ggml-metal.metal`:
- `dequantize_q4_hqq` and `dequantize_q4_hqq_t4` — block dequant (copies of the
  Q4_1 versions, retyped).
- `block_q_n_dot_y(block_q4_hqq*, ...)` overload — the matvec inner product,
  computing `d*Σ(q*y) + sumy*m` (the GPU form of the asymmetric offset term).
- `kernel_mul_mv_q4_hqq_f32` — the mat-vec kernel (decode path).
- `kernel_get_rows_q4_hqq` — embedding/output row gather.
- `kernel_mul_mm_q4_hqq_f32` / `_f16` — the mat-mul kernel (prefill/batched).

In `ggml-metal-impl.h`: tuning constants `N_R0_Q4_HQQ`, `N_SG_Q4_HQQ` (= Q4_1's).

In `ggml-metal-device.cpp` / `ggml-metal-device.m`: registered Q4_HQQ in the
mat-vec parameter switches and the supported-type switches (get_rows, mul_mat,
flash-attn type checks). Pipeline names are auto-constructed from
`ggml_type_name` (→ "q4_hqq"), which is why the kernel host-names must match
exactly.

## Scoping decision

Q4_HQQ was deliberately **not** added to the "ext" small-batch mat-vec path in
`ggml-metal-ops.cpp`. That path requires four additional kernel template
variants (`kernel_mul_mv_ext_q4_hqq_f32_r1_{2..5}`). Omitting them makes
Q4_HQQ fall through to the standard mat-vec kernel — fully correct, just without
the small-batch (BS 2–8) micro-optimization. This keeps the GPU surface minimal
while delivering correct, performant single-stream generation.

## Results (Llama-3.2-3B, Metal, M4 GPU)

| Path | Gen speed (tg128) |
|---|---|
| Q4_HQQ CPU scalar | 13.5 t/s |
| Q4_HQQ CPU NEON | 30.27 t/s |
| **Q4_HQQ Metal GPU** | **44.85 ± 0.21 t/s** |
| Q4_0 Metal GPU (ref) | 49.23 t/s |

GPU execution confirmed by the `MTL,BLAS` backend tag and the timing profile.
Q4_HQQ reaches 91% of Q4_0's GPU throughput; the gap is the same asymmetric
overhead (extra offset term, heavier activation handling) seen on CPU.

Output quality on GPU is coherent and consistent with the CPU runs.

## Full execution-path summary

Q4_HQQ runs correctly on all three paths with a clean speedup ladder:
**13.5 (scalar) → 30.3 (NEON) → 44.85 (Metal) t/s.**

