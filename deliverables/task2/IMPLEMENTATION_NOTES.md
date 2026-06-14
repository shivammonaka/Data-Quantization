# Task 2 Tier 1 — what the patch changes and why

13 files, ~146 lines. Grouped by purpose. The PDF listed 6 files; the current
tree needs more because (a) ggml split the CPU backend into its own directory
with its **own** trait table, and (b) the core/CPU quant code is split.

## A. Give the format an identity
1. **ggml/include/ggml.h** — `GGML_TYPE_Q4_HQQ = 42`, bump `GGML_TYPE_COUNT` to 43.
   Stable integer ID used everywhere downstream; COUNT is the size of every
   per-type array, so it must grow or the new entry writes out of bounds.
2. **include/llama.h** — `LLAMA_FTYPE_MOSTLY_Q4_HQQ = 99`. The llama-level handle
   the quantizer and loader speak in. (99 chosen to avoid colliding with used
   and reserved enum slots.)

## B. Define the data layout
3. **ggml/src/ggml-common.h** — `block_q4_hqq`: `ggml_half d, m` + 16 bytes of
   nibbles = 20 bytes / 32 weights = 5.0 bpw. `static_assert` guards the size.
   We store derived `d = 1/scale = (max-min)/15` and `m = min`, NOT raw
   scale/zero — this makes reconstruction `w = q*d + m`, identical to Q4_1.

## C. The three core functions (ggml-quants.{h,c})
4. `quantize_row_q4_hqq_ref` — implements the spec's formula verbatim:
   `scale=15/(max-min)`, `zero=-min*scale`, `q=clamp(round(w*scale+zero),0,15)`,
   then stores derived d/m. (Equivalence verified numerically vs NumPy.)
5. `dequantize_row_q4_hqq` — `w = q*d + m`.
6. `quantize_q4_hqq` — per-row dispatch entry the quantization driver calls.
   Tier 1: ignores the importance matrix (a future improvement).

## D. The dot product (ggml-cpu/quants.{h,c}, arch-fallback.h)
7. `ggml_vec_dot_q4_hqq_q8_0_generic` — scalar dot.
   **Key design point flagged for the writeup:** the spec NAMES this `_q8_0`,
   but an asymmetric format (`w = q*d + m`) cannot dot against Q8_0. Expanding
   `sum_i (q_i*d + m)*a_i = d*da*sum(q_i*qa_i) + m*(da*sum qa_i)`, the second
   term needs `sum(qa)`, which only **Q8_1** precomputes (its `s` field). So we
   keep the spec's name but pair with Q8_1 internally — exactly what Q4_1 does.
   arch-fallback.h maps the public name to the generic impl on ALL arches
   (no SIMD yet; NEON is Tier 3/bonus).

## E. Register in BOTH trait tables
8. **ggml/src/ggml.c** (core table): block size, name "q4_hqq", is_quantized,
   to_float=dequant, from_float_ref=quant_ref. Backend-independent facts.
9. **ggml/src/ggml-cpu/ggml-cpu.c** (CPU table): from_float=quantize_row_q4_hqq,
   vec_dot=our dot, **vec_dot_type=GGML_TYPE_Q8_1** (this is what tells ggml to
   quantize activations to Q8_1 before the dot — makes the m*s term valid).
   This two-table split is the #1 thing the PDF gets wrong.

## F. llama load/quantize wiring + CLI
10. **src/llama-quant.cpp** — ftype → GGML_TYPE_Q4_HQQ (quantization).
11. **src/llama-model-loader.cpp** — name string "Q4_HQQ" + reverse
    GGML_TYPE_Q4_HQQ → ftype (so a loaded model reports its type correctly).
12. **tools/quantize/quantize.cpp** — the `"Q4_HQQ"` CLI option row.

## Verified in sandbox before shipping
- Algorithm correctness: standalone C harness + independent NumPy reference
  agree on dequant RMSE (0.0389) and confirm spec formula == stored d/m form.
- `ggml-quants.c` and `ggml-cpu/quants.c` compile clean as isolated TUs.
- Full link/build happens on your Mac (Metal SDK needed for the whole binary).

## Expected results
- File size: ~2.0 GB (5.0 bpw vs Q4_0's 4.75 effective → ~10% larger).
- Quality: ~identical to Q4_1, slightly better than Q4_0 on asymmetric blocks.
- Speed: marginally slower than Q4_0 (extra m*s term), close to Q4_1.
- CPU only for now: `-ngl 0`. Offloading to Metal needs the Tier 3 kernel.
