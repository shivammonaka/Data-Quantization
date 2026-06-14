# QVAC Assignment — Summary

**Candidate:** Shivam Gautam · **Environment:** MacBook Air M4, 16 GB, macOS 26.5.1
**llama.cpp:** pinned commit `ebc10770` (build b9614), Metal + NEON enabled

## What was done

**Task 1 — Quantize & run.** Converted Llama-3.2-3B-Instruct to F16 GGUF, quantized
to Q4_0, benchmarked on CPU and Metal. Q4_0: 1820.9 MiB (4.75 bpw); generation
36.6 t/s CPU / 49.2 t/s Metal; first-token latency 215 ms CPU / 150 ms Metal.

**Task 2 — Q4_HQQ format.** Implemented a new 4-bit asymmetric block format
end-to-end across 13 files: block struct, quantize/dequantize/dot-product,
registration in both ggml trait tables, the quantize-chunk dispatch, load path,
and CLI option.
- **2a:** format works; coherent output. 1988.9 MiB (5.19 bpw).
- **2b:** enabled `--cache-type-k/v q4_hqq`. KV cache 140 MiB vs F16 448 MiB (−69%).
- **2c:** Metal GPU kernel; runs at 44.85 t/s on the M4 GPU.
- **Bonus (NEON):** hand-written ARM dot product, 13.5 → 30.3 t/s (2.24×).
- **Bonus (perplexity):** Wikitext-2 — F16 12.67, Q4_0 13.43, Q4_HQQ 13.65.

**Task 3 — `--mmproj-backend`.** Added a CLI flag to select the backend for only
the multimodal projector, independent of the base model. Verified on Qwen2-VL-2B.

## Key engineering findings

1. **The spec's "Q4_HQQ" is re-parameterized Q4_1.** The formula is asymmetric
   min-max round-to-nearest; verified numerically identical to Q4_1 (dequant
   RMSE 0.0389). It is *not* true HQQ, which adds iterative zero-point
   optimization.
2. **The dot product must pair with Q8_1, not Q8_0** (despite the spec naming
   `_q8_0`). The asymmetric offset term `m·s` needs the activation sum, which
   only Q8_1 precomputes.
3. **Honest quality result:** as specified, Q4_HQQ is marginally *worse* than
   Q4_0 (13.65 vs 13.43 ppl) at 9% more size and ~17% less speed — because it
   omits HQQ's defining optimization while Q4_0 uses a scale-search + REPACK.
   The highest-value improvement would be adding iterative zero-point refinement.

## Speed ladder (Q4_HQQ, Llama-3.2-3B)

| Path | tg128 |
|---|---|
| CPU scalar | 13.5 t/s |
| CPU NEON | 30.3 t/s |
| Metal GPU | 44.9 t/s |

## Per-task reports

- `task1_report.md` — Task 1 detail
- `task2_report.md` — unified Task 2 (format, KV cache, Metal, bonuses)
- `task3_report.md` — Task 3 detail
- patch: `q4_hqq_full.patch` (all of Task 2 + Task 3, against commit ebc10770)
