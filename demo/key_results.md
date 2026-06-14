# Key Results — Proof at a Glance

Every claim below is backed by the exact log line that produced it. Logs are in
`deliverables/task*/logs/`. No build required to verify these.

---

## Task 1 — Q4_0 baseline (Llama-3.2-3B)

| Metric | CPU (-ngl 0) | Metal (-ngl 99) |
|---|---|---|
| First-token latency | 215 ms | 150 ms |
| Generation (tg128) | 36.60 ± 0.07 t/s | 49.23 ± 0.19 t/s |
| Peak RSS | 4.2 GB | 2.4 GB |
| File size | 1820.9 MiB (4.75 bpw) | — |

Proof (llama-bench):
```
llama 3B Q4_0 | MTL,BLAS | 4 | tg128 | 36.60 ± 0.07     (CPU, -ngl 0)
llama 3B Q4_0 | MTL,BLAS | 4 | tg128 | 49.23 ± 0.19     (Metal)
```

---

## Task 2a — Q4_HQQ format

Quantize succeeds and produces a runnable model:
```
$ llama-quantize llama-3.2-3b-f16.gguf llama-3.2-3b-q4hqq.gguf Q4_HQQ
... size = 1988.93 MiB (5.19 bpw)
```

Coherent output (CPU):
```
> What is bitcoin?
Bitcoin is a decentralized digital currency that uses cryptography to secure
and verify transactions. It was created in 2009 by an anonymous individual or
group using the pseudonym Satoshi Nakamoto...
```

### Comparison table (Q4_HQQ vs Q4_0)

| | Q4_0 | Q4_HQQ |
|---|---|---|
| File size | 1820.9 MiB (4.75 bpw) | 1988.9 MiB (5.19 bpw) |
| Gen speed (CPU, NEON) | 36.6 t/s | 30.3 t/s |
| Gen speed (Metal) | 49.2 t/s | 44.9 t/s |
| Perplexity (Wikitext-2) | 13.43 | 13.65 |
| Output quality | coherent | coherent |

---

## Task 2 bonus — NEON SIMD speedup

| Q4_HQQ path | tg128 |
|---|---|
| CPU scalar (generic C) | 13.5 t/s |
| CPU NEON (hand-written) | 30.27 ± 1.29 t/s |
| Metal GPU | 44.85 ± 0.21 t/s |

Proof (llama-bench, NEON CPU):
```
llama 3B Q4_HQQ | MTL,BLAS | 4 | tg128 | 30.27 ± 1.29
```

---

## Task 2 bonus — Perplexity (Wikitext-2, 12 chunks, CPU, equal conditions)

| Model | Perplexity (lower=better) | Δ vs F16 |
|---|---|---|
| F16 | 12.6661 ± 0.65 | — |
| Q4_0 | 13.4273 ± 0.69 | +0.76 |
| Q4_HQQ | 13.6534 ± 0.71 | +0.99 |

Proof (final estimate lines):
```
F16    : Final estimate: PPL = 12.6661 +/- 0.64655
Q4_0   : Final estimate: PPL = 13.4273 +/- 0.68921
Q4_HQQ : Final estimate: PPL = 13.6534 +/- 0.70610
```

Honest reading: Q4_HQQ is marginally *worse* than Q4_0 — expected, because the
spec's formula is asymmetric min-max RTN (≈ Q4_1), while Q4_0 uses a
scale-search + REPACK. True HQQ's iterative zero-point optimization (not in the
spec) is what would close this. See task2 report.

---

## Task 2c — Metal GPU

Q4_HQQ runs on the M4 GPU (-ngl 99), coherent output at 44.85 t/s:
```
llama 3B Q4_HQQ | MTL,BLAS | tg128 | 44.85 ± 0.21
[GPU smoke] "Bitcoin is a decentralized digital currency..." (coherent)
```

---

## Task 2b — KV cache quantization (ctx=4096, measured)

| Cache type | Total (K+V) | Per side | vs F16 |
|---|---|---|---|
| F16 (default) | 448.00 MiB | 224.00 | — |
| Q4_0 | 126.00 MiB | 63.00 | −71.9% |
| Q4_HQQ | 140.00 MiB | 70.00 | −68.8% |

Proof (llama.cpp init line):
```
llama_kv_cache: size = 140.00 MiB (4096 cells, 28 layers, 1/1 seqs),
                K (q4_hqq): 70.00 MiB, V (q4_hqq): 70.00 MiB
```
(`--cache-type-v q4_hqq` requires `-fa on`; output coherent in all configs.)

---

## Task 3 — --mmproj-backend

Projector backend selected independently of the base model (which stays on GPU).
The flag drives `ggml_backend_init_by_name` end-to-end:

```
[--mmproj-backend CPU]   -> projector on CPU, correct image description
[--mmproj-backend Metal] -> clip_ctx: Failed to initialize "Metal" backend,
                            falling back to default GPU backend
                            (proves the string reaches the init call; the exact
                             device name is MTL0)
```

All three runs produced the same correct output:
```
"a newspaper clipping from The New York Times, dated July 21, 1969 ...
 Men Walk on Moon"
```
