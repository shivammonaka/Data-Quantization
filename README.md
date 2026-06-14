# QVAC Foundation Engineer — Assignment Submission

**Candidate:** Shivam Gautam
**Environment:** MacBook Air M4 (10-core, 16 GB), macOS 26.5.1, llama.cpp commit `ebc10770` (build b9614)

All three tasks are complete, including both Task 2 bonuses (NEON SIMD,
perplexity) and the Task 2c GPU bonus. The format runs on **CPU and GPU**.

---

## Verify in 2 minutes (no build required)

1. Read `SUMMARY.md` (one page: what was done, key findings, results).
2. Open `demo/key_results.md` — every claimed number with the exact log line
   that proves it (file sizes, speeds, perplexity, KV-cache memory, GPU output).

## If you want to build and check the code

1. `SETUP.md` — exact environment and commands.
2. `deliverables/q4_hqq_full.patch` — the complete implementation (Task 2 + Task 3),
   applies cleanly to llama.cpp commit `ebc10770`.
3. Per-task folders under `deliverables/` contain the report, scripts, and logs.

---


## Folder map

```
.
├── README.md                  (this file)
├── SETUP.md                   environment + exact commands (required deliverable)
├── SUMMARY.md                 1-page summary (required deliverable)
├── deliverables/              EXACTLY what the assignment asked for
│   ├── q4_hqq_full.patch       git patch: Q4_HQQ (2a/2b/2c) + --mmproj-backend (T3)
│   ├── task1/                  README + scripts/ + logs/
│   ├── task2/                  README + IMPLEMENTATION_NOTES + scripts/ + logs/
│   └── task3/                  README + scripts/ + logs/
└── demo/                       proof-at-a-glance
    └── key_results.md          every number + the log line that proves it
```

## One-line status per task

- **Task 1** — convert → Q4_0 → benchmark. 36.6 t/s CPU / 49.2 t/s Metal.
- **Task 2a** — Q4_HQQ format, 13 files, coherent output. Comparison table in report.
- **Task 2b** — `--cache-type-k/v q4_hqq`; KV cache 140 MiB vs F16 448 MiB.
- **Task 2c** — Metal GPU kernel; 44.85 t/s on the M4 GPU.
- **Task 3** — `--mmproj-backend`; projector backend independent of base model.
- **Bonuses** — NEON dot product (13.5 → 30.3 t/s); Wikitext-2 perplexity.
