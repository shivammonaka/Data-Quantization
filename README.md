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
3. Watch `demo/screen_recording.mp4` (~4 min) to see each task run live.

## If you want to build and check the code

1. `SETUP.md` — exact environment and commands.
2. `deliverables/q4_hqq_full.patch` — the complete implementation (Task 2 + Task 3),
   applies cleanly to llama.cpp commit `ebc10770`.
3. Per-task folders under `deliverables/` contain the report, scripts, and logs.

---

## Automated verification

Run the included script to check everything at once — it applies the patch,
builds, runs each task, and prints PASS/FAIL against the expected numbers:

```bash
./verify.sh                      # full: build + all tasks
./verify.sh --no-build           # skip build (use an already-built tree)
./verify.sh --skip-task3         # skip the multimodal test (no VLM needed)
```

Paths are auto-detected but configurable via env vars at the top of the script
(`REPO`, `GGUF`, `PATCH`). For manual reproduction of any single result, see `SETUP.md`.

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
    ├── key_results.md          all numbers + the log lines that prove them
    ├── screen_recording.mp4    live run of every task
    └── screenshots/            terminal captures per task
```

## One-line status per task

- **Task 1** — convert → Q4_0 → benchmark. 36.6 t/s CPU / 49.2 t/s Metal.
- **Task 2a** — Q4_HQQ format, 13 files, coherent output. Comparison table in report.
- **Task 2b** — `--cache-type-k/v q4_hqq`; KV cache 140 MiB vs F16 448 MiB.
- **Task 2c** — Metal GPU kernel; 44.85 t/s on the M4 GPU.
- **Task 3** — `--mmproj-backend`; projector backend independent of base model.
- **Bonuses** — NEON dot product (13.5 → 30.3 t/s); Wikitext-2 perplexity.
