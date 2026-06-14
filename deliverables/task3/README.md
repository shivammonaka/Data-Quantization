# Task 3 Report — `--mmproj-backend` CLI Flag

## Summary

Added `--mmproj-backend NAME`, which selects the backend device used for **only**
the multimodal projector (CLIP/vision encoder), leaving the base text model's
backend (set by `-ngl`) unchanged. Verified working on Qwen2-VL-2B: the
projector backend is chosen independently of the base model, which stays on GPU.

## Design

llama.cpp already contained projector backend-selection logic in `clip_ctx`,
but it was driven only by an environment variable (`MTMD_BACKEND_DEVICE`) and
had no CLI surface. Task 3 exposes it as a first-class flag and threads the
value cleanly through the existing layers:

```
--mmproj-backend NAME            (common/arg.cpp)
  -> common_params.mmproj_backend          (common/common.h)
    -> mtmd_context_params.mmproj_backend  (tools/mtmd/mtmd.h / .cpp)
      -> clip_context_params.backend_name  (tools/mtmd/clip.h)
        -> clip_ctx: ggml_backend_init_by_name(name)   (tools/mtmd/clip.cpp)
```

In `clip_ctx`, the explicit flag takes priority over the env var, then falls
back to automatic GPU/iGPU selection:

```c
const char * backend_name = (ctx_params.backend_name && ctx_params.backend_name[0])
                            ? ctx_params.backend_name
                            : std::getenv("MTMD_BACKEND_DEVICE");
if (backend_name) {
    backend = ggml_backend_init_by_name(backend_name, nullptr);
    if (!backend) { /* warn, fall back to default GPU */ }
}
```

Wired into both consumers: `llama-mtmd-cli` and `llama-server`. The base model
backend is untouched because it is initialized separately (via `-ngl` /
`llama_model_load`), so the projector and the LLM are independently placed.

Files: `common/arg.cpp`, `common/common.h`, `tools/mtmd/mtmd.h`,
`tools/mtmd/mtmd.cpp`, `tools/mtmd/clip.h`, `tools/mtmd/clip.cpp`,
`tools/mtmd/mtmd-cli.cpp`, `tools/server/server-context.cpp`.

## Backend names

`ggml_backend_init_by_name` matches the **registered ggml device name**, not a
friendly alias. On Apple Silicon these are:
- `CPU` — the CPU backend
- `MTL0` (and `MTL0_Private`) — the Metal GPU device (`GGML_METAL_NAME` = "MTL"
  + device index)

So `--mmproj-backend CPU` and `--mmproj-backend MTL0` are valid;
`--mmproj-backend Metal` is not an exact device name and triggers the
documented fallback (see below).

## Verification (Qwen2-VL-2B, base model on GPU via -ngl 99)

| Run | Flag | Result |
|---|---|---|
| default | (none) | projector auto → GPU; correct image description |
| proj_cpu | `--mmproj-backend CPU` | projector on CPU (no warning); correct description |
| proj_metal | `--mmproj-backend Metal` | `clip_ctx: Failed to initialize "Metal" backend, falling back to default GPU backend` → correct description |

All three produced the same correct output ("a newspaper clipping from The New
York Times, dated July 21, 1969 ... Men Walk on Moon").

The `proj_metal` run is informative: the string `Metal` propagated all the way
into `ggml_backend_init_by_name("Metal")`, which did not match the device name
`MTL0`, so the code logged the warning and gracefully fell back to the default
GPU — exactly the designed behavior, and direct evidence that the flag drives
the projector init path end-to-end. `--mmproj-backend MTL0` selects Metal by
its exact name. In every case the base model remained on GPU, confirming the
projector backend is selected independently.

## Usage

```bash
# projector forced to CPU, base model on GPU
llama-mtmd-cli -m model.gguf --mmproj mmproj.gguf --image img.jpg \
  -p "Describe this image." -ngl 99 --mmproj-backend CPU

# projector explicitly on the Metal device (exact name)
llama-mtmd-cli ... -ngl 99 --mmproj-backend MTL0
```

## Suggested improvement

Accepting a friendly alias (`Metal`, `GPU`) and resolving it to the concrete
device name would be more ergonomic than requiring `MTL0`. The current behavior
(try-by-name, warn, fall back) is safe but the alias mapping would remove the
need for users to know ggml's internal device naming.
