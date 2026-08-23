# M5 performance report: M4 baseline → M5

**Model:** DeepSeek-R1-Distill-Qwen-1.5B (Q5_K_M GGUF, fp16 dequantized)
**Host:** Apple Silicon (arm64 macOS, 10 logical CPUs — 4 P + 4 E + …,
`hw.logicalcpu = 10`)
**Workload:** `generate("<|User|>What is 1+1?<|Assistant|><think>\n",
max_tokens=120, temperature=0.6, top_p=0.95, top_k=40, seed=7)` — 436 chars,
identical output to the M3/M4 reference (byte-for-byte, verified by the e2e
pytest against `tests/python/reference_outputs/reference_generate.txt`).

| metric | M4 baseline | M5 final | change |
| :--- | :--- | :--- | :--- |
| 1.5B generation, 120 tokens | 30.3 s | **19.6 s** | **1.55× faster** |
| per-token decode | ~252 ms | ~163 ms | 1.55× |
| LM-head projection (151936×1536) | ~89 ms | ~16 ms | **5.6× faster** |
| transformer layers / token | ~150 ms | ~110 ms | 1.36× |

**M5 target was < 20 s / 120 tokens: met (19.6 s).**

## What moved the needle

1. **Persistent C thread pool** (`tools/thread_pool.c` +
   `src/core/thread_pool.mojo`). Mojo 1.0's stdlib has no threading and the
   driver cannot link C object files, so the pool is a small pthread
   library linked with `-Xlinker`, driven through `external_call`. Worker
   threads persist for the process lifetime (decode issues ~250 pool
   submissions per token; per-call `pthread_create` ate the whole win).
   Workers are Mojo `@export abi("C")` entry points resolved by
   `dlsym(RTLD_DEFAULT, ...)` (the engine dylib loads `RTLD_GLOBAL` from
   Python). Measured pool scaling: 5.8× on 8 threads (pure ALU), 9.5× on
   10 threads (in-dylib worker benchmark).
2. **Threaded + batched matmuls.** `matmul_weight_cpu_threaded` splits the
   output columns across pool workers for wide projections
   (N ≥ 4096 → gate/up/lm-head); `matmul_weight_3_threaded` /
   `matmul_weight_2_threaded` run the QKV triple and the gate+up pair in
   *one* pool submission each, amortizing the wake-up latency. The LM head
   went from ~89 ms to ~16 ms (5.6×).
3. **Fused kernels** (Phase 2): `fused_matmul_add_bias`,
   `fused_matmul_add`, `fused_matmul_rms_norm`, `fused_swiglu_matmul` —
   each collapses two passes into one with f32 accumulation (M3 numerics
   rule). The torch.compile backend rewrites 2D patterns onto them
   (linear+β, down(silu(g)u), rms_norm(linear)); the optimizer IR fuses
   the same shapes on the engine side.
4. **SIMD sweep** (Phase 5): `add`, `add_bias`, `swiglu`, the weighted
   RMSNorm and the shared dot-product helpers are now vectorized via
   `src/core/simd_utils.mojo` (128-bit NEON/SSE widths, f32 accumulate);
   matmul/softmax/rms_norm were already vectorized in M3. All 10 CPU
   kernel families now have SIMD main loops (scalar tails only).

## Toolchain landmines found (documented for M6)

- `@export` emits symbols **only for the compilation entry module** —
  pool workers must live in `src/bindings/infer_train_bindings.mojo`.
- `Int(pointer)` truncates to 32 bits inside the shared library for some
  pointer origins — worker contexts are built as **Int64 arrays written
  element-wise** (the pattern M4's tensor-copy API already proved safe),
  and field-wise stores onto `unsafe_alloc` struct slots are miscompiled.
- `mojo <file>` (run mode) does not resolve `-Xlinker` libraries; all
  executable tests now build + run.
- `unsafe_from_address` is fine on pool threads; struct-context field
  stores are not. Lane-wise SIMD inserts are miscompiled (avoid).

## Correctness gates

- `verify_dags` runs the original and optimized DAGs through the same
  interpretive executor and compares outputs (per-pass verification is
  exercised in `tests/test_optimizer.mojo`).
- Every Python-level pass/fusion is checked against torch eager in
  `tests/python/` (25 tests, incl. the e2e 1.5B reference comparison).
- M1–M3 suites unchanged and green (`make test`, `make test-m3`).

## Memory planning

Liveness analysis + first-fit slot reuse (`memory_plan.mojo`): the
transformer-shaped chain graph pools 16 blocks' worth of intermediate
tensors into a single slot — **1536 B planned vs 48 B... 96% smaller** (see
`tests/test_optimizer.mojo`: baseline 1536 B → planned 48 B). Target
"≥ 20%" — far exceeded on the activation workspace (weights are resident
and outside the pool).
