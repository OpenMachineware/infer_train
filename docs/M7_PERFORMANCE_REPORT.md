# M7 Performance Report

Machine: **Apple M1 Max, 64 GB unified memory**.  All InferTrain numbers are
CPU multi-threaded (the C pthread task pool drives the matmul kernels; SIMD
covers the norm/dot kernels).  llama.cpp baselines are the **CPU-only**
build (`GGML_METAL=OFF`, Apple AMX-enabled CPU backend) from the same source
tree, same prompt, same sampling settings.

## 1. Benchmark results

Prompt: `The quick brown fox jumps over the lazy dog.` (10 tokens), 16-token
generation, temp 0.6 / top-k 40 / top-p 0.95, seed 7, 512-token context.

| model | InferTrain prefill | InferTrain decode | llama.cpp (CPU) prefill | llama.cpp (CPU) decode | decode ratio |
| :--- | :--- | :--- | :--- | :--- | :--- |
| DeepSeek-R1-Distill-Qwen-1.5B (Q5_K_M) | 4.4 t/s | 4.2 t/s | 171 t/s | 60.3 t/s | **7%** |
| Hy-MT2-7B (Q4_K_M, hunyuan-dense) | 0.82 t/s | 0.82 t/s | 63.8 t/s | 19.6 t/s | **4%** |
| Qwen3.8-27B (UD-Q5_K_M, qwen35 hybrid) | 0.23 t/s | 0.22 t/s | 12.2 t/s | 4.4 t/s | **5%** |

Peak memory (InferTrain, fp16 dequantized weights + KV cache + mmap file):

| model | weights (fp16) | KV cache (512 ctx) | observed RSS |
| :--- | :--- | :--- | :--- |
| 1.5B | ~3.1 GB | ~0.4 GB | ~5 GB |
| 7B | ~14 GB | ~0.5 GB | ~15 GB |
| 27B | ~51 GB | ~2.1 GB | ~55 GB |

The 27B runs stably at 32K context (the KV cache adapts via
`KVCache.adjust_capacity`; the SSM state is ~150 MB, constant in context
length), so the memory envelope stays inside 64 GB.

## 2. Status vs the M7 performance target

**The 85%-of-llama.cpp target is NOT met.**  Decode throughput is 4–7% of
the CPU llama.cpp baseline on all three models.  Correctness is unaffected —
the same binaries are validated token-for-token against llama.cpp (see
below) — the gap is purely in kernel efficiency.

### Gap analysis

1. **The Gated DeltaNet recurrence is scalar.**  48 of 64 layers of the
   27B model run a 128×128 outer-product/decay update per value head
   (48 heads).  That is ~110 M pointer loads/stores per token, each
   currently compiled to a scalar load/store with no vectorization.
   llama.cpp fuses this into a SIMD `ggml_gated_delta_net` op.
2. **The attention decode path is scalar per element.**  `mha_forward_v2`
   computes scores element-by-element (the dense-pointer fast path added
   in M7 helped, but there is no Q·K^T blocking or SIMD dot reduction).
3. **Matmuls use the C pool without data repacking.**  Each projection is
   a generic fp16 dot with fp32 accumulation; llama.cpp's Apple CPU
   backend repacks weights for AMX (the Apple Matrix coprocessor), which
   is ~15–20× faster on the M1 Max than portable SIMD loops.
4. **First-token page faults.**  Model files are `mmap`'d; the first
   forwards pay the page-fault cost (a warm-up pass is required for
   stable prefill numbers).

### Roadmap (v1.1)

* vectorize the DeltaNet recurrence (SIMD over the 128-dim state rows);
* block the attention Q·K^T dot products and keep softmax in fp32 SIMD;
* repack fp16 weights for AMX-style outer-product matmuls (or a Metal
  backend — the porting contract is documented in PORTING_GUIDE.md);
* i8/i16 quantized dot kernels (Q8_0/Q4_K weights computed in packed form
  instead of dequantize-on-load).

## 3. Correctness (validated against llama.cpp)

* **Hy-MT2-7B (hunyuan-dense)**: 4 consecutive greedy decode steps match
  llama.cpp token-for-token; full 128K logits vectors correlate ≥ 0.9995
  (fp16-activation rounding vs llama's fp32).
* **Qwen3.8-27B (qwen35 hybrid)**: single-token logits correlate 0.9991;
  6 consecutive greedy tokens match llama.cpp exactly.  Layer-by-layer
  intermediate dumps (projections, conv+SiLU, L2 norms, recurrence,
  gated norm, attention QK-norms, MRoPE) were compared during bring-up.
* **1.5B (qwen2)**: the M3 reference logits (`reference_logits_5.npy`)
  reproduce bit-for-bit after the M7 architecture refactor.
* All GGUF dequantizers (Q4_K/Q5_K/Q6_K/Q8_0/IQ4_NL/IQ4_XS) validated
  against gguf-py; the new Q4_K/Q8_0/NF4 *quantizers* round-trip within
  format bounds (max error 0.003 / 0.053 / 0.119 on a smooth test signal).

## 4. Reproduce

```bash
make tp
pixi run mojo build -I . /tmp/bench27.mojo -Xlinker python/infer_train/_lib/libinfer_train_tp.dylib -o bench
./bench                      # 27B (or adapt the model path)
# llama.cpp CPU baseline (source build with -DGGML_METAL=OFF)
llama-cli -m MODEL -p "The quick brown fox jumps over the lazy dog." \
    -n 16 --temp 0.6 --seed 7 --single-turn --simple-io
```
