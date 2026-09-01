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
  The FP32 dequantization path is bit-exact with ggml-quants.c and the
  fp16 path's precision gap is quantified in §4.

## 4. Dequantization precision: fp16 inference path vs FP32 path

The inference path dequantizes GGUF block-quantized weights into **fp16**
(`dequantize_into`).  The decode arithmetic itself is FP32 and matches
llama.cpp's `dequantize_row_*` (ggml-quants.c) exactly; the only precision
question is what happens when the exact value is stored into the fp16
weight buffer.  Two entry points now exist in
`src/core/ops/quantized/dequantize.mojo`:

| entry point | output | error vs ggml-quants.c (FP32 reference) |
| :--- | :--- | :--- |
| `dequantize_into` (inference, existing) | fp16 | one fp16 rounding: ≤ 2⁻¹¹ ≈ 4.88e-4 relative |
| `dequantize_into_f32` / `dequantize_q4_k_m_f32` (new) | fp32 | **0 — bit-exact** |

Measured on real model tensors (1024 values each; Q4_K/Q5_K/Q6_K/Q8_0/
IQ4_NL/IQ4_XS/F32 from Hy-MT2-7B-Q4_K_M and Qwen3.8-27B-UD-Q5_K_M):

| format | FP32 path | FP16 path (max rel. error) |
| :--- | :--- | :--- |
| Q4_K | 0/1024 mismatches | 4.83e-4 |
| Q5_K | 0/1024 mismatches | 4.85e-4 |
| Q6_K | 0/1024 mismatches | 4.85e-4 |
| Q8_0 | 0/1024 mismatches | 4.81e-4 |
| IQ4_NL | 0/1024 mismatches | 4.83e-4 |
| IQ4_XS | 0/1024 mismatches | 4.73e-4 |
| F32 | 0/1024 mismatches | 0.0 – 4.57e-4 |

The FP32 path is bit-exact because all arithmetic is done in FP32 in the
same order as the C reference; the FP16 path is exactly the fp16 rounding
of that value — bounded, deterministic (the same GGUF bytes always produce
the same fp16 weights), and ~3 orders of magnitude below the
re-quantization error of `infer_train quantize` (max 0.003 / 0.053 /
0.119 for Q4_K/Q8_0/NF4, §3).

### Impact on training

* **Initialization.**  Inference-time fine-tuning seeds its fp32 master
  copy of the output head from the fp16-dequantized head
  (`infer_train_finetune_create`), so the master inherits the one-time
  fp16 rounding (≤ 4.88e-4 relative per element).  In dot products the
  per-element roundings have pseudo-random signs and largely cancel —
  consistent with the measured logits correlation ≥ 0.9995 vs llama.cpp —
  so the effect on the loss landscape is a small fixed offset, not
  per-step noise, and it is far below the re-quantization error the
  weights receive on export.
* **Per-update re-rounding.**  After each AdamW update the fp32 head is
  re-synced into the fp16 live head, so the head that actually serves —
  and whose forward pass produces the fine-tuning gradients — is always
  one fp16 rounding away from the fp32 master.  Bounded, but fine-tuning
  effectively runs on a slightly different (deterministic) weight than
  the master.
* **Reproducibility.**  Within InferTrain everything is deterministic, so
  runs are bit-reproducible (the M6 suite is bit-identical to PyTorch
  eager — it trains a randomly initialized fp32 `TrainModel` and is
  unaffected by this gap).  What the gap breaks is *cross-engine*
  reproducibility: a run initialized from FP32-dequantized weights (e.g.
  PyTorch + gguf-py) diverges from an InferTrain run by the fp16 amount.
* **Where it matters.**  Distillation / logit-matching work and any
  workflow that diffs logits across engines at the 1e-4 level.  For those,
  initialize from `dequantize_into_f32` instead.

### Improvement directions

1. **Quantized dot kernels** (already on the v1.1 roadmap, §2): compute
   matmuls directly on the packed Q4_K/Q8_0 weights (llama.cpp-style
   `ggml_vec_dot_q4_K_q8_0`).  This removes the dequantize-on-load fp16
   rounding entirely — the exact quantized values enter the fp32
   accumulation — and keeps the packed weights in RAM (27B: ~20 GB
   instead of ~51 GB of fp16).  Same work item as the AMX repack; it
   closes the precision gap for free.
2. **FP32 fine-tune initialization**: seed the fine-tune fp32 master from
   `dequantize_into_f32` instead of the fp16 head (one-time, at session
   create; negligible cost).
3. **Per-tensor precision option for inference**: dequantize numerically
   sensitive tensors (norms, small projections) to fp32 and keep the large
   matmul weights in fp16.  Full-fp32 weights are only affordable for
   small models on this machine (27B would need ~108 GB).
4. **Tighten the regression test**: `test_dequant_m7` currently checks the
   fp16 output against the FP32 reference with a 5e-3 tolerance; add the
   bit-exact FP32 comparison (`dequantize_into_f32`) so the gap above is
   regression-tested at 0 rather than at the fp16 bound.

## 5. Reproduce

```bash
make tp
pixi run mojo build -I . /tmp/bench27.mojo -Xlinker python/infer_train/_lib/libinfer_train_tp.dylib -o bench
./bench                      # 27B (or adapt the model path)
# llama.cpp CPU baseline (source build with -DGGML_METAL=OFF)
llama-cli -m MODEL -p "The quick brown fox jumps over the lazy dog." \
    -n 16 --temp 0.6 --seed 7 --single-turn --simple-io
```
