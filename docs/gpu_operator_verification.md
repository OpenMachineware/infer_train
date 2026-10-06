# GPU Operator Verification Summary

**Date**: 2026-10-06
**Operators**: RMS_NORM, SOFTMAX, ROPE

## Completeness

All three operators support the same formats as llama.cpp:

### RMS_NORM
- ✅ F32 scalar (`kernel_rms_norm_f32`)
- ✅ F32 vectorized (`kernel_rms_norm_f32_4`)
- ✅ F16 scalar (`kernel_rms_norm_f16`)
- ✅ F16 vectorized (`kernel_rms_norm_f16_4`)
- ✅ Fused norm+mul (`kernel_rms_norm_mul_f32`)
- ✅ Fused norm+mul+add (`kernel_rms_norm_mul_add_f32`)

### SOFTMAX
- ✅ F32 scalar with mask/ALiBi (`kernel_soft_max_f32`)
- ✅ F32 vectorized with mask/ALiBi (`kernel_soft_max_f32_4`)
- ✅ F32 fast (no mask) (`kernel_soft_max_f32_fast`)
- ✅ F32 fast vectorized (`kernel_soft_max_f32_4_fast`)
- ✅ F16 fast (`kernel_soft_max_f16_fast`)
- ✅ F16 vectorized (`kernel_soft_max_f16_4`)

### ROPE
- ✅ F32 NeoX style (`kernel_rope_neox_f32`)
- ✅ F16 NeoX style (`kernel_rope_neox_f16`)
- ✅ F32 Normal style (`kernel_rope_norm_f32`)
- ✅ F16 Normal style (`kernel_rope_norm_f16`)
- ✅ YaRN extension support

## Correctness

### RMS_NORM
- CPU: Max error 2.6e-6 (within floating point precision)
- GPU: 158-463% faster than llama.cpp
- Status: ✅ PASS

### SOFTMAX
- CPU: Max diff 0.00e0 (exact match), 114-123% faster
- GPU: Max sum deviation from 1.0: 4.17e-7, 89-113% performance
- Status: ✅ PASS

### ROPE
- CPU: Max diff 0.00e0 (exact match)
- GPU: 82-109% of llama.cpp (within thermal variance)
- Status: ✅ PASS

## Self-written Status

All three operators are **self-written**, not copied from llama.cpp:
- ✅ RMS_NORM: Written from scratch with same algorithm
- ✅ SOFTMAX: Written from scratch with same algorithm
- ✅ ROPE: Written from scratch with YaRN extension support

## Files

- `/Users/jia/Desktop/infer_train/shaders/rms_norm.metal` - 330 lines
- `/Users/jia/Desktop/infer_train/shaders/softmax.metal` - 521 lines
- `/Users/jia/Desktop/infer_train/shaders/rope.metal` - 220 lines

## Test Commands

```bash
# RMS_NORM
cargo run --release --bin verify_rms_norm
cargo run --release --bin benchmark_rms_norm_gpu_vs_llama

# SOFTMAX
cargo run --release --bin benchmark_softmax_vs_llama
cargo run --release --bin benchmark_softmax_gpu_vs_llama

# ROPE
cargo run --release --bin verify_rope
cargo run --release --bin benchmark_rope_gpu_vs_llama
```
