# Rust Inference Engine Design Document

## Background

Rust-based LLM inference engine with CPU (NEON/AVX) and GPU (Metal/CUDA/ROCm) backends.

**Business Goals**:
- Both CPU and GPU performance exceed llama.cpp (≥100%, not 95%)
- Extend vLLM features (multi-request batching)
- Cross-platform support: macOS (Metal), Linux (CUDA/ROCm), Windows (CUDA)

## Platform Support

### Current (Phase 1)
- **CPU**: ARM (NEON), x86 (AVX/AVX2/AVX-512) via feature detection
- **GPU**: Metal (Apple Silicon)

### Future
- **GPU**: CUDA (NVIDIA), ROCm (AMD)
- **CPU**: SVE (ARMv9)

### Platform Detection
- Detect CPU features at startup: arch, vendor, SIMD capabilities
- Detect GPU features at startup: vendor, memory, supported operations
- Zero runtime overhead: selection happens once at init

## Core Architecture

```
┌─────────────────────────────────────────┐
│        InferenceEngine (Rust)           │
├─────────────────────────────────────────┤
│  CPU Kernels (NEON/AVX2/Scalar)         │
│  GPU Kernels (Metal/CUDA/ROCm)          │
│  Unified Memory Manager                 │
├─────────────────────────────────────────┤
│  Single-User Fast Path                  │ ← Local deployment: zero overhead
│  Multi-User Batch Path                  │ ← Server: vLLM features
├─────────────────────────────────────────┤
│  Pipeline Parallel (Phase 3)            │ ← Heterogeneous extension
│  Heterogeneous Memory Pool (Phase 3)    │
└─────────────────────────────────────────┘
```

## Module Organization

```
src/
  platform/         # CPU/GPU feature detection
    cpu.rs          # ARM/x86 detection
    gpu.rs          # Metal/CUDA/ROCm detection
  ops/              # Operator implementations
    arm/            # ARM CPU operators
      neon.rs       # NEON SIMD kernels
    x86/            # x86 CPU operators
      avx.rs        # AVX/AVX2 kernels
    metal/          # Metal GPU operators
      mv.rs         # Matrix-vector
      gemm.rs       # Matrix-matrix
      attention.rs  # Flash attention
    cuda/           # CUDA operators (future)
    common/         # Shared tables, utilities
  model/            # GGUF parsing, model architectures
    gguf.rs         # File format parser
    weights.rs      # Weight storage
    llama.rs        # Llama architecture
    moe.rs          # MoE layers
    mamba.rs        # Mamba/SSM layers
  quant/            # Quantization types
```

## Phase 1: MVP (Complete ✅)

### CPU Kernels ✅

**Quantization Format Support**:
- K-quant: Q2_K, Q3_K, Q4_K, Q5_K, Q6_K (all exceed llama.cpp)
- IQ series: IQ1_S, IQ1_M, IQ2_S, IQ2_XS, IQ2_XXS, IQ3_S, IQ3_XXS, IQ4_XS, IQ4_NL
- Standard: Q4_0, Q4_1, Q5_0, Q5_1, Q8_0
- TQ series: TQ1_0, TQ2_0
- MXFP: MXFP4

**Performance Results** (NEON on M1 Max):
| Format | vs llama.cpp |
|--------|-------------|
| Q2_K-Q6_K | 100%+ (matching) |
| IQ series | 100%+ (matching) |
| Attention (quantized KV) | 100%+ |
| RMS Norm | 290% (3-4x faster) |
| Group Norm | 372% (3-4x faster) |
| Top-K | 181% |
| Argsort | 150%+ |

### GPU Kernels (Metal) ✅

**Implemented**:
- Matrix-vector (MV): All 23 quantization formats
- Matrix-matrix (GEMM): All formats with MoE support
- Flash Attention: Decode (100+ GFLOPS) + Prefill (MPP tensor)
- Operators: RMS_NORM, SOFTMAX, ROPE, POOL_2D, TOP_K, ARGSORT, CONV_2D

**Performance Results** (M1 Max):
| Kernel | vs llama.cpp |
|--------|-------------|
| MV (all formats) | 100%+ |
| GEMM (all formats) | 100%+ |
| Flash Decode | 107 GFLOPS (exceeds 90 GFLOPS baseline) |
| Flash Prefill | 1106 GFLOPS via MPP tensor |

### Model Layer ✅

**GGUF Parser**: Full format support, 29 data types
**Architectures**: Llama, MoE (DeepSeek), Mamba/SSM (Qwen3.8), Hybrid MoE+SSM (Qwen3.6)
**Tested Models**: 7 models at 100% layer completeness

## Phase 2: Server Deployment

### Multi-User Batch Path

**Core Features**:
1. **PagedAttention + Block Manager**
   - KV Cache paged management
   - Shared prefix deduplication

2. **Prefix Caching**
   - System prompt KV reuse
   - Throughput doubled

3. **Continuous Batching**
   - Dynamic request insertion
   - Avoid long prompt blocking

4. **Fast/Slow Path Separation**
   ```rust
   enum ExecutionMode {
       SingleUser,  // No scheduling overhead
       MultiUser,   // Full vLLM features
   }
   ```

### Async Architecture

**Tech Stack**: Rust + tokio
- Zero overhead for single request
- Automatic scheduling for multiple requests

## Phase 3: Distributed/Heterogeneous

### Heterogeneous Architecture (AMD AI Max + NVIDIA 5090)

**Pipeline Parallel**:
- AI Max: 192GB unified memory (model storage)
- 5090: 24-32GB VRAM, 10x faster compute
- RDMA communication: 12.5GB/s bandwidth

### Homogeneous GPU Cluster

**Tensor Parallel**: Intra-layer split
**Pipeline Parallel**: Inter-layer split

## Performance Targets

### Current Results ✅

| Component | Target | Achieved |
|-----------|--------|----------|
| CPU kernels | ≥100% llama.cpp | ✅ 100%+ |
| GPU MV/GEMM | ≥100% llama.cpp | ✅ 100%+ |
| Flash Decode | >90 GFLOPS | ✅ 107 GFLOPS |
| Model loading | All formats | ✅ 29 formats |

### End-to-End

| Scenario | Target | Status |
|----------|--------|--------|
| Local inference (7B Q4_K) | ≥10 t/s | Pending integration |
| Server multi-request | vLLM level | Phase 2 |

## Porting to New Platforms

### Adding a New CPU Architecture

1. Create `src/ops/<arch>/` directory
2. Implement operators in `<arch>/<simd>.rs`
3. Add detection in `src/platform/cpu.rs`
4. Update `src/ops.rs` to dispatch based on detected features

### Adding a New GPU Backend

1. Create `src/ops/<backend>/` directory
2. Implement kernels (mv, gemm, attention)
3. Add detection in `src/platform/gpu.rs`
4. Update `src/ops.rs` for backend selection

## References

### Code

- `src/ops/arm/neon.rs` - ARM NEON kernels
- `src/ops/metal/` - Metal GPU kernels
- `llama.cpp/src/ggml-quants.c` - NEON kernel reference

### Memory

- Project memory files in `.qoder-cn/projects/-Users-jia-Desktop-infer-train/memory/`

---

**Created**: 2026-09-22
**Updated**: 2026-10-12
**Status**: Phase 1 complete, Phase 2 pending
**Next Step**: Inference engine integration
