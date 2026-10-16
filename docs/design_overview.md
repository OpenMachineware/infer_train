# Inference Engine Design Overview

## Core Philosophy

**Load Model → Build DAG → Traverse DAG for Execution**

The engine separates execution planning from execution itself:

1. **Model Loading**: Parse GGUF, load weights into memory
2. **DAG Construction**: Build execution graph from model architecture
3. **DAG Traversal**: Execute operators in dependency order

This separation enables systematic optimization across all backends.

---

## Optimization Strategy

### Three-Phase Approach

| Phase | Description | Expected Gain | Status |
|-------|-------------|---------------|--------|
| **1. Baseline** | DAG traversal, no fusion | Match llama.cpp (~167 t/s) | Current |
| **2. Platform-Independent Fusion** | DAG-level optimizations | +20% | Future |
| **3. Backend-Specific Fusion** | Metal/CUDA/ROCm-specific | +5% | Future |

**Key Principle**: Each phase builds on the previous. No skipping to fusion before baseline is correct and performant.

### Why DAG-Level Fusion First?

llama.cpp does per-backend fusion (limited gains). Our approach:

- **Platform-independent fusion** applies to all backends (CPU NEON, GPU Metal/CUDA/ROCm)
- **Backend-specific fusion** adds Metal/CUDA/ROCm optimizations on top
- **Result**: Same optimization logic works across all hardware

### Fusion Opportunities

**Platform-Independent (DAG-Level)**:
- SiLU + Mul (FFN gate)
- Residual Add (inline into previous op)
- RMS Norm + input caching

**Backend-Specific**:
- Metal: SIMDgroup operations, tile memory
- CUDA: Tensor cores, cooperative groups
- ROCm: Wavefront operations

---

## Platform Support

### Current (Phase 1)

- **CPU**: ARM (NEON), x86 (AVX/AVX2/AVX-512)
- **GPU**: Metal (Apple Silicon)

### Future

- **GPU**: CUDA (NVIDIA), ROCm (AMD)
- **CPU**: SVE (ARMv9)

### Feature Detection

- Detect CPU features at startup: arch, vendor, SIMD capabilities
- Detect GPU features at startup: vendor, memory, supported operations
- Zero runtime overhead: selection happens once at init

---

## Core Architecture

```
┌─────────────────────────────────────────┐
│        InferenceEngine (Rust)           │
├─────────────────────────────────────────┤
│  Model Loader (GGUF Parser)             │
│  DAG Builder (Architecture → Graph)     │
│  DAG Executor (Traverse + Dispatch)     │
├─────────────────────────────────────────┤
│  CPU Kernels (NEON/AVX2/Scalar)         │
│  GPU Kernels (Metal/CUDA/ROCm)          │
│  Unified Memory Manager                 │
├─────────────────────────────────────────┤
│  Single-User Fast Path                  │ ← Zero overhead
│  Multi-User Batch Path                  │ ← vLLM features
└─────────────────────────────────────────┘
```

---

## Module Organization

```
src/
  platform/         # CPU/GPU feature detection
    cpu.rs          # ARM/x86 detection
    gpu.rs          # Metal/CUDA/ROCm detection
  dag/              # NEW: DAG construction and optimization
    builder.rs      # Model architecture → DAG
    optimizer.rs    # Platform-independent fusion
    executor.rs     # DAG traversal and dispatch
  ops/              # Operator implementations
    arm/            # ARM CPU operators
    x86/            # x86 CPU operators
    metal/          # Metal GPU operators
    cuda/           # CUDA operators (future)
    common/         # Shared utilities
  model/            # GGUF parsing, model architectures
    gguf.rs         # File format parser
    weights.rs      # Weight storage
    llama.rs        # Llama architecture
    moe.rs          # MoE layers
    mamba.rs        # Mamba/SSM layers
  quant/            # Quantization types
```

---

## Current Status

### Phase 1: Baseline Performance

**CPU Kernels**: ✅ Complete
- K-quant family: Q2_K-Q6_K (100%+ llama.cpp)
- IQ series: 9 formats (100%+ llama.cpp)
- Standard: Q4_0, Q4_1, Q5_0, Q5_1, Q8_0
- Attention, Norm, Top-K, Argsort: 100%+

**GPU Kernels (Metal)**: ⚠️ Debugging
- MV/GEMM: All 23 formats implemented, performance bugs pending
- Flash Attention: Decode (107 GFLOPS), Prefill (1106 GFLOPS)
- Operators: RMS_NORM, SOFTMAX, ROPE, etc.

**Model Layer**: ✅ Complete
- GGUF Parser: 29 data types
- Architectures: Llama, MoE, Mamba/SSM, Hybrid
- Tested: 7 models at 100% layer completeness

**End-to-End**: 🔧 In Progress
- Current: 1.09 t/s (CPU fallback for stability)
- Target: 167 t/s (llama.cpp baseline)
- Blockers: GPU GEMV kernel bugs, command buffer batching

---

## Performance Targets

### Baseline (Phase 1)

| Component | Target | Status |
|-----------|--------|--------|
| CPU kernels | ≥100% llama.cpp | ✅ 100%+ |
| GPU kernels | ≥100% llama.cpp | 🔧 Debugging |
| End-to-end (7B Q4_K) | ≥167 t/s | 🔧 1.09 t/s |

### Platform-Independent Fusion (Phase 2)

| Optimization | Expected Gain |
|--------------|---------------|
| SiLU+Mul fusion | +5% |
| Residual inlining | +3% |
| RMS Norm caching | +2% |
| **Total** | **+20%** |

### Backend-Specific Fusion (Phase 3)

| Backend | Optimization | Expected Gain |
|---------|--------------|---------------|
| Metal | SIMDgroup, tile memory | +5% |
| CUDA | Tensor cores, cooperative groups | +5% |
| ROCm | Wavefront operations | +5% |

---

## Next Steps

1. **Fix GPU kernel bugs** (GEMV, KV update, command buffer batching)
2. **Achieve baseline performance** (167 t/s on 7B Q4_K)
3. **Implement DAG builder** (separate execution planning from execution)
4. **Add platform-independent fusion** (DAG optimizer)
5. **Add backend-specific fusion** (Metal optimizer)

---

**Created**: 2026-09-22
**Updated**: 2026-10-16
**Status**: Phase 1 (Baseline) - Debugging GPU kernels
