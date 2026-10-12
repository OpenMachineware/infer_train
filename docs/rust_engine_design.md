# Rust Inference Engine Design Document

## Background

Mojo CPU path was abandoned due to loop overhead issues (11-16x vs C), deciding to rewrite CPU inference engine in Rust.

**Business Goals**:
- Both CPU and GPU performance exceed llama.cpp (≥100%, not 95%)
- Extend vLLM features (multi-request batching)
- Find acquaintances to promote, try using in their companies

**Platform Scope**: macOS only (Metal), other platforms CPU fallback

## Core Architecture

```
┌─────────────────────────────────────────┐
│        InferenceEngine (Rust)           │
├─────────────────────────────────────────┤
│  CPU Kernels (NEON/AVX2/Scalar)         │
│  GPU Kernels (Metal/CUDA)               │
│  Unified Memory Manager                 │
├─────────────────────────────────────────┤
│  Single-User Fast Path                  │ ← Local deployment: zero overhead
│  Multi-User Batch Path                  │ ← Server: vLLM features
├─────────────────────────────────────────┤
│  Pipeline Parallel (Phase 3)            │ ← Heterogeneous extension
│  Heterogeneous Memory Pool (Phase 3)    │
└─────────────────────────────────────────┘
```

## Phase 1: MVP (Local Single Machine)

### Goals
- CPU NEON kernels performance ≥ llama.cpp
- Metal GPU kernels usable
- Single request inference usable

### CPU Kernels

**Quantization Format Support**:
- K-quant: Q2_K, Q3_K, Q4_K, Q5_K, Q6_K
- IQ series: IQ1_S, IQ1_M, IQ2_S, IQ2_XS, IQ2_XXS, IQ3_S, IQ3_XXS, IQ4_XS, IQ4_NL
- Quantized weights + FP16 activations

**Key Optimizations** (from Mojo experience):
1. **Q8_K Quantization Reuse**:
   - Quantize activations to Q8_K once, reuse for multiple projections (gate+up, QKV)
   - Quantization overhead: ~10µs vs multiple quantizations ~100µs

2. **Threaded Kernel**:
   - Threshold: N >= 2048 (threading only beneficial above this)
   - Batch prefill: Use threaded kernel when M > 1
   - Decode: Use single-threaded kernel when M = 1

3. **IQ4_XS Processing**:
   - Non-K-quant format, but uses Q8_K vec_dot kernel
   - Block size: 136 bytes
   - Dispatch: ggml_type == 23

**Performance Targets** (reference Mojo benchmark):
| Format | GFLOPS | vs llama.cpp |
|--------|--------|-------------|
| Q4_K | 117-121 | +5-21% |
| Q5_K | ~95 | +10% |
| Q6_K | ~85 | +10% |
| IQ4_XS | ~70 | +21% |
| IQ3_S | ~20 | +11% |

**Verification Methodology**:
- **Bottom-up**: Each kernel ≥100% llama.cpp before building up
- **Assembly comparison**: Ensure same SIMD instructions generated
- **Test data**: Use llama.cpp verbatim test data (no simplification)

### GPU Kernels (Metal)

**Tech Stack**: Rust + `metal-rs`

**Kernel List**:
- Matmul (quantized weights + FP16 activations)
- Flash Attention (prefill optimized)
- RMSNorm, Softmax
- RoPE

**Performance Targets**:
- Prefill: Close to llama.cpp (Metal path)
- Decode: GPU slower than CPU is normal (for small batches), but acceptable

**Unified Memory**:
- CPU and GPU share memory, zero-copy
- macOS unique advantage

### Execution Mode

**Single-User Fast Path**:
```rust
fn forward_fast(&mut self, tokens: &[Token]) -> Logits {
    // No scheduler, no locks, no caching checks
    // Contiguous KV Cache (not paged)
    // Direct execution, close to llama.cpp overhead
}
```

**Design Principle**:
- Local deployment performance ≥ llama.cpp
- Don't sacrifice single-user performance for vLLM features

## Phase 2: Server Deployment

### Multi-User Batch Path

**Core Features**:
1. **PagedAttention + Block Manager**
   - KV Cache paged management, similar to OS virtual memory
   - Shared prefix (system prompt) automatic deduplication
   - Avoid wasting pre-allocated `max_context × n_requests`

2. **Prefix Caching**
   - Multiple users share system prompt → KV Cache reuse
   - Throughput doubled (in real scenarios)
   - Implementation: KV block reference counting + LRU eviction

3. **Continuous Batching**
   - Don't wait for all requests to complete, insert new requests when slots available
   - Avoid "long prompt blocks all short requests"

4. **Fast/Slow Path Separation**
   ```rust
   enum ExecutionMode {
       SingleUser,      // Local deployment: no scheduling overhead
       MultiUser,       // Server: full vLLM features
   }
   ```

**Single-User Overhead Analysis**:
| Feature | Single-User Overhead | Reason |
|---------|---------------------|--------|
| PagedAttention | 2-5% | Pointer indirection vs contiguous memory |
| Continuous Batching | <1% | If branch, CPU prediction accurate |
| Prefix Caching | <1% | LRU lookup |

**Goal**: Single-user mode performance ≥ llama.cpp

### Async Architecture

**Tech Stack**: Rust + tokio

**Design**:
- async/await + tokio runtime
- Zero overhead for single request (fast path)
- Automatic scheduling for multiple requests

**vs Mojo**:
- Mojo pthread: ms-level scheduling overhead
- Rust tokio: µs-level scheduling overhead
- Solves "single request thread pool eats benefits" problem

### Other vLLM Features (Optional)

**Chunked Prefill** (Phase 2.5):
- Execute long prompt in chunks (32 tokens per chunk)
- Interleave decode requests
- Avoid long prompt blocking

**KV Cache Quantization** (Phase 2.5):
- Q8/Q6/Q4 quantized KV Cache
- Significant benefit in memory bandwidth sensitive scenarios
- Accuracy loss needs trade-off

**Speculative Decoding** (Phase 3):
- Small model predicts, large model verifies
- Decoding speed ×3 (high accuracy scenarios)
- Requires: two models + verification logic

**Attention Sink** (Phase 3):
- Retain first few tokens' KV during streaming generation
- Solves rolling window attention collapse

## Phase 3: Distributed/Heterogeneous

### Heterogeneous Architecture (AMD AI Max + NVIDIA 5090)

**Scenario**:
- AI Max: 192GB unified memory (fits entire large model)
- 5090: 24-32GB VRAM, but 10x faster compute
- Combination: AI Max as memory pool, 5090 as compute

**Pipeline Parallel + Local Weights**:
```
AI Max (Memory Server)       5090 (Compute Node)
┌──────────────────┐         ┌──────────────────┐
│ 192GB Memory Pool│ ←RDMA──→│ 24GB VRAM        │
│ Activation/KV    │ Zero    │ All Computation  │
│ Model Weights    │ Copy    │ Read from AI Max │
└──────────────────┘         └──────────────────┘
```

**Communication Methods**:
1. **Same-machine PCIe** (Optimal):
   - AI Max motherboard with 5090
   - Unified memory + PCIe = zero-copy

2. **RDMA Network**:
   - GPU Direct RDMA
   - Bandwidth: ~12.5GB/s (100Gbps)
   - Latency: <10µs

**Scheduling Strategy**:
```rust
struct PipelineSchedule {
    layers_on_device_a: Range<usize>,  // AI Max: 0-15
    layers_on_device_b: Range<usize>,  // 5090: 16-31
    activation_channel: Channel<Tensor>,  // KB-level
}
```

**Performance Analysis**:
- Activation transfer: 8KB per token
- Network latency: negligible (µs-level)
- Computation: dominant factor (ms-level)

### Homogeneous GPU Cluster

**Benchmark vLLM**:
- Tensor Parallel (intra-layer split)
- Pipeline Parallel (inter-layer split)
- RDMA + GPU Direct communication

**RPC Mode** (Not Recommended):
- Traditional network transfer weights too slow
- 40GB × 10Gbps = 32 seconds
- Only suitable for occasional calls, not continuous inference

## Technology Selection

### Rust GPU Libraries

| Platform | Library | Performance vs Mojo/C++ | Maturity |
|----------|---------|-------------------------|----------|
| Metal | metal-rs | = Mojo | Community library, sufficient |
| CUDA | cust, rust-cuda | = C++ | Mature |
| ROCm | rocm-rs | = C++ | Weak, may need FFI |

**Key Points**:
- GPU kernel performance independent of host language (same shader/PTX)
- API call overhead negligible (µs-level vs ms-level kernel)
- Performance not an issue, ecosystem maturity is

### Quantization Kernel Implementation

**Reference**: llama.cpp NEON kernels

**Verification Method**:
1. Copy llama.cpp test data (no simplification)
2. Compare output function by function
3. Assembly-level SIMD instruction comparison

**Key Code Paths** (port from Mojo):
- `ggml_vec_dot_q2_k_q8_k` → Rust NEON
- `ggml_vec_dot_q4_k_q8_k` → Rust NEON
- `quantize_row_q8_k` → Rust NEON

## Performance Target Summary

### CPU Kernels

| Format | Mojo GFLOPS | Target Rust GFLOPS | vs llama.cpp |
|--------|-------------|-------------------|-------------|
| Q4_K | 117-121 | ≥120 | ≥105% |
| Q5_K | ~95 | ≥95 | ≥110% |
| Q6_K | ~85 | ≥85 | ≥110% |
| IQ4_XS | ~70 | ≥70 | ≥121% |

### End-to-End

| Scenario | Target | vs llama.cpp |
|----------|--------|-------------|
| Local single request (7B Q4_K) | ≥10 t/s | ≥100% |
| Server multi-request | Throughput priority | vLLM level |

### GPU

| Scenario | Target | vs llama.cpp |
|----------|--------|-------------|
| Prefill | 300+ t/s | ~100% |
| Decode | 70% of CPU | Acceptable (small batch) |

## Development Priority

**Phase 1** (2-4 weeks):
- [ ] CPU NEON kernels (quantization formats)
- [ ] Metal GPU kernels
- [ ] Single-user fast path
- [ ] Performance verification ≥ llama.cpp

**Phase 2** (2-4 weeks):
- [ ] Multi-user batch path
- [ ] PagedAttention
- [ ] Prefix Caching
- [ ] Continuous Batching

**Phase 3** (As needed):
- [ ] Pipeline Parallel
- [ ] Heterogeneous memory pool
- [ ] RDMA communication

## References

### Code

- `src/core/ops/quantized/qweight.mojo` - Quantized projection dispatch
- `src/core/ops/cpu/matmul_q8k_threaded.mojo` - Threaded kernel, fused projection
- `src/core/ops/attention/mha.mojo` - Attention implementation, QKV fused
- `llama.cpp/src/ggml-quants.c` - NEON kernel reference

### Memory

- `Q4_K Final Performance` - 117-121 GFLOPS, +5-21% vs llama.cpp
- `IQ Series Final Status` - 5 formats exceed llama.cpp 11-67%
- `Mojo Loop Overhead` - 11-16x vs C, abandoned CPU path
- `vLLM Design Strategy` - vLLM feature priorities
- `Heterogeneous Distributed Architecture` - Heterogeneous architecture design

### Methodology

- **Bottom-up verification**: Each layer ≥100% llama.cpp before building up
- **Assembly comparison**: Ensure SIMD instructions match
- **Test data**: llama.cpp verbatim data, no simplification

---

**Created**: 2026-09-22
**Status**: Design complete, awaiting implementation
**Next Step**: Phase 1 CPU kernels
