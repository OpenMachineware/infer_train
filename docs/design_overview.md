# Inference Engine Design Overview

## Core Philosophy

**Load Model → Build DAG → Traverse DAG for Execution**

The engine separates execution planning from execution itself:

1. **Model Loading**: Parse GGUF, load weights into memory
2. **DAG Construction**: Build execution graph from model architecture
3. **DAG Traversal**: Execute operators in dependency order

This separation enables systematic optimization across all backends.

---

## Two Design Dimensions

The codebase has two independent "layer" concepts that serve different purposes:

### 1. Metal Resource Layers (Hardware Abstraction)

```
┌─────────────────────────────────────────────────────────┐
│  MetalDevice (singleton)                                │
│  - Shared MTLDevice, MTLCommandQueue, MTLLibrary        │
│  - "a single global queue shared by all backends"       │
│  - Prevents Metal state corruption                      │
└─────────────────────────────────────────────────────────┘
                         ↓
┌─────────────────────────────────────────────────────────┐
│  MetalContext (per-backend)                             │
│  - Pipelines, cached buffers, execution state           │
│  - Lightweight, just references shared resources        │
└─────────────────────────────────────────────────────────┘
```

**Why**: Metal device is a singleton (`MTLCreateSystemDefaultDevice`). Creating multiple `CommandQueue`s or `Library`s on the same device corrupts Metal's internal state. llama.cpp explicitly shares these resources at device level.

### 2. Execution Layers (Inference Engine)

```
┌─────────────────────────────────────────────────────────┐
│  Layer 3: Forward Orchestrator                          │
│  - Architecture detection, path selection               │
│  - Expert scheduling, top-level coordination            │
└─────────────────────────────────────────────────────────┘
                         ↓
┌─────────────────────────────────────────────────────────┐
│  Layer 2: Operation Executor                            │
│  - GEMV, Attention, FFN dispatchers                     │
│  - Multi-backend support (CPU/GPU/Remote)               │
└─────────────────────────────────────────────────────────┘
                         ↓
┌─────────────────────────────────────────────────────────┐
│  Layer 1: Buffer Manager                                │
│  - Unified buffer pool, location tracking               │
│  - KV cache management, transfer queue                  │
└─────────────────────────────────────────────────────────┘
```

**Why**: Separation of concerns. Buffer Manager knows WHERE data is, Operation Executor knows HOW to compute, Forward Orchestrator knows WHAT to compute.

---

## Core Architecture

### Three-Layer Execution Design

The inference engine is organized into three distinct layers, each with clear responsibilities:

```
┌─────────────────────────────────────────────────────────┐
│  Layer 3: Forward Orchestrator                          │
│  - Architecture detection (Standard/MQA/FusedQKV)       │
│  - Path selection (GPU vs CPU fallback)                 │
│  - Expert scheduling (local/remote)                     │
│  - Top-level forward coordination                       │
└─────────────────────────────────────────────────────────┘
                         ↓
┌─────────────────────────────────────────────────────────┐
│  Layer 2: Operation Executor                            │
│  - GEMV with auto GPU/CPU fallback                      │
│  - Attention (standard/MQA/fused) dispatcher            │
│  - FFN (SwiGLU/MLP) dispatcher                          │
│  - Output projection with transposed detection          │
│  - Multi-backend support (CPU/GPU/Remote)               │
└─────────────────────────────────────────────────────────┘
                         ↓
┌─────────────────────────────────────────────────────────┐
│  Layer 1: Buffer Manager                                │
│  - Unified buffer pool (create once, reuse across tokens)│
│  - Buffer location tracking (Local GPU/Remote/CPU)      │
│  - Transfer queue for heterogeneous memory              │
│  - KV cache management (GPU kernels for F32→F16)        │
│  - Prefetch policy for expert scheduling                │
└─────────────────────────────────────────────────────────┘
```

### Layer Responsibilities

#### Layer 1: Buffer Manager

**Core Responsibility**: Manage memory locations and data movement.

```rust
enum BufferLocation {
    Local(Rc<Buffer>),        // GPU VRAM
    Remote(RemoteHandle),     // Storage node (AI Max or remote)
    Host(Box<[u8]>),          // CPU memory
}

pub struct BufferManager {
    device: MTLDevice,
    registry: HashMap<BufferId, BufferLocation>,
    local_cache: HashMap<BufferId, Rc<Buffer>>,
    transfer_queue: TransferQueue,
}

impl BufferManager {
    // Core interface
    fn get_or_load(&mut self, id: BufferId) -> Rc<Buffer>;
    fn prefetch(&mut self, ids: &[BufferId]);
    fn pin(&mut self, id: BufferId);      // Keep in local cache
    fn unpin(&mut self, id: BufferId);    // Allow eviction
}
```

**Key Features**:
- Location-aware: Tracks where each tensor resides (GPU VRAM / CPU Memory / Remote)
- CPU buffer management: Manages host memory for CPU fallback operations
- Transfer interface: Moves data between GPU and CPU as needed
- Transfer queue: Manages async data movement between devices
- Prefetch support: Predicts next-layer needs for smoother execution
- Single-user fast path: Direct access for local inference

**CPU Support Details**:

```rust
impl BufferManager {
    /// Get buffer on specific device (GPU or CPU)
    pub fn get_or_load_on(&mut self, id: BufferId, size: usize, device: ComputeDevice) -> BufferRef {
        match device {
            ComputeDevice::Cpu => {
                // CPU buffer pool for fallback operations
                self.host_pool.get(size)
            }
            ComputeDevice::LocalGpu => {
                // Existing GPU logic
                self.get_or_load(id, size)
            }
        }
    }

    /// Transfer buffer between devices
    pub fn transfer(&mut self, id: BufferId, from: ComputeDevice, to: ComputeDevice) {
        // GPU -> CPU: Copy from Metal buffer to host memory
        // CPU -> GPU: Upload to Metal buffer
    }

    /// Get buffer location
    pub fn get_location(&self, id: &BufferId) -> Option<&BufferLocation> {
        self.registry.get(id)
    }
}

/// Unified buffer reference
pub enum BufferRef {
    #[cfg(all(target_arch = "aarch64", feature = "metal"))]
    Gpu(Rc<Buffer>),
    Cpu(Box<[u8]>),
}
```

**Use Cases**:
1. **Pure CPU inference**: All buffers in `host_pool`, no GPU usage
2. **GPU-first with CPU fallback**: Most operations on GPU, unsupported formats fall back to CPU
3. **Mixed execution**: Some layers on GPU, some on CPU (e.g., MQA architectures)
4. **Data transfer**: Automatically move data when operation needs it on different device

#### Layer 2: Operation Executor

**Core Responsibility**: Execute compute operations across backends.

```rust
enum ComputeDevice {
    LocalGpu,
    RemoteGpu,
    Cpu,
}

pub struct OperationExecutor {
    device: ComputeDevice,
    buffer_mgr: Arc<Mutex<BufferManager>>,
}

impl OperationExecutor {
    // Async execution with cross-device dependencies
    fn execute_async(
        &mut self,
        op: &Op,
        inputs: &[BufferId],
        output: BufferId,
    ) -> impl Future<Output = Rc<Buffer>>;

    // Synchronous convenience method
    fn execute(&mut self, op: &Op, inputs: &[BufferId], output: BufferId) -> Rc<Buffer>;
}
```

**Key Features**:
- Multi-backend: CPU (NEON/AVX2), GPU (Metal/CUDA/ROCm), Remote
- Cross-device sync: Handles data dependencies across devices
- Format support: All quantization formats (Q4_K, Q5_K, etc.)
- Fallback logic: Automatic CPU fallback for unsupported formats

#### Layer 3: Forward Orchestrator

**Core Responsibility**: Coordinate forward pass execution and expert scheduling.

```rust
pub struct ForwardOrchestrator {
    buffer_mgr: Arc<Mutex<BufferManager>>,
    executor: OperationExecutor,
    scheduler: ExpertScheduler,
    model: Arc<LlamaModel>,
}

pub trait ExpertScheduler {
    fn should_prefetch(&self, expert_id: u32) -> bool;
    fn eviction_candidate(&self) -> Option<BufferId>;
    fn update_stats(&mut self, expert_id: u32, access_time: Instant);
}
```

**Key Features**:
- Architecture detection: Standard/MQA/MLA/FusedQKV
- Path selection: GPU-first with CPU fallback for unsupported architectures
- Expert scheduling: Static partitioning or dynamic replacement (LRU)
- KV cache lifecycle: Manages cache growth and cleanup

### Design Principles

#### Zero-Cost Abstraction

```rust
// ✅ Use enums for zero-cost dispatch via monomorphization
enum Executor {
    GPU(GPUContext),
    CPU,
}

impl Executor {
    #[inline]
    fn gemv(&self, ...) -> Vec<f32> {
        match self {
            GPU(ctx) => ctx.gemv_gpu(...),
            CPU => self.gemv_cpu(...),
        }
    }
}
```

#### Separation of Concerns

- **Buffer Manager**: Knows WHERE data is, not HOW to compute
- **Operation Executor**: Knows HOW to compute, not WHERE data comes from
- **Forward Orchestrator**: Knows WHAT to compute, delegates the rest

#### Performance-Critical Paths Stay Fast

```rust
// GPU kernels (60-80% of execution time) unchanged by refactoring
gpu_ctx.flash_attn_f16_encode(...);
gpu_ctx.mv_q4_k_encode(...);
```

### Current Implementation Status

The three-layer architecture is implemented with the following approach:

**GPU Path (Fast Path)**:
- `ForwardOrchestrator.forward_gpu()` directly calls `LlamaModel::forward_gpu_native_batched()`
- Preserves batched command buffer optimization (85.8x speedup)
- No overhead from OperationExecutor for GPU operations
- Used for all GPU-supported architectures (Standard attention)

**CPU Path (Fallback)**:
- `ForwardOrchestrator.forward()` calls `LlamaModel::forward()`
- Automatic fallback for unsupported architectures (MQA, MLA)
- OperationExecutor provides future extension point for fine-grained control

**Architecture Detection**:
- Automatically detects Standard/MQA/MLA/FusedQkv
- GPU support status determines execution path
- MQA/MLA models fall back to CPU automatically

---

## Module Organization

```
src/
  orchestrator/           # Layer 3: Forward coordination
    forward.rs            # Main forward pass orchestrator
    architecture.rs       # Architecture detection (Standard/MQA/MLA)
    scheduler.rs          # Expert scheduling (static/dynamic)
    kv_lifecycle.rs       # KV cache management

  executor/               # Layer 2: Operation execution
    mod.rs                # Executor trait and dispatcher
    gemv.rs               # GEMV with GPU/CPU fallback
    attention.rs          # Attention variants (standard/MQA/fused)
    ffn.rs                # FFN variants (SwiGLU/MLP)
    output.rs             # Output projection (transposed detection)
    cpu.rs                # CPU backend (NEON/AVX2)
    gpu.rs                # GPU backend (Metal/CUDA/ROCm)
    remote.rs             # Remote execution (future)

  buffer/                 # Layer 1: Memory management
    manager.rs            # Buffer pool and location tracking
    location.rs           # BufferLocation enum
    transfer.rs           # Transfer queue for heterogeneity
    prefetch.rs           # Prefetch policy
    kv_cache.rs           # KV cache GPU management

  dag/                    # DAG construction and optimization
    builder.rs            # Model architecture → DAG
    optimizer.rs          # Platform-independent fusion
    executor.rs           # DAG traversal and dispatch

  model/                  # GGUF parsing, model architectures
    gguf.rs               # File format parser
    weights.rs            # Weight storage
    llama.rs              # Llama architecture
    moe.rs                # MoE layers
    mamba.rs              # Mamba/SSM layers

  quant/                  # Quantization types and kernels
    k_quant.rs            # K-quant family (Q2_K-Q6_K)
    iq_quant.rs           # IQ series
    standard.rs           # Q4_0, Q4_1, Q5_0, Q5_1, Q8_0

  platform/               # CPU/GPU feature detection
    cpu.rs                # ARM/x86 detection
    gpu.rs                # Metal/CUDA/ROCm detection
```

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

### Heterogeneous Architectures

The three-layer design supports future heterogeneous deployments:

#### Scenario 1: Separate Machines (PC + AI Max)

**Topology**:
- Compute node: Consumer GPU (high FLOPS, limited VRAM)
- Storage node: AMD AI Max (large unified memory, lower FLOPS)
- Communication: Network (1-10ms latency, ~3GB/s USB4) or PCIe

**Use Case**: MoE models with more experts than fit in GPU VRAM.

**Buffer Manager Role**:
- Tracks expert locations (local GPU vs remote)
- Manages transfer queue for expert loading
- Prefetch strategy based on token prediction

#### Scenario 2: Single Machine (AI Max motherboard + GPU card)

**Topology**:
- AMD AI Max motherboard with unified memory
- Consumer GPU card via PCIe or special interconnect
- Communication: Memory-VRAM direct (potentially 16-32GB/s)

**Advantage**: Lower latency, higher bandwidth for expert swapping.

**Buffer Manager Role**:
- Same as Scenario 1, but with faster transfers
- Potential for finer-grained expert scheduling

#### Expert Scheduling Strategy

For MoE models (e.g., DeepSeek-V3 with 256 experts, ~250MB each in Q4_K):

**Static Partitioning**:
- Analyze historical access patterns
- Pin most-used experts in GPU VRAM
- Store remaining experts in remote memory

**Dynamic Replacement**:
- LRU-based eviction when VRAM full
- Prefetch predicted experts based on current token
- Latency budget: Must complete transfer before expert needed

**Buffer Manager Abstraction**:
```rust
impl BufferManager {
    fn ensure_available(&mut self, expert_id: u32) -> Rc<Buffer> {
        if let Some(buf) = self.local_cache.get(&expert_id) {
            return buf.clone();
        }
        // Trigger transfer, possibly evict another expert
        self.transfer_and_evict(expert_id)
    }
}
```

### Feature Detection

- Detect CPU features at startup: arch, vendor, SIMD capabilities
- Detect GPU features at startup: vendor, memory, supported operations
- Zero runtime overhead: selection happens once at init

---

## Design Comparison

### Metal Resource Layers vs Execution Layers

| Aspect | Metal Resource Layers | Execution Layers |
|--------|----------------------|------------------|
| **Purpose** | Hardware abstraction | Inference orchestration |
| **Scope** | Metal/GPU only | All backends (CPU/GPU/Remote) |
| **Problem Solved** | Prevent state corruption | Enable optimization & heterogeneity |
| **Inspiration** | llama.cpp | Custom design |

### Metal Resource Layers: Why Two Layers?

**Problem**: Creating multiple `MetalContext` instances caused NaN on Layer 1 operations.

**Root Cause**: `MTLCreateSystemDefaultDevice()` returns the same device object. Creating multiple `CommandQueue`s or `Library`s on the same device corrupts Metal's internal state.

**llama.cpp's Solution**:
```c
struct ggml_metal_device {
    id<MTLDevice> mtl_device;
    id<MTLCommandQueue> mtl_queue;  // "shared by all Metal backends"
    ggml_metal_library_t library;
};

struct ggml_metal {
    ggml_metal_device_t dev;  // Just a reference
    // Only state, not shared resources
};
```

**Our Solution**:
```rust
// Singleton, shared across all contexts
pub struct MetalDevice {
    device: Device,
    queue: CommandQueue,
}

// Per-backend, lightweight
pub struct MetalContext {
    // Can reference MetalDevice::global()
    pipelines: Vec<Pipeline>,
    buffers: RefCell<...>,
}
```

**Benefits**:
1. **Correctness**: No Metal state corruption
2. **Efficiency**: Compile shaders once, reuse across contexts
3. **Clarity**: Explicit separation of shared vs. per-context state

### Execution Layers: Why Three Layers?

**Layer 1 - Buffer Manager**:
- Manages WHERE data is (GPU VRAM / CPU Memory / Remote)
- Handles data movement between devices
- Single-user fast path for local inference

**Layer 2 - Operation Executor**:
- Knows HOW to compute (GPU kernel vs CPU fallback)
- Automatic format detection and fallback
- Cross-device synchronization

**Layer 3 - Forward Orchestrator**:
- Knows WHAT to compute (which layers, which experts)
- Architecture detection (Standard/MQA/MLA)
- Expert scheduling for MoE models

**Benefits**:
1. **Flexibility**: Each layer can be optimized independently
2. **Extensibility**: Easy to add new backends or architectures
3. **Heterogeneity**: Natural support for multi-device scenarios

---

## Current Implementation Status

### Metal Resource Layers

| Component | Status |
|-----------|--------|
| MetalDevice singleton | ✅ Implemented |
| MetalContext Clone | ✅ Implemented (workaround) |
| Full two-layer refactoring | 📋 Planned |

### Execution Layers

### Phase 1: Baseline Performance

**CPU Kernels**: ✅ Complete
- K-quant family: Q2_K-Q6_K (100%+ llama.cpp)
- IQ series: 9 formats (100%+ llama.cpp)
- Standard: Q4_0, Q4_1, Q5_0, Q5_1, Q8_0
- Attention, Norm, Top-K, Argsort: 100%+

**GPU Kernels (Metal)**: 🔧 Debugging
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

**Architecture Refactoring**: 📋 Planned
- Three-layer design finalized
- Heterogeneous support designed into Layer 1
- Implementation starting with Buffer Manager

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

### Quantization Advantage

| Quantization | VRAM Reduction | Concurrent Users |
|--------------|----------------|------------------|
| FP16 | 1x (baseline) | 1x |
| Q8_0 | 2x | 2x |
| Q5_K | 3.2x | 3x |
| Q4_K | 4x | 4x |

**Server Market Strategy**: Q4_K can run 4x more concurrent requests than FP16, enabling higher throughput at lower cost.

---

## Next Steps

### Phase 1: Fix Immediate Issues (P0)

1. **Fix GPU kernel bugs**
   - GEMV correctness for all quantization formats
   - KV cache update kernel validation
   - Command buffer batching

2. **Achieve baseline performance**
   - Target: 167 t/s on 7B Q4_K
   - GPU must be faster than CPU (≥10%)

3. **Validate architecture coverage**
   - Standard LLaMA ✅
   - MQA/MLA (DeepSeek) - CPU fallback
   - Fused QKV (Qwen)

### Phase 2: Architecture Refactoring (P1)

1. **Implement Layer 1: Buffer Manager**
   - Unified buffer pool with reuse
   - Buffer location tracking
   - Transfer queue interface (future heterogeneity)
   - Expected speedup: 5-10%

2. **Implement Layer 2: Operation Executor**
   - Consolidate GPU/CPU fallback logic
   - Multi-backend interface

3. **Implement Layer 3: Forward Orchestrator**
   - Architecture detection in single place
   - Expert scheduler interface (future MoE)

### Phase 3: DAG-Based Optimization (P2)

1. **Implement DAG Builder**
   - Model architecture → compute graph
   - Separate execution planning from execution

2. **Add Platform-Independent Fusion**
   - SiLU + Mul, Residual inlining
   - Expected speedup: +20%

3. **Add Backend-Specific Fusion**
   - Metal-specific optimizations
   - Expected speedup: +5%

---

**Created**: 2026-09-22
**Updated**: 2026-10-19
**Status**: Phase 1 (Baseline) - Metal resource layers implemented, execution layers designed
