# GPU Flash Attention Design Document

## Overview

This document describes the implementation of GPU Flash Attention for the infer_train project, following llama.cpp's template-based approach.

## Architecture

### Two Kernels

1. **Vec Kernel (kernel_flash_attn_ext_vec)**: For decode (batch=1) case
   - Each simdgroup handles Q=1 queries and C=256 cache items
   - Uses online softmax algorithm
   - Efficient for small batches (decode scenario)

2. **Main Kernel (kernel_flash_attn_ext)**: For prefill (batch>1) case
   - Uses simdgroup_matrix for efficient computation
   - Handles multiple queries in parallel
   - Optimized for large batches (prefill scenario)

### Template Parameters

#### Vec Kernel Template Signature

```metal
template<
    typename q4_t,    // query type in shared memory (half4)
    typename k4_t,    // key type in shared memory (half4)
    typename v4_t,    // value type in shared memory (half4)
    typename qk_t,    // Q*K type (float)
    typename s_t,     // softmax type (float)
    typename s4_t,    // softmax type 4 (float4)
    typename o4_t,    // output type (float4)
    typename kd4_t,   // key type in device memory
    short nl_k,       // number of 4-element groups per block (QK/4 = 8)
    void (*deq_k_t4)(device const kd4_t *, short, thread k4_t &),
    typename vd4_t,   // value type in device memory
    short nl_v,       // number of 4-element groups per block
    void (*deq_v_t4)(device const vd4_t *, short, thread v4_t &),
    short DK,         // K head dimension (128)
    short DV,         // V head dimension (128)
    short NE = 4,     // elements per thread (1 for head_dim=128)
    short Q  = 1,     // queries per simdgroup
    short C  = 256>   // cache items per simdgroup
```

### Key Template Parameter Values

| Format | kd4_t | nl_k | nl_v | NE |
|--------|-------|------|------|----|
| F16 | half4 | 1 | 1 | 1 |
| Q4_0 | block_q4_0 | 8 | 8 | 1 |
| Q4_1 | block_q4_1 | 8 | 8 | 1 |
| Q5_0 | block_q5_0 | 8 | 8 | 1 |
| Q5_1 | block_q5_1 | 8 | 8 | 1 |
| Q8_0 | block_q8_0 | 8 | 8 | 1 |

**Note**: `nl_k` = number of 4-element groups per block = QK/4 = 32/4 = 8

### Function Constants

| Constant | Index | Description | Value (F16) |
|----------|-------|-------------|-------------|
| FC_has_mask | 0 | Has attention mask | false |
| FC_has_bias | 1 | Has ALiBi bias | false |
| FC_has_scap | 2 | Has logit softcap | false |
| FC_has_kvpad | 3 | Has KV padding | false |
| FC_nsg | 4 | Number of SIMD groups | 1 |
| FC_nwg | 5 | Number of workgroups | 1 |
| FC_ns10 | 6 | K stride in bytes | 256 |
| FC_ns20 | 7 | V stride in bytes | 256 |

### Buffer Indices

| Buffer | Index | Description |
|--------|-------|-------------|
| args | 0 | Arguments (via set_bytes) |
| q | 1 | Query tensor |
| k | 2 | Key tensor |
| v | 3 | Value tensor |
| mask | 4 | Attention mask (optional) |
| sinks | 5 | Attention sinks (not used) |
| pad | 6 | Padding buffer (not used) |
| dst | 7 | Output tensor |
| idx | 8 | Sparse indices (not used) |

### Shared Memory Layout

For Q=1, NSG=1, C=256, DK=DV=128:

```
Offset (bytes)  Size (bytes)  Content
-------------   ------------  -------
0               256           sq4 (query data: Q*NSG*PK half)
256             4096          ss (scores: NSG*SH float)
4352            512           sm (mask: Q*C half)
4864            2048          so4 (output: NSG*Q*PV float4)
Total:          6912 bytes
```

Where:
- PK = PAD2(DK, 128) = 128
- PV = PAD2(DV, 128) = 128
- SH = 4*Q*C = 1024

### Multi-SIMD Reduction

When NSG > 1, the kernel performs a parallel reduction across SIMD groups:

1. Each SIMD group computes partial S and M values
2. Reduction loop merges S and M values using online softmax
3. Final output is scaled by merged S value

## Implementation Status

### Completed
- [x] Template structure matching llama.cpp
- [x] F16 kernel template instantiation
- [x] Q4_0/Q4_1/Q5_0/Q5_1/Q8_0 template instantiations
- [x] Function constants setup
- [x] Shared memory layout
- [x] Multi-SIMD reduction logic
- [x] Output writing logic

### Pending
- [ ] KV padding for seq_len not multiple of C=256
- [ ] Main kernel for prefill
- [ ] Correctness verification
- [ ] Performance benchmarking

## Critical Issues

### Bounds Checking

**Problem**: The kernel processes C=256 cache items per iteration, but doesn't check if seq_len is a multiple of C. This causes out-of-bounds memory access.

**Solution**:
1. Set FC_has_kvpad=true and provide a pad buffer
2. OR pad K/V tensors to be a multiple of C
3. OR add bounds checking to the Q*K and O computation loops

**Recommendation**: Use KV padding (FC_has_kvpad=true) like llama.cpp does.

### Testing

Before running any GPU tests:
1. Ensure seq_len is a multiple of C=256, OR
2. Implement proper KV padding

## References

- llama.cpp's Flash Attention implementation: `llama.cpp-0.5.0/ggml/src/ggml-metal/kernels/fa.metal`
- Flash Attention paper: https://arxiv.org/pdf/2307.08691.pdf
