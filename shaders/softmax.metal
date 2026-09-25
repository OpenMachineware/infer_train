// Softmax Metal kernel
// Based on llama.cpp's kernel_soft_max
// Supports: scale, mask, ALiBi

#include <metal_stdlib>
using namespace metal;

constant float N_INFINITY = 1e30f;

struct SoftmaxArgs {
    int32_t ne00;       // row length
    int32_t ne01;       // number of rows per batch
    int32_t ne02;       // batch size
    uint64_t nb01;      // row stride (bytes)
    uint64_t nb02;      // batch stride (bytes)
    uint64_t nb03;      // outer stride (bytes)
    int32_t ne11;       // mask rows
    int32_t ne12;       // mask batch
    int32_t ne13;       // mask outer batch
    uint64_t nb11;      // mask row stride
    uint64_t nb12;      // mask batch stride
    uint64_t nb13;      // mask outer stride
    uint64_t nb1;       // output row stride
    uint64_t nb2;       // output batch stride
    uint64_t nb3;       // output outer stride
    float scale;        // temperature scaling
    float max_bias;     // ALiBi max bias (0 if not using ALiBi)
    float m0;           // ALiBi m0
    float m1;           // ALiBi m1
    int32_t n_head_log2;// ALiBi log2(n_heads)
};

// Fast path kernel - no mask, no ALiBi
// Matches llama.cpp's simple kernel for maximum performance
// Note: This kernel assumes ne01 (n_heads) is passed via buffer(3)
kernel void kernel_soft_max_f32_fast(
    constant int32_t& ne00 [[buffer(0)]],
    device const float* src0 [[buffer(1)]],
    device float* dst [[buffer(2)]],
    constant int32_t& ne01 [[buffer(3)]],
    threadgroup float* buf [[threadgroup(0)]],
    uint3 tgpig [[threadgroup_position_in_grid]],
    uint3 tpitg [[thread_position_in_threadgroup]],
    uint sgitg [[simdgroup_index_in_threadgroup]],
    uint tiisg [[thread_index_in_simdgroup]],
    uint3 tptg [[threads_per_threadgroup]]
) {
    const int32_t i02 = tgpig.y;  // batch index
    const int32_t i01 = tgpig.x;  // head index
    const uint64_t nb01 = ne00 * 4;  // row stride
    const uint64_t nb02 = ne01 * nb01;  // batch stride = n_heads * seq_len * 4

    const uint64_t row_offset = i02 * nb02 + i01 * nb01;
    device const float* psrc = src0 + row_offset / 4;
    device float* pdst = dst + row_offset / 4;

    // Step 1: Find max
    float lmax = -N_INFINITY;
    for (int i00 = tpitg.x; i00 < ne00; i00 += tptg.x) {
        lmax = max(lmax, psrc[i00]);
    }

    float max_val = simd_max(lmax);
    if (tptg.x > 32) {
        if (sgitg == 0) { buf[tiisg] = -N_INFINITY; }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tiisg == 0) { buf[sgitg] = max_val; }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        max_val = simd_max(buf[tiisg]);
    }

    // Step 2: Compute exp and sum
    float lsum = 0.0f;
    for (int i00 = tpitg.x; i00 < ne00; i00 += tptg.x) {
        const float exp_val = exp(psrc[i00] - max_val);
        lsum += exp_val;
        pdst[i00] = exp_val;
    }

    float sum = simd_sum(lsum);
    if (tptg.x > 32) {
        if (sgitg == 0) { buf[tiisg] = 0.0f; }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tiisg == 0) { buf[sgitg] = sum; }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        sum = simd_sum(buf[tiisg]);
    }

    // Step 3: Normalize
    const float inv_sum = 1.0f / sum;
    for (int i00 = tpitg.x; i00 < ne00; i00 += tptg.x) {
        pdst[i00] *= inv_sum;
    }
}

// Fast path kernel - float4 vectorized, no mask, no ALiBi
kernel void kernel_soft_max_f32_4_fast(
    constant int32_t& ne00 [[buffer(0)]],
    device const float* src0 [[buffer(1)]],
    device float* dst [[buffer(2)]],
    constant int32_t& ne01 [[buffer(3)]],
    threadgroup float* buf [[threadgroup(0)]],
    uint3 tgpig [[threadgroup_position_in_grid]],
    uint3 tpitg [[thread_position_in_threadgroup]],
    uint sgitg [[simdgroup_index_in_threadgroup]],
    uint tiisg [[thread_index_in_simdgroup]],
    uint3 tptg [[threads_per_threadgroup]]
) {
    const int32_t i02 = tgpig.y;
    const int32_t i01 = tgpig.x;
    const uint64_t nb01 = ne00 * 4;
    const uint64_t nb02 = ne01 * nb01;

    const uint64_t row_offset = i02 * nb02 + i01 * nb01;
    device const float4* psrc4 = (device const float4*)(src0 + row_offset / 4);
    device float4* pdst4 = (device float4*)(dst + row_offset / 4);
    const int32_t ne00_4 = ne00 / 4;

    // Step 1: Find max
    float4 lmax4 = -N_INFINITY;
    for (int i00 = tpitg.x; i00 < ne00_4; i00 += tptg.x) {
        lmax4 = fmax(lmax4, psrc4[i00]);
    }
    float lmax = max(max(lmax4[0], lmax4[1]), max(lmax4[2], lmax4[3]));

    float max_val = simd_max(lmax);
    if (tptg.x > 32) {
        if (sgitg == 0) { buf[tiisg] = -N_INFINITY; }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tiisg == 0) { buf[sgitg] = max_val; }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        max_val = simd_max(buf[tiisg]);
    }

    // Step 2: Compute exp and sum
    float4 lsum4 = 0.0f;
    for (int i00 = tpitg.x; i00 < ne00_4; i00 += tptg.x) {
        const float4 exp_val = exp(psrc4[i00] - max_val);
        lsum4 += exp_val;
        pdst4[i00] = exp_val;
    }
    float lsum = lsum4[0] + lsum4[1] + lsum4[2] + lsum4[3];

    float sum = simd_sum(lsum);
    if (tptg.x > 32) {
        if (sgitg == 0) { buf[tiisg] = 0.0f; }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tiisg == 0) { buf[sgitg] = sum; }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        sum = simd_sum(buf[tiisg]);
    }

    // Step 3: Normalize
    const float inv_sum = 1.0f / sum;
    for (int i00 = tpitg.x; i00 < ne00_4; i00 += tptg.x) {
        pdst4[i00] *= inv_sum;
    }
}

// Scalar version - each thread processes one element
kernel void kernel_soft_max_f32(
    constant SoftmaxArgs& args [[buffer(0)]],
    device const float* src0 [[buffer(1)]],
    device const char* src1 [[buffer(2)]],  // mask (optional, FP16 or FP32)
    device float* dst [[buffer(3)]],
    threadgroup float* buf [[threadgroup(0)]],
    uint3 tgpig [[threadgroup_position_in_grid]],
    uint3 tpitg [[thread_position_in_threadgroup]],
    uint sgitg [[simdgroup_index_in_threadgroup]],
    uint tiisg [[thread_index_in_simdgroup]],
    uint3 tptg [[threads_per_threadgroup]]
) {
    const int32_t i03 = tgpig.z;
    const int32_t i02 = tgpig.y;  // head index
    const int32_t i01 = tgpig.x;  // row index within head

    // Source row pointer
    device const float* psrc0 = (device const float*)((device const char*)src0 + i01 * args.nb01 + i02 * args.nb02 + i03 * args.nb03);

    // Mask pointer (optional)
    device const char* pmask = (src1 != (device const char*)src0) ?
        (src1 + i01 * args.nb11 + (i02 % args.ne12) * args.nb12 + (i03 % args.ne13) * args.nb13) : nullptr;

    // Output row pointer
    device float* pdst = (device float*)((device const char*)dst + i01 * args.nb1 + i02 * args.nb2 + i03 * args.nb3);

    const int32_t ne00 = args.ne00;
    const float scale = args.scale;

    // ALiBi slope
    float slope = 1.0f;
    if (args.max_bias > 0.0f) {
        const int32_t h = i02;
        const float base = h < args.n_head_log2 ? args.m0 : args.m1;
        const int exp = h < args.n_head_log2 ? h + 1 : 2 * (h - args.n_head_log2) + 1;
        slope = pow(base, (float)exp);
    }

    // Step 1: Find max in parallel
    float lmax = -N_INFINITY;
    if (pmask) {
        for (int i00 = tpitg.x; i00 < ne00; i00 += tptg.x) {
            float val = psrc0[i00] * scale;
            float mask_val = *(device const float*)(pmask + i00 * 4);
            val += slope * mask_val;
            lmax = max(lmax, val);
        }
    } else {
        for (int i00 = tpitg.x; i00 < ne00; i00 += tptg.x) {
            lmax = max(lmax, psrc0[i00] * scale);
        }
    }

    // Reduce max across SIMD group
    float max_val = simd_max(lmax);

    // If more than one SIMD group, reduce across threadgroup
    if (tptg.x > 32) {
        if (sgitg == 0) {
            buf[tiisg] = -N_INFINITY;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (tiisg == 0) {
            buf[sgitg] = max_val;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        max_val = simd_max(buf[tiisg]);
    }

    // Step 2: Compute exp(x - max) and sum in parallel
    float lsum = 0.0f;
    if (pmask) {
        for (int i00 = tpitg.x; i00 < ne00; i00 += tptg.x) {
            float val = psrc0[i00] * scale;
            float mask_val = *(device const float*)(pmask + i00 * 4);
            val += slope * mask_val;
            const float exp_val = exp(val - max_val);
            lsum += exp_val;
            pdst[i00] = exp_val;
        }
    } else {
        for (int i00 = tpitg.x; i00 < ne00; i00 += tptg.x) {
            const float exp_val = exp(psrc0[i00] * scale - max_val);
            lsum += exp_val;
            pdst[i00] = exp_val;
        }
    }

    // Reduce sum across SIMD group
    float sum = simd_sum(lsum);

    // If more than one SIMD group, reduce across threadgroup
    if (tptg.x > 32) {
        if (sgitg == 0) {
            buf[tiisg] = 0.0f;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (tiisg == 0) {
            buf[sgitg] = sum;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        sum = simd_sum(buf[tiisg]);
    }

    // Step 3: Normalize
    const float inv_sum = 1.0f / sum;
    for (int i00 = tpitg.x; i00 < ne00; i00 += tptg.x) {
        pdst[i00] *= inv_sum;
    }
}

// Vectorized version - each thread processes 4 elements
kernel void kernel_soft_max_f32_4(
    constant SoftmaxArgs& args [[buffer(0)]],
    device const float* src0 [[buffer(1)]],
    device const char* src1 [[buffer(2)]],
    device float* dst [[buffer(3)]],
    threadgroup float* buf [[threadgroup(0)]],
    uint3 tgpig [[threadgroup_position_in_grid]],
    uint3 tpitg [[thread_position_in_threadgroup]],
    uint sgitg [[simdgroup_index_in_threadgroup]],
    uint tiisg [[thread_index_in_simdgroup]],
    uint3 tptg [[threads_per_threadgroup]]
) {
    const int32_t i03 = tgpig.z;
    const int32_t i02 = tgpig.y;
    const int32_t i01 = tgpig.x;

    device const float4* psrc4 = (device const float4*)((device const char*)src0 + i01 * args.nb01 + i02 * args.nb02 + i03 * args.nb03);
    device const char* pmask = (src1 != (device const char*)src0) ?
        (src1 + i01 * args.nb11 + (i02 % args.ne12) * args.nb12 + (i03 % args.ne13) * args.nb13) : nullptr;
    device float4* pdst4 = (device float4*)((device const char*)dst + i01 * args.nb1 + i02 * args.nb2 + i03 * args.nb3);

    const int32_t ne00_4 = args.ne00 / 4;
    const float scale = args.scale;

    // ALiBi slope
    float slope = 1.0f;
    if (args.max_bias > 0.0f) {
        const int32_t h = i02;
        const float base = h < args.n_head_log2 ? args.m0 : args.m1;
        const int exp = h < args.n_head_log2 ? h + 1 : 2 * (h - args.n_head_log2) + 1;
        slope = pow(base, (float)exp);
    }

    // Step 1: Find max
    float4 lmax4 = -N_INFINITY;
    if (pmask) {
        for (int i00 = tpitg.x; i00 < ne00_4; i00 += tptg.x) {
            float4 val = psrc4[i00] * scale;
            float4 mask_val = *(device const float4*)(pmask + i00 * 16);
            val += slope * mask_val;
            lmax4 = fmax(lmax4, val);
        }
    } else {
        for (int i00 = tpitg.x; i00 < ne00_4; i00 += tptg.x) {
            lmax4 = fmax(lmax4, psrc4[i00] * scale);
        }
    }
    float lmax = max(max(lmax4[0], lmax4[1]), max(lmax4[2], lmax4[3]));

    float max_val = simd_max(lmax);

    if (tptg.x > 32) {
        if (sgitg == 0) {
            buf[tiisg] = -N_INFINITY;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tiisg == 0) {
            buf[sgitg] = max_val;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        max_val = simd_max(buf[tiisg]);
    }

    // Step 2: Compute exp and sum
    float4 lsum4 = 0.0f;
    if (pmask) {
        for (int i00 = tpitg.x; i00 < ne00_4; i00 += tptg.x) {
            float4 val = psrc4[i00] * scale;
            float4 mask_val = *(device const float4*)(pmask + i00 * 16);
            val += slope * mask_val;
            const float4 exp_val = exp(val - max_val);
            lsum4 += exp_val;
            pdst4[i00] = exp_val;
        }
    } else {
        for (int i00 = tpitg.x; i00 < ne00_4; i00 += tptg.x) {
            const float4 exp_val = exp(psrc4[i00] * scale - max_val);
            lsum4 += exp_val;
            pdst4[i00] = exp_val;
        }
    }
    float lsum = lsum4[0] + lsum4[1] + lsum4[2] + lsum4[3];

    float sum = simd_sum(lsum);

    if (tptg.x > 32) {
        if (sgitg == 0) {
            buf[tiisg] = 0.0f;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tiisg == 0) {
            buf[sgitg] = sum;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        sum = simd_sum(buf[tiisg]);
    }

    // Step 3: Normalize
    const float inv_sum = 1.0f / sum;
    for (int i00 = tpitg.x; i00 < ne00_4; i00 += tptg.x) {
        pdst4[i00] *= inv_sum;
    }
}
