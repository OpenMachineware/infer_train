// SPDX-License-Identifier: Apache-2.0-or-later
// Copyright 2026 Jia Liu

#![allow(dead_code, non_snake_case)]

use crate::quant::types::{
    BlockIQ1M, BlockIQ1S, BlockIQ2S, BlockIQ2XS, BlockIQ2XXS, BlockIQ3S,
    BlockIQ3XXS, BlockIQ4NL, BlockIQ4XS, BlockMXFP4, BlockQ1_0, BlockQ2_0,
    BlockQ2K, BlockQ3K, BlockQ4_0, BlockQ4_1, BlockQ4K, BlockQ5_0, BlockQ5_1,
    BlockQ5K, BlockQ6K, BlockQ8_0, BlockTQ1_0, BlockTQ2_0,
};
use metal::{
    Buffer, CommandBuffer, CommandQueue, ComputeCommandEncoder,
    ComputeCommandEncoderRef, ComputePipelineState, Device,
    FunctionConstantValues, Library, MTLDataType, MTLSize, NSUInteger,
};
use std::cell::RefCell;
use std::os::raw::c_void;

/// GEMM kernel arguments matching ggml_metal_kargs_mul_mm in mul_mm.metal
/// Used for matrix-matrix and matrix-vector operations
#[repr(C)]
struct GemmArgs {
    ne00: i32, // K dimension
    ne02: i32, // batch dimension
    nb01: u64, // row stride for A (bytes between rows)
    nb02: u64, // batch stride for A
    nb03: u64, // batch3 stride for A
    ne12: i32, // batch dimension for B
    nb10: u64, // element stride for B (bytes between elements)
    nb11: u64, // row stride for B (bytes between rows)
    nb12: u64, // batch stride for B
    nb13: u64, // batch3 stride for B
    ne0: i32,  // M dimension (output rows)
    ne1: i32,  // N dimension (output cols)
    r2: i16,   // broadcast ratio for batch dimension
    r3: i16,   // broadcast ratio for batch3 dimension
}

impl GemmArgs {
    /// Create args for matrix-vector multiplication (N=1)
    fn for_mv(m: usize, k: usize, nb01: u64) -> Self {
        Self {
            ne00: k as i32,
            ne02: 1,
            nb01,
            nb02: 0,
            nb03: 0,
            ne12: 1,
            nb10: 4,
            nb11: k as u64 * 4,
            nb12: 0,
            nb13: 0,
            ne0: m as i32,
            ne1: 1,
            r2: 1,
            r3: 1,
        }
    }
}

/// K-quant MV kernel arguments (matching kernel's kquant_mv_args)
#[repr(C)]
struct KquantMvArgs {
    ne00: u32, // K dimension (input size)
    ne01: u32, // M dimension (output rows)
    nb01: u64, // Byte stride between rows of weight matrix
}

impl KquantMvArgs {
    fn new(m: usize, k: usize, nb01: u64) -> Self {
        Self { ne00: k as u32, ne01: m as u32, nb01 }
    }
}

// Note: We do NOT derive(Clone) because we need to clear caches on clone
// to avoid stale buffer references when MetalContext is shared across contexts.
pub struct MetalContext {
    device: Device,
    queue: CommandQueue,
    vec_dot_library: Library,
    mv_library: Library,
    mv_simd_library: Library,
    mv_iq_tq_library: Library,
    // Self-written GEMM template library (all formats)
    gemm_library: Library,
    // Self-written K-quant MV library (Q2_K, Q3_K, Q4_K, Q5_K, Q6_K)
    kquant_mv_library: Library,
    // Transposed GEMV library (for output layer with tied embeddings)
    gemv_transposed_library: Library,
    mv_q6_k_transposed_pipeline: ComputePipelineState,
    // F16 GEMV library (for transposed output layer with F16 weights)
    mv_f16_library: Library,
    mv_f16_short_pipeline: ComputePipelineState,
    mv_f16_basic_pipeline: ComputePipelineState,
    mv_f16_f16_pipeline: ComputePipelineState, // F16 weights + F16 input
    mv_f16_4_nsg1_pipeline: ComputePipelineState,
    mv_f16_4_nsg2_pipeline: ComputePipelineState,
    mv_f16_4_nsg4_pipeline: ComputePipelineState,
    // Optimized F16 pipelines with nxpsg support (llama.cpp-style)
    mv_f16_opt_r1_1_nxpsg16_pipeline: ComputePipelineState,
    // Cached MV pipelines - Standard Q
    mv_q1_0_pipeline: ComputePipelineState,
    mv_q2_0_pipeline: ComputePipelineState,
    mv_q4_0_pipeline: ComputePipelineState,
    mv_q4_1_pipeline: ComputePipelineState,
    mv_q5_0_pipeline: ComputePipelineState,
    mv_q5_1_pipeline: ComputePipelineState,
    mv_q8_0_pipeline: ComputePipelineState,
    // Cached MV pipelines - K-quant
    mv_q4_k_pipeline: ComputePipelineState,
    mv_q5_k_pipeline: ComputePipelineState,
    // Fused GEMV+Add pipelines for residual connections
    fused_gemv_add_q4_k_pipeline: ComputePipelineState,
    fused_gemv_add_q5_k_pipeline: ComputePipelineState,
    fused_gemv_add_q6_k_pipeline: ComputePipelineState,
    mv_q2_k_pipeline: ComputePipelineState,
    mv_q6_k_pipeline: ComputePipelineState,
    mv_q3_k_pipeline: ComputePipelineState,
    mv_iq4_nl_pipeline: ComputePipelineState,
    mv_tq2_0_pipeline: ComputePipelineState,
    mv_tq1_0_pipeline: ComputePipelineState,
    mv_iq4_xs_pipeline: ComputePipelineState,
    mv_iq1_s_pipeline: ComputePipelineState,
    mv_iq1_m_pipeline: ComputePipelineState,
    mv_iq2_xxs_pipeline: ComputePipelineState,
    mv_iq2_xs_pipeline: ComputePipelineState,
    mv_iq2_s_pipeline: ComputePipelineState,
    mv_iq3_xxs_pipeline: ComputePipelineState,
    mv_iq3_s_pipeline: ComputePipelineState,
    // MXFP
    mv_mxfp4_pipeline: ComputePipelineState,
    // Split=1 pipelines for large K sizes
    mv_iq2_xxs_split1_pipeline: ComputePipelineState,
    mv_iq2_xs_split1_pipeline: ComputePipelineState,
    mv_iq2_s_split1_pipeline: ComputePipelineState,
    mv_iq3_xxs_split1_pipeline: ComputePipelineState,
    mv_iq3_s_split1_pipeline: ComputePipelineState,
    gemm_template_q4_k_pipeline: ComputePipelineState,
    gemm_template_q2_k_pipeline: ComputePipelineState,
    gemm_template_q3_k_pipeline: ComputePipelineState,
    gemm_template_q5_k_pipeline: ComputePipelineState,
    gemm_template_q6_k_pipeline: ComputePipelineState,
    // RMS Norm, Softmax, ROPE, Activation libraries and pipelines
    rms_norm_library: Library,
    rms_norm_pipeline: ComputePipelineState,
    rms_norm_4_pipeline: ComputePipelineState,
    softmax_library: Library,
    softmax_fast_pipeline: ComputePipelineState,
    softmax_4_fast_pipeline: ComputePipelineState,
    softmax_pipeline: ComputePipelineState,
    softmax_4_pipeline: ComputePipelineState,
    rope_library: Library,
    rope_pipeline: ComputePipelineState,
    rope_f16_pipeline: ComputePipelineState, // For KV cache F16 output
    activation_library: Library,
    add_pipeline: ComputePipelineState,
    mul_pipeline: ComputePipelineState,
    silu_pipeline: ComputePipelineState,
    fused_swi_glu_pipeline: ComputePipelineState,
    silu_4_pipeline: ComputePipelineState,
    gelu_pipeline: ComputePipelineState,
    gelu_4_pipeline: ComputePipelineState,
    gelu_quick_pipeline: ComputePipelineState,
    gelu_quick_4_pipeline: ComputePipelineState,
    // F32 <-> F16 conversion
    f32_to_f16_pipeline: ComputePipelineState,
    f16_to_f32_pipeline: ComputePipelineState,
    f32_to_f16_strided_pipeline: ComputePipelineState,
    // Fused kernels
    fused_swiglu_pipeline: ComputePipelineState,
    fused_swiglu_4_pipeline: ComputePipelineState,
    fused_norm_linear_f16_pipeline: ComputePipelineState,
    fused_norm_linear_f32_pipeline: ComputePipelineState,
    fused_norm_linear_f16_4_pipeline: ComputePipelineState,
    fused_norm_linear_q4_k_pipeline: ComputePipelineState,
    // TODO: Add Q3_K, Q2_K, IQ4_NL, IQ4_XS fused norm linear pipelines
    fused_norm_linear_q5_k_pipeline: ComputePipelineState,
    fused_norm_linear_q6_k_pipeline: ComputePipelineState,
    // Split QKV kernel
    split_qkv_library: Library,
    split_qkv_pipeline: ComputePipelineState,
    // FusedAddNorm (Add + RMSNorm)
    fused_add_norm_f32_pipeline: ComputePipelineState,
    fused_add_norm_f32_4_pipeline: ComputePipelineState,
    fused_add_norm_f16_pipeline: ComputePipelineState,
    fused_add_norm_f16_4_pipeline: ComputePipelineState,
    // FusedGateUp (gate + up projections in one kernel)
    fused_gate_up_f16_pipeline: ComputePipelineState,
    fused_gate_up_q4_k_pipeline: ComputePipelineState,
    fused_gate_up_q5_k_pipeline: ComputePipelineState,
    fused_gate_up_q6_k_pipeline: ComputePipelineState,
    // Flash Attention
    flash_attn_library: Library,
    // F16 decode pipelines for each head_dim (NE=4 variant)
    flash_decode_f16_32_pipeline: ComputePipelineState,
    flash_decode_f16_64_pipeline: ComputePipelineState,
    flash_decode_f16_80_pipeline: ComputePipelineState,
    flash_decode_f16_128_pipeline: ComputePipelineState,
    flash_decode_f16_160_pipeline: ComputePipelineState,
    flash_decode_f16_213_pipeline: ComputePipelineState, // Special kernel for head_dim=213
    // MMA-based decode (experimental, for performance comparison)
    flash_decode_mma_pipeline: ComputePipelineState,
    flash_decode_mma_v2_pipeline: ComputePipelineState, // Optimized version
    // flash_decode_mma_true_pipeline: ComputePipelineState, // True MMA with simdgroup_float8x8 (DISABLED)
    flash_decode_mma_batched_pipeline: ComputePipelineState, // Batched processing (C=8)
    // NWG parallelized Flash Attention
    flash_decode_nwg_pass1_pipeline: ComputePipelineState, // First pass: partial results
    flash_decode_nwg_pass2_pipeline: ComputePipelineState, // Second pass: reduction
    // Multi-head Flash Attention (llama.cpp style, Q=32, NSG=8)
    flash_attn_multihead_128_pipeline: ComputePipelineState,
    // Flash Prefill (for batched prefill)
    flash_prefill_f16_pipeline: ComputePipelineState,
    // Batched flash prefill (llama.cpp style)
    flash_prefill_batch_pipeline: ComputePipelineState,
    // Quantize
    quantize_library: Library,
    quantize_f16_pipeline: ComputePipelineState,
    quantize_f16_4_pipeline: ComputePipelineState,
    quantize_q4_0_pipeline: ComputePipelineState,
    quantize_q8_0_pipeline: ComputePipelineState,
    // SSM (State Space Model)
    ssm_library: Library,
    ssm_conv1d_d4_pipeline: ComputePipelineState,
    ssm_scan_d16_pipeline: ComputePipelineState,
    ssm_scan_ssd_pipeline: ComputePipelineState,
    ssm_scan_ssd_mma_pipeline: ComputePipelineState,
    // Conv1D batched pipelines for different batch sizes
    ssm_conv1d_d4_batched_256_pipeline: ComputePipelineState,
    ssm_conv1d_d4_batched_128_pipeline: ComputePipelineState,
    ssm_conv1d_d4_batched_64_pipeline: ComputePipelineState,
    ssm_conv1d_d4_batched_32_pipeline: ComputePipelineState,
    ssm_conv1d_d4_batched_16_pipeline: ComputePipelineState,
    ssm_conv1d_d4_batched_8_pipeline: ComputePipelineState,
    ssm_conv1d_d4_batched_2_pipeline: ComputePipelineState,
    // KV Cache Update (F32 -> F16 with strided write)
    kv_update_library: Library,
    kv_update_k_pipeline: ComputePipelineState,
    kv_update_k_f16_pipeline: ComputePipelineState, // F16->F16 (no conversion)
    kv_update_v_pipeline: ComputePipelineState,
    // Batched KV update for prefill
    kv_update_batched_f16_pipeline: ComputePipelineState,
    kv_update_batched_f16_strided_pipeline: ComputePipelineState,
    // Cached buffers for MV operations
    // Once created with data, reuse without copy
    mv_weights_buffer: RefCell<Option<Buffer>>,
    mv_input_buffer: RefCell<Option<Buffer>>,
    mv_output_buffer: RefCell<Option<Buffer>>,
    // Track buffer sizes and data pointers to know if we need to recreate
    mv_weights_size: RefCell<usize>,
    mv_weights_ptr: RefCell<usize>, // Track data pointer to detect different tensors
    mv_input_size: RefCell<usize>,
    mv_output_size: RefCell<usize>,
    // Weight buffer cache: key = data pointer, value = GPU buffer
    // This allows each weight tensor to have its own cached buffer
    weight_buffer_cache:
        RefCell<std::collections::HashMap<usize, std::rc::Rc<Buffer>>>,
    // Cached buffers for GEMM operations
    gemm_weights_buffer: RefCell<Option<Buffer>>,
    gemm_input_buffer: RefCell<Option<Buffer>>,
    gemm_output_buffer: RefCell<Option<Buffer>>,
    gemm_weights_size: RefCell<usize>,
    gemm_input_size: RefCell<usize>,
    gemm_output_size: RefCell<usize>,
    // Cached pos buffer for RoPE (avoid allocation on every call)
    rope_pos_buffer: RefCell<Option<Buffer>>,
    // Cached embedding buffer for decode (avoid allocation on every call)
    embed_buffer: RefCell<Option<Buffer>>,
    embed_size: RefCell<usize>,
}

impl std::fmt::Debug for MetalContext {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MetalContext")
            .field("device", &self.device.name())
            .finish_non_exhaustive()
    }
}

impl Clone for MetalContext {
    fn clone(&self) -> Self {
        // Clone all immutable resources (device, queues, libraries, pipelines)
        // but CLEAR all caches to avoid stale buffer references
        Self {
            device: self.device.clone(),
            queue: self.queue.clone(),
            vec_dot_library: self.vec_dot_library.clone(),
            mv_library: self.mv_library.clone(),
            mv_simd_library: self.mv_simd_library.clone(),
            mv_iq_tq_library: self.mv_iq_tq_library.clone(),
            gemm_library: self.gemm_library.clone(),
            kquant_mv_library: self.kquant_mv_library.clone(),
            gemv_transposed_library: self.gemv_transposed_library.clone(),
            mv_q6_k_transposed_pipeline: self
                .mv_q6_k_transposed_pipeline
                .clone(),
            mv_f16_library: self.mv_f16_library.clone(),
            mv_f16_short_pipeline: self.mv_f16_short_pipeline.clone(),
            mv_f16_basic_pipeline: self.mv_f16_basic_pipeline.clone(),
            mv_f16_f16_pipeline: self.mv_f16_f16_pipeline.clone(),
            mv_f16_4_nsg1_pipeline: self.mv_f16_4_nsg1_pipeline.clone(),
            mv_f16_4_nsg2_pipeline: self.mv_f16_4_nsg2_pipeline.clone(),
            mv_f16_4_nsg4_pipeline: self.mv_f16_4_nsg4_pipeline.clone(),
            mv_f16_opt_r1_1_nxpsg16_pipeline: self
                .mv_f16_opt_r1_1_nxpsg16_pipeline
                .clone(),
            mv_q1_0_pipeline: self.mv_q1_0_pipeline.clone(),
            mv_q2_0_pipeline: self.mv_q2_0_pipeline.clone(),
            mv_q4_0_pipeline: self.mv_q4_0_pipeline.clone(),
            mv_q4_1_pipeline: self.mv_q4_1_pipeline.clone(),
            mv_q5_0_pipeline: self.mv_q5_0_pipeline.clone(),
            mv_q5_1_pipeline: self.mv_q5_1_pipeline.clone(),
            mv_q8_0_pipeline: self.mv_q8_0_pipeline.clone(),
            mv_q4_k_pipeline: self.mv_q4_k_pipeline.clone(),
            mv_q5_k_pipeline: self.mv_q5_k_pipeline.clone(),
            mv_q2_k_pipeline: self.mv_q2_k_pipeline.clone(),
            mv_q6_k_pipeline: self.mv_q6_k_pipeline.clone(),
            mv_q3_k_pipeline: self.mv_q3_k_pipeline.clone(),
            fused_gemv_add_q4_k_pipeline: self
                .fused_gemv_add_q4_k_pipeline
                .clone(),
            fused_gemv_add_q5_k_pipeline: self
                .fused_gemv_add_q5_k_pipeline
                .clone(),
            fused_gemv_add_q6_k_pipeline: self
                .fused_gemv_add_q6_k_pipeline
                .clone(),
            mv_iq4_nl_pipeline: self.mv_iq4_nl_pipeline.clone(),
            mv_tq2_0_pipeline: self.mv_tq2_0_pipeline.clone(),
            mv_tq1_0_pipeline: self.mv_tq1_0_pipeline.clone(),
            mv_iq4_xs_pipeline: self.mv_iq4_xs_pipeline.clone(),
            mv_iq1_s_pipeline: self.mv_iq1_s_pipeline.clone(),
            mv_iq1_m_pipeline: self.mv_iq1_m_pipeline.clone(),
            mv_iq2_xxs_pipeline: self.mv_iq2_xxs_pipeline.clone(),
            mv_iq2_xs_pipeline: self.mv_iq2_xs_pipeline.clone(),
            mv_iq2_s_pipeline: self.mv_iq2_s_pipeline.clone(),
            mv_iq3_xxs_pipeline: self.mv_iq3_xxs_pipeline.clone(),
            mv_iq3_s_pipeline: self.mv_iq3_s_pipeline.clone(),
            mv_mxfp4_pipeline: self.mv_mxfp4_pipeline.clone(),
            mv_iq2_xxs_split1_pipeline: self.mv_iq2_xxs_split1_pipeline.clone(),
            mv_iq2_xs_split1_pipeline: self.mv_iq2_xs_split1_pipeline.clone(),
            mv_iq2_s_split1_pipeline: self.mv_iq2_s_split1_pipeline.clone(),
            mv_iq3_xxs_split1_pipeline: self.mv_iq3_xxs_split1_pipeline.clone(),
            mv_iq3_s_split1_pipeline: self.mv_iq3_s_split1_pipeline.clone(),
            gemm_template_q4_k_pipeline: self
                .gemm_template_q4_k_pipeline
                .clone(),
            gemm_template_q2_k_pipeline: self
                .gemm_template_q2_k_pipeline
                .clone(),
            gemm_template_q3_k_pipeline: self
                .gemm_template_q3_k_pipeline
                .clone(),
            gemm_template_q5_k_pipeline: self
                .gemm_template_q5_k_pipeline
                .clone(),
            gemm_template_q6_k_pipeline: self
                .gemm_template_q6_k_pipeline
                .clone(),
            rms_norm_library: self.rms_norm_library.clone(),
            rms_norm_pipeline: self.rms_norm_pipeline.clone(),
            rms_norm_4_pipeline: self.rms_norm_4_pipeline.clone(),
            softmax_library: self.softmax_library.clone(),
            softmax_fast_pipeline: self.softmax_fast_pipeline.clone(),
            softmax_4_fast_pipeline: self.softmax_4_fast_pipeline.clone(),
            softmax_pipeline: self.softmax_pipeline.clone(),
            softmax_4_pipeline: self.softmax_4_pipeline.clone(),
            rope_library: self.rope_library.clone(),
            rope_pipeline: self.rope_pipeline.clone(),
            rope_f16_pipeline: self.rope_f16_pipeline.clone(),
            activation_library: self.activation_library.clone(),
            add_pipeline: self.add_pipeline.clone(),
            mul_pipeline: self.mul_pipeline.clone(),
            silu_pipeline: self.silu_pipeline.clone(),
            fused_swi_glu_pipeline: self.fused_swi_glu_pipeline.clone(),
            silu_4_pipeline: self.silu_4_pipeline.clone(),
            gelu_pipeline: self.gelu_pipeline.clone(),
            gelu_4_pipeline: self.gelu_4_pipeline.clone(),
            gelu_quick_pipeline: self.gelu_quick_pipeline.clone(),
            gelu_quick_4_pipeline: self.gelu_quick_4_pipeline.clone(),
            // F32 <-> F16 conversion
            f32_to_f16_pipeline: self.f32_to_f16_pipeline.clone(),
            f16_to_f32_pipeline: self.f16_to_f32_pipeline.clone(),
            f32_to_f16_strided_pipeline: self
                .f32_to_f16_strided_pipeline
                .clone(),
            // Fused kernels
            fused_swiglu_pipeline: self.fused_swiglu_pipeline.clone(),
            fused_swiglu_4_pipeline: self.fused_swiglu_4_pipeline.clone(),
            fused_norm_linear_f16_pipeline: self
                .fused_norm_linear_f16_pipeline
                .clone(),
            fused_norm_linear_f32_pipeline: self
                .fused_norm_linear_f32_pipeline
                .clone(),
            fused_norm_linear_f16_4_pipeline: self
                .fused_norm_linear_f16_4_pipeline
                .clone(),
            fused_norm_linear_q4_k_pipeline: self
                .fused_norm_linear_q4_k_pipeline
                .clone(),
            fused_norm_linear_q5_k_pipeline: self
                .fused_norm_linear_q5_k_pipeline
                .clone(),
            fused_norm_linear_q6_k_pipeline: self
                .fused_norm_linear_q6_k_pipeline
                .clone(),
            split_qkv_library: self.split_qkv_library.clone(),
            split_qkv_pipeline: self.split_qkv_pipeline.clone(),
            fused_add_norm_f32_pipeline: self
                .fused_add_norm_f32_pipeline
                .clone(),
            fused_add_norm_f32_4_pipeline: self
                .fused_add_norm_f32_4_pipeline
                .clone(),
            fused_add_norm_f16_pipeline: self
                .fused_add_norm_f16_pipeline
                .clone(),
            fused_add_norm_f16_4_pipeline: self
                .fused_add_norm_f16_4_pipeline
                .clone(),
            // FusedGateUp
            fused_gate_up_f16_pipeline: self.fused_gate_up_f16_pipeline.clone(),
            fused_gate_up_q4_k_pipeline: self
                .fused_gate_up_q4_k_pipeline
                .clone(),
            fused_gate_up_q5_k_pipeline: self
                .fused_gate_up_q5_k_pipeline
                .clone(),
            fused_gate_up_q6_k_pipeline: self
                .fused_gate_up_q6_k_pipeline
                .clone(),
            flash_attn_library: self.flash_attn_library.clone(),
            flash_decode_f16_32_pipeline: self
                .flash_decode_f16_32_pipeline
                .clone(),
            flash_decode_f16_64_pipeline: self
                .flash_decode_f16_64_pipeline
                .clone(),
            flash_decode_f16_80_pipeline: self
                .flash_decode_f16_80_pipeline
                .clone(),
            flash_decode_f16_128_pipeline: self
                .flash_decode_f16_128_pipeline
                .clone(),
            flash_decode_f16_160_pipeline: self
                .flash_decode_f16_160_pipeline
                .clone(),
            flash_decode_f16_213_pipeline: self
                .flash_decode_f16_213_pipeline
                .clone(),
            flash_decode_mma_pipeline: self.flash_decode_mma_pipeline.clone(),
            flash_decode_mma_v2_pipeline: self
                .flash_decode_mma_v2_pipeline
                .clone(),
            // flash_decode_mma_true_pipeline: self
            //     .flash_decode_mma_true_pipeline
            //     .clone(), // DISABLED
            flash_decode_mma_batched_pipeline: self
                .flash_decode_mma_batched_pipeline
                .clone(),
            flash_decode_nwg_pass1_pipeline: self
                .flash_decode_nwg_pass1_pipeline
                .clone(),
            flash_decode_nwg_pass2_pipeline: self
                .flash_decode_nwg_pass2_pipeline
                .clone(),
            flash_attn_multihead_128_pipeline: self
                .flash_attn_multihead_128_pipeline
                .clone(),
            flash_prefill_f16_pipeline: self.flash_prefill_f16_pipeline.clone(),
            flash_prefill_batch_pipeline: self
                .flash_prefill_batch_pipeline
                .clone(),
            quantize_library: self.quantize_library.clone(),
            quantize_f16_pipeline: self.quantize_f16_pipeline.clone(),
            quantize_f16_4_pipeline: self.quantize_f16_4_pipeline.clone(),
            quantize_q4_0_pipeline: self.quantize_q4_0_pipeline.clone(),
            quantize_q8_0_pipeline: self.quantize_q8_0_pipeline.clone(),
            ssm_library: self.ssm_library.clone(),
            ssm_conv1d_d4_pipeline: self.ssm_conv1d_d4_pipeline.clone(),
            ssm_scan_d16_pipeline: self.ssm_scan_d16_pipeline.clone(),
            ssm_scan_ssd_pipeline: self.ssm_scan_ssd_pipeline.clone(),
            ssm_scan_ssd_mma_pipeline: self.ssm_scan_ssd_mma_pipeline.clone(),
            ssm_conv1d_d4_batched_256_pipeline: self
                .ssm_conv1d_d4_batched_256_pipeline
                .clone(),
            ssm_conv1d_d4_batched_128_pipeline: self
                .ssm_conv1d_d4_batched_128_pipeline
                .clone(),
            ssm_conv1d_d4_batched_64_pipeline: self
                .ssm_conv1d_d4_batched_64_pipeline
                .clone(),
            ssm_conv1d_d4_batched_32_pipeline: self
                .ssm_conv1d_d4_batched_32_pipeline
                .clone(),
            ssm_conv1d_d4_batched_16_pipeline: self
                .ssm_conv1d_d4_batched_16_pipeline
                .clone(),
            ssm_conv1d_d4_batched_8_pipeline: self
                .ssm_conv1d_d4_batched_8_pipeline
                .clone(),
            ssm_conv1d_d4_batched_2_pipeline: self
                .ssm_conv1d_d4_batched_2_pipeline
                .clone(),
            kv_update_library: self.kv_update_library.clone(),
            kv_update_k_pipeline: self.kv_update_k_pipeline.clone(),
            kv_update_k_f16_pipeline: self.kv_update_k_f16_pipeline.clone(),
            kv_update_v_pipeline: self.kv_update_v_pipeline.clone(),
            // Batched KV update
            kv_update_batched_f16_pipeline: self
                .kv_update_batched_f16_pipeline
                .clone(),
            kv_update_batched_f16_strided_pipeline: self
                .kv_update_batched_f16_strided_pipeline
                .clone(),
            // Clear all cached buffers - they may contain stale data
            mv_weights_buffer: RefCell::new(None),
            mv_input_buffer: RefCell::new(None),
            mv_output_buffer: RefCell::new(None),
            mv_weights_size: RefCell::new(0),
            mv_weights_ptr: RefCell::new(0),
            mv_input_size: RefCell::new(0),
            mv_output_size: RefCell::new(0),
            weight_buffer_cache: RefCell::new(std::collections::HashMap::new()),
            gemm_weights_buffer: RefCell::new(None),
            gemm_input_buffer: RefCell::new(None),
            gemm_output_buffer: RefCell::new(None),
            gemm_weights_size: RefCell::new(0),
            gemm_input_size: RefCell::new(0),
            gemm_output_size: RefCell::new(0),
            rope_pos_buffer: RefCell::new(None),
            embed_buffer: RefCell::new(None),
            embed_size: RefCell::new(0),
        }
    }
}

#[derive(Clone, Copy)]
#[repr(C)]
struct RopeArgs {
    hidden_dim: i32,     // dimension
    num_heads_seqs: i32, // heads * sequences
    num_seqs: i32,       // sequences
    rotate_dims: i32,    // dimensions to rotate
    offset: i32,         // rotation offset
    freq_base: f32,      // base frequency (usually 10000)
    freq_scale: f32,     // frequency scaling
    ext_factor: f32,     // YaRN extension factor (0 = disabled)
    attn_factor: f32,    // YaRN attention factor
    beta_fast: f32,      // YaRN fast beta
    beta_slow: f32,      // YaRN slow beta
    ctx_orig: i32,       // original context length
}

#[derive(Clone, Copy)]
#[repr(C)]
struct FlashAttnArgs {
    ne01: u32,    // Q rows
    ne02: u32,    // Q heads
    ne03: u32,    // Q batches
    ne11: u32,    // K rows (seq_len)
    ne_12_2: u32, // KV heads
    ne_12_3: u32, // KV batches
    nb01: u32,    // Q stride (bytes)
    nb02: u32,    // Q stride (bytes)
    nb03: u32,    // Q stride (bytes)
    nb11: u32,    // K stride (bytes)
    nb12: u32,    // K stride (bytes)
    nb13: u32,    // K stride (bytes)
    nb21: u32,    // V stride (bytes)
    nb22: u32,    // V stride (bytes)
    nb23: u32,    // V stride (bytes)
    nb31: u32,    // mask stride
    nb32: u32,    // mask stride
    nb33: u32,    // mask stride
    ne32: u32,    // mask rows
    ne33: u32,    // mask batches
    scale: f32,   // 1/sqrt(head_dim)
    max_bias: f32,
    logit_softcap: f32,
    n_head_log2: u32,
    m0: f32,
    m1: f32,
    nsg: u32,
    nwg: u32,
}

const FC_MUL_MV_NSG: NSUInteger = 600;
const FC_MUL_MV_NXPSG: NSUInteger = 601;
const FC_MUL_MV_NE12: NSUInteger = 602;
const FC_MUL_MV_R2: NSUInteger = 603;
const FC_MUL_MV_R3: NSUInteger = 604;
const FC_MUL_MV_SPLIT: NSUInteger = 605;

/// Create function constants for MV kernel specialization
/// nsg: number of simdgroups per threadgroup (typically 2 for decode)
/// ne12: batch dimension of src1 (typically 1 for single vector)
/// r2: ne12 / ne02 (typically 1)
/// r3: ne13 / ne03 (typically 1)
fn create_mv_function_constants(
    nsg: i16,
    ne12: i16,
    r2: i16,
    r3: i16,
) -> FunctionConstantValues {
    let cv = FunctionConstantValues::new();

    // Set NSG (number of simdgroups)
    let nsg_ptr: *const c_void = &nsg as *const i16 as *const c_void;
    cv.set_constant_value_at_index(nsg_ptr, MTLDataType::Short, FC_MUL_MV_NSG);

    // Set NE12
    let ne12_ptr: *const c_void = &ne12 as *const i16 as *const c_void;
    cv.set_constant_value_at_index(
        ne12_ptr,
        MTLDataType::Short,
        FC_MUL_MV_NE12,
    );

    // Set R2
    let r2_ptr: *const c_void = &r2 as *const i16 as *const c_void;
    cv.set_constant_value_at_index(r2_ptr, MTLDataType::Short, FC_MUL_MV_R2);

    // Set R3
    let r3_ptr: *const c_void = &r3 as *const i16 as *const c_void;
    cv.set_constant_value_at_index(r3_ptr, MTLDataType::Short, FC_MUL_MV_R3);

    cv
}

impl MetalContext {
    pub fn new() -> Result<Self, String> {
        let device = Device::system_default().ok_or("No Metal device found")?;

        let queue = device.new_command_queue();

        // Set optimization level for Metal shader compilation
        let compile_options = metal::CompileOptions::new();
        compile_options.set_fast_math_enabled(true);
        // Note: metal-rs 0.31 doesn't expose set_optimization_level directly
        // but the default should be equivalent to -O3

        let vec_dot_source =
            include_str!("../../../ops/metal/gemv/gemv_q80.metal");
        let vec_dot_library = device
            .new_library_with_source(vec_dot_source, &compile_options)
            .map_err(|e| format!("Failed to compile vec_dot library: {}", e))?;

        let mv_source = include_str!("../../../ops/metal/gemv/gemv_q80.metal");
        let mv_library = device
            .new_library_with_source(mv_source, &compile_options)
            .map_err(|e| format!("Failed to compile mv library: {}", e))?;

        let mv_simd_source =
            include_str!("../../../ops/metal/gemv/gemv_q80.metal");
        let mv_simd_library = device
            .new_library_with_source(mv_simd_source, &compile_options)
            .map_err(|e| format!("Failed to compile mv_simd library: {}", e))?;

        // Compile IQ/TQ MV library (includes all IQ formats)
        let mv_iq_tq_source =
            include_str!("../../../ops/metal/gemv/mul_mv_split.metal");
        let iq_grid_tables =
            include_str!("../../../ops/metal/include/iq_grid_tables.h");
        let mv_iq_tq_combined = mv_iq_tq_source.replace(
            "#include \"../include/iq_grid_tables.h\"",
            iq_grid_tables,
        );
        let mv_iq_tq_library = device
            .new_library_with_source(&mv_iq_tq_combined, &compile_options)
            .map_err(|e| {
                format!("Failed to compile mv_iq_tq library: {}", e)
            })?;

        // Compile self-written GEMM library (all formats, self-contained)
        let gemm_source = include_str!("../../../ops/metal/gemm/mul_mm.metal");
        let blocks_h = include_str!("../../../ops/metal/include/blocks.h");
        let iq_grid_tables =
            include_str!("../../../ops/metal/include/iq_grid_tables.h");
        let gemm_combined = gemm_source
            .replace("#include \"../include/blocks.h\"", blocks_h)
            .replace(
                "#include \"../include/iq_grid_tables.h\"",
                iq_grid_tables,
            );
        let gemm_library = device
            .new_library_with_source(&gemm_combined, &compile_options)
            .map_err(|e| format!("Failed to compile gemm library: {}", e))?;

        // Compile self-written K-quant MV library (Q2_K, Q3_K, Q4_K, Q5_K, Q6_K)
        let kquant_mv_source =
            include_str!("../../../ops/metal/gemv/kquant_mv.metal");
        let kquant_mv_combined = kquant_mv_source
            .replace("#include \"../include/blocks.h\"", blocks_h);
        let kquant_mv_library = device
            .new_library_with_source(&kquant_mv_combined, &compile_options)
            .map_err(|e| {
                format!("Failed to compile kquant_mv library: {}", e)
            })?;

        // Compile transposed GEMV library (for output layer with tied embeddings)
        let gemv_transposed_source =
            include_str!("../../../ops/metal/gemv/gemv_transposed.metal");
        let gemv_transposed_library = device
            .new_library_with_source(gemv_transposed_source, &compile_options)
            .map_err(|e| {
                format!("Failed to compile gemv_transposed library: {}", e)
            })?;

        // Create Q6_K transposed MV pipeline
        let mv_q6_k_transposed_kernel = gemv_transposed_library
            .get_function("mv_q6_k_f32_transposed", None)
            .map_err(|e| {
                format!("Failed to get Q6_K transposed kernel: {}", e)
            })?;
        let mv_q6_k_transposed_pipeline = device
            .new_compute_pipeline_state_with_function(
                &mv_q6_k_transposed_kernel,
            )
            .map_err(|e| {
                format!("Failed to create Q6_K transposed pipeline: {}", e)
            })?;

        let mv_f16_kernels_source =
            include_str!("../../../ops/metal/gemv/mul_mv_f16_kernels.metal");
        let mv_f16_library = device
            .new_library_with_source(mv_f16_kernels_source, &compile_options)
            .map_err(|e| {
                format!("Failed to compile F16 GEMV library: {}", e)
            })?;

        // Create three pipelines
        let mv_f16_short_kernel = mv_f16_library
            .get_function("kernel_mul_mv_f16_f32_short", None)
            .map_err(|e| {
                format!("Failed to get F16 GEMV short kernel: {}", e)
            })?;
        let mv_f16_short_pipeline = device
            .new_compute_pipeline_state_with_function(&mv_f16_short_kernel)
            .map_err(|e| {
                format!("Failed to create F16 GEMV short pipeline: {}", e)
            })?;

        let mv_f16_basic_kernel = mv_f16_library
            .get_function("kernel_mul_mv_f16_f32", None)
            .map_err(|e| {
                format!("Failed to get F16 GEMV basic kernel: {}", e)
            })?;
        let mv_f16_basic_pipeline = device
            .new_compute_pipeline_state_with_function(&mv_f16_basic_kernel)
            .map_err(|e| {
                format!("Failed to create F16 GEMV basic pipeline: {}", e)
            })?;

        // F16×F16 GEMV pipeline (for F16 weights + F16 input)
        let mv_f16_f16_kernel = mv_f16_library
            .get_function("kernel_mul_mv_f16_f16", None)
            .map_err(|e| format!("Failed to get F16×F16 GEMV kernel: {}", e))?;
        let mv_f16_f16_pipeline = device
            .new_compute_pipeline_state_with_function(&mv_f16_f16_kernel)
            .map_err(|e| {
                format!("Failed to create F16×F16 GEMV pipeline: {}", e)
            })?;

        // Create separate pipelines for each NSG value
        // Create pipelines with function constants (matching llama.cpp approach)
        // NSG=1
        let mut fc_nsg1 = metal::FunctionConstantValues::new();
        let nsg1: i16 = 1;
        fc_nsg1.set_constant_value_at_index(
            &nsg1 as *const i16 as *const std::ffi::c_void,
            metal::MTLDataType::Short,
            600, // FC_MUL_MV base index
        );
        let mv_f16_4_nsg1_kernel = mv_f16_library
            .get_function("kernel_mul_mv_f16_f32_4", Some(fc_nsg1))
            .map_err(|e| {
                format!("Failed to get F16 GEMV _4 kernel with NSG=1: {}", e)
            })?;
        let mv_f16_4_nsg1_pipeline = device
            .new_compute_pipeline_state_with_function(&mv_f16_4_nsg1_kernel)
            .map_err(|e| {
                format!("Failed to create F16 GEMV _4_nsg1 pipeline: {}", e)
            })?;

        // NSG=2
        let mut fc_nsg2 = metal::FunctionConstantValues::new();
        let nsg2: i16 = 2;
        fc_nsg2.set_constant_value_at_index(
            &nsg2 as *const i16 as *const std::ffi::c_void,
            metal::MTLDataType::Short,
            600,
        );
        let mv_f16_4_nsg2_kernel = mv_f16_library
            .get_function("kernel_mul_mv_f16_f32_4", Some(fc_nsg2))
            .map_err(|e| {
                format!("Failed to get F16 GEMV _4 kernel with NSG=2: {}", e)
            })?;
        let mv_f16_4_nsg2_pipeline = device
            .new_compute_pipeline_state_with_function(&mv_f16_4_nsg2_kernel)
            .map_err(|e| {
                format!("Failed to create F16 GEMV _4_nsg2 pipeline: {}", e)
            })?;

        // NSG=4
        let mut fc_nsg4 = metal::FunctionConstantValues::new();
        let nsg4: i16 = 4;
        fc_nsg4.set_constant_value_at_index(
            &nsg4 as *const i16 as *const std::ffi::c_void,
            metal::MTLDataType::Short,
            600,
        );
        let mv_f16_4_nsg4_kernel = mv_f16_library
            .get_function("kernel_mul_mv_f16_f32_4", Some(fc_nsg4))
            .map_err(|e| {
                format!("Failed to get F16 GEMV _4 kernel with NSG=4: {}", e)
            })?;
        let mv_f16_4_nsg4_pipeline = device
            .new_compute_pipeline_state_with_function(&mv_f16_4_nsg4_kernel)
            .map_err(|e| {
                format!("Failed to create F16 GEMV _4_nsg4 pipeline: {}", e)
            })?;

        // Load optimized F16 GEMV library (llama.cpp-style with nxpsg)
        let mv_f16_optimized_source =
            include_str!("../../../ops/metal/gemv/mul_mv_f16_optimized.metal");
        let mv_f16_optimized_library = device
            .new_library_with_source(mv_f16_optimized_source, &compile_options)
            .map_err(|e| {
                format!("Failed to compile F16 GEMV optimized library: {}", e)
            })?;

        // Create optimized pipelines for r1ptg=1 (decode scenario)
        // For K >= 256 && batch=1, use nxpsg=16
        let mut fc_opt_nxpsg16 = metal::FunctionConstantValues::new();
        let nsg_opt: i16 = 1;
        let nxpsg16: i16 = 16;
        let ne12_opt: i16 = 1;
        let r2_opt: i16 = 1;
        let r3_opt: i16 = 1;
        fc_opt_nxpsg16.set_constant_value_at_index(
            &nsg_opt as *const i16 as *const std::ffi::c_void,
            metal::MTLDataType::Short,
            600,
        );
        fc_opt_nxpsg16.set_constant_value_at_index(
            &nxpsg16 as *const i16 as *const std::ffi::c_void,
            metal::MTLDataType::Short,
            601,
        );
        fc_opt_nxpsg16.set_constant_value_at_index(
            &ne12_opt as *const i16 as *const std::ffi::c_void,
            metal::MTLDataType::Short,
            602,
        );
        fc_opt_nxpsg16.set_constant_value_at_index(
            &r2_opt as *const i16 as *const std::ffi::c_void,
            metal::MTLDataType::Short,
            603,
        );
        fc_opt_nxpsg16.set_constant_value_at_index(
            &r3_opt as *const i16 as *const std::ffi::c_void,
            metal::MTLDataType::Short,
            604,
        );
        let mv_f16_opt_r1_1_nxpsg16_kernel = mv_f16_optimized_library
            .get_function(
                "kernel_mul_mv_ext_f16_f32_r1_1",
                Some(fc_opt_nxpsg16),
            )
            .map_err(|e| {
                format!(
                    "Failed to get optimized F16 kernel r1=1 nxpsg=16: {}",
                    e
                )
            })?;
        let mv_f16_opt_r1_1_nxpsg16_pipeline = device
            .new_compute_pipeline_state_with_function(
                &mv_f16_opt_r1_1_nxpsg16_kernel,
            )
            .map_err(|e| {
                format!(
                    "Failed to create optimized F16 pipeline r1=1 nxpsg=16: {}",
                    e
                )
            })?;

        // Compile RMS Norm library
        let rms_norm_source =
            include_str!("../../../ops/metal/ops/rms_norm.metal");
        let rms_norm_library = device
            .new_library_with_source(rms_norm_source, &compile_options)
            .map_err(|e| {
                format!("Failed to compile rms_norm library: {}", e)
            })?;

        // Compile Softmax library
        let softmax_source =
            include_str!("../../../ops/metal/ops/softmax.metal");
        let softmax_library = device
            .new_library_with_source(softmax_source, &compile_options)
            .map_err(|e| format!("Failed to compile softmax library: {}", e))?;

        // Compile ROPE library
        let rope_source = include_str!("../../../ops/metal/ops/rope.metal");
        let rope_library = device
            .new_library_with_source(rope_source, &compile_options)
            .map_err(|e| format!("Failed to compile rope library: {}", e))?;

        // Compile Activation library
        let activation_source =
            include_str!("../../../ops/metal/ops/activation.metal");
        let activation_library = device
            .new_library_with_source(activation_source, &compile_options)
            .map_err(|e| {
                format!("Failed to compile activation library: {}", e)
            })?;

        // Compile Fused Norm Linear library
        let fused_norm_linear_source =
            include_str!("../../../ops/metal/ops/fused_norm_linear.metal");
        // Replace include with actual content (needed for Q4_K fused kernel)
        let fused_norm_linear_combined = fused_norm_linear_source
            .replace("#include \"../include/blocks.h\"", blocks_h);
        let fused_norm_linear_library = device
            .new_library_with_source(
                &fused_norm_linear_combined,
                &compile_options,
            )
            .map_err(|e| {
                format!("Failed to compile fused_norm_linear library: {}", e)
            })?;

        // Compile Fused Add Norm library
        let fused_add_norm_source =
            include_str!("../../../ops/metal/ops/fused_add_norm.metal");
        let fused_add_norm_library = device
            .new_library_with_source(fused_add_norm_source, &compile_options)
            .map_err(|e| {
                format!("Failed to compile fused_add_norm library: {}", e)
            })?;

        // Compile FusedGateUp library
        let fused_gate_up_source =
            include_str!("../../../ops/metal/ops/fused_gate_up.metal");
        // Replace include with actual content (needed for quantized kernels)
        let fused_gate_up_combined = fused_gate_up_source
            .replace("#include \"../include/blocks.h\"", blocks_h);
        let fused_gate_up_library = device
            .new_library_with_source(&fused_gate_up_combined, &compile_options)
            .map_err(|e| {
                format!("Failed to compile fused_gate_up library: {}", e)
            })?;

        // Compile Flash Attention library (vectorized decode kernel)
        let flash_attn_source =
            include_str!("../../../ops/metal/attention/flash_decode_vec.metal");
        let flash_attn_library = device
            .new_library_with_source(flash_attn_source, &compile_options)
            .map_err(|e| {
                format!("Failed to compile flash_attn library: {}", e)
            })?;

        // Compile Flash Attention MMA library (experimental)
        let flash_attn_mma_source =
            include_str!("../../../ops/metal/attention/flash_decode_mma.metal");
        let flash_attn_mma_library = device
            .new_library_with_source(flash_attn_mma_source, &compile_options)
            .map_err(|e| {
                format!("Failed to compile flash_attn_mma library: {}", e)
            })?;

        // Compile Flash Attention MMA v2 library (optimized)
        let flash_attn_mma_v2_source = include_str!(
            "../../../ops/metal/attention/flash_decode_mma_v2.metal"
        );
        let flash_attn_mma_v2_library = device
            .new_library_with_source(flash_attn_mma_v2_source, &compile_options)
            .map_err(|e| {
                format!("Failed to compile flash_attn_mma_v2 library: {}", e)
            })?;

        // Compile Flash Prefill library (for batched attention)
        let flash_prefill_source =
            include_str!("../../../ops/metal/attention/flash_attn_quant.metal");
        let flash_prefill_library = device
            .new_library_with_source(flash_prefill_source, &compile_options)
            .map_err(|e| {
                format!("Failed to compile flash_prefill library: {}", e)
            })?;

        // Compile Quantize library
        let quantize_source =
            include_str!("../../../ops/metal/ops/quantize.metal");
        let quantize_library = device
            .new_library_with_source(quantize_source, &compile_options)
            .map_err(|e| {
                format!("Failed to compile quantize library: {}", e)
            })?;

        // Compile KV Update library (F32->F16 for KV cache)
        // Note: KV update kernels already exist in ops/kv_update.metal
        // We use the existing implementation which includes:
        // - kv_update_f32_to_f16 for K cache
        // - kv_update_v_f32_to_f16 for V cache

        // Create cached pipeline for Q4_0 MV (from template)
        // Q4_0 kernel also requires function constants
        let mv_q4_0_kernel = gemm_library
            .get_function(
                "kernel_mul_mm_q4_0_f32",
                Some(create_mv_function_constants(2, 1, 1, 1)),
            )
            .map_err(|e| format!("Failed to get Q4_0 kernel: {}", e))?;
        let mv_q4_0_pipeline = device
            .new_compute_pipeline_state_with_function(&mv_q4_0_kernel)
            .map_err(|e| format!("Failed to create Q4_0 pipeline: {}", e))?;

        // Standard Q kernels (from gemm.metal)
        let mv_q1_0_kernel = gemm_library
            .get_function(
                "kernel_mul_mm_q1_0_f32",
                Some(create_mv_function_constants(8, 1, 1, 1)),
            )
            .map_err(|e| format!("Failed to get Q1_0 kernel: {}", e))?;
        let mv_q1_0_pipeline = device
            .new_compute_pipeline_state_with_function(&mv_q1_0_kernel)
            .map_err(|e| format!("Failed to create Q1_0 pipeline: {}", e))?;

        let mv_q2_0_kernel = gemm_library
            .get_function(
                "kernel_mul_mm_q2_0_f32",
                Some(create_mv_function_constants(8, 1, 1, 1)),
            )
            .map_err(|e| format!("Failed to get Q2_0 kernel: {}", e))?;
        let mv_q2_0_pipeline = device
            .new_compute_pipeline_state_with_function(&mv_q2_0_kernel)
            .map_err(|e| format!("Failed to create Q2_0 pipeline: {}", e))?;

        let mv_q4_1_kernel = gemm_library
            .get_function(
                "kernel_mul_mm_q4_1_f32",
                Some(create_mv_function_constants(4, 1, 1, 1)),
            )
            .map_err(|e| format!("Failed to get Q4_1 kernel: {}", e))?;
        let mv_q4_1_pipeline = device
            .new_compute_pipeline_state_with_function(&mv_q4_1_kernel)
            .map_err(|e| format!("Failed to create Q4_1 pipeline: {}", e))?;

        let mv_q5_0_kernel = gemm_library
            .get_function(
                "kernel_mul_mm_q5_0_f32",
                Some(create_mv_function_constants(4, 1, 1, 1)),
            )
            .map_err(|e| format!("Failed to get Q5_0 kernel: {}", e))?;
        let mv_q5_0_pipeline = device
            .new_compute_pipeline_state_with_function(&mv_q5_0_kernel)
            .map_err(|e| format!("Failed to create Q5_0 pipeline: {}", e))?;

        let mv_q5_1_kernel = gemm_library
            .get_function(
                "kernel_mul_mm_q5_1_f32",
                Some(create_mv_function_constants(4, 1, 1, 1)),
            )
            .map_err(|e| format!("Failed to get Q5_1 kernel: {}", e))?;
        let mv_q5_1_pipeline = device
            .new_compute_pipeline_state_with_function(&mv_q5_1_kernel)
            .map_err(|e| format!("Failed to create Q5_1 pipeline: {}", e))?;

        let mv_q8_0_kernel = gemm_library
            .get_function(
                "kernel_mul_mm_q8_0_f32",
                Some(create_mv_function_constants(2, 1, 1, 1)),
            )
            .map_err(|e| format!("Failed to get Q8_0 kernel: {}", e))?;
        let mv_q8_0_pipeline = device
            .new_compute_pipeline_state_with_function(&mv_q8_0_kernel)
            .map_err(|e| format!("Failed to create Q8_0 pipeline: {}", e))?;

        // Create cached pipelines for K-quant MV (using self-written kernels)
        // Self-written kernels don't use function constants

        let mv_q4_k_kernel = kquant_mv_library
            .get_function("kernel_mul_mv_q4_K_f32", None)
            .map_err(|e| format!("Failed to get Q4_K kernel: {}", e))?;
        let mv_q4_k_pipeline = device
            .new_compute_pipeline_state_with_function(&mv_q4_k_kernel)
            .map_err(|e| format!("Failed to create Q4_K pipeline: {}", e))?;

        let mv_q5_k_kernel = kquant_mv_library
            .get_function("kernel_mul_mv_q5_K_f32", None)
            .map_err(|e| format!("Failed to get Q5_K kernel: {}", e))?;
        let mv_q5_k_pipeline = device
            .new_compute_pipeline_state_with_function(&mv_q5_k_kernel)
            .map_err(|e| format!("Failed to create Q5_K pipeline: {}", e))?;

        let mv_q2_k_kernel = kquant_mv_library
            .get_function("kernel_mul_mv_q2_K_f32", None)
            .map_err(|e| format!("Failed to get Q2_K kernel: {}", e))?;
        let mv_q2_k_pipeline = device
            .new_compute_pipeline_state_with_function(&mv_q2_k_kernel)
            .map_err(|e| format!("Failed to create Q2_K pipeline: {}", e))?;

        let mv_q6_k_kernel = kquant_mv_library
            .get_function("kernel_mul_mv_q6_K_f32", None)
            .map_err(|e| format!("Failed to get Q6_K kernel: {}", e))?;
        let mv_q6_k_pipeline = device
            .new_compute_pipeline_state_with_function(&mv_q6_k_kernel)
            .map_err(|e| format!("Failed to create Q6_K pipeline: {}", e))?;

        let mv_q3_k_kernel = kquant_mv_library
            .get_function("kernel_mul_mv_q3_K_f32", None)
            .map_err(|e| format!("Failed to get Q3_K kernel: {}", e))?;
        let mv_q3_k_pipeline = device
            .new_compute_pipeline_state_with_function(&mv_q3_k_kernel)
            .map_err(|e| format!("Failed to create Q3_K pipeline: {}", e))?;

        // Create fused GEMV+Add pipelines for residual connections
        let fused_gemv_add_q4_k_kernel = kquant_mv_library
            .get_function("kernel_fused_gemv_add_q4_K", None)
            .map_err(|e| {
                format!("Failed to get fused GEMV+Add Q4_K kernel: {}", e)
            })?;
        let fused_gemv_add_q4_k_pipeline = device
            .new_compute_pipeline_state_with_function(
                &fused_gemv_add_q4_k_kernel,
            )
            .map_err(|e| {
                format!("Failed to create fused GEMV+Add Q4_K pipeline: {}", e)
            })?;

        let fused_gemv_add_q5_k_kernel = kquant_mv_library
            .get_function("kernel_fused_gemv_add_q5_K", None)
            .map_err(|e| {
                format!("Failed to get fused GEMV+Add Q5_K kernel: {}", e)
            })?;
        let fused_gemv_add_q5_k_pipeline = device
            .new_compute_pipeline_state_with_function(
                &fused_gemv_add_q5_k_kernel,
            )
            .map_err(|e| {
                format!("Failed to create fused GEMV+Add Q5_K pipeline: {}", e)
            })?;

        let fused_gemv_add_q6_k_kernel = kquant_mv_library
            .get_function("kernel_fused_gemv_add_q6_K", None)
            .map_err(|e| {
                format!("Failed to get fused GEMV+Add Q6_K kernel: {}", e)
            })?;
        let fused_gemv_add_q6_k_pipeline = device
            .new_compute_pipeline_state_with_function(
                &fused_gemv_add_q6_k_kernel,
            )
            .map_err(|e| {
                format!("Failed to create fused GEMV+Add Q6_K pipeline: {}", e)
            })?;

        // Create cached pipeline for IQ4_NL MV
        let mv_iq4_nl_kernel = gemm_library
            .get_function(
                "kernel_mul_mm_iq4_nl_f32",
                Some(create_mv_function_constants(2, 1, 1, 1)),
            )
            .map_err(|e| format!("Failed to get IQ4_NL kernel: {}", e))?;
        let mv_iq4_nl_pipeline = device
            .new_compute_pipeline_state_with_function(&mv_iq4_nl_kernel)
            .map_err(|e| format!("Failed to create IQ4_NL pipeline: {}", e))?;

        // Create cached pipeline for TQ2_0 MV
        let mv_tq2_0_kernel = gemm_library
            .get_function(
                "kernel_mul_mm_tq2_0_f32",
                Some(create_mv_function_constants(2, 1, 1, 1)),
            )
            .map_err(|e| format!("Failed to get TQ2_0 kernel: {}", e))?;
        let mv_tq2_0_pipeline = device
            .new_compute_pipeline_state_with_function(&mv_tq2_0_kernel)
            .map_err(|e| format!("Failed to create TQ2_0 pipeline: {}", e))?;

        // Create cached pipeline for IQ4_XS MV (use GEMV kernel, not GEMM!)
        let mv_iq4_xs_kernel = mv_iq_tq_library
            .get_function("kernel_mul_mv_iq4_xs_f32", None)
            .map_err(|e| format!("Failed to get IQ4_XS MV kernel: {}", e))?;
        let mv_iq4_xs_pipeline = device
            .new_compute_pipeline_state_with_function(&mv_iq4_xs_kernel)
            .map_err(|e| {
                format!("Failed to create IQ4_XS MV pipeline: {}", e)
            })?;

        // Create cached pipeline for TQ1_0 MV
        let mv_tq1_0_kernel = gemm_library
            .get_function(
                "kernel_mul_mm_tq1_0_f32",
                Some(create_mv_function_constants(4, 1, 1, 1)),
            )
            .map_err(|e| format!("Failed to get TQ1_0 kernel: {}", e))?;
        let mv_tq1_0_pipeline = device
            .new_compute_pipeline_state_with_function(&mv_tq1_0_kernel)
            .map_err(|e| format!("Failed to create TQ1_0 pipeline: {}", e))?;

        // Create pipelines for IQ series
        let mv_iq1_s_kernel = gemm_library
            .get_function(
                "kernel_mul_mm_iq1_s_f32",
                Some(create_mv_function_constants(4, 1, 1, 1)),
            )
            .map_err(|e| format!("Failed to get IQ1_S kernel: {}", e))?;
        let mv_iq1_s_pipeline = device
            .new_compute_pipeline_state_with_function(&mv_iq1_s_kernel)
            .map_err(|e| format!("Failed to create IQ1_S pipeline: {}", e))?;

        let mv_iq1_m_kernel = gemm_library
            .get_function(
                "kernel_mul_mm_iq1_m_f32",
                Some(create_mv_function_constants(4, 1, 1, 1)),
            )
            .map_err(|e| format!("Failed to get IQ1_M kernel: {}", e))?;
        let mv_iq1_m_pipeline = device
            .new_compute_pipeline_state_with_function(&mv_iq1_m_kernel)
            .map_err(|e| format!("Failed to create IQ1_M pipeline: {}", e))?;

        let mv_iq2_xxs_kernel = gemm_library
            .get_function(
                "kernel_mul_mm_iq2_xxs_f32",
                Some(create_mv_function_constants(2, 1, 1, 1)),
            )
            .map_err(|e| format!("Failed to get IQ2_XXS kernel: {}", e))?;
        let mv_iq2_xxs_pipeline = device
            .new_compute_pipeline_state_with_function(&mv_iq2_xxs_kernel)
            .map_err(|e| format!("Failed to create IQ2_XXS pipeline: {}", e))?;

        let mv_iq2_xxs_split1_kernel = mv_iq_tq_library
            .get_function("kernel_mul_mv_iq2_xxs_f32_split1", None)
            .map_err(|e| {
                format!("Failed to get IQ2_XXS split1 kernel: {}", e)
            })?;
        let mv_iq2_xxs_split1_pipeline = device
            .new_compute_pipeline_state_with_function(&mv_iq2_xxs_split1_kernel)
            .map_err(|e| {
                format!("Failed to create IQ2_XXS split1 pipeline: {}", e)
            })?;

        let mv_iq2_xs_kernel = gemm_library
            .get_function(
                "kernel_mul_mm_iq2_xs_f32",
                Some(create_mv_function_constants(2, 1, 1, 1)),
            )
            .map_err(|e| format!("Failed to get IQ2_XS kernel: {}", e))?;
        let mv_iq2_xs_pipeline = device
            .new_compute_pipeline_state_with_function(&mv_iq2_xs_kernel)
            .map_err(|e| format!("Failed to create IQ2_XS pipeline: {}", e))?;

        let mv_iq2_xs_split1_kernel = mv_iq_tq_library
            .get_function("kernel_mul_mv_iq2_xs_f32_split1", None)
            .map_err(|e| {
                format!("Failed to get IQ2_XS split1 kernel: {}", e)
            })?;
        let mv_iq2_xs_split1_pipeline = device
            .new_compute_pipeline_state_with_function(&mv_iq2_xs_split1_kernel)
            .map_err(|e| {
                format!("Failed to create IQ2_XS split1 pipeline: {}", e)
            })?;

        let mv_iq2_s_kernel = mv_iq_tq_library
            .get_function("kernel_mul_mv_iq2_s_f32_split0", None)
            .map_err(|e| format!("Failed to get IQ2_S kernel: {}", e))?;
        let mv_iq2_s_pipeline = device
            .new_compute_pipeline_state_with_function(&mv_iq2_s_kernel)
            .map_err(|e| format!("Failed to create IQ2_S pipeline: {}", e))?;

        let mv_iq2_s_split1_kernel = mv_iq_tq_library
            .get_function("kernel_mul_mv_iq2_s_f32_split1", None)
            .map_err(|e| format!("Failed to get IQ2_S split1 kernel: {}", e))?;
        let mv_iq2_s_split1_pipeline = device
            .new_compute_pipeline_state_with_function(&mv_iq2_s_split1_kernel)
            .map_err(|e| {
                format!("Failed to create IQ2_S split1 pipeline: {}", e)
            })?;

        let mv_iq3_xxs_kernel = gemm_library
            .get_function(
                "kernel_mul_mm_iq3_xxs_f32",
                Some(create_mv_function_constants(2, 1, 1, 1)),
            )
            .map_err(|e| format!("Failed to get IQ3_XXS kernel: {}", e))?;
        let mv_iq3_xxs_pipeline = device
            .new_compute_pipeline_state_with_function(&mv_iq3_xxs_kernel)
            .map_err(|e| format!("Failed to create IQ3_XXS pipeline: {}", e))?;

        let mv_iq3_xxs_split1_kernel = mv_iq_tq_library
            .get_function("kernel_mul_mv_iq3_xxs_f32_split1", None)
            .map_err(|e| {
                format!("Failed to get IQ3_XXS split1 kernel: {}", e)
            })?;
        let mv_iq3_xxs_split1_pipeline = device
            .new_compute_pipeline_state_with_function(&mv_iq3_xxs_split1_kernel)
            .map_err(|e| {
                format!("Failed to create IQ3_XXS split1 pipeline: {}", e)
            })?;

        let mv_iq3_s_kernel = gemm_library
            .get_function(
                "kernel_mul_mm_iq3_s_f32",
                Some(create_mv_function_constants(2, 1, 1, 1)),
            )
            .map_err(|e| format!("Failed to get IQ3_S kernel: {}", e))?;
        let mv_iq3_s_pipeline = device
            .new_compute_pipeline_state_with_function(&mv_iq3_s_kernel)
            .map_err(|e| format!("Failed to create IQ3_S pipeline: {}", e))?;

        let mv_iq3_s_split1_kernel = mv_iq_tq_library
            .get_function("kernel_mul_mv_iq3_s_f32_split1", None)
            .map_err(|e| format!("Failed to get IQ3_S split1 kernel: {}", e))?;
        let mv_iq3_s_split1_pipeline = device
            .new_compute_pipeline_state_with_function(&mv_iq3_s_split1_kernel)
            .map_err(|e| {
                format!("Failed to create IQ3_S split1 pipeline: {}", e)
            })?;

        // MXFP4 kernel
        let mv_mxfp4_kernel = gemm_library
            .get_function(
                "kernel_mul_mm_mxfp4_f32",
                Some(create_mv_function_constants(2, 1, 1, 1)),
            )
            .map_err(|e| format!("Failed to get MXFP4 kernel: {}", e))?;
        let mv_mxfp4_pipeline = device
            .new_compute_pipeline_state_with_function(&mv_mxfp4_kernel)
            .map_err(|e| format!("Failed to create MXFP4 pipeline: {}", e))?;

        // Requires function constant specialization
        // Function constant IDs (from mul_mm.metal):
        //   700: FC_mul_mm_bc_inp (bool)
        //   701: FC_mul_mm_bc_out (bool)
        //   702: FC_mul_mm_ne12 (short)
        //   703: FC_mul_mm_ne13 (short)
        //   704: FC_mul_mm_r2 (short)
        //   705: FC_mul_mm_r3 (short)
        let gemm_template_q4_k_kernel = {
            let constant_values = metal::FunctionConstantValues::new();
            let false_val: bool = false;
            let one_val: i16 = 1;
            constant_values.set_constant_value_at_index(
                &false_val as *const _ as *const std::ffi::c_void,
                metal::MTLDataType::Bool,
                700,
            );
            constant_values.set_constant_value_at_index(
                &false_val as *const _ as *const std::ffi::c_void,
                metal::MTLDataType::Bool,
                701,
            );
            constant_values.set_constant_value_at_index(
                &one_val as *const _ as *const std::ffi::c_void,
                metal::MTLDataType::Short,
                702,
            );
            constant_values.set_constant_value_at_index(
                &one_val as *const _ as *const std::ffi::c_void,
                metal::MTLDataType::Short,
                703,
            );
            constant_values.set_constant_value_at_index(
                &one_val as *const _ as *const std::ffi::c_void,
                metal::MTLDataType::Short,
                704,
            );
            constant_values.set_constant_value_at_index(
                &one_val as *const _ as *const std::ffi::c_void,
                metal::MTLDataType::Short,
                705,
            );
            let descriptor = metal::FunctionDescriptor::new();
            descriptor.set_name("kernel_mul_mm_q4_K_f32");
            descriptor.set_constant_values(&constant_values);
            gemm_library.new_function_with_descriptor(&descriptor).map_err(
                |e| format!("Failed to get templated Q4_K kernel: {}", e),
            )?
        };
        let gemm_template_q4_k_pipeline = device
            .new_compute_pipeline_state_with_function(
                &gemm_template_q4_k_kernel,
            )
            .map_err(|e| {
                format!("Failed to create templated Q4_K pipeline: {}", e)
            })?;

        // Create remaining template kernels with same constants
        let gemm_template_q2_k_kernel = {
            let constant_values = metal::FunctionConstantValues::new();
            let false_val: bool = false;
            let one_val: i16 = 1;
            constant_values.set_constant_value_at_index(
                &false_val as *const _ as *const std::ffi::c_void,
                metal::MTLDataType::Bool,
                700,
            );
            constant_values.set_constant_value_at_index(
                &false_val as *const _ as *const std::ffi::c_void,
                metal::MTLDataType::Bool,
                701,
            );
            constant_values.set_constant_value_at_index(
                &one_val as *const _ as *const std::ffi::c_void,
                metal::MTLDataType::Short,
                702,
            );
            constant_values.set_constant_value_at_index(
                &one_val as *const _ as *const std::ffi::c_void,
                metal::MTLDataType::Short,
                703,
            );
            constant_values.set_constant_value_at_index(
                &one_val as *const _ as *const std::ffi::c_void,
                metal::MTLDataType::Short,
                704,
            );
            constant_values.set_constant_value_at_index(
                &one_val as *const _ as *const std::ffi::c_void,
                metal::MTLDataType::Short,
                705,
            );
            let descriptor = metal::FunctionDescriptor::new();
            descriptor.set_name("kernel_mul_mm_q2_K_f32");
            descriptor.set_constant_values(&constant_values);
            gemm_library.new_function_with_descriptor(&descriptor).map_err(
                |e| format!("Failed to get templated Q2_K kernel: {}", e),
            )?
        };
        let gemm_template_q2_k_pipeline = device
            .new_compute_pipeline_state_with_function(
                &gemm_template_q2_k_kernel,
            )
            .map_err(|e| {
                format!("Failed to create templated Q2_K pipeline: {}", e)
            })?;

        let gemm_template_q3_k_kernel = {
            let constant_values = metal::FunctionConstantValues::new();
            let false_val: bool = false;
            let one_val: i16 = 1;
            constant_values.set_constant_value_at_index(
                &false_val as *const _ as *const std::ffi::c_void,
                metal::MTLDataType::Bool,
                700,
            );
            constant_values.set_constant_value_at_index(
                &false_val as *const _ as *const std::ffi::c_void,
                metal::MTLDataType::Bool,
                701,
            );
            constant_values.set_constant_value_at_index(
                &one_val as *const _ as *const std::ffi::c_void,
                metal::MTLDataType::Short,
                702,
            );
            constant_values.set_constant_value_at_index(
                &one_val as *const _ as *const std::ffi::c_void,
                metal::MTLDataType::Short,
                703,
            );
            constant_values.set_constant_value_at_index(
                &one_val as *const _ as *const std::ffi::c_void,
                metal::MTLDataType::Short,
                704,
            );
            constant_values.set_constant_value_at_index(
                &one_val as *const _ as *const std::ffi::c_void,
                metal::MTLDataType::Short,
                705,
            );
            let descriptor = metal::FunctionDescriptor::new();
            descriptor.set_name("kernel_mul_mm_q3_K_f32");
            descriptor.set_constant_values(&constant_values);
            gemm_library.new_function_with_descriptor(&descriptor).map_err(
                |e| format!("Failed to get templated Q3_K kernel: {}", e),
            )?
        };
        let gemm_template_q3_k_pipeline = device
            .new_compute_pipeline_state_with_function(
                &gemm_template_q3_k_kernel,
            )
            .map_err(|e| {
                format!("Failed to create templated Q3_K pipeline: {}", e)
            })?;

        let gemm_template_q5_k_kernel = {
            let constant_values = metal::FunctionConstantValues::new();
            let false_val: bool = false;
            let one_val: i16 = 1;
            constant_values.set_constant_value_at_index(
                &false_val as *const _ as *const std::ffi::c_void,
                metal::MTLDataType::Bool,
                700,
            );
            constant_values.set_constant_value_at_index(
                &false_val as *const _ as *const std::ffi::c_void,
                metal::MTLDataType::Bool,
                701,
            );
            constant_values.set_constant_value_at_index(
                &one_val as *const _ as *const std::ffi::c_void,
                metal::MTLDataType::Short,
                702,
            );
            constant_values.set_constant_value_at_index(
                &one_val as *const _ as *const std::ffi::c_void,
                metal::MTLDataType::Short,
                703,
            );
            constant_values.set_constant_value_at_index(
                &one_val as *const _ as *const std::ffi::c_void,
                metal::MTLDataType::Short,
                704,
            );
            constant_values.set_constant_value_at_index(
                &one_val as *const _ as *const std::ffi::c_void,
                metal::MTLDataType::Short,
                705,
            );
            let descriptor = metal::FunctionDescriptor::new();
            descriptor.set_name("kernel_mul_mm_q5_K_f32");
            descriptor.set_constant_values(&constant_values);
            gemm_library.new_function_with_descriptor(&descriptor).map_err(
                |e| format!("Failed to get templated Q5_K kernel: {}", e),
            )?
        };
        let gemm_template_q5_k_pipeline = device
            .new_compute_pipeline_state_with_function(
                &gemm_template_q5_k_kernel,
            )
            .map_err(|e| {
                format!("Failed to create templated Q5_K pipeline: {}", e)
            })?;

        let gemm_template_q6_k_kernel = {
            let constant_values = metal::FunctionConstantValues::new();
            let false_val: bool = false;
            let one_val: i16 = 1;
            constant_values.set_constant_value_at_index(
                &false_val as *const _ as *const std::ffi::c_void,
                metal::MTLDataType::Bool,
                700,
            );
            constant_values.set_constant_value_at_index(
                &false_val as *const _ as *const std::ffi::c_void,
                metal::MTLDataType::Bool,
                701,
            );
            constant_values.set_constant_value_at_index(
                &one_val as *const _ as *const std::ffi::c_void,
                metal::MTLDataType::Short,
                702,
            );
            constant_values.set_constant_value_at_index(
                &one_val as *const _ as *const std::ffi::c_void,
                metal::MTLDataType::Short,
                703,
            );
            constant_values.set_constant_value_at_index(
                &one_val as *const _ as *const std::ffi::c_void,
                metal::MTLDataType::Short,
                704,
            );
            constant_values.set_constant_value_at_index(
                &one_val as *const _ as *const std::ffi::c_void,
                metal::MTLDataType::Short,
                705,
            );
            let descriptor = metal::FunctionDescriptor::new();
            descriptor.set_name("kernel_mul_mm_q6_K_f32");
            descriptor.set_constant_values(&constant_values);
            gemm_library.new_function_with_descriptor(&descriptor).map_err(
                |e| format!("Failed to get templated Q6_K kernel: {}", e),
            )?
        };
        let gemm_template_q6_k_pipeline = device
            .new_compute_pipeline_state_with_function(
                &gemm_template_q6_k_kernel,
            )
            .map_err(|e| {
                format!("Failed to create templated Q6_K pipeline: {}", e)
            })?;

        // Create RMS Norm pipelines
        let rms_norm_kernel = rms_norm_library
            .get_function("kernel_rms_norm_f32", None)
            .map_err(|e| format!("Failed to get RMS Norm kernel: {}", e))?;
        let rms_norm_pipeline = device
            .new_compute_pipeline_state_with_function(&rms_norm_kernel)
            .map_err(|e| {
                format!("Failed to create RMS Norm pipeline: {}", e)
            })?;

        let rms_norm_4_kernel = rms_norm_library
            .get_function("kernel_rms_norm_f32_4", None)
            .map_err(|e| format!("Failed to get RMS Norm 4 kernel: {}", e))?;
        let rms_norm_4_pipeline = device
            .new_compute_pipeline_state_with_function(&rms_norm_4_kernel)
            .map_err(|e| {
                format!("Failed to create RMS Norm 4 pipeline: {}", e)
            })?;

        // Create Softmax pipelines
        let softmax_fast_kernel = softmax_library
            .get_function("kernel_soft_max_f32_fast", None)
            .map_err(|e| format!("Failed to get Softmax fast kernel: {}", e))?;
        let softmax_fast_pipeline = device
            .new_compute_pipeline_state_with_function(&softmax_fast_kernel)
            .map_err(|e| {
                format!("Failed to create Softmax fast pipeline: {}", e)
            })?;

        let softmax_4_fast_kernel = softmax_library
            .get_function("kernel_soft_max_f32_4_fast", None)
            .map_err(|e| {
                format!("Failed to get Softmax 4 fast kernel: {}", e)
            })?;
        let softmax_4_fast_pipeline = device
            .new_compute_pipeline_state_with_function(&softmax_4_fast_kernel)
            .map_err(|e| {
                format!("Failed to create Softmax 4 fast pipeline: {}", e)
            })?;

        let softmax_kernel = softmax_library
            .get_function("kernel_soft_max_f32_fast", None)
            .map_err(|e| format!("Failed to get Softmax kernel: {}", e))?;
        let softmax_pipeline = device
            .new_compute_pipeline_state_with_function(&softmax_kernel)
            .map_err(|e| format!("Failed to create Softmax pipeline: {}", e))?;

        let softmax_4_kernel = softmax_library
            .get_function("kernel_soft_max_f32_4_fast", None)
            .map_err(|e| format!("Failed to get Softmax 4 kernel: {}", e))?;
        let softmax_4_pipeline = device
            .new_compute_pipeline_state_with_function(&softmax_4_kernel)
            .map_err(|e| {
                format!("Failed to create Softmax 4 pipeline: {}", e)
            })?;

        // Create ROPE pipeline
        let rope_kernel = rope_library
            .get_function("kernel_rope_neox_f32", None)
            .map_err(|e| format!("Failed to get ROPE kernel: {}", e))?;
        let rope_pipeline = device
            .new_compute_pipeline_state_with_function(&rope_kernel)
            .map_err(|e| format!("Failed to create ROPE pipeline: {}", e))?;

        // Create RoPE F16 pipeline (for KV cache)
        let rope_f16_kernel = rope_library
            .get_function("kernel_rope_neox_f16", None)
            .map_err(|e| format!("Failed to get ROPE F16 kernel: {}", e))?;
        let rope_f16_pipeline = device
            .new_compute_pipeline_state_with_function(&rope_f16_kernel)
            .map_err(|e| {
                format!("Failed to create ROPE F16 pipeline: {}", e)
            })?;

        // Create Activation pipelines
        let add_kernel = activation_library
            .get_function("kernel_add_f32", None)
            .map_err(|e| format!("Failed to get Add kernel: {}", e))?;
        let add_pipeline = device
            .new_compute_pipeline_state_with_function(&add_kernel)
            .map_err(|e| format!("Failed to create Add pipeline: {}", e))?;

        let mul_kernel = activation_library
            .get_function("kernel_mul_f32", None)
            .map_err(|e| format!("Failed to get Mul kernel: {}", e))?;
        let mul_pipeline = device
            .new_compute_pipeline_state_with_function(&mul_kernel)
            .map_err(|e| format!("Failed to create Mul pipeline: {}", e))?;

        let silu_kernel = activation_library
            .get_function("kernel_silu_f32_4", None)
            .map_err(|e| format!("Failed to get SiLU kernel: {}", e))?;
        let silu_pipeline = device
            .new_compute_pipeline_state_with_function(&silu_kernel)
            .map_err(|e| format!("Failed to create SiLU pipeline: {}", e))?;

        let fused_swi_glu_kernel = activation_library
            .get_function("kernel_fused_swi_glu", None)
            .map_err(|e| format!("Failed to get FusedSwiGLU kernel: {}", e))?;
        let fused_swi_glu_pipeline = device
            .new_compute_pipeline_state_with_function(&fused_swi_glu_kernel)
            .map_err(|e| {
                format!("Failed to create FusedSwiGLU pipeline: {}", e)
            })?;

        let silu_4_pipeline = silu_pipeline.clone();

        let gelu_kernel = activation_library
            .get_function("kernel_gelu_f32_4", None)
            .map_err(|e| format!("Failed to get GELU kernel: {}", e))?;
        let gelu_pipeline = device
            .new_compute_pipeline_state_with_function(&gelu_kernel)
            .map_err(|e| format!("Failed to create GELU pipeline: {}", e))?;

        let gelu_4_pipeline = gelu_pipeline.clone();

        let gelu_quick_kernel = activation_library
            .get_function("kernel_gelu_quick_f32_4", None)
            .map_err(|e| format!("Failed to get GELU Quick kernel: {}", e))?;
        let gelu_quick_pipeline = device
            .new_compute_pipeline_state_with_function(&gelu_quick_kernel)
            .map_err(|e| {
                format!("Failed to create GELU Quick pipeline: {}", e)
            })?;

        let gelu_quick_4_pipeline = gelu_quick_pipeline.clone();

        // Create F32 <-> F16 conversion pipelines
        let f32_to_f16_kernel = activation_library
            .get_function("kernel_f32_to_f16", None)
            .map_err(|e| format!("Failed to get kernel_f32_to_f16: {}", e))?;
        let f32_to_f16_pipeline = device
            .new_compute_pipeline_state_with_function(&f32_to_f16_kernel)
            .map_err(|e| {
                format!("Failed to create f32_to_f16 pipeline: {}", e)
            })?;

        let f16_to_f32_kernel = activation_library
            .get_function("kernel_f16_to_f32", None)
            .map_err(|e| format!("Failed to get kernel_f16_to_f32: {}", e))?;
        let f16_to_f32_pipeline = device
            .new_compute_pipeline_state_with_function(&f16_to_f32_kernel)
            .map_err(|e| {
                format!("Failed to create f16_to_f32 pipeline: {}", e)
            })?;

        let f32_to_f16_strided_kernel = activation_library
            .get_function("kernel_f32_to_f16_strided", None)
            .map_err(|e| {
                format!("Failed to get kernel_f32_to_f16_strided: {}", e)
            })?;
        let f32_to_f16_strided_pipeline = device
            .new_compute_pipeline_state_with_function(
                &f32_to_f16_strided_kernel,
            )
            .map_err(|e| {
                format!("Failed to create f32_to_f16_strided pipeline: {}", e)
            })?;

        // Create fused kernel pipelines
        let fused_swiglu_kernel = activation_library
            .get_function("kernel_fused_swiglu_f32", None)
            .map_err(|e| {
                format!("Failed to get kernel_fused_swiglu_f32 kernel: {}", e)
            })?;
        let fused_swiglu_pipeline = device
            .new_compute_pipeline_state_with_function(&fused_swiglu_kernel)
            .map_err(|e| {
                format!("Failed to create fused_swiglu pipeline: {}", e)
            })?;

        let fused_swiglu_4_kernel = activation_library
            .get_function("kernel_fused_swiglu_f32_4", None)
            .map_err(|e| {
                format!("Failed to get kernel_fused_swiglu_f32_4 kernel: {}", e)
            })?;
        let fused_swiglu_4_pipeline = device
            .new_compute_pipeline_state_with_function(&fused_swiglu_4_kernel)
            .map_err(|e| {
                format!("Failed to create fused_swiglu_4 pipeline: {}", e)
            })?;

        // Create FusedNormLinear pipelines
        let fused_norm_linear_f16_kernel = fused_norm_linear_library
            .get_function("kernel_fused_norm_linear_f16", None)
            .map_err(|e| {
                format!(
                    "Failed to get kernel_fused_norm_linear_f16 kernel: {}",
                    e
                )
            })?;
        let fused_norm_linear_f16_pipeline = device
            .new_compute_pipeline_state_with_function(
                &fused_norm_linear_f16_kernel,
            )
            .map_err(|e| {
                format!(
                    "Failed to create fused_norm_linear_f16 pipeline: {}",
                    e
                )
            })?;

        let fused_norm_linear_f32_kernel = fused_norm_linear_library
            .get_function("kernel_fused_norm_linear_f32", None)
            .map_err(|e| {
                format!(
                    "Failed to get kernel_fused_norm_linear_f32 kernel: {}",
                    e
                )
            })?;
        let fused_norm_linear_f32_pipeline = device
            .new_compute_pipeline_state_with_function(
                &fused_norm_linear_f32_kernel,
            )
            .map_err(|e| {
                format!(
                    "Failed to create fused_norm_linear_f32 pipeline: {}",
                    e
                )
            })?;

        let fused_norm_linear_f16_4_kernel = fused_norm_linear_library
            .get_function("kernel_fused_norm_linear_f16_4", None)
            .map_err(|e| {
                format!(
                    "Failed to get kernel_fused_norm_linear_f16_4 kernel: {}",
                    e
                )
            })?;
        let fused_norm_linear_f16_4_pipeline = device
            .new_compute_pipeline_state_with_function(
                &fused_norm_linear_f16_4_kernel,
            )
            .map_err(|e| {
                format!(
                    "Failed to create fused_norm_linear_f16_4 pipeline: {}",
                    e
                )
            })?;

        // Create FusedNormLinear Q4_K pipeline
        let fused_norm_linear_q4_k_kernel = fused_norm_linear_library
            .get_function("kernel_fused_norm_linear_q4_k", None)
            .map_err(|e| {
                format!(
                    "Failed to get kernel_fused_norm_linear_q4_k kernel: {}",
                    e
                )
            })?;
        let fused_norm_linear_q4_k_pipeline = device
            .new_compute_pipeline_state_with_function(
                &fused_norm_linear_q4_k_kernel,
            )
            .map_err(|e| {
                format!(
                    "Failed to create fused_norm_linear_q4_k pipeline: {}",
                    e
                )
            })?;

        // Create FusedNormLinear Q5_K pipeline
        let fused_norm_linear_q5_k_kernel = fused_norm_linear_library
            .get_function("kernel_fused_norm_linear_q5_k", None)
            .map_err(|e| {
                format!(
                    "Failed to get kernel_fused_norm_linear_q5_k kernel: {}",
                    e
                )
            })?;
        let fused_norm_linear_q5_k_pipeline = device
            .new_compute_pipeline_state_with_function(
                &fused_norm_linear_q5_k_kernel,
            )
            .map_err(|e| {
                format!(
                    "Failed to create fused_norm_linear_q5_k pipeline: {}",
                    e
                )
            })?;

        // Create FusedNormLinear Q6_K pipeline
        let fused_norm_linear_q6_k_kernel = fused_norm_linear_library
            .get_function("kernel_fused_norm_linear_q6_k", None)
            .map_err(|e| {
                format!(
                    "Failed to get kernel_fused_norm_linear_q6_k kernel: {}",
                    e
                )
            })?;
        let fused_norm_linear_q6_k_pipeline = device
            .new_compute_pipeline_state_with_function(
                &fused_norm_linear_q6_k_kernel,
            )
            .map_err(|e| {
                format!(
                    "Failed to create fused_norm_linear_q6_k pipeline: {}",
                    e
                )
            })?;

        // Create FusedAddNorm pipelines
        let fused_add_norm_f32_kernel = fused_add_norm_library
            .get_function("kernel_fused_add_norm_f32", None)
            .map_err(|e| {
                format!("Failed to get kernel_fused_add_norm_f32 kernel: {}", e)
            })?;
        let fused_add_norm_f32_pipeline = device
            .new_compute_pipeline_state_with_function(
                &fused_add_norm_f32_kernel,
            )
            .map_err(|e| {
                format!("Failed to create fused_add_norm_f32 pipeline: {}", e)
            })?;

        let fused_add_norm_f32_4_kernel = fused_add_norm_library
            .get_function("kernel_fused_add_norm_f32_4", None)
            .map_err(|e| {
                format!(
                    "Failed to get kernel_fused_add_norm_f32_4 kernel: {}",
                    e
                )
            })?;
        let fused_add_norm_f32_4_pipeline = device
            .new_compute_pipeline_state_with_function(
                &fused_add_norm_f32_4_kernel,
            )
            .map_err(|e| {
                format!("Failed to create fused_add_norm_f32_4 pipeline: {}", e)
            })?;

        let fused_add_norm_f16_kernel = fused_add_norm_library
            .get_function("kernel_fused_add_norm_f16", None)
            .map_err(|e| {
                format!("Failed to get kernel_fused_add_norm_f16 kernel: {}", e)
            })?;
        let fused_add_norm_f16_pipeline = device
            .new_compute_pipeline_state_with_function(
                &fused_add_norm_f16_kernel,
            )
            .map_err(|e| {
                format!("Failed to create fused_add_norm_f16 pipeline: {}", e)
            })?;

        let fused_add_norm_f16_4_kernel = fused_add_norm_library
            .get_function("kernel_fused_add_norm_f16_4", None)
            .map_err(|e| {
                format!(
                    "Failed to get kernel_fused_add_norm_f16_4 kernel: {}",
                    e
                )
            })?;
        let fused_add_norm_f16_4_pipeline = device
            .new_compute_pipeline_state_with_function(
                &fused_add_norm_f16_4_kernel,
            )
            .map_err(|e| {
                format!("Failed to create fused_add_norm_f16_4 pipeline: {}", e)
            })?;

        // Create FusedGateUp pipelines
        let fused_gate_up_f16_kernel = fused_gate_up_library
            .get_function("kernel_fused_gate_up_f16", None)
            .map_err(|e| {
                format!("Failed to get kernel_fused_gate_up_f16 kernel: {}", e)
            })?;
        let fused_gate_up_f16_pipeline = device
            .new_compute_pipeline_state_with_function(&fused_gate_up_f16_kernel)
            .map_err(|e| {
                format!("Failed to create fused_gate_up_f16 pipeline: {}", e)
            })?;

        let fused_gate_up_q4_k_kernel = fused_gate_up_library
            .get_function("kernel_fused_gate_up_q4_k", None)
            .map_err(|e| {
                format!("Failed to get kernel_fused_gate_up_q4_k kernel: {}", e)
            })?;
        let fused_gate_up_q4_k_pipeline = device
            .new_compute_pipeline_state_with_function(
                &fused_gate_up_q4_k_kernel,
            )
            .map_err(|e| {
                format!("Failed to create fused_gate_up_q4_k pipeline: {}", e)
            })?;

        let fused_gate_up_q5_k_kernel = fused_gate_up_library
            .get_function("kernel_fused_gate_up_q5_k", None)
            .map_err(|e| {
                format!("Failed to get kernel_fused_gate_up_q5_k kernel: {}", e)
            })?;
        let fused_gate_up_q5_k_pipeline = device
            .new_compute_pipeline_state_with_function(
                &fused_gate_up_q5_k_kernel,
            )
            .map_err(|e| {
                format!("Failed to create fused_gate_up_q5_k pipeline: {}", e)
            })?;

        let fused_gate_up_q6_k_kernel = fused_gate_up_library
            .get_function("kernel_fused_gate_up_q6_k", None)
            .map_err(|e| {
                format!("Failed to get kernel_fused_gate_up_q6_k kernel: {}", e)
            })?;
        let fused_gate_up_q6_k_pipeline = device
            .new_compute_pipeline_state_with_function(
                &fused_gate_up_q6_k_kernel,
            )
            .map_err(|e| {
                format!("Failed to create fused_gate_up_q6_k pipeline: {}", e)
            })?;

        // Create Flash Attention pipeline (template specialization, may fail on some Metal versions)
        // Create Flash Attention pipelines for each head_dim
        let flash_decode_f16_32_kernel = flash_attn_library
            .get_function("flash_decode_vec_f16_32", None)
            .map_err(|e| {
                format!("Failed to get flash_decode_vec_f16_32 kernel: {}", e)
            })?;
        let flash_decode_f16_32_pipeline = device
            .new_compute_pipeline_state_with_function(
                &flash_decode_f16_32_kernel,
            )
            .map_err(|e| {
                format!("Failed to create flash_decode_f16_32 pipeline: {}", e)
            })?;

        let flash_decode_f16_64_kernel = flash_attn_library
            .get_function("flash_decode_vec_f16_64", None)
            .map_err(|e| {
                format!("Failed to get flash_decode_vec_f16_64 kernel: {}", e)
            })?;
        let flash_decode_f16_64_pipeline = device
            .new_compute_pipeline_state_with_function(
                &flash_decode_f16_64_kernel,
            )
            .map_err(|e| {
                format!("Failed to create flash_decode_f16_64 pipeline: {}", e)
            })?;

        let flash_decode_f16_80_kernel = flash_attn_library
            .get_function("flash_decode_vec_f16_80", None)
            .map_err(|e| {
                format!("Failed to get flash_decode_vec_f16_80 kernel: {}", e)
            })?;
        let flash_decode_f16_80_pipeline = device
            .new_compute_pipeline_state_with_function(
                &flash_decode_f16_80_kernel,
            )
            .map_err(|e| {
                format!("Failed to create flash_decode_f16_80 pipeline: {}", e)
            })?;

        let flash_decode_f16_128_kernel = flash_attn_library
            .get_function("flash_decode_vec_f16", None) // Default name for HEAD_DIM=128
            .map_err(|e| {
                format!("Failed to get flash_decode_vec_f16 kernel: {}", e)
            })?;
        let flash_decode_f16_128_pipeline = device
            .new_compute_pipeline_state_with_function(
                &flash_decode_f16_128_kernel,
            )
            .map_err(|e| {
                format!("Failed to create flash_decode_vec_f16 pipeline: {}", e)
            })?;

        let flash_decode_f16_160_kernel = flash_attn_library
            .get_function("flash_decode_vec_f16_160", None)
            .map_err(|e| {
                format!("Failed to get flash_decode_vec_f16_160 kernel: {}", e)
            })?;
        let flash_decode_f16_160_pipeline = device
            .new_compute_pipeline_state_with_function(
                &flash_decode_f16_160_kernel,
            )
            .map_err(|e| {
                format!(
                    "Failed to create flash_decode_vec_f16_160 pipeline: {}",
                    e
                )
            })?;

        // Create Flash Attention pipeline for HEAD_DIM=213
        let flash_decode_f16_213_kernel = flash_attn_library
            .get_function("flash_decode_vec_f16_213", None)
            .map_err(|e| {
                format!("Failed to get flash_decode_vec_f16_213 kernel: {}", e)
            })?;
        let flash_decode_f16_213_pipeline = device
            .new_compute_pipeline_state_with_function(
                &flash_decode_f16_213_kernel,
            )
            .map_err(|e| {
                format!(
                    "Failed to create flash_decode_vec_f16_213 pipeline: {}",
                    e
                )
            })?;

        // Create Flash Attention MMA pipeline (experimental)
        let flash_decode_mma_kernel = flash_attn_mma_library
            .get_function("flash_decode_mma_f16", None)
            .map_err(|e| {
                format!("Failed to get flash_decode_mma_f16 kernel: {}", e)
            })?;
        let flash_decode_mma_pipeline = device
            .new_compute_pipeline_state_with_function(&flash_decode_mma_kernel)
            .map_err(|e| {
                format!("Failed to create flash_decode_mma pipeline: {}", e)
            })?;

        // Create Flash Attention MMA v2 pipeline (optimized)
        let flash_decode_mma_v2_kernel = flash_attn_mma_v2_library
            .get_function("flash_decode_mma_v2_f16", None)
            .map_err(|e| {
                format!("Failed to get flash_decode_mma_v2_f16 kernel: {}", e)
            })?;
        let flash_decode_mma_v2_pipeline = device
            .new_compute_pipeline_state_with_function(
                &flash_decode_mma_v2_kernel,
            )
            .map_err(|e| {
                format!("Failed to create flash_decode_mma_v2 pipeline: {}", e)
            })?;

        // Create Flash Attention MMA True pipeline (with simdgroup_float8x8)
        // TEMPORARILY DISABLED: kernel has compilation errors
        // TODO: Fix simdgroup_fill and simdgroup_add usage in flash_decode_mma_true.metal
        /*
        let flash_decode_mma_true_source = include_str!(
            "../../../ops/metal/attention/flash_decode_mma_true.metal"
        );
        let flash_decode_mma_true_library = device
            .new_library_with_source(
                flash_decode_mma_true_source,
                &compile_options,
            )
            .map_err(|e| {
                format!(
                    "Failed to compile flash_decode_mma_true library: {}",
                    e
                )
            })?;
        let flash_decode_mma_true_kernel = flash_decode_mma_true_library
            .get_function("flash_decode_mma_true_f16", None)
            .map_err(|e| {
                format!("Failed to get flash_decode_mma_true_f16 kernel: {}", e)
            })?;
        let flash_decode_mma_true_pipeline = device
            .new_compute_pipeline_state_with_function(
                &flash_decode_mma_true_kernel,
            )
            .map_err(|e| {
                format!(
                    "Failed to create flash_decode_mma_true pipeline: {}",
                    e
                )
            })?;
        */

        // Create Flash Decode Batched pipeline (C=8 positions per iteration)
        let flash_decode_batched_source = include_str!(
            "../../../ops/metal/attention/flash_decode_batched.metal"
        );
        let flash_decode_batched_library = device
            .new_library_with_source(
                flash_decode_batched_source,
                &compile_options,
            )
            .map_err(|e| {
                format!("Failed to compile flash_decode_batched library: {}", e)
            })?;
        let flash_decode_batched_kernel = flash_decode_batched_library
            .get_function("flash_decode_batched_f16", None)
            .map_err(|e| {
                format!("Failed to get flash_decode_batched_f16 kernel: {}", e)
            })?;
        let flash_decode_mma_batched_pipeline = device
            .new_compute_pipeline_state_with_function(
                &flash_decode_batched_kernel,
            )
            .map_err(|e| {
                format!(
                    "Failed to create flash_decode_mma_batched pipeline: {}",
                    e
                )
            })?;

        // Create NWG parallelized Flash Attention pipelines
        let flash_decode_nwg_source =
            include_str!("../../../ops/metal/attention/flash_decode_nwg.metal");
        let flash_decode_nwg_library = device
            .new_library_with_source(flash_decode_nwg_source, &compile_options)
            .map_err(|e| {
                format!("Failed to compile flash_decode_nwg library: {}", e)
            })?;
        let flash_decode_nwg_pass1_kernel = flash_decode_nwg_library
            .get_function("flash_decode_nwg_pass1_f16", None)
            .map_err(|e| {
                format!(
                    "Failed to get flash_decode_nwg_pass1_f16 kernel: {}",
                    e
                )
            })?;
        let flash_decode_nwg_pass1_pipeline = device
            .new_compute_pipeline_state_with_function(
                &flash_decode_nwg_pass1_kernel,
            )
            .map_err(|e| {
                format!(
                    "Failed to create flash_decode_nwg_pass1 pipeline: {}",
                    e
                )
            })?;
        let flash_decode_nwg_pass2_kernel = flash_decode_nwg_library
            .get_function("flash_decode_nwg_pass2_f16", None)
            .map_err(|e| {
                format!(
                    "Failed to get flash_decode_nwg_pass2_f16 kernel: {}",
                    e
                )
            })?;
        let flash_decode_nwg_pass2_pipeline = device
            .new_compute_pipeline_state_with_function(
                &flash_decode_nwg_pass2_kernel,
            )
            .map_err(|e| {
                format!(
                    "Failed to create flash_decode_nwg_pass2 pipeline: {}",
                    e
                )
            })?;

        // Create Multi-Head Flash Attention pipeline (llama.cpp style)
        // Q=32, NSG=8 for 32x dispatch overhead reduction
        let flash_attn_multihead_source = include_str!(
            "../../../ops/metal/attention/flash_attn_multihead.metal"
        );
        let flash_attn_multihead_library = device
            .new_library_with_source(
                flash_attn_multihead_source,
                &compile_options,
            )
            .map_err(|e| {
                format!("Failed to compile flash_attn_multihead library: {}", e)
            })?;
        let flash_attn_multihead_128_kernel = flash_attn_multihead_library
            .get_function("kernel_flash_attn_multihead_d128", None)
            .map_err(|e| {
                format!(
                    "Failed to get kernel_flash_attn_multihead_d128 kernel: {}",
                    e
                )
            })?;
        let flash_attn_multihead_128_pipeline = device
            .new_compute_pipeline_state_with_function(
                &flash_attn_multihead_128_kernel,
            )
            .map_err(|e| {
                format!(
                    "Failed to create flash_attn_multihead_128 pipeline: {}",
                    e
                )
            })?;

        // Create Flash Prefill pipeline
        let flash_prefill_f16_kernel = flash_prefill_library
            .get_function("flash_prefill_attention_f16", None)
            .map_err(|e| {
                format!(
                    "Failed to get flash_prefill_attention_f16 kernel: {}",
                    e
                )
            })?;
        let flash_prefill_f16_pipeline = device
            .new_compute_pipeline_state_with_function(&flash_prefill_f16_kernel)
            .map_err(|e| {
                format!("Failed to create flash_prefill_f16 pipeline: {}", e)
            })?;

        // Create Batched Flash Prefill pipeline (llama.cpp style architecture)
        let flash_prefill_batch_source = include_str!(
            "../../../ops/metal/attention/flash_prefill_batch.metal"
        );
        let flash_prefill_batch_library = device
            .new_library_with_source(
                flash_prefill_batch_source,
                &compile_options,
            )
            .map_err(|e| {
                format!("Failed to compile flash_prefill_batch library: {}", e)
            })?;
        let flash_prefill_batch_kernel = flash_prefill_batch_library
            .get_function("kernel_flash_prefill_batch", None)
            .map_err(|e| {
                format!(
                    "Failed to get kernel_flash_prefill_batch kernel: {}",
                    e
                )
            })?;
        let flash_prefill_batch_pipeline = device
            .new_compute_pipeline_state_with_function(
                &flash_prefill_batch_kernel,
            )
            .map_err(|e| {
                format!("Failed to create flash_prefill_batch pipeline: {}", e)
            })?;

        // Create Quantize pipelines
        let quantize_f16_kernel = quantize_library
            .get_function("kernel_quantize_f16", None)
            .map_err(|e| format!("Failed to get Quantize F16 kernel: {}", e))?;
        let quantize_f16_pipeline = device
            .new_compute_pipeline_state_with_function(&quantize_f16_kernel)
            .map_err(|e| {
                format!("Failed to create Quantize F16 pipeline: {}", e)
            })?;

        let quantize_f16_4_kernel = quantize_library
            .get_function("kernel_quantize_f16_4", None)
            .map_err(|e| {
                format!("Failed to get Quantize F16_4 kernel: {}", e)
            })?;
        let quantize_f16_4_pipeline = device
            .new_compute_pipeline_state_with_function(&quantize_f16_4_kernel)
            .map_err(|e| {
                format!("Failed to create Quantize F16_4 pipeline: {}", e)
            })?;

        let quantize_q4_0_kernel = quantize_library
            .get_function("kernel_quantize_q4_0", None)
            .map_err(|e| {
                format!("Failed to get Quantize Q4_0 kernel: {}", e)
            })?;
        let quantize_q4_0_pipeline = device
            .new_compute_pipeline_state_with_function(&quantize_q4_0_kernel)
            .map_err(|e| {
                format!("Failed to create Quantize Q4_0 pipeline: {}", e)
            })?;

        let quantize_q8_0_kernel = quantize_library
            .get_function("kernel_quantize_q8_0", None)
            .map_err(|e| {
                format!("Failed to get Quantize Q8_0 kernel: {}", e)
            })?;
        let quantize_q8_0_pipeline = device
            .new_compute_pipeline_state_with_function(&quantize_q8_0_kernel)
            .map_err(|e| {
                format!("Failed to create Quantize Q8_0 pipeline: {}", e)
            })?;

        // Compile Split QKV library
        let split_qkv_source =
            include_str!("../../../ops/metal/ops/split_qkv.metal");
        let split_qkv_library = device
            .new_library_with_source(split_qkv_source, &compile_options)
            .map_err(|e| {
                format!("Failed to compile split_qkv library: {}", e)
            })?;

        // Create Split QKV pipeline
        let split_qkv_kernel = split_qkv_library
            .get_function("kernel_split_qkv", None)
            .map_err(|e| format!("Failed to get split_qkv kernel: {}", e))?;
        let split_qkv_pipeline = device
            .new_compute_pipeline_state_with_function(&split_qkv_kernel)
            .map_err(|e| {
                format!("Failed to create split_qkv pipeline: {}", e)
            })?;

        // Compile SSM library (State Space Model kernels)
        let ssm_conv_source =
            include_str!("../../../ops/metal/ops/ssm_conv.metal");
        let ssm_scan_source =
            include_str!("../../../ops/metal/ops/ssm_scan.metal");
        let ssd_optimized_source =
            include_str!("../../../ops/metal/ops/ssd_optimized.metal");
        let ssm_combined = format!(
            "{}\n{}\n{}",
            ssm_conv_source, ssm_scan_source, ssd_optimized_source
        );
        let ssm_library = device
            .new_library_with_source(&ssm_combined, &compile_options)
            .map_err(|e| format!("Failed to compile SSM library: {}", e))?;

        // Create SSM Conv1D pipeline (d_conv=4 for Qwen3.8)
        let ssm_conv1d_d4_kernel =
            ssm_library.get_function("ssm_conv1d_d4_f32", None).map_err(
                |e| format!("Failed to get SSM Conv1D D4 kernel: {}", e),
            )?;
        let ssm_conv1d_d4_pipeline = device
            .new_compute_pipeline_state_with_function(&ssm_conv1d_d4_kernel)
            .map_err(|e| {
                format!("Failed to create SSM Conv1D D4 pipeline: {}", e)
            })?;

        // Create Conv1D batched pipelines for different batch sizes
        let ssm_conv1d_d4_batched_256_kernel = ssm_library
            .get_function("ssm_conv1d_d4_batched_256", None)
            .map_err(|e| {
                format!("Failed to get Conv1D batched 256 kernel: {}", e)
            })?;
        let ssm_conv1d_d4_batched_256_pipeline = device
            .new_compute_pipeline_state_with_function(
                &ssm_conv1d_d4_batched_256_kernel,
            )
            .map_err(|e| {
                format!("Failed to create Conv1D batched 256 pipeline: {}", e)
            })?;

        let ssm_conv1d_d4_batched_128_kernel = ssm_library
            .get_function("ssm_conv1d_d4_batched_128", None)
            .map_err(|e| {
                format!("Failed to get Conv1D batched 128 kernel: {}", e)
            })?;
        let ssm_conv1d_d4_batched_128_pipeline = device
            .new_compute_pipeline_state_with_function(
                &ssm_conv1d_d4_batched_128_kernel,
            )
            .map_err(|e| {
                format!("Failed to create Conv1D batched 128 pipeline: {}", e)
            })?;

        let ssm_conv1d_d4_batched_64_kernel = ssm_library
            .get_function("ssm_conv1d_d4_batched_64", None)
            .map_err(|e| {
                format!("Failed to get Conv1D batched 64 kernel: {}", e)
            })?;
        let ssm_conv1d_d4_batched_64_pipeline = device
            .new_compute_pipeline_state_with_function(
                &ssm_conv1d_d4_batched_64_kernel,
            )
            .map_err(|e| {
                format!("Failed to create Conv1D batched 64 pipeline: {}", e)
            })?;

        let ssm_conv1d_d4_batched_32_kernel = ssm_library
            .get_function("ssm_conv1d_d4_batched_32", None)
            .map_err(|e| {
                format!("Failed to get Conv1D batched 32 kernel: {}", e)
            })?;
        let ssm_conv1d_d4_batched_32_pipeline = device
            .new_compute_pipeline_state_with_function(
                &ssm_conv1d_d4_batched_32_kernel,
            )
            .map_err(|e| {
                format!("Failed to create Conv1D batched 32 pipeline: {}", e)
            })?;

        let ssm_conv1d_d4_batched_16_kernel = ssm_library
            .get_function("ssm_conv1d_d4_batched_16", None)
            .map_err(|e| {
                format!("Failed to get Conv1D batched 16 kernel: {}", e)
            })?;
        let ssm_conv1d_d4_batched_16_pipeline = device
            .new_compute_pipeline_state_with_function(
                &ssm_conv1d_d4_batched_16_kernel,
            )
            .map_err(|e| {
                format!("Failed to create Conv1D batched 16 pipeline: {}", e)
            })?;

        let ssm_conv1d_d4_batched_8_kernel =
            ssm_library.get_function("ssm_conv1d_d4_batched_8", None).map_err(
                |e| format!("Failed to get Conv1D batched 8 kernel: {}", e),
            )?;
        let ssm_conv1d_d4_batched_8_pipeline = device
            .new_compute_pipeline_state_with_function(
                &ssm_conv1d_d4_batched_8_kernel,
            )
            .map_err(|e| {
                format!("Failed to create Conv1D batched 8 pipeline: {}", e)
            })?;

        let ssm_conv1d_d4_batched_2_kernel =
            ssm_library.get_function("ssm_conv1d_d4_batched_2", None).map_err(
                |e| format!("Failed to get Conv1D batched 2 kernel: {}", e),
            )?;
        let ssm_conv1d_d4_batched_2_pipeline = device
            .new_compute_pipeline_state_with_function(
                &ssm_conv1d_d4_batched_2_kernel,
            )
            .map_err(|e| {
                format!("Failed to create Conv1D batched 2 pipeline: {}", e)
            })?;

        // Create SSM Scan pipelines
        let ssm_scan_d16_kernel = ssm_library
            .get_function("ssm_scan_d16_f32", None)
            .map_err(|e| format!("Failed to get SSM Scan D16 kernel: {}", e))?;
        let ssm_scan_d16_pipeline = device
            .new_compute_pipeline_state_with_function(&ssm_scan_d16_kernel)
            .map_err(|e| {
                format!("Failed to create SSM Scan D16 pipeline: {}", e)
            })?;

        // Create SSM SSD pipeline (State Space Duality for long sequences)
        let ssm_scan_ssd_kernel = ssm_library
            .get_function("ssm_scan_ssd_f32", None)
            .map_err(|e| format!("Failed to get SSM Scan SSD kernel: {}", e))?;
        let ssm_scan_ssd_pipeline = device
            .new_compute_pipeline_state_with_function(&ssm_scan_ssd_kernel)
            .map_err(|e| {
                format!("Failed to create SSM Scan SSD pipeline: {}", e)
            })?;

        // Create SSD MMA pipeline (optimized with simdgroup matrix operations)
        let ssm_scan_ssd_mma_kernel = ssm_library
            .get_function("kernel_ssm_scan_ssd_mma", None)
            .map_err(|e| format!("Failed to get SSD MMA kernel: {}", e))?;
        let ssm_scan_ssd_mma_pipeline = device
            .new_compute_pipeline_state_with_function(&ssm_scan_ssd_mma_kernel)
            .map_err(|e| format!("Failed to create SSD MMA pipeline: {}", e))?;

        // Compile KV update library (F32 -> F16 with strided write)
        let kv_update_source =
            include_str!("../../../ops/metal/ops/kv_update.metal");
        let kv_update_library = device
            .new_library_with_source(kv_update_source, &compile_options)
            .map_err(|e| {
                format!("Failed to compile kv_update library: {}", e)
            })?;

        // Create KV update K pipeline
        let kv_update_k_kernel = kv_update_library
            .get_function("kv_update_f32_to_f16", None)
            .map_err(|e| format!("Failed to get KV update K kernel: {}", e))?;
        let kv_update_k_pipeline = device
            .new_compute_pipeline_state_with_function(&kv_update_k_kernel)
            .map_err(|e| {
                format!("Failed to create KV update K pipeline: {}", e)
            })?;

        // Create KV update V pipeline
        let kv_update_v_kernel = kv_update_library
            .get_function("kv_update_v_f32_to_f16", None)
            .map_err(|e| format!("Failed to get KV update V kernel: {}", e))?;
        let kv_update_v_pipeline = device
            .new_compute_pipeline_state_with_function(&kv_update_v_kernel)
            .map_err(|e| {
                format!("Failed to create KV update V pipeline: {}", e)
            })?;

        // Create KV update K F16 pipeline (F16->F16, no conversion)
        let kv_update_k_f16_kernel = kv_update_library
            .get_function("kv_update_f16_to_f16", None)
            .map_err(|e| {
                format!("Failed to get KV update K F16 kernel: {}", e)
            })?;
        let kv_update_k_f16_pipeline = device
            .new_compute_pipeline_state_with_function(&kv_update_k_f16_kernel)
            .map_err(|e| {
                format!("Failed to create KV update K F16 pipeline: {}", e)
            })?;

        // Create batched KV update pipelines for prefill
        let kv_update_batched_f16_kernel = kv_update_library
            .get_function("kv_update_batched_f16", None)
            .map_err(|e| {
                format!("Failed to get KV update batched F16 kernel: {}", e)
            })?;
        let kv_update_batched_f16_pipeline = device
            .new_compute_pipeline_state_with_function(
                &kv_update_batched_f16_kernel,
            )
            .map_err(|e| {
                format!(
                    "Failed to create KV update batched F16 pipeline: {}",
                    e
                )
            })?;

        let kv_update_batched_f16_strided_kernel = kv_update_library
            .get_function("kv_update_batched_f16_strided", None)
            .map_err(|e| {
                format!(
                    "Failed to get KV update batched F16 strided kernel: {}",
                    e
                )
            })?;
        let kv_update_batched_f16_strided_pipeline = device
            .new_compute_pipeline_state_with_function(
                &kv_update_batched_f16_strided_kernel,
            )
            .map_err(|e| {
                format!(
                    "Failed to create KV update batched F16 strided pipeline: {}",
                    e
                )
            })?;

        Ok(Self {
            device,
            queue,
            vec_dot_library,
            mv_library,
            mv_simd_library,
            mv_iq_tq_library,
            // Self-written GEMM template library
            gemm_library,
            // Self-written K-quant MV library
            kquant_mv_library,
            // Transposed GEMV library and pipeline
            gemv_transposed_library,
            mv_q6_k_transposed_pipeline,
            // F16 GEMV library and pipelines
            mv_f16_library,
            mv_f16_short_pipeline,
            mv_f16_basic_pipeline,
            mv_f16_f16_pipeline,
            mv_f16_4_nsg1_pipeline,
            mv_f16_4_nsg2_pipeline,
            mv_f16_4_nsg4_pipeline,
            // Optimized F16 pipeline with nxpsg
            mv_f16_opt_r1_1_nxpsg16_pipeline,
            // MV pipelines - Standard Q
            mv_q1_0_pipeline,
            mv_q2_0_pipeline,
            mv_q4_0_pipeline,
            mv_q4_1_pipeline,
            mv_q5_0_pipeline,
            mv_q5_1_pipeline,
            mv_q8_0_pipeline,
            // MV pipelines - K-quant
            mv_q4_k_pipeline,
            mv_q5_k_pipeline,
            mv_q2_k_pipeline,
            mv_q6_k_pipeline,
            mv_q3_k_pipeline,
            // Fused GEMV+Add pipelines for residual connections
            fused_gemv_add_q4_k_pipeline,
            fused_gemv_add_q5_k_pipeline,
            fused_gemv_add_q6_k_pipeline,
            mv_iq4_nl_pipeline,
            mv_tq2_0_pipeline,
            mv_tq1_0_pipeline,
            mv_iq4_xs_pipeline,
            mv_iq1_s_pipeline,
            mv_iq1_m_pipeline,
            mv_iq2_xxs_pipeline,
            mv_iq2_xs_pipeline,
            mv_iq2_s_pipeline,
            mv_iq3_xxs_pipeline,
            mv_iq3_s_pipeline,
            mv_iq2_xxs_split1_pipeline,
            mv_iq2_xs_split1_pipeline,
            mv_iq2_s_split1_pipeline,
            mv_iq3_xxs_split1_pipeline,
            mv_iq3_s_split1_pipeline,
            // MXFP
            mv_mxfp4_pipeline,
            // Templated GEMM pipelines
            gemm_template_q4_k_pipeline,
            gemm_template_q2_k_pipeline,
            gemm_template_q3_k_pipeline,
            gemm_template_q5_k_pipeline,
            gemm_template_q6_k_pipeline,
            // RMS Norm, Softmax, ROPE, Activation
            rms_norm_library,
            rms_norm_pipeline,
            rms_norm_4_pipeline,
            softmax_library,
            softmax_fast_pipeline,
            softmax_4_fast_pipeline,
            softmax_pipeline,
            softmax_4_pipeline,
            rope_library,
            rope_pipeline,
            rope_f16_pipeline,
            activation_library,
            add_pipeline,
            mul_pipeline,
            silu_pipeline,
            fused_swi_glu_pipeline,
            silu_4_pipeline,
            gelu_pipeline,
            gelu_4_pipeline,
            gelu_quick_pipeline,
            gelu_quick_4_pipeline,
            // F32 <-> F16 conversion
            f32_to_f16_pipeline,
            f16_to_f32_pipeline,
            f32_to_f16_strided_pipeline,
            // Fused kernels
            fused_swiglu_pipeline,
            fused_swiglu_4_pipeline,
            fused_norm_linear_f16_pipeline,
            fused_norm_linear_f32_pipeline,
            fused_norm_linear_f16_4_pipeline,
            fused_norm_linear_q4_k_pipeline,
            fused_norm_linear_q5_k_pipeline,
            fused_norm_linear_q6_k_pipeline,
            // Split QKV
            split_qkv_library,
            split_qkv_pipeline,
            fused_add_norm_f32_pipeline,
            fused_add_norm_f32_4_pipeline,
            fused_add_norm_f16_pipeline,
            fused_add_norm_f16_4_pipeline,
            // FusedGateUp
            fused_gate_up_f16_pipeline,
            fused_gate_up_q4_k_pipeline,
            fused_gate_up_q5_k_pipeline,
            fused_gate_up_q6_k_pipeline,
            // Flash Attention
            flash_attn_library,
            flash_decode_f16_32_pipeline,
            flash_decode_f16_64_pipeline,
            flash_decode_f16_80_pipeline,
            flash_decode_f16_128_pipeline,
            flash_decode_f16_160_pipeline,
            flash_decode_f16_213_pipeline,
            flash_decode_mma_pipeline,
            flash_decode_mma_v2_pipeline,
            // flash_decode_mma_true_pipeline, // DISABLED
            flash_decode_mma_batched_pipeline,
            // NWG parallelized Flash Attention
            flash_decode_nwg_pass1_pipeline,
            flash_decode_nwg_pass2_pipeline,
            flash_attn_multihead_128_pipeline,
            // Flash Prefill
            flash_prefill_f16_pipeline,
            flash_prefill_batch_pipeline,
            // Quantize
            quantize_library,
            quantize_f16_pipeline,
            quantize_f16_4_pipeline,
            quantize_q4_0_pipeline,
            quantize_q8_0_pipeline,
            // SSM
            ssm_library,
            ssm_conv1d_d4_pipeline,
            ssm_conv1d_d4_batched_256_pipeline,
            ssm_conv1d_d4_batched_128_pipeline,
            ssm_conv1d_d4_batched_64_pipeline,
            ssm_conv1d_d4_batched_32_pipeline,
            ssm_conv1d_d4_batched_16_pipeline,
            ssm_conv1d_d4_batched_8_pipeline,
            ssm_conv1d_d4_batched_2_pipeline,
            ssm_scan_d16_pipeline,
            ssm_scan_ssd_pipeline,
            ssm_scan_ssd_mma_pipeline,
            // KV Cache Update
            kv_update_library,
            kv_update_k_pipeline,
            kv_update_k_f16_pipeline,
            kv_update_v_pipeline,
            // Batched KV update
            kv_update_batched_f16_pipeline,
            kv_update_batched_f16_strided_pipeline,
            // MV buffers
            mv_weights_buffer: RefCell::new(None),
            mv_input_buffer: RefCell::new(None),
            mv_output_buffer: RefCell::new(None),
            mv_weights_size: RefCell::new(0),
            mv_weights_ptr: RefCell::new(0),
            mv_input_size: RefCell::new(0),
            mv_output_size: RefCell::new(0),
            // GEMM buffers
            gemm_weights_buffer: RefCell::new(None),
            gemm_input_buffer: RefCell::new(None),
            gemm_output_buffer: RefCell::new(None),
            gemm_weights_size: RefCell::new(0),
            gemm_input_size: RefCell::new(0),
            gemm_output_size: RefCell::new(0),
            rope_pos_buffer: RefCell::new(None),
            embed_buffer: RefCell::new(None),
            embed_size: RefCell::new(0),
            weight_buffer_cache: RefCell::new(std::collections::HashMap::new()),
        })
    }

    /// Get the Metal device
    pub fn device(&self) -> &Device {
        &self.device
    }

    pub fn add_pipeline(&self) -> &ComputePipelineState {
        &self.add_pipeline
    }

    pub fn queue(&self) -> &CommandQueue {
        &self.queue
    }

    /// Clear all cached buffers to prevent stale references
    /// Call this before reusing MetalContext for a new forward pass
    pub fn clear_caches(&self) {
        // Clear MV buffers
        *self.mv_weights_buffer.borrow_mut() = None;
        *self.mv_input_buffer.borrow_mut() = None;
        *self.mv_output_buffer.borrow_mut() = None;
        *self.mv_weights_size.borrow_mut() = 0;
        *self.mv_weights_ptr.borrow_mut() = 0;
        *self.mv_input_size.borrow_mut() = 0;
        *self.mv_output_size.borrow_mut() = 0;

        // Clear weight buffer cache
        self.weight_buffer_cache.borrow_mut().clear();

        // Clear GEMM buffers
        *self.gemm_weights_buffer.borrow_mut() = None;
        *self.gemm_input_buffer.borrow_mut() = None;
        *self.gemm_output_buffer.borrow_mut() = None;
        *self.gemm_weights_size.borrow_mut() = 0;
        *self.gemm_input_size.borrow_mut() = 0;
        *self.gemm_output_size.borrow_mut() = 0;

        // Clear embed buffer
        *self.embed_buffer.borrow_mut() = None;
        *self.embed_size.borrow_mut() = 0;
    }

    /// Get or create cached embedding buffer
    /// Reuses buffer if size matches, creates new one if needed
    pub fn get_or_create_embed_buffer(&self, size: usize) -> Buffer {
        let mut embed_buffer = self.embed_buffer.borrow_mut();
        let mut embed_size = self.embed_size.borrow_mut();

        if *embed_size != size {
            // Size changed, create new buffer
            *embed_buffer = Some(self.device.new_buffer(
                size as u64,
                metal::MTLResourceOptions::StorageModeShared,
            ));
            *embed_size = size;
        }

        embed_buffer.as_ref().unwrap().clone()
    }

    /// Insert a GPU fence to ensure all previous commands complete
    /// This can help with thread-local Metal state issues
    pub fn insert_fence(&self) {
        let command_buffer = self.queue.new_command_buffer();
        command_buffer.commit();
        command_buffer.wait_until_completed();
    }

    /// Get Conv1D batched pipeline for specific batch size
    pub fn ssm_conv1d_d4_batched_pipeline(
        &self,
        batch_size: usize,
    ) -> &ComputePipelineState {
        match batch_size {
            256 => &self.ssm_conv1d_d4_batched_256_pipeline,
            128 => &self.ssm_conv1d_d4_batched_128_pipeline,
            64 => &self.ssm_conv1d_d4_batched_64_pipeline,
            32 => &self.ssm_conv1d_d4_batched_32_pipeline,
            16 => &self.ssm_conv1d_d4_batched_16_pipeline,
            8 => &self.ssm_conv1d_d4_batched_8_pipeline,
            _ => &self.ssm_conv1d_d4_batched_2_pipeline,
        }
    }

    /// Get SSD MMA pipeline
    pub fn ssm_scan_ssd_mma_pipeline(&self) -> &ComputePipelineState {
        &self.ssm_scan_ssd_mma_pipeline
    }

    /// Get or create GPU buffer for a tensor (lazy initialization)
    pub fn get_or_create_buffer(
        &self,
        tensor: &crate::model::tensor::Tensor,
    ) -> std::rc::Rc<Buffer> {
        let mut gpu_buffer = tensor.gpu_buffer.borrow_mut();

        if let Some(ref buffer) = *gpu_buffer {
            std::rc::Rc::clone(buffer)
        } else {
            // Create new buffer with zero-copy mapping
            // Tensor data is owned by the model and persists for the model's lifetime
            let buffer =
                std::rc::Rc::new(self.device.new_buffer_with_bytes_no_copy(
                    tensor.bytes().as_ptr() as *const std::ffi::c_void,
                    tensor.bytes().len() as u64,
                    metal::MTLResourceOptions::StorageModeShared,
                    None, // No deallocator - tensor owns the memory
                ));
            *gpu_buffer = Some(std::rc::Rc::clone(&buffer));
            buffer
        }
    }

    pub fn vec_dot_iq4_nl_q8_0(
        &self,
        n: usize,
        x: &[BlockIQ4NL],
        y: &[BlockQ8_0],
    ) -> Result<f32, String> {
        let nb = n / 32;

        let x_buffer = self.device.new_buffer_with_data(
            x.as_ptr() as *const std::ffi::c_void,
            (x.len() * std::mem::size_of::<BlockIQ4NL>()) as u64,
            metal::MTLResourceOptions::StorageModeShared,
        );

        let y_buffer = self.device.new_buffer_with_data(
            y.as_ptr() as *const std::ffi::c_void,
            (y.len() * std::mem::size_of::<BlockQ8_0>()) as u64,
            metal::MTLResourceOptions::StorageModeShared,
        );

        let result_buffer = self.device.new_buffer(
            std::mem::size_of::<f32>() as u64,
            metal::MTLResourceOptions::StorageModeShared,
        );

        let kernel = self
            .vec_dot_library
            .get_function("vec_dot_iq4_nl_q8_0", None)
            .map_err(|e| format!("Failed to get kernel: {}", e))?;
        let pipeline = self
            .device
            .new_compute_pipeline_state_with_function(&kernel)
            .map_err(|e| format!("Failed to create pipeline: {}", e))?;

        let command_buffer = self.queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();

        encoder.set_compute_pipeline_state(&pipeline);
        encoder.set_buffer(0, Some(&x_buffer), 0);
        encoder.set_buffer(1, Some(&y_buffer), 0);
        encoder.set_buffer(2, Some(&result_buffer), 0);
        encoder.set_bytes(
            3,
            std::mem::size_of::<u32>() as u64,
            &(nb as u32) as *const u32 as *const std::ffi::c_void,
        );

        let thread_group_count = MTLSize { width: 1, height: 1, depth: 1 };
        let thread_group_size = MTLSize { width: 1, height: 1, depth: 1 };

        encoder.dispatch_thread_groups(thread_group_count, thread_group_size);
        encoder.end_encoding();

        command_buffer.commit();
        command_buffer.wait_until_completed();

        let result_ptr = result_buffer.contents() as *const f32;
        Ok(unsafe { *result_ptr })
    }

    pub fn mv_iq4_nl_q8_0(
        &self,
        m: usize,
        k: usize,
        weights: &[BlockIQ4NL],
        input: &[BlockQ8_0],
    ) -> Result<Vec<f32>, String> {
        let nb = k / 32;

        // Time buffer creation
        let t0 = std::time::Instant::now();

        let weights_buffer = self.device.new_buffer_with_data(
            weights.as_ptr() as *const std::ffi::c_void,
            (weights.len() * std::mem::size_of::<BlockIQ4NL>()) as u64,
            metal::MTLResourceOptions::StorageModeShared,
        );

        let input_buffer = self.device.new_buffer_with_data(
            input.as_ptr() as *const std::ffi::c_void,
            (input.len() * std::mem::size_of::<BlockQ8_0>()) as u64,
            metal::MTLResourceOptions::StorageModeShared,
        );

        let output_buffer = self.device.new_buffer(
            (m * std::mem::size_of::<f32>()) as u64,
            metal::MTLResourceOptions::StorageModeShared,
        );

        let buffer_time = t0.elapsed();

        let t1 = std::time::Instant::now();

        let kernel = self
            .mv_library
            .get_function("mv_iq4_nl_q8_0", None)
            .map_err(|e| format!("Failed to get kernel: {}", e))?;
        let pipeline = self
            .device
            .new_compute_pipeline_state_with_function(&kernel)
            .map_err(|e| format!("Failed to create pipeline: {}", e))?;

        let pipeline_time = t1.elapsed();

        let t2 = std::time::Instant::now();

        let command_buffer = self.queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();

        encoder.set_compute_pipeline_state(&pipeline);
        encoder.set_buffer(0, Some(&weights_buffer), 0);
        encoder.set_buffer(1, Some(&input_buffer), 0);
        encoder.set_buffer(2, Some(&output_buffer), 0);
        encoder.set_bytes(
            3,
            std::mem::size_of::<u32>() as u64,
            &(m as u32) as *const u32 as *const std::ffi::c_void,
        );
        encoder.set_bytes(
            4,
            std::mem::size_of::<u32>() as u64,
            &(nb as u32) as *const u32 as *const std::ffi::c_void,
        );

        // NSG = 2 SIMD groups per threadgroup, each SIMD group has 32 threads
        // Each SIMD group processes NR0 = 4 rows
        // Total: 2 * 32 = 64 threads, processing 2 * 4 = 8 rows per threadgroup
        const NSG: u64 = 2;
        const NR0: u64 = 4;
        const ROWS_PER_THREADGROUP: u64 = NSG * NR0; // 8
        const THREADS_PER_THREADGROUP: u64 = NSG * 32; // 64

        let thread_group_size =
            MTLSize { width: THREADS_PER_THREADGROUP, height: 1, depth: 1 };
        let thread_group_count = MTLSize {
            width: (m as u64 + ROWS_PER_THREADGROUP - 1) / ROWS_PER_THREADGROUP,
            height: 1,
            depth: 1,
        };

        encoder.dispatch_thread_groups(thread_group_count, thread_group_size);
        encoder.end_encoding();

        let encode_time = t2.elapsed();

        let t3 = std::time::Instant::now();

        command_buffer.commit();
        command_buffer.wait_until_completed();

        let gpu_time = t3.elapsed();

        // Only log timing in debug builds
        #[cfg(debug_assertions)]
        eprintln!(
            "MV timing: buffer={:?} pipeline={:?} encode={:?} gpu={:?}",
            buffer_time, pipeline_time, encode_time, gpu_time
        );

        let output_ptr = output_buffer.contents() as *const f32;
        Ok(unsafe { std::slice::from_raw_parts(output_ptr, m).to_vec() })
    }

    /// SIMD-optimized MV kernel (32 threads per row)
    pub fn mv_iq4_nl_q8_0_simd(
        &self,
        m: usize,
        k: usize,
        weights: &[BlockIQ4NL],
        input: &[BlockQ8_0],
    ) -> Result<Vec<f32>, String> {
        let nb = k / 32;

        let weights_buffer = self.device.new_buffer_with_data(
            weights.as_ptr() as *const std::ffi::c_void,
            (weights.len() * std::mem::size_of::<BlockIQ4NL>()) as u64,
            metal::MTLResourceOptions::StorageModeShared,
        );

        let input_buffer = self.device.new_buffer_with_data(
            input.as_ptr() as *const std::ffi::c_void,
            (input.len() * std::mem::size_of::<BlockQ8_0>()) as u64,
            metal::MTLResourceOptions::StorageModeShared,
        );

        let output_buffer = self.device.new_buffer(
            (m * std::mem::size_of::<f32>()) as u64,
            metal::MTLResourceOptions::StorageModeShared,
        );

        let kernel = self
            .mv_simd_library
            .get_function("mv_iq4_nl_q8_0_simd", None)
            .map_err(|e| format!("Failed to get kernel: {}", e))?;
        let pipeline = self
            .device
            .new_compute_pipeline_state_with_function(&kernel)
            .map_err(|e| format!("Failed to create pipeline: {}", e))?;

        let command_buffer = self.queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();

        encoder.set_compute_pipeline_state(&pipeline);
        encoder.set_buffer(0, Some(&weights_buffer), 0);
        encoder.set_buffer(1, Some(&input_buffer), 0);
        encoder.set_buffer(2, Some(&output_buffer), 0);
        encoder.set_bytes(
            3,
            std::mem::size_of::<u32>() as u64,
            &(m as u32) as *const u32 as *const std::ffi::c_void,
        );
        encoder.set_bytes(
            4,
            std::mem::size_of::<u32>() as u64,
            &(nb as u32) as *const u32 as *const std::ffi::c_void,
        );

        // One threadgroup per row (32 threads per threadgroup)
        let thread_group_size = MTLSize { width: 32, height: 1, depth: 1 };
        let thread_group_count =
            MTLSize { width: m as u64, height: 1, depth: 1 };

        encoder.dispatch_thread_groups(thread_group_count, thread_group_size);
        encoder.end_encoding();

        command_buffer.commit();
        command_buffer.wait_until_completed();

        let output_ptr = output_buffer.contents() as *const f32;
        Ok(unsafe { std::slice::from_raw_parts(output_ptr, m).to_vec() })
    }

    pub fn mv_q4_0_f32(
        &self,
        m: usize,
        k: usize,
        weights: &[BlockQ4_0],
        input: &[f32],
    ) -> Result<Vec<f32>, String> {
        let nb = k / 32;
        let nb01 = (nb * std::mem::size_of::<BlockQ4_0>()) as u64; // Byte stride per row

        #[repr(C)]
        struct MvArgs {
            ne00: i32,
            ne01: i32,
            ne02: i32,
            nb00: u64,
            nb01: u64,
            nb02: u64,
            nb03: u64,
            ne10: i32,
            ne11: i32,
            ne12: i32,
            nb10: u64,
            nb11: u64,
            nb12: u64,
            nb13: u64,
            ne0: i32,
            ne1: i32,
            nr0: i32,
            r2: i16,
            r3: i16,
        }

        let args = MvArgs {
            ne00: k as i32,
            ne01: m as i32,
            ne02: 1,
            nb00: 1,
            nb01,
            nb02: 0,
            nb03: 0,
            ne10: k as i32,
            ne11: 1,
            ne12: 1,
            nb10: 4,
            nb11: k as u64 * 4,
            nb12: 0,
            nb13: 0,
            ne0: m as i32,
            ne1: 1,
            nr0: 4,
            r2: 1,
            r3: 1, // N_R0_Q4_0 = 4
        };

        // Get or create cached buffers (NO data copy on repeated calls with same size)
        let weights_size = weights.len() * std::mem::size_of::<BlockQ4_0>();
        let input_size = input.len() * std::mem::size_of::<f32>();
        let output_size = m * std::mem::size_of::<f32>();

        // Weights buffer - cache by data pointer (weights are static during inference)
        let weights_buffer = {
            let data_ptr = weights.as_ptr() as usize;
            let mut cache = self.weight_buffer_cache.borrow_mut();

            if let Some(buffer) = cache.get(&data_ptr) {
                // Buffer already exists for this tensor
                std::rc::Rc::clone(buffer)
            } else {
                // Create new buffer and cache it (zero-copy mapping)
                let buffer = std::rc::Rc::new(
                    self.device.new_buffer_with_bytes_no_copy(
                        weights.as_ptr() as *const std::ffi::c_void,
                        weights_size as u64,
                        metal::MTLResourceOptions::StorageModeShared,
                        None, // No deallocator - weights are static model data
                    ),
                );
                cache.insert(data_ptr, std::rc::Rc::clone(&buffer));
                buffer
            }
        };

        // Input buffer - data changes every layer, so always upload
        let input_buffer = {
            let mut buf_cell = self.mv_input_buffer.borrow_mut();
            let mut size_cell = self.mv_input_size.borrow_mut();

            if *size_cell != input_size {
                // Size changed, create new buffer with data
                let buf = self.device.new_buffer_with_data(
                    input.as_ptr() as *const std::ffi::c_void,
                    input_size as u64,
                    metal::MTLResourceOptions::StorageModeShared,
                );
                *buf_cell = Some(buf.clone());
                *size_cell = input_size;
                buf
            } else {
                // Size matches, reuse buffer but copy new data
                let buf = buf_cell.as_ref().unwrap().clone();
                let contents = buf.contents();
                unsafe {
                    std::ptr::copy_nonoverlapping(
                        input.as_ptr() as *const std::ffi::c_void,
                        contents,
                        input_size,
                    );
                }
                buf
            }
        };

        // Output buffer
        let output_buffer = {
            let mut buf_cell = self.mv_output_buffer.borrow_mut();
            let mut size_cell = self.mv_output_size.borrow_mut();

            if *size_cell != output_size {
                let buf = self.device.new_buffer(
                    output_size as u64,
                    metal::MTLResourceOptions::StorageModeShared,
                );
                *buf_cell = Some(buf.clone());
                *size_cell = output_size;
                buf
            } else {
                buf_cell.as_ref().unwrap().clone()
            }
        };

        // Use cached pipeline
        let command_buffer = self.queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();

        encoder.set_compute_pipeline_state(&self.mv_q4_0_pipeline);
        // Template kernel signature: (args, src0, src1, dst)
        // Index 0: args struct
        // Index 1: src0 (weights)
        // Index 2: src1 (input)
        // Index 3: dst (output)
        encoder.set_bytes(
            0,
            std::mem::size_of::<MvArgs>() as u64,
            &args as *const MvArgs as *const std::ffi::c_void,
        );
        encoder.set_buffer(1, Some(&weights_buffer), 0);
        encoder.set_buffer(2, Some(&input_buffer), 0);
        encoder.set_buffer(3, Some(&output_buffer), 0);

        // Q4_0: N_R0 = 4, NSG = 2
        const NSG: u64 = 2;
        const NR0: u64 = 4; // N_R0_Q4_0 = 4
        const ROWS_PER_THREADGROUP: u64 = NSG * NR0; // 8

        let thread_group_size = MTLSize {
            width: 32,
            height: NSG, // 2D: Y = 2 for 2 SIMD groups
            depth: 1,
        };
        let thread_group_count = MTLSize {
            width: (m as u64 + ROWS_PER_THREADGROUP - 1) / ROWS_PER_THREADGROUP,
            height: 1,
            depth: 1,
        };

        encoder.dispatch_thread_groups(thread_group_count, thread_group_size);
        encoder.end_encoding();

        command_buffer.commit();
        command_buffer.wait_until_completed();

        let output_ptr = output_buffer.contents() as *const f32;
        Ok(unsafe { std::slice::from_raw_parts(output_ptr, m).to_vec() })
    }

    /// Q4_0 x F32 Matrix-Vector (zero-copy version, returns GPU buffer)
    pub fn mv_q4_0_f32_gpu(
        &self,
        m: usize,
        k: usize,
        weights: &[BlockQ4_0],
        input_buffer: &Buffer,
    ) -> Result<std::rc::Rc<Buffer>, String> {
        let nb = k / 32;
        let nb01 = (nb * std::mem::size_of::<BlockQ4_0>()) as u64;

        #[repr(C)]
        struct MvArgs {
            ne00: i32,
            ne01: i32,
            ne02: i32,
            nb00: u64,
            nb01: u64,
            nb02: u64,
            nb03: u64,
            ne10: i32,
            ne11: i32,
            ne12: i32,
            nb10: u64,
            nb11: u64,
            nb12: u64,
            nb13: u64,
            ne0: i32,
            ne1: i32,
            nr0: i32,
            r2: i16,
            r3: i16,
        }

        let args = MvArgs {
            ne00: k as i32,
            ne01: m as i32,
            ne02: 1,
            nb00: 1,
            nb01,
            nb02: 0,
            nb03: 0,
            ne10: k as i32,
            ne11: 1,
            ne12: 1,
            nb10: 4,
            nb11: k as u64 * 4,
            nb12: 0,
            nb13: 0,
            ne0: m as i32,
            ne1: 1,
            nr0: 4,
            r2: 1,
            r3: 1,
        };

        let weights_size = weights.len() * std::mem::size_of::<BlockQ4_0>();
        let output_size = m * std::mem::size_of::<f32>();

        // Weights buffer - cache by data pointer
        let weights_buffer = {
            let data_ptr = weights.as_ptr() as usize;
            let mut cache = self.weight_buffer_cache.borrow_mut();

            if let Some(buffer) = cache.get(&data_ptr) {
                std::rc::Rc::clone(buffer)
            } else {
                let buffer =
                    std::rc::Rc::new(self.device.new_buffer_with_data(
                        weights.as_ptr() as *const std::ffi::c_void,
                        weights_size as u64,
                        metal::MTLResourceOptions::StorageModeShared,
                    ));
                cache.insert(data_ptr, std::rc::Rc::clone(&buffer));
                buffer
            }
        };

        // Output buffer
        let output_buffer = std::rc::Rc::new(self.device.new_buffer(
            output_size as u64,
            metal::MTLResourceOptions::StorageModeShared,
        ));

        let command_buffer = self.queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();

        encoder.set_compute_pipeline_state(&self.mv_q4_0_pipeline);
        encoder.set_bytes(
            0,
            std::mem::size_of::<MvArgs>() as u64,
            &args as *const MvArgs as *const std::ffi::c_void,
        );
        encoder.set_buffer(1, Some(&weights_buffer), 0);
        encoder.set_buffer(2, Some(input_buffer), 0);
        encoder.set_buffer(3, Some(&output_buffer), 0);

        const NSG: u64 = 2;
        const NR0: u64 = 4;
        const ROWS_PER_THREADGROUP: u64 = NSG * NR0;

        let thread_group_size = MTLSize { width: 32, height: NSG, depth: 1 };
        let thread_group_count = MTLSize {
            width: (m as u64 + ROWS_PER_THREADGROUP - 1) / ROWS_PER_THREADGROUP,
            height: 1,
            depth: 1,
        };

        encoder.dispatch_thread_groups(thread_group_count, thread_group_size);
        encoder.end_encoding();

        command_buffer.commit();
        command_buffer.wait_until_completed();

        Ok(output_buffer)
    }

    /// Q4_0 x F32 Matrix-Vector (fully zero-copy, both weights and input as GPU buffers)
    pub fn mv_q4_0_f32_gpu_weights(
        &self,
        m: usize,
        k: usize,
        weights_buffer: &Buffer,
        input_buffer: &Buffer,
    ) -> Result<std::rc::Rc<Buffer>, String> {
        let nb = k / 32;
        let nb01 = (nb * std::mem::size_of::<BlockQ4_0>()) as u64;

        #[repr(C)]
        struct MvArgs {
            ne00: i32,
            ne01: i32,
            ne02: i32,
            nb00: u64,
            nb01: u64,
            nb02: u64,
            nb03: u64,
            ne10: i32,
            ne11: i32,
            ne12: i32,
            nb10: u64,
            nb11: u64,
            nb12: u64,
            nb13: u64,
            ne0: i32,
            ne1: i32,
            nr0: i32,
            r2: i16,
            r3: i16,
        }

        let args = MvArgs {
            ne00: k as i32,
            ne01: m as i32,
            ne02: 1,
            nb00: 1,
            nb01,
            nb02: 0,
            nb03: 0,
            ne10: k as i32,
            ne11: 1,
            ne12: 1,
            nb10: 4,
            nb11: k as u64 * 4,
            nb12: 0,
            nb13: 0,
            ne0: m as i32,
            ne1: 1,
            nr0: 4,
            r2: 1,
            r3: 1,
        };

        let output_size = m * std::mem::size_of::<f32>();
        let output_buffer = std::rc::Rc::new(self.device.new_buffer(
            output_size as u64,
            metal::MTLResourceOptions::StorageModeShared,
        ));

        let command_buffer = self.queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();

        encoder.set_compute_pipeline_state(&self.mv_q4_0_pipeline);
        encoder.set_bytes(
            0,
            std::mem::size_of::<MvArgs>() as u64,
            &args as *const MvArgs as *const std::ffi::c_void,
        );
        encoder.set_buffer(1, Some(weights_buffer), 0);
        encoder.set_buffer(2, Some(input_buffer), 0);
        encoder.set_buffer(3, Some(&output_buffer), 0);

        const NSG: u64 = 2;
        const NR0: u64 = 4;
        const ROWS_PER_THREADGROUP: u64 = NSG * NR0;

        let thread_group_size = MTLSize { width: 32, height: NSG, depth: 1 };
        let thread_group_count = MTLSize {
            width: (m as u64 + ROWS_PER_THREADGROUP - 1) / ROWS_PER_THREADGROUP,
            height: 1,
            depth: 1,
        };

        encoder.dispatch_thread_groups(thread_group_count, thread_group_size);
        encoder.end_encoding();

        command_buffer.commit();
        command_buffer.wait_until_completed();

        Ok(output_buffer)
    }

    /// Q4_0 MV (batched, encode into shared encoder)
    pub fn mv_q4_0_encode(
        &self,
        encoder: &ComputeCommandEncoderRef,
        weights_buffer: &Buffer,
        input_buffer: &Buffer,
        output_buffer: &Buffer,
        m: usize,
        k: usize,
    ) {
        let nb = k / 32;
        let nb01 = (nb * std::mem::size_of::<BlockQ4_0>()) as u64;

        #[repr(C)]
        struct MvArgs {
            ne00: i32,
            ne01: i32,
            ne02: i32,
            nb00: u64,
            nb01: u64,
            nb02: u64,
            nb03: u64,
            ne10: i32,
            ne11: i32,
            ne12: i32,
            nb10: u64,
            nb11: u64,
            nb12: u64,
            nb13: u64,
            ne0: i32,
            ne1: i32,
            nr0: i32,
            r2: i16,
            r3: i16,
        }

        let args = MvArgs {
            ne00: k as i32,
            ne01: m as i32,
            ne02: 1,
            nb00: 1,
            nb01,
            nb02: 0,
            nb03: 0,
            ne10: k as i32,
            ne11: 1,
            ne12: 1,
            nb10: 4,
            nb11: k as u64 * 4,
            nb12: 0,
            nb13: 0,
            ne0: m as i32,
            ne1: 1,
            nr0: 4,
            r2: 1,
            r3: 1,
        };

        encoder.set_compute_pipeline_state(&self.mv_q4_0_pipeline);
        encoder.set_bytes(
            0,
            std::mem::size_of::<MvArgs>() as u64,
            &args as *const MvArgs as *const std::ffi::c_void,
        );
        encoder.set_buffer(1, Some(weights_buffer), 0);
        encoder.set_buffer(2, Some(input_buffer), 0);
        encoder.set_buffer(3, Some(output_buffer), 0);

        const NSG: u64 = 2;
        const ROWS_PER_THREADGROUP: u64 = NSG * 4;

        let thread_group_size = MTLSize { width: 32, height: NSG, depth: 1 };
        let thread_group_count = MTLSize {
            width: (m as u64 + ROWS_PER_THREADGROUP - 1) / ROWS_PER_THREADGROUP,
            height: 1,
            depth: 1,
        };

        encoder.dispatch_thread_groups(thread_group_count, thread_group_size);
    }

    /// Generic MV encode that dispatches based on dtype
    /// Returns true if successful, false if format not supported
    pub fn mv_encode(
        &self,
        encoder: &ComputeCommandEncoderRef,
        weights_buffer: &Buffer,
        input_buffer: &Buffer,
        output_buffer: &Buffer,
        m: usize,
        k: usize,
        dtype: crate::model::tensor::DataType,
    ) -> bool {
        match dtype {
            crate::model::tensor::DataType::Q4_0 => {
                self.mv_q4_0_encode(
                    encoder,
                    weights_buffer,
                    input_buffer,
                    output_buffer,
                    m,
                    k,
                );
                true
            }
            crate::model::tensor::DataType::Q4_1 => self.mv_standard_q_encode(
                encoder,
                weights_buffer,
                input_buffer,
                output_buffer,
                m,
                k,
                dtype,
            ),
            crate::model::tensor::DataType::Q5_0 => self.mv_standard_q_encode(
                encoder,
                weights_buffer,
                input_buffer,
                output_buffer,
                m,
                k,
                dtype,
            ),
            crate::model::tensor::DataType::Q5_1 => self.mv_standard_q_encode(
                encoder,
                weights_buffer,
                input_buffer,
                output_buffer,
                m,
                k,
                dtype,
            ),
            crate::model::tensor::DataType::Q8_0 => self.mv_standard_q_encode(
                encoder,
                weights_buffer,
                input_buffer,
                output_buffer,
                m,
                k,
                dtype,
            ),
            crate::model::tensor::DataType::Q4_K
            | crate::model::tensor::DataType::Q5_K
            | crate::model::tensor::DataType::Q2_K
            | crate::model::tensor::DataType::Q6_K
            | crate::model::tensor::DataType::Q3_K => self.mv_kquant_encode(
                encoder,
                weights_buffer,
                input_buffer,
                output_buffer,
                m,
                k,
                dtype,
            ),
            // IQ formats
            crate::model::tensor::DataType::IQ4_XS
            | crate::model::tensor::DataType::IQ1_S
            | crate::model::tensor::DataType::IQ2_XXS
            | crate::model::tensor::DataType::IQ2_XS
            | crate::model::tensor::DataType::IQ2_S
            | crate::model::tensor::DataType::IQ3_XXS
            | crate::model::tensor::DataType::IQ3_S
            | crate::model::tensor::DataType::IQ4_NL => self.mv_iq_encode(
                encoder,
                weights_buffer,
                input_buffer,
                output_buffer,
                m,
                k,
                dtype,
            ),
            // F16
            crate::model::tensor::DataType::F16 => self.mv_f16_encode(
                encoder,
                weights_buffer,
                input_buffer,
                output_buffer,
                m,
                k,
            ),
            // Add more formats as needed
            _ => false,
        }
    }

    /// Standard Q format MV encode (Q4_1, Q5_0, Q5_1, Q8_0)
    fn mv_standard_q_encode(
        &self,
        encoder: &ComputeCommandEncoderRef,
        weights_buffer: &Buffer,
        input_buffer: &Buffer,
        output_buffer: &Buffer,
        m: usize,
        k: usize,
        dtype: crate::model::tensor::DataType,
    ) -> bool {
        let (block_size, pipeline) = match dtype {
            crate::model::tensor::DataType::Q4_1 => {
                (32, &self.mv_q4_1_pipeline)
            }
            crate::model::tensor::DataType::Q5_0 => {
                (32, &self.mv_q5_0_pipeline)
            }
            crate::model::tensor::DataType::Q5_1 => {
                (32, &self.mv_q5_1_pipeline)
            }
            crate::model::tensor::DataType::Q8_0 => {
                (32, &self.mv_q8_0_pipeline)
            }
            _ => return false,
        };

        let nb = k / block_size;
        let nb01 = match dtype {
            crate::model::tensor::DataType::Q4_1 => {
                (nb * std::mem::size_of::<BlockQ4_1>()) as u64
            }
            crate::model::tensor::DataType::Q5_0 => {
                (nb * std::mem::size_of::<BlockQ5_0>()) as u64
            }
            crate::model::tensor::DataType::Q5_1 => {
                (nb * std::mem::size_of::<BlockQ5_1>()) as u64
            }
            crate::model::tensor::DataType::Q8_0 => {
                (nb * std::mem::size_of::<BlockQ8_0>()) as u64
            }
            _ => return false,
        };

        #[repr(C)]
        struct MvArgs {
            ne00: i32,
            ne01: i32,
            ne02: i32,
            nb00: u64,
            nb01: u64,
            nb02: u64,
            nb03: u64,
            ne10: i32,
            ne11: i32,
            ne12: i32,
            nb10: u64,
            nb11: u64,
            nb12: u64,
            nb13: u64,
            ne0: i32,
            ne1: i32,
            nr0: i32,
            r2: i16,
            r3: i16,
        }

        let args = MvArgs {
            ne00: k as i32,
            ne01: m as i32,
            ne02: 1,
            nb00: 1,
            nb01,
            nb02: 0,
            nb03: 0,
            ne10: k as i32,
            ne11: 1,
            ne12: 1,
            nb10: 4,
            nb11: k as u64 * 4,
            nb12: 0,
            nb13: 0,
            ne0: m as i32,
            ne1: 1,
            nr0: 4,
            r2: 1,
            r3: 1,
        };

        encoder.set_compute_pipeline_state(pipeline);
        encoder.set_bytes(
            0,
            std::mem::size_of::<MvArgs>() as u64,
            &args as *const MvArgs as *const std::ffi::c_void,
        );
        encoder.set_buffer(1, Some(weights_buffer), 0);
        encoder.set_buffer(2, Some(input_buffer), 0);
        encoder.set_buffer(3, Some(output_buffer), 0);

        const NSG: u64 = 2;
        const ROWS_PER_THREADGROUP: u64 = NSG * 4;

        let thread_group_size = MTLSize { width: 32, height: NSG, depth: 1 };
        let thread_group_count = MTLSize {
            width: (m as u64 + ROWS_PER_THREADGROUP - 1) / ROWS_PER_THREADGROUP,
            height: 1,
            depth: 1,
        };

        encoder.dispatch_thread_groups(thread_group_count, thread_group_size);
        true
    }

    /// F16 GEMV encode
    fn mv_f16_encode(
        &self,
        encoder: &ComputeCommandEncoderRef,
        weights_buffer: &Buffer,
        input_buffer: &Buffer,
        output_buffer: &Buffer,
        m: usize,
        k: usize,
    ) -> bool {
        // Parameter struct matching mul_mv_args
        #[repr(C)]
        struct MvArgs {
            ne00: i32, // K
            ne01: i32, // M (rows)
            ne02: i32,
            nb00: u64,
            nb01: u64, // row stride
            nb02: u64,
            nb03: u64,
            ne10: i32,
            ne11: i32,
            ne12: i32,
            nb10: u64,
            nb11: u64,
            nb12: u64,
            nb13: u64,
            ne0: i32,
            ne1: i32,
            nr0: i32, // NR0 (2 for normal kernels, 32 for short)
            r2: i16,
            r3: i16,
            nsg: i16,        // Number of simdgroups
            ne12_batch: i16, // batch dimension
        }

        if k < 32 {
            // Short kernel: nsg=1, nr0=32, each thread processes one row
            encoder.set_compute_pipeline_state(&self.mv_f16_short_pipeline);

            let args = MvArgs {
                ne00: k as i32,
                ne01: m as i32,
                ne02: 1,
                nb00: 2, // sizeof(half)
                nb01: (k * 2) as u64,
                nb02: 0,
                nb03: 0,
                ne10: k as i32,
                ne11: 1,
                ne12: 1,
                nb10: 4, // sizeof(float)
                nb11: (k * 4) as u64,
                nb12: 0,
                nb13: 0,
                ne0: m as i32,
                ne1: 1,
                nr0: 32,
                r2: 1,
                r3: 1,
                nsg: 1, // NSG=1 for short kernel
                ne12_batch: 1,
            };

            let thread_group_size = MTLSize { width: 32, height: 1, depth: 1 };
            let thread_group_count =
                MTLSize { width: ((m + 31) / 32) as u64, height: 1, depth: 1 };

            encoder.set_bytes(
                0,
                std::mem::size_of::<MvArgs>() as u64,
                &args as *const MvArgs as *const std::ffi::c_void,
            );
            encoder.set_buffer(1, Some(weights_buffer), 0);
            encoder.set_buffer(2, Some(input_buffer), 0);
            encoder.set_buffer(3, Some(output_buffer), 0);
            encoder
                .dispatch_thread_groups(thread_group_count, thread_group_size);
        } else {
            let use_4 = k % 4 == 0; // Use float4 version if K % 4 == 0

            // Try optimized kernel for large K (llama.cpp-style with nxpsg)
            // For K >= 256, use nxpsg=16 (matching llama.cpp: ne00 % 256 == 0 && ne11 < 3)
            if use_4 && k >= 256 {
                // Use optimized kernel with nxpsg=16
                encoder.set_compute_pipeline_state(
                    &self.mv_f16_opt_r1_1_nxpsg16_pipeline,
                );

                // mul_mv_ext_args structure (matching Metal kernel)
                #[repr(C)]
                struct MvExtArgs {
                    ne00: i32,
                    ne01: i32,
                    ne02: i32,
                    nb00: u64,
                    nb01: u64,
                    nb02: u64,
                    nb03: u64,
                    ne10: i32,
                    ne11: i32,
                    ne12: i32,
                    nb10: u64,
                    nb11: u64,
                    nb12: u64,
                    nb13: u64,
                    ne0: i32,
                    ne1: i32,
                }

                let args = MvExtArgs {
                    ne00: k as i32,
                    ne01: m as i32,
                    ne02: 1,
                    nb00: 2, // sizeof(half)
                    nb01: (k * 2) as u64,
                    nb02: 0,
                    nb03: 0,
                    ne10: k as i32,
                    ne11: 1, // batch=1 for decode
                    ne12: 1,
                    nb10: 4, // sizeof(float)
                    nb11: (k * 4) as u64,
                    nb12: 0,
                    nb13: 0,
                    ne0: m as i32,
                    ne1: 1,
                };

                encoder.set_bytes(
                    0,
                    std::mem::size_of::<MvExtArgs>() as u64,
                    &args as *const MvExtArgs as *const std::ffi::c_void,
                );
                encoder.set_buffer(1, Some(weights_buffer), 0);
                encoder.set_buffer(2, Some(input_buffer), 0);
                encoder.set_buffer(3, Some(output_buffer), 0);

                // Threadgroup configuration for nxpsg=16, nsg=1
                // nypsg = 32/nxpsg = 2 (threads per simdgroup in Y dimension)
                // Each threadgroup has 1 simdgroup = 32 threads
                let thread_group_size =
                    MTLSize { width: 32, height: 1, depth: 1 };
                // Number of threadgroups = (M / nypsg) = (M / 2)
                let thread_group_count = MTLSize {
                    width: ((m + 1) / 2) as u64,
                    height: 1,
                    depth: 1,
                };

                encoder.dispatch_thread_groups(
                    thread_group_count,
                    thread_group_size,
                );
                return true;
            }

            // Fall back to original kernels
            // Match llama.cpp NSG formula: min(4, (K+127)/128)
            let nsg = if use_4 { std::cmp::min(4, (k + 127) / 128) } else { 1 };

            let pipeline = if use_4 {
                match nsg {
                    1 => &self.mv_f16_4_nsg1_pipeline,
                    2 => &self.mv_f16_4_nsg2_pipeline,
                    _ => &self.mv_f16_4_nsg4_pipeline, // 4 or larger
                }
            } else {
                &self.mv_f16_basic_pipeline
            };

            encoder.set_compute_pipeline_state(pipeline);

            // Match llama.cpp NSG formula: min(4, (K+127)/128)
            let nsg: i16 = if use_4 {
                std::cmp::min(4, (k + 127) / 128) as i16
            } else {
                1
            };

            let args = MvArgs {
                ne00: k as i32,
                ne01: m as i32,
                ne02: 1,
                nb00: 2,
                nb01: (k * 2) as u64,
                nb02: 0,
                nb03: 0,
                ne10: k as i32,
                ne11: 1,
                ne12: 1,
                nb10: 4,
                nb11: (k * 4) as u64,
                nb12: 0,
                nb13: 0,
                ne0: m as i32,
                ne1: 1,
                nr0: 2,
                r2: 1,
                r3: 1,
                nsg: nsg as i16, // Store in args for reference
                ne12_batch: 1,
            };

            encoder.set_bytes(
                0,
                std::mem::size_of::<MvArgs>() as u64,
                &args as *const MvArgs as *const std::ffi::c_void,
            );
            encoder.set_buffer(1, Some(weights_buffer), 0);
            encoder.set_buffer(2, Some(input_buffer), 0);
            encoder.set_buffer(3, Some(output_buffer), 0);

            // Threadgroup size scales with nsg
            let thread_group_size =
                MTLSize { width: (nsg as u64 * 32), height: 1, depth: 1 };
            let thread_group_count =
                MTLSize { width: ((m + 1) / 2) as u64, height: 1, depth: 1 };

            // Allocate threadgroup memory for reduction (2 rows * 32 threads * nsg simdgroups)
            let threadgroup_mem_size =
                (2 * 32 * nsg as usize) * std::mem::size_of::<f32>();
            encoder
                .set_threadgroup_memory_length(0, threadgroup_mem_size as u64);

            encoder
                .dispatch_thread_groups(thread_group_count, thread_group_size);
        }

        true
    }

    /// K-quant family MV encode (shared implementation)
    fn mv_kquant_encode(
        &self,
        encoder: &ComputeCommandEncoderRef,
        weights_buffer: &Buffer,
        input_buffer: &Buffer,
        output_buffer: &Buffer,
        m: usize,
        k: usize,
        dtype: crate::model::tensor::DataType,
    ) -> bool {
        let nb = k / 256;
        let nb01 = match dtype {
            crate::model::tensor::DataType::Q4_K => {
                (nb * std::mem::size_of::<BlockQ4K>()) as u64
            }
            crate::model::tensor::DataType::Q5_K => {
                (nb * std::mem::size_of::<BlockQ5K>()) as u64
            }
            crate::model::tensor::DataType::Q2_K => {
                (nb * std::mem::size_of::<BlockQ2K>()) as u64
            }
            crate::model::tensor::DataType::Q6_K => {
                (nb * std::mem::size_of::<BlockQ6K>()) as u64
            }
            crate::model::tensor::DataType::Q3_K => {
                (nb * std::mem::size_of::<BlockQ3K>()) as u64
            }
            _ => return false,
        };

        let args = KquantMvArgs::new(m, k, nb01);

        let pipeline = match dtype {
            crate::model::tensor::DataType::Q4_K => &self.mv_q4_k_pipeline,
            crate::model::tensor::DataType::Q5_K => &self.mv_q5_k_pipeline,
            crate::model::tensor::DataType::Q2_K => &self.mv_q2_k_pipeline,
            crate::model::tensor::DataType::Q6_K => &self.mv_q6_k_pipeline,
            crate::model::tensor::DataType::Q3_K => &self.mv_q3_k_pipeline,
            _ => return false,
        };

        encoder.set_compute_pipeline_state(pipeline);
        encoder.set_buffer(0, Some(weights_buffer), 0);
        encoder.set_buffer(1, Some(input_buffer), 0);
        encoder.set_buffer(2, Some(output_buffer), 0);
        encoder.set_bytes(
            3,
            std::mem::size_of::<KquantMvArgs>() as u64,
            &args as *const KquantMvArgs as *const std::ffi::c_void,
        );

        // Threadgroup configuration (matching kernel expectations)
        // Q4_K: NSG=2, N_R0=2 (2 simdgroups, each processes 2 rows)
        // Q5_K: NSG=2, N_R0=1 (2 simdgroups, each processes 1 row)
        // Q6_K: NSG=1, N_R0=2 (1 simdgroup, processes 2 rows)
        // Q2_K, Q3_K: NSG=2, N_R0=2 (similar to Q4_K)
        let (nsg, nr0) = match dtype {
            crate::model::tensor::DataType::Q6_K => (1, 2),
            crate::model::tensor::DataType::Q5_K => (2, 1),
            _ => (2, 2), // Q4_K, Q2_K, Q3_K
        };

        let threads_per_tg = nsg * 32; // 32 threads per simdgroup
        let tgs = (m + nsg * nr0 - 1) / (nsg * nr0); // threadgroups needed

        // Use 2D threadgroup size to create multiple simdgroups
        // width: 32 threads per simdgroup
        // height: nsg simdgroups
        let thread_group_size =
            MTLSize { width: 32, height: nsg as u64, depth: 1 };
        let thread_group_count =
            MTLSize { width: tgs as u64, height: 1, depth: 1 };

        encoder.dispatch_thread_groups(thread_group_count, thread_group_size);
        true
    }

    /// IQ format MV encode (IQ4_XS, IQ1_S, IQ2_XXS, IQ2_XS, IQ2_S, IQ3_XXS, IQ3_S, IQ4_NL)
    fn mv_iq_encode(
        &self,
        encoder: &ComputeCommandEncoderRef,
        weights_buffer: &Buffer,
        input_buffer: &Buffer,
        output_buffer: &Buffer,
        m: usize,
        k: usize,
        dtype: crate::model::tensor::DataType,
    ) -> bool {
        let nb = k / 256; // QK_K = 256 for all IQ formats

        // Block sizes and NR0 values differ by format
        let (nb01, nr0, pipeline) = match dtype {
            crate::model::tensor::DataType::IQ4_XS => {
                let nb01 = (nb * std::mem::size_of::<BlockIQ4XS>()) as u64;
                (nb01, 2, &self.mv_iq4_xs_pipeline)
            }
            crate::model::tensor::DataType::IQ1_S => {
                let nb01 = (nb * std::mem::size_of::<BlockIQ1S>()) as u64;
                (nb01, 2, &self.mv_iq1_s_pipeline)
            }
            crate::model::tensor::DataType::IQ2_XXS => {
                let nb01 = (nb * std::mem::size_of::<BlockIQ2XXS>()) as u64;
                (nb01, 2, &self.mv_iq2_xxs_pipeline)
            }
            crate::model::tensor::DataType::IQ2_XS => {
                let nb01 = (nb * std::mem::size_of::<BlockIQ2XS>()) as u64;
                (nb01, 2, &self.mv_iq2_xs_pipeline)
            }
            crate::model::tensor::DataType::IQ2_S => {
                let nb01 = (nb * std::mem::size_of::<BlockIQ2S>()) as u64;
                (nb01, 2, &self.mv_iq2_s_pipeline)
            }
            crate::model::tensor::DataType::IQ3_XXS => {
                let nb01 = (nb * std::mem::size_of::<BlockIQ3XXS>()) as u64;
                (nb01, 2, &self.mv_iq3_xxs_pipeline)
            }
            crate::model::tensor::DataType::IQ3_S => {
                let nb01 = (nb * std::mem::size_of::<BlockIQ3S>()) as u64;
                (nb01, 2, &self.mv_iq3_s_pipeline)
            }
            crate::model::tensor::DataType::IQ4_NL => {
                let nb01 = (nb * std::mem::size_of::<BlockIQ4NL>()) as u64;
                (nb01, 2, &self.mv_iq4_nl_pipeline)
            }
            _ => return false,
        };

        #[repr(C)]
        struct MvArgs {
            ne00: u32,
            ne01: u32,
            nb01: u64,
        }

        let args = MvArgs { ne00: k as u32, ne01: m as u32, nb01 };

        encoder.set_compute_pipeline_state(pipeline);
        encoder.set_buffer(0, Some(weights_buffer), 0);
        encoder.set_buffer(1, Some(input_buffer), 0);
        encoder.set_buffer(2, Some(output_buffer), 0);
        encoder.set_bytes(
            3,
            std::mem::size_of::<MvArgs>() as u64,
            &args as *const MvArgs as *const std::ffi::c_void,
        );

        const THREADGROUP_MEM_SIZE: u64 =
            32 * std::mem::size_of::<f32>() as u64;
        encoder.set_threadgroup_memory_length(THREADGROUP_MEM_SIZE, 0);

        const NSG: u64 = 2;
        let rows_per_threadgroup = NSG * nr0;

        let thread_group_size = MTLSize { width: 32, height: NSG, depth: 1 };
        let thread_group_count = MTLSize {
            width: (m as u64 + rows_per_threadgroup - 1) / rows_per_threadgroup,
            height: 1,
            depth: 1,
        };

        encoder.dispatch_thread_groups(thread_group_count, thread_group_size);
        true
    }

    /// Q1_0 x F32 Matrix-Vector
    pub fn mv_q1_0_f32(
        &self,
        m: usize,
        k: usize,
        weights: &[BlockQ1_0],
        input: &[f32],
    ) -> Result<Vec<f32>, String> {
        let nb = k / 32;
        let nb01 = (nb * std::mem::size_of::<BlockQ1_0>()) as u64;

        #[repr(C)]
        struct MvArgs {
            ne00: i32,
            ne01: i32,
            ne02: i32,
            nb00: u64,
            nb01: u64,
            nb02: u64,
            nb03: u64,
            ne10: i32,
            ne11: i32,
            ne12: i32,
            nb10: u64,
            nb11: u64,
            nb12: u64,
            nb13: u64,
            ne0: i32,
            ne1: i32,
            nr0: i32,
            r2: i16,
            r3: i16,
        }
        let args = MvArgs {
            ne00: k as i32,
            ne01: m as i32,
            ne02: 1,
            nb00: std::mem::size_of::<BlockQ1_0>() as u64,
            nb01,
            nb02: nb01,
            nb03: nb01,
            ne10: k as i32,
            ne11: 1,
            ne12: 1,
            nb10: 4,
            nb11: k as u64 * 4,
            nb12: 0,
            nb13: 0,
            ne0: m as i32,
            ne1: 1,
            nr0: 8,
            r2: 1,
            r3: 1,
        };

        let weights_buffer = self.device.new_buffer_with_data(
            weights.as_ptr() as *const std::ffi::c_void,
            (weights.len() * std::mem::size_of::<BlockQ1_0>()) as u64,
            metal::MTLResourceOptions::StorageModeShared,
        );
        let input_buffer = self.device.new_buffer_with_data(
            input.as_ptr() as *const std::ffi::c_void,
            (input.len() * 4) as u64,
            metal::MTLResourceOptions::StorageModeShared,
        );
        let output_buffer = self.device.new_buffer(
            (m * 4) as u64,
            metal::MTLResourceOptions::StorageModeShared,
        );

        let command_buffer = self.queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(&self.mv_q1_0_pipeline);
        encoder.set_bytes(
            0,
            std::mem::size_of::<MvArgs>() as u64,
            &args as *const _ as *const std::ffi::c_void,
        );
        encoder.set_buffer(1, Some(&weights_buffer), 0);
        encoder.set_buffer(2, Some(&input_buffer), 0);
        encoder.set_buffer(3, Some(&output_buffer), 0);
        encoder.dispatch_thread_groups(
            metal::MTLSize::new(((m + 7) / 8) as u64, 1, 1),
            metal::MTLSize::new(32, 8, 1),
        );
        encoder.end_encoding();
        command_buffer.commit();
        command_buffer.wait_until_completed();

        Ok(unsafe {
            std::slice::from_raw_parts(
                output_buffer.contents() as *const f32,
                m,
            )
            .to_vec()
        })
    }

    /// Q2_0 x F32 Matrix-Vector
    pub fn mv_q2_0_f32(
        &self,
        m: usize,
        k: usize,
        weights: &[BlockQ2_0],
        input: &[f32],
    ) -> Result<Vec<f32>, String> {
        let nb = k / 64;
        let nb01 = (nb * std::mem::size_of::<BlockQ2_0>()) as u64;

        #[repr(C)]
        struct MvArgs {
            ne00: i32,
            ne01: i32,
            ne02: i32,
            nb00: u64,
            nb01: u64,
            nb02: u64,
            nb03: u64,
            ne10: i32,
            ne11: i32,
            ne12: i32,
            nb10: u64,
            nb11: u64,
            nb12: u64,
            nb13: u64,
            ne0: i32,
            ne1: i32,
            nr0: i32,
            r2: i16,
            r3: i16,
        }
        let args = MvArgs {
            ne00: k as i32,
            ne01: m as i32,
            ne02: 1,
            nb00: std::mem::size_of::<BlockQ2_0>() as u64,
            nb01,
            nb02: nb01,
            nb03: nb01,
            ne10: k as i32,
            ne11: 1,
            ne12: 1,
            nb10: 4,
            nb11: k as u64 * 4,
            nb12: 0,
            nb13: 0,
            ne0: m as i32,
            ne1: 1,
            nr0: 8,
            r2: 1,
            r3: 1,
        };

        let weights_buffer = self.device.new_buffer_with_data(
            weights.as_ptr() as *const std::ffi::c_void,
            (weights.len() * std::mem::size_of::<BlockQ2_0>()) as u64,
            metal::MTLResourceOptions::StorageModeShared,
        );
        let input_buffer = self.device.new_buffer_with_data(
            input.as_ptr() as *const std::ffi::c_void,
            (input.len() * 4) as u64,
            metal::MTLResourceOptions::StorageModeShared,
        );
        let output_buffer = self.device.new_buffer(
            (m * 4) as u64,
            metal::MTLResourceOptions::StorageModeShared,
        );

        let command_buffer = self.queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(&self.mv_q2_0_pipeline);
        encoder.set_bytes(
            0,
            std::mem::size_of::<MvArgs>() as u64,
            &args as *const _ as *const std::ffi::c_void,
        );
        encoder.set_buffer(1, Some(&weights_buffer), 0);
        encoder.set_buffer(2, Some(&input_buffer), 0);
        encoder.set_buffer(3, Some(&output_buffer), 0);
        encoder.dispatch_thread_groups(
            metal::MTLSize::new(((m + 7) / 8) as u64, 1, 1),
            metal::MTLSize::new(32, 8, 1),
        );
        encoder.end_encoding();
        command_buffer.commit();
        command_buffer.wait_until_completed();

        Ok(unsafe {
            std::slice::from_raw_parts(
                output_buffer.contents() as *const f32,
                m,
            )
            .to_vec()
        })
    }

    /// Q4_1 x F32 Matrix-Vector (fully zero-copy, GPU buffers)
    pub fn mv_q4_1_f32_gpu_weights(
        &self,
        m: usize,
        k: usize,
        weights_buffer: &Buffer,
        input_buffer: &Buffer,
    ) -> Result<std::rc::Rc<Buffer>, String> {
        let nb = k / 32;
        let nb01 = (nb * std::mem::size_of::<BlockQ4_1>()) as u64;

        #[repr(C)]
        struct MvArgs {
            ne00: i32,
            ne01: i32,
            ne02: i32,
            nb00: u64,
            nb01: u64,
            nb02: u64,
            nb03: u64,
            ne10: i32,
            ne11: i32,
            ne12: i32,
            nb10: u64,
            nb11: u64,
            nb12: u64,
            nb13: u64,
            ne0: i32,
            ne1: i32,
            nr0: i32,
            r2: i16,
            r3: i16,
        }
        let args = MvArgs {
            ne00: k as i32,
            ne01: m as i32,
            ne02: 1,
            nb00: std::mem::size_of::<BlockQ4_1>() as u64,
            nb01,
            nb02: nb01,
            nb03: nb01,
            ne10: k as i32,
            ne11: 1,
            ne12: 1,
            nb10: 4,
            nb11: k as u64 * 4,
            nb12: 0,
            nb13: 0,
            ne0: m as i32,
            ne1: 1,
            nr0: 4,
            r2: 1,
            r3: 1,
        };

        let output_buffer = std::rc::Rc::new(self.device.new_buffer(
            (m * 4) as u64,
            metal::MTLResourceOptions::StorageModeShared,
        ));

        let command_buffer = self.queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(&self.mv_q4_1_pipeline);
        encoder.set_bytes(
            0,
            std::mem::size_of::<MvArgs>() as u64,
            &args as *const MvArgs as *const std::ffi::c_void,
        );
        encoder.set_buffer(1, Some(weights_buffer), 0);
        encoder.set_buffer(2, Some(input_buffer), 0);
        encoder.set_buffer(3, Some(&output_buffer), 0);

        let thread_group_size = MTLSize { width: 32, height: 4, depth: 1 };
        let thread_group_count =
            MTLSize { width: (m as u64 + 3) / 4, height: 1, depth: 1 };

        encoder.dispatch_thread_groups(thread_group_count, thread_group_size);
        encoder.end_encoding();
        command_buffer.commit();
        command_buffer.wait_until_completed();

        Ok(output_buffer)
    }

    /// Q5_0 x F32 Matrix-Vector
    pub fn mv_q4_1_f32(
        &self,
        m: usize,
        k: usize,
        weights: &[BlockQ4_1],
        input: &[f32],
    ) -> Result<Vec<f32>, String> {
        let nb = k / 32;
        let nb01 = (nb * std::mem::size_of::<BlockQ4_1>()) as u64;

        #[repr(C)]
        struct MvArgs {
            ne00: i32,
            ne01: i32,
            ne02: i32,
            nb00: u64,
            nb01: u64,
            nb02: u64,
            nb03: u64,
            ne10: i32,
            ne11: i32,
            ne12: i32,
            nb10: u64,
            nb11: u64,
            nb12: u64,
            nb13: u64,
            ne0: i32,
            ne1: i32,
            nr0: i32,
            r2: i16,
            r3: i16,
        }
        let args = MvArgs {
            ne00: k as i32,
            ne01: m as i32,
            ne02: 1,
            nb00: std::mem::size_of::<BlockQ4_1>() as u64,
            nb01,
            nb02: nb01,
            nb03: nb01,
            ne10: k as i32,
            ne11: 1,
            ne12: 1,
            nb10: 4,
            nb11: k as u64 * 4,
            nb12: 0,
            nb13: 0,
            ne0: m as i32,
            ne1: 1,
            nr0: 4,
            r2: 1,
            r3: 1,
        };

        let weights_buffer = self.device.new_buffer_with_data(
            weights.as_ptr() as *const std::ffi::c_void,
            (weights.len() * std::mem::size_of::<BlockQ4_1>()) as u64,
            metal::MTLResourceOptions::StorageModeShared,
        );
        let input_buffer = self.device.new_buffer_with_data(
            input.as_ptr() as *const std::ffi::c_void,
            (input.len() * 4) as u64,
            metal::MTLResourceOptions::StorageModeShared,
        );
        let output_buffer = self.device.new_buffer(
            (m * 4) as u64,
            metal::MTLResourceOptions::StorageModeShared,
        );

        let command_buffer = self.queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(&self.mv_q4_1_pipeline);
        encoder.set_bytes(
            0,
            std::mem::size_of::<MvArgs>() as u64,
            &args as *const _ as *const std::ffi::c_void,
        );
        encoder.set_buffer(1, Some(&weights_buffer), 0);
        encoder.set_buffer(2, Some(&input_buffer), 0);
        encoder.set_buffer(3, Some(&output_buffer), 0);
        encoder.dispatch_thread_groups(
            metal::MTLSize::new(((m + 3) / 4) as u64, 1, 1),
            metal::MTLSize::new(32, 4, 1),
        );
        encoder.end_encoding();
        command_buffer.commit();
        command_buffer.wait_until_completed();

        Ok(unsafe {
            std::slice::from_raw_parts(
                output_buffer.contents() as *const f32,
                m,
            )
            .to_vec()
        })
    }

    /// Q5_0 x F32 Matrix-Vector
    pub fn mv_q5_0_f32(
        &self,
        m: usize,
        k: usize,
        weights: &[BlockQ5_0],
        input: &[f32],
    ) -> Result<Vec<f32>, String> {
        let nb = k / 32;
        let nb01 = (nb * std::mem::size_of::<BlockQ5_0>()) as u64;

        #[repr(C)]
        struct MvArgs {
            ne00: i32,
            ne01: i32,
            ne02: i32,
            nb00: u64,
            nb01: u64,
            nb02: u64,
            nb03: u64,
            ne10: i32,
            ne11: i32,
            ne12: i32,
            nb10: u64,
            nb11: u64,
            nb12: u64,
            nb13: u64,
            ne0: i32,
            ne1: i32,
            nr0: i32,
            r2: i16,
            r3: i16,
        }
        let args = MvArgs {
            ne00: k as i32,
            ne01: m as i32,
            ne02: 1,
            nb00: std::mem::size_of::<BlockQ5_0>() as u64,
            nb01,
            nb02: nb01,
            nb03: nb01,
            ne10: k as i32,
            ne11: 1,
            ne12: 1,
            nb10: 4,
            nb11: k as u64 * 4,
            nb12: 0,
            nb13: 0,
            ne0: m as i32,
            ne1: 1,
            nr0: 4,
            r2: 1,
            r3: 1,
        };

        let weights_buffer = self.device.new_buffer_with_data(
            weights.as_ptr() as *const std::ffi::c_void,
            (weights.len() * std::mem::size_of::<BlockQ5_0>()) as u64,
            metal::MTLResourceOptions::StorageModeShared,
        );
        let input_buffer = self.device.new_buffer_with_data(
            input.as_ptr() as *const std::ffi::c_void,
            (input.len() * 4) as u64,
            metal::MTLResourceOptions::StorageModeShared,
        );
        let output_buffer = self.device.new_buffer(
            (m * 4) as u64,
            metal::MTLResourceOptions::StorageModeShared,
        );

        let command_buffer = self.queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(&self.mv_q5_0_pipeline);
        encoder.set_bytes(
            0,
            std::mem::size_of::<MvArgs>() as u64,
            &args as *const _ as *const std::ffi::c_void,
        );
        encoder.set_buffer(1, Some(&weights_buffer), 0);
        encoder.set_buffer(2, Some(&input_buffer), 0);
        encoder.set_buffer(3, Some(&output_buffer), 0);
        encoder.dispatch_thread_groups(
            metal::MTLSize::new(((m + 3) / 4) as u64, 1, 1),
            metal::MTLSize::new(32, 4, 1),
        );
        encoder.end_encoding();
        command_buffer.commit();
        command_buffer.wait_until_completed();

        Ok(unsafe {
            std::slice::from_raw_parts(
                output_buffer.contents() as *const f32,
                m,
            )
            .to_vec()
        })
    }

    /// Q5_0 x F32 Matrix-Vector (fully zero-copy, GPU buffers)
    pub fn mv_q5_0_f32_gpu_weights(
        &self,
        m: usize,
        k: usize,
        weights_buffer: &Buffer,
        input_buffer: &Buffer,
    ) -> Result<std::rc::Rc<Buffer>, String> {
        let nb = k / 32;
        let nb01 = (nb * std::mem::size_of::<BlockQ5_0>()) as u64;

        #[repr(C)]
        struct MvArgs {
            ne00: i32,
            ne01: i32,
            ne02: i32,
            nb00: u64,
            nb01: u64,
            nb02: u64,
            nb03: u64,
            ne10: i32,
            ne11: i32,
            ne12: i32,
            nb10: u64,
            nb11: u64,
            nb12: u64,
            nb13: u64,
            ne0: i32,
            ne1: i32,
            nr0: i32,
            r2: i16,
            r3: i16,
        }
        let args = MvArgs {
            ne00: k as i32,
            ne01: m as i32,
            ne02: 1,
            nb00: std::mem::size_of::<BlockQ5_0>() as u64,
            nb01,
            nb02: nb01,
            nb03: nb01,
            ne10: k as i32,
            ne11: 1,
            ne12: 1,
            nb10: 4,
            nb11: k as u64 * 4,
            nb12: 0,
            nb13: 0,
            ne0: m as i32,
            ne1: 1,
            nr0: 4,
            r2: 1,
            r3: 1,
        };

        let output_buffer = std::rc::Rc::new(self.device.new_buffer(
            (m * 4) as u64,
            metal::MTLResourceOptions::StorageModeShared,
        ));

        let command_buffer = self.queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(&self.mv_q5_0_pipeline);
        encoder.set_bytes(
            0,
            std::mem::size_of::<MvArgs>() as u64,
            &args as *const MvArgs as *const std::ffi::c_void,
        );
        encoder.set_buffer(1, Some(weights_buffer), 0);
        encoder.set_buffer(2, Some(input_buffer), 0);
        encoder.set_buffer(3, Some(&output_buffer), 0);

        let thread_group_size = MTLSize { width: 32, height: 2, depth: 1 };
        let thread_group_count =
            MTLSize { width: (m as u64 + 1) / 2, height: 1, depth: 1 };

        encoder.dispatch_thread_groups(thread_group_count, thread_group_size);
        encoder.end_encoding();
        command_buffer.commit();
        command_buffer.wait_until_completed();

        Ok(output_buffer)
    }

    /// Q5_1 x F32 Matrix-Vector
    pub fn mv_q5_1_f32(
        &self,
        m: usize,
        k: usize,
        weights: &[BlockQ5_1],
        input: &[f32],
    ) -> Result<Vec<f32>, String> {
        let nb = k / 32;
        let nb01 = (nb * std::mem::size_of::<BlockQ5_1>()) as u64;

        #[repr(C)]
        struct MvArgs {
            ne00: i32,
            ne01: i32,
            ne02: i32,
            nb00: u64,
            nb01: u64,
            nb02: u64,
            nb03: u64,
            ne10: i32,
            ne11: i32,
            ne12: i32,
            nb10: u64,
            nb11: u64,
            nb12: u64,
            nb13: u64,
            ne0: i32,
            ne1: i32,
            nr0: i32,
            r2: i16,
            r3: i16,
        }
        let args = MvArgs {
            ne00: k as i32,
            ne01: m as i32,
            ne02: 1,
            nb00: std::mem::size_of::<BlockQ5_1>() as u64,
            nb01,
            nb02: nb01,
            nb03: nb01,
            ne10: k as i32,
            ne11: 1,
            ne12: 1,
            nb10: 4,
            nb11: k as u64 * 4,
            nb12: 0,
            nb13: 0,
            ne0: m as i32,
            ne1: 1,
            nr0: 4,
            r2: 1,
            r3: 1,
        };

        let weights_buffer = self.device.new_buffer_with_data(
            weights.as_ptr() as *const std::ffi::c_void,
            (weights.len() * std::mem::size_of::<BlockQ5_1>()) as u64,
            metal::MTLResourceOptions::StorageModeShared,
        );
        let input_buffer = self.device.new_buffer_with_data(
            input.as_ptr() as *const std::ffi::c_void,
            (input.len() * 4) as u64,
            metal::MTLResourceOptions::StorageModeShared,
        );
        let output_buffer = self.device.new_buffer(
            (m * 4) as u64,
            metal::MTLResourceOptions::StorageModeShared,
        );

        let command_buffer = self.queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(&self.mv_q5_1_pipeline);
        encoder.set_bytes(
            0,
            std::mem::size_of::<MvArgs>() as u64,
            &args as *const _ as *const std::ffi::c_void,
        );
        encoder.set_buffer(1, Some(&weights_buffer), 0);
        encoder.set_buffer(2, Some(&input_buffer), 0);
        encoder.set_buffer(3, Some(&output_buffer), 0);
        encoder.dispatch_thread_groups(
            metal::MTLSize::new(((m + 3) / 4) as u64, 1, 1),
            metal::MTLSize::new(32, 4, 1),
        );
        encoder.end_encoding();
        command_buffer.commit();
        command_buffer.wait_until_completed();

        Ok(unsafe {
            std::slice::from_raw_parts(
                output_buffer.contents() as *const f32,
                m,
            )
            .to_vec()
        })
    }

    /// Q5_1 x F32 Matrix-Vector (fully zero-copy, GPU buffers)
    pub fn mv_q5_1_f32_gpu_weights(
        &self,
        m: usize,
        k: usize,
        weights_buffer: &Buffer,
        input_buffer: &Buffer,
    ) -> Result<std::rc::Rc<Buffer>, String> {
        let nb = k / 32;
        let nb01 = (nb * std::mem::size_of::<BlockQ5_1>()) as u64;

        #[repr(C)]
        struct MvArgs {
            ne00: i32,
            ne01: i32,
            ne02: i32,
            nb00: u64,
            nb01: u64,
            nb02: u64,
            nb03: u64,
            ne10: i32,
            ne11: i32,
            ne12: i32,
            nb10: u64,
            nb11: u64,
            nb12: u64,
            nb13: u64,
            ne0: i32,
            ne1: i32,
            nr0: i32,
            r2: i16,
            r3: i16,
        }
        let args = MvArgs {
            ne00: k as i32,
            ne01: m as i32,
            ne02: 1,
            nb00: std::mem::size_of::<BlockQ5_1>() as u64,
            nb01,
            nb02: nb01,
            nb03: nb01,
            ne10: k as i32,
            ne11: 1,
            ne12: 1,
            nb10: 4,
            nb11: k as u64 * 4,
            nb12: 0,
            nb13: 0,
            ne0: m as i32,
            ne1: 1,
            nr0: 4,
            r2: 1,
            r3: 1,
        };

        let output_buffer = std::rc::Rc::new(self.device.new_buffer(
            (m * 4) as u64,
            metal::MTLResourceOptions::StorageModeShared,
        ));

        let command_buffer = self.queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(&self.mv_q5_1_pipeline);
        encoder.set_bytes(
            0,
            std::mem::size_of::<MvArgs>() as u64,
            &args as *const MvArgs as *const std::ffi::c_void,
        );
        encoder.set_buffer(1, Some(weights_buffer), 0);
        encoder.set_buffer(2, Some(input_buffer), 0);
        encoder.set_buffer(3, Some(&output_buffer), 0);
        encoder.dispatch_thread_groups(
            metal::MTLSize::new(((m + 3) / 4) as u64, 1, 1),
            metal::MTLSize::new(32, 4, 1),
        );
        encoder.end_encoding();
        command_buffer.commit();
        command_buffer.wait_until_completed();

        Ok(output_buffer)
    }

    /// Q8_0 x F32 Matrix-Vector
    pub fn mv_q8_0_f32(
        &self,
        m: usize,
        k: usize,
        weights: &[BlockQ8_0],
        input: &[f32],
    ) -> Result<Vec<f32>, String> {
        let nb = k / 32;
        let nb01 = (nb * std::mem::size_of::<BlockQ8_0>()) as u64;

        #[repr(C)]
        struct MvArgs {
            ne00: i32,
            ne01: i32,
            ne02: i32,
            nb00: u64,
            nb01: u64,
            nb02: u64,
            nb03: u64,
            ne10: i32,
            ne11: i32,
            ne12: i32,
            nb10: u64,
            nb11: u64,
            nb12: u64,
            nb13: u64,
            ne0: i32,
            ne1: i32,
            nr0: i32,
            r2: i16,
            r3: i16,
        }
        let args = MvArgs {
            ne00: k as i32,
            ne01: m as i32,
            ne02: 1,
            nb00: std::mem::size_of::<BlockQ8_0>() as u64,
            nb01,
            nb02: nb01,
            nb03: nb01,
            ne10: k as i32,
            ne11: 1,
            ne12: 1,
            nb10: 4,
            nb11: k as u64 * 4,
            nb12: 0,
            nb13: 0,
            ne0: m as i32,
            ne1: 1,
            nr0: 2,
            r2: 1,
            r3: 1,
        };

        let weights_buffer = self.device.new_buffer_with_data(
            weights.as_ptr() as *const std::ffi::c_void,
            (weights.len() * std::mem::size_of::<BlockQ8_0>()) as u64,
            metal::MTLResourceOptions::StorageModeShared,
        );
        let input_buffer = self.device.new_buffer_with_data(
            input.as_ptr() as *const std::ffi::c_void,
            (input.len() * 4) as u64,
            metal::MTLResourceOptions::StorageModeShared,
        );
        let output_buffer = self.device.new_buffer(
            (m * 4) as u64,
            metal::MTLResourceOptions::StorageModeShared,
        );

        let command_buffer = self.queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(&self.mv_q8_0_pipeline);
        encoder.set_bytes(
            0,
            std::mem::size_of::<MvArgs>() as u64,
            &args as *const _ as *const std::ffi::c_void,
        );
        encoder.set_buffer(1, Some(&weights_buffer), 0);
        encoder.set_buffer(2, Some(&input_buffer), 0);
        encoder.set_buffer(3, Some(&output_buffer), 0);
        encoder.dispatch_thread_groups(
            metal::MTLSize::new(((m + 1) / 2) as u64, 1, 1),
            metal::MTLSize::new(32, 2, 1),
        );
        encoder.end_encoding();
        command_buffer.commit();
        command_buffer.wait_until_completed();

        Ok(unsafe {
            std::slice::from_raw_parts(
                output_buffer.contents() as *const f32,
                m,
            )
            .to_vec()
        })
    }

    /// Q8_0 x F32 Matrix-Vector (fully zero-copy, both weights and input as GPU buffers)
    pub fn mv_q8_0_f32_gpu_weights(
        &self,
        m: usize,
        k: usize,
        weights_buffer: &Buffer,
        input_buffer: &Buffer,
    ) -> Result<std::rc::Rc<Buffer>, String> {
        let nb = k / 32;
        let nb01 = (nb * std::mem::size_of::<BlockQ8_0>()) as u64;

        #[repr(C)]
        struct MvArgs {
            ne00: i32,
            ne01: i32,
            ne02: i32,
            nb00: u64,
            nb01: u64,
            nb02: u64,
            nb03: u64,
            ne10: i32,
            ne11: i32,
            ne12: i32,
            nb10: u64,
            nb11: u64,
            nb12: u64,
            nb13: u64,
            ne0: i32,
            ne1: i32,
            nr0: i32,
            r2: i16,
            r3: i16,
        }
        let args = MvArgs {
            ne00: k as i32,
            ne01: m as i32,
            ne02: 1,
            nb00: std::mem::size_of::<BlockQ8_0>() as u64,
            nb01,
            nb02: nb01,
            nb03: nb01,
            ne10: k as i32,
            ne11: 1,
            ne12: 1,
            nb10: 4,
            nb11: k as u64 * 4,
            nb12: 0,
            nb13: 0,
            ne0: m as i32,
            ne1: 1,
            nr0: 2,
            r2: 1,
            r3: 1,
        };

        let output_buffer = std::rc::Rc::new(self.device.new_buffer(
            (m * 4) as u64,
            metal::MTLResourceOptions::StorageModeShared,
        ));

        let command_buffer = self.queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(&self.mv_q8_0_pipeline);
        encoder.set_bytes(
            0,
            std::mem::size_of::<MvArgs>() as u64,
            &args as *const MvArgs as *const std::ffi::c_void,
        );
        encoder.set_buffer(1, Some(weights_buffer), 0);
        encoder.set_buffer(2, Some(input_buffer), 0);
        encoder.set_buffer(3, Some(&output_buffer), 0);

        let thread_group_size = MTLSize { width: 32, height: 2, depth: 1 };
        let thread_group_count =
            MTLSize { width: (m as u64 + 3) / 4, height: 1, depth: 1 };

        encoder.dispatch_thread_groups(thread_group_count, thread_group_size);
        encoder.end_encoding();
        command_buffer.commit();
        command_buffer.wait_until_completed();

        Ok(output_buffer)
    }

    pub fn mv_q4_k_f32(
        &self,
        m: usize,
        k: usize,
        weights: &[BlockQ4K],
        input: &[f32],
    ) -> Result<Vec<f32>, String> {
        let nb = k / 256; // QK_K = 256
        let nb01 = (nb * std::mem::size_of::<BlockQ4K>()) as u64; // Byte stride per row

        #[repr(C)]
        struct MvArgs {
            ne00: i32,
            ne01: i32,
            ne02: i32,
            nb00: u64,
            nb01: u64,
            nb02: u64,
            nb03: u64,
            ne10: i32,
            ne11: i32,
            ne12: i32,
            nb10: u64,
            nb11: u64,
            nb12: u64,
            nb13: u64,
            ne0: i32,
            ne1: i32,
            nr0: i32,
            r2: i16,
            r3: i16,
        }

        let args = MvArgs {
            ne00: k as i32,
            ne01: m as i32,
            ne02: 1,
            nb00: 1,
            nb01,
            nb02: 0,
            nb03: 0,
            ne10: k as i32,
            ne11: 1,
            ne12: 1,
            nb10: 4,
            nb11: k as u64 * 4,
            nb12: 0,
            nb13: 0,
            ne0: m as i32,
            ne1: 1,
            nr0: 2,
            r2: 1,
            r3: 1, // N_R0_Q4_K = 2
        };

        // Get or create cached buffers (NO data copy on repeated calls with same size)
        let weights_size = weights.len() * std::mem::size_of::<BlockQ4K>();
        let input_size = input.len() * std::mem::size_of::<f32>();
        let output_size = m * std::mem::size_of::<f32>();

        // Weights buffer - cache by data pointer (weights are static during inference)
        let weights_buffer = {
            let data_ptr = weights.as_ptr() as usize;
            let mut cache = self.weight_buffer_cache.borrow_mut();

            if let Some(buffer) = cache.get(&data_ptr) {
                // Buffer already exists for this tensor
                std::rc::Rc::clone(buffer)
            } else {
                // Create new buffer and cache it (zero-copy mapping)
                let buffer = std::rc::Rc::new(
                    self.device.new_buffer_with_bytes_no_copy(
                        weights.as_ptr() as *const std::ffi::c_void,
                        weights_size as u64,
                        metal::MTLResourceOptions::StorageModeShared,
                        None, // No deallocator - weights are static model data
                    ),
                );
                cache.insert(data_ptr, std::rc::Rc::clone(&buffer));
                buffer
            }
        };

        // Input buffer - data changes every layer, so always upload
        let input_buffer = {
            let mut buf_cell = self.mv_input_buffer.borrow_mut();
            let mut size_cell = self.mv_input_size.borrow_mut();

            if *size_cell != input_size {
                // Size changed, create new buffer with data
                let buf = self.device.new_buffer_with_data(
                    input.as_ptr() as *const std::ffi::c_void,
                    input_size as u64,
                    metal::MTLResourceOptions::StorageModeShared,
                );
                *buf_cell = Some(buf.clone());
                *size_cell = input_size;
                buf
            } else {
                // Size matches, reuse buffer but copy new data
                let buf = buf_cell.as_ref().unwrap().clone();
                let contents = buf.contents();
                unsafe {
                    std::ptr::copy_nonoverlapping(
                        input.as_ptr() as *const std::ffi::c_void,
                        contents,
                        input_size,
                    );
                }
                buf
            }
        };

        // Output buffer
        let output_buffer = {
            let mut buf_cell = self.mv_output_buffer.borrow_mut();
            let mut size_cell = self.mv_output_size.borrow_mut();

            if *size_cell != output_size {
                let buf = self.device.new_buffer(
                    output_size as u64,
                    metal::MTLResourceOptions::StorageModeShared,
                );
                *buf_cell = Some(buf.clone());
                *size_cell = output_size;
                buf
            } else {
                buf_cell.as_ref().unwrap().clone()
            }
        };

        // Use cached pipeline
        let command_buffer = self.queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();

        encoder.set_compute_pipeline_state(&self.mv_q4_k_pipeline);
        // Template kernel signature: (args, src0, src1, dst)
        // Index 0: args struct
        // Index 1: src0 (weights)
        // Index 2: src1 (input)
        // Index 3: dst (output)
        encoder.set_bytes(
            0,
            std::mem::size_of::<MvArgs>() as u64,
            &args as *const MvArgs as *const std::ffi::c_void,
        );
        encoder.set_buffer(1, Some(&weights_buffer), 0);
        encoder.set_buffer(2, Some(&input_buffer), 0);
        encoder.set_buffer(3, Some(&output_buffer), 0);

        // Q4_K: N_R0 = 2, NSG = 2
        const NSG: u64 = 2;
        const NR0: u64 = 2; // N_R0_Q4_K = 2
        const ROWS_PER_THREADGROUP: u64 = NSG * NR0; // 4

        let thread_group_size = MTLSize {
            width: 32,
            height: NSG, // 2D: Y = 2 for 2 SIMD groups
            depth: 1,
        };
        let thread_group_count = MTLSize {
            width: (m as u64 + ROWS_PER_THREADGROUP - 1) / ROWS_PER_THREADGROUP,
            height: 1,
            depth: 1,
        };

        encoder.dispatch_thread_groups(thread_group_count, thread_group_size);
        encoder.end_encoding();

        command_buffer.commit();
        command_buffer.wait_until_completed();

        let output_ptr = output_buffer.contents() as *const f32;
        Ok(unsafe { std::slice::from_raw_parts(output_ptr, m).to_vec() })
    }

    pub fn mv_q5_k_f32(
        &self,
        m: usize,
        k: usize,
        weights: &[BlockQ5K],
        input: &[f32],
    ) -> Result<Vec<f32>, String> {
        let nb = k / 256; // QK_K = 256
        let nb01 = (nb * std::mem::size_of::<BlockQ5K>()) as u64; // Byte stride per row

        // Simple args structure matching kernel's kquant_mv_args
        #[repr(C)]
        struct KquantMvArgs {
            ne00: u32, // K dimension
            ne01: u32, // M dimension
            nb01: u64, // Byte stride
        }

        let args = KquantMvArgs { ne00: k as u32, ne01: m as u32, nb01 };

        // Get or create cached buffers (NO data copy on repeated calls with same size)
        let weights_size = weights.len() * std::mem::size_of::<BlockQ5K>();
        let input_size = input.len() * std::mem::size_of::<f32>();
        let output_size = m * std::mem::size_of::<f32>();

        // Weights buffer - cache by data pointer (weights are static during inference)
        let weights_buffer = {
            let data_ptr = weights.as_ptr() as usize;
            let mut cache = self.weight_buffer_cache.borrow_mut();

            if let Some(buffer) = cache.get(&data_ptr) {
                // Buffer already exists for this tensor
                std::rc::Rc::clone(buffer)
            } else {
                // Create new buffer and cache it (zero-copy mapping)
                let buffer = std::rc::Rc::new(
                    self.device.new_buffer_with_bytes_no_copy(
                        weights.as_ptr() as *const std::ffi::c_void,
                        weights_size as u64,
                        metal::MTLResourceOptions::StorageModeShared,
                        None, // No deallocator - weights are static model data
                    ),
                );
                cache.insert(data_ptr, std::rc::Rc::clone(&buffer));
                buffer
            }
        };

        // Input buffer - data changes every layer, so always upload
        let input_buffer = {
            let mut buf_cell = self.mv_input_buffer.borrow_mut();
            let mut size_cell = self.mv_input_size.borrow_mut();

            if *size_cell != input_size {
                // Size changed, create new buffer with data
                let buf = self.device.new_buffer_with_data(
                    input.as_ptr() as *const std::ffi::c_void,
                    input_size as u64,
                    metal::MTLResourceOptions::StorageModeShared,
                );
                *buf_cell = Some(buf.clone());
                *size_cell = input_size;
                buf
            } else {
                // Size matches, reuse buffer but copy new data
                let buf = buf_cell.as_ref().unwrap().clone();
                let contents = buf.contents();
                unsafe {
                    std::ptr::copy_nonoverlapping(
                        input.as_ptr() as *const std::ffi::c_void,
                        contents,
                        input_size,
                    );
                }
                buf
            }
        };

        // Output buffer - create once, reuse
        let output_buffer = {
            let mut buf_cell = self.mv_output_buffer.borrow_mut();
            let mut size_cell = self.mv_output_size.borrow_mut();

            if *size_cell != output_size {
                let buf = self.device.new_buffer(
                    output_size as u64,
                    metal::MTLResourceOptions::StorageModeShared,
                );
                *buf_cell = Some(buf.clone());
                *size_cell = output_size;
                buf
            } else {
                buf_cell.as_ref().unwrap().clone()
            }
        };

        let command_buffer = self.queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();

        encoder.set_compute_pipeline_state(&self.mv_q5_k_pipeline);
        // Kernel signature: buffer(0)=weights, 1=input, 2=output, 3=args
        encoder.set_buffer(0, Some(&weights_buffer), 0);
        encoder.set_buffer(1, Some(&input_buffer), 0);
        encoder.set_buffer(2, Some(&output_buffer), 0);
        encoder.set_bytes(
            3,
            std::mem::size_of::<KquantMvArgs>() as u64,
            &args as *const KquantMvArgs as *const std::ffi::c_void,
        );

        // Thread group configuration - Simple version (1 row per threadgroup)
        // Step 1: Correctness first
        let thread_group_size = MTLSize {
            width: 32, // 1 SIMD group
            height: 1,
            depth: 1,
        };
        let thread_group_count = MTLSize {
            width: m as u64, // 1 threadgroup per row
            height: 1,
            depth: 1,
        };

        encoder.dispatch_thread_groups(thread_group_count, thread_group_size);
        encoder.end_encoding();

        command_buffer.commit();
        command_buffer.wait_until_completed();

        let output_ptr = output_buffer.contents() as *const f32;
        Ok(unsafe { std::slice::from_raw_parts(output_ptr, m).to_vec() })
    }

    pub fn mv_q2_k_f32(
        &self,
        m: usize,
        k: usize,
        weights: &[BlockQ2K],
        input: &[f32],
    ) -> Result<Vec<f32>, String> {
        let nb = k / 256; // QK_K = 256
        let nb01 = (nb * std::mem::size_of::<BlockQ2K>()) as u64; // Byte stride per row

        #[repr(C)]
        struct MvArgs {
            ne00: i32,
            ne01: i32,
            nb00: u64,
            nb01: u64,
            nb02: u64,
            nb03: u64,
            ne10: i32,
            ne11: i32,
            ne12: i32,
            nb10: u64,
            nb11: u64,
            nb12: u64,
            nb13: u64,
            ne0: i32,
            ne1: i32,
            nr0: i32,
            r2: i16,
            r3: i16,
        }

        let args = MvArgs {
            ne00: k as i32,
            ne01: m as i32,
            nb00: 1,
            nb01,
            nb02: 0,
            nb03: 0,
            ne10: k as i32,
            ne11: 1,
            ne12: 1,
            nb10: 4,
            nb11: k as u64 * 4,
            nb12: 0,
            nb13: 0,
            ne0: m as i32,
            ne1: 1,
            nr0: 4,
            r2: 1,
            r3: 1, // N_R0_Q2_K = 4
        };

        // Get or create cached buffers (NO data copy on repeated calls with same size)
        let weights_size = weights.len() * std::mem::size_of::<BlockQ2K>();
        let input_size = input.len() * std::mem::size_of::<f32>();
        let output_size = m * std::mem::size_of::<f32>();

        // Weights buffer - cache by data pointer (weights are static during inference)
        let weights_buffer = {
            let data_ptr = weights.as_ptr() as usize;
            let mut cache = self.weight_buffer_cache.borrow_mut();

            if let Some(buffer) = cache.get(&data_ptr) {
                // Buffer already exists for this tensor
                std::rc::Rc::clone(buffer)
            } else {
                // Create new buffer and cache it (zero-copy mapping)
                let buffer = std::rc::Rc::new(
                    self.device.new_buffer_with_bytes_no_copy(
                        weights.as_ptr() as *const std::ffi::c_void,
                        weights_size as u64,
                        metal::MTLResourceOptions::StorageModeShared,
                        None, // No deallocator - weights are static model data
                    ),
                );
                cache.insert(data_ptr, std::rc::Rc::clone(&buffer));
                buffer
            }
        };

        // Input buffer - data changes every layer, so always upload
        let input_buffer = {
            let mut buf_cell = self.mv_input_buffer.borrow_mut();
            let mut size_cell = self.mv_input_size.borrow_mut();

            if *size_cell != input_size {
                // Size changed, create new buffer with data
                let buf = self.device.new_buffer_with_data(
                    input.as_ptr() as *const std::ffi::c_void,
                    input_size as u64,
                    metal::MTLResourceOptions::StorageModeShared,
                );
                *buf_cell = Some(buf.clone());
                *size_cell = input_size;
                buf
            } else {
                // Size matches, reuse buffer but copy new data
                let buf = buf_cell.as_ref().unwrap().clone();
                let contents = buf.contents();
                unsafe {
                    std::ptr::copy_nonoverlapping(
                        input.as_ptr() as *const std::ffi::c_void,
                        contents,
                        input_size,
                    );
                }
                buf
            }
        };

        // Output buffer - create once, reuse
        let output_buffer = {
            let mut buf_cell = self.mv_output_buffer.borrow_mut();
            let mut size_cell = self.mv_output_size.borrow_mut();

            if *size_cell != output_size {
                let buf = self.device.new_buffer(
                    output_size as u64,
                    metal::MTLResourceOptions::StorageModeShared,
                );
                *buf_cell = Some(buf.clone());
                *size_cell = output_size;
                buf
            } else {
                buf_cell.as_ref().unwrap().clone()
            }
        };

        let command_buffer = self.queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();

        encoder.set_compute_pipeline_state(&self.mv_q2_k_pipeline);
        // Template kernel signature: (args, src0, src1, dst)
        // Index 0: args struct
        // Index 1: src0 (weights)
        // Index 2: src1 (input)
        // Index 3: dst (output)
        encoder.set_bytes(
            0,
            std::mem::size_of::<MvArgs>() as u64,
            &args as *const MvArgs as *const std::ffi::c_void,
        );
        encoder.set_buffer(1, Some(&weights_buffer), 0);
        encoder.set_buffer(2, Some(&input_buffer), 0);
        encoder.set_buffer(3, Some(&output_buffer), 0);

        //   threadgroup_count = ((ne01 + nr0*nsg - 1)/(nr0*nsg)), 1, 1
        //   threads_per_threadgroup = (32, nsg, 1)
        // Q2_K: N_R0 = 4, NSG = 2
        const NSG: u64 = 2;
        const NR0: u64 = 4; // N_R0_Q2_K = 4
        const ROWS_PER_THREADGROUP: u64 = NSG * NR0; // 8

        let thread_group_size = MTLSize {
            width: 32,
            height: NSG, // 2D: Y = 2 for 2 SIMD groups
            depth: 1,
        };
        let thread_group_count = MTLSize {
            width: (m as u64 + ROWS_PER_THREADGROUP - 1) / ROWS_PER_THREADGROUP,
            height: 1,
            depth: 1,
        };

        encoder.dispatch_thread_groups(thread_group_count, thread_group_size);
        encoder.end_encoding();

        command_buffer.commit();
        command_buffer.wait_until_completed();

        let output_ptr = output_buffer.contents() as *const f32;
        Ok(unsafe { std::slice::from_raw_parts(output_ptr, m).to_vec() })
    }

    pub fn mv_q6_k_f32(
        &self,
        m: usize,
        k: usize,
        weights: &[BlockQ6K],
        input: &[f32],
    ) -> Result<Vec<f32>, String> {
        let nb = k / 256; // QK_K = 256
        let nb01 = (nb * std::mem::size_of::<BlockQ6K>()) as u64; // Byte stride per row

        let args = KquantMvArgs::new(m, k, nb01);

        let weights_size = weights.len() * std::mem::size_of::<BlockQ6K>();
        let input_size = input.len() * std::mem::size_of::<f32>();
        let output_size = m * std::mem::size_of::<f32>();

        // Weights buffer - cache by data pointer (weights are static during inference)
        let weights_buffer = {
            let data_ptr = weights.as_ptr() as usize;
            let mut cache = self.weight_buffer_cache.borrow_mut();

            if let Some(buffer) = cache.get(&data_ptr) {
                // Buffer already exists for this tensor
                std::rc::Rc::clone(buffer)
            } else {
                // Create new buffer and cache it (zero-copy mapping)
                let buffer = std::rc::Rc::new(
                    self.device.new_buffer_with_bytes_no_copy(
                        weights.as_ptr() as *const std::ffi::c_void,
                        weights_size as u64,
                        metal::MTLResourceOptions::StorageModeShared,
                        None, // No deallocator - weights are static model data
                    ),
                );
                cache.insert(data_ptr, std::rc::Rc::clone(&buffer));
                buffer
            }
        };

        // Input buffer - data changes every layer, so always upload
        let input_buffer = {
            let mut buf_cell = self.mv_input_buffer.borrow_mut();
            let mut size_cell = self.mv_input_size.borrow_mut();

            if *size_cell != input_size {
                // Size changed, create new buffer with data
                let buf = self.device.new_buffer_with_data(
                    input.as_ptr() as *const std::ffi::c_void,
                    input_size as u64,
                    metal::MTLResourceOptions::StorageModeShared,
                );
                *buf_cell = Some(buf.clone());
                *size_cell = input_size;
                buf
            } else {
                // Size matches, reuse buffer but copy new data
                let buf = buf_cell.as_ref().unwrap().clone();
                let contents = buf.contents();
                unsafe {
                    std::ptr::copy_nonoverlapping(
                        input.as_ptr() as *const std::ffi::c_void,
                        contents,
                        input_size,
                    );
                }
                buf
            }
        };

        // Output buffer - create once, reuse
        let output_buffer = {
            let mut buf_cell = self.mv_output_buffer.borrow_mut();
            let mut size_cell = self.mv_output_size.borrow_mut();

            if *size_cell != output_size {
                let buf = self.device.new_buffer(
                    output_size as u64,
                    metal::MTLResourceOptions::StorageModeShared,
                );
                *buf_cell = Some(buf.clone());
                *size_cell = output_size;
                buf
            } else {
                buf_cell.as_ref().unwrap().clone()
            }
        };

        let command_buffer = self.queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();

        encoder.set_compute_pipeline_state(&self.mv_q6_k_pipeline);
        // Kernel signature: buffer(0)=weights, 1=input, 2=output, 3=args
        encoder.set_buffer(0, Some(&weights_buffer), 0);
        encoder.set_buffer(1, Some(&input_buffer), 0);
        encoder.set_buffer(2, Some(&output_buffer), 0);
        encoder.set_bytes(
            3,
            std::mem::size_of::<KquantMvArgs>() as u64,
            &args as *const KquantMvArgs as *const std::ffi::c_void,
        );

        // Thread group configuration - matches kernel expectations (NSG=1, N_R0=2)
        // 1 simdgroup per threadgroup, each processes 2 rows
        let thread_group_size = MTLSize {
            width: 32, // 1 SIMD group
            height: 1,
            depth: 1,
        };
        let thread_group_count = MTLSize {
            width: ((m + 1) / 2) as u64, // Ceiling division, 2 rows per threadgroup
            height: 1,
            depth: 1,
        };

        encoder.dispatch_thread_groups(thread_group_count, thread_group_size);
        encoder.end_encoding();

        command_buffer.commit();
        command_buffer.wait_until_completed();

        let output_ptr = output_buffer.contents() as *const f32;
        Ok(unsafe { std::slice::from_raw_parts(output_ptr, m).to_vec() })
    }

    pub fn mv_q3_k_f32(
        &self,
        m: usize,
        k: usize,
        weights: &[BlockQ3K],
        input: &[f32],
    ) -> Result<Vec<f32>, String> {
        let nb = k / 256;
        let nb01 = (nb * std::mem::size_of::<BlockQ3K>()) as u64;

        // Simple args structure matching kernel's kquant_mv_args
        #[repr(C)]
        struct KquantMvArgs {
            ne00: u32, // K dimension
            ne01: u32, // M dimension
            nb01: u64, // Byte stride
        }

        let args = KquantMvArgs { ne00: k as u32, ne01: m as u32, nb01 };

        let weights_size = weights.len() * std::mem::size_of::<BlockQ3K>();
        let input_size = input.len() * std::mem::size_of::<f32>();
        let output_size = m * std::mem::size_of::<f32>();

        // Weights buffer - cache by data pointer (weights are static during inference)
        let weights_buffer = {
            let data_ptr = weights.as_ptr() as usize;
            let mut cache = self.weight_buffer_cache.borrow_mut();

            if let Some(buffer) = cache.get(&data_ptr) {
                // Buffer already exists for this tensor
                std::rc::Rc::clone(buffer)
            } else {
                // Create new buffer and cache it (zero-copy mapping)
                let buffer = std::rc::Rc::new(
                    self.device.new_buffer_with_bytes_no_copy(
                        weights.as_ptr() as *const std::ffi::c_void,
                        weights_size as u64,
                        metal::MTLResourceOptions::StorageModeShared,
                        None, // No deallocator - weights are static model data
                    ),
                );
                cache.insert(data_ptr, std::rc::Rc::clone(&buffer));
                buffer
            }
        };

        // Input buffer - data changes every layer, so always upload
        let input_buffer = {
            let mut buf_cell = self.mv_input_buffer.borrow_mut();
            let mut size_cell = self.mv_input_size.borrow_mut();

            if *size_cell != input_size {
                // Size changed, create new buffer with data
                let buf = self.device.new_buffer_with_data(
                    input.as_ptr() as *const std::ffi::c_void,
                    input_size as u64,
                    metal::MTLResourceOptions::StorageModeShared,
                );
                *buf_cell = Some(buf.clone());
                *size_cell = input_size;
                buf
            } else {
                // Size matches, reuse buffer but copy new data
                let buf = buf_cell.as_ref().unwrap().clone();
                let contents = buf.contents();
                unsafe {
                    std::ptr::copy_nonoverlapping(
                        input.as_ptr() as *const std::ffi::c_void,
                        contents,
                        input_size,
                    );
                }
                buf
            }
        };

        // Output buffer - create once, reuse
        let output_buffer = {
            let mut buf_cell = self.mv_output_buffer.borrow_mut();
            let mut size_cell = self.mv_output_size.borrow_mut();

            if *size_cell != output_size {
                let buf = self.device.new_buffer(
                    output_size as u64,
                    metal::MTLResourceOptions::StorageModeShared,
                );
                *buf_cell = Some(buf.clone());
                *size_cell = output_size;
                buf
            } else {
                buf_cell.as_ref().unwrap().clone()
            }
        };

        let command_buffer = self.queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();

        encoder.set_compute_pipeline_state(&self.mv_q3_k_pipeline);
        // Kernel signature: buffer(0)=weights, 1=input, 2=output, 3=args
        encoder.set_buffer(0, Some(&weights_buffer), 0);
        encoder.set_buffer(1, Some(&input_buffer), 0);
        encoder.set_buffer(2, Some(&output_buffer), 0);
        encoder.set_bytes(
            3,
            std::mem::size_of::<KquantMvArgs>() as u64,
            &args as *const KquantMvArgs as *const std::ffi::c_void,
        );

        // Thread group configuration - Simple version (1 row per threadgroup)
        let thread_group_size = MTLSize {
            width: 32, // 1 SIMD group
            height: 1,
            depth: 1,
        };
        let thread_group_count = MTLSize {
            width: m as u64, // 1 threadgroup per row
            height: 1,
            depth: 1,
        };

        encoder.dispatch_thread_groups(thread_group_count, thread_group_size);
        encoder.end_encoding();

        command_buffer.commit();
        command_buffer.wait_until_completed();

        let output_ptr = output_buffer.contents() as *const f32;
        Ok(unsafe { std::slice::from_raw_parts(output_ptr, m).to_vec() })
    }

    /// Q4_K x F32 Matrix-Vector (fully zero-copy, both weights and input as GPU buffers)
    pub fn mv_q4_k_f32_gpu_weights(
        &self,
        m: usize,
        k: usize,
        weights_buffer: &Buffer,
        input_buffer: &Buffer,
    ) -> Result<std::rc::Rc<Buffer>, String> {
        let nb = k / 256; // QK_K = 256
        let nb01 = (nb * std::mem::size_of::<BlockQ4K>()) as u64;
        let args = KquantMvArgs::new(m, k, nb01);

        let output_buffer = std::rc::Rc::new(self.device.new_buffer(
            (m * 4) as u64,
            metal::MTLResourceOptions::StorageModeShared,
        ));

        let command_buffer = self.queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(&self.mv_q4_k_pipeline);
        // Kernel expects: buffer 0=weights, 1=input, 2=output, 3=args
        encoder.set_buffer(0, Some(weights_buffer), 0);
        encoder.set_buffer(1, Some(input_buffer), 0);
        encoder.set_buffer(2, Some(&output_buffer), 0);
        encoder.set_bytes(
            3,
            std::mem::size_of::<KquantMvArgs>() as u64,
            &args as *const KquantMvArgs as *const std::ffi::c_void,
        );

        // 2D dispatch for multiple simdgroups per threadgroup
        // Q4_K: NSG=2, N_R0=2 (2 simdgroups, each processes 2 rows)
        let nsg: u64 = 2;
        let nr0: u64 = 2;
        let tgs = (m as u64 + nsg * nr0 - 1) / (nsg * nr0); // threadgroups needed

        let thread_group_size = MTLSize { width: 32, height: nsg, depth: 1 };
        let thread_group_count = MTLSize { width: tgs, height: 1, depth: 1 };

        encoder.dispatch_thread_groups(thread_group_count, thread_group_size);
        encoder.end_encoding();
        command_buffer.commit();
        command_buffer.wait_until_completed();

        Ok(output_buffer)
    }

    /// Q5_K x F32 Matrix-Vector (fully zero-copy, both weights and input as GPU buffers)
    pub fn mv_q5_k_f32_gpu_weights(
        &self,
        m: usize,
        k: usize,
        weights_buffer: &Buffer,
        input_buffer: &Buffer,
    ) -> Result<std::rc::Rc<Buffer>, String> {
        let nb = k / 256;
        let nb01 = (nb * std::mem::size_of::<BlockQ5K>()) as u64;
        let args = KquantMvArgs::new(m, k, nb01);

        let output_buffer = std::rc::Rc::new(self.device.new_buffer(
            (m * 4) as u64,
            metal::MTLResourceOptions::StorageModeShared,
        ));

        let command_buffer = self.queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(&self.mv_q5_k_pipeline);
        // Kernel expects: buffer 0=weights, 1=input, 2=output, 3=args
        encoder.set_buffer(0, Some(weights_buffer), 0);
        encoder.set_buffer(1, Some(input_buffer), 0);
        encoder.set_buffer(2, Some(&output_buffer), 0);
        encoder.set_bytes(
            3,
            std::mem::size_of::<KquantMvArgs>() as u64,
            &args as *const KquantMvArgs as *const std::ffi::c_void,
        );

        // 2D dispatch for multiple simdgroups per threadgroup
        // Q5_K: NSG=2, N_R0=1 (2 simdgroups, each processes 1 row)
        let nsg: u64 = 2;
        let nr0: u64 = 1;
        let tgs = (m as u64 + nsg * nr0 - 1) / (nsg * nr0); // threadgroups needed

        let thread_group_size = MTLSize { width: 32, height: nsg, depth: 1 };
        let thread_group_count = MTLSize { width: tgs, height: 1, depth: 1 };

        encoder.dispatch_thread_groups(thread_group_count, thread_group_size);
        encoder.end_encoding();
        command_buffer.commit();
        command_buffer.wait_until_completed();

        Ok(output_buffer)
    }

    /// Q2_K x F32 Matrix-Vector (fully zero-copy, both weights and input as GPU buffers)
    pub fn mv_q2_k_f32_gpu_weights(
        &self,
        m: usize,
        k: usize,
        weights_buffer: &Buffer,
        input_buffer: &Buffer,
    ) -> Result<std::rc::Rc<Buffer>, String> {
        let nb = k / 256;
        let nb01 = (nb * std::mem::size_of::<BlockQ2K>()) as u64;
        let args = KquantMvArgs::new(m, k, nb01);

        let output_buffer = std::rc::Rc::new(self.device.new_buffer(
            (m * 4) as u64,
            metal::MTLResourceOptions::StorageModeShared,
        ));

        let command_buffer = self.queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(&self.mv_q2_k_pipeline);
        // Kernel expects: buffer 0=weights, 1=input, 2=output, 3=args
        encoder.set_buffer(0, Some(weights_buffer), 0);
        encoder.set_buffer(1, Some(input_buffer), 0);
        encoder.set_buffer(2, Some(&output_buffer), 0);
        encoder.set_bytes(
            3,
            std::mem::size_of::<KquantMvArgs>() as u64,
            &args as *const KquantMvArgs as *const std::ffi::c_void,
        );

        // 2D dispatch for multiple simdgroups per threadgroup
        // Q2_K: NSG=2, N_R0=4 (2 simdgroups, each processes 4 rows)
        let nsg: u64 = 2;
        let nr0: u64 = 4;
        let tgs = (m as u64 + nsg * nr0 - 1) / (nsg * nr0); // threadgroups needed

        let thread_group_size = MTLSize { width: 32, height: nsg, depth: 1 };
        let thread_group_count = MTLSize { width: tgs, height: 1, depth: 1 };

        encoder.dispatch_thread_groups(thread_group_count, thread_group_size);
        encoder.end_encoding();
        command_buffer.commit();
        command_buffer.wait_until_completed();

        Ok(output_buffer)
    }

    /// Q6_K x F32 Matrix-Vector (fully zero-copy, both weights and input as GPU buffers)
    pub fn mv_q6_k_f32_gpu_weights(
        &self,
        m: usize,
        k: usize,
        weights_buffer: &Buffer,
        input_buffer: &Buffer,
    ) -> Result<std::rc::Rc<Buffer>, String> {
        let nb = k / 256;
        let nb01 = (nb * std::mem::size_of::<BlockQ6K>()) as u64;
        let args = KquantMvArgs::new(m, k, nb01);

        let output_buffer = std::rc::Rc::new(self.device.new_buffer(
            (m * 4) as u64,
            metal::MTLResourceOptions::StorageModeShared,
        ));

        let command_buffer = self.queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(&self.mv_q6_k_pipeline);
        // Kernel expects: buffer 0=weights, 1=input, 2=output, 3=args
        encoder.set_buffer(0, Some(weights_buffer), 0);
        encoder.set_buffer(1, Some(input_buffer), 0);
        encoder.set_buffer(2, Some(&output_buffer), 0);
        encoder.set_bytes(
            3,
            std::mem::size_of::<KquantMvArgs>() as u64,
            &args as *const KquantMvArgs as *const std::ffi::c_void,
        );

        // 2D dispatch for multiple simdgroups per threadgroup
        // Q6_K: NSG=2, N_R0=2 (2 simdgroups, each processes 2 rows)
        let nsg: u64 = 2;
        let nr0: u64 = 2;
        let tgs = (m as u64 + nsg * nr0 - 1) / (nsg * nr0); // threadgroups needed

        let thread_group_size = MTLSize { width: 32, height: nsg, depth: 1 };
        let thread_group_count = MTLSize { width: tgs, height: 1, depth: 1 };

        encoder.dispatch_thread_groups(thread_group_count, thread_group_size);
        encoder.end_encoding();
        command_buffer.commit();
        command_buffer.wait_until_completed();

        Ok(output_buffer)
    }

    /// Q3_K x F32 Matrix-Vector (fully zero-copy, both weights and input as GPU buffers)
    pub fn mv_q3_k_f32_gpu_weights(
        &self,
        m: usize,
        k: usize,
        weights_buffer: &Buffer,
        input_buffer: &Buffer,
    ) -> Result<std::rc::Rc<Buffer>, String> {
        let nb = k / 256;
        let nb01 = (nb * std::mem::size_of::<BlockQ3K>()) as u64;
        let args = KquantMvArgs::new(m, k, nb01);

        let output_buffer = std::rc::Rc::new(self.device.new_buffer(
            (m * 4) as u64,
            metal::MTLResourceOptions::StorageModeShared,
        ));

        let command_buffer = self.queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(&self.mv_q3_k_pipeline);
        // Kernel expects: buffer 0=weights, 1=input, 2=output, 3=args
        encoder.set_buffer(0, Some(weights_buffer), 0);
        encoder.set_buffer(1, Some(input_buffer), 0);
        encoder.set_buffer(2, Some(&output_buffer), 0);
        encoder.set_bytes(
            3,
            std::mem::size_of::<KquantMvArgs>() as u64,
            &args as *const KquantMvArgs as *const std::ffi::c_void,
        );

        // 2D dispatch for multiple simdgroups per threadgroup
        // Q3_K: NSG=2, N_R0=2 (2 simdgroups, each processes 2 rows)
        let nsg: u64 = 2;
        let nr0: u64 = 2;
        let tgs = (m as u64 + nsg * nr0 - 1) / (nsg * nr0); // threadgroups needed

        let thread_group_size = MTLSize { width: 32, height: nsg, depth: 1 };
        let thread_group_count = MTLSize { width: tgs, height: 1, depth: 1 };

        encoder.dispatch_thread_groups(thread_group_count, thread_group_size);
        encoder.end_encoding();
        command_buffer.commit();
        command_buffer.wait_until_completed();

        Ok(output_buffer)
    }

    /// Q6_K transposed Matrix-Vector: y = W^T * x
    /// W is stored as [K, M] instead of [M, K] (for tied embeddings)
    /// K = input_size (hidden_size), M = output_size (vocab_size)
    pub fn mv_q6_k_f32_transposed(
        &self,
        m: usize, // Output size (vocab_size)
        k: usize, // Input size (hidden_size)
        weights: &[BlockQ6K],
        input: &[f32],
    ) -> Result<Vec<f32>, String> {
        let weights_size = weights.len() * std::mem::size_of::<BlockQ6K>();
        let input_size = input.len() * std::mem::size_of::<f32>();
        let output_size = m * std::mem::size_of::<f32>();

        let weights_buffer = self.device.new_buffer_with_data(
            weights.as_ptr() as *const std::ffi::c_void,
            weights_size as u64,
            metal::MTLResourceOptions::StorageModeShared,
        );
        let input_buffer = self.device.new_buffer_with_data(
            input.as_ptr() as *const std::ffi::c_void,
            input_size as u64,
            metal::MTLResourceOptions::StorageModeShared,
        );
        let output_buffer = self.device.new_buffer(
            output_size as u64,
            metal::MTLResourceOptions::StorageModeShared,
        );

        let command_buffer = self.queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(&self.mv_q6_k_transposed_pipeline);

        // Kernel expects: buffer 0=weights, 1=input, 2=output, 3=m, 4=k
        encoder.set_buffer(0, Some(&weights_buffer), 0);
        encoder.set_buffer(1, Some(&input_buffer), 0);
        encoder.set_buffer(2, Some(&output_buffer), 0);
        encoder.set_bytes(3, 4, &m as *const usize as *const std::ffi::c_void);
        encoder.set_bytes(4, 4, &k as *const usize as *const std::ffi::c_void);

        // One thread per output element
        let thread_group_size = MTLSize { width: 256, height: 1, depth: 1 };
        let thread_group_count =
            MTLSize { width: ((m + 255) / 256) as u64, height: 1, depth: 1 };

        encoder.dispatch_thread_groups(thread_group_count, thread_group_size);
        encoder.end_encoding();
        command_buffer.commit();
        command_buffer.wait_until_completed();

        // Read output
        let output_ptr = output_buffer.contents() as *const f32;
        let output =
            unsafe { std::slice::from_raw_parts(output_ptr, m).to_vec() };

        Ok(output)
    }

    /// Q6_K transposed Matrix-Vector with GPU buffers (zero-copy)
    /// W is stored as [K, M] instead of [M, K] (for tied embeddings)
    pub fn mv_q6_k_f32_transposed_gpu_weights(
        &self,
        m: usize, // Output size (vocab_size)
        k: usize, // Input size (hidden_size)
        weights_buffer: &Buffer,
        input_buffer: &Buffer,
    ) -> Result<std::rc::Rc<Buffer>, String> {
        let output_size = m * std::mem::size_of::<f32>();
        let output_buffer = std::rc::Rc::new(self.device.new_buffer(
            output_size as u64,
            metal::MTLResourceOptions::StorageModeShared,
        ));

        let command_buffer = self.queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(&self.mv_q6_k_transposed_pipeline);

        // Kernel expects: buffer 0=weights, 1=input, 2=output, 3=m, 4=k
        encoder.set_buffer(0, Some(weights_buffer), 0);
        encoder.set_buffer(1, Some(input_buffer), 0);
        encoder.set_buffer(2, Some(&output_buffer), 0);
        encoder.set_bytes(3, 4, &m as *const usize as *const std::ffi::c_void);
        encoder.set_bytes(4, 4, &k as *const usize as *const std::ffi::c_void);

        // One thread per output element
        let thread_group_size = MTLSize { width: 256, height: 1, depth: 1 };
        let thread_group_count =
            MTLSize { width: ((m + 255) / 256) as u64, height: 1, depth: 1 };

        encoder.dispatch_thread_groups(thread_group_count, thread_group_size);
        encoder.end_encoding();
        command_buffer.commit();
        command_buffer.wait_until_completed();

        Ok(output_buffer)
    }

    /// F16 x F32 Matrix-Vector Multiplication
    /// For transposed output layer with F16 weights
    /// weights: [M, K] in F16 format (row-major)
    /// input: [K] in F32 format
    /// Returns: [M] in F32 format
    pub fn mv_f16_f32_gpu_weights(
        &self,
        m: usize,
        k: usize,
        weights_buffer: &Buffer,
        input_buffer: &Buffer,
    ) -> Result<std::rc::Rc<Buffer>, String> {
        let output_size = m * std::mem::size_of::<f32>();
        let output_buffer = std::rc::Rc::new(self.device.new_buffer(
            output_size as u64,
            metal::MTLResourceOptions::StorageModeShared,
        ));

        let command_buffer = self.queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();

        // Parameter struct matching mul_mv_args
        #[repr(C)]
        struct MvArgs {
            ne00: i32, // K
            ne01: i32, // M (rows)
            ne02: i32,
            nb00: u64,
            nb01: u64, // row stride
            nb02: u64,
            nb03: u64,
            ne10: i32,
            ne11: i32,
            ne12: i32,
            nb10: u64,
            nb11: u64,
            nb12: u64,
            nb13: u64,
            ne0: i32,
            ne1: i32,
            nr0: i32, // NR0 (2 for normal kernels, 32 for short)
            r2: i16,
            r3: i16,
            nsg: i16,        // Number of simdgroups
            ne12_batch: i16, // batch dimension
        }

        if k < 32 {
            // Short kernel: nsg=1, nr0=32, each thread processes one row
            encoder.set_compute_pipeline_state(&self.mv_f16_short_pipeline);

            let args = MvArgs {
                ne00: k as i32,
                ne01: m as i32,
                ne02: 1,
                nb00: 2, // sizeof(half)
                nb01: (k * 2) as u64,
                nb02: 0,
                nb03: 0,
                ne10: k as i32,
                ne11: 1,
                ne12: 1,
                nb10: 4, // sizeof(float)
                nb11: (k * 4) as u64,
                nb12: 0,
                nb13: 0,
                ne0: m as i32,
                ne1: 1,
                nr0: 32,
                r2: 1,
                r3: 1,
                nsg: 1, // NSG=1 for short kernel
                ne12_batch: 1,
            };

            let thread_group_size = MTLSize { width: 32, height: 1, depth: 1 };
            let thread_group_count =
                MTLSize { width: ((m + 31) / 32) as u64, height: 1, depth: 1 };

            encoder.set_bytes(
                0,
                std::mem::size_of::<MvArgs>() as u64,
                &args as *const MvArgs as *const std::ffi::c_void,
            );
            encoder.set_buffer(1, Some(weights_buffer), 0);
            encoder.set_buffer(2, Some(input_buffer), 0);
            encoder.set_buffer(3, Some(&output_buffer), 0);
            encoder
                .dispatch_thread_groups(thread_group_count, thread_group_size);
        } else {
            // Use optimized kernel with nxpsg support (matches llama.cpp)
            // Calculate nxpsg based on K value
            let nxpsg: i16 = if k % 256 == 0 {
                16 // K divisible by 256
            } else if k % 128 == 0 {
                8 // K divisible by 128
            } else {
                4 // Default
            };

            // NSG = number of simdgroups
            let nsg: i16 = std::cmp::min(4, (k + 127) / 128) as i16;

            // r1ptg = rows per threadgroup in batch dimension (batch=1 for decode)
            let r1ptg: i16 = 1;

            // Use optimized pipeline with correct nxpsg
            // For now, use the nxpsg=16 pipeline (we need to create all variants)
            encoder.set_compute_pipeline_state(
                &self.mv_f16_opt_r1_1_nxpsg16_pipeline,
            );

            // Use mul_mv_ext_args structure matching optimized kernel
            #[repr(C)]
            struct ExtArgs {
                ne00: i32,
                ne01: i32,
                ne02: i32,
                nb00: u64,
                nb01: u64,
                nb02: u64,
                nb03: u64,
                ne10: i32,
                ne11: i32,
                ne12: i32,
                nb10: u64,
                nb11: u64,
                nb12: u64,
                nb13: u64,
                ne0: i32,
                ne1: i32,
            }

            let args = ExtArgs {
                ne00: k as i32,
                ne01: m as i32,
                ne02: 1,
                nb00: 2, // sizeof(half)
                nb01: (k * 2) as u64,
                nb02: 0,
                nb03: 0,
                ne10: k as i32,
                ne11: 1, // batch size
                ne12: 1,
                nb10: 4, // sizeof(float)
                nb11: (k * 4) as u64,
                nb12: 0,
                nb13: 0,
                ne0: m as i32,
                ne1: 1,
            };

            encoder.set_bytes(
                0,
                std::mem::size_of::<ExtArgs>() as u64,
                &args as *const ExtArgs as *const std::ffi::c_void,
            );
            encoder.set_buffer(1, Some(weights_buffer), 0);
            encoder.set_buffer(2, Some(input_buffer), 0);
            encoder.set_buffer(3, Some(&output_buffer), 0);

            // Threadgroup size: 32 threads (one simdgroup)
            // Number of simdgroups scales with nxpsg and nypsg
            let nypsg = 32 / nxpsg as u64; // threads per simdgroup in Y dimension
            let thread_group_size = MTLSize { width: 32, height: 1, depth: 1 };
            let thread_group_count = MTLSize {
                width: ((m + nypsg as usize * nsg as usize - 1)
                    / (nypsg as usize * nsg as usize))
                    as u64,
                height: 1,
                depth: 1,
            };

            encoder
                .dispatch_thread_groups(thread_group_count, thread_group_size);
        }

        encoder.end_encoding();
        command_buffer.commit();
        command_buffer.wait_until_completed();

        Ok(output_buffer)
    }

    pub fn mv_iq4_nl_f32(
        &self,
        m: usize,
        k: usize,
        weights: &[BlockIQ4NL],
        input: &[f32],
    ) -> Result<Vec<f32>, String> {
        let nb = k / 32; // QK4_NL = 32
        let nb01 = (nb * std::mem::size_of::<BlockIQ4NL>()) as u64;
        const NR0: u64 = 2; // Keep at 2 (optimal)

        #[repr(C)]
        struct MvArgs {
            ne00: u32,
            ne01: u32,
            nb01: u64,
        }

        let args = MvArgs { ne00: k as u32, ne01: m as u32, nb01 };

        let weights_size = weights.len() * std::mem::size_of::<BlockIQ4NL>();
        let input_size = input.len() * std::mem::size_of::<f32>();
        let output_size = m * std::mem::size_of::<f32>();

        let weights_buffer = {
            let mut buf_cell = self.mv_weights_buffer.borrow_mut();
            let mut size_cell = self.mv_weights_size.borrow_mut();

            if *size_cell != weights_size {
                let buf = self.device.new_buffer_with_data(
                    weights.as_ptr() as *const std::ffi::c_void,
                    weights_size as u64,
                    metal::MTLResourceOptions::StorageModeShared,
                );
                *buf_cell = Some(buf.clone());
                *size_cell = weights_size;
                buf
            } else {
                buf_cell.as_ref().unwrap().clone()
            }
        };

        let input_buffer = {
            let mut buf_cell = self.mv_input_buffer.borrow_mut();
            let mut size_cell = self.mv_input_size.borrow_mut();

            if *size_cell != input_size {
                let buf = self.device.new_buffer_with_data(
                    input.as_ptr() as *const std::ffi::c_void,
                    input_size as u64,
                    metal::MTLResourceOptions::StorageModeShared,
                );
                *buf_cell = Some(buf.clone());
                *size_cell = input_size;
                buf
            } else {
                buf_cell.as_ref().unwrap().clone()
            }
        };

        let output_buffer = {
            let mut buf_cell = self.mv_output_buffer.borrow_mut();
            let mut size_cell = self.mv_output_size.borrow_mut();

            if *size_cell != output_size {
                let buf = self.device.new_buffer(
                    output_size as u64,
                    metal::MTLResourceOptions::StorageModeShared,
                );
                *buf_cell = Some(buf.clone());
                *size_cell = output_size;
                buf
            } else {
                buf_cell.as_ref().unwrap().clone()
            }
        };

        let command_buffer = self.queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();

        encoder.set_compute_pipeline_state(&self.mv_iq4_nl_pipeline);
        encoder.set_buffer(0, Some(&weights_buffer), 0);
        encoder.set_buffer(1, Some(&input_buffer), 0);
        encoder.set_buffer(2, Some(&output_buffer), 0);
        encoder.set_bytes(
            3,
            std::mem::size_of::<MvArgs>() as u64,
            &args as *const MvArgs as *const std::ffi::c_void,
        );

        // Set threadgroup memory: 32 floats for kvalues lookup table
        // Round up to multiple of 16 as required by Metal
        const THREADGROUP_MEM_SIZE: u64 =
            32 * std::mem::size_of::<f32>() as u64; // 128 bytes
        encoder.set_threadgroup_memory_length(THREADGROUP_MEM_SIZE, 0);

        const NSG: u64 = 2;
        const ROWS_PER_THREADGROUP: u64 = NSG * NR0;

        let thread_group_size = MTLSize { width: 32, height: NSG, depth: 1 };
        let thread_group_count = MTLSize {
            width: (m as u64 + ROWS_PER_THREADGROUP - 1) / ROWS_PER_THREADGROUP,
            height: 1,
            depth: 1,
        };

        encoder.dispatch_thread_groups(thread_group_count, thread_group_size);
        encoder.end_encoding();

        command_buffer.commit();
        command_buffer.wait_until_completed();

        let output_ptr = output_buffer.contents() as *const f32;
        Ok(unsafe { std::slice::from_raw_parts(output_ptr, m).to_vec() })
    }

    /// IQ4_NL x F32 Matrix-Vector (fully zero-copy, GPU buffers)
    pub fn mv_iq4_nl_f32_gpu_weights(
        &self,
        m: usize,
        k: usize,
        weights_buffer: &Buffer,
        input_buffer: &Buffer,
    ) -> Result<std::rc::Rc<Buffer>, String> {
        let nb = k / 32;
        let nb01 = (nb * std::mem::size_of::<BlockIQ4NL>()) as u64;
        const NR0: u64 = 2;

        #[repr(C)]
        struct MvArgs {
            ne00: u32,
            ne01: u32,
            nb01: u64,
        }

        let args = MvArgs { ne00: k as u32, ne01: m as u32, nb01 };

        let output_buffer = std::rc::Rc::new(self.device.new_buffer(
            (m * 4) as u64,
            metal::MTLResourceOptions::StorageModeShared,
        ));

        let command_buffer = self.queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();

        encoder.set_compute_pipeline_state(&self.mv_iq4_nl_pipeline);
        encoder.set_buffer(0, Some(weights_buffer), 0);
        encoder.set_buffer(1, Some(input_buffer), 0);
        encoder.set_buffer(2, Some(&output_buffer), 0);
        encoder.set_bytes(
            3,
            std::mem::size_of::<MvArgs>() as u64,
            &args as *const MvArgs as *const std::ffi::c_void,
        );

        const THREADGROUP_MEM_SIZE: u64 =
            32 * std::mem::size_of::<f32>() as u64;
        encoder.set_threadgroup_memory_length(THREADGROUP_MEM_SIZE, 0);

        const NSG: u64 = 2;
        const ROWS_PER_THREADGROUP: u64 = NSG * NR0;

        let thread_group_size = MTLSize { width: 32, height: NSG, depth: 1 };
        let thread_group_count = MTLSize {
            width: (m as u64 + ROWS_PER_THREADGROUP - 1) / ROWS_PER_THREADGROUP,
            height: 1,
            depth: 1,
        };

        encoder.dispatch_thread_groups(thread_group_count, thread_group_size);
        encoder.end_encoding();
        command_buffer.commit();
        command_buffer.wait_until_completed();

        Ok(output_buffer)
    }

    /// IQ4_XS x F32 Matrix-Vector (fully zero-copy, both weights and input as GPU buffers)
    pub fn mv_iq4_xs_f32_gpu_weights(
        &self,
        m: usize,
        k: usize,
        weights_buffer: &Buffer,
        input_buffer: &Buffer,
    ) -> Result<std::rc::Rc<Buffer>, String> {
        let nb = k / 256; // QK_K = 256
        let nb01 = (nb * std::mem::size_of::<BlockIQ4XS>()) as u64;
        const NR0: u64 = 2;

        #[repr(C)]
        struct MvArgs {
            ne00: u32,
            ne01: u32,
            nb01: u64,
        }

        let args = MvArgs { ne00: k as u32, ne01: m as u32, nb01 };

        let output_buffer = std::rc::Rc::new(self.device.new_buffer(
            (m * 4) as u64,
            metal::MTLResourceOptions::StorageModeShared,
        ));

        let command_buffer = self.queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();

        encoder.set_compute_pipeline_state(&self.mv_iq4_xs_pipeline);
        encoder.set_buffer(0, Some(weights_buffer), 0);
        encoder.set_buffer(1, Some(input_buffer), 0);
        encoder.set_buffer(2, Some(&output_buffer), 0);
        encoder.set_bytes(
            3,
            std::mem::size_of::<MvArgs>() as u64,
            &args as *const MvArgs as *const std::ffi::c_void,
        );

        const THREADGROUP_MEM_SIZE: u64 =
            32 * std::mem::size_of::<f32>() as u64;
        encoder.set_threadgroup_memory_length(THREADGROUP_MEM_SIZE, 0);

        const NSG: u64 = 2;
        const ROWS_PER_THREADGROUP: u64 = NSG * NR0;

        let thread_group_size = MTLSize { width: 32, height: NSG, depth: 1 };
        let thread_group_count = MTLSize {
            width: (m as u64 + ROWS_PER_THREADGROUP - 1) / ROWS_PER_THREADGROUP,
            height: 1,
            depth: 1,
        };

        encoder.dispatch_thread_groups(thread_group_count, thread_group_size);
        encoder.end_encoding();
        command_buffer.commit();
        command_buffer.wait_until_completed();

        Ok(output_buffer)
    }

    /// IQ1_S x F32 Matrix-Vector (fully zero-copy, both weights and input as GPU buffers)
    pub fn mv_iq1_s_f32_gpu_weights(
        &self,
        m: usize,
        k: usize,
        weights_buffer: &Buffer,
        input_buffer: &Buffer,
    ) -> Result<std::rc::Rc<Buffer>, String> {
        let nb = k / 256;
        let nb01 = (nb * std::mem::size_of::<BlockIQ1S>()) as u64;
        const NR0: u64 = 2;

        #[repr(C)]
        struct MvArgs {
            ne00: u32,
            ne01: u32,
            nb01: u64,
        }

        let args = MvArgs { ne00: k as u32, ne01: m as u32, nb01 };

        let output_buffer = std::rc::Rc::new(self.device.new_buffer(
            (m * 4) as u64,
            metal::MTLResourceOptions::StorageModeShared,
        ));

        let command_buffer = self.queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();

        encoder.set_compute_pipeline_state(&self.mv_iq1_s_pipeline);
        encoder.set_buffer(0, Some(weights_buffer), 0);
        encoder.set_buffer(1, Some(input_buffer), 0);
        encoder.set_buffer(2, Some(&output_buffer), 0);
        encoder.set_bytes(
            3,
            std::mem::size_of::<MvArgs>() as u64,
            &args as *const MvArgs as *const std::ffi::c_void,
        );

        const THREADGROUP_MEM_SIZE: u64 =
            32 * std::mem::size_of::<f32>() as u64;
        encoder.set_threadgroup_memory_length(THREADGROUP_MEM_SIZE, 0);

        const NSG: u64 = 2;
        const ROWS_PER_THREADGROUP: u64 = NSG * NR0;

        let thread_group_size = MTLSize { width: 32, height: NSG, depth: 1 };
        let thread_group_count = MTLSize {
            width: (m as u64 + ROWS_PER_THREADGROUP - 1) / ROWS_PER_THREADGROUP,
            height: 1,
            depth: 1,
        };

        encoder.dispatch_thread_groups(thread_group_count, thread_group_size);
        encoder.end_encoding();
        command_buffer.commit();
        command_buffer.wait_until_completed();

        Ok(output_buffer)
    }

    /// IQ2_XXS x F32 Matrix-Vector (fully zero-copy, both weights and input as GPU buffers)
    pub fn mv_iq2_xxs_f32_gpu_weights(
        &self,
        m: usize,
        k: usize,
        weights_buffer: &Buffer,
        input_buffer: &Buffer,
    ) -> Result<std::rc::Rc<Buffer>, String> {
        let nb = k / 256;
        let nb01 = (nb * std::mem::size_of::<BlockIQ2XXS>()) as u64;
        const NR0: u64 = 2;

        #[repr(C)]
        struct MvArgs {
            ne00: u32,
            ne01: u32,
            nb01: u64,
        }

        let args = MvArgs { ne00: k as u32, ne01: m as u32, nb01 };

        let output_buffer = std::rc::Rc::new(self.device.new_buffer(
            (m * 4) as u64,
            metal::MTLResourceOptions::StorageModeShared,
        ));

        let command_buffer = self.queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();

        encoder.set_compute_pipeline_state(&self.mv_iq2_xxs_pipeline);
        encoder.set_buffer(0, Some(weights_buffer), 0);
        encoder.set_buffer(1, Some(input_buffer), 0);
        encoder.set_buffer(2, Some(&output_buffer), 0);
        encoder.set_bytes(
            3,
            std::mem::size_of::<MvArgs>() as u64,
            &args as *const MvArgs as *const std::ffi::c_void,
        );

        const THREADGROUP_MEM_SIZE: u64 =
            32 * std::mem::size_of::<f32>() as u64;
        encoder.set_threadgroup_memory_length(THREADGROUP_MEM_SIZE, 0);

        const NSG: u64 = 2;
        const ROWS_PER_THREADGROUP: u64 = NSG * NR0;

        let thread_group_size = MTLSize { width: 32, height: NSG, depth: 1 };
        let thread_group_count = MTLSize {
            width: (m as u64 + ROWS_PER_THREADGROUP - 1) / ROWS_PER_THREADGROUP,
            height: 1,
            depth: 1,
        };

        encoder.dispatch_thread_groups(thread_group_count, thread_group_size);
        encoder.end_encoding();
        command_buffer.commit();
        command_buffer.wait_until_completed();

        Ok(output_buffer)
    }

    /// IQ2_XS x F32 Matrix-Vector (fully zero-copy, both weights and input as GPU buffers)
    pub fn mv_iq2_xs_f32_gpu_weights(
        &self,
        m: usize,
        k: usize,
        weights_buffer: &Buffer,
        input_buffer: &Buffer,
    ) -> Result<std::rc::Rc<Buffer>, String> {
        let nb = k / 256;
        let nb01 = (nb * std::mem::size_of::<BlockIQ2XS>()) as u64;
        const NR0: u64 = 2;

        #[repr(C)]
        struct MvArgs {
            ne00: u32,
            ne01: u32,
            nb01: u64,
        }

        let args = MvArgs { ne00: k as u32, ne01: m as u32, nb01 };

        let output_buffer = std::rc::Rc::new(self.device.new_buffer(
            (m * 4) as u64,
            metal::MTLResourceOptions::StorageModeShared,
        ));

        let command_buffer = self.queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();

        encoder.set_compute_pipeline_state(&self.mv_iq2_xs_pipeline);
        encoder.set_buffer(0, Some(weights_buffer), 0);
        encoder.set_buffer(1, Some(input_buffer), 0);
        encoder.set_buffer(2, Some(&output_buffer), 0);
        encoder.set_bytes(
            3,
            std::mem::size_of::<MvArgs>() as u64,
            &args as *const MvArgs as *const std::ffi::c_void,
        );

        const THREADGROUP_MEM_SIZE: u64 =
            32 * std::mem::size_of::<f32>() as u64;
        encoder.set_threadgroup_memory_length(THREADGROUP_MEM_SIZE, 0);

        const NSG: u64 = 2;
        const ROWS_PER_THREADGROUP: u64 = NSG * NR0;

        let thread_group_size = MTLSize { width: 32, height: NSG, depth: 1 };
        let thread_group_count = MTLSize {
            width: (m as u64 + ROWS_PER_THREADGROUP - 1) / ROWS_PER_THREADGROUP,
            height: 1,
            depth: 1,
        };

        encoder.dispatch_thread_groups(thread_group_count, thread_group_size);
        encoder.end_encoding();
        command_buffer.commit();
        command_buffer.wait_until_completed();

        Ok(output_buffer)
    }

    /// IQ2_S x F32 Matrix-Vector (fully zero-copy, both weights and input as GPU buffers)
    pub fn mv_iq2_s_f32_gpu_weights(
        &self,
        m: usize,
        k: usize,
        weights_buffer: &Buffer,
        input_buffer: &Buffer,
    ) -> Result<std::rc::Rc<Buffer>, String> {
        let nb = k / 256;
        let nb01 = (nb * std::mem::size_of::<BlockIQ2S>()) as u64;
        const NR0: u64 = 2;

        #[repr(C)]
        struct MvArgs {
            ne00: u32,
            ne01: u32,
            nb01: u64,
        }

        let args = MvArgs { ne00: k as u32, ne01: m as u32, nb01 };

        let output_buffer = std::rc::Rc::new(self.device.new_buffer(
            (m * 4) as u64,
            metal::MTLResourceOptions::StorageModeShared,
        ));

        let command_buffer = self.queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();

        encoder.set_compute_pipeline_state(&self.mv_iq2_s_pipeline);
        encoder.set_buffer(0, Some(weights_buffer), 0);
        encoder.set_buffer(1, Some(input_buffer), 0);
        encoder.set_buffer(2, Some(&output_buffer), 0);
        encoder.set_bytes(
            3,
            std::mem::size_of::<MvArgs>() as u64,
            &args as *const MvArgs as *const std::ffi::c_void,
        );

        const THREADGROUP_MEM_SIZE: u64 =
            32 * std::mem::size_of::<f32>() as u64;
        encoder.set_threadgroup_memory_length(THREADGROUP_MEM_SIZE, 0);

        const NSG: u64 = 2;
        const ROWS_PER_THREADGROUP: u64 = NSG * NR0;

        let thread_group_size = MTLSize { width: 32, height: NSG, depth: 1 };
        let thread_group_count = MTLSize {
            width: (m as u64 + ROWS_PER_THREADGROUP - 1) / ROWS_PER_THREADGROUP,
            height: 1,
            depth: 1,
        };

        encoder.dispatch_thread_groups(thread_group_count, thread_group_size);
        encoder.end_encoding();
        command_buffer.commit();
        command_buffer.wait_until_completed();

        Ok(output_buffer)
    }

    /// IQ3_XXS x F32 Matrix-Vector (fully zero-copy, both weights and input as GPU buffers)
    pub fn mv_iq3_xxs_f32_gpu_weights(
        &self,
        m: usize,
        k: usize,
        weights_buffer: &Buffer,
        input_buffer: &Buffer,
    ) -> Result<std::rc::Rc<Buffer>, String> {
        let nb = k / 256;
        let nb01 = (nb * std::mem::size_of::<BlockIQ3XXS>()) as u64;
        const NR0: u64 = 2;

        #[repr(C)]
        struct MvArgs {
            ne00: u32,
            ne01: u32,
            nb01: u64,
        }

        let args = MvArgs { ne00: k as u32, ne01: m as u32, nb01 };

        let output_buffer = std::rc::Rc::new(self.device.new_buffer(
            (m * 4) as u64,
            metal::MTLResourceOptions::StorageModeShared,
        ));

        let command_buffer = self.queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();

        encoder.set_compute_pipeline_state(&self.mv_iq3_xxs_pipeline);
        encoder.set_buffer(0, Some(weights_buffer), 0);
        encoder.set_buffer(1, Some(input_buffer), 0);
        encoder.set_buffer(2, Some(&output_buffer), 0);
        encoder.set_bytes(
            3,
            std::mem::size_of::<MvArgs>() as u64,
            &args as *const MvArgs as *const std::ffi::c_void,
        );

        const THREADGROUP_MEM_SIZE: u64 =
            32 * std::mem::size_of::<f32>() as u64;
        encoder.set_threadgroup_memory_length(THREADGROUP_MEM_SIZE, 0);

        const NSG: u64 = 2;
        const ROWS_PER_THREADGROUP: u64 = NSG * NR0;

        let thread_group_size = MTLSize { width: 32, height: NSG, depth: 1 };
        let thread_group_count = MTLSize {
            width: (m as u64 + ROWS_PER_THREADGROUP - 1) / ROWS_PER_THREADGROUP,
            height: 1,
            depth: 1,
        };

        encoder.dispatch_thread_groups(thread_group_count, thread_group_size);
        encoder.end_encoding();
        command_buffer.commit();
        command_buffer.wait_until_completed();

        Ok(output_buffer)
    }

    /// IQ3_S x F32 Matrix-Vector (fully zero-copy, both weights and input as GPU buffers)
    pub fn mv_iq3_s_f32_gpu_weights(
        &self,
        m: usize,
        k: usize,
        weights_buffer: &Buffer,
        input_buffer: &Buffer,
    ) -> Result<std::rc::Rc<Buffer>, String> {
        let nb = k / 256;
        let nb01 = (nb * std::mem::size_of::<BlockIQ3S>()) as u64;
        const NR0: u64 = 2;

        #[repr(C)]
        struct MvArgs {
            ne00: u32,
            ne01: u32,
            nb01: u64,
        }

        let args = MvArgs { ne00: k as u32, ne01: m as u32, nb01 };

        let output_buffer = std::rc::Rc::new(self.device.new_buffer(
            (m * 4) as u64,
            metal::MTLResourceOptions::StorageModeShared,
        ));

        let command_buffer = self.queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();

        encoder.set_compute_pipeline_state(&self.mv_iq3_s_pipeline);
        encoder.set_buffer(0, Some(weights_buffer), 0);
        encoder.set_buffer(1, Some(input_buffer), 0);
        encoder.set_buffer(2, Some(&output_buffer), 0);
        encoder.set_bytes(
            3,
            std::mem::size_of::<MvArgs>() as u64,
            &args as *const MvArgs as *const std::ffi::c_void,
        );

        const THREADGROUP_MEM_SIZE: u64 =
            32 * std::mem::size_of::<f32>() as u64;
        encoder.set_threadgroup_memory_length(THREADGROUP_MEM_SIZE, 0);

        const NSG: u64 = 2;
        const ROWS_PER_THREADGROUP: u64 = NSG * NR0;

        let thread_group_size = MTLSize { width: 32, height: NSG, depth: 1 };
        let thread_group_count = MTLSize {
            width: (m as u64 + ROWS_PER_THREADGROUP - 1) / ROWS_PER_THREADGROUP,
            height: 1,
            depth: 1,
        };

        encoder.dispatch_thread_groups(thread_group_count, thread_group_size);
        encoder.end_encoding();
        command_buffer.commit();
        command_buffer.wait_until_completed();

        Ok(output_buffer)
    }

    pub fn mv_tq2_0_f32(
        &self,
        m: usize,
        k: usize,
        weights: &[BlockTQ2_0],
        input: &[f32],
    ) -> Result<Vec<f32>, String> {
        let nb = k / 256; // QK_K = 256
        let nb01 = (nb * std::mem::size_of::<BlockTQ2_0>()) as u64; // Byte stride per row
        const NR0: u64 = 2; // N_R0_TQ2_0 = 2

        #[repr(C)]
        struct MvArgs {
            ne00: u32,
            ne01: u32,
            nb01: u64,
        }

        let args = MvArgs { ne00: k as u32, ne01: m as u32, nb01 };

        let weights_size = weights.len() * std::mem::size_of::<BlockTQ2_0>();
        let input_size = input.len() * std::mem::size_of::<f32>();
        let output_size = m * std::mem::size_of::<f32>();

        let weights_buffer = {
            let mut buf_cell = self.mv_weights_buffer.borrow_mut();
            let mut size_cell = self.mv_weights_size.borrow_mut();

            if *size_cell != weights_size {
                let buf = self.device.new_buffer_with_data(
                    weights.as_ptr() as *const std::ffi::c_void,
                    weights_size as u64,
                    metal::MTLResourceOptions::StorageModeShared,
                );
                *buf_cell = Some(buf.clone());
                *size_cell = weights_size;
                buf
            } else {
                buf_cell.as_ref().unwrap().clone()
            }
        };

        let input_buffer = {
            let mut buf_cell = self.mv_input_buffer.borrow_mut();
            let mut size_cell = self.mv_input_size.borrow_mut();

            if *size_cell != input_size {
                let buf = self.device.new_buffer_with_data(
                    input.as_ptr() as *const std::ffi::c_void,
                    input_size as u64,
                    metal::MTLResourceOptions::StorageModeShared,
                );
                *buf_cell = Some(buf.clone());
                *size_cell = input_size;
                buf
            } else {
                buf_cell.as_ref().unwrap().clone()
            }
        };

        let output_buffer = {
            let mut buf_cell = self.mv_output_buffer.borrow_mut();
            let mut size_cell = self.mv_output_size.borrow_mut();

            if *size_cell != output_size {
                let buf = self.device.new_buffer(
                    output_size as u64,
                    metal::MTLResourceOptions::StorageModeShared,
                );
                *buf_cell = Some(buf.clone());
                *size_cell = output_size;
                buf
            } else {
                buf_cell.as_ref().unwrap().clone()
            }
        };

        let command_buffer = self.queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();

        encoder.set_compute_pipeline_state(&self.mv_tq2_0_pipeline);
        encoder.set_buffer(0, Some(&weights_buffer), 0);
        encoder.set_buffer(1, Some(&input_buffer), 0);
        encoder.set_buffer(2, Some(&output_buffer), 0);
        encoder.set_bytes(
            3,
            std::mem::size_of::<MvArgs>() as u64,
            &args as *const MvArgs as *const std::ffi::c_void,
        );

        const NSG: u64 = 2;
        const ROWS_PER_THREADGROUP: u64 = NSG * NR0;

        let thread_group_size = MTLSize { width: 32, height: NSG, depth: 1 };
        let thread_group_count = MTLSize {
            width: (m as u64 + ROWS_PER_THREADGROUP - 1) / ROWS_PER_THREADGROUP,
            height: 1,
            depth: 1,
        };

        encoder.dispatch_thread_groups(thread_group_count, thread_group_size);
        encoder.end_encoding();

        command_buffer.commit();
        command_buffer.wait_until_completed();

        let output_ptr = output_buffer.contents() as *const f32;
        Ok(unsafe { std::slice::from_raw_parts(output_ptr, m).to_vec() })
    }

    pub fn mv_iq4_xs_f32(
        &self,
        m: usize,
        k: usize,
        weights: &[BlockIQ4XS],
        input: &[f32],
    ) -> Result<Vec<f32>, String> {
        let nb = k / 256; // QK_K = 256
        let nb01 = (nb * std::mem::size_of::<BlockIQ4XS>()) as u64; // Byte stride per row
        const NR0: u64 = 2; // N_R0_IQ4_XS = 2

        #[repr(C)]
        struct MvArgs {
            ne00: u32,
            ne01: u32,
            nb01: u64,
        }

        let args = MvArgs { ne00: k as u32, ne01: m as u32, nb01 };

        let weights_size = weights.len() * std::mem::size_of::<BlockIQ4XS>();
        let input_size = input.len() * std::mem::size_of::<f32>();
        let output_size = m * std::mem::size_of::<f32>();

        let weights_buffer = {
            let mut buf_cell = self.mv_weights_buffer.borrow_mut();
            let mut size_cell = self.mv_weights_size.borrow_mut();

            if *size_cell != weights_size {
                let buf = self.device.new_buffer_with_data(
                    weights.as_ptr() as *const std::ffi::c_void,
                    weights_size as u64,
                    metal::MTLResourceOptions::StorageModeShared,
                );
                *buf_cell = Some(buf.clone());
                *size_cell = weights_size;
                buf
            } else {
                buf_cell.as_ref().unwrap().clone()
            }
        };

        let input_buffer = {
            let mut buf_cell = self.mv_input_buffer.borrow_mut();
            let mut size_cell = self.mv_input_size.borrow_mut();

            if *size_cell != input_size {
                let buf = self.device.new_buffer_with_data(
                    input.as_ptr() as *const std::ffi::c_void,
                    input_size as u64,
                    metal::MTLResourceOptions::StorageModeShared,
                );
                *buf_cell = Some(buf.clone());
                *size_cell = input_size;
                buf
            } else {
                buf_cell.as_ref().unwrap().clone()
            }
        };

        let output_buffer = {
            let mut buf_cell = self.mv_output_buffer.borrow_mut();
            let mut size_cell = self.mv_output_size.borrow_mut();

            if *size_cell != output_size {
                let buf = self.device.new_buffer(
                    output_size as u64,
                    metal::MTLResourceOptions::StorageModeShared,
                );
                *buf_cell = Some(buf.clone());
                *size_cell = output_size;
                buf
            } else {
                buf_cell.as_ref().unwrap().clone()
            }
        };

        let command_buffer = self.queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();

        encoder.set_compute_pipeline_state(&self.mv_iq4_xs_pipeline);
        encoder.set_buffer(0, Some(&weights_buffer), 0);
        encoder.set_buffer(1, Some(&input_buffer), 0);
        encoder.set_buffer(2, Some(&output_buffer), 0);
        encoder.set_bytes(
            3,
            std::mem::size_of::<MvArgs>() as u64,
            &args as *const MvArgs as *const std::ffi::c_void,
        );

        const NSG: u64 = 2;
        const ROWS_PER_THREADGROUP: u64 = NSG * NR0;

        let thread_group_size = MTLSize { width: 32, height: NSG, depth: 1 };
        let thread_group_count = MTLSize {
            width: (m as u64 + ROWS_PER_THREADGROUP - 1) / ROWS_PER_THREADGROUP,
            height: 1,
            depth: 1,
        };

        encoder.dispatch_thread_groups(thread_group_count, thread_group_size);
        encoder.end_encoding();

        command_buffer.commit();
        command_buffer.wait_until_completed();

        let output_ptr = output_buffer.contents() as *const f32;
        Ok(unsafe { std::slice::from_raw_parts(output_ptr, m).to_vec() })
    }

    /// TQ1_0 x F32 Matrix-Vector
    pub fn mv_tq1_0_f32(
        &self,
        m: usize,
        k: usize,
        weights: &[BlockTQ1_0],
        input: &[f32],
    ) -> Result<Vec<f32>, String> {
        let nb = k / 256;
        let nb01 = (nb * std::mem::size_of::<BlockTQ1_0>()) as u64;
        const NR0: u64 = 2;

        #[repr(C)]
        struct MvArgs {
            ne00: u32,
            ne01: u32,
            nb01: u64,
        }

        let args = MvArgs { ne00: k as u32, ne01: m as u32, nb01 };

        let weights_size = weights.len() * std::mem::size_of::<BlockTQ1_0>();
        let input_size = input.len() * std::mem::size_of::<f32>();
        let output_size = m * std::mem::size_of::<f32>();

        let weights_buffer = {
            let mut buf_cell = self.mv_weights_buffer.borrow_mut();
            let mut size_cell = self.mv_weights_size.borrow_mut();

            if *size_cell != weights_size {
                let buf = self.device.new_buffer_with_data(
                    weights.as_ptr() as *const std::ffi::c_void,
                    weights_size as u64,
                    metal::MTLResourceOptions::StorageModeShared,
                );
                *buf_cell = Some(buf.clone());
                *size_cell = weights_size;
                buf
            } else {
                buf_cell.as_ref().unwrap().clone()
            }
        };

        let input_buffer = {
            let mut buf_cell = self.mv_input_buffer.borrow_mut();
            let mut size_cell = self.mv_input_size.borrow_mut();

            if *size_cell != input_size {
                let buf = self.device.new_buffer_with_data(
                    input.as_ptr() as *const std::ffi::c_void,
                    input_size as u64,
                    metal::MTLResourceOptions::StorageModeShared,
                );
                *buf_cell = Some(buf.clone());
                *size_cell = input_size;
                buf
            } else {
                buf_cell.as_ref().unwrap().clone()
            }
        };

        let output_buffer = {
            let mut buf_cell = self.mv_output_buffer.borrow_mut();
            let mut size_cell = self.mv_output_size.borrow_mut();

            if *size_cell != output_size {
                let buf = self.device.new_buffer(
                    output_size as u64,
                    metal::MTLResourceOptions::StorageModeShared,
                );
                *buf_cell = Some(buf.clone());
                *size_cell = output_size;
                buf
            } else {
                buf_cell.as_ref().unwrap().clone()
            }
        };

        let command_buffer = self.queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();

        encoder.set_compute_pipeline_state(&self.mv_tq1_0_pipeline);
        encoder.set_buffer(0, Some(&weights_buffer), 0);
        encoder.set_buffer(1, Some(&input_buffer), 0);
        encoder.set_buffer(2, Some(&output_buffer), 0);
        encoder.set_bytes(
            3,
            std::mem::size_of::<MvArgs>() as u64,
            &args as *const MvArgs as *const std::ffi::c_void,
        );

        const NSG: u64 = 2;
        const ROWS_PER_THREADGROUP: u64 = NSG * NR0;

        let thread_group_size = MTLSize { width: 32, height: NSG, depth: 1 };
        let thread_group_count = MTLSize {
            width: (m as u64 + ROWS_PER_THREADGROUP - 1) / ROWS_PER_THREADGROUP,
            height: 1,
            depth: 1,
        };

        encoder.dispatch_thread_groups(thread_group_count, thread_group_size);
        encoder.end_encoding();

        command_buffer.commit();
        command_buffer.wait_until_completed();

        let output_ptr = output_buffer.contents() as *const f32;
        Ok(unsafe { std::slice::from_raw_parts(output_ptr, m).to_vec() })
    }

    /// IQ1_S x F32 Matrix-Vector
    pub fn mv_iq1_s_f32(
        &self,
        m: usize,
        k: usize,
        weights: &[BlockIQ1S],
        input: &[f32],
    ) -> Result<Vec<f32>, String> {
        let nb = k / 256;
        let nb01 = (nb * std::mem::size_of::<BlockIQ1S>()) as u64;
        const NR0: u64 = 4; // N_R0_IQ1_S = 4

        #[repr(C)]
        struct MvArgs {
            ne00: u32,
            ne01: u32,
            nb01: u64,
        }

        let args = MvArgs { ne00: k as u32, ne01: m as u32, nb01 };
        let weights_size = weights.len() * std::mem::size_of::<BlockIQ1S>();
        let input_size = input.len() * std::mem::size_of::<f32>();
        let output_size = m * std::mem::size_of::<f32>();

        let weights_buffer = {
            let mut buf_cell = self.mv_weights_buffer.borrow_mut();
            let mut size_cell = self.mv_weights_size.borrow_mut();
            if *size_cell != weights_size {
                let buf = self.device.new_buffer_with_data(
                    weights.as_ptr() as *const std::ffi::c_void,
                    weights_size as u64,
                    metal::MTLResourceOptions::StorageModeShared,
                );
                *buf_cell = Some(buf.clone());
                *size_cell = weights_size;
                buf
            } else {
                buf_cell.as_ref().unwrap().clone()
            }
        };

        let input_buffer = {
            let mut buf_cell = self.mv_input_buffer.borrow_mut();
            let mut size_cell = self.mv_input_size.borrow_mut();
            if *size_cell != input_size {
                let buf = self.device.new_buffer_with_data(
                    input.as_ptr() as *const std::ffi::c_void,
                    input_size as u64,
                    metal::MTLResourceOptions::StorageModeShared,
                );
                *buf_cell = Some(buf.clone());
                *size_cell = input_size;
                buf
            } else {
                buf_cell.as_ref().unwrap().clone()
            }
        };

        let output_buffer = {
            let mut buf_cell = self.mv_output_buffer.borrow_mut();
            let mut size_cell = self.mv_output_size.borrow_mut();
            if *size_cell != output_size {
                let buf = self.device.new_buffer(
                    output_size as u64,
                    metal::MTLResourceOptions::StorageModeShared,
                );
                *buf_cell = Some(buf.clone());
                *size_cell = output_size;
                buf
            } else {
                buf_cell.as_ref().unwrap().clone()
            }
        };

        let command_buffer = self.queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(&self.mv_iq1_s_pipeline);
        encoder.set_buffer(0, Some(&weights_buffer), 0);
        encoder.set_buffer(1, Some(&input_buffer), 0);
        encoder.set_buffer(2, Some(&output_buffer), 0);
        encoder.set_bytes(
            3,
            std::mem::size_of::<MvArgs>() as u64,
            &args as *const MvArgs as *const std::ffi::c_void,
        );

        const NSG: u64 = 2;
        const ROWS_PER_THREADGROUP: u64 = NSG * NR0;
        let thread_group_size = MTLSize { width: 32, height: NSG, depth: 1 };
        let thread_group_count = MTLSize {
            width: (m as u64 + ROWS_PER_THREADGROUP - 1) / ROWS_PER_THREADGROUP,
            height: 1,
            depth: 1,
        };

        encoder.dispatch_thread_groups(thread_group_count, thread_group_size);
        encoder.end_encoding();
        command_buffer.commit();
        command_buffer.wait_until_completed();

        let output_ptr = output_buffer.contents() as *const f32;
        Ok(unsafe { std::slice::from_raw_parts(output_ptr, m).to_vec() })
    }

    /// IQ1_M x F32 Matrix-Vector
    pub fn mv_iq1_m_f32(
        &self,
        m: usize,
        k: usize,
        weights: &[BlockIQ1M],
        input: &[f32],
    ) -> Result<Vec<f32>, String> {
        let nb = k / 256;
        let nb01 = (nb * std::mem::size_of::<BlockIQ1M>()) as u64;
        const NR0: u64 = 4; // N_R0_IQ1_M = 4

        #[repr(C)]
        struct MvArgs {
            ne00: u32,
            ne01: u32,
            nb01: u64,
        }

        let args = MvArgs { ne00: k as u32, ne01: m as u32, nb01 };
        let weights_size = weights.len() * std::mem::size_of::<BlockIQ1M>();
        let input_size = input.len() * std::mem::size_of::<f32>();
        let output_size = m * std::mem::size_of::<f32>();

        let weights_buffer = {
            let mut buf_cell = self.mv_weights_buffer.borrow_mut();
            let mut size_cell = self.mv_weights_size.borrow_mut();
            if *size_cell != weights_size {
                let buf = self.device.new_buffer_with_data(
                    weights.as_ptr() as *const std::ffi::c_void,
                    weights_size as u64,
                    metal::MTLResourceOptions::StorageModeShared,
                );
                *buf_cell = Some(buf.clone());
                *size_cell = weights_size;
                buf
            } else {
                buf_cell.as_ref().unwrap().clone()
            }
        };

        let input_buffer = {
            let mut buf_cell = self.mv_input_buffer.borrow_mut();
            let mut size_cell = self.mv_input_size.borrow_mut();
            if *size_cell != input_size {
                let buf = self.device.new_buffer_with_data(
                    input.as_ptr() as *const std::ffi::c_void,
                    input_size as u64,
                    metal::MTLResourceOptions::StorageModeShared,
                );
                *buf_cell = Some(buf.clone());
                *size_cell = input_size;
                buf
            } else {
                buf_cell.as_ref().unwrap().clone()
            }
        };

        let output_buffer = {
            let mut buf_cell = self.mv_output_buffer.borrow_mut();
            let mut size_cell = self.mv_output_size.borrow_mut();
            if *size_cell != output_size {
                let buf = self.device.new_buffer(
                    output_size as u64,
                    metal::MTLResourceOptions::StorageModeShared,
                );
                *buf_cell = Some(buf.clone());
                *size_cell = output_size;
                buf
            } else {
                buf_cell.as_ref().unwrap().clone()
            }
        };

        let command_buffer = self.queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(&self.mv_iq1_m_pipeline);
        encoder.set_buffer(0, Some(&weights_buffer), 0);
        encoder.set_buffer(1, Some(&input_buffer), 0);
        encoder.set_buffer(2, Some(&output_buffer), 0);
        encoder.set_bytes(
            3,
            std::mem::size_of::<MvArgs>() as u64,
            &args as *const MvArgs as *const std::ffi::c_void,
        );

        const NSG: u64 = 2;
        const ROWS_PER_THREADGROUP: u64 = NSG * NR0;
        let thread_group_size = MTLSize { width: 32, height: NSG, depth: 1 };
        let thread_group_count = MTLSize {
            width: (m as u64 + ROWS_PER_THREADGROUP - 1) / ROWS_PER_THREADGROUP,
            height: 1,
            depth: 1,
        };

        encoder.dispatch_thread_groups(thread_group_count, thread_group_size);
        encoder.end_encoding();
        command_buffer.commit();
        command_buffer.wait_until_completed();

        let output_ptr = output_buffer.contents() as *const f32;
        Ok(unsafe { std::slice::from_raw_parts(output_ptr, m).to_vec() })
    }

    /// IQ2_XXS x F32 Matrix-Vector with dynamic kernel selection
    pub fn mv_iq2_xxs_f32(
        &self,
        m: usize,
        k: usize,
        weights: &[BlockIQ2XXS],
        input: &[f32],
    ) -> Result<Vec<f32>, String> {
        let nb = k / 256;
        let nb01 = (nb * std::mem::size_of::<BlockIQ2XXS>()) as u64;
        let nb32 = nb * (256 / 32);

        // Dynamic dispatch: use split=1 for small nb32
        let use_split = nb32 < 32;
        let nr0 = if use_split { 8u64 } else { 4u64 };

        #[repr(C)]
        struct MvArgs {
            ne00: u32,
            ne01: u32,
            nb01: u64,
        }

        let args = MvArgs { ne00: k as u32, ne01: m as u32, nb01 };
        let weights_size = weights.len() * std::mem::size_of::<BlockIQ2XXS>();
        let input_size = input.len() * std::mem::size_of::<f32>();
        let output_size = m * std::mem::size_of::<f32>();

        let weights_buffer = {
            let mut buf_cell = self.mv_weights_buffer.borrow_mut();
            let mut size_cell = self.mv_weights_size.borrow_mut();
            if *size_cell != weights_size {
                let buf = self.device.new_buffer_with_data(
                    weights.as_ptr() as *const std::ffi::c_void,
                    weights_size as u64,
                    metal::MTLResourceOptions::StorageModeShared,
                );
                *buf_cell = Some(buf.clone());
                *size_cell = weights_size;
                buf
            } else {
                buf_cell.as_ref().unwrap().clone()
            }
        };

        let input_buffer = {
            let mut buf_cell = self.mv_input_buffer.borrow_mut();
            let mut size_cell = self.mv_input_size.borrow_mut();
            if *size_cell != input_size {
                let buf = self.device.new_buffer_with_data(
                    input.as_ptr() as *const std::ffi::c_void,
                    input_size as u64,
                    metal::MTLResourceOptions::StorageModeShared,
                );
                *buf_cell = Some(buf.clone());
                *size_cell = input_size;
                buf
            } else {
                buf_cell.as_ref().unwrap().clone()
            }
        };

        let output_buffer = {
            let mut buf_cell = self.mv_output_buffer.borrow_mut();
            let mut size_cell = self.mv_output_size.borrow_mut();
            if *size_cell != output_size {
                let buf = self.device.new_buffer(
                    output_size as u64,
                    metal::MTLResourceOptions::StorageModeShared,
                );
                *buf_cell = Some(buf.clone());
                *size_cell = output_size;
                buf
            } else {
                buf_cell.as_ref().unwrap().clone()
            }
        };

        let command_buffer = self.queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();

        // Select appropriate pipeline
        if use_split {
            encoder
                .set_compute_pipeline_state(&self.mv_iq2_xxs_split1_pipeline);
        } else {
            encoder.set_compute_pipeline_state(&self.mv_iq2_xxs_pipeline);
        }

        encoder.set_buffer(0, Some(&weights_buffer), 0);
        encoder.set_buffer(1, Some(&input_buffer), 0);
        encoder.set_buffer(2, Some(&output_buffer), 0);
        encoder.set_bytes(
            3,
            std::mem::size_of::<MvArgs>() as u64,
            &args as *const MvArgs as *const std::ffi::c_void,
        );

        // Threadgroup memory for IQ2_XXS: svalues (256*8) + ssigns (128) = 2176 bytes
        const THREADGROUP_MEM_SIZE: u64 = 256 * 8 + 128;
        encoder.set_threadgroup_memory_length(THREADGROUP_MEM_SIZE, 0);

        const NSG: u64 = 2;
        let rows_per_threadgroup = NSG * nr0;
        let thread_group_size = MTLSize { width: 32, height: NSG, depth: 1 };
        let thread_group_count = MTLSize {
            width: (m as u64 + rows_per_threadgroup - 1) / rows_per_threadgroup,
            height: 1,
            depth: 1,
        };

        encoder.dispatch_thread_groups(thread_group_count, thread_group_size);
        encoder.end_encoding();
        command_buffer.commit();
        command_buffer.wait_until_completed();

        let output_ptr = output_buffer.contents() as *const f32;
        Ok(unsafe { std::slice::from_raw_parts(output_ptr, m).to_vec() })
    }

    /// IQ2_XS x F32 Matrix-Vector with dynamic kernel selection
    pub fn mv_iq2_xs_f32(
        &self,
        m: usize,
        k: usize,
        weights: &[BlockIQ2XS],
        input: &[f32],
    ) -> Result<Vec<f32>, String> {
        let nb = k / 256;
        let nb01 = (nb * std::mem::size_of::<BlockIQ2XS>()) as u64;
        let nb32 = nb * (256 / 32);

        // Dynamic dispatch: use split=1 for small nb32
        let use_split = nb32 < 32;
        let nr0 = if use_split { 8u64 } else { 4u64 };

        #[repr(C)]
        struct MvArgs {
            ne00: u32,
            ne01: u32,
            nb01: u64,
        }

        let args = MvArgs { ne00: k as u32, ne01: m as u32, nb01 };
        let weights_size = weights.len() * std::mem::size_of::<BlockIQ2XS>();
        let input_size = input.len() * std::mem::size_of::<f32>();
        let output_size = m * std::mem::size_of::<f32>();

        let weights_buffer = {
            let mut buf_cell = self.mv_weights_buffer.borrow_mut();
            let mut size_cell = self.mv_weights_size.borrow_mut();
            if *size_cell != weights_size {
                let buf = self.device.new_buffer_with_data(
                    weights.as_ptr() as *const std::ffi::c_void,
                    weights_size as u64,
                    metal::MTLResourceOptions::StorageModeShared,
                );
                *buf_cell = Some(buf.clone());
                *size_cell = weights_size;
                buf
            } else {
                buf_cell.as_ref().unwrap().clone()
            }
        };

        let input_buffer = {
            let mut buf_cell = self.mv_input_buffer.borrow_mut();
            let mut size_cell = self.mv_input_size.borrow_mut();
            if *size_cell != input_size {
                let buf = self.device.new_buffer_with_data(
                    input.as_ptr() as *const std::ffi::c_void,
                    input_size as u64,
                    metal::MTLResourceOptions::StorageModeShared,
                );
                *buf_cell = Some(buf.clone());
                *size_cell = input_size;
                buf
            } else {
                buf_cell.as_ref().unwrap().clone()
            }
        };

        let output_buffer = {
            let mut buf_cell = self.mv_output_buffer.borrow_mut();
            let mut size_cell = self.mv_output_size.borrow_mut();
            if *size_cell != output_size {
                let buf = self.device.new_buffer(
                    output_size as u64,
                    metal::MTLResourceOptions::StorageModeShared,
                );
                *buf_cell = Some(buf.clone());
                *size_cell = output_size;
                buf
            } else {
                buf_cell.as_ref().unwrap().clone()
            }
        };

        let command_buffer = self.queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();

        // Select appropriate pipeline
        if use_split {
            encoder.set_compute_pipeline_state(&self.mv_iq2_xs_split1_pipeline);
        } else {
            encoder.set_compute_pipeline_state(&self.mv_iq2_xs_pipeline);
        }
        encoder.set_buffer(0, Some(&weights_buffer), 0);
        encoder.set_buffer(1, Some(&input_buffer), 0);
        encoder.set_buffer(2, Some(&output_buffer), 0);
        encoder.set_bytes(
            3,
            std::mem::size_of::<MvArgs>() as u64,
            &args as *const MvArgs as *const std::ffi::c_void,
        );

        // Set threadgroup memory size: 512*8 (grid) + 128 (signs) = 4224 bytes
        // Round up to multiple of 16 as required by Metal
        const THREADGROUP_MEM_SIZE: u64 = 4224;
        encoder.set_threadgroup_memory_length(THREADGROUP_MEM_SIZE, 0);

        const NSG: u64 = 2;
        let rows_per_threadgroup = NSG * nr0;
        let thread_group_size = MTLSize { width: 32, height: NSG, depth: 1 };
        let thread_group_count = MTLSize {
            width: (m as u64 + rows_per_threadgroup - 1) / rows_per_threadgroup,
            height: 1,
            depth: 1,
        };

        encoder.dispatch_thread_groups(thread_group_count, thread_group_size);
        encoder.end_encoding();
        command_buffer.commit();
        command_buffer.wait_until_completed();

        let output_ptr = output_buffer.contents() as *const f32;
        Ok(unsafe { std::slice::from_raw_parts(output_ptr, m).to_vec() })
    }

    /// IQ2_S x F32 Matrix-Vector with dynamic kernel selection
    pub fn mv_iq2_s_f32(
        &self,
        m: usize,
        k: usize,
        weights: &[BlockIQ2S],
        input: &[f32],
    ) -> Result<Vec<f32>, String> {
        let nb = k / 256;
        let nb01 = (nb * std::mem::size_of::<BlockIQ2S>()) as u64;
        let nb32 = nb * (256 / 32); // nb32 = nb * 8

        // Dynamic dispatch: use split=1 for small nb32
        let use_split = nb32 < 32;
        let nr0 = if use_split { 8u64 } else { 4u64 };

        #[repr(C)]
        struct MvArgs {
            ne00: u32,
            ne01: u32,
            nb01: u64,
        }

        let args = MvArgs { ne00: k as u32, ne01: m as u32, nb01 };
        let weights_size = weights.len() * std::mem::size_of::<BlockIQ2S>();
        let input_size = input.len() * std::mem::size_of::<f32>();
        let output_size = m * std::mem::size_of::<f32>();

        let weights_buffer = {
            let mut buf_cell = self.mv_weights_buffer.borrow_mut();
            let mut size_cell = self.mv_weights_size.borrow_mut();
            if *size_cell != weights_size {
                let buf = self.device.new_buffer_with_data(
                    weights.as_ptr() as *const std::ffi::c_void,
                    weights_size as u64,
                    metal::MTLResourceOptions::StorageModeShared,
                );
                *buf_cell = Some(buf.clone());
                *size_cell = weights_size;
                buf
            } else {
                buf_cell.as_ref().unwrap().clone()
            }
        };

        let input_buffer = {
            let mut buf_cell = self.mv_input_buffer.borrow_mut();
            let mut size_cell = self.mv_input_size.borrow_mut();
            if *size_cell != input_size {
                let buf = self.device.new_buffer_with_data(
                    input.as_ptr() as *const std::ffi::c_void,
                    input_size as u64,
                    metal::MTLResourceOptions::StorageModeShared,
                );
                *buf_cell = Some(buf.clone());
                *size_cell = input_size;
                buf
            } else {
                buf_cell.as_ref().unwrap().clone()
            }
        };

        let output_buffer = {
            let mut buf_cell = self.mv_output_buffer.borrow_mut();
            let mut size_cell = self.mv_output_size.borrow_mut();
            if *size_cell != output_size {
                let buf = self.device.new_buffer(
                    output_size as u64,
                    metal::MTLResourceOptions::StorageModeShared,
                );
                *buf_cell = Some(buf.clone());
                *size_cell = output_size;
                buf
            } else {
                buf_cell.as_ref().unwrap().clone()
            }
        };

        let command_buffer = self.queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();

        // Select appropriate pipeline
        if use_split {
            encoder.set_compute_pipeline_state(&self.mv_iq2_s_split1_pipeline);
        } else {
            encoder.set_compute_pipeline_state(&self.mv_iq2_s_pipeline);
        }

        encoder.set_buffer(0, Some(&weights_buffer), 0);
        encoder.set_buffer(1, Some(&input_buffer), 0);
        encoder.set_buffer(2, Some(&output_buffer), 0);
        encoder.set_bytes(
            3,
            std::mem::size_of::<MvArgs>() as u64,
            &args as *const MvArgs as *const std::ffi::c_void,
        );

        const NSG: u64 = 2;
        let rows_per_threadgroup = NSG * nr0;
        let thread_group_size = MTLSize { width: 32, height: NSG, depth: 1 };
        let thread_group_count = MTLSize {
            width: (m as u64 + rows_per_threadgroup - 1) / rows_per_threadgroup,
            height: 1,
            depth: 1,
        };

        encoder.dispatch_thread_groups(thread_group_count, thread_group_size);
        encoder.end_encoding();
        command_buffer.commit();
        command_buffer.wait_until_completed();

        let output_ptr = output_buffer.contents() as *const f32;
        Ok(unsafe { std::slice::from_raw_parts(output_ptr, m).to_vec() })
    }

    /// IQ3_XXS x F32 Matrix-Vector with dynamic kernel selection
    pub fn mv_iq3_xxs_f32(
        &self,
        m: usize,
        k: usize,
        weights: &[BlockIQ3XXS],
        input: &[f32],
    ) -> Result<Vec<f32>, String> {
        let nb = k / 256;
        let nb01 = (nb * std::mem::size_of::<BlockIQ3XXS>()) as u64;
        let nb32 = nb * (256 / 32);

        // Dynamic dispatch: use split=1 for small nb32
        let use_split = nb32 < 32;
        let nr0 = if use_split { 8u64 } else { 4u64 };

        #[repr(C)]
        struct MvArgs {
            ne00: u32,
            ne01: u32,
            nb01: u64,
        }

        let args = MvArgs { ne00: k as u32, ne01: m as u32, nb01 };
        let weights_size = weights.len() * std::mem::size_of::<BlockIQ3XXS>();
        let input_size = input.len() * std::mem::size_of::<f32>();
        let output_size = m * std::mem::size_of::<f32>();

        let weights_buffer = {
            let mut buf_cell = self.mv_weights_buffer.borrow_mut();
            let mut size_cell = self.mv_weights_size.borrow_mut();
            if *size_cell != weights_size {
                let buf = self.device.new_buffer_with_data(
                    weights.as_ptr() as *const std::ffi::c_void,
                    weights_size as u64,
                    metal::MTLResourceOptions::StorageModeShared,
                );
                *buf_cell = Some(buf.clone());
                *size_cell = weights_size;
                buf
            } else {
                buf_cell.as_ref().unwrap().clone()
            }
        };

        let input_buffer = {
            let mut buf_cell = self.mv_input_buffer.borrow_mut();
            let mut size_cell = self.mv_input_size.borrow_mut();
            if *size_cell != input_size {
                let buf = self.device.new_buffer_with_data(
                    input.as_ptr() as *const std::ffi::c_void,
                    input_size as u64,
                    metal::MTLResourceOptions::StorageModeShared,
                );
                *buf_cell = Some(buf.clone());
                *size_cell = input_size;
                buf
            } else {
                buf_cell.as_ref().unwrap().clone()
            }
        };

        let output_buffer = {
            let mut buf_cell = self.mv_output_buffer.borrow_mut();
            let mut size_cell = self.mv_output_size.borrow_mut();
            if *size_cell != output_size {
                let buf = self.device.new_buffer(
                    output_size as u64,
                    metal::MTLResourceOptions::StorageModeShared,
                );
                *buf_cell = Some(buf.clone());
                *size_cell = output_size;
                buf
            } else {
                buf_cell.as_ref().unwrap().clone()
            }
        };

        let command_buffer = self.queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();

        // Select appropriate pipeline
        if use_split {
            encoder
                .set_compute_pipeline_state(&self.mv_iq3_xxs_split1_pipeline);
        } else {
            encoder.set_compute_pipeline_state(&self.mv_iq3_xxs_pipeline);
        }

        encoder.set_buffer(0, Some(&weights_buffer), 0);
        encoder.set_buffer(1, Some(&input_buffer), 0);
        encoder.set_buffer(2, Some(&output_buffer), 0);
        encoder.set_bytes(
            3,
            std::mem::size_of::<MvArgs>() as u64,
            &args as *const MvArgs as *const std::ffi::c_void,
        );

        // Threadgroup memory for IQ3_XXS: svalues (256*4) + ssigns (128) = 1152 bytes
        const THREADGROUP_MEM_SIZE: u64 = 256 * 4 + 128;
        encoder.set_threadgroup_memory_length(THREADGROUP_MEM_SIZE, 0);

        const NSG: u64 = 2;
        let rows_per_threadgroup = NSG * nr0;
        let thread_group_size = MTLSize { width: 32, height: NSG, depth: 1 };
        let thread_group_count = MTLSize {
            width: (m as u64 + rows_per_threadgroup - 1) / rows_per_threadgroup,
            height: 1,
            depth: 1,
        };

        encoder.dispatch_thread_groups(thread_group_count, thread_group_size);
        encoder.end_encoding();
        command_buffer.commit();
        command_buffer.wait_until_completed();

        let output_ptr = output_buffer.contents() as *const f32;
        Ok(unsafe { std::slice::from_raw_parts(output_ptr, m).to_vec() })
    }

    /// IQ3_S x F32 Matrix-Vector with dynamic kernel selection
    pub fn mv_iq3_s_f32(
        &self,
        m: usize,
        k: usize,
        weights: &[BlockIQ3S],
        input: &[f32],
    ) -> Result<Vec<f32>, String> {
        let nb = k / 256;
        let nb01 = (nb * std::mem::size_of::<BlockIQ3S>()) as u64;
        let nb32 = nb * (256 / 32);

        // Dynamic dispatch: use split=1 for small nb32
        let use_split = nb32 < 32;
        let nr0 = if use_split { 8u64 } else { 4u64 };

        #[repr(C)]
        struct MvArgs {
            ne00: u32,
            ne01: u32,
            nb01: u64,
        }

        let args = MvArgs { ne00: k as u32, ne01: m as u32, nb01 };
        let weights_size = weights.len() * std::mem::size_of::<BlockIQ3S>();
        let input_size = input.len() * std::mem::size_of::<f32>();
        let output_size = m * std::mem::size_of::<f32>();

        let weights_buffer = {
            let mut buf_cell = self.mv_weights_buffer.borrow_mut();
            let mut size_cell = self.mv_weights_size.borrow_mut();
            if *size_cell != weights_size {
                let buf = self.device.new_buffer_with_data(
                    weights.as_ptr() as *const std::ffi::c_void,
                    weights_size as u64,
                    metal::MTLResourceOptions::StorageModeShared,
                );
                *buf_cell = Some(buf.clone());
                *size_cell = weights_size;
                buf
            } else {
                buf_cell.as_ref().unwrap().clone()
            }
        };

        let input_buffer = {
            let mut buf_cell = self.mv_input_buffer.borrow_mut();
            let mut size_cell = self.mv_input_size.borrow_mut();
            if *size_cell != input_size {
                let buf = self.device.new_buffer_with_data(
                    input.as_ptr() as *const std::ffi::c_void,
                    input_size as u64,
                    metal::MTLResourceOptions::StorageModeShared,
                );
                *buf_cell = Some(buf.clone());
                *size_cell = input_size;
                buf
            } else {
                buf_cell.as_ref().unwrap().clone()
            }
        };

        let output_buffer = {
            let mut buf_cell = self.mv_output_buffer.borrow_mut();
            let mut size_cell = self.mv_output_size.borrow_mut();
            if *size_cell != output_size {
                let buf = self.device.new_buffer(
                    output_size as u64,
                    metal::MTLResourceOptions::StorageModeShared,
                );
                *buf_cell = Some(buf.clone());
                *size_cell = output_size;
                buf
            } else {
                buf_cell.as_ref().unwrap().clone()
            }
        };

        let command_buffer = self.queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();

        // Select appropriate pipeline
        if use_split {
            encoder.set_compute_pipeline_state(&self.mv_iq3_s_split1_pipeline);
        } else {
            encoder.set_compute_pipeline_state(&self.mv_iq3_s_pipeline);
        }

        encoder.set_buffer(0, Some(&weights_buffer), 0);
        encoder.set_buffer(1, Some(&input_buffer), 0);
        encoder.set_buffer(2, Some(&output_buffer), 0);
        encoder.set_bytes(
            3,
            std::mem::size_of::<MvArgs>() as u64,
            &args as *const MvArgs as *const std::ffi::c_void,
        );

        // Threadgroup memory for IQ3_S: svalues (512*4) = 2048 bytes
        const THREADGROUP_MEM_SIZE: u64 = 512 * 4;
        encoder.set_threadgroup_memory_length(THREADGROUP_MEM_SIZE, 0);

        const NSG: u64 = 2;
        let rows_per_threadgroup = NSG * nr0;
        let thread_group_size = MTLSize { width: 32, height: NSG, depth: 1 };
        let thread_group_count = MTLSize {
            width: (m as u64 + rows_per_threadgroup - 1) / rows_per_threadgroup,
            height: 1,
            depth: 1,
        };

        encoder.dispatch_thread_groups(thread_group_count, thread_group_size);
        encoder.end_encoding();
        command_buffer.commit();
        command_buffer.wait_until_completed();

        let output_ptr = output_buffer.contents() as *const f32;
        Ok(unsafe { std::slice::from_raw_parts(output_ptr, m).to_vec() })
    }

    /// MXFP4 x F32 Matrix-Vector
    pub fn mv_mxfp4_f32(
        &self,
        m: usize,
        k: usize,
        weights: &[BlockMXFP4],
        input: &[f32],
    ) -> Result<Vec<f32>, String> {
        let nb = k / 32;
        let nb01 = (nb * std::mem::size_of::<BlockMXFP4>()) as u64;

        #[repr(C)]
        struct MvArgs {
            ne00: i32,
            ne01: i32,
            ne02: i32,
            nb00: u64,
            nb01: u64,
            nb02: u64,
            nb03: u64,
            ne10: i32,
            ne11: i32,
            ne12: i32,
            nb10: u64,
            nb11: u64,
            nb12: u64,
            nb13: u64,
            ne0: i32,
            ne1: i32,
            nr0: i32,
            r2: i16,
            r3: i16,
        }
        let args = MvArgs {
            ne00: k as i32,
            ne01: m as i32,
            ne02: 1,
            nb00: std::mem::size_of::<BlockMXFP4>() as u64,
            nb01,
            nb02: nb01,
            nb03: nb01,
            ne10: k as i32,
            ne11: 1,
            ne12: 1,
            nb10: 4,
            nb11: k as u64 * 4,
            nb12: 0,
            nb13: 0,
            ne0: m as i32,
            ne1: 1,
            nr0: 2,
            r2: 1,
            r3: 1,
        };

        let weights_buffer = self.device.new_buffer_with_data(
            weights.as_ptr() as *const std::ffi::c_void,
            (weights.len() * std::mem::size_of::<BlockMXFP4>()) as u64,
            metal::MTLResourceOptions::StorageModeShared,
        );
        let input_buffer = self.device.new_buffer_with_data(
            input.as_ptr() as *const std::ffi::c_void,
            (input.len() * 4) as u64,
            metal::MTLResourceOptions::StorageModeShared,
        );
        let output_buffer = self.device.new_buffer(
            (m * 4) as u64,
            metal::MTLResourceOptions::StorageModeShared,
        );

        let command_buffer = self.queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(&self.mv_mxfp4_pipeline);
        encoder.set_bytes(
            0,
            std::mem::size_of::<MvArgs>() as u64,
            &args as *const _ as *const std::ffi::c_void,
        );
        encoder.set_buffer(1, Some(&weights_buffer), 0);
        encoder.set_buffer(2, Some(&input_buffer), 0);
        encoder.set_buffer(3, Some(&output_buffer), 0);
        encoder.dispatch_thread_groups(
            metal::MTLSize::new(((m + 1) / 2) as u64, 1, 1),
            metal::MTLSize::new(32, 2, 1),
        );
        encoder.end_encoding();
        command_buffer.commit();
        command_buffer.wait_until_completed();

        Ok(unsafe {
            std::slice::from_raw_parts(
                output_buffer.contents() as *const f32,
                m,
            )
            .to_vec()
        })
    }

    // These use the kernel_mul_mm template with dequantize functions

    /// Note: Output is in column-major format (column * M + row)
    pub fn gemm_template_q4_k_f32(
        &self,
        m: usize,
        n: usize,
        k: usize,
        weights: &[BlockQ4K],
        input: &[f32],
    ) -> Result<Vec<f32>, String> {
        let nb = k / 256;
        let nb01 = (nb * std::mem::size_of::<BlockQ4K>()) as u64;
        let _nb00 = std::mem::size_of::<BlockQ4K>() as u64;

        #[repr(C)]
        struct GemmArgs {
            ne00: i32, // K
            ne02: i32, // batch dim (1 for simple case)
            nb01: u64, // stride for A rows
            nb02: u64, // batch stride (0 for simple case)
            nb03: u64, // batch3 stride (0)
            ne12: i32, // batch dim for B (1)
            nb10: u64, // element stride for B (2 for FP16)
            nb11: u64, // row stride for B (K * 2)
            nb12: u64, // batch stride (0)
            nb13: u64, // batch3 stride (0)
            ne0: i32,  // M (output rows)
            ne1: i32,  // N (output cols)
            r2: i16,   // broadcast ratio (1)
            r3: i16,   // broadcast ratio (1)
        }

        let args = GemmArgs {
            ne00: k as i32,
            ne02: 1, // Weights have no batch dim
            nb01,
            nb02: 0,
            nb03: 0,
            ne12: n as i32,       // Input batch dimension
            nb10: 2,              // FP16 element size
            nb11: (k * 2) as u64, // K * sizeof(FP16)
            nb12: (k * 2) as u64, // Batch stride for input: K elements per batch
            nb13: 0,
            ne0: m as i32,
            ne1: n as i32,
            r2: 1,
            r3: 1,
        };

        let weights_size = weights.len() * std::mem::size_of::<BlockQ4K>();

        // Convert FP32 input to FP16
        let input_f16: Vec<u16> =
            input.iter().map(|&x| half::f16::from_f32(x).to_bits()).collect();
        let input_size = input_f16.len() * 2;
        let output_size = m * n * std::mem::size_of::<f32>();

        let weights_buffer = self.device.new_buffer_with_data(
            weights.as_ptr() as *const std::ffi::c_void,
            weights_size as u64,
            metal::MTLResourceOptions::StorageModeShared,
        );
        let input_buffer = self.device.new_buffer_with_data(
            input_f16.as_ptr() as *const std::ffi::c_void,
            input_size as u64,
            metal::MTLResourceOptions::StorageModeShared,
        );
        let output_buffer = self.device.new_buffer(
            output_size as u64,
            metal::MTLResourceOptions::StorageModeShared,
        );

        let command_buffer = self.queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(&self.gemm_template_q4_k_pipeline);
        encoder.set_bytes(
            0,
            std::mem::size_of::<GemmArgs>() as u64,
            &args as *const GemmArgs as *const std::ffi::c_void,
        );
        encoder.set_buffer(1, Some(&weights_buffer), 0);
        encoder.set_buffer(2, Some(&input_buffer), 0);
        encoder.set_buffer(3, Some(&output_buffer), 0);

        encoder.set_threadgroup_memory_length(0, 8192);

        // For quantized types: threadgroup is 32xnsg, grid is (M/nr0*nsg) x (N/nr1)
        // This groups multiple simdgroups per threadgroup for better efficiency
        const NR0: u64 = 64;
        const NR1: u64 = 32;
        const NSG: u64 = 4; // N_MM_SIMD_GROUP_X * N_MM_SIMD_GROUP_Y

        let thread_group_size = MTLSize { width: 32, height: NSG, depth: 1 };
        let thread_group_count = MTLSize {
            width: ((m as u64 + NR0 * NSG - 1) / (NR0 * NSG)),
            height: ((n as u64 + NR1 - 1) / NR1),
            depth: 1,
        };

        encoder.dispatch_thread_groups(thread_group_count, thread_group_size);
        encoder.end_encoding();
        command_buffer.commit();
        command_buffer.wait_until_completed();

        let output_ptr = output_buffer.contents() as *const f32;
        Ok(unsafe { std::slice::from_raw_parts(output_ptr, m * n).to_vec() })
    }

    /// Templated Q2_K × F32 GEMM
    pub fn gemm_template_q2_k_f32(
        &self,
        m: usize,
        n: usize,
        k: usize,
        weights: &[BlockQ2K],
        input: &[f32],
    ) -> Result<Vec<f32>, String> {
        let nb = k / 256;
        let nb01 = (nb * std::mem::size_of::<BlockQ2K>()) as u64;
        let _nb00 = std::mem::size_of::<BlockQ2K>() as u64;

        #[repr(C)]
        struct GemmArgs {
            ne00: i32, // K
            ne02: i32, // batch dim (1 for simple case)
            nb01: u64, // stride for A rows
            nb02: u64, // batch stride (0 for simple case)
            nb03: u64, // batch3 stride (0)
            ne12: i32, // batch dim for B
            nb10: u64, // element stride for B (2 for FP16)
            nb11: u64, // row stride for B (K * 2)
            nb12: u64, // batch stride for input
            nb13: u64, // batch3 stride (0)
            ne0: i32,  // M (output rows)
            ne1: i32,  // N (output cols)
            r2: i16,   // broadcast ratio (1)
            r3: i16,   // broadcast ratio (1)
        }

        let args = GemmArgs {
            ne00: k as i32,
            ne02: 1, // Weights have no batch dim
            nb01,
            nb02: 0,
            nb03: 0,
            ne12: n as i32,       // Input batch dimension
            nb10: 2,              // FP16 element size
            nb11: (k * 2) as u64, // K * sizeof(FP16)
            nb12: (k * 2) as u64, // Batch stride for input: K elements per batch
            nb13: 0,
            ne0: m as i32,
            ne1: n as i32,
            r2: 1,
            r3: 1,
        };

        let weights_size = weights.len() * std::mem::size_of::<BlockQ2K>();

        // Convert FP32 input to FP16
        let input_f16: Vec<u16> =
            input.iter().map(|&x| half::f16::from_f32(x).to_bits()).collect();
        let input_size = input_f16.len() * 2;
        let output_size = m * n * std::mem::size_of::<f32>();

        let weights_buffer = self.device.new_buffer_with_data(
            weights.as_ptr() as *const std::ffi::c_void,
            weights_size as u64,
            metal::MTLResourceOptions::StorageModeShared,
        );
        let input_buffer = self.device.new_buffer_with_data(
            input_f16.as_ptr() as *const std::ffi::c_void,
            input_size as u64,
            metal::MTLResourceOptions::StorageModeShared,
        );
        let output_buffer = self.device.new_buffer(
            output_size as u64,
            metal::MTLResourceOptions::StorageModeShared,
        );

        let command_buffer = self.queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(&self.gemm_template_q2_k_pipeline);
        encoder.set_bytes(
            0,
            std::mem::size_of::<GemmArgs>() as u64,
            &args as *const GemmArgs as *const std::ffi::c_void,
        );
        encoder.set_buffer(1, Some(&weights_buffer), 0);
        encoder.set_buffer(2, Some(&input_buffer), 0);
        encoder.set_buffer(3, Some(&output_buffer), 0);

        encoder.set_threadgroup_memory_length(0, 8192);

        let thread_group_size = MTLSize { width: 128, height: 1, depth: 1 };
        let thread_group_count = MTLSize {
            width: ((n + 31) / 32) as u64,
            height: ((m + 63) / 64) as u64,
            depth: 1,
        };

        encoder.dispatch_thread_groups(thread_group_count, thread_group_size);
        encoder.end_encoding();
        command_buffer.commit();
        command_buffer.wait_until_completed();

        let output_ptr = output_buffer.contents() as *const f32;
        Ok(unsafe { std::slice::from_raw_parts(output_ptr, m * n).to_vec() })
    }

    /// Templated Q3_K × F32 GEMM
    pub fn gemm_template_q3_k_f32(
        &self,
        m: usize,
        n: usize,
        k: usize,
        weights: &[BlockQ3K],
        input: &[f32],
    ) -> Result<Vec<f32>, String> {
        let nb = k / 256;
        let nb01 = (nb * std::mem::size_of::<BlockQ3K>()) as u64;
        let _nb00 = std::mem::size_of::<BlockQ3K>() as u64;

        #[repr(C)]
        struct GemmArgs {
            ne00: i32, // K
            ne02: i32, // batch dim (1 for simple case)
            nb01: u64, // stride for A rows
            nb02: u64, // batch stride (0 for simple case)
            nb03: u64, // batch3 stride (0)
            ne12: i32, // batch dim for B
            nb10: u64, // element stride for B (2 for FP16)
            nb11: u64, // row stride for B (K * 2)
            nb12: u64, // batch stride for input
            nb13: u64, // batch3 stride (0)
            ne0: i32,  // M (output rows)
            ne1: i32,  // N (output cols)
            r2: i16,   // broadcast ratio (1)
            r3: i16,   // broadcast ratio (1)
        }

        let args = GemmArgs {
            ne00: k as i32,
            ne02: 1, // Weights have no batch dim
            nb01,
            nb02: 0,
            nb03: 0,
            ne12: n as i32,       // Input batch dimension
            nb10: 2,              // FP16 element size
            nb11: (k * 2) as u64, // K * sizeof(FP16)
            nb12: (k * 2) as u64, // Batch stride for input: K elements per batch
            nb13: 0,
            ne0: m as i32,
            ne1: n as i32,
            r2: 1,
            r3: 1,
        };

        let weights_size = weights.len() * std::mem::size_of::<BlockQ3K>();

        // Convert FP32 input to FP16
        let input_f16: Vec<u16> =
            input.iter().map(|&x| half::f16::from_f32(x).to_bits()).collect();
        let input_size = input_f16.len() * 2;
        let output_size = m * n * std::mem::size_of::<f32>();

        let weights_buffer = self.device.new_buffer_with_data(
            weights.as_ptr() as *const std::ffi::c_void,
            weights_size as u64,
            metal::MTLResourceOptions::StorageModeShared,
        );
        let input_buffer = self.device.new_buffer_with_data(
            input_f16.as_ptr() as *const std::ffi::c_void,
            input_size as u64,
            metal::MTLResourceOptions::StorageModeShared,
        );
        let output_buffer = self.device.new_buffer(
            output_size as u64,
            metal::MTLResourceOptions::StorageModeShared,
        );

        let command_buffer = self.queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(&self.gemm_template_q3_k_pipeline);
        encoder.set_bytes(
            0,
            std::mem::size_of::<GemmArgs>() as u64,
            &args as *const GemmArgs as *const std::ffi::c_void,
        );
        encoder.set_buffer(1, Some(&weights_buffer), 0);
        encoder.set_buffer(2, Some(&input_buffer), 0);
        encoder.set_buffer(3, Some(&output_buffer), 0);

        encoder.set_threadgroup_memory_length(0, 8192);

        let thread_group_size = MTLSize { width: 128, height: 1, depth: 1 };
        let thread_group_count = MTLSize {
            width: ((n + 31) / 32) as u64,
            height: ((m + 63) / 64) as u64,
            depth: 1,
        };

        encoder.dispatch_thread_groups(thread_group_count, thread_group_size);
        encoder.end_encoding();
        command_buffer.commit();
        command_buffer.wait_until_completed();

        let output_ptr = output_buffer.contents() as *const f32;
        Ok(unsafe { std::slice::from_raw_parts(output_ptr, m * n).to_vec() })
    }

    /// Templated Q5_K × F32 GEMM
    pub fn gemm_template_q5_k_f32(
        &self,
        m: usize,
        n: usize,
        k: usize,
        weights: &[BlockQ5K],
        input: &[f32],
    ) -> Result<Vec<f32>, String> {
        let nb = k / 256;
        let nb01 = (nb * std::mem::size_of::<BlockQ5K>()) as u64;
        let _nb00 = std::mem::size_of::<BlockQ5K>() as u64;

        #[repr(C)]
        struct GemmArgs {
            ne00: i32, // K
            ne02: i32, // batch dim (1 for simple case)
            nb01: u64, // stride for A rows
            nb02: u64, // batch stride (0 for simple case)
            nb03: u64, // batch3 stride (0)
            ne12: i32, // batch dim for B
            nb10: u64, // element stride for B (2 for FP16)
            nb11: u64, // row stride for B (K * 2)
            nb12: u64, // batch stride for input
            nb13: u64, // batch3 stride (0)
            ne0: i32,  // M (output rows)
            ne1: i32,  // N (output cols)
            r2: i16,   // broadcast ratio (1)
            r3: i16,   // broadcast ratio (1)
        }

        let args = GemmArgs {
            ne00: k as i32,
            ne02: 1, // Weights have no batch dim
            nb01,
            nb02: 0,
            nb03: 0,
            ne12: n as i32,       // Input batch dimension
            nb10: 2,              // FP16 element size
            nb11: (k * 2) as u64, // K * sizeof(FP16)
            nb12: (k * 2) as u64, // Batch stride for input: K elements per batch
            nb13: 0,
            ne0: m as i32,
            ne1: n as i32,
            r2: 1,
            r3: 1,
        };

        let weights_size = weights.len() * std::mem::size_of::<BlockQ5K>();

        // Convert FP32 input to FP16
        let input_f16: Vec<u16> =
            input.iter().map(|&x| half::f16::from_f32(x).to_bits()).collect();
        let input_size = input_f16.len() * 2;
        let output_size = m * n * std::mem::size_of::<f32>();

        let weights_buffer = self.device.new_buffer_with_data(
            weights.as_ptr() as *const std::ffi::c_void,
            weights_size as u64,
            metal::MTLResourceOptions::StorageModeShared,
        );
        let input_buffer = self.device.new_buffer_with_data(
            input_f16.as_ptr() as *const std::ffi::c_void,
            input_size as u64,
            metal::MTLResourceOptions::StorageModeShared,
        );
        let output_buffer = self.device.new_buffer(
            output_size as u64,
            metal::MTLResourceOptions::StorageModeShared,
        );

        let command_buffer = self.queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(&self.gemm_template_q5_k_pipeline);
        encoder.set_bytes(
            0,
            std::mem::size_of::<GemmArgs>() as u64,
            &args as *const GemmArgs as *const std::ffi::c_void,
        );
        encoder.set_buffer(1, Some(&weights_buffer), 0);
        encoder.set_buffer(2, Some(&input_buffer), 0);
        encoder.set_buffer(3, Some(&output_buffer), 0);

        encoder.set_threadgroup_memory_length(0, 8192);

        let thread_group_size = MTLSize { width: 128, height: 1, depth: 1 };
        let thread_group_count = MTLSize {
            width: ((n + 31) / 32) as u64,
            height: ((m + 63) / 64) as u64,
            depth: 1,
        };

        encoder.dispatch_thread_groups(thread_group_count, thread_group_size);
        encoder.end_encoding();
        command_buffer.commit();
        command_buffer.wait_until_completed();

        let output_ptr = output_buffer.contents() as *const f32;
        Ok(unsafe { std::slice::from_raw_parts(output_ptr, m * n).to_vec() })
    }

    /// Templated Q6_K × F32 GEMM
    pub fn gemm_template_q6_k_f32(
        &self,
        m: usize,
        n: usize,
        k: usize,
        weights: &[BlockQ6K],
        input: &[f32],
    ) -> Result<Vec<f32>, String> {
        let nb = k / 256;
        let nb01 = (nb * std::mem::size_of::<BlockQ6K>()) as u64;
        let _nb00 = std::mem::size_of::<BlockQ6K>() as u64;

        #[repr(C)]
        struct GemmArgs {
            ne00: i32, // K
            ne02: i32, // batch dim (1 for simple case)
            nb01: u64, // stride for A rows
            nb02: u64, // batch stride (0 for simple case)
            nb03: u64, // batch3 stride (0)
            ne12: i32, // batch dim for B
            nb10: u64, // element stride for B (2 for FP16)
            nb11: u64, // row stride for B (K * 2)
            nb12: u64, // batch stride for input
            nb13: u64, // batch3 stride (0)
            ne0: i32,  // M (output rows)
            ne1: i32,  // N (output cols)
            r2: i16,   // broadcast ratio (1)
            r3: i16,   // broadcast ratio (1)
        }

        let args = GemmArgs {
            ne00: k as i32,
            ne02: 1, // Weights have no batch dim
            nb01,
            nb02: 0,
            nb03: 0,
            ne12: n as i32,       // Input batch dimension
            nb10: 2,              // FP16 element size
            nb11: (k * 2) as u64, // K * sizeof(FP16)
            nb12: (k * 2) as u64, // Batch stride for input: K elements per batch
            nb13: 0,
            ne0: m as i32,
            ne1: n as i32,
            r2: 1,
            r3: 1,
        };

        let weights_size = weights.len() * std::mem::size_of::<BlockQ6K>();

        // Convert FP32 input to FP16
        let input_f16: Vec<u16> =
            input.iter().map(|&x| half::f16::from_f32(x).to_bits()).collect();
        let input_size = input_f16.len() * 2;
        let output_size = m * n * std::mem::size_of::<f32>();

        let weights_buffer = self.device.new_buffer_with_data(
            weights.as_ptr() as *const std::ffi::c_void,
            weights_size as u64,
            metal::MTLResourceOptions::StorageModeShared,
        );
        let input_buffer = self.device.new_buffer_with_data(
            input_f16.as_ptr() as *const std::ffi::c_void,
            input_size as u64,
            metal::MTLResourceOptions::StorageModeShared,
        );
        let output_buffer = self.device.new_buffer(
            output_size as u64,
            metal::MTLResourceOptions::StorageModeShared,
        );

        let command_buffer = self.queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(&self.gemm_template_q6_k_pipeline);
        encoder.set_bytes(
            0,
            std::mem::size_of::<GemmArgs>() as u64,
            &args as *const GemmArgs as *const std::ffi::c_void,
        );
        encoder.set_buffer(1, Some(&weights_buffer), 0);
        encoder.set_buffer(2, Some(&input_buffer), 0);
        encoder.set_buffer(3, Some(&output_buffer), 0);

        encoder.set_threadgroup_memory_length(0, 8192);

        let thread_group_size = MTLSize { width: 128, height: 1, depth: 1 };
        let thread_group_count = MTLSize {
            width: ((n + 31) / 32) as u64,
            height: ((m + 63) / 64) as u64,
            depth: 1,
        };

        encoder.dispatch_thread_groups(thread_group_count, thread_group_size);
        encoder.end_encoding();
        command_buffer.commit();
        command_buffer.wait_until_completed();

        let output_ptr = output_buffer.contents() as *const f32;
        Ok(unsafe { std::slice::from_raw_parts(output_ptr, m * n).to_vec() })
    }

    /// Unified dispatch: uses MV for decode (N=1), GEMM template for batch (N≥32)
    pub fn matmul_q4_k_f32(
        &self,
        m: usize,
        n: usize,
        k: usize,
        weights: &[BlockQ4K],
        input: &[f32],
    ) -> Result<Vec<f32>, String> {
        // Use GEMM template for all batch sizes (reduces dispatch overhead)
        self.gemm_template_q4_k_f32(m, n, k, weights, input)
    }

    pub fn matmul_q2_k_f32(
        &self,
        m: usize,
        n: usize,
        k: usize,
        weights: &[BlockQ2K],
        input: &[f32],
    ) -> Result<Vec<f32>, String> {
        // Use GEMM template for all batch sizes (reduces dispatch overhead)
        self.gemm_template_q2_k_f32(m, n, k, weights, input)
    }

    pub fn matmul_q3_k_f32(
        &self,
        m: usize,
        n: usize,
        k: usize,
        weights: &[BlockQ3K],
        input: &[f32],
    ) -> Result<Vec<f32>, String> {
        // Use GEMM template for all batch sizes (reduces dispatch overhead)
        self.gemm_template_q3_k_f32(m, n, k, weights, input)
    }

    pub fn matmul_q5_k_f32(
        &self,
        m: usize,
        n: usize,
        k: usize,
        weights: &[BlockQ5K],
        input: &[f32],
    ) -> Result<Vec<f32>, String> {
        // Use GEMM template for all batch sizes (reduces dispatch overhead)
        self.gemm_template_q5_k_f32(m, n, k, weights, input)
    }

    pub fn matmul_q6_k_f32(
        &self,
        m: usize,
        n: usize,
        k: usize,
        weights: &[BlockQ6K],
        input: &[f32],
    ) -> Result<Vec<f32>, String> {
        // Use GEMM template for all batch sizes (reduces dispatch overhead)
        self.gemm_template_q6_k_f32(m, n, k, weights, input)
    }

    pub fn matmul_iq4_xs_f32(
        &self,
        m: usize,
        n: usize,
        k: usize,
        weights: &[BlockIQ4XS],
        input: &[f32],
    ) -> Result<Vec<f32>, String> {
        // Use sequential MV for all batch sizes (no GEMM template for IQ4_XS yet)
        let mut result = Vec::with_capacity(n * m);
        for i in 0..n {
            let x_token = &input[i * k..(i + 1) * k];
            let out = self.mv_iq4_xs_f32(m, k, weights, x_token)?;
            result.extend(out);
        }
        Ok(result)
    }

    // ========================================================================
    // RMS Norm
    // ========================================================================

    pub fn rms_norm_f32(
        &self,
        n_rows: usize,
        n_cols: usize,
        src: &[f32],
        weight: &[f32],
        eps: f32,
    ) -> Result<Vec<f32>, String> {
        let src_buffer = self.device.new_buffer_with_data(
            src.as_ptr() as *const std::ffi::c_void,
            (src.len() * std::mem::size_of::<f32>()) as u64,
            metal::MTLResourceOptions::StorageModeShared,
        );
        let weight_buffer = self.device.new_buffer_with_data(
            weight.as_ptr() as *const std::ffi::c_void,
            (weight.len() * std::mem::size_of::<f32>()) as u64,
            metal::MTLResourceOptions::StorageModeShared,
        );
        let output_buffer = self.device.new_buffer(
            (n_rows * n_cols * std::mem::size_of::<f32>()) as u64,
            metal::MTLResourceOptions::StorageModeShared,
        );

        let command_buffer = self.queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();

        let ne00 = n_cols as i32;
        encoder.set_compute_pipeline_state(&self.rms_norm_pipeline);
        encoder.set_buffer(0, Some(&src_buffer), 0);
        encoder.set_buffer(1, Some(&weight_buffer), 0);
        encoder.set_buffer(2, Some(&output_buffer), 0);
        encoder.set_bytes(3, 4, &ne00 as *const i32 as *const std::ffi::c_void);
        encoder.set_bytes(4, 4, &eps as *const f32 as *const std::ffi::c_void);

        let thread_group_size = MTLSize { width: 64, height: 1, depth: 1 };
        let thread_group_count =
            MTLSize { width: n_rows as u64, height: 1, depth: 1 };

        // Allocate threadgroup memory for reduction (32 floats = 128 bytes)
        encoder.set_threadgroup_memory_length(0, 128);
        encoder.dispatch_thread_groups(thread_group_count, thread_group_size);
        encoder.end_encoding();
        command_buffer.commit();
        command_buffer.wait_until_completed();

        let output_ptr = output_buffer.contents() as *const f32;
        Ok(unsafe {
            std::slice::from_raw_parts(output_ptr, n_rows * n_cols).to_vec()
        })
    }

    /// RMS Norm (zero-copy GPU version)
    pub fn rms_norm_f32_gpu(
        &self,
        src_buffer: &Buffer,
        weight_buffer: &Buffer,
        n_rows: usize,
        n_cols: usize,
        eps: f32,
    ) -> Result<std::rc::Rc<Buffer>, String> {
        let output_buffer = std::rc::Rc::new(self.device.new_buffer(
            (n_rows * n_cols * std::mem::size_of::<f32>()) as u64,
            metal::MTLResourceOptions::StorageModeShared,
        ));

        let ne00 = n_cols as i32;
        let command_buffer = self.queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();

        encoder.set_compute_pipeline_state(&self.rms_norm_pipeline);
        encoder.set_buffer(0, Some(src_buffer), 0);
        encoder.set_buffer(1, Some(weight_buffer), 0);
        encoder.set_buffer(2, Some(&output_buffer), 0);
        encoder.set_bytes(3, 4, &ne00 as *const i32 as *const std::ffi::c_void);
        encoder.set_bytes(4, 4, &eps as *const f32 as *const std::ffi::c_void);

        let thread_group_size = MTLSize { width: 64, height: 1, depth: 1 };
        let thread_group_count =
            MTLSize { width: n_rows as u64, height: 1, depth: 1 };

        encoder.set_threadgroup_memory_length(0, 128);
        encoder.dispatch_thread_groups(thread_group_count, thread_group_size);
        encoder.end_encoding();
        command_buffer.commit();
        command_buffer.wait_until_completed();

        Ok(output_buffer)
    }

    /// RMS Norm (batched, encode into shared encoder)
    pub fn rms_norm_encode(
        &self,
        encoder: &ComputeCommandEncoderRef,
        src_buffer: &Buffer,
        weight_buffer: &Buffer,
        output_buffer: &Buffer,
        n_rows: usize,
        n_cols: usize,
        eps: f32,
    ) {
        let ne00 = n_cols as i32;
        encoder.set_compute_pipeline_state(&self.rms_norm_pipeline);
        encoder.set_buffer(0, Some(src_buffer), 0);
        encoder.set_buffer(1, Some(weight_buffer), 0);
        encoder.set_buffer(2, Some(output_buffer), 0);
        encoder.set_bytes(3, 4, &ne00 as *const i32 as *const std::ffi::c_void);
        encoder.set_bytes(4, 4, &eps as *const f32 as *const std::ffi::c_void);

        let thread_group_size = MTLSize { width: 64, height: 1, depth: 1 };
        let thread_group_count =
            MTLSize { width: n_rows as u64, height: 1, depth: 1 };

        encoder.set_threadgroup_memory_length(0, 128);
        encoder.dispatch_thread_groups(thread_group_count, thread_group_size);
    }

    /// Add (batched, encode into shared encoder)
    pub fn add_encode(
        &self,
        encoder: &ComputeCommandEncoderRef,
        a_buffer: &Buffer,
        b_buffer: &Buffer,
        output_buffer: &Buffer,
        n: usize,
    ) {
        encoder.set_compute_pipeline_state(&self.add_pipeline);
        encoder.set_buffer(0, Some(a_buffer), 0);
        encoder.set_buffer(1, Some(b_buffer), 0);
        encoder.set_buffer(2, Some(output_buffer), 0);

        let thread_group_size = MTLSize { width: 256, height: 1, depth: 1 };
        let thread_group_count =
            MTLSize { width: (n as u64 + 255) / 256, height: 1, depth: 1 };

        encoder.dispatch_thread_groups(thread_group_count, thread_group_size);
    }

    /// SiLU (batched, encode into shared encoder)
    pub fn silu_encode(
        &self,
        encoder: &ComputeCommandEncoderRef,
        src_buffer: &Buffer,
        output_buffer: &Buffer,
        n: usize,
    ) {
        encoder.set_compute_pipeline_state(&self.silu_pipeline);
        encoder.set_buffer(0, Some(src_buffer), 0);
        encoder.set_buffer(1, Some(output_buffer), 0);

        let thread_group_size = MTLSize { width: 256, height: 1, depth: 1 };
        let thread_group_count =
            MTLSize { width: (n as u64 + 255) / 256, height: 1, depth: 1 };

        encoder.dispatch_thread_groups(thread_group_count, thread_group_size);
    }

    /// Mul (batched, encode into shared encoder)
    pub fn mul_encode(
        &self,
        encoder: &ComputeCommandEncoderRef,
        a_buffer: &Buffer,
        b_buffer: &Buffer,
        output_buffer: &Buffer,
        n: usize,
    ) {
        encoder.set_compute_pipeline_state(&self.mul_pipeline);
        encoder.set_buffer(0, Some(a_buffer), 0);
        encoder.set_buffer(1, Some(b_buffer), 0);
        encoder.set_buffer(2, Some(output_buffer), 0);

        let thread_group_size = MTLSize { width: 256, height: 1, depth: 1 };
        let thread_group_count =
            MTLSize { width: (n as u64 + 255) / 256, height: 1, depth: 1 };

        encoder.dispatch_thread_groups(thread_group_count, thread_group_size);
    }

    /// FusedSwiGLU: SiLU(gate) * up in single kernel
    pub fn fused_swi_glu_encode(
        &self,
        encoder: &ComputeCommandEncoderRef,
        gate_buffer: &Buffer,
        up_buffer: &Buffer,
        output_buffer: &Buffer,
        n: usize,
    ) {
        encoder.set_compute_pipeline_state(&self.fused_swi_glu_pipeline);
        encoder.set_buffer(0, Some(gate_buffer), 0);
        encoder.set_buffer(1, Some(up_buffer), 0);
        encoder.set_buffer(2, Some(output_buffer), 0);

        let n_u32 = n as u32;
        encoder.set_bytes(
            3,
            std::mem::size_of::<u32>() as u64,
            &n_u32 as *const u32 as *const std::ffi::c_void,
        );

        let thread_group_size = MTLSize { width: 32, height: 1, depth: 1 };
        let thread_group_count =
            MTLSize { width: (n as u64 + 31) / 32, height: 1, depth: 1 };

        encoder.dispatch_thread_groups(thread_group_count, thread_group_size);
    }

    // ========================================================================
    // Type Conversion
    // ========================================================================

    /// F32 to F16 conversion (encode into shared encoder)
    pub fn f32_to_f16_encode(
        &self,
        encoder: &ComputeCommandEncoderRef,
        src_buffer: &Buffer,
        dst_buffer: &Buffer,
        total_size: usize,
    ) {
        encoder.set_compute_pipeline_state(&self.f32_to_f16_pipeline);
        encoder.set_buffer(0, Some(src_buffer), 0);
        encoder.set_buffer(1, Some(dst_buffer), 0);
        let size_u32 = total_size as u32;
        encoder.set_bytes(
            2,
            4,
            &size_u32 as *const u32 as *const std::ffi::c_void,
        );

        let thread_group_size = MTLSize { width: 256, height: 1, depth: 1 };
        let thread_group_count = MTLSize {
            width: (total_size as u64 + 255) / 256,
            height: 1,
            depth: 1,
        };

        encoder.dispatch_thread_groups(thread_group_count, thread_group_size);
    }

    /// F16 to F32 conversion (encode into shared encoder)
    pub fn f16_to_f32_encode(
        &self,
        encoder: &ComputeCommandEncoderRef,
        src_buffer: &Buffer,
        dst_buffer: &Buffer,
        total_size: usize,
    ) {
        encoder.set_compute_pipeline_state(&self.f16_to_f32_pipeline);
        encoder.set_buffer(0, Some(src_buffer), 0);
        encoder.set_buffer(1, Some(dst_buffer), 0);
        let size_u32 = total_size as u32;
        encoder.set_bytes(
            2,
            4,
            &size_u32 as *const u32 as *const std::ffi::c_void,
        );

        let thread_group_size = MTLSize { width: 256, height: 1, depth: 1 };
        let thread_group_count = MTLSize {
            width: (total_size as u64 + 255) / 256,
            height: 1,
            depth: 1,
        };

        encoder.dispatch_thread_groups(thread_group_count, thread_group_size);
    }

    /// F32 to F16 with strided input (for non-contiguous data)
    pub fn f32_to_f16_strided_encode(
        &self,
        encoder: &ComputeCommandEncoderRef,
        src_buffer: &Buffer,
        dst_buffer: &Buffer,
        src_stride: usize,
        dst_stride: usize,
        rows: usize,
        cols: usize,
    ) {
        encoder.set_compute_pipeline_state(&self.f32_to_f16_strided_pipeline);
        encoder.set_buffer(0, Some(src_buffer), 0);
        encoder.set_buffer(1, Some(dst_buffer), 0);
        let src_stride_u32 = src_stride as u32;
        let dst_stride_u32 = dst_stride as u32;
        let rows_u32 = rows as u32;
        let cols_u32 = cols as u32;
        encoder.set_bytes(
            2,
            4,
            &src_stride_u32 as *const u32 as *const std::ffi::c_void,
        );
        encoder.set_bytes(
            3,
            4,
            &dst_stride_u32 as *const u32 as *const std::ffi::c_void,
        );
        encoder.set_bytes(
            4,
            4,
            &rows_u32 as *const u32 as *const std::ffi::c_void,
        );
        encoder.set_bytes(
            5,
            4,
            &cols_u32 as *const u32 as *const std::ffi::c_void,
        );

        // 2D dispatch: [cols, rows]
        let thread_group_size = MTLSize { width: 16, height: 16, depth: 1 };
        let thread_group_count = MTLSize {
            width: ((cols as u64 + 15) / 16),
            height: ((rows as u64 + 15) / 16),
            depth: 1,
        };

        encoder.dispatch_thread_groups(thread_group_count, thread_group_size);
    }

    // ========================================================================
    // Fused Kernels
    // ========================================================================

    /// Fused SwiGLU: output = SiLU(gate) * up
    /// Combines SiLU activation and element-wise multiplication into one kernel
    /// Reduces dispatch overhead and intermediate memory writes
    pub fn fused_swiglu_encode(
        &self,
        encoder: &ComputeCommandEncoderRef,
        gate_buffer: &Buffer,   // Gate projection output
        up_buffer: &Buffer,     // Up projection output
        output_buffer: &Buffer, // Output: SiLU(gate) * up
        n: usize,
    ) {
        // Use float4 kernel for better throughput when size is large enough
        if n >= 256 && n % 4 == 0 {
            encoder.set_compute_pipeline_state(&self.fused_swiglu_4_pipeline);
            encoder.set_buffer(0, Some(gate_buffer), 0);
            encoder.set_buffer(1, Some(up_buffer), 0);
            encoder.set_buffer(2, Some(output_buffer), 0);
            let n_i32 = n as i32;
            encoder.set_bytes(
                3,
                4,
                &n_i32 as *const i32 as *const std::ffi::c_void,
            );

            let thread_group_size = MTLSize { width: 256, height: 1, depth: 1 };
            let thread_group_count = MTLSize {
                width: ((n / 4) as u64 + 255) / 256,
                height: 1,
                depth: 1,
            };

            encoder
                .dispatch_thread_groups(thread_group_count, thread_group_size);
        } else {
            encoder.set_compute_pipeline_state(&self.fused_swiglu_pipeline);
            encoder.set_buffer(0, Some(gate_buffer), 0);
            encoder.set_buffer(1, Some(up_buffer), 0);
            encoder.set_buffer(2, Some(output_buffer), 0);
            let n_i32 = n as i32;
            encoder.set_bytes(
                3,
                4,
                &n_i32 as *const i32 as *const std::ffi::c_void,
            );

            let thread_group_size = MTLSize { width: 256, height: 1, depth: 1 };
            let thread_group_count =
                MTLSize { width: (n as u64 + 255) / 256, height: 1, depth: 1 };

            encoder
                .dispatch_thread_groups(thread_group_count, thread_group_size);
        }
    }

    /// Fused RMSNorm + Linear: output = RMSNorm(input, norm_weight, eps) @ weights
    /// Combines normalization and linear projection into one kernel
    /// Reduces dispatch overhead and avoids intermediate memory writes
    ///
    /// # Arguments
    /// * `input_buffer` - Input vector [K] in F32
    /// * `norm_weight_buffer` - RMSNorm scale weights [K] in F32
    /// * `weights_buffer` - Linear weights [M, K] (F16 or F32 depending on pipeline)
    /// * `output_buffer` - Output vector [M] in F32
    /// * `m` - Output dimension (number of rows)
    /// * `k` - Input dimension (vector size)
    /// * `epsilon` - RMSNorm epsilon
    pub fn fused_norm_linear_f16_encode(
        &self,
        encoder: &ComputeCommandEncoderRef,
        input_buffer: &Buffer,
        norm_weight_buffer: &Buffer,
        weights_buffer: &Buffer,
        output_buffer: &Buffer,
        m: usize,
        k: usize,
        epsilon: f32,
    ) {
        // Use float4 kernel for better memory throughput when K is divisible by 4
        let (pipeline, threadgroup_size) = if k % 4 == 0 && k >= 256 {
            (&self.fused_norm_linear_f16_4_pipeline, 64)
        } else {
            (&self.fused_norm_linear_f16_pipeline, 32)
        };

        encoder.set_compute_pipeline_state(pipeline);
        encoder.set_buffer(0, Some(input_buffer), 0);
        encoder.set_buffer(1, Some(norm_weight_buffer), 0);
        encoder.set_buffer(2, Some(weights_buffer), 0);
        encoder.set_buffer(3, Some(output_buffer), 0);
        encoder.set_bytes(
            4,
            4,
            &(m as u32) as *const u32 as *const std::ffi::c_void,
        );
        encoder.set_bytes(
            5,
            4,
            &(k as u32) as *const u32 as *const std::ffi::c_void,
        );
        encoder.set_bytes(
            6,
            4,
            &epsilon as *const f32 as *const std::ffi::c_void,
        );

        // Threadgroup memory size (K elements as F32) + reduction buffer
        let tg_mem_size = std::cmp::max(k * 4 + 256, 4096);
        encoder.set_threadgroup_memory_length(0, tg_mem_size as u64);

        // One threadgroup per output row
        let thread_group_count =
            MTLSize { width: m as u64, height: 1, depth: 1 };
        let thread_group_size =
            MTLSize { width: threadgroup_size as u64, height: 1, depth: 1 };

        encoder.dispatch_thread_groups(thread_group_count, thread_group_size);
    }

    /// Fused RMSNorm + Linear with F32 weights
    pub fn fused_norm_linear_f32_encode(
        &self,
        encoder: &ComputeCommandEncoderRef,
        input_buffer: &Buffer,
        norm_weight_buffer: &Buffer,
        weights_buffer: &Buffer,
        output_buffer: &Buffer,
        m: usize,
        k: usize,
        epsilon: f32,
    ) {
        encoder
            .set_compute_pipeline_state(&self.fused_norm_linear_f32_pipeline);
        encoder.set_buffer(0, Some(input_buffer), 0);
        encoder.set_buffer(1, Some(norm_weight_buffer), 0);
        encoder.set_buffer(2, Some(weights_buffer), 0);
        encoder.set_buffer(3, Some(output_buffer), 0);
        encoder.set_bytes(
            4,
            4,
            &(m as u32) as *const u32 as *const std::ffi::c_void,
        );
        encoder.set_bytes(
            5,
            4,
            &(k as u32) as *const u32 as *const std::ffi::c_void,
        );
        encoder.set_bytes(
            6,
            4,
            &epsilon as *const f32 as *const std::ffi::c_void,
        );

        // Threadgroup memory size (K elements as F32) + reduction buffer
        let tg_mem_size = std::cmp::max(k * 4 + 256, 4096);
        encoder.set_threadgroup_memory_length(0, tg_mem_size as u64);

        // One threadgroup per output row
        let thread_group_count =
            MTLSize { width: m as u64, height: 1, depth: 1 };
        let thread_group_size = MTLSize { width: 32, height: 1, depth: 1 };

        encoder.dispatch_thread_groups(thread_group_count, thread_group_size);
    }

    /// Fused RMSNorm + Linear with Q4_K quantized weights
    pub fn fused_norm_linear_q4_k_encode(
        &self,
        encoder: &ComputeCommandEncoderRef,
        input_buffer: &Buffer,
        norm_weight_buffer: &Buffer,
        weights_buffer: &Buffer,
        output_buffer: &Buffer,
        m: usize,
        k: usize,
        epsilon: f32,
        nb01: u64,
    ) {
        encoder
            .set_compute_pipeline_state(&self.fused_norm_linear_q4_k_pipeline);
        encoder.set_buffer(0, Some(input_buffer), 0);
        encoder.set_buffer(1, Some(norm_weight_buffer), 0);
        encoder.set_buffer(2, Some(weights_buffer), 0);
        encoder.set_buffer(3, Some(output_buffer), 0);
        encoder.set_bytes(
            4,
            4,
            &(m as u32) as *const u32 as *const std::ffi::c_void,
        );
        encoder.set_bytes(
            5,
            4,
            &(k as u32) as *const u32 as *const std::ffi::c_void,
        );
        encoder.set_bytes(
            6,
            4,
            &epsilon as *const f32 as *const std::ffi::c_void,
        );
        encoder.set_bytes(7, 8, &nb01 as *const u64 as *const std::ffi::c_void);

        // Threadgroup memory size (K elements as F32) + reduction buffer
        let tg_mem_size = std::cmp::max(k * 4 + 256, 4096);
        encoder.set_threadgroup_memory_length(0, tg_mem_size as u64);

        // One threadgroup per output row, 32 threads for SIMD reduction
        let thread_group_count =
            MTLSize { width: m as u64, height: 1, depth: 1 };
        let thread_group_size = MTLSize { width: 32, height: 1, depth: 1 };

        encoder.dispatch_thread_groups(thread_group_count, thread_group_size);
    }

    /// Fused RMSNorm + Linear with Q5_K quantized weights
    pub fn fused_norm_linear_q5_k_encode(
        &self,
        encoder: &ComputeCommandEncoderRef,
        input_buffer: &Buffer,
        norm_weight_buffer: &Buffer,
        weights_buffer: &Buffer,
        output_buffer: &Buffer,
        m: usize,
        k: usize,
        epsilon: f32,
        nb01: u64,
    ) {
        encoder
            .set_compute_pipeline_state(&self.fused_norm_linear_q5_k_pipeline);
        encoder.set_buffer(0, Some(input_buffer), 0);
        encoder.set_buffer(1, Some(norm_weight_buffer), 0);
        encoder.set_buffer(2, Some(weights_buffer), 0);
        encoder.set_buffer(3, Some(output_buffer), 0);
        encoder.set_bytes(
            4,
            4,
            &(m as u32) as *const u32 as *const std::ffi::c_void,
        );
        encoder.set_bytes(
            5,
            4,
            &(k as u32) as *const u32 as *const std::ffi::c_void,
        );
        encoder.set_bytes(
            6,
            4,
            &epsilon as *const f32 as *const std::ffi::c_void,
        );
        encoder.set_bytes(7, 8, &nb01 as *const u64 as *const std::ffi::c_void);

        // Threadgroup memory size (K elements as F32) + reduction buffer
        let tg_mem_size = std::cmp::max(k * 4 + 256, 4096);
        encoder.set_threadgroup_memory_length(0, tg_mem_size as u64);

        // One threadgroup per output row, 32 threads for SIMD reduction
        let thread_group_count =
            MTLSize { width: m as u64, height: 1, depth: 1 };
        let thread_group_size = MTLSize { width: 32, height: 1, depth: 1 };

        encoder.dispatch_thread_groups(thread_group_count, thread_group_size);
    }

    /// Fused RMSNorm + Linear with Q6_K quantized weights
    pub fn fused_norm_linear_q6_k_encode(
        &self,
        encoder: &ComputeCommandEncoderRef,
        input_buffer: &Buffer,
        norm_weight_buffer: &Buffer,
        weights_buffer: &Buffer,
        output_buffer: &Buffer,
        m: usize,
        k: usize,
        epsilon: f32,
        nb01: u64,
    ) {
        encoder
            .set_compute_pipeline_state(&self.fused_norm_linear_q6_k_pipeline);
        encoder.set_buffer(0, Some(input_buffer), 0);
        encoder.set_buffer(1, Some(norm_weight_buffer), 0);
        encoder.set_buffer(2, Some(weights_buffer), 0);
        encoder.set_buffer(3, Some(output_buffer), 0);
        encoder.set_bytes(
            4,
            4,
            &(m as u32) as *const u32 as *const std::ffi::c_void,
        );
        encoder.set_bytes(
            5,
            4,
            &(k as u32) as *const u32 as *const std::ffi::c_void,
        );
        encoder.set_bytes(
            6,
            4,
            &epsilon as *const f32 as *const std::ffi::c_void,
        );
        encoder.set_bytes(7, 8, &nb01 as *const u64 as *const std::ffi::c_void);

        // Threadgroup memory size (K elements as F32) + reduction buffer
        let tg_mem_size = std::cmp::max(k * 4 + 256, 4096);
        encoder.set_threadgroup_memory_length(0, tg_mem_size as u64);

        // One threadgroup per output row, 32 threads for SIMD reduction
        let thread_group_count =
            MTLSize { width: m as u64, height: 1, depth: 1 };
        let thread_group_size = MTLSize { width: 32, height: 1, depth: 1 };

        encoder.dispatch_thread_groups(thread_group_count, thread_group_size);
    }

    /// Split fused QKV buffer into separate Q, K, V buffers
    /// Used for models with merged qkv weights (e.g., Qwen3.8)
    pub fn split_qkv_encode(
        &self,
        encoder: &ComputeCommandEncoderRef,
        fused_qkv_buffer: &Buffer,
        q_out_buffer: &Buffer,
        k_out_buffer: &Buffer,
        v_out_buffer: &Buffer,
        q_size: usize,
        k_size: usize,
        v_size: usize,
    ) {
        encoder.set_compute_pipeline_state(&self.split_qkv_pipeline);
        encoder.set_buffer(0, Some(fused_qkv_buffer), 0);
        encoder.set_buffer(1, Some(q_out_buffer), 0);
        encoder.set_buffer(2, Some(k_out_buffer), 0);
        encoder.set_buffer(3, Some(v_out_buffer), 0);
        encoder.set_bytes(
            4,
            4,
            &(q_size as u32) as *const u32 as *const std::ffi::c_void,
        );
        encoder.set_bytes(
            5,
            4,
            &(k_size as u32) as *const u32 as *const std::ffi::c_void,
        );
        encoder.set_bytes(
            6,
            4,
            &(v_size as u32) as *const u32 as *const std::ffi::c_void,
        );

        let total_size = q_size + k_size + v_size;
        let thread_group_size = MTLSize { width: 256, height: 1, depth: 1 };
        let thread_group_count = MTLSize {
            width: ((total_size as u64 + 255) / 256),
            height: 1,
            depth: 1,
        };

        encoder.dispatch_thread_groups(thread_group_count, thread_group_size);
    }

    // ========================================================================
    // FusedGateUp (gate + up projections in one kernel)
    // ========================================================================

    /// Fused gate and up projections with F16 weights
    /// Computes: gate = W_gate @ input, up = W_up @ input in one kernel
    pub fn fused_gate_up_f16_encode(
        &self,
        encoder: &ComputeCommandEncoderRef,
        input_buffer: &Buffer,
        gate_weights_buffer: &Buffer,
        up_weights_buffer: &Buffer,
        gate_output_buffer: &Buffer,
        up_output_buffer: &Buffer,
        m: usize,
        k: usize,
    ) {
        encoder.set_compute_pipeline_state(&self.fused_gate_up_f16_pipeline);
        encoder.set_buffer(0, Some(input_buffer), 0);
        encoder.set_buffer(1, Some(gate_weights_buffer), 0);
        encoder.set_buffer(2, Some(up_weights_buffer), 0);
        encoder.set_buffer(3, Some(gate_output_buffer), 0);
        encoder.set_buffer(4, Some(up_output_buffer), 0);
        encoder.set_bytes(
            5,
            4,
            &(m as u32) as *const u32 as *const std::ffi::c_void,
        );
        encoder.set_bytes(
            6,
            4,
            &(k as u32) as *const u32 as *const std::ffi::c_void,
        );

        let thread_group_count =
            MTLSize { width: m as u64, height: 1, depth: 1 };
        let thread_group_size = MTLSize { width: 64, height: 1, depth: 1 };
        encoder.dispatch_thread_groups(thread_group_count, thread_group_size);
    }

    /// Fused gate and up projections with Q4_K quantized weights
    pub fn fused_gate_up_q4_k_encode(
        &self,
        encoder: &ComputeCommandEncoderRef,
        input_buffer: &Buffer,
        gate_weights_buffer: &Buffer,
        up_weights_buffer: &Buffer,
        gate_output_buffer: &Buffer,
        up_output_buffer: &Buffer,
        m: usize,
        k: usize,
        gate_nb01: u64,
        up_nb01: u64,
    ) {
        encoder.set_compute_pipeline_state(&self.fused_gate_up_q4_k_pipeline);
        encoder.set_buffer(0, Some(input_buffer), 0);
        encoder.set_buffer(1, Some(gate_weights_buffer), 0);
        encoder.set_buffer(2, Some(up_weights_buffer), 0);
        encoder.set_buffer(3, Some(gate_output_buffer), 0);
        encoder.set_buffer(4, Some(up_output_buffer), 0);
        encoder.set_bytes(
            5,
            4,
            &(m as u32) as *const u32 as *const std::ffi::c_void,
        );
        encoder.set_bytes(
            6,
            4,
            &(k as u32) as *const u32 as *const std::ffi::c_void,
        );
        encoder.set_bytes(
            7,
            8,
            &gate_nb01 as *const u64 as *const std::ffi::c_void,
        );
        encoder.set_bytes(
            8,
            8,
            &up_nb01 as *const u64 as *const std::ffi::c_void,
        );

        let thread_group_count =
            MTLSize { width: m as u64, height: 1, depth: 1 };
        let thread_group_size = MTLSize { width: 64, height: 1, depth: 1 };
        encoder.dispatch_thread_groups(thread_group_count, thread_group_size);
    }

    /// Fused gate and up projections with Q5_K quantized weights
    pub fn fused_gate_up_q5_k_encode(
        &self,
        encoder: &ComputeCommandEncoderRef,
        input_buffer: &Buffer,
        gate_weights_buffer: &Buffer,
        up_weights_buffer: &Buffer,
        gate_output_buffer: &Buffer,
        up_output_buffer: &Buffer,
        m: usize,
        k: usize,
        gate_nb01: u64,
        up_nb01: u64,
    ) {
        encoder.set_compute_pipeline_state(&self.fused_gate_up_q5_k_pipeline);
        encoder.set_buffer(0, Some(input_buffer), 0);
        encoder.set_buffer(1, Some(gate_weights_buffer), 0);
        encoder.set_buffer(2, Some(up_weights_buffer), 0);
        encoder.set_buffer(3, Some(gate_output_buffer), 0);
        encoder.set_buffer(4, Some(up_output_buffer), 0);
        encoder.set_bytes(
            5,
            4,
            &(m as u32) as *const u32 as *const std::ffi::c_void,
        );
        encoder.set_bytes(
            6,
            4,
            &(k as u32) as *const u32 as *const std::ffi::c_void,
        );
        encoder.set_bytes(
            7,
            8,
            &gate_nb01 as *const u64 as *const std::ffi::c_void,
        );
        encoder.set_bytes(
            8,
            8,
            &up_nb01 as *const u64 as *const std::ffi::c_void,
        );

        let thread_group_count =
            MTLSize { width: m as u64, height: 1, depth: 1 };
        let thread_group_size = MTLSize { width: 64, height: 1, depth: 1 };
        encoder.dispatch_thread_groups(thread_group_count, thread_group_size);
    }

    /// Fused gate and up projections with Q6_K quantized weights
    pub fn fused_gate_up_q6_k_encode(
        &self,
        encoder: &ComputeCommandEncoderRef,
        input_buffer: &Buffer,
        gate_weights_buffer: &Buffer,
        up_weights_buffer: &Buffer,
        gate_output_buffer: &Buffer,
        up_output_buffer: &Buffer,
        m: usize,
        k: usize,
        gate_nb01: u64,
        up_nb01: u64,
    ) {
        encoder.set_compute_pipeline_state(&self.fused_gate_up_q6_k_pipeline);
        encoder.set_buffer(0, Some(input_buffer), 0);
        encoder.set_buffer(1, Some(gate_weights_buffer), 0);
        encoder.set_buffer(2, Some(up_weights_buffer), 0);
        encoder.set_buffer(3, Some(gate_output_buffer), 0);
        encoder.set_buffer(4, Some(up_output_buffer), 0);
        encoder.set_bytes(
            5,
            4,
            &(m as u32) as *const u32 as *const std::ffi::c_void,
        );
        encoder.set_bytes(
            6,
            4,
            &(k as u32) as *const u32 as *const std::ffi::c_void,
        );
        encoder.set_bytes(
            7,
            8,
            &gate_nb01 as *const u64 as *const std::ffi::c_void,
        );
        encoder.set_bytes(
            8,
            8,
            &up_nb01 as *const u64 as *const std::ffi::c_void,
        );

        let thread_group_count =
            MTLSize { width: m as u64, height: 1, depth: 1 };
        let thread_group_size = MTLSize { width: 64, height: 1, depth: 1 };
        encoder.dispatch_thread_groups(thread_group_count, thread_group_size);
    }

    /// Unified fused gate-up dispatch that routes to dtype-specific kernel
    pub fn fused_gate_up_encode(
        &self,
        encoder: &ComputeCommandEncoderRef,
        gate_weights_buffer: &Buffer,
        up_weights_buffer: &Buffer,
        input_buffer: &Buffer,
        gate_output_buffer: &Buffer,
        up_output_buffer: &Buffer,
        m: usize,
        k: usize,
        dtype: crate::model::tensor::DataType,
    ) {
        let nb = k / 256; // QK_K = 256

        match dtype {
            crate::model::tensor::DataType::Q4_K => {
                let gate_nb01 = (nb * std::mem::size_of::<BlockQ4K>()) as u64;
                let up_nb01 = (nb * std::mem::size_of::<BlockQ4K>()) as u64;
                self.fused_gate_up_q4_k_encode(
                    encoder,
                    input_buffer,
                    gate_weights_buffer,
                    up_weights_buffer,
                    gate_output_buffer,
                    up_output_buffer,
                    m,
                    k,
                    gate_nb01,
                    up_nb01,
                );
            }
            crate::model::tensor::DataType::Q5_K => {
                let gate_nb01 = (nb * std::mem::size_of::<BlockQ5K>()) as u64;
                let up_nb01 = (nb * std::mem::size_of::<BlockQ5K>()) as u64;
                self.fused_gate_up_q5_k_encode(
                    encoder,
                    input_buffer,
                    gate_weights_buffer,
                    up_weights_buffer,
                    gate_output_buffer,
                    up_output_buffer,
                    m,
                    k,
                    gate_nb01,
                    up_nb01,
                );
            }
            crate::model::tensor::DataType::Q6_K => {
                let gate_nb01 = (nb * std::mem::size_of::<BlockQ6K>()) as u64;
                let up_nb01 = (nb * std::mem::size_of::<BlockQ6K>()) as u64;
                self.fused_gate_up_q6_k_encode(
                    encoder,
                    input_buffer,
                    gate_weights_buffer,
                    up_weights_buffer,
                    gate_output_buffer,
                    up_output_buffer,
                    m,
                    k,
                    gate_nb01,
                    up_nb01,
                );
            }
            _ => {
                eprintln!(
                    "[WARN] FusedGateUp not implemented for dtype {:?}, using separate GEMV",
                    dtype
                );
            }
        }
    }

    // ========================================================================
    // FusedAddNorm (Add + RMSNorm)
    // ========================================================================

    /// Fused Add + RMSNorm kernel
    /// Pattern from vLLM's fused_add_rms_norm
    /// Computes: output = RMSNorm(lhs + rhs, norm_weight, eps)
    ///
    /// # Arguments
    /// * `lhs_buffer` - Left-hand side of add (residual)
    /// * `rhs_buffer` - Right-hand side of add (attention/FFN output)
    /// * `norm_weight_buffer` - RMSNorm scale weights
    /// * `output_buffer` - Output tensor
    /// * `dimension` - Vector dimension
    /// * `epsilon` - RMSNorm epsilon
    pub fn fused_add_norm_encode(
        &self,
        encoder: &ComputeCommandEncoderRef,
        lhs_buffer: &Buffer,
        rhs_buffer: &Buffer,
        norm_weight_buffer: &Buffer,
        output_buffer: &Buffer,
        dimension: usize,
        epsilon: f32,
    ) {
        // Use vectorized kernel for better memory throughput
        let (pipeline, threadgroup_size) =
            if dimension % 4 == 0 && dimension >= 256 {
                (&self.fused_add_norm_f32_4_pipeline, 64)
            } else {
                (&self.fused_add_norm_f32_pipeline, 32)
            };

        encoder.set_compute_pipeline_state(pipeline);
        encoder.set_buffer(0, Some(lhs_buffer), 0);
        encoder.set_buffer(1, Some(rhs_buffer), 0);
        encoder.set_buffer(2, Some(norm_weight_buffer), 0);
        encoder.set_buffer(3, Some(output_buffer), 0);
        encoder.set_bytes(
            4,
            4,
            &(dimension as u32) as *const u32 as *const std::ffi::c_void,
        );
        encoder.set_bytes(
            5,
            4,
            &epsilon as *const f32 as *const std::ffi::c_void,
        );

        // Threadgroup memory for reduction
        let tg_mem_size = std::cmp::max(dimension * 4 + 256, 4096);
        encoder.set_threadgroup_memory_length(0, tg_mem_size as u64);

        // One threadgroup per row
        let thread_group_count = MTLSize { width: 1, height: 1, depth: 1 };
        let thread_group_size =
            MTLSize { width: threadgroup_size as u64, height: 1, depth: 1 };

        encoder.dispatch_thread_groups(thread_group_count, thread_group_size);
    }

    /// Fused Add + RMSNorm with F16 inputs
    pub fn fused_add_norm_f16_encode(
        &self,
        encoder: &ComputeCommandEncoderRef,
        lhs_buffer: &Buffer,
        rhs_buffer: &Buffer,
        norm_weight_buffer: &Buffer,
        output_buffer: &Buffer,
        dimension: usize,
        epsilon: f32,
    ) {
        // Use vectorized kernel for better memory throughput
        let (pipeline, threadgroup_size) =
            if dimension % 4 == 0 && dimension >= 256 {
                (&self.fused_add_norm_f16_4_pipeline, 64)
            } else {
                (&self.fused_add_norm_f16_pipeline, 32)
            };

        encoder.set_compute_pipeline_state(pipeline);
        encoder.set_buffer(0, Some(lhs_buffer), 0);
        encoder.set_buffer(1, Some(rhs_buffer), 0);
        encoder.set_buffer(2, Some(norm_weight_buffer), 0);
        encoder.set_buffer(3, Some(output_buffer), 0);
        encoder.set_bytes(
            4,
            4,
            &(dimension as u32) as *const u32 as *const std::ffi::c_void,
        );
        encoder.set_bytes(
            5,
            4,
            &epsilon as *const f32 as *const std::ffi::c_void,
        );

        // Threadgroup memory for reduction
        let tg_mem_size = std::cmp::max(dimension * 4 + 256, 4096);
        encoder.set_threadgroup_memory_length(0, tg_mem_size as u64);

        // One threadgroup per row
        let thread_group_count = MTLSize { width: 1, height: 1, depth: 1 };
        let thread_group_size =
            MTLSize { width: threadgroup_size as u64, height: 1, depth: 1 };

        encoder.dispatch_thread_groups(thread_group_count, thread_group_size);
    }

    // ========================================================================
    // KV Cache Update (F32 -> F16 with strided write)
    // ========================================================================

    /// Update K cache: Convert F32 to F16 and write to strided position
    /// src: F32 buffer [n_kv_heads * head_dim]
    /// dst: F16 KV cache [n_kv_heads, max_seq, head_dim]
    pub fn kv_update_k_encode(
        &self,
        encoder: &ComputeCommandEncoderRef,
        src_buffer: &Buffer,
        dst_buffer: &Buffer,
        pos: usize,
        max_seq: usize,
        n_kv_heads: usize,
        head_dim: usize,
    ) {
        encoder.set_compute_pipeline_state(&self.kv_update_k_pipeline);
        encoder.set_buffer(0, Some(src_buffer), 0);
        encoder.set_buffer(1, Some(dst_buffer), 0);
        encoder.set_bytes(
            2,
            4,
            &(pos as u32) as *const u32 as *const std::ffi::c_void,
        );
        encoder.set_bytes(
            3,
            4,
            &(max_seq as u32) as *const u32 as *const std::ffi::c_void,
        );
        encoder.set_bytes(
            4,
            4,
            &(n_kv_heads as u32) as *const u32 as *const std::ffi::c_void,
        );
        encoder.set_bytes(
            5,
            4,
            &(head_dim as u32) as *const u32 as *const std::ffi::c_void,
        );

        let total_elements = n_kv_heads * head_dim;
        let thread_group_size = MTLSize { width: 256, height: 1, depth: 1 };
        let thread_group_count = MTLSize {
            width: ((total_elements as u64 + 255) / 256),
            height: 1,
            depth: 1,
        };

        encoder.dispatch_thread_groups(thread_group_count, thread_group_size);
    }

    /// Update V cache: Convert F32 to F16 and write to strided position
    pub fn kv_update_v_encode(
        &self,
        encoder: &ComputeCommandEncoderRef,
        src_buffer: &Buffer,
        dst_buffer: &Buffer,
        pos: usize,
        max_seq: usize,
        n_kv_heads: usize,
        head_dim: usize,
    ) {
        encoder.set_compute_pipeline_state(&self.kv_update_v_pipeline);
        encoder.set_buffer(0, Some(src_buffer), 0);
        encoder.set_buffer(1, Some(dst_buffer), 0);
        encoder.set_bytes(
            2,
            4,
            &(pos as u32) as *const u32 as *const std::ffi::c_void,
        );
        encoder.set_bytes(
            3,
            4,
            &(max_seq as u32) as *const u32 as *const std::ffi::c_void,
        );
        encoder.set_bytes(
            4,
            4,
            &(n_kv_heads as u32) as *const u32 as *const std::ffi::c_void,
        );
        encoder.set_bytes(
            5,
            4,
            &(head_dim as u32) as *const u32 as *const std::ffi::c_void,
        );

        let total_elements = n_kv_heads * head_dim;
        let thread_group_size = MTLSize { width: 256, height: 1, depth: 1 };
        let thread_group_count = MTLSize {
            width: ((total_elements as u64 + 255) / 256),
            height: 1,
            depth: 1,
        };

        encoder.dispatch_thread_groups(thread_group_count, thread_group_size);
    }

    /// Update K cache: Copy F16 to strided position (no conversion)
    /// Used with RoPE F16 output to avoid redundant F32->F16 conversion
    pub fn kv_update_k_f16_encode(
        &self,
        encoder: &ComputeCommandEncoderRef,
        src_buffer: &Buffer,
        dst_buffer: &Buffer,
        pos: usize,
        max_seq: usize,
        n_kv_heads: usize,
        head_dim: usize,
    ) {
        encoder.set_compute_pipeline_state(&self.kv_update_k_f16_pipeline);
        encoder.set_buffer(0, Some(src_buffer), 0);
        encoder.set_buffer(1, Some(dst_buffer), 0);
        encoder.set_bytes(
            2,
            4,
            &(pos as u32) as *const u32 as *const std::ffi::c_void,
        );
        encoder.set_bytes(
            3,
            4,
            &(max_seq as u32) as *const u32 as *const std::ffi::c_void,
        );
        encoder.set_bytes(
            4,
            4,
            &(n_kv_heads as u32) as *const u32 as *const std::ffi::c_void,
        );
        encoder.set_bytes(
            5,
            4,
            &(head_dim as u32) as *const u32 as *const std::ffi::c_void,
        );

        let total_elements = n_kv_heads * head_dim;
        let thread_group_size = MTLSize { width: 256, height: 1, depth: 1 };
        let thread_group_count = MTLSize {
            width: ((total_elements as u64 + 255) / 256),
            height: 1,
            depth: 1,
        };

        encoder.dispatch_thread_groups(thread_group_count, thread_group_size);
    }

    /// Batched KV cache update for prefill (encode into shared encoder)
    /// Update multiple positions at once from batched input
    /// src: [batch, n_kv_heads, head_dim] F16
    /// dst: [n_kv_heads, max_seq, head_dim] F16 KV cache
    pub fn kv_update_batched_f16_encode(
        &self,
        encoder: &ComputeCommandEncoderRef,
        src_buffer: &Buffer,
        dst_buffer: &Buffer,
        start_pos: usize,
        max_seq: usize,
        n_kv_heads: usize,
        head_dim: usize,
        batch: usize,
    ) {
        encoder
            .set_compute_pipeline_state(&self.kv_update_batched_f16_pipeline);
        encoder.set_buffer(0, Some(src_buffer), 0);
        encoder.set_buffer(1, Some(dst_buffer), 0);
        encoder.set_bytes(
            2,
            4,
            &(start_pos as u32) as *const u32 as *const std::ffi::c_void,
        );
        encoder.set_bytes(
            3,
            4,
            &(max_seq as u32) as *const u32 as *const std::ffi::c_void,
        );
        encoder.set_bytes(
            4,
            4,
            &(n_kv_heads as u32) as *const u32 as *const std::ffi::c_void,
        );
        encoder.set_bytes(
            5,
            4,
            &(head_dim as u32) as *const u32 as *const std::ffi::c_void,
        );
        encoder.set_bytes(
            6,
            4,
            &(batch as u32) as *const u32 as *const std::ffi::c_void,
        );

        // 3D dispatch: [head_dim, n_kv_heads, batch]
        let thread_group_size = MTLSize { width: 64, height: 4, depth: 4 };
        let thread_group_count = MTLSize {
            width: ((head_dim as u64 + 63) / 64),
            height: ((n_kv_heads as u64 + 3) / 4),
            depth: ((batch as u64 + 3) / 4),
        };

        encoder.dispatch_thread_groups(thread_group_count, thread_group_size);
    }

    /// Batched KV cache update with strided input (encode into shared encoder)
    /// Used when input is not contiguous (e.g., from GEMM output)
    pub fn kv_update_batched_f16_strided_encode(
        &self,
        encoder: &ComputeCommandEncoderRef,
        src_buffer: &Buffer,
        dst_buffer: &Buffer,
        start_pos: usize,
        max_seq: usize,
        n_kv_heads: usize,
        head_dim: usize,
        batch: usize,
        src_stride: usize,
    ) {
        encoder.set_compute_pipeline_state(
            &self.kv_update_batched_f16_strided_pipeline,
        );
        encoder.set_buffer(0, Some(src_buffer), 0);
        encoder.set_buffer(1, Some(dst_buffer), 0);
        encoder.set_bytes(
            2,
            4,
            &(start_pos as u32) as *const u32 as *const std::ffi::c_void,
        );
        encoder.set_bytes(
            3,
            4,
            &(max_seq as u32) as *const u32 as *const std::ffi::c_void,
        );
        encoder.set_bytes(
            4,
            4,
            &(n_kv_heads as u32) as *const u32 as *const std::ffi::c_void,
        );
        encoder.set_bytes(
            5,
            4,
            &(head_dim as u32) as *const u32 as *const std::ffi::c_void,
        );
        encoder.set_bytes(
            6,
            4,
            &(batch as u32) as *const u32 as *const std::ffi::c_void,
        );
        encoder.set_bytes(
            7,
            4,
            &(src_stride as u32) as *const u32 as *const std::ffi::c_void,
        );

        // 3D dispatch: [head_dim, n_kv_heads, batch]
        let thread_group_size = MTLSize { width: 64, height: 4, depth: 4 };
        let thread_group_count = MTLSize {
            width: ((head_dim as u64 + 63) / 64),
            height: ((n_kv_heads as u64 + 3) / 4),
            depth: ((batch as u64 + 3) / 4),
        };

        encoder.dispatch_thread_groups(thread_group_count, thread_group_size);
    }

    // ========================================================================
    // Softmax
    // ========================================================================

    pub fn softmax_f32_fast(
        &self,
        n_rows: usize,
        n_cols: usize,
        src: &[f32],
    ) -> Result<Vec<f32>, String> {
        let src_buffer = self.device.new_buffer_with_data(
            src.as_ptr() as *const std::ffi::c_void,
            (src.len() * std::mem::size_of::<f32>()) as u64,
            metal::MTLResourceOptions::StorageModeShared,
        );
        let output_buffer = self.device.new_buffer(
            (n_rows * n_cols * std::mem::size_of::<f32>()) as u64,
            metal::MTLResourceOptions::StorageModeShared,
        );

        let command_buffer = self.queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();

        let ne00 = n_cols as i32;
        let ne01 = n_rows as i32;
        encoder.set_compute_pipeline_state(&self.softmax_fast_pipeline);
        encoder.set_bytes(0, 4, &ne00 as *const i32 as *const std::ffi::c_void);
        encoder.set_buffer(1, Some(&src_buffer), 0);
        encoder.set_buffer(2, Some(&output_buffer), 0);
        encoder.set_bytes(3, 4, &ne01 as *const i32 as *const std::ffi::c_void);

        let thread_group_size = MTLSize { width: 32, height: 1, depth: 1 };
        let thread_group_count =
            MTLSize { width: n_rows as u64, height: 1, depth: 1 };

        encoder.set_threadgroup_memory_length(0, 128);
        encoder.dispatch_thread_groups(thread_group_count, thread_group_size);
        encoder.end_encoding();
        command_buffer.commit();
        command_buffer.wait_until_completed();

        let output_ptr = output_buffer.contents() as *const f32;
        Ok(unsafe {
            std::slice::from_raw_parts(output_ptr, n_rows * n_cols).to_vec()
        })
    }

    /// Softmax (zero-copy GPU version)
    pub fn softmax_f32_fast_gpu(
        &self,
        src_buffer: &Buffer,
        n_rows: usize,
        n_cols: usize,
    ) -> Result<std::rc::Rc<Buffer>, String> {
        let output_buffer = std::rc::Rc::new(self.device.new_buffer(
            (n_rows * n_cols * std::mem::size_of::<f32>()) as u64,
            metal::MTLResourceOptions::StorageModeShared,
        ));

        let command_buffer = self.queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();

        let ne00 = n_cols as i32;
        let ne01 = n_rows as i32;
        encoder.set_compute_pipeline_state(&self.softmax_fast_pipeline);
        encoder.set_bytes(0, 4, &ne00 as *const i32 as *const std::ffi::c_void);
        encoder.set_buffer(1, Some(src_buffer), 0);
        encoder.set_buffer(2, Some(&output_buffer), 0);
        encoder.set_bytes(3, 4, &ne01 as *const i32 as *const std::ffi::c_void);

        let thread_group_size = MTLSize { width: 32, height: 1, depth: 1 };
        let thread_group_count =
            MTLSize { width: n_rows as u64, height: 1, depth: 1 };

        encoder.set_threadgroup_memory_length(0, 128);
        encoder.dispatch_thread_groups(thread_group_count, thread_group_size);
        encoder.end_encoding();
        command_buffer.commit();
        command_buffer.wait_until_completed();

        Ok(output_buffer)
    }

    // ========================================================================
    // ROPE
    // ========================================================================

    pub fn rope_neox_f32(
        &self,
        n_seqs: usize,
        n_heads: usize,
        n_dims: usize,
        src: &[f32],
        pos: &[i32],
        freq_base: f32,
        freq_scale: f32,
    ) -> Result<Vec<f32>, String> {
        let src_buffer = self.device.new_buffer_with_data(
            src.as_ptr() as *const std::ffi::c_void,
            (src.len() * std::mem::size_of::<f32>()) as u64,
            metal::MTLResourceOptions::StorageModeShared,
        );
        let pos_buffer = self.device.new_buffer_with_data(
            pos.as_ptr() as *const std::ffi::c_void,
            (pos.len() * std::mem::size_of::<i32>()) as u64,
            metal::MTLResourceOptions::StorageModeShared,
        );
        let output_buffer = self.device.new_buffer(
            (src.len() * std::mem::size_of::<f32>()) as u64,
            metal::MTLResourceOptions::StorageModeShared,
        );

        let args = RopeArgs {
            hidden_dim: n_dims as i32,
            num_heads_seqs: n_heads as i32, // Each sequence has n_heads heads
            num_seqs: n_seqs as i32,
            rotate_dims: n_dims as i32,
            offset: 0,
            freq_base,
            freq_scale,
            ext_factor: 0.0,
            attn_factor: 1.0,
            beta_fast: 32.0,
            beta_slow: 1.0,
            ctx_orig: 0,
        };

        let command_buffer = self.queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();

        encoder.set_compute_pipeline_state(&self.rope_pipeline);
        encoder.set_bytes(
            0,
            std::mem::size_of::<RopeArgs>() as u64,
            &args as *const RopeArgs as *const std::ffi::c_void,
        );
        encoder.set_buffer(1, Some(&src_buffer), 0);
        encoder.set_buffer(2, Some(&pos_buffer), 0);
        encoder.set_buffer(3, Some(&output_buffer), 0);

        let thread_group_size = MTLSize { width: 32, height: 1, depth: 1 };
        let thread_group_count =
            MTLSize { width: n_heads as u64, height: n_seqs as u64, depth: 1 };

        encoder.dispatch_thread_groups(thread_group_count, thread_group_size);
        encoder.end_encoding();
        command_buffer.commit();
        command_buffer.wait_until_completed();

        let output_ptr = output_buffer.contents() as *const f32;
        Ok(unsafe {
            std::slice::from_raw_parts(output_ptr, src.len()).to_vec()
        })
    }

    /// RoPE (zero-copy GPU version)
    pub fn rope_neox_f32_gpu(
        &self,
        src_buffer: &Buffer,
        pos: &[i32],
        n_seqs: usize,
        n_heads: usize,
        n_dims: usize,
        freq_base: f32,
        freq_scale: f32,
    ) -> Result<std::rc::Rc<Buffer>, String> {
        let pos_buffer = self.device.new_buffer_with_data(
            pos.as_ptr() as *const std::ffi::c_void,
            (pos.len() * std::mem::size_of::<i32>()) as u64,
            metal::MTLResourceOptions::StorageModeShared,
        );
        let output_buffer = std::rc::Rc::new(self.device.new_buffer(
            (n_seqs * n_heads * n_dims * std::mem::size_of::<f32>()) as u64,
            metal::MTLResourceOptions::StorageModeShared,
        ));

        let args = RopeArgs {
            hidden_dim: n_dims as i32,
            num_heads_seqs: n_heads as i32, // Each sequence has n_heads heads
            num_seqs: n_seqs as i32,
            rotate_dims: n_dims as i32,
            offset: 0,
            freq_base,
            freq_scale,
            ext_factor: 0.0,
            attn_factor: 1.0,
            beta_fast: 32.0,
            beta_slow: 1.0,
            ctx_orig: 0,
        };

        let command_buffer = self.queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();

        encoder.set_compute_pipeline_state(&self.rope_pipeline);
        encoder.set_bytes(
            0,
            std::mem::size_of::<RopeArgs>() as u64,
            &args as *const RopeArgs as *const std::ffi::c_void,
        );
        encoder.set_buffer(1, Some(src_buffer), 0);
        encoder.set_buffer(2, Some(&pos_buffer), 0);
        encoder.set_buffer(3, Some(&output_buffer), 0);

        let thread_group_size = MTLSize { width: 32, height: 1, depth: 1 };
        let thread_group_count =
            MTLSize { width: n_heads as u64, height: n_seqs as u64, depth: 1 };

        encoder.dispatch_thread_groups(thread_group_count, thread_group_size);
        encoder.end_encoding();
        command_buffer.commit();
        command_buffer.wait_until_completed();

        Ok(output_buffer)
    }

    // ========================================================================
    // Activation Functions
    // ========================================================================

    pub fn silu_f32(&self, n: usize, src: &[f32]) -> Result<Vec<f32>, String> {
        let src_buffer = self.device.new_buffer_with_data(
            src.as_ptr() as *const std::ffi::c_void,
            (src.len() * std::mem::size_of::<f32>()) as u64,
            metal::MTLResourceOptions::StorageModeShared,
        );
        let output_buffer = self.device.new_buffer(
            (n * std::mem::size_of::<f32>()) as u64,
            metal::MTLResourceOptions::StorageModeShared,
        );

        let command_buffer = self.queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();

        encoder.set_compute_pipeline_state(&self.silu_pipeline);
        encoder.set_buffer(0, Some(&src_buffer), 0);
        encoder.set_buffer(1, Some(&output_buffer), 0);

        let thread_group_size = MTLSize { width: 256, height: 1, depth: 1 };
        let thread_group_count = MTLSize {
            width: ((n / 4 + 255) / 256) as u64,
            height: 1,
            depth: 1,
        };

        encoder.dispatch_thread_groups(thread_group_count, thread_group_size);
        encoder.end_encoding();
        command_buffer.commit();
        command_buffer.wait_until_completed();

        let output_ptr = output_buffer.contents() as *const f32;
        Ok(unsafe { std::slice::from_raw_parts(output_ptr, n).to_vec() })
    }

    // ========================================================================
    // Binary Operations (GPU-native)
    // ========================================================================

    /// Add two tensors (zero-copy GPU version)
    pub fn add_f32_gpu(
        &self,
        src0_buffer: &Buffer,
        src1_buffer: &Buffer,
        n: usize,
    ) -> Result<std::rc::Rc<Buffer>, String> {
        let output_buffer = std::rc::Rc::new(self.device.new_buffer(
            (n * std::mem::size_of::<f32>()) as u64,
            metal::MTLResourceOptions::StorageModeShared,
        ));

        let command_buffer = self.queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();

        encoder.set_compute_pipeline_state(&self.add_pipeline);
        encoder.set_buffer(0, Some(src0_buffer), 0);
        encoder.set_buffer(1, Some(src1_buffer), 0);
        encoder.set_buffer(2, Some(&output_buffer), 0);

        let thread_group_size = MTLSize { width: 256, height: 1, depth: 1 };
        let thread_group_count =
            MTLSize { width: ((n + 255) / 256) as u64, height: 1, depth: 1 };

        encoder.dispatch_thread_groups(thread_group_count, thread_group_size);
        encoder.end_encoding();
        command_buffer.commit();
        command_buffer.wait_until_completed();

        Ok(output_buffer)
    }

    /// Multiply two tensors (zero-copy GPU version)
    pub fn mul_f32_gpu(
        &self,
        src0_buffer: &Buffer,
        src1_buffer: &Buffer,
        n: usize,
    ) -> Result<std::rc::Rc<Buffer>, String> {
        let output_buffer = std::rc::Rc::new(self.device.new_buffer(
            (n * std::mem::size_of::<f32>()) as u64,
            metal::MTLResourceOptions::StorageModeShared,
        ));

        let command_buffer = self.queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();

        encoder.set_compute_pipeline_state(&self.mul_pipeline);
        encoder.set_buffer(0, Some(src0_buffer), 0);
        encoder.set_buffer(1, Some(src1_buffer), 0);
        encoder.set_buffer(2, Some(&output_buffer), 0);

        let thread_group_size = MTLSize { width: 256, height: 1, depth: 1 };
        let thread_group_count =
            MTLSize { width: ((n + 255) / 256) as u64, height: 1, depth: 1 };

        encoder.dispatch_thread_groups(thread_group_count, thread_group_size);
        encoder.end_encoding();
        command_buffer.commit();
        command_buffer.wait_until_completed();

        Ok(output_buffer)
    }

    /// RoPE NeoX with F16 output (for KV cache)
    /// Input is F32, output is F16
    pub fn rope_neox_f16_gpu(
        &self,
        src_buffer: &Buffer,
        pos: &[i32],
        n_seqs: usize,
        n_heads: usize,
        n_dims: usize,
        freq_base: f32,
        freq_scale: f32,
    ) -> Result<std::rc::Rc<Buffer>, String> {
        let pos_buffer = self.device.new_buffer_with_data(
            pos.as_ptr() as *const std::ffi::c_void,
            (pos.len() * std::mem::size_of::<i32>()) as u64,
            metal::MTLResourceOptions::StorageModeShared,
        );
        // Output is F16 (half the size of F32)
        let output_size =
            n_seqs * n_heads * n_dims * std::mem::size_of::<u16>();
        let output_buffer = std::rc::Rc::new(self.device.new_buffer(
            output_size as u64,
            metal::MTLResourceOptions::StorageModeShared,
        ));

        let args = RopeArgs {
            hidden_dim: n_dims as i32,
            num_heads_seqs: n_heads as i32,
            num_seqs: n_seqs as i32,
            rotate_dims: n_dims as i32,
            offset: 0,
            freq_base,
            freq_scale,
            ext_factor: 0.0,
            attn_factor: 1.0,
            beta_fast: 32.0,
            beta_slow: 1.0,
            ctx_orig: 0,
        };

        let command_buffer = self.queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();

        encoder.set_compute_pipeline_state(&self.rope_f16_pipeline);
        encoder.set_bytes(
            0,
            std::mem::size_of::<RopeArgs>() as u64,
            &args as *const RopeArgs as *const std::ffi::c_void,
        );
        encoder.set_buffer(1, Some(src_buffer), 0);
        encoder.set_buffer(2, Some(&pos_buffer), 0);
        encoder.set_buffer(3, Some(&output_buffer), 0);

        let thread_group_size = MTLSize { width: 32, height: 1, depth: 1 };
        let thread_group_count =
            MTLSize { width: n_heads as u64, height: n_seqs as u64, depth: 1 };

        encoder.dispatch_thread_groups(thread_group_count, thread_group_size);
        encoder.end_encoding();
        command_buffer.commit();
        command_buffer.wait_until_completed();

        Ok(output_buffer)
    }

    /// SiLU (zero-copy GPU version)
    pub fn silu_f32_gpu(
        &self,
        src_buffer: &Buffer,
        n: usize,
    ) -> Result<std::rc::Rc<Buffer>, String> {
        let output_buffer = std::rc::Rc::new(self.device.new_buffer(
            (n * std::mem::size_of::<f32>()) as u64,
            metal::MTLResourceOptions::StorageModeShared,
        ));

        let command_buffer = self.queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();

        encoder.set_compute_pipeline_state(&self.silu_pipeline);
        encoder.set_buffer(0, Some(src_buffer), 0);
        encoder.set_buffer(1, Some(&output_buffer), 0);

        let thread_group_size = MTLSize { width: 256, height: 1, depth: 1 };
        let thread_group_count = MTLSize {
            width: ((n / 4 + 255) / 256) as u64,
            height: 1,
            depth: 1,
        };

        encoder.dispatch_thread_groups(thread_group_count, thread_group_size);
        encoder.end_encoding();
        command_buffer.commit();
        command_buffer.wait_until_completed();

        Ok(output_buffer)
    }

    pub fn gelu_f32(&self, n: usize, src: &[f32]) -> Result<Vec<f32>, String> {
        let src_buffer = self.device.new_buffer_with_data(
            src.as_ptr() as *const std::ffi::c_void,
            (src.len() * std::mem::size_of::<f32>()) as u64,
            metal::MTLResourceOptions::StorageModeShared,
        );
        let output_buffer = self.device.new_buffer(
            (n * std::mem::size_of::<f32>()) as u64,
            metal::MTLResourceOptions::StorageModeShared,
        );

        let command_buffer = self.queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();

        encoder.set_compute_pipeline_state(&self.gelu_pipeline);
        encoder.set_buffer(0, Some(&src_buffer), 0);
        encoder.set_buffer(1, Some(&output_buffer), 0);

        let thread_group_size = MTLSize { width: 256, height: 1, depth: 1 };
        let thread_group_count = MTLSize {
            width: ((n / 4 + 255) / 256) as u64,
            height: 1,
            depth: 1,
        };

        encoder.dispatch_thread_groups(thread_group_count, thread_group_size);
        encoder.end_encoding();
        command_buffer.commit();
        command_buffer.wait_until_completed();

        let output_ptr = output_buffer.contents() as *const f32;
        Ok(unsafe { std::slice::from_raw_parts(output_ptr, n).to_vec() })
    }

    pub fn gelu_quick_f32(
        &self,
        n: usize,
        src: &[f32],
    ) -> Result<Vec<f32>, String> {
        let src_buffer = self.device.new_buffer_with_data(
            src.as_ptr() as *const std::ffi::c_void,
            (src.len() * std::mem::size_of::<f32>()) as u64,
            metal::MTLResourceOptions::StorageModeShared,
        );
        let output_buffer = self.device.new_buffer(
            (n * std::mem::size_of::<f32>()) as u64,
            metal::MTLResourceOptions::StorageModeShared,
        );

        let command_buffer = self.queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();

        encoder.set_compute_pipeline_state(&self.gelu_quick_pipeline);
        encoder.set_buffer(0, Some(&src_buffer), 0);
        encoder.set_buffer(1, Some(&output_buffer), 0);

        let thread_group_size = MTLSize { width: 256, height: 1, depth: 1 };
        let thread_group_count = MTLSize {
            width: ((n / 4 + 255) / 256) as u64,
            height: 1,
            depth: 1,
        };

        encoder.dispatch_thread_groups(thread_group_count, thread_group_size);
        encoder.end_encoding();
        command_buffer.commit();
        command_buffer.wait_until_completed();

        let output_ptr = output_buffer.contents() as *const f32;
        Ok(unsafe { std::slice::from_raw_parts(output_ptr, n).to_vec() })
    }

    // ========================================================================
    // Flash Attention
    // ========================================================================

    /// Flash Attention for F16 KV cache
    /// Supports: head_dim 32/64/80/128/160, GQA (grouped query attention)
    pub fn flash_attn_f16(
        &self,
        n_seqs: usize,      // batch size
        n_heads: usize,     // number of query heads
        _n_kv_heads: usize, // number of KV heads (for GQA)
        head_dim: usize,    // head dimension (32/64/80/128/160)
        seq_len: usize,     // KV cache length
        q: &[f32],          // query [n_seqs, n_heads, head_dim]
        k: &[u16], // key cache [n_seqs, n_kv_heads, seq_len, head_dim] as F16
        v: &[u16], // value cache [n_seqs, n_kv_heads, seq_len, head_dim] as F16
    ) -> Result<Vec<f32>, String> {
        // Select pipeline based on head_dim
        let pipeline = match head_dim {
            32 => &self.flash_decode_f16_32_pipeline,
            64 => &self.flash_decode_f16_64_pipeline,
            80 => &self.flash_decode_f16_80_pipeline,
            128 => &self.flash_decode_f16_128_pipeline,
            160 => &self.flash_decode_f16_160_pipeline,
            213 => &self.flash_decode_f16_213_pipeline,
            _ => {
                return Err(format!(
                    "Flash Attention does not support head_dim={}, only 32/64/80/128/160/213",
                    head_dim
                ));
            }
        };

        let q_buffer = self.device.new_buffer_with_data(
            q.as_ptr() as *const std::ffi::c_void,
            (q.len() * std::mem::size_of::<f32>()) as u64,
            metal::MTLResourceOptions::StorageModeShared,
        );
        let k_buffer = self.device.new_buffer_with_data(
            k.as_ptr() as *const std::ffi::c_void,
            (k.len() * std::mem::size_of::<u16>()) as u64,
            metal::MTLResourceOptions::StorageModeShared,
        );
        let v_buffer = self.device.new_buffer_with_data(
            v.as_ptr() as *const std::ffi::c_void,
            (v.len() * std::mem::size_of::<u16>()) as u64,
            metal::MTLResourceOptions::StorageModeShared,
        );
        let output_buffer = self.device.new_buffer(
            (q.len() * std::mem::size_of::<f32>()) as u64,
            metal::MTLResourceOptions::StorageModeShared,
        );

        let scale = 1.0f32 / (head_dim as f32).sqrt();
        let seq_k = seq_len as u32;

        let command_buffer = self.queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();

        encoder.set_compute_pipeline_state(pipeline);

        // Buffer 0: seq_k (uint)
        encoder.set_bytes(
            0,
            4,
            &seq_k as *const u32 as *const std::ffi::c_void,
        );

        // Buffer 1: scale (float)
        encoder.set_bytes(
            1,
            4,
            &scale as *const f32 as *const std::ffi::c_void,
        );

        // Buffer 2: query
        encoder.set_buffer(2, Some(&q_buffer), 0);

        // Buffer 3: key_cache
        encoder.set_buffer(3, Some(&k_buffer), 0);

        // Buffer 4: value_cache
        encoder.set_buffer(4, Some(&v_buffer), 0);

        // Buffer 5: output
        encoder.set_buffer(5, Some(&output_buffer), 0);

        // Grid: [1, n_heads, n_seqs]
        let thread_group_size = MTLSize { width: 32, height: 1, depth: 1 };
        let thread_group_count =
            MTLSize { width: 1, height: n_heads as u64, depth: n_seqs as u64 };

        // Threadgroup memory for shared query and scores
        // Threadgroup memory:
        // s_query: float4[32] = 512 bytes (fixed size, kernel uses HEAD_DIM/4 elements)
        // s_scores: float[C] where C varies by HEAD_DIM
        //   HEAD_DIM=32: C=8, s_scores=32 bytes
        //   HEAD_DIM=64: C=16, s_scores=64 bytes
        //   HEAD_DIM=80: C=20, s_scores=80 bytes
        //   HEAD_DIM=128: C=32, s_scores=128 bytes
        //   HEAD_DIM=160: C=40, s_scores=160 bytes
        let c_per_iter = match head_dim {
            32 => 8,
            64 => 16,
            80 => 20,
            128 => 32,
            160 => 40,
            213 => 32, // Specialized kernel uses 32 items per iteration
            _ => 32,
        };
        let shmem_size = 512 + c_per_iter * 4; // s_query (fixed 512) + s_scores (C floats)
        encoder.set_threadgroup_memory_length(0, shmem_size as u64);

        encoder.dispatch_thread_groups(thread_group_count, thread_group_size);
        encoder.end_encoding();
        command_buffer.commit();
        command_buffer.wait_until_completed();

        let output_ptr = output_buffer.contents() as *const f32;
        Ok(unsafe { std::slice::from_raw_parts(output_ptr, q.len()).to_vec() })
    }

    /// Flash Attention for F16 KV cache (fully zero-copy, GPU buffers)
    /// Supports head_dim: 32, 64, 80, 128, 160
    /// For head_dim=128 and seq_len >= 64, uses NWG parallelization
    pub fn flash_attn_f16_gpu(
        &self,
        n_seqs: usize,
        n_heads: usize,
        _n_kv_heads: usize,
        head_dim: usize,
        seq_len: usize,
        kv_stride: usize, // max_seq for KV cache layout
        q_buffer: &Buffer,
        k_buffer: &Buffer,
        v_buffer: &Buffer,
    ) -> Result<std::rc::Rc<Buffer>, String> {
        // Use NWG parallelization for head_dim=128 with longer sequences
        // NWG is beneficial when seq_len >= 64 (2+ iterations per workgroup)
        // For short sequences, single-workgroup sequential is faster
        let use_nwg = head_dim == 128 && seq_len >= 64;
        if use_nwg {
            return self.flash_attn_nwg_f16_gpu(
                n_seqs, n_heads, head_dim, seq_len, kv_stride, q_buffer,
                k_buffer, v_buffer,
            );
        }

        // Use MMA pipeline for head_dim=128 (DISABLED - using standard kernel)
        // TODO: Re-enable MMA when flash_decode_mma_true.metal is fixed
        let _use_mma = head_dim == 128; // Placeholder, not used

        // Select pipeline based on head_dim
        let pipeline = match head_dim {
            32 => &self.flash_decode_f16_32_pipeline,
            64 => &self.flash_decode_f16_64_pipeline,
            80 => &self.flash_decode_f16_80_pipeline,
            128 => &self.flash_decode_f16_128_pipeline,
            160 => &self.flash_decode_f16_160_pipeline,
            213 => &self.flash_decode_f16_213_pipeline,
            _ => {
                return Err(format!(
                    "Flash Attention does not support head_dim={}, only 32/64/80/128/160/213",
                    head_dim
                ));
            }
        };

        // Output buffer: [n_seqs, n_heads, head_dim]
        let output_size =
            n_seqs * n_heads * head_dim * std::mem::size_of::<f32>();
        let output_buffer = std::rc::Rc::new(self.device.new_buffer(
            output_size as u64,
            metal::MTLResourceOptions::StorageModeShared,
        ));

        let scale = 1.0f32 / (head_dim as f32).sqrt();
        let seq_k = seq_len as u32;
        let kv_stride_u32 = kv_stride as u32;

        let command_buffer = self.queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();

        encoder.set_compute_pipeline_state(pipeline);

        // Buffer 0: seq_k (uint)
        encoder.set_bytes(
            0,
            4,
            &seq_k as *const u32 as *const std::ffi::c_void,
        );

        // Buffer 1: kv_stride (uint) - max_seq for KV cache layout
        encoder.set_bytes(
            1,
            4,
            &kv_stride_u32 as *const u32 as *const std::ffi::c_void,
        );

        // Buffer 2: scale (float)
        encoder.set_bytes(
            2,
            4,
            &scale as *const f32 as *const std::ffi::c_void,
        );

        // Buffer 3: query
        encoder.set_buffer(3, Some(q_buffer), 0);

        // Buffer 4: key_cache
        encoder.set_buffer(4, Some(k_buffer), 0);

        // Buffer 5: value_cache
        encoder.set_buffer(5, Some(v_buffer), 0);

        // Buffer 6: output
        encoder.set_buffer(6, Some(&output_buffer), 0);

        // Grid: [1, n_heads, n_seqs]
        // Each simdgroup processes one head
        let thread_group_size = MTLSize {
            width: 32, // 32 threads per simdgroup
            height: 1,
            depth: 1,
        };
        let thread_group_count = MTLSize {
            width: n_heads as u64, // One simdgroup per head
            height: n_seqs as u64,
            depth: 1,
        };

        // Threadgroup memory for shared query and scores
        // s_query: float4[32] = 512 bytes (fixed size, kernel uses HEAD_DIM/4 elements)
        // s_scores: float[C] where C varies by HEAD_DIM
        //   HEAD_DIM=32: C=8, s_scores=32 bytes
        //   HEAD_DIM=64: C=16, s_scores=64 bytes
        //   HEAD_DIM=80: C=20, s_scores=80 bytes
        //   HEAD_DIM=128: C=32, s_scores=128 bytes
        //   HEAD_DIM=160: C=40, s_scores=160 bytes
        let c_per_iter = match head_dim {
            32 => 8,
            64 => 16,
            80 => 20,
            128 => 32,
            160 => 40,
            213 => 32, // Specialized kernel uses 32 items per iteration
            _ => 32,
        };
        let shmem_size = 512 + c_per_iter * 4; // s_query (fixed 512) + s_scores (C floats)
        encoder.set_threadgroup_memory_length(0, shmem_size as u64);

        encoder.dispatch_thread_groups(thread_group_count, thread_group_size);
        encoder.end_encoding();
        command_buffer.commit();
        command_buffer.wait_until_completed();

        Ok(output_buffer)
    }

    /// Flash Attention with NWG parallelization for decode
    /// Uses 32 workgroups per head to parallelize across KV positions
    /// Expected 30-50% speedup for decode
    pub fn flash_attn_nwg_f16_gpu(
        &self,
        n_seqs: usize,
        n_heads: usize,
        head_dim: usize,
        seq_len: usize,
        kv_stride: usize,
        q_buffer: &Buffer,
        k_buffer: &Buffer,
        v_buffer: &Buffer,
    ) -> Result<std::rc::Rc<Buffer>, String> {
        if head_dim != 128 {
            return Err(format!(
                "NWG Flash Attention only supports head_dim=128, got {}",
                head_dim
            ));
        }

        let n_wg = 32; // Number of workgroups per head (matching llama.cpp)
        let scale = 1.0f32 / (head_dim as f32).sqrt();
        let seq_k = seq_len as u32;
        let kv_stride_u32 = kv_stride as u32;
        let n_wg_u32 = n_wg as u32;

        // Buffers for partial results
        // partial_O: [n_wg, n_heads, head_dim]
        let partial_o_size =
            n_wg * n_heads * head_dim * std::mem::size_of::<f32>();
        let partial_o_buffer = std::rc::Rc::new(self.device.new_buffer(
            partial_o_size as u64,
            metal::MTLResourceOptions::StorageModeShared,
        ));
        // partial_S: [n_wg, n_heads]
        let partial_s_size = n_wg * n_heads * std::mem::size_of::<f32>();
        let partial_s_buffer = std::rc::Rc::new(self.device.new_buffer(
            partial_s_size as u64,
            metal::MTLResourceOptions::StorageModeShared,
        ));
        // partial_M: [n_wg, n_heads]
        let partial_m_size = n_wg * n_heads * std::mem::size_of::<f32>();
        let partial_m_buffer = std::rc::Rc::new(self.device.new_buffer(
            partial_m_size as u64,
            metal::MTLResourceOptions::StorageModeShared,
        ));

        // Final output buffer
        let output_size =
            n_seqs * n_heads * head_dim * std::mem::size_of::<f32>();
        let output_buffer = std::rc::Rc::new(self.device.new_buffer(
            output_size as u64,
            metal::MTLResourceOptions::StorageModeShared,
        ));

        // Pass 1: Compute partial results
        let command_buffer = self.queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();

        encoder
            .set_compute_pipeline_state(&self.flash_decode_nwg_pass1_pipeline);

        // Buffer 0: seq_k
        encoder.set_bytes(
            0,
            4,
            &seq_k as *const u32 as *const std::ffi::c_void,
        );
        // Buffer 1: kv_stride
        encoder.set_bytes(
            1,
            4,
            &kv_stride_u32 as *const u32 as *const std::ffi::c_void,
        );
        // Buffer 2: scale
        encoder.set_bytes(
            2,
            4,
            &scale as *const f32 as *const std::ffi::c_void,
        );
        // Buffer 3: query
        encoder.set_buffer(3, Some(q_buffer), 0);
        // Buffer 4: key_cache
        encoder.set_buffer(4, Some(k_buffer), 0);
        // Buffer 5: value_cache
        encoder.set_buffer(5, Some(v_buffer), 0);
        // Buffer 6: partial_O
        encoder.set_buffer(6, Some(&partial_o_buffer), 0);
        // Buffer 7: partial_S
        encoder.set_buffer(7, Some(&partial_s_buffer), 0);
        // Buffer 8: partial_M
        encoder.set_buffer(8, Some(&partial_m_buffer), 0);
        // Buffer 9: n_wg
        encoder.set_bytes(
            9,
            4,
            &n_wg_u32 as *const u32 as *const std::ffi::c_void,
        );
        // Buffer 10: n_heads
        let n_heads_u32 = n_heads as u32;
        encoder.set_bytes(
            10,
            4,
            &n_heads_u32 as *const u32 as *const std::ffi::c_void,
        );

        // Grid: [n_heads, n_seqs, n_wg]
        let thread_group_size = MTLSize { width: 32, height: 1, depth: 1 };
        let thread_group_count = MTLSize {
            width: n_heads as u64,
            height: n_seqs as u64,
            depth: n_wg as u64,
        };

        encoder.dispatch_thread_groups(thread_group_count, thread_group_size);
        encoder.end_encoding();

        // Pass 2: Reduce partial results
        let encoder2 = command_buffer.new_compute_command_encoder();
        encoder2
            .set_compute_pipeline_state(&self.flash_decode_nwg_pass2_pipeline);

        // Buffer 0: partial_O
        encoder2.set_buffer(0, Some(&partial_o_buffer), 0);
        // Buffer 1: partial_S
        encoder2.set_buffer(1, Some(&partial_s_buffer), 0);
        // Buffer 2: partial_M
        encoder2.set_buffer(2, Some(&partial_m_buffer), 0);
        // Buffer 3: output
        encoder2.set_buffer(3, Some(&output_buffer), 0);
        // Buffer 4: n_wg
        encoder2.set_bytes(
            4,
            4,
            &n_wg_u32 as *const u32 as *const std::ffi::c_void,
        );
        // Buffer 5: n_heads
        encoder2.set_bytes(
            5,
            4,
            &n_heads_u32 as *const u32 as *const std::ffi::c_void,
        );

        // Grid: [n_heads, n_seqs, 1]
        let thread_group_count2 =
            MTLSize { width: n_heads as u64, height: n_seqs as u64, depth: 1 };

        encoder2.dispatch_thread_groups(thread_group_count2, thread_group_size);
        encoder2.end_encoding();

        command_buffer.commit();
        command_buffer.wait_until_completed();

        Ok(output_buffer)
    }

    // ========================================================================
    // Quantize
    // ========================================================================

    /// Quantize FP32 to F16
    pub fn quantize_f16(
        &self,
        n: usize,
        src: &[f32],
    ) -> Result<Vec<u16>, String> {
        let src_buffer = self.device.new_buffer_with_data(
            src.as_ptr() as *const std::ffi::c_void,
            (src.len() * std::mem::size_of::<f32>()) as u64,
            metal::MTLResourceOptions::StorageModeShared,
        );
        let output_buffer = self.device.new_buffer(
            (n * std::mem::size_of::<u16>()) as u64,
            metal::MTLResourceOptions::StorageModeShared,
        );

        let n_val = n as i32;
        let command_buffer = self.queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();

        // Use vectorized kernel for better performance
        encoder.set_compute_pipeline_state(&self.quantize_f16_4_pipeline);
        encoder.set_buffer(0, Some(&src_buffer), 0);
        encoder.set_buffer(1, Some(&output_buffer), 0);
        encoder.set_bytes(
            2,
            4,
            &n_val as *const i32 as *const std::ffi::c_void,
        );

        let thread_group_size = MTLSize { width: 256, height: 1, depth: 1 };
        let thread_group_count = MTLSize {
            width: ((n / 4 + 255) / 256) as u64,
            height: 1,
            depth: 1,
        };

        encoder.dispatch_thread_groups(thread_group_count, thread_group_size);
        encoder.end_encoding();
        command_buffer.commit();
        command_buffer.wait_until_completed();

        let output_ptr = output_buffer.contents() as *const u16;
        Ok(unsafe { std::slice::from_raw_parts(output_ptr, n).to_vec() })
    }

    /// Quantize FP32 to Q4_0
    /// Each block has 32 elements
    pub fn quantize_q4_0(
        &self,
        n: usize,
        src: &[f32],
    ) -> Result<Vec<BlockQ4_0>, String> {
        if n % 32 != 0 {
            return Err(
                "Q4_0 quantization requires n to be multiple of 32".to_string()
            );
        }

        let nb = n / 32;
        let src_buffer = self.device.new_buffer_with_data(
            src.as_ptr() as *const std::ffi::c_void,
            (src.len() * std::mem::size_of::<f32>()) as u64,
            metal::MTLResourceOptions::StorageModeShared,
        );
        let output_buffer = self.device.new_buffer(
            (nb * std::mem::size_of::<BlockQ4_0>()) as u64,
            metal::MTLResourceOptions::StorageModeShared,
        );

        let n_val = n as i32;
        let command_buffer = self.queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();

        encoder.set_compute_pipeline_state(&self.quantize_q4_0_pipeline);
        encoder.set_buffer(0, Some(&src_buffer), 0);
        encoder.set_buffer(1, Some(&output_buffer), 0);
        encoder.set_bytes(
            2,
            4,
            &n_val as *const i32 as *const std::ffi::c_void,
        );

        let thread_group_size = MTLSize { width: 256, height: 1, depth: 1 };
        let thread_group_count =
            MTLSize { width: nb as u64, height: 1, depth: 1 };

        encoder.dispatch_thread_groups(thread_group_count, thread_group_size);
        encoder.end_encoding();
        command_buffer.commit();
        command_buffer.wait_until_completed();

        let output_ptr = output_buffer.contents() as *const BlockQ4_0;
        Ok(unsafe { std::slice::from_raw_parts(output_ptr, nb).to_vec() })
    }

    /// Quantize FP32 to Q8_0
    /// Each block has 32 elements
    pub fn quantize_q8_0(
        &self,
        n: usize,
        src: &[f32],
    ) -> Result<Vec<BlockQ8_0>, String> {
        if n % 32 != 0 {
            return Err(
                "Q8_0 quantization requires n to be multiple of 32".to_string()
            );
        }

        let nb = n / 32;
        let src_buffer = self.device.new_buffer_with_data(
            src.as_ptr() as *const std::ffi::c_void,
            (src.len() * std::mem::size_of::<f32>()) as u64,
            metal::MTLResourceOptions::StorageModeShared,
        );
        let output_buffer = self.device.new_buffer(
            (nb * std::mem::size_of::<BlockQ8_0>()) as u64,
            metal::MTLResourceOptions::StorageModeShared,
        );

        let n_val = n as i32;
        let command_buffer = self.queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();

        encoder.set_compute_pipeline_state(&self.quantize_q8_0_pipeline);
        encoder.set_buffer(0, Some(&src_buffer), 0);
        encoder.set_buffer(1, Some(&output_buffer), 0);
        encoder.set_bytes(
            2,
            4,
            &n_val as *const i32 as *const std::ffi::c_void,
        );

        let thread_group_size = MTLSize { width: 256, height: 1, depth: 1 };
        let thread_group_count =
            MTLSize { width: nb as u64, height: 1, depth: 1 };

        encoder.dispatch_thread_groups(thread_group_count, thread_group_size);
        encoder.end_encoding();
        command_buffer.commit();
        command_buffer.wait_until_completed();

        let output_ptr = output_buffer.contents() as *const BlockQ8_0;
        Ok(unsafe { std::slice::from_raw_parts(output_ptr, nb).to_vec() })
    }

    // ========================================================================
    // ArgSort (simplified for sampling)
    // ========================================================================

    /// Get indices for top-k elements (simplified version)
    /// Returns indices of sorted elements in descending order
    pub fn argsort_top_k(
        &self,
        _n: usize,
        src: &[f32],
        k: usize,
    ) -> Result<Vec<i32>, String> {
        // For small sizes, use CPU version for simplicity
        // GPU version would need bitonic sort or similar
        let mut indexed: Vec<(f32, i32)> =
            src.iter().enumerate().map(|(i, &v)| (v, i as i32)).collect();
        indexed.sort_by(|a, b| {
            b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal)
        });
        Ok(indexed.iter().take(k).map(|(_, i)| *i).collect())
    }

    // ========================================================================
    // Binary Operations with Broadcasting
    // ========================================================================

    /// Add two tensors with broadcasting
    pub fn add_f32(
        &self,
        n: usize,
        src0: &[f32],
        src1: &[f32],
    ) -> Result<Vec<f32>, String> {
        let mut dst = vec![0.0f32; n];
        for i in 0..n {
            dst[i] = src0[i] + src1[i];
        }
        Ok(dst)
    }

    /// Subtract two tensors with broadcasting
    pub fn sub_f32(
        &self,
        n: usize,
        src0: &[f32],
        src1: &[f32],
    ) -> Result<Vec<f32>, String> {
        let mut dst = vec![0.0f32; n];
        for i in 0..n {
            dst[i] = src0[i] - src1[i];
        }
        Ok(dst)
    }

    /// Multiply two tensors with broadcasting
    pub fn mul_f32(
        &self,
        n: usize,
        src0: &[f32],
        src1: &[f32],
    ) -> Result<Vec<f32>, String> {
        let mut dst = vec![0.0f32; n];
        for i in 0..n {
            dst[i] = src0[i] * src1[i];
        }
        Ok(dst)
    }

    /// Divide two tensors with broadcasting
    pub fn div_f32(
        &self,
        n: usize,
        src0: &[f32],
        src1: &[f32],
    ) -> Result<Vec<f32>, String> {
        let mut dst = vec![0.0f32; n];
        for i in 0..n {
            dst[i] = src0[i] / src1[i];
        }
        Ok(dst)
    }

    // ========================================================================
    // Reduce Operations
    // ========================================================================

    /// Reduce sum
    pub fn reduce_sum_f32(
        &self,
        _n: usize,
        src: &[f32],
    ) -> Result<f32, String> {
        Ok(src.iter().sum())
    }

    /// Reduce max
    pub fn reduce_max_f32(
        &self,
        _n: usize,
        src: &[f32],
    ) -> Result<f32, String> {
        let max_val = src.iter().cloned().fold(-f32::INFINITY, f32::max);
        Ok(max_val)
    }

    /// Row-wise reduce sum
    pub fn reduce_sum_f32_row(
        &self,
        n_rows: usize,
        n_cols: usize,
        src: &[f32],
    ) -> Result<Vec<f32>, String> {
        let mut dst = vec![0.0f32; n_rows];
        for i in 0..n_rows {
            let row_start = i * n_cols;
            dst[i] = src[row_start..row_start + n_cols].iter().sum();
        }
        Ok(dst)
    }

    // ========================================================================
    // SSM (Mamba) Operations
    // ========================================================================

    /// Selective scan for Mamba/SSM
    /// u: input [d_model, seq_len]
    /// delta: time step [d_model, seq_len]
    /// A, B, C, D: SSM parameters
    /// Returns output [d_model, seq_len]
    pub fn ssm_scan_f32(
        &self,
        d_model: usize,
        d_state: usize,
        seq_len: usize,
        u: &[f32],
        delta: &[f32],
        a: &[f32],
        b: &[f32],
        c: &[f32],
        d: &[f32],
    ) -> Result<Vec<f32>, String> {
        // CPU implementation for correctness
        let mut y = vec![0.0f32; d_model * seq_len];

        for ch in 0..d_model {
            let mut h = 0.0f32;
            let d_val = d[ch];

            for t in 0..seq_len {
                let u_t = u[ch * seq_len + t];
                let delta_t = delta[ch * seq_len + t];

                // Discretize A (simplified diagonal)
                let da = (-delta_t * a[0]).exp();

                // Accumulate B
                let b_t: f32 = (0..d_state).map(|s| b[s * seq_len + t]).sum();
                h = da * h + delta_t * b_t * u_t;

                // Accumulate C
                let c_t: f32 = (0..d_state).map(|s| c[s * seq_len + t]).sum();
                y[ch * seq_len + t] = c_t * h + d_val * u_t;
            }
        }

        Ok(y)
    }

    // ========================================================================
    // WKV (RWKV) Operations
    // ========================================================================

    /// WKV time mixing for RWKV
    /// k, v, r: [seq_len, d_model]
    /// w, u: time decay and bonus [d_model]
    /// Returns output [seq_len, d_model]
    pub fn wkv_f32(
        &self,
        seq_len: usize,
        d_model: usize,
        k: &[f32],
        v: &[f32],
        r: &[f32],
        w: &[f32],
        u: &[f32],
    ) -> Result<Vec<f32>, String> {
        let mut y = vec![0.0f32; seq_len * d_model];

        for ch in 0..d_model {
            let mut wkv_n = 0.0f32;
            let mut wkv_d = 1.0f32;

            let w_val = (-w[ch].exp()).exp();
            let u_val = u[ch].exp();

            for t in 0..seq_len {
                let k_t = k[t * d_model + ch];
                let v_t = v[t * d_model + ch];
                let r_t = r[t * d_model + ch];

                let wkv = (wkv_n + u_val * k_t * v_t) / (wkv_d + u_val * k_t);
                y[t * d_model + ch] = r_t * wkv;

                wkv_n = w_val * wkv_n + w_val * k_t * v_t;
                wkv_d = w_val * wkv_d + w_val * k_t;
            }
        }

        Ok(y)
    }

    // ========================================================================
    // CNN Operations (for multimodal models)
    // ========================================================================

    /// 2D Convolution (simplified)
    /// input: [batch, in_channels, in_h, in_w]
    /// weight: [out_channels, in_channels, kH, kW]
    /// bias: [out_channels]
    /// Returns [batch, out_channels, out_h, out_w]
    pub fn conv2d_f32(
        &self,
        batch_size: usize,
        in_channels: usize,
        out_channels: usize,
        in_h: usize,
        in_w: usize,
        kH: usize,
        kW: usize,
        stride_h: usize,
        stride_w: usize,
        pad_h: usize,
        pad_w: usize,
        input: &[f32],
        weight: &[f32],
        bias: &[f32],
    ) -> Result<Vec<f32>, String> {
        let out_h_size = (in_h + 2 * pad_h - kH) / stride_h + 1;
        let out_w_size = (in_w + 2 * pad_w - kW) / stride_w + 1;
        let mut output =
            vec![0.0f32; batch_size * out_channels * out_h_size * out_w_size];

        for batch in 0..batch_size {
            for out_ch in 0..out_channels {
                for out_h_idx in 0..out_h_size {
                    for out_w_idx in 0..out_w_size {
                        let mut sum = bias[out_ch];

                        for in_ch in 0..in_channels {
                            for kh in 0..kH {
                                for kw in 0..kW {
                                    let in_h_idx =
                                        out_h_idx * stride_h - pad_h + kh;
                                    let in_w_idx =
                                        out_w_idx * stride_w - pad_w + kw;

                                    if in_h_idx < in_h && in_w_idx < in_w {
                                        let input_idx =
                                            batch * in_channels * in_h * in_w
                                                + in_ch * in_h * in_w
                                                + in_h_idx * in_w
                                                + in_w_idx;
                                        let weight_idx =
                                            out_ch * in_channels * kH * kW
                                                + in_ch * kH * kW
                                                + kh * kW
                                                + kw;
                                        sum += input[input_idx]
                                            * weight[weight_idx];
                                    }
                                }
                            }
                        }

                        let output_idx =
                            batch * out_channels * out_h_size * out_w_size
                                + out_ch * out_h_size * out_w_size
                                + out_h_idx * out_w_size
                                + out_w_idx;
                        output[output_idx] = sum;
                    }
                }
            }
        }

        Ok(output)
    }

    /// Max Pooling 2D
    pub fn pool2d_max_f32(
        &self,
        batch_size: usize,
        channels: usize,
        in_h: usize,
        in_w: usize,
        kH: usize,
        kW: usize,
        stride_h: usize,
        stride_w: usize,
        input: &[f32],
    ) -> Result<Vec<f32>, String> {
        let out_h_size = (in_h - kH) / stride_h + 1;
        let out_w_size = (in_w - kW) / stride_w + 1;
        let mut output =
            vec![0.0f32; batch_size * channels * out_h_size * out_w_size];

        for batch in 0..batch_size {
            for ch in 0..channels {
                for out_h_idx in 0..out_h_size {
                    for out_w_idx in 0..out_w_size {
                        let mut max_val = f32::NEG_INFINITY;

                        for kh in 0..kH {
                            for kw in 0..kW {
                                let in_h_idx = out_h_idx * stride_h + kh;
                                let in_w_idx = out_w_idx * stride_w + kw;

                                let input_idx = batch * channels * in_h * in_w
                                    + ch * in_h * in_w
                                    + in_h_idx * in_w
                                    + in_w_idx;
                                max_val = max_val.max(input[input_idx]);
                            }
                        }

                        let output_idx =
                            batch * channels * out_h_size * out_w_size
                                + ch * out_h_size * out_w_size
                                + out_h_idx * out_w_size
                                + out_w_idx;
                        output[output_idx] = max_val;
                    }
                }
            }
        }

        Ok(output)
    }

    /// Nearest neighbor upscale
    pub fn upscale_nearest_f32(
        &self,
        batch_size: usize,
        channels: usize,
        in_h: usize,
        in_w: usize,
        scale_factor: usize,
        input: &[f32],
    ) -> Result<Vec<f32>, String> {
        let out_h = in_h * scale_factor;
        let out_w = in_w * scale_factor;
        let mut output = vec![0.0f32; batch_size * channels * out_h * out_w];

        for batch in 0..batch_size {
            for ch in 0..channels {
                for out_h_idx in 0..out_h {
                    for out_w_idx in 0..out_w {
                        let in_h_idx = out_h_idx / scale_factor;
                        let in_w_idx = out_w_idx / scale_factor;

                        let input_idx = batch * channels * in_h * in_w
                            + ch * in_h * in_w
                            + in_h_idx * in_w
                            + in_w_idx;
                        let output_idx = batch * channels * out_h * out_w
                            + ch * out_h * out_w
                            + out_h_idx * out_w
                            + out_w_idx;

                        output[output_idx] = input[input_idx];
                    }
                }
            }
        }

        Ok(output)
    }

    // ========================================================================
    // SSM (State Space Model) Operators
    // ========================================================================

    /// SSM Conv1D: 1D convolution for state space models
    /// Input x: [d_inner, n_tok + d_conv - 1]
    /// Weight w: [d_inner, d_conv]
    /// Bias: [d_inner] (optional)
    /// Output y: [d_inner, n_tok]
    pub fn ssm_conv1d_d4(
        &self,
        x: &[f32],
        w: &[f32],
        bias: Option<&[f32]>,
        d_inner: usize,
        n_tok: usize,
    ) -> Result<Vec<f32>, String> {
        let d_conv = 4;
        let x_size = d_inner * (n_tok + d_conv - 1);
        let w_size = d_inner * d_conv;

        if x.len() != x_size {
            return Err(format!(
                "SSM Conv1D: x size mismatch, expected {}, got {}",
                x_size,
                x.len()
            ));
        }
        if w.len() != w_size {
            return Err(format!(
                "SSM Conv1D: w size mismatch, expected {}, got {}",
                w_size,
                w.len()
            ));
        }
        if let Some(b) = bias {
            if b.len() != d_inner {
                return Err(format!(
                    "SSM Conv1D: bias size mismatch, expected {}, got {}",
                    d_inner,
                    b.len()
                ));
            }
        }

        // Create buffers
        let x_buffer = self.device.new_buffer_with_data(
            x.as_ptr() as *const c_void,
            (x_size * std::mem::size_of::<f32>()) as u64,
            metal::MTLResourceOptions::StorageModeShared,
        );

        let w_buffer = self.device.new_buffer_with_data(
            w.as_ptr() as *const c_void,
            (w_size * std::mem::size_of::<f32>()) as u64,
            metal::MTLResourceOptions::StorageModeShared,
        );

        let bias_buffer = if let Some(b) = bias {
            Some(self.device.new_buffer_with_data(
                b.as_ptr() as *const c_void,
                (d_inner * std::mem::size_of::<f32>()) as u64,
                metal::MTLResourceOptions::StorageModeShared,
            ))
        } else {
            None
        };

        let y_size = d_inner * n_tok;
        let y_buffer = self.device.new_buffer(
            (y_size * std::mem::size_of::<f32>()) as u64,
            metal::MTLResourceOptions::StorageModeShared,
        );

        // Create command buffer and encoder
        let command_buffer = self.queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();

        encoder.set_compute_pipeline_state(&self.ssm_conv1d_d4_pipeline);
        encoder.set_buffer(0, Some(&x_buffer), 0);
        encoder.set_buffer(1, Some(&w_buffer), 0);
        if let Some(ref b) = bias_buffer {
            encoder.set_buffer(2, Some(b), 0);
        } else {
            encoder.set_buffer(2, None, 0);
        }
        encoder.set_buffer(3, Some(&y_buffer), 0);
        let d_inner_u32 = d_inner as u32;
        let n_tok_u32 = n_tok as u32;
        encoder.set_bytes(
            4,
            std::mem::size_of::<u32>() as u64,
            &d_inner_u32 as *const u32 as *const c_void,
        );
        encoder.set_bytes(
            5,
            std::mem::size_of::<u32>() as u64,
            &n_tok_u32 as *const u32 as *const c_void,
        );

        // Use batched kernel for prefill (n_tok > 1)
        if n_tok > 1 {
            // Determine batch size (power of 2, up to 256)
            let batch_size = if n_tok > 128 {
                256u16
            } else if n_tok > 64 {
                128u16
            } else if n_tok > 32 {
                64u16
            } else if n_tok > 16 {
                32u16
            } else if n_tok > 8 {
                16u16
            } else if n_tok > 4 {
                8u16
            } else {
                2u16
            };

            let n_token_batches =
                (n_tok + batch_size as usize - 1) / batch_size as usize;

            // Use specialized pipeline for this batch size
            encoder.set_compute_pipeline_state(
                self.ssm_conv1d_d4_batched_pipeline(batch_size as usize),
            );

            // Grid: (d_inner rows, token batches, 1 sequence)
            let grid_size = MTLSize {
                width: d_inner as u64,
                height: n_token_batches as u64,
                depth: 1,
            };
            let threadgroup_size =
                MTLSize { width: batch_size as u64, height: 1, depth: 1 };

            encoder.dispatch_thread_groups(grid_size, threadgroup_size);
        } else {
            // Small case: use non-batched kernel
            encoder.set_compute_pipeline_state(&self.ssm_conv1d_d4_pipeline);

            let grid_size = MTLSize {
                width: n_tok as u64,
                height: d_inner as u64,
                depth: 1,
            };
            let threadgroup_size = MTLSize { width: 16, height: 16, depth: 1 };

            encoder.dispatch_threads(grid_size, threadgroup_size);
        }
        encoder.end_encoding();

        command_buffer.commit();
        command_buffer.wait_until_completed();

        // Read result
        let y_ptr = y_buffer.contents() as *const f32;
        let y_slice = unsafe { std::slice::from_raw_parts(y_ptr, y_size) };
        Ok(y_slice.to_vec())
    }

    /// SSM Scan: Recurrent state update for state space models
    /// Input x: [d_inner, n_tok]
    /// dt: [d_inner, n_tok]
    /// A: [d_inner, d_state]
    /// B: [d_state, n_tok]
    /// C: [d_state, n_tok]
    /// s0: [d_inner, d_state] (initial state, optional)
    /// Output y: [d_inner, n_tok], s: [d_inner, d_state]
    pub fn ssm_scan_d16(
        &self,
        x: &[f32],
        dt: &[f32],
        a: &[f32],
        b: &[f32],
        c: &[f32],
        s0: Option<&[f32]>,
        d_inner: usize,
        n_tok: usize,
    ) -> Result<(Vec<f32>, Vec<f32>), String> {
        let d_state = 16;
        let x_size = d_inner * n_tok;
        let a_size = d_inner * d_state;
        let bc_size = d_state * n_tok;

        if x.len() != x_size {
            return Err(format!(
                "SSM Scan: x size mismatch, expected {}, got {}",
                x_size,
                x.len()
            ));
        }
        if dt.len() != x_size {
            return Err(format!(
                "SSM Scan: dt size mismatch, expected {}, got {}",
                x_size,
                dt.len()
            ));
        }
        if a.len() != a_size {
            return Err(format!(
                "SSM Scan: A size mismatch, expected {}, got {}",
                a_size,
                a.len()
            ));
        }
        if b.len() != bc_size {
            return Err(format!(
                "SSM Scan: B size mismatch, expected {}, got {}",
                bc_size,
                b.len()
            ));
        }
        if c.len() != bc_size {
            return Err(format!(
                "SSM Scan: C size mismatch, expected {}, got {}",
                bc_size,
                c.len()
            ));
        }
        if let Some(s) = s0 {
            if s.len() != a_size {
                return Err(format!(
                    "SSM Scan: s0 size mismatch, expected {}, got {}",
                    a_size,
                    s.len()
                ));
            }
        }

        // Create buffers
        let x_buffer = self.device.new_buffer_with_data(
            x.as_ptr() as *const c_void,
            (x_size * std::mem::size_of::<f32>()) as u64,
            metal::MTLResourceOptions::StorageModeShared,
        );

        let dt_buffer = self.device.new_buffer_with_data(
            dt.as_ptr() as *const c_void,
            (x_size * std::mem::size_of::<f32>()) as u64,
            metal::MTLResourceOptions::StorageModeShared,
        );

        let a_buffer = self.device.new_buffer_with_data(
            a.as_ptr() as *const c_void,
            (a_size * std::mem::size_of::<f32>()) as u64,
            metal::MTLResourceOptions::StorageModeShared,
        );

        let b_buffer = self.device.new_buffer_with_data(
            b.as_ptr() as *const c_void,
            (bc_size * std::mem::size_of::<f32>()) as u64,
            metal::MTLResourceOptions::StorageModeShared,
        );

        let c_buffer = self.device.new_buffer_with_data(
            c.as_ptr() as *const c_void,
            (bc_size * std::mem::size_of::<f32>()) as u64,
            metal::MTLResourceOptions::StorageModeShared,
        );

        let s0_buffer = if let Some(s) = s0 {
            Some(self.device.new_buffer_with_data(
                s.as_ptr() as *const c_void,
                (a_size * std::mem::size_of::<f32>()) as u64,
                metal::MTLResourceOptions::StorageModeShared,
            ))
        } else {
            None
        };

        // Create command buffer and encoder
        let command_buffer = self.queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();

        // Create combined output buffer for y + s_out (kernel expects single dst buffer)
        let output_buffer = self.device.new_buffer(
            ((x_size + a_size) * std::mem::size_of::<f32>()) as u64,
            metal::MTLResourceOptions::StorageModeShared,
        );

        // Buffer layout matching kernel:
        // 0: s0 (initial state)
        // 1: x (input)
        // 2: dt (delta time)
        // 3: A (decay parameter)
        // 4: B (input projection)
        // 5: C (output projection)
        // 6: dst (y output + s_out final state)
        encoder.set_compute_pipeline_state(&self.ssm_scan_d16_pipeline);
        if let Some(ref s) = s0_buffer {
            encoder.set_buffer(0, Some(s), 0);
        } else {
            encoder.set_buffer(0, None, 0);
        }
        encoder.set_buffer(1, Some(&x_buffer), 0);
        encoder.set_buffer(2, Some(&dt_buffer), 0);
        encoder.set_buffer(3, Some(&a_buffer), 0);
        encoder.set_buffer(4, Some(&b_buffer), 0);
        encoder.set_buffer(5, Some(&c_buffer), 0);
        encoder.set_buffer(6, Some(&output_buffer), 0);

        let d_state_u32 = 16u32;
        let d_inner_u32 = d_inner as u32;
        let n_tok_u32 = n_tok as u32;
        encoder.set_bytes(
            7,
            std::mem::size_of::<u32>() as u64,
            &d_state_u32 as *const u32 as *const c_void,
        );
        encoder.set_bytes(
            8,
            std::mem::size_of::<u32>() as u64,
            &d_inner_u32 as *const u32 as *const c_void,
        );
        encoder.set_bytes(
            9,
            std::mem::size_of::<u32>() as u64,
            &n_tok_u32 as *const u32 as *const c_void,
        );

        // Threadgroup-based dispatch matching kernel
        // One threadgroup per sgptg d_inner elements
        // Each threadgroup has sgptg * 32 threads
        // Each SIMD group (32 threads) handles one d_inner element
        let sgptg: u64 = 4; // simdgroups per threadgroup
        let n_simdwidth: u64 = 32;
        let n_inner_groups = (d_inner as u64 + sgptg - 1) / sgptg;
        let threadgroup_mem = (sgptg * n_simdwidth + sgptg * 2) * 4; // shared_sums + shared_x_dt + shared_dA
        encoder.set_threadgroup_memory_length(0, threadgroup_mem);

        let grid_size = MTLSize { width: n_inner_groups, height: 1, depth: 1 };
        let threadgroup_size =
            MTLSize { width: sgptg * n_simdwidth, height: 1, depth: 1 };

        encoder.dispatch_thread_groups(grid_size, threadgroup_size);
        encoder.end_encoding();

        command_buffer.commit();
        command_buffer.wait_until_completed();

        // Read results from combined output buffer
        // Layout: [y (d_inner * n_tok), s_out (d_inner * d_state)]
        let output_ptr = output_buffer.contents() as *const f32;
        let y_slice = unsafe { std::slice::from_raw_parts(output_ptr, x_size) };
        let s_slice = unsafe {
            std::slice::from_raw_parts(output_ptr.add(x_size), a_size)
        };

        Ok((y_slice.to_vec(), s_slice.to_vec()))
    }

    /// SSD (State Space Duality) scan for long sequences
    /// Uses matrix multiply-accumulate for efficient parallel processing
    /// Specialized for: CS=64, HD=64 (Mamba-2)
    /// Memory layout (token-major for efficiency):
    /// x: [n_tok, HD] - token-major
    /// dt: [n_tok] - one value per token
    /// A: single scalar
    /// B: [n_tok, d_state] - token-major
    /// C: [n_tok, d_state] - token-major
    /// s0: [HD, d_state] (initial state)
    /// Output y: [n_tok, HD], s: [HD, d_state]
    pub fn ssm_scan_ssd(
        &self,
        x: &[f32],
        dt: &[f32],
        a: &[f32],
        b: &[f32],
        c: &[f32],
        s0: Option<&[f32]>,
        d_inner: usize,
        n_tok: usize,
        d_state: usize,
    ) -> Result<(Vec<f32>, Vec<f32>), String> {
        const HD: usize = 64; // Head Dim (must match d_inner for SSD)

        if d_inner != HD {
            return Err(format!(
                "SSD: d_inner must be {} (head dim), got {}",
                HD, d_inner
            ));
        }

        // Memory layout: token-major for x, B, C
        let x_size = n_tok * d_inner; // [n_tok, HD]
        let bc_size = n_tok * d_state; // [n_tok, d_state]

        if x.len() != x_size {
            return Err(format!(
                "SSD: x size mismatch, expected [n_tok={}, HD={}] = {}, got {}",
                n_tok,
                d_inner,
                x_size,
                x.len()
            ));
        }
        if dt.len() != n_tok {
            return Err(format!(
                "SSD: dt size mismatch, expected [n_tok={}] = {}, got {}",
                n_tok,
                n_tok,
                dt.len()
            ));
        }
        if a.len() != 1 {
            return Err(format!(
                "SSD: A size mismatch, expected 1 (scalar), got {}",
                a.len()
            ));
        }
        if b.len() != bc_size {
            return Err(format!(
                "SSD: B size mismatch, expected [n_tok={}, d_state={}] = {}, got {}",
                n_tok,
                d_state,
                bc_size,
                b.len()
            ));
        }
        if c.len() != bc_size {
            return Err(format!(
                "SSD: C size mismatch, expected [n_tok={}, d_state={}] = {}, got {}",
                n_tok,
                d_state,
                bc_size,
                c.len()
            ));
        }
        if let Some(s) = s0 {
            if s.len() != d_inner * d_state {
                return Err(format!(
                    "SSD: s0 size mismatch, expected [HD={}, d_state={}] = {}, got {}",
                    d_inner,
                    d_state,
                    d_inner * d_state,
                    s.len()
                ));
            }
        }

        // Create buffers
        let x_buffer = self.device.new_buffer_with_data(
            x.as_ptr() as *const c_void,
            (x_size * std::mem::size_of::<f32>()) as u64,
            metal::MTLResourceOptions::StorageModeShared,
        );

        let dt_buffer = self.device.new_buffer_with_data(
            dt.as_ptr() as *const c_void,
            (n_tok * std::mem::size_of::<f32>()) as u64,
            metal::MTLResourceOptions::StorageModeShared,
        );

        let a_buffer = self.device.new_buffer_with_data(
            a.as_ptr() as *const c_void,
            std::mem::size_of::<f32>() as u64,
            metal::MTLResourceOptions::StorageModeShared,
        );

        let b_buffer = self.device.new_buffer_with_data(
            b.as_ptr() as *const c_void,
            (bc_size * std::mem::size_of::<f32>()) as u64,
            metal::MTLResourceOptions::StorageModeShared,
        );

        let c_buffer = self.device.new_buffer_with_data(
            c.as_ptr() as *const c_void,
            (bc_size * std::mem::size_of::<f32>()) as u64,
            metal::MTLResourceOptions::StorageModeShared,
        );

        let s0_buffer = if let Some(s) = s0 {
            self.device.new_buffer_with_data(
                s.as_ptr() as *const c_void,
                (d_inner * d_state * std::mem::size_of::<f32>()) as u64,
                metal::MTLResourceOptions::StorageModeShared,
            )
        } else {
            // Create zero-initialized buffer for initial state
            let buf = self.device.new_buffer(
                (d_inner * d_state * std::mem::size_of::<f32>()) as u64,
                metal::MTLResourceOptions::StorageModeShared,
            );
            // Zero the buffer
            let ptr = buf.contents() as *mut f32;
            for i in 0..(d_inner * d_state) {
                unsafe {
                    *ptr.add(i) = 0.0f32;
                }
            }
            buf
        };

        let y_buffer = self.device.new_buffer(
            (x_size * std::mem::size_of::<f32>()) as u64,
            metal::MTLResourceOptions::StorageModeShared,
        );

        let s_buffer = self.device.new_buffer(
            (d_inner * d_state * std::mem::size_of::<f32>()) as u64,
            metal::MTLResourceOptions::StorageModeShared,
        );

        // Create command buffer and encoder
        let command_buffer = self.queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();

        encoder.set_compute_pipeline_state(&self.ssm_scan_ssd_pipeline);
        encoder.set_buffer(0, Some(&x_buffer), 0);
        encoder.set_buffer(1, Some(&dt_buffer), 0);
        encoder.set_buffer(2, Some(&a_buffer), 0);
        encoder.set_buffer(3, Some(&b_buffer), 0);
        encoder.set_buffer(4, Some(&c_buffer), 0);
        encoder.set_buffer(5, Some(&s0_buffer), 0);
        encoder.set_buffer(6, Some(&y_buffer), 0);
        encoder.set_buffer(7, Some(&s_buffer), 0);
        let d_state_u32 = d_state as u32;
        let n_tok_u32 = n_tok as u32;
        encoder.set_bytes(
            8,
            std::mem::size_of::<u32>() as u64,
            &d_state_u32 as *const u32 as *const c_void,
        );
        encoder.set_bytes(
            9,
            std::mem::size_of::<u32>() as u64,
            &n_tok_u32 as *const u32 as *const c_void,
        );

        // Grid: 1 threadgroup per (head, sequence) - processes chunks of 64 tokens
        // For now, single head and sequence
        let grid_size = MTLSize {
            width: 64, // One thread per channel (HD=64)
            height: 1,
            depth: 1,
        };
        let threadgroup_size = MTLSize { width: 64, height: 1, depth: 1 };

        encoder.dispatch_threads(grid_size, threadgroup_size);
        encoder.end_encoding();

        command_buffer.commit();
        command_buffer.wait_until_completed();

        // Read results
        let y_ptr = y_buffer.contents() as *const f32;
        let s_ptr = s_buffer.contents() as *const f32;
        let y_slice = unsafe { std::slice::from_raw_parts(y_ptr, x_size) };
        let s_slice =
            unsafe { std::slice::from_raw_parts(s_ptr, d_inner * d_state) };

        Ok((y_slice.to_vec(), s_slice.to_vec()))
    }

    /// SSD (State Space Duality) scan with simdgroup MMA optimization
    /// Uses Metal's matrix multiply-accelerate for parallel scan
    /// Designed for Mamba-2: d_state=64, head_dim=64
    pub fn ssm_scan_ssd_mma(
        &self,
        x: &[f32],
        dt: &[f32],
        a: &[f32],
        b: &[f32],
        c: &[f32],
        s0: Option<&[f32]>,
        head_dim: usize,
        n_tok: usize,
        d_state: usize,
        n_head: usize,
    ) -> Result<(Vec<f32>, Vec<f32>), String> {
        const HD: usize = 64; // Head Dim
        const CS: usize = 64; // Chunk size (64 fits in 32KB threadgroup memory)

        if head_dim != HD {
            return Err(format!(
                "SSD MMA: head_dim must be {} (head dim), got {}",
                HD, head_dim
            ));
        }

        // Memory layout: token-major for x, B, C
        let x_size = n_tok * head_dim; // [n_tok, HD]
        let bc_size = n_tok * d_state; // [n_tok, d_state]
        let state_size = head_dim * d_state; // [HD, d_state]
        let y_size = n_tok * n_head * head_dim;

        if x.len() != x_size {
            return Err(format!(
                "SSD MMA: x size mismatch, expected [n_tok={}, HD={}] = {}, got {}",
                n_tok,
                head_dim,
                x_size,
                x.len()
            ));
        }
        if dt.len() != n_tok {
            return Err(format!(
                "SSD MMA: dt size mismatch, expected [n_tok={}] = {}, got {}",
                n_tok,
                n_tok,
                dt.len()
            ));
        }
        if a.len() != 1 {
            return Err(format!(
                "SSD MMA: A size mismatch, expected 1 (scalar), got {}",
                a.len()
            ));
        }
        if b.len() != bc_size {
            return Err(format!(
                "SSD MMA: B size mismatch, expected [n_tok={}, d_state={}] = {}, got {}",
                n_tok,
                d_state,
                bc_size,
                b.len()
            ));
        }
        if c.len() != bc_size {
            return Err(format!(
                "SSD MMA: C size mismatch, expected [n_tok={}, d_state={}] = {}, got {}",
                n_tok,
                d_state,
                bc_size,
                c.len()
            ));
        }

        // Create buffers
        let s0_buffer = if let Some(s) = s0 {
            self.device.new_buffer_with_data(
                s.as_ptr() as *const c_void,
                (state_size * std::mem::size_of::<f32>()) as u64,
                metal::MTLResourceOptions::StorageModeShared,
            )
        } else {
            let buf = self.device.new_buffer(
                (state_size * std::mem::size_of::<f32>()) as u64,
                metal::MTLResourceOptions::StorageModeShared,
            );
            let ptr = buf.contents() as *mut f32;
            for i in 0..state_size {
                unsafe { *ptr.add(i) = 0.0f32 };
            }
            buf
        };

        let x_buffer = self.device.new_buffer_with_data(
            x.as_ptr() as *const c_void,
            (x_size * std::mem::size_of::<f32>()) as u64,
            metal::MTLResourceOptions::StorageModeShared,
        );

        let dt_buffer = self.device.new_buffer_with_data(
            dt.as_ptr() as *const c_void,
            (n_tok * std::mem::size_of::<f32>()) as u64,
            metal::MTLResourceOptions::StorageModeShared,
        );

        let a_buffer = self.device.new_buffer_with_data(
            a.as_ptr() as *const c_void,
            std::mem::size_of::<f32>() as u64,
            metal::MTLResourceOptions::StorageModeShared,
        );

        let b_buffer = self.device.new_buffer_with_data(
            b.as_ptr() as *const c_void,
            (bc_size * std::mem::size_of::<f32>()) as u64,
            metal::MTLResourceOptions::StorageModeShared,
        );

        let c_buffer = self.device.new_buffer_with_data(
            c.as_ptr() as *const c_void,
            (bc_size * std::mem::size_of::<f32>()) as u64,
            metal::MTLResourceOptions::StorageModeShared,
        );

        // Output: y + s_out concatenated
        let output_size = y_size + state_size;
        let output_buffer = self.device.new_buffer(
            (output_size * std::mem::size_of::<f32>()) as u64,
            metal::MTLResourceOptions::StorageModeShared,
        );

        // Threadgroup memory size calculation for CS=64:
        // acs[CS] + exp_acs[CS] + state_decay[CS] + dtx[CS*HD] + sam[NSG*TC*CS] + tiles[NSG*2*TC*TC]
        // = 64 + 64 + 64 + 64*64 + 4*8*64 + 4*2*64 = 192 + 4096 + 2048 + 512 = 6848 floats = 27.4 KB
        // Fits under 32KB Metal threadgroup memory limit
        let threadgroup_memory_size =
            (CS * 3 + CS * HD + 4 * 8 * CS + 4 * 2 * 64) * 4; // bytes

        let command_buffer = self.queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();

        encoder.set_compute_pipeline_state(&self.ssm_scan_ssd_mma_pipeline);
        encoder.set_buffer(0, Some(&s0_buffer), 0); // src0 (s0)
        encoder.set_buffer(1, Some(&x_buffer), 0); // src1 (x)
        encoder.set_buffer(2, Some(&dt_buffer), 0); // src2 (dt)
        encoder.set_buffer(3, Some(&a_buffer), 0); // src3 (A)
        encoder.set_buffer(4, Some(&b_buffer), 0); // src4 (B)
        encoder.set_buffer(5, Some(&c_buffer), 0); // src5 (C)
        // src6 (ids) - not used for now
        encoder.set_buffer(7, Some(&output_buffer), 0); // dst (y + s_out)

        // Set constants
        let d_state_u32 = d_state as u32;
        let head_dim_u32 = head_dim as u32;
        let n_head_u32 = n_head as u32;
        let n_tok_u32 = n_tok as u32;
        let n_seqs_u32 = 1u32;

        encoder.set_bytes(
            8,
            std::mem::size_of::<u32>() as u64,
            &d_state_u32 as *const u32 as *const c_void,
        );
        encoder.set_bytes(
            9,
            std::mem::size_of::<u32>() as u64,
            &head_dim_u32 as *const u32 as *const c_void,
        );
        encoder.set_bytes(
            10,
            std::mem::size_of::<u32>() as u64,
            &n_head_u32 as *const u32 as *const c_void,
        );
        encoder.set_bytes(
            11,
            std::mem::size_of::<u32>() as u64,
            &n_tok_u32 as *const u32 as *const c_void,
        );
        encoder.set_bytes(
            12,
            std::mem::size_of::<u32>() as u64,
            &n_seqs_u32 as *const u32 as *const c_void,
        );

        // Set threadgroup memory size
        encoder
            .set_threadgroup_memory_length(0, threadgroup_memory_size as u64);

        // Grid: one threadgroup per (head, sequence)
        // Threadgroup size: NSG * 32 = 128 threads
        let grid_size = MTLSize {
            width: 1,              // Not used
            height: n_head as u64, // One threadgroup per head
            depth: 1,              // One sequence
        };
        let threadgroup_size = MTLSize {
            width: 128, // NSG * 32
            height: 1,
            depth: 1,
        };

        encoder.dispatch_thread_groups(grid_size, threadgroup_size);
        encoder.end_encoding();

        command_buffer.commit();
        command_buffer.wait_until_completed();

        // Read results
        let output_ptr = output_buffer.contents() as *const f32;
        let y_slice = unsafe { std::slice::from_raw_parts(output_ptr, y_size) };
        let s_slice = unsafe {
            std::slice::from_raw_parts(output_ptr.add(y_size), state_size)
        };

        Ok((y_slice.to_vec(), s_slice.to_vec()))
    }

    // ========================================================================
    // Batched Encode Methods (for command buffer batching)
    // ========================================================================

    /// RoPE (batched, encode into shared encoder)
    pub fn rope_neox_encode(
        &self,
        encoder: &ComputeCommandEncoderRef,
        src_buffer: &Buffer,
        pos: &[i32],
        n_seqs: usize,
        n_heads: usize,
        n_dims: usize,
        freq_base: f32,
        freq_scale: f32,
        output_buffer: &Buffer,
    ) {
        // Reuse cached pos buffer to avoid allocation overhead
        let pos_size = pos.len() * std::mem::size_of::<i32>();
        let pos_buffer = {
            let mut cache = self.rope_pos_buffer.borrow_mut();
            let needs_new = match cache.as_ref() {
                Some(buf) => (buf.length() as usize) < pos_size,
                None => true,
            };

            if needs_new {
                let new_buf = self.device.new_buffer_with_data(
                    pos.as_ptr() as *const std::ffi::c_void,
                    pos_size as u64,
                    metal::MTLResourceOptions::StorageModeShared,
                );
                *cache = Some(new_buf);
            } else if let Some(buf) = cache.as_ref() {
                // Update existing buffer with new data
                unsafe {
                    std::ptr::copy_nonoverlapping(
                        pos.as_ptr() as *const std::ffi::c_void,
                        buf.contents(),
                        pos_size,
                    );
                }
            }
            cache.as_ref().unwrap().clone()
        };

        let args = RopeArgs {
            hidden_dim: n_dims as i32,
            num_heads_seqs: n_heads as i32, // Each sequence has n_heads heads
            num_seqs: n_seqs as i32,
            rotate_dims: n_dims as i32,
            offset: 0,
            freq_base,
            freq_scale,
            ext_factor: 0.0,
            attn_factor: 1.0,
            beta_fast: 32.0,
            beta_slow: 1.0,
            ctx_orig: 0,
        };

        encoder.set_compute_pipeline_state(&self.rope_pipeline);
        encoder.set_bytes(
            0,
            std::mem::size_of::<RopeArgs>() as u64,
            &args as *const RopeArgs as *const std::ffi::c_void,
        );
        encoder.set_buffer(1, Some(src_buffer), 0);
        encoder.set_buffer(2, Some(&pos_buffer), 0);
        encoder.set_buffer(3, Some(output_buffer), 0);

        let thread_group_size = MTLSize { width: 32, height: 1, depth: 1 };
        let thread_group_count =
            MTLSize { width: n_heads as u64, height: n_seqs as u64, depth: 1 };

        encoder.dispatch_thread_groups(thread_group_count, thread_group_size);
        // Do NOT end_encoding() or commit() - let caller handle that
    }

    /// RoPE with F16 output (batched, encode into shared encoder)
    /// Input is F32, output is F16 for KV cache
    pub fn rope_neox_f16_encode(
        &self,
        encoder: &ComputeCommandEncoderRef,
        src_buffer: &Buffer,
        pos: &[i32],
        n_seqs: usize,
        n_heads: usize,
        n_dims: usize,
        freq_base: f32,
        freq_scale: f32,
        output_buffer: &Buffer,
    ) {
        let pos_buffer = self.device.new_buffer_with_data(
            pos.as_ptr() as *const std::ffi::c_void,
            (pos.len() * std::mem::size_of::<i32>()) as u64,
            metal::MTLResourceOptions::StorageModeShared,
        );

        let args = RopeArgs {
            hidden_dim: n_dims as i32,
            num_heads_seqs: n_heads as i32, // Each sequence has n_heads heads
            num_seqs: n_seqs as i32,
            rotate_dims: n_dims as i32,
            offset: 0,
            freq_base,
            freq_scale,
            ext_factor: 0.0,
            attn_factor: 1.0,
            beta_fast: 32.0,
            beta_slow: 1.0,
            ctx_orig: 0,
        };

        encoder.set_compute_pipeline_state(&self.rope_f16_pipeline);
        encoder.set_bytes(
            0,
            std::mem::size_of::<RopeArgs>() as u64,
            &args as *const RopeArgs as *const std::ffi::c_void,
        );
        encoder.set_buffer(1, Some(src_buffer), 0);
        encoder.set_buffer(2, Some(&pos_buffer), 0);
        encoder.set_buffer(3, Some(output_buffer), 0);

        let thread_group_size = MTLSize { width: 32, height: 1, depth: 1 };
        let thread_group_count =
            MTLSize { width: n_heads as u64, height: n_seqs as u64, depth: 1 };

        encoder.dispatch_thread_groups(thread_group_count, thread_group_size);
    }

    /// Flash Attention for F16 KV cache (batched, encode into shared encoder)
    /// Returns true if GPU kernel was dispatched, false if unsupported (need CPU fallback)
    pub fn flash_attn_f16_encode(
        &self,
        encoder: &ComputeCommandEncoderRef,
        n_seqs: usize,
        n_heads: usize,
        _n_kv_heads: usize,
        head_dim: usize,
        seq_len: usize,
        kv_stride: usize,
        q_buffer: &Buffer,
        k_buffer: &Buffer,
        v_buffer: &Buffer,
        output_buffer: &Buffer,
    ) -> bool {
        let pipeline = match head_dim {
            32 => &self.flash_decode_f16_32_pipeline,
            64 => &self.flash_decode_f16_64_pipeline,
            80 => &self.flash_decode_f16_80_pipeline,
            128 => &self.flash_decode_f16_128_pipeline,
            160 => &self.flash_decode_f16_160_pipeline,
            213 => &self.flash_decode_f16_213_pipeline,
            _ => {
                // GPU kernel not available, return false to indicate CPU fallback needed
                return false;
            }
        };

        let scale = 1.0f32 / (head_dim as f32).sqrt();
        let seq_k = seq_len as u32;
        let kv_stride_u32 = kv_stride as u32;

        encoder.set_compute_pipeline_state(pipeline);
        encoder.set_bytes(
            0,
            4,
            &seq_k as *const u32 as *const std::ffi::c_void,
        );
        encoder.set_bytes(
            1,
            4,
            &kv_stride_u32 as *const u32 as *const std::ffi::c_void,
        );
        encoder.set_bytes(
            2,
            4,
            &scale as *const f32 as *const std::ffi::c_void,
        );
        encoder.set_buffer(3, Some(q_buffer), 0);
        encoder.set_buffer(4, Some(k_buffer), 0);
        encoder.set_buffer(5, Some(v_buffer), 0);
        encoder.set_buffer(6, Some(output_buffer), 0);

        // Pass GQA parameters (buffer indices 7 and 8)
        let n_kv_heads_u32 = _n_kv_heads as u32;
        let n_heads_u32 = n_heads as u32;
        encoder.set_bytes(
            7,
            4,
            &n_kv_heads_u32 as *const u32 as *const std::ffi::c_void,
        );
        encoder.set_bytes(
            8,
            4,
            &n_heads_u32 as *const u32 as *const std::ffi::c_void,
        );

        let thread_group_size = MTLSize { width: 32, height: 1, depth: 1 };
        let thread_group_count =
            MTLSize { width: 1, height: n_heads as u64, depth: n_seqs as u64 };

        let c_per_iter = match head_dim {
            32 => 8,
            64 => 16,
            80 => 20,
            128 => 32,
            160 => 40,
            213 => 32, // Specialized kernel uses 32 items per iteration
            _ => 32,
        };
        let shmem_size = 512 + c_per_iter * 4;
        encoder.set_threadgroup_memory_length(0, shmem_size as u64);

        encoder.dispatch_thread_groups(thread_group_count, thread_group_size);
        // Do NOT end_encoding() or commit() - let caller handle that

        true // GPU kernel dispatched successfully
    }

    /// Multi-Head Flash Attention (llama.cpp style)
    /// Process 32 heads per threadgroup for 32x dispatch overhead reduction
    /// Q=32, NSG=8, C=64
    /// Returns true if GPU kernel was dispatched, false if unsupported (need CPU fallback)
    pub fn flash_attn_multihead_encode(
        &self,
        encoder: &ComputeCommandEncoderRef,
        n_seqs: usize,
        n_heads: usize,
        n_kv_heads: usize,
        head_dim: usize,
        seq_len: usize,
        kv_stride: usize,
        q_buffer: &Buffer,
        k_buffer: &Buffer,
        v_buffer: &Buffer,
        output_buffer: &Buffer,
    ) -> bool {
        // Only support head_dim=128 for now
        if head_dim != 128 {
            return false;
        }

        let scale = 1.0f32 / (head_dim as f32).sqrt();
        let seq_k = seq_len as u32;
        let kv_stride_u32 = kv_stride as u32;

        encoder.set_compute_pipeline_state(
            &self.flash_attn_multihead_128_pipeline,
        );
        encoder.set_bytes(
            0,
            4,
            &seq_k as *const u32 as *const std::ffi::c_void,
        );
        encoder.set_bytes(
            1,
            4,
            &kv_stride_u32 as *const u32 as *const std::ffi::c_void,
        );
        encoder.set_bytes(
            2,
            4,
            &scale as *const f32 as *const std::ffi::c_void,
        );
        encoder.set_buffer(3, Some(q_buffer), 0);
        encoder.set_buffer(4, Some(k_buffer), 0);
        encoder.set_buffer(5, Some(v_buffer), 0);
        encoder.set_buffer(6, Some(output_buffer), 0);

        // Pass GQA parameters (buffer indices 7 and 8)
        let n_kv_heads_u32 = n_kv_heads as u32;
        let n_heads_u32 = n_heads as u32;
        encoder.set_bytes(
            7,
            4,
            &n_kv_heads_u32 as *const u32 as *const std::ffi::c_void,
        );
        encoder.set_bytes(
            8,
            4,
            &n_heads_u32 as *const u32 as *const std::ffi::c_void,
        );

        // Grid: [n_heads / 32, n_seqs, 1]
        // Each threadgroup processes 32 heads with 8 simdgroups (256 threads)
        let n_head_groups = (n_heads + 31) / 32;
        let thread_group_size = MTLSize {
            width: 256, // 8 simdgroups
            height: 1,
            depth: 1,
        };
        let thread_group_count = MTLSize {
            width: n_head_groups as u64,
            height: n_seqs as u64,
            depth: 1,
        };

        encoder.dispatch_thread_groups(thread_group_count, thread_group_size);

        true // GPU kernel dispatched successfully
    }

    /// Flash Attention MMA kernel (experimental, for performance comparison)
    /// Process 4 heads per simdgroup for better memory coalescing
    pub fn flash_attn_mma_encode(
        &self,
        encoder: &ComputeCommandEncoderRef,
        n_heads: usize,
        n_kv_heads: usize,
        seq_len: usize,
        kv_stride: usize,
        q_buffer: &Buffer,
        k_buffer: &Buffer,
        v_buffer: &Buffer,
        output_buffer: &Buffer,
    ) {
        let head_dim = 128; // MMA kernel only supports head_dim=128

        let scale = 1.0f32 / (head_dim as f32).sqrt();
        let seq_k = seq_len as u32;
        let kv_stride_u32 = kv_stride as u32;
        let gqa_ratio = (n_heads / n_kv_heads) as u32;

        encoder.set_compute_pipeline_state(&self.flash_decode_mma_pipeline);
        encoder.set_bytes(
            0,
            4,
            &seq_k as *const u32 as *const std::ffi::c_void,
        );
        encoder.set_bytes(
            1,
            4,
            &kv_stride_u32 as *const u32 as *const std::ffi::c_void,
        );
        encoder.set_bytes(
            2,
            4,
            &scale as *const f32 as *const std::ffi::c_void,
        );
        encoder.set_buffer(3, Some(q_buffer), 0);
        encoder.set_buffer(4, Some(k_buffer), 0);
        encoder.set_buffer(5, Some(v_buffer), 0);
        encoder.set_buffer(6, Some(output_buffer), 0);
        encoder.set_bytes(
            7,
            4,
            &gqa_ratio as *const u32 as *const std::ffi::c_void,
        );

        // 4 heads per threadgroup (MMA version)
        let n_head_groups = (n_heads + 3) / 4;
        let thread_group_size = MTLSize { width: 32, height: 1, depth: 1 };
        let thread_group_count =
            MTLSize { width: n_head_groups as u64, height: 1, depth: 1 };

        encoder.dispatch_thread_groups(thread_group_count, thread_group_size);
    }

    /// Flash Attention MMA v2 encode (optimized version)
    pub fn flash_attn_mma_v2_encode(
        &self,
        encoder: &ComputeCommandEncoderRef,
        n_heads: usize,
        n_kv_heads: usize,
        seq_len: usize,
        kv_stride: usize,
        q_buffer: &Buffer,
        k_buffer: &Buffer,
        v_buffer: &Buffer,
        output_buffer: &Buffer,
    ) {
        let head_dim = 128; // MMA v2 kernel only supports head_dim=128

        let scale = 1.0f32 / (head_dim as f32).sqrt();
        let seq_k = seq_len as u32;
        let kv_stride_u32 = kv_stride as u32;
        let gqa_ratio = (n_heads / n_kv_heads) as u32;

        encoder.set_compute_pipeline_state(&self.flash_decode_mma_v2_pipeline);
        encoder.set_bytes(
            0,
            4,
            &seq_k as *const u32 as *const std::ffi::c_void,
        );
        encoder.set_bytes(
            1,
            4,
            &kv_stride_u32 as *const u32 as *const std::ffi::c_void,
        );
        encoder.set_bytes(
            2,
            4,
            &scale as *const f32 as *const std::ffi::c_void,
        );
        encoder.set_buffer(3, Some(q_buffer), 0);
        encoder.set_buffer(4, Some(k_buffer), 0);
        encoder.set_buffer(5, Some(v_buffer), 0);
        encoder.set_buffer(6, Some(output_buffer), 0);
        encoder.set_bytes(
            7,
            4,
            &gqa_ratio as *const u32 as *const std::ffi::c_void,
        );

        // 4 heads per threadgroup (same as MMA v1)
        let n_head_groups = (n_heads + 3) / 4;
        let thread_group_size = MTLSize { width: 32, height: 1, depth: 1 };
        let thread_group_count =
            MTLSize { width: n_head_groups as u64, height: 1, depth: 1 };

        encoder.dispatch_thread_groups(thread_group_count, thread_group_size);
    }

    /// Flash Attention MMA Batched encode (C=8 positions per iteration)
    /// Reduces iteration count by 8x for better performance
    /// Returns true if GPU kernel was dispatched, false if unsupported (need CPU fallback)
    pub fn flash_attn_mma_batched_encode(
        &self,
        encoder: &ComputeCommandEncoderRef,
        n_heads: usize,
        n_kv_heads: usize,
        seq_len: usize,
        kv_stride: usize,
        q_buffer: &Buffer,
        k_buffer: &Buffer,
        v_buffer: &Buffer,
        output_buffer: &Buffer,
    ) -> bool {
        let head_dim = 128;

        let scale = 1.0f32 / (head_dim as f32).sqrt();

        // Pack arguments into struct
        #[repr(C)]
        #[derive(Clone, Copy)]
        struct FlashDecodeMMAArgs {
            seq_k: u32,
            n_heads: u32,
            n_kv_heads: u32,
            kv_stride: u32,
            scale: f32,
            _padding: [u8; 12], // Align to 32 bytes
        }

        let args = FlashDecodeMMAArgs {
            seq_k: seq_len as u32,
            n_heads: n_heads as u32,
            n_kv_heads: n_kv_heads as u32,
            kv_stride: kv_stride as u32,
            scale,
            _padding: [0; 12],
        };

        encoder.set_compute_pipeline_state(
            &self.flash_decode_mma_batched_pipeline,
        );
        encoder.set_bytes(
            0,
            std::mem::size_of::<FlashDecodeMMAArgs>() as u64,
            &args as *const FlashDecodeMMAArgs as *const std::ffi::c_void,
        );
        encoder.set_buffer(1, Some(q_buffer), 0);
        encoder.set_buffer(2, Some(k_buffer), 0);
        encoder.set_buffer(3, Some(v_buffer), 0);
        encoder.set_buffer(4, Some(output_buffer), 0);

        // Each threadgroup processes 1 head
        let thread_group_size = MTLSize { width: 32, height: 1, depth: 1 };
        let thread_group_count =
            MTLSize { width: n_heads as u64, height: 1, depth: 1 };

        encoder.dispatch_thread_groups(thread_group_count, thread_group_size);

        true // GPU kernel dispatched successfully
    }

    /// Flash Prefill for batched queries (encode into shared encoder)
    /// Process N queries in parallel, each query attending to seq_k KV cache entries
    /// q: [batch, n_heads, seq_q, head_dim]
    /// k: [batch, n_kv_heads, seq_k, head_dim]
    /// v: [batch, n_kv_heads, seq_k, head_dim]
    /// output: [batch, n_heads, seq_q, head_dim]
    /// start_pos: Starting position in KV cache for causal mask
    pub fn flash_prefill_f16_encode(
        &self,
        encoder: &ComputeCommandEncoderRef,
        batch: usize,
        n_heads: usize,
        n_kv_heads: usize,
        seq_q: usize,
        seq_k: usize,
        head_dim: usize,
        q_buffer: &Buffer,
        k_buffer: &Buffer,
        v_buffer: &Buffer,
        output_buffer: &Buffer,
        q_stride_b: usize,
        q_stride_h: usize,
        q_stride_s: usize,
        k_stride_b: usize,
        k_stride_h: usize,
        k_stride_s: usize,
        v_stride_b: usize,
        v_stride_h: usize,
        v_stride_s: usize,
        o_stride_b: usize,
        o_stride_h: usize,
        o_stride_s: usize,
        start_pos: usize,
    ) {
        let scale = 1.0f32 / (head_dim as f32).sqrt();

        #[repr(C)]
        struct FlashPrefillArgs {
            batch: u32,
            heads: u32,
            seq_q: u32,
            seq_k: u32,
            head_dim: u32,
            scale: f32,
            q_stride_b: u32,
            q_stride_h: u32,
            q_stride_s: u32,
            k_stride_b: u32,
            k_stride_h: u32,
            k_stride_s: u32,
            v_stride_b: u32,
            v_stride_h: u32,
            v_stride_s: u32,
            o_stride_b: u32,
            o_stride_h: u32,
            o_stride_s: u32,
            start_pos: u32, // For causal mask
        }

        let args = FlashPrefillArgs {
            batch: batch as u32,
            heads: n_heads as u32,
            seq_q: seq_q as u32,
            seq_k: seq_k as u32,
            head_dim: head_dim as u32,
            scale,
            q_stride_b: q_stride_b as u32,
            q_stride_h: q_stride_h as u32,
            q_stride_s: q_stride_s as u32,
            k_stride_b: k_stride_b as u32,
            k_stride_h: k_stride_h as u32,
            k_stride_s: k_stride_s as u32,
            v_stride_b: v_stride_b as u32,
            v_stride_h: v_stride_h as u32,
            v_stride_s: v_stride_s as u32,
            o_stride_b: o_stride_b as u32,
            o_stride_h: o_stride_h as u32,
            o_stride_s: o_stride_s as u32,
            start_pos: start_pos as u32,
        };

        encoder.set_compute_pipeline_state(&self.flash_prefill_f16_pipeline);
        encoder.set_bytes(
            0,
            std::mem::size_of::<FlashPrefillArgs>() as u64,
            &args as *const FlashPrefillArgs as *const std::ffi::c_void,
        );
        encoder.set_buffer(1, Some(q_buffer), 0);
        encoder.set_buffer(2, Some(k_buffer), 0);
        encoder.set_buffer(3, Some(v_buffer), 0);
        encoder.set_buffer(4, Some(output_buffer), 0);

        // Threadgroup: each query position gets one threadgroup
        // Grid: [seq_q, n_heads, batch]
        let thread_group_count = MTLSize {
            width: seq_q as u64,
            height: n_heads as u64,
            depth: batch as u64,
        };

        // Threadgroup size: 128 threads (4 simdgroups)
        let thread_group_size = MTLSize { width: 128, height: 1, depth: 1 };

        // Shared memory for threadgroup reduction
        // tg_max: 4 floats, tg_sum: 4 floats, tg_out: 128 * 4 floats
        let shmem_size = (4 + 4 + 128 * 4) * 4;
        encoder.set_threadgroup_memory_length(0, shmem_size as u64);

        encoder.dispatch_thread_groups(thread_group_count, thread_group_size);
    }

    /// Batched Flash Prefill (llama.cpp style architecture)
    /// One threadgroup per Q_PER_TG (8) queries, each processing all KV positions
    /// This maximizes GPU parallelism by launching more threadgroups
    pub fn flash_prefill_batch_encode(
        &self,
        encoder: &ComputeCommandEncoderRef,
        batch: usize,
        n_heads: usize,
        seq_q: usize,
        seq_k: usize,
        head_dim: usize,
        q_buffer: &Buffer,
        k_buffer: &Buffer,
        v_buffer: &Buffer,
        output_buffer: &Buffer,
        q_stride_b: usize,
        q_stride_h: usize,
        q_stride_s: usize,
        k_stride_b: usize,
        k_stride_h: usize,
        k_stride_s: usize,
        v_stride_b: usize,
        v_stride_h: usize,
        v_stride_s: usize,
        o_stride_b: usize,
        o_stride_h: usize,
        o_stride_s: usize,
        start_pos: usize,
    ) {
        let scale = 1.0f32 / (head_dim as f32).sqrt();

        #[repr(C)]
        struct FlashPrefillBatchArgs {
            batch: u32,
            n_heads: u32,
            seq_q: u32,
            seq_k: u32,
            head_dim: u32,
            scale: f32,
            q_stride_b: u32,
            q_stride_h: u32,
            q_stride_s: u32,
            k_stride_b: u32,
            k_stride_h: u32,
            k_stride_s: u32,
            v_stride_b: u32,
            v_stride_h: u32,
            v_stride_s: u32,
            o_stride_b: u32,
            o_stride_h: u32,
            o_stride_s: u32,
            start_pos: u32,
        }

        let args = FlashPrefillBatchArgs {
            batch: batch as u32,
            n_heads: n_heads as u32,
            seq_q: seq_q as u32,
            seq_k: seq_k as u32,
            head_dim: head_dim as u32,
            scale,
            q_stride_b: q_stride_b as u32,
            q_stride_h: q_stride_h as u32,
            q_stride_s: q_stride_s as u32,
            k_stride_b: k_stride_b as u32,
            k_stride_h: k_stride_h as u32,
            k_stride_s: k_stride_s as u32,
            v_stride_b: v_stride_b as u32,
            v_stride_h: v_stride_h as u32,
            v_stride_s: v_stride_s as u32,
            o_stride_b: o_stride_b as u32,
            o_stride_h: o_stride_h as u32,
            o_stride_s: o_stride_s as u32,
            start_pos: start_pos as u32,
        };

        encoder.set_compute_pipeline_state(&self.flash_prefill_batch_pipeline);
        encoder.set_bytes(
            0,
            std::mem::size_of::<FlashPrefillBatchArgs>() as u64,
            &args as *const FlashPrefillBatchArgs as *const std::ffi::c_void,
        );
        encoder.set_buffer(1, Some(q_buffer), 0);
        encoder.set_buffer(2, Some(k_buffer), 0);
        encoder.set_buffer(3, Some(v_buffer), 0);
        encoder.set_buffer(4, Some(output_buffer), 0);

        // Grid: [(seq_q + 7) / 8, n_heads, batch]
        // Each threadgroup handles 8 query positions
        const Q_PER_TG: usize = 8;
        let num_q_blocks = (seq_q + Q_PER_TG - 1) / Q_PER_TG;
        let thread_group_count = MTLSize {
            width: num_q_blocks as u64,
            height: n_heads as u64,
            depth: batch as u64,
        };

        // Threadgroup size: 128 threads (4 simdgroups × 32 threads)
        let thread_group_size = MTLSize { width: 128, height: 1, depth: 1 };

        // Shared memory:
        // sq: 8 * 128 = 1024 half (2048 bytes)
        // so: 8 * 128 = 1024 half (2048 bytes)
        // ss: 8 * 64 = 512 float (2048 bytes)
        let shmem_size = 2048 + 2048 + 2048;
        encoder.set_threadgroup_memory_length(0, shmem_size as u64);

        encoder.dispatch_thread_groups(thread_group_count, thread_group_size);
    }

    // ========================================================================
    // GEMM (Matrix-Matrix Multiplication) for Batched Prefill
    // ========================================================================

    /// Q5_K x F16 Matrix-Matrix multiplication (encode into shared encoder)
    /// weights: Column-major [K, M] quantized as Q5_K (GGUF format)
    /// input: [N, K] as F16 (batch_size rows, hidden_size cols)
    /// output: [M, N] as F32
    ///
    /// For Column-major W[K, M]:
    /// - Element W[i, j] is at offset i + j*K (skip K elements per column)
    /// - This equals W^T stored as Row-major [M, K]
    ///
    /// The kernel expects Row-major A[ne0, ne00]:
    /// - ne0 = M (output rows)
    /// - ne00 = K (reduction dim)
    /// - nb01 = stride between rows = K/256 blocks (column stride in col-major)
    pub fn mul_mm_q5_k_encode(
        &self,
        encoder: &ComputeCommandEncoderRef,
        weights_buffer: &Buffer,
        input_buffer: &Buffer,
        output_buffer: &Buffer,
        m: usize, // output rows (q_out_size)
        n: usize, // output cols (batch_size)
        k: usize, // reduction dimension (hidden_size)
    ) {
        // Q5_K: 256 elements per block
        // Column-major [K, M] = Row-major [M, K] (transposed)
        // Stride between rows in row-major = column stride in col-major = K/256 blocks
        let nb_per_row = k / 256;
        let nb01 = (nb_per_row * std::mem::size_of::<BlockQ5K>()) as u64;

        #[repr(C)]
        struct MulMmArgs {
            ne00: i32, // reduction dim = K
            ne02: i32, // batch for weights (1)
            nb01: u64, // row stride for kernel
            nb02: u64, // batch stride for weights
            nb03: u64, // batch3 stride for weights
            ne12: i32, // batch for input (1)
            nb10: u64, // element stride for input
            nb11: u64, // row stride for input
            nb12: u64, // batch stride for input
            nb13: u64, // batch3 stride for input
            ne0: i32,  // output rows = M
            ne1: i32,  // output cols = batch
            r2: i16,   // broadcast ratio
            r3: i16,   // broadcast ratio
        }

        // Input is [N, K] in row-major order:
        // - nb10: stride between elements in a row = 2 (size of F16)
        // - nb11: stride between rows = K * 2
        let args = MulMmArgs {
            ne00: k as i32, // reduction dim = K
            ne02: 1,
            nb01,
            nb02: 0,
            nb03: 0,
            ne12: 1,
            nb10: 2, // Stride between elements in a row (F16 = 2 bytes)
            nb11: (k * 2) as u64, // Stride between rows = K * sizeof(F16)
            nb12: 0,
            nb13: 0,
            ne0: m as i32, // output rows = M
            ne1: n as i32,
            r2: 1,
            r3: 1,
        };

        encoder.set_compute_pipeline_state(&self.gemm_template_q5_k_pipeline);
        encoder.set_bytes(
            0,
            std::mem::size_of::<MulMmArgs>() as u64,
            &args as *const MulMmArgs as *const std::ffi::c_void,
        );
        encoder.set_buffer(1, Some(weights_buffer), 0);
        encoder.set_buffer(2, Some(input_buffer), 0);
        encoder.set_buffer(3, Some(output_buffer), 0);

        // Threadgroup configuration from mul_mm.metal:
        // NR0 = 64, NR1 = 32
        // Grid: [ceil(N/32), ceil(M/64), batch]
        let thread_group_count = MTLSize {
            width: ((n + 31) / 32) as u64,
            height: ((m + 63) / 64) as u64,
            depth: 1,
        };

        // Threadgroup size: 2 simdgroups = 64 threads (from template)
        let thread_group_size = MTLSize { width: 32, height: 2, depth: 1 };

        // Shared memory for simdgroup matrix operations
        // sa: 4096 bytes, sb: 4096 bytes = 8192 total
        encoder.set_threadgroup_memory_length(0, 8192);

        encoder.dispatch_thread_groups(thread_group_count, thread_group_size);
    }

    /// Q4_K GEMM: Y = W @ X
    /// W is Q4_K quantized [M, K], X is F16 [N, K], Y is F32 [N, M]
    pub fn mul_mm_q4_k_encode(
        &self,
        encoder: &ComputeCommandEncoderRef,
        weights_buffer: &Buffer,
        input_buffer: &Buffer,
        output_buffer: &Buffer,
        m: usize,
        n: usize,
        k: usize,
    ) {
        let nb_per_row = k / 256;
        let nb01 = (nb_per_row * std::mem::size_of::<BlockQ4K>()) as u64;

        #[repr(C)]
        struct MulMmArgs {
            ne00: i32,
            ne02: i32,
            nb01: u64,
            nb02: u64,
            nb03: u64,
            ne12: i32,
            nb10: u64,
            nb11: u64,
            nb12: u64,
            nb13: u64,
            ne0: i32,
            ne1: i32,
            r2: i16,
            r3: i16,
        }

        let args = MulMmArgs {
            ne00: k as i32,
            ne02: 1,
            nb01,
            nb02: 0,
            nb03: 0,
            ne12: 1,
            nb10: 2,
            nb11: (k * 2) as u64,
            nb12: 0,
            nb13: 0,
            ne0: m as i32,
            ne1: n as i32,
            r2: 1,
            r3: 1,
        };

        encoder.set_compute_pipeline_state(&self.gemm_template_q4_k_pipeline);
        encoder.set_bytes(
            0,
            std::mem::size_of::<MulMmArgs>() as u64,
            &args as *const MulMmArgs as *const std::ffi::c_void,
        );
        encoder.set_buffer(1, Some(weights_buffer), 0);
        encoder.set_buffer(2, Some(input_buffer), 0);
        encoder.set_buffer(3, Some(output_buffer), 0);

        let thread_group_count = MTLSize {
            width: ((n + 31) / 32) as u64,
            height: ((m + 63) / 64) as u64,
            depth: 1,
        };
        let thread_group_size = MTLSize { width: 32, height: 2, depth: 1 };
        encoder.set_threadgroup_memory_length(0, 8192);
        encoder.dispatch_thread_groups(thread_group_count, thread_group_size);
    }

    /// Q6_K GEMM: Y = W @ X
    /// W is Q6_K quantized [M, K], X is F16 [N, K], Y is F32 [N, M]
    pub fn mul_mm_q6_k_encode(
        &self,
        encoder: &ComputeCommandEncoderRef,
        weights_buffer: &Buffer,
        input_buffer: &Buffer,
        output_buffer: &Buffer,
        m: usize,
        n: usize,
        k: usize,
    ) {
        let nb_per_row = k / 256;
        let nb01 = (nb_per_row * std::mem::size_of::<BlockQ6K>()) as u64;

        #[repr(C)]
        struct MulMmArgs {
            ne00: i32,
            ne02: i32,
            nb01: u64,
            nb02: u64,
            nb03: u64,
            ne12: i32,
            nb10: u64,
            nb11: u64,
            nb12: u64,
            nb13: u64,
            ne0: i32,
            ne1: i32,
            r2: i16,
            r3: i16,
        }

        let args = MulMmArgs {
            ne00: k as i32,
            ne02: 1,
            nb01,
            nb02: 0,
            nb03: 0,
            ne12: 1,
            nb10: 2,
            nb11: (k * 2) as u64,
            nb12: 0,
            nb13: 0,
            ne0: m as i32,
            ne1: n as i32,
            r2: 1,
            r3: 1,
        };

        encoder.set_compute_pipeline_state(&self.gemm_template_q6_k_pipeline);
        encoder.set_bytes(
            0,
            std::mem::size_of::<MulMmArgs>() as u64,
            &args as *const MulMmArgs as *const std::ffi::c_void,
        );
        encoder.set_buffer(1, Some(weights_buffer), 0);
        encoder.set_buffer(2, Some(input_buffer), 0);
        encoder.set_buffer(3, Some(output_buffer), 0);

        let thread_group_count = MTLSize {
            width: ((n + 31) / 32) as u64,
            height: ((m + 63) / 64) as u64,
            depth: 1,
        };
        let thread_group_size = MTLSize { width: 32, height: 2, depth: 1 };
        encoder.set_threadgroup_memory_length(0, 8192);
        encoder.dispatch_thread_groups(thread_group_count, thread_group_size);
    }

    /// Fused GEMV + Add for residual connections
    /// Combines matrix-vector multiply with residual addition in single kernel
    /// Pattern: output = (W @ input) + residual
    pub fn fused_gemv_add_encode(
        &self,
        encoder: &ComputeCommandEncoderRef,
        weights_buffer: &Buffer,
        input_buffer: &Buffer,
        residual_buffer: &Buffer,
        output_buffer: &Buffer,
        m: usize,
        k: usize,
        dtype: crate::model::tensor::DataType,
    ) -> bool {
        // Only support K-quant formats for now
        if !matches!(
            dtype,
            crate::model::tensor::DataType::Q4_K
                | crate::model::tensor::DataType::Q5_K
                | crate::model::tensor::DataType::Q6_K
        ) {
            return false;
        }

        let nb = k / 256; // QK_K = 256
        let nb01: u64 = match dtype {
            crate::model::tensor::DataType::Q4_K => {
                (nb * std::mem::size_of::<BlockQ4K>()) as u64
            }
            crate::model::tensor::DataType::Q5_K => {
                (nb * std::mem::size_of::<BlockQ5K>()) as u64
            }
            crate::model::tensor::DataType::Q6_K => {
                (nb * std::mem::size_of::<BlockQ6K>()) as u64
            }
            _ => return false,
        };

        // Args structure for fused kernel
        #[repr(C)]
        #[derive(Clone, Copy)]
        struct FusedGemvAddArgs {
            ne00: u32, // K dimension (input size)
            ne01: u32, // M dimension (output rows)
            nb01: u64, // Byte stride between rows
        }

        let args = FusedGemvAddArgs { ne00: k as u32, ne01: m as u32, nb01 };

        let pipeline = match dtype {
            crate::model::tensor::DataType::Q4_K => {
                &self.fused_gemv_add_q4_k_pipeline
            }
            crate::model::tensor::DataType::Q5_K => {
                &self.fused_gemv_add_q5_k_pipeline
            }
            crate::model::tensor::DataType::Q6_K => {
                &self.fused_gemv_add_q6_k_pipeline
            }
            _ => return false,
        };

        encoder.set_compute_pipeline_state(pipeline);
        encoder.set_buffer(0, Some(weights_buffer), 0);
        encoder.set_buffer(1, Some(input_buffer), 0);
        encoder.set_buffer(2, Some(residual_buffer), 0);
        encoder.set_buffer(3, Some(output_buffer), 0);
        encoder.set_bytes(
            4,
            std::mem::size_of::<FusedGemvAddArgs>() as u64,
            &args as *const FusedGemvAddArgs as *const std::ffi::c_void,
        );

        // Threadgroup configuration (same as regular GEMV)
        let (nsg, nr0) = match dtype {
            crate::model::tensor::DataType::Q6_K => (2, 2),
            _ => (2, 1), // Q4_K, Q5_K
        };

        let threads_per_tg = nsg * 32;
        let tgs = (m + nsg * nr0 - 1) / (nsg * nr0);

        let thread_group_size =
            MTLSize { width: threads_per_tg as u64, height: 1, depth: 1 };
        let thread_group_count =
            MTLSize { width: tgs as u64, height: 1, depth: 1 };

        encoder.dispatch_thread_groups(thread_group_count, thread_group_size);
        true
    }

    /// F16 GEMV: y = W @ x
    /// W is stored row-major as [M, K] in F16 format
    /// x is F32 vector of size K
    /// y is F32 vector of size M
    pub fn mul_mv_f16_encode(
        &self,
        encoder: &ComputeCommandEncoderRef,
        weights_buffer: &Buffer, // [M, K] in F16
        input_buffer: &Buffer,   // [K] in F32
        output_buffer: &Buffer,  // [M] in F32
        m: usize,                // output size
        k: usize,                // input size
    ) {
        // Choose optimal pipeline based on K size
        let (pipeline, grid_size) = if k < 32 {
            // Short vector: use simple kernel, one thread per output
            let pipeline = &self.mv_f16_short_pipeline;
            let grid = MTLSize { width: m as u64, height: 1, depth: 1 };
            (pipeline, grid)
        } else if k < 256 {
            // Medium vector: use basic simd kernel
            let pipeline = &self.mv_f16_basic_pipeline;
            // One threadgroup per row, 32 threads per threadgroup
            let grid = MTLSize { width: m as u64, height: 1, depth: 1 };
            (pipeline, grid)
        } else {
            // Large vector: use optimized kernel with adaptive NSG
            let pipeline = &self.mv_f16_4_nsg2_pipeline;
            // One threadgroup per row
            let grid = MTLSize { width: m as u64, height: 1, depth: 1 };
            (pipeline, grid)
        };

        encoder.set_compute_pipeline_state(pipeline);
        encoder.set_buffer(0, Some(weights_buffer), 0);
        encoder.set_buffer(1, Some(input_buffer), 0);
        encoder.set_buffer(2, Some(output_buffer), 0);
        encoder.set_bytes(
            3,
            4,
            &(m as u32) as *const u32 as *const std::ffi::c_void,
        );
        encoder.set_bytes(
            4,
            4,
            &(k as u32) as *const u32 as *const std::ffi::c_void,
        );

        let thread_group_size = MTLSize { width: 32, height: 1, depth: 1 };
        encoder.dispatch_thread_groups(grid_size, thread_group_size);
    }

    /// F16 GEMM: Y = W @ X
    /// W is stored row-major as [M, K] in F16 format
    /// X is [K, N] in F16 format
    /// Y is [M, N] in F32 format
    /// For decode (N=1), uses GEMV; for prefill (N>1), uses batched GEMV
    pub fn mul_mm_f16_encode(
        &self,
        encoder: &ComputeCommandEncoderRef,
        weights_buffer: &Buffer, // [M, K] in F16
        input_buffer: &Buffer,   // [N, K] in F16 (batched input)
        output_buffer: &Buffer,  // [N, M] in F32 (batched output)
        m: usize,                // output rows (vocab_size)
        n: usize,                // batch size
        k: usize,                // reduction dim (hidden_size)
    ) {
        // Simple approach: process each batch separately
        // TODO: Implement proper F16 GEMM using kernel_mul_mm_f16_f16
        for batch_idx in 0..n {
            let input_offset = (batch_idx * k * 2) as u64; // F16 = 2 bytes
            let output_offset = (batch_idx * m * 4) as u64; // F32 = 4 bytes

            // Use F16×F16 GEMV kernel for each batch
            // Kernel expects mul_mv_args struct as first argument
            #[repr(C)]
            struct MulMvArgs {
                ne00: i32, // K
                ne01: i32, // M
                ne02: i32,
                nb00: u64,
                nb01: u64, // row stride for weights = K * 2
                nb02: u64,
                nb03: u64,
                ne10: i32,
                ne11: i32,
                ne12: i32,
                nb10: u64,
                nb11: u64,
                nb12: u64,
                nb13: u64,
                ne0: i32,
                ne1: i32,
                nr0: i32,
                r2: i16,
                r3: i16,
                nsg: i16,
                ne12_batch: i16,
            }

            let args = MulMvArgs {
                ne00: k as i32,
                ne01: m as i32,
                ne02: 1,
                nb00: 2,
                nb01: (k * 2) as u64, // F16 row stride
                nb02: 0,
                nb03: 0,
                ne10: k as i32,
                ne11: 1,
                ne12: 1,
                nb10: 2,
                nb11: (k * 2) as u64,
                nb12: 0,
                nb13: 0,
                ne0: m as i32,
                ne1: 1,
                nr0: 1,
                r2: 1,
                r3: 1,
                nsg: 2,
                ne12_batch: 1,
            };

            encoder.set_compute_pipeline_state(&self.mv_f16_f16_pipeline);
            encoder.set_bytes(
                0,
                std::mem::size_of::<MulMvArgs>() as u64,
                &args as *const MulMvArgs as *const std::ffi::c_void,
            );
            encoder.set_buffer(1, Some(weights_buffer), 0);
            encoder.set_buffer(2, Some(input_buffer), input_offset);
            encoder.set_buffer(3, Some(output_buffer), output_offset);

            // One threadgroup per output row
            let grid = MTLSize { width: m as u64, height: 1, depth: 1 };
            let tg_size = MTLSize { width: 32, height: 1, depth: 1 };
            encoder.set_threadgroup_memory_length(0, 256);
            encoder.dispatch_thread_groups(grid, tg_size);
        }
    }
}
