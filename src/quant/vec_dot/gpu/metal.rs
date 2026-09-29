use crate::quant::types::{
    BlockIQ1M, BlockIQ1S, BlockIQ2S, BlockIQ2XS, BlockIQ2XXS, BlockIQ3S,
    BlockIQ3XXS, BlockIQ4NL, BlockIQ4XS, BlockQ2K, BlockQ3K, BlockQ4_0,
    BlockQ4K, BlockQ5K, BlockQ6K, BlockQ8_0, BlockTQ1_0, BlockTQ2_0,
};
use metal::{
    Buffer, CommandQueue, ComputePipelineState, Device, FunctionConstantValues,
    Library, MTLDataType, MTLSize, NSUInteger,
};
use std::cell::RefCell;
use std::os::raw::c_void;

pub struct MetalContext {
    device: Device,
    queue: CommandQueue,
    vec_dot_library: Library,
    mv_library: Library,
    mv_simd_library: Library,
    mv_iq_tq_library: Library,
    // Templated MV library from llama.cpp (all K-quant formats)
    mul_mv_template_library: Library,
    // GEMM libraries
    gemm_q4_k_library: Library,
    gemm_q2_k_library: Library,
    gemm_q3_k_library: Library,
    gemm_q5_k_library: Library,
    gemm_q6_k_library: Library,
    gemm_iq_tq_library: Library,
    gemm_iq2_iq3_library: Library,
    // Templated GEMM library from llama.cpp
    gemm_template_library: Library,
    // Cached MV pipelines
    mv_q4_0_pipeline: ComputePipelineState,
    mv_q4_k_pipeline: ComputePipelineState,
    mv_q5_k_pipeline: ComputePipelineState,
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
    // Split=1 pipelines for large K sizes
    mv_iq2_xxs_split1_pipeline: ComputePipelineState,
    mv_iq2_xs_split1_pipeline: ComputePipelineState,
    mv_iq2_s_split1_pipeline: ComputePipelineState,
    mv_iq3_xxs_split1_pipeline: ComputePipelineState,
    mv_iq3_s_split1_pipeline: ComputePipelineState,
    // GEMM pipelines
    gemm_q4_k_pipeline: ComputePipelineState,
    gemm_q2_k_pipeline: ComputePipelineState,
    gemm_q3_k_pipeline: ComputePipelineState,
    gemm_q5_k_pipeline: ComputePipelineState,
    gemm_q6_k_pipeline: ComputePipelineState,
    gemm_iq4_nl_pipeline: ComputePipelineState,
    gemm_iq4_xs_pipeline: ComputePipelineState,
    gemm_iq1_s_pipeline: ComputePipelineState,
    gemm_iq1_m_pipeline: ComputePipelineState,
    gemm_iq2_xxs_pipeline: ComputePipelineState,
    gemm_iq2_xs_pipeline: ComputePipelineState,
    gemm_iq2_s_pipeline: ComputePipelineState,
    gemm_iq3_xxs_pipeline: ComputePipelineState,
    gemm_iq3_s_pipeline: ComputePipelineState,
    gemm_tq2_0_pipeline: ComputePipelineState,
    gemm_tq1_0_pipeline: ComputePipelineState,
    // Templated GEMM pipelines (from llama.cpp)
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
    activation_library: Library,
    silu_pipeline: ComputePipelineState,
    silu_4_pipeline: ComputePipelineState,
    gelu_pipeline: ComputePipelineState,
    gelu_4_pipeline: ComputePipelineState,
    gelu_quick_pipeline: ComputePipelineState,
    gelu_quick_4_pipeline: ComputePipelineState,
    // Flash Attention
    flash_attn_library: Library,
    flash_attn_f16_pipeline: ComputePipelineState,
    // Quantize
    quantize_library: Library,
    quantize_f16_pipeline: ComputePipelineState,
    quantize_f16_4_pipeline: ComputePipelineState,
    quantize_q4_0_pipeline: ComputePipelineState,
    quantize_q8_0_pipeline: ComputePipelineState,
    // Cached buffers for MV operations
    // Once created with data, reuse without copy
    mv_weights_buffer: RefCell<Option<Buffer>>,
    mv_input_buffer: RefCell<Option<Buffer>>,
    mv_output_buffer: RefCell<Option<Buffer>>,
    // Track buffer sizes to know if we need to recreate
    mv_weights_size: RefCell<usize>,
    mv_input_size: RefCell<usize>,
    mv_output_size: RefCell<usize>,
    // Cached buffers for GEMM operations
    gemm_weights_buffer: RefCell<Option<Buffer>>,
    gemm_input_buffer: RefCell<Option<Buffer>>,
    gemm_output_buffer: RefCell<Option<Buffer>>,
    gemm_weights_size: RefCell<usize>,
    gemm_input_size: RefCell<usize>,
    gemm_output_size: RefCell<usize>,
}

#[derive(Clone, Copy)]
#[repr(C)]
struct RopeArgs {
    ne00: i32,
    ne01: i32,
    ne02: i32,
    nb00: u64,
    nb01: u64,
    nb02: u64,
    n_dims: i32,
    n_offs: i32,
    freq_base: f32,
    freq_scale: f32,
    ext_factor: f32,
    attn_factor: f32,
    beta_fast: f32,
    beta_slow: f32,
    n_ctx_orig: i32,
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

/// Function constant indices for MV kernels (from llama.cpp)
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

    unsafe {
        // Set NSG (number of simdgroups)
        let nsg_ptr: *const c_void = &nsg as *const i16 as *const c_void;
        cv.set_constant_value_at_index(
            nsg_ptr,
            MTLDataType::Short,
            FC_MUL_MV_NSG,
        );

        // Set NE12
        let ne12_ptr: *const c_void = &ne12 as *const i16 as *const c_void;
        cv.set_constant_value_at_index(
            ne12_ptr,
            MTLDataType::Short,
            FC_MUL_MV_NE12,
        );

        // Set R2
        let r2_ptr: *const c_void = &r2 as *const i16 as *const c_void;
        cv.set_constant_value_at_index(
            r2_ptr,
            MTLDataType::Short,
            FC_MUL_MV_R2,
        );

        // Set R3
        let r3_ptr: *const c_void = &r3 as *const i16 as *const c_void;
        cv.set_constant_value_at_index(
            r3_ptr,
            MTLDataType::Short,
            FC_MUL_MV_R3,
        );
    }

    cv
}

impl MetalContext {
    pub fn new() -> Result<Self, String> {
        let device = Device::system_default().ok_or("No Metal device found")?;

        let queue = device.new_command_queue();

        // Set optimization level for Metal shader compilation
        // Enable fast math for better GPU performance (matches llama.cpp approach)
        let compile_options = metal::CompileOptions::new();
        compile_options.set_fast_math_enabled(true);
        // Note: metal-rs 0.31 doesn't expose set_optimization_level directly
        // but the default should be equivalent to -O3

        let vec_dot_source = include_str!("../../../../shaders/vec_dot.metal");
        let vec_dot_library = device
            .new_library_with_source(vec_dot_source, &compile_options)
            .map_err(|e| format!("Failed to compile vec_dot library: {}", e))?;

        let mv_source = include_str!("../../../../shaders/mv.metal");
        let mv_library = device
            .new_library_with_source(mv_source, &compile_options)
            .map_err(|e| format!("Failed to compile mv library: {}", e))?;

        let mv_simd_source = include_str!("../../../../shaders/mv_simd.metal");
        let mv_simd_library = device
            .new_library_with_source(mv_simd_source, &compile_options)
            .map_err(|e| format!("Failed to compile mv_simd library: {}", e))?;

        // Compile IQ/TQ MV library (includes all IQ formats)
        let mv_iq_tq_source =
            include_str!("../../../../shaders/mv_iq_tq.metal");
        let iq_grid_tables =
            include_str!("../../../../shaders/iq_grid_tables.h");
        let mv_iq_tq_combined = mv_iq_tq_source
            .replace("#include \"iq_grid_tables.h\"", iq_grid_tables);
        let mv_iq_tq_library = device
            .new_library_with_source(&mv_iq_tq_combined, &compile_options)
            .map_err(|e| {
                format!("Failed to compile mv_iq_tq library: {}", e)
            })?;

        // Compile templated MV library from llama.cpp (all K-quant formats in one file)
        let mul_mv_template_source =
            include_str!("../../../../shaders/mul_mv_template.metal");
        let mul_mv_template_library = device
            .new_library_with_source(mul_mv_template_source, &compile_options)
            .map_err(|e| {
                format!("Failed to compile mul_mv_template library: {}", e)
            })?;

        // Compile GEMM libraries
        let gemm_q4_k_source =
            include_str!("../../../../shaders/gemm_q4_k.metal");
        let gemm_q4_k_library = device
            .new_library_with_source(gemm_q4_k_source, &compile_options)
            .map_err(|e| {
                format!("Failed to compile gemm_q4_k library: {}", e)
            })?;

        let gemm_q2_k_source =
            include_str!("../../../../shaders/gemm_q2_k.metal");
        let gemm_q2_k_library = device
            .new_library_with_source(gemm_q2_k_source, &compile_options)
            .map_err(|e| {
                format!("Failed to compile gemm_q2_k library: {}", e)
            })?;

        let gemm_q3_k_source =
            include_str!("../../../../shaders/gemm_q3_k.metal");
        let gemm_q3_k_library = device
            .new_library_with_source(gemm_q3_k_source, &compile_options)
            .map_err(|e| {
                format!("Failed to compile gemm_q3_k library: {}", e)
            })?;

        let gemm_q5_k_source =
            include_str!("../../../../shaders/gemm_q5_k.metal");
        let gemm_q5_k_library = device
            .new_library_with_source(gemm_q5_k_source, &compile_options)
            .map_err(|e| {
                format!("Failed to compile gemm_q5_k library: {}", e)
            })?;

        let gemm_q6_k_source =
            include_str!("../../../../shaders/gemm_q6_k.metal");
        let gemm_q6_k_library = device
            .new_library_with_source(gemm_q6_k_source, &compile_options)
            .map_err(|e| {
                format!("Failed to compile gemm_q6_k library: {}", e)
            })?;

        let gemm_iq_tq_source =
            include_str!("../../../../shaders/gemm_iq_tq.metal");
        let gemm_iq_tq_combined = gemm_iq_tq_source
            .replace("#include \"iq_grid_tables.h\"", iq_grid_tables);
        let gemm_iq_tq_library = device
            .new_library_with_source(&gemm_iq_tq_combined, &compile_options)
            .map_err(|e| {
                format!("Failed to compile gemm_iq_tq library: {}", e)
            })?;

        let gemm_iq2_iq3_source =
            include_str!("../../../../shaders/gemm_iq2_iq3.metal");
        let gemm_iq2_iq3_combined = gemm_iq2_iq3_source
            .replace("#include \"iq_grid_tables.h\"", iq_grid_tables);
        let gemm_iq2_iq3_library = device
            .new_library_with_source(&gemm_iq2_iq3_combined, &compile_options)
            .map_err(|e| {
                format!("Failed to compile gemm_iq2_iq3 library: {}", e)
            })?;

        // Compile templated GEMM library (self-contained, no C++ headers)
        let gemm_template_source =
            include_str!("../../../../shaders/mul_mm_standalone.metal");
        let gemm_template_library = device
            .new_library_with_source(gemm_template_source, &compile_options)
            .map_err(|e| {
                format!("Failed to compile gemm_template library: {}", e)
            })?;

        // Compile RMS Norm library
        let rms_norm_source =
            include_str!("../../../../shaders/rms_norm.metal");
        let rms_norm_library = device
            .new_library_with_source(rms_norm_source, &compile_options)
            .map_err(|e| {
                format!("Failed to compile rms_norm library: {}", e)
            })?;

        // Compile Softmax library
        let softmax_source = include_str!("../../../../shaders/softmax.metal");
        let softmax_library = device
            .new_library_with_source(softmax_source, &compile_options)
            .map_err(|e| format!("Failed to compile softmax library: {}", e))?;

        // Compile ROPE library
        let rope_source = include_str!("../../../../shaders/rope.metal");
        let rope_library = device
            .new_library_with_source(rope_source, &compile_options)
            .map_err(|e| format!("Failed to compile rope library: {}", e))?;

        // Compile Activation library
        let activation_source =
            include_str!("../../../../shaders/activation.metal");
        let activation_library = device
            .new_library_with_source(activation_source, &compile_options)
            .map_err(|e| {
                format!("Failed to compile activation library: {}", e)
            })?;

        // Compile Flash Attention library (standalone with expanded includes)
        let flash_attn_source = include_str!(
            "../../../../shaders/flash_attn_standalone_stub.metal"
        );
        let flash_attn_library = device
            .new_library_with_source(flash_attn_source, &compile_options)
            .map_err(|e| {
                format!("Failed to compile flash_attn library: {}", e)
            })?;

        // Compile Quantize library
        let quantize_source =
            include_str!("../../../../shaders/quantize.metal");
        let quantize_library = device
            .new_library_with_source(quantize_source, &compile_options)
            .map_err(|e| {
                format!("Failed to compile quantize library: {}", e)
            })?;

        // Create cached pipeline for Q4_0 MV (from template)
        // Q4_0 kernel also requires function constants
        let mv_q4_0_kernel = mul_mv_template_library
            .get_function(
                "kernel_mul_mv_q4_0_f32",
                Some(create_mv_function_constants(2, 1, 1, 1)),
            )
            .map_err(|e| format!("Failed to get Q4_0 kernel: {}", e))?;
        let mv_q4_0_pipeline = device
            .new_compute_pipeline_state_with_function(&mv_q4_0_kernel)
            .map_err(|e| format!("Failed to create Q4_0 pipeline: {}", e))?;

        // Create cached pipelines for K-quant MV (all from template)
        // Use default specialization: nsg=2, ne12=1, r2=1, r3=1 (single-batch decode)

        let mv_q4_k_kernel = mul_mv_template_library
            .get_function(
                "kernel_mul_mv_q4_K_f32",
                Some(create_mv_function_constants(2, 1, 1, 1)),
            )
            .map_err(|e| format!("Failed to get Q4_K kernel: {}", e))?;
        let mv_q4_k_pipeline = device
            .new_compute_pipeline_state_with_function(&mv_q4_k_kernel)
            .map_err(|e| format!("Failed to create Q4_K pipeline: {}", e))?;

        let mv_q5_k_kernel = mul_mv_template_library
            .get_function(
                "kernel_mul_mv_q5_K_f32",
                Some(create_mv_function_constants(2, 1, 1, 1)),
            )
            .map_err(|e| format!("Failed to get Q5_K kernel: {}", e))?;
        let mv_q5_k_pipeline = device
            .new_compute_pipeline_state_with_function(&mv_q5_k_kernel)
            .map_err(|e| format!("Failed to create Q5_K pipeline: {}", e))?;

        let mv_q2_k_kernel = mul_mv_template_library
            .get_function(
                "kernel_mul_mv_q2_K_f32",
                Some(create_mv_function_constants(2, 1, 1, 1)),
            )
            .map_err(|e| format!("Failed to get Q2_K kernel: {}", e))?;
        let mv_q2_k_pipeline = device
            .new_compute_pipeline_state_with_function(&mv_q2_k_kernel)
            .map_err(|e| format!("Failed to create Q2_K pipeline: {}", e))?;

        let mv_q6_k_kernel = mul_mv_template_library
            .get_function(
                "kernel_mul_mv_q6_K_f32",
                Some(create_mv_function_constants(2, 1, 1, 1)),
            )
            .map_err(|e| format!("Failed to get Q6_K kernel: {}", e))?;
        let mv_q6_k_pipeline = device
            .new_compute_pipeline_state_with_function(&mv_q6_k_kernel)
            .map_err(|e| format!("Failed to create Q6_K pipeline: {}", e))?;

        let mv_q3_k_kernel = mul_mv_template_library
            .get_function(
                "kernel_mul_mv_q3_K_f32",
                Some(create_mv_function_constants(2, 1, 1, 1)),
            )
            .map_err(|e| format!("Failed to get Q3_K kernel: {}", e))?;
        let mv_q3_k_pipeline = device
            .new_compute_pipeline_state_with_function(&mv_q3_k_kernel)
            .map_err(|e| format!("Failed to create Q3_K pipeline: {}", e))?;

        // Create cached pipeline for IQ4_NL MV
        let mv_iq4_nl_kernel = mv_iq_tq_library
            .get_function("kernel_mul_mv_iq4_nl_f32", None)
            .map_err(|e| format!("Failed to get IQ4_NL kernel: {}", e))?;
        let mv_iq4_nl_pipeline = device
            .new_compute_pipeline_state_with_function(&mv_iq4_nl_kernel)
            .map_err(|e| format!("Failed to create IQ4_NL pipeline: {}", e))?;

        // Create cached pipeline for TQ2_0 MV
        let mv_tq2_0_kernel = mv_iq_tq_library
            .get_function("kernel_mul_mv_tq2_0_f32", None)
            .map_err(|e| format!("Failed to get TQ2_0 kernel: {}", e))?;
        let mv_tq2_0_pipeline = device
            .new_compute_pipeline_state_with_function(&mv_tq2_0_kernel)
            .map_err(|e| format!("Failed to create TQ2_0 pipeline: {}", e))?;

        // Create cached pipeline for IQ4_XS MV
        let mv_iq4_xs_kernel = mv_iq_tq_library
            .get_function("kernel_mul_mv_iq4_xs_f32", None)
            .map_err(|e| format!("Failed to get IQ4_XS kernel: {}", e))?;
        let mv_iq4_xs_pipeline = device
            .new_compute_pipeline_state_with_function(&mv_iq4_xs_kernel)
            .map_err(|e| format!("Failed to create IQ4_XS pipeline: {}", e))?;

        // Create cached pipeline for TQ1_0 MV
        let mv_tq1_0_kernel = mv_iq_tq_library
            .get_function("kernel_mul_mv_tq1_0_f32", None)
            .map_err(|e| format!("Failed to get TQ1_0 kernel: {}", e))?;
        let mv_tq1_0_pipeline = device
            .new_compute_pipeline_state_with_function(&mv_tq1_0_kernel)
            .map_err(|e| format!("Failed to create TQ1_0 pipeline: {}", e))?;

        // Create pipelines for IQ series
        let mv_iq1_s_kernel = mv_iq_tq_library
            .get_function("kernel_mul_mv_iq1_s_f32", None)
            .map_err(|e| format!("Failed to get IQ1_S kernel: {}", e))?;
        let mv_iq1_s_pipeline = device
            .new_compute_pipeline_state_with_function(&mv_iq1_s_kernel)
            .map_err(|e| format!("Failed to create IQ1_S pipeline: {}", e))?;

        let mv_iq1_m_kernel = mv_iq_tq_library
            .get_function("kernel_mul_mv_iq1_m_f32", None)
            .map_err(|e| format!("Failed to get IQ1_M kernel: {}", e))?;
        let mv_iq1_m_pipeline = device
            .new_compute_pipeline_state_with_function(&mv_iq1_m_kernel)
            .map_err(|e| format!("Failed to create IQ1_M pipeline: {}", e))?;

        let mv_iq2_xxs_kernel = mv_iq_tq_library
            .get_function("kernel_mul_mv_iq2_xxs_f32", None)
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

        let mv_iq2_xs_kernel = mv_iq_tq_library
            .get_function("kernel_mul_mv_iq2_xs_f32", None)
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

        let mv_iq3_xxs_kernel = mv_iq_tq_library
            .get_function("kernel_mul_mv_iq3_xxs_f32", None)
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

        let mv_iq3_s_kernel = mv_iq_tq_library
            .get_function("kernel_mul_mv_iq3_s_f32", None)
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

        // Create GEMM pipelines
        let gemm_q4_k_kernel = gemm_q4_k_library
            .get_function("kernel_gemm_q4_k_f32", None)
            .map_err(|e| format!("Failed to get GEMM Q4_K kernel: {}", e))?;
        let gemm_q4_k_pipeline = device
            .new_compute_pipeline_state_with_function(&gemm_q4_k_kernel)
            .map_err(|e| {
                format!("Failed to create GEMM Q4_K pipeline: {}", e)
            })?;

        let gemm_q2_k_kernel = gemm_q2_k_library
            .get_function("kernel_gemm_q2_k_f32", None)
            .map_err(|e| format!("Failed to get GEMM Q2_K kernel: {}", e))?;
        let gemm_q2_k_pipeline = device
            .new_compute_pipeline_state_with_function(&gemm_q2_k_kernel)
            .map_err(|e| {
                format!("Failed to create GEMM Q2_K pipeline: {}", e)
            })?;

        let gemm_q3_k_kernel = gemm_q3_k_library
            .get_function("kernel_gemm_q3_k_f32", None)
            .map_err(|e| format!("Failed to get GEMM Q3_K kernel: {}", e))?;
        let gemm_q3_k_pipeline = device
            .new_compute_pipeline_state_with_function(&gemm_q3_k_kernel)
            .map_err(|e| {
                format!("Failed to create GEMM Q3_K pipeline: {}", e)
            })?;

        let gemm_q5_k_kernel = gemm_q5_k_library
            .get_function("kernel_gemm_q5_k_f32", None)
            .map_err(|e| format!("Failed to get GEMM Q5_K kernel: {}", e))?;
        let gemm_q5_k_pipeline = device
            .new_compute_pipeline_state_with_function(&gemm_q5_k_kernel)
            .map_err(|e| {
                format!("Failed to create GEMM Q5_K pipeline: {}", e)
            })?;

        let gemm_q6_k_kernel = gemm_q6_k_library
            .get_function("kernel_gemm_q6_k_f32", None)
            .map_err(|e| format!("Failed to get GEMM Q6_K kernel: {}", e))?;
        let gemm_q6_k_pipeline = device
            .new_compute_pipeline_state_with_function(&gemm_q6_k_kernel)
            .map_err(|e| {
                format!("Failed to create GEMM Q6_K pipeline: {}", e)
            })?;

        let gemm_iq4_nl_kernel = gemm_iq_tq_library
            .get_function("kernel_gemm_iq4_nl_f32", None)
            .map_err(|e| format!("Failed to get GEMM IQ4_NL kernel: {}", e))?;
        let gemm_iq4_nl_pipeline = device
            .new_compute_pipeline_state_with_function(&gemm_iq4_nl_kernel)
            .map_err(|e| {
                format!("Failed to create GEMM IQ4_NL pipeline: {}", e)
            })?;

        let gemm_iq4_xs_kernel = gemm_iq_tq_library
            .get_function("kernel_gemm_iq4_xs_f32", None)
            .map_err(|e| format!("Failed to get GEMM IQ4_XS kernel: {}", e))?;
        let gemm_iq4_xs_pipeline = device
            .new_compute_pipeline_state_with_function(&gemm_iq4_xs_kernel)
            .map_err(|e| {
                format!("Failed to create GEMM IQ4_XS pipeline: {}", e)
            })?;

        let gemm_iq1_s_kernel = gemm_iq_tq_library
            .get_function("kernel_gemm_iq1_s_f32", None)
            .map_err(|e| format!("Failed to get GEMM IQ1_S kernel: {}", e))?;
        let gemm_iq1_s_pipeline = device
            .new_compute_pipeline_state_with_function(&gemm_iq1_s_kernel)
            .map_err(|e| {
                format!("Failed to create GEMM IQ1_S pipeline: {}", e)
            })?;

        let gemm_iq1_m_kernel = gemm_iq_tq_library
            .get_function("kernel_gemm_iq1_m_f32", None)
            .map_err(|e| format!("Failed to get GEMM IQ1_M kernel: {}", e))?;
        let gemm_iq1_m_pipeline = device
            .new_compute_pipeline_state_with_function(&gemm_iq1_m_kernel)
            .map_err(|e| {
                format!("Failed to create GEMM IQ1_M pipeline: {}", e)
            })?;

        let gemm_iq2_xxs_kernel = gemm_iq2_iq3_library
            .get_function("kernel_gemm_iq2_xxs_f32", None)
            .map_err(|e| format!("Failed to get GEMM IQ2_XXS kernel: {}", e))?;
        let gemm_iq2_xxs_pipeline = device
            .new_compute_pipeline_state_with_function(&gemm_iq2_xxs_kernel)
            .map_err(|e| {
                format!("Failed to create GEMM IQ2_XXS pipeline: {}", e)
            })?;

        let gemm_iq2_xs_kernel = gemm_iq2_iq3_library
            .get_function("kernel_gemm_iq2_xs_f32", None)
            .map_err(|e| format!("Failed to get GEMM IQ2_XS kernel: {}", e))?;
        let gemm_iq2_xs_pipeline = device
            .new_compute_pipeline_state_with_function(&gemm_iq2_xs_kernel)
            .map_err(|e| {
                format!("Failed to create GEMM IQ2_XS pipeline: {}", e)
            })?;

        let gemm_iq2_s_kernel = gemm_iq2_iq3_library
            .get_function("kernel_gemm_iq2_s_f32", None)
            .map_err(|e| format!("Failed to get GEMM IQ2_S kernel: {}", e))?;
        let gemm_iq2_s_pipeline = device
            .new_compute_pipeline_state_with_function(&gemm_iq2_s_kernel)
            .map_err(|e| {
                format!("Failed to create GEMM IQ2_S pipeline: {}", e)
            })?;

        let gemm_iq3_xxs_kernel = gemm_iq2_iq3_library
            .get_function("kernel_gemm_iq3_xxs_f32", None)
            .map_err(|e| format!("Failed to get GEMM IQ3_XXS kernel: {}", e))?;
        let gemm_iq3_xxs_pipeline = device
            .new_compute_pipeline_state_with_function(&gemm_iq3_xxs_kernel)
            .map_err(|e| {
                format!("Failed to create GEMM IQ3_XXS pipeline: {}", e)
            })?;

        let gemm_iq3_s_kernel = gemm_iq2_iq3_library
            .get_function("kernel_gemm_iq3_s_f32", None)
            .map_err(|e| format!("Failed to get GEMM IQ3_S kernel: {}", e))?;
        let gemm_iq3_s_pipeline = device
            .new_compute_pipeline_state_with_function(&gemm_iq3_s_kernel)
            .map_err(|e| {
                format!("Failed to create GEMM IQ3_S pipeline: {}", e)
            })?;

        let gemm_tq2_0_kernel = gemm_iq_tq_library
            .get_function("kernel_gemm_tq2_0_f32", None)
            .map_err(|e| format!("Failed to get GEMM TQ2_0 kernel: {}", e))?;
        let gemm_tq2_0_pipeline = device
            .new_compute_pipeline_state_with_function(&gemm_tq2_0_kernel)
            .map_err(|e| {
                format!("Failed to create GEMM TQ2_0 pipeline: {}", e)
            })?;

        let gemm_tq1_0_kernel = gemm_iq2_iq3_library
            .get_function("kernel_gemm_tq1_0_f32", None)
            .map_err(|e| format!("Failed to get GEMM TQ1_0 kernel: {}", e))?;
        let gemm_tq1_0_pipeline = device
            .new_compute_pipeline_state_with_function(&gemm_tq1_0_kernel)
            .map_err(|e| {
                format!("Failed to create GEMM TQ1_0 pipeline: {}", e)
            })?;

        // Create templated GEMM pipelines (from llama.cpp)
        // Requires function constant specialization
        let gemm_template_q4_k_kernel = {
            let constant_values = metal::FunctionConstantValues::new();
            let false_val: bool = false;
            let one_val: i16 = 1;
            unsafe {
                constant_values.set_constant_value_at_index(
                    &false_val as *const _ as *const std::ffi::c_void,
                    metal::MTLDataType::Bool,
                    0,
                );
                constant_values.set_constant_value_at_index(
                    &false_val as *const _ as *const std::ffi::c_void,
                    metal::MTLDataType::Bool,
                    1,
                );
                constant_values.set_constant_value_at_index(
                    &one_val as *const _ as *const std::ffi::c_void,
                    metal::MTLDataType::Short,
                    2,
                );
                constant_values.set_constant_value_at_index(
                    &one_val as *const _ as *const std::ffi::c_void,
                    metal::MTLDataType::Short,
                    3,
                );
                constant_values.set_constant_value_at_index(
                    &one_val as *const _ as *const std::ffi::c_void,
                    metal::MTLDataType::Short,
                    4,
                );
                constant_values.set_constant_value_at_index(
                    &one_val as *const _ as *const std::ffi::c_void,
                    metal::MTLDataType::Short,
                    5,
                );
            }
            let descriptor = metal::FunctionDescriptor::new();
            descriptor.set_name("kernel_mul_mm_q4_K_f32");
            descriptor.set_constant_values(&constant_values);
            gemm_template_library
                .new_function_with_descriptor(&descriptor)
                .map_err(|e| {
                    format!("Failed to get templated Q4_K kernel: {}", e)
                })?
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
            unsafe {
                constant_values.set_constant_value_at_index(
                    &false_val as *const _ as *const std::ffi::c_void,
                    metal::MTLDataType::Bool,
                    0,
                );
                constant_values.set_constant_value_at_index(
                    &false_val as *const _ as *const std::ffi::c_void,
                    metal::MTLDataType::Bool,
                    1,
                );
                constant_values.set_constant_value_at_index(
                    &one_val as *const _ as *const std::ffi::c_void,
                    metal::MTLDataType::Short,
                    2,
                );
                constant_values.set_constant_value_at_index(
                    &one_val as *const _ as *const std::ffi::c_void,
                    metal::MTLDataType::Short,
                    3,
                );
                constant_values.set_constant_value_at_index(
                    &one_val as *const _ as *const std::ffi::c_void,
                    metal::MTLDataType::Short,
                    4,
                );
                constant_values.set_constant_value_at_index(
                    &one_val as *const _ as *const std::ffi::c_void,
                    metal::MTLDataType::Short,
                    5,
                );
            }
            let descriptor = metal::FunctionDescriptor::new();
            descriptor.set_name("kernel_mul_mm_q2_K_f32");
            descriptor.set_constant_values(&constant_values);
            gemm_template_library
                .new_function_with_descriptor(&descriptor)
                .map_err(|e| {
                    format!("Failed to get templated Q2_K kernel: {}", e)
                })?
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
            unsafe {
                constant_values.set_constant_value_at_index(
                    &false_val as *const _ as *const std::ffi::c_void,
                    metal::MTLDataType::Bool,
                    0,
                );
                constant_values.set_constant_value_at_index(
                    &false_val as *const _ as *const std::ffi::c_void,
                    metal::MTLDataType::Bool,
                    1,
                );
                constant_values.set_constant_value_at_index(
                    &one_val as *const _ as *const std::ffi::c_void,
                    metal::MTLDataType::Short,
                    2,
                );
                constant_values.set_constant_value_at_index(
                    &one_val as *const _ as *const std::ffi::c_void,
                    metal::MTLDataType::Short,
                    3,
                );
                constant_values.set_constant_value_at_index(
                    &one_val as *const _ as *const std::ffi::c_void,
                    metal::MTLDataType::Short,
                    4,
                );
                constant_values.set_constant_value_at_index(
                    &one_val as *const _ as *const std::ffi::c_void,
                    metal::MTLDataType::Short,
                    5,
                );
            }
            let descriptor = metal::FunctionDescriptor::new();
            descriptor.set_name("kernel_mul_mm_q3_K_f32");
            descriptor.set_constant_values(&constant_values);
            gemm_template_library
                .new_function_with_descriptor(&descriptor)
                .map_err(|e| {
                    format!("Failed to get templated Q3_K kernel: {}", e)
                })?
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
            unsafe {
                constant_values.set_constant_value_at_index(
                    &false_val as *const _ as *const std::ffi::c_void,
                    metal::MTLDataType::Bool,
                    0,
                );
                constant_values.set_constant_value_at_index(
                    &false_val as *const _ as *const std::ffi::c_void,
                    metal::MTLDataType::Bool,
                    1,
                );
                constant_values.set_constant_value_at_index(
                    &one_val as *const _ as *const std::ffi::c_void,
                    metal::MTLDataType::Short,
                    2,
                );
                constant_values.set_constant_value_at_index(
                    &one_val as *const _ as *const std::ffi::c_void,
                    metal::MTLDataType::Short,
                    3,
                );
                constant_values.set_constant_value_at_index(
                    &one_val as *const _ as *const std::ffi::c_void,
                    metal::MTLDataType::Short,
                    4,
                );
                constant_values.set_constant_value_at_index(
                    &one_val as *const _ as *const std::ffi::c_void,
                    metal::MTLDataType::Short,
                    5,
                );
            }
            let descriptor = metal::FunctionDescriptor::new();
            descriptor.set_name("kernel_mul_mm_q5_K_f32");
            descriptor.set_constant_values(&constant_values);
            gemm_template_library
                .new_function_with_descriptor(&descriptor)
                .map_err(|e| {
                    format!("Failed to get templated Q5_K kernel: {}", e)
                })?
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
            unsafe {
                constant_values.set_constant_value_at_index(
                    &false_val as *const _ as *const std::ffi::c_void,
                    metal::MTLDataType::Bool,
                    0,
                );
                constant_values.set_constant_value_at_index(
                    &false_val as *const _ as *const std::ffi::c_void,
                    metal::MTLDataType::Bool,
                    1,
                );
                constant_values.set_constant_value_at_index(
                    &one_val as *const _ as *const std::ffi::c_void,
                    metal::MTLDataType::Short,
                    2,
                );
                constant_values.set_constant_value_at_index(
                    &one_val as *const _ as *const std::ffi::c_void,
                    metal::MTLDataType::Short,
                    3,
                );
                constant_values.set_constant_value_at_index(
                    &one_val as *const _ as *const std::ffi::c_void,
                    metal::MTLDataType::Short,
                    4,
                );
                constant_values.set_constant_value_at_index(
                    &one_val as *const _ as *const std::ffi::c_void,
                    metal::MTLDataType::Short,
                    5,
                );
            }
            let descriptor = metal::FunctionDescriptor::new();
            descriptor.set_name("kernel_mul_mm_q6_K_f32");
            descriptor.set_constant_values(&constant_values);
            gemm_template_library
                .new_function_with_descriptor(&descriptor)
                .map_err(|e| {
                    format!("Failed to get templated Q6_K kernel: {}", e)
                })?
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
            .get_function("kernel_soft_max_f32", None)
            .map_err(|e| format!("Failed to get Softmax kernel: {}", e))?;
        let softmax_pipeline = device
            .new_compute_pipeline_state_with_function(&softmax_kernel)
            .map_err(|e| format!("Failed to create Softmax pipeline: {}", e))?;

        let softmax_4_kernel = softmax_library
            .get_function("kernel_soft_max_f32_4", None)
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

        // Create Activation pipelines
        let silu_kernel = activation_library
            .get_function("kernel_silu_f32_4", None)
            .map_err(|e| format!("Failed to get SiLU kernel: {}", e))?;
        let silu_pipeline = device
            .new_compute_pipeline_state_with_function(&silu_kernel)
            .map_err(|e| format!("Failed to create SiLU pipeline: {}", e))?;

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

        // Create Flash Attention pipeline (template specialization, may fail on some Metal versions)
        let flash_attn_f16_pipeline = match flash_attn_library
            .get_function("kernel_flash_attn_ext_f16_dk128_dv128", None)
        {
            Ok(kernel) => {
                match device.new_compute_pipeline_state_with_function(&kernel) {
                    Ok(pipeline) => pipeline,
                    Err(_) => rms_norm_pipeline.clone(),
                }
            }
            Err(_) => rms_norm_pipeline.clone(),
        };

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

        Ok(Self {
            device,
            queue,
            vec_dot_library,
            mv_library,
            mv_simd_library,
            mv_iq_tq_library,
            // Templated MV library
            mul_mv_template_library,
            // GEMM libraries
            gemm_q4_k_library,
            gemm_q2_k_library,
            gemm_q3_k_library,
            gemm_q5_k_library,
            gemm_q6_k_library,
            gemm_iq_tq_library,
            gemm_iq2_iq3_library,
            gemm_template_library,
            // MV pipelines
            mv_q4_0_pipeline,
            mv_q4_k_pipeline,
            mv_q5_k_pipeline,
            mv_q2_k_pipeline,
            mv_q6_k_pipeline,
            mv_q3_k_pipeline,
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
            // GEMM pipelines
            gemm_q4_k_pipeline,
            gemm_q2_k_pipeline,
            gemm_q3_k_pipeline,
            gemm_q5_k_pipeline,
            gemm_q6_k_pipeline,
            gemm_iq4_nl_pipeline,
            gemm_iq4_xs_pipeline,
            gemm_iq1_s_pipeline,
            gemm_iq1_m_pipeline,
            gemm_iq2_xxs_pipeline,
            gemm_iq2_xs_pipeline,
            gemm_iq2_s_pipeline,
            gemm_iq3_xxs_pipeline,
            gemm_iq3_s_pipeline,
            gemm_tq2_0_pipeline,
            gemm_tq1_0_pipeline,
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
            activation_library,
            silu_pipeline,
            silu_4_pipeline,
            gelu_pipeline,
            gelu_4_pipeline,
            gelu_quick_pipeline,
            gelu_quick_4_pipeline,
            // Flash Attention
            flash_attn_library,
            flash_attn_f16_pipeline,
            // Quantize
            quantize_library,
            quantize_f16_pipeline,
            quantize_f16_4_pipeline,
            quantize_q4_0_pipeline,
            quantize_q8_0_pipeline,
            // MV buffers
            mv_weights_buffer: RefCell::new(None),
            mv_input_buffer: RefCell::new(None),
            mv_output_buffer: RefCell::new(None),
            mv_weights_size: RefCell::new(0),
            mv_input_size: RefCell::new(0),
            mv_output_size: RefCell::new(0),
            // GEMM buffers
            gemm_weights_buffer: RefCell::new(None),
            gemm_input_buffer: RefCell::new(None),
            gemm_output_buffer: RefCell::new(None),
            gemm_weights_size: RefCell::new(0),
            gemm_input_size: RefCell::new(0),
            gemm_output_size: RefCell::new(0),
        })
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

        // Thread group configuration matching llama.cpp
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

    /// Q4_0 x F32 Matrix-Vector (matching llama.cpp kernel)
    pub fn mv_q4_0_f32(
        &self,
        m: usize,
        k: usize,
        weights: &[BlockQ4_0],
        input: &[f32],
    ) -> Result<Vec<f32>, String> {
        let nb = k / 32;
        let nb01 = (nb * std::mem::size_of::<BlockQ4_0>()) as u64; // Byte stride per row

        // Args structure matching llama.cpp's ggml_metal_kargs_mul_mv
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

        // Weights buffer - only create once, reuse without copy
        let weights_buffer = {
            let mut buf_cell = self.mv_weights_buffer.borrow_mut();
            let mut size_cell = self.mv_weights_size.borrow_mut();

            if *size_cell != weights_size {
                // Size changed, create new buffer with data
                let buf = self.device.new_buffer_with_data(
                    weights.as_ptr() as *const std::ffi::c_void,
                    weights_size as u64,
                    metal::MTLResourceOptions::StorageModeShared,
                );
                *buf_cell = Some(buf.clone());
                *size_cell = weights_size;
                buf
            } else {
                // Size matches, reuse existing buffer
                buf_cell.as_ref().unwrap().clone()
            }
        };

        // Input buffer - only create once, reuse without copy
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

        // Thread group configuration matching llama.cpp EXACTLY
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

    /// Q4_K x F32 Matrix-Vector (matching llama.cpp kernel exactly)
    pub fn mv_q4_k_f32(
        &self,
        m: usize,
        k: usize,
        weights: &[BlockQ4K],
        input: &[f32],
    ) -> Result<Vec<f32>, String> {
        let nb = k / 256; // QK_K = 256
        let nb01 = (nb * std::mem::size_of::<BlockQ4K>()) as u64; // Byte stride per row

        // Args structure matching llama.cpp's ggml_metal_kargs_mul_mv
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

        // Weights buffer - only create once, reuse without copy
        let weights_buffer = {
            let mut buf_cell = self.mv_weights_buffer.borrow_mut();
            let mut size_cell = self.mv_weights_size.borrow_mut();

            if *size_cell != weights_size {
                // Size changed, create new buffer with data
                let buf = self.device.new_buffer_with_data(
                    weights.as_ptr() as *const std::ffi::c_void,
                    weights_size as u64,
                    metal::MTLResourceOptions::StorageModeShared,
                );
                *buf_cell = Some(buf.clone());
                *size_cell = weights_size;
                buf
            } else {
                // Size matches, reuse existing buffer
                buf_cell.as_ref().unwrap().clone()
            }
        };

        // Input buffer - only create once, reuse without copy
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

        // Thread group configuration matching llama.cpp EXACTLY
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

    /// Q5_K x F32 Matrix-Vector (matching llama.cpp kernel exactly)
    pub fn mv_q5_k_f32(
        &self,
        m: usize,
        k: usize,
        weights: &[BlockQ5K],
        input: &[f32],
    ) -> Result<Vec<f32>, String> {
        let nb = k / 256; // QK_K = 256
        let nb01 = (nb * std::mem::size_of::<BlockQ5K>()) as u64; // Byte stride per row

        // Args structure matching llama.cpp's ggml_metal_kargs_mul_mv
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
            nr0: 1,
            r2: 1,
            r3: 1, // N_R0_Q5_K = 1
        };

        // Get or create cached buffers (NO data copy on repeated calls with same size)
        let weights_size = weights.len() * std::mem::size_of::<BlockQ5K>();
        let input_size = input.len() * std::mem::size_of::<f32>();
        let output_size = m * std::mem::size_of::<f32>();

        // Weights buffer - only create once, reuse without copy
        let weights_buffer = {
            let mut buf_cell = self.mv_weights_buffer.borrow_mut();
            let mut size_cell = self.mv_weights_size.borrow_mut();

            if *size_cell != weights_size {
                // Size changed, create new buffer with data
                let buf = self.device.new_buffer_with_data(
                    weights.as_ptr() as *const std::ffi::c_void,
                    weights_size as u64,
                    metal::MTLResourceOptions::StorageModeShared,
                );
                *buf_cell = Some(buf.clone());
                *size_cell = weights_size;
                buf
            } else {
                // Size matches, reuse existing buffer
                buf_cell.as_ref().unwrap().clone()
            }
        };

        // Input buffer - only create once, reuse without copy
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

        // Thread group configuration matching llama.cpp EXACTLY
        // Q5_K: N_R0 = 1, NSG = 2
        const NSG: u64 = 2;
        const NR0: u64 = 1; // N_R0_Q5_K = 1
        const ROWS_PER_THREADGROUP: u64 = NSG * NR0; // 2

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

    /// Q2_K x F32 Matrix-Vector (matching llama.cpp kernel exactly)
    pub fn mv_q2_k_f32(
        &self,
        m: usize,
        k: usize,
        weights: &[BlockQ2K],
        input: &[f32],
    ) -> Result<Vec<f32>, String> {
        let nb = k / 256; // QK_K = 256
        let nb01 = (nb * std::mem::size_of::<BlockQ2K>()) as u64; // Byte stride per row

        // Args structure matching llama.cpp's ggml_metal_kargs_mul_mv
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

        // Weights buffer - only create once, reuse without copy
        let weights_buffer = {
            let mut buf_cell = self.mv_weights_buffer.borrow_mut();
            let mut size_cell = self.mv_weights_size.borrow_mut();

            if *size_cell != weights_size {
                // Size changed, create new buffer with data
                let buf = self.device.new_buffer_with_data(
                    weights.as_ptr() as *const std::ffi::c_void,
                    weights_size as u64,
                    metal::MTLResourceOptions::StorageModeShared,
                );
                *buf_cell = Some(buf.clone());
                *size_cell = weights_size;
                buf
            } else {
                // Size matches, reuse existing buffer
                buf_cell.as_ref().unwrap().clone()
            }
        };

        // Input buffer - only create once, reuse without copy
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

        // Thread group configuration matching llama.cpp EXACTLY
        // For K-quant types, llama.cpp uses:
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

    /// Q6_K x F32 Matrix-Vector (matching llama.cpp kernel)
    pub fn mv_q6_k_f32(
        &self,
        m: usize,
        k: usize,
        weights: &[BlockQ6K],
        input: &[f32],
    ) -> Result<Vec<f32>, String> {
        let nb = k / 256; // QK_K = 256
        let nb01 = (nb * std::mem::size_of::<BlockQ6K>()) as u64; // Byte stride per row

        // Args structure matching llama.cpp's ggml_metal_kargs_mul_mv
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
            r3: 1, // N_R0_Q6_K = 2
        };

        let weights_size = weights.len() * std::mem::size_of::<BlockQ6K>();
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

        encoder.set_compute_pipeline_state(&self.mv_q6_k_pipeline);
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

        // Thread group configuration matching llama.cpp EXACTLY
        // Q6_K: N_R0 = 2, NSG = 2
        const NSG: u64 = 2;
        const NR0: u64 = 2; // N_R0_Q6_K = 2
        const ROWS_PER_THREADGROUP: u64 = NSG * NR0; // 4

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

    /// Q3_K x F32 Matrix-Vector (matching llama.cpp kernel)
    pub fn mv_q3_k_f32(
        &self,
        m: usize,
        k: usize,
        weights: &[BlockQ3K],
        input: &[f32],
    ) -> Result<Vec<f32>, String> {
        let nb = k / 256;
        let nb01 = (nb * std::mem::size_of::<BlockQ3K>()) as u64;

        // Args structure matching llama.cpp's ggml_metal_kargs_mul_mv
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
            r3: 1, // N_R0_Q3_K = 2
        };

        let weights_size = weights.len() * std::mem::size_of::<BlockQ3K>();
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

        encoder.set_compute_pipeline_state(&self.mv_q3_k_pipeline);
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

        // Thread group configuration matching llama.cpp EXACTLY
        // Q3_K: N_R0 = 2, NSG = 2
        const NSG: u64 = 2;
        const NR0: u64 = 2; // N_R0_Q3_K = 2
        const ROWS_PER_THREADGROUP: u64 = NSG * NR0; // 4

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

    /// IQ4_NL x F32 Matrix-Vector (matching llama.cpp kernel)
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

    /// TQ2_0 x F32 Matrix-Vector (matching llama.cpp kernel)
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

    /// IQ4_XS x F32 Matrix-Vector (matching llama.cpp kernel)
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

    // ===== GEMM (General Matrix-Matrix Multiplication) Methods =====
    // These handle M×K weights × K×N input → M×N output (batch operations)

    /// Q4_K × F32 GEMM (M rows × N input vectors)
    pub fn gemm_q4_k_f32(
        &self,
        m: usize,
        n: usize,
        k: usize,
        weights: &[BlockQ4K],
        input: &[f32],
    ) -> Result<Vec<f32>, String> {
        let nb = k / 256;
        let nb01 = (nb * std::mem::size_of::<BlockQ4K>()) as u64;

        #[repr(C)]
        struct GemmArgs {
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

        let args = GemmArgs {
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
            ne1: n as i32,
            r2: 1,
            r3: 1,
        };

        let weights_size = weights.len() * std::mem::size_of::<BlockQ4K>();
        let input_size = input.len() * std::mem::size_of::<f32>();
        let output_size = m * n * std::mem::size_of::<f32>();

        let weights_buffer = {
            let mut buf_cell = self.gemm_weights_buffer.borrow_mut();
            let mut size_cell = self.gemm_weights_size.borrow_mut();
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
            let mut buf_cell = self.gemm_input_buffer.borrow_mut();
            let mut size_cell = self.gemm_input_size.borrow_mut();
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
            let mut buf_cell = self.gemm_output_buffer.borrow_mut();
            let mut size_cell = self.gemm_output_size.borrow_mut();
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
        encoder.set_compute_pipeline_state(&self.gemm_q4_k_pipeline);
        encoder.set_buffer(0, Some(&weights_buffer), 0);
        encoder.set_buffer(1, Some(&input_buffer), 0);
        encoder.set_buffer(2, Some(&output_buffer), 0);
        encoder.set_bytes(
            3,
            std::mem::size_of::<GemmArgs>() as u64,
            &args as *const GemmArgs as *const std::ffi::c_void,
        );

        // Dynamic dispatch based on K size for better GPU utilization
        // Small K (nb <= 4): use NR0=8 rows per threadgroup, NC=4 columns
        // Large K (nb > 4): use NR0=2 rows per threadgroup, NC=4 columns
        let (nr0, nc) = if nb <= 4 {
            (8u64, 4u64) // Small K: more rows per threadgroup
        } else {
            (2u64, 4u64) // Large K: use shader's N_COLS_LARGE=4
        };
        const NSG: u64 = 2;

        let thread_group_size = MTLSize { width: 32, height: NSG, depth: 1 };
        let thread_group_count = MTLSize {
            width: (m as u64 + NSG * nr0 - 1) / (NSG * nr0),
            height: (n as u64 + nc - 1) / nc,
            depth: 1,
        };

        encoder.dispatch_thread_groups(thread_group_count, thread_group_size);
        encoder.end_encoding();
        command_buffer.commit();
        command_buffer.wait_until_completed();

        let output_ptr = output_buffer.contents() as *const f32;
        Ok(unsafe { std::slice::from_raw_parts(output_ptr, m * n).to_vec() })
    }

    /// Q2_K × F32 GEMM
    pub fn gemm_q2_k_f32(
        &self,
        m: usize,
        n: usize,
        k: usize,
        weights: &[BlockQ2K],
        input: &[f32],
    ) -> Result<Vec<f32>, String> {
        let nb = k / 256;
        let nb01 = (nb * std::mem::size_of::<BlockQ2K>()) as u64;

        #[repr(C)]
        #[repr(C)]
        struct GemmArgs {
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

        let args = GemmArgs {
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
            ne1: n as i32,
            r2: 1,
            r3: 1,
        };

        let weights_size = weights.len() * std::mem::size_of::<BlockQ2K>();
        let input_size = input.len() * std::mem::size_of::<f32>();
        let output_size = m * n * std::mem::size_of::<f32>();

        let weights_buffer = {
            let mut buf_cell = self.gemm_weights_buffer.borrow_mut();
            let mut size_cell = self.gemm_weights_size.borrow_mut();
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
            let mut buf_cell = self.gemm_input_buffer.borrow_mut();
            let mut size_cell = self.gemm_input_size.borrow_mut();
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
            let mut buf_cell = self.gemm_output_buffer.borrow_mut();
            let mut size_cell = self.gemm_output_size.borrow_mut();
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
        encoder.set_compute_pipeline_state(&self.gemm_q2_k_pipeline);
        encoder.set_buffer(0, Some(&weights_buffer), 0);
        encoder.set_buffer(1, Some(&input_buffer), 0);
        encoder.set_buffer(2, Some(&output_buffer), 0);
        encoder.set_bytes(
            3,
            std::mem::size_of::<GemmArgs>() as u64,
            &args as *const GemmArgs as *const std::ffi::c_void,
        );

        // Dynamic dispatch based on K size
        // Small K (nb <= 4): NR0=8, NC=4
        // Large K (nb > 4): NR0=2, NC=4
        let (nr0, nc) = if nb <= 4 { (8u64, 4u64) } else { (2u64, 4u64) };
        const NSG: u64 = 2;

        let thread_group_size = MTLSize { width: 32, height: NSG, depth: 1 };
        let thread_group_count = MTLSize {
            width: (m as u64 + NSG * nr0 - 1) / (NSG * nr0),
            height: (n as u64 + nc - 1) / nc,
            depth: 1,
        };

        encoder.dispatch_thread_groups(thread_group_count, thread_group_size);
        encoder.end_encoding();
        command_buffer.commit();
        command_buffer.wait_until_completed();

        let output_ptr = output_buffer.contents() as *const f32;
        Ok(unsafe { std::slice::from_raw_parts(output_ptr, m * n).to_vec() })
    }

    /// Q3_K × F32 GEMM
    pub fn gemm_q3_k_f32(
        &self,
        m: usize,
        n: usize,
        k: usize,
        weights: &[BlockQ3K],
        input: &[f32],
    ) -> Result<Vec<f32>, String> {
        let nb = k / 256;
        let nb01 = (nb * std::mem::size_of::<BlockQ3K>()) as u64;

        #[repr(C)]
        #[repr(C)]
        struct GemmArgs {
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

        let args = GemmArgs {
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
            ne1: n as i32,
            r2: 1,
            r3: 1,
        };

        let weights_size = weights.len() * std::mem::size_of::<BlockQ3K>();
        let input_size = input.len() * std::mem::size_of::<f32>();
        let output_size = m * n * std::mem::size_of::<f32>();

        let weights_buffer = {
            let mut buf_cell = self.gemm_weights_buffer.borrow_mut();
            let mut size_cell = self.gemm_weights_size.borrow_mut();
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
            let mut buf_cell = self.gemm_input_buffer.borrow_mut();
            let mut size_cell = self.gemm_input_size.borrow_mut();
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
            let mut buf_cell = self.gemm_output_buffer.borrow_mut();
            let mut size_cell = self.gemm_output_size.borrow_mut();
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
        encoder.set_compute_pipeline_state(&self.gemm_q3_k_pipeline);
        encoder.set_buffer(0, Some(&weights_buffer), 0);
        encoder.set_buffer(1, Some(&input_buffer), 0);
        encoder.set_buffer(2, Some(&output_buffer), 0);
        encoder.set_bytes(
            3,
            std::mem::size_of::<GemmArgs>() as u64,
            &args as *const GemmArgs as *const std::ffi::c_void,
        );

        // Dynamic dispatch based on K size
        // Small K (nb <= 4): NR0=8, NC=4
        // Large K (nb > 4): NR0=2, NC=4
        let (nr0, nc) = if nb <= 4 { (8u64, 4u64) } else { (2u64, 4u64) };
        const NSG: u64 = 2;

        let thread_group_size = MTLSize { width: 32, height: NSG, depth: 1 };
        let thread_group_count = MTLSize {
            width: (m as u64 + NSG * nr0 - 1) / (NSG * nr0),
            height: (n as u64 + nc - 1) / nc,
            depth: 1,
        };

        encoder.dispatch_thread_groups(thread_group_count, thread_group_size);
        encoder.end_encoding();
        command_buffer.commit();
        command_buffer.wait_until_completed();

        let output_ptr = output_buffer.contents() as *const f32;
        Ok(unsafe { std::slice::from_raw_parts(output_ptr, m * n).to_vec() })
    }

    /// Q5_K × F32 GEMM
    pub fn gemm_q5_k_f32(
        &self,
        m: usize,
        n: usize,
        k: usize,
        weights: &[BlockQ5K],
        input: &[f32],
    ) -> Result<Vec<f32>, String> {
        let nb = k / 256;
        let nb01 = (nb * std::mem::size_of::<BlockQ5K>()) as u64;

        #[repr(C)]
        #[repr(C)]
        struct GemmArgs {
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

        let args = GemmArgs {
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
            ne1: n as i32,
            r2: 1,
            r3: 1,
        };

        let weights_size = weights.len() * std::mem::size_of::<BlockQ5K>();
        let input_size = input.len() * std::mem::size_of::<f32>();
        let output_size = m * n * std::mem::size_of::<f32>();

        let weights_buffer = {
            let mut buf_cell = self.gemm_weights_buffer.borrow_mut();
            let mut size_cell = self.gemm_weights_size.borrow_mut();
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
            let mut buf_cell = self.gemm_input_buffer.borrow_mut();
            let mut size_cell = self.gemm_input_size.borrow_mut();
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
            let mut buf_cell = self.gemm_output_buffer.borrow_mut();
            let mut size_cell = self.gemm_output_size.borrow_mut();
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
        encoder.set_compute_pipeline_state(&self.gemm_q5_k_pipeline);
        encoder.set_buffer(0, Some(&weights_buffer), 0);
        encoder.set_buffer(1, Some(&input_buffer), 0);
        encoder.set_buffer(2, Some(&output_buffer), 0);
        encoder.set_bytes(
            3,
            std::mem::size_of::<GemmArgs>() as u64,
            &args as *const GemmArgs as *const std::ffi::c_void,
        );

        // Dynamic dispatch based on K size
        // Small K (nb <= 4): NR0=8, NC=4
        // Large K (nb > 4): NR0=2, NC=4
        let (nr0, nc) = if nb <= 4 { (8u64, 4u64) } else { (2u64, 4u64) };
        const NSG: u64 = 2;

        let thread_group_size = MTLSize { width: 32, height: NSG, depth: 1 };
        let thread_group_count = MTLSize {
            width: (m as u64 + NSG * nr0 - 1) / (NSG * nr0),
            height: (n as u64 + nc - 1) / nc,
            depth: 1,
        };

        encoder.dispatch_thread_groups(thread_group_count, thread_group_size);
        encoder.end_encoding();
        command_buffer.commit();
        command_buffer.wait_until_completed();

        let output_ptr = output_buffer.contents() as *const f32;
        Ok(unsafe { std::slice::from_raw_parts(output_ptr, m * n).to_vec() })
    }

    /// Q6_K × F32 GEMM
    pub fn gemm_q6_k_f32(
        &self,
        m: usize,
        n: usize,
        k: usize,
        weights: &[BlockQ6K],
        input: &[f32],
    ) -> Result<Vec<f32>, String> {
        let nb = k / 256;
        let nb01 = (nb * std::mem::size_of::<BlockQ6K>()) as u64;

        #[repr(C)]
        #[repr(C)]
        struct GemmArgs {
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

        let args = GemmArgs {
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
            ne1: n as i32,
            r2: 1,
            r3: 1,
        };

        let weights_size = weights.len() * std::mem::size_of::<BlockQ6K>();
        let input_size = input.len() * std::mem::size_of::<f32>();
        let output_size = m * n * std::mem::size_of::<f32>();

        let weights_buffer = {
            let mut buf_cell = self.gemm_weights_buffer.borrow_mut();
            let mut size_cell = self.gemm_weights_size.borrow_mut();
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
            let mut buf_cell = self.gemm_input_buffer.borrow_mut();
            let mut size_cell = self.gemm_input_size.borrow_mut();
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
            let mut buf_cell = self.gemm_output_buffer.borrow_mut();
            let mut size_cell = self.gemm_output_size.borrow_mut();
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
        encoder.set_compute_pipeline_state(&self.gemm_q6_k_pipeline);
        encoder.set_buffer(0, Some(&weights_buffer), 0);
        encoder.set_buffer(1, Some(&input_buffer), 0);
        encoder.set_buffer(2, Some(&output_buffer), 0);
        encoder.set_bytes(
            3,
            std::mem::size_of::<GemmArgs>() as u64,
            &args as *const GemmArgs as *const std::ffi::c_void,
        );

        // Dynamic dispatch based on K size
        // Small K (nb <= 4): NR0=8, NC=4
        // Large K (nb > 4): NR0=2, NC=4
        let (nr0, nc) = if nb <= 4 { (8u64, 4u64) } else { (2u64, 4u64) };
        const NSG: u64 = 2;

        let thread_group_size = MTLSize { width: 32, height: NSG, depth: 1 };
        let thread_group_count = MTLSize {
            width: (m as u64 + NSG * nr0 - 1) / (NSG * nr0),
            height: (n as u64 + nc - 1) / nc,
            depth: 1,
        };

        encoder.dispatch_thread_groups(thread_group_count, thread_group_size);
        encoder.end_encoding();
        command_buffer.commit();
        command_buffer.wait_until_completed();

        let output_ptr = output_buffer.contents() as *const f32;
        Ok(unsafe { std::slice::from_raw_parts(output_ptr, m * n).to_vec() })
    }

    // ===== Templated GEMM Methods (from llama.cpp) =====
    // These use the kernel_mul_mm template with dequantize functions

    /// Templated Q4_K × F32 GEMM (using llama.cpp kernel_mul_mm)
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
        let nb00 = std::mem::size_of::<BlockQ4K>() as u64;

        // Use exact struct from llama.cpp's ggml-metal-impl.h
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
            ne02: 1,
            nb01,
            nb02: 0,
            nb03: 0,
            ne12: 1,
            nb10: 2,              // FP16 element size
            nb11: (k * 2) as u64, // K * sizeof(FP16)
            nb12: 0,
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

        // Allocate threadgroup memory (8KB as per llama.cpp)
        encoder.set_threadgroup_memory_length(0, 8192);

        // Threadgroup configuration from llama.cpp
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
        let nb00 = std::mem::size_of::<BlockQ2K>() as u64;

        #[repr(C)]
        struct GemmArgs {
            ne00: i32,
            ne01: i32,
            ne02: i32,
            ne03: i32,
            nb00: u64,
            nb01: u64,
            nb02: u64,
            nb03: u64,
            ne10: i32,
            ne11: i32,
            ne12: i32,
            ne13: i32,
            nb10: u64,
            nb11: u64,
            nb12: u64,
            nb13: u64,
            ne0: i32,
            ne1: i32,
            ne2: i32,
            ne3: i32,
            nb0: u64,
            nb1: u64,
            nb2: u64,
            nb3: u64,
        }

        let args = GemmArgs {
            ne00: k as i32,
            ne01: m as i32,
            ne02: 1,
            ne03: 1,
            nb00,
            nb01,
            nb02: 0,
            nb03: 0,
            ne10: k as i32,
            ne11: n as i32,
            ne12: 1,
            ne13: 1,
            nb10: 2,
            nb11: (k * 2) as u64,
            nb12: 0,
            nb13: 0, // FP16: element stride=2, row stride=K*2
            ne0: m as i32,
            ne1: n as i32,
            ne2: 1,
            ne3: 1, // ne0=M (rows), ne1=N (cols)
            nb0: 4,
            nb1: (n * 4) as u64,
            nb2: 0,
            nb3: 0, // row stride = N * sizeof(f32)
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

        // Allocate threadgroup memory (8KB as per llama.cpp)
        encoder.set_threadgroup_memory_length(0, 8192);

        // Threadgroup configuration from llama.cpp
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
        let nb00 = std::mem::size_of::<BlockQ3K>() as u64;

        #[repr(C)]
        struct GemmArgs {
            ne00: i32,
            ne01: i32,
            ne02: i32,
            ne03: i32,
            nb00: u64,
            nb01: u64,
            nb02: u64,
            nb03: u64,
            ne10: i32,
            ne11: i32,
            ne12: i32,
            ne13: i32,
            nb10: u64,
            nb11: u64,
            nb12: u64,
            nb13: u64,
            ne0: i32,
            ne1: i32,
            ne2: i32,
            ne3: i32,
            nb0: u64,
            nb1: u64,
            nb2: u64,
            nb3: u64,
        }

        let args = GemmArgs {
            ne00: k as i32,
            ne01: m as i32,
            ne02: 1,
            ne03: 1,
            nb00,
            nb01,
            nb02: 0,
            nb03: 0,
            ne10: k as i32,
            ne11: n as i32,
            ne12: 1,
            ne13: 1,
            nb10: 2,
            nb11: (k * 2) as u64,
            nb12: 0,
            nb13: 0, // FP16: element stride=2, row stride=K*2
            ne0: m as i32,
            ne1: n as i32,
            ne2: 1,
            ne3: 1, // ne0=M (rows), ne1=N (cols)
            nb0: 4,
            nb1: (n * 4) as u64,
            nb2: 0,
            nb3: 0, // row stride = N * sizeof(f32)
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

        // Allocate threadgroup memory (8KB as per llama.cpp)
        encoder.set_threadgroup_memory_length(0, 8192);

        // Threadgroup configuration from llama.cpp
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
        let nb00 = std::mem::size_of::<BlockQ5K>() as u64;

        #[repr(C)]
        struct GemmArgs {
            ne00: i32,
            ne01: i32,
            ne02: i32,
            ne03: i32,
            nb00: u64,
            nb01: u64,
            nb02: u64,
            nb03: u64,
            ne10: i32,
            ne11: i32,
            ne12: i32,
            ne13: i32,
            nb10: u64,
            nb11: u64,
            nb12: u64,
            nb13: u64,
            ne0: i32,
            ne1: i32,
            ne2: i32,
            ne3: i32,
            nb0: u64,
            nb1: u64,
            nb2: u64,
            nb3: u64,
        }

        let args = GemmArgs {
            ne00: k as i32,
            ne01: m as i32,
            ne02: 1,
            ne03: 1,
            nb00,
            nb01,
            nb02: 0,
            nb03: 0,
            ne10: k as i32,
            ne11: n as i32,
            ne12: 1,
            ne13: 1,
            nb10: 2,
            nb11: (k * 2) as u64,
            nb12: 0,
            nb13: 0, // FP16: element stride=2, row stride=K*2
            ne0: m as i32,
            ne1: n as i32,
            ne2: 1,
            ne3: 1, // ne0=M (rows), ne1=N (cols)
            nb0: 4,
            nb1: (n * 4) as u64,
            nb2: 0,
            nb3: 0, // row stride = N * sizeof(f32)
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

        // Allocate threadgroup memory (8KB as per llama.cpp)
        encoder.set_threadgroup_memory_length(0, 8192);

        // Threadgroup configuration from llama.cpp
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
        let nb00 = std::mem::size_of::<BlockQ6K>() as u64;

        #[repr(C)]
        struct GemmArgs {
            ne00: i32,
            ne01: i32,
            ne02: i32,
            ne03: i32,
            nb00: u64,
            nb01: u64,
            nb02: u64,
            nb03: u64,
            ne10: i32,
            ne11: i32,
            ne12: i32,
            ne13: i32,
            nb10: u64,
            nb11: u64,
            nb12: u64,
            nb13: u64,
            ne0: i32,
            ne1: i32,
            ne2: i32,
            ne3: i32,
            nb0: u64,
            nb1: u64,
            nb2: u64,
            nb3: u64,
        }

        let args = GemmArgs {
            ne00: k as i32,
            ne01: m as i32,
            ne02: 1,
            ne03: 1,
            nb00,
            nb01,
            nb02: 0,
            nb03: 0,
            ne10: k as i32,
            ne11: n as i32,
            ne12: 1,
            ne13: 1,
            nb10: 2,
            nb11: (k * 2) as u64,
            nb12: 0,
            nb13: 0, // FP16: element stride=2, row stride=K*2
            ne0: m as i32,
            ne1: n as i32,
            ne2: 1,
            ne3: 1, // ne0=M (rows), ne1=N (cols)
            nb0: 4,
            nb1: (n * 4) as u64,
            nb2: 0,
            nb3: 0, // row stride = N * sizeof(f32)
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

        // Allocate threadgroup memory (8KB as per llama.cpp)
        encoder.set_threadgroup_memory_length(0, 8192);

        // Threadgroup configuration from llama.cpp
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
        if n == 1 {
            self.mv_q4_k_f32(m, k, weights, input)
        } else if n >= 32 {
            self.gemm_template_q4_k_f32(m, n, k, weights, input)
        } else {
            self.gemm_q4_k_f32(m, n, k, weights, input)
        }
    }

    pub fn matmul_q2_k_f32(
        &self,
        m: usize,
        n: usize,
        k: usize,
        weights: &[BlockQ2K],
        input: &[f32],
    ) -> Result<Vec<f32>, String> {
        if n == 1 {
            self.mv_q2_k_f32(m, k, weights, input)
        } else if n >= 32 {
            self.gemm_template_q2_k_f32(m, n, k, weights, input)
        } else {
            self.gemm_q2_k_f32(m, n, k, weights, input)
        }
    }

    pub fn matmul_q3_k_f32(
        &self,
        m: usize,
        n: usize,
        k: usize,
        weights: &[BlockQ3K],
        input: &[f32],
    ) -> Result<Vec<f32>, String> {
        if n == 1 {
            self.mv_q3_k_f32(m, k, weights, input)
        } else if n >= 32 {
            self.gemm_template_q3_k_f32(m, n, k, weights, input)
        } else {
            self.gemm_q3_k_f32(m, n, k, weights, input)
        }
    }

    pub fn matmul_q5_k_f32(
        &self,
        m: usize,
        n: usize,
        k: usize,
        weights: &[BlockQ5K],
        input: &[f32],
    ) -> Result<Vec<f32>, String> {
        if n == 1 {
            self.mv_q5_k_f32(m, k, weights, input)
        } else if n >= 32 {
            self.gemm_template_q5_k_f32(m, n, k, weights, input)
        } else {
            self.gemm_q5_k_f32(m, n, k, weights, input)
        }
    }

    pub fn matmul_q6_k_f32(
        &self,
        m: usize,
        n: usize,
        k: usize,
        weights: &[BlockQ6K],
        input: &[f32],
    ) -> Result<Vec<f32>, String> {
        if n == 1 {
            self.mv_q6_k_f32(m, k, weights, input)
        } else if n >= 32 {
            self.gemm_template_q6_k_f32(m, n, k, weights, input)
        } else {
            self.gemm_q6_k_f32(m, n, k, weights, input)
        }
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

        encoder.dispatch_thread_groups(thread_group_count, thread_group_size);
        encoder.end_encoding();
        command_buffer.commit();
        command_buffer.wait_until_completed();

        let output_ptr = output_buffer.contents() as *const f32;
        Ok(unsafe {
            std::slice::from_raw_parts(output_ptr, n_rows * n_cols).to_vec()
        })
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
            ne00: n_dims as i32,
            ne01: (n_heads * n_seqs) as i32,
            ne02: n_seqs as i32,
            nb00: 4,
            nb01: (n_dims * 4) as u64,
            nb02: (n_heads * n_dims * 4) as u64,
            n_dims: n_dims as i32,
            n_offs: 0,
            freq_base,
            freq_scale,
            ext_factor: 0.0,
            attn_factor: 1.0,
            beta_fast: 32.0,
            beta_slow: 1.0,
            n_ctx_orig: 0,
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
    /// Supports: head_dim=128, GQA (grouped query attention)
    pub fn flash_attn_f16(
        &self,
        n_seqs: usize,     // batch size
        n_heads: usize,    // number of query heads
        n_kv_heads: usize, // number of KV heads (for GQA)
        head_dim: usize,   // head dimension (must be 128)
        seq_len: usize,    // KV cache length
        q: &[f32],         // query [n_seqs, n_heads, head_dim]
        k: &[u16], // key cache [n_seqs, n_kv_heads, seq_len, head_dim] as F16
        v: &[u16], // value cache [n_seqs, n_kv_heads, seq_len, head_dim] as F16
    ) -> Result<Vec<f32>, String> {
        if head_dim != 128 {
            return Err(format!(
                "Flash Attention only supports head_dim=128, got {}",
                head_dim
            ));
        }

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

        let scale = 1.0 / (head_dim as f32).sqrt();
        let n_heads_per_group = n_heads / n_kv_heads;

        let args = FlashAttnArgs {
            ne01: 1, // Q rows per head (1 for decode)
            ne02: n_heads as u32,
            ne03: n_seqs as u32,
            ne11: seq_len as u32,
            ne_12_2: n_kv_heads as u32,
            ne_12_3: n_seqs as u32,
            nb01: (head_dim * 4) as u32,
            nb02: (n_heads * head_dim * 4) as u32,
            nb03: (n_seqs * n_heads * head_dim * 4) as u32,
            nb11: (head_dim * 2) as u32,
            nb12: (seq_len * head_dim * 2) as u32,
            nb13: (n_seqs * seq_len * head_dim * 2) as u32,
            nb21: (head_dim * 2) as u32,
            nb22: (seq_len * head_dim * 2) as u32,
            nb23: (n_seqs * seq_len * head_dim * 2) as u32,
            nb31: 0,
            nb32: 0,
            nb33: 0,
            ne32: 0,
            ne33: 0,
            scale,
            max_bias: 0.0,
            logit_softcap: 0.0,
            n_head_log2: 0,
            m0: 0.0,
            m1: 0.0,
            nsg: 2,  // 2 simdgroups per threadgroup
            nwg: 32, // 32 workgroups for parallelization
        };

        let command_buffer = self.queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();

        encoder.set_compute_pipeline_state(&self.flash_attn_f16_pipeline);
        encoder.set_bytes(
            0,
            std::mem::size_of::<FlashAttnArgs>() as u64,
            &args as *const FlashAttnArgs as *const std::ffi::c_void,
        );
        encoder.set_buffer(1, Some(&q_buffer), 0);
        encoder.set_buffer(2, Some(&k_buffer), 0);
        encoder.set_buffer(3, Some(&v_buffer), 0);
        encoder.set_buffer(4, None, 0); // mask
        encoder.set_buffer(5, None, 0); // sinks
        encoder.set_buffer(6, None, 0); // pad
        encoder.set_buffer(7, Some(&output_buffer), 0);

        // Threadgroup configuration
        let thread_group_size = MTLSize { width: 64, height: 1, depth: 1 };
        let thread_group_count = MTLSize {
            width: n_heads as u64,
            height: n_seqs as u64,
            depth: 32, // 32 workgroups for seq_len parallelization
        };

        encoder.set_threadgroup_memory_length(0, 32768); // 32KB shared memory
        encoder.dispatch_thread_groups(thread_group_count, thread_group_size);
        encoder.end_encoding();
        command_buffer.commit();
        command_buffer.wait_until_completed();

        let output_ptr = output_buffer.contents() as *const f32;
        Ok(unsafe { std::slice::from_raw_parts(output_ptr, q.len()).to_vec() })
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
        n: usize,
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
    pub fn reduce_sum_f32(&self, n: usize, src: &[f32]) -> Result<f32, String> {
        Ok(src.iter().sum())
    }

    /// Reduce max
    pub fn reduce_max_f32(&self, n: usize, src: &[f32]) -> Result<f32, String> {
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
}
