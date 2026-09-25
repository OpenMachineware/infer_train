use metal::{Device, CommandQueue, Library, ComputePipelineState, MTLSize, Buffer};
use crate::quant::types::{BlockIQ4NL, BlockQ8_0, BlockQ4_0, BlockQ4K, BlockQ5K, BlockQ2K, BlockQ6K, BlockQ3K, BlockTQ2_0, BlockTQ1_0, BlockIQ4XS, BlockIQ1S, BlockIQ1M, BlockIQ2XXS, BlockIQ2XS, BlockIQ2S, BlockIQ3XXS, BlockIQ3S};
use std::cell::RefCell;

pub struct MetalContext {
    device: Device,
    queue: CommandQueue,
    vec_dot_library: Library,
    mv_library: Library,
    mv_simd_library: Library,
    mv_q4_0_library: Library,
    mv_q4_k_library: Library,
    mv_q5_k_library: Library,
    mv_q2_k_library: Library,
    mv_q6_k_library: Library,
    mv_q3_k_library: Library,
    mv_iq_tq_library: Library,
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

impl MetalContext {
    pub fn new() -> Result<Self, String> {
        let device = Device::system_default()
            .ok_or("No Metal device found")?;

        let queue = device.new_command_queue();

        // Set optimization level for Metal shader compilation
        let compile_options = metal::CompileOptions::new();
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

        let mv_q4_0_source = include_str!("../../../../shaders/mv_q4_0.metal");
        let mv_q4_0_library = device
            .new_library_with_source(mv_q4_0_source, &compile_options)
            .map_err(|e| format!("Failed to compile mv_q4_0 library: {}", e))?;

        let mv_q4_k_source = include_str!("../../../../shaders/mv_q4_k.metal");
        let mv_q4_k_library = device
            .new_library_with_source(mv_q4_k_source, &compile_options)
            .map_err(|e| format!("Failed to compile mv_q4_k library: {}", e))?;

        let mv_q5_k_source = include_str!("../../../../shaders/mv_q5_k.metal");
        let mv_q5_k_library = device
            .new_library_with_source(mv_q5_k_source, &compile_options)
            .map_err(|e| format!("Failed to compile mv_q5_k library: {}", e))?;

        let mv_q2_k_source = include_str!("../../../../shaders/mv_q2_k.metal");
        let mv_q2_k_library = device
            .new_library_with_source(mv_q2_k_source, &compile_options)
            .map_err(|e| format!("Failed to compile mv_q2_k library: {}", e))?;

        let mv_q6_k_source = include_str!("../../../../shaders/mv_q6_k.metal");
        let mv_q6_k_library = device
            .new_library_with_source(mv_q6_k_source, &compile_options)
            .map_err(|e| format!("Failed to compile mv_q6_k library: {}", e))?;

        let mv_q3_k_source = include_str!("../../../../shaders/mv_q3_k.metal");
        let mv_q3_k_library = device
            .new_library_with_source(mv_q3_k_source, &compile_options)
            .map_err(|e| format!("Failed to compile mv_q3_k library: {}", e))?;

        let mv_iq_tq_source = include_str!("../../../../shaders/mv_iq_tq.metal");
        let iq_grid_tables = include_str!("../../../../shaders/iq_grid_tables.h");
        let mv_iq_tq_combined = mv_iq_tq_source.replace("#include \"iq_grid_tables.h\"", iq_grid_tables);
        let mv_iq_tq_library = device
            .new_library_with_source(&mv_iq_tq_combined, &compile_options)
            .map_err(|e| format!("Failed to compile mv_iq_tq library: {}", e))?;

        // Compile GEMM libraries
        let gemm_q4_k_source = include_str!("../../../../shaders/gemm_q4_k.metal");
        let gemm_q4_k_library = device
            .new_library_with_source(gemm_q4_k_source, &compile_options)
            .map_err(|e| format!("Failed to compile gemm_q4_k library: {}", e))?;

        let gemm_q2_k_source = include_str!("../../../../shaders/gemm_q2_k.metal");
        let gemm_q2_k_library = device
            .new_library_with_source(gemm_q2_k_source, &compile_options)
            .map_err(|e| format!("Failed to compile gemm_q2_k library: {}", e))?;

        let gemm_q3_k_source = include_str!("../../../../shaders/gemm_q3_k.metal");
        let gemm_q3_k_library = device
            .new_library_with_source(gemm_q3_k_source, &compile_options)
            .map_err(|e| format!("Failed to compile gemm_q3_k library: {}", e))?;

        let gemm_q5_k_source = include_str!("../../../../shaders/gemm_q5_k.metal");
        let gemm_q5_k_library = device
            .new_library_with_source(gemm_q5_k_source, &compile_options)
            .map_err(|e| format!("Failed to compile gemm_q5_k library: {}", e))?;

        let gemm_q6_k_source = include_str!("../../../../shaders/gemm_q6_k.metal");
        let gemm_q6_k_library = device
            .new_library_with_source(gemm_q6_k_source, &compile_options)
            .map_err(|e| format!("Failed to compile gemm_q6_k library: {}", e))?;

        let gemm_iq_tq_source = include_str!("../../../../shaders/gemm_iq_tq.metal");
        let gemm_iq_tq_combined = gemm_iq_tq_source.replace("#include \"iq_grid_tables.h\"", iq_grid_tables);
        let gemm_iq_tq_library = device
            .new_library_with_source(&gemm_iq_tq_combined, &compile_options)
            .map_err(|e| format!("Failed to compile gemm_iq_tq library: {}", e))?;

        let gemm_iq2_iq3_source = include_str!("../../../../shaders/gemm_iq2_iq3.metal");
        let gemm_iq2_iq3_combined = gemm_iq2_iq3_source.replace("#include \"iq_grid_tables.h\"", iq_grid_tables);
        let gemm_iq2_iq3_library = device
            .new_library_with_source(&gemm_iq2_iq3_combined, &compile_options)
            .map_err(|e| format!("Failed to compile gemm_iq2_iq3 library: {}", e))?;

        // Compile templated GEMM library (self-contained, no C++ headers)
        let gemm_template_source = include_str!("../../../../shaders/mul_mm_standalone.metal");
        let gemm_template_library = device
            .new_library_with_source(gemm_template_source, &compile_options)
            .map_err(|e| format!("Failed to compile gemm_template library: {}", e))?;

        // Create cached pipeline for Q4_0 MV
        let mv_q4_0_kernel = mv_q4_0_library.get_function("kernel_mul_mv_q4_0_f32", None)
            .map_err(|e| format!("Failed to get Q4_0 kernel: {}", e))?;
        let mv_q4_0_pipeline = device.new_compute_pipeline_state_with_function(&mv_q4_0_kernel)
            .map_err(|e| format!("Failed to create Q4_0 pipeline: {}", e))?;

        // Create cached pipeline for Q4_K MV
        let mv_q4_k_kernel = mv_q4_k_library.get_function("kernel_mul_mv_q4_K_f32", None)
            .map_err(|e| format!("Failed to get Q4_K kernel: {}", e))?;
        let mv_q4_k_pipeline = device.new_compute_pipeline_state_with_function(&mv_q4_k_kernel)
            .map_err(|e| format!("Failed to create Q4_K pipeline: {}", e))?;

        // Create cached pipeline for Q5_K MV
        let mv_q5_k_kernel = mv_q5_k_library.get_function("kernel_mul_mv_q5_K_f32", None)
            .map_err(|e| format!("Failed to get Q5_K kernel: {}", e))?;
        let mv_q5_k_pipeline = device.new_compute_pipeline_state_with_function(&mv_q5_k_kernel)
            .map_err(|e| format!("Failed to create Q5_K pipeline: {}", e))?;

        // Create cached pipeline for Q2_K MV
        let mv_q2_k_kernel = mv_q2_k_library.get_function("kernel_mul_mv_q2_K_f32", None)
            .map_err(|e| format!("Failed to get Q2_K kernel: {}", e))?;
        let mv_q2_k_pipeline = device.new_compute_pipeline_state_with_function(&mv_q2_k_kernel)
            .map_err(|e| format!("Failed to create Q2_K pipeline: {}", e))?;

        // Create cached pipeline for Q6_K MV
        let mv_q6_k_kernel = mv_q6_k_library.get_function("kernel_mul_mv_q6_K_f32", None)
            .map_err(|e| format!("Failed to get Q6_K kernel: {}", e))?;
        let mv_q6_k_pipeline = device.new_compute_pipeline_state_with_function(&mv_q6_k_kernel)
            .map_err(|e| format!("Failed to create Q6_K pipeline: {}", e))?;

        // Create cached pipeline for Q3_K MV
        let mv_q3_k_kernel = mv_q3_k_library.get_function("kernel_mul_mv_q3_K_f32", None)
            .map_err(|e| format!("Failed to get Q3_K kernel: {}", e))?;
        let mv_q3_k_pipeline = device.new_compute_pipeline_state_with_function(&mv_q3_k_kernel)
            .map_err(|e| format!("Failed to create Q3_K pipeline: {}", e))?;

        // Create cached pipeline for IQ4_NL MV
        let mv_iq4_nl_kernel = mv_iq_tq_library.get_function("kernel_mul_mv_iq4_nl_f32", None)
            .map_err(|e| format!("Failed to get IQ4_NL kernel: {}", e))?;
        let mv_iq4_nl_pipeline = device.new_compute_pipeline_state_with_function(&mv_iq4_nl_kernel)
            .map_err(|e| format!("Failed to create IQ4_NL pipeline: {}", e))?;

        // Create cached pipeline for TQ2_0 MV
        let mv_tq2_0_kernel = mv_iq_tq_library.get_function("kernel_mul_mv_tq2_0_f32", None)
            .map_err(|e| format!("Failed to get TQ2_0 kernel: {}", e))?;
        let mv_tq2_0_pipeline = device.new_compute_pipeline_state_with_function(&mv_tq2_0_kernel)
            .map_err(|e| format!("Failed to create TQ2_0 pipeline: {}", e))?;

        // Create cached pipeline for IQ4_XS MV
        let mv_iq4_xs_kernel = mv_iq_tq_library.get_function("kernel_mul_mv_iq4_xs_f32", None)
            .map_err(|e| format!("Failed to get IQ4_XS kernel: {}", e))?;
        let mv_iq4_xs_pipeline = device.new_compute_pipeline_state_with_function(&mv_iq4_xs_kernel)
            .map_err(|e| format!("Failed to create IQ4_XS pipeline: {}", e))?;

        // Create cached pipeline for TQ1_0 MV
        let mv_tq1_0_kernel = mv_iq_tq_library.get_function("kernel_mul_mv_tq1_0_f32", None)
            .map_err(|e| format!("Failed to get TQ1_0 kernel: {}", e))?;
        let mv_tq1_0_pipeline = device.new_compute_pipeline_state_with_function(&mv_tq1_0_kernel)
            .map_err(|e| format!("Failed to create TQ1_0 pipeline: {}", e))?;

        // Create pipelines for IQ series
        let mv_iq1_s_kernel = mv_iq_tq_library.get_function("kernel_mul_mv_iq1_s_f32", None)
            .map_err(|e| format!("Failed to get IQ1_S kernel: {}", e))?;
        let mv_iq1_s_pipeline = device.new_compute_pipeline_state_with_function(&mv_iq1_s_kernel)
            .map_err(|e| format!("Failed to create IQ1_S pipeline: {}", e))?;

        let mv_iq1_m_kernel = mv_iq_tq_library.get_function("kernel_mul_mv_iq1_m_f32", None)
            .map_err(|e| format!("Failed to get IQ1_M kernel: {}", e))?;
        let mv_iq1_m_pipeline = device.new_compute_pipeline_state_with_function(&mv_iq1_m_kernel)
            .map_err(|e| format!("Failed to create IQ1_M pipeline: {}", e))?;

        let mv_iq2_xxs_kernel = mv_iq_tq_library.get_function("kernel_mul_mv_iq2_xxs_f32", None)
            .map_err(|e| format!("Failed to get IQ2_XXS kernel: {}", e))?;
        let mv_iq2_xxs_pipeline = device.new_compute_pipeline_state_with_function(&mv_iq2_xxs_kernel)
            .map_err(|e| format!("Failed to create IQ2_XXS pipeline: {}", e))?;

        let mv_iq2_xxs_split1_kernel = mv_iq_tq_library.get_function("kernel_mul_mv_iq2_xxs_f32_split1", None)
            .map_err(|e| format!("Failed to get IQ2_XXS split1 kernel: {}", e))?;
        let mv_iq2_xxs_split1_pipeline = device.new_compute_pipeline_state_with_function(&mv_iq2_xxs_split1_kernel)
            .map_err(|e| format!("Failed to create IQ2_XXS split1 pipeline: {}", e))?;

        let mv_iq2_xs_kernel = mv_iq_tq_library.get_function("kernel_mul_mv_iq2_xs_f32", None)
            .map_err(|e| format!("Failed to get IQ2_XS kernel: {}", e))?;
        let mv_iq2_xs_pipeline = device.new_compute_pipeline_state_with_function(&mv_iq2_xs_kernel)
            .map_err(|e| format!("Failed to create IQ2_XS pipeline: {}", e))?;

        let mv_iq2_xs_split1_kernel = mv_iq_tq_library.get_function("kernel_mul_mv_iq2_xs_f32_split1", None)
            .map_err(|e| format!("Failed to get IQ2_XS split1 kernel: {}", e))?;
        let mv_iq2_xs_split1_pipeline = device.new_compute_pipeline_state_with_function(&mv_iq2_xs_split1_kernel)
            .map_err(|e| format!("Failed to create IQ2_XS split1 pipeline: {}", e))?;

        let mv_iq2_s_kernel = mv_iq_tq_library.get_function("kernel_mul_mv_iq2_s_f32_split0", None)
            .map_err(|e| format!("Failed to get IQ2_S kernel: {}", e))?;
        let mv_iq2_s_pipeline = device.new_compute_pipeline_state_with_function(&mv_iq2_s_kernel)
            .map_err(|e| format!("Failed to create IQ2_S pipeline: {}", e))?;

        let mv_iq2_s_split1_kernel = mv_iq_tq_library.get_function("kernel_mul_mv_iq2_s_f32_split1", None)
            .map_err(|e| format!("Failed to get IQ2_S split1 kernel: {}", e))?;
        let mv_iq2_s_split1_pipeline = device.new_compute_pipeline_state_with_function(&mv_iq2_s_split1_kernel)
            .map_err(|e| format!("Failed to create IQ2_S split1 pipeline: {}", e))?;

        let mv_iq3_xxs_kernel = mv_iq_tq_library.get_function("kernel_mul_mv_iq3_xxs_f32", None)
            .map_err(|e| format!("Failed to get IQ3_XXS kernel: {}", e))?;
        let mv_iq3_xxs_pipeline = device.new_compute_pipeline_state_with_function(&mv_iq3_xxs_kernel)
            .map_err(|e| format!("Failed to create IQ3_XXS pipeline: {}", e))?;

        let mv_iq3_xxs_split1_kernel = mv_iq_tq_library.get_function("kernel_mul_mv_iq3_xxs_f32_split1", None)
            .map_err(|e| format!("Failed to get IQ3_XXS split1 kernel: {}", e))?;
        let mv_iq3_xxs_split1_pipeline = device.new_compute_pipeline_state_with_function(&mv_iq3_xxs_split1_kernel)
            .map_err(|e| format!("Failed to create IQ3_XXS split1 pipeline: {}", e))?;

        let mv_iq3_s_kernel = mv_iq_tq_library.get_function("kernel_mul_mv_iq3_s_f32", None)
            .map_err(|e| format!("Failed to get IQ3_S kernel: {}", e))?;
        let mv_iq3_s_pipeline = device.new_compute_pipeline_state_with_function(&mv_iq3_s_kernel)
            .map_err(|e| format!("Failed to create IQ3_S pipeline: {}", e))?;

        let mv_iq3_s_split1_kernel = mv_iq_tq_library.get_function("kernel_mul_mv_iq3_s_f32_split1", None)
            .map_err(|e| format!("Failed to get IQ3_S split1 kernel: {}", e))?;
        let mv_iq3_s_split1_pipeline = device.new_compute_pipeline_state_with_function(&mv_iq3_s_split1_kernel)
            .map_err(|e| format!("Failed to create IQ3_S split1 pipeline: {}", e))?;

        // Create GEMM pipelines
        let gemm_q4_k_kernel = gemm_q4_k_library.get_function("kernel_gemm_q4_k_f32", None)
            .map_err(|e| format!("Failed to get GEMM Q4_K kernel: {}", e))?;
        let gemm_q4_k_pipeline = device.new_compute_pipeline_state_with_function(&gemm_q4_k_kernel)
            .map_err(|e| format!("Failed to create GEMM Q4_K pipeline: {}", e))?;

        let gemm_q2_k_kernel = gemm_q2_k_library.get_function("kernel_gemm_q2_k_f32", None)
            .map_err(|e| format!("Failed to get GEMM Q2_K kernel: {}", e))?;
        let gemm_q2_k_pipeline = device.new_compute_pipeline_state_with_function(&gemm_q2_k_kernel)
            .map_err(|e| format!("Failed to create GEMM Q2_K pipeline: {}", e))?;

        let gemm_q3_k_kernel = gemm_q3_k_library.get_function("kernel_gemm_q3_k_f32", None)
            .map_err(|e| format!("Failed to get GEMM Q3_K kernel: {}", e))?;
        let gemm_q3_k_pipeline = device.new_compute_pipeline_state_with_function(&gemm_q3_k_kernel)
            .map_err(|e| format!("Failed to create GEMM Q3_K pipeline: {}", e))?;

        let gemm_q5_k_kernel = gemm_q5_k_library.get_function("kernel_gemm_q5_k_f32", None)
            .map_err(|e| format!("Failed to get GEMM Q5_K kernel: {}", e))?;
        let gemm_q5_k_pipeline = device.new_compute_pipeline_state_with_function(&gemm_q5_k_kernel)
            .map_err(|e| format!("Failed to create GEMM Q5_K pipeline: {}", e))?;

        let gemm_q6_k_kernel = gemm_q6_k_library.get_function("kernel_gemm_q6_k_f32", None)
            .map_err(|e| format!("Failed to get GEMM Q6_K kernel: {}", e))?;
        let gemm_q6_k_pipeline = device.new_compute_pipeline_state_with_function(&gemm_q6_k_kernel)
            .map_err(|e| format!("Failed to create GEMM Q6_K pipeline: {}", e))?;

        let gemm_iq4_nl_kernel = gemm_iq_tq_library.get_function("kernel_gemm_iq4_nl_f32", None)
            .map_err(|e| format!("Failed to get GEMM IQ4_NL kernel: {}", e))?;
        let gemm_iq4_nl_pipeline = device.new_compute_pipeline_state_with_function(&gemm_iq4_nl_kernel)
            .map_err(|e| format!("Failed to create GEMM IQ4_NL pipeline: {}", e))?;

        let gemm_iq4_xs_kernel = gemm_iq_tq_library.get_function("kernel_gemm_iq4_xs_f32", None)
            .map_err(|e| format!("Failed to get GEMM IQ4_XS kernel: {}", e))?;
        let gemm_iq4_xs_pipeline = device.new_compute_pipeline_state_with_function(&gemm_iq4_xs_kernel)
            .map_err(|e| format!("Failed to create GEMM IQ4_XS pipeline: {}", e))?;

        let gemm_iq1_s_kernel = gemm_iq_tq_library.get_function("kernel_gemm_iq1_s_f32", None)
            .map_err(|e| format!("Failed to get GEMM IQ1_S kernel: {}", e))?;
        let gemm_iq1_s_pipeline = device.new_compute_pipeline_state_with_function(&gemm_iq1_s_kernel)
            .map_err(|e| format!("Failed to create GEMM IQ1_S pipeline: {}", e))?;

        let gemm_iq1_m_kernel = gemm_iq_tq_library.get_function("kernel_gemm_iq1_m_f32", None)
            .map_err(|e| format!("Failed to get GEMM IQ1_M kernel: {}", e))?;
        let gemm_iq1_m_pipeline = device.new_compute_pipeline_state_with_function(&gemm_iq1_m_kernel)
            .map_err(|e| format!("Failed to create GEMM IQ1_M pipeline: {}", e))?;

        let gemm_iq2_xxs_kernel = gemm_iq2_iq3_library.get_function("kernel_gemm_iq2_xxs_f32", None)
            .map_err(|e| format!("Failed to get GEMM IQ2_XXS kernel: {}", e))?;
        let gemm_iq2_xxs_pipeline = device.new_compute_pipeline_state_with_function(&gemm_iq2_xxs_kernel)
            .map_err(|e| format!("Failed to create GEMM IQ2_XXS pipeline: {}", e))?;

        let gemm_iq2_xs_kernel = gemm_iq2_iq3_library.get_function("kernel_gemm_iq2_xs_f32", None)
            .map_err(|e| format!("Failed to get GEMM IQ2_XS kernel: {}", e))?;
        let gemm_iq2_xs_pipeline = device.new_compute_pipeline_state_with_function(&gemm_iq2_xs_kernel)
            .map_err(|e| format!("Failed to create GEMM IQ2_XS pipeline: {}", e))?;

        let gemm_iq2_s_kernel = gemm_iq2_iq3_library.get_function("kernel_gemm_iq2_s_f32", None)
            .map_err(|e| format!("Failed to get GEMM IQ2_S kernel: {}", e))?;
        let gemm_iq2_s_pipeline = device.new_compute_pipeline_state_with_function(&gemm_iq2_s_kernel)
            .map_err(|e| format!("Failed to create GEMM IQ2_S pipeline: {}", e))?;

        let gemm_iq3_xxs_kernel = gemm_iq2_iq3_library.get_function("kernel_gemm_iq3_xxs_f32", None)
            .map_err(|e| format!("Failed to get GEMM IQ3_XXS kernel: {}", e))?;
        let gemm_iq3_xxs_pipeline = device.new_compute_pipeline_state_with_function(&gemm_iq3_xxs_kernel)
            .map_err(|e| format!("Failed to create GEMM IQ3_XXS pipeline: {}", e))?;

        let gemm_iq3_s_kernel = gemm_iq2_iq3_library.get_function("kernel_gemm_iq3_s_f32", None)
            .map_err(|e| format!("Failed to get GEMM IQ3_S kernel: {}", e))?;
        let gemm_iq3_s_pipeline = device.new_compute_pipeline_state_with_function(&gemm_iq3_s_kernel)
            .map_err(|e| format!("Failed to create GEMM IQ3_S pipeline: {}", e))?;

        let gemm_tq2_0_kernel = gemm_iq_tq_library.get_function("kernel_gemm_tq2_0_f32", None)
            .map_err(|e| format!("Failed to get GEMM TQ2_0 kernel: {}", e))?;
        let gemm_tq2_0_pipeline = device.new_compute_pipeline_state_with_function(&gemm_tq2_0_kernel)
            .map_err(|e| format!("Failed to create GEMM TQ2_0 pipeline: {}", e))?;

        let gemm_tq1_0_kernel = gemm_iq2_iq3_library.get_function("kernel_gemm_tq1_0_f32", None)
            .map_err(|e| format!("Failed to get GEMM TQ1_0 kernel: {}", e))?;
        let gemm_tq1_0_pipeline = device.new_compute_pipeline_state_with_function(&gemm_tq1_0_kernel)
            .map_err(|e| format!("Failed to create GEMM TQ1_0 pipeline: {}", e))?;

        // Create templated GEMM pipelines (from llama.cpp)
        // Requires function constant specialization
        let gemm_template_q4_k_kernel = {
            let constant_values = metal::FunctionConstantValues::new();
            let false_val: bool = false;
            let one_val: i16 = 1;
            unsafe {
                constant_values.set_constant_value_at_index(
                    &false_val as *const _ as *const std::ffi::c_void, metal::MTLDataType::Bool, 0);
                constant_values.set_constant_value_at_index(
                    &false_val as *const _ as *const std::ffi::c_void, metal::MTLDataType::Bool, 1);
                constant_values.set_constant_value_at_index(
                    &one_val as *const _ as *const std::ffi::c_void, metal::MTLDataType::Short, 2);
                constant_values.set_constant_value_at_index(
                    &one_val as *const _ as *const std::ffi::c_void, metal::MTLDataType::Short, 3);
                constant_values.set_constant_value_at_index(
                    &one_val as *const _ as *const std::ffi::c_void, metal::MTLDataType::Short, 4);
                constant_values.set_constant_value_at_index(
                    &one_val as *const _ as *const std::ffi::c_void, metal::MTLDataType::Short, 5);
            }
            let descriptor = metal::FunctionDescriptor::new();
            descriptor.set_name("kernel_mul_mm_q4_K_f32");
            descriptor.set_constant_values(&constant_values);
            gemm_template_library.new_function_with_descriptor(&descriptor)
                .map_err(|e| format!("Failed to get templated Q4_K kernel: {}", e))?
        };
        let gemm_template_q4_k_pipeline = device.new_compute_pipeline_state_with_function(&gemm_template_q4_k_kernel)
            .map_err(|e| format!("Failed to create templated Q4_K pipeline: {}", e))?;

        // Create remaining template kernels with same constants
        let gemm_template_q2_k_kernel = {
            let constant_values = metal::FunctionConstantValues::new();
            let false_val: bool = false;
            let one_val: i16 = 1;
            unsafe {
                constant_values.set_constant_value_at_index(
                    &false_val as *const _ as *const std::ffi::c_void, metal::MTLDataType::Bool, 0);
                constant_values.set_constant_value_at_index(
                    &false_val as *const _ as *const std::ffi::c_void, metal::MTLDataType::Bool, 1);
                constant_values.set_constant_value_at_index(
                    &one_val as *const _ as *const std::ffi::c_void, metal::MTLDataType::Short, 2);
                constant_values.set_constant_value_at_index(
                    &one_val as *const _ as *const std::ffi::c_void, metal::MTLDataType::Short, 3);
                constant_values.set_constant_value_at_index(
                    &one_val as *const _ as *const std::ffi::c_void, metal::MTLDataType::Short, 4);
                constant_values.set_constant_value_at_index(
                    &one_val as *const _ as *const std::ffi::c_void, metal::MTLDataType::Short, 5);
            }
            let descriptor = metal::FunctionDescriptor::new();
            descriptor.set_name("kernel_mul_mm_q2_K_f32");
            descriptor.set_constant_values(&constant_values);
            gemm_template_library.new_function_with_descriptor(&descriptor)
                .map_err(|e| format!("Failed to get templated Q2_K kernel: {}", e))?
        };
        let gemm_template_q2_k_pipeline = device.new_compute_pipeline_state_with_function(&gemm_template_q2_k_kernel)
            .map_err(|e| format!("Failed to create templated Q2_K pipeline: {}", e))?;

        let gemm_template_q3_k_kernel = {
            let constant_values = metal::FunctionConstantValues::new();
            let false_val: bool = false;
            let one_val: i16 = 1;
            unsafe {
                constant_values.set_constant_value_at_index(
                    &false_val as *const _ as *const std::ffi::c_void, metal::MTLDataType::Bool, 0);
                constant_values.set_constant_value_at_index(
                    &false_val as *const _ as *const std::ffi::c_void, metal::MTLDataType::Bool, 1);
                constant_values.set_constant_value_at_index(
                    &one_val as *const _ as *const std::ffi::c_void, metal::MTLDataType::Short, 2);
                constant_values.set_constant_value_at_index(
                    &one_val as *const _ as *const std::ffi::c_void, metal::MTLDataType::Short, 3);
                constant_values.set_constant_value_at_index(
                    &one_val as *const _ as *const std::ffi::c_void, metal::MTLDataType::Short, 4);
                constant_values.set_constant_value_at_index(
                    &one_val as *const _ as *const std::ffi::c_void, metal::MTLDataType::Short, 5);
            }
            let descriptor = metal::FunctionDescriptor::new();
            descriptor.set_name("kernel_mul_mm_q3_K_f32");
            descriptor.set_constant_values(&constant_values);
            gemm_template_library.new_function_with_descriptor(&descriptor)
                .map_err(|e| format!("Failed to get templated Q3_K kernel: {}", e))?
        };
        let gemm_template_q3_k_pipeline = device.new_compute_pipeline_state_with_function(&gemm_template_q3_k_kernel)
            .map_err(|e| format!("Failed to create templated Q3_K pipeline: {}", e))?;

        let gemm_template_q5_k_kernel = {
            let constant_values = metal::FunctionConstantValues::new();
            let false_val: bool = false;
            let one_val: i16 = 1;
            unsafe {
                constant_values.set_constant_value_at_index(
                    &false_val as *const _ as *const std::ffi::c_void, metal::MTLDataType::Bool, 0);
                constant_values.set_constant_value_at_index(
                    &false_val as *const _ as *const std::ffi::c_void, metal::MTLDataType::Bool, 1);
                constant_values.set_constant_value_at_index(
                    &one_val as *const _ as *const std::ffi::c_void, metal::MTLDataType::Short, 2);
                constant_values.set_constant_value_at_index(
                    &one_val as *const _ as *const std::ffi::c_void, metal::MTLDataType::Short, 3);
                constant_values.set_constant_value_at_index(
                    &one_val as *const _ as *const std::ffi::c_void, metal::MTLDataType::Short, 4);
                constant_values.set_constant_value_at_index(
                    &one_val as *const _ as *const std::ffi::c_void, metal::MTLDataType::Short, 5);
            }
            let descriptor = metal::FunctionDescriptor::new();
            descriptor.set_name("kernel_mul_mm_q5_K_f32");
            descriptor.set_constant_values(&constant_values);
            gemm_template_library.new_function_with_descriptor(&descriptor)
                .map_err(|e| format!("Failed to get templated Q5_K kernel: {}", e))?
        };
        let gemm_template_q5_k_pipeline = device.new_compute_pipeline_state_with_function(&gemm_template_q5_k_kernel)
            .map_err(|e| format!("Failed to create templated Q5_K pipeline: {}", e))?;

        let gemm_template_q6_k_kernel = {
            let constant_values = metal::FunctionConstantValues::new();
            let false_val: bool = false;
            let one_val: i16 = 1;
            unsafe {
                constant_values.set_constant_value_at_index(
                    &false_val as *const _ as *const std::ffi::c_void, metal::MTLDataType::Bool, 0);
                constant_values.set_constant_value_at_index(
                    &false_val as *const _ as *const std::ffi::c_void, metal::MTLDataType::Bool, 1);
                constant_values.set_constant_value_at_index(
                    &one_val as *const _ as *const std::ffi::c_void, metal::MTLDataType::Short, 2);
                constant_values.set_constant_value_at_index(
                    &one_val as *const _ as *const std::ffi::c_void, metal::MTLDataType::Short, 3);
                constant_values.set_constant_value_at_index(
                    &one_val as *const _ as *const std::ffi::c_void, metal::MTLDataType::Short, 4);
                constant_values.set_constant_value_at_index(
                    &one_val as *const _ as *const std::ffi::c_void, metal::MTLDataType::Short, 5);
            }
            let descriptor = metal::FunctionDescriptor::new();
            descriptor.set_name("kernel_mul_mm_q6_K_f32");
            descriptor.set_constant_values(&constant_values);
            gemm_template_library.new_function_with_descriptor(&descriptor)
                .map_err(|e| format!("Failed to get templated Q6_K kernel: {}", e))?
        };
        let gemm_template_q6_k_pipeline = device.new_compute_pipeline_state_with_function(&gemm_template_q6_k_kernel)
            .map_err(|e| format!("Failed to create templated Q6_K pipeline: {}", e))?;

        Ok(Self {
            device,
            queue,
            vec_dot_library,
            mv_library,
            mv_simd_library,
            mv_q4_0_library,
            mv_q4_k_library,
            mv_q5_k_library,
            mv_q2_k_library,
            mv_q6_k_library,
            mv_q3_k_library,
            mv_iq_tq_library,
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

    pub fn vec_dot_iq4_nl_q8_0(&self, n: usize, x: &[BlockIQ4NL], y: &[BlockQ8_0]) -> Result<f32, String> {
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

        let kernel = self.vec_dot_library.get_function("vec_dot_iq4_nl_q8_0", None)
            .map_err(|e| format!("Failed to get kernel: {}", e))?;
        let pipeline = self.device.new_compute_pipeline_state_with_function(&kernel)
            .map_err(|e| format!("Failed to create pipeline: {}", e))?;

        let command_buffer = self.queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();

        encoder.set_compute_pipeline_state(&pipeline);
        encoder.set_buffer(0, Some(&x_buffer), 0);
        encoder.set_buffer(1, Some(&y_buffer), 0);
        encoder.set_buffer(2, Some(&result_buffer), 0);
        encoder.set_bytes(3, std::mem::size_of::<u32>() as u64, &(nb as u32) as *const u32 as *const std::ffi::c_void);

        let thread_group_count = MTLSize { width: 1, height: 1, depth: 1 };
        let thread_group_size = MTLSize { width: 1, height: 1, depth: 1 };

        encoder.dispatch_thread_groups(thread_group_count, thread_group_size);
        encoder.end_encoding();

        command_buffer.commit();
        command_buffer.wait_until_completed();

        let result_ptr = result_buffer.contents() as *const f32;
        Ok(unsafe { *result_ptr })
    }

    pub fn mv_iq4_nl_q8_0(&self, m: usize, k: usize, weights: &[BlockIQ4NL], input: &[BlockQ8_0]) -> Result<Vec<f32>, String> {
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

        let kernel = self.mv_library.get_function("mv_iq4_nl_q8_0", None)
            .map_err(|e| format!("Failed to get kernel: {}", e))?;
        let pipeline = self.device.new_compute_pipeline_state_with_function(&kernel)
            .map_err(|e| format!("Failed to create pipeline: {}", e))?;

        let pipeline_time = t1.elapsed();

        let t2 = std::time::Instant::now();

        let command_buffer = self.queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();

        encoder.set_compute_pipeline_state(&pipeline);
        encoder.set_buffer(0, Some(&weights_buffer), 0);
        encoder.set_buffer(1, Some(&input_buffer), 0);
        encoder.set_buffer(2, Some(&output_buffer), 0);
        encoder.set_bytes(3, std::mem::size_of::<u32>() as u64, &(m as u32) as *const u32 as *const std::ffi::c_void);
        encoder.set_bytes(4, std::mem::size_of::<u32>() as u64, &(nb as u32) as *const u32 as *const std::ffi::c_void);

        // Thread group configuration matching llama.cpp
        // NSG = 2 SIMD groups per threadgroup, each SIMD group has 32 threads
        // Each SIMD group processes NR0 = 4 rows
        // Total: 2 * 32 = 64 threads, processing 2 * 4 = 8 rows per threadgroup
        const NSG: u64 = 2;
        const NR0: u64 = 4;
        const ROWS_PER_THREADGROUP: u64 = NSG * NR0; // 8
        const THREADS_PER_THREADGROUP: u64 = NSG * 32; // 64

        let thread_group_size = MTLSize {
            width: THREADS_PER_THREADGROUP,
            height: 1,
            depth: 1,
        };
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
        eprintln!("MV timing: buffer={:?} pipeline={:?} encode={:?} gpu={:?}",
                  buffer_time, pipeline_time, encode_time, gpu_time);

        let output_ptr = output_buffer.contents() as *const f32;
        Ok(unsafe { std::slice::from_raw_parts(output_ptr, m).to_vec() })
    }

    /// SIMD-optimized MV kernel (32 threads per row)
    pub fn mv_iq4_nl_q8_0_simd(&self, m: usize, k: usize, weights: &[BlockIQ4NL], input: &[BlockQ8_0]) -> Result<Vec<f32>, String> {
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

        let kernel = self.mv_simd_library.get_function("mv_iq4_nl_q8_0_simd", None)
            .map_err(|e| format!("Failed to get kernel: {}", e))?;
        let pipeline = self.device.new_compute_pipeline_state_with_function(&kernel)
            .map_err(|e| format!("Failed to create pipeline: {}", e))?;

        let command_buffer = self.queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();

        encoder.set_compute_pipeline_state(&pipeline);
        encoder.set_buffer(0, Some(&weights_buffer), 0);
        encoder.set_buffer(1, Some(&input_buffer), 0);
        encoder.set_buffer(2, Some(&output_buffer), 0);
        encoder.set_bytes(3, std::mem::size_of::<u32>() as u64, &(m as u32) as *const u32 as *const std::ffi::c_void);
        encoder.set_bytes(4, std::mem::size_of::<u32>() as u64, &(nb as u32) as *const u32 as *const std::ffi::c_void);

        // One threadgroup per row (32 threads per threadgroup)
        let thread_group_size = MTLSize { width: 32, height: 1, depth: 1 };
        let thread_group_count = MTLSize { width: m as u64, height: 1, depth: 1 };

        encoder.dispatch_thread_groups(thread_group_count, thread_group_size);
        encoder.end_encoding();

        command_buffer.commit();
        command_buffer.wait_until_completed();

        let output_ptr = output_buffer.contents() as *const f32;
        Ok(unsafe { std::slice::from_raw_parts(output_ptr, m).to_vec() })
    }

    /// Q4_0 x F32 Matrix-Vector (matching llama.cpp kernel)
    pub fn mv_q4_0_f32(&self, m: usize, k: usize, weights: &[BlockQ4_0], input: &[f32]) -> Result<Vec<f32>, String> {
        let nb = k / 32;
        let nb01 = (nb * std::mem::size_of::<BlockQ4_0>()) as u64; // Byte stride per row
        const NR0: u64 = 4;

        // Args structure matching Metal kernel
        #[repr(C)]
        struct MvArgs {
            ne00: u32,  // K
            ne01: u32,  // M
            nb01: u64,  // Byte stride per row
        }

        let args = MvArgs {
            ne00: k as u32,
            ne01: m as u32,
            nb01,
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
        encoder.set_buffer(0, Some(&weights_buffer), 0);
        encoder.set_buffer(1, Some(&input_buffer), 0);
        encoder.set_buffer(2, Some(&output_buffer), 0);
        encoder.set_bytes(3, std::mem::size_of::<MvArgs>() as u64, &args as *const MvArgs as *const std::ffi::c_void);

        // Thread group configuration matching llama.cpp EXACTLY
        // 2D threadgroup: (32, 2, 1) where Y dimension = NSG = 2
        // Metal organizes threads linearly: Y=0 -> sgitg=0, Y=1 -> sgitg=1
        const NSG: u64 = 2;
        const ROWS_PER_THREADGROUP: u64 = NSG * NR0; // 8

        let thread_group_size = MTLSize {
            width: 32,
            height: NSG,  // 2D: Y = 2 for 2 SIMD groups
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

    /// Q4_K x F32 Matrix-Vector (matching llama.cpp kernel)
    pub fn mv_q4_k_f32(&self, m: usize, k: usize, weights: &[BlockQ4K], input: &[f32]) -> Result<Vec<f32>, String> {
        let nb = k / 256;  // QK_K = 256
        let nb01 = (nb * std::mem::size_of::<BlockQ4K>()) as u64; // Byte stride per row
        const NR0: u64 = 2;  // N_R0_Q4_K = 2

        // Args structure matching Metal kernel
        #[repr(C)]
        struct MvArgs {
            ne00: u32,  // K
            ne01: u32,  // M
            nb01: u64,  // Byte stride per row
        }

        let args = MvArgs {
            ne00: k as u32,
            ne01: m as u32,
            nb01,
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
        encoder.set_buffer(0, Some(&weights_buffer), 0);
        encoder.set_buffer(1, Some(&input_buffer), 0);
        encoder.set_buffer(2, Some(&output_buffer), 0);
        encoder.set_bytes(3, std::mem::size_of::<MvArgs>() as u64, &args as *const MvArgs as *const std::ffi::c_void);

        // Thread group configuration matching llama.cpp EXACTLY
        // 2D threadgroup: (32, 2, 1) where Y dimension = NSG = 2
        // Metal organizes threads linearly: Y=0 -> sgitg=0, Y=1 -> sgitg=1
        const NSG: u64 = 2;
        const ROWS_PER_THREADGROUP: u64 = NSG * NR0; // 4

        let thread_group_size = MTLSize {
            width: 32,
            height: NSG,  // 2D: Y = 2 for 2 SIMD groups
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

    /// Q5_K x F32 Matrix-Vector (matching llama.cpp kernel)
    pub fn mv_q5_k_f32(&self, m: usize, k: usize, weights: &[BlockQ5K], input: &[f32]) -> Result<Vec<f32>, String> {
        let nb = k / 256;  // QK_K = 256
        let nb01 = (nb * std::mem::size_of::<BlockQ5K>()) as u64; // Byte stride per row
        const NR0: u64 = 2;  // N_R0_Q5_K = 2

        // Args structure matching Metal kernel
        #[repr(C)]
        struct MvArgs {
            ne00: u32,  // K
            ne01: u32,  // M
            nb01: u64,  // Byte stride per row
        }

        let args = MvArgs {
            ne00: k as u32,
            ne01: m as u32,
            nb01,
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
        encoder.set_buffer(0, Some(&weights_buffer), 0);
        encoder.set_buffer(1, Some(&input_buffer), 0);
        encoder.set_buffer(2, Some(&output_buffer), 0);
        encoder.set_bytes(3, std::mem::size_of::<MvArgs>() as u64, &args as *const MvArgs as *const std::ffi::c_void);

        // Thread group configuration matching llama.cpp EXACTLY
        // 2D threadgroup: (32, 2, 1) where Y dimension = NSG = 2
        // Metal organizes threads linearly: Y=0 -> sgitg=0, Y=1 -> sgitg=1
        const NSG: u64 = 2;
        const ROWS_PER_THREADGROUP: u64 = NSG * NR0; // 4

        let thread_group_size = MTLSize {
            width: 32,
            height: NSG,  // 2D: Y = 2 for 2 SIMD groups
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

    /// Q2_K x F32 Matrix-Vector (matching llama.cpp kernel)
    pub fn mv_q2_k_f32(&self, m: usize, k: usize, weights: &[BlockQ2K], input: &[f32]) -> Result<Vec<f32>, String> {
        let nb = k / 256;  // QK_K = 256
        let nb01 = (nb * std::mem::size_of::<BlockQ2K>()) as u64; // Byte stride per row
        const NR0: u64 = 2;  // N_R0_Q2_K = 2

        // Args structure matching Metal kernel
        #[repr(C)]
        struct MvArgs {
            ne00: u32,  // K
            ne01: u32,  // M
            nb01: u64,  // Byte stride per row
        }

        let args = MvArgs {
            ne00: k as u32,
            ne01: m as u32,
            nb01,
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
        encoder.set_buffer(0, Some(&weights_buffer), 0);
        encoder.set_buffer(1, Some(&input_buffer), 0);
        encoder.set_buffer(2, Some(&output_buffer), 0);
        encoder.set_bytes(3, std::mem::size_of::<MvArgs>() as u64, &args as *const MvArgs as *const std::ffi::c_void);

        // Thread group configuration matching llama.cpp EXACTLY
        // 2D threadgroup: (32, 2, 1) where Y dimension = NSG = 2
        // Metal organizes threads linearly: Y=0 -> sgitg=0, Y=1 -> sgitg=1
        const NSG: u64 = 2;
        const ROWS_PER_THREADGROUP: u64 = NSG * NR0; // 4

        let thread_group_size = MTLSize {
            width: 32,
            height: NSG,  // 2D: Y = 2 for 2 SIMD groups
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
    pub fn mv_q6_k_f32(&self, m: usize, k: usize, weights: &[BlockQ6K], input: &[f32]) -> Result<Vec<f32>, String> {
        let nb = k / 256;  // QK_K = 256
        let nb01 = (nb * std::mem::size_of::<BlockQ6K>()) as u64; // Byte stride per row
        const NR0: u64 = 2;  // N_R0_Q6_K = 2

        #[repr(C)]
        struct MvArgs {
            ne00: u32,
            ne01: u32,
            nb01: u64,
        }

        let args = MvArgs {
            ne00: k as u32,
            ne01: m as u32,
            nb01,
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
        encoder.set_buffer(0, Some(&weights_buffer), 0);
        encoder.set_buffer(1, Some(&input_buffer), 0);
        encoder.set_buffer(2, Some(&output_buffer), 0);
        encoder.set_bytes(3, std::mem::size_of::<MvArgs>() as u64, &args as *const MvArgs as *const std::ffi::c_void);

        const NSG: u64 = 2;
        const ROWS_PER_THREADGROUP: u64 = NSG * NR0;

        let thread_group_size = MTLSize {
            width: 32,
            height: NSG,
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

    /// Q3_K x F32 Matrix-Vector (matching llama.cpp kernel)
    pub fn mv_q3_k_f32(&self, m: usize, k: usize, weights: &[BlockQ3K], input: &[f32]) -> Result<Vec<f32>, String> {
        let nb = k / 256;
        let nb01 = (nb * std::mem::size_of::<BlockQ3K>()) as u64;
        const NR0: u64 = 2;

        #[repr(C)]
        struct MvArgs {
            ne00: u32,
            ne01: u32,
            nb01: u64,
        }

        let args = MvArgs {
            ne00: k as u32,
            ne01: m as u32,
            nb01,
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
        encoder.set_buffer(0, Some(&weights_buffer), 0);
        encoder.set_buffer(1, Some(&input_buffer), 0);
        encoder.set_buffer(2, Some(&output_buffer), 0);
        encoder.set_bytes(3, std::mem::size_of::<MvArgs>() as u64, &args as *const MvArgs as *const std::ffi::c_void);

        const NSG: u64 = 2;
        const ROWS_PER_THREADGROUP: u64 = NSG * NR0;

        let thread_group_size = MTLSize {
            width: 32,
            height: NSG,
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

    /// IQ4_NL x F32 Matrix-Vector (matching llama.cpp kernel)
    pub fn mv_iq4_nl_f32(&self, m: usize, k: usize, weights: &[BlockIQ4NL], input: &[f32]) -> Result<Vec<f32>, String> {
        let nb = k / 32;  // QK4_NL = 32
        let nb01 = (nb * std::mem::size_of::<BlockIQ4NL>()) as u64;
        const NR0: u64 = 2;  // Keep at 2 (optimal)

        #[repr(C)]
        struct MvArgs {
            ne00: u32,
            ne01: u32,
            nb01: u64,
        }

        let args = MvArgs {
            ne00: k as u32,
            ne01: m as u32,
            nb01,
        };

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
        encoder.set_bytes(3, std::mem::size_of::<MvArgs>() as u64, &args as *const MvArgs as *const std::ffi::c_void);

        // Set threadgroup memory: 32 floats for kvalues lookup table
        // Round up to multiple of 16 as required by Metal
        const THREADGROUP_MEM_SIZE: u64 = 32 * std::mem::size_of::<f32>() as u64;  // 128 bytes
        encoder.set_threadgroup_memory_length(THREADGROUP_MEM_SIZE, 0);

        const NSG: u64 = 2;
        const ROWS_PER_THREADGROUP: u64 = NSG * NR0;

        let thread_group_size = MTLSize {
            width: 32,
            height: NSG,
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

    /// TQ2_0 x F32 Matrix-Vector (matching llama.cpp kernel)
    pub fn mv_tq2_0_f32(&self, m: usize, k: usize, weights: &[BlockTQ2_0], input: &[f32]) -> Result<Vec<f32>, String> {
        let nb = k / 256;  // QK_K = 256
        let nb01 = (nb * std::mem::size_of::<BlockTQ2_0>()) as u64; // Byte stride per row
        const NR0: u64 = 2;  // N_R0_TQ2_0 = 2

        #[repr(C)]
        struct MvArgs {
            ne00: u32,
            ne01: u32,
            nb01: u64,
        }

        let args = MvArgs {
            ne00: k as u32,
            ne01: m as u32,
            nb01,
        };

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
        encoder.set_bytes(3, std::mem::size_of::<MvArgs>() as u64, &args as *const MvArgs as *const std::ffi::c_void);

        const NSG: u64 = 2;
        const ROWS_PER_THREADGROUP: u64 = NSG * NR0;

        let thread_group_size = MTLSize {
            width: 32,
            height: NSG,
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

    /// IQ4_XS x F32 Matrix-Vector (matching llama.cpp kernel)
    pub fn mv_iq4_xs_f32(&self, m: usize, k: usize, weights: &[BlockIQ4XS], input: &[f32]) -> Result<Vec<f32>, String> {
        let nb = k / 256;  // QK_K = 256
        let nb01 = (nb * std::mem::size_of::<BlockIQ4XS>()) as u64; // Byte stride per row
        const NR0: u64 = 2;  // N_R0_IQ4_XS = 2

        #[repr(C)]
        struct MvArgs {
            ne00: u32,
            ne01: u32,
            nb01: u64,
        }

        let args = MvArgs {
            ne00: k as u32,
            ne01: m as u32,
            nb01,
        };

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
        encoder.set_bytes(3, std::mem::size_of::<MvArgs>() as u64, &args as *const MvArgs as *const std::ffi::c_void);

        const NSG: u64 = 2;
        const ROWS_PER_THREADGROUP: u64 = NSG * NR0;

        let thread_group_size = MTLSize {
            width: 32,
            height: NSG,
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

    /// TQ1_0 x F32 Matrix-Vector
    pub fn mv_tq1_0_f32(&self, m: usize, k: usize, weights: &[BlockTQ1_0], input: &[f32]) -> Result<Vec<f32>, String> {
        let nb = k / 256;
        let nb01 = (nb * std::mem::size_of::<BlockTQ1_0>()) as u64;
        const NR0: u64 = 2;

        #[repr(C)]
        struct MvArgs {
            ne00: u32,
            ne01: u32,
            nb01: u64,
        }

        let args = MvArgs {
            ne00: k as u32,
            ne01: m as u32,
            nb01,
        };

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
        encoder.set_bytes(3, std::mem::size_of::<MvArgs>() as u64, &args as *const MvArgs as *const std::ffi::c_void);

        const NSG: u64 = 2;
        const ROWS_PER_THREADGROUP: u64 = NSG * NR0;

        let thread_group_size = MTLSize {
            width: 32,
            height: NSG,
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

    /// IQ1_S x F32 Matrix-Vector
    pub fn mv_iq1_s_f32(&self, m: usize, k: usize, weights: &[BlockIQ1S], input: &[f32]) -> Result<Vec<f32>, String> {
        let nb = k / 256;
        let nb01 = (nb * std::mem::size_of::<BlockIQ1S>()) as u64;
        const NR0: u64 = 4;  // N_R0_IQ1_S = 4

        #[repr(C)]
        struct MvArgs { ne00: u32, ne01: u32, nb01: u64 }

        let args = MvArgs { ne00: k as u32, ne01: m as u32, nb01 };
        let weights_size = weights.len() * std::mem::size_of::<BlockIQ1S>();
        let input_size = input.len() * std::mem::size_of::<f32>();
        let output_size = m * std::mem::size_of::<f32>();

        let weights_buffer = {
            let mut buf_cell = self.mv_weights_buffer.borrow_mut();
            let mut size_cell = self.mv_weights_size.borrow_mut();
            if *size_cell != weights_size {
                let buf = self.device.new_buffer_with_data(weights.as_ptr() as *const std::ffi::c_void, weights_size as u64, metal::MTLResourceOptions::StorageModeShared);
                *buf_cell = Some(buf.clone()); *size_cell = weights_size; buf
            } else { buf_cell.as_ref().unwrap().clone() }
        };

        let input_buffer = {
            let mut buf_cell = self.mv_input_buffer.borrow_mut();
            let mut size_cell = self.mv_input_size.borrow_mut();
            if *size_cell != input_size {
                let buf = self.device.new_buffer_with_data(input.as_ptr() as *const std::ffi::c_void, input_size as u64, metal::MTLResourceOptions::StorageModeShared);
                *buf_cell = Some(buf.clone()); *size_cell = input_size; buf
            } else { buf_cell.as_ref().unwrap().clone() }
        };

        let output_buffer = {
            let mut buf_cell = self.mv_output_buffer.borrow_mut();
            let mut size_cell = self.mv_output_size.borrow_mut();
            if *size_cell != output_size {
                let buf = self.device.new_buffer(output_size as u64, metal::MTLResourceOptions::StorageModeShared);
                *buf_cell = Some(buf.clone()); *size_cell = output_size; buf
            } else { buf_cell.as_ref().unwrap().clone() }
        };

        let command_buffer = self.queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(&self.mv_iq1_s_pipeline);
        encoder.set_buffer(0, Some(&weights_buffer), 0);
        encoder.set_buffer(1, Some(&input_buffer), 0);
        encoder.set_buffer(2, Some(&output_buffer), 0);
        encoder.set_bytes(3, std::mem::size_of::<MvArgs>() as u64, &args as *const MvArgs as *const std::ffi::c_void);

        const NSG: u64 = 2;
        const ROWS_PER_THREADGROUP: u64 = NSG * NR0;
        let thread_group_size = MTLSize { width: 32, height: NSG, depth: 1 };
        let thread_group_count = MTLSize { width: (m as u64 + ROWS_PER_THREADGROUP - 1) / ROWS_PER_THREADGROUP, height: 1, depth: 1 };

        encoder.dispatch_thread_groups(thread_group_count, thread_group_size);
        encoder.end_encoding();
        command_buffer.commit();
        command_buffer.wait_until_completed();

        let output_ptr = output_buffer.contents() as *const f32;
        Ok(unsafe { std::slice::from_raw_parts(output_ptr, m).to_vec() })
    }

    /// IQ1_M x F32 Matrix-Vector
    pub fn mv_iq1_m_f32(&self, m: usize, k: usize, weights: &[BlockIQ1M], input: &[f32]) -> Result<Vec<f32>, String> {
        let nb = k / 256;
        let nb01 = (nb * std::mem::size_of::<BlockIQ1M>()) as u64;
        const NR0: u64 = 4;  // N_R0_IQ1_M = 4

        #[repr(C)]
        struct MvArgs { ne00: u32, ne01: u32, nb01: u64 }

        let args = MvArgs { ne00: k as u32, ne01: m as u32, nb01 };
        let weights_size = weights.len() * std::mem::size_of::<BlockIQ1M>();
        let input_size = input.len() * std::mem::size_of::<f32>();
        let output_size = m * std::mem::size_of::<f32>();

        let weights_buffer = {
            let mut buf_cell = self.mv_weights_buffer.borrow_mut();
            let mut size_cell = self.mv_weights_size.borrow_mut();
            if *size_cell != weights_size {
                let buf = self.device.new_buffer_with_data(weights.as_ptr() as *const std::ffi::c_void, weights_size as u64, metal::MTLResourceOptions::StorageModeShared);
                *buf_cell = Some(buf.clone()); *size_cell = weights_size; buf
            } else { buf_cell.as_ref().unwrap().clone() }
        };

        let input_buffer = {
            let mut buf_cell = self.mv_input_buffer.borrow_mut();
            let mut size_cell = self.mv_input_size.borrow_mut();
            if *size_cell != input_size {
                let buf = self.device.new_buffer_with_data(input.as_ptr() as *const std::ffi::c_void, input_size as u64, metal::MTLResourceOptions::StorageModeShared);
                *buf_cell = Some(buf.clone()); *size_cell = input_size; buf
            } else { buf_cell.as_ref().unwrap().clone() }
        };

        let output_buffer = {
            let mut buf_cell = self.mv_output_buffer.borrow_mut();
            let mut size_cell = self.mv_output_size.borrow_mut();
            if *size_cell != output_size {
                let buf = self.device.new_buffer(output_size as u64, metal::MTLResourceOptions::StorageModeShared);
                *buf_cell = Some(buf.clone()); *size_cell = output_size; buf
            } else { buf_cell.as_ref().unwrap().clone() }
        };

        let command_buffer = self.queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(&self.mv_iq1_m_pipeline);
        encoder.set_buffer(0, Some(&weights_buffer), 0);
        encoder.set_buffer(1, Some(&input_buffer), 0);
        encoder.set_buffer(2, Some(&output_buffer), 0);
        encoder.set_bytes(3, std::mem::size_of::<MvArgs>() as u64, &args as *const MvArgs as *const std::ffi::c_void);

        const NSG: u64 = 2;
        const ROWS_PER_THREADGROUP: u64 = NSG * NR0;
        let thread_group_size = MTLSize { width: 32, height: NSG, depth: 1 };
        let thread_group_count = MTLSize { width: (m as u64 + ROWS_PER_THREADGROUP - 1) / ROWS_PER_THREADGROUP, height: 1, depth: 1 };

        encoder.dispatch_thread_groups(thread_group_count, thread_group_size);
        encoder.end_encoding();
        command_buffer.commit();
        command_buffer.wait_until_completed();

        let output_ptr = output_buffer.contents() as *const f32;
        Ok(unsafe { std::slice::from_raw_parts(output_ptr, m).to_vec() })
    }

    /// IQ2_XXS x F32 Matrix-Vector with dynamic kernel selection
    pub fn mv_iq2_xxs_f32(&self, m: usize, k: usize, weights: &[BlockIQ2XXS], input: &[f32]) -> Result<Vec<f32>, String> {
        let nb = k / 256;
        let nb01 = (nb * std::mem::size_of::<BlockIQ2XXS>()) as u64;
        let nb32 = nb * (256 / 32);

        // Dynamic dispatch: use split=1 for small nb32
        let use_split = nb32 < 32;
        let nr0 = if use_split { 8u64 } else { 4u64 };

        #[repr(C)]
        struct MvArgs { ne00: u32, ne01: u32, nb01: u64 }

        let args = MvArgs { ne00: k as u32, ne01: m as u32, nb01 };
        let weights_size = weights.len() * std::mem::size_of::<BlockIQ2XXS>();
        let input_size = input.len() * std::mem::size_of::<f32>();
        let output_size = m * std::mem::size_of::<f32>();

        let weights_buffer = {
            let mut buf_cell = self.mv_weights_buffer.borrow_mut();
            let mut size_cell = self.mv_weights_size.borrow_mut();
            if *size_cell != weights_size {
                let buf = self.device.new_buffer_with_data(weights.as_ptr() as *const std::ffi::c_void, weights_size as u64, metal::MTLResourceOptions::StorageModeShared);
                *buf_cell = Some(buf.clone()); *size_cell = weights_size; buf
            } else { buf_cell.as_ref().unwrap().clone() }
        };

        let input_buffer = {
            let mut buf_cell = self.mv_input_buffer.borrow_mut();
            let mut size_cell = self.mv_input_size.borrow_mut();
            if *size_cell != input_size {
                let buf = self.device.new_buffer_with_data(input.as_ptr() as *const std::ffi::c_void, input_size as u64, metal::MTLResourceOptions::StorageModeShared);
                *buf_cell = Some(buf.clone()); *size_cell = input_size; buf
            } else { buf_cell.as_ref().unwrap().clone() }
        };

        let output_buffer = {
            let mut buf_cell = self.mv_output_buffer.borrow_mut();
            let mut size_cell = self.mv_output_size.borrow_mut();
            if *size_cell != output_size {
                let buf = self.device.new_buffer(output_size as u64, metal::MTLResourceOptions::StorageModeShared);
                *buf_cell = Some(buf.clone()); *size_cell = output_size; buf
            } else { buf_cell.as_ref().unwrap().clone() }
        };

        let command_buffer = self.queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();

        // Select appropriate pipeline
        if use_split {
            encoder.set_compute_pipeline_state(&self.mv_iq2_xxs_split1_pipeline);
        } else {
            encoder.set_compute_pipeline_state(&self.mv_iq2_xxs_pipeline);
        }

        encoder.set_buffer(0, Some(&weights_buffer), 0);
        encoder.set_buffer(1, Some(&input_buffer), 0);
        encoder.set_buffer(2, Some(&output_buffer), 0);
        encoder.set_bytes(3, std::mem::size_of::<MvArgs>() as u64, &args as *const MvArgs as *const std::ffi::c_void);

        // Threadgroup memory for IQ2_XXS: svalues (256*8) + ssigns (128) = 2176 bytes
        const THREADGROUP_MEM_SIZE: u64 = 256 * 8 + 128;
        encoder.set_threadgroup_memory_length(THREADGROUP_MEM_SIZE, 0);

        const NSG: u64 = 2;
        let rows_per_threadgroup = NSG * nr0;
        let thread_group_size = MTLSize { width: 32, height: NSG, depth: 1 };
        let thread_group_count = MTLSize { width: (m as u64 + rows_per_threadgroup - 1) / rows_per_threadgroup, height: 1, depth: 1 };

        encoder.dispatch_thread_groups(thread_group_count, thread_group_size);
        encoder.end_encoding();
        command_buffer.commit();
        command_buffer.wait_until_completed();

        let output_ptr = output_buffer.contents() as *const f32;
        Ok(unsafe { std::slice::from_raw_parts(output_ptr, m).to_vec() })
    }

    /// IQ2_XS x F32 Matrix-Vector with dynamic kernel selection
    pub fn mv_iq2_xs_f32(&self, m: usize, k: usize, weights: &[BlockIQ2XS], input: &[f32]) -> Result<Vec<f32>, String> {
        let nb = k / 256;
        let nb01 = (nb * std::mem::size_of::<BlockIQ2XS>()) as u64;
        let nb32 = nb * (256 / 32);

        // Dynamic dispatch: use split=1 for small nb32
        let use_split = nb32 < 32;
        let nr0 = if use_split { 8u64 } else { 4u64 };

        #[repr(C)]
        struct MvArgs { ne00: u32, ne01: u32, nb01: u64 }

        let args = MvArgs { ne00: k as u32, ne01: m as u32, nb01 };
        let weights_size = weights.len() * std::mem::size_of::<BlockIQ2XS>();
        let input_size = input.len() * std::mem::size_of::<f32>();
        let output_size = m * std::mem::size_of::<f32>();

        let weights_buffer = {
            let mut buf_cell = self.mv_weights_buffer.borrow_mut();
            let mut size_cell = self.mv_weights_size.borrow_mut();
            if *size_cell != weights_size {
                let buf = self.device.new_buffer_with_data(weights.as_ptr() as *const std::ffi::c_void, weights_size as u64, metal::MTLResourceOptions::StorageModeShared);
                *buf_cell = Some(buf.clone()); *size_cell = weights_size; buf
            } else { buf_cell.as_ref().unwrap().clone() }
        };

        let input_buffer = {
            let mut buf_cell = self.mv_input_buffer.borrow_mut();
            let mut size_cell = self.mv_input_size.borrow_mut();
            if *size_cell != input_size {
                let buf = self.device.new_buffer_with_data(input.as_ptr() as *const std::ffi::c_void, input_size as u64, metal::MTLResourceOptions::StorageModeShared);
                *buf_cell = Some(buf.clone()); *size_cell = input_size; buf
            } else { buf_cell.as_ref().unwrap().clone() }
        };

        let output_buffer = {
            let mut buf_cell = self.mv_output_buffer.borrow_mut();
            let mut size_cell = self.mv_output_size.borrow_mut();
            if *size_cell != output_size {
                let buf = self.device.new_buffer(output_size as u64, metal::MTLResourceOptions::StorageModeShared);
                *buf_cell = Some(buf.clone()); *size_cell = output_size; buf
            } else { buf_cell.as_ref().unwrap().clone() }
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
        encoder.set_bytes(3, std::mem::size_of::<MvArgs>() as u64, &args as *const MvArgs as *const std::ffi::c_void);

        // Set threadgroup memory size: 512*8 (grid) + 128 (signs) = 4224 bytes
        // Round up to multiple of 16 as required by Metal
        const THREADGROUP_MEM_SIZE: u64 = 4224;
        encoder.set_threadgroup_memory_length(THREADGROUP_MEM_SIZE, 0);

        const NSG: u64 = 2;
        let rows_per_threadgroup = NSG * nr0;
        let thread_group_size = MTLSize { width: 32, height: NSG, depth: 1 };
        let thread_group_count = MTLSize { width: (m as u64 + rows_per_threadgroup - 1) / rows_per_threadgroup, height: 1, depth: 1 };

        encoder.dispatch_thread_groups(thread_group_count, thread_group_size);
        encoder.end_encoding();
        command_buffer.commit();
        command_buffer.wait_until_completed();

        let output_ptr = output_buffer.contents() as *const f32;
        Ok(unsafe { std::slice::from_raw_parts(output_ptr, m).to_vec() })
    }

    /// IQ2_S x F32 Matrix-Vector with dynamic kernel selection
    pub fn mv_iq2_s_f32(&self, m: usize, k: usize, weights: &[BlockIQ2S], input: &[f32]) -> Result<Vec<f32>, String> {
        let nb = k / 256;
        let nb01 = (nb * std::mem::size_of::<BlockIQ2S>()) as u64;
        let nb32 = nb * (256 / 32);  // nb32 = nb * 8

        // Dynamic dispatch: use split=1 for small nb32
        let use_split = nb32 < 32;
        let nr0 = if use_split { 8u64 } else { 4u64 };

        #[repr(C)]
        struct MvArgs { ne00: u32, ne01: u32, nb01: u64 }

        let args = MvArgs { ne00: k as u32, ne01: m as u32, nb01 };
        let weights_size = weights.len() * std::mem::size_of::<BlockIQ2S>();
        let input_size = input.len() * std::mem::size_of::<f32>();
        let output_size = m * std::mem::size_of::<f32>();

        let weights_buffer = {
            let mut buf_cell = self.mv_weights_buffer.borrow_mut();
            let mut size_cell = self.mv_weights_size.borrow_mut();
            if *size_cell != weights_size {
                let buf = self.device.new_buffer_with_data(weights.as_ptr() as *const std::ffi::c_void, weights_size as u64, metal::MTLResourceOptions::StorageModeShared);
                *buf_cell = Some(buf.clone()); *size_cell = weights_size; buf
            } else { buf_cell.as_ref().unwrap().clone() }
        };

        let input_buffer = {
            let mut buf_cell = self.mv_input_buffer.borrow_mut();
            let mut size_cell = self.mv_input_size.borrow_mut();
            if *size_cell != input_size {
                let buf = self.device.new_buffer_with_data(input.as_ptr() as *const std::ffi::c_void, input_size as u64, metal::MTLResourceOptions::StorageModeShared);
                *buf_cell = Some(buf.clone()); *size_cell = input_size; buf
            } else { buf_cell.as_ref().unwrap().clone() }
        };

        let output_buffer = {
            let mut buf_cell = self.mv_output_buffer.borrow_mut();
            let mut size_cell = self.mv_output_size.borrow_mut();
            if *size_cell != output_size {
                let buf = self.device.new_buffer(output_size as u64, metal::MTLResourceOptions::StorageModeShared);
                *buf_cell = Some(buf.clone()); *size_cell = output_size; buf
            } else { buf_cell.as_ref().unwrap().clone() }
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
        encoder.set_bytes(3, std::mem::size_of::<MvArgs>() as u64, &args as *const MvArgs as *const std::ffi::c_void);

        const NSG: u64 = 2;
        let rows_per_threadgroup = NSG * nr0;
        let thread_group_size = MTLSize { width: 32, height: NSG, depth: 1 };
        let thread_group_count = MTLSize { width: (m as u64 + rows_per_threadgroup - 1) / rows_per_threadgroup, height: 1, depth: 1 };

        encoder.dispatch_thread_groups(thread_group_count, thread_group_size);
        encoder.end_encoding();
        command_buffer.commit();
        command_buffer.wait_until_completed();

        let output_ptr = output_buffer.contents() as *const f32;
        Ok(unsafe { std::slice::from_raw_parts(output_ptr, m).to_vec() })
    }

    /// IQ3_XXS x F32 Matrix-Vector with dynamic kernel selection
    pub fn mv_iq3_xxs_f32(&self, m: usize, k: usize, weights: &[BlockIQ3XXS], input: &[f32]) -> Result<Vec<f32>, String> {
        let nb = k / 256;
        let nb01 = (nb * std::mem::size_of::<BlockIQ3XXS>()) as u64;
        let nb32 = nb * (256 / 32);

        // Dynamic dispatch: use split=1 for small nb32
        let use_split = nb32 < 32;
        let nr0 = if use_split { 8u64 } else { 4u64 };

        #[repr(C)]
        struct MvArgs { ne00: u32, ne01: u32, nb01: u64 }

        let args = MvArgs { ne00: k as u32, ne01: m as u32, nb01 };
        let weights_size = weights.len() * std::mem::size_of::<BlockIQ3XXS>();
        let input_size = input.len() * std::mem::size_of::<f32>();
        let output_size = m * std::mem::size_of::<f32>();

        let weights_buffer = {
            let mut buf_cell = self.mv_weights_buffer.borrow_mut();
            let mut size_cell = self.mv_weights_size.borrow_mut();
            if *size_cell != weights_size {
                let buf = self.device.new_buffer_with_data(weights.as_ptr() as *const std::ffi::c_void, weights_size as u64, metal::MTLResourceOptions::StorageModeShared);
                *buf_cell = Some(buf.clone()); *size_cell = weights_size; buf
            } else { buf_cell.as_ref().unwrap().clone() }
        };

        let input_buffer = {
            let mut buf_cell = self.mv_input_buffer.borrow_mut();
            let mut size_cell = self.mv_input_size.borrow_mut();
            if *size_cell != input_size {
                let buf = self.device.new_buffer_with_data(input.as_ptr() as *const std::ffi::c_void, input_size as u64, metal::MTLResourceOptions::StorageModeShared);
                *buf_cell = Some(buf.clone()); *size_cell = input_size; buf
            } else { buf_cell.as_ref().unwrap().clone() }
        };

        let output_buffer = {
            let mut buf_cell = self.mv_output_buffer.borrow_mut();
            let mut size_cell = self.mv_output_size.borrow_mut();
            if *size_cell != output_size {
                let buf = self.device.new_buffer(output_size as u64, metal::MTLResourceOptions::StorageModeShared);
                *buf_cell = Some(buf.clone()); *size_cell = output_size; buf
            } else { buf_cell.as_ref().unwrap().clone() }
        };

        let command_buffer = self.queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();

        // Select appropriate pipeline
        if use_split {
            encoder.set_compute_pipeline_state(&self.mv_iq3_xxs_split1_pipeline);
        } else {
            encoder.set_compute_pipeline_state(&self.mv_iq3_xxs_pipeline);
        }

        encoder.set_buffer(0, Some(&weights_buffer), 0);
        encoder.set_buffer(1, Some(&input_buffer), 0);
        encoder.set_buffer(2, Some(&output_buffer), 0);
        encoder.set_bytes(3, std::mem::size_of::<MvArgs>() as u64, &args as *const MvArgs as *const std::ffi::c_void);

        // Threadgroup memory for IQ3_XXS: svalues (256*4) + ssigns (128) = 1152 bytes
        const THREADGROUP_MEM_SIZE: u64 = 256 * 4 + 128;
        encoder.set_threadgroup_memory_length(THREADGROUP_MEM_SIZE, 0);

        const NSG: u64 = 2;
        let rows_per_threadgroup = NSG * nr0;
        let thread_group_size = MTLSize { width: 32, height: NSG, depth: 1 };
        let thread_group_count = MTLSize { width: (m as u64 + rows_per_threadgroup - 1) / rows_per_threadgroup, height: 1, depth: 1 };

        encoder.dispatch_thread_groups(thread_group_count, thread_group_size);
        encoder.end_encoding();
        command_buffer.commit();
        command_buffer.wait_until_completed();

        let output_ptr = output_buffer.contents() as *const f32;
        Ok(unsafe { std::slice::from_raw_parts(output_ptr, m).to_vec() })
    }

    /// IQ3_S x F32 Matrix-Vector with dynamic kernel selection
    pub fn mv_iq3_s_f32(&self, m: usize, k: usize, weights: &[BlockIQ3S], input: &[f32]) -> Result<Vec<f32>, String> {
        let nb = k / 256;
        let nb01 = (nb * std::mem::size_of::<BlockIQ3S>()) as u64;
        let nb32 = nb * (256 / 32);

        // Dynamic dispatch: use split=1 for small nb32
        let use_split = nb32 < 32;
        let nr0 = if use_split { 8u64 } else { 4u64 };

        #[repr(C)]
        struct MvArgs { ne00: u32, ne01: u32, nb01: u64 }

        let args = MvArgs { ne00: k as u32, ne01: m as u32, nb01 };
        let weights_size = weights.len() * std::mem::size_of::<BlockIQ3S>();
        let input_size = input.len() * std::mem::size_of::<f32>();
        let output_size = m * std::mem::size_of::<f32>();

        let weights_buffer = {
            let mut buf_cell = self.mv_weights_buffer.borrow_mut();
            let mut size_cell = self.mv_weights_size.borrow_mut();
            if *size_cell != weights_size {
                let buf = self.device.new_buffer_with_data(weights.as_ptr() as *const std::ffi::c_void, weights_size as u64, metal::MTLResourceOptions::StorageModeShared);
                *buf_cell = Some(buf.clone()); *size_cell = weights_size; buf
            } else { buf_cell.as_ref().unwrap().clone() }
        };

        let input_buffer = {
            let mut buf_cell = self.mv_input_buffer.borrow_mut();
            let mut size_cell = self.mv_input_size.borrow_mut();
            if *size_cell != input_size {
                let buf = self.device.new_buffer_with_data(input.as_ptr() as *const std::ffi::c_void, input_size as u64, metal::MTLResourceOptions::StorageModeShared);
                *buf_cell = Some(buf.clone()); *size_cell = input_size; buf
            } else { buf_cell.as_ref().unwrap().clone() }
        };

        let output_buffer = {
            let mut buf_cell = self.mv_output_buffer.borrow_mut();
            let mut size_cell = self.mv_output_size.borrow_mut();
            if *size_cell != output_size {
                let buf = self.device.new_buffer(output_size as u64, metal::MTLResourceOptions::StorageModeShared);
                *buf_cell = Some(buf.clone()); *size_cell = output_size; buf
            } else { buf_cell.as_ref().unwrap().clone() }
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
        encoder.set_bytes(3, std::mem::size_of::<MvArgs>() as u64, &args as *const MvArgs as *const std::ffi::c_void);

        // Threadgroup memory for IQ3_S: svalues (512*4) = 2048 bytes
        const THREADGROUP_MEM_SIZE: u64 = 512 * 4;
        encoder.set_threadgroup_memory_length(THREADGROUP_MEM_SIZE, 0);

        const NSG: u64 = 2;
        let rows_per_threadgroup = NSG * nr0;
        let thread_group_size = MTLSize { width: 32, height: NSG, depth: 1 };
        let thread_group_count = MTLSize { width: (m as u64 + rows_per_threadgroup - 1) / rows_per_threadgroup, height: 1, depth: 1 };

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
    pub fn gemm_q4_k_f32(&self, m: usize, n: usize, k: usize, weights: &[BlockQ4K], input: &[f32]) -> Result<Vec<f32>, String> {
        let nb = k / 256;
        let nb01 = (nb * std::mem::size_of::<BlockQ4K>()) as u64;

        #[repr(C)]
        struct GemmArgs {
            ne00: u32,  // K
            ne01: u32,  // M
            ne02: u32,  // N
            nb01: u64,  // Byte stride for weights rows
            nb11: u64,  // Byte stride for input rows
        }

        let args = GemmArgs {
            ne00: k as u32,
            ne01: m as u32,
            ne02: n as u32,
            nb01,
            nb11: k as u64 * 4,  // K * sizeof(f32)
        };

        let weights_size = weights.len() * std::mem::size_of::<BlockQ4K>();
        let input_size = input.len() * std::mem::size_of::<f32>();
        let output_size = m * n * std::mem::size_of::<f32>();

        let weights_buffer = {
            let mut buf_cell = self.gemm_weights_buffer.borrow_mut();
            let mut size_cell = self.gemm_weights_size.borrow_mut();
            if *size_cell != weights_size {
                let buf = self.device.new_buffer_with_data(weights.as_ptr() as *const std::ffi::c_void, weights_size as u64, metal::MTLResourceOptions::StorageModeShared);
                *buf_cell = Some(buf.clone()); *size_cell = weights_size; buf
            } else { buf_cell.as_ref().unwrap().clone() }
        };

        let input_buffer = {
            let mut buf_cell = self.gemm_input_buffer.borrow_mut();
            let mut size_cell = self.gemm_input_size.borrow_mut();
            if *size_cell != input_size {
                let buf = self.device.new_buffer_with_data(input.as_ptr() as *const std::ffi::c_void, input_size as u64, metal::MTLResourceOptions::StorageModeShared);
                *buf_cell = Some(buf.clone()); *size_cell = input_size; buf
            } else { buf_cell.as_ref().unwrap().clone() }
        };

        let output_buffer = {
            let mut buf_cell = self.gemm_output_buffer.borrow_mut();
            let mut size_cell = self.gemm_output_size.borrow_mut();
            if *size_cell != output_size {
                let buf = self.device.new_buffer(output_size as u64, metal::MTLResourceOptions::StorageModeShared);
                *buf_cell = Some(buf.clone()); *size_cell = output_size; buf
            } else { buf_cell.as_ref().unwrap().clone() }
        };

        let command_buffer = self.queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(&self.gemm_q4_k_pipeline);
        encoder.set_buffer(0, Some(&weights_buffer), 0);
        encoder.set_buffer(1, Some(&input_buffer), 0);
        encoder.set_buffer(2, Some(&output_buffer), 0);
        encoder.set_bytes(3, std::mem::size_of::<GemmArgs>() as u64, &args as *const GemmArgs as *const std::ffi::c_void);

        // Dynamic dispatch based on K size for better GPU utilization
        // Small K (nb <= 4): use NR0=8 rows per threadgroup, NC=4 columns
        // Large K (nb > 4): use NR0=2 rows per threadgroup, NC=4 columns
        let (nr0, nc) = if nb <= 4 {
            (8u64, 4u64)  // Small K: more rows per threadgroup
        } else {
            (2u64, 4u64)  // Large K: use shader's N_COLS_LARGE=4
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
    pub fn gemm_q2_k_f32(&self, m: usize, n: usize, k: usize, weights: &[BlockQ2K], input: &[f32]) -> Result<Vec<f32>, String> {
        let nb = k / 256;
        let nb01 = (nb * std::mem::size_of::<BlockQ2K>()) as u64;

        #[repr(C)]
        struct GemmArgs { ne00: u32, ne01: u32, ne02: u32, nb01: u64, nb11: u64 }
        let args = GemmArgs { ne00: k as u32, ne01: m as u32, ne02: n as u32, nb01, nb11: k as u64 * 4 };

        let weights_size = weights.len() * std::mem::size_of::<BlockQ2K>();
        let input_size = input.len() * std::mem::size_of::<f32>();
        let output_size = m * n * std::mem::size_of::<f32>();

        let weights_buffer = {
            let mut buf_cell = self.gemm_weights_buffer.borrow_mut();
            let mut size_cell = self.gemm_weights_size.borrow_mut();
            if *size_cell != weights_size {
                let buf = self.device.new_buffer_with_data(weights.as_ptr() as *const std::ffi::c_void, weights_size as u64, metal::MTLResourceOptions::StorageModeShared);
                *buf_cell = Some(buf.clone()); *size_cell = weights_size; buf
            } else { buf_cell.as_ref().unwrap().clone() }
        };

        let input_buffer = {
            let mut buf_cell = self.gemm_input_buffer.borrow_mut();
            let mut size_cell = self.gemm_input_size.borrow_mut();
            if *size_cell != input_size {
                let buf = self.device.new_buffer_with_data(input.as_ptr() as *const std::ffi::c_void, input_size as u64, metal::MTLResourceOptions::StorageModeShared);
                *buf_cell = Some(buf.clone()); *size_cell = input_size; buf
            } else { buf_cell.as_ref().unwrap().clone() }
        };

        let output_buffer = {
            let mut buf_cell = self.gemm_output_buffer.borrow_mut();
            let mut size_cell = self.gemm_output_size.borrow_mut();
            if *size_cell != output_size {
                let buf = self.device.new_buffer(output_size as u64, metal::MTLResourceOptions::StorageModeShared);
                *buf_cell = Some(buf.clone()); *size_cell = output_size; buf
            } else { buf_cell.as_ref().unwrap().clone() }
        };

        let command_buffer = self.queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(&self.gemm_q2_k_pipeline);
        encoder.set_buffer(0, Some(&weights_buffer), 0);
        encoder.set_buffer(1, Some(&input_buffer), 0);
        encoder.set_buffer(2, Some(&output_buffer), 0);
        encoder.set_bytes(3, std::mem::size_of::<GemmArgs>() as u64, &args as *const GemmArgs as *const std::ffi::c_void);

        // Dynamic dispatch based on K size
        // Small K (nb <= 4): NR0=8, NC=4
        // Large K (nb > 4): NR0=2, NC=4
        let (nr0, nc) = if nb <= 4 {
            (8u64, 4u64)
        } else {
            (2u64, 4u64)
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

    /// Q3_K × F32 GEMM
    pub fn gemm_q3_k_f32(&self, m: usize, n: usize, k: usize, weights: &[BlockQ3K], input: &[f32]) -> Result<Vec<f32>, String> {
        let nb = k / 256;
        let nb01 = (nb * std::mem::size_of::<BlockQ3K>()) as u64;

        #[repr(C)]
        struct GemmArgs { ne00: u32, ne01: u32, ne02: u32, nb01: u64, nb11: u64 }
        let args = GemmArgs { ne00: k as u32, ne01: m as u32, ne02: n as u32, nb01, nb11: k as u64 * 4 };

        let weights_size = weights.len() * std::mem::size_of::<BlockQ3K>();
        let input_size = input.len() * std::mem::size_of::<f32>();
        let output_size = m * n * std::mem::size_of::<f32>();

        let weights_buffer = {
            let mut buf_cell = self.gemm_weights_buffer.borrow_mut();
            let mut size_cell = self.gemm_weights_size.borrow_mut();
            if *size_cell != weights_size {
                let buf = self.device.new_buffer_with_data(weights.as_ptr() as *const std::ffi::c_void, weights_size as u64, metal::MTLResourceOptions::StorageModeShared);
                *buf_cell = Some(buf.clone()); *size_cell = weights_size; buf
            } else { buf_cell.as_ref().unwrap().clone() }
        };

        let input_buffer = {
            let mut buf_cell = self.gemm_input_buffer.borrow_mut();
            let mut size_cell = self.gemm_input_size.borrow_mut();
            if *size_cell != input_size {
                let buf = self.device.new_buffer_with_data(input.as_ptr() as *const std::ffi::c_void, input_size as u64, metal::MTLResourceOptions::StorageModeShared);
                *buf_cell = Some(buf.clone()); *size_cell = input_size; buf
            } else { buf_cell.as_ref().unwrap().clone() }
        };

        let output_buffer = {
            let mut buf_cell = self.gemm_output_buffer.borrow_mut();
            let mut size_cell = self.gemm_output_size.borrow_mut();
            if *size_cell != output_size {
                let buf = self.device.new_buffer(output_size as u64, metal::MTLResourceOptions::StorageModeShared);
                *buf_cell = Some(buf.clone()); *size_cell = output_size; buf
            } else { buf_cell.as_ref().unwrap().clone() }
        };

        let command_buffer = self.queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(&self.gemm_q3_k_pipeline);
        encoder.set_buffer(0, Some(&weights_buffer), 0);
        encoder.set_buffer(1, Some(&input_buffer), 0);
        encoder.set_buffer(2, Some(&output_buffer), 0);
        encoder.set_bytes(3, std::mem::size_of::<GemmArgs>() as u64, &args as *const GemmArgs as *const std::ffi::c_void);

        // Dynamic dispatch based on K size
        // Small K (nb <= 4): NR0=8, NC=4
        // Large K (nb > 4): NR0=2, NC=4
        let (nr0, nc) = if nb <= 4 {
            (8u64, 4u64)
        } else {
            (2u64, 4u64)
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

    /// Q5_K × F32 GEMM
    pub fn gemm_q5_k_f32(&self, m: usize, n: usize, k: usize, weights: &[BlockQ5K], input: &[f32]) -> Result<Vec<f32>, String> {
        let nb = k / 256;
        let nb01 = (nb * std::mem::size_of::<BlockQ5K>()) as u64;

        #[repr(C)]
        struct GemmArgs { ne00: u32, ne01: u32, ne02: u32, nb01: u64, nb11: u64 }
        let args = GemmArgs { ne00: k as u32, ne01: m as u32, ne02: n as u32, nb01, nb11: k as u64 * 4 };

        let weights_size = weights.len() * std::mem::size_of::<BlockQ5K>();
        let input_size = input.len() * std::mem::size_of::<f32>();
        let output_size = m * n * std::mem::size_of::<f32>();

        let weights_buffer = {
            let mut buf_cell = self.gemm_weights_buffer.borrow_mut();
            let mut size_cell = self.gemm_weights_size.borrow_mut();
            if *size_cell != weights_size {
                let buf = self.device.new_buffer_with_data(weights.as_ptr() as *const std::ffi::c_void, weights_size as u64, metal::MTLResourceOptions::StorageModeShared);
                *buf_cell = Some(buf.clone()); *size_cell = weights_size; buf
            } else { buf_cell.as_ref().unwrap().clone() }
        };

        let input_buffer = {
            let mut buf_cell = self.gemm_input_buffer.borrow_mut();
            let mut size_cell = self.gemm_input_size.borrow_mut();
            if *size_cell != input_size {
                let buf = self.device.new_buffer_with_data(input.as_ptr() as *const std::ffi::c_void, input_size as u64, metal::MTLResourceOptions::StorageModeShared);
                *buf_cell = Some(buf.clone()); *size_cell = input_size; buf
            } else { buf_cell.as_ref().unwrap().clone() }
        };

        let output_buffer = {
            let mut buf_cell = self.gemm_output_buffer.borrow_mut();
            let mut size_cell = self.gemm_output_size.borrow_mut();
            if *size_cell != output_size {
                let buf = self.device.new_buffer(output_size as u64, metal::MTLResourceOptions::StorageModeShared);
                *buf_cell = Some(buf.clone()); *size_cell = output_size; buf
            } else { buf_cell.as_ref().unwrap().clone() }
        };

        let command_buffer = self.queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(&self.gemm_q5_k_pipeline);
        encoder.set_buffer(0, Some(&weights_buffer), 0);
        encoder.set_buffer(1, Some(&input_buffer), 0);
        encoder.set_buffer(2, Some(&output_buffer), 0);
        encoder.set_bytes(3, std::mem::size_of::<GemmArgs>() as u64, &args as *const GemmArgs as *const std::ffi::c_void);

        // Dynamic dispatch based on K size
        // Small K (nb <= 4): NR0=8, NC=4
        // Large K (nb > 4): NR0=2, NC=4
        let (nr0, nc) = if nb <= 4 {
            (8u64, 4u64)
        } else {
            (2u64, 4u64)
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

    /// Q6_K × F32 GEMM
    pub fn gemm_q6_k_f32(&self, m: usize, n: usize, k: usize, weights: &[BlockQ6K], input: &[f32]) -> Result<Vec<f32>, String> {
        let nb = k / 256;
        let nb01 = (nb * std::mem::size_of::<BlockQ6K>()) as u64;

        #[repr(C)]
        struct GemmArgs { ne00: u32, ne01: u32, ne02: u32, nb01: u64, nb11: u64 }
        let args = GemmArgs { ne00: k as u32, ne01: m as u32, ne02: n as u32, nb01, nb11: k as u64 * 4 };

        let weights_size = weights.len() * std::mem::size_of::<BlockQ6K>();
        let input_size = input.len() * std::mem::size_of::<f32>();
        let output_size = m * n * std::mem::size_of::<f32>();

        let weights_buffer = {
            let mut buf_cell = self.gemm_weights_buffer.borrow_mut();
            let mut size_cell = self.gemm_weights_size.borrow_mut();
            if *size_cell != weights_size {
                let buf = self.device.new_buffer_with_data(weights.as_ptr() as *const std::ffi::c_void, weights_size as u64, metal::MTLResourceOptions::StorageModeShared);
                *buf_cell = Some(buf.clone()); *size_cell = weights_size; buf
            } else { buf_cell.as_ref().unwrap().clone() }
        };

        let input_buffer = {
            let mut buf_cell = self.gemm_input_buffer.borrow_mut();
            let mut size_cell = self.gemm_input_size.borrow_mut();
            if *size_cell != input_size {
                let buf = self.device.new_buffer_with_data(input.as_ptr() as *const std::ffi::c_void, input_size as u64, metal::MTLResourceOptions::StorageModeShared);
                *buf_cell = Some(buf.clone()); *size_cell = input_size; buf
            } else { buf_cell.as_ref().unwrap().clone() }
        };

        let output_buffer = {
            let mut buf_cell = self.gemm_output_buffer.borrow_mut();
            let mut size_cell = self.gemm_output_size.borrow_mut();
            if *size_cell != output_size {
                let buf = self.device.new_buffer(output_size as u64, metal::MTLResourceOptions::StorageModeShared);
                *buf_cell = Some(buf.clone()); *size_cell = output_size; buf
            } else { buf_cell.as_ref().unwrap().clone() }
        };

        let command_buffer = self.queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(&self.gemm_q6_k_pipeline);
        encoder.set_buffer(0, Some(&weights_buffer), 0);
        encoder.set_buffer(1, Some(&input_buffer), 0);
        encoder.set_buffer(2, Some(&output_buffer), 0);
        encoder.set_bytes(3, std::mem::size_of::<GemmArgs>() as u64, &args as *const GemmArgs as *const std::ffi::c_void);

        // Dynamic dispatch based on K size
        // Small K (nb <= 4): NR0=8, NC=4
        // Large K (nb > 4): NR0=2, NC=4
        let (nr0, nc) = if nb <= 4 {
            (8u64, 4u64)
        } else {
            (2u64, 4u64)
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

    // ===== Templated GEMM Methods (from llama.cpp) =====
    // These use the kernel_mul_mm template with dequantize functions

    /// Templated Q4_K × F32 GEMM (using llama.cpp kernel_mul_mm)
    /// Note: Output is in column-major format (column * M + row)
    pub fn gemm_template_q4_k_f32(&self, m: usize, n: usize, k: usize, weights: &[BlockQ4K], input: &[f32]) -> Result<Vec<f32>, String> {
        let nb = k / 256;
        let nb01 = (nb * std::mem::size_of::<BlockQ4K>()) as u64;
        let nb00 = std::mem::size_of::<BlockQ4K>() as u64;

        // Use exact struct from llama.cpp's ggml-metal-impl.h
        #[repr(C)]
        struct GemmArgs {
            ne00: i32,   // K
            ne02: i32,   // batch dim (1 for simple case)
            nb01: u64,   // stride for A rows
            nb02: u64,   // batch stride (0 for simple case)
            nb03: u64,   // batch3 stride (0)
            ne12: i32,   // batch dim for B (1)
            nb10: u64,   // element stride for B (2 for FP16)
            nb11: u64,   // row stride for B (K * 2)
            nb12: u64,   // batch stride (0)
            nb13: u64,   // batch3 stride (0)
            ne0: i32,    // M (output rows)
            ne1: i32,    // N (output cols)
            r2: i16,     // broadcast ratio (1)
            r3: i16,     // broadcast ratio (1)
        }

        let args = GemmArgs {
            ne00: k as i32,
            ne02: 1,
            nb01,
            nb02: 0,
            nb03: 0,
            ne12: 1,
            nb10: 2,  // FP16 element size
            nb11: (k * 2) as u64,  // K * sizeof(FP16)
            nb12: 0,
            nb13: 0,
            ne0: m as i32,
            ne1: n as i32,
            r2: 1,
            r3: 1,
        };

        let weights_size = weights.len() * std::mem::size_of::<BlockQ4K>();

        // Convert FP32 input to FP16
        let input_f16: Vec<u16> = input.iter()
            .map(|&x| half::f16::from_f32(x).to_bits())
            .collect();
        let input_size = input_f16.len() * 2;
        let output_size = m * n * std::mem::size_of::<f32>();

        let weights_buffer = self.device.new_buffer_with_data(weights.as_ptr() as *const std::ffi::c_void, weights_size as u64, metal::MTLResourceOptions::StorageModeShared);
        let input_buffer = self.device.new_buffer_with_data(input_f16.as_ptr() as *const std::ffi::c_void, input_size as u64, metal::MTLResourceOptions::StorageModeShared);
        let output_buffer = self.device.new_buffer(output_size as u64, metal::MTLResourceOptions::StorageModeShared);

        let command_buffer = self.queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(&self.gemm_template_q4_k_pipeline);
        encoder.set_bytes(0, std::mem::size_of::<GemmArgs>() as u64, &args as *const GemmArgs as *const std::ffi::c_void);
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
        const NSG: u64 = 4;  // N_MM_SIMD_GROUP_X * N_MM_SIMD_GROUP_Y

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
    pub fn gemm_template_q2_k_f32(&self, m: usize, n: usize, k: usize, weights: &[BlockQ2K], input: &[f32]) -> Result<Vec<f32>, String> {
        let nb = k / 256;
        let nb01 = (nb * std::mem::size_of::<BlockQ2K>()) as u64;
        let nb00 = std::mem::size_of::<BlockQ2K>() as u64;

        #[repr(C)]
        struct GemmArgs {
            ne00: i32, ne01: i32, ne02: i32, ne03: i32,
            nb00: u64, nb01: u64, nb02: u64, nb03: u64,
            ne10: i32, ne11: i32, ne12: i32, ne13: i32,
            nb10: u64, nb11: u64, nb12: u64, nb13: u64,
            ne0: i32, ne1: i32, ne2: i32, ne3: i32,
            nb0: u64, nb1: u64, nb2: u64, nb3: u64,
        }

        let args = GemmArgs {
            ne00: k as i32, ne01: m as i32, ne02: 1, ne03: 1,
            nb00, nb01, nb02: 0, nb03: 0,
            ne10: k as i32, ne11: n as i32, ne12: 1, ne13: 1,
            nb10: 2, nb11: (k * 2) as u64, nb12: 0, nb13: 0, // FP16: element stride=2, row stride=K*2
            ne0: m as i32, ne1: n as i32, ne2: 1, ne3: 1,  // ne0=M (rows), ne1=N (cols)
            nb0: 4, nb1: (n * 4) as u64, nb2: 0, nb3: 0,  // row stride = N * sizeof(f32)
        };

        let weights_size = weights.len() * std::mem::size_of::<BlockQ2K>();

        // Convert FP32 input to FP16
        let input_f16: Vec<u16> = input.iter()
            .map(|&x| half::f16::from_f32(x).to_bits())
            .collect();
        let input_size = input_f16.len() * 2;
        let output_size = m * n * std::mem::size_of::<f32>();

        let weights_buffer = self.device.new_buffer_with_data(weights.as_ptr() as *const std::ffi::c_void, weights_size as u64, metal::MTLResourceOptions::StorageModeShared);
        let input_buffer = self.device.new_buffer_with_data(input_f16.as_ptr() as *const std::ffi::c_void, input_size as u64, metal::MTLResourceOptions::StorageModeShared);
        let output_buffer = self.device.new_buffer(output_size as u64, metal::MTLResourceOptions::StorageModeShared);

        let command_buffer = self.queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(&self.gemm_template_q2_k_pipeline);
        encoder.set_bytes(0, std::mem::size_of::<GemmArgs>() as u64, &args as *const GemmArgs as *const std::ffi::c_void);
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
    pub fn gemm_template_q3_k_f32(&self, m: usize, n: usize, k: usize, weights: &[BlockQ3K], input: &[f32]) -> Result<Vec<f32>, String> {
        let nb = k / 256;
        let nb01 = (nb * std::mem::size_of::<BlockQ3K>()) as u64;
        let nb00 = std::mem::size_of::<BlockQ3K>() as u64;

        #[repr(C)]
        struct GemmArgs {
            ne00: i32, ne01: i32, ne02: i32, ne03: i32,
            nb00: u64, nb01: u64, nb02: u64, nb03: u64,
            ne10: i32, ne11: i32, ne12: i32, ne13: i32,
            nb10: u64, nb11: u64, nb12: u64, nb13: u64,
            ne0: i32, ne1: i32, ne2: i32, ne3: i32,
            nb0: u64, nb1: u64, nb2: u64, nb3: u64,
        }

        let args = GemmArgs {
            ne00: k as i32, ne01: m as i32, ne02: 1, ne03: 1,
            nb00, nb01, nb02: 0, nb03: 0,
            ne10: k as i32, ne11: n as i32, ne12: 1, ne13: 1,
            nb10: 2, nb11: (k * 2) as u64, nb12: 0, nb13: 0, // FP16: element stride=2, row stride=K*2
            ne0: m as i32, ne1: n as i32, ne2: 1, ne3: 1,  // ne0=M (rows), ne1=N (cols)
            nb0: 4, nb1: (n * 4) as u64, nb2: 0, nb3: 0,  // row stride = N * sizeof(f32)
        };

        let weights_size = weights.len() * std::mem::size_of::<BlockQ3K>();

        // Convert FP32 input to FP16
        let input_f16: Vec<u16> = input.iter()
            .map(|&x| half::f16::from_f32(x).to_bits())
            .collect();
        let input_size = input_f16.len() * 2;
        let output_size = m * n * std::mem::size_of::<f32>();

        let weights_buffer = self.device.new_buffer_with_data(weights.as_ptr() as *const std::ffi::c_void, weights_size as u64, metal::MTLResourceOptions::StorageModeShared);
        let input_buffer = self.device.new_buffer_with_data(input_f16.as_ptr() as *const std::ffi::c_void, input_size as u64, metal::MTLResourceOptions::StorageModeShared);
        let output_buffer = self.device.new_buffer(output_size as u64, metal::MTLResourceOptions::StorageModeShared);

        let command_buffer = self.queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(&self.gemm_template_q3_k_pipeline);
        encoder.set_bytes(0, std::mem::size_of::<GemmArgs>() as u64, &args as *const GemmArgs as *const std::ffi::c_void);
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
    pub fn gemm_template_q5_k_f32(&self, m: usize, n: usize, k: usize, weights: &[BlockQ5K], input: &[f32]) -> Result<Vec<f32>, String> {
        let nb = k / 256;
        let nb01 = (nb * std::mem::size_of::<BlockQ5K>()) as u64;
        let nb00 = std::mem::size_of::<BlockQ5K>() as u64;

        #[repr(C)]
        struct GemmArgs {
            ne00: i32, ne01: i32, ne02: i32, ne03: i32,
            nb00: u64, nb01: u64, nb02: u64, nb03: u64,
            ne10: i32, ne11: i32, ne12: i32, ne13: i32,
            nb10: u64, nb11: u64, nb12: u64, nb13: u64,
            ne0: i32, ne1: i32, ne2: i32, ne3: i32,
            nb0: u64, nb1: u64, nb2: u64, nb3: u64,
        }

        let args = GemmArgs {
            ne00: k as i32, ne01: m as i32, ne02: 1, ne03: 1,
            nb00, nb01, nb02: 0, nb03: 0,
            ne10: k as i32, ne11: n as i32, ne12: 1, ne13: 1,
            nb10: 2, nb11: (k * 2) as u64, nb12: 0, nb13: 0, // FP16: element stride=2, row stride=K*2
            ne0: m as i32, ne1: n as i32, ne2: 1, ne3: 1,  // ne0=M (rows), ne1=N (cols)
            nb0: 4, nb1: (n * 4) as u64, nb2: 0, nb3: 0,  // row stride = N * sizeof(f32)
        };

        let weights_size = weights.len() * std::mem::size_of::<BlockQ5K>();

        // Convert FP32 input to FP16
        let input_f16: Vec<u16> = input.iter()
            .map(|&x| half::f16::from_f32(x).to_bits())
            .collect();
        let input_size = input_f16.len() * 2;
        let output_size = m * n * std::mem::size_of::<f32>();

        let weights_buffer = self.device.new_buffer_with_data(weights.as_ptr() as *const std::ffi::c_void, weights_size as u64, metal::MTLResourceOptions::StorageModeShared);
        let input_buffer = self.device.new_buffer_with_data(input_f16.as_ptr() as *const std::ffi::c_void, input_size as u64, metal::MTLResourceOptions::StorageModeShared);
        let output_buffer = self.device.new_buffer(output_size as u64, metal::MTLResourceOptions::StorageModeShared);

        let command_buffer = self.queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(&self.gemm_template_q5_k_pipeline);
        encoder.set_bytes(0, std::mem::size_of::<GemmArgs>() as u64, &args as *const GemmArgs as *const std::ffi::c_void);
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
    pub fn gemm_template_q6_k_f32(&self, m: usize, n: usize, k: usize, weights: &[BlockQ6K], input: &[f32]) -> Result<Vec<f32>, String> {
        let nb = k / 256;
        let nb01 = (nb * std::mem::size_of::<BlockQ6K>()) as u64;
        let nb00 = std::mem::size_of::<BlockQ6K>() as u64;

        #[repr(C)]
        struct GemmArgs {
            ne00: i32, ne01: i32, ne02: i32, ne03: i32,
            nb00: u64, nb01: u64, nb02: u64, nb03: u64,
            ne10: i32, ne11: i32, ne12: i32, ne13: i32,
            nb10: u64, nb11: u64, nb12: u64, nb13: u64,
            ne0: i32, ne1: i32, ne2: i32, ne3: i32,
            nb0: u64, nb1: u64, nb2: u64, nb3: u64,
        }

        let args = GemmArgs {
            ne00: k as i32, ne01: m as i32, ne02: 1, ne03: 1,
            nb00, nb01, nb02: 0, nb03: 0,
            ne10: k as i32, ne11: n as i32, ne12: 1, ne13: 1,
            nb10: 2, nb11: (k * 2) as u64, nb12: 0, nb13: 0, // FP16: element stride=2, row stride=K*2
            ne0: m as i32, ne1: n as i32, ne2: 1, ne3: 1,  // ne0=M (rows), ne1=N (cols)
            nb0: 4, nb1: (n * 4) as u64, nb2: 0, nb3: 0,  // row stride = N * sizeof(f32)
        };

        let weights_size = weights.len() * std::mem::size_of::<BlockQ6K>();

        // Convert FP32 input to FP16
        let input_f16: Vec<u16> = input.iter()
            .map(|&x| half::f16::from_f32(x).to_bits())
            .collect();
        let input_size = input_f16.len() * 2;
        let output_size = m * n * std::mem::size_of::<f32>();

        let weights_buffer = self.device.new_buffer_with_data(weights.as_ptr() as *const std::ffi::c_void, weights_size as u64, metal::MTLResourceOptions::StorageModeShared);
        let input_buffer = self.device.new_buffer_with_data(input_f16.as_ptr() as *const std::ffi::c_void, input_size as u64, metal::MTLResourceOptions::StorageModeShared);
        let output_buffer = self.device.new_buffer(output_size as u64, metal::MTLResourceOptions::StorageModeShared);

        let command_buffer = self.queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(&self.gemm_template_q6_k_pipeline);
        encoder.set_bytes(0, std::mem::size_of::<GemmArgs>() as u64, &args as *const GemmArgs as *const std::ffi::c_void);
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
    /// This is the main entry point for quantized matrix multiplication
    pub fn matmul_q4_k_f32(&self, m: usize, n: usize, k: usize, weights: &[BlockQ4K], input: &[f32]) -> Result<Vec<f32>, String> {
        // Dispatch strategy based on batch size:
        // N=1 (decode): use MV kernel
        // N≥32 (batch): use GEMM template kernel
        if n == 1 {
            self.mv_q4_k_f32(m, k, weights, input)
        } else if n >= 32 {
            // Use GEMM template for batch processing
            self.gemm_template_q4_k_f32(m, n, k, weights, input)
        } else {
            // Fallback to old GEMM for intermediate sizes (N=2-31)
            // This path should be optimized later
            self.gemm_q4_k_f32(m, n, k, weights, input)
        }
    }
}
