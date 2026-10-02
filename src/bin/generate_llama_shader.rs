// Generate combined llama.cpp shader for Standard Q + TQ + MXFP
// Run this once to create /tmp/combined_llama.metal

use std::fs;

fn main() {
    let base_dir = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("llama.cpp-0.5.0/ggml/src");

    // Read ggml-common.h (block definitions)
    let ggml_common = fs::read_to_string(base_dir.join("ggml-common.h"))
        .expect("Failed to read ggml-common.h");

    // Read kernels/common.h
    let common_h =
        fs::read_to_string(base_dir.join("ggml-metal/kernels/common.h"))
            .expect("Failed to read common.h");

    // Read kernels/dequantize.h (contains block_q_n_dot_y)
    let dequantize_h =
        fs::read_to_string(base_dir.join("ggml-metal/kernels/dequantize.h"))
            .expect("Failed to read dequantize.h");

    // Read kernels/mul_mv.metal (kernel implementations)
    let mul_mv =
        fs::read_to_string(base_dir.join("ggml-metal/kernels/mul_mv.metal"))
            .expect("Failed to read mul_mv.metal");

    // Note: We use the grid tables from ggml-common.h (enabled by GGML_COMMON_IMPL_METAL)
    // rather than our separate iq_grid_tables.h to avoid redefinitions.

    // Combine all sources
    let combined = format!(
        r#"
// Combined llama.cpp shader for Standard Q + TQ + MXFP

#include <metal_stdlib>
using namespace metal;

// Enable Metal implementation in ggml-common.h
#define GGML_COMMON_DECL_METAL
#define GGML_COMMON_IMPL_METAL

// Block type definitions and grid tables from ggml-common.h
{}

// Common definitions
{}

// Dequantize functions (contains block_q_n_dot_y)
{}

// Kernel implementations
{}
"#,
        extract_metal_blocks(&ggml_common),
        extract_metal_common(&common_h),
        extract_metal_dequant(&dequantize_h),
        extract_metal_kernels(&mul_mv)
    );

    // Write to /tmp
    fs::write("/tmp/combined_llama.metal", &combined).expect("Failed to write");

    // Also write ggml-common.h to /tmp (just Metal typedefs, block types are in combined_llama.metal)
    let ggml_common_metal = r#"
using namespace metal;
typedef half ggml_half;
typedef half2 ggml_half2;
"#;
    fs::write("/tmp/ggml-common.h", ggml_common_metal)
        .expect("Failed to write ggml-common.h");

    println!("Written to /tmp/combined_llama.metal ({} bytes)", combined.len());
    println!(
        "Written to /tmp/ggml-common.h ({} bytes)",
        ggml_common_metal.len()
    );
}

fn extract_metal_blocks(source: &str) -> String {
    // Simply return the entire ggml-common.h content
    // The GGML_COMMON_IMPL_METAL define at the top will enable the Metal parts
    source.to_string()
}

fn extract_metal_common(source: &str) -> String {
    let mut result = String::new();

    // Add FC_MUL_MV and related constants from ggml-metal-impl.h
    result.push_str("// Function constants for MV kernels\n");
    result.push_str("#define FC_MUL_MV 600\n");
    result.push_str("#define N_SIMDWIDTH 32\n\n");

    // Add N_R0 and N_SG constants for each format
    result.push_str("// Rows per simdgroup and simdgroups per threadgroup\n");
    result.push_str("#define N_R0_Q1_0 8\n");
    result.push_str("#define N_SG_Q1_0 2\n");
    result.push_str("#define N_R0_Q2_0 8\n");
    result.push_str("#define N_SG_Q2_0 2\n");
    result.push_str("#define N_R0_Q4_0 4\n");
    result.push_str("#define N_SG_Q4_0 2\n");
    result.push_str("#define N_R0_Q4_1 4\n");
    result.push_str("#define N_SG_Q4_1 2\n");
    result.push_str("#define N_R0_Q5_0 4\n");
    result.push_str("#define N_SG_Q5_0 2\n");
    result.push_str("#define N_R0_Q5_1 4\n");
    result.push_str("#define N_SG_Q5_1 2\n");
    result.push_str("#define N_R0_Q8_0 2\n");
    result.push_str("#define N_SG_Q8_0 4\n");
    result.push_str("#define N_R0_MXFP4 2\n");
    result.push_str("#define N_SG_MXFP4 2\n");
    result.push_str("#define N_R0_Q2_K 4\n");
    result.push_str("#define N_SG_Q2_K 2\n");
    result.push_str("#define N_R0_Q3_K 2\n");
    result.push_str("#define N_SG_Q3_K 2\n");
    result.push_str("#define N_R0_Q4_K 2\n");
    result.push_str("#define N_SG_Q4_K 2\n");
    result.push_str("#define N_R0_Q5_K 1\n");
    result.push_str("#define N_SG_Q5_K 2\n");
    result.push_str("#define N_R0_Q6_K 2\n");
    result.push_str("#define N_SG_Q6_K 2\n");
    result.push_str("#define N_R0_IQ1_S 4\n");
    result.push_str("#define N_SG_IQ1_S 2\n");
    result.push_str("#define N_R0_IQ1_S_SPLIT 8\n");
    result.push_str("#define N_R0_IQ1_M 4\n");
    result.push_str("#define N_SG_IQ1_M 2\n");
    result.push_str("#define N_R0_IQ1_M_SPLIT 8\n");
    result.push_str("#define N_R0_IQ2_XXS 4\n");
    result.push_str("#define N_SG_IQ2_XXS 2\n");
    result.push_str("#define N_R0_IQ2_XXS_SPLIT 8\n");
    result.push_str("#define N_R0_IQ2_XS 4\n");
    result.push_str("#define N_SG_IQ2_XS 2\n");
    result.push_str("#define N_R0_IQ2_XS_SPLIT 8\n");
    result.push_str("#define N_R0_IQ2_S 4\n");
    result.push_str("#define N_SG_IQ2_S 2\n");
    result.push_str("#define N_R0_IQ2_S_SPLIT 8\n");
    result.push_str("#define N_R0_IQ3_XXS 4\n");
    result.push_str("#define N_SG_IQ3_XXS 2\n");
    result.push_str("#define N_R0_IQ3_XXS_SPLIT 8\n");
    result.push_str("#define N_R0_IQ3_S 4\n");
    result.push_str("#define N_SG_IQ3_S 2\n");
    result.push_str("#define N_R0_IQ3_S_SPLIT 8\n");
    result.push_str("#define N_R0_IQ4_NL 2\n");
    result.push_str("#define N_SG_IQ4_NL 2\n");
    result.push_str("#define N_R0_IQ4_XS 2\n");
    result.push_str("#define N_SG_IQ4_XS 2\n");
    result.push_str("#define N_R0_TQ2_0 4\n");
    result.push_str("#define N_SG_TQ2_0 2\n\n");

    // Additional QK constants
    result.push_str("#define QK4_NL 32\n\n");

    // Add argument struct for mul_mv
    result.push_str("typedef struct {\n");
    result.push_str("    int32_t  ne00;\n");
    result.push_str("    int32_t  ne01;\n");
    result.push_str("    int32_t  ne02;\n");
    result.push_str("    uint64_t nb00;\n");
    result.push_str("    uint64_t nb01;\n");
    result.push_str("    uint64_t nb02;\n");
    result.push_str("    uint64_t nb03;\n");
    result.push_str("    int32_t  ne10;\n");
    result.push_str("    int32_t  ne11;\n");
    result.push_str("    int32_t  ne12;\n");
    result.push_str("    uint64_t nb10;\n");
    result.push_str("    uint64_t nb11;\n");
    result.push_str("    uint64_t nb12;\n");
    result.push_str("    uint64_t nb13;\n");
    result.push_str("    int32_t  ne0;\n");
    result.push_str("    int32_t  ne1;\n");
    result.push_str("    int32_t  nr0;\n");
    result.push_str("    int16_t  r2;\n");
    result.push_str("    int16_t  r3;\n");
    result.push_str("} ggml_metal_kargs_mul_mv;\n\n");

    // Add argument struct for mul_mv_ext (for IQ formats)
    result.push_str("typedef struct {\n");
    result.push_str("    int32_t  ne00;\n");
    result.push_str("    int32_t  ne01;\n");
    result.push_str("    int32_t  ne02;\n");
    result.push_str("    uint64_t nb00;\n");
    result.push_str("    uint64_t nb01;\n");
    result.push_str("    uint64_t nb02;\n");
    result.push_str("    uint64_t nb03;\n");
    result.push_str("    int32_t  ne10;\n");
    result.push_str("    int32_t  ne11;\n");
    result.push_str("    int32_t  ne12;\n");
    result.push_str("    uint64_t nb10;\n");
    result.push_str("    uint64_t nb11;\n");
    result.push_str("    uint64_t nb12;\n");
    result.push_str("    uint64_t nb13;\n");
    result.push_str("    int32_t  ne0;\n");
    result.push_str("    int32_t  ne1;\n");
    result.push_str("    int16_t  r2;\n");
    result.push_str("    int16_t  r3;\n");
    result.push_str("} ggml_metal_kargs_mul_mv_ext;\n\n");

    // Add argument struct for mul_mv_id (for MoE - not used in this benchmark but needed for compilation)
    result.push_str("typedef struct {\n");
    result.push_str("    int32_t  nei0;\n");
    result.push_str("    int32_t  nei1;\n");
    result.push_str("    uint64_t nbi1;\n");
    result.push_str("    int32_t  ne00;\n");
    result.push_str("    int32_t  ne01;\n");
    result.push_str("    int32_t  ne02;\n");
    result.push_str("    uint64_t nb00;\n");
    result.push_str("    uint64_t nb01;\n");
    result.push_str("    uint64_t nb02;\n");
    result.push_str("    int32_t  ne10;\n");
    result.push_str("    int32_t  ne11;\n");
    result.push_str("    int32_t  ne12;\n");
    result.push_str("    int32_t  ne13;\n");
    result.push_str("    uint64_t nb10;\n");
    result.push_str("    uint64_t nb11;\n");
    result.push_str("    uint64_t nb12;\n");
    result.push_str("    int32_t  ne0;\n");
    result.push_str("    int32_t  ne1;\n");
    result.push_str("    uint64_t nb1;\n");
    result.push_str("    int32_t  nr0;\n");
    result.push_str("} ggml_metal_kargs_mul_mv_id;\n\n");

    // Add iq1m_scale_t type for IQ1_M (only if not already defined in ggml-common.h)
    // Note: ggml-common.h already defines this, so we skip it here
    // result.push_str("typedef union { ggml_half f16; uint16_t u16; } iq1m_scale_t;\n\n");

    for line in source.lines() {
        // Skip includes and pragma
        if line.starts_with("#include") || line.starts_with("#pragma") {
            continue;
        }
        result.push_str(line);
        result.push('\n');
    }
    result
}

fn extract_metal_dequant(source: &str) -> String {
    let mut result = String::new();
    let mut skip_embed = false;
    let mut embed_depth = 0;

    for line in source.lines() {
        // Skip embed blocks
        if line.contains("GGML_METAL_EMBED_LIBRARY") {
            skip_embed = true;
            embed_depth = 1;
            continue;
        }
        if skip_embed {
            if line.starts_with("#if") {
                embed_depth += 1;
            } else if line.starts_with("#endif") {
                embed_depth -= 1;
                if embed_depth == 0 {
                    skip_embed = false;
                }
            }
            continue;
        }

        // Skip includes
        if line.starts_with("#include") || line.starts_with("#pragma") {
            continue;
        }

        result.push_str(line);
        result.push('\n');
    }
    result
}

fn extract_metal_kernels(source: &str) -> String {
    let mut result = String::new();

    for line in source.lines() {
        // Skip includes
        if line.starts_with("#include") {
            continue;
        }
        result.push_str(line);
        result.push('\n');
    }
    result
}

fn extract_iq_tables(source: &str) -> String {
    let mut result = String::new();

    // Add the extra tables needed for IQ kernels
    result.push_str("constexpr constant static uint8_t kmask_iq2xs[8] = {1, 2, 4, 8, 16, 32, 64, 128};\n");
    result.push_str("constexpr constant static int8_t ksigns_iq2xs[128] = {\n");
    result.push_str(
        "     1,  1,  1,  1,  1,  1,  1,  1, -1,  1,  1,  1,  1,  1,  1, -1,\n",
    );
    result.push_str(
        "     1, -1,  1,  1,  1,  1,  1, -1, -1, -1,  1,  1,  1,  1,  1,  1,\n",
    );
    result.push_str(
        "     1,  1, -1,  1,  1,  1,  1, -1, -1,  1, -1,  1,  1,  1,  1,  1,\n",
    );
    result.push_str(
        "     1, -1, -1,  1,  1,  1,  1,  1, -1, -1, -1,  1,  1,  1,  1, -1,\n",
    );
    result.push_str(
        "     1,  1,  1, -1,  1,  1,  1, -1, -1,  1,  1, -1,  1,  1,  1,  1,\n",
    );
    result.push_str(
        "     1, -1,  1, -1,  1,  1,  1,  1, -1, -1,  1, -1,  1,  1,  1, -1,\n",
    );
    result.push_str(
        "     1,  1, -1, -1,  1,  1,  1,  1, -1,  1, -1, -1,  1,  1,  1, -1,\n",
    );
    result.push_str(
        "     1, -1, -1, -1,  1,  1,  1, -1, -1, -1, -1, -1,  1,  1,  1,  1,\n",
    );
    result.push_str("};\n\n");

    for line in source.lines() {
        // Skip pragma
        if line.starts_with("#pragma") {
            continue;
        }
        result.push_str(line);
        result.push('\n');
    }
    result
}
