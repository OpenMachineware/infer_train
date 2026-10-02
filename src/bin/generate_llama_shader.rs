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

    // Read IQ grid tables from our project
    let project_dir = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let iq_grid_tables =
        fs::read_to_string(project_dir.join("shaders/iq_grid_tables.h"))
            .expect("Failed to read iq_grid_tables.h");

    // Combine all sources
    let combined = format!(
        r#"
// Combined llama.cpp shader for Standard Q + TQ + MXFP

#include <metal_stdlib>
using namespace metal;

// Block type definitions from ggml-common.h (Metal version)
#define GGML_COMMON_DECL_METAL
{}

// Common definitions
{}

// IQ grid tables
{}

// Dequantize functions (contains block_q_n_dot_y)
{}

// Kernel implementations
{}
"#,
        extract_metal_blocks(&ggml_common),
        extract_metal_common(&common_h),
        extract_iq_tables(&iq_grid_tables),
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
    let mut result = String::new();

    result.push_str("using namespace metal;\n");
    result.push_str("typedef half ggml_half;\n");
    result.push_str("typedef half2 ggml_half2;\n\n");

    // Extract constants
    result.push_str("#define QK_K 256\n");
    result.push_str("#define K_SCALE_SIZE 12\n\n");

    // Extract block structures by searching for them
    let blocks = vec![
        ("QK1_0", "128", "block_q1_0"),
        ("QK2_0", "64", "block_q2_0"),
        ("QK4_0", "32", "block_q4_0"),
        ("QK4_1", "32", "block_q4_1"),
        ("QK5_0", "32", "block_q5_0"),
        ("QK5_1", "32", "block_q5_1"),
        ("QK8_0", "32", "block_q8_0"),
        ("QK_MXFP4", "32", "block_mxfp4"),
    ];

    for (qk_name, qk_val, _block_name) in &blocks {
        result.push_str(&format!("#define {} {}\n", qk_name, qk_val));
    }
    result.push_str("\n");

    // Standard block definitions (simplified, matching llama.cpp)
    result.push_str(
        "typedef struct { ggml_half d; uint8_t qs[QK1_0/8]; } block_q1_0;\n",
    );
    result.push_str(
        "typedef struct { ggml_half d; uint8_t qs[QK2_0/4]; } block_q2_0;\n",
    );
    result.push_str(
        "typedef struct { ggml_half d; uint8_t qs[QK4_0/2]; } block_q4_0;\n",
    );
    result.push_str("typedef struct { union { struct { ggml_half d; ggml_half m; }; ggml_half2 dm; }; uint8_t qs[QK4_1/2]; } block_q4_1;\n");
    result.push_str("typedef struct { ggml_half d; uint8_t qh[4]; uint8_t qs[QK5_0/2]; } block_q5_0;\n");
    result.push_str("typedef struct { union { struct { ggml_half d; ggml_half m; }; ggml_half2 dm; }; uint8_t qh[4]; uint8_t qs[QK5_1/2]; } block_q5_1;\n");
    result.push_str(
        "typedef struct { ggml_half d; int8_t qs[QK8_0]; } block_q8_0;\n",
    );
    result.push_str("typedef struct { uint8_t e; uint8_t qs[QK_MXFP4/2]; } block_mxfp4;\n\n");

    // TQ blocks
    result.push_str("typedef struct { uint8_t qs[(QK_K-4*QK_K/64)/5]; uint8_t qh[QK_K/64]; ggml_half d; } block_tq1_0;\n");
    result.push_str(
        "typedef struct { uint8_t qs[QK_K/4]; ggml_half d; } block_tq2_0;\n\n",
    );

    // K-quant blocks
    result.push_str("typedef struct { uint8_t scales[QK_K/16]; uint8_t qs[QK_K/4]; union { struct { ggml_half d; ggml_half dmin; }; ggml_half2 dm; }; } block_q2_K;\n");
    result.push_str("typedef struct { uint8_t hmask[QK_K/8]; uint8_t qs[QK_K/4]; uint8_t scales[12]; ggml_half d; } block_q3_K;\n");
    result.push_str("typedef struct { union { struct { ggml_half d; ggml_half dmin; }; ggml_half2 dm; }; uint8_t scales[K_SCALE_SIZE]; uint8_t qs[QK_K/2]; } block_q4_K;\n");
    result.push_str("typedef struct { union { struct { ggml_half d; ggml_half dmin; }; ggml_half2 dm; }; uint8_t scales[K_SCALE_SIZE]; uint8_t qh[QK_K/8]; uint8_t qs[QK_K/2]; } block_q5_K;\n");
    result.push_str("typedef struct { uint8_t ql[QK_K/2]; uint8_t qh[QK_K/4]; int8_t scales[QK_K/16]; ggml_half d; } block_q6_K;\n\n");

    // IQ blocks
    result.push_str(
        "typedef struct { ggml_half d; uint16_t qs[QK_K/8]; } block_iq2_xxs;\n",
    );
    result.push_str("typedef struct { ggml_half d; uint16_t qs[QK_K/8]; uint8_t scales[QK_K/32]; } block_iq2_xs;\n");
    result.push_str("typedef struct { ggml_half d; uint8_t qs[QK_K/4]; uint8_t qh[QK_K/32]; uint8_t scales[QK_K/32]; } block_iq2_s;\n");
    result.push_str("typedef struct { ggml_half d; uint8_t qs[3*QK_K/8]; } block_iq3_xxs;\n");
    result.push_str("typedef struct { ggml_half d; uint8_t qs[QK_K/4]; uint8_t qh[QK_K/32]; uint8_t signs[QK_K/8]; uint8_t scales[QK_K/64]; } block_iq3_s;\n");
    result.push_str("typedef struct { ggml_half d; uint8_t qs[QK_K/8]; uint16_t qh[QK_K/32]; } block_iq1_s;\n");
    result.push_str("typedef struct { uint8_t qs[QK_K/8]; uint8_t qh[QK_K/16]; uint8_t scales[QK_K/32]; } block_iq1_m;\n");
    result.push_str(
        "typedef struct { ggml_half d; uint8_t qs[QK4_0/2]; } block_iq4_nl;\n",
    );
    result.push_str("typedef struct { ggml_half d; uint16_t scales_h; uint8_t scales_l[QK_K/64]; uint8_t qs[QK_K/2]; } block_iq4_xs;\n");

    result
}

fn extract_metal_common(source: &str) -> String {
    let mut result = String::new();

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
