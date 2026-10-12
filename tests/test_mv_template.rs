// Test MV template from llama.cpp
// Verify Rust can call the template kernels

use metal::{CompileOptions, Device};
use std::path::PathBuf;

fn expand_includes(
    source: &str,
    shader_dir: &PathBuf,
    visited: &mut Vec<String>,
) -> String {
    let mut result = String::new();

    for line in source.lines() {
        if line.starts_with("#include \"") {
            // Extract filename
            let start = line.find('"').unwrap() + 1;
            let end = line.rfind('"').unwrap();
            let filename = &line[start..end];

            // Skip if already included
            if visited.contains(&filename.to_string()) {
                result
                    .push_str(&format!("// Already included: {}\n", filename));
                continue;
            }

            visited.push(filename.to_string());

            // Read and expand the included file
            let include_path = shader_dir.join(filename);
            if let Ok(include_content) = std::fs::read_to_string(&include_path)
            {
                result.push_str(&format!("// --- {} ---\n", filename));
                // Remove #pragma once from included files
                let include_content =
                    include_content.replace("#pragma once", "// #pragma once");
                result.push_str(&expand_includes(
                    &include_content,
                    shader_dir,
                    visited,
                ));
                result.push_str(&format!("// --- end {} ---\n", filename));
            } else {
                result.push_str(&format!(
                    "// WARNING: Could not find: {}\n",
                    filename
                ));
                result.push_str(line);
                result.push('\n');
            }
        } else {
            result.push_str(line);
            result.push('\n');
        }
    }

    result
}

fn main() {
    let device = Device::system_default().expect("No Metal device");

    // Compile options
    let compile_options = CompileOptions::new();
    compile_options.set_fast_math_enabled(true);

    // Load all required sources and expand includes
    let shader_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("shaders");

    // Read the main template
    let source = std::fs::read_to_string(shader_dir.join("gemm.metal"))
        .expect("Failed to read gemm.metal");

    // Expand all includes recursively
    let mut visited = Vec::new();
    let expanded = expand_includes(&source, &shader_dir, &mut visited);

    // Compile
    println!("Compiling MV template (expanded: {} bytes)...", expanded.len());
    let library = device
        .new_library_with_source(&expanded, &compile_options)
        .expect("Failed to compile MV template");

    // List all function names
    println!("\n=== Available kernel functions ===");
    let functions = library.function_names();
    for name in functions.iter() {
        println!("{}", name);
    }

    // Check for specific Q4_K kernels
    println!("\n=== Q4_K kernels ===");
    let q4k_kernels =
        functions.iter().filter(|n| n.contains("q4_K")).collect::<Vec<_>>();
    for name in &q4k_kernels {
        println!("{}", name);
    }

    // Check for Q2_K kernels
    println!("\n=== Q2_K kernels ===");
    let q2k_kernels =
        functions.iter().filter(|n| n.contains("q2_K")).collect::<Vec<_>>();
    for name in &q2k_kernels {
        println!("{}", name);
    }

    // Try to get a specific kernel
    println!("\n=== Testing kernel retrieval ===");
    match library.get_function("kernel_mul_mm_ext_q4_K_f32_r1_2", None) {
        Ok(func) => println!("✓ Got kernel: {:?}", func.name()),
        Err(e) => println!("✗ Failed to get kernel: {}", e),
    }

    println!("\nTotal kernels: {}", functions.len());
}
