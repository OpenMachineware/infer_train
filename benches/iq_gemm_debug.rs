// Debug test for IQ GEMM Metal compilation
// Just checks if the library compiles and the kernel exists

use metal::Device;

fn main() {
    println!("=== IQ GEMM Metal Debug ===");

    let device = Device::system_default().expect("No Metal device");

    // Load self-written mul_mm_standalone.metal
    let ggml_common = include_str!("../../shaders/ggml-common.h");
    let ggml_common_metal =
        format!("#define GGML_COMMON_IMPL_METAL\n{}", ggml_common);
    let standalone_source =
        include_str!("../../shaders/mul_mm_standalone.metal");

    // Remove the #include line from standalone_source since we're prepending the content
    let standalone_no_include = standalone_source
        .lines()
        .filter(|line| !line.contains("#include \"iq_grid_tables.h\""))
        .collect::<Vec<_>>()
        .join("\n");

    let combined = format!("{}\n{}", ggml_common_metal, standalone_no_include);

    // Save the combined source to a file for inspection
    std::fs::write("/tmp/iq_gemm_debug.metal", &combined)
        .expect("Failed to write debug file");
    println!("Combined source written to /tmp/iq_gemm_debug.metal");

    let compile_opts = metal::CompileOptions::new();
    match device.new_library_with_source(&combined, &compile_opts) {
        Ok(lib) => {
            println!("✅ Library compiled successfully");

            // List all function names
            let functions = lib.function_names();
            println!("Found {} functions:", functions.len());
            for name in &functions {
                println!("  - {}", name);
            }

            // Try to get each IQ kernel
            let iq_kernels = [
                "kernel_mul_mm_iq1_s_f32",
                "kernel_mul_mm_iq1_m_f32",
                "kernel_mul_mm_iq2_xxs_f32",
                "kernel_mul_mm_iq2_xs_f32",
                "kernel_mul_mm_iq2_s_f32",
                "kernel_mul_mm_iq3_xxs_f32",
                "kernel_mul_mm_iq3_s_f32",
                "kernel_mul_mm_iq4_nl_f32",
                "kernel_mul_mm_iq4_xs_f32",
            ];

            println!("\nChecking IQ kernel functions:");
            for kernel in &iq_kernels {
                let cv = metal::FunctionConstantValues::new();
                match lib.get_function(kernel, Some(cv)) {
                    Ok(_) => println!("  ✅ {} found", kernel),
                    Err(e) => println!("  ❌ {} NOT found: {}", kernel, e),
                }
            }
        }
        Err(e) => {
            println!("❌ Library compilation failed:");
            println!("{}", e);
        }
    }
}
