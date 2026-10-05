// List all kernel names in a Metal library
use metal::*;

fn main() {
    let device = Device::system_default().expect("No Metal device");

    println!("=== llama.cpp GEMM kernels ===");
    let llama_lib = device
        .new_library_with_file("shaders/mul_mm_template.metallib")
        .expect("Failed to load llama.cpp library");

    let mut kernels: Vec<String> =
        llama_lib.function_names().into_iter().collect();
    kernels.sort();

    for name in &kernels {
        if name.contains("mul_mm") {
            println!("  {}", name);
        }
    }

    println!("\n=== Self-written GEMM kernels ===");
    let self_lib = device
        .new_library_with_file("shaders/mul_mm_standalone.metallib")
        .expect("Failed to load self-written library");

    let mut kernels2: Vec<String> =
        self_lib.function_names().into_iter().collect();
    kernels2.sort();

    for name in &kernels2 {
        if name.contains("mul_mm") {
            println!("  {}", name);
        }
    }

    println!("\n=== Comparison ===");
    let llama_kernels: std::collections::HashSet<String> =
        kernels.into_iter().filter(|n| n.contains("mul_mm")).collect();
    let self_kernels: std::collections::HashSet<String> =
        kernels2.into_iter().filter(|n| n.contains("mul_mm")).collect();

    let missing: Vec<_> = llama_kernels.difference(&self_kernels).collect();
    if missing.is_empty() {
        println!("✓ All llama.cpp kernels implemented");
    } else {
        println!("Missing kernels:");
        for k in missing {
            println!("  - {}", k);
        }
    }

    let extra: Vec<_> = self_kernels.difference(&llama_kernels).collect();
    if !extra.is_empty() {
        println!("\nExtra kernels (not in llama.cpp):");
        for k in extra {
            println!("  + {}", k);
        }
    }
}
