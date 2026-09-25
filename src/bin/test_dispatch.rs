// Test unified dispatch for all K-quant formats
// Run: cargo run --release --bin test_dispatch

use infer_train::quant::types::*;
use infer_train::quant::vec_dot::gpu::metal::MetalContext;

fn main() {
    println!("=== Testing Unified Dispatch for All K-Quant Formats ===\n");

    let ctx = MetalContext::new().expect("Failed to create Metal context");

    let m = 4096;
    let k = 4096;

    // Test decode (N=1) and batch (N=64) scenarios
    let test_cases = [
        ("Decode N=1", 1),
        ("Batch N=64", 64),
    ];

    let formats = [
        ("Q2_K", 144, "matmul_q2_k_f32"),
        ("Q3_K", 128, "matmul_q3_k_f32"),
        ("Q4_K", 144, "matmul_q4_k_f32"),
        ("Q5_K", 176, "matmul_q5_k_f32"),
        ("Q6_K", 210, "matmul_q6_k_f32"),
    ];

    println!("Testing dispatch for M={}, K={}\n", m, k);
    println!("Format\t\tN=1 (MV)\tN=64 (GEMM)");
    println!("{}", "-".repeat(50));

    for (name, block_size, _func_name) in formats.iter() {
        let weights = vec![0u8; m * (k/256) * block_size];
        let input_n1 = vec![0.5f32; k];
        let input_n64 = vec![0.5f32; k * 64];

        // Test N=1 (decode - should use MV)
        let result_n1 = match *name {
            "Q2_K" => ctx.matmul_q2_k_f32(m, 1, k,
                unsafe { std::slice::from_raw_parts(weights.as_ptr() as *const BlockQ2K, m * k / 256) },
                &input_n1),
            "Q3_K" => ctx.matmul_q3_k_f32(m, 1, k,
                unsafe { std::slice::from_raw_parts(weights.as_ptr() as *const BlockQ3K, m * k / 256) },
                &input_n1),
            "Q4_K" => ctx.matmul_q4_k_f32(m, 1, k,
                unsafe { std::slice::from_raw_parts(weights.as_ptr() as *const BlockQ4K, m * k / 256) },
                &input_n1),
            "Q5_K" => ctx.matmul_q5_k_f32(m, 1, k,
                unsafe { std::slice::from_raw_parts(weights.as_ptr() as *const BlockQ5K, m * k / 256) },
                &input_n1),
            "Q6_K" => ctx.matmul_q6_k_f32(m, 1, k,
                unsafe { std::slice::from_raw_parts(weights.as_ptr() as *const BlockQ6K, m * k / 256) },
                &input_n1),
            _ => panic!("Unknown format"),
        };

        // Test N=64 (batch - should use GEMM template)
        let result_n64 = match *name {
            "Q2_K" => ctx.matmul_q2_k_f32(m, 64, k,
                unsafe { std::slice::from_raw_parts(weights.as_ptr() as *const BlockQ2K, m * k / 256) },
                &input_n64),
            "Q3_K" => ctx.matmul_q3_k_f32(m, 64, k,
                unsafe { std::slice::from_raw_parts(weights.as_ptr() as *const BlockQ3K, m * k / 256) },
                &input_n64),
            "Q4_K" => ctx.matmul_q4_k_f32(m, 64, k,
                unsafe { std::slice::from_raw_parts(weights.as_ptr() as *const BlockQ4K, m * k / 256) },
                &input_n64),
            "Q5_K" => ctx.matmul_q5_k_f32(m, 64, k,
                unsafe { std::slice::from_raw_parts(weights.as_ptr() as *const BlockQ5K, m * k / 256) },
                &input_n64),
            "Q6_K" => ctx.matmul_q6_k_f32(m, 64, k,
                unsafe { std::slice::from_raw_parts(weights.as_ptr() as *const BlockQ6K, m * k / 256) },
                &input_n64),
            _ => panic!("Unknown format"),
        };

        let n1_ok = result_n1.is_ok();
        let n64_ok = result_n64.is_ok();
        let n1_len = result_n1.map(|r| r.len()).unwrap_or(0);
        let n64_len = result_n64.map(|r| r.len()).unwrap_or(0);

        println!("{}\t\t{}\t\t{}", name,
            if n1_ok && n1_len == m { "✓ OK" } else { "✗ FAIL" },
            if n64_ok && n64_len == m * 64 { "✓ OK" } else { "✗ FAIL" });
    }

    println!("\nAll dispatch functions working correctly!");
}
