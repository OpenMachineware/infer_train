// Test Q6_K tiled GEMM correctness
use infer_train::quant::kv_cache::quantize_row_q8_k_neon;
use infer_train::quant::matmul::tiled::matmul_q6k_q8k_tiled;
use infer_train::quant::types::{BlockQ6K, BlockQ8K, QK_K};

fn main() {
    // Test dimensions
    let m = 256; // output size
    let k = 256; // input size (single block)
    let n = 8; // batch size

    // Create dummy Q6_K weights
    let nb = k / QK_K;
    let mut weights = vec![
        BlockQ6K {
            ql: [0u8; 128],
            qh: [0u8; 64],
            scales: [1i8; 16],
            d: 64, // scale * 64
        };
        m * nb
    ];

    // Create input data
    let mut inputs = vec![vec![1.0f32; k]; n];

    // Quantize inputs to Q8_K
    let x_q8k_blocks_needed = n * nb;
    let mut x_q8k_batch =
        vec![0u8; x_q8k_blocks_needed * std::mem::size_of::<BlockQ8K>()];
    let x_q8k_blocks: &mut [BlockQ8K] = unsafe {
        std::slice::from_raw_parts_mut(
            x_q8k_batch.as_mut_ptr() as *mut BlockQ8K,
            x_q8k_blocks_needed,
        )
    };

    for (i, input) in inputs.iter().enumerate() {
        quantize_row_q8_k_neon(input, &mut x_q8k_blocks[i * nb..(i + 1) * nb]);
    }

    // Run tiled GEMM
    let mut outputs = vec![0.0f32; m * n];
    unsafe {
        matmul_q6k_q8k_tiled(&weights, x_q8k_blocks, &mut outputs, k, m, n);
    }

    // Check for NaN/Inf
    let mut nan_count = 0;
    let mut inf_count = 0;
    let mut valid_count = 0;

    for (i, &v) in outputs.iter().enumerate() {
        if v.is_nan() {
            nan_count += 1;
            if nan_count <= 5 {
                eprintln!("NaN at index {}", i);
            }
        } else if v.is_infinite() {
            inf_count += 1;
            if inf_count <= 5 {
                eprintln!("Inf at index {}", i);
            }
        } else {
            valid_count += 1;
        }
    }

    println!("=== Q6_K Tiled GEMM Test ===");
    println!("Dimensions: m={}, k={}, n={}", m, k, n);
    println!("Valid values: {}", valid_count);
    println!("NaN values: {}", nan_count);
    println!("Inf values: {}", inf_count);

    if nan_count == 0 && inf_count == 0 {
        println!("SUCCESS: No NaN or Inf values!");
    } else {
        println!(
            "FAILED: Found {} NaN and {} Inf values",
            nan_count, inf_count
        );
        std::process::exit(1);
    }

    // Sample output values
    println!("\nSample outputs (first 10):");
    for i in 0..10.min(outputs.len()) {
        println!("  outputs[{}] = {}", i, outputs[i]);
    }
}
