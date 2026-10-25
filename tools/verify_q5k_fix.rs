// Verify Q5_K GEMM doesn't produce NaN
// Run with: cargo run --release --features metal --bin verify-q5k-fix

use bytemuck::Zeroable;
use infer_train::quant::types::BlockQ5K;
use infer_train::quant::vec_dot::gpu::metal::MetalContext;

fn main() {
    let m = 32;
    let k = 256;

    // Create test data
    let n_blocks = k / 256;
    let mut weights = vec![BlockQ5K::zeroed(); m * n_blocks];

    // Fill with test data
    for (i, block) in weights.iter_mut().enumerate() {
        let d_val = 0.1 * (i as f32 + 1.0);
        block.d = half::f16::from_f32(d_val).to_bits();
        block.dmin = half::f16::from_f32(d_val).to_bits();

        // Set scales: scale=1, min=16
        for j in 0..4 {
            block.scales[j] = 1;
            block.scales[j + 4] = 16;
        }
        for j in 0..4 {
            block.scales[j + 8] = (16 << 4) | 1;
        }

        // Fill qs and qh with test data
        for j in 0..128 {
            block.qs[j] = (j % 32) as u8;
        }
        for j in 0..32 {
            block.qh[j] = (j % 4) as u8;
        }
    }

    // Create input
    let input: Vec<f32> = (0..k).map(|i| (i % 100) as f32 * 0.01).collect();

    // Setup Metal context
    let ctx = MetalContext::new().expect("Failed to create Metal context");

    // Upload weights to GPU
    let weights_bytes = unsafe {
        std::slice::from_raw_parts(
            weights.as_ptr() as *const u8,
            weights.len() * std::mem::size_of::<BlockQ5K>(),
        )
    };
    let weights_buffer = ctx.device().new_buffer_with_data(
        weights_bytes.as_ptr() as *const _,
        weights_bytes.len() as u64,
        metal::MTLResourceOptions::StorageModeManaged,
    );

    // Upload input to GPU
    let input_buffer = ctx.device().new_buffer_with_data(
        input.as_ptr() as *const _,
        (input.len() * std::mem::size_of::<f32>()) as u64,
        metal::MTLResourceOptions::StorageModeManaged,
    );

    // Run GEMV
    let output_buffer = ctx
        .mv_q5_k_f32_gpu_weights(m, k, &weights_buffer, &input_buffer)
        .expect("GEMV failed");

    // Download results
    let result = unsafe {
        std::slice::from_raw_parts(output_buffer.contents() as *const f32, m)
    };

    // Check for NaN/Inf
    let nan_count = result.iter().filter(|x| x.is_nan()).count();
    let inf_count = result.iter().filter(|x| x.is_infinite()).count();

    println!("\n=== Q5_K GEMV Test ===");
    println!("M: {}, K: {}", m, k);
    println!();

    for i in 0..std::cmp::min(10, m) {
        println!("  Row {}: {:.6}", i, result[i]);
    }
    println!("  ...");

    println!();
    println!("NaN count: {}/{}", nan_count, m);
    println!("Inf count: {}/{}", inf_count, m);

    let valid: Vec<f32> =
        result.iter().filter(|x| x.is_finite()).cloned().collect();
    if !valid.is_empty() {
        let min = valid.iter().cloned().fold(f32::INFINITY, f32::min);
        let max = valid.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        println!("Valid range: [{:.6}, {:.6}]", min, max);
    }

    if nan_count > 0 {
        println!("\nERROR: NaN detected!");
        std::process::exit(1);
    }

    if inf_count > 0 {
        println!("\nERROR: Inf detected!");
        std::process::exit(1);
    }

    println!("\nSUCCESS: Q5_K GEMV produces valid output!");
}
