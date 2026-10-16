use infer_train::inference::dequant::dequant_q5_k;
use infer_train::quant::types::BlockQ5K;
use infer_train::quant::vec_dot::gpu::metal::MetalContext;

#[test]
fn test_q5k_error_detailed() {
    // Start with just 2 rows to debug SIMD group issue
    let m = 2;
    let k = 256;

    // Random non-zero data
    let mut weights_bytes = vec![0u8; m * std::mem::size_of::<BlockQ5K>()];
    for i in 0..weights_bytes.len() {
        weights_bytes[i] = ((i * 17 + 13) % 256) as u8;
    }
    let weights: &[BlockQ5K] = unsafe {
        std::slice::from_raw_parts(weights_bytes.as_ptr() as *const BlockQ5K, m)
    };
    let input: Vec<f32> =
        (0..k).map(|i| ((i % 20) as f32 - 10.0) * 0.1).collect();

    // CPU computation
    let mut cpu_output = vec![0.0f32; m];
    for row in 0..m {
        let row_offset = row * std::mem::size_of::<BlockQ5K>();
        let mut dequant = vec![0.0f32; k];
        dequant_q5_k(&weights_bytes[row_offset..], &mut dequant, k);

        for i in 0..k {
            cpu_output[row] += dequant[i] * input[i];
        }
    }

    // GPU computation
    let ctx = MetalContext::new().unwrap();
    let gpu_output = ctx.mv_q5_k_f32(m, k, weights, &input).unwrap();

    println!("2-row test (debug SIMD group issue):");
    let mut max_diff = 0.0f32;
    for i in 0..m {
        let diff = (cpu_output[i] - gpu_output[i]).abs();
        let rel_err = if cpu_output[i].abs() > 1.0 {
            diff / cpu_output[i].abs()
        } else {
            diff
        };

        println!(
            "  Row {}: CPU={:.4}, GPU={:.4}, diff={:.6}, rel_err={:.6}",
            i, cpu_output[i], gpu_output[i], diff, rel_err
        );

        max_diff = max_diff.max(diff);
    }

    println!("\nMax diff: {:.6}", max_diff);

    // Verify
    assert!(max_diff < 0.5, "Max diff too large: {}", max_diff);
}
